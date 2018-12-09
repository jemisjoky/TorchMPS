#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torchvision
import random
import time
import sys
from math import ceil
from misc import load_HV_data, convert_to_onehot, joint_shuffle

# IMPORTANT
# Deal with non-periodic boundary conditions
# Write MPS compression function using iterated SVD's, make sure you don't backpropagate through 
#   the SVD though, sounds like this is numerically ill-conditioned
# Deal with variable local bond dimensions
#   * This should mostly involve changes in _contract_batch_input, right?
# Rewrite loop over input images in forward function to make use of batch multiplication

# REALLY MINOR
# Write code to move location of bond index

class MPSClassifier(nn.Module):
    def __init__(self, size, D, d=2, num_labels=10, **args):
        """
        Define variables for holding our MPS cores (trainable)
        """
        super(MPSClassifier, self).__init__()

        # Number of sites in the MPS
        self.size = size
        # Dimension of local embedding space (one per pixel)
        self.d = d
        # Number of distinct labels for our classification
        self.num_labels = num_labels

        # Maximum global bond dimension D. If a tuple is passed, the
        # bond dim is adaptive, with the first element the minimum 
        # bond dim and the latter the maximum.
        if type(D) not in [list, tuple] or len(D) == 1:
            self.D = D
            self.adaptive = False
        elif len(D) == 2:
            self.max_D = D[1]
            self.min_D = D[0]
            self.adaptive = True

            self.D = D[1]      # Redundant, but improves consistency
            D = self.D
        else:
            raise ValueError("D must be integer or tuple of integers")

        # Commonly used tensor shapes for our classifier
        full_shape = [size, D, d, D]
        label_shape = [D, num_labels, D]
        single_shape = torch.tensor([D, d, D]).unsqueeze(0)
        
        # Information about the shapes of our tensors
        self.core_shapes = single_shape.expand([size, 3]).clone()
        self.label_shape = torch.tensor(label_shape)

        # Location of the label index
        self.label_site = size // 2

        # TODO: Deal with label_site at first/last sites in the
        # presence of open boundary conditions
        assert self.label_site not in [0, size]

        # If args is specified as dict, unpack it
        if 'args' in args.keys():
            args = args['args']

        # Type of boundary conditions for our MPS, either 
        # 'open' or 'periodic' (default 'periodic')
        if 'bc' in args.keys():
            self.bc = args['bc']
        else:
            self.bc = 'periodic'

        # Method used to initialize our tensor weights, either
        # 'random' or 'random_eye' (default 'random')
        if 'weight_init_method' in args.keys():
            if 'weight_init_scale' not in args.keys():
                raise ValueError("If 'weight_init_method' is set, "
                        "'weight_init_scale' must also be set")
            init_method = args['weight_init_method']
            init_std = args['weight_init_scale']
        else:
            init_method = 'random_eye'
            init_std = 0.01

        self.init_method = init_method
        self.init_std = init_std

        # Initialize the core tensors, which live on input sites,
        # and the label tensor, which outputs label predictions.
        # Note that 'random' sets our tensors completely randomly,
        # while 'random_eye' sets them close to the identity
        if init_method == 'random':
            core_tensors = init_std * torch.randn(full_shape)
            label_tensor = init_std * torch.randn(label_shape)
        elif init_method == 'random_eye':
            core_tensors = torch.eye(D).view([1, D, 1, D])
            core_tensors = core_tensors.expand(full_shape) + \
                           init_std * torch.randn(full_shape)
            label_tensor = torch.eye(D).view([D, 1, D])
            label_tensor = label_tensor.expand(label_shape) + \
                           init_std * torch.randn(label_shape)

        # With open boundary conditions, project the first and last
        # core tensors onto a rank-1 matrix
        if self.bc == 'open':
            core_tensors[0, 1:] = 0
            self.core_shapes[0] = torch.tensor([1, d, D])
            
            core_tensors[-1, :, :, 1:] = 0
            self.core_shapes[-1] = torch.tensor([D, d, 1])
            
        self.core_tensors = nn.Parameter(core_tensors)
        self.label_tensor = nn.Parameter(label_tensor)

        # Flag for preventing backpropagation through compression
        self._compress_called = False

    def _contract_batch_input(self, batch_input):
        """
        Contract batch of input data with MPS cores and return 
        the resulting batch of matrices as a torch tensor
        """
        size, D, d = self.size, self.D, self.d
        batch_size = batch_input.shape[0]

        # Massage the shape of the core tensors and data 
        # vectors, then batch multiply them together
        core_mats = self.core_tensors.permute([0, 1, 3, 2])
        core_mats = core_mats.contiguous().view(
                              [size, D*D, d]).unsqueeze(0)
        core_mats = core_mats.expand([batch_size, size, D*D, d])

        core_mats = core_mats.contiguous().view(
                              [batch_size*size, D*D, d])
        batch_input = batch_input.view([batch_size*size, d, 1])

        batch_mats = torch.bmm(core_mats, batch_input)
        return batch_mats.view([batch_size, size, D, D])

    def embedding_map(self, datum):
        """
        Take a single input and embed it into a local
        feature space of dimension d.

        I'm using a particularly simple affine embedding 
        map of the form x --> [1-x, x]
        """
        if self.d != 2:
            raise ValueError("Embedding map needs d=2, but "
                "self.d={}".format(self.d))
        else:
            return torch.tensor([1-datum, datum])

    def _embed_batch_input(self, batch_input):
        """
        Take batch input data and embed each data point into a 
        local feature space of dimension d using embedding_map
        """
        batch_size = batch_input.shape[0]
        size = batch_input.shape[1]
        batch_input = batch_input.view(batch_size*size)
        
        embed_data = [self.embedding_map(pixel).unsqueeze(0) 
                      for pixel in batch_input]
        embed_data = torch.stack(embed_data, 0)

        return embed_data.view([batch_size, size, self.d])


    def _pack_tensors(self, core_tensors, label_tensor):
        """
        Take a list of core tensors and a label tensor with 
        different shapes and pack them into the parameter tensors
        self.core_tensors and self.label_tensor. This also updates
        the shape information, self.core_shape and self.label_shape.
        """
        assert len(core_tensors) == self.size
        size = self.size
        D = self.D
        d = self.d
        label_site = self.label_site
        num_labels = self.num_labels

        # Shapes of all input tensors
        core_shapes = torch.stack([torch.tensor(ct.shape)
                                   for ct in core_tensors])
        label_shape = torch.tensor(label_tensor.shape)
        full_shape = [size, D, d, D]

        # With open boundaries, check that our end tensors terminate
        # and reformat to add singleton dimensions if necessary
        if self.bc == 'open':
            # Check on the left...
            if len(core_shapes[0]) == 2:
                core_tensors[0] = core_tensors[0].unsqueeze(0)
                core_shapes[0] = core_shapes[0].unsqueeze(0)
            if core_shapes[0, 0] != 1:
                raise RuntimeError("Open boundaries need leftmost "
                      "tensor with shape[0] = 1 (input_shape[0] = "
                      "{0})".format(core_shapes[0, 0]))

            # ...and on the right
            if len(core_shapes[-1]) == 2:
                core_tensors[-1] = core_tensors[-1].unsqueeze(2)
                core_shapes[-1] = core_shapes[-1].unsqueeze(2)
            if core_shapes[-1, 2] != 1:
                raise RuntimeError("Open boundaries need rightmost "
                      "tensor with shape[2] = 1 (input_shape[2] = "
                      "{0})".format(core_shapes[-1, 1]))

        # Check that the input shapes are internally consistent
        try:
            for i in range(size):
                # Typical sites
                if i < label_site-1:
                    left_shape = core_shapes[i]
                    right_shape = core_shapes[i+1]

                # Site to the left of the label site
                elif i == label_site-1:
                    left_shape = core_shapes[i]
                    right_shape = label_shape

                # Site to the right of the label site
                elif i == label_site:
                    left_shape = label_shape
                    right_shape = core_shapes[i]

                elif i > label_site:
                    left_shape = core_shapes[i-1]
                    right_shape = core_shapes[i]

                assert left_shape[2] == right_shape[0]
                assert torch.max(left_shape[0], right_shape[2]) <= D
        except AssertionError:
            i_adj = -1 if i < label_site else 0
            left_id = ("label site" if i == label_site else 
                       "site {0}".format(i + i_adj))
            right_id = ("label site" if i == label_site-1 else 
                       "site {0}".format(i + i_adj + 1))
            msg = ("Shape mismatch, {0} (left) has shape {1}, but "
                  "{2} (right) has shape {3}. Max D = {4}".format(
                    left_id, left_shape, right_id, right_shape, D))
            raise RuntimeError(msg)

        # Populate our tensors with the inputs
        with torch.no_grad():
            for i, shape in enumerate(core_shapes):
                s0, s2 = shape[0], shape[2]
                self.core_tensors.data[i, 0:s0,:,0:s2] = \
                                            core_tensors[i]

            s0, s2 = label_shape[0], label_shape[2]
            self.label_tensor.data[0:s0,:,0:s2] = label_tensor

        # Just want to make sure these are trainable
        self.core_tensors.requires_grad = True
        self.label_tensor.requires_grad = True

        # Finally, update the shape information
        self.core_shapes = core_shapes
        self.label_shape = label_shape

    def _unpack_tensors(self):
        """
        Take our single parameter tensor self.core_tensors and 
        return a (hard copy) list of core tensors with shapes 
        matching up with self.core_shapes.
        """
        core_shapes = self.core_shapes
        label_shape = self.label_shape

        core_tensors = []
        for i, shape in enumerate(core_shapes):
            s0, s2 = shape[0], shape[2]
            core_tensors.append(
                         self.core_tensors[i, 0:s0,:,0:s2].detach())

        s0, s2 = label_shape[0], label_shape[2]
        label_tensor = self.label_tensor[0:s0,:,0:s2].detach()

        # Might be overkill, but want to ensure no autograd tracking
        for tensor in core_tensors:
            tensor.requires_grad = False
        label_tensor.requires_grad = False

        # Also, make sure all the core tensors are contiguous
        core_tensors = [t.contiguous() for t in core_tensors]

        assert len(core_tensors) == self.size
        return core_tensors, label_tensor

    def compress(self, cutoff=10e-10):
        """
        Uses iterative singular value decompositions to minimize the
        bond dimensions between all neighboring sites.

        `cutoff` is the singular value threshold for truncating the
        bond dimension, which is ignored if such a cutoff would move
        the bond dimension outside of the range [min_D, D].
        """
        if self.bc != 'open':
            raise ValueError("`compress` only defined for open "
                  "boundary conditions, but self.bc={0}".format(
                                                    self.bc))
        elif self.adaptive == False:
            print("self.adaptive is False, can't compress. Need "
                  "to initialize classifier with D=[min_D, max_D]")

        size = self.size
        max_D = self.max_D
        min_D = self.min_D
        d = self.d
        label_site = self.label_site
        num_labels = self.num_labels

        core_tensors, label_tensor = self._unpack_tensors()
        core_shapes, label_shape = self.core_shapes, self.label_shape

        # Initialize leftover matrices (lom) and bond dimensions
        # for the left and right sides, along with loop limit
        left_D, right_D = 1, 1
        left_lom, right_lom = torch.eye(1), torch.eye(1)
        max_i = max(label_site, size-label_site)

        # Used to maintain numerical stability during compression
        log_scale = 0.

        # Start from the left and compress up to the label site
        for i in range(max_i):
            left_i, right_i = i, (size-1) - i

            # Start on the left side...
            if left_i < label_site:
                # Tack on the leftover, reshape, and take the SVD
                left_mshape = [core_shapes[left_i,0], 
                               d * core_shapes[left_i,2]]
                left_mat = torch.mm(left_lom, 
                           core_tensors[left_i].view(left_mshape))

                left_mshape = [left_D * d, core_shapes[left_i,2]]
                U,S,V = torch.svd(left_mat.view(left_mshape))

                # For stability set the maximum singular value to 1
                max_sv = S[0]
                S = S / max_sv
                log_scale += torch.log(max_sv)

                # Filter against our cutoff to get the new D
                assert all(S[:-1] >= S[1:]) # Assume decreasing SV's
                S = torch.stack([sv for sv in S if sv > cutoff])
                new_D = len(S)

                # Truncate matrices and reassign loop variables
                left_tshape = [left_D, d, new_D]
                V = V[:,:new_D]    # (SVD outputs transpose of V)

                core_tensors[left_i] = U[:,:new_D].view(left_tshape)
                left_D = new_D
                left_lom = torch.mm(torch.diag(S), torch.t(V))

            # ...then work on the right side
            if right_i >= label_site:
                # Tack on the leftover, reshape, and take the SVD
                right_mshape = [core_shapes[right_i,0] * d, 
                                core_shapes[right_i,2]]

                right_mat = torch.mm(
                         core_tensors[right_i].view(right_mshape),
                         right_lom)

                right_mshape = [core_shapes[right_i,0], d * right_D]
                U, S, V = torch.svd(right_mat.view(right_mshape))

                # For stability set the maximum singular value to 1
                max_sv = S[0]
                S = S / max_sv
                log_scale += torch.log(max_sv)

                # Filter against our cutoff to get the new D
                assert all(S[:-1] >= S[1:]) # Assume decreasing SV's
                S = torch.stack([sv for sv in S if sv > cutoff])
                new_D = len(S)

                # Truncate matrices and reassign loop variables
                right_tshape = [new_D, d, right_D]
                U = U[:,:new_D]

                core_tensors[right_i] = torch.t(V[:,:new_D]
                                        ).view(right_tshape)
                right_D = new_D
                right_lom = torch.mm(U, torch.diag(S))

        # Merge the leftover matrices with label_tensor
        lshape = [label_shape[0], num_labels * label_shape[2]]
        label_tensor = torch.mm(left_lom,
                                label_tensor.view(lshape))
        lshape = [left_D * num_labels, label_shape[2]]
        label_tensor = torch.mm(label_tensor.view(lshape),
                                right_lom)
        lshape = [left_D, num_labels, right_D]
        label_tensor = label_tensor.view(lshape)

        # Finally, restore all the magnitude taken away above
        scale = torch.exp(log_scale / size)
        core_tensors = [scale * tensor for tensor in core_tensors]

        self._pack_tensors(core_tensors, label_tensor)
        self._compress_called = True

    def forward(self, batch_input):
        """
        Evaluate our classifier on a batch of input data 
        and return a vector of logit scores over labels

        batch_input must be formatted as a torch tensor of size 
        [batch_size, input_size], where every entry lies in
        the interval [0, 1].

        The internals of this evaluation are different for open vs
        periodic boundary conditions, and also take into account
        the location of the label index within our classifier.
        """
        size, D, d = self.size, self.D, self.d
        num_labels = self.num_labels
        label_site = self.label_site

        batch_size = batch_input.shape[0]
        input_size = batch_input.shape[1]

        if input_size != size:
            raise ValueError(("len(batch_input) = {0}, but "
                  "needs to be {1}").format(input_size, size))

        # Get local embedded images of input pixels
        batch_input = self._embed_batch_input(batch_input)
        batch_input.requires_grad = False

        # Contract the input images to get a batch of matrices 
        batch_mats = self._contract_batch_input(batch_input)

        # Interchange site and batch dimensions (for ease of 
        # indexing), then divide into regions left and right of
        # the label site
        batch_mats = batch_mats.permute([1, 0, 2, 3])
        left_mats = batch_mats[:label_site]
        right_mats = batch_mats[label_site:]

        # Size of the left and right matrix products, which
        # decrease by half on each iteration of the following
        left_size = label_site
        right_size = size - label_site

        # Iteratively multiply nearest neighboring pairs of 
        # matrices until we've reduced the left and right regions
        # to a single matrix on each side
        while left_size > 1 or right_size > 1:
            if left_size > 1:
                odd_size = (left_size % 2) == 1
                left_size = left_size // 2
            
                # If our size is odd, set aside the leftover 
                # (right-most) matrices to get an even size
                if odd_size:
                    lone_mats = left_mats[-1].unsqueeze(0)
                    left_mats = left_mats[:-1]
                else:
                    lone_mats = None

                # Divide matrices into neighboring pairs and 
                # contract all pairs using batch multiplication
                mats1, mats2 = left_mats[0::2].contiguous(), \
                               left_mats[1::2].contiguous()
                mats1 = mats1.view([left_size*batch_size, D, D])
                mats2 = mats2.view([left_size*batch_size, D, D])
                left_mats = torch.bmm(mats1, mats2)
                
                # Reshape and append any leftover matrices
                left_mats = left_mats.view(
                            [left_size, batch_size, D, D])
                if odd_size:
                    left_size = left_size + 1
                    left_mats = torch.cat([left_mats, lone_mats], 0)

            if right_size > 1:
                odd_size = (right_size % 2) == 1
                right_size = right_size // 2
            
                # If our size is odd, set aside the leftover 
                # (right-most) matrices to get an even size
                if odd_size:
                    lone_mats = right_mats[-1].unsqueeze(0)
                    right_mats = right_mats[:-1]
                else:
                    lone_mats = None

                # Divide matrices into neighboring pairs and 
                # contract all pairs using batch multiplication
                mats1, mats2 = right_mats[0::2].contiguous(), \
                               right_mats[1::2].contiguous()
                mats1 = mats1.view([right_size*batch_size, D, D])
                mats2 = mats2.view([right_size*batch_size, D, D])
                right_mats = torch.bmm(mats1, mats2)
                
                # Reshape and append any leftover matrices
                right_mats = right_mats.view(
                            [right_size, batch_size, D, D])
                if odd_size:
                    right_size = right_size + 1
                    right_mats = torch.cat([right_mats, lone_mats], 0)

        # For each image, we now have one product matrix on the left
        # and one on the right, which will be contracted with the
        # central label tensor. In order to use batch multiplication
        # for this, we need to first do some expanding and reshaping
        left_stack = left_mats.squeeze().unsqueeze(1).expand(
                               [batch_size, num_labels, D, D])
        left_stack = left_stack.contiguous().view(
                                [batch_size*num_labels, D, D])

        right_stack = right_mats.squeeze().unsqueeze(1).expand(
                                 [batch_size, num_labels, D, D])
        right_stack = right_stack.contiguous().view(
                                  [batch_size*num_labels, D, D])

        label_tensor = self.label_tensor.permute([1, 0, 2])
        label_tensor = label_tensor.unsqueeze(0).expand(
                                    [batch_size, num_labels, D, D])
        label_tensor = label_tensor.contiguous().view(
                                    [batch_size*num_labels, D, D])

        # And here's the actual contraction with the label tensor
        label_tensor = torch.bmm(left_stack, label_tensor)
        label_tensor = torch.bmm(label_tensor, right_stack)
        
        # If we have periodic boundary conditions then we take the
        # partial trace over the bond indices to get (logit) scores
        # (In case of open boundaries, this does nothing)
        # (eye_vecs is just a convenient way of doing a partial
        #  trace over the bond dimension using batch multiplication)
        label_tensor = label_tensor.view(
                        [batch_size, num_labels, D*D])
        eye_vecs = torch.eye(D).view(D*D).unsqueeze(0).expand(
                                [batch_size, D*D]).unsqueeze(2)
        batch_scores = torch.bmm(label_tensor, eye_vecs)

        self._compress_called = False
        return batch_scores.squeeze()



    # def backward():
    #     if self._compress_called == True:
    #         raise RuntimeError("Can't backpropagate through "
    #               ".compress(), must call before .forward()")
    #     # TODO: Call parent backward routine 



    def num_correct(self, input_imgs, labels, batch_size=100):
        """
        Use our classifier to predict the labels associated with a
        batch of input images, then compare with the correct labels
        and return the number of correct guesses. For the sake of
        memory, the input is processed in batches of size batch_size
        """
        bs = batch_size
        num_imgs = input_imgs.size(0)
        num_batches = ceil(num_imgs / bs)
        num_corr = 0.

        for b in range(num_batches):
            if b == num_batches-1:
                # Last batch might be smaller
                batch_input = input_imgs[b*bs:]
                batch_labels = labels[b*bs:]
            else:
                batch_input = input_imgs[b*bs:(b+1)*bs]
                batch_labels = labels[b*bs:(b+1)*bs]

            with torch.no_grad():
                scores = self.forward(batch_input)
            predictions = torch.argmax(scores, 1)
            
            num_corr += torch.sum(
                        torch.eq(predictions, batch_labels)).float()

        return num_corr

if __name__ == "__main__":
    # Experimental parameters
    length = 28                 # Linear dimension of input images
    size = length**2            # Number of pixels in input images
    num_train_imgs = 5000       # Total number of training and
    num_test_imgs = 5000        #   testing images
    
    D = 40                      # Maximum bond dimension
    d = 2                       # Local feature dimension
    num_labels = 10             # Number of classification labels

    epochs = 10                 # Rounds of training
    batch_size = 100            # Size of minibatches
    weight_decay = 1e-3         # L2 regularizer weight
    loss_type = 'crossentropy'  # Either 'mse' or 'crossentropy'

    # Parameters for defining our classifier
    args = {'bc': 'open',
            'weight_init_method': 'random_eye',
            'weight_init_scale': 0.01}
    
    # We drop the last batch, so check that we have enough for one
    batches = num_train_imgs // batch_size
    if batches < 1:
        raise ValueError("Batch size < # of training images")

    # Initialize timing variables
    start_point = time.time()
    forward_time = 0.
    back_time = 0.
    diag_time = 0.
    torch.manual_seed(23)

    # Get and set GPU-related parameters
    if len(sys.argv) == 1:
        want_gpu = True
    else:
        want_gpu = False if sys.argv[1]=='--no_gpu' else True

    use_gpu = want_gpu and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor'
                  if use_gpu else 'torch.FloatTensor')
    print("Using device:", device)

    # Load the training and test sets
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='./mnist',
                train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./mnist',
                train=False, download=True, transform=transform)
    
    # Initialize image and label tensors
    train_imgs = torch.stack([data[0].view(size)
                          for data in train_set])
    test_imgs = torch.stack([data[0].view(size)
                          for data in test_set])
    train_lbls = torch.stack([data[1] for data in train_set])
    test_lbls = torch.stack([data[1] for data in test_set])
    label_vecs = convert_to_onehot(train_lbls, num_labels)

    # If we don't want to train on all of MNIST, pare down a bit
    if len(train_imgs) > num_train_imgs:
        train_imgs, train_lbls = joint_shuffle(train_imgs, train_lbls)
        train_imgs = train_imgs[:num_train_imgs]
        train_lbls = train_lbls[:num_train_imgs]

    if len(test_imgs) > num_test_imgs:
        test_imgs, test_lbls = joint_shuffle(test_imgs, test_lbls)
        test_imgs = test_imgs[:num_test_imgs]
        test_lbls = test_lbls[:num_test_imgs]

    print("Training on {0} images of size "
          "{1}x{1} for {2} epochs".format(
                                   num_train_imgs, length, epochs))
    print("Using bond dimension D =", D)
    print()

    # Build our MPS classifier using our chosen parameters
    classifier = MPSClassifier(size=size, D=D, d=d,
                               num_labels=num_labels, args=args)

    if use_gpu:
        classifier = classifier.cuda(device=device)
        train_imgs = train_imgs.cuda(device=device)
        train_lbls = train_lbls.cuda(device=device)
        test_imgs = test_imgs.cuda(device=device)
        test_lbls = test_lbls.cuda(device=device)

    # Initialize loss function and optimizer
    if loss_type == 'mse':
        loss_f = nn.MSELoss()
    elif loss_type == 'crossentropy':
        loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1E-3, 
                                 weight_decay=weight_decay)
    init_time = time.time() - start_point

    # Compute and print out the initial accuracy
    diag_point = time.time()
    train_correct = classifier.num_correct(train_imgs,
                               train_lbls, batch_size)
    train_acc = train_correct / num_train_imgs
    test_correct = classifier.num_correct(test_imgs,
                              test_lbls, batch_size)
    test_acc = test_correct / num_test_imgs
    diag_time += time.time() - diag_point

    print("### before training ###")
    print("training accuracy = {:.4f}".format(train_acc.item()))
    print("test accuracy = {:.4f}".format(test_acc.item()))
    print()

    # Start the actual training
    for epoch in range(epochs):
        av_loss = 0.

        for batch in range(batches):
            # Get data for this batch
            batch_imgs = train_imgs[batch*batch_size:
                                   (batch+1)*batch_size]
            batch_lbls = train_lbls[batch*batch_size:
                                   (batch+1)*batch_size]
            batch_vecs = convert_to_onehot(batch_lbls, num_labels)

            # Compute the loss and add it to the running total
            forward_point = time.time()
            scores = classifier(batch_imgs)
            if loss_type == 'mse':
                loss = loss_f(scores, batch_vecs.float())
            elif loss_type == 'crossentropy':
                loss = loss_f(scores, batch_lbls)
            av_loss += loss
            forward_time += time.time() - forward_point

            # Get the gradients and take an optimization step
            back_point = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            back_time += time.time() - back_point

        # Compute the training information, repeat
        loss = av_loss / batches

        print("### epoch", epoch, "###")
        print("average loss = {:.4e}".format(loss.item()))

        # Every so often, calculate the diagnostic information
        if epoch % 1 == 0:
            diag_point = time.time()
            train_correct = classifier.num_correct(train_imgs, train_lbls)
            train_acc = train_correct / num_train_imgs
            test_correct = classifier.num_correct(test_imgs, test_lbls)
            test_acc = test_correct / num_test_imgs
            diag_time += time.time() - diag_point

            current_point = time.time()
            run_time = current_point - start_point
            print("training accuracy = {:.4f}".format(train_acc.item()))
            print("test accuracy = {:.4f}".format(test_acc.item()))
            print("  forward time  = {0:.2f} sec".format(forward_time))
            print("  backprop time = {0:.2f} sec".format(back_time))
            print("  error time    = {0:.2f} sec".format(diag_time))
            print("  ---------------------------")
            print("  runtime so far = {0:.2f} sec".format(run_time))
        print()

        # Compress the classifier's bond dimensions
        compress_point = time.time()
        compress_time += time.time() - compress_point
        classifier.compress()

        # Shuffle our training data for the next epoch
        train_imgs, train_lbls = joint_shuffle(train_imgs, train_lbls)

    # Get relevant runtimes
    run_time = time.time() - start_point

    print("loading time  = {0:.2f} sec".format(init_time))
    print("forward time  = {0:.2f} sec".format(forward_time))
    print("backprop time = {0:.2f} sec".format(back_time))
    print("error time    = {0:.2f} sec".format(diag_time))
    print("---------------------------")
    print("total runtime = {0:.2f} sec".format(run_time))
