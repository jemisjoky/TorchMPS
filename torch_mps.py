#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import random
import time
import sys
from math import ceil
from misc import load_HV_data, convert_to_onehot, joint_shuffle

# NOT AS CRUCIAL
# Allow placement of label index in any location
# Deal with non-periodic boundary conditions
# Write MPS compression function using iterated SVD's, make sure you don't backpropagate through 
#   the SVD though, sounds like this is numerically ill-conditioned
# Deal with variable local bond dimensions
#   * This should mostly involve changes in _contract_batch_input, right?

# REALLY MINOR
# Write code to move location of bond index
# Add ability to specify type of datatype/scalar (currently only float allowed)
# Rewrite for loop in batch prediction (logits) function to use batch multiplication

class MPSClassifier(nn.Module):
    def __init__(self, size, D, d=2, num_labels=10, **args):
        """
        Define variables for holding our MPS cores (trainable)
        """
        super(MPSClassifier, self).__init__()

        # Number of sites in the MPS
        self.size = size
        # Global bond dimension
        self.D = D 
        # Dimension of local embedding space (one per pixel)
        self.d = d
        # Number of distinct labels for our classification
        self.num_labels = num_labels

        # If args is specified as dict, unpack it
        if 'args' in args.keys():
            args = args['args']

        # Type of boundary conditions for our MPS, either 
        # 'open' or 'periodic' (default 'periodic')
        if 'bc' in args.keys():
            self.bc = args['bc']
        else:
            self.bc = 'periodic'

        # Information about the shapes of our tensors
        full_shape = [size, D, D, d]
        label_shape = [D, D, num_labels]
        single_shape = torch.tensor([D, D, d]).unsqueeze(0)
        self.core_shapes = single_shape.expand([size, 3])     
        self.label_shape = torch.tensor(label_shape)

        # Location of the label index
        # TODO: Make this work by changing forward() appropriately
        self.label_location = size // 2

        # Method used to initialize our tensor weights, either
        # 'random' or 'random_eye' (default 'random')
        if 'weight_init_method' in args.keys():
            if 'weight_init_scale' not in args.keys():
                raise ValueError("Need to set 'weight_init_scale'")
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
            self.core_tensors = nn.Parameter(
                                init_std * torch.randn(full_shape))
            self.label_tensor = nn.Parameter(
                                init_std * torch.randn(label_shape))
        elif init_method == 'random_eye':
            core_tensors = torch.eye(D).view([1, D, D, 1])
            core_tensors = core_tensors.expand(full_shape) + \
                           init_std * torch.randn(full_shape)
            label_tensor = torch.eye(D).unsqueeze(2)
            label_tensor = label_tensor.expand(label_shape) + \
                           init_std * torch.randn(label_shape)
            
            self.core_tensors = nn.Parameter(core_tensors)
            self.label_tensor = nn.Parameter(label_tensor)
        
    def _tensors_as_mats(self):
        """
        Return the size x D x D x d tensor holding our MPS cores,
        but reshaped into a size x (D*D) x d tensor we can feed 
        into batch multiplication
        """
        full_shape = [self.size, self.D**2, self.d]
        return self.core_tensors.view(full_shape)

    def _contract_batch_input(self, batch_input):
        """
        Contract batch of input data with MPS cores and return 
        the resulting batch of matrices as a torch tensor
        """
        size, D, d = self.size, self.D, self.d
        batch_size = batch_input.shape[0]

        # Massage the shape of the core tensors and data 
        # vectors, then batch multiply them together
        core_mats = self._tensors_as_mats()
        # core_mats = [core.as_mat() for core in self.cores]
        core_mats = core_mats.unsqueeze(0)
        core_mats = core_mats.expand([batch_size, size, D*D, d])

        core_mats = core_mats.contiguous()
        batch_input = batch_input.contiguous()

        core_mats = core_mats.view(batch_size*size, D*D, d)
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
        
        embed_data = [self.embedding_map(dat).unsqueeze(0) 
                      for dat in batch_input]
        embed_data = torch.stack(embed_data, 0)

        return embed_data.view([batch_size, size, self.d])

    def forward(self, batch_input):
        """
        Evaluate our classifier on a batch of input data 
        and return a vector of logit scores over labels

        batch_input must be formatted as a torch tensor of size 
        [batch_size, input_size], where every entry lies in
        the interval [0, 1]
        """
        size, D, d = self.size, self.D, self.d
        num_labels = self.num_labels
        num_layers = size.bit_length() - 1

        batch_size = batch_input.shape[0]
        input_size = batch_input.shape[1]

        # Get local embedded images of input pixels
        batch_input = self._embed_batch_input(batch_input)

        if input_size != size:
            raise ValueError(("len(batch_input) = {0}, but "
                  "needs to be {1}").format(input_size, size))

        # Inputs aren't trainable
        batch_input.requires_grad = False

        # Get a batch of matrices, which we will multiply 
        # together to get our batch of logit scores
        batch_mats = self._contract_batch_input(batch_input)
        batch_scores = torch.zeros([batch_size, num_labels])

        for i in range(batch_size):
            mats = batch_mats[i]
            new_size = size

            # Iteratively multiply nearest neighboring pairs of matrices
            # until we've reduced this to a single product matrix
            for j in range(num_layers):
                odd_size = (new_size % 2) == 1
                new_size = new_size // 2
                
                # If our size is odd, leave the last matrix aside
                # and multiply together all other matrix pairs
                if odd_size:
                    mats = mats[:-1]
                    lone_mat = mats[-1].unsqueeze(0)

                mats = mats.view([2, new_size, D, D])
                mats = torch.bmm(mats[0], mats[1])

                if odd_size:
                    mats = torch.cat([mats, lone_mat], 0)
                    new_size = new_size + 1
            
            # Multiply our product matrix with the label tensor
            label_tensor = self.label_tensor.permute([2,0,1])
            mat_stack = mats[0].unsqueeze(0).expand([num_labels,D,D])
            logit_tensor = torch.bmm(mat_stack, label_tensor)
            
            # Take the partial trace over the bond indices, 
            # which leaves us with an output logit score
            logit_tensor = logit_tensor.view([num_labels, D*D])
            eye_vec = torch.eye(D).view(D*D)
            batch_scores[i] = torch.mv(logit_tensor, eye_vec)

        return batch_scores

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
    start_point = time.time()
    forward_time = 0.
    back_time = 0.
    diag_time = 0.
    torch.manual_seed(23)

    if len(sys.argv) == 1:
        want_gpu = True
    else:
        want_gpu = False if sys.argv[1]=='no_gpu' else True

    use_gpu = want_gpu and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print("Using device:", device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor'
                  if use_gpu else 'torch.FloatTensor')

    # Experiment settings
    length = 14
    size = length**2
    num_train_imgs = 3*(2**(length-1)-1)
    num_test_imgs = 1*(2**(length-1)-1)
    D = 50
    d = 2
    num_labels = 2
    epochs = 5
    batch_size = 100            # Size of minibatches
    loss_type = 'crossentropy'  # Either 'mse' or 'crossentropy'
    args = {'bc': 'open',
            'weight_init_method': 'random_eye',
            'weight_init_scale': 0.01}
    batches = num_train_imgs // batch_size
    if batches == 0:
        raise ValueError("Batch size < # of training images")

    # Build the training environment and load data
    train_imgs,train_lbls,test_imgs,test_lbls = load_HV_data(length)
    train_imgs = train_imgs.view([-1, size]).to(device)
    test_imgs = test_imgs.view([-1, size]).to(device)
    label_vecs = convert_to_onehot(train_lbls, d)
    print("Training on {0} images of size "
          "{1}x{1}".format(num_train_imgs, length))
    print()

    classifier = MPSClassifier(size=size, D=D, d=d,
                               num_labels=num_labels, args=args)

    if use_gpu:
        classifier = classifier.cuda(device=device)
        train_imgs = train_imgs.cuda(device=device)
        train_lbls = train_lbls.cuda(device=device)
        test_imgs = test_imgs.cuda(device=device)
        test_lbls = test_lbls.cuda(device=device)

    if loss_type == 'mse':
        loss_f = nn.MSELoss()
    elif loss_type == 'crossentropy':
        loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1E-3)
    init_point = time.time()
    
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

    for epoch in range(epochs):
        
        av_loss = 0.
        for batch in range(batches):
            # Get data for this batch
            batch_imgs = train_imgs[batch*batch_size:
                                   (batch+1)*batch_size]
            batch_lbls = train_lbls[batch*batch_size:
                                   (batch+1)*batch_size]
            batch_vecs = convert_to_onehot(batch_lbls, d)

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

        diag_point = time.time()
        train_correct = classifier.num_correct(train_imgs, train_lbls)
        train_acc = train_correct / num_train_imgs
        test_correct = classifier.num_correct(test_imgs, test_lbls)
        test_acc = test_correct / num_test_imgs
        diag_time += time.time() - diag_point

        # Print the training information
        if epoch % 1 == 0:
            print("### epoch", epoch, "###")
            print("average loss = {:.4e}".format(loss.item()))
            print("training accuracy = {:.4f}".format(train_acc.item()))
            print("test accuracy = {:.4f}".format(test_acc.item()))
            print()

        # Shuffle our training data for the next epoch
        train_imgs,train_lbls = joint_shuffle(train_imgs,train_lbls)

    # Get relevant runtimes
    end_point = time.time()
    init_time = init_point - start_point
    train_time = end_point - init_point
    run_time = end_point - start_point

    print("Loading time  = {0:.2f} sec".format(init_time))
    print("Forward time  = {0:.2f} sec".format(forward_time))
    print("Backprop time = {0:.2f} sec".format(back_time))
    print("Error time    = {0:.2f} sec".format(diag_time))
    print("---------------------------")
    print("Training time = {0:.2f} sec".format(train_time))
    print("Total runtime = {0:.2f} sec".format(run_time))