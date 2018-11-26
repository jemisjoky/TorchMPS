#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from misc import load_HV_data, convert_to_onehot, joint_shuffle

# CRUCIAL
# Expand data, separate it into separate train and test data

# NOT AS CRUCIAL
# Allow placement of label index in any location
# Deal with non-periodic boundary conditions
# Write MPS compression function using iterated SVD's, make sure you don't backpropagate through 
#   the SVD though, sounds like this is numerically ill-conditioned
# Choose better initialization (potentially building on above compression routine)
# Deal with variable local bond dimensions
#   * This should mostly involve changes in _contract_batch_input, right?
# Allow for mini-batches of our training data

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
        single_shape = torch.Tensor([D, D, d]).unsqueeze(0)
        self.core_shapes = single_shape.expand([size, 3])     
        self.label_shape = torch.Tensor(label_shape)

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
            return torch.Tensor([1-datum, datum])

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

    def num_correct(self, batch_input, labels):
        """
        Use our classifier to predict the labels associated with a
        batch of input images, then compare with the correct labels
        and return the number of correct guesses
        """
        scores = self.forward(batch_input)
        predictions = torch.argmax(scores, 1)
        return torch.sum(torch.eq(predictions, labels)).float()


if __name__ == "__main__":
    torch.manual_seed(23)

    # Experiment settings
    length = 14
    size = length**2
    num_train_imgs = 3*(2**(length-1)-1)
    num_test_imgs = 1*(2**(length-1)-1)
    D = 5
    d = 2
    num_labels = 2
    epochs = 1000
    batch_size = 100    # Size of minibatches
    batches = num_train_imgs // batch_size
    loss_type = 'crossentropy'  # Either 'mse' or 'crossentropy'
    args = {'bc': 'open',
            'weight_init_method': 'random_eye',
            'weight_init_scale': 0.01}

    # Build the training environment and load data
    classifier = MPSClassifier(size=size, D=D, d=d,
                               num_labels=num_labels, args=args)

    train_imgs,train_lbls,test_imgs,test_lbls = load_HV_data(length)
    train_imgs = train_imgs.view([-1, size])
    test_imgs = test_imgs.view([-1, size])
    label_vecs = convert_to_onehot(train_lbls, d)

    if loss_type == 'mse':
        loss_f = nn.MSELoss()
    elif loss_type == 'crossentropy':
        loss_f = nn.CrossEntropyLoss()
    optimi = torch.optim.Adam(classifier.parameters(), lr=1E-3)

    # Compute the initial training information
    scores = classifier(train_imgs)
    if loss_type == 'mse':
        loss = loss_f(scores, label_vecs.float())
    elif loss_type == 'crossentropy':
        loss = loss_f(scores, train_lbls)
    
    train_correct = classifier.num_correct(train_imgs, train_lbls)
    train_acc = train_correct / num_train_imgs
    test_correct = classifier.num_correct(test_imgs, test_lbls)
    test_acc = test_correct / num_test_imgs

    print("Training on {0} images of size "
          "{1}x{1}".format(num_train_imgs, length))
    print()

    for epoch in range(epochs):
        # Print the training information
        if epoch % 1 == 0:
            print("### epoch", epoch, "###")
            print("loss = {:.4e}".format(loss.item()))
            print("training accuracy = {:.4f}".format(train_acc.item()))
            print("test accuracy = {:.4f}".format(test_acc.item()))
            print()
        
        for batch in range(batches):
            # Get data for this batch
            batch_imgs = train_imgs[batch*batch_size:
                                   (batch+1)*batch_size]
            batch_lbls = train_lbls[batch*batch_size:
                                   (batch+1)*batch_size]
            batch_vecs = convert_to_onehot(batch_lbls, d)

            # Compute the loss
            scores = classifier(batch_imgs)
            if loss_type == 'mse':
                loss = loss_f(scores, batch_vecs.float())
            elif loss_type == 'crossentropy':
                loss = loss_f(scores, batch_lbls)

            # Get the gradients and take an optimization step
            optimi.zero_grad()
            loss.backward()
            optimi.step()

        # Compute the training information, repeat
        scores = classifier(train_imgs)
        if loss_type == 'mse':
            loss = loss_f(scores, train_vecs.float())
        elif loss_type == 'crossentropy':
            loss = loss_f(scores, train_lbls)

        train_correct = classifier.num_correct(train_imgs, train_lbls)
        train_acc = train_correct / num_train_imgs
        test_correct = classifier.num_correct(test_imgs, test_lbls)
        test_acc = test_correct / num_test_imgs

        # Shuffle our training data for the next epoch
        train_imgs,train_lbls = joint_shuffle(train_imgs,train_lbls)