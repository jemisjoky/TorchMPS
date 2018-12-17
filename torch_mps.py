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
        self.label_site = size // 2

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

        label_tensor = self.label_tensor.permute([2, 0, 1])
        label_tensor = label_tensor.unsqueeze(0).expand(
                                    [batch_size, num_labels, D, D])
        label_tensor = label_tensor.contiguous().view(
                                    [batch_size*num_labels, D, D])

        # And here's the actual contraction with the label tensor
        label_tensor = torch.bmm(left_stack, label_tensor)
        label_tensor = torch.bmm(label_tensor, right_stack)
        
        # Finally, taking the partial trace over the bond indices
        # leaves us with a batch of output (logit) scores.
        # (FYI, eye_vecs is just a convenient way of doing a partial
        #  trace over the bond dimension using batch multiplication)
        label_tensor = label_tensor.view([batch_size, num_labels, D*D])
        eye_vecs = torch.eye(D).view(D*D).unsqueeze(0).expand(
                                [batch_size, D*D]).unsqueeze(2)
        batch_scores = torch.bmm(label_tensor, eye_vecs)

        return batch_scores.squeeze()

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