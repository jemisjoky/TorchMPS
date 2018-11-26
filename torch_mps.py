#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from misc import load_HV_data, convert_to_onehot

# CRUCIAL
# Compute loss function and write code for training the whole thing
# Make some data and test the whole thing out!!!

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
        # Type of boundary conditions for our MPS, either 
        # 'open' or 'periodic' (default 'periodic')
        if 'bc' in args.keys():
            self.bc = args['bc']
        else:
            self.bc = 'periodic'

        # Weights and information about all MPS core tensors, 
        # initialized randomly
        full_shape = [size, D, D, d]
        self.core_tensors = nn.Parameter(torch.randn(full_shape))
        single_shape = torch.Tensor([D, D, d]).unsqueeze(0)
        self.core_shapes = single_shape.expand([size, 3])     

        # The output label core, whose non-bond index gives
        # the logit score for each classification label
        label_shape = [D, D, num_labels]
        self.label_tensor = nn.Parameter(torch.randn(label_shape))
        self.label_shape = torch.Tensor(label_shape)

        # Location of the label index
        # TODO: Make this work by changing forward() appropriately
        self.label_location = size // 2
        
    def tensors_as_mats(self):
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
        core_mats = self.tensors_as_mats()
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
            mat_stack = mats[0].unsqueeze(0).expand([num_labels, D, D])
            logit_tensor = torch.bmm(mat_stack, label_tensor)
            
            # Take the partial trace over the bond indices, 
            # which leaves us with an output logit score
            logit_tensor = logit_tensor.view([num_labels, D*D])
            eye_vec = torch.eye(D).view(D*D)
            batch_scores[i] = torch.mv(logit_tensor, eye_vec)

        return batch_scores


### TESTING STUFF BELOW ###
if __name__ == "__main__":
    torch.manual_seed(23)

    length = 3
    batch_size = 4*length
    size = length**2
    D = 2
    d = 2
    num_labels = 2

    classifier = MPSClassifier(size=size, D=D, d=d,
                                  num_labels=num_labels, bc='open')

    images, labels = load_HV_data(length)
    images = images.view([-1,size])
    label_vecs = convert_to_onehot(labels, d)

    loss_f = nn.MSELoss()
    optimi = torch.optim.Adam(classifier.parameters(), lr=1E-3)
    epochs = 1

    # Compute the training information
    scores = classifier(images)
    predictions = torch.argmax(scores,1)
    loss = loss_f(predictions, labels)
    num_correct = torch.sum(torch.eq(predictions, labels)).float()
    accuracy = num_correct / batch_size

    for epoch in range(epochs):
        # Print the training information
        print("accuracy =", accuracy)
        print("epoch", epoch)
        print("loss =", loss.item())
        
        # Get the gradients and step
        optimi.zero_grad()
        loss.backward()
        optimi.step()

        # Shuffle to be safe
        images, labels = random.shuffle(zip(images, labels))

        # Compute the training information, repeat
        scores = classifier(images)
        predictions = torch.argmax(scores,1)
        loss = loss_f(predictions, labels)
        num_correct = torch.sum(torch.eq(predictions, labels)).float()
        accuracy = num_correct / batch_size



    # print("label_vecs =", label_vecs)
    # print("scores =", scores)
    # print("labels =     ", labels)
    # print("predictions =", predictions)
    # print("predictions==labels =", torch.eq(predictions, labels))