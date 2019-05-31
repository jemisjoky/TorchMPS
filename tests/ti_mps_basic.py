#!/usr/bin/env python3
import torch
import sys
from random import randint

sys.path.append('..')
from torchmps import TI_MPS

feature_dim = 3
batch_size  = 7
seq_length  = 4
output_dim  = 2
bond_dim    = 5

# Generate a random batch input tensor
batch_input = torch.randn([batch_size, seq_length, feature_dim])

# Generate a random list of input sequences with different lengths
seq_input = [torch.randn([randint(1,seq_length), feature_dim]) for 
             _ in range(batch_size)]

for parallel_eval in [False, True]:
    mps_module = TI_MPS(feature_dim, output_dim, bond_dim, parallel_eval)

    # Feed both types of input to our MPS, and check that the outputs have the 
    # correct batch size and output dimension
    batch_output = mps_module(batch_input)
    seq_output = mps_module(seq_input)

    assert list(batch_output.shape) == [batch_size, output_dim]
    assert list(seq_output.shape) == [batch_size, output_dim]

    # Sum the outputs and generate gradients
    seq_sum = torch.sum(seq_output)
    seq_sum.backward()

    # Get the parameter tensors and check that all of them have gradients
    for tensor in list(mps_module.parameters()):
        assert tensor.grad is not None