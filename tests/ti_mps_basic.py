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

# for parallel_eval in [False, True]:
for parallel_eval in [False]:
    mps_module = TI_MPS(feature_dim, output_dim, bond_dim, parallel_eval)

    # Feed both types of input to our MPS, and check that the outputs have the 
    # correct batch size and output dimension
    batch_output = mps_module(batch_input)
    seq_output = mps_module(seq_input)

    assert list(batch_output.shape) == [batch_size, output_dim]
    assert list(seq_output.shape) == [batch_size, output_dim]

    # Grab the core tensor from the MPS
    param_gen = mps_module.parameters()
    core_tensor = next(param_gen)
    
    # There should be exactly one tensor in the TI_MPS parameters
    try:
        next(param_gen)
        assert False
    except StopIteration:
        pass

    # Sum the outputs and generate gradients
    seq_sum = torch.sum(seq_output)
    seq_sum.backward()

    # The gradient with respect to the core_tensor should be defined
    assert core_tensor.grad is not None