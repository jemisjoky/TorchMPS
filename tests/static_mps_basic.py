#!/usr/bin/env python3
import torch
import sys

sys.path.append('/home/jemis/torch_mps')
from realizables import MPS

batch_size = 11
input_size = 21
output_dim = 4
bond_dim = 5

input_data = torch.randn([batch_size, input_size])

# Place the label site in different locations and check that the basic
# behavior is correct
for num_params, label_site in [(3, None), (2, 0), (2, input_size)]:
    mps_module = MPS(input_size, output_dim, bond_dim, 
                     d=2, label_site=label_site)
    assert len(list(mps_module.parameters())) == num_params

    output = mps_module(input_data)
    assert list(output.shape) == [batch_size, output_dim]
