#!/usr/bin/env python3
import torch
import sys

sys.path.append('/home/jemis/torch_mps')
from modules import MPS

batch_size = 11
input_size = 21
output_dim = 4
bond_dim = 5
threshold = 3 * batch_size

input_data = torch.randn([batch_size, input_size])

# For both open and periodic boundary conditions, place the label site in 
# different locations and check that the basic behavior is correct
for bc in [False, True]:
    for num_params, label_site in [([3,5], None), ([2,3], 0), ([2,4,5], 1), 
                                   ([2,3], input_size), ([2,4], input_size-1)]:
        # MPS(input_size, output_dim, bond_dim, d=2, label_site=None,
        #     periodic_bc=False, parallel_eval=False, dynamic_mode=False, 
        #     cutoff=1e-10, threshold=1000)
        mps_module = MPS(input_size, output_dim, bond_dim, periodic_bc=bc, 
                         label_site=label_site, dynamic_mode=True, 
                         threshold=threshold)
        assert len(list(mps_module.parameters())) in num_params
        assert mps_module.linear_region.offset == 0

        for _ in range(6):
            output = mps_module(input_data)
            assert list(output.shape) == [batch_size, output_dim]

        # At this point we should have flipped our offset from 0 to 1, but are
        # on the threshold so that the next call will flip offset back to 0
        assert len(list(mps_module.parameters())) in num_params
        assert mps_module.linear_region.offset == 1
        
        output = mps_module(input_data)
        assert list(output.shape) == [batch_size, output_dim]
        assert mps_module.linear_region.offset == 0
        assert len(list(mps_module.parameters())) in num_params
