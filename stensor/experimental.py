### Workspace for testing out different functionality ###

from stensor import *

# Possible rescaling functions
def rescale_1_(stens):
    """Simple in-place one-norm based rescaling"""
    nb, nt = stens.num_batch, len(stens.shape)
    tens_scale = torch.sum(stens.data.abs(), dim=list(range(nb, nt)), keepdim=True)
    log_shift = torch.floor(TARGET_SCALE(stens.shape, nb) - torch.log2(tens_scale))
    stens.data *= 2**log_shift
    stens.scale -= log_shift.view_as(stens.scale)

def rescale_1(stens):
    """Simple one-norm based rescaling"""
    nb, nt = stens.num_batch, len(stens.shape)
    tens_scale = torch.sum(stens.data.abs(), dim=list(range(nb, nt)), keepdim=True)
    log_shift = torch.floor(TARGET_SCALE(stens.shape, nb) - torch.log2(tens_scale))
    return STensor(stens.data*(2**log_shift), 
                   stens.scale-log_shift.view_as(stens.scale))

def rescale_2_(stens):
    """Simple in-place two-norm based rescaling"""
    nb, nt = stens.num_batch, len(stens.shape)
    tens_scale = torch.sum(stens.data.abs()**2, dim=list(range(nb, nt)), keepdim=True)
    log_shift = torch.floor(TARGET_SCALE(stens.shape, nb) - torch.log2(tens_scale)/2)
    stens.data *= 2**log_shift
    stens.scale -= log_shift.view_as(stens.scale)

def rescale_2(stens):
    """Simple two-norm based rescaling"""
    nb, nt = stens.num_batch, len(stens.shape)
    tens_scale = torch.sum(stens.data.abs()**2, dim=list(range(nb, nt)), keepdim=True)
    log_shift = torch.floor(TARGET_SCALE(stens.shape, nb) - torch.log2(tens_scale)/2)
    return STensor(stens.data*(2**log_shift), 
                   stens.scale-log_shift.view_as(stens.scale))

def rescale_1p_(stens):
    """Simple in-place Pytorch one-norm based rescaling"""
    nd, nb, bs = stens.num_data, stens.num_batch, stens.batch_shape
    flat_shape, long_shape = bs + (-1,), bs + (1,)*nd
    tens_scale = torch.norm(stens.data.view(flat_shape), dim=-1, p=1)
    log_shift = torch.floor(TARGET_SCALE(stens.shape, nb) - torch.log2(tens_scale))
    stens.data *= 2**log_shift.view(long_shape)
    stens.scale -= log_shift

def rescale_1p(stens):
    """Simple Pytorch one-norm based rescaling"""
    nd, nb, bs = stens.num_data, stens.num_batch, stens.batch_shape
    flat_shape, long_shape = bs + (-1,), bs + (1,)*nd
    tens_scale = torch.norm(stens.data.view(flat_shape), dim=-1, p=1)
    log_shift = torch.floor(TARGET_SCALE(stens.shape, nb) - torch.log2(tens_scale))
    return STensor(stens.data*(2**log_shift.view(long_shape)), 
                   stens.scale-log_shift)

def rescale_2p_(stens):
    """Simple in-place Pytorch two-norm based rescaling"""
    nd, nb, bs = stens.num_data, stens.num_batch, stens.batch_shape
    flat_shape, long_shape = bs + (-1,), bs + (1,)*nd
    tens_scale = torch.norm(stens.data.view(flat_shape), dim=-1)
    log_shift = torch.floor(TARGET_SCALE(stens.shape, nb) - torch.log2(tens_scale))
    stens.data *= 2**log_shift.view(long_shape)
    stens.scale -= log_shift

def rescale_2p(stens):
    """Simple Pytorch two-norm based rescaling"""
    nd, nb, bs = stens.num_data, stens.num_batch, stens.batch_shape
    flat_shape, long_shape = bs + (-1,), bs + (1,)*nd
    tens_scale = torch.norm(stens.data.view(flat_shape), dim=-1)
    log_shift = torch.floor(TARGET_SCALE(stens.shape, nb) - torch.log2(tens_scale))
    return STensor(stens.data*(2**log_shift.view(long_shape)), 
                   stens.scale-log_shift)

# # Testing out how to add in-place methods externally
# class MyClass:
#     def __init__(self, data):
#         self.data = data
# def increase_data(self):
#     self.data = self.data + 1
# setattr(MyClass, 'increase_data', increase_data)

if __name__ == '__main__':
    # myc = MyClass(0)
    # print(myc.data)
    # myc.increase_data()
    # print(myc.data)


    # # Define tensors and stensors
    # torch.manual_seed(0)
    # scale_tensor = torch.zeros((100,))
    # small_tensor = torch.randn((100, 2, 2, 2))
    # med_tensor = torch.randn((100, 5, 7, 10))
    # big_tensor = torch.randn((100, 10, 20, 5, 4))
    # all_tensors = [small_tensor, med_tensor, big_tensor]
    # small_st, med_st, big_st = [STensor(t, scale_tensor) for t in all_tensors]

    # # Fill the cache
    # for t in all_tensors: TARGET_SCALE(t.shape, 1)

    # # Benchmarking experiments for rescaling
    # import timeit
    # from itertools import product

    # for tup, OP in product(zip(['sml', 'med', 'lrg'], 
    #                            [small_st, med_st, big_st]), 
    #                        ['rescale_1', 'rescale_1_', 
    #                         'rescale_2', 'rescale_2_', 
    #                         'rescale_1p', 'rescale_1p_', 
    #                         'rescale_2p', 'rescale_2p_', 
    #                         ]):
    #     size, st = tup
    #     command = f"{OP}(st)"
    #     loading = f"from __main__ import {OP}, st"
    #     print(f"({size}, {OP}): {timeit.timeit(command, setup=loading, number=10000)}")