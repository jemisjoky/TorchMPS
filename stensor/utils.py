import torch

def bad_conversion(stens, tensor):
    """Check if conversion to tensor led to underflow/overflow problems"""
    underflow = torch.any(torch.logical_and(tensor==0, stens.data!=0))
    overflow = torch.any(torch.logical_and(torch.isinf(tensor), 
                                           torch.isfinite(stens.data)))
    return underflow or overflow

def tupleize(dim, ndim):
    """Convert one or more dims to a tuple of non-negative indices"""
    if not isinstance(dim, tuple):
        if hasattr(dim, '__iter__'):
            dim = tuple(dim)
        else:
            dim = (dim,)
    return tuple((i if i >=0 else ndim+i) for i in dim)

def squeeze_dims(tensor, dims):
    """Squeeze multiple singleton dimensions from a tensor input"""
    shape = tensor.shape
    assert all(shape[i] == 1 for i in dims)
    new_shape = tuple(d for i, d in enumerate(shape) if i not in dims)
    return tensor.view(new_shape)

def flatten_index(idx, scale):
    """Convert index object for data tensor into one for scale tensor"""
    shape = scale.shape
    real_idx = lambda i: isinstance(i, (int, slice))
    flatdict = {int:0, slice:slice(None,None,None)}
    # Map from indexing object type to flattened indexing object
    flatmap = lambda i, n: flatdict[type(i)] if shape[n] == 1 else i
    
    # Generate scale index, different for single vs tuple indices
    if isinstance(idx, tuple):
        # Handle relative indexing coming from ... (e.x. tens[..., idx])
        if ... in idx:
            assert len([i for i in idx if i == ...]) == 1
            # How many dims the ... counts for
            eps_dims = scale.ndim - len([i for i in idx if real_idx(i)])

        # Iteratively generate the scale tensor indices
        scale_idx, n = [], 0
        for i in idx:
            if real_idx(i):
                scale_idx.append(flatmap(i, n))
                n += 1
            else:
                if i == ...: n += eps_dims
                scale_idx.append(i)

        return tuple(scale_idx)

    else:
        # Singleton indexing object
        return flatmap(idx, 0) if real_idx(idx) else idx

def scalar_scale(stens):
    return stens.scale.numel() == 1

def is_vector(stens):
    return len(stens.shape) == 1