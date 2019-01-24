class Reducible:
    """
    An object which can be put on a line and contracted with other reducibles

    The purpose of this class is making my MPS contraction code more modular,
    while also allowing outlets for improving the speed and parallelism of the
    code. I'm imagining a small number of subclasses here, each of which
    represents a certain kind of reduction or evaluation operation:

    * MatRegion, a contiguous region of matrices which can be multiplied 
      together in parallel (reduced) to get a single matrix, or evaluated 
      serially using iterated matrix-vector multiplication
    * SingleMat, a single matrix
    * SingleVec, a single vector
    * LabelCore, a single MPS core of shape [out_dim, left_D, right_D]

    Every one of these objects has a batch dimension as its first index

    WHAT'S NEEDED FOR THIS CLASS?
    (1) The defining tensor which holds our intermediate data
    (2) core_shape, the shape of the tensor *excluding its batch dimension*
    (3) bond_indices = [left_ind, right_ind], the location of the left and 
        right bond indices within core_shape. If we don't have a given bond 
        (e.g. Scalar or Vec), then left_ind and/or right_ind is None
    (4) l_mult(), a method which multiplies another reducible by our reducible
        on the left 
    (5) r_mult(), a method which multiplies another reducible by our reducible
        on the right
    (6) reduce(), a method which converts our reducible into a simpler form.
        reduce() is non-trivial only when we have some composite reducible, for 
        which our reduction operation lets us convert to an atomic reducible.
        reduce() is only used when we have good parallel computing resources
    """
    def __init__(self, tensor, bond_indices):
        assert len(tensor.shape) == 1 + len(core_shape)
        assert tensor.shape[1:] == core_shape

        self.tensor = tensor
        self.core_shape = core_shape
        self.left_ind, self.right_ind = bond_indices

    def l_mult(self, right_reducible):
        """
        
        """
        raise NotImplementedError

    def r_mult(self, left_reducible):
        """
        
        """
        raise NotImplementedError

    def reduce(self):
        """
        
        """
        return self


class MatRegion(Reducible):
    """
    A contiguous collection of matrices which are multiplied together

    The input tensor defining our MatRegion must have shape 
    [batch_size, num_mats, D, D], so that core_shape = [num_mats, D, D]
    """
    def __init__(self, tensor):
        core_shape = tensor.shape
        len_shape = len(core_shape)
        
        if len(core_shape) != 3 or core_shape[-2] != core_shape[-1]:
            raise ValueError("tensor must have shape "
                             "[batch_size, num_mats, D, D]")
        
        core_shape = core_shape[1:]
        bond_indices = [1, 2]

        # Initialize a reducible with our parameters
        super().__init__(tensor, core_shape, bond_indices)

    def l_mult(self, right_reducible):
        """
        Iteratively multiply an input vector on the left with all our matrices 
        """
        assert isinstance(right_reducible, Reducible)

        # The input must be an instance of SingleVec
        if not isinstance(right_reducible, SingleVec):
            raise UnsupportedReducible()




    def reduce(self):
        """
        Multiplies together all matrices and returns residual SingleMat

        Iterated batch multiplication is used to evaluate the full matrix 
        product in depth log(num_mats)
        """
        mats = self.tensor
        core_shape = self.core_shape
        batch_size = mats.size(0)
        size, D = core_shape[:2]

        # Iteratively multiply pairs of matrices until there is only one
        while size > 1:
            odd_size = (size % 2 == 1)
            half_size = size // 2
            nice_size = 2 * half_size
        
            even_mats = mats[:, 0:nice_size:2].contiguous()
            odd_mats = mats[:, 1:nice_size:2].contiguous()
            leftover = mats[:, nice_size:]

            # Could use einsum here, but this is likely faster
            even_mats = even_mats.view([batch_size * half_size, D, D])
            odd_mats = odd_mats.view([batch_size * half_size, D, D])
            mats = torch.cat([torch.bmm(even_mats, odd_mats), leftover], 1)

            size = half_size + int(odd_size)

        core_shape = [core_shape[0]] + core_shape[-2:]
        mats.squeeze(1)

        # Since we only have a single matrix, wrap it as a SingleMat
        return SingleMat(mats)




class Scalar(Reducible):
    """
    A batch of scalars
    """
    def __init__(self, scalar):
        core_shape = scalar.shape
        
        if len(core_shape) != 1:
            raise ValueError("input scalar must be a torch tensor with shape "
                             "[batch_size] (or [1] for no batch)")
        
        # Initialize a reducible with our parameters
        super().__init__(scalar, core_shape, [None, None])

    def l_mult(self, reducible):
        scalar = self.tensor
        has_batch

        tensor = reducible.tensor


    def r_mult(self, reducible):
        # Scalar multiplication is commutative, so...
        return self.l_mult(reducible)


class UnsupportedReducible(Exception):
    """
    Indicates that l_mult or r_mult isn't supported on the input Reducible
    """
    pass