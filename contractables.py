class Contractable:
    """
    An object which can be contracted with other contractables

    The purpose of this class is making my MPS contraction code more modular,
    while also allowing outlets for improving the speed and parallelism of the
    code. I'm imagining a small number of subclasses here, each of which
    represents contraction, reduction, and/or evaluation operations:

    * MatRegion, a contiguous region of matrices which can be multiplied 
      together in parallel (reduced) to get a single matrix, or contracted 
      serially using iterated matrix-vector multiplication
    * SingleMat, a single matrix
    * SingleVec, a single vector
    * LabelCore, a single MPS core of shape [out_dim, left_D, right_D]

    Every one of these objects has a batch dimension as its first index

    WHAT'S NEEDED FOR THIS CLASS?
    (1) The defining tensor which holds our intermediate data
    (3) bond_inds, the location of the bond indices in our tensor's shape
        which can be contracted by the contraction engine
    (4) l_mult(), a method which multiplies another contractable by our 
        contractable on the left 
    (5) r_mult(), a method which multiplies another contractable by our 
        contractable on the right
    (6) reduce(), a method which converts our contractable into a simpler form.
        reduce() is non-trivial only when we have some composite contractable,  
        which our reduction operation lets us convert to an atomic contractable.
        reduce() only serves to make our code more parallelizable
    """
    # The batch size for our contraction engine, which is initialized when we 
    # receive the first input tensor with a batch index
    # Shared by all Contractable instances
    _bs = None

    def __init__(self, tensor, bond_inds):
        # Set the defining instance attributes
        self.tensor = tensor
        self.bond_inds = bond_inds
        batch_size = tensor.size(0)

        # Set the global batch size if it hasn't been set yet
        if not Contractable._bs:
            Contractable._bs = batch_size
        else:
            global_bs = Contractable._bs
            if global_bs != batch_size:
                raise RuntimeError(f"Batch size previously set to {global_bs}"
                                    ", but input tensor has batch size "
                                   f"{batch_size}. Try calling "
                                    "Contractable.reset_batch_size() first")

    def l_mult(self, right_contractable):
        """
        
        """
        raise UnsupportedContractable

    def r_mult(self, left_contractable):
        """
        
        """
        raise UnsupportedContractable

    def reduce(self):
        """
        
        """
        return self

    def reset_batch_size(self):
        """

        """
        Contractable._bs = None


class MatRegion(Contractable):
    """
    A contiguous collection of matrices which are multiplied together

    The input tensor defining our MatRegion must have shape 
    [batch_size, num_mats, D, D], or [num_mats, D, D] when the global batch
    size is already known
    """
    def __init__(self, mats):
        # Check the input shape
        shape = list(mats.shape)
        if len(shape) not in [3, 4] or shape[-2] != shape[-1]:
            raise ValueError("MatRegion tensors must have shape "
                             "[batch_size, num_mats, D, D], or else [num_mats,"
                             " D, D] if batch size has already been set")

        # Add a batch dimension if needed
        if len(shape) == 3:
            if Contractable._bs:
                mats = mats.unsqueeze(0).expand([Contractable._bs] + shape)
            else:
                raise RuntimeError("Input tensor has no batch dimension, and "
                                   "no previously set batch size")

        # Initialize our contractable
        assert len(mats.size) == 4
        super().__init__(mats, bond_inds=[2, 3])

    def l_mult(self, right_vec):
        """
        Iteratively multiply an input vector on the left with all our matrices 
        """
        # The input must be an instance of SingleVec
        if not isinstance(right_vec, SingleVec):
            raise UnsupportedContractable

        mats = self.tensor
        num_mats = mats.size(1)
        batch_size = mats.size(0)

        # Load our vector and matrix batches
        vec = right_vec.tensor.unsqueeze(2)
        mat_list = torch.chunk(mats, num_mats, 1)[::-1]

        # Do the repeated matrix-vector multiplications in right-to-left order
        for mat in mat_list:
            vec = torch.bmm(mat, vec)

        # Since we only have a single vector, wrap it as a SingleVec
        return SingleVec(vec.squeeze(2))

    def r_mult(self, left_vec):
        """
        Iteratively multiply an input vector on the right with all our matrices 
        """
        # The input must be an instance of SingleVec
        if not isinstance(left_vec, SingleVec):
            raise UnsupportedContractable

        mats = self.tensor
        num_mats = mats.size(1)
        batch_size = mats.size(0)

        # Load our vector and matrix batches
        vec = left_vec.tensor.unsqueeze(1)
        mat_list = torch.chunk(mats, num_mats, 1)

        # Do the repeated matrix-vector multiplications in right-to-left order
        for mat in mat_list:
            vec = torch.bmm(vec, mat)

        # Since we only have a single vector, wrap it as a SingleVec
        return SingleVec(vec.squeeze(1))

    def reduce(self):
        """
        Multiplies together all matrices and returns residual SingleMat

        Iterated batch multiplication is used to evaluate the full matrix 
        product in depth log(num_mats)
        """
        mats = self.tensor
        shape = list(mats.shape)
        batch_size = mats.size(0)
        size, D = shape[1:3]

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

        # Since we only have a single matrix, wrap it as a SingleMat
        return SingleMat(mats.squeeze(1))

class SingleMat(Contractable):
    """
    A batch of matrices associated with a single location in our MPS
    """
    def __init__(self, mat):
        # Check the input shape
        shape = list(mat.shape)
        if len(shape) not in [2, 3]:
            raise ValueError("SingleMat tensors must have shape [batch_size, "
                             "D_l, D_r], or else [D_l, D_r] if batch size "
                             "has already been set")

        # Add a batch dimension if needed
        if len(shape) == 2:
            if Contractable._bs:
                mats = mats.unsqueeze(0).expand([Contractable._bs] + shape)
            else:
                raise RuntimeError("Input tensor has no batch dimension, and "
                                   "no previously set batch size")

        # Initialize our contractable
        assert len(mats.size) == 3
        super().__init__(mats, bond_inds=[1, 2])

    def l_mult(self, right_contractable):
        """
        Multiply an input vector or matrix on the left by our matrix
        """
        # The input must be an instance of SingleVec or SingleMat
        if not isinstance(right_contractable, (SingleVec, SingleMat)):
            raise UnsupportedContractable
        is_vec = isinstance(right_contractable, SingleVec)

        left_mat = self.tensor
        right_obj = right_contractable.tensor
        batch_size = left_mat.size(0)

        # Add an extra dimension if we have an input vector
        if is_vec:
            right_obj.unsqueeze_(2)

        # Do the batch multiplication
        out_obj = torch.bmm(left_mat, right_vec)

        # Wrap our output in the appropriate Contractable constructor
        if is_vec:
            return SingleVec(out_obj.squeeze(2))
        else:
            return SingleMat(out_obj)

    def r_mult(self, left_vec):
        """
        Multiply an input vector on the right with our matrix
        """
        # The input must be an instance of SingleVec
        if not isinstance(left_vec, SingleVec):
            raise UnsupportedContractable

        mat = self.tensor
        left_vec = left_vec.tensor.unsqueeze(1)
        batch_size = mat.size(0)

        # Do the batch multiplication
        vec = torch.bmm(left_vec, mat)

        # Since we only have a single vector, wrap it as a SingleVec
        return SingleVec(vec.squeeze(1))

class SingleVec(Contractable):
    """
    A batch of vectors associated with an edge of our MPS
    """
    def __init__(self, vec):
        # Check the input shape
        shape = list(vec.shape)
        if len(shape) not in [1, 2]:
            raise ValueError("SingleVec tensors must have shape "
                             "[batch_size, D], or else [D] if batch size "
                             "has already been set")

        # Add a batch dimension if needed
        if len(shape) == 1:
            if Contractable._bs:
                mats = mats.unsqueeze(0).expand([Contractable._bs] + shape)
            else:
                raise RuntimeError("Input tensor has no batch dimension, and "
                                   "no previously set batch size")

        # Initialize our contractable
        assert len(mats.size) == 2
        super().__init__(mats, bond_inds=[1])

    def l_mult(self, right_vec):
        """
        Take the inner product of our vector with another vector
        """
        # The input must be an instance of SingleVec
        if not isinstance(right_vec, SingleVec):
            raise UnsupportedContractable

        left_vec = self.tensor.unsqueeze(1)
        right_vec = right_vec.tensor.unsqueeze(2)
        batch_size = left_vec.size(0)

        # Do the batch inner product
        scalar = torch.bmm(left_vec, right_vec).view([batch_size])

        # Since we only have a single scalar, wrap it as a Scalar
        return Scalar(scalar)

class Scalar(Contractable):
    """
    A batch of scalars
    """
    def __init__(self, scalar):
        # Add dummy dimension if we have a torch scalar
        shape = list(scalar.shape)
        if shape is []:
            scalar = scalar.view([1])
            shape = [1]
            
        # Check the input shape
        if len(shape) != 1:
            raise ValueError("input scalar must be a torch tensor with shape "
                             "[batch_size], or [] or [1] if batch size has "
                             "been set")

        # Add a batch dimension if needed
        if shape[0] == 1:
            if Contractable._bs:
                batch_size = Contractable._bs
                scalar = scalar.expand([batch_size])
            else:
                raise RuntimeError("Input scalar has no batch dimension, and "
                                   "no previously set batch size")

        # Initialize our contractable
        super().__init__(scalar, bond_inds=[])

    def l_mult(self, right_contractable):
        """
        
        """
        raise UnsupportedContractable

    def r_mult(self, left_contractable):
        """
        
        """
        raise UnsupportedContractable

class UnsupportedContractable(Exception):
    """
    Indicates that l_mult or r_mult isn't supported on the input Contractable

    Used to signal to our contraction engine that we want to use the x_mult
    operation defined in the other contractable
    """
    pass