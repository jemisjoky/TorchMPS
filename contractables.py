"""
TODO:
    * Write multiplication operations for OutputCore

    * Write docstrings for everything
    
"""
from reducibles import Reducible 
from realizables import Realizable

class Contractable(Realizable):
    """
    Container for tensors with labeled indices and a global batch size

    The labels for our indices give some high-level knowledge of the tensor
    layout, and permit the contraction of pairs of indices in a more 
    systematic manner. However, much of the actual heavy lifting is done 
    through specific contraction routines in different subclasses

    Attributes:
        tensor (Tensor):    A Pytorch tensor whose first index must be a batch
                            index. Sub-classes of Contractable may put other 
                            restrictions on tensor
        bonds (dict):       A dictionary whose keys are names for bond indices, 
                            and whose values point to the index of `tensor` 
                            which holds that index
        _batch_size (int):  The batch size associated with all Contractables.
                            This is thus shared between all instances as a 
                            class attribute
    """
    # The key in `bonds` which points to our batch index
    batch_bond = 'B'
    # The global batch size
    _batch_size = None

    def __init__(self, tensor, bonds):
        # Get the batch size and add it to our bond dict if it isn't there
        batch_size = tensor.size(0)
        if batch_bond not in bonds:
            bonds[batch_bond] = 0

        # Set the global batch size if it hasn't been set yet
        if not Contractable._batch_size:
            Contractable._batch_size = batch_size
        else:
            global_bs = Contractable._batch_size
            if global_bs != batch_size:
                raise RuntimeError(f"Batch size previously set to {global_bs}"
                                    ", but input tensor has batch size "
                                   f"{batch_size}. Try calling "
                                    "Contractable.unset_batch_size() first")
        
        # Set the defining attributes of our Contractable
        self.tensor = tensor
        self.bonds = bonds

    def set_batch_size(self, batch_size):
        """

        """
        Contractable._batch_size = batch_size

    def unset_batch_size(self):
        """

        """
        Contractable._batch_size = None

    def reduce(self):
        """
        A Reducable is just something which can be reduced to a Contractable,
        and any Contractable can trivially do so by returning itself
        """
        return self

class LinearContractable(Contractable):
    """
    An object which can be contracted with other contractables on a line

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
    """
    # The keys in `bonds` which point to the left and right bond indices
    linear_bonds = ['l', 'r']

    def __init__(self, tensor, bonds):
        super().__init__(tensor, bonds)

    def __mul__(self, right_contractable):
        """
        
        """
        raise NotImplemented

    def __rmul__(self, left_contractable):
        """
        
        """
        raise NotImplemented

class MatRegion(LinearContractable):
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
            global_bs = Contractable._batch_size
            if global_bs:
                mats = mats.unsqueeze(0).expand([global_bs] + shape)
            else:
                raise RuntimeError("Input tensor has no batch dimension, and "
                                   "no previously set batch size")

        # Initialize our contractable
        assert len(mats.size) == 4
        super().__init__(mats, bonds={'j': 1, 'l': 2, 'r': 3})

    def __mul__(self, right_vec):
        """
        Iteratively multiply an input vector on the left with all our matrices
        """
        # The input must be an instance of SingleVec
        if not isinstance(right_vec, SingleVec):
            raise NotImplemented

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

    def __rmul__(self, left_vec):
        """
        Iteratively multiply an input vector on the right with all our matrices 
        """
        # The input must be an instance of SingleVec
        if not isinstance(left_vec, SingleVec):
            raise NotImplemented

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
        Multiplies together all matrices and returns resultant SingleMat

        This method uses iterated batch multiplication to evaluate the full 
        matrix product in depth log(num_mats)
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

            # Could use einsum here, but this is probably faster
            even_mats = even_mats.view([batch_size * half_size, D, D])
            odd_mats = odd_mats.view([batch_size * half_size, D, D])
            mats = torch.cat([torch.bmm(even_mats, odd_mats), leftover], 1)

            size = half_size + int(odd_size)

        # Since we only have a single matrix, wrap it as a SingleMat
        return SingleMat(mats.squeeze(1))

class OutputCore(LinearContractable, Realizable):
    """
    A single MPS core with no input and a single output index
    """
    def __init__(self, core):
        # Check the input shape
        shape = list(core.shape)
        if len(shape) not in [3, 4]:
            raise ValueError("OutputCore tensors must have shape [batch_size, "
                             "output_dim, D_l, D_r], or else [output_dim, "
                             "D_l, D_r] if batch size has already been set")

        # Add a batch dimension if needed
        if len(shape) == 3:
            global_bs = Contractable._batch_size
            if global_bs:
                core = core.unsqueeze(0).expand([global_bs] + shape)
            else:
                raise RuntimeError("Input tensor has no batch dimension, and "
                                   "no previously set batch size")

        # Initialize our contractable
        assert len(core.size) == 3
        super().__init__(core, bonds={'o': 1, 'l': 2, 'r': 3})

    def __mul__(self, right_contractable):
        """

        """
        pass

    def __rmul__(self, left_contractable):
        """

        """
        pass


    def realize(self, input_data):
        """
        Since our OutputCore has no inputs, it's already in realized form
        """
        return self

class SingleMat(LinearContractable):
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
            global_bs = Contractable._batch_size
            if global_bs:
                mat = mat.unsqueeze(0).expand([global_bs] + shape)
            else:
                raise RuntimeError("Input tensor has no batch dimension, and "
                                   "no previously set batch size")

        # Initialize our contractable
        assert len(mat.size) == 3
        super().__init__(mat, bonds={'l': 1, 'r': 2})

    def __mul__(self, right_contractable):
        """
        Multiply an input vector or matrix on the left by our matrix
        """
        # The input must be an instance of SingleVec or SingleMat
        if not isinstance(right_contractable, (SingleVec, SingleMat)):
            raise NotImplemented
        is_vec = isinstance(right_contractable, SingleVec)

        left_mat = self.tensor
        right_obj = right_contractable.tensor
        batch_size = left_mat.size(0)

        # Add an extra dimension if we have an input vector
        if is_vec:
            right_obj.unsqueeze_(2)

        # Do the batch multiplication
        out_obj = torch.bmm(left_mat, right_obj)

        # Wrap our output in the appropriate LinearContractable constructor
        if is_vec:
            return SingleVec(out_obj.squeeze(2))
        else:
            return SingleMat(out_obj)

    def __rmul__(self, left_vec):
        """
        Multiply an input vector on the right with our matrix
        """
        # The input must be an instance of SingleVec
        if not isinstance(left_vec, SingleVec):
            raise NotImplemented

        mat = self.tensor
        left_vec = left_vec.tensor.unsqueeze(1)
        batch_size = mat.size(0)

        # Do the batch multiplication
        vec = torch.bmm(left_vec, mat)

        # Since we only have a single vector, wrap it as a SingleVec
        return SingleVec(vec.squeeze(1))

class SingleVec(LinearContractable):
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
            global_bs = Contractable._batch_size
            if global_bs:
                vec = vec.unsqueeze(0).expand([global_bs] + shape)
            else:
                raise RuntimeError("Input tensor has no batch dimension, and "
                                   "no previously set batch size")

        # Initialize our contractable
        assert len(vec.size) == 2
        super().__init__(vec, bonds={'l': 1, 'r': 1})

    def __mul__(self, right_vec):
        """
        Take the inner product of our vector with another vector
        """
        # The input must be an instance of SingleVec
        if not isinstance(right_vec, SingleVec):
            raise NotImplemented

        left_vec = self.tensor.unsqueeze(1)
        right_vec = right_vec.tensor.unsqueeze(2)
        batch_size = left_vec.size(0)

        # Do the batch inner product
        scalar = torch.bmm(left_vec, right_vec).view([batch_size])

        # Since we only have a single scalar, wrap it as a Scalar
        return Scalar(scalar)

class OutputVec(Contractable):
    """
    A batch of vectors which forms an output to another layer
    """
    def __init__(self, vec):
        # Check the input shape
        shape = list(vec.shape)
        if len(shape) not in [1, 2]:
            raise ValueError("OutputVec tensors must have shape "
                             "[batch_size, output_dim], or else [output_dim] "
                             "if batch size has already been set")

        # Add a batch dimension if needed
        if len(shape) == 1:
            global_bs = Contractable._batch_size
            if global_bs:
                vec = vec.unsqueeze(0).expand([global_bs] + shape)
            else:
                raise RuntimeError("Input tensor has no batch dimension, and "
                                   "no previously set batch size")

        # Initialize our contractable
        assert len(vec.size) == 2
        super().__init__(vec, bonds={'o': 1})


class Scalar(LinearContractable):
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
            if Contractable._batch_size:
                batch_size = Contractable._batch_size
                scalar = scalar.expand([batch_size])
            else:
                raise RuntimeError("Input scalar has no batch dimension, and "
                                   "no previously set batch size")

        # Initialize our contractable
        super().__init__(scalar, bonds={})

    def __mul__(self, right_contractable):
        """
        Multiply a contractable by our scalar and return the result
        """
        assert isinstance(right_contractable, Contractable)

        # Do the scalar multiplication of right_contractable's tensor
        tensor = right_contractable.tensor
        scalar_shape = [b if i==0 else 1 for (i, b) in enumerate(tensor.shape)]
        out_tensor = tensor * self.tensor.view(scalar_shape)

        # Wrap the result in the same class right_contractable belongs to
        contract_class = type(right_contractable)
        return contract_class(out_tensor)

    def __rmul__(self, left_contractable):
        # Scalar multiplication is commutative
        return self.__mul__(left_contractable)