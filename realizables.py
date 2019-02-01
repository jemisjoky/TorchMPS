"""
TODO:
    * Implement other classes which are built from embeddable objects, and 
      manage the full pipeline from input to contractables to output. Objects
      from these classes are actual Pytorch modules, but the majority of their 
      'code' is written in terms of higher-level embed(), reduce(), and 
      multiplication operations.

      I already have the core of a lot of these classes, and some examples are:
        (1) PeriodicBC, which will now hold a list of embeddables and then 
            a) call embed() on each one, b) load the output list into 
            LinearRegion, and c) reduce the linear region to get an output
        (2) OpenBC, which has the same general workflow as above.

      The big open question here is best practices for routing our input. 

NOTE: To correctly register parameters, I need to set them as attributes of a 
      Pytorch Module. When I start defining Modules, I'll need to include a
      ParameterList which has a reference to everything in my sub-realizables.
"""
import torch
import torch.nn as nn
from contractables import SingleMat, MatRegion, OutputCore

def init_tensor(shape, bond_str, init_method):
    """
    Initialize a tensor of a given shape

    Args:
        shape:       The shape of our output parameter tensor

        bond_str:    The bond string describing our output parameter tensor,
                     which is used in 'random_eye' initialization method

        init_method: The method used to initialize the entries of our tensor.
                     This can be either a string, or else a tuple whose first
                     entry is an initialization method and whose second entry
                     is a scale/standard deviation parameter
    """
    # Unpack init_method if needed
    if not isinstance(init_method, str):
        init_str = init_method[0]
        std = init_method[1]
        init_method = init_str
    else:
        std = 0.01

    # Check that bond_str is properly sized and doesn't have repeat indices
    assert len(shape) == len(bond_str)
    assert len(set(bond_str)) == len(bond_str)

    if init_method not in ["random_eye", "full_random"]:
        raise ValueError(f"Unknown initialization method: {init_method}")

    if init_method == 'random_eye':
        bond_chars = ['l', 'r']
        assert all([c in bond_str for c in bond_chars])

        # Initialize our tensor as an expanded identity matrix 
        eye_shape = [shape[i] if c in bond_chars else 1
                     for i, c in enumerate(bond_str)]
        bond_dims = [shape[bond_str.index(c)] for c in bond_chars]
        tensor = torch.eye(bond_dims[0], bond_dims[1]).view(eye_shape)

        # Add on a bit of random noise
        tensor += std * torch.randn(shape)

    elif init_method == 'full_random':
        tensor = torch.randn(shape)

    return tensor

class InputSite(nn.Module):
    """
    A single MPS core which takes in a single input datum
    """
    def __init__(self, d, D_l, D_r=None):
        super().__init__()

        # Initialize our core tensor
        bond_str = 'lri'
        shape = [D_l, (D_r if D_r else D_l), d]
        tensor = init_tensor(shape, bond_str, 'random_eye')

        # Register our tensor as a Pytorch Parameter
        self.tensor = nn.Parameter(tensor)

    def forward(self, input_data):
        """
        Contract input with MPS core and return result as a SingleMat

        Args:
            input_data (Tensor): Input with shape [batch_size, d]
        """
        # Check that input_data has the correct shape
        tensor = self.tensor
        assert len(input_data.shape) == 2
        assert input_data.size(1) == tensor.size(2)

        # Contract the input with our core tensor
        mat = torch.einsum('lri,bi->blr', [tensor, input_data])

        return SingleMat(mat)

    def __len__(self):
        return 1

class InputRegion(nn.Module):
    """
    Contiguous region of MPS cores which takes in a collection of input data
    """
    def __init__(self, num_sites, d, D):
        super().__init__()

        # Initialize our site-indexed core tensor
        bond_str = 'slri'
        shape = [num_sites, D, D, d]
        tensor = init_tensor(shape, bond_str, 'random_eye')

        # Register our tensor as a Pytorch Parameter
        self.tensor = nn.Parameter(tensor)

    def forward(self, input_data):
        """
        Contract input with MPS cores and return result as a MatRegion

        Args:
            input_data (Tensor): Input with shape [batch_size, num_sites, d]
        """
        # Check that input_data has the correct shape
        tensor = self.tensor
        assert len(input_data.shape) == 3
        assert input_data.size(1) == tensor.size(0)
        assert input_data.size(2) == tensor.size(3)

        # Contract the input with our core tensor
        mats = torch.einsum('slri,bsi->bslr', [tensor, input_data])

        return MatRegion(mats)

    def __len__(self):
        return self.tensor.size(0)

class MergedInput(nn.Module):
    """
    Contiguous region of merged MPS cores, each taking in a pair of input data

    Since MergedInput arises after contracting together existing input cores,
    a merged input tensor is required for initialization
    """
    def __init__(self, tensor):
        # Check that our input tensor has the correct shape
        shape = tensor.shape
        assert len(shape) == 5
        assert shape[1] == shape[2]
        assert shape[3] == shape[4]

        super().__init__()

        # Register our tensor as a Pytorch Parameter
        self.tensor = nn.Parameter(tensor)

    def forward(self, input_data):
        """
        Contract input with merged MPS cores and return result as a MatRegion

        Args:
            input_data (Tensor): Input with shape [batch_size, num_sites, d], 
                                 where num_sites must be even (each merged
                                 core takes 2 inputs)
        """
        # Check that input_data has the correct shape
        tensor = self.tensor
        assert len(input_data.shape) == 3
        assert input_data.size(1) == len(self)
        assert input_data.size(2) == tensor.size(3)
        assert input_data.size(1) % 2 == 0
        
        # Divide input_data into inputs living on even and on odd sites
        inputs = [input_data[:, 0::2], input_data[:, 1::2]]

        # Contract the odd (right-most) and even inputs with merged cores
        tensor = torch.einsum('slrij,bsj->bslri', [tensor, inputs[1]])
        mats = torch.einsum('bslri,bsi->bslr', [tensor, inputs[0]])

        return MatRegion(mats)

    def __len__(self):
        """
        Returns the number of input sites, which is twice the number of cores
        """
        return 2 * self.tensor.size(0)

class OutputSite(nn.Module):
    """
    A single MPS core with no input and a single output index
    """
    def __init__(self, output_dim, D_l, D_r=None):
        super().__init__()

        # Initialize our core tensor
        bond_str = 'olr'
        shape = [output_dim, D_l, (D_r if D_r else D_l)]
        tensor = init_tensor(shape, bond_str, 'random_eye')

        # Register our tensor as a Pytorch Parameter
        self.tensor = nn.Parameter(tensor)

    def forward(self, input_data):
        """
        Return the OutputSite wrapped as an OutputCore contractable
        """
        return OutputCore(self.tensor)

    def __len__(self):
        return 0

class MergedOutput(nn.Module):
    """
    Merged MPS core taking in one input datum and returning an output vector

    Since MergedInput arises after contracting together an input and an output
    core, an existing merged tensor is required for initialization
    """
    def __init__(self, tensor, left_input=True):
        """
        left_input specifies if the input core was contracted on the left 
        (True), or on the right (False). This information isn't needed here
        but is essential when unmerging a MergedOutput into two separate cores
        """
        # Check that our input tensor has the correct shape
        assert len(tensor.shape) == 4
        super().__init__()

        # Register our tensor as a Pytorch Parameter
        self.tensor = nn.Parameter(tensor)
        self.left_input = left_input

    def forward(self, input_data):
        """
        Contract input with input index of core and return an OutputCore

        Args:
            input_data (Tensor): Input with shape [batch_size, d]
        """
        # Check that input_data has the correct shape
        tensor = self.tensor
        assert len(input_data.shape) == 2
        assert input_data.size(1) == tensor.size(3)

        # Contract the input with our core tensor
        mat = torch.einsum('olri,bi->bolr', [tensor, input_data])

        return SingleMat(mat)

    def __len__(self):
        return 1