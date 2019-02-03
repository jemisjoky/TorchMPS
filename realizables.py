"""
TODO:
"""
import torch
import torch.nn as nn
from contractables import SingleMat, MatRegion, OutputCore

class MPS(nn.Module):
    """
    Matrix product state which converts input into a single output vector
    """
    def __init__(self, input_size, output_dim, bond_dim, d=2, label_site=None,
                 periodic_bc=False, parallel_eval=False, dynamic_mode=False):
        if not label_site:
            label_site = input_size // 2
        assert label_site >= 0 and label_site <= input_size

        # Our MPS is made of two InputRegions separated by an OutputSite.
        # If our output is at an end of the MPS, we only have one InputRegion
        if label_site > 0:
            module_list = [InputRegion(label_site, bond_dim, d)]
        else:
            module_list = []

        module_list.append(OutputSite(output_dim, bond_dim))

        if label_site < input_size:
            module_list.append(InputRegion(input_size-label_site, bond_dim, d))

        # Initialize linear_region according to our dynamic_mode specification
        if dynamic_mode:
            self.linear_region = MergedLinearRegion(module_list, periodic_bc,
                                                    parallel_eval)
        else:
            self.linear_region = LinearRegion(module_list, periodic_bc,
                                              parallel_eval)
        assert len(self.linear_region) == input_size

        self.label_site = label_site
        self.periodic_bc = periodic_bc
        self.dynamic_mode = dynamic_mode

    def forward(self, input_data):
        """

        """
        # IF DOING CUSTOM ROUTING, THAT CODE GOES HERE

        return self.linear_region(input_data)

class MergedLinearRegion(LinearRegion):
    """
    Dynamic variant of LinearRegion that periodically rearranges its submodules
    """
    def __init__(self, module_list, periodic_bc=False, parallel_eval=False,
                 threshold=1000):
        # Initialize a LinearRegion with our given module_list
        super().__init__(module_list, periodic_bc, parallel_eval)
        self.left_merged = None

        # Merge all of our parameter tensors, which rewrites self.module_list
        self.merge(merge_left=True)

        self.input_counter = 0
        self.threshold = threshold

    def unmerge(self):
        """
        Convert merged modules in self.module_list to unmerged counterparts
        """
        assert isinstance(self.left_merged, bool)
        module_list = self.module_list

        pass

        self.left_merged = None

    def merge(self, merge_left=True):
        """
        Convert unmerged modules in self.module_list to merged counterparts
        """
        assert self.left_merged is None
        module_list = self.module_list

        pass

        self.left_merged = merge_left

class LinearRegion(nn.Module):
    """
    List of modules which feeds input to each module and returns reduced output
    """
    def __init__(self, module_list, periodic_bc=False, parallel_eval=False):
        # Check that module_list is a list whose entries are Pytorch modules
        if not isinstance(module_list, list) or module_list is []:
            raise ValueError("Input to LinearRegion must be nonempty list")
        for i, item in enumerate(module_list):
            if not isinstance(item, nn.Module):
                raise ValueError("Input items to LinearRegion must be PyTorch "
                                f"Module instances, but item {i} is not")
        super().__init__()

        # Wrap as a ModuleList for proper parameter registration
        self.module_list = nn.ModuleList(module_list)
        self.periodic_bc = periodic_bc
        self.parallel_eval = parallel_eval

    def forward(self, input_data):
        """
        Contract input with list of MPS cores and return result as contractable

        Args:
            input_data (Tensor): Input with shape [batch_size, input_size, d]
        """
        # Check that input_data has the correct shape
        assert len(input_data.shape) == 3
        assert input_data.size(1) == len(self)
        periodic_bc = self.periodic_bc
        parallel_eval = self.parallel_eval
        lin_bonds = ['l', 'r']

        # For each module, pull out the number of pixels needed and call that
        # module's forward() method, putting the result in contractable_list
        ind = 0
        contractable_list = []
        for module in self.module_list:
            mod_len = len(module)
            mod_input = input_data[:, ind:(ind+mod_len)]
            ind += mod_len

            contractable_list.append(module(mod_input))

        # For periodic boundary conditions, reduce contractable_list and 
        # trace over the left and right indices to get our output
        if periodic_bc:
            contractable_list = ContractableList(contractable_list)
            contractable = contractable_list.reduce(parallel_eval=True)

            # Unpack the output (atomic) contractable
            tensor, bond_str = contractable.tensor, contractable.bond_str
            assert all(c in bond_str for c in lin_bonds)

            # Build einsum string for the trace of tensor
            in_str, out_str = "", ""
            for c in bond_str:
                if c in lin_bonds:
                    in_str += 'l'
                else:
                    in_str += c
                    out_str += c
            ein_str = in_str + "->" + out_str

            # Return the trace over left and right indices
            return torch.einsum(ein_str, [tensor])

        # For open boundary conditions, add dummy edge vectors to 
        # contractable_list and reduce everything to get our output
        else:
            # Get the dimension of left and right bond indices
            end_items = [contractable_list[i]for i in [0, -1]]
            bond_strs = [item.bond_str for item in end_items]
            bond_inds = [bs.index(c) for (bs, c) in zip(bond_strs, lin_bonds)]
            bond_dims = [item.tensor.size(ind) for (item, ind) in 
                                               zip(end_items, bond_inds)]

            # Build dummy end vectors and insert them at the ends of our list
            end_vecs = [torch.zeros(dim) for dim in bond_dims]
            for vec in end_vecs:
                vec[0] = 1
            contractable_list.insert(0, EdgeVec(end_vecs[0], is_left_vec=True))
            contractable_list.append(EdgeVec(end_vecs[1], is_left_vec=False))

            # Multiply together everything in contractable_list
            contractable_list = ContractableList(contractable_list)
            output = contractable_list.reduce(parallel_eval=parallel_eval)

            return output.tensor

    def __len__(self):
        """
        Returns the number of input sites, which is the required size of input
        """
        return sum([len(module) for module in self.module_list])

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
    def __init__(self, input_size, bond_dim, d):
        super().__init__()

        # Initialize our site-indexed core tensor
        bond_str = 'slri'
        shape = [input_size, bond_dim, bond_dim, d]
        tensor = init_tensor(shape, bond_str, 'random_eye')

        # Register our tensor as a Pytorch Parameter
        self.tensor = nn.Parameter(tensor)

    def forward(self, input_data):
        """
        Contract input with MPS cores and return result as a MatRegion

        Args:
            input_data (Tensor): Input with shape [batch_size, input_size, d]
        """
        # Check that input_data has the correct shape
        tensor = self.tensor
        assert len(input_data.shape) == 3
        assert input_data.size(1) == len(self)
        assert input_data.size(2) == tensor.size(3)

        # Contract the input with our core tensor
        mats = torch.einsum('slri,bsi->bslr', [tensor, input_data])

        return MatRegion(mats)

    def __len__(self):
        return self.tensor.size(0)

class _MergedInput(nn.Module):
    """
    Contiguous region of merged MPS cores, each taking in a pair of input data

    Since _MergedInput arises after contracting together existing input cores,
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
            input_data (Tensor): Input with shape [batch_size, input_size, d], 
                                 where input_size must be even (each merged
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

class _MergedOutput(nn.Module):
    """
    Merged MPS core taking in one input datum and returning an output vector

    Since _MergedOutput arises after contracting together an input and an 
    output core, an existing merged tensor is required for initialization
    """
    def __init__(self, tensor, left_input=True):
        """
        left_input specifies if the input core was contracted on the left 
        (True), or on the right (False). This information isn't needed here
        but is essential when unmerging a _MergedOutput into two separate cores
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
