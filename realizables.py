import torch
import torch.nn as nn
from contractables import SingleMat, MatRegion, OutputCore, ContractableList, \
                          EdgeVec

class MPS(nn.Module):
    """
    Matrix product state which converts input into a single output vector
    """
    def __init__(self, input_size, output_dim, bond_dim, d=2, label_site=None,
                 periodic_bc=False, parallel_eval=False, dynamic_mode=False):
        super().__init__()

        if label_site is None:
            label_site = input_size // 2
        assert label_site >= 0 and label_site <= input_size

        # Our MPS is made of two InputRegions separated by an OutputSite.
        # If our output is at an end of the MPS, we only have one InputRegion
        module_list = []
        if label_site > 0:
            module_list.append(InputRegion(None, label_site, bond_dim, d))

        module_list.append(OutputSite(None, output_dim, bond_dim))

        if label_site < input_size:
            module_list.append(InputRegion(None, 
                               input_size - label_site, bond_dim, d))

        # Initialize linear_region according to our dynamic_mode specification
        if dynamic_mode:
            self.linear_region = MergedLinearRegion(module_list, periodic_bc,
                                                    parallel_eval)
        else:
            self.linear_region = LinearRegion(module_list, periodic_bc,
                                              parallel_eval)
        assert len(self.linear_region) == input_size

        self.input_size = input_size
        self.output_dim = output_dim
        self.label_site = label_site
        self.bond_dim = bond_dim
        self.d = d

        self.periodic_bc = periodic_bc
        self.dynamic_mode = dynamic_mode

    def embed_input(self, input_data):
        """
        Embed pixels of input_data into separate d-dimensional spaces

        Args:
            input_data (Tensor):    Input with shape [batch_size, input_size]

        Returns:
            embedded_data (Tensor): Input embedded into a tensor with shape
                                    [batch_size, input_size, d]
        """
        assert len(input_data.shape) == 2
        assert input_data.size(1) == self.input_size

        embedded_shape = list(input_data.shape) + [self.d]
        embedded_data = torch.empty(embedded_shape)

        # A simple linear embedding map
        embedded_data[:,:,0] = input_data
        embedded_data[:,:,1] = 1 - input_data

        return embedded_data

    def __len__(self):
        """
        Returns the number of input sites, which is the required input size
        """
        return self.input_size

    def forward(self, input_data):
        """

        """
        # IF DOING CUSTOM ROUTING, THAT CODE GOES HERE

        # Embed our input data before feeding it into our linear region
        input_data = self.embed_input(input_data)

        return self.linear_region(input_data)

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

    def literal_len(self):
        """
        Returns the number of cores, which is at least the required input size
        """
        return sum([module.literal_len() for module in self.module_list])

    def __len__(self):
        """
        Returns the number of input sites, which is the required input size
        """
        return sum([len(module) for module in self.module_list])

class MergedLinearRegion(LinearRegion):
    """
    Dynamic variant of LinearRegion that periodically rearranges its submodules
    """
    def __init__(self, module_list, periodic_bc=False, parallel_eval=False,
                 threshold=1000):
        # Initialize a LinearRegion with our given module_list
        super().__init__(module_list, periodic_bc, parallel_eval)

        # Merge all of our parameter tensors, which rewrites self.module_list
        self.merge_left = True
        self.merge(merge_left=self.merge_left)

        self.input_counter = 0
        self.threshold = threshold

    def unmerge(self, cutoff=1e-10):
        """
        Convert merged modules in self.module_list to unmerged counterparts
        """
        # DON'T FORGET TO DO EVERYTHING WITH NO_GRAD

        module_list = self.module_list

        pass

    def merge(self, merge_left):
        """
        Convert unmerged modules in self.module_list to merged counterparts
        """
        unmerged_list = self.module_list
        merged_list = []

        # DON'T FORGET TO DO EVERYTHING WITH NO_GRAD
        
        # Cores that admit merging
        for core in unmerged_list:
            pass

    def forward(self, input_data):
        """
        Contract input with list of MPS cores and return result as contractable

        MergedLinearRegion keeps an input counter of the number of inputs, and
        when this exceeds its threshold, triggers an unmerging and remerging of
        its parameter tensors.

        Args:
            input_data (Tensor): Input with shape [batch_size, input_size, d]
        """
        # Check if we've hit our threshold yet, and if so flip our merge state
        if self.input_counter >= self.threshold:
            self.unmerge()
            self.merge_left = not self.merge_left
            self.merge(merge_left=self.merge_left)

        # Increment our counter and call the real forward method
        self.input_counter += input_data.size(0)
        return super().forward(input_data)

    def literal_len(self):
        """
        Returns the number of cores, which is at least the required input size
        """
        return sum([module.literal_len() for module in self.module_list])

    def __len__(self):
        """
        Returns the number of input sites, which is the required input size
        """
        return sum([len(module) for module in self.module_list])

class InputRegion(nn.Module):
    """
    Contiguous region of MPS cores which takes in a collection of input data
    """
    def __init__(self, tensor=None, input_size=None, bond_dim=None, d=None):
        super().__init__()
        bond_str = 'slri'

        # If it isn't given, initialize our site-indexed core tensor
        if tensor is None:
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

    def merge(self, offset=0):
        """
        Merge all pairs of neighboring cores and return a new list of cores

        offset is either 0 or 1, which gives the first core at which we start 
        our merging. Depending on the length of our InputRegion, the output of
        merge may have 1, 2, or 3 entries, with the majority of sites ending in
        a MergedInput instance
        """
        assert offset in [0, 1]
        num_sites = self.literal_len()
        parity = num_sites % 2

        # Cases with empty tensors might arise in recursion below
        if num_sites == 0:
            return [None]

        # Simplify the problem into one where offset=0 and num_sites is even
        if (offset, parity) == (1, 1):
            out_list = [self[0], self[1:].merge()[0]]
            return [x for x in out_list if x is not None]
        elif (offset, parity) == (1, 0):
            out_list = [self[0], self[1:-1].merge()[0], self[-1]]
            return [x for x in out_list if x is not None]
        elif (offset, parity) == (0, 1):
            out_list = [self[:-1].merge()[0], self[-1]]
            return [x for x in out_list if x is not None]

        # The main case of interest, with no offset and an even number of sites
        else:
            tensor = self.tensor

            even_cores, odd_cores = tensor[0::2], tensor[1::2]
            assert len(even_cores) == len(odd_cores)

            # Multiply all pairs of cores, keeping inputs separate
            merged_cores = einsum('slui,surj->slrij', [even_cores, odd_cores])

            return [MergedInput(merged_cores)]

    def __getitems__(self, key):
        """
        Returns an InputRegion instance sliced along the site index
        """
        assert isinstance(key, int) or isinstance(key, slice)

        if isinstance(key, slice):
            return InputRegion(self.tensor[key])
        else:
            return InputSite(self.tensor[key])

    def literal_len(self):
        return len(self)

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
        bond_str = 'slrij'
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

    def unmerge(self, cutoff=1e-10):
        """
        Separate the cores in our MergedInput and return an InputRegion

        The length of the resultant InputRegion will be identical to our 
        original MergedInput (same number of inputs), but its literal_len will
        be doubled (twice as many individual cores)
        """
        bond_str = 'slrij'
        tensor = self.tensor
        svd_string = 'lrij->lui,urj'
        max_D = tensor.size(1)

        # Split every one of the cores into two and add them both to core_list
        core_list, bond_list = [], []
        for merged_core in tensor:
            left_core, right_core, bond_dim = svd_flex(merged_core, svd_string,
                                                       max_D, cutoff)
            core_list += [left_core, right_core]
            bond_list += [bond_dim]

        # Collate the split cores into one tensor and return as an InputRegion
        tensor = torch.stack(core_list)
        return [InputRegion(tensor)]

    def literal_len(self):
        return len(self)

    def __len__(self):
        """
        Returns the number of input sites, which is twice the number of cores
        """
        return 2 * self.tensor.size(0)

class InputSite(nn.Module):
    """
    A single MPS core which takes in a single input datum
    """
    def __init__(self, tensor=None, d=None, D_l=None, D_r=None):
        super().__init__()

        # If it isn't given, initialize our site-indexed core tensor
        if tensor is None:
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

    def literal_len(self):
        return 1

    def __len__(self):
        return 1

class OutputSite(nn.Module):
    """
    A single MPS core with no input and a single output index
    """
    def __init__(self, tensor, output_dim=None, D_l=None, D_r=None):
        super().__init__()
        bond_str = 'olr'

        # If it isn't given, initialize our core tensor
        if tensor is None:
            shape = [output_dim, D_l, (D_r if D_r else D_l)]
            tensor = init_tensor(shape, bond_str, 'random_eye')

        # Register our tensor as a Pytorch Parameter
        self.tensor = nn.Parameter(tensor)

    def forward(self, input_data):
        """
        Return the OutputSite wrapped as an OutputCore contractable
        """
        return OutputCore(self.tensor)

    def merge(self, other_core=None, left_output=True):
        """
        Merge OutputSite with an InputSite and return a MergedOutput

        If left_output is True, our Output site is on the left side, otherwise 
        it appears on the right side of the MergedOutput. If no inputs are 
        given, this just returns our existing OutputSite
        """
        if other_core is None:
            return self
        assert isinstance(other_core, InputSite)

        if offset == 0:
            merged_core = torch.einsum('olu,uri->olri',
                                       [self.tensor, other_core.tensor])
        else:
            merged_core = torch.einsum('lui,our->olri',
                                       [other_core.tensor, self.tensor])

        return [MergedOutput(merged_core, left_output=(offset==0))]

    def literal_len(self):
        return 1

    def __len__(self):
        return 0

class MergedOutput(nn.Module):
    """
    Merged MPS core taking in one input datum and returning an output vector

    Since MergedOutput arises after contracting together an input and an 
    output core, an existing merged tensor is required for initialization

    Args:
        tensor (Tensor):    Value that our merged core is initialized to
        left_output (bool): Specifies if the output core is on the left side of
                            the input core (True), or on the right (False)
        bond_dims (list):   Bond dimensions of the left and right bonds
    """
    def __init__(self, tensor, left_output, bond_dims):
        # Check that our input tensor has the correct shape
        bond_str = 'olri'
        assert len(tensor.shape) == 4
        super().__init__()

        # Register our tensor as a Pytorch Parameter
        self.tensor = nn.Parameter(tensor)
        self.left_output = left_output
        self.bond_dims = bond_dims

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

    def unmerge(self, cutoff=1e-10):
        """
        Split our MergedOutput into an OutputSite and an InputSite

        The non-zero entries of our tensors are dynamically sized according to 
        the SVD cutoff, but will generally be padded with zeros to give the 
        new index a regular size.
        """
        bond_str = 'olri'
        tensor = self.tensor
        left_output = self.left_output
        if left_output:
            svd_string = 'olri->olu,uri'
            max_D = tensor.size(2)
            output_core, input_core, bond_dim = svd_flex(tensor, svd_string, 
                                                         max_D, cutoff)
            return [OutputSite(output_core), InputSite(input_core)]

        else:
            svd_string = 'olri->our,lui'
            max_D = tensor.size(1)
            output_core, input_core, bond_dim = svd_flex(tensor, svd_string, 
                                                         max_D, cutoff)
            return [InputSite(input_core), OutputSite(output_core)]

    def literal_len(self):
        return 2

    def __len__(self):
        return 1

