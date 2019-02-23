import torch
import torch.nn as nn
from utils import init_tensor, svd_flex
from contractables import SingleMat, MatRegion, OutputCore, ContractableList, \
                          EdgeVec

class MPS(nn.Module):
    """
    Matrix product state which converts input into a single output vector
    """
    def __init__(self, input_dim, output_dim, bond_dim, d=2, label_site=None,
                 periodic_bc=False, parallel_eval=False, dynamic_mode=False, 
                 cutoff=1e-10, threshold=2000, init_std=1e-9):
        super().__init__()

        if label_site is None:
            label_site = input_dim // 2
        assert label_site >= 0 and label_site <= input_dim

        # Our MPS is made of two InputRegions separated by an OutputSite.
        # If our output is at an end of the MPS, we only have one InputRegion
        module_list = []
        if label_site > 0:
            module_list.append(InputRegion(None, label_site, bond_dim, d))

        module_list.append(OutputSite(None, output_dim, bond_dim))

        if label_site < input_dim:
            module_list.append(InputRegion(None, 
                               input_dim - label_site, bond_dim, d))

        # Initialize linear_region according to our dynamic_mode specification
        if dynamic_mode:
            self.linear_region = MergedLinearRegion(module_list=module_list, 
                                 periodic_bc=periodic_bc, 
                                 parallel_eval=parallel_eval, cutoff=cutoff,
                                 threshold=threshold)
        else:
            self.linear_region = LinearRegion(module_list=module_list, 
                                 periodic_bc=periodic_bc,
                                 parallel_eval=parallel_eval)
        assert len(self.linear_region) == input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.label_site = label_site
        self.bond_dim = bond_dim
        self.d = d

        self.periodic_bc = periodic_bc
        self.dynamic_mode = dynamic_mode
        self.cutoff = cutoff
        self.threshold = threshold

    def embed_input(self, input_data):
        """
        Embed pixels of input_data into separate d-dimensional spaces

        Args:
            input_data (Tensor):    Input with shape [batch_size, input_dim]

        Returns:
            embedded_data (Tensor): Input embedded into a tensor with shape
                                    [batch_size, input_dim, d]
        """
        assert len(input_data.shape) == 2
        assert input_data.size(1) == self.input_dim

        embedded_shape = list(input_data.shape) + [self.d]
        embedded_data = torch.empty(embedded_shape)

        # A simple linear embedding map
        embedded_data[:,:,0] = input_data
        embedded_data[:,:,1] = 1 - input_data

        return embedded_data

    def core_len(self):
        """
        Returns the number of cores, which is at least the required input size
        """
        return self.linear_region.core_len()

    def __len__(self):
        """
        Returns the number of input sites, which is the required input size
        """
        return self.input_dim

    def forward(self, input_data):
        """
        Embed our data and pass it to an MPS with a single output site
        """
        # WHEN IMPLEMENTING ROUTING FOR CUSTOM PATHS, THAT CODE GOES HERE

        # Embed our input data before feeding it into our linear region
        input_data = self.embed_input(input_data)
        output = self.linear_region(input_data)

        # return output
        return torch.abs(output)

class LinearRegion(nn.Module):
    """
    List of modules which feeds input to each module and returns reduced output
    """
    def __init__(self, module_list, periodic_bc=False, parallel_eval=False,
                 module_states=None):
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
            input_data (Tensor): Input with shape [batch_size, input_dim, d]
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
            if mod_len == 1:
                mod_input = input_data[:, ind]
            else:
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

    def core_len(self):
        """
        Returns the number of cores, which is at least the required input size
        """
        return sum([module.core_len() for module in self.module_list])

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
                 cutoff=1e-10, threshold=2000):
        # Initialize a LinearRegion with our given module_list
        super().__init__(module_list, periodic_bc, parallel_eval)
        
        # Initialize attributes self.module_list_0 and self.module_list_1 
        # using the unmerged self.module_list, then redefine the latter in
        # terms of one of the former lists
        self.offset = 0
        self.merge(offset=self.offset)
        self.merge(offset=(self.offset+1)%2)
        self.module_list = getattr(self, f"module_list_{self.offset}")

        # Initialize variables used during switching
        self.input_counter = 0
        self.threshold = threshold
        self.cutoff = cutoff

    def forward(self, input_data):
        """
        Contract input with list of MPS cores and return result as contractable

        MergedLinearRegion keeps an input counter of the number of inputs, and
        when this exceeds its threshold, triggers an unmerging and remerging of
        its parameter tensors.

        Args:
            input_data (Tensor): Input with shape [batch_size, input_dim, d]
        """
        # If we've hit our threshold, flip the merge state of our tensors
        if self.input_counter >= self.threshold:
            self.unmerge(cutoff=self.cutoff)
            self.offset = (self.offset + 1) % 2
            self.merge(offset=self.offset)
            self.input_counter -= self.threshold

            # Point self.module_list to the appropriate merged module
            self.module_list = getattr(self, f"module_list_{self.offset}")

        # Increment our counter and call the LinearRegion's forward method
        self.input_counter += input_data.size(0)
        return super().forward(input_data)

    def merge(self, offset):
        """
        Convert unmerged modules in self.module_list to merged counterparts

        This proceeds by first merging all unmerged cores internally, then
        merging lone cores when possible during a second sweep
        """
        assert offset in [0, 1]

        with torch.no_grad():
            unmerged_list = self.module_list

            # Merge each core internally and add the results to midway_list
            site_num = offset
            merged_list = []
            for core in unmerged_list:
                assert not isinstance(core, MergedInput)
                assert not isinstance(core, MergedOutput)

                # Apply internal merging routine if our core supports it
                if hasattr(core, 'merge'):
                    merged_list.extend(core.merge(offset=site_num%2))
                else:
                    merged_list.append(core)

                site_num += core.core_len()

            # Merge pairs of cores when possible (currently only with 
            # InputSites), making sure to respect the offset for merging. 
            while True:
                mod_num, site_num = 0, 0
                combined_list = []
                
                while mod_num < len(merged_list) - 1:
                    left_core, right_core = merged_list[mod_num: mod_num+2]
                    new_core = self.combine(left_core, right_core, 
                                                       merging=True)
                    
                    # If cores aren't combinable, move our sliding window by 1
                    if new_core is None or offset != site_num % 2:
                        combined_list.append(left_core)
                        mod_num += 1
                        site_num += left_core.core_len()
                    
                    # If we get something new, move to the next distinct pair
                    else:
                        assert new_core.core_len() == left_core.core_len() + \
                                                      right_core.core_len()
                        combined_list.append(new_core)
                        mod_num += 2
                        site_num += new_core.core_len()
                    
                    # Add the last core if there's nothing to merge it with
                    if mod_num == len(merged_list)-1:
                        combined_list.append(merged_list[mod_num])
                        mod_num += 1

                # We're finished when unmerged_list remains unchanged
                if len(combined_list) == len(merged_list):
                    break
                else:
                    merged_list = combined_list

            # Finally, update the appropriate merged module list
            list_name = f"module_list_{offset}"
            # If the merged module list hasn't been set yet, initialize it
            if not hasattr(self, list_name):
                setattr(self, list_name, nn.ModuleList(merged_list))
            
            # Otherwise, do an in-place update so that all tensors remain 
            # properly registered with whatever optimizer we use
            else:
                module_list = getattr(self, list_name)
                assert len(module_list) == len(merged_list)
                for i in range(len(module_list)):
                    assert module_list[i].tensor.shape == \
                           merged_list[i].tensor.shape
                    module_list[i].tensor[:] = merged_list[i].tensor

    def unmerge(self, cutoff=1e-10):
        """
        Convert merged modules in self.module_list to unmerged counterparts

        This proceeds by first unmerging all merged cores internally, then
        combining lone cores where possible
        """
        with torch.no_grad():
            list_name = f"module_list_{self.offset}"
            merged_list = getattr(self, list_name)

            # Unmerge each core internally and add results to unmerged_list
            unmerged_list = []
            for core in merged_list:

                # Apply internal unmerging routine if our core supports it
                if hasattr(core, 'unmerge'):
                    unmerged_list.extend(core.unmerge(cutoff))
                else:
                    unmerged_list.append(core)

            # Combine all combinable pairs of cores. This occurs in several
            # passes, and for now acts nontrivially only on InputSite instances
            while True:
                mod_num = 0
                combined_list = []
                
                while mod_num < len(unmerged_list) - 1:
                    left_core, right_core = unmerged_list[mod_num: mod_num+2]
                    new_core = self.combine(left_core, right_core, 
                                                       merging=False)

                    # If cores aren't combinable, move our sliding window by 1
                    if new_core is None:
                        combined_list.append(left_core)
                        mod_num += 1
                    # If we get something new, move to the next distinct pair
                    else:
                        combined_list.append(new_core)
                        mod_num += 2

                    # Add the last core if there's nothing to combine it with
                    if mod_num == len(unmerged_list)-1:
                        combined_list.append(unmerged_list[mod_num])
                        mod_num += 1

                # We're finished when unmerged_list remains unchanged
                if len(combined_list) == len(unmerged_list):
                    break
                else:
                    unmerged_list = combined_list

            # Finally, add our unmerged module list as a new attribute
            self.module_list = nn.ModuleList(unmerged_list)

    def combine(self, left_core, right_core, merging):
        """
        Combine a pair of cores into a new core using context-dependent rules

        Depending on the types of left_core and right_core, along with whether
        we're currently merging (merging=True) or unmerging (merging=False), 
        either return a new core, or None if no rule exists for this context
        """
        # Combine an OutputSite with a stray InputSite, return a MergedOutput
        if merging and ((isinstance(left_core, OutputSite) and 
                         isinstance(right_core, InputSite)) or
                            (isinstance(left_core, InputSite) and 
                            isinstance(right_core, OutputSite))):

            left_site = isinstance(left_core, InputSite)
            if left_site:
                new_tensor = torch.einsum('lui,our->olri', [left_core.tensor, 
                                                            right_core.tensor])
            else:
                new_tensor = torch.einsum('olu,uri->olri', [left_core.tensor, 
                                                            right_core.tensor])
            return MergedOutput(new_tensor, left_output=(not left_site))

        # Combine an InputRegion with a stray InputSite, return an InputRegion
        elif not merging and ((isinstance(left_core, InputRegion) and 
                               isinstance(right_core, InputSite)) or
                                    (isinstance(left_core, InputSite) and 
                                    isinstance(right_core, InputRegion))):

            left_site = isinstance(left_core, InputSite)
            if left_site:
                left_tensor = left_core.tensor.unsqueeze(0)
                right_tensor = right_core.tensor
            else:
                left_tensor = left_core.tensor
                right_tensor = right_core.tensor.unsqueeze(0)

            assert left_tensor.shape[1:] == right_tensor.shape[1:]
            new_tensor = torch.cat([left_tensor, right_tensor])

            return InputRegion(new_tensor)

        # If this situation doesn't belong to the above cases, return None
        else:
            return None

    def core_len(self):
        """
        Returns the number of cores, which is at least the required input size
        """
        return sum([module.core_len() for module in self.module_list])

    def __len__(self):
        """
        Returns the number of input sites, which is the required input size
        """
        return sum([len(module) for module in self.module_list])

class InputRegion(nn.Module):
    """
    Contiguous region of MPS cores which takes in a collection of input data
    """
    def __init__(self, tensor=None, input_dim=None, bond_dim=None, d=None,
                 init_std=1e-9):
        super().__init__()
        bond_str = 'slri'

        # If it isn't given, initialize our site-indexed core tensor
        if tensor is None:
            shape = [input_dim, bond_dim, bond_dim, d]
            tensor = init_tensor(shape, bond_str, ('random_eye', init_std))

        # Register our tensor as a Pytorch Parameter
        self.tensor = nn.Parameter(tensor.contiguous())

    def forward(self, input_data):
        """
        Contract input with MPS cores and return result as a MatRegion

        Args:
            input_data (Tensor): Input with shape [batch_size, input_dim, d]
        """
        # Check that input_data has the correct shape
        tensor = self.tensor
        assert len(input_data.shape) == 3
        assert input_data.size(1) == len(self)
        assert input_data.size(2) == tensor.size(3)

        # Contract the input with our core tensor
        mats = torch.einsum('slri,bsi->bslr', [tensor, input_data])

        return MatRegion(mats)

    def merge(self, offset):
        """
        Merge all pairs of neighboring cores and return a new list of cores

        offset is either 0 or 1, which gives the first core at which we start 
        our merging. Depending on the length of our InputRegion, the output of
        merge may have 1, 2, or 3 entries, with the majority of sites ending in
        a MergedInput instance
        """
        assert offset in [0, 1]
        num_sites = self.core_len()
        parity = num_sites % 2

        # Cases with empty tensors might arise in recursion below
        if num_sites == 0:
            return [None]

        # Simplify the problem into one where offset=0 and num_sites is even
        if (offset, parity) == (1, 1):
            out_list = [self[0], self[1:].merge(offset=0)[0]]
        elif (offset, parity) == (1, 0):
            out_list = [self[0], self[1:-1].merge(offset=0)[0], self[-1]]
        elif (offset, parity) == (0, 1):
            out_list = [self[:-1].merge(offset=0)[0], self[-1]]

        # The main case of interest, with no offset and an even number of sites
        else:
            tensor = self.tensor
            even_cores, odd_cores = tensor[0::2], tensor[1::2]
            assert len(even_cores) == len(odd_cores)

            # Multiply all pairs of cores, keeping inputs separate
            merged_cores = torch.einsum('slui,surj->slrij', [even_cores, 
                                                             odd_cores])
            out_list = [MergedInput(merged_cores)]

        # Remove empty MergedInputs, which appear in very small InputRegions
        return [x for x in out_list if x is not None]

    def __getitem__(self, key):
        """
        Returns an InputRegion instance sliced along the site index
        """
        assert isinstance(key, int) or isinstance(key, slice)

        if isinstance(key, slice):
            return InputRegion(self.tensor[key])
        else:
            return InputSite(self.tensor[key])

    def core_len(self):
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
        self.tensor = nn.Parameter(tensor.contiguous())

    def forward(self, input_data):
        """
        Contract input with merged MPS cores and return result as a MatRegion

        Args:
            input_data (Tensor): Input with shape [batch_size, input_dim, d], 
                                 where input_dim must be even (each merged
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
        original MergedInput (same number of inputs), but its core_len will
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

    def core_len(self):
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
        self.tensor = nn.Parameter(tensor.contiguous())

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

    def core_len(self):
        return 1

    def __len__(self):
        return 1

class OutputSite(nn.Module):
    """
    A single MPS core with no input and a single output index
    """
    def __init__(self, tensor, output_dim=None, D_l=None, D_r=None, 
                 init_std=1e-9):
        super().__init__()
        bond_str = 'olr'

        # If it isn't given, initialize our core tensor
        if tensor is None:
            shape = [output_dim, D_l, (D_r if D_r else D_l)]
            tensor = init_tensor(shape, bond_str, ('random_eye', init_std))

        # Register our tensor as a Pytorch Parameter
        self.tensor = nn.Parameter(tensor.contiguous())

    def forward(self, input_data):
        """
        Return the OutputSite wrapped as an OutputCore contractable
        """
        return OutputCore(self.tensor)

    def core_len(self):
        return 1

    def __len__(self):
        return 0

class MergedOutput(nn.Module):
    """
    Merged MPS core taking in one input datum and returning an output vector

    Since MergedOutput arises after contracting together an existing input and 
    output core, an already-merged tensor is required for initialization

    Args:
        tensor (Tensor):    Value that our merged core is initialized to
        left_output (bool): Specifies if the output core is on the left side of
                            the input core (True), or on the right (False)
    """
    def __init__(self, tensor, left_output):
        # Check that our input tensor has the correct shape
        bond_str = 'olri'
        assert len(tensor.shape) == 4
        super().__init__()

        # Register our tensor as a Pytorch Parameter
        self.tensor = nn.Parameter(tensor.contiguous())
        self.left_output = left_output

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
        tensor = torch.einsum('olri,bi->bolr', [tensor, input_data])

        return OutputCore(tensor)

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

    def core_len(self):
        return 2

    def __len__(self):
        return 1
