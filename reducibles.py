"""
TODO:
    * Implement the following Reducible subclasses:

        (1) LinearRegion, which holds a list of Reducibles and just reduces
            all of them before multiplying each in turn
        (2) Periodic, which takes in a LinearRegion and first reduces it before
            taking the trace of the resultant linear contractable
        (3) Open, which takes in a LinearRegion, inserts a SingleVec on either
            end, and multiplies everything together. Alternately, if the 
            optional arg parallel_reduction is True, then first reduce 
            everything in the list before multiplying it out
        (4) 

    * Modify my trace code to work on arbitrary output contractables. Right now
      it just checks specific cases and applies a type-specific operation
"""
class Reducible:
    """
    An object which can be 'reduced' in some way to get a contractable

    This class includes our various procedures to take a collection of 
    contractables and multiply them all together. When implementing a Pytorch
    module using TorchMPS, the last step will typically be calling a reduce
    operation and then returning the tensor associated with the simple output
    contractable

    The Reducible class itself is just an empty template class, working
    implementations should just subclass this for type checking
    """
    def __init__(self):
        raise NotImplementedError

    def reduce(self):
        raise NotImplementedError

class LinearRegion(Reducible):
    """
    A list of reducibles which can all be multiplied together in order

    Calling reduce on a LinearRegion instance will simply reduce every item
    to a linear contractable, before multiplying all such contractables 
    in a manner set by the open_bc and right_to_left options
    """
    def __init__(self, reducible_list):
        if not reducible_list:
            raise ValueError("Input to LinearRegion must be nonempty list")

        for i, item in enumerate(reducible_list):
            if not isinstance(item, Reducible):
                raise ValueError("Inputs to LinearRegion must be reducibles, "
                                f"but item {i} is not")

        self.reducible_list = reducible_list

    def reduce(self, open_bc, right_to_left=False):
        """
        Reduce all the reducibles in our list before multiplying them together

        If open_bc is False, reduce is only called on items which can't be 
        linearly contracted together. If right_to_left is True, multiplication 
        is done right to left. This doesn't change the result, but might be 
        more efficient in some situations
        """
        reducible_list = self.reducible_list
        contract_list = []

        # Reduce our reducibles and put the outputs into contract_list
        for item in reducible_list:
            if open_bc or not hasattr(item, "__mul__"):
                item = item.reduce()
            
            assert hasattr(item, "__mul__")
            contract_list.append(item)

        # Multiply together contractables in the correct order 
        if right_to_left:
            contractable = contract_list[-1]
            for item in contract_list[-2::-1]:
                contractable = item * contractable
        else:
            contractable = contract_list[0]
            for item in contract_list[1:]:
                contractable = contractable * item

        return contractable

class PeriodicBC(Reducible):
    """
    A list of reducibles with periodic boundary conditions

    Calling reduce on a PeriodicBC instance will proceed as in LinearRegion, 
    followed by a trace over the left and right bonds to get output tensor
    """
    def __init__(self, reducible_list):
        # We'll typically get a list, but LinearRegions are OK too
        if isinstance(reducible_list, list):
            self.linear_region = LinearRegion(reducible_list)
        elif instance(reducible_list, LinearRegion):
            self.linear_region = reducible_list
        else:
            raise TypeError

    def reduce(self):
        # Contract linear region to an irreducible contractable
        contractable = self.linear_region.reduce(open_bc=False)
        tensor = contractable.tensor
        bond_string = contractable.bond_string

        # It takes two different indices to trace over the output
        assert 'l' in bond_string and 'r' in bond_string

        # Build einsum string for trace of our tensor
        in_str, out_str = "", ""
        for c in bond_string:
            if c in ['l', 'r']:
                in_str += 'l'
            else:
                in_str += c
                out_str += c
        ein_str = in_str + "->" + out_str

        # Return the trace over linear indices
        return torch.einsum(ein_str, tensor)

class Open(Reducible):
    """
    A list of reducibles with open boundary conditions

    Calling reduce on an OpenBC instance will introduce SingleVec instances on
    both ends, before contracting all the items in the linear region. 
    """
    def __init__(self, reducible_list):
        # We'll typically get a list, but LinearRegions are OK too
        if isinstance(reducible_list, LinearRegion):
            self.reducible_list = reducible_list.reducible_list
        elif isinstance(reducible_list, list):
            self.reducible_list = reducible_list
        else:
            raise TypeError

    def reduce(self, parallel_reduce=False):
        # Add terminal vectors on each end
        red_list = self.reducible_list

        # Get size of left and right bond dimensions
        left_item, right_item = red_list[0], red_list[-1]
        left_ind, right_ind = left_item.index('l'), right_item.index('r')
        left_D, right_D = left_item.size(left_ind), right_item.size(right_ind)
        global_bs = left_item.global_bs

        # Build dummy end vectors and insert them at the ends of our list
        left_vec, right_vec = torch.zeros([left_D]), torch.zeros([right_D])
        left_vec[0], right_vec[0] = 1, 1
        red_list.insert(0, EdgeVec(left_vec, is_left_vec=True))
        red_list.append(EdgeVec(right_vec, is_left_vec=False))

        # Multiply everything in the list together to get our output
        return red_list.reduce(open_bc=True).tensor
