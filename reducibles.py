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
from contractables import LinearContractable

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
    in a manner set by the parallel_reduce and right_to_left options
    """
    def __init__(self, reducible_list):
        if not reducible_list:
            raise ValueError("Input to LinearRegion must be nonempty list")

        for i, item in enumerate(reducible_list):
            if not isinstance(item, Reducible):
                raise ValueError("Inputs to LinearRegion must be reducibles, "
                                f"but item {i} is not")

        self.reducible_list = reducible_list

    def reduce(self, parallel_reduce=True, right_to_left=False):
        """
        Reduce all the reducibles in our list before multiplying them together

        If parallel_reduce is False, reduce is only called on items which
        aren't already a LinearContractable. If right_to_left is True, the 
        multiplication proceeds in right-to-left order
        """
        contract_list = []

        # Reduce our reducibles and put the outputs into contract_list
        for item in reducible_list:
            if parallel_reduce or not isinstance(item, LinearContractable):
                item = item.reduce()
            contract_list.append(item)

        # Reverse if we're multiplying right-to-left
        if right_to_left:
            contract_list = contract_list[::-1]

        # Now multiply together all the contractables in contract_list
        contractable = contract_list[:1]
        for item in contract_list[1:]:
            contractable = contractable * item

        assert isinstance(contractable, LinearContractable)
        return contractable

class PeriodicBC(Reducible):
    """
    A list of reducibles with periodic boundary conditions

    Calling reduce on a PeriodicBC instance will proceed in the same manner as
    LinearRegion, followed by tracing over the left and right bonds to obtain
    an output contractable
    """
    def __init__(self, reducible_list):
        # We'll typically get a list, although it's possible to feed in a 
        # LinearRegion instance instead
        if not isinstance(reducible_list, LinearRegion):
            self.linear_region = LinearRegion(reducible_list)
        else:
            self.linear_region = reducible_list

    def reduce(self):
        contractable = self.linear_region.reduce()
        tensor = contractable.tensor

        # Check the type of my linear contractable and trace appropriately
        if isinstance(contractable, SingleMat):
            scalar = torch.einsum("Bll->B", tensor)
            return Scalar(scalar)

        elif isinstance(contractable, OutputCore):
            vector = torch.einsum("Boll->Bo", tensor)
            return OutputVec(vector)

        # TODO: Write a general routine that works for other contractables
        else:
            raise NotImplementedError

class Open(Reducible):
    """
    A list of reducibles with open boundary conditions

    Calling reduce on an OpenBC instance will introduce SingleVec instances on
    both ends, before contracting all the  in the same manner as
    LinearRegion, followed by tracing over the left and right bonds to obtain
    an output contractable
    """
    def __init__(self, reducible_list):
        self.linear_region = LinearRegion(reducible_list)

    def reduce(self):
        contractable = self.linear_region.reduce()
        tensor = contractable.tensor

        # Check the type of my linear contractable and trace appropriately
        if isinstance(contractable, SingleMat):
            scalar = torch.einsum("Bll->B", tensor)
            return Scalar(scalar)

        elif isinstance(contractable, OutputCore):
            vector = torch.einsum("Boll->Bo", tensor)
            return OutputVec(vector)

        # TODO: Write a general routine for other contractables
        else:
            raise NotImplementedError

