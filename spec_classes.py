class CoreSpec:
    """
    Specification of a named MPS core or collection of similar MPS cores

    WHAT'S NEEDED FOR THIS CLASS?
    (1) The name of the parameter tensor being defined
    (2) The shape attribute for the size of the Parameter tensor defined by 
        our CoreSpec instance
    (3) bond_indices = [left_ind, right_ind], the location of the left and
        right bond indices within shape ([] if no bonds)
    (4) input_indices = [ind1, ind2, ...], the location of different input 
        indices within shape ([] if no inputs)
    (5) output_indices = [ind1, ind2, ...], the location of different output
        indices within shape ([] if no outputs)
    """
    def __init__(self, name, shape, bond_indices=[], input_indices=[],
                 output_indices=[]):
        self.name = name
        self.shape = shape
        self.bond_indices = bond_indices
        self.input_indices = input_indices
        self.output_indices = output_indices

class MPSSpec:
    """
    Specification of an MPS in terms of its geometry and constituent MPS cores

    WHAT'S NEEDED FOR THIS CLASS?
    (1) A dictionary of CoreSpec objects, one for each parameter tensor
    (2) input_size, the size of a single input to the MPS (not including batch
        dimension). Even if our MPS doesn't use every part of the input, this 
        argument must still agree with the input size
    (3) repackage(), a method which takes our input and rearranges it into a
        dictionary of inputs (one for each parameter tensor) which can each be
        contracted with the corresponding parameter tensor
    (4) contract_input(), a method which takes our repackaged input and returns
        a list of Reducible objects
    """
    def __init__(self, core_specs, input_size, repackage, contract_input):
        self.core_specs = core_specs
        self.input_size = input_size
        self.repackage = repackage
        self.contract_input = contract_input
