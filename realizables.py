"""
TODO:
    * Write base Realizable class, which has the method realize()
      
    * Implement the following realizable subclasses:

        (1) InputCore, holding a trainable MPS core which takes a single input
        (2) InputRegion, holding a region of trainable MPS cores, which takes
            in an input of length equal to the number of cores
        (3) MergedInput, holding a region of trainable merged MPS cores, which
            takes in an input of length exactly twice the number of cores
        (4) OutputCore, holding a trainable MPS core which doesn't use input
        (5) MergedOutput, which was a trainable MPS core taking in one input,
            coming from a nearby merged site. The location of 

        { (1) returns a SingleMat, (2) and (3) return a MatRegion }

        (6) Region, which holds a list of generic linear reducibles 
            and realizes every one in turn before returning the corresponding
            list of contractables. This needs a list of routing maps, one for 
            each reducible in the list.

        (7) PeriodicString, which has a single BaseRegion and a single 
            OutputCore. These are both realized to give a Periodic reducible

        { For (6) and (7) it looks like there's a close relationship between
          the composite realizable and the composite reducible it spawns.
          I'm assuming this will involve some mindless unpacking and repacking
          attributes, but maybe there's something more systematic possible? }
"""
class Realizable:
    """
    
    """