#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from math import ceil

""" TODO ITEMS
    * 
"""

"""
VALID_ARGS = {'args':   If we want to pass arguments as a single dictionary,
                        include them as key: value pairs in args,
                 'd':   Dimension of the local embedding space at each site,
                'bc':   Boundary conditions, either 'open' (default) or 
                        'periodic',
        'label_site':   Location of label core in MPS, must satisfy
                        0 <= label_site <= size (default = size // 2),
'weight_init_method':   Preferred method for initializing our core weights,
                        can be either 'random' or 'random_eye' (default),
 'weight_init_scale':   The standard deviation for initialization methods,
        'train_mode':   Initial training mode, either 'static' (default) or
                        'dynamic'}
"""

class MPSModule(nn.Module):
    def __init__(self, size, D=20, output_dim=10, **args):
        """
        Define variables for holding our trainable MPS cores
        """
        super(MPSModule, self).__init__()

        # Number of sites in the MPS
        self.size = size
        # Global (maximum) bond dimension
        self.D = D 
        # Dimension of our module output (when used for classification,
        # this is the number of classification labels)
        self.output_dim = output_dim

        # If args includes a dict `args`, add its contents to args
        if 'args' in args:
            args.update(args['args'])
            del args['args']

        # Dimension of local embedding space
        if 'd' in args:
            d = args['d']
        else:
            d = 2
        self.d = d

        # Specify open or periodic boundary conditions for the MPS
        if 'bc' in args:
            bc = args['bc']
            if args['bc'] not in ['open', 'periodic']:
                raise ValueError("Unrecognized value for option 'bc': "
                                f"{args['bc']}, must be 'open' or 'periodic'")
        else:
            bc = 'open'
        self.bc = bc

        # Set location of the label site within the MPS
        if 'label_site' in args:
            label_site = args['label_site']
            if not (0 <= label_site and label_site <= size):
                raise ValueError("label_site must be between 0 and size="
                                 f"{size}, but input label_site={label_site}")
        else:
            label_site = size // 2
        self.label_site = label_site

        # Information about the default shapes of our tensors
        full_shape = [size, D, D, d]
        base_shapes = size * [[D, D, d]]
        label_shape = [D, D, output_dim]

        # Method used to initialize our tensor weights, either
        # 'random' or 'random_eye' (default 'random_eye')
        if 'weight_init_method' in args:
            if 'weight_init_scale' in args:
                init_method = args['weight_init_method']
                init_std = args['weight_init_scale']
            else:
                raise ValueError("If 'weigh_init_method' is set, "
                            "'weight_init_scale' must also be set")
        else:
            init_method = 'random_eye'
            init_std = 0.01

        # Initialize the base MPS cores, which live on input sites,
        # and the label core, which outputs label predictions.
        # Note that 'random' sets our tensors completely randomly,
        # while 'random_eye' sets them close to the identity
        if init_method == 'random':
            base_cores = init_std * torch.randn(full_shape)
            label_core = init_std * torch.randn(label_shape)

        elif init_method == 'random_eye':
            base_cores = torch.eye(D).view([1, D, D, 1]).expand(full_shape)
            label_core = torch.eye(D).unsqueeze(2).expand(label_shape)

            base_cores = base_cores + init_std * torch.randn(full_shape)
            label_core = label_core + init_std * torch.randn(label_shape)
            
        # Adjust for open boundary conditions by projecting the first and last
        # core tensors onto a 1D subspace and update the shape information
        if bc == 'open':
            if label_site != 0:
                base_cores[0, 1:] = 0
                base_shapes[0] = [1, D, d]
            else:
                label_core[1:] = 0
                label_shape = [1, D, output_dim]

            if label_site != size:
                base_cores[-1, :, 1:] = 0
                base_shapes[-1] = [D, 1, d]
            else:
                label_core[:, 1:] = 0
                label_shape = [D, 1, output_dim]
        
        self.base_shapes = base_shapes
        self.label_shape = torch.tensor(label_shape)
        self.base_cores = nn.Parameter(base_cores)
        self.label_core = nn.Parameter(label_core)

        # train_mode is either 'static' or 'dynamic', while dynamic_mode
        # can be 'no_merge', 'merge_left', or 'merge_right'
        if 'train_mode' in args:
            train_mode = args['train_mode']
            if train_mode not in ['static', 'dynamic']:
                raise ValueError("Unrecognized value for option 'train_mode': "
                                f"{train_mode}, must be 'static' or 'dynamic'")
        else:
            train_mode = 'static'

        self.train_mode = train_mode
        self.dynamic_mode = 'no_merge'

        # Effective size and label site after merging cores in dynamic mode
        self.dyn_size = (size + (1 if bc == 'open' else 0)) // 2
        self.dyn_label_site = label_site // 2
        dyn_full_shape = [self.dyn_size, D, D, d, d]

        # Merged cores for dynamic train mode (set after mode switch)
        self.dyn_base_shapes = self.dyn_size * [[D, D, d, d]]
        self.dyn_label_shape = [D, D, output_dim, d]
        self.dyn_base_cores = nn.Parameter(torch.zeros(dyn_full_shape))
        self.dyn_label_core = nn.Parameter(torch.zeros(self.dyn_label_shape))

        # Each input datum increments train_counter by 1, and after reaching
        # toggle_threshold, dynamic_mode is toggled between 'even' and 'odd'
        self.train_counter = 0
        self.toggle_threshold = 1000

        # Sets truncation during un-merging process
        self.svd_cutoff = 1e-10

        # Make sure our merged cores are properly initialized
        if train_mode == 'dynamic':
            self._update_dynamic_mode()

    def _contract_batch_input(self, batch_input):
        """
        Contract input data with MPS cores, return matrices and label tensor.

        Args:
            batch_input (Tensor): Input data, with shape of [batch_size, size].
                Optionally, if batch_input has size [batch_size, size, d], then
                input is already assumed to be in embedded form.

        Returns:
            base_mats (Tensor): Data-dependent matrices coming from contraction
                of input data with base cores of MPS, has shape of
                [mode_size, batch_size, D, D], where mode_size is either `size`
                or `dyn_size`, depending on train_mode.
            label_cores (Tensor): Batch of label cores with shape of 
                [batch_size, output_dim, D, D]. In 'static' dynamic mode, this
                is just a permuted and expanded version of self.label_core, but
                in the presence of core merging this output is data-dependent
        """
        size = self.size
        D, d = self.D, self.d
        output_dim = self.output_dim

        batch_shape = batch_input.shape
        batch_size = batch_shape[0]

        # Interchange batch and site indices, and embed data if necessary
        if len(batch_shape) == 2:
            batch_input = batch_input.permute([1, 0]).contiguous()
            batch_input = self._embed_batch_input(batch_input)
        else:
            batch_input = batch_input.permute([1, 0, 2]).contiguous()

        # If train_mode is static, then contraction is pretty straightforward.
        # Just massage the shapes of our base cores and embedded data and
        # batch multiply them all together
        if self.train_mode == 'static' or self.dynamic_mode == 'no_merge':
            batch_input = batch_input.view([size*batch_size, d, 1])
            
            base_cores = self.base_cores.view([size, 1, D*D, d])
            base_cores = base_cores.expand([size, batch_size, D*D, d])
            base_cores = base_cores.contiguous()
            base_cores = base_cores.view([size*batch_size, D*D, d])

            # The actual contraction of inputs with (reshaped) base cores
            base_mats = torch.bmm(base_cores, batch_input)
            base_mats = base_mats.view([size, batch_size, D, D])
            
            label_cores = self.label_core.permute([2, 0, 1]).unsqueeze(0)
            label_cores = label_cores.expand([batch_size, output_dim, D, D])
            label_cores = label_cores.contiguous()

            return base_mats, label_cores

        # If train_mode is dynamic, we need to contract input data with the
        # merged cores. This involves some tricky geometric details regarding 
        # the different ways we can merge the cores of our MPS, and the 
        # different (literal) edge cases which emerge at the ends of the MPS
        else:
            bc = self.bc
            dyn_size = self.dyn_size
            label_site = self.label_site
            dynamic_mode = self.dynamic_mode
            dyn_label_site = self.dyn_label_site

            # Flags that indicate different edge cases in the MPS geometry
            lone_left_core, lone_right_core = False, False
            lone_label_core, wrap_around = False, False

            if dynamic_mode == 'merge_left':
                special_site = 2 * (label_site // 2)
                
                # Even number of base cores leaves the last core unmerged
                if size % 2 == 0:
                    if label_site == size:
                        lone_label_core = True
                    else:
                        lone_right_core = True

            elif dynamic_mode == 'merge_right':
                special_site = 2 * ((label_site - 1) // 2) + 1

                # Even number of base cores leaves the first core unmerged
                if size % 2 == 0:
                    if label_site == 0:
                        lone_label_core = True
                    else:
                        lone_left_core = True

                # Odd number of base cores picks up all the base cases, which
                # is either wrap-around, or else an unmerged core on each end
                elif size % 2 == 1:
                    if bc == 'periodic':
                        wrap_around = True
                        special_site = special_site % size
                    elif label_site == 0:
                        lone_label_core = True
                        lone_right_core = True
                    elif label_site == size:
                        lone_label_core = True
                        lone_left_core = True
                    else:
                        lone_left_core = True
                        lone_right_core = True

            # Get label_cores, usually by contraction with special_site pixel
            label_cores = self.dyn_label_core.permute([2,0,1,3]).unsqueeze(0)
            label_cores = label_cores.expand([batch_size, output_dim, D, D, d])
            label_cores = label_cores.contiguous()

            if not lone_label_core:
                special_batch = batch_input[special_site].unsqueeze(2)
                label_cores = label_cores.view([batch_size, output_dim*D*D, d])
                label_cores = torch.bmm(label_cores, special_batch)

                label_cores = label_cores.view([batch_size, output_dim, D, D])
            else:
                label_cores = label_cores[:, :, :, :, 0]

            # Until return statement, the following generates base_mats using 
            # a contraction of our input pixels with the merged cores 
            base_cores = self.dyn_base_cores.unsqueeze(1)
            base_cores = base_cores.expand([dyn_size, batch_size, D, D, d, d])
            base_cores = base_cores.contiguous()
            base_cores = base_cores.view([dyn_size*batch_size, D*D*d, d])
            
            # Split our input pixels into those contracted on sites with even
            # vs odd (i.e. left vs right) parity relative to our merge grouping
            if dynamic_mode == 'merge_left':
                even_pixels = [batch_input[0:special_site:2]]
                odd_pixels = [batch_input[1:special_site:2]]
            elif dynamic_mode == 'merge_right':
                even_pixels = [batch_input[1:special_site:2]]
                odd_pixels = [batch_input[2:special_site:2]]
            even_pixels.append(batch_input[special_site + 1:size:2])
            odd_pixels.append(batch_input[special_site + 2:size:2])

            # Filler data for padding odd_pixels if we have edge cases
            padding_pixel = torch.zeros([1, 1, d])
            padding_pixel[0, 0, 0] = 1
            padding_pixel = padding_pixel.expand([1, batch_size, d])
            
            # Adjust even_pixels and odd_pixels to handle different edge cases
            if lone_left_core:
                even_pixels.insert(0, padding_pixel)
                odd_pixels.insert(0, batch_input[:1])
            if lone_right_core:
                odd_pixels.append(padding_pixel)
            if wrap_around:
                odd_pixels.append(batch_input[:1])

            # Bring our pixels together and reshape for batch multiplication
            even_pixels = torch.cat(even_pixels)
            odd_pixels = torch.cat(odd_pixels)

            even_pixels = even_pixels.view([dyn_size*batch_size, d, 1])
            odd_pixels = odd_pixels.view([dyn_size*batch_size, d, 1])
            
            # Contract base_cores with odd_pixels, then even_pixels
            base_cores = torch.bmm(base_cores, odd_pixels)
            base_cores = base_cores.view([dyn_size*batch_size, D*D, d])
            base_mats = torch.bmm(base_cores, even_pixels)

            return base_mats.view([dyn_size, batch_size, D, D]), label_cores

    def _embed_batch_input(self, batch_input):
        """
        Embed input data using a d-dimensional embedding map on each pixel.

        If the user has specified an embedding_map() method for the module,
        then this map is used, otherwise use the d=2 linear map x -> [1-x, x].

        Args:
            batch_input (Tensor): Input data with shape [size, batch_size].

        Returns:
            embedded_data (Tensor): Embedded images of input data, with shape
                [size, batch_size, d].
        """
        d = self.d
        size = batch_input.shape[0]
        batch_size = batch_input.shape[1]
        batch_input = batch_input.view(size * batch_size)
        
        if 'embedding_map' in dir(self):
            embedded_data = [self.embedding_map(pixel).unsqueeze(0) 
                             for pixel in batch_input]
            embedded_data = torch.stack(embedded_data, 0)
        else:
            assert d == 2
            embedded_data = torch.empty([size * batch_size, d])
            embedded_data[:, 0] = 1 - batch_input
            embedded_data[:, 1] = batch_input

        return embedded_data.view([size, batch_size, d])

    def forward(self, batch_input):
        """
        Evaluate module on a batch of input data and return vector output.

        Args:
            batch_input (Tensor): Input data with shape [batch_size, size], or
                optionally [batch_size, size, d]. In the former case the 
                individual inputs are embedded using an affine map (or user-
                defined embedding function), otherwise they are used as is.

        Returns:
            batch_output (Tensor): Output data with shape 
                [batch_size, output_dim].
        
        TODO:
            (1) [BIG TASK] Check contract_mode to see if I should run the 
                current algorithm ('parallel'), or a new 'serial' algorithm
                (1a) Write serial contraction algorithm with batch parallelism
                (1b) See if you can refactor everything so the two contraction 
                     algorithms are stored as two different (internal) methods
        """
        size, D, d = self.size, self.D, self.d
        output_dim = self.output_dim
        train_mode = self.train_mode

        # Get input shape and check that it's valid
        input_shape = batch_input.shape
        batch_size = input_shape[0]
        if input_shape[1] != size or \
            (len(input_shape) == 3 and input_shape[2] != d):
            return ValueError(f"batch_input has shape {list(input_shape)}, but"
                              f" must be either [{batch_size}, {size}] or "
                              f"[{batch_size}, {size}, {d}]")

        # Contract batch_input with MPS cores to get matrices and label cores
        base_mats, label_cores = self._contract_batch_input(batch_input)

        # Depending on our training mode, we might have matrices/tensors which
        # correspond to a different MPS geometry, so adjust for that here
        if train_mode == 'dynamic':
            size = self.dyn_size
            label_site = self.dyn_label_site
            self._update_dynamic_mode()
        else:
            label_site = self.label_site

        # Divide into regions left and right of the label site
        left_mats = base_mats[:label_site]
        right_mats = base_mats[label_site:]
        all_mats = [left_mats, right_mats]

        # Number of matrices on the left and right products, which decreases 
        # by about half on each successive batch matrix multiplication
        left_length = label_site
        right_length = size - label_site
        lengths = [left_length, right_length]
        
        # Iteratively multiply nearest neighboring pairs of matrices until the 
        # left and right regions are reduced to (at most) a single matrix each
        while max(lengths) > 1:
            for s in range(2): 
                if lengths[s] > 1:
                    # Unpack our length and matrices for this case
                    length = lengths[s]
                    mats = all_mats[s]
                    leftover_mat = (length % 2) == 1
                    length //= 2
                
                    # Set aside leftover matrix when length is odd
                    lone_mats = torch.tensor([])
                    if leftover_mat:
                        lone_mats = mats[-1:]
                        mats = mats[:-1]

                    # Divide matrices into neighboring pairs and 
                    # contract all pairs using batch multiplication
                    even_mats = mats[0::2].contiguous()
                    odd_mats = mats[1::2].contiguous()
                    even_mats = even_mats.view([length*batch_size, D, D])
                    odd_mats = odd_mats.view([length*batch_size, D, D])
                    mats = torch.bmm(even_mats, odd_mats)
                    
                    # Append leftover matrix
                    mats = mats.view([length, batch_size, D, D])
                    mats = torch.cat([mats, lone_mats])
                    length += 1 if leftover_mat else 0

                    lengths[s] = length
                    all_mats[s] = mats


        # The following just contracts left_mats, label_cores, and right_mats
        # using expanding, reshaping, and batch multiplication
        left_mats, right_mats = all_mats
        label_cores = label_cores.view([batch_size*output_dim, D, D])

        # Left contraction
        if left_mats.nelement() > 0:
                left_mats = left_mats.squeeze().unsqueeze(1).expand(
                                        [batch_size, output_dim, D, D])
                left_mats = left_mats.contiguous().view(
                                        [batch_size*output_dim, D, D])
                label_cores = torch.bmm(left_mats, label_cores)

        # Right contraction
        if right_mats.nelement() > 0:
                right_mats = right_mats.squeeze().unsqueeze(1).expand(
                                        [batch_size, output_dim, D, D])
                right_mats = right_mats.contiguous().view(
                                        [batch_size*output_dim, D, D])
                label_cores = torch.bmm(label_cores, right_mats)
        
        # Taking the partial trace over the bond indices gives the outputs in
        # a manner which works for both open and periodic boundary conditions
        label_cores = label_cores.view([batch_size, output_dim, D*D])
        eye_vecs = torch.eye(D).view(D*D).unsqueeze(0).expand(
                                [batch_size, D*D]).unsqueeze(2)

        batch_output = torch.bmm(label_cores, eye_vecs).squeeze()

        # Update our counter to reflect the new inputs
        if train_mode == 'dynamic':
            self.train_counter += batch_size

        return batch_output

    def num_correct(self, input_data, labels, batch_size=100):
        """
        Use our module as a classifier to predict the labels associated with a
        batch of input data, then compare with the correct labels and return
        the number of correct guesses. For the sake of memory, the input is 
        processed in batches of size batch_size
        """
        bs = batch_size
        num_inputs = input_data.size(0)
        num_batches = ceil(num_inputs / bs)
        num_corr = 0.

        for b in range(num_batches):
            if b == num_batches-1:
                # Last batch might be smaller
                batch_input = input_data[b*bs:]
                batch_labels = labels[b*bs:]
            else:
                batch_input = input_data[b*bs:(b+1)*bs]
                batch_labels = labels[b*bs:(b+1)*bs]

            with torch.no_grad():
                scores = self.forward(batch_input)
            predictions = torch.argmax(scores, 1)
            
            num_corr += torch.sum(torch.eq(predictions, batch_labels)).float()

        return num_corr

    # def embedding_map(self, datum):
    #     """
    #     Embed a single scalar input into a local feature space of dimension d

    #     embedding_map currently uses a simple affine map of the form 
    #     x -> [1-x, x], but this can be replaced by user-specified embeddings
    #     """
    #     if self.d != 2:
    #         raise ValueError(f"Embedding map needs d=2, but self.d={self.d}")
    #     else:
    #         return torch.tensor([1-datum, datum])
