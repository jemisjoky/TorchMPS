#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from math import ceil

""" TODO ITEMS
    * Implement mode to allow for variable bond dimensions during training.
      This involves (at least) one new collections of parameters, which handles
      the even-contracted and odd-contracted bonds. I also need a mode flag, 
      called `train_mode`, which is toggled by a change_mode() method. 
      `train_mode` can be either 'static' or 'dynamic', with 'static' what
      I have now, and 'dynamic' involving an alternation between adjusting
      even bonds and odd bonds. This necessarily requires a `train_counter`
      which counts the number of data that are passed to forward(), and 
      automatically changes 'dynamic_mode', a flag which is either 'even' or
      'odd', for adjusting even or odd bonds.

        * Initialize new parameters and flags in __init__(), make sure to
          set defaults appropriately. Before doing this, think of the geometry
          of the new system and how many contracted core tensors you'll need.

        * Write change_mode(train_mode) method, which rearranges our core
          and label tensors so that the relevant parameter is up-to-date.
          In addition to doing some contractions or SVD's (depending on the
          mode change), we also need to toggle the trainability of the 
          different parameters.

        * Change _contract_batch_input() to contract according to our current
          train mode. This will now return not just base matrices, but also
          our label matrix. It is also where most of the geometry of the 
          problem comes in, so work out those details first (during __init__
          modifications)

        * Make any necessary changes to forward(). The actual evaluation
          should be pretty similar, but I need to add some code which checks
          and updates `train_counter` and toggles dynamic_mode if we've 
          passed our threshold.

    * Add different contract modes, which can be either 'parallel' or 
      'serial'. I currently am using the former, which is the only possible 
      mode in the presence of periodic boundary conditions. 
      However, 'serial' is D times faster (if less parallelizable). Working 
      with just one GPU, the batch-level parallelization is probably enough.

    * Add 'no_embed' option, where our batch inputs are taken as vectors and
      no embedding map is explicitly defined

"""

class MPSModule(nn.Module):
    def __init__(self, size, D=20, d=2, output_dim=10, **args):
        """
        Define variables for holding our trainable MPS cores
        """
        super(MPSModule, self).__init__()

        # Number of sites in the MPS
        self.size = size
        # Global (maximum) bond dimension
        self.D = D 
        # Dimension of local embedding space
        self.d = d
        # Dimension of our module output (when used for classification,
        # this is the number of classification labels)
        self.output_dim = output_dim

        # If args includes a dict `args`, add its contents to args
        if 'args' in args.keys():
            args.update(args['args'])
            del args['args']

        # Specify open or periodic boundary conditions for the MPS
        if 'bc' in args.keys():
            if args['bc'] in ['open', 'periodic']:
                bc = args['bc']
            else:
                raise ValueError("Unrecognized value for option 'bc': "
                                 f"{args['bc']}")
        else:
            bc = 'open'
        self.bc = bc

        # Set location of the label site within the MPS
        if 'label_site' in args.keys():
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
        if 'weight_init_method' in args.keys():
            if 'weight_init_scale' in args.keys():
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
        self.train_mode = 'static'
        self.dynamic_mode = 'no_merge'

        # Effective size and label site after merging cores in dynamic mode
        self.dyn_size = (size + (1 if bc == 'periodic' else 0)) // 2
        self.dyn_label_site = label_site // 2
        dyn_full_shape = [self.dyn_size, D, D, d, d]

        # Merged cores for dynamic train mode (set after mode switch)
        self.dyn_base_shapes = self.dyn_size * [[D, D, d, d]]
        self.dyn_label_shape = [D, D, output_dim, d]
        self.dyn_base_cores = nn.Parameter(torch.empty(dyn_full_shape))
        self.dyn_label_core = nn.Parameter(torch.empty(self.dyn_label_shape))

        # Each input datum increments train_counter by 1, and after reaching
        # toggle_threshold, dynamic_mode is toggled between 'even' and 'odd'
        self.train_counter = 0
        self.toggle_threshold = 1000

        # Sets truncation during un-merging process
        self.svd_cutoff = 1e-10

    def _contract_batch_input(self, batch_input):
        """
        Contract input data with MPS cores, return matrices and label tensor.

        Args:
            batch_input (Tensor): Input data, with shape of [batch_size, size].
                Optionally, if batch_input has size [batch_size, size, d], then
                input is already assumed to be in embedded form, which skips a
                call to _embed_batch_input.

        Returns:
            base_mats (Tensor): Data-dependent matrices coming from contraction
                of input data with base cores of MPS, has shape of
                [mode_size, batch_size, D, D], where mode_size is either `size`
                or `dyn_size`, depending on train_mode.
            label_cores (Tensor): Batch of label cores with shape of 
                [batch_size, output_dim, D, D]. In 'static' dynamic mode, this
                is just a permuted and expanded version of self.label_core, but
                in the presence of core merging this output is data-dependent

        TODO:
            (1) [IMPORTANT] Refactor to make the site index dominant
            (2) Check shape of input and call _embed_batch_input if need be
            (3) If we're in a merged dynamic mode, iterate through pairs of
                sites and use .bmm() to merge pairs. Condition on label_site
            (4) Stack outputs as base_mats, return with site index dominant
        """
        size = self.size
        D, d = self.D, self.d
        dyn_size = self.dyn_size
        label_site = self.label_site
        output_dim = self.output_dim

        batch_shape = batch_input.shape
        batch_size = batch_shape[0]

        # Interchange batch and site indices, and embed data if necessary
        if len(batch_shape) == 2:
            batch_input = batch_input.permute([1, 0]).contiguous()
            batch_input = self._embed_batch_input(batch_input)
        else:
            batch_input = batch_input.permute([1, 0, 2])
        batch_input = batch_input.contiguous().view([size*batch_size, d, 1])

        # If train_mode is static, then contraction is pretty straightforward.
        # Just massage the shapes of our base cores and embedded data and
        # batch multiply them all together
        if self.train_mode == 'static' or self.dynamic_mode == 'no_merge':
            batch_mats = self.base_cores.view([size, 1, D**2, d])
            batch_mats = batch_mats.expand([size, batch_size, D*D, d])
            batch_mats = batch_mats.contiguous()
            batch_mats = batch_mats.view([size*batch_size, D*D, d])

            base_mats = torch.bmm(batch_mats, batch_input)
            base_mats = base_mats.view([size, batch_size, D, D])
            
            label_cores = self.label_core.permute([2, 0, 1]).unsqueeze(0)
            label_cores = label_cores.expand([batch_size, output_dim, D, D])
            label_cores = label_cores.contiguous()

            return base_mats, label_cores

        # If train_mode is dynamic, we need to iterate by site index and do
        # batch multiplications only over batch index. This section invokes all
        # of the merged geometry of the MPS
        else:
            pass

            return base_mats, label_cores

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
            (0) Check value of train_counter and possibly switch dynamic modes 
                (0a) Write method to change train modes
                (0b) Write method to change dynamic modes
            (1) If dynamic mode is merged, load the correct size and label site
            (2) Account for _*_batch_input() methods in site dominant format
            
            (3) [BIG TASK] Check contract_mode to see if I should run the 
                current algorithm ('parallel'), or a new 'serial' algorithm
                (3a) Write serial contraction algorithm with batch parallelism
                (3b) See if you can refactor everything so the two contraction 
                     algorithms are stored as two different (internal) methods
            
            (4) (Output should remain the same with all of these changes)
        """
        size, D, d = self.size, self.D, self.d
        output_dim = self.output_dim
        label_site = self.label_site

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

        # REWRITE FOLLOWING IN TERMS OF base_mats AND label_cores


        # Divide into regions left and right of the label site
        left_mats = base_mats[:label_site]
        right_mats = base_mats[label_site:]

        # Size of the left and right matrix products, which decrease by half on 
        # each iteration below
        left_size = label_site
        right_size = size - label_site
        
        # lr_size and lr_mats let us treat both sides in a uniform manner
        lr_size = [left_size, right_size]
        lr_mats = [left_mats, right_mats]

        # Iteratively multiply nearest neighboring pairs of matrices until we've
        # reduced the left and right regions to a single matrix each
        while max(lr_size) > 1:
            # s is the left/right index
            for s in [s for s in range(2) if lr_size[s] > 1]:
                # size (or mats) is either left_size or right_size
                size = lr_size[s]
                mats = lr_mats[s]
                odd_size = (size % 2) == 1
                size = size // 2
            
                # If our size is odd, set aside extra matrix to make size even
                if odd_size:
                    lone_mats = mats[-1].unsqueeze(0)
                    mats = mats[:-1]
                else:
                    lone_mats = None

                # Divide matrices into neighboring pairs and 
                # contract all pairs using batch multiplication
                mats1, mats2 = mats[0::2].contiguous(), mats[1::2].contiguous()
                mats1 = mats1.view([size*batch_size, D, D])
                mats2 = mats2.view([size*batch_size, D, D])
                mats = torch.bmm(mats1, mats2)
                
                # Reshape and append any leftover matrices
                mats = mats.view([size, batch_size, D, D])
                if odd_size:
                    size += 1
                    mats = torch.cat([mats, lone_mats], 0)

                lr_size[s] = size
                lr_mats[s] = mats

        # For each input, we now have (at most) one matrix on the left and
        # (at most) one on the right (empty if `label_site` is 0 or `size`)
        # We now contract these three (two) objects to obtain our output

        # Expand and reshape the (nonempty) left and right matrices
        lr_stack = [None, None]
        for s in range(2):
            if lr_mats[s].nelement() > 0:
                lr_stack[s] = lr_mats[s].squeeze().unsqueeze(1).expand(
                                       [batch_size, output_dim, D, D])
                lr_stack[s] = lr_stack[s].contiguous().view(
                                        [batch_size*output_dim, D, D])
        left_stack, right_stack = lr_stack

        # Perform the actual contraction of nonempty mats with the label tensor
        label_cores = label_cores.view([batch_size*output_dim, D, D])
        if left_stack is not None:
            label_cores = torch.bmm(left_stack, label_cores)
        if right_stack is not None:
            label_cores = torch.bmm(label_cores, right_stack)
        
        # Taking the partial trace over the bond indices gives the outputs in
        # a way which works for both open and periodic boundary conditions
        label_cores = label_cores.view([batch_size, output_dim, D*D])
        eye_vecs = torch.eye(D).view(D*D).unsqueeze(0).expand(
                                [batch_size, D*D]).unsqueeze(2)
        batch_output = torch.bmm(label_cores, eye_vecs)

        return batch_output.squeeze()

    def change_train_mode(self, new_mode):
        pass

    def _change_dynamic_mode(self, new_mode):
        pass

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
