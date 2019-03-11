# TorchMPS: Matrix Product States in Pytorch

TorchMPS is a framework for working with matrix product state (aka MPS, 
aka tensor train) models within Pytorch. Our MPS models are written 
as Pytorch Modules, and so can simply be viewed as differentiable black 
boxes that are interchangeable with standard neural network layers. However, 
the rich structure of MPS's allows for more interesting behavior, such as:

 * A novel adaptive training algorithm (inspired by [Stoudenmire and Schwab 2016][S&S]),
   which dynamically varies the MPS hyperparameters (MPS bond dimensions) during 
   the course of training.
 * Mechanisms for controlling MPS geometry, such as custom "routing" of the MPS 
   through different regions of the input data, or periodic boundary conditions 
   that give the MPS a circular topology.
 * Choice of tensor contraction strategies, which represent flexible tradeoffs 
   between computational cost and parallelizability of the computation graph.

## What our MPS Models Do

The function computed by our MPS Module comes from embedding 
input data in a high-dimensional feature space before contracting it with an 
MPS living in this space, as first described in [Novikov, Trofimov, and Oseledets 2016] 
and [Stoudenmire and Schwab 2016]. For scalar outputs, this contraction step is
formally identical to linear regression, but the (exponentially) large feature space and
MPS-parameterized weight vector makes the overall function significantly more
expressive. In general, the output is associated with a single site of the MPS,
whose placement within the network is a hyperparameter that varies the inductive
bias towards different regions of the input data.

## How to Use

As our models are built on Pytorch, users will need to have this installed and
available in PYTHONPATH. Torchvision is also used in our example script 
`train_script.py`, but not anywhere else.

After cloning the repo, running `train_script.py` on the command
line shows how our MPS can be used as a classifier for images from the MNIST dataset. 
More generally, MPS models can be invoked by simply importing 
the class `MPS` from `torchmps.py`, and then creating a new `MPS` instance. For 
example, an MPS which classifies 32x32 images into one of 10 categories can be 
created and used as follows:

```
from torchmps.py import MPS

my_mps = MPS(input_dim=32**2, output_dim=10, bond_dim=16)

# Now get a batch of (flattened) images to classify...

batch_scores = my_mps(batch_images)
```

That's all! After creation, `my_mps` acts as a stateful function whose internal parameters can be trained exactly as any other Pytorch Module (e.g. nn.Linear, nn.Conv1d, nn.Sequential, etc).

The arguments given to MPS are:

 * `input_dim`: The dimension of the input we feed to our MPS
 * `output_dim`: The dimension of the output we get from each input
 * `bond_dim`: The internal bond dimension, a hyperparameter that sets the
   expressivity of our MPS. When in adaptive training mode, `bond_dim`
   instead specifies the **maximum** possible bond dimension, with the initial
   bond dimension set to half of `bond_dim`
 * `feature_dim`: The dimension of the local feature spaces we embed each datum
   in (_default = 2_)
 * `adaptive_mode`: Whether our MPS is trained with its bond dimensions chosen
   adaptively or are fixed at creation (_default = False (fixed bonds)_)
 * `periodic_bc`: Whether our MPS has periodic boundary conditions (making it 
   a tensor ring) or open boundary conditions (_default = False (open boundaries)_)
 * `parallel_eval`: For open boundary conditions, whether contraction of tensors
   is performed serially or in parallel (_default = False (serial)_)
 * `label_site`: The location in the MPS chain where our output lives after
   contracting all other sites with inputs (_default = input_dim // 2_)
 * `path`: A list specifying the path our MPS takes through the input data. For
   example, `path = [0, 1, ..., input_dim-1]` gives the standard in-order 
   traversal (used if `path = None`), while `path = [0, 2, ..., input_dim-1]`
   defines an MPS which only acts on even-valued sites within our input 
   (_default = None_)
 * `cutoff`: The singular value cutoff which controls adaptation of bond 
   dimensions (_default = 1e-9_)
 * `merge_threshold`: The number of inputs before our MPS dynamically shifts
   its merge state, which updates half the bond dimensions (_default = 2000, 
   only used in adaptive mode_)
 * `init_std`: The size of the random terms used during initialization 
   (_default = 1e-9_)

## Similar Software

[NEED TO EXPAND MORE HERE]

    * T3F (written using TensorFlow)
    * TNML (written using ITensor)
    * tntorch (written using Pytorch)
    * scikit_tt (written in Python)

[S&S]: https://arxiv.org/abs/1605.05775
[NTO]: https://arxiv.org/abs/1605.03795
