# Copyright (C) 2021 Jacob Miller
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Uniform and non-uniform probabilistic MPS classes"""
# from typing import Union, Sequence, Optional

import torch
from torch import Tensor, nn

from torchmps.mps_base import contract_matseq, near_eye_init, get_mat_slices
from torchmps.utils2 import phaseify

# TensorSeq = Union[Tensor, Sequence[Tensor]]


class ProbMPS(nn.Module):
    r"""
    Fixed-length MPS model using L2 probabilities for generative modeling

    Probabilities of fixed-length inputs are obtained via the Born rule of
    quantum mechanics, making ProbMPS a "Born machine" model. For a model
    acting on length-n inputs, the probability assigned to the sequence
    :math:`x = x_1 x_2 \dots x_n` is :math:`P(x) = |h_n^T \omega|^2 / Z`,
    where :math:`Z` is a normalization constant and the hidden state
    vectors :math:`h_t` are updated according to:

    .. math::
        h_t = (A_t[x_t] + B) h_{t-1},

    with :math:`h_0 := \alpha` (for :math:`\alpha, \omega` trainable
    parameter vectors), :math:`A_t[i]` the i'th matrix slice of a
    third-order core tensor for the t'th input, and :math:`B` an optional
    bias matrix.

    Note that calling a :class:`ProbMPS` instance with given input will
    return the **logarithm** of the input probabilities, to avoid underflow
    in the case of longer sequences. To get the negative log likelihood
    loss for a batch of inputs, use the :attr:`loss` function of the
    :class:`ProbMPS`.

    Args:
        seq_len: Length of fixed-length discrete sequence inputs. Inputs
            can be either batches of discrete sequences, with a shape of
            `(input_len, batch)`, or batches of vector sequences, with a
            shape of `(input_len, batch, input_dim)`.
        input_dim: Dimension of the inputs to each core. For vector
            sequence inputs this is the dimension of the input vectors,
            while for discrete sequence inputs this is the size of the
            discrete alphabet.
        bond_dim: Dimension of the bond spaces linking adjacent MPS cores,
            which are assumed to be equal everywhere.
        complex_params: Whether model parameters are complex or real. The
            former allows more expressivity, but is less common in Pytorch.
            Default: ``False``
        periodic_bc: Whether MPS has periodic boundary conditions (i.e. is
            a tensor ring) or open boundary conditions (i.e. is a tensor
            train). Default: ``False``
        parallel_eval: Whether to force parallel contraction of tensor
            network. This leads to greater total computational cost, but
            can be faster in the presence of GPUs. Default: ``False``
        use_bias: Whether to use a trainable bias matrix in evaluation.
            Default: ``False``
    """

    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        bond_dim: int,
        complex_params: bool = False,
        periodic_bc: bool = False,
        parallel_eval: bool = False,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        assert min(seq_len, input_dim, bond_dim) > 0

        # Initialize core tensor and edge vectors
        self.core_tensors = near_eye_init(
            (seq_len, input_dim, bond_dim, bond_dim), complex_params
        )
        self.edge_vecs = torch.randn(2, bond_dim) / torch.sqrt(bond_dim)
        if complex_params:
            self.core_tensors = phaseify(self.core_tensors)
            self.edge_vecs = phaseify(self.edge_vecs)

        # Initialize (optional) bias matrices at zero
        if use_bias:
            self.bias_mat = torch.zeros(bond_dim, bond_dim)
            if complex_params:
                self.bias_mat = phaseify(self.bias_mat)

        # Set other MPS attributes
        self.complex_params = complex_params
        self.periodic_bc = periodic_bc
        self.parallel_eval = parallel_eval

    def forward(self, input_data: Tensor) -> Tensor:
        """
        Get the log probabilities of batch of input data

        Args:
            input_data: Sequential with shape `(seq_len, batch)`, for
                discrete inputs, or shape `(seq_len, batch, input_dim)`,
                for vector inputs.

        Returns:
            log_probs: Vector with shape `(batch,)` giving the natural
                logarithm of the probability of each input sequence.
        """
        pass

    def loss(self, input_data: Tensor) -> Tensor:
        """
        Get the negative log likelihood loss for batch of input data

        Args:
            input_data: Sequential with shape `(seq_len, batch)`, for
                discrete inputs, or shape `(seq_len, batch, input_dim)`,
                for vector inputs.

        Returns:
            loss_val: Scalar value giving average of the negative log
                likelihood loss of all sequences in input batch.
        """
        return -torch.mean(self.forward(input_data))

    @property
    def seq_len(self):
        return self.core_tensors.shape[0]

    @property
    def input_dim(self):
        return self.core_tensors.shape[1]

    @property
    def bond_dim(self):
        return self.core_tensors.shape[2]

    @property
    def use_bias(self):
        return hasattr(self, "bias_mat")
