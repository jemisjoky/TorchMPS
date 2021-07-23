# MIT License
#
# Copyright (c) 2021 Jacob Miller
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Uniform and non-uniform probabilistic MPS classes"""
# from typing import Union, Sequence, Optional
from math import sqrt

import torch
from torch import Tensor, nn

from torchmps.mps_base import (
    contract_matseq,
    near_eye_init,
    get_mat_slices,
    get_log_norm,
)
from torchmps.utils2 import phaseify, floor2

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
        parallel_eval: bool = False,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        assert min(seq_len, input_dim, bond_dim) > 0

        # Initialize core tensor and edge vectors
        core_tensors = near_eye_init(
            (seq_len, input_dim, bond_dim, bond_dim), complex_params
        )
        edge_vecs = torch.randn(2, bond_dim) / sqrt(bond_dim)
        if complex_params:
            edge_vecs = phaseify(edge_vecs)
        self.core_tensors = nn.Parameter(core_tensors)
        self.edge_vecs = nn.Parameter(edge_vecs)

        # Initialize (optional) bias matrices at zero
        if use_bias:
            bias_mat = torch.zeros(bond_dim, bond_dim)
            if complex_params:
                bias_mat = phaseify(bias_mat)
            self.bias_mat = nn.Parameter(bias_mat)

        # Set other MPS attributes
        self.complex_params = complex_params
        self.parallel_eval = parallel_eval
        self.rescale_factor = None

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
        # TODO: Convert input to STensors first

        # Contract inputs with core tensors and add bias matrices
        mat_slices = get_mat_slices(input_data, self.core_tensors)
        if self.use_bias:
            mat_slices = mat_slices + self.bias_mat[None, None]

        # Put the batch axis, since contract_matseq expects that
        mat_slices.transpose_(0, 1)

        #  Contract all bond dims to get (unnormalized) prob amplitudes
        psi_vals = contract_matseq(
            mat_slices, self.edge_vecs[0], self.edge_vecs[1], self.parallel_eval
        )

        # Get log normalization and check for infinities
        log_norm = self.log_norm()
        assert log_norm.isfinite()
        assert torch.all(psi_vals.isfinite())

        # Compute unnormalized log probabilities and rescale factor
        log_uprobs = torch.log(torch.abs(psi_vals))
        self.rescale_factor = torch.exp(log_uprobs.mean() / len(input_data))

        # Return normalized probabilities
        return 2 * log_uprobs - log_norm

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
        # Rescale the core tensors and boundary vectors
        if self.rescale_factor is not None:
            state_dict = self.state_dict()
            vec_rescale = floor2(self.edge_vecs.norm(dim=1, keepdim=True))
            new_edgevecs = self.edge_vecs / vec_rescale
            new_coretensors = self.core_tensors / self.rescale_factor
            state_dict["edge_vecs"] = new_edgevecs
            state_dict["core_tensors"] = new_coretensors
            self.load_state_dict(state_dict)

        return -torch.mean(self.forward(input_data))

    def log_norm(self) -> Tensor:
        r"""
        Compute the log normalization of the MPS for its fixed-size input

        Uses iterated tensor contraction to compute :math:`\log(|\psi|^2)`,
        where :math:`\psi` is the n'th order tensor described by the
        contraction of MPS parameter cores. In the Born machine paradigm,
        this is also :math:`\log(Z)`, for :math:`Z` the normalization
        constant for the probability.

        Returns:
            l_norm: Scalar value giving the log squared L2 norm of the
                n'th order prob. amp. tensor described by the MPS.
        """
        # Account for bias matrices before calling log norm implementation
        if self.use_bias:
            core_tensors = self.core_tensors + self.bias_mat[None, None]
        else:
            core_tensors = self.core_tensors

        return get_log_norm(core_tensors, self.edge_vecs)

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


class ProbUnifMPS(ProbMPS):
    r"""
    Uniform MPS model using L2 probabilities for generative modeling

    Probabilities of sequential inputs are obtained via the Born rule of
    quantum mechanics, making ProbUnifMPS a "Born machine" model. Given an
    input sequence of length n, the probability assigned to the sequence
    :math:`x = x_1 x_2 \dots x_n` is :math:`P(x) = |h_n^T \omega|^2 / Z`,
    where :math:`Z` is a normalization constant and the hidden state
    vectors :math:`h_t` are updated according to:

    .. math::
        h_t = (A_t[x_t] + B) h_{t-1},

    with :math:`h_0 := \alpha` (for :math:`\alpha, \omega` trainable
    parameter vectors), :math:`A_t[i]` the i'th matrix slice of a
    third-order core tensor for the t'th input, and :math:`B` an optional
    bias matrix.

    Note that calling a :class:`ProbUnifMPS` instance with given input will
    return the **logarithm** of the input probabilities, to avoid underflow
    in the case of longer sequences. To get the negative log likelihood
    loss for a batch of inputs, use the :attr:`loss` function of the
    :class:`ProbUnifMPS`.

    Args:
        input_dim: Dimension of the inputs to the uMPS core. For vector
            sequence inputs this is the dimension of the input vectors,
            while for discrete sequence inputs this is the size of the
            discrete alphabet.
        bond_dim: Dimension of the bond spaces linking copies of uMPS core.
        complex_params: Whether model parameters are complex or real. The
            former allows more expressivity, but is less common in Pytorch.
            Default: ``False``
        parallel_eval: Whether to force parallel contraction of tensor
            network. This leads to greater total computational cost, but
            can be faster in the presence of GPUs. Default: ``False``
        use_bias: Whether to use a trainable bias matrix in evaluation.
            Default: ``False``
    """

    def __init__(
        self,
        input_dim: int,
        bond_dim: int,
        complex_params: bool = False,
        parallel_eval: bool = False,
        use_bias: bool = False,
    ) -> None:
        super(ProbMPS, self).__init__()
        assert min(input_dim, bond_dim) > 0

        # Initialize core tensor and edge vectors
        core_tensors = near_eye_init((input_dim, bond_dim, bond_dim), complex_params)
        edge_vecs = torch.randn(2, bond_dim) / sqrt(bond_dim)
        if complex_params:
            edge_vecs = phaseify(edge_vecs)
        self.core_tensors = nn.Parameter(core_tensors)
        self.edge_vecs = nn.Parameter(edge_vecs)

        # Initialize (optional) bias matrices at zero
        if use_bias:
            bias_mat = torch.zeros(bond_dim, bond_dim)
            if complex_params:
                bias_mat = phaseify(bias_mat)
            self.bias_mat = nn.Parameter(bias_mat)

        # Set other MPS attributes
        self.complex_params = complex_params
        self.parallel_eval = parallel_eval
        self.rescale_factor = None

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
        # TODO: Convert input to STensors first

        # Contract inputs with core tensors and add bias matrices
        mat_slices = get_mat_slices(input_data, self.core_tensors)
        if self.use_bias:
            mat_slices = mat_slices + self.bias_mat[None, None]

        # Put the batch axis, since contract_matseq expects that
        mat_slices.transpose_(0, 1)

        #  Contract all bond dims to get (unnormalized) prob amplitudes
        psi_vals = contract_matseq(
            mat_slices, self.edge_vecs[0], self.edge_vecs[1], self.parallel_eval
        )

        # Get log normalization and check for infinities
        log_norm = self.log_norm(len(input_data))
        assert log_norm.isfinite()
        assert torch.all(psi_vals.isfinite())

        # Compute unnormalized log probabilities and rescale factor
        log_uprobs = torch.log(torch.abs(psi_vals))
        self.rescale_factor = torch.exp(log_uprobs.mean() / len(input_data))

        # Return normalized probabilities
        return 2 * log_uprobs - log_norm

    # def loss(self, input_data: Tensor) -> Tensor:
    #     """
    #     Get the negative log likelihood loss for batch of input data

    #     Args:
    #         input_data: Sequential with shape `(seq_len, batch)`, for
    #             discrete inputs, or shape `(seq_len, batch, input_dim)`,
    #             for vector inputs.

    #     Returns:
    #         loss_val: Scalar value giving average of the negative log
    #             likelihood loss of all sequences in input batch.
    #     """
    #     # Rescale the core tensors and boundary vectors
    #     if self.rescale_factor is not None:
    #         state_dict = self.state_dict()
    #         vec_rescale = floor2(self.edge_vecs.norm(dim=1, keepdim=True))
    #         new_edgevecs = self.edge_vecs / vec_rescale
    #         new_coretensors = self.core_tensors / self.rescale_factor
    #         state_dict["edge_vecs"] = new_edgevecs
    #         state_dict["core_tensors"] = new_coretensors
    #         self.load_state_dict(state_dict)

    #     return -torch.mean(self.forward(input_data))

    def log_norm(self, data_len) -> Tensor:
        r"""
        Compute the log normalization of the MPS for its fixed-size input

        Uses iterated tensor contraction to compute :math:`\log(|\psi|^2)`,
        where :math:`\psi` is the n'th order tensor described by the
        contraction of MPS parameter cores. In the Born machine paradigm,
        this is also :math:`\log(Z)`, for :math:`Z` the normalization
        constant for the probability.

        Returns:
            l_norm: Scalar value giving the log squared L2 norm of the
                n'th order prob. amp. tensor described by the MPS.
        """
        # Account for bias matrices before calling log norm implementation
        if self.use_bias:
            core_tensors = self.core_tensors + self.bias_mat[None, None]
        else:
            core_tensors = self.core_tensors

        return get_log_norm(core_tensors, self.edge_vecs, length=data_len)

    @property
    def input_dim(self):
        return self.core_tensors.shape[0]

    @property
    def bond_dim(self):
        return self.core_tensors.shape[1]

    @property
    def use_bias(self):
        return hasattr(self, "bias_mat")
