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

"""Basic MPS functions used for uniform and non-uniform models"""
from typing import Union, Tuple, Sequence, Optional

import torch
from torch import Tensor

from torchmps.utils2 import bundle_tensors

TensorSeq = Union[Tensor, Sequence[Tensor]]


def contract_matrices(
    matrices: TensorSeq,
    bnd_vecs: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    parallel_eval: bool = False,
) -> Tensor:
    """
    Matrix-multiply sequence of matrices, with optional boundary vectors

    The output is a single matrix, or a scalar if a pair of boundary
    vectors are given. In the latter case, the first vector is treated as
    a row vector and the last as a column vector and multiplied at the
    beginning and end of the sequence of matrices to reduce to a scalar.

    Parallel matrix-matrix multiplications can be used to make the
    computation more GPU-friendly, at the cost of a larger overall compute
    cost. By default, this method is only used when an output matrix is
    desired, but can be forced by setting parallel_eval to True.

    Args:
        matrices: Single tensor of shape (len, D, D), or sequence of
            matrices with compatible shapes :math:`(D_i, D_{i+1})`, for
            :math:`i = 0, 1, \ldots, len`
        bnd_vecs: Pair of vectors to place on the boundaries, specified as
            matrix of shape (2, D) or pair of vectors
        parallel_eval: Whether or not to force parallel evaluation of
            matrix contraction

    Returns:
        contracted: Single scalar (or matrix if bnd_vecs isn't given)
            giving the sequential contraction of input matrices and vectors
    """
    # Convert matrices to single batch tensor, provided all shapes agree
    matrices = bundle_tensors(matrices)
    same_shape = isinstance(matrices, Tensor)

    # Decide whether to use parallel evaluation algorithm
    no_bvecs = bnd_vecs is None
    use_parallel = same_shape and (parallel_eval or no_bvecs)

    if use_parallel:
        prod_mat = mat_reduce_par(matrices)

        # TODO: Contraction with boundary matrices

    else:
        return matvec_reduce_seq(matrices, bnd_vecs)


def mat_reduce_par(matrices: Tensor) -> Tensor:
    """
    Contract sequence of square matrices using parallel mat-mat multiplies
    """
    pass


def matvec_reduce_seq(
    matrices: TensorSeq, bnd_vecs: Union[Tensor, Tuple[Tensor, Tensor]]
) -> Tensor:
    """
    Contract sequence of matrices with left/right boundary vectors

    Args:
        matrices: Single batch tensor or sequence of matrices to multiply in order. When additional batch indices are present,
    """
    pass
