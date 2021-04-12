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

from torchmps.utils2 import bundle_tensors, batch_broadcast

TensorSeq = Union[Tensor, Sequence[Tensor]]


def contract_matseq(
    matrices: TensorSeq,
    bnd_vecs: Tuple[Optional[Tensor], Optional[Tensor]] = (None, None),
    parallel_eval: bool = False,
) -> Tensor:
    """
    Batch matrix-multiply sequence of matrices, with optional boundary vectors

    The output is a single matrix, or a scalar if a pair of boundary
    vectors are given. In the latter case, the first vector is treated as
    a row vector and the last as a column vector and multiplied at the
    beginning and end of the sequence of matrices to reduce to a scalar.

    Parallel matrix-matrix multiplications can be used to make the
    computation more GPU-friendly, at the cost of a larger overall compute
    cost. By default, this method is only used when an output matrix is
    desired, but can be forced by setting parallel_eval to True.

    When matrices or boundary vectors contain additional batch indices
    (assumed to be left-most indices), then matrix multiplication is
    carried out over all batch indices, which are broadcast together.
    Shapes described below neglect these additional batch indices.

    Args:
        matrices: Single tensor of shape (L, D, D), or sequence of
            matrices with compatible shapes :math:`(D_i, D_{i+1})`, for
            :math:`i = 0, 1, \ldots, L`.
        bnd_vecs: Pair of vectors `(l_vec, r_vec)` to place on the
            boundaries of contracted matrices. Either value can be None,
            in which case that matrix index is left open.
        parallel_eval: Whether or not to force parallel evaluation in
            matrix contraction, which requires all input matrices to have
            same shape.

    Returns:
        contraction: Single scalar, vector, or matrix, equal to the
            sequential contraction of the input matrices with (resp.)
            two, one, or zero boundary vectors.
    """
    # Count number of boundary vectors
    bnd_vecs = list(bnd_vecs)
    num_vecs = sum(v is not None for v in bnd_vecs)
    assert all(v is None or isinstance(v, Tensor) for v in bnd_vecs)
    assert num_vecs <= 2

    # Convert matrices to single batch tensor, provided all shapes agree
    matrices = bundle_tensors(matrices)
    same_shape = isinstance(matrices, Tensor)
    num_mats = matrices.shape[-3] if same_shape else len(matrices)

    # Decide whether to use parallel evaluation algorithm
    use_parallel = same_shape and (parallel_eval or num_vecs == 0)

    # Broadcast batch dimensions of matrices and boundary vectors
    if num_vecs == 0:
        if not same_shape:
            matrices = batch_broadcast(matrices, (2,) * num_mats)
    elif num_vecs == 1:
        v_ind = [v is not None for v in bnd_vecs].index(True)
        vec = bnd_vecs[v_ind]
        if same_shape:
            vec, matrices = batch_broadcast((vec, matrices), (1, 3))
        else:
            outs = batch_broadcast((vec,) + tuple(matrices), (1,) + (2,) * num_mats)
            vec, matrices = outs[0], outs[1:]
        bnd_vecs[v_ind] = vec
    else:
        if same_shape:
            outs = batch_broadcast(bnd_vecs + [matrices], (1, 1, 3))
            bnd_vecs, matrices = outs[:2], outs[2]
        else:
            outs = batch_broadcast(bnd_vecs + list(matrices), (1, 1) + (2,) * num_mats)
            bnd_vecs, matrices = outs[:2], outs[2:]

    if use_parallel:
        # Reduce product of all matrices in parallel
        product = mat_reduce_par(matrices)

        # Contract with boundary vectors
        if bnd_vecs[1] is not None:
            product = torch.matmul(product, bnd_vecs[1][..., None])
            product = product[..., 0]
        if bnd_vecs[0] is not None:
            product = torch.matmul(bnd_vecs[0][..., None, :], product)
            product = product[..., 0, :]

        return product

    else:
        if num_vecs == 0:
            return matvec_reduce_seq(matrices)
        elif num_vecs == 1:
            pass


def mat_reduce_par(matrices: Tensor) -> Tensor:
    """
    Contract sequence of square matrices using parallel mat-mat multiplies

    Args:
        matrices: Sequence of matrices to multiply.

    Returns:
        prod_mat: Product of input matrices
    """
    pass


def mat_reduce_seq(matrices: Sequence[Tensor]) -> Tensor:
    """
    Multiply sequence of matrices sequentially, from left to right

    Args:
        matrices: Sequence of matrices to multiply.

    Returns:
        prod_mat: Product of input matrices
    """
    prod_mat, matrices = matrices[0], matrices[1:]

    for mat in matrices:
        pass
