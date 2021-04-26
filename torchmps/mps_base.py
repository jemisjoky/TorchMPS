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
from typing import Union, Sequence, Optional

import torch
from torch import Tensor

from torchmps.utils2 import bundle_tensors, batch_broadcast

TensorSeq = Union[Tensor, Sequence[Tensor]]


def contract_matseq(
    matrices: TensorSeq,
    left_vec: Optional[Tensor] = None,
    right_vec: Optional[Tensor] = None,
    parallel_eval: bool = False,
) -> Tensor:
    """
    Matrix-multiply sequence of matrices with optional boundary vectors

    The output is a single matrix, or a vector/scalar if one/both boundary
    vectors are given. In the latter case, the first vector is treated as
    a row vector and the last as a column vector and put at the beginning
    or end of the sequence of matrices to reduce to a vector/scalar.

    Parallel matrix-matrix multiplications can be used to make the
    computation more GPU-friendly, at the cost of a larger overall compute
    cost. By default, this method is only used when an output matrix is
    desired, but can be forced by setting parallel_eval to True.

    When matrices or boundary vectors contain additional batch indices
    (assumed to be left-most indices), then batch matrix multiplication is
    carried out over all batch indices, which are broadcast together.
    Shapes described below neglect these additional batch indices.

    Args:
        matrices: Single tensor of shape (L, D, D), or sequence of
            matrices with compatible shapes :math:`(D_i, D_{i+1})`, for
            :math:`i = 0, 1, \ldots, L`.
        left_vec: Left boundary vector with shape `(D_0,)`, or None if no
            left boundary is present.
        right_vec: Left boundary vector with shape `(D_L,)`, or None if no
            right boundary is present.
        parallel_eval: Whether or not to force parallel evaluation in
            matrix contraction, which requires all input matrices to have
            same shape.

    Returns:
        contraction: Single scalar, vector, or matrix, equal to the
            sequential contraction of the input matrices with (resp.)
            two, one, or zero boundary vectors.
    """
    # Count number of boundary vectors
    bnd_vecs = [left_vec, right_vec]
    real_vec = [v is not None for v in bnd_vecs]
    num_vecs = sum(real_vec)
    assert all(v is None or isinstance(v, Tensor) for v in bnd_vecs)
    assert num_vecs <= 2

    # Convert matrices to single batch tensor, provided all shapes agree
    matrices = bundle_tensors(matrices, dim=-3)
    same_shape = isinstance(matrices, Tensor)
    num_mats = matrices.shape[-3] if same_shape else len(matrices)

    # Decide whether to use parallel evaluation algorithm
    use_parallel = same_shape and (parallel_eval or num_vecs == 0)

    # Broadcast batch dimensions of matrices and boundary vectors
    if num_vecs == 0:
        if not same_shape:
            matrices = batch_broadcast(matrices, (2,) * num_mats)
    elif num_vecs == 1:
        v_ind = real_vec.index(True)
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

        # Contract with boundary vectors, using intermediate dummy axes
        if real_vec[0]:
            product = torch.matmul(bnd_vecs[0][..., None, :], product)
        if real_vec[1]:
            product = torch.matmul(product, bnd_vecs[1][..., None])
        if real_vec[0]:
            product.squeeze_(-2)
        if real_vec[1]:
            product.squeeze_(-1)

    else:
        if num_vecs == 0 and len(matrices) == 0:
            raise ValueError(
                "Must input at least one matrix or boundary vector to contract_matseq"
            )

        # Prepend/append boundary vectors, augmented with dummy dimensions
        if same_shape:
            matrices = [matrices[..., i, :, :] for i in range(num_mats)]
        else:
            matrices = list(matrices)
        if real_vec[0]:
            matrices = [bnd_vecs[0][..., None, :]] + matrices
        if real_vec[1]:
            matrices.append(bnd_vecs[1][..., None])

        # Compute product sequentially and strip away dummy dimensions
        product = mat_reduce_seq(matrices)
        if real_vec[0]:
            product.squeeze_(-2)
        if real_vec[1]:
            product.squeeze_(-1)

    return product


def mat_reduce_par(matrices: Tensor) -> Tensor:
    """
    Contract sequence of square matrices with parallel mat-mat multiplies

    Args:
        matrices: Sequence of matrices to multiply.

    Returns:
        prod_mat: Product of input matrices
    """
    assert matrices.ndim >= 3
    s_dim = -3  # Dimension which has spatial arrangement of matrices
    n_mats = matrices.shape[s_dim]

    # In case of empty collection of matrices, return the identity
    if n_mats == 0:
        eye = torch.eye(matrices.shape[-1], dtype=matrices.dtype)
        matrices, _ = batch_broadcast((eye, matrices), (2, 3))
        return matrices
    elif n_mats == 1:
        return matrices.squeeze(dim=s_dim)

    # Iteratively multiply pairs of matrices until there is only one left
    assert matrices.shape[-2] == matrices.shape[-1]
    while n_mats > 1:
        half_n = n_mats // 2
        floor_n = half_n * 2

        # Split matrices up into even and odd numbers (maybe w/ leftover)
        even_mats = matrices[..., 0:floor_n:2, :, :]
        odd_mats = matrices[..., 1:floor_n:2, :, :]
        leftover = matrices[..., floor_n:, :, :]

        # Batch multiply everything, append remainder
        matrices = even_mats @ odd_mats
        matrices = torch.cat((matrices, leftover), dim=s_dim)
        n_mats = matrices.shape[s_dim]

    return matrices.squeeze(dim=s_dim)


def mat_reduce_seq(matrices: Sequence[Tensor]) -> Tensor:
    """
    Multiply sequence of matrices sequentially, from left to right

    Args:
        matrices: Sequence of matrices to multiply.

    Returns:
        prod_mat: Product of input matrices
    """
    # Multiplication from left to right, so flip matrices if it's cheaper
    # to multiply from right to left
    r2l = matrices[-1].size(-1) < matrices[0].size(-2)
    if r2l:
        matrices = tuple(m.transpose(-2, -1) for m in matrices[::-1])

    # Multiply all matrices sequentially, from left to right
    product, matrices = matrices[0], matrices[1:]
    for mat in matrices:
        product = torch.matmul(product, mat)

    # Revert to original form before returning
    return product.transpose(-2, -1) if r2l else product
