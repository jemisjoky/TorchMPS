# Copyright (C) 2021 Jacob Ellwyn Miller
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

"""Tests for MPS base functions"""
import pytest
from math import sqrt

import torch
from hypothesis import given, strategies as st

from torchmps.mps_base import contract_matseq
from torchmps.utils2 import batch_broadcast, batch_to

bool_st = st.booleans()
bond_dim_st = st.integers(1, 20)
batch_shape_st = lambda l: st.lists(st.integers(1, 10), min_size=0, max_size=l)


def naive_contraction(mats, lvec, rvec, use_lvec, use_rvec):
    """Handle conditional contraction with boundary vectors"""
    # For empty batch of matrices, replace with single identity matrix
    if isinstance(mats, torch.Tensor) and mats.shape[-3] == 0:
        assert isinstance(mats, torch.Tensor)
        assert mats.shape[-2] == mats.shape[-1]
        eye = torch.eye(mats.shape[-1])
        mats = [batch_to(eye, mats.shape[:-3], 2)]

    # Convert mats to list, add phony dims to vecs, broadcast all batch dims
    if isinstance(mats, torch.Tensor):
        mats = [mats[..., i, :, :] for i in range(mats.shape[-3])]
    else:
        assert hasattr(mats, "__len__")
        mats = list(mats)
    lvec, rvec = lvec[..., None, :], rvec[..., None]
    out = batch_broadcast([lvec, rvec] + mats, (2,) * (len(mats) + 2))
    lvec, rvec, mats = out[0], out[1], out[2:]

    # Matrix/vector multiplication which respects batch dimensions
    if use_rvec:
        for mat in mats[::-1]:
            rvec = torch.matmul(mat, rvec)
        rvec = rvec[..., 0]
        if not use_lvec:
            return rvec
        else:
            return torch.sum(lvec[..., 0, :] * rvec, dim=-1)
    elif use_lvec:
        for mat in mats:
            lvec = torch.matmul(lvec, mat)
        lvec = lvec[..., 0, :]
        return lvec
    else:
        pmat = mats[0]
        for mat in mats[1:]:
            pmat = torch.matmul(pmat, mat)
        return pmat


@given(batch_shape_st(4), bond_dim_st, st.integers(0, 10), bool_st, bool_st, bool_st)
def test_contract_matseq_identity_batches(
    batch_shape, bond_dim, seq_len, use_lvec, use_rvec, parallel_eval
):
    """
    Multipy random multiples of the identity matrix w/ variable batch size
    """
    # Case of empty matrices and no boundary vectors is special
    empty_case = seq_len == 0 and not (use_lvec or use_rvec)

    # Generate identity matrices and boundary vectors
    eye = torch.eye(bond_dim)
    eye_mats = batch_to(eye, tuple(batch_shape) + (seq_len,), 2)
    eye_mats2 = [eye_mats[..., i, :, :] for i in range(seq_len)]
    left_vec, right_vec = torch.randn(2, bond_dim)
    lvec = left_vec if use_lvec else None
    rvec = right_vec if use_rvec else None

    # Contract with the naive algorithm, compare to contract_matseq output
    naive_result = naive_contraction(eye_mats, left_vec, right_vec, use_lvec, use_rvec)
    lib_result = contract_matseq(eye_mats, lvec, rvec, parallel_eval)

    # Can't call contract_matseq with empty list, no boundary vectors
    if empty_case:
        lib_result2 = lib_result
    else:
        lib_result2 = contract_matseq(eye_mats2, lvec, rvec, parallel_eval)

    # Both ways of calling contract_matseq should agree,
    # except for empty matrix sequences
    if not torch.equal(lib_result, lib_result2):
        assert seq_len == 0
        assert lib_result.ndim > lib_result2.ndim
    assert torch.allclose(lib_result, naive_result)


@given(
    st.lists(bond_dim_st, min_size=1, max_size=10),
    batch_shape_st(3),
    bool_st,
    bool_st,
    bool_st,
    bool_st,
)
def test_contract_matseq_random_inhom_bonddim(
    bonddim_list, vec_batch, use_lvec, use_rvec, use_tuple, parallel_eval
):
    """
    Multiply random matrices with inhom bond dimensions, boundary vecs have batch dims
    """
    # Generate random matrices and batch boundary vectors
    num_bd, bd_lst = len(bonddim_list), bonddim_list
    rescales = [sqrt(bd_lst[i] * bd_lst[i + 1]) for i in range(num_bd - 1)]
    matrices = [
        torch.randn(bd_lst[i], bd_lst[i + 1]) / r for i, r in enumerate(rescales)
    ]
    if use_tuple:
        matrices = tuple(matrices)
    left_vec = torch.randn(*(vec_batch + [bd_lst[0]]))
    right_vec = torch.randn(*(vec_batch + [bd_lst[-1]]))
    lvec = left_vec if use_lvec else None
    rvec = right_vec if use_rvec else None

    # If no matrices and no boundary vecs are given, contract_matseq
    # should raise an error
    if len(bonddim_list) == 1 and not (use_lvec or use_rvec):
        with pytest.raises(ValueError):
            contract_matseq(matrices, lvec, rvec, parallel_eval)
        return

    # Contract with the naive algorithm, compare to contract_matseq output
    naive_result = naive_contraction(matrices, left_vec, right_vec, use_lvec, use_rvec)
    lib_result = contract_matseq(matrices, lvec, rvec, parallel_eval)

    # Numerical error is sometimes greater than tolerance of allclose
    if not torch.allclose(lib_result, naive_result):
        assert (lib_result - naive_result).norm() < 1e-5


@given(st.booleans())
def test_contract_matseq_empty(parallel_eval):
    """Verify that no boundary vectors and empty sequence raises error"""
    with pytest.raises(ValueError):
        contract_matseq((), None, None, parallel_eval)
