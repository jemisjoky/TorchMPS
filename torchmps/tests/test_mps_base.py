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

import torch
from hypothesis import given, settings, strategies as st

from torchmps.utils2 import batch_broadcast
from torchmps.mps_base import contract_matseq

bool_st = st.booleans()
bond_dim_st = st.integers(1, 20)
batch_shape_st = st.lists(st.integers(1, 10), min_size=0, max_size=4)


def naive_contraction(mats, lvec, rvec, use_lvec, use_rvec):
    """Handle conditional contraction with boundary vectors"""
    # Add phony boundary dimensions, broadcast batch dimensions to agree
    num_mats = len(mats)
    lvec, rvec = lvec[..., None, :], rvec[..., None]
    out = batch_broadcast((lvec, rvec) + tuple(mats), (2,) * (num_mats + 2))
    lvec, rvec, mats = out[0], out[1], out[2:]

    # Matrix/vector multiplication which respects batch dimensions
    if use_rvec:
        for mat in mats[::-1]:
            rvec = torch.matmul(mat, rvec)
        rvec = rvec[..., :, 0]
        return torch.sum(lvec * rvec, dim=-1)
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


@given(batch_shape_st, bond_dim_st, st.integers(0, 10), bool_st, bool_st)
def test_contract_matseq_identity_batches(batch_shape, bond_dim, seq_len, use_lvec, use_rvec):
    """
    Multipy random multiples of the identity matrix w/ variable batch size
    """
    pass


@given(st.lists(bond_dim_st, min_size=1, max_size=10), bool_st, bool_st, bool_st)
def test_contract_matseq_random_inhom_bonddim(bonddim_list, use_lvec, use_rvec, use_list):
    """Multiply random matrices with inhomogeneous bond dimensions"""
    pass


@given(st.booleans())
def test_contract_matseq_empty(parallel_eval):
    with pytest.raises(ValueError):
        contract_matseq((), None, None, parallel_eval)