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

from torchmps.mps_base import contract_matseq

bool_st = st.booleans()
bond_dim_st = st.integers(1, 20)
batch_shape_st = st.lists(st.integers(1, 10), min_size=0, max_size=4)


def boundary_contraction(mats, lvec, rvec, use_lvec, use_rvec):
    """Handle conditional contraction with boundary vectors"""
    pass


@given(batch_shape_st, bond_dim_st, st.integers(0, 10), bool_st, bool_st)
def test_contract_matseq_identity_batches(batch_shape, bond_dim, seq_len, use_lvec, use_rvec):
    """
    Multipy random multiples of the identity matrix w/ variable batch size
    """
    pass


@given(st.lists(bond_dim_st, min_size=1, max_size=10), bool_st, bool_st)
def test_contract_matseq_random_inhom_bonddim(bonddim_list, use_lvec, use_rvec):
    """Multiply random matrices with inhomogeneous bond dimensions"""
    pass
