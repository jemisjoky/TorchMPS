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

"""Tests for utility functions"""
import pytest

import torch

from torchmps.utils2 import batch_broadcast


def test_batch_broadcast_good_shapes():
    t_shape = (5, 3, 2, 2, 2)
    m_shape = (1, 3, 3, 4)
    v_shape = (3,)
    s_shape = (2, 1, 1)

    tensors = [torch.ones(*shp) for shp in (t_shape, m_shape, v_shape, s_shape)]
    non_batch = (3, 2, 1, 0)
    out_tensors = batch_broadcast(tensors, non_batch)
    shapes = [tuple(t.shape) for t in out_tensors]

    assert shapes == [(2, 5, 3, 2, 2, 2), (2, 5, 3, 3, 4), (2, 5, 3, 3), (2, 5, 3)]
