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

"""Utility functions"""
from typing import Union
from collections.abc import Sequence

import torch
from torch import Tensor

TensorSeq = Union[Tensor, Sequence[Tensor]]


def bundle_tensors(tensors: TensorSeq, dim: int = 0) -> TensorSeq:
    """
    When possible, converts a sequence of tensors into single batch tensor

    When all input tensors have the same shape or only one tensor is input,
    a batch tensor is produced with a new batch index. Collections of
    tensors with inhomogeneous shapes are returned unchanged.

    Args:
        tensors: Sequence of tensors
        dim: Location of the new batch dimension

    Returns:
        out_tens: Single batched tensor, when possible, or unchanged input
    """
    if isinstance(tensors, Tensor):
        return tensors

    if len(set(t.shape for t in tensors)) > 1:
        return tensors
    else:
        return torch.stack(tensors)