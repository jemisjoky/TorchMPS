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

"""Utility functions"""
from typing import Union, Sequence

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


def batch_broadcast(tens_list: Sequence[Tensor], num_nonbatch: Sequence[int]):
    """
    Broadcast collection of tensors to have matching batch indices

    Broadcasting behavior is identical to standard PyTorch/NumPy but with
    broadcasting only performed on batch indices, which are always assumed
    to be the left-most indices. The separation between batch and non-batch
    indices is set by `num_nonbatch`, which gives the number of non-batch
    indices in each tensor.

    Args:
        tens_list: Sequence of tensors whose batch indices are being
            broadcast together. If the shape of batch indices cannot be
            broadcast, then `batch_broadcast` will throw an error
        num_nonbatch: Sequence of integers describing the number of
            non-batch indices in each of the tensors in `tens_list`. These
            non-batch indices are assumed to be the right-most indices of
            each respective tensor

    Returns:
        out_list: Sequence of tensors, which are broadcasted versions of
            those input in `tens_list`
    """
    assert not isinstance(tens_list, Tensor)
    assert len(tens_list) == len(num_nonbatch)
    assert all(i >= 0 for i in num_nonbatch)
    assert all(t.ndim >= nnb for t, nnb in zip(tens_list, num_nonbatch))

    # Compute shape of broadcasted batch dimensions
    b_shapes = [t.shape[: (t.ndim - nnb)] for t, nnb in zip(tens_list, num_nonbatch)]
    try:
        full_batch = shape_broadcast(b_shapes)
        bdims = len(full_batch)
    except ValueError:
        raise ValueError(
            f"Following batch shapes couldn't be broadcast: {tuple(b_shapes)}"
        )

    # Add singletons and expand batch dims of each tensor
    tens_list = [
        t[(None,) * (bdims + nnb - t.ndim)] for t, nnb in zip(tens_list, num_nonbatch)
    ]
    shapes = [full_batch + t.shape[bdims:] for t in tens_list]

    return tuple(t.expand(*shp) for t, shp in zip(tens_list, shapes))


def shape_broadcast(shape_list: Sequence[tuple]):
    """
    Predict shape of broadcasted tensors with given input shapes

    Code based on Stack Overflow post `here <https://stackoverflow.com/questions/54859286/is-there-a-function-that-can-apply-numpys-broadcasting-rules-to-a-list-of-shape/>`_

    Args:
        shape_list: Sequence of shapes, each input as a tuple

    Returns:
        b_shape: Broadcasted shape of those input in `shape_list`
    """
    ml = max(shape_list, key=len)
    out = list(ml)
    for l in shape_list:
        if l is ml:
            continue
        for i, x in enumerate(l, -len(l)):
            if x != 1 and x != out[i]:
                if out[i] != 1:
                    raise ValueError
                out[i] = x
    return tuple(out)
