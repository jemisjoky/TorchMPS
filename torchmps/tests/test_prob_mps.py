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
from functools import partial

import torch
from hypothesis import given, settings, strategies as st

from torchmps.prob_mps import ProbMPS

bool_st = st.booleans
seq_len_st = partial(st.integers, 1, 1000)
bond_dim_st = partial(st.integers, 1, 20)
input_dim_st = partial(st.integers, 1, 10)


@settings(deadline=None)
@given(
    seq_len_st(),
    input_dim_st(),
    bond_dim_st(),
    bool_st(),
    bool_st(),
    bool_st(),
    bool_st(),
    bool_st(),
    bool_st(),
)
def test_model_forward(
    seq_len,
    input_dim,
    bond_dim,
    complex_params,
    periodic_bc,
    parallel_eval,
    use_bias,
    vec_input,
    big_batch,
):
    """
    Verify that model forward function runs and gives reasonable output
    """
    # Initialize probabilistic MPS and the sequence data it will be fed
    prob_mps = ProbMPS(
        seq_len,
        input_dim,
        bond_dim,
        complex_params,
        periodic_bc,
        parallel_eval,
        use_bias,
    )
    batch_dim = 100 if big_batch else 1
    if vec_input:
        fake_data = torch.randn(seq_len, batch_dim, input_dim).abs()
        fake_data /= fake_data.sum(dim=2, keepdim=True)
    else:
        fake_data = torch.randint(input_dim, (seq_len, batch_dim))

    # Call the model on the fake data, verify that it looks alright
    log_probs = prob_mps(fake_data)
    assert log_probs.shape == (batch_dim,)
    assert log_probs.is_floating_point()
    if not torch.all(log_probs <= 0):
        assert input_dim == 1
