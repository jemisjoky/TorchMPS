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


def init_model_and_data(
    seq_len,
    input_dim,
    bond_dim,
    complex_params,
    parallel_eval,
    use_bias,
    vec_input,
    big_batch,
):
    """Initialize probabilistic MPS and the sequence data it will be fed"""
    prob_mps = ProbMPS(
        seq_len,
        input_dim,
        bond_dim,
        complex_params,
        parallel_eval,
        use_bias,
    )
    batch_dim = 25 if big_batch else 1
    if vec_input:
        fake_data = torch.randn(seq_len, batch_dim, input_dim).abs()
        fake_data /= fake_data.sum(dim=2, keepdim=True)
    else:
        fake_data = torch.randint(input_dim, (seq_len, batch_dim))

    return prob_mps, fake_data


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
)
def test_model_forward(
    seq_len,
    input_dim,
    bond_dim,
    complex_params,
    parallel_eval,
    use_bias,
    vec_input,
    big_batch,
):
    """
    Verify that model forward function runs and gives reasonable output
    """
    # Initialize probabilistic MPS and dataset
    prob_mps, fake_data = init_model_and_data(
        seq_len,
        input_dim,
        bond_dim,
        complex_params,
        parallel_eval,
        use_bias,
        vec_input,
        big_batch,
    )
    batch_dim = 25 if big_batch else 1

    # Call the model on the fake data, verify that it looks alright
    log_probs = prob_mps(fake_data)
    assert log_probs.shape == (batch_dim,)
    assert torch.all(log_probs.isfinite())
    assert log_probs.is_floating_point()
    if not torch.all(log_probs <= 0):
        assert input_dim == 1


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
)
def test_model_backward(
    seq_len,
    input_dim,
    bond_dim,
    complex_params,
    parallel_eval,
    use_bias,
    vec_input,
    big_batch,
):
    """
    Verify that model backward pass runs and updates model params
    """
    # Initialize probabilistic MPS, dataset, and optimizer
    prob_mps, fake_data = init_model_and_data(
        seq_len,
        input_dim,
        bond_dim,
        complex_params,
        parallel_eval,
        use_bias,
        vec_input,
        big_batch,
    )
    optimizer = torch.optim.Adam(prob_mps.parameters())
    old_params = tuple(p.detach().clone() for p in prob_mps.parameters())

    # Call the model on fake data, backpropagate, and take gradient step
    loss = prob_mps.loss(fake_data)
    loss.backward()
    optimizer.step()

    # Verify that the new parameters are different from the old ones
    param_diff = False
    new_params = tuple(prob_mps.parameters())
    assert len(old_params) == len(new_params)
    assert all(p.grad is not None for p in new_params)
    for old_p, new_p in zip(old_params, new_params):
        assert old_p.shape == new_p.shape
        assert torch.all(old_p.isfinite())
        assert torch.all(new_p.isfinite())

        # For input dimension of 1, the gradients should be trivial
        if input_dim != 1:
            param_diff = param_diff or not torch.all(old_p == new_p)
        else:
            assert torch.allclose(old_p, new_p, rtol=1e-3, atol=1e-3)
            param_diff = True

    assert param_diff
