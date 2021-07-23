# MIT License
#
# Copyright (c) 2021 Jacob Miller
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Tests for MPS base functions"""
from functools import partial

import torch
import pytest
from hypothesis import given, settings, strategies as st

from torchmps import ProbMPS, ProbUnifMPS
from .utils_for_tests import complete_binary_dataset, allcloseish

# Frequently used Hypothesis strategies
bool_st = st.booleans
seq_len_st = partial(st.integers, 1, 1000)
bond_dim_st = partial(st.integers, 1, 20)
input_dim_st = partial(st.integers, 1, 10)
model_list = ["fixed-len", "uniform"]


# Parameterization over fixed-len and uniform models
def parametrize_models():
    return pytest.mark.parametrize("model", model_list, ids=model_list)


def init_model_and_data(
    model,
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
    if model == "fixed-len":
        prob_mps = ProbMPS(
            seq_len, input_dim, bond_dim, complex_params, parallel_eval, use_bias
        )
    elif model == "uniform":
        prob_mps = ProbUnifMPS(
            input_dim, bond_dim, complex_params, parallel_eval, use_bias
        )

    batch_dim = 25 if big_batch else 1
    if vec_input:
        fake_data = torch.randn(seq_len, batch_dim, input_dim).abs()
        fake_data /= fake_data.sum(dim=2, keepdim=True)
    else:
        fake_data = torch.randint(input_dim, (seq_len, batch_dim))

    return prob_mps, fake_data


@parametrize_models()
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
    model,
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
        model,
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


@parametrize_models()
# @settings(deadline=None)
@given(input_dim_st(), bond_dim_st(), bool_st(), bool_st(), bool_st())
def test_valid_binary_probs(
    model, seq_len, bond_dim, complex_params, parallel_eval, use_bias
):
    """
    Verify that for binary distributions, all probabilities sum up to 1
    """
    # Initialize dataset and model
    all_seqs = complete_binary_dataset(seq_len).T
    prob_mps, _ = init_model_and_data(
        model,
        seq_len,
        2,
        bond_dim,
        complex_params,
        parallel_eval,
        use_bias,
        False,
        False,
    )

    # Get model probabilities and verify they are close to 1
    probs = torch.exp(prob_mps(all_seqs))
    assert allcloseish(probs.sum(), 1.0, tol=5e-3)


@parametrize_models()
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
    model,
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
        model,
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
            assert allcloseish(old_p, new_p, tol=1e-3)
            param_diff = True

    assert param_diff
