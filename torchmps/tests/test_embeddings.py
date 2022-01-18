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

"""Tests for embedding functions"""
import pytest
from functools import partial
from itertools import product

import torch
from hypothesis import given, settings, strategies as st

from torchmps import ProbMPS
from torchmps.embeddings import (
    DataDomain,
    unit_interval,
    FixedEmbedding,
    onehot_embed,
    trig_embed,
    init_mlp_embed,
    legendre_embed,
)
from torchmps.tests.utils_for_tests import allcloseish


@given(st.booleans(), st.floats(-100, 100), st.floats(0.0001, 1000))
def test_data_domain(continuous, max_val, length):
    """
    Verify that DataDomain class initializes properly
    """
    min_val = max_val - length
    if not continuous:
        max_val = abs(int(max_val)) + 1
    data_domain = DataDomain(continuous=continuous, max_val=max_val, min_val=min_val)

    assert data_domain.continuous == continuous
    assert data_domain.max_val == max_val
    if continuous:
        assert data_domain.min_val == min_val
    else:
        assert not hasattr(data_domain, "min_val")


@given(st.integers(1, 100))
def test_onehot_embedding(emb_dim):
    """
    Verify that FixedEmbedding works as expected when given one-hot embedding
    """
    data_domain = DataDomain(continuous=False, max_val=emb_dim)
    fixed_embed = FixedEmbedding(partial(onehot_embed, emb_dim=emb_dim), data_domain)
    assert torch.allclose(fixed_embed.lamb_mat, torch.ones(()))

    # Verify that the embedding function can be called and works fine
    rand_inds1 = torch.randint(emb_dim, (10,))
    rand_inds2 = torch.randint(emb_dim, (10, 5))
    rand_vecs1 = fixed_embed(rand_inds1)
    rand_vecs2 = fixed_embed(rand_inds2).reshape(50, emb_dim)
    rand_inds2 = rand_inds2.reshape(50)
    for vec, idx in zip(rand_vecs1, rand_inds1):
        assert torch.all((vec == 1) == (torch.arange(emb_dim) == idx))
    for vec, idx in zip(rand_vecs2, rand_inds2):
        assert torch.all((vec == 1) == (torch.arange(emb_dim) == idx))


@given(st.integers(1, 10), st.integers(1, 3))
def test_mlp_embedding(input_dim, num_layers):
    """
    Verify that MLP embedding function runs and gives properly normalized probs
    """
    bond_dim = 8
    hidden_dim = 10
    embed_fun = init_mlp_embed(input_dim, num_layers=num_layers, hidden_dims=hidden_dim)
    mps = ProbMPS(
        seq_len=1,
        input_dim=input_dim,
        bond_dim=bond_dim,
        complex_params=False,
        embed_fun=embed_fun,
    )
    points = torch.linspace(0, 1, 100)[:, None]  # Add phony spatial dim
    log_prob_densities = mps(points)
    assert log_prob_densities.shape == (100,)
    prob_densities = torch.exp(log_prob_densities)
    total_prob = torch.trapz(prob_densities, points[:, 0])
    assert torch.allclose(total_prob, torch.ones(()))


@pytest.mark.parametrize("raw_embed", [trig_embed, legendre_embed])
@given(st.integers(1, 10))
def test_frameified_embeddings(raw_embed, input_dim):
    """
    Verify that frameified trig and legendre embeddings are indeed a frame
    """
    embed_fun = partial(raw_embed, emb_dim=input_dim)
    data_domain = DataDomain(continuous=True, min_val=0.0, max_val=1.0)
    fixed_embed = FixedEmbedding(embed_fun, data_domain, frameify=True)

    # Manually compute the lambda matrix for frameified embedding
    num_points = 1000
    points = torch.linspace(0, 1, steps=num_points)
    emb_vecs = fixed_embed(points)
    emb_mats = torch.einsum("bi,bj->bij", emb_vecs, emb_vecs.conj())
    lamb_mat = torch.trapz(emb_mats, points, dim=0)

    # Verify the lambda matrix for framified embedding is identity
    assert allcloseish(lamb_mat, torch.eye(input_dim), tol=input_dim * 1e-2)


@given(st.integers(1, 50), st.booleans())
def test_legendre_embedding(input_dim, use_emb_class):
    """
    Verify that Legendre polynomial embedding is indeed a frame
    """
    embed_fun = partial(legendre_embed, emb_dim=input_dim)
    if use_emb_class:
        embed_fun = FixedEmbedding(embed_fun, unit_interval, frameify=False)

    # Manually compute the lambda matrix for frameified embedding
    points = torch.linspace(0, 1, steps=1000)
    emb_vecs = embed_fun(points)
    emb_mats = torch.einsum("bi,bj->bij", emb_vecs, emb_vecs.conj())
    lamb_mat = torch.trapz(emb_mats, points, dim=0)

    # Verify the lambda matrix for framified embedding is identity
    assert allcloseish(lamb_mat, torch.eye(input_dim), tol=input_dim * 1e-2)


@pytest.mark.parametrize(
    "raw_embed, frameify", list(product([trig_embed, legendre_embed], [False, True]))
)
@settings(deadline=None)
# @given(st.integers(1, 15))
# def test_normalization(raw_embed, frameify, input_dim):
@given(st.integers(1, 12), st.booleans())
def test_normalization(raw_embed, frameify, input_dim, complex_params):
    """
    For a 2-mode MPS, verify that probabilities integrate to 1
    """
    torch.manual_seed(0)
    bond_dim = 10
    num_points = 200
    embed_fun = FixedEmbedding(
        partial(raw_embed, emb_dim=input_dim), unit_interval, frameify=frameify
    )
    mps = ProbMPS(
        seq_len=2,
        input_dim=input_dim,
        bond_dim=bond_dim,
        complex_params=complex_params,
        embed_fun=embed_fun,
    )

    # Cast to double precision, since we need it for the following
    if complex_params:
        model_dtype = torch.complex128
    else:
        model_dtype = torch.float64
    mps.embedding.dtype = model_dtype
    mps.to(model_dtype)

    # Define a 2D mesh grid covering the unit square, stack x and y coordinates
    points = torch.linspace(0, 1, num_points)
    x, y = torch.meshgrid(points, points, indexing="ij")
    all_points = torch.stack((x, y), dim=-1)

    # Feed flattened (x, y) pairs to MPS, then integrate over both axes
    log_pdf = mps(all_points.reshape(-1, 2)).reshape(num_points, num_points)
    pdf = log_pdf.exp()
    int_prob = torch.trapz(torch.trapz(pdf, points, dim=1), points, dim=0)

    assert allcloseish(int_prob, torch.ones((), dtype=int_prob.dtype), tol=1e-1)


# # TODO: Get the following test actually working
# @given(st.integers(1, 10), st.integers(1, 3))
# def test_frameified_mlp_embedding(input_dim, num_layers):
#     """
#     Verify that frameified MLP embedding is indeed a frame
#     """
#     hidden_dim = 10
#     train_embed = init_mlp_embed(
#         input_dim, num_layers=num_layers, hidden_dims=hidden_dim, frameify=True
#     )

#     # Manually compute the lambda matrix for frameified embedding
#     points = torch.linspace(0, 1, steps=1000)
#     emb_vecs = train_embed(points)
#     emb_mats = torch.einsum("bi,bj->bij", emb_vecs, emb_vecs.conj())
#     lamb_mat = torch.trapz(emb_mats, points, dim=0)

#     # Verify the lambda matrix for framified embedding is identity
#     assert allcloseish(lamb_mat, torch.eye(input_dim), tol=input_dim * 1e-2)
