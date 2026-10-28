"""Tests for the parametric sweep grid construction (subset support)."""

import jax
import jax.numpy as jnp
import pytest

from gpu_ehl.sweep import make_parameter_grid

jax.config.update("jax_enable_x64", True)

PUBLISHED_STOKES_COUNT = 13
PUBLISHED_SHAPES = {1.05, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}


def _unique(arr: jnp.ndarray) -> set[float]:
    return {float(v) for v in jnp.unique(arr)}


def test_default_grid_matches_published_sweep():
    """The default grid must reproduce the published 130-combination sweep."""
    S_flat, E_flat = make_parameter_grid()

    assert S_flat.shape == E_flat.shape == (130,)
    stokes = _unique(S_flat)
    assert len(stokes) == PUBLISHED_STOKES_COUNT
    assert min(stokes) == 10.0
    assert max(stokes) == 10000.0
    assert _unique(E_flat) == PUBLISHED_SHAPES


def test_subset_grid_is_rectangular_product():
    """Explicit lists produce the full rectangular product."""
    stokes = [100.0, 1000.0]
    exponents = [2.0, 6.0, 10.0]

    S_flat, E_flat = make_parameter_grid(stokes_list=stokes, exponent_list=exponents)

    assert S_flat.shape == E_flat.shape == (6,)
    assert _unique(S_flat) == set(stokes)
    assert _unique(E_flat) == set(exponents)


def test_subset_grid_high_stokes_values_preserved():
    """Full-precision high-Stokes values must survive the round trip."""
    stokes = [3162.2776601683795, 5623.413251903491, 10000.0]
    exponents = [2.0, 10.0]

    S_flat, E_flat = make_parameter_grid(stokes_list=stokes, exponent_list=exponents)

    assert S_flat.shape == E_flat.shape == (6,)
    assert S_flat.dtype == jnp.float64
    assert jnp.allclose(jnp.sort(_unique_array(S_flat)), jnp.array(stokes))


def _unique_array(arr: jnp.ndarray) -> jnp.ndarray:
    return jnp.unique(arr)


def test_partial_override_keeps_default_exponents():
    """Omitting the exponent list falls back to the published 10-value set."""
    S_flat, E_flat = make_parameter_grid(stokes_list=[100.0])

    assert S_flat.shape == E_flat.shape == (10,)
    assert _unique(E_flat) == PUBLISHED_SHAPES
    assert _unique(S_flat) == {100.0}


def test_empty_stokes_list_raises():
    """An explicitly provided but empty Stokes list must fail loudly."""
    with pytest.raises(ValueError):
        make_parameter_grid(stokes_list=[])
