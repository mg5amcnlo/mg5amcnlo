"""EnergyScale: the dynamical scale choices and `scale_factor`, which multiplies
the dynamical scale only (as the LO `scalefact`); fixed scales are left alone.
"""

import math

import pytest
import torch
from madspace_testing import generator, rand

import madspace as ms

M_TOP = 173.0


def _momenta():
    """A few p p -> t t~ events, (batch, n, 4) with (E, px, py, pz)."""
    rng = generator(20260917)
    batch = 32
    pt = 10.0 + 390.0 * rand((batch, 2), generator=rng)
    phi = 2 * math.pi * rand((batch, 2), generator=rng)
    pz = -600.0 + 1200.0 * rand((batch, 2), generator=rng)
    e = torch.sqrt(M_TOP**2 + pt**2 + pz**2)
    out = torch.stack(
        [e, pt * torch.cos(phi), pt * torch.sin(phi), pz], dim=-1
    )  # (batch, 2, 4)
    e_in = out[:, :, 0].sum(dim=1) / 2
    incoming = torch.zeros((batch, 2, 4))
    incoming[:, 0, 0] = e_in
    incoming[:, 0, 3] = e_in
    incoming[:, 1, 0] = e_in
    incoming[:, 1, 3] = -e_in
    return torch.cat([incoming, out], dim=1)


def _ht(momenta):
    """sum_i sqrt(m_i^2 + pT_i^2) over the final state."""
    final = momenta[:, 2:, :]
    return torch.sqrt(
        torch.clamp(final[..., 0] ** 2 - final[..., 3] ** 2, min=0.0)
    ).sum(dim=1)


def assert_approx(actual, expected, rel=1e-6, abs=1e-12):
    """pytest.approx-like elementwise comparison of tensors."""
    torch.testing.assert_close(actual, expected, rtol=rel, atol=abs)


@pytest.mark.parametrize(
    "dyn_type, expected",
    [
        (ms.EnergyScale.transverse_mass, 1.0),
        (ms.EnergyScale.half_transverse_mass, 0.5),
    ],
)
def test_dynamical_scale_definitions(dyn_type, expected):
    momenta = _momenta()
    scale = ms.EnergyScale(momenta.shape[1], dyn_type, False, False, 0.0, 0.0, 0.0)
    ren, fact1, fact2 = scale(momenta)
    reference = expected * _ht(momenta)
    assert_approx(ren, reference)
    assert_approx(fact1, reference)
    assert_approx(fact2, reference)


@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
def test_scale_factor_multiplies_the_dynamical_scale(factor):
    momenta = _momenta()
    scale = ms.EnergyScale(
        momenta.shape[1],
        ms.EnergyScale.half_transverse_mass,
        False,
        False,
        0.0,
        0.0,
        0.0,
        factor,
    )
    ren, fact1, fact2 = scale(momenta)
    reference = factor * 0.5 * _ht(momenta)
    assert_approx(ren, reference)
    assert_approx(fact1, reference)
    assert_approx(fact2, reference)


def test_scale_factor_default_is_one():
    momenta = _momenta()
    args = (
        momenta.shape[1],
        ms.EnergyScale.half_transverse_mass,
        False,
        False,
        0.0,
        0.0,
        0.0,
    )
    without = ms.EnergyScale(*args)(momenta)
    with_one = ms.EnergyScale(*args, 1.0)(momenta)
    for a, b in zip(without, with_one):
        assert_approx(a, b)


def test_scale_factor_leaves_a_fixed_scale_alone():
    momenta = _momenta()
    fixed = 345.0
    scale = ms.EnergyScale(
        momenta.shape[1],
        ms.EnergyScale.half_transverse_mass,
        True,
        True,
        fixed,
        fixed,
        fixed,
        0.5,
    )
    ren, fact1, fact2 = scale(momenta)
    assert_approx(ren, torch.full((momenta.shape[0],), fixed))
    assert_approx(fact1, torch.full((momenta.shape[0],), fixed))
    assert_approx(fact2, torch.full((momenta.shape[0],), fixed))


def test_scale_factor_with_a_fixed_renormalisation_scale_only():
    """mu_R fixed, mu_F dynamical: only mu_F picks the factor up."""
    momenta = _momenta()
    fixed = 91.188
    scale = ms.EnergyScale(
        momenta.shape[1],
        ms.EnergyScale.half_transverse_mass,
        True,
        False,
        fixed,
        0.0,
        0.0,
        0.25,
    )
    ren, fact1, fact2 = scale(momenta)
    assert_approx(ren, torch.full((momenta.shape[0],), fixed))
    assert_approx(fact1, 0.25 * 0.5 * _ht(momenta))
    assert_approx(fact2, 0.25 * 0.5 * _ht(momenta))
