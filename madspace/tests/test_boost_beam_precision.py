"""Precision of boost_beam out of the lab frame.

PhaseSpaceMapping generates the partonic momenta in their centre-of-mass frame
and boosts them into the lab with boost_beam; its inverse boosts them back with
boost_beam_inverse. A momentum moving against the boost loses e^{2|y|} of
relative precision in the linear boost E cosh y + p_z sinh y, so boost_beam
works in light-cone components with the smaller one from the mass shell. That
has to hold in both directions: the partonic momenta recovered from the lab
were off by up to 5e-9 of sqrt(s_hat) for partonic rapidities |y| ~ 7 (13 TeV,
x ~ 1e-6), and the random numbers of the round trip by as much.
"""

import numpy as np
import pytest
import torch
from madspace_testing import default_context, to_numpy

import madspace as ms

SQRT_S = 13000.0
N = 100000


def boost_beam_inverse(p, x1, x2, masses):
    """the partonic momenta from the lab momenta, as torch tensors"""
    n = p.shape[1]
    vec = ms.batch_four_vec_array(n)
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("p", vec), ("x1", ms.batch_float), ("x2", ms.batch_float)]),
        ms.NamedTypes([("p_out", vec)]),
    )
    fb.output(
        0,
        fb.boost_beam_inverse(fb.input(0), ms.Value(masses), fb.input(1), fb.input(2)),
    )
    runtime = ms.FunctionRuntime(fb.function(), default_context())
    return to_numpy(runtime.call([p, x1, x2])[0])


def light_cone_boost(p, masses, y):
    """Boost along z by y with both light-cone components to rounding."""
    q = p.copy()
    trans2 = np.asarray(masses) ** 2 + p[..., 1] ** 2 + p[..., 2] ** 2
    forward = p[..., 3] >= 0
    big = np.where(forward, p[..., 0] + p[..., 3], p[..., 0] - p[..., 3])
    small = trans2 / big
    plus = np.where(forward, big, small) * np.exp(y)[:, None]
    minus = np.where(forward, small, big) * np.exp(-y)[:, None]
    q[..., 0] = 0.5 * (plus + minus)
    q[..., 3] = 0.5 * (plus - minus)
    return q


# (masses, rapidity beyond which the round trip is checked: t t~ j reaches
# |y| ~ 3.6 at most)
CASES = [([0.0] * 5, 4.0), ([0.0, 0.0, 173.0, 173.0, 0.0], 2.0), ([0.0] * 6, 4.0)]


@pytest.mark.parametrize("masses,y_far", CASES, ids=["jjj", "ttj", "jjjj"])
def test_partonic_frame_from_the_lab(masses, y_far):
    mapping = ms.PhaseSpaceMapping(
        masses, SQRT_S, mode=ms.PhaseSpaceMapping.TChannelMode.rambo
    )
    r = np.random.default_rng(7).random((N, mapping.random_dim()))
    # torch in, torch out: numpy inputs give read-only numpy outputs
    lab = mapping.map_forward([torch.as_tensor(r)], [])
    p, x1, x2, det = map(to_numpy, lab)
    rapidity = 0.5 * np.log(x1 / x2)
    sqrt_s_hat = SQRT_S * np.sqrt(x1 * x2)
    ok = det > 0

    # the partonic frame comes back as precisely as an exact boost gives it
    partonic = boost_beam_inverse(*lab[:3], masses)
    exact = light_cone_boost(p, masses, -rapidity)
    error = np.abs(partonic - exact).max((1, 2)) / sqrt_s_hat
    assert error[ok].max() < 1e-14

    # and with it the random numbers at large rapidities, to the conditioning
    # of RAMBO itself (below 1e-10 on two million events; it was 4e-9 for
    # |y| > 6)
    r_inv, det_inv = map(to_numpy, mapping.map_inverse(lab[:3], []))
    far = ok & (np.abs(rapidity) > y_far)
    assert far.sum() > 100
    assert np.abs(r_inv - r)[far].max() < 1e-9
    assert np.abs(det * det_inv - 1)[far].max() < 1e-9

    # the lab momenta conserve momentum as the partonic ones do (RAMBO, up to
    # 1e-11 of sqrt(s_hat) near the edge of phase space), the violation
    # stretched by at most e^{|y|} by the boost, and stay on their mass shell
    scale = p[..., 0].max(1)
    violation = np.abs(p[:, 2:].sum(1) - p[:, :2].sum(1)).max(1)
    partonic_violation = np.abs(exact[:, 2:].sum(1) - exact[:, :2].sum(1)).max(1)
    stretch = np.exp(np.abs(rapidity))
    bound = stretch * partonic_violation + 1e-14 * scale
    assert (violation[ok] <= bound[ok]).all()
    m2 = p[..., 0] ** 2 - (p[..., 1:] ** 2).sum(-1)
    off_shell = np.abs(m2 - np.asarray(masses) ** 2) / scale[:, None] ** 2
    assert off_shell[ok].max() < 1e-13
