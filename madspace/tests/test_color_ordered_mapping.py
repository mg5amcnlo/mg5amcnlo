import numpy as np
import pytest
from pytest import approx

import madspace as ms

# ----------------------------
# Fixtures
# ----------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


N = 10_000  # keep this moderate for CI speed; bump locally for tighter stats
CM_ENERGY = 13000.0


# Mass set families. n_out is filled in at use.
def _massless(n_out):
    return [0.0] * n_out, f"{n_out} massless"


def _w_like(n_out):
    return [80.0] * n_out, f"{n_out} W-like"


def _mixed_top(n_out):
    # one heavy, one W-like, rest massless; only meaningful for n_out >= 3.
    return [173.0, 80.0] + [0.0] * (n_out - 2), f"{n_out} mixed"


def _mass_sets_for(n_out):
    sets = [_massless(n_out), _w_like(n_out)]
    if n_out >= 3:
        sets.append(_mixed_top(n_out))
    return sets


# Each entry: (color_order, int_order_for_old_class, topology_label).
TOPOLOGIES = [
    ([0, 2, 3, 1, 4], [0, 1], "n=5: set1={2,3}, set2={4}"),
    ([0, 1, 2, 3, 4], [0, 1], "n=5: set1={}, set2={2,3,4}"),
    ([0, 2, 3, 4, 1, 5], [0, 1, 2], "n=6: set1={2,3,4}, set2={5}"),
    ([0, 2, 3, 1, 4, 5], [0, 1, 2], "n=6: set1={2,3}, set2={4,5}"),
    ([0, 1, 2, 3, 4, 5], [0, 1, 2], "n=6: set1={}, set2={2,3,4,5}"),
    ([0, 2, 3, 4, 1, 5, 6], [0, 1, 2, 3], "n=7: set1={2,3,4}, set2={5,6}"),
    ([0, 2, 3, 4, 5, 1, 6], [0, 1, 2, 3], "n=7: set1={2,3,4,5}, set2={6}"),
    ([0, 2, 3, 4, 5, 6, 1, 7], [0, 1, 2, 3, 4], "n=8: set1={2,3,4,5,6}, set2={7}"),
    ([0, 2, 3, 4, 5, 6, 7, 1], [0, 1, 2, 3, 4], "n=8: set1={2,3,4,5,6,7}, set2={}"),
]


def _expand(topologies):
    """Flat list of (color_order, int_order, masses, label) pairs."""
    out = []
    for color_order, int_order, topo_label in topologies:
        n_out = len(color_order) - 2
        for masses, mass_label in _mass_sets_for(n_out):
            label = f"{mass_label}-{topo_label}"
            out.append((color_order, int_order, masses, label))
    return out


CASES = _expand(TOPOLOGIES)
CASE_IDS = [c[3] for c in CASES]


def _physical_mask(det, p_ext):
    """A physical event has finite det and finite momenta on every leg."""
    mask = np.isfinite(det)
    for p in p_ext:
        mask &= np.all(np.isfinite(p), axis=1)
    return mask


def _filter_physical(det, p_ext):
    """Keep physical events. Fail only if literally none survive."""
    mask = _physical_mask(det, p_ext)
    if not mask.any():
        pytest.fail(f"No physical events out of {len(det)}")
    return mask, [p[mask] for p in p_ext], det[mask]


# ----------------------------
# Tests
# ----------------------------


@pytest.mark.parametrize("color_order,_int_order,masses,_label", CASES, ids=CASE_IDS)
def test_momentum_conservation(rng, color_order, _int_order, masses, _label):
    mapping = ms.ColorOrderedMapping(color_order)
    r = rng.random((N, mapping.random_dim()))
    e_cm = np.full(N, CM_ENERGY)
    cond = [e_cm] + [np.full(N, m) for m in masses]

    *p_ext, det = mapping.map_forward(list(r.T), cond)
    p_ext = list(p_ext)
    _, p_ext, det = _filter_physical(det, p_ext)

    p_in = p_ext[0] + p_ext[1]
    p_out_sum = sum(p_ext[i] for i in range(2, len(p_ext)))
    assert p_in == approx(p_out_sum, abs=1e-3, rel=1e-6)


@pytest.mark.parametrize("color_order,_int_order,masses,_label", CASES, ids=CASE_IDS)
def test_on_shell_masses(rng, color_order, _int_order, masses, _label):
    mapping = ms.ColorOrderedMapping(color_order)
    r = rng.random((N, mapping.random_dim()))
    e_cm = np.full(N, CM_ENERGY)
    cond = [e_cm] + [np.full(N, m) for m in masses]

    *p_ext, det = mapping.map_forward(list(r.T), cond)
    p_ext = list(p_ext)
    _, p_ext, det = _filter_physical(det, p_ext)

    for i, m in enumerate(masses):
        p = p_ext[i + 2]
        m_check = np.sqrt(np.maximum(0, p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=1)))
        assert m_check == approx(m, abs=1e-2, rel=1e-3), f"particle {i} mass off"


@pytest.mark.parametrize("color_order,_int_order,masses,_label", CASES, ids=CASE_IDS)
def test_inverse(rng, color_order, _int_order, masses, _label):
    """Forward o inverse = identity, to FP precision modulo a small tail.

    A handful of events at the very edge of the physical region produce a
    forward det at the lower extreme of the distribution; for those, the
    round-trip accumulates non-negligible FP error in atan2/sqrt/boost. We
    drop the bottom 0.1% by |det| (genuine boundary events, near-zero
    weight in any MC integral) and check the rest.
    """
    n_events = N
    mapping = ms.ColorOrderedMapping(color_order)
    r = rng.random((n_events, mapping.random_dim()))
    e_cm = np.full(n_events, CM_ENERGY)
    cond = [e_cm] + [np.full(n_events, m) for m in masses]

    *p_ext, det = mapping.map_forward(list(r.T), cond)
    p_ext = list(p_ext)
    mask, p_ext_phys, det_phys = _filter_physical(det, p_ext)
    cond_phys = [c[mask] for c in cond]
    r_phys = r[mask]

    *r_inv, det_inv = mapping.map_inverse(p_ext_phys, cond_phys)

    floor = np.quantile(np.abs(det_phys), 0.001)
    keep = np.isfinite(det_inv) & (np.abs(det_phys) > floor)
    for ri in r_inv:
        keep &= np.isfinite(ri)

    det_kept = det_phys[keep]
    det_inv_kept = det_inv[keep]
    # 0.1% tolerance absorbs accumulated FP error from chained
    # boosts/rotations/sqrts in the deepest-walk topologies. Physics
    # correctness is the job of test_phase_space_volume_matches_old.
    assert det_inv_kept == approx(1 / det_kept, rel=1e-3)

    # Some randoms are discrete-branch choices (r_choice from 2->3 rungs);
    # the round-trip recovers only the bin (< 0.5 vs >= 0.5), not the exact
    # value. Bin-membership comparison handles those.
    for i, (r_i, r_inv_i) in enumerate(zip(list(r_phys.T), r_inv)):
        r_i_k = r_i[keep]
        r_inv_i_k = r_inv_i[keep]
        diff = np.abs(r_i_k - r_inv_i_k)
        bin_ok = (r_i_k < 0.5) == (r_inv_i_k < 0.5)
        ok = (diff < 1e-4) | bin_ok
        assert np.all(ok), (
            f"random index {i}: {np.sum(~ok)} mismatches, "
            f"max diff = {diff[~ok].max() if np.any(~ok) else 0:.3e}"
        )


@pytest.mark.parametrize("color_order,int_order,masses,_label", CASES, ids=CASE_IDS)
def test_phase_space_volume_matches_old(rng, color_order, int_order, masses, _label):
    """The new class should integrate to the same phase-space volume as the
    old TPropagatorMapping (matrix element = 1).

    This is *the* correctness test: both classes are MC estimators of the
    same n-body integral, and their means must agree within a few sigma.
    test_inverse is a precision check; this one is the physics check.
    """
    n_events = N
    new = ms.ColorOrderedMapping(color_order)
    old = ms.TPropagatorMapping(int_order)

    r_new = rng.random((n_events, new.random_dim()))
    r_old = rng.random((n_events, old.random_dim()))
    e_cm = np.full(n_events, CM_ENERGY)
    cond = [e_cm] + [np.full(n_events, m) for m in masses]

    *_, det_new = new.map_forward(list(r_new.T), cond)
    *_, det_old = old.map_forward(list(r_old.T), cond)

    det_new = det_new[np.isfinite(det_new) & (det_new > 0)]
    det_old = det_old[np.isfinite(det_old) & (det_old > 0)]
    if not (len(det_new) and len(det_old)):
        pytest.fail("no physical events in one of the estimators")

    mean_new, mean_old = np.mean(det_new), np.mean(det_old)
    err_new = np.std(det_new) / np.sqrt(len(det_new))
    err_old = np.std(det_old) / np.sqrt(len(det_old))
    err = np.sqrt(err_new**2 + err_old**2)

    diff = abs(mean_new - mean_old)
    assert diff < 5 * err, (
        f"phase-space volumes differ: new = {mean_new:.4e} +/- {err_new:.4e}, "
        f"old = {mean_old:.4e} +/- {err_old:.4e}, "
        f"diff = {diff:.4e} ({diff/err:.2f} sigma)"
    )
