"""A minimum on the partonic sqrt(s), and the phase-space boundary it implies.

A cut on sqrt(s_hat) is a statement about the whole final state: nothing below
it can ever be generated, so the root propagator -- the virtuality the
luminosity mapping samples -- has a floor. Handing that floor to the sampler is
the difference between generating the region the cut allows and generating
everything and throwing nearly all of it away.

The process is g g > t t~, which has a kinematic threshold of its own at
2 m_t = 346 GeV. That threshold is what makes the central invariant testable:
a cut placed *below* it cannot remove a single point, so it must leave the
integral -- and here even the sampled points themselves -- untouched. A cut
placed above it must move the boundary without moving the integral.

Pinned down here:

  * obs_sqrt_s is the invariant mass of the two incoming partons,
  * a cut below the kinematic threshold changes nothing at all,
  * with a cut above it, no point below the cut is generated,
  * and doing so does not change the volume of the cut region, i.e. the
    boundary is a consequence of the cut and not an extra cut of its own,
  * a cut above the beam energy leaves an empty, finite result rather than an
    inverted sampling range.
"""

import functools
import math

import phasespace_helpers
import torch

import madspace as ms

CM_ENERGY = 13000.0
M_TOP = 173.0
BATCH_SIZE = 100_000
SEED = 20260912

# g g > t t~
PIDS = [21, 21, 6, -6]
MASSES = [0.0, 0.0, M_TOP, M_TOP]
# the process cannot produce less than this, whatever the cut says
THRESHOLD = 2 * M_TOP
# below the threshold: the cut cannot bite, so nothing may change
NO_OP_CUT = 300.0
# above it: the cut removes a large fraction of the points and enough of the
# volume that a boundary which failed to compensate could not hide in the
# statistics (see test_the_cut_removes_more_than_the_comparison_could_hide)
BITING_CUT = 4000.0


def sqrt_s_cuts(minimum):
    O = ms.Observable
    return ms.Cuts([ms.CutItem(O(PIDS, O.obs_sqrt_s, []), min=minimum)])


def mapping(cuts=None, leptonic=False):
    return ms.PhaseSpaceMapping(MASSES, CM_ENERGY, cuts=cuts, leptonic=leptonic)


sample = functools.partial(phasespace_helpers.sample, seed=SEED, n=BATCH_SIZE)


def s_hat(p):
    """sqrt of the partonic Mandelstam s, from the two incoming momenta."""
    return phasespace_helpers.invariant_mass(p, 0, 1)


def volume(det, p):
    """The integral the mapping estimates, with non-finite points dropped."""
    return phasespace_helpers.finite_weight(p, det)


# --------------------------------------------------------------------------
# the observable itself
# --------------------------------------------------------------------------


def test_obs_sqrt_s_is_the_partonic_invariant_mass():
    p_ext, _ = sample(mapping())
    O = ms.Observable
    computed = O(PIDS, O.obs_sqrt_s, [])(p_ext).reshape(-1)
    torch.testing.assert_close(computed, s_hat(p_ext), rtol=1e-9, atol=1e-12)
    # and it is the invariant mass of the final state too, i.e. it really is
    # the quantity the whole event has to clear
    out = p_ext[:, 2, :] + p_ext[:, 3, :]
    m_out = torch.sqrt(
        torch.clamp(out[:, 0] ** 2 - torch.sum(out[:, 1:] ** 2, dim=1), min=0.0)
    )
    torch.testing.assert_close(computed, m_out, rtol=1e-6, atol=1e-12)


def test_the_process_has_a_threshold_of_its_own():
    """The rest of the file leans on this: 2 m_t is a floor the mapping already
    respects, so a cut below it has nothing left to forbid."""
    p_ext, _ = sample(mapping())
    assert s_hat(p_ext).min() >= THRESHOLD - 1e-6
    assert NO_OP_CUT < THRESHOLD < BITING_CUT


# --------------------------------------------------------------------------
# a cut that cannot bite
# --------------------------------------------------------------------------


def test_a_cut_below_the_threshold_changes_nothing():
    """The invariant the boundary must respect: the floor is allowed to narrow
    the sampled range only as far as the cut actually forbids something. Below
    the kinematic threshold it forbids nothing, so the same random numbers must
    give back the very same points with the very same weights -- which makes
    the integral unchanged in the sharpest possible sense."""
    p_free, det_free = sample(mapping())
    p_cut, det_cut = sample(mapping(cuts=sqrt_s_cuts(NO_OP_CUT)))
    assert torch.equal(det_free, det_cut)
    assert torch.equal(p_free, p_cut)


def test_a_cut_at_zero_changes_nothing():
    """The same statement for the value the run card ships with."""
    p_free, det_free = sample(mapping())
    p_cut, det_cut = sample(mapping(cuts=sqrt_s_cuts(0.0)))
    assert torch.equal(det_free, det_cut)
    assert torch.equal(p_free, p_cut)


# --------------------------------------------------------------------------
# the boundary a cut above the threshold implies
# --------------------------------------------------------------------------


def test_cut_bounds_the_sampled_s_hat():
    """With the cut handed to the mapping nothing below it is generated, and
    nothing generated has to be thrown away either."""
    p_ext, det = sample(mapping(cuts=sqrt_s_cuts(BITING_CUT)))
    s = s_hat(p_ext)
    assert s.min() >= BITING_CUT - 1e-6
    # the region has not collapsed onto its own boundary
    assert s.max() > 2 * BITING_CUT
    # and the cut filters nothing out
    assert torch.all(det != 0.0)


def test_without_the_bound_the_same_region_is_wasted():
    """The counterpart of the test above: without the boundary the sampler puts
    a real fraction of its points where the cut throws them away. That waste is
    what the boundary removes, so if it ever stops being true the test above
    has stopped proving anything."""
    p_ext, _ = sample(mapping())
    assert (s_hat(p_ext) < BITING_CUT).double().mean() > 0.25


def test_cut_volume_is_unchanged_by_the_boundary():
    """The boundary must follow from the cut, not add to it: integrating the
    cut region has to give the same answer whether the mapping knows about the
    cut or only the filtering does."""
    p_free, det_free = sample(mapping(), seed=SEED)
    passes = s_hat(p_free) >= BITING_CUT
    external = torch.where(passes, volume(det_free, p_free), 0.0)

    p_cut, det_cut = sample(mapping(cuts=sqrt_s_cuts(BITING_CUT)), seed=SEED + 1)
    piped = volume(det_cut, p_cut)

    error = math.sqrt(
        external.std(correction=0).item() ** 2 / len(external)
        + piped.std(correction=0).item() ** 2 / len(piped)
    )
    assert error > 0.0
    assert abs(external.mean() - piped.mean()) < 5.0 * error


def test_the_cut_removes_more_than_the_comparison_could_hide():
    """Keeps the comparison above honest: if the volume the cut removes were
    smaller than the precision of that comparison, the two numbers would agree
    for uninteresting reasons. It has to be well outside the tolerance used
    there, not merely nonzero."""
    p_free, det_free = sample(mapping())
    total = volume(det_free, p_free)
    removed = torch.where(s_hat(p_free) < BITING_CUT, total, 0.0)
    error = total.std(correction=0).item() / math.sqrt(len(total))
    assert error > 0.0
    assert removed.mean() > 10.0 * error


# --------------------------------------------------------------------------
# the cases where there is no range to narrow
# --------------------------------------------------------------------------


def test_a_cut_above_the_beam_energy_is_empty_and_finite():
    """Such a cut leaves nothing to sample at all. It must come out as an empty
    result, not as an inverted sampling range with nan weights."""
    p_ext, det = sample(mapping(cuts=sqrt_s_cuts(2 * CM_ENERGY)))
    assert torch.all(det == 0.0)
    assert torch.all(torch.isfinite(det))
    assert torch.all(torch.isfinite(p_ext))


def test_leptonic_s_hat_is_fixed_so_the_cut_only_filters():
    """A leptonic collision has s_hat fixed at the beam energy: there is no
    luminosity to sample and hence no range to narrow, so the cut stays a plain
    filter and must still be applied as one."""
    p_ext, det = sample(mapping(leptonic=True))
    s = s_hat(p_ext)
    torch.testing.assert_close(s, torch.full_like(s, CM_ENERGY), rtol=1e-9, atol=1e-12)
    _, det_kept = sample(mapping(cuts=sqrt_s_cuts(BITING_CUT), leptonic=True))
    assert torch.equal(det_kept, det)
    _, det_gone = sample(mapping(cuts=sqrt_s_cuts(2 * CM_ENERGY), leptonic=True))
    assert torch.all(det_gone == 0.0)
