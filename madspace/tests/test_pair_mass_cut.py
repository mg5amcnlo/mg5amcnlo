"""Two-particle invariant mass cuts, and the phase-space boundary they imply.

A cut on the invariant mass of a pair is not only a filter applied to finished
events. Where the pair is exactly what a propagator decays into, the cut is a
floor on that propagator's invariant mass, and handing it to the sampler is the
difference between generating the allowed region and generating everything and
throwing most of it away.

The process here is g g > t t~ g g, whose first diagram has the top pair coming
off a single propagator ('o1', 'o0' -> 'p0'), so a cut on m(t t~) lands exactly
on one node of the decay tree.

Three things are pinned down:

  * obs_pair_mass computes the invariant mass of the pair,
  * with the cut handed to the mapping, no sampled point falls below it,
  * doing so does not change the volume of the cut region, i.e. the boundary
    is a consequence of the cut and not an extra cut of its own.
"""

import functools
import json
import math
import os

import phasespace_helpers
import torch
from phasespace_helpers import invariant_mass

import madspace as ms

CM_ENERGY = 13000.0
M_TOP = 173.0
BATCH_SIZE = 100_000
SEED = 20260908

TEST_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data")

# g g > t t~ g g: two gluons in, then t, t~, g, g
PIDS = [21, 21, 6, -6, 21, 21]
TOP_PIDS = [6, -6]
# well above threshold, so the cut removes a real part of both the points
# and the volume rather than shaving off a tail
PAIR_MASS_CUT = 2500.0


def load_topology():
    with open(os.path.join(TEST_DATA, "ttgg.json")) as f:
        diagram = json.load(f)[0]
    assert ["o1", "o0", "p0"] in diagram["vertices"], (
        "this test needs the diagram in which the top pair comes off one "
        "propagator; the fixture has changed"
    )
    return ms.Topology(
        ms.Diagram(
            diagram["incoming_masses"],
            diagram["outgoing_masses"],
            [ms.Propagator(*p) for p in diagram["propagators"]],
            diagram["vertices"],
        )
    )


def pair_mass_cuts(minimum):
    O = ms.Observable
    return ms.Cuts([ms.CutItem(O(PIDS, O.obs_pair_mass, [TOP_PIDS]), min=minimum)])


def mapping(cuts=None):
    return ms.PhaseSpaceMapping(load_topology(), CM_ENERGY, cuts=cuts)


sample = functools.partial(phasespace_helpers.sample, seed=SEED, n=BATCH_SIZE)


# --------------------------------------------------------------------------
# the observable itself
# --------------------------------------------------------------------------


def test_obs_pair_mass_is_the_invariant_mass_of_the_pair():
    p_ext, _ = sample(mapping())
    O = ms.Observable
    observable = O(PIDS, O.obs_pair_mass, [TOP_PIDS])
    computed = observable(p_ext).reshape(p_ext.shape[0], -1)
    # the two tops are the only pair the selection can form
    assert computed.shape[1] == 1
    torch.testing.assert_close(
        computed[:, 0], invariant_mass(p_ext, 2, 3), rtol=1e-9, atol=1e-12
    )


def test_obs_pair_mass_is_not_the_single_particle_mass():
    """Guards against obs_pair_mass quietly resolving to obs_mass, which would
    return the top mass for every event and make the cut meaningless."""
    p_ext, _ = sample(mapping())
    O = ms.Observable
    pair = O(PIDS, O.obs_pair_mass, [TOP_PIDS])(p_ext).reshape(-1)
    single = O(PIDS, O.obs_mass, [TOP_PIDS])(p_ext).reshape(p_ext.shape[0], -1)
    torch.testing.assert_close(
        single, torch.full_like(single, M_TOP), rtol=1e-6, atol=1e-12
    )
    assert torch.all(pair > 2 * M_TOP - 1e-6)
    assert pair.median() > 2.2 * M_TOP


# --------------------------------------------------------------------------
# the boundary the cut implies
# --------------------------------------------------------------------------


def test_cut_bounds_the_sampled_invariant():
    """With the cut handed to the mapping nothing below it is generated."""
    p_ext, _ = sample(mapping(cuts=pair_mass_cuts(PAIR_MASS_CUT)))
    pair = invariant_mass(p_ext, 2, 3)
    assert pair.min() >= PAIR_MASS_CUT - 1e-6
    # and the cut has not collapsed the region onto its own boundary
    assert pair.max() > 1.5 * PAIR_MASS_CUT


def test_without_the_cut_the_same_region_is_mostly_wasted():
    """The counterpart of the test above: without the cut the sampler spends
    most of its points where the cut would have thrown them away. This is what
    the boundary is there to avoid, so if it ever stops being true the test
    above has stopped proving anything."""
    p_ext, _ = sample(mapping())
    pair = invariant_mass(p_ext, 2, 3)
    assert (pair < PAIR_MASS_CUT).double().mean() > 0.5


def test_cut_volume_is_unchanged_by_the_boundary():
    """The boundary must follow from the cut, not add to it: integrating the
    cut region has to give the same answer whether the mapping knows about the
    cut or only the filtering does."""
    p_free, det_free = sample(mapping(), seed=SEED)
    passes = invariant_mass(p_free, 2, 3) >= PAIR_MASS_CUT
    external = torch.where(
        passes, phasespace_helpers.finite_weight(p_free, det_free), 0.0
    )

    p_cut, det_cut = sample(mapping(cuts=pair_mass_cuts(PAIR_MASS_CUT)), seed=SEED + 1)
    piped = phasespace_helpers.finite_weight(p_cut, det_cut)

    mean_external = external.mean().item()
    mean_piped = piped.mean().item()
    error = math.sqrt(
        external.std(correction=0).item() ** 2 / len(external)
        + piped.std(correction=0).item() ** 2 / len(piped)
    )
    assert error > 0.0
    assert abs(mean_external - mean_piped) < 5.0 * error


def test_cut_volume_is_a_real_fraction_of_the_total():
    """Keeps the comparison above honest: if the cut removed nothing, the two
    volumes would agree for uninteresting reasons."""
    _, det_free = sample(mapping())
    p_free, det_full = sample(mapping())
    passes = invariant_mass(p_free, 2, 3) >= PAIR_MASS_CUT
    kept = torch.where(passes & torch.isfinite(det_full), det_full, 0.0).mean()
    total = torch.where(torch.isfinite(det_full), det_full, 0.0).mean()
    assert 0.01 < kept / total < 0.95
