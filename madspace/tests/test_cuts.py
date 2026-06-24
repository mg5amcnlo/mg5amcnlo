"""Cut-piping consistency check across t-channel modes, variants, and cut levels.

The cut phase-space volume V_cut = INT dPhi * Theta(cuts) is a property of the
integrand, not of how it is sampled.  So it must come out the same regardless of

  * which t-channel mode PhaseSpaceMapping uses (rambo / propagator /
    color_ordered), and
  * whether the cut is applied only externally (sample the full phase space, then
    reject points failing the physical cut) or also piped through the mapping
    (cuts=Cuts(...), so the mapping concentrates its sampling in the cut region).

Piping the cut changes the sampling distribution (and the variance), never the
integral.  rambo (no t-channel cut machinery at all) with the cut applied
externally is the ground-truth reference; every (mode, piped) combination must
agree with it within MC errors.

Cut levels
----------
Each variant is probed at several pt cuts spanning a soft, LHC-like floor up to a
hard, high-energy one.  The cut is scaled per variant as
CUT_FRACTION * sqrt(s) / n_jets
"""

import math

import numpy as np
import pytest

import madspace as ms

PSM = ms.PhaseSpaceMapping

CM_ENERGY = 13000.0  # fixed leptonic CM energy (no PDF / x1,x2 convolution)
N_SAMPLES = 300_000  # MC points per (variant, cut level, configuration)
TOL_SIGMA = 4.0  # agreement required within this many combined MC sigmas
BASE_SEED = 2024

# pt cut = CUT_FRACTION * sqrt(s) / n_jets.  Soft (LHC-like, ~30-65 GeV) to hard
# (~650-1625 GeV).  For reference, at sqrt(s) = 13 TeV and a di-jet final state:
#   0.01 -> 65 GeV,  0.05 -> 325 GeV,  0.15 -> 975 GeV,  0.25 -> 1625 GeV.
CUT_FRACTIONS = [0.01, 0.05, 0.15, 0.25, 0.5]

M_TOP = 173.0
M_W = 80.4
M_Z = 91.19

# (label, outgoing masses, outgoing PDG ids).  Incoming is always two gluons.
# Only gluons (pid 21) are jets and thus carry the pt cut.
VARIANTS = [
    ("gg", [0.0, 0.0], [21, 21]),
    ("ggg", [0.0, 0.0, 0.0], [21, 21, 21]),
    ("gggg", [0.0, 0.0, 0.0, 0.0], [21, 21, 21, 21]),
    ("ggggg", [0.0, 0.0, 0.0, 0.0, 0.0], [21, 21, 21, 21, 21]),
    ("ttg", [M_TOP, M_TOP, 0.0], [6, -6, 21]),
    ("ttgg", [M_TOP, M_TOP, 0.0, 0.0], [6, -6, 21, 21]),
    ("ttggg", [M_TOP, M_TOP, 0.0, 0.0, 0.0], [6, -6, 21, 21, 21]),
    ("ttgggg", [M_TOP, M_TOP, 0.0, 0.0, 0.0, 0.0], [6, -6, 21, 21, 21, 21]),
    ("Wg", [M_W, 0.0], [24, 21]),
    ("Wgg", [M_W, 0.0, 0.0], [24, 21, 21]),
    ("Wggg", [M_W, 0.0, 0.0, 0.0], [24, 21, 21, 21]),
    ("Zgg", [M_Z, 0.0, 0.0], [23, 21, 21]),
]

CONFIGS = [
    (PSM.propagator, False),
    (PSM.propagator, True),
    (PSM.color_ordered, False),
    (PSM.color_ordered, True),
]


def _mode_name(mode):
    return str(mode).rsplit(".", 1)[-1]


def _n_jets(pids_out):
    return sum(1 for pid in pids_out if pid == 21)


def _pt_min(pids_out, fraction):
    """Per-variant pt cut, scaled with jet multiplicity so the cut region is
    always non-vacuous (n_jets jets each clearing pt need n_jets*pt < sqrt(s))."""
    return fraction * CM_ENERGY / max(_n_jets(pids_out), 1)


def _color_order(mode, n_out):
    if mode != PSM.color_ordered:
        return None
    return [0] + [i + 2 for i in range(n_out)] + [1]


def _inputs(mapping, rng, n):
    inputs = [rng.random((n, mapping.random_dim()))]
    dd = mapping.discrete_dim()
    if dd:
        inputs.append(rng.integers(0, 2, size=(n, dd)).astype(np.int32))
    return inputs


def _cuts(pids, pt_min):
    O = ms.Observable
    return ms.Cuts([ms.CutItem(O(pids, O.obs_pt, [O.jet_pids]), min=pt_min)])


def _pt(p):  # p[..., (E, px, py, pz)]
    return np.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)


def _cut_volume(masses_out, pids_out, mode, piped, pt_min, seed, n=N_SAMPLES):
    """MC estimate of INT dPhi * Theta(jet pt > pt_min) and its statistical error."""
    masses = [0.0, 0.0] + list(masses_out)
    pids = [21, 21] + list(pids_out)
    jet_idx = [i for i, pid in enumerate(pids_out) if pid == 21]

    mapping = PSM(
        masses,
        CM_ENERGY,
        mode=mode,
        leptonic=True,
        color_order=_color_order(mode, len(masses_out)),
        cuts=_cuts(pids, pt_min) if piped else None,
    )
    rng = np.random.default_rng(seed)
    p_ext, _x1, _x2, det = mapping.map_forward(_inputs(mapping, rng, n))
    # det is the per-branch Jacobian; restore the 2-solution multiplicity per 2->3
    # peel (owned externally) to recover the full volume.
    det = np.asarray(det) * 2.0 ** mapping.discrete_dim()
    p_out = np.asarray(p_ext)[:, 2:, :]

    if jet_idx:
        passes = np.all(_pt(p_out)[:, jet_idx] >= pt_min, axis=1)
    else:
        passes = np.ones(n, dtype=bool)
    finite = np.isfinite(det) & np.all(np.isfinite(p_out), axis=(1, 2))
    weights = np.nan_to_num(np.where(passes & finite, det, 0.0))

    return float(weights.mean()), float(weights.std() / math.sqrt(n))


# (variant index, fraction index, label, masses, pids, fraction) for every
# (variant, cut level) combination.
_CASES = [
    (vi, fi, label, masses, pids, frac)
    for vi, (label, masses, pids) in enumerate(VARIANTS)
    for fi, frac in enumerate(CUT_FRACTIONS)
]


@pytest.mark.parametrize(
    "vi, fi, label, masses_out, pids_out, fraction",
    _CASES,
    ids=[f"{c[2]}-f{c[5]:g}" for c in _CASES],
)
def test_cut_volume_consistent(vi, fi, label, masses_out, pids_out, fraction):
    """Cut volume must match the rambo+external reference for every mode/piping."""
    pt_min = _pt_min(pids_out, fraction)
    seed0 = BASE_SEED + 1000 * vi + 100 * fi

    ref, ref_err = _cut_volume(masses_out, pids_out, PSM.rambo, False, pt_min, seed0)
    if ref <= 0.0 or not math.isfinite(ref):
        pytest.skip(
            f"{label}: cut volume vanishes at pt_min={pt_min:.0f} GeV "
            f"({_n_jets(pids_out)} jets exceed sqrt(s)); nothing to compare."
        )

    rows = []
    for ci, (mode, piped) in enumerate(CONFIGS, start=1):
        val, err = _cut_volume(masses_out, pids_out, mode, piped, pt_min, seed0 + ci)
        sigma = math.sqrt(err**2 + ref_err**2)
        pull = abs(val - ref) / sigma if sigma > 0 else math.inf
        rows.append((mode, piped, val, err, pull))

    failures = [r for r in rows if r[4] > TOL_SIGMA]
    if failures:
        lines = [
            f"{label}: cut volume disagrees with the rambo+external reference "
            f"(tolerance {TOL_SIGMA:g} sigma, pt_min={pt_min:.0f} GeV, "
            f"fraction={fraction:g}, N={N_SAMPLES:_}, {_n_jets(pids_out)} jets).",
            f"  reference  rambo            external : " f"{ref:.6e} +/- {ref_err:.2e}",
        ]
        for mode, piped, val, err, pull in rows:
            mark = "FAIL" if pull > TOL_SIGMA else "ok  "
            kind = "piped   " if piped else "external"
            lines.append(
                f"  {mark}       {_mode_name(mode):14} {kind} : "
                f"{val:.6e} +/- {err:.2e}   ({pull:5.1f} sigma)"
            )
        pytest.fail("\n".join(lines), pytrace=False)


# Concentration is only expected when at least two jets are peeled with a t_min
# cut; a single (residual) jet is fixed by momentum conservation, not sampled with
# the cut, so piping cannot concentrate it.  Checked at the hardest cut level,
# where the effect is largest.
_CONC = [v for v in VARIANTS if _n_jets(v[2]) >= 2 and len(v[1]) >= 3]


@pytest.mark.parametrize(
    "idx, label, masses_out, pids_out",
    [(i, *v) for i, v in enumerate(_CONC)],
    ids=[v[0] for v in _CONC],
)
def test_piping_concentrates_samples(idx, label, masses_out, pids_out):
    """Sanity: piping the cut must actually change the sampling (not be ignored).

    With the cut piped in, the mapping should land in the cut region more often
    than when the same cut is only applied externally -- a direct check that the
    cut reaches the t-channel layers rather than being silently dropped.
    """
    pt_min = _pt_min(pids_out, CUT_FRACTIONS[-1])
    pids = [21, 21] + list(pids_out)
    jet_idx = [i for i, pid in enumerate(pids_out) if pid == 21]

    def pass_fraction(piped, seed):
        mapping = PSM(
            [0.0, 0.0] + list(masses_out),
            CM_ENERGY,
            mode=PSM.color_ordered,
            leptonic=True,
            color_order=_color_order(PSM.color_ordered, len(masses_out)),
            cuts=_cuts(pids, pt_min) if piped else None,
        )
        rng = np.random.default_rng(seed)
        p_ext, *_ = mapping.map_forward(_inputs(mapping, rng, 100_000))
        p_out = np.asarray(p_ext)[:, 2:, :]
        return float(np.all(_pt(p_out)[:, jet_idx] >= pt_min, axis=1).mean())

    frac_ext = pass_fraction(False, BASE_SEED + 100 * idx + 11)
    frac_pipe = pass_fraction(True, BASE_SEED + 100 * idx + 12)
    assert frac_pipe > frac_ext, (
        f"{label}: piping the cut did not concentrate sampling at "
        f"pt_min={pt_min:.0f} GeV: external pass-fraction={frac_ext:.3f}, "
        f"piped pass-fraction={frac_pipe:.3f} (expected piped to be higher)."
    )
