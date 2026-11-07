"""Cut volume of PhaseSpaceMapping modes (with and without the cuts handed to the
mapping) against RAMBO with the cuts applied afterwards."""

import math
from dataclasses import dataclass

import pytest
import torch
from madspace_testing import generator, rand, randint

import madspace as ms

PSM = ms.PhaseSpaceMapping

CM_ENERGY = 13000.0
N_SAMPLES = 300_000
TOL_SIGMA = 5.0
BASE_SEED = 2024

CUT_FRACTIONS = [0.05, 0.10, 0.20, 0.50]
CUT_DELTAR_MIN = 0.4

M_TOP, M_W, M_Z = 173.0, 80.4, 91.19


VARIANTS = [
    ("ggg", [0.0, 0.0, 0.0], [21, 21, 21]),
    ("gggg", [0.0, 0.0, 0.0, 0.0], [21, 21, 21, 21]),
    ("ggggg", [0.0, 0.0, 0.0, 0.0, 0.0], [21, 21, 21, 21, 21]),
    ("ttgg", [M_TOP, M_TOP, 0.0, 0.0], [6, -6, 21, 21]),
    ("Wgg", [M_W, 0.0, 0.0], [24, 21, 21]),
    ("Zgg", [M_Z, 0.0, 0.0], [23, 21, 21]),
]


@dataclass(frozen=True)
class Config:
    topology: str
    mode: any
    color_order: list[int] | None
    piped: bool


def _n_jets(pids_out):
    return sum(1 for pid in pids_out if pid == 21)


def _pt_min(pids_out, fraction):
    return fraction * CM_ENERGY / max(_n_jets(pids_out), 1)


def _color_orders(n_out):
    """Each ordering is a distinct physical topology."""
    base = [i + 2 for i in range(n_out)]

    orders = [("empty-set (single chain)", [0, *base, 1])]

    for pos in range(1, n_out):
        left, right = base[:pos], base[pos:]
        orders.append(
            (f"split topology | left={left} right={right}", [0, *left, 1, *right])
        )

    return orders


def _inputs(mapping, g, n):
    inputs = [rand((n, mapping.random_dim()), generator=g)]
    dd = mapping.discrete_dim()
    if dd:
        inputs.append(randint(0, 2, (n, dd), generator=g, dtype=torch.int32))
    return inputs


def _cuts(pids, pt_min):
    O = ms.Observable
    return ms.Cuts(
        [
            ms.CutItem(O(pids, O.obs_pt, [O.jet_pids]), min=pt_min),
            ms.CutItem(O(pids, O.obs_delta_r, [O.jet_pids]), min=CUT_DELTAR_MIN),
        ]
    )


def _feats(p):
    pt = torch.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)
    safe_pt = torch.where(pt > 0, pt, 1.0)
    eta = torch.arcsinh(torch.where(pt > 0, p[..., 3] / safe_pt, 0.0))
    phi = torch.atan2(p[..., 2], p[..., 1])
    return pt, eta, phi


def _passes(p_out, jet_idx, pt_min):
    if not jet_idx:
        return torch.ones(p_out.shape[0], dtype=torch.bool)

    pt, eta, phi = _feats(p_out)

    ok = torch.all(pt[:, jet_idx] >= pt_min, dim=1)

    for a in range(len(jet_idx)):
        for b in range(a + 1, len(jet_idx)):
            i, j = jet_idx[a], jet_idx[b]
            dphi = torch.abs(phi[:, i] - phi[:, j])
            dphi = torch.minimum(dphi, 2 * math.pi - dphi)
            dR = torch.sqrt((eta[:, i] - eta[:, j]) ** 2 + dphi**2)
            ok = ok & (dR >= CUT_DELTAR_MIN)

    return ok


def _cut_volume(
    masses_out, pids_out, mode, color_order, piped, pt_min, seed, n=N_SAMPLES
):
    masses = [0.0, 0.0] + list(masses_out)
    pids = [21, 21] + list(pids_out)
    jet_idx = [i for i, pid in enumerate(pids_out) if pid == 21]

    mapping = PSM(
        masses,
        CM_ENERGY,
        mode=mode,
        leptonic=True,
        color_ordered=(
            None if color_order is None else ms.ColorOrderedMapping.Options(color_order)
        ),
        cuts=_cuts(pids, pt_min) if piped else None,
    )

    g = generator(seed)
    p_ext, _x1, _x2, det = mapping.map_forward(_inputs(mapping, g, n))

    det = det * 2.0 ** mapping.discrete_dim()
    p_out = p_ext[:, 2:, :]

    passes = _passes(p_out, jet_idx, pt_min)
    finite = torch.isfinite(det) & torch.isfinite(p_out).all(dim=2).all(dim=1)

    weights = torch.nan_to_num(torch.where(passes & finite, det, 0.0))

    return weights.mean().item(), weights.std(correction=0).item() / math.sqrt(n)


def _configs(n_out):
    cfgs = []

    cfgs.append(
        Config(
            topology="propagator (external)",
            mode=PSM.propagator,
            color_order=None,
            piped=False,
        )
    )

    cfgs.append(
        Config(
            topology="propagator (piped)",
            mode=PSM.propagator,
            color_order=None,
            piped=True,
        )
    )

    for topo_name, order in _color_orders(n_out):
        cfgs.append(
            Config(
                topology=f"color_ordered (external) | {topo_name}",
                mode=PSM.color_ordered,
                color_order=order,
                piped=False,
            )
        )

        cfgs.append(
            Config(
                topology=f"color_ordered (piped) | {topo_name}",
                mode=PSM.color_ordered,
                color_order=order,
                piped=True,
            )
        )

    return cfgs


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
    pt_min = _pt_min(pids_out, fraction)
    seed0 = BASE_SEED + 1000 * vi + 100 * fi

    ref, ref_err = _cut_volume(
        masses_out, pids_out, PSM.rambo, None, False, pt_min, seed0
    )

    if ref <= 0.0 or not math.isfinite(ref):
        pytest.skip(f"{label}: cut volume vanishes at pt_min={pt_min:.0f} GeV.")

    rows = []

    for ci, cfg in enumerate(_configs(len(masses_out)), start=1):
        val, err = _cut_volume(
            masses_out,
            pids_out,
            cfg.mode,
            cfg.color_order,
            cfg.piped,
            pt_min,
            seed0 + ci,
        )

        sigma = math.sqrt(err**2 + ref_err**2)
        pull = abs(val - ref) / sigma if sigma > 0 else math.inf

        rows.append((cfg.topology, val, err, pull))

    failures = [r for r in rows if r[3] > TOL_SIGMA]

    if failures:
        lines = [
            f"{label}: cut volume disagrees with rambo+external reference "
            f"(tol {TOL_SIGMA:g} sigma, pt_min={pt_min:.0f} GeV, N={N_SAMPLES:_}).",
            f"  reference rambo external: {ref:.6e} +/- {ref_err:.2e}",
        ]

        for topo, val, err, pull in rows:
            mark = "FAIL" if pull > TOL_SIGMA else "ok  "
            lines.append(
                f"  {mark} {topo:45}: {val:.6e} +/- {err:.2e} "
                f"(ratio {val/ref:5.3f}, {pull:6.1f} sigma)"
            )

        pytest.fail("\n".join(lines), pytrace=False)
