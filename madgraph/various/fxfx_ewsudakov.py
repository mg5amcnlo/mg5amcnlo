################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""
FxFx clustering and EW Sudakov reweighting for LHE events.

This module provides:
- FxFx event clustering (n+1 body -> n body Born-like events)
- EW Sudakov reweighting with scalar and density-matrix paths
- Helper functions for charge conservation and flavor identification

The main class FxFxEWSudakovMixin is designed to be mixed into ReweightInterface.
"""
from __future__ import division

import copy
import contextlib
import logging
import math
import os
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import reduce
from itertools import product
from operator import mul
from typing import List, Optional, Tuple
from xml.etree import ElementTree
from xml.sax.saxutils import escape, quoteattr

import madgraph.various.lhe_parser as lhe_parser
import numpy as np

FourMomentum = lhe_parser.FourMomentum
logger = logging.getLogger("madgraph.various.fxfx_ewsudakov")


class MissingSudakovProcessError(Exception):
    """The compiled Sudakov library does not contain a required process."""

# Event counter set by reweight_interface.py during the event loop (1-based).
CURRENT_EVENT_ID = 0

_DIM_OVERFLOW_WARNED = False

# Fixed dimensions in the generated density dispatcher/Fortran wrapper.
MAX_DENSITY_DIM = 36
MAX_DENSITY_RESONANCES = 10
MAX_DENSITY_ALLOW_HEL = 300


def limit_sudakov_factors(factors, accounting_key=None):
    """Set all factors to unity when the central factor exceeds 200 in magnitude."""
    factors = list(factors)
    if abs(factors[0]) > 200:
        if accounting_key is not None:
            _count_fks_event(accounting_key)
        return [1.0] * len(factors)
    return factors


def density_dimensions_supported(total_dim, nres):
    """Return whether a density call fits every fixed generated buffer."""
    return (
        total_dim <= MAX_DENSITY_DIM
        and nres <= MAX_DENSITY_RESONANCES
        and total_dim * nres <= MAX_DENSITY_ALLOW_HEL
    )

# =============================================================================
# FKS-mapping accounting and fallback policy
# =============================================================================
# The FSR mapping can fail to reach the pole mass (see _fks_fsr_mapping): each
# outcome is identified by a method string. Successful methods leave the mother
# on its pole mass (within 1e-4 M, or 1e-3 √q² for a massless mother); every
# other method returns an off-pole mother.
FKS_SUCCESS_METHODS = frozenset(
    [
        "fks_fsr_massive_onshell",
        "fks_fsr_no_spectators_beam_scaled",
        "fks_isr_cm_boost",
    ]
)

# Per-launch counters, reset by fks_reset_accounting and reported by
# fks_accounting_summary.
FKS_METHOD_COUNTS = Counter()  # method string -> number of clusterings
FKS_QUALITY_COUNTS = Counter()  # coarse quality events (aborts, damping, ...)

# Policy for events whose clustering history contains any FSR fallback:
#   "accept"      : keep the event, compute weights on the off-pole Born-like
#                   object (mass label is set to the actual invariant mass so
#                   downstream kinematics checks remain consistent);
#   "unit_weight" : force all Sudakov variants to 1 for such events.
# An event whose clustering fails gets unit weights under either policy.
FKS_FALLBACK_POLICY = "accept"

# Mapping counters use the first clustering pass of each main-loop event.
# Event outcome counters cover the full driver chain and every xi evaluation.
# Both exclude the catalogue prescan (CURRENT_EVENT_ID == 0) and are per worker.
_FKS_COUNT_ACTIVE = False
_FKS_COUNTED_UPTO = 0
_FKS_QUALITY_EVENT_IDS = {}


def _count_fks_event(name):
    """Count an outcome once for the current main-loop event."""
    if CURRENT_EVENT_ID > 0 and _FKS_QUALITY_EVENT_IDS.get(name) != CURRENT_EVENT_ID:
        FKS_QUALITY_COUNTS[name] += 1
        _FKS_QUALITY_EVENT_IDS[name] = CURRENT_EVENT_ID


def fks_reset_accounting():
    """Reset the per-launch counters (called at the start of each event loop)."""
    global _FKS_COUNT_ACTIVE, _FKS_COUNTED_UPTO
    FKS_METHOD_COUNTS.clear()
    FKS_QUALITY_COUNTS.clear()
    _FKS_QUALITY_EVENT_IDS.clear()
    _FKS_COUNT_ACTIVE = False
    _FKS_COUNTED_UPTO = 0


def fks_accounting_summary():
    """Multi-line summary of FKS-mapping outcomes accumulated in this launch.

    Mapping counts use the first clustering pass; outcomes count each event
    once across driver fallbacks and xi evaluations. All counts are per worker.
    """
    total = sum(FKS_METHOD_COUNTS.values())
    n_fallback = sum(
        c for m, c in FKS_METHOD_COUNTS.items() if m not in FKS_SUCCESS_METHODS
    )
    lines = [
        "FKS mapping outcomes (once per event, per worker): "
        "%d clusterings, %d fallback(s), policy=%s"
        % (total, n_fallback, FKS_FALLBACK_POLICY)
    ]
    for method, count in sorted(FKS_METHOD_COUNTS.items()):
        marker = "" if method in FKS_SUCCESS_METHODS else "  <-- fallback"
        lines.append("    %-42s %d%s" % (method, count, marker))
    for name, count in sorted(FKS_QUALITY_COUNTS.items()):
        lines.append("    %-42s %d" % (name, count))
    return "\n".join(lines)


# Pole masses in GeV: defaults, overridden from the param_card by
# _init_pole_masses_from_banner().
MW_POLE = 80.385
MZ_POLE = 91.1876
MT_POLE = 173.3
MH_POLE = 125.0
_POLE_MASSES_INITIALIZED = False


@dataclass
class ForcedClusterStep:
    """One forced clustering step reversing a MadSpin decay (e.g. W+ -> l+ nu).

    LHE indices are 1-based; scale is the children's invariant mass in GeV;
    depth is the distance to the leaves (inner decays have lower depth and
    are clustered first).
    """

    children_lhe_idx: List[int]
    mother_lhe_idx: int
    mother_pdg: int
    scale: float
    depth: int


FXFX_CLUSTER_STEP_RE = re.compile(r'step_(\d+)="([^"]+)"')
FXFX_CLUSTER_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

# Flavour families that may be relabelled to restore charge conservation.
FXFX_FLAVOR_FAMILIES = {
    "quark": set(range(1, 7)),
    "lepton": {11, 13, 15},
    "neutrino": {12, 14, 16},
}


# =============================================================================
# Clustering helpers
# =============================================================================


def _choose_keep(i_pos, j_pos):
    """Choose which position to keep when clustering two particles.

    Prioritizes initial-state particles (positions 0, 1) over final-state.
    For same-category positions, keeps the one with lower index.
    """
    if i_pos <= 1 and j_pos > 1:
        return i_pos
    if j_pos <= 1 and i_pos > 1:
        return j_pos
    return min(i_pos, j_pos)


def _flavor_family(pdg):
    """Return a flavor family identifier for PDGs we can safely relabel."""
    try:
        pid = abs(int(pdg))
    except Exception:
        return None
    for family, pdgs in FXFX_FLAVOR_FAMILIES.items():
        if pid in pdgs:
            return family
    return None


def _get_charge_from_pdg(model, pdg):
    """Return electric charge for a PDG, handling antiparticles when needed."""
    if model is None:
        return 0.0
    try:
        part_dict = model.get("particle_dict") or {}
    except Exception:
        return 0.0
    abs_pdg = abs(int(pdg))
    part = part_dict.get(abs_pdg)
    if part is None:
        return 0.0
    try:
        charge = part.get("charge")
    except Exception:
        charge = 0.0
    if charge is None:
        charge = 0.0
    try:
        self_antipart = part.get("self_antipart")
    except Exception:
        self_antipart = False
    if pdg < 0 and not self_antipart:
        return -charge
    return charge


def _event_charges_after_merge(model, buff_event, keep_pos, drop_pos, mother_pdg):
    """
    Compute (Q_initial, Q_final) after clustering the legs at keep_pos and drop_pos
    into a single mother with PDG = mother_pdg (simulation only, no mutation).
    """
    q_init = 0.0
    q_final = 0.0
    for idx, p in enumerate(buff_event):
        if idx == drop_pos:
            continue

        status = int(getattr(p, "status", 0))
        pdg = int(mother_pdg) if idx == keep_pos else int(getattr(p, "pid"))
        q = _get_charge_from_pdg(model, pdg)

        if status == -1:
            q_init += q
        else:
            q_final += q

    return q_init, q_final


def _get_mother_lhe_idx(mother):
    """1-based LHE index of a mother (Particle: event_id + 1, number: int, None: 0)."""
    if mother is None:
        return 0
    if hasattr(mother, "event_id"):
        return mother.event_id + 1
    elif isinstance(mother, (int, float)):
        return int(mother)
    return 0


def _fix_mother_sign_global(model, buff_event, pos_i, pos_j, mother_pdg, tol=1e-6):
    """Fix the mother PDG sign/flavour by global charge conservation.

    Tries mother_pdg, then -mother_pdg, then the other flavours of its family
    (the LHE coin flip can pick a different flavour than the FxFx clustering).
    Returns (pdg, changed); raises RuntimeError if no candidate conserves charge.
    """
    if mother_pdg is None:
        return mother_pdg, False

    keep_pos = _choose_keep(pos_i, pos_j)
    drop_pos = pos_j if keep_pos == pos_i else pos_i

    q_init_after, q_final_after = _event_charges_after_merge(
        model, buff_event, keep_pos, drop_pos, mother_pdg
    )

    if abs(q_init_after - q_final_after) < tol:
        return mother_pdg, False

    flipped = -int(mother_pdg)
    q_init_flip, q_final_flip = _event_charges_after_merge(
        model, buff_event, keep_pos, drop_pos, flipped
    )

    if abs(q_init_flip - q_final_flip) < tol:
        return flipped, True

    original_family = _flavor_family(mother_pdg)
    if original_family:
        family_pdgs = list(FXFX_FLAVOR_FAMILIES.get(original_family, set()))
        candidates = []
        for pdg in family_pdgs:
            candidates.extend([pdg, -pdg])

        for cand_pdg in candidates:
            if cand_pdg == mother_pdg or cand_pdg == flipped:
                continue
            q_init_cand, q_final_cand = _event_charges_after_merge(
                model, buff_event, keep_pos, drop_pos, cand_pdg
            )
            if abs(q_init_cand - q_final_cand) < tol:
                return cand_pdg, True

    msg = (
        "Global charge conservation FAILED for clustering of positions "
        f"{pos_i+1} and {pos_j+1} with mother PDG {mother_pdg}. "
        f"With mother={mother_pdg}: Q_init={q_init_after:.6f}, Q_final={q_final_after:.6f}; "
        f"with mother={flipped}: Q_init={q_init_flip:.6f}, Q_final={q_final_flip:.6f}"
    )
    raise RuntimeError(msg)


def suggest_mothers_from_interactions(model, perts, child_i, child_j, emitter_state):
    """Suggest possible mother legs from UFO 3-point interactions."""
    if model is None:
        return []

    try:
        part_dict = model.get("particle_dict") or {}
    except Exception:
        part_dict = {}

    if not part_dict:
        return []

    def _to_incoming_pdg(pdg, is_final):
        part = part_dict.get(pdg)
        if part is None:
            return None
        if is_final:
            return pdg
        return part.get_anti_pdg_code()

    pi_int = _to_incoming_pdg(child_i["pdg"], child_i["state"])
    pj_int = _to_incoming_pdg(child_j["pdg"], child_j["state"])
    if pi_int is None or pj_int is None:
        return []

    # Look up particles, falling back to abs(pdg) for models that don't have
    # explicit antiparticle entries in particle_dict
    part_i = part_dict.get(pi_int)
    if part_i is None and pi_int is not None:
        part_i = part_dict.get(abs(pi_int))
    part_j = part_dict.get(pj_int)
    if part_j is None and pj_int is not None:
        part_j = part_dict.get(abs(pj_int))
    if part_i is None or part_j is None:
        return []

    try:
        interactions = model.get("interactions") or []
    except Exception:
        interactions = []

    candidates = []
    for inter in interactions:
        try:
            participants = inter.get("particles") or []
        except Exception:
            continue
        if len(participants) != 3:
            continue

        if perts:
            try:
                orders = inter.get("orders") or {}
            except Exception:
                orders = {}
            if not any(orders.get(p, 0) > 0 for p in perts):
                continue

        tmp = list(participants)
        try:
            tmp.remove(part_i)
        except ValueError:
            continue
        try:
            tmp.remove(part_j)
        except ValueError:
            continue
        if len(tmp) != 1:
            continue

        remaining = tmp[0]
        if emitter_state:
            mother_pdg = remaining.get_anti_pdg_code()
        else:
            mother_pdg = remaining.get_pdg_code()

        candidates.append({"mother_pdg": mother_pdg})

    return candidates


# =============================================================================
# FxFx clustering kinematics (FKS mappings)
# =============================================================================


def _m2(p):
    """Invariant mass squared."""
    return p.E**2 - p.px**2 - p.py**2 - p.pz**2


def _get_mother_mass(mother_pid):
    """Pole mass of a mother for t, Z, W, H; 0 for every other PDG."""
    abs_pid = abs(mother_pid)
    if abs_pid == 6:
        return MT_POLE
    elif abs_pid == 23:
        return MZ_POLE
    elif abs_pid == 24:
        return MW_POLE
    elif abs_pid == 25:
        return MH_POLE
    else:
        return 0.0


def _boost_along_direction(p, beta, direction):
    """Boost p with velocity beta along the unit vector n̂ = direction.

    This is the boost B of FKS Sec. 5.2 (Eqs. 5.25-5.27):
        E' = γ(E - β p·n̂),   p'_∥ = γ(p·n̂ - β E),
    i.e. to a frame moving with +β along n̂. For a momentum |p| n̂ and
    β = |p|/E, +β goes to its rest frame and -β back from it.
    """
    if abs(beta) < 1e-12:
        return FourMomentum(p)

    nx, ny, nz = direction
    gamma = 1.0 / math.sqrt(1.0 - beta**2)

    p_dot_n = p.px * nx + p.py * ny + p.pz * nz

    E_new = gamma * (p.E - beta * p_dot_n)
    p_parallel_new = gamma * (p_dot_n - beta * p.E)

    p_perp_x = p.px - p_dot_n * nx
    p_perp_y = p.py - p_dot_n * ny
    p_perp_z = p.pz - p_dot_n * nz

    return FourMomentum(
        [
            E_new,
            p_parallel_new * nx + p_perp_x,
            p_parallel_new * ny + p_perp_y,
            p_parallel_new * nz + p_perp_z,
        ]
    )


def _apply_fxfx_frame_transform(momentum, frame_transform):
    """Apply ordered clustering and event-preparation Lorentz operations.

    Event.boost uses the HELAS boost convention: it negates the spatial part
    of the total incoming momentum before calling FourMomentum.boost.  Mirror
    that implementation exactly so decay products can follow the same Lorentz
    map as the clustered production event.
    """
    transformed = FourMomentum(momentum)
    for operation, reference in frame_transform or []:
        if operation == "boost_to_cm":
            pboost = FourMomentum(reference)
            pboost.px *= -1.0
            pboost.py *= -1.0
            pboost.pz *= -1.0
            transformed = transformed.boost(pboost)
        elif operation == "rotate_to_z":
            transformed = transformed.rotate_to_z(prot=FourMomentum(reference))
        elif operation == "pt_boost":
            transformed = transformed.pt_boost(pboost=FourMomentum(reference))
        elif operation == "boost_along_direction":
            beta, direction = reference
            transformed = _boost_along_direction(transformed, beta, direction)
        elif operation == "boost_from_rest":
            transformed = transformed.boost(FourMomentum(reference))
        elif operation == "rescale":
            # Rescaling: on a lightlike momentum, the boost along its own direction.
            transformed = FourMomentum(
                [
                    reference * transformed.E,
                    reference * transformed.px,
                    reference * transformed.py,
                    reference * transformed.pz,
                ]
            )
        else:
            raise ValueError("Unknown FxFx frame operation: %s" % operation)
    return transformed


def _transform_decay_products_to_onshell(
    decay_momenta, original_res_p4, onshell_res_p4, frame_transform=None,
):
    """Project decay products onto the resonance momentum onshell_res_p4.

    original_res_p4 is the sum of decay_momenta. frame_transform, if given,
    is the ordered list of Lorentz operations recorded while preparing the
    clustered production event; it is applied to the decay system first. The
    products are then boosted to the rest frame of original_res_p4, rebuilt
    with the mass of onshell_res_p4 keeping their rest-frame directions, and
    boosted so that they sum to onshell_res_p4. If M <= m1 + m2 (two-body),
    or the source has neither mass nor momentum, the transported inputs are
    returned unprojected. Returns momenta in input order.
    """
    if not decay_momenta:
        return []

    # Express the input decay system and the target parent in the same frame.
    if frame_transform:
        decay_momenta = [
            _apply_fxfx_frame_transform(p, frame_transform) for p in decay_momenta
        ]
        original_res_p4 = _apply_fxfx_frame_transform(
            original_res_p4, frame_transform
        )

    n_particles = len(decay_momenta)

    orig_m2 = _m2(original_res_p4)
    orig_m = math.sqrt(max(0, orig_m2))
    orig_p = math.sqrt(original_res_p4.px**2 + original_res_p4.py**2 + original_res_p4.pz**2)

    onshell_m2 = _m2(onshell_res_p4)
    onshell_m = math.sqrt(max(0, onshell_m2))
    onshell_p = math.sqrt(onshell_res_p4.px**2 + onshell_res_p4.py**2 + onshell_res_p4.pz**2)

    # Vanishing source momentum and mass: nothing to project.
    if orig_p < 1e-10 and orig_m < 1e-10:
        return [FourMomentum(p) for p in decay_momenta]

    if orig_p > 1e-10:
        orig_dir = (
            original_res_p4.px / orig_p,
            original_res_p4.py / orig_p,
            original_res_p4.pz / orig_p,
        )
        orig_beta = orig_p / original_res_p4.E
    else:
        orig_dir = (0, 0, 1)
        orig_beta = 0

    # +β = |p|/E along p̂: to the source rest frame.
    rest_momenta = []
    for p in decay_momenta:
        p_rest = _boost_along_direction(p, orig_beta, orig_dir)
        rest_momenta.append(p_rest)

    rest_directions = []
    rest_masses = []
    for i, p_rest in enumerate(rest_momenta):
        p_mag = math.sqrt(p_rest.px**2 + p_rest.py**2 + p_rest.pz**2)
        m2 = _m2(p_rest)
        m = math.sqrt(max(0, m2))
        rest_masses.append(m)
        if p_mag > 1e-10:
            direction = (p_rest.px / p_mag, p_rest.py / p_mag, p_rest.pz / p_mag)
        else:
            direction = (0, 0, 1)
        rest_directions.append(direction)

    if len(decay_momenta) == 2:
        # Two-body: keep m1, m2 and the direction of product 1; back to back
        # with |p| = λ^{1/2}(M², m1², m2²) / (2M) in the on-shell rest frame.
        m1, m2 = rest_masses[0], rest_masses[1]
        if onshell_m > m1 + m2:
            lambda_val = (onshell_m2 - (m1 + m2) ** 2) * (onshell_m2 - (m1 - m2) ** 2)
            p_new = math.sqrt(max(0, lambda_val)) / (2 * onshell_m)
        else:
            # Below threshold M <= m1 + m2: return the (transported) inputs.
            return [FourMomentum(p) for p in decay_momenta]

        E1_new = math.sqrt(p_new**2 + m1**2)
        E2_new = math.sqrt(p_new**2 + m2**2)

        new_rest_momenta = [
            FourMomentum(
                [
                    E1_new,
                    p_new * rest_directions[0][0],
                    p_new * rest_directions[0][1],
                    p_new * rest_directions[0][2],
                ]
            ),
            FourMomentum(
                [
                    E2_new,
                    -p_new * rest_directions[0][0],
                    -p_new * rest_directions[0][1],
                    -p_new * rest_directions[0][2],
                ]
            ),
        ]
    else:
        # Multi-body: rescale all components uniformly so that sum E = M; this
        # keeps the rest-frame directions but not the product masses.
        sum_E_rest = sum(p.E for p in rest_momenta)
        scale = onshell_m / sum_E_rest if sum_E_rest > 0 else 1.0
        new_rest_momenta = []
        for p_rest in rest_momenta:
            p_new = FourMomentum(
                [p_rest.E * scale, p_rest.px * scale, p_rest.py * scale, p_rest.pz * scale]
            )
            new_rest_momenta.append(p_new)

    if onshell_p > 1e-10:
        onshell_dir = (
            onshell_res_p4.px / onshell_p,
            onshell_res_p4.py / onshell_p,
            onshell_res_p4.pz / onshell_p,
        )
        onshell_beta = onshell_p / onshell_res_p4.E
    else:
        onshell_dir = (0, 0, 1)
        onshell_beta = 0

    # -β: from the on-shell rest frame to the frame of onshell_res_p4.
    lab_momenta = []
    for p_rest in new_rest_momenta:
        p_lab = _boost_along_direction(p_rest, -onshell_beta, onshell_dir)
        lab_momenta.append(p_lab)

    return lab_momenta


def _transport_decay_products(products, frame_transform):
    """Transport leaf momenta while retaining their original LHE labels."""
    return {
        label: _apply_fxfx_frame_transform(momentum, frame_transform)
        for label, momentum in products.items()
    }


def _project_cluster_decay_products(children, momenta, mother):
    """Project two children onto a parent and transport each child's leaves.

    A timelike parent's two-body projection preserves child masses and
    rest-frame directions. Lightlike jets carry collinear constituents with
    the input energy fractions.
    Each composite child carries its leaves through the same canonical boosts,
    preserving the invariant masses of its internal decay systems.
    """
    source_parent = momenta[0] + momenta[1]
    if _m2(mother) <= 1e-12 * mother.E**2:
        leaves = dict(children[0])
        leaves.update(children[1])
        total_energy = sum(p.E for p in leaves.values())
        return {
            label: FourMomentum([component * p.E / total_energy
                                 for component in (mother.E, mother.px, mother.py, mother.pz)])
            for label, p in leaves.items()
        }
    projected = _transform_decay_products_to_onshell(
        momenta, source_parent, mother,
    )
    products = {}
    for leaves, source, target in zip(children, momenta, projected):
        if len(leaves) == 1:
            products[next(iter(leaves))] = FourMomentum(target)
        elif leaves and _m2(source) <= 1e-12 * source.E**2:
            products.update({
                label: FourMomentum([component * p.E / source.E
                                     for component in (target.E, target.px, target.py, target.pz)])
                for label, p in leaves.items()
            })
        elif leaves:
            products.update(_transport_decay_products(leaves, [
                ("boost_to_cm", source), ("boost_from_rest", target),
            ]))
    return products


def _fks_isr_mapping(fks_i, fks_j, moth, orig_momenta):
    """FKS initial-state mapping (FKS paper Sec. 5.1).

    The radiation k = orig_momenta[fks_j] is removed from the initial state,
    q̄ = q - k, √ŝ' = √(q̄²); the Born-like beams are massless along z with
    E = √ŝ'/2 in the q̄ rest frame. This changes x± and leaves the final state
    unchanged up to the transverse boost that puts q̄ along z.
    orig_momenta is [beam0, beam1, final state...]; fks_i and moth are unused.

    Returns {'beams': [b0, b1], 'final_state': {idx: p} without the
    radiation, 'sqrt_s_new', 'method', 'final_state_transform': the ordered
    Lorentz operations applied to the surviving final state}.
    """
    p_initial = orig_momenta[0] + orig_momenta[1]
    p_radiation = orig_momenta[fks_j]

    p_clustered = FourMomentum(
        [
            p_initial.E - p_radiation.E,
            p_initial.px - p_radiation.px,
            p_initial.py - p_radiation.py,
            p_initial.pz - p_radiation.pz,
        ]
    )

    s_hat = _m2(p_clustered)
    sqrt_s = math.sqrt(max(0, s_hat))

    # A nearly lightlike q̄ has no usable rest frame: below √(q̄²) = 0.1 GeV,
    # γ > E/0.1 GeV. The caller stops clustering on this "light-like" error.
    if sqrt_s < 0.1:
        raise ValueError(f"ISR mapping: light-like p_clustered (√s={sqrt_s:.4f} GeV), cannot boost")

    result = {
        "final_state": {},
        "sqrt_s_new": sqrt_s,
        "method": "fks_isr_cm_boost",
        "final_state_transform": [],
    }

    p_clustered_pt = math.sqrt(p_clustered.px**2 + p_clustered.py**2)
    needs_pt_boost = p_clustered_pt > 1e-10

    # The pT boost is not undone: the output frame has q̄ along z, as a
    # Born-level LHE event needs (beams along z). The Sudakov computation
    # boosts to the CM frame and is unaffected.
    if needs_pt_boost:
        result["final_state_transform"] = [("pt_boost", FourMomentum(p_clustered))]
        after_pt = [p.pt_boost(pboost=p_clustered) for p in orig_momenta]
        p_clustered_after_pt = p_clustered.pt_boost(pboost=p_clustered)
    else:
        after_pt = list(orig_momenta)
        p_clustered_after_pt = FourMomentum(p_clustered)

    cm_momenta = [p.zboost(pboost=p_clustered_after_pt) for p in after_pt]

    # Beams in the q̄ rest frame, keeping the orientation of beam 0.
    E_beam_cm = sqrt_s / 2.0
    if cm_momenta[0].pz >= 0:
        beam0_new = FourMomentum([E_beam_cm, 0, 0, E_beam_cm])
        beam1_new = FourMomentum([E_beam_cm, 0, 0, -E_beam_cm])
    else:
        beam0_new = FourMomentum([E_beam_cm, 0, 0, -E_beam_cm])
        beam1_new = FourMomentum([E_beam_cm, 0, 0, E_beam_cm])

    # Longitudinal boost back to the frame of q̄ (after the pT boost).
    for ip, p_cm in enumerate(cm_momenta):
        if ip <= 1:
            continue
        if ip == fks_j:
            continue
        p_lab = p_cm.zboost_inv(pboost=p_clustered_after_pt)
        result["final_state"][ip] = p_lab

    result["beams"] = [
        beam0_new.zboost_inv(pboost=p_clustered_after_pt),
        beam1_new.zboost_inv(pboost=p_clustered_after_pt),
    ]

    return result


def _fks_fsr_mapping(fks_i, fks_j, moth, orig_momenta):
    """FKS final-state mapping (FKS paper Sec. 5.2) with an on-shell mother.

    The spectators (recoil system k_rec) are boosted along n̂ = k⃗_rec/|k⃗_rec|
    so that the mother q - B k_rec satisfies (q - B k_rec)² = M² instead of
    the FKS lightlike condition (q - B k_rec)² = 0. With q_n = q⃗·n̂,
    p_n = |k⃗_rec|,
        A = q_E E_rec - q_n p_n
        B = q_n E_rec - q_E p_n
        R = (q² + m²_rec - M²)/2 + q⃗_perp·k⃗_perp    (k⃗_perp ≡ 0 for the
                                                       total recoil system)
    the on-shell condition reads γ(A + β B) = R; squaring gives
        (B² + R²) β² + 2AB β + (A² - R²) = 0,
        β = (-AB ± R √(R² + B² - A²)) / (B² + R²).

    Spurious roots: A ± B = (q_E ± q_n)(E_rec ∓ p_n) ≥ 0 for timelike q and
    physical k_rec, hence A ≥ |B| and γ(A + βB) > 0 for all |β| < 1. For R > 0
    every root with |β| < 1 satisfies the unsquared (+R) equation. For R ≤ 0
    the target is unreachable (q·k̄ ≥ 0 for future-pointing momenta), yet the
    squared quadratic still has |β| < 1 roots, all on the spurious (-R)
    branch: R ≤ 0 is rejected ("fks_fsr_unreachable_target"), and the chosen
    root is checked against the unsquared constraint ("fks_fsr_spurious_root";
    reached in floating point for soft recoil systems, E_rec up to about 1 GeV
    at TeV-scale q²).

    A recoil system is lightlike when |m²_rec| < 1e-6 GeV² and
    |m²_rec| ≤ 1e-6 E²_rec. The boost along n̂ then acts on it as the rescaling
    k̄ = r k_rec with r = √((1-β)/(1+β)), applied directly (frame operation
    "rescale"); r solves (q - r k_rec)² = M² for the input momenta.

    A mapping is reported as on-shell only if the mother is within 1e-4 M of a
    massive pole (ten times inside the 0.1% mass gate of
    lhe_parser.Event.check_kinematics_only), or within 1e-3 √q² of zero mass;
    otherwise the mother is the naive pair momentum (method "fks_fsr_inexact").

    fks_i becomes the mother (PDG moth[0]['id']) and fks_j is removed;
    orig_momenta is [beam0, beam1, final state...]. Returns {'mother',
    'beams' (changed only without spectators), 'spectators': {idx: p},
    'spectator_transform': the ordered frame operations applied to the
    spectators, 'method'}.
    """
    mother_pid = abs(moth[0]["id"])
    mother_mass = _get_mother_mass(mother_pid)
    M2 = mother_mass**2

    q = orig_momenta[0] + orig_momenta[1]
    q2 = _m2(q)

    spectator_indices = [
        idx for idx in range(len(orig_momenta)) if idx >= 2 and idx != fks_i and idx != fks_j
    ]

    result = {
        "beams": [FourMomentum(orig_momenta[0]), FourMomentum(orig_momenta[1])],
        "spectators": {},
        "spectator_transform": [],
    }

    if len(spectator_indices) == 0:
        # No spectators: the mother is q. Rescaling both beams by M/√(q²)
        # puts it on the pole and keeps its rapidity (E1/E2 fixed).
        actual_m = math.sqrt(max(0, q2))

        # Clustering stops before a 2 -> 1 configuration with a massless
        # mother (γ, g).
        if mother_mass < 1e-6:
            raise ValueError(
                f"FSR no-spectators: massless mother (M={mother_mass:.2e}) should have been "
                "prevented by clustering stop logic. This is a bug."
            )

        if actual_m > 1e-10:
            scale = mother_mass / actual_m
        else:
            scale = 1.0

        beam0_new = FourMomentum(
            [
                orig_momenta[0].E * scale,
                orig_momenta[0].px * scale,
                orig_momenta[0].py * scale,
                orig_momenta[0].pz * scale,
            ]
        )
        beam1_new = FourMomentum(
            [
                orig_momenta[1].E * scale,
                orig_momenta[1].px * scale,
                orig_momenta[1].py * scale,
                orig_momenta[1].pz * scale,
            ]
        )
        result["beams"] = [beam0_new, beam1_new]

        q_new = beam0_new + beam1_new
        result["mother"] = q_new
        result["method"] = "fks_fsr_no_spectators_beam_scaled"

        return result

    k_rec = FourMomentum()
    for idx in spectator_indices:
        k_rec += orig_momenta[idx]

    k_rec_E = k_rec.E
    k_rec_p = math.sqrt(k_rec.px**2 + k_rec.py**2 + k_rec.pz**2)
    k_rec_m2 = _m2(k_rec)

    if k_rec_p < 1e-10:
        # |k⃗_rec| = 0 defines no boost direction: the mother is q - k_rec.
        result["mother"] = FourMomentum(
            [
                q.E - k_rec.E,
                q.px - k_rec.px,
                q.py - k_rec.py,
                q.pz - k_rec.pz,
            ]
        )
        for idx in spectator_indices:
            result["spectators"][idx] = FourMomentum(orig_momenta[idx])
        result["method"] = "fks_fsr_zero_krec_momentum"
        return result

    # A naive pair mass below M is not rejected: a boost with βB < 0 (β > 0 in
    # the q rest frame, where B = -√q² |k⃗_rec|) lowers q·k̄ and raises the
    # mother mass; the quadratic decides whether such a |β| < 1 exists.
    n_rec = (k_rec.px / k_rec_p, k_rec.py / k_rec_p, k_rec.pz / k_rec_p)

    # Lightlike test: m²_rec negligible both absolutely and relative to E²_rec
    # (a soft wide-angle system is massive).
    rescale = None
    m2_negligible = abs(k_rec_m2) < 1e-6 and abs(k_rec_m2) <= 1e-6 * k_rec_E**2
    if m2_negligible:
        # For lightlike k̄ ∥ n̂: q·k̄ = k̄_E (q_E - q⃗·n̂) = (q² - M²)/2 fixes
        # k̄_E, which must be positive.
        q_dot_n = q.px * n_rec[0] + q.py * n_rec[1] + q.pz * n_rec[2]
        denom_qn = q.E - q_dot_n

        if abs(denom_qn) < 1e-10:
            result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
            for idx in spectator_indices:
                result["spectators"][idx] = FourMomentum(orig_momenta[idx])
            result["method"] = "fks_fsr_degenerate"
            return result

        k_bar_E_target = (q2 - M2) / (2.0 * denom_qn)

        if k_bar_E_target < 0:
            result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
            for idx in spectator_indices:
                result["spectators"][idx] = FourMomentum(orig_momenta[idx])
            result["method"] = "fks_fsr_negative_energy"
            return result

        # The boost of a lightlike k along n̂ is k̄ = r k with
        # r = (1 - β) / √(1 - β²) = √((1-β)/(1+β)), i.e. β = (1 - r²) / (1 + r²).
        # r solves (q - r k_rec)² = M² for the input momenta, whose m²_rec is
        # not exactly zero: m²_rec r² - 2 (q·k_rec) r + (q² - M²) = 0, on the
        # root continuous with the lightlike limit r = (q² - M²) / (2 q·k_rec).
        q_dot_k = q.E * k_rec.E - q.px * k_rec.px - q.py * k_rec.py - q.pz * k_rec.pz
        r = (q2 - M2) / (
            q_dot_k + math.sqrt(max(0.0, q_dot_k**2 - k_rec_m2 * (q2 - M2)))
        )

        if r <= 0:
            result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
            for idx in spectator_indices:
                result["spectators"][idx] = FourMomentum(orig_momenta[idx])
            result["method"] = "fks_fsr_negative_r"
            return result

        r2 = r * r
        beta = (1.0 - r2) / (1.0 + r2)

        # Applied as the rescaling k̄ = r k, which keeps the mother on its pole
        # for r far from 1 (soft recoil), where 1 - β² cancels.
        rescale = r

    else:
        # Massive recoil system: solve γ(A + βB) = R for β (see docstring).
        q_dot_n = q.px * n_rec[0] + q.py * n_rec[1] + q.pz * n_rec[2]
        p_n = k_rec.px * n_rec[0] + k_rec.py * n_rec[1] + k_rec.pz * n_rec[2]

        k_perp_x = k_rec.px - p_n * n_rec[0]
        k_perp_y = k_rec.py - p_n * n_rec[1]
        k_perp_z = k_rec.pz - p_n * n_rec[2]
        q_dot_k_perp = q.px * k_perp_x + q.py * k_perp_y + q.pz * k_perp_z

        target = (q2 + k_rec_m2 - M2) / 2.0

        A = q.E * k_rec_E - q_dot_n * p_n
        B = q_dot_n * k_rec_E - q.E * p_n
        rhs = target + q_dot_k_perp

        # rhs = R ≤ 0 (q² + m²_rec ≤ M²: too light for an on-shell mother plus
        # this recoil system) is unreachable; the squared quadratic would still
        # return roots of γ(A + βB) = -R (see docstring).
        if rhs <= 0:
            result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
            for idx in spectator_indices:
                result["spectators"][idx] = FourMomentum(orig_momenta[idx])
            result["method"] = "fks_fsr_unreachable_target"
            return result

        if abs(B) < 1e-8 * max(abs(A), abs(rhs), 1.0):
            # B ≈ 0: the quadratic degenerates; γ A = R gives β² = 1 - A²/R².
            if abs(rhs) < 1e-10:
                result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
                for idx in spectator_indices:
                    result["spectators"][idx] = FourMomentum(orig_momenta[idx])
                result["method"] = "fks_fsr_degenerate_collinear"
                return result
            gamma_sq = (rhs / A) ** 2 if abs(A) > 1e-10 else float("inf")
            if gamma_sq < 1.0:
                result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
                for idx in spectator_indices:
                    result["spectators"][idx] = FourMomentum(orig_momenta[idx])
                result["method"] = "fks_fsr_collinear_unphysical"
                return result
            beta_sq = 1.0 - 1.0 / gamma_sq
            # The sign of β does not enter γ A = R; β ≥ 0 is taken.
            beta = math.sqrt(beta_sq) if beta_sq > 0 else 0.0
        else:
            coef_a = B * B + rhs * rhs
            coef_b = 2 * A * B
            coef_c = A * A - rhs * rhs

            disc = coef_b * coef_b - 4 * coef_a * coef_c
            if disc < 0:
                result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
                for idx in spectator_indices:
                    result["spectators"][idx] = FourMomentum(orig_momenta[idx])
                result["method"] = "fks_fsr_no_solution"
                return result

            sqrt_disc = math.sqrt(disc)
            beta1 = (-coef_b + sqrt_disc) / (2 * coef_a)
            beta2 = (-coef_b - sqrt_disc) / (2 * coef_a)

            # The smallest |β| among the roots with |β| < 1.
            valid_betas = [b for b in [beta1, beta2] if abs(b) < 1.0]
            if not valid_betas:
                result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
                for idx in spectator_indices:
                    result["spectators"][idx] = FourMomentum(orig_momenta[idx])
                result["method"] = "fks_fsr_superluminal"
                return result

            beta = min(valid_betas, key=abs)

            # The root must solve the unsquared constraint γ(A + βB) = R.
            _gamma_chk = 1.0 / math.sqrt(1.0 - beta**2)
            if abs(_gamma_chk * (A + beta * B) - rhs) > 1e-6 * max(1.0, abs(rhs)):
                result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
                for idx in spectator_indices:
                    result["spectators"][idx] = FourMomentum(orig_momenta[idx])
                result["method"] = "fks_fsr_spurious_root"
                return result

    # A boost needs |β| < 1; the lightlike rescaling needs only r > 0.
    if rescale is None and abs(beta) >= 1.0:
        result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
        for idx in spectator_indices:
            result["spectators"][idx] = FourMomentum(orig_momenta[idx])
        result["method"] = "fks_fsr_superluminal"
        return result

    if rescale is None:
        result["spectator_transform"] = [("boost_along_direction", (beta, n_rec))]
        k_rec_boosted = _boost_along_direction(k_rec, beta, n_rec)

        for idx in spectator_indices:
            p_orig = orig_momenta[idx]
            k_bar_i = _boost_along_direction(p_orig, beta, n_rec)
            result["spectators"][idx] = k_bar_i
    else:
        result["spectator_transform"] = [("rescale", rescale)]
        k_rec_boosted = _apply_fxfx_frame_transform(
            k_rec, result["spectator_transform"]
        )

        for idx in spectator_indices:
            result["spectators"][idx] = _apply_fxfx_frame_transform(
                orig_momenta[idx], result["spectator_transform"]
            )

    mother = FourMomentum(
        [
            q.E - k_rec_boosted.E,
            q.px - k_rec_boosted.px,
            q.py - k_rec_boosted.py,
            q.pz - k_rec_boosted.pz,
        ]
    )

    if mother_mass > 0.0:
        off_pole = abs(math.sqrt(max(0.0, _m2(mother))) - mother_mass)
        off_shell = off_pole > 1e-4 * mother_mass
    else:
        off_shell = abs(_m2(mother)) > 1e-6 * q2
    if off_shell:
        result["mother"] = orig_momenta[fks_i] + orig_momenta[fks_j]
        for idx in spectator_indices:
            result["spectators"][idx] = FourMomentum(orig_momenta[idx])
        result["spectator_transform"] = []
        result["method"] = "fks_fsr_inexact"
        return result

    result["mother"] = mother
    result["method"] = "fks_fsr_massive_onshell"

    return result


def _fks_fsr_mapping_cm(fks_i, fks_j, moth, orig_momenta):
    """_fks_fsr_mapping applied in the partonic CM frame and boosted back.

    The FSR map boosts the recoil system along its own direction, so it depends
    on the frame; it is defined with q at rest (FKS Sec. 5.2). The beams lie
    along z, so the CM boost is along z. Returns the mapping in the input frame,
    with 'spectator_transform' = to_cm + map + from_cm, together with 'to_cm',
    'from_cm', 'momenta_cm' and 'mother_cm' for projecting decay products in
    the CM frame.
    """
    q = orig_momenta[0] + orig_momenta[1]
    beta_cm = q.pz / q.E if q.E > 1e-10 else 0.0
    z_hat = (0, 0, 1)
    momenta_cm = [_boost_along_direction(p, beta_cm, z_hat) for p in orig_momenta]
    result = _fks_fsr_mapping(fks_i, fks_j, moth, momenta_cm)
    to_cm = [("boost_along_direction", (beta_cm, z_hat))]
    from_cm = [("boost_along_direction", (-beta_cm, z_hat))]
    return {
        "mother": _boost_along_direction(result["mother"], -beta_cm, z_hat),
        "beams": [_boost_along_direction(p, -beta_cm, z_hat) for p in result["beams"]],
        "spectators": {
            idx: _boost_along_direction(p, -beta_cm, z_hat)
            for idx, p in result["spectators"].items()
        },
        "spectator_transform": to_cm + result["spectator_transform"] + from_cm,
        "method": result["method"],
        "to_cm": to_cm,
        "from_cm": from_cm,
        "momenta_cm": momenta_cm,
        "mother_cm": result["mother"],
    }


def fxfx_merge_particles_kinematics(event, i, j, moth):
    """Cluster particles i and j of event in place with an FKS mapping.

    ISR (FKS Sec. 5.1) if the surviving particle is a beam, otherwise FSR
    (Sec. 5.2: spectators boosted along k⃗_rec, mother = q - B k_rec, with β
    fixed by the on-shell condition instead of the lightlike one; see
    _fks_fsr_mapping). moth[0] = {'number': 1-based index of the surviving
    particle, 'id': mother PDG}. Decay products attached as
    event._fxfx_decay_products follow the same Lorentz maps. Returns the FKS
    method string (FKS_SUCCESS_METHODS lists the successful outcomes: on-pole
    FSR and ISR).
    """
    if i == moth[0].get("number") - 1:
        fks_i = i
        fks_j = j
    elif j == moth[0].get("number") - 1:
        fks_i = j
        fks_j = i
    else:
        fks_i = min(i, j)
        fks_j = max(i, j)

    to_remove = fks_j

    orig_momenta = [FourMomentum(p) for p in event]
    decay_products = getattr(event, "_fxfx_decay_products", None)

    if fks_i <= 1:
        isr_result = _fks_isr_mapping(fks_i, fks_j, moth, orig_momenta)
        method = isr_result.get("method", "fks_isr_unknown")

        if decay_products is not None:
            for idx in isr_result["final_state"]:
                decay_products[idx] = _transport_decay_products(
                    decay_products[idx], isr_result["final_state_transform"],
                )

        event[0].set_momentum(isr_result["beams"][0])
        event[1].set_momentum(isr_result["beams"][1])

        for idx, p in isr_result["final_state"].items():
            event[idx].set_momentum(p)

        event[fks_i].pid = moth[0]["id"]

    else:
        fks_result = _fks_fsr_mapping_cm(fks_i, fks_j, moth, orig_momenta)
        method = fks_result.get("method", "fks_fsr_unknown")

        if decay_products is not None:
            children = [
                _transport_decay_products(decay_products[idx], fks_result["to_cm"])
                for idx in (fks_i, fks_j)
            ]
            projected = _project_cluster_decay_products(
                children,
                [fks_result["momenta_cm"][fks_i], fks_result["momenta_cm"][fks_j]],
                fks_result["mother_cm"],
            )
            decay_products[fks_i] = _transport_decay_products(
                projected, fks_result["from_cm"]
            )
            for idx in fks_result["spectators"]:
                decay_products[idx] = _transport_decay_products(
                    decay_products[idx], fks_result["spectator_transform"],
                )

        event[0].set_momentum(fks_result["beams"][0])
        event[1].set_momentum(fks_result["beams"][1])
        event[fks_i].set_momentum(fks_result["mother"])
        event[fks_i].pid = moth[0]["id"]

        # The mass label is the pole mass only if the mapping reached the pole,
        # else the actual invariant mass: Event.check_kinematics_only (called
        # from _prepare_fxfx_sudakov_inputs) raises on a >0.1% mismatch
        # between the label and the four-momentum.
        if method in FKS_SUCCESS_METHODS:
            event[fks_i].mass = _get_mother_mass(abs(moth[0]["id"]))
        else:
            event[fks_i].mass = math.sqrt(max(0.0, _m2(fks_result["mother"])))

        for idx, p in fks_result["spectators"].items():
            event[idx].set_momentum(p)

    event.pop(to_remove)
    if decay_products is not None:
        decay_products.pop(to_remove)

    # Counted on the first clustering pass of each event only. Every mapping,
    # fallbacks included, conserves momentum, so a conservation violation
    # signals an error, not a fallback.
    if _FKS_COUNT_ACTIVE:
        FKS_METHOD_COUNTS[method] += 1
        p_in_sum = FourMomentum(event[0]) + FourMomentum(event[1])
        p_out_sum = FourMomentum()
        for idx in range(2, len(event)):
            p_out_sum += FourMomentum(event[idx])
        cons_scale = max(abs(p_in_sum.E), 1.0)
        if (
            max(
                abs(p_in_sum.E - p_out_sum.E),
                abs(p_in_sum.px - p_out_sum.px),
                abs(p_in_sum.py - p_out_sum.py),
                abs(p_in_sum.pz - p_out_sum.pz),
            )
            > 1e-6 * cons_scale
        ):
            FKS_QUALITY_COUNTS["merge_momentum_conservation_violation"] += 1

    return method


# =============================================================================
# Density matrices
# =============================================================================

# Helicity-space dimension per resonance, keyed by abs(PDG).
SPIN_DIMS = {
    6: 2,
    23: 3,
    24: 3,
}

# Resonances of the density-matrix path (particle and antiparticle codes).
RESONANCE_PDGS = frozenset([6, -6, 23, 24, -24])


def _total_dimension(dims: List[int]) -> int:
    """Compute total Hilbert space dimension as product of individual dims."""
    return reduce(mul, dims, 1)


def _fortran_to_python_hel_permutation(res_dims):
    """Build permutation to reorder Fortran delta indices to Python B indices.

    The Fortran density_matrix_wrapper enumerates helicity combinations using
    little-endian mixed-radix (first resonance varies fastest) with ascending
    helicity values (0->-1, 1->+1 for fermions; 0->-1, 1->0, 2->+1 for vectors).

    The Python _build_allowed_helicities uses itertools.product (big-endian,
    first resonance varies slowest) with descending values ([+1,-1] for fermions;
    [+1,0,-1] for vectors).

    Returns perm such that delta_python[i] = delta_fortran[perm[i]],
    i.e. perm[i] is the Fortran index corresponding to Python index i.
    """
    total_dim = 1
    for d in res_dims:
        total_dim *= d

    perm = [0] * total_dim
    for ih_fortran in range(total_dim):
        k = ih_fortran
        hels = []
        for d in res_dims:
            idx = k % d
            k //= d
            if d == 2:
                hels.append(2 * idx - 1)
            elif d == 3:
                hels.append(idx - 1)
            else:
                hels.append(idx - (d - 1) // 2)

        ih_python = 0
        for n, d in enumerate(res_dims):
            h = hels[n]
            if d == 2:
                local_idx = 0 if h == +1 else 1
            elif d == 3:
                local_idx = {+1: 0, 0: 1, -1: 2}[h]
            else:
                local_idx = (d - 1) // 2 - h
            ih_python = ih_python * d + local_idx

        perm[ih_python] = ih_fortran

    return perm


def unpack_density_triangle(packed, len_tri=None):
    """Full density matrix from the packed triangle returned by GET_DENSITY.

    GET_ALL_INTER stores the upper triangle row by row,
    packed[i*n + k - i*(i+1)//2] = B[i,k] = M_i * conj(M_k) for i <= k
    (summed over colours and over the helicities not in the matrix).
    The upper triangle (diagonal included) is kept as stored and the lower
    triangle is filled with B[k,i] = conj(B[i,k]), giving B[h,h'] = M_h M*_h'
    as DensityMatrix.apply_sudakov and DensityMatrix.contract use it.
    Density_functions.DensityMatrixObservables.square_matrix returns the
    transpose of this matrix.

    Only the first len_tri entries are used (all if None). Returns a complex
    (n, n) array; raises ValueError unless they form a 1D line of n(n+1)/2
    entries with n >= 1.
    """
    line = np.asarray(packed, dtype=np.complex128)
    if line.ndim == 1 and len_tri is not None:
        line = line[:len_tri]
    n = (int(math.sqrt(8 * line.size + 1)) - 1) // 2
    if line.ndim != 1 or n < 1 or n * (n + 1) // 2 != line.size:
        raise ValueError(
            f"Packed density triangle of shape {line.shape} is not a 1D line of n(n+1)/2 entries"
        )
    matrix = np.zeros((n, n), dtype=np.complex128)
    matrix[np.triu_indices(n)] = line
    lower = np.tril_indices(n, -1)
    matrix[lower] = np.conj(matrix.T[lower])
    return matrix


# =============================================================================
# DensityHelper: density matrices through the DensityInterface machinery
# =============================================================================


class DensityHelper:
    """Density-matrix evaluation without a full DensityInterface.

    Copies the needed attributes from a ReweightInterface, uses the
    density-specific id_to_path and f2pylib (standalone exports), and provides
    the methods that DensityInterface.calculate_matrix_element needs:

        helper = DensityHelper(reweight_interface, density_prod_id_to_path, density_prod_f2pylib)
        matrix = helper.compute_density_matrix(event, target_pdgs=[6, -6], res_dims=[2, 2])
    """

    def __init__(self, reweight_interface, id_to_path, f2pylib):
        """id_to_path, f2pylib: the density-specific maps (e.g. density_prod_*)."""
        self.keep_ordering = getattr(reweight_interface, "keep_ordering", True)
        self.helicity_reweighting = getattr(reweight_interface, "helicity_reweighting", False)
        self.options = getattr(
            reweight_interface, "options", {"identical_particle_in_prod_and_decay": "average"}
        )
        self.me_dir = getattr(reweight_interface, "me_dir", os.getcwd())

        self.id_to_path = id_to_path
        self.f2pylib = f2pylib

        # No boost, rotation or initial-state symmetrisation for EW Sudakov.
        self.momenta_boost = ([0], "", [])
        self.helicity_direction = ([0], "", [])
        self.axis_referential = [0]
        self.symmetrise_initial_state = False

        # Set per call in compute_density_matrix.
        self.particle_in_density_matrix = None
        self.allowed_helicities = None
        self.number_changing_helicities = None
        self.number_combinations = None
        self.spins = None
        self.flag_particle_in_density_matrix = True
        self._initialized_modules = set()

    @staticmethod
    def invert_momenta(p):
        """(n, 4) momenta of get_momenta() -> Fortran-ordered (4, n) array.

        This is the F2PY layout of P(0:3, NEXTERNAL).
        """
        p_arr = np.asarray(p, dtype=np.float64)

        return np.asfortranarray(p_arr.T, dtype=np.float64)

    def method_boost_event(self, event, all_p, orig_order, hypp_id, boost_corrected):
        """No boost for EW Sudakov: returns all_p unchanged."""
        if boost_corrected == [-1]:
            return all_p
        return all_p

    def calculate_angles_rotation(self, position_particles, all_p, module):
        """No rotation for EW Sudakov: zero angles."""
        if position_particles == [-1]:
            return [0] * len(all_p), [0] * len(all_p)
        return [0] * len(all_p), [0] * len(all_p)

    def rotation_density(self, module, all_p, phi, theta):
        """No rotation for EW Sudakov: momentum sets converted to the Fortran layout."""
        for i in range(len(all_p)):
            all_p[i] = self.invert_momenta(all_p[i])
        return all_p

    def compute_density_matrix(self, event, target_pdgs, res_dims):
        """Density matrix (production B or decay C) of target_pdgs.

        res_dims are the spin dimensions (2 fermion, 3 vector).

        Returns None if no density module or subprocess matches the event.
        """

        self._configure_for_pdgs(target_pdgs, res_dims)

        tag, order = event.get_tag_and_order()
        if self.keep_ordering:
            # Unsorted IS and FS ordering, as in the id_to_path keys.
            tag = (tuple(order[0]), tuple(order[1]))

        if tag not in self.id_to_path:
            return None

        orig_order, Pdir, hel_dict = self.id_to_path[tag]
        base = os.path.basename(os.path.dirname(Pdir))
        moduletag = (base, 2)

        if moduletag not in self.f2pylib:
            return None

        module = self.f2pylib[moduletag]

        if self.keep_ordering:
            all_p = [event.get_momenta(orig_order)]
        else:
            all_p = event.get_all_momenta(orig_order)

        boost_corrected = [-1]
        all_p = self.method_boost_event(event, all_p, orig_order, 0, boost_corrected)

        refChoice_corrected = [-1]
        phi, theta = self.calculate_angles_rotation(refChoice_corrected, all_p, module)
        all_p = self.rotation_density(module, all_p, phi, theta)

        pos_corrected = self._find_particle_positions(orig_order, target_pdgs)

        # The module serves several subprocesses: take the prefix whose PDG
        # list matches this process (up to ordering within IS and FS).
        all_pdgs_raw, all_pids = module.get_pdg_order()
        all_pdgs = [[pdg for pdg in pdgs if pdg != 0] for pdgs in all_pdgs_raw]
        PREFIX = module.get_prefix()
        all_prefixes = [bytes(j).decode(errors="ignore").strip().lower() for j in PREFIX]

        # nincoming = 1 for a decay, 2 for production.
        nincoming = len(tag[0])
        lookup_incoming = sorted(orig_order[0])
        lookup_outgoing = sorted(orig_order[1])
        lookup_tag = (tuple(lookup_incoming), tuple(lookup_outgoing))

        proc_idx = None
        for i, pdg in enumerate(all_pdgs):
            incoming = pdg[:nincoming]
            outgoing = pdg[nincoming:]
            incoming_sorted = sorted(incoming)
            outgoing_sorted = sorted(outgoing)
            cmp_tag = (tuple(incoming_sorted), tuple(outgoing_sorted))
            if cmp_tag == lookup_tag:
                proc_idx = i
                break

        if proc_idx is None:
            return None

        prefix = all_prefixes[proc_idx]

        # Load the param card once per module: Pdir is <me_dir>/SubProcesses,
        # the card <me_dir>/Cards/param_card.dat (me_dir: rw_me_density_prod or
        # rw_me_density_decay).
        module_id = id(module)
        if module_id not in self._initialized_modules:
            density_me_dir = os.path.dirname(Pdir)
            Card_dir = os.path.join(density_me_dir, "Cards", "param_card.dat")
            if os.path.exists(Card_dir):
                Initialise = getattr(module, "initialise", None)
                if Initialise:
                    Initialise(Card_dir)
                InitModel = getattr(module, prefix + "initialisemodel", None)
                if InitModel:
                    InitModel(Card_dir)
            self._initialized_modules.add(module_id)

        # GET_DENSITY via F2PY, in one of two conventions (tried in this order):
        #   <prefix>get_density(p, pos, n_changing, allow_hel, n_comb, alphas),
        #     per subprocess, exported by density modules such as ttHj;
        #   py_get_density(pdgs, procid, p, pos, allow_hel, alphas, scale2), the
        #     PDG dispatcher, the only GET_DENSITY entry point of the MG5
        #     standalone f2py wrapper (f2py_wrapper_all.inc). F77_DENSITY selects
        #     the subprocess by PDGS when procid <= 0. In this MG5 3.8.1 f2py
        #     signature n_comb, n_changing and npdg are hidden and inferred
        #     from len(allow_hel)/len(pos), len(pos), len(pdgs).
        func_name = prefix + "get_density"
        get_density_func = getattr(module, func_name, None)
        use_density_dispatcher = False
        if get_density_func is None:
            get_density_func = getattr(module, "py_get_density", None)
            use_density_dispatcher = True
        if get_density_func is None:
            return None

        # Dispatcher only: PDGS in the momentum order (orig_order = (incoming,
        # outgoing)) and procid = -1 (select by PDGS), as in the standard
        # reweight call smatrixhel(pdg, pid=-1, ...).
        density_pdgs = np.asarray(
            list(orig_order[0]) + list(orig_order[1]), dtype=np.int32
        )
        density_procid = -1

        me_value = None

        pos_arr = np.asarray(pos_corrected, dtype=np.int32)
        hel_arr = np.asarray(self.allowed_helicities, dtype=np.int32)

        for i in range(len(all_p)):
            # all_p[i] is already in the Fortran layout (rotation_density).
            pinv = np.asfortranarray(all_p[i], dtype=np.float64)
            if use_density_dispatcher:
                # The inferred n_comb = len(allow_hel)/len(pos) is number_combinations.
                # scale2 is the event scale squared, as in reweight_interface's own
                # density call (a dummy argument for LO matrix elements).
                production_matrix = get_density_func(
                    density_pdgs,
                    density_procid,
                    pinv,
                    pos_arr,
                    hel_arr,
                    event.aqcd,
                    event.scale ** 2,
                )
            else:
                production_matrix = get_density_func(
                    pinv,
                    pos_arr,
                    self.number_changing_helicities,
                    hel_arr,
                    self.number_combinations,
                    event.aqcd,
                )

            len_tri = self.number_combinations * (self.number_combinations + 1) // 2
            me_value = unpack_density_triangle(production_matrix, len_tri)

        return np.array(me_value) if me_value is not None else None

    def _configure_for_pdgs(self, pdg_codes, res_dims):
        """Configure density attributes for given PDGs."""
        self.particle_in_density_matrix = (list(pdg_codes), "", [])
        self.number_changing_helicities = len(pdg_codes)
        self.spins = list(res_dims)

        self.number_combinations = 1
        for spin in self.spins:
            self.number_combinations *= spin

        self.allowed_helicities = self._build_allowed_helicities(res_dims)

    def _build_allowed_helicities(self, res_dims):
        """Flat int32 array of all helicity combinations, in itertools.product order.

        Values are [+1, -1] for fermions and [+1, 0, -1] for vectors.
        """
        hel_lists = []
        for dim in res_dims:
            if dim == 2:
                hel_lists.append([+1, -1])
            elif dim == 3:
                hel_lists.append([+1, 0, -1])
            else:
                hel_lists.append(list(range(-(dim - 1) // 2, (dim - 1) // 2 + 1)))

        combos = list(product(*hel_lists))
        flat = []
        for combo in combos:
            flat.extend(combo)

        return np.asarray(flat, dtype=np.int32)

    def _find_particle_positions(self, orig_order, target_pdgs):
        """1-based (Fortran) positions of target_pdgs as an int32 array ([-1] if none).

        The initial state is searched first (the resonance of a decay), then
        the final state (the resonances of a production process).
        """
        positions = []
        n_initial = len(orig_order[0])

        for pid in target_pdgs:
            found = False
            for idx, initial_pid in enumerate(orig_order[0]):
                if initial_pid == pid:
                    pos = idx + 1
                    if pos not in positions:
                        positions.append(pos)
                        found = True
                        break
            if found:
                continue
            for idx, final_pid in enumerate(orig_order[1]):
                if final_pid == pid:
                    pos = n_initial + idx + 1
                    if pos not in positions:
                        positions.append(pos)
                        break

        return (
            np.asarray(positions, dtype=np.int32) if positions else np.array([-1], dtype=np.int32)
        )


class DensityMatrix:
    """Spin density matrix ρ[λ₁…λN; λ₁'…λN'] of a system of resonances.

    matrix is a complex (D, D) array, D = ∏ᵢ dᵢ, with dims[i] the spin
    dimension of pdgs[i] (e.g. [6, -6] for tt̄, [24, -24] for W+W-).
    """

    def __init__(self, resonance_pdgs: List[int]):
        self.pdgs = list(resonance_pdgs)
        self.dims = [SPIN_DIMS[abs(pdg)] for pdg in self.pdgs]
        self.total_dim = _total_dimension(self.dims)
        self.matrix = np.zeros((self.total_dim, self.total_dim), dtype=complex)

    @classmethod
    def from_full_matrix(cls, matrix: np.ndarray, resonance_pdgs: List[int]) -> "DensityMatrix":
        """DensityMatrix from a full D×D matrix (ValueError on a shape mismatch)."""
        dm = cls(resonance_pdgs)
        if matrix.shape != (dm.total_dim, dm.total_dim):
            raise ValueError(
                f"Matrix shape {matrix.shape} doesn't match "
                f"expected ({dm.total_dim}, {dm.total_dim})"
            )
        dm.matrix = matrix.astype(complex)
        return dm

    def copy(self) -> "DensityMatrix":
        """Create a deep copy of this density matrix."""
        dm = DensityMatrix(self.pdgs)
        dm.matrix = self.matrix.copy()
        return dm

    def apply_sudakov(self, delta: np.ndarray) -> "DensityMatrix":
        """Linearised (NLO) EW Sudakov correction of B[h,h'] = M_h M*_h'.

        M_h → M_h (1 + Δ_h) gives B[h,h'] (1 + Δ_h)(1 + Δ*_h'); dropping
        O(α²), B^EWSL[h,h'] = B[h,h'] (1 + Δ_h + Δ*_h'). delta has shape
        (total_dim,); returns a new DensityMatrix.
        """
        if len(delta) != self.total_dim:
            raise ValueError(
                f"Delta length {len(delta)} doesn't match "
                f"density matrix dimension {self.total_dim}"
            )

        result = self.copy()

        for h in range(self.total_dim):
            for hp in range(self.total_dim):
                factor = 1.0 + delta[h] + np.conj(delta[hp])
                result.matrix[h, hp] = self.matrix[h, hp] * factor

        return result

    def trace(self) -> float:
        """Real part of the trace."""
        return np.trace(self.matrix).real

    def contract(self, other: "DensityMatrix") -> float:
        """Real part of <ρ₁, ρ₂> = Σ_{h,h'} ρ₁[h,h'] ρ₂[h,h'] = Tr[ρ₁ · ρ₂ᵀ].

        Both matrices are stored as X[h,h'] = A_h A*_h': the production matrix
        from amplitudes with the resonances outgoing, the decay matrix from
        amplitudes with them incoming, in the same helicity ordering.  The
        coherent production-decay rate |Σ_h M_h D_h|² is then
        Σ_{h,h'} B[h,h'] C[h,h'].
        """
        if self.total_dim != other.total_dim:
            raise ValueError(f"Dimension mismatch: {self.total_dim} vs {other.total_dim}")

        return np.sum(self.matrix * other.matrix).real

    def __repr__(self) -> str:
        return (
            f"DensityMatrix(pdgs={self.pdgs}, dims={self.dims}, "
            f"total_dim={self.total_dim}, trace={self.trace():.6f})"
        )

    def __str__(self) -> str:
        return f"DensityMatrix({self.total_dim}×{self.total_dim}) for {self.pdgs}"


def _tensor_product(dm1: DensityMatrix, dm2: DensityMatrix) -> DensityMatrix:
    """Tensor product ρ₁ ⊗ ρ₂ (np.kron, pdgs concatenated)."""
    combined_pdgs = dm1.pdgs + dm2.pdgs
    result = DensityMatrix(combined_pdgs)
    result.matrix = np.kron(dm1.matrix, dm2.matrix)
    return result


def build_decay_tensor_product(decay_matrices: List[DensityMatrix]) -> DensityMatrix:
    """Decay density matrix C = C₁ ⊗ C₂ ⊗ ... ⊗ Cₙ in list order."""
    if not decay_matrices:
        raise ValueError("Need at least one decay matrix")

    result = decay_matrices[0].copy()
    for dm in decay_matrices[1:]:
        result = _tensor_product(result, dm)

    return result


class ResonanceIdentifier:
    """Resonances and their decay products in LHE events."""

    @staticmethod
    def find_resonances(event) -> List[Tuple[int, int]]:
        """(0-based index, pdg) of each resonance with status 1 or 2."""
        resonances = []
        for i, particle in enumerate(event):
            pdg = getattr(particle, "pdg", None) or getattr(particle, "pid", None)
            status = getattr(particle, "status", 1)

            if pdg is None:
                continue

            if abs(pdg) in RESONANCE_PDGS and status in (1, 2):
                resonances.append((i, int(pdg)))

        return resonances

    @staticmethod
    def _mother_matches(mother, resonance_idx: int) -> bool:
        """Whether a mother pointer is the particle at 0-based resonance_idx.

        The pointer is either a Particle (0-based event_id) or a raw 1-based
        LHE number.
        """
        if mother is None:
            return False
        if hasattr(mother, "event_id"):
            return mother.event_id == resonance_idx
        elif isinstance(mother, (int, float)):
            return int(mother) == resonance_idx + 1
        return False

    @staticmethod
    def _identify_decay_products(event, resonance_idx: int) -> List[int]:
        """Indices of the particles whose mother1 or mother2 is resonance_idx."""
        decay_products = []

        for i, particle in enumerate(event):
            m1 = getattr(particle, "mother1", None)
            m2 = getattr(particle, "mother2", None)

            if ResonanceIdentifier._mother_matches(
                m1, resonance_idx
            ) or ResonanceIdentifier._mother_matches(m2, resonance_idx):
                decay_products.append(i)

        return decay_products

    @staticmethod
    def find_decayed_resonances(event) -> List[Tuple[int, int, List[int]]]:
        """(resonance_idx, pdg, [decay product indices]) for each decayed resonance.

        Daughters are found through the mother pointers, which covers decays
        in the hard matrix element and MadSpin decays alike.
        """
        results = []

        for i, particle in enumerate(event):
            pdg = getattr(particle, "pdg", None) or getattr(particle, "pid", None)

            if pdg is None:
                continue

            if abs(pdg) in RESONANCE_PDGS:
                decay_products = ResonanceIdentifier._identify_decay_products(event, i)

                if decay_products:
                    results.append((i, int(pdg), decay_products))

        return results


def compute_density_weight(
    B: DensityMatrix, C: Optional[DensityMatrix], delta: np.ndarray
) -> float:
    """EW Sudakov weight w = <B^EWSL, C> / <B, C> with density matrices.

    <X, C> = Σ_{h,h'} X[h,h'] C[h,h'] (DensityMatrix.contract) and
    B^EWSL[h,h'] = B[h,h'] (1 + Δ_h + Δ*_h') is the NLO-corrected Born-like
    production matrix. C = None is the spin-summed limit (C = identity:
    traces). Returns 1 if <B, C> vanishes.
    """
    B_ewsl = B.apply_sudakov(delta)

    if C is not None:
        numerator = B_ewsl.contract(C)
        denominator = B.contract(C)
    else:
        numerator = B_ewsl.trace()
        denominator = B.trace()

    if abs(denominator) < 1e-15:
        return 1.0

    return numerator / denominator


class DensitySudakovLogger:
    """Counter of the events processed by the density-matrix Sudakov reweighting."""

    def __init__(self):
        self.events_processed = 0

    def log_event(
        self,
        mode: str,
        resonances: List[Tuple[int, int]],
        dimension: int,
        weight: float,
        clustering_info: Optional[dict] = None,
    ):
        """Count one processed event."""
        self.events_processed += 1


# =============================================================================
# FxFxEWSudakovMixin
# =============================================================================


class FxFxEWSudakovMixin:
    """FxFx clustering and EW Sudakov reweighting methods for ReweightInterface.

    Uses model, banner, lhe_input, me_dir, keep_ordering, options and the
    density attributes (density_prod_id_to_path, ...) of self.
    """

    def _init_pole_masses_from_banner(self):
        """Set the module pole masses from the banner's param_card.

        Runs once per process; a default stays if the banner or its MASS
        entry is unavailable.
        """
        global MW_POLE, MZ_POLE, MT_POLE, MH_POLE, _POLE_MASSES_INITIALIZED
        if _POLE_MASSES_INITIALIZED:
            return
        _POLE_MASSES_INITIALIZED = True

        banner = getattr(self, 'banner', None)
        if banner is None:
            return
        mass_map = {24: 'MW_POLE', 23: 'MZ_POLE', 6: 'MT_POLE', 25: 'MH_POLE'}
        for pdg, name in mass_map.items():
            try:
                val = float(banner.get_detail('param_card', 'mass', pdg).value)
                globals()[name] = val
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # MadSpin forced clustering (applied before the FxFx clustering)
    # -------------------------------------------------------------------------

    def _build_madspin_decay_tree(self, event):
        """Decay tree of the status-2 resonances from the mother1/mother2 pointers.

        Returns {lhe_idx (1-based): {'pdg', 'status', 'children': [lhe_idx, ...],
        'depth', 'mother'}}; depth is the maximum distance to a leaf (0 for a
        leaf).
        """
        tree = {}

        for i, particle in enumerate(event):
            lhe_idx = i + 1
            pdg = int(getattr(particle, "pid", 0))
            status = int(getattr(particle, "status", 0))
            tree[lhe_idx] = {
                "pdg": pdg,
                "status": status,
                "children": [],
                "depth": 0,
                "mother": None,
            }

        for i, particle in enumerate(event):
            lhe_idx = i + 1
            m1 = _get_mother_lhe_idx(getattr(particle, "mother1", None))
            m2 = _get_mother_lhe_idx(getattr(particle, "mother2", None))

            if m1 > 0 and m1 in tree and tree[m1]["status"] == 2:
                tree[m1]["children"].append(lhe_idx)
                tree[lhe_idx]["mother"] = m1
            elif m2 > 0 and m2 in tree and tree[m2]["status"] == 2:
                if m2 != m1:
                    tree[m2]["children"].append(lhe_idx)
                    tree[lhe_idx]["mother"] = m2

        def compute_depth(lhe_idx):
            node = tree[lhe_idx]
            if not node["children"]:
                return 0
            max_child_depth = 0
            for child_idx in node["children"]:
                child_depth = compute_depth(child_idx)
                max_child_depth = max(max_child_depth, child_depth)
            node["depth"] = max_child_depth + 1
            return node["depth"]

        for lhe_idx in tree:
            if tree[lhe_idx]["status"] == 2 and tree[lhe_idx]["children"]:
                compute_depth(lhe_idx)

        return tree

    def _generate_forced_cluster_steps(self, event, decay_tree):
        """ForcedClusterSteps of the decay tree, inner decays first.

        Steps are sorted by depth, so nested decays cluster W -> l nu
        (depth 1) before t -> b W (depth 2). The scale is the children's
        invariant mass.
        """
        steps = []

        resonances = [
            (lhe_idx, node)
            for lhe_idx, node in decay_tree.items()
            if node["status"] == 2 and node["children"]
        ]

        resonances.sort(key=lambda x: x[1]["depth"])

        for lhe_idx, node in resonances:
            children = node["children"]
            sum_E = sum_px = sum_py = sum_pz = 0.0

            for child_idx in children:
                p = event[child_idx - 1]
                sum_E += float(p.E)
                sum_px += float(p.px)
                sum_py += float(p.py)
                sum_pz += float(p.pz)

            m2 = sum_E**2 - sum_px**2 - sum_py**2 - sum_pz**2
            scale = math.sqrt(max(0, m2))

            step = ForcedClusterStep(
                children_lhe_idx=children,
                mother_lhe_idx=lhe_idx,
                mother_pdg=node["pdg"],
                scale=scale,
                depth=node["depth"],
            )
            steps.append(step)

        return steps

    def _apply_forced_clustering(self, event, steps, track_decays=False):
        """Reverse the MadSpin decays with the FKS FSR mapping (mother on its pole).

        Each 1 -> 2 step replaces the two children by their status-2 mother,
        which becomes final state; the other final-state particles recoil.
        track_decays carries the decay leaf momenta through every mapping.
        Returns (event, groups): the event with the hard process restored,
        with the per-step methods in event.fks_methods, and per position the
        set of original LHE indices it contains.
        """
        buff_event = copy.deepcopy(event)
        decay_products = None
        if track_decays:
            decay_products = [
                {i + 1: FourMomentum(p)} if p.status == 1 else {}
                for i, p in enumerate(event)
            ]
            buff_event._fxfx_decay_products = decay_products

        # _cluster_fxfx_event_internal seeds its method history from these.
        forced_methods = []

        # groups[pos]: original LHE indices at each position (every particle,
        # intermediates included).
        groups = []
        pos_to_lhe = {}
        lhe_to_pos = {}

        for i, p in enumerate(buff_event):
            lhe_idx = i + 1
            pos_to_lhe[i] = lhe_idx
            lhe_to_pos[lhe_idx] = i
            groups.append(set([lhe_idx]))

        for step_num, step in enumerate(steps):

            # Only 1 -> 2 decays are reversed. A skipped mother stays status 2
            # (stripped later), so the hard process is incomplete: the skip is
            # recorded for the fallback policy and the counters.
            if len(step.children_lhe_idx) != 2:
                forced_methods.append("forced_step_skipped")
                if _FKS_COUNT_ACTIVE:
                    FKS_METHOD_COUNTS["forced_step_skipped"] += 1
                continue

            child1_lhe, child2_lhe = step.children_lhe_idx
            mother_lhe = step.mother_lhe_idx

            if child1_lhe not in lhe_to_pos or child2_lhe not in lhe_to_pos:
                forced_methods.append("forced_step_skipped")
                if _FKS_COUNT_ACTIVE:
                    FKS_METHOD_COUNTS["forced_step_skipped"] += 1
                continue
            if mother_lhe not in lhe_to_pos:
                forced_methods.append("forced_step_skipped")
                if _FKS_COUNT_ACTIVE:
                    FKS_METHOD_COUNTS["forced_step_skipped"] += 1
                continue

            child1_pos = lhe_to_pos[child1_lhe]
            child2_pos = lhe_to_pos[child2_lhe]
            mother_pos = lhe_to_pos[mother_lhe]

            # The FKS FSR mapping puts the mother on its pole, the recoil going
            # to the other final-state particles (or the beams). The mother is
            # still a status-2 intermediate, so only the beams and the status
            # ±1 particles ("active") enter momentum conservation.
            active_positions = []
            for pos in range(len(buff_event)):
                p = buff_event[pos]
                status = getattr(p, "status", 1)
                if pos <= 1:
                    active_positions.append(pos)
                elif abs(status) == 1:
                    active_positions.append(pos)

            active_to_buff = {i: pos for i, pos in enumerate(active_positions)}
            buff_to_active = {pos: i for i, pos in enumerate(active_positions)}

            orig_momenta = [FourMomentum(buff_event[pos]) for pos in active_positions]

            if child1_pos not in buff_to_active or child2_pos not in buff_to_active:
                # A child is not active (an intermediate whose own decay was
                # not reversed): the mother is the off-shell sum of the children.
                p1 = buff_event[child1_pos]
                p2 = buff_event[child2_pos]
                mother_E = float(p1.E) + float(p2.E)
                mother_px = float(p1.px) + float(p2.px)
                mother_py = float(p1.py) + float(p2.py)
                mother_pz = float(p1.pz) + float(p2.pz)
                mother_p = buff_event[mother_pos]
                mother_p.E = mother_E
                mother_p.px = mother_px
                mother_p.py = mother_py
                mother_p.pz = mother_pz
                mother_p.status = 1
                # The mass label must match the momentum (check_kinematics_only).
                mother_p.mass = math.sqrt(
                    max(0.0, mother_E**2 - mother_px**2 - mother_py**2 - mother_pz**2)
                )
                if _FKS_COUNT_ACTIVE:
                    FKS_METHOD_COUNTS["forced_children_not_active"] += 1
                forced_methods.append("forced_children_not_active")
                if decay_products is not None:
                    decay_products[mother_pos] = dict(decay_products[child1_pos])
                    decay_products[mother_pos].update(decay_products[child2_pos])
            else:
                child1_active = buff_to_active[child1_pos]
                child2_active = buff_to_active[child2_pos]

                p1 = buff_event[child1_pos]
                p2 = buff_event[child2_pos]
                mother_E = float(p1.E) + float(p2.E)
                mother_px = float(p1.px) + float(p2.px)
                mother_py = float(p1.py) + float(p2.py)
                mother_pz = float(p1.pz) + float(p2.pz)
                off_shell_m2 = mother_E**2 - mother_px**2 - mother_py**2 - mother_pz**2
                off_shell_m = math.sqrt(max(0, off_shell_m2))

                # Pole mass, or the pair mass for a mother without one.
                target_mass = _get_mother_mass(abs(step.mother_pdg)) or off_shell_m

                moth = [{"number": child1_active + 1, "id": step.mother_pdg}]

                # Mapped in the partonic CM frame, as every FSR clustering step.
                fks_result = _fks_fsr_mapping_cm(child1_active, child2_active, moth, orig_momenta)

                if decay_products is not None:
                    children = [
                        _transport_decay_products(decay_products[pos], fks_result["to_cm"])
                        for pos in (child1_pos, child2_pos)
                    ]
                    projected = _project_cluster_decay_products(
                        children,
                        [
                            fks_result["momenta_cm"][child1_active],
                            fks_result["momenta_cm"][child2_active],
                        ],
                        fks_result["mother_cm"],
                    )
                    decay_products[mother_pos] = _transport_decay_products(
                        projected, fks_result["from_cm"]
                    )
                    for active_idx in fks_result["spectators"]:
                        pos = active_to_buff[active_idx]
                        decay_products[pos] = _transport_decay_products(
                            decay_products[pos], fks_result["spectator_transform"],
                        )

                onshell_mom_lab = fks_result["mother"]
                beam0_lab = fks_result["beams"][0]
                beam1_lab = fks_result["beams"][1]
                spectators_lab = fks_result["spectators"]

                mother_p = buff_event[mother_pos]
                mother_p.E = onshell_mom_lab.E
                mother_p.px = onshell_mom_lab.px
                mother_p.py = onshell_mom_lab.py
                mother_p.pz = onshell_mom_lab.pz
                mother_p.status = 1

                onshell_m2 = _m2(onshell_mom_lab)
                onshell_m = math.sqrt(max(0, onshell_m2))

                # Pole-mass label only for an on-pole mapping, else the actual
                # mass (see fxfx_merge_particles_kinematics).
                fks_method = fks_result.get("method", "")
                if _FKS_COUNT_ACTIVE:
                    FKS_METHOD_COUNTS[fks_method] += 1
                forced_methods.append(fks_method)
                if fks_method in FKS_SUCCESS_METHODS:
                    mother_p.mass = target_mass
                else:
                    mother_p.mass = onshell_m

                buff_event[0].E = beam0_lab.E
                buff_event[0].px = beam0_lab.px
                buff_event[0].py = beam0_lab.py
                buff_event[0].pz = beam0_lab.pz
                buff_event[1].E = beam1_lab.E
                buff_event[1].px = beam1_lab.px
                buff_event[1].py = beam1_lab.py
                buff_event[1].pz = beam1_lab.pz

                for active_idx, boosted_p in spectators_lab.items():
                    buff_pos = active_to_buff[active_idx]
                    if buff_pos != child1_pos and buff_pos != child2_pos:
                        buff_event[buff_pos].E = boosted_p.E
                        buff_event[buff_pos].px = boosted_p.px
                        buff_event[buff_pos].py = boosted_p.py
                        buff_event[buff_pos].pz = boosted_p.pz

            mother_group = groups[mother_pos]
            child1_group = groups[child1_pos]
            child2_group = groups[child2_pos]
            mother_group.update(child1_group)
            mother_group.update(child2_group)

            children_pos = sorted([child1_pos, child2_pos], reverse=True)
            for pos in children_pos:
                buff_event.pop(pos)
                groups.pop(pos)
                if decay_products is not None:
                    decay_products.pop(pos)

            pos_to_lhe = {}
            lhe_to_pos = {}
            for i, grp in enumerate(groups):
                rep_lhe = min(grp)
                pos_to_lhe[i] = rep_lhe
                for lhe_idx in grp:
                    lhe_to_pos[lhe_idx] = i

        buff_event.fks_methods = forced_methods
        return buff_event, groups

    # -------------------------------------------------------------------------
    # FxFx clustering from the <clustering> history
    # -------------------------------------------------------------------------

    def _fxfx_parse_clustering(self, event):
        """Parse the FxFx <clustering ...> block: its attributes and 'steps'.

        'steps' is [{'n', 'idxs', 'mom'}] sorted by n; None without a block.
        """
        ev_lines = str(event).splitlines()
        for line in ev_lines:
            if "<clustering" not in line:
                continue
            attrs = dict(FXFX_CLUSTER_ATTR_RE.findall(line))
            steps = []
            for step, payload in FXFX_CLUSTER_STEP_RE.findall(line):
                parts = [p.strip() for p in payload.split(";")]
                idx_part = parts[0] if parts else ""
                mom_pdg = None
                for comp in parts[1:]:
                    if comp.startswith("mom="):
                        try:
                            mom_pdg = int(comp.split("=", 1)[1])
                        except Exception:
                            mom_pdg = None
                try:
                    step_int = int(step)
                except Exception:
                    continue
                idx_clean = idx_part.strip()
                if idx_clean.startswith("(") and ")" in idx_clean:
                    idx_clean = idx_clean[1 : idx_clean.find(")")]
                idxs = []
                for token in idx_clean.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    try:
                        idxs.append(int(token))
                    except Exception:
                        continue
                steps.append({"n": step_int, "idxs": idxs, "mom": mom_pdg})
            steps.sort(key=lambda s: s["n"])
            attrs["steps"] = steps
            return attrs
        return None

    def _cluster_fxfx_event(self, event, record_tag=False, return_groups=False):
        """Cluster an event to its Born-like hard process.

        MadSpin decays (status-2 resonances with children) are reversed first
        by forced clustering, then the FxFx <clustering> history is applied.
        Returns (clustered_event, sorted_tag), or with return_groups
        (clustered_event, sorted_tag, groups), groups[pos] being the original
        LHE indices at each position and the event carrying the decay leaf
        momenta; None on failure. record_tag adds sorted_tag to
        self.fxfx_clustered_final_states.
        """

        # Mapping counters include only the first clustering pass of each
        # main-loop event; event outcomes are counted by _count_fks_event.
        global _FKS_COUNT_ACTIVE, _FKS_COUNTED_UPTO
        if CURRENT_EVENT_ID > 0 and CURRENT_EVENT_ID > _FKS_COUNTED_UPTO:
            _FKS_COUNT_ACTIVE = True
            _FKS_COUNTED_UPTO = CURRENT_EVENT_ID
        else:
            _FKS_COUNT_ACTIVE = False

        decayed_resonances = ResonanceIdentifier.find_decayed_resonances(event)

        if decayed_resonances:

            decay_tree = self._build_madspin_decay_tree(event)

            forced_steps = self._generate_forced_cluster_steps(event, decay_tree)

            if forced_steps:
                hard_event, forced_groups = self._apply_forced_clustering(
                    event, forced_steps, track_decays=return_groups,
                )

                return self._cluster_fxfx_event_internal(
                    hard_event,
                    original_event=event,
                    initial_groups=forced_groups,
                    record_tag=record_tag,
                    return_groups=return_groups,
                )

        return self._cluster_fxfx_event_internal(
            event,
            original_event=event,
            initial_groups=None,
            record_tag=record_tag,
            return_groups=return_groups,
        )

    def _cluster_fxfx_event_internal(
        self, event, original_event=None, initial_groups=None, record_tag=False, return_groups=False
    ):
        """Apply the FxFx <clustering> steps of original_event to event.

        event may be the forced-clustering output, with initial_groups its
        position -> LHE-index groups. Returns as _cluster_fxfx_event.
        """
        if original_event is None:
            original_event = event

        # The <clustering> tag of the original event survives MadSpin.
        clustering = self._fxfx_parse_clustering(original_event)
        if not clustering:
            if initial_groups is not None:
                # No FxFx history: the forced-clustered hard process is the result.
                try:
                    sorted_tag = event.get_tag_and_order()[0]
                except Exception:
                    sorted_tag = None
                if record_tag and sorted_tag:
                    if not hasattr(self, "fxfx_clustered_final_states"):
                        self.fxfx_clustered_final_states = set()
                    self.fxfx_clustered_final_states.add(sorted_tag)
                if not hasattr(event, "fks_methods"):
                    event.fks_methods = []
                if return_groups:
                    return (event, sorted_tag, initial_groups)
                return (event, sorted_tag)
            return None

        buff_event = copy.deepcopy(event)
        to_pop = [
            ip
            for ip, part in enumerate(buff_event)
            if (abs(getattr(part, "status", 0)) != 1) and (getattr(part, "status", 0) != -1)
        ]
        for ip in reversed(to_pop):
            buff_event.pop(ip)
            if hasattr(buff_event, "_fxfx_decay_products"):
                buff_event._fxfx_decay_products.pop(ip)

        if return_groups and not hasattr(buff_event, "_fxfx_decay_products"):
            buff_event._fxfx_decay_products = [
                {i + 1: FourMomentum(p)} if p.status == 1 else {}
                for i, p in enumerate(event) if i not in to_pop
            ]

        # Method history: the forced-clustering methods, then one entry per
        # FxFx step.
        fks_methods = list(getattr(event, "fks_methods", []))

        if initial_groups is not None:
            groups = [set(g) for g in initial_groups]
            lhe_external_indices = []
            for g in groups:
                lhe_external_indices.append(min(g))
            ext_index_to_lhe = {
                i + 1: lhe_external_indices[i] for i in range(len(lhe_external_indices))
            }
            ext_index_max = len(lhe_external_indices)
        else:
            lhe_external_indices = [
                i
                for i, p in enumerate(event, start=1)
                if (abs(getattr(p, "status", 0)) == 1) or (getattr(p, "status", 0) == -1)
            ]
            # 1-based external index -> 1-based LHE index
            ext_index_to_lhe = {
                i + 1: lhe_external_indices[i] for i in range(len(lhe_external_indices))
            }
            ext_index_max = len(lhe_external_indices)
            groups = [set([lhe_external_indices[k]]) for k in range(len(buff_event))]

        def _find_position_single(i1b):
            for k, g in enumerate(groups):
                if i1b in g:
                    return k
            return None

        def _group_momentum_labels(grp):
            """LHE labels whose original momenta add up to this object's momentum.

            A forced-clustering group keeps the restored resonance's own LHE
            index next to its children (the density path uses it to identify
            the decay system), so summing every member would count the
            resonance twice (2 p_W for W -> l nu, 2 p_t + p_W for nested
            t -> b W(-> l nu)). Status-2 members are dropped whenever the group
            has a non-status-2 member, which also covers nested decays. A group
            of intermediates only (a decay that was not reversed) keeps its
            entries, and labels outside the original event are kept, so that a
            stale index fails in the momentum sum instead of being dropped.
            """
            leaves = []
            for lab in grp:
                if 1 <= lab <= len(original_event):
                    if int(getattr(original_event[lab - 1], "status", 1)) == 2:
                        continue
                leaves.append(lab)
            return leaves if leaves else list(grp)

        def _s_groups_lhe_from_pos(pos_i, pos_j):
            # LHE labels index original_event.
            Ei = pxi = pyi = pzi = 0.0
            Ej = pxj = pyj = pzj = 0.0
            for lab in _group_momentum_labels(groups[pos_i]):
                q = original_event[lab - 1]
                Ei += float(q.E)
                pxi += float(q.px)
                pyi += float(q.py)
                pzi += float(q.pz)
            for lab in _group_momentum_labels(groups[pos_j]):
                q = original_event[lab - 1]
                Ej += float(q.E)
                pxj += float(q.px)
                pyj += float(q.py)
                pzj += float(q.pz)
            E = Ei + Ej
            px = pxi + pxj
            py = pyi + pyj
            pz = pzi + pzj
            return E * E - px * px - py * py - pz * pz

        def _group_rep(pos):
            """PDG and state (False for status -1) of the current buff_event[pos]."""
            if pos < 0 or pos >= len(buff_event):
                return None
            part = buff_event[pos]
            pdg = int(getattr(part, "pid", 0))
            state = int(getattr(part, "status", 0)) != -1
            return {"pdg": pdg, "state": state}

        mW2 = MW_POLE**2
        # Clustering continues while some pair has s_ij < mW2_cluster_scale M_W²;
        # the default 1.5 reaches √s_ij ≈ 98 GeV and so covers M_Z.
        mW2_cluster_scale = getattr(self, "mW2_cluster_scale", 1.5)
        mW2_threshold = mW2_cluster_scale * mW2

        # Steps in the tag's order, which is scale ordered (soft to hard).
        original_steps = clustering.get("steps", [])
        ordered_steps = sorted(original_steps, key=lambda t: t["n"])

        stop_clustering = False
        for _step in ordered_steps:
            idxs = _step["idxs"]
            _step_mom = _step.get("mom")

            while True:
                all_positions = list(range(len(groups)))
                if len(all_positions) < 2:
                    stop_clustering = True
                    break

                all_pairs_sij = []
                for ii in range(len(all_positions)):
                    for jj in range(ii + 1, len(all_positions)):
                        pos_i = all_positions[ii]
                        pos_j = all_positions[jj]
                        sij = _s_groups_lhe_from_pos(pos_i, pos_j)
                        all_pairs_sij.append((pos_i, pos_j, sij))

                if not all_pairs_sij:
                    stop_clustering = True
                    break

                all_above = True
                for pos_i, pos_j, sij in all_pairs_sij:
                    above = sij >= mW2_threshold
                    if not above:
                        all_above = False

                if all_above:
                    stop_clustering = True
                    break

                # Tag indices count external particles only (no status-2
                # intermediates).
                idxs_ext = idxs
                valid = all((1 <= k <= ext_index_max) for k in idxs_ext)
                if not valid:
                    stop_clustering = True
                    break

                idxs_lhe = []
                for ext_idx in idxs_ext:
                    lhe_idx = ext_index_to_lhe.get(ext_idx)
                    if lhe_idx is None:
                        break
                    idxs_lhe.append(lhe_idx)
                if len(idxs_lhe) != len(idxs_ext):
                    stop_clustering = True
                    break

                parent_pos = []
                seen = set()
                for lab in idxs_lhe:
                    pos = _find_position_single(lab)
                    if pos is None:
                        continue
                    if pos not in seen:
                        parent_pos.append(pos)
                        seen.add(pos)

                if len(idxs_lhe) == 3 and len(parent_pos) != 2:
                    stop_clustering = True
                    break

                if len(parent_pos) < 2:
                    break

                if len(parent_pos) == 2:
                    ai, bj = parent_pos
                    pair = (ai, bj)
                else:
                    best = None
                    best_pair = None
                    for ii in range(len(parent_pos)):
                        for jj in range(ii + 1, len(parent_pos)):
                            sij_loc = _s_groups_lhe_from_pos(parent_pos[ii], parent_pos[jj])
                            if (best is None) or (sij_loc < best):
                                best = sij_loc
                                best_pair = (parent_pos[ii], parent_pos[jj])
                    if best_pair is None:
                        break
                    pair = best_pair

                # The selected pair is clustered even above threshold: the
                # threshold enters only through the all-pairs check above.
                a, b = sorted(pair)
                keep = _choose_keep(a, b)

                if _step_mom is not None and self.model:
                    rep_a = _group_rep(a)
                    rep_b = _group_rep(b)
                    if rep_a and rep_b:
                        pdg_a = rep_a["pdg"]
                        pdg_b = rep_b["pdg"]
                        abs_a = abs(pdg_a)
                        abs_b = abs(pdg_b)

                        # Mother flavour from the 3-point vertex (g/γ + f -> f
                        # takes the fermion flavour and the tag sign; same-flavour
                        # q + q~ -> g; other quark and lepton pairs keep the tag
                        # mother), then global charge conservation fixes the sign
                        # (or the flavour within the family).
                        is_quark_a = 1 <= abs_a <= 6
                        is_quark_b = 1 <= abs_b <= 6
                        is_lepton_a = 11 <= abs_a <= 16
                        is_lepton_b = 11 <= abs_b <= 16
                        is_fermion_a = is_quark_a or is_lepton_a
                        is_fermion_b = is_quark_b or is_lepton_b
                        is_gluon_a = abs_a == 21
                        is_gluon_b = abs_b == 21
                        is_photon_a = abs_a == 22
                        is_photon_b = abs_b == 22
                        is_gauge_boson_a = is_gluon_a or is_photon_a
                        is_gauge_boson_b = is_gluon_b or is_photon_b

                        if is_gauge_boson_a and is_fermion_b:
                            sign = 1 if _step_mom >= 0 else -1
                            _step_mom = (
                                sign * abs_b
                            )
                        elif is_fermion_a and is_gauge_boson_b:
                            sign = 1 if _step_mom >= 0 else -1
                            _step_mom = (
                                sign * abs_a
                            )
                        elif is_quark_a and is_quark_b:
                            if abs_a == abs_b and pdg_a * pdg_b < 0:
                                if abs(_step_mom) != 21:
                                    _step_mom = 21
                        elif is_lepton_a and is_lepton_b:
                            pass
                        else:
                            # Other pairs: the first UFO 3-point mother in the
                            # flavour family of the tag mother.
                            emitter_state = False if (keep <= 1) else True
                            suggestions = suggest_mothers_from_interactions(
                                self.model, ("QCD", "QED"), rep_a, rep_b, emitter_state
                            )
                            family = _flavor_family(_step_mom)
                            if family and suggestions:
                                for cand in suggestions:
                                    if _flavor_family(cand["mother_pdg"]) == family:
                                        sug_pdg = int(cand["mother_pdg"])
                                        sign = 1 if _step_mom >= 0 else -1
                                        _step_mom = sign * abs(sug_pdg)
                                        break

                    if _step_mom is not None:
                        try:
                            _step_mom, charge_changed = _fix_mother_sign_global(
                                self.model, buff_event, a, b, _step_mom
                            )
                        except RuntimeError:
                            raise

                if _step_mom is None:
                    break

                moth_id = int(_step_mom)

                # Stop before a 2 -> 1 configuration with a massless final
                # state, which has no rest frame.
                current_fs = [i for i in range(len(groups)) if i > 1]
                a_is_fs, b_is_fs = (a > 1), (b > 1)

                fs_after_pdgs = []
                if a_is_fs and b_is_fs:
                    fs_after_pdgs.append(moth_id)
                    for i in current_fs:
                        if i != a and i != b:
                            rep = _group_rep(i)
                            if rep:
                                fs_after_pdgs.append(rep["pdg"])
                else:
                    # ISR: only the spectators stay in the final state.
                    for i in current_fs:
                        if i != a and i != b:
                            rep = _group_rep(i)
                            if rep:
                                fs_after_pdgs.append(rep["pdg"])

                if len(fs_after_pdgs) == 1:
                    fs_pdg = fs_after_pdgs[0]
                    fs_mass = _get_mother_mass(abs(fs_pdg))
                    if fs_mass < 1e-6:
                        stop_clustering = True
                        break

                moth_loc = [{"number": keep + 1, "id": moth_id}]

                try:
                    step_method = fxfx_merge_particles_kinematics(buff_event, a, b, moth_loc)
                except ValueError as e:
                    if "light-like" in str(e):
                        if _FKS_COUNT_ACTIVE:
                            FKS_QUALITY_COUNTS["clustering_stopped_lightlike"] += 1
                        stop_clustering = True
                        break
                    if "math domain error" in str(e):
                        # The event is kept with unit Sudakov weights.
                        if _FKS_COUNT_ACTIVE:
                            FKS_QUALITY_COUNTS["clustering_aborted_math_domain"] += 1
                        return None
                    raise
                fks_methods.append(step_method)

                if keep == a:
                    groups[a] = groups[a] | groups[b]
                    del groups[b]
                else:
                    groups[b] = groups[a] | groups[b]
                    del groups[a]
                continue

            if stop_clustering:
                break

        try:
            mapped_tag, mapped_order = buff_event.get_tag_and_order()
            sorted_tag = (tuple(mapped_order[0]), tuple(sorted(mapped_order[1])))
        except Exception:
            return None

        if record_tag:
            self.fxfx_clustered_final_states.add(sorted_tag)

        buff_event.fks_methods = fks_methods
        if return_groups:
            return buff_event, sorted_tag, groups
        return buff_event, sorted_tag

    def _needs_fxfx_cluster_catalogue(self):
        if not self.inc_sudakov:
            return False
        try:
            ickkw = int(self.banner.get("run_card", "ickkw"))
        except Exception:
            return False
        return ickkw == 3

    def _ensure_fxfx_cluster_catalogue(self):
        if not self._needs_fxfx_cluster_catalogue():
            return
        if getattr(self, "lhe_input", None) is None:
            return
        source = getattr(self, "fxfx_catalogue_source", None) or self.lhe_input.path
        source = os.path.realpath(source)
        stat = os.stat(source)
        scale = getattr(self, "mW2_cluster_scale", 1.5)
        xis = tuple(getattr(self, "sudakov_xi_scan", None) or [scale])
        catalogue_key = (source, stat.st_size, stat.st_mtime_ns, xis)
        if (getattr(self, "_fxfx_cluster_catalog_ready", False)
                and getattr(self, "_fxfx_cluster_catalog_key", None) == catalogue_key):
            return
        self._fxfx_cluster_catalog_ready = False
        self.fxfx_clustered_final_states = set()
        self.fxfx_clustered_decay_processes = set()
        self._sudakov_source_ids = set()
        self._init_pole_masses_from_banner()
        logger.info("Collecting FxFx processes from %s at xi=%s", source, xis)
        try:
            with contextlib.closing(lhe_parser.EventFile(source)) as events:
                for event in events:
                    self._sudakov_source_ids.update(event.parse_reweight() or {"1001": event.wgt})
                    for xi in xis:
                        self.mW2_cluster_scale = xi
                        cluster_result = self._cluster_fxfx_event(event, record_tag=True, return_groups=True)
                        if not cluster_result:
                            continue
                        clustered_event, _, groups = cluster_result
                        for res_idx, res_pdg in ResonanceIdentifier.find_resonances(clustered_event):
                            if res_idx >= len(groups):
                                continue
                            decay_pdgs = [
                                int(event[i - 1].pid) for i in sorted(groups[res_idx])
                                if 1 <= i <= len(event) and event[i - 1].status == 1
                            ]
                            if len(decay_pdgs) >= 2:
                                self.fxfx_clustered_decay_processes.add(
                                    (int(res_pdg), tuple(sorted(decay_pdgs))))
        finally:
            self.mW2_cluster_scale = scale
        self._fxfx_cluster_catalog_key = catalogue_key
        self._fxfx_cluster_catalog_ready = True
        logger.info("Collected %d FxFx production and %d decay processes",
                    len(self.fxfx_clustered_final_states), len(self.fxfx_clustered_decay_processes))

    def _validate_fxfx_sudakov_catalogue(self, sud_mod):
        """Require compiled coverage for each clustered multi-particle process."""
        if not self._needs_fxfx_cluster_catalogue():
            return
        missing = [tag for tag in sorted(self.fxfx_clustered_final_states)
                   if len(tag[1]) > 1 and tag not in sud_mod.original_pdg_list_dict]
        if missing:
            raise MissingSudakovProcessError(
                "Sudakov library is missing FxFx processes %s. "
                "Generate the library from the complete event input and xi scan." % missing)

    # -------------------------------------------------------------------------
    # Process generation
    # -------------------------------------------------------------------------

    def _pdg_to_model_name(self, pdg):
        """Return the MG5 name for a PDG id, keeping track of antiparticles."""
        if not self.model:
            return None
        particle_dict = self.model.get("particle_dict") or {}
        part = particle_dict.get(pdg)
        if not part:
            part = particle_dict.get(-pdg)
        if not part:
            return None
        if pdg < 0 and not part.get("self_antipart"):
            return part.get("antiname") or ("anti_" + (part.get("name") or str(abs(pdg))))
        return part.get("name")

    def _sorted_tag_to_process(self, sorted_tag):
        if not sorted_tag or not self.model:
            return None
        incoming, outgoing = sorted_tag
        in_names = [self._pdg_to_model_name(pdg) for pdg in incoming]
        out_names = [self._pdg_to_model_name(pdg) for pdg in outgoing]
        if any(name is None for name in in_names + out_names):
            return None
        return "%s > %s" % (" ".join(in_names), " ".join(out_names))

    def _decay_pdgs_to_process(self, res_pdg, decay_pdgs):
        """Return a 1->N decay process string from PDG codes."""
        if not self.model:
            return None
        res_name = self._pdg_to_model_name(res_pdg)
        out_names = [self._pdg_to_model_name(pdg) for pdg in decay_pdgs]
        if res_name is None or any(name is None for name in out_names):
            return None
        return "%s > %s" % (res_name, " ".join(out_names))

    def _get_fxfx_extra_processes(self):
        if not (
            self._needs_fxfx_cluster_catalogue() and self.fxfx_clustered_final_states and self.model
        ):
            return []
        extra = []
        seen = set()
        for sorted_tag in sorted(self.fxfx_clustered_final_states):
            proc = self._sorted_tag_to_process(sorted_tag)
            if not proc or proc in seen:
                continue
            seen.add(proc)
            extra.append(proc)
        return extra

    def _get_fxfx_density_production_processes(self):
        if not (
            self._needs_fxfx_cluster_catalogue() and self.fxfx_clustered_final_states and self.model
        ):
            return []
        # Undecayed resonances (e.g. stable tops in ttH) have no decay matrix C
        # and use the scalar path at runtime: only final states with a decayed
        # resonance need a density production process.
        if not self.fxfx_clustered_decay_processes:
            return []
        decayed_resonance_pdgs = set()
        for res_pdg, decay_pdgs in self.fxfx_clustered_decay_processes:
            decayed_resonance_pdgs.add(abs(res_pdg))
        extra = []
        seen = set()
        for sorted_tag in sorted(self.fxfx_clustered_final_states):
            outgoing = sorted_tag[1]
            if not any(abs(pdg) in decayed_resonance_pdgs for pdg in outgoing):
                continue
            proc = self._sorted_tag_to_process(sorted_tag)
            if not proc or proc in seen:
                continue
            seen.add(proc)
            extra.append(proc)
        return extra

    def _get_fxfx_density_decay_processes(self):
        if not (
            self._needs_fxfx_cluster_catalogue()
            and self.fxfx_clustered_decay_processes
            and self.model
        ):
            return []
        extra = []
        seen = set()
        for res_pdg, decay_pdgs in sorted(self.fxfx_clustered_decay_processes):
            proc = self._decay_pdgs_to_process(res_pdg, decay_pdgs)
            if not proc or proc in seen:
                continue
            seen.add(proc)
            extra.append(proc)
        return extra

    def _prepare_fxfx_sudakov_inputs(self, event, sud_mod, event_to_sud):
        """Sudakov inputs: CM-frame kinematics in the canonical library order."""
        frame_transform = []

        p_in_sum = lhe_parser.FourMomentum()
        for part in event_to_sud:
            if getattr(part, "status", 0) == -1:
                p_in_sum += part
        needs_boost = not (
            (abs(p_in_sum.px) < 1e-6 * p_in_sum.E)
            and (abs(p_in_sum.py) < 1e-6 * p_in_sum.E)
            and (abs(p_in_sum.pz) < 1e-6 * p_in_sum.E)
        )
        if needs_boost:
            frame_transform.append(("boost_to_cm", FourMomentum(p_in_sum)))
            event_to_sud.boost(p_in_sum)

        # Initial state along z, as in the ickkw=0 path.
        initial = copy.deepcopy(event_to_sud[0])
        if not ((abs(initial.px) < 1e-6 * initial.E) and (abs(initial.py) < 1e-6 * initial.E)):
            frame_transform.append(("rotate_to_z", FourMomentum(initial)))
            for p in event_to_sud:
                p.set_momentum(
                    lhe_parser.FourMomentum(p).rotate_to_z(prot=lhe_parser.FourMomentum(initial))
                )

        # Remove the ~1e-7 E round-off of rotate_to_z: set_initial_mass_to_zero
        # strictly requires px = py = 0 and pz0 = -pz1 for the two beams.
        p0, p1 = event_to_sud[0], event_to_sud[1]
        E_in   = abs(p0.E) + abs(p1.E)
        pz_abs = (abs(p0.pz) + abs(p1.pz)) / 2.0
        sign0  = +1.0 if p0.pz >= 0 else -1.0
        p0.set_momentum(lhe_parser.FourMomentum([E_in/2.0, 0.0, 0.0,  sign0 * pz_abs]))
        p1.set_momentum(lhe_parser.FourMomentum([E_in/2.0, 0.0, 0.0, -sign0 * pz_abs]))

        event_to_sud.set_final_jet_mass_to_zero()
        event_to_sud.set_initial_mass_to_zero()
        event_to_sud.check_kinematics_only()

        gstr = math.sqrt(4.0 * math.pi * event.aqcd)  # G = √(4πα_s)

        mapped_tag, mapped_order = event_to_sud.get_tag_and_order()
        sorted_tag = (tuple(mapped_order[0]), tuple(sorted(mapped_order[1])))

        try:
            canon = list(sud_mod.original_pdg_list_dict[sorted_tag][1])
        except KeyError as exc:
            raise MissingSudakovProcessError(
                "Sudakov library has no process %s. Generate the library from "
                "the complete event input and xi scan." % (sorted_tag,)) from exc

        # perm[i]: position in canon of the final-state particle mapped_order[1][i].
        perm = []
        used = set()
        for r in mapped_order[1]:
            for idx, c in enumerate(canon):
                if c == r and idx not in used:
                    perm.append(idx)
                    used.add(idx)
                    break

        event_to_sud_order = copy.deepcopy(event_to_sud)
        event_to_sud_order[: len(mapped_tag[0])] = event_to_sud[: len(mapped_tag[0])]
        offset = len(mapped_tag[0])
        for i, target_pos in enumerate(perm):
            event_to_sud_order[target_pos + offset] = event_to_sud[i + offset]

        p_in = np.zeros(shape=(len(event_to_sud_order), 4))
        for i, el in enumerate(event_to_sud_order):
            p_in[i] = [float(el.E), float(el.px), float(el.py), float(el.pz)]

        mapped_tag2, mapped_order2 = event_to_sud_order.get_tag_and_order()
        expected_order = list(sud_mod.original_pdg_list_dict[sorted_tag][1])
        actual_order = mapped_order2[1]
        if expected_order != actual_order:
            raise RuntimeError("Order in particle momenta does not match MG convention")

        incoming_pdgs = list(mapped_order2[0])
        outgoing_pdgs = list(mapped_order2[1])
        iflist = [-1] * len(incoming_pdgs) + [1] * len(outgoing_pdgs)
        pdg_order = incoming_pdgs + outgoing_pdgs

        return {
            "sorted_tag": sorted_tag,
            "p_in": p_in,
            "gstr": gstr,
            "iflist": iflist,
            "pdg_order": pdg_order,
            "perm": perm,  # LHE -> canonical MG5 order, final state only
            # Lorentz operations applied before the non-Lorentz mass
            # normalisations/reordering.  The decay system must follow these
            # operations before its on-shell adjustment.
            "frame_transform": frame_transform,
        }

    def _get_fxfx_ewsudpy_module(self, sud_mod, sorted_tag):
        """Return the ewsudpy module for a given sorted_tag (if available)."""
        try:
            result = sud_mod.pdg2ewsud_dict.get(sorted_tag)
            return result
        except Exception:
            return None

    def _set_ewsud_rij_ge_mw(self, ewsud_mod, disable_clamp):
        """Set fxfx_ignore_invariant_checks = disable_clamp (else rij_ge_mw = not it).

        Returns (kind, previous value) for _restore_ewsud_rij_ge_mw, or None if
        nothing was set.
        """
        if ewsud_mod is None:
            return None
        if hasattr(ewsud_mod, "fxfx_ignore_invariant_checks"):
            prev = getattr(ewsud_mod, "fxfx_ignore_invariant_checks")
            try:
                ewsud_mod.fxfx_ignore_invariant_checks = disable_clamp
                return ("fxfx", prev)
            except Exception:
                try:
                    ewsud_mod.fxfx_ignore_invariant_checks[...] = disable_clamp
                    return ("fxfx", prev)
                except Exception:
                    return None
        if not hasattr(ewsud_mod, "rij_ge_mw"):
            return None
        prev = getattr(ewsud_mod, "rij_ge_mw")
        value = not disable_clamp
        try:
            ewsud_mod.rij_ge_mw = value
            return ("rij", prev)
        except Exception:
            try:
                ewsud_mod.rij_ge_mw[...] = value
                return ("rij", prev)
            except Exception:
                return None

    def _restore_ewsud_rij_ge_mw(self, ewsud_mod, prev):
        """Restore invariant-clamp setting on the ewsudpy module if it was set."""
        if prev is None or ewsud_mod is None:
            return
        kind = "rij"
        value = prev
        if isinstance(prev, tuple) and len(prev) == 2:
            kind, value = prev
        if kind == "fxfx" and hasattr(ewsud_mod, "fxfx_ignore_invariant_checks"):
            try:
                ewsud_mod.fxfx_ignore_invariant_checks = value
            except Exception:
                try:
                    ewsud_mod.fxfx_ignore_invariant_checks[...] = value
                except Exception:
                    pass
            return
        if not hasattr(ewsud_mod, "rij_ge_mw"):
            return
        try:
            ewsud_mod.rij_ge_mw = value
        except Exception:
            try:
                ewsud_mod.rij_ge_mw[...] = value
            except Exception:
                pass

    def _is_2to1_topology(self, clustered_event):
        """Whether the clustered event is 2 -> 1.

        Its only scale is s = M², with no invariant for Sudakov logarithms.
        """
        n_initial = sum(1 for p in clustered_event if p.status == -1)
        n_final = sum(1 for p in clustered_event if p.status == 1)
        is_2to1 = n_initial == 2 and n_final == 1
        return is_2to1

    def _has_small_invariants(self, p_in, iflist, mw2=None):
        """Whether any pair has |s_ij| < M_W² (soft/collinear: pass-through).

        s_ij = (p_i + sign·p_j)² = m_i² + m_j² + sign·2·(p_i·p_j), the
        convention of Source/kin_functions.f::SumDot. The mass terms matter
        for legs on shell at a non-light scale (the Z in Z+jets:
        m_Z² ≈ 1.28·M_W²): without them forward-Z events with r_ij ≈ 0 in
        Fortran would pass and the kernel's rij_ge_mw clamp would act on the
        log argument.
        """
        if mw2 is None:
            mw2 = MW_POLE**2

        nlegs = len(iflist)
        for i in range(nlegs):
            for j in range(i + 1, nlegs):
                sign = float(iflist[i] * iflist[j])
                e  = p_in[i][0] + sign * p_in[j][0]
                px = p_in[i][1] + sign * p_in[j][1]
                py = p_in[i][2] + sign * p_in[j][2]
                pz = p_in[i][3] + sign * p_in[j][3]
                sij = e * e - px * px - py * py - pz * pz
                if abs(sij) < mw2:
                    return True
        return False

    # FxFx applies the invariant cut before evaluating these three variants.
    SUDAKOV_VARIANT_NAMES = ("central", "s_to_rij_off", "sherpa_qed")
    SUDAKOV_LO_VARIANT_NAMES = SUDAKOV_VARIANT_NAMES + ("both_off", "rij_ge_mw_off")

    def _collect_sudakov_weight_sources(self):
        """Collect full source IDs from the input header and event stream."""
        header = self.banner["initrwgt"] if "initrwgt" in self.banner else ""
        tree = ElementTree.fromstring("<initrwgt>" + header + "</initrwgt>")
        for group in tree.findall("weightgroup"):
            if group.get("name") == "ew_sudakov":
                raise ValueError("Input declares EW Sudakov weights; supply source event weights.")
        declared = set()
        for entry in tree.iter("weight"):
            declared.add(entry.attrib["id"])
            if re.search(r"scale_\S+_sud(?:\s|$)", "".join(entry.itertext())):
                raise ValueError("Input declares EW Sudakov weights; supply source event weights.")
        self._sudakov_input_header = header
        self._sudakov_declared_source_ids = declared
        if not self._needs_fxfx_cluster_catalogue():
            self._sudakov_source_ids = set()
            source = getattr(self, "fxfx_catalogue_source", None) or self.lhe_input.path
            with contextlib.closing(lhe_parser.EventFile(source)) as events:
                for event in events:
                    self._sudakov_source_ids.update(event.parse_reweight() or {"1001": event.wgt})
        self._sudakov_source_ids = set(getattr(self, "_sudakov_source_ids", ())) | declared
        if not self._sudakov_source_ids:
            self._sudakov_source_ids.add("1001")

    def _configure_sudakov_weight_ids(self, source_ids, xis, variant_names=None):
        """Assign a unique numeric ID to each source, xi index, and variant."""
        sources = sorted(set(map(str, source_ids)),
                         key=lambda key: (0, int(key), key) if key.isdigit() else (1, 0, key))
        names = tuple(variant_names or self.SUDAKOV_VARIANT_NAMES)
        self._sudakov_xi_list = list(xis)
        self._sudakov_variant_names = names
        self._sudakov_source_ids = set(sources)
        self._sudakov_weight_map = {}
        self._sudakov_weight_metadata = {}
        reserved = set(sources) | {str(int(key)) for key in sources if key.isdigit()}
        slots = [(k, v, source) for k in range(len(xis))
                 for v in range(len(names)) for source in sources]
        for k, v, source in slots:
            if re.fullmatch(r"10[0-9]{2}", source):
                candidate = str((20 + 5*k + v)*100 + int(source) - 1000)
                if candidate not in reserved:
                    self._sudakov_weight_map[k, v, source] = candidate
                    reserved.add(candidate)
        candidate = 10000
        for k, v, source in slots:
            slot = (k, v, source)
            if slot not in self._sudakov_weight_map:
                while str(candidate) in reserved:
                    candidate += 1
                self._sudakov_weight_map[slot] = str(candidate)
                reserved.add(str(candidate))
                candidate += 1
            output_id = self._sudakov_weight_map[slot]
            self._sudakov_weight_metadata[output_id] = (source, k, xis[k], names[v])
        return list(self._sudakov_weight_metadata)

    def _prepare_sudakov_weight_ids(self):
        """Select the xi scan and variant layout for the configured event mode."""
        try:
            self._sudakov_ickkw = int(self.banner.get('run_card', 'ickkw'))
        except Exception:
            self._sudakov_ickkw = 0
        xis = [getattr(self, 'mW2_cluster_scale', 1.5)]
        names = self.SUDAKOV_LO_VARIANT_NAMES
        if self._sudakov_ickkw == 3:
            xis = self.sudakov_xi_scan or xis
            names = self.SUDAKOV_VARIANT_NAMES
        elif self.sudakov_xi_scan and len(self.sudakov_xi_scan) > 1:
            logger.warning("The xi scan requires ickkw=3; using one set of Sudakov variants.")
        return self._configure_sudakov_weight_ids(self._sudakov_source_ids, xis, names)

    def _sudakov_weight_header(self):
        """Describe every output ID by its full source ID, xi, and variant."""
        header = getattr(self, "_sudakov_input_header", "")
        missing = self._sudakov_source_ids - getattr(self, "_sudakov_declared_source_ids", set())
        if missing:
            header += '\n<weightgroup name="source_weights">\n'
            for source in sorted(missing):
                header += '<weight id=%s>Source weight %s</weight>\n' % (quoteattr(source), escape(source))
            header += '</weightgroup>\n'
        header += '\n<weightgroup name="ew_sudakov" weight_name_strategy="includeIdInWeightName">\n'
        for output_id, (source, k, xi, variant) in self._sudakov_weight_metadata.items():
            header += ('<weight id=%s source_id=%s xi_index="%d" xi="%s" variant=%s>'
                       'source=%s xi=%s variant=%s</weight>\n' %
                       (quoteattr(output_id), quoteattr(source), k, repr(float(xi)), quoteattr(variant),
                        escape(source), repr(float(xi)), escape(variant)))
        return header + '</weightgroup>\n'

    def _build_sudakov_rwgt_dict(self, event, weights, variant_names=None):
        """Multiply each full source weight by the selected Sudakov factors."""
        sources = event.parse_reweight() or {"1001": event.wgt}
        names = tuple(variant_names or getattr(self, "_sudakov_variant_names", self.SUDAKOV_VARIANT_NAMES))
        xi_idx = getattr(self, "_current_xi_idx", 0)
        if not hasattr(self, "_sudakov_weight_map"):
            xis = getattr(self, "sudakov_xi_scan", None) or [getattr(self, "mW2_cluster_scale", 1.5)]
            self._configure_sudakov_weight_ids(sources, xis, names)
        factors = list(weights)[:len(names)]
        factors += [1.0] * (len(names) - len(factors))
        result = {"orig": event.wgt} if xi_idx == 0 else {}
        for source, value in sources.items():
            for variant, factor in enumerate(factors):
                slot = (xi_idx, variant, str(source))
                if slot not in self._sudakov_weight_map:
                    raise ValueError("Sudakov weight layout does not contain source/xi/variant %s" % (slot,))
                result[self._sudakov_weight_map[slot]] = value * factor
        return result

    def _compute_ewsudakov_fxfx_reweight(self, event, sud_mod):
        """FxFx-aware scalar Sudakov reweighting."""
        self._init_pole_masses_from_banner()
        # A 2 -> 1 input (qq̄ -> resonance, no radiation) has no Sudakov logs:
        # unit weights, before clustering.
        n_init  = sum(1 for p in event if p.status == -1)
        n_final = sum(1 for p in event if p.status ==  1)
        if n_init == 2 and n_final == 1:
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        cluster_result = self._cluster_fxfx_event(event, record_tag=True)
        if not cluster_result:
            # Without a Born-like object there is no Sudakov factor: unit weights.
            _count_fks_event("events_clustering_failed")
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        event_to_sud, _ = cluster_result

        # Policy "unit_weight": unit weights for any clustering history with a
        # non-success FKS method.
        _fks_hist = getattr(event_to_sud, "fks_methods", [])
        if any(m not in FKS_SUCCESS_METHODS for m in _fks_hist):
            _count_fks_event("events_with_fsr_fallback")
            if FKS_FALLBACK_POLICY == "unit_weight":
                _count_fks_event("events_unit_weight_by_policy")
                return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        if self._is_2to1_topology(event_to_sud):
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        try:
            prep = self._prepare_fxfx_sudakov_inputs(event, sud_mod, event_to_sud)
        except RuntimeError:
            sys.exit(3)

        # Unit weights for small invariants. This cut makes the Fortran
        # rij_ge_mw clamp inert, so both_off and rij_ge_mw_off would equal
        # s_to_rij_off and central; they are not emitted.
        if self._has_small_invariants(prep["p_in"], prep["iflist"]):
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        rij_override_mod = None
        rij_override_prev = None
        _ignore_invs = getattr(self, "fxfx_ignore_invariant_checks", False)
        if _ignore_invs:
            rij_override_mod = self._get_fxfx_ewsudpy_module(sud_mod, prep["sorted_tag"])
            rij_override_prev = self._set_ewsud_rij_ge_mw(rij_override_mod, False)
        try:
            res = sud_mod.ewsudakov(prep["sorted_tag"], prep["p_in"], prep["gstr"])
        finally:
            if rij_override_prev is not None:
                self._restore_ewsud_rij_ge_mw(rij_override_mod, rij_override_prev)

        # res[0] = Born, res[1] = sherpa_qed (sud_mod=0), res[2] = central,
        # res[3] = s_to_rij_off; res[4] (both_off) and res[5] (rij_ge_mw_off)
        # equal res[3] and res[2] after the small-invariant cut.
        if abs(res[0]) > 1e-30:
            sudrats = [
                1.0 + res[2] / res[0],   # central
                1.0 + res[3] / res[0],   # s_to_rij_off
                1.0 + res[1] / res[0],   # sherpa_qed
            ]
        else:
            sudrats = [1.0] * 3

        sudrats = limit_sudakov_factors(sudrats, "events_runaway_damped_scalar")

        return self._build_sudakov_rwgt_dict(event, sudrats)

    def _compute_density_ewsudakov_reweight(self, event, sud_mod):
        """Density-matrix EW Sudakov reweighting (FxFx, ickkw=3).

        For events with decayed t, W, Z resonances: Δ_h is applied per
        resonance helicity on the Born-like density matrix B of the clustered
        event and contracted with the decay matrix C,
        w = <B^EWSL, C> / <B, C> (compute_density_weight). Events without a
        decayed resonance, or whose density evaluation fails, use the scalar
        FxFx path. Returns the rwgt dict of _build_sudakov_rwgt_dict.
        """
        self._init_pole_masses_from_banner()

        density_logger = getattr(self, "_density_logger", None)
        if density_logger is None:
            density_logger = DensitySudakovLogger()
            self._density_logger = density_logger

        # 2 -> 1 input: unit weights before clustering (see
        # _compute_ewsudakov_fxfx_reweight).
        n_init  = sum(1 for p in event if p.status == -1)
        n_final = sum(1 for p in event if p.status ==  1)
        if n_init == 2 and n_final == 1:
            density_logger.log_event("passthrough_2to1_raw", [], 0, 1.0)
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        cluster_result = self._cluster_fxfx_event(event, record_tag=True, return_groups=True)
        if not cluster_result:
            # Evaluate the scalar path when clustering information is unavailable.
            density_logger.log_event("fallback_no_cluster", [], 0, 1.0)
            return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

        clustered_event, cluster_sorted_tag, groups = cluster_result

        # Fallback policy as in _compute_ewsudakov_fxfx_reweight.
        _fks_hist = getattr(clustered_event, "fks_methods", [])
        if any(m not in FKS_SUCCESS_METHODS for m in _fks_hist):
            _count_fks_event("events_with_fsr_fallback_density")
            if FKS_FALLBACK_POLICY == "unit_weight":
                _count_fks_event("events_unit_weight_by_policy")
                return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        if self._is_2to1_topology(clustered_event):
            density_logger.log_event("passthrough_2to1", [], 0, 1.0)
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        incoming = [p for p in clustered_event if p.status < 0]
        if len(incoming) >= 2:
            E1, E2 = incoming[0].E, incoming[1].E
            pz1, pz2 = incoming[0].pz, incoming[1].pz
            s_hat = (E1 + E2) ** 2 - (pz1 + pz2) ** 2  # beams along z

            # Degenerate beams (a zero-energy beam or ŝ < 1 GeV², e.g. a
            # lightlike final state) give no CM frame for the density matrices.
            if E1 < 1e-6 or E2 < 1e-6 or s_hat < 1.0:
                return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

        resonances = ResonanceIdentifier.find_resonances(clustered_event)

        # Only resonances rebuilt from decay products count: a singleton group
        # is a stable final-state particle.
        resonances = [
            (idx, pdg) for idx, pdg in resonances if idx < len(groups) and len(groups[idx]) > 1
        ]

        res_pdgs = [r[1] for r in resonances]
        res_dims = [SPIN_DIMS[abs(pdg)] for pdg in res_pdgs]
        total_dim = 1
        for d in res_dims:
            total_dim *= d
        if not resonances:
            total_dim = 0

        # No decayed resonance (only γ/g mothers, or stable resonances such as
        # the tops in ttH): scalar path.
        if not resonances:
            return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

        # Fixed buffers of the generated density kernel: density_delta and
        # born_diag(36), res arrays(10), allow_hel(300). Beyond them the
        # assumed-size Fortran arrays overrun silently: scalar path.
        if not density_dimensions_supported(total_dim, len(resonances)):
            _count_fks_event("events_density_dim_overflow")
            global _DIM_OVERFLOW_WARNED
            if not _DIM_OVERFLOW_WARNED:
                logger.warning(
                    "Resonance-helicity space too large for the density kernel "
                    "(total_dim=%d, nres=%d; caps %d/%d/%d): falling back "
                    "to the scalar Sudakov path.",
                    total_dim,
                    len(resonances),
                    MAX_DENSITY_DIM,
                    MAX_DENSITY_RESONANCES,
                    MAX_DENSITY_ALLOW_HEL,
                )
                _DIM_OVERFLOW_WARNED = True
            return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

        # Decay systems of C from the clustering groups (evaluated later).
        C_def = []
        for res_idx, res_pdg in resonances:
            if res_idx >= len(groups):
                msg = f"Resonance index {res_idx} (pdg={res_pdg}) out of groups range (len={len(groups)}), skipping"
                warnings.warn(msg, RuntimeWarning)
                continue
            group_lhe = sorted(groups[res_idx])

            decay_lhe = []
            decay_pdgs = []
            for lhe_idx in group_lhe:
                if lhe_idx < 1 or lhe_idx > len(event):
                    continue
                particle = event[lhe_idx - 1]
                if int(getattr(particle, "status", 0)) != 1:
                    continue
                decay_lhe.append(lhe_idx)
                decay_pdgs.append(int(getattr(particle, "pid", 0)))
            C_def.append(
                {
                    "res_cluster_idx": res_idx,
                    "res_pdg": res_pdg,
                    "decay_lhe": decay_lhe,
                    "decay_pdgs": decay_pdgs,
                }
            )

        self._density_last_C_def = C_def

        event_to_sud = clustered_event
        try:
            prep = self._prepare_fxfx_sudakov_inputs(event, sud_mod, event_to_sud)
        except RuntimeError:
            sys.exit(3)

        if self._has_small_invariants(prep["p_in"], prep["iflist"]):
            density_logger.log_event("passthrough_small_sij", resonances, total_dim, 1.0)
            return self._build_sudakov_rwgt_dict(event, [1.0, 1.0, 1.0])

        weight_var = None
        try:
            use_density_path = False

            if hasattr(sud_mod, "density_sudakov") and resonances:
                try:
                    # 1-based resonance positions in the canonical MG5 order
                    # (prep["perm"] maps the final state of clustered_event).
                    res_indices_fortran = []
                    offset = len(prep["sorted_tag"][0])
                    perm = prep["perm"]
                    for idx, pdg in resonances:
                        if idx < offset:
                            fortran_idx = idx + 1
                        else:
                            final_idx = idx - offset
                            canonical_final_idx = perm[final_idx]
                            canonical_idx = canonical_final_idx + offset
                            fortran_idx = canonical_idx + 1
                        res_indices_fortran.append(fortran_idx)

                    rij_override_mod = None
                    rij_override_prev = None
                    if getattr(self, "fxfx_ignore_invariant_checks", False):
                        rij_override_mod = self._get_fxfx_ewsudpy_module(
                            sud_mod, prep["sorted_tag"]
                        )
                        rij_override_prev = self._set_ewsud_rij_ge_mw(rij_override_mod, False)
                    try:
                        density_result = sud_mod.density_sudakov(
                            prep["sorted_tag"],
                            prep["p_in"],
                            prep["gstr"],
                            res_indices_fortran,
                            res_dims,
                        )
                    finally:
                        if rij_override_prev is not None:
                            self._restore_ewsud_rij_ge_mw(rij_override_mod, rij_override_prev)

                    # The Fortran wrapper returns zeros for density_born; B comes
                    # from DensityHelper and the standalone density_prod modules
                    # (GET_DENSITY).
                    total_dim = density_result["total_dim"]
                    B = None

                    if getattr(self, "density_prod_id_to_path", None) and getattr(
                        self, "density_prod_f2pylib", None
                    ):
                        try:
                            helper = DensityHelper(
                                self, self.density_prod_id_to_path, self.density_prod_f2pylib
                            )
                            B_matrix = helper.compute_density_matrix(
                                clustered_event, res_pdgs, res_dims
                            )
                            if B_matrix is not None:
                                B = DensityMatrix.from_full_matrix(B_matrix, res_pdgs)
                        except Exception:
                            pass

                    if B is None:
                        raise ValueError(
                            "DensityHelper failed to compute B matrix. "
                            "Check that rw_me_density_prod modules are generated and compiled."
                        )
                    delta_raw = density_result["density_delta"]
                    # s_to_rij=False variation
                    delta_var_raw = density_result.get("density_delta_var", None)
                    # Per-helicity Born from Fortran (summed over spectators, /IDEN)
                    born_diag_fortran_raw = density_result["born_diag"]

                    # Fortran -> Python helicity order.
                    _hel_perm = _fortran_to_python_hel_permutation(res_dims)
                    delta = np.array([delta_raw[_hel_perm[i]] for i in range(len(delta_raw))])
                    delta_var = None
                    if delta_var_raw is not None:
                        delta_var = np.array(
                            [delta_var_raw[_hel_perm[i]] for i in range(len(delta_var_raw))]
                        )
                    born_diag_fortran = np.array(
                        [born_diag_fortran_raw[_hel_perm[i]] for i in range(len(born_diag_fortran_raw))]
                    )

                    # Fortran returns Δ_h times the per-helicity Born, and that
                    # Born, both divided by IDEN, which cancels in
                    # Δ_h = density_delta[h] / born_diag[h].
                    if np.any(np.abs(delta) > 1e-30):
                        delta_normalized = np.zeros_like(delta)
                        for h in range(len(delta)):
                            if abs(born_diag_fortran[h]) > 1e-30:
                                delta_normalized[h] = delta[h] / born_diag_fortran[h]
                            else:
                                delta_normalized[h] = 0.0
                        delta = delta_normalized
                        if delta_var is not None and np.any(np.abs(delta_var) > 1e-30):
                            delta_var_normalized = np.zeros_like(delta_var)
                            for h in range(len(delta_var)):
                                if abs(born_diag_fortran[h]) > 1e-30:
                                    delta_var_normalized[h] = delta_var[h] / born_diag_fortran[h]
                                else:
                                    delta_var_normalized[h] = 0.0
                            delta_var = delta_var_normalized

                    C = self._get_decay_density_matrix(
                        event,
                        clustered_event,
                        resonances,
                        frame_transform=prep.get("frame_transform"),
                    )
                    if C is not None:
                        weight = compute_density_weight(B, C, delta)

                        weight_var = None
                        if delta_var is not None:
                            weight_var = compute_density_weight(B, C, delta_var)

                        use_density_path = True
                    else:
                        use_density_path = False

                except Exception as e:
                    warnings.warn(
                        f"Density matrix computation failed, falling back to scalar: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    use_density_path = False

            # Scalar path if the density evaluation is unavailable or failed.
            if not use_density_path:
                res = sud_mod.ewsudakov(prep["sorted_tag"], prep["p_in"], prep["gstr"])
                born, sud_weak = res[0], res[2]
                weight = 1.0 + sud_weak / born if abs(born) > 1e-30 else 1.0
                # s_to_rij_off variant res[3], as in the scalar FxFx path.
                weight_var = (1.0 + res[3] / born) if abs(born) > 1e-30 else 1.0

            density_logger.log_event(
                mode="density" if use_density_path else "scalar",
                resonances=resonances,
                dimension=total_dim,
                weight=weight,
            )

        except Exception:
            return self._compute_ewsudakov_fxfx_reweight(event, sud_mod)

        # The scalar kernel supplies the Sherpa QED factor; the density kernel
        # supplies the central and s_to_rij_off factors.
        weight_sherpa_qed = 1.0
        try:
            res_for_sq = sud_mod.ewsudakov(prep["sorted_tag"], prep["p_in"], prep["gstr"])
            if abs(res_for_sq[0]) > 1e-30:
                weight_sherpa_qed = 1.0 + res_for_sq[1] / res_for_sq[0]
        except Exception:  # noqa: BLE001 — robustness: never let extraction failure abort the event
            weight_sherpa_qed = 1.0
        weight, weight_var, weight_sherpa_qed = limit_sudakov_factors(
            [weight, weight_var if weight_var is not None else 1.0, weight_sherpa_qed],
            "events_runaway_damped_density")

        return self._build_sudakov_rwgt_dict(event, [weight, weight_var, weight_sherpa_qed])

    # -------------------------------------------------------------------------
    # Decay density matrices
    # -------------------------------------------------------------------------

    def _get_decay_density_matrix(
        self, full_event, clustered_event, resonances, frame_transform=None
    ):
        """Decay density matrix C = C_1 ⊗ ... ⊗ C_n of self._density_last_C_def.

        None if there is no decay system or any C_r cannot be computed.
        """
        c_defs = getattr(self, "_density_last_C_def", None)
        if not c_defs:
            return None

        decay_matrices = []

        for entry in c_defs:
            res_pdg = entry["res_pdg"]
            res_idx = entry["res_cluster_idx"]
            decay_lhe = entry["decay_lhe"]
            if not decay_lhe:
                continue

            decay_event = self._build_decay_event_from_cluster(
                full_event,
                clustered_event,
                res_idx,
                res_pdg,
                decay_lhe,
                frame_transform=frame_transform,
            )
            dim = SPIN_DIMS.get(abs(res_pdg), 2)
            C_r = self._compute_density_matrix_from_event(
                decay_event, [res_pdg], [dim], label=f"C:{res_pdg}"
            )
            if C_r is None:
                return None

            decay_matrices.append(C_r)

        if not decay_matrices:
            return None

        if len(decay_matrices) == 1:
            return decay_matrices[0]

        result = build_decay_tensor_product(decay_matrices)
        return result

    def _build_decay_event_from_cluster(
        self,
        full_event,
        clustered_event,
        res_idx,
        res_pdg,
        decay_lhe,
        frame_transform=None,
    ):
        """Minimal decay event: the clustered resonance -> its decay products.

        Clustering supplies leaf momenta transported with each recoil and parent
        projection. Event preparation supplies the final CM boost and rotation.
        Without stored transport, the LHE momenta define the decay projection.
        """
        decay_event = lhe_parser.Event()
        decay_event.wgt = getattr(full_event, "wgt", 0.0)
        decay_event.aqcd = getattr(full_event, "aqcd", 0.0)

        res_part = clustered_event[res_idx]
        onshell_res_p4 = FourMomentum([res_part.E, res_part.px, res_part.py, res_part.pz])

        # LHE labels identify the transported leaves and their particle metadata.
        transported = getattr(clustered_event, "_fxfx_decay_products", None)
        original_momenta = []
        decay_pids = []
        decay_masses = []
        sum_E = sum_px = sum_py = sum_pz = 0.0

        for lhe_idx in decay_lhe:
            if lhe_idx < 1 or lhe_idx > len(full_event):
                continue
            src = full_event[lhe_idx - 1]
            p4 = (FourMomentum(transported[res_idx][lhe_idx]) if transported is not None
                  else FourMomentum(src))
            original_momenta.append(p4)
            decay_pids.append(int(getattr(src, "pid", 0)))
            decay_masses.append(float(getattr(src, "mass", 0.0)))
            sum_E += p4.E
            sum_px += p4.px
            sum_py += p4.py
            sum_pz += p4.pz

        # Parent momentum carried by the input daughters.
        original_res_p4 = FourMomentum([sum_E, sum_px, sum_py, sum_pz])

        transformed_momenta = _transform_decay_products_to_onshell(
            original_momenta,
            original_res_p4,
            onshell_res_p4,
            frame_transform=frame_transform,
        )

        res_particle = lhe_parser.Particle(event=decay_event)
        res_particle.pid = int(res_pdg)
        res_particle.status = -1
        res_particle.px = float(res_part.px)
        res_particle.py = float(res_part.py)
        res_particle.pz = float(res_part.pz)
        res_particle.E = float(res_part.E)
        res_particle.mass = float(getattr(res_part, "mass", 0.0))
        decay_event.append(res_particle)

        for i, p4 in enumerate(transformed_momenta):
            p = lhe_parser.Particle(event=decay_event)
            p.pid = decay_pids[i]
            p.status = 1
            p.px = float(p4.px)
            p.py = float(p4.py)
            p.pz = float(p4.pz)
            p.E = float(p4.E)
            p.mass = decay_masses[i]
            decay_event.append(p)
        return decay_event

    def _compute_density_matrix_from_event(self, event, target_pdgs, res_dims, label="density"):
        """DensityMatrix of target_pdgs from GET_DENSITY of the standalone modules.

        A decay event (one incoming particle) uses rw_me_density_decay (C), a
        production event rw_me_density_prod (B). Returns None on failure;
        label is not used in the computation.
        """

        try:
            tag_raw, order = event.get_tag_and_order()
            nincoming = len(tag_raw[0])
        except Exception:
            nincoming = 2

        if nincoming == 1:
            if not getattr(self, "density_decay_id_to_path", None):
                return None
            id_to_path = self.density_decay_id_to_path
            f2pylib = self.density_decay_f2pylib
        else:
            if not getattr(self, "density_prod_id_to_path", None):
                return None
            id_to_path = self.density_prod_id_to_path
            f2pylib = self.density_prod_f2pylib

        helper = DensityHelper(self, id_to_path, f2pylib)

        try:
            density_matrix = helper.compute_density_matrix(event, target_pdgs, res_dims)

            if density_matrix is None:
                return None

            return DensityMatrix.from_full_matrix(density_matrix, target_pdgs)

        except Exception:
            return None
