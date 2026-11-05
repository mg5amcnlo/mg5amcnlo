# SPDX-License-Identifier: 0BSD
"""The unstable s-channel propagators a process may put on shell.

A phase-space mapping has to follow these resonances, or it undersamples their
peaks. This is a species-level reachability sweep over the subsets of external
legs, like the DAG prepass: it ignores helicity and colour, keeps the coupling
orders, and may overestimate. Masses and widths are only known at launch, so
every candidate whose mass and width are not zero in the model is kept.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..models.base import CouplingOrders, Model, Particle
from ..processes.ir import CanonicalProcessIR
from .dag_reachability import (
    _combine_coupling_order_tuples,
    _coupling_orders_within_limits,
    _masks_by_size,
    _normalize_coupling_order_limits,
    _ordered_splits,
)


def s_channel_resonances(
    process: CanonicalProcessIR,
    model: Model,
    max_coupling_orders: Mapping[str, int] | None = None,
) -> list[dict[str, object]]:
    """``[{"pdg", "labels", "color"}]`` of every possibly unstable particle (see
    ``_possibly_unstable``) that a tree current of two or more final-state legs
    can carry, with a complementary current to close the amplitude with. The
    pdg carries the summed charge of the outgoing legs it decays to."""

    if not any(_possibly_unstable(model, p) for p in model.particles.values()):
        return []
    limits = _normalize_coupling_order_limits(max_coupling_orders)
    legs = process.legs
    every = final = 0
    species: dict[int, dict[int, set[CouplingOrders]]] = {}
    for leg in legs:
        bit = 1 << (leg.label - 1)
        every |= bit
        if leg.is_final:
            final |= bit
        if leg.outgoing_pdg is not None:
            species.setdefault(bit, {}).setdefault(int(leg.outgoing_pdg), set()).add(())
    for mask in _masks_by_size(every):
        if mask & (mask - 1) == 0 or mask == every:
            continue
        current = species.setdefault(mask, {})
        for left_mask, right_mask in _ordered_splits(mask):
            left, right = species.get(left_mask), species.get(right_mask)
            if not left or not right:
                continue
            for left_pdg, left_orders in left.items():
                for right_pdg, right_orders in right.items():
                    for vertex in model.vertices_accepting(
                        left_pdg, right_pdg, color_accuracy=process.color_accuracy
                    ):
                        if model.skip_duplicate_vertex_orientation(vertex):
                            continue
                        vertex_orders = model.vertex_coupling_orders(vertex)
                        for a in left_orders:
                            for b in right_orders:
                                orders = _combine_coupling_order_tuples(
                                    a, b, vertex_orders
                                )
                                if _coupling_orders_within_limits(orders, limits):
                                    current.setdefault(vertex.particles[2], set()).add(
                                        orders
                                    )
    found: dict[tuple[int, tuple[int, ...]], dict[str, object]] = {}
    for mask, particles in species.items():
        if mask & ~final or mask.bit_count() < 2:
            continue
        decay_legs = [leg for leg in legs if mask >> (leg.label - 1) & 1]
        rest = species.get(every & ~mask, {})
        for pdg, orders in particles.items():
            if not _possibly_unstable(model, model.particle(pdg)):
                continue
            closing = rest.get(model.anti_particle(pdg), ())
            if not any(
                _coupling_orders_within_limits(
                    _combine_coupling_order_tuples(a, b), limits
                )
                for a in orders
                for b in closing
            ):
                continue
            charge = sum(model.charge(int(leg.pdg)) for leg in decay_legs)
            if abs(model.charge(pdg) - charge) > 1e-9:
                pdg = model.anti_particle(pdg)
            labels = tuple(leg.label for leg in decay_legs)
            found.setdefault(
                (pdg, labels),
                {
                    "pdg": pdg,
                    "labels": list(labels),
                    "color": int(model.color_rep(pdg)),
                },
            )
    return [found[key] for key in sorted(found)]


def _possibly_unstable(model: Model, particle: Particle) -> bool:
    """Whether neither the mass nor the width of a particle is zero in the
    model: a zero value without a parameter (ZERO in UFO) is final."""

    def zero(kind: str) -> bool:
        if getattr(particle, kind) != 0.0:
            return False
        provider = getattr(model, f"runtime_{kind}_parameter_name", None)
        if not callable(provider):
            return True
        try:
            return provider(particle.pdg) is None
        except KeyError:
            return False

    return not zero("mass") and not zero("width")
