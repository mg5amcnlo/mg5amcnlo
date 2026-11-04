# SPDX-License-Identifier: 0BSD
"""Export UMAMI lookup tables without changing numerical artifact contents.

Identical final particles permit relabelling of physical colour contributions.
Other equivalences require generation certificates or exact structural keys;
equal masses or parameter defaults are not a proof. The exception is the
explicit ``--flavor-blind`` assumption: its ungenerated group members are
exported through their representative's runtime and recorded in
``grouping.assumptions``. Integration multiplicities live in the JSON, not in
the C provider.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from itertools import combinations_with_replacement, islice, permutations
from typing import TYPE_CHECKING, Any

from .api_bundle import ApiBundlePayload

if TYPE_CHECKING:
    from ..generation.process_grouping import GroupMember

_LOGGER = logging.getLogger(__name__)

FLAVOR_BLIND_ASSUMPTION = (
    "flavor-blind matrix elements: ungenerated process-group members are "
    "evaluated through their relabelled representative (assumed, not proven)"
)


@dataclass(frozen=True)
class UmamiProcessInput:
    process_id: str
    expression: str
    color_accuracy: str
    external_pdgs: tuple[int, ...]
    physics: Mapping[str, Any]
    aliases: tuple[Mapping[str, Any], ...] = ()
    color_sectors: tuple[Mapping[str, Any], ...] = ()
    # Keys certify equality of the normalized, available-helicity-summed
    # squared contribution, in this process's external momentum order.
    structural_keys: Mapping[str, str] | None = None
    # Used when re-exporting an appended compact OTF process: its previous SDK
    # contains authoritative physical LHE tags, while physics.json stays compact.
    color_flows: Mapping[str, Sequence[Sequence[int]]] | None = None
    # Ungenerated group members evaluated through this process's runtime:
    # ME_member(p) = factor * ME_this(q) with q[r] = p[permutation[r]].
    group_members: tuple[GroupMember, ...] = ()


def _records(value: Any) -> list[Mapping[str, Any]]:
    return [record for record in value or () if isinstance(record, Mapping)]


def _finite(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"UMAMI {context} must be finite")
    return result


def _pair(value: Any) -> tuple[float, float]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return _finite(value[0], "parameter"), _finite(value[1], "parameter")
    return _finite(value, "parameter"), 0.0


def _model_tables(model: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    ir = model.get("ir", model.get("model", model))
    particles = {int(p["pdg_code"]): p for p in _records(ir.get("particles"))}
    parameters = {str(p["name"]): p for p in _records(ir.get("parameters"))}
    defaults = {
        str(k): _pair(v) for k, v in model.get("parameter_defaults", {}).items()
    }
    for name, parameter in parameters.items():
        if name not in defaults and parameter.get("value") is not None:
            defaults[name] = _pair(parameter["value"])
        if name not in defaults:
            with suppress(TypeError, ValueError):
                defaults[name] = _pair(parameter.get("resolved_expression"))
    return particles, parameters, defaults


def _mass(
    particle: Mapping, parameters: Mapping, defaults: Mapping
) -> tuple[str, float]:
    name = str(particle["mass"])
    if name in defaults:
        real, imaginary = defaults[name]
    else:
        try:
            real, imaginary = float(name), 0.0
        except ValueError as exc:
            # UFO's reserved ZERO is a literal, not an independent parameter.
            if name.upper() != "ZERO":
                raise ValueError(f"UMAMI cannot resolve default mass {name!r}") from exc
            real, imaginary = 0.0, 0.0
    if imaginary or real < 0.0 or not math.isfinite(real):
        raise ValueError(f"UMAMI external mass {name!r} must be nonnegative and real")
    declaration = parameters.get(name, {})
    semantic = (
        name
        if declaration.get("nature") == "external"
        else str(declaration.get("resolved_expression", name))
    )
    if not declaration and real == 0.0:
        semantic = "0"
    return semantic, real


def _alpha_s_name(model: Mapping, physics: Mapping, parameters: Mapping) -> str | None:
    public = {p["name"]: p for p in _records(physics.get("model_parameters"))}
    explicit = model.get("extensions", {}).get("alpha_s_parameter")
    candidates = (
        [str(explicit)]
        if explicit
        else [
            name
            for name, p in parameters.items()
            if str(p.get("lhablock", "")).upper() == "SMINPUTS"
            and list(p.get("lhacode", ())) == [3]
            and p.get("nature") == "external"
        ]
    )
    candidates = [name for name in candidates if public.get(name, {}).get("mutable")]
    return candidates[0] if len(candidates) == 1 else None


def _slha_reference(declaration: Mapping) -> dict[str, Any]:
    """The parameter card entry of an external UFO parameter, if it has one."""
    if declaration.get("nature") != "external" or not declaration.get("lhablock"):
        return {}
    return {
        "lhablock": str(declaration["lhablock"]).upper(),
        "lhacode": [int(code) for code in declaration.get("lhacode", ())],
    }


def _flow_id(word: Sequence[int]) -> str:
    return "flow:" + (",".join(str(label) for label in word) if word else "singlet")


def _word(sector: Mapping) -> tuple[int, ...]:
    return tuple(
        sector.get("word_labels") or next(iter(sector.get("color_words", ())), ())
    )


def _permutation(values: Sequence[int], count: int) -> tuple[int, ...]:
    result = tuple(int(i) for i in values)
    if sorted(result) != list(range(count)):
        raise ValueError("UMAMI external permutation is not a complete bijection")
    return result


def _member_permutation(
    member: GroupMember, count: int, incoming: int
) -> tuple[int, ...]:
    """The ``representative leg -> member leg`` map; incoming swaps are allowed."""
    permutation = _permutation(member.permutation, count)
    if len(member.pdgs) != count or any(
        (leg < incoming) != (target < incoming)
        for leg, target in enumerate(permutation)
    ):
        raise ValueError(
            f"UMAMI group member {member.process_id!r} must keep initial legs initial"
        )
    return permutation


def _color_flow(
    word: Sequence[int],
    external: Sequence[Mapping],
    particles: Mapping,
    sector: Mapping | None = None,
) -> list[list[int]]:
    """Assign LHE tags from all-outgoing fundamental chains/adjoint cycles."""
    labels = {int(p["label"]): i for i, p in enumerate(external)}
    roles = {}
    for label, slot in labels.items():
        p = external[slot]
        representation = int(particles[int(p["pdg"])]["color"])
        if p["role"] == "initial" and abs(representation) == 3:
            representation = -representation
        roles[label] = representation
    colored = [int(label) for label in word if roles[int(label)] != 1]
    if set(colored) != {label for label, rep in roles.items() if rep != 1}:
        raise ValueError("UMAMI LC word must contain every coloured external leg once")
    if len(colored) != len(set(colored)):
        raise ValueError("UMAMI LC word repeats an external leg")
    if any(rep not in (1, 3, -3, 8) for rep in roles.values()):
        raise ValueError(
            "UMAMI LHE tags require singlet, fundamental or adjoint colour"
        )
    result = [[0, 0] for _ in external]
    edge = 501
    if sector is not None and sector.get("kind") == "open-lines":
        lines = _records(sector.get("open_color_lines"))
        chains = [
            [
                int(line["fundamental_label"]),
                *(int(label) for label in line.get("adjoint_labels", ())),
                int(line["antifundamental_label"]),
            ]
            for line in lines
        ]
        flattened = [label for chain in chains for label in chain]
        if sorted(flattened) != sorted(colored):
            raise ValueError("UMAMI colour-sector lines do not cover the LC word")
        for chain in chains:
            if (
                roles[chain[0]] != 3
                or roles[chain[-1]] != -3
                or any(roles[label] != 8 for label in chain[1:-1])
            ):
                raise ValueError(
                    "UMAMI colour-sector endpoints disagree with the model"
                )
            result[labels[chain[0]]][0] = edge
            for label in chain[1:-1]:
                result[labels[label]] = [edge + 1, edge]
                edge += 1
            result[labels[chain[-1]]][1] = edge
            edge += 1
    elif colored and all(roles[label] == 8 for label in colored):
        for index, label in enumerate(colored):
            result[labels[label]] = [edge + index, edge + (index - 1) % len(colored)]
    else:
        current = None
        for label in colored:
            role = roles[label]
            if role == 3:
                if current is not None:
                    raise ValueError(
                        "UMAMI LC word has an unterminated fundamental chain"
                    )
                current = edge
                result[labels[label]][0] = current
            elif role == 8 and current is not None:
                result[labels[label]] = [current + 1, current]
                current += 1
            elif role == -3 and current is not None:
                result[labels[label]][1] = current
                edge = current + 1
                current = None
            else:
                raise ValueError(
                    "UMAMI LC word does not describe complete fundamental chains"
                )
        if current is not None:
            raise ValueError("UMAMI LC word has an unterminated fundamental chain")
    for index, p in enumerate(external):
        if p["role"] == "initial":
            result[index].reverse()
    return result


def _safe_replay(process: UmamiProcessInput) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Convert certified label maps; weights may fold reflection, not one flow."""
    physics = process.physics
    external = _records(physics["external_particles"])
    count = len(external)
    labels = {int(p["label"]): i for i, p in enumerate(external)}
    identity = tuple(range(count))
    colors = {str(c["id"]): c for c in _records(physics.get("color_components"))}
    result = {key: (key, identity) for key in colors}
    certificate = physics.get("extensions", {}).get("lc_topology_replay")
    if not certificate:
        # These public nontrivial mappings are certified trace reflections.
        for key, color in colors.items():
            representative = str(color.get("representative_id", key))
            if representative in colors and float(color.get("coefficient", 1.0)) == 1.0:
                result[key] = (representative, identity)
        return result
    sectors = {int(s["id"]): s for s in process.color_sectors}
    sector_flow_ids = {_flow_id(_word(s)) for s in sectors.values()}
    helicities = {tuple(h["values"]) for h in _records(physics.get("helicities"))}
    for partition in _records(certificate.get("partitions")):
        if (partition.get("proof") or {}).get("status") != "proven":
            continue
        representative = sectors.get(int(partition["representative_sector_id"]))
        if representative is None:
            continue
        rep_id = _flow_id(_word(representative))
        for sector_id, pairs in zip(
            partition["active_sector_ids"], partition["label_permutations"], strict=True
        ):
            sector = sectors.get(int(sector_id))
            if sector is None:
                continue
            mapping = dict(pairs)
            try:
                perm = _permutation(
                    [labels[int(mapping[int(p["label"])])] for p in external], count
                )
            except (KeyError, ValueError):
                continue
            # No beam/PDF symmetry is assumed. Restricted helicity coverage
            # must also be invariant under the proposed external relabelling.
            if any(
                perm[i] != i for i, p in enumerate(external) if p["role"] == "initial"
            ):
                continue
            if {tuple(h[i] for i in perm) for h in helicities} != helicities:
                continue
            word = _word(sector)
            target_ids = [_flow_id(word)]
            if sector.get("kind") == "single-trace" and len(word) > 2:
                reflected = _flow_id((word[0], *reversed(word[1:])))
                # Only an additional public flow absent from the sector plan
                # is a folded reflection. An explicit reversed sector has its
                # own proof/permutation and must not be overwritten here.
                if reflected not in sector_flow_ids:
                    target_ids.append(reflected)
            for target in target_ids:
                if target in colors and rep_id in colors:
                    result[target] = (rep_id, perm)
    return result


def _channel_pattern(order, incoming, masses, classes=None):
    """The cyclic colour order from the first incoming leg, with the final-state
    legs known only by their mass and class: orders with equal patterns share
    one phase-space map up to relabelling final-state legs of equal mass and
    class. The classes are those of _leg_classes."""
    start = order.index(0)
    rotated = order[start:] + order[:start]
    pattern = tuple(
        ("initial", slot)
        if slot < incoming
        else ("final", masses[slot], None if classes is None else classes[slot])
        for slot in rotated
    )
    return pattern, rotated


def _leg_classes(pdgs, roles):
    """The legs that flavour-blind cuts treat alike: coloured partons of any
    flavour, and colour singlets of the same PDG only."""
    return ["coloured" if role != 1 else int(pdg) for pdg, role in zip(pdgs, roles)]


def _relabelled(values, relabel):
    result = [None] * len(values)
    for slot, value in enumerate(values):
        result[relabel[slot]] = value
    return result


def _relabelled_runtime(runtime, relabel):
    return {
        **runtime,
        "permutation": [relabel[slot] for slot in runtime["permutation"]],
    }


# Phase-space partners of one colour order beyond this many are dropped: any
# subset gives correct channel weights, as each selector lists its own.
_MAX_PARTNERS = 64
# Resonance configurations of one process beyond this many are not enumerated.
_MAX_CONFIGURATIONS = 1024


def _roles(pdgs, incoming, particles):
    """The colour representation of each leg in the all-outgoing convention."""
    roles = []
    for slot, pdg in enumerate(pdgs):
        representation = int(particles[int(pdg)]["color"])
        if slot < incoming and abs(representation) == 3:
            representation = -representation
        roles.append(representation)
    return roles


def _line_blocks(word_slots, roles):
    """The complete open colour lines of an LC word in its order, or None for a
    trace or a word without fundamental lines."""
    blocks, current = [], None
    for slot in word_slots:
        if roles[slot] == 3:
            current = [slot]
        elif current is not None:
            current.append(slot)
            if roles[slot] == -3:
                blocks.append(current)
                current = None
    if not blocks or current is not None or sum(map(len, blocks)) != len(word_slots):
        return None
    return blocks


def _classify_resonances(resonances, roles):
    """Split the candidates ``(pdg, legs, color)`` of a process into the colour
    singlets a phase-space channel can follow, ``(pdg, legs)``, the coloured
    ones and the singlets decaying to coloured and colourless particles."""
    mappable, coloured, mixed = [], [], []
    for pdg, legs, color in resonances:
        if color != 1:
            coloured.append((pdg, legs, color))
        elif len({roles[slot] == 1 for slot in legs}) > 1:
            mixed.append((pdg, legs))
        else:
            mappable.append((pdg, legs))
    return mappable, coloured, mixed


# A decay of the phase-space mapping has at most this many daughters.
_MAX_DECAY_DAUGHTERS = 3


def _decays_fit(chosen):
    """Whether every resonance of the set decays to at most three daughters,
    its nested resonances and its legs that no nested resonance contains."""
    for resonance in chosen:
        children = [
            other
            for other in chosen
            if other[1] < resonance[1]
            and not any(other[1] < r[1] < resonance[1] for r in chosen)
        ]
        covered = set().union(*(other[1] for other in children))
        if len(children) + len(resonance[1] - covered) > _MAX_DECAY_DAUGHTERS:
            return False
    return True


def _configurations(resonances, limit=_MAX_CONFIGURATIONS):
    """Every set of resonances that can be mapped together, pairwise nested or
    disjoint, each decaying to at most three daughters: the non-resonant one
    first, then the larger ones first. Returns
    ``(configurations, complete)``, enumerating at most ``limit`` of them."""

    def compatible(a, b):
        return a < b or b < a or not a & b

    found, truncated = [], False

    def extend(chosen, start):
        nonlocal truncated
        # a set the mapping cannot follow may still grow into one it can
        if _decays_fit(chosen):
            found.append(tuple(chosen))
        for index in range(start, len(resonances)):
            candidate = resonances[index]
            if all(compatible(candidate[1], r[1]) for r in chosen):
                if len(found) >= limit:
                    truncated = True
                    return
                extend([*chosen, candidate], index + 1)

    extend([], 0)
    found[1:] = sorted(found[1:], key=len, reverse=True)
    return found, not truncated


def _resonance_legs(resonance, configuration):
    """The legs of a resonance with its nested resonances kept contiguous."""
    inner = [
        other
        for other in configuration
        if other[1] < resonance[1]
        and not any(other[1] < r[1] < resonance[1] for r in configuration)
    ]
    covered = set().union(*(other[1] for other in inner))
    blocks = [_resonance_legs(other, configuration) for other in inner]
    blocks += [[slot] for slot in resonance[1] - covered]
    return [slot for block in sorted(blocks, key=min) for slot in block]


def _placements(word_slots, blocks, units):
    """Phase-space orders of one colour order, generated lazily: as in AmpliCol,
    the colour singlet units follow the antiquark ends of the lines, in every
    order and every distribution over the lines, and with three or more lines
    every non-cyclic order of the lines is a partner too (the same colour
    tensor)."""
    if blocks is None:
        yield list(word_slots) + [slot for unit in units for slot in unit]
        return
    for rest in permutations(blocks[1:]):
        lines = [blocks[0], *rest]
        for unit_order in permutations(units):
            for cuts in combinations_with_replacement(
                range(len(units) + 1), len(lines) - 1
            ):
                bounds = (0, *cuts, len(units))
                order = []
                for index, line in enumerate(lines):
                    order += line
                    for unit in unit_order[bounds[index] : bounds[index + 1]]:
                        order += unit
                yield order


def _phasespace_partners(word_slots, singlet_slots, roles, configurations):
    """The phase-space orders and resonance configurations integrating one
    colour order together, the first one its own: ``[(order, configuration)]``
    with ``configuration`` one of the process's ``configurations`` of
    ``(pdg, legs)``. A resonance decaying to coloured particles applies only if
    they are a complete colour line of this order. At most ``_MAX_PARTNERS``
    configurations are kept, in their order, and as many placements of each
    as fit."""
    blocks = _line_blocks(word_slots, roles) if word_slots else None
    lines = {frozenset(block) for block in blocks or ()}
    applicable = [
        configuration
        for configuration in configurations
        if all(
            legs in lines or all(roles[slot] == 1 for slot in legs)
            for _, legs in configuration
        )
    ][:_MAX_PARTNERS]
    per_configuration = max(1, _MAX_PARTNERS // len(applicable))
    partners = []
    for configuration in applicable:
        top = [
            r
            for r in configuration
            if not any(r[1] < other[1] for other in configuration)
        ]
        units = [
            _resonance_legs(r, configuration)
            for r in top
            if all(roles[slot] == 1 for slot in r[1])
        ]
        covered = {slot for unit in units for slot in unit}
        units += [[slot] for slot in singlet_slots if slot not in covered]
        units.sort(key=min)
        orders = islice(_placements(word_slots, blocks, units), per_configuration)
        partners += [(order, configuration) for order in orders]
    return partners


class _InternTable(list):
    """A list with a hash index of its entries, so interning is O(1)."""

    def __init__(self):
        super().__init__()
        self.positions: dict[Any, int] = {}


def _frozen(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_frozen(v) for v in value)
    return value


def _intern(table: _InternTable, value: Any) -> int:
    key = _frozen(value)
    index = table.positions.get(key)
    if index is None:
        index = table.positions[key] = len(table)
        table.append(value)
    return index


def _identical_orbit(order, flow, pdgs, incoming, helicities, word_size):
    """An oriented physical colour tensor modulo identical outgoing labels.

    Equal keys induce the explicit bijection between the two complete orders.
    It fixes incoming slots, preserves particle identity, directed colour edges
    and the available helicity domain. Exchanging identical fermions changes
    the overall amplitude sign, not this squared physical contribution. This
    argument is independent of runtime topology/reflection reuse certificates.
    """

    def structure(candidate):
        tags = {}
        directed_edges = []
        for slot in candidate:
            for tag in flow[slot] if flow is not None else ():
                directed_edges.append(tags.setdefault(tag, len(tags) + 1) if tag else 0)
        return (
            tuple(
                ("incoming", slot) if slot < incoming else ("final", pdgs[slot])
                for slot in candidate
            ),
            tuple(directed_edges),
        )

    def helicity_domain(candidate):
        return tuple(
            sorted(tuple(h["values"][slot] for slot in candidate) for h in helicities)
        )

    starts = [0]
    if flow is not None and word_size:
        # Undo incoming crossing to identify oriented open-line starts from
        # the actual colour tensor. A closed trace has no such endpoints.
        pairs = [
            tuple(reversed(flow[s])) if s < incoming else tuple(flow[s])
            for s in order[:word_size]
        ]
        if all(any(pair) for pair in pairs):
            starts = [i for i, (left, right) in enumerate(pairs) if left and not right]
            if not starts:
                starts = list(range(word_size)) if all(all(p) for p in pairs) else [0]
    # Re-anchoring a trace, or cyclically rotating complete open-line blocks,
    # changes no colour tensor. Never reverse it or arbitrarily sort blocks.
    candidates = [order[i:word_size] + order[:i] + order[word_size:] for i in starts]
    # The key is (legs, colour edges, helicity domain), compared in that order.
    # The helicity domain dominates its cost and can only decide between
    # candidates tied on the cheaper structure, so build it for those alone.
    scored = [(structure(candidate), candidate) for candidate in candidates]
    best = min(score for score, _ in scored)
    return min(
        ((*score, helicity_domain(candidate)), candidate)
        for score, candidate in scored
        if score == best
    )


def _add_terms(entry, terms):
    for term in terms:
        existing = next(
            (
                value
                for value in entry["matrix_elements"]
                if value["pdg_ids"] == term["pdg_ids"]
            ),
            None,
        )
        if existing is None:
            entry["matrix_elements"].append(dict(term))
        else:
            existing["factor"] += term["factor"]


def _exported_entry(data, relabel, member, contribution, helicity_groups, mappings):
    """An exported contribution with its final-state legs relabelled; its
    ``multichannels`` are filled in by _group_phasespace_channels."""
    pdgs, order, flow, runtime = contribution
    color_index = (
        None if flow is None else _intern(data["color_flows"], _relabelled(flow, relabel))
    )
    entry_runtime = _relabelled_runtime(runtime, relabel)
    if color_index is not None:
        entry_runtime["color_index"] = color_index
    return {
        "matrix_elements": [
            {
                "pdg_ids": _intern(data["pdg_ids"], _relabelled(list(pdgs), relabel)),
                "factor": 1.0,
            }
        ],
        "color_order": _intern(data["color_orders"], [relabel[slot] for slot in order]),
        "color_flows": color_index,
        "helicities": [
            [_intern(data["helicities"], _relabelled(v, relabel)) for v in group]
            for group in helicity_groups.values()
        ],
        "helicity_mappings": [
            {
                **mapping,
                "helicity": _intern(
                    data["helicities"], _relabelled(mapping["helicity"], relabel)
                ),
            }
            for mapping in mappings
        ],
        "runtime": entry_runtime,
        "members": [] if member is None else [member],
        "multichannels": [],
    }


def _unique_partners(partners, incoming, masses):
    """The partners ``(order, configuration)`` without repeated phase-space maps:
    equal cyclic orders following the same resonances."""
    unique, seen = [], set()
    for order, configuration in partners:
        _, rotated = _channel_pattern(order, incoming, masses)
        key = (tuple(rotated), frozenset((abs(pdg), legs) for pdg, legs in configuration))
        if key not in seen:
            seen.add(key)
            unique.append((order, configuration))
    return unique


def _phasespace_key(order, configuration, relabel, incoming, masses, channels):
    """The phase-space channel of a partner in relabelled legs, recorded in
    ``channels`` under its (cyclic order, resonances) key."""
    _, rotated = _channel_pattern([relabel[slot] for slot in order], incoming, masses)
    resonances = sorted(
        (sorted(relabel[slot] for slot in legs), pdg) for pdg, legs in configuration
    )
    key = (
        tuple(rotated),
        tuple((tuple(legs), abs(pdg)) for legs, pdg in resonances),
    )
    channels.setdefault(
        key,
        {
            "phasespace_order": rotated,
            "resonances": [{"pdg": pdg, "legs": legs} for legs, pdg in resonances],
        },
    )
    return key


def _group_phasespace_channels(data, selectors, phasespace_channels):
    """One UMAMI channel per canonical phase-space channel, which samples it: the
    copies of contributions it integrates. Each copy lists its own channel first
    in ``multichannels`` (index 0 of ``phasespace_channels``), then those of its
    partners, which only enter its weight g_0 / sum_k g_k: every partner has a
    copy of its own in its channel, so the copies add up to the contribution.
    The coloured resonances of its contributions are listed for the launch."""
    groups: dict[tuple, dict] = {}
    for entry, channel_key, keys, colored in selectors:
        group = groups.setdefault(
            channel_key, {"lookup": {}, "processes": [], "colored": []}
        )
        entry["multichannels"] = [
            group["lookup"].setdefault(key, len(group["lookup"])) for key in keys
        ]
        group["processes"].append(entry)
        group["colored"] += [r for r in colored if r not in group["colored"]]
    data["channels"] = []
    for group in groups.values():
        channels = [phasespace_channels[key] for key in group["lookup"]]
        data["channels"].append(
            {
                "phasespace_order": channels[0]["phasespace_order"],
                "phasespace_channels": channels,
                "colored_resonances": group["colored"],
                "processes": group["processes"],
            }
        )


def _merge_flavour_orbits(data):
    """Merge different flavours with a common proven computation.

    Do not use runtime reflection reuse to further collapse same-flavour colour
    orders: their integration partition was already constructed physically.
    A merged flavour is evaluated by the kept selector, so the complete runtime
    evaluation (permutation, factor and colour tags included) must agree.
    """
    incoming = data["provider"]["incoming_count"]
    for channel in data["channels"]:
        candidates = {}
        kept = []
        for entry in channel["processes"]:
            runtime = entry["runtime"]
            pdgs = {term["pdg_ids"] for term in entry["matrix_elements"]}
            key = (
                runtime["process_index"],
                runtime["color_id"],
                runtime["color_index"],
                tuple(runtime["permutation"]),
                runtime["factor"],
                tuple(data["pdg_ids"][next(iter(pdgs))][:incoming]),
                tuple(entry["multichannels"]),
            )
            previous = next(
                (
                    other
                    for other in candidates.get(key, ())
                    if pdgs.isdisjoint(t["pdg_ids"] for t in other["matrix_elements"])
                ),
                None,
            )
            if previous is None:
                candidates.setdefault(key, []).append(entry)
                kept.append(entry)
            else:
                _add_terms(previous, entry["matrix_elements"])
                previous["members"].extend(entry["members"])
        channel["processes"] = kept
    data["channels"] = [c for c in data["channels"] if c["processes"]]


def _export_helicities(data, runtime_helicities):
    """Map runtime helicities to exported ones, given in the input leg order."""
    entries = [e for c in data["channels"] for e in c["processes"]]
    for runtime in [e["runtime"] for e in entries] + [
        m["runtime"] for e in entries for m in e["members"]
    ]:
        permutation = runtime["permutation"]
        exported = []
        for values in runtime_helicities[runtime["process_index"]]:
            ordered = [0] * len(values)
            for runtime_leg, input_leg in enumerate(permutation):
                ordered[input_leg] = values[runtime_leg]
            exported.append(_intern(data["helicities"], ordered))
        runtime["helicity_export"] = exported


def build_umami_metadata(
    *,
    processes: Sequence[UmamiProcessInput],
    compiled_model: Mapping[str, Any],
    artifact_id: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return separate, directly consumable metadata documents per provider.

    Each provider exports sources ``(process, member)``: a generated process
    (``member`` is None) or one of its ungenerated group members, evaluated
    through the process's runtime without a runtime row of its own.
    """
    particles, parameters, defaults = _model_tables(compiled_model)
    groups: list[tuple[tuple, list[tuple], list[tuple[str, float]]]] = []

    def physical(source, incoming: int) -> tuple:
        process, member = source
        pdgs = process.external_pdgs if member is None else tuple(member.pdgs)
        return pdgs[:incoming], tuple(sorted(pdgs[incoming:]))

    def place(signature, source, masses, preferred=None):
        # Alternative selections of the same external process are not additive.
        key = physical(source, signature[1])
        compatible = next(
            (
                g
                for g in ([preferred] if preferred else []) + groups
                if g[0] == signature
                and all(physical(other, signature[1]) != key for other in g[1])
            ),
            None,
        )
        if compatible is None:
            compatible = (signature, [], masses)
            groups.append(compatible)
        compatible[1].append(source)
        return compatible

    for process in processes:
        external = _records(process.physics.get("external_particles"))
        if not external or len(external) != len(process.external_pdgs):
            raise ValueError(
                f"UMAMI process {process.process_id!r} lacks resolved external metadata"
            )
        if tuple(int(p["pdg"]) for p in external) != process.external_pdgs:
            raise ValueError("UMAMI external PDGs disagree with resolved physics")
        incoming = sum(p["role"] == "initial" for p in external)
        if [p["role"] for p in external] != ["initial"] * incoming + ["final"] * (
            len(external) - incoming
        ):
            raise ValueError("UMAMI requires initial legs before final legs")
        # The public catalog has already evaluated internal/derived parameters
        # for the artifact's defaults. Reuse those values instead of adding a
        # second symbolic evaluator here (the parameter card can omit them).
        process_defaults = {
            **defaults,
            **{
                str(p["name"]): (
                    _finite(p["default_real"], "parameter"),
                    _finite(p.get("default_imaginary", 0.0), "parameter"),
                )
                for p in _records(process.physics.get("model_parameters"))
            },
        }
        masses = [
            _mass(particles[pdg], parameters, process_defaults)
            for pdg in process.external_pdgs
        ]
        # The available helicity set, not its enumeration order.
        helicity_domain = tuple(
            sorted(
                tuple(h["values"]) for h in _records(process.physics.get("helicities"))
            )
        )
        signature = (
            len(external),
            incoming,
            process.color_accuracy,
            tuple(masses),
            helicity_domain,
        )
        home = place(signature, (process, None), masses)
        for member in process.group_members:
            # A member's legs carry its own masses; a member whose relabelling
            # moves unequal masses needs a provider of its own.
            relation = _member_permutation(member, len(external), incoming)
            member_masses = [
                _mass(particles[int(pdg)], parameters, process_defaults)
                for pdg in member.pdgs
            ]
            domain = tuple(
                sorted(tuple(_relabelled(v, relation)) for v in helicity_domain)
            )
            place(
                (*signature[:3], tuple(member_masses), domain),
                (process, member),
                member_masses,
                home,
            )
    documents = []
    for index, (_, sources, masses) in enumerate(groups):
        provider_id = f"p{index}"
        documents.append(
            _build_provider(
                provider_id,
                sources,
                masses,
                compiled_model,
                particles,
                parameters,
                len(groups) == 1,
            )
        )
        documents[-1]["provider"]["artifact_id"] = artifact_id
    return tuple(documents)


def _build_provider(
    provider_id, sources, masses, model, particles, parameters, single
):
    first_external = _records(sources[0][0].physics["external_particles"])
    count = len(first_external)
    incoming = sum(p["role"] == "initial" for p in first_external)
    identity = list(range(count))
    assumptions = [
        "identical-final-particle permutation-invariant observables and cuts",
        "final-state flavour-blind observables and cuts",
    ]
    if any(member is not None for _, member in sources):
        assumptions.append(FLAVOR_BLIND_ASSUMPTION)
    data: dict[str, Any] = {
        "kind": "pyamplicol-umami-metadata",
        "schema_version": 3,
        "provider": {
            "id": provider_id,
            "library": "libumami.so" if single else f"libumami_{provider_id}.so",
            "particle_count": count,
            "incoming_count": incoming,
            "masses": [m[1] for m in masses],
            "mass_parameters": [m[0] for m in masses],
            "color_accuracy": sources[0][0].color_accuracy,
        },
        "grouping": {"assumptions": assumptions},
        "channels": [],
        "pdg_ids": _InternTable(),
        "color_orders": _InternTable(),
        "color_flows": _InternTable(),
        "helicities": _InternTable(),
        "runtime_processes": [],
        "aliases": [],
        "xml_header": (
            '<generator name="pyAmpliCol">UMAMI artifact provider</generator>'
        ),
    }
    structural_representatives: dict[str, dict] = {}
    orbit_entries: dict[tuple, tuple[list[tuple[dict, list[int]]], list[int], dict]] = {}
    channel_patterns: dict[tuple, list[int]] = {}
    leg_masses = data["provider"]["masses"]
    # (order, resonances) of every phase-space channel, and the exported
    # contributions with the channels integrating them and coloured resonances
    phasespace_channels: dict[tuple, dict] = {}
    selectors: list[tuple[dict, tuple, list[tuple], list[dict]]] = []
    physical_count = 0
    runtime_helicities: list[list[list[int]]] = []
    runtime_indices: dict[str, int] = {}
    for process, group_member in sources:
        physics = process.physics
        external = _records(physics["external_particles"])
        labels = {int(p["label"]): i for i, p in enumerate(external)}
        helicities = _records(physics.get("helicities"))
        if process.process_id not in runtime_indices:
            runtime_indices[process.process_id] = len(data["runtime_processes"])
            data["runtime_processes"].append(
                {
                    "id": process.process_id,
                    "expression": process.expression,
                    "pdgs": list(process.external_pdgs),
                    "color_accuracy": process.color_accuracy,
                    "alpha_s_parameter": _alpha_s_name(model, physics, parameters),
                    "parameters": [
                        {
                            **p,
                            **_slha_reference(parameters.get(p["name"], {})),
                            "is_complex": (
                                parameters.get(p["name"], {}).get("parameter_type")
                                == "complex"
                                or p.get("kind") == "coupling"
                                or p.get("default_imaginary", 0.0) != 0.0
                            ),
                        }
                        for p in _records(physics.get("model_parameters"))
                    ],
                    "helicity_ids": [h["id"] for h in helicities],
                    "coverage": dict(physics.get("coverage", {})),
                }
            )
            data["aliases"].extend(
                {"process_id": process.process_id, **dict(a)} for a in process.aliases
            )
            runtime_helicities.append([list(h["values"]) for h in helicities])
        process_index = runtime_indices[process.process_id]
        source_id, pdgs = process.process_id, process.external_pdgs
        relation = None
        if group_member is not None:
            # Representative leg r is the member's leg relation[r]: helicity
            # values and colour tags move with it, momenta through the runtime.
            relation = _member_permutation(group_member, count, incoming)
            source_id, pdgs = group_member.process_id, tuple(group_member.pdgs)
            helicities = [
                {**h, "values": _relabelled(h["values"], relation)} for h in helicities
            ]
        # Helicity values in the input leg order, interned per contribution
        # once its final-state legs are relabelled to the channel.
        helicity_groups: dict[str, list[list[int]]] = {}
        helicity_mappings = []
        for h in helicities:
            key = str(h.get("representative_id", h["id"]))
            helicity_groups.setdefault(key, []).append(list(h["values"]))
            helicity_mappings.append(
                {
                    "helicity": list(h["values"]),
                    "representative_id": h.get("representative_id", h["id"]),
                    "coefficient": h.get("coefficient", 1.0),
                    "structural_zero": h.get("structural_zero", False),
                }
            )
        replay = _safe_replay(process)
        sectors_by_word = {_word(sector): sector for sector in process.color_sectors}
        roles = _roles(pdgs, incoming, particles)
        resonances = [
            (
                int(r["pdg"]),
                frozenset(labels[int(label)] for label in r["labels"]),
                int(r["color"]),
            )
            for r in _records(
                physics.get("extensions", {}).get("s_channel_resonances", ())
            )
        ]
        if relation is not None:
            resonances = [
                (pdg, frozenset(relation[slot] for slot in legs), color)
                for pdg, legs, color in resonances
            ]
        mappable, coloured, unmapped = _classify_resonances(resonances, roles)
        configurations, complete = _configurations(mappable)
        if not complete and process.color_accuracy == "lc":
            _LOGGER.warning(
                "%s: only the first %d s-channel resonance configurations are "
                "followed by the [amplicol] phase space",
                source_id,
                len(configurations),
            )
        # the legs of a complete colour line in some colour order of the process
        line_legs = set()
        colors = sorted(
            _records(physics.get("color_components")),
            key=lambda color: tuple(color.get("word", ())),
        )
        for color in colors:
            physical_count += 1
            color_id = str(color["id"])
            contracted = process.color_accuracy != "lc"
            word = tuple(int(v) for v in color.get("word", ()))
            order = [labels[label] for label in word]
            order += [i for i in identity if i not in order]
            order = list(_permutation(order, count))
            flow = (
                None
                if contracted
                else [list(pair) for pair in process.color_flows[color_id]]
                if process.color_flows is not None
                else _color_flow(word, external, particles, sectors_by_word.get(word))
            )
            physical_color = None if contracted else color_id
            if relation is not None:
                # Charge conjugation reverses the word and swaps the colour
                # and anticolour tags of every leg.
                word_slots = [relation[slot] for slot in order[: len(word)]]
                if group_member.conjugated:
                    word_slots.reverse()
                order = word_slots + [i for i in identity if i not in word_slots]
                if flow is not None:
                    if group_member.conjugated:
                        flow = [pair[::-1] for pair in flow]
                    flow = _relabelled(flow, relation)
                    physical_color = _flow_id(
                        [int(external[slot]["label"]) for slot in word_slots]
                    )
            color_index = None if flow is None else _intern(data["color_flows"], flow)
            if flow is not None:
                coloured_slots = [slot for slot in order if roles[slot] != 1]
                blocks = _line_blocks(coloured_slots, roles) if coloured_slots else None
                line_legs.update(frozenset(block) for block in blocks or ())
            rep_color, perm = replay.get(color_id, (color_id, tuple(identity)))
            runtime = {
                "process_index": process_index,
                "process_id": process.process_id,
                "color_id": None if contracted else rep_color,
                "color_index": 0 if color_index is None else color_index,
                "permutation": list(perm),
                "factor": 1.0,
            }
            semantic_key = (process.structural_keys or {}).get(color_id)
            if semantic_key:
                if semantic_key in structural_representatives:
                    runtime = dict(structural_representatives[semantic_key])
                else:
                    # The proof key already includes normalization and colour
                    # contraction; exact cross-flavour equality needs no cuts.
                    structural_representatives[semantic_key] = dict(runtime)
            if relation is not None:
                runtime = {
                    **runtime,
                    "permutation": [relation[slot] for slot in runtime["permutation"]],
                    "factor": runtime["factor"] * float(group_member.factor),
                }
            member = {
                "process_id": source_id,
                "pdgs": list(pdgs),
                "color_id": physical_color,
                "color_flows": color_index,
                "runtime": dict(runtime),
                "factor": 1.0,
            }
            orbit_key, aligned_order = _identical_orbit(
                order,
                flow,
                pdgs,
                incoming,
                helicities,
                len(word),
            )

            orbit = (source_id, orbit_key)
            if orbit in orbit_entries:
                copies, representative_order, representative_runtime = (
                    orbit_entries[orbit]
                )
                # Evaluate each physical member through the exported entry,
                # independently of any additional internal runtime reuse.
                slots = dict(zip(representative_order, aligned_order, strict=True))
                member["runtime"] = {
                    **representative_runtime,
                    "permutation": [
                        slots[i] for i in representative_runtime["permutation"]
                    ],
                }
                for entry, relabel in copies:
                    pdg_index = _intern(
                        data["pdg_ids"], _relabelled(list(pdgs), relabel)
                    )
                    _add_terms(entry, [{"pdg_ids": pdg_index, "factor": 1.0}])
                copies[0][0]["members"].append(member)
                continue
            # The phase-space channels integrating the contribution: its own
            # colour order and its partners, in the physical legs.
            partners = _unique_partners(
                [(order, ())]
                if flow is None
                else _phasespace_partners(
                    [slot for slot in order if roles[slot] != 1],
                    [slot for slot in order if roles[slot] == 1],
                    roles,
                    configurations,
                ),
                incoming,
                leg_masses,
            )
            classes = _leg_classes(pdgs, roles)
            copies = []
            for index, (partner_order, configuration) in enumerate(partners):
                # One exported copy per channel, its final-state legs relabelled
                # so that the channel is the canonical one of its pattern; members
                # keep physical labels and are listed by the first copy only.
                pattern, rotated = _channel_pattern(
                    partner_order, incoming, leg_masses, classes
                )
                position = {slot: i for i, slot in enumerate(rotated)}
                channel_key = (
                    pattern,
                    tuple(
                        sorted(
                            (tuple(sorted(position[slot] for slot in legs)), abs(pdg))
                            for pdg, legs in configuration
                        )
                    ),
                )
                representative = channel_patterns.setdefault(channel_key, rotated)
                relabel = list(identity)
                for slot, target in zip(rotated, representative, strict=True):
                    relabel[slot] = target
                entry = _exported_entry(
                    data,
                    relabel,
                    member if index == 0 else None,
                    (pdgs, order, flow, runtime),
                    helicity_groups,
                    helicity_mappings,
                )
                # Its own channel first: the copy is weighted by its density
                # over those of all its partners, in the copy's legs.
                keys = [
                    _phasespace_key(
                        other_order,
                        other_configuration,
                        relabel,
                        incoming,
                        leg_masses,
                        phasespace_channels,
                    )
                    for other_order, other_configuration in [
                        partners[index],
                        *partners[:index],
                        *partners[index + 1 :],
                    ]
                ]
                # not followed by the channels: the launch rejects those that go
                # on shell
                colored_resonances = (
                    []
                    if flow is None
                    else [
                        {
                            "pdg": pdg,
                            "legs": sorted(relabel[slot] for slot in legs),
                            "color": color,
                        }
                        for pdg, legs, color in coloured
                    ]
                )
                selectors.append((entry, channel_key, keys, colored_resonances))
                copies.append((entry, relabel))
            orbit_entries[orbit] = (copies, aligned_order, runtime)
        unmapped += [
            (pdg, legs)
            for pdg, legs in mappable
            if legs not in line_legs and any(roles[slot] != 1 for slot in legs)
        ]
        if unmapped and process.color_accuracy == "lc":
            _LOGGER.warning(
                "%s: the [amplicol] phase space does not follow the s-channel "
                "resonances %s, which decay to coloured and colourless particles "
                "or not to one complete colour line",
                source_id,
                ", ".join(
                    f"{pdg} > {' '.join(str(pdgs[slot]) for slot in sorted(legs))}"
                    for pdg, legs in sorted(
                        unmapped, key=lambda r: (r[0], sorted(r[1]))
                    )
                ),
            )
    _group_phasespace_channels(data, selectors, phasespace_channels)
    _merge_flavour_orbits(data)
    _export_helicities(data, runtime_helicities)
    alpha_names = {p["alpha_s_parameter"] for p in data["runtime_processes"]}
    data["provider"]["alpha_s_parameter"] = (
        next(iter(alpha_names)) if len(alpha_names) == 1 else None
    )
    data["provider"]["supports_alpha_s"] = all(name is not None for name in alpha_names)
    data["provider"]["helicity_count"] = max(
        (len(p["helicity_ids"]) for p in data["runtime_processes"]), default=0
    )
    data["provider"]["color_count"] = max(1, len(data["color_flows"]))
    data["grouping"]["physical_contributions"] = physical_count
    data["grouping"]["exported_contributions"] = sum(
        len(c["processes"]) for c in data["channels"]
    )

    # Convert to plain lists; the interning indices are only needed while building.
    for name in ("pdg_ids", "color_orders", "color_flows", "helicities"):
        data[name] = list(data[name])
    return data


def _json_payload(path: str, value: Any) -> ApiBundlePayload:
    return ApiBundlePayload(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        "sdk-metadata",
        "application/json",
    )


def umami_bundle_payloads(
    *,
    processes: Sequence[UmamiProcessInput],
    compiled_model: Mapping[str, Any],
    artifact_id: str | None = None,
) -> tuple[ApiBundlePayload, ...]:
    """Create JSON, provider headers and Make variables from one representation."""
    from .umami_c import provider_header

    documents = build_umami_metadata(
        processes=processes,
        compiled_model=compiled_model,
        artifact_id=artifact_id,
    )
    payloads = []
    entries = []
    for data in documents:
        provider = data["provider"]
        identifier = provider["id"]
        relative = (
            "metadata.json"
            if len(documents) == 1
            else f"providers/{identifier}/metadata.json"
        )
        payloads.append(_json_payload(f"API/umami/{relative}", data))
        entries.append(
            {"id": identifier, "metadata": relative, "library": provider["library"]}
        )
        payloads.append(
            ApiBundlePayload(
                f"API/umami/provider_{identifier}.h",
                provider_header(data).encode(),
                "api-source",
                "text/x-chdr",
            )
        )
    if len(documents) != 1:
        payloads.append(
            _json_payload(
                "API/umami/metadata.json",
                {
                    "kind": "pyamplicol-umami-provider-index",
                    "schema_version": 1,
                    "providers": entries,
                },
            )
        )
    make = "UMAMI_PROVIDERS := " + " ".join(p["id"] for p in entries) + "\n"
    make += f"UMAMI_SINGLE_PROVIDER := {int(len(documents) == 1)}\n"
    payloads.append(
        ApiBundlePayload(
            "API/umami/providers.mk", make.encode(), "api-build-file", "text/x-makefile"
        )
    )
    return tuple(payloads)


__all__ = [
    "FLAVOR_BLIND_ASSUMPTION",
    "UmamiProcessInput",
    "build_umami_metadata",
    "umami_bundle_payloads",
]
