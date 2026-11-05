# SPDX-License-Identifier: 0BSD
"""Numerical discovery of process groups in a generated artifact.

Every stored representative is evaluated in f64 at three on-shell points. Each
point has its own deterministic variation of the free model inputs (external
and normalization parameters other than masses and widths, within
``PARAMETER_VARIATION``); a member and a representative are always compared
with the same values. A relation is
accepted when, through one leg mapping of
:func:`~pyamplicol.generation.process_grouping.candidate_mappings`, every
resolved helicity and colour component of the member is one common rational
factor times the matching component of the representative, at every point.

Helicity values map unchanged, also under charge conjugation. LC colour-flow
words are relabelled and, for conjugated mappings, reversed; they are compared
as sets of open fundamental chains or as one cyclic adjoint trace.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from ..api.errors import ArtifactError, PyAmpliColError
from ..api.results import ResolvedEvaluation
from ..artifacts.manifest import load_manifest
from ..artifacts.process_groups import (
    initial_process_groups,
    read_process_groups,
    write_process_groups,
)
from ..artifacts.security import confined_path
from .phase_space import massive_rambo_final_state
from .process_grouping import (
    GroupingAnalysis,
    GroupingBasis,
    GroupingProcess,
    GroupMember,
    LegMapping,
    ParticleTable,
    ProcessGroupMap,
    candidate_mappings,
    coarse_key,
    identity_permutation,
    invert_permutation,
    particle_table,
    rational_factor,
    representative_key,
    resolved_parameter_name,
)

POINTS = 3
PARAMETER_VARIATION = 0.1
TOLERANCE = 1.0e-9
DEFAULT_SEED = 20260917
_VARIED_KINDS = frozenset({"external", "normalization"})
# Components below _ZERO times the largest one at a point count as zero. The
# others must agree to TOLERANCE relative to themselves, but never more tightly
# than TOLERANCE times _FLOOR times the largest component.
_ZERO = 1.0e-12
_FLOOR = 1.0e-4
_MINIMUM_SQRT_S = 1000.0
_LOGGER = logging.getLogger("pyamplicol.generation")

FourMomentum = tuple[float, float, float, float]
Point = tuple[FourMomentum, ...]
ComponentKey = tuple[tuple[int, ...], Hashable]
Components = dict[ComponentKey, complex]


class _Unanalysable(Exception):
    """A representative that cannot take part in the comparison."""


# --------------------------------------------------------------------------
# Pure helpers


def parameter_variation(
    parameters: Sequence[Any],
    *,
    point: int,
    seed: int = DEFAULT_SEED,
    fixed: Collection[str] = frozenset(),
) -> dict[str, float | complex]:
    """Varied values of the free inputs among ``parameters`` at ``point``.

    ``parameters`` are ``ModelParameter``-like records; ``fixed`` names the
    mass and width parameters, which UFO models declare as external inputs.
    The factor of each parameter depends only on its name, the point and the
    seed, so every process sees the same values for the parameters it shares.
    """

    values: dict[str, float | complex] = {}
    for parameter in parameters:
        if (
            parameter.kind not in _VARIED_KINDS
            or not parameter.mutable
            or parameter.name in fixed
        ):
            continue
        digest = hashlib.sha256(f"{seed}:{point}:{parameter.name}".encode()).digest()
        uniform = int.from_bytes(digest[:8], "big") / float(1 << 64)
        factor = 1.0 + PARAMETER_VARIATION * (2.0 * uniform - 1.0)
        if parameter.default_imaginary:
            values[parameter.name] = factor * complex(
                parameter.default_real, parameter.default_imaginary
            )
        else:
            values[parameter.name] = factor * parameter.default_real
    return values


def on_shell_point(
    initial_masses: Sequence[float],
    final_masses: Sequence[float],
    *,
    seed: int,
) -> Point | None:
    """A deterministic on-shell point in the centre-of-mass frame.

    Beams may be massive; a single initial particle decays at rest. Returns
    ``None`` when the masses leave no phase space.
    """

    final_masses = tuple(final_masses)
    if len(initial_masses) == 1:
        (mass,) = initial_masses
        if len(final_masses) < 2 or mass <= sum(final_masses):
            return None
        return ((mass, 0.0, 0.0, 0.0), *_final_state(final_masses, mass, seed))
    first, second = initial_masses
    if len(final_masses) == 1:
        sqrt_s = final_masses[0]
        if sqrt_s <= first + second:
            return None
        final: tuple[FourMomentum, ...] = ((sqrt_s, 0.0, 0.0, 0.0),)
    else:
        threshold = max(first + second, sum(final_masses))
        sqrt_s = max(_MINIMUM_SQRT_S, 2.0 * threshold)
        final = _final_state(final_masses, sqrt_s, seed)
    s = sqrt_s * sqrt_s
    first_energy = (s + first * first - second * second) / (2.0 * sqrt_s)
    momentum = math.sqrt(max(0.0, first_energy * first_energy - first * first))
    return (
        (first_energy, 0.0, 0.0, momentum),
        (sqrt_s - first_energy, 0.0, 0.0, -momentum),
        *final,
    )


def _final_state(
    masses: Sequence[float],
    sqrt_s: float,
    seed: int,
) -> tuple[FourMomentum, ...]:
    return massive_rambo_final_state(
        len(masses), sqrt_s=sqrt_s, masses=masses, seed=seed
    )


def _point_seed(seed: int, point: int) -> int:
    digest = hashlib.sha256(f"{seed}:{point}:phase-space".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def member_point(representative: Point, permutation: Sequence[int]) -> Point:
    """Momenta of the member such that ``q[r] = p[permutation[r]]``."""

    member: list[FourMomentum] = [representative[0]] * len(representative)
    for representative_index, member_index in enumerate(permutation):
        member[member_index] = representative[representative_index]
    return tuple(member)


def color_roles(
    pdgs: Sequence[int],
    initial_count: int,
    colors: Mapping[int, int],
) -> tuple[int, ...]:
    """Colour representation of each leg with all legs outgoing."""

    roles = []
    for index, pdg in enumerate(pdgs):
        color = colors.get(pdg)
        if color is None:
            color = -colors[-pdg] if abs(colors[-pdg]) == 3 else colors[-pdg]
        if index < initial_count and abs(color) == 3:
            color = -color
        roles.append(color)
    return tuple(roles)


def flow_key(
    word: Sequence[int],
    roles: Sequence[int],
    relabel: Sequence[int],
    *,
    reverse: bool,
) -> Hashable:
    """Canonical colour tensor of an LC flow word in target leg indices.

    ``word`` holds zero-based source leg indices, ``roles`` their all-outgoing
    representations and ``relabel`` maps source to target indices. Open lines
    run from a fundamental through adjoints to an antifundamental and are
    compared as a set; a pure-adjoint word is one trace, compared up to
    rotation. ``reverse`` reverses every line and trace (charge conjugation).
    A word that is neither is kept verbatim.
    """

    chains: list[tuple[int, ...]] = []
    trace: list[int] = []
    current: list[int] | None = None
    decodable = True
    for index in word:
        role = roles[index]
        if role == 3 and current is None:
            current = [index]
        elif role == -3 and current is not None:
            current.append(index)
            chains.append(tuple(current))
            current = None
        elif role == 8:
            (trace if current is None else current).append(index)
        elif role != 1:
            decodable = False

    def mapped(line: Sequence[int]) -> tuple[int, ...]:
        result = tuple(relabel[index] for index in line)
        return result[::-1] if reverse else result

    if not decodable or current is not None or (chains and trace):
        return ("word", mapped(word))
    if chains:
        return ("lines", tuple(sorted(mapped(chain) for chain in chains)))
    cycle = mapped(trace)
    if not cycle:
        return ("singlet",)
    start = cycle.index(min(cycle))
    return ("trace", cycle[start:] + cycle[:start])


def resolved_components(
    resolved: ResolvedEvaluation,
    point: int,
    *,
    helicities: Mapping[str, Sequence[int]],
    words: Mapping[str, Sequence[int]],
    roles: Sequence[int],
    mapping: LegMapping | None,
) -> Components:
    """One point of a member's resolved values keyed in representative legs.

    ``helicities`` and ``words`` give the member's helicity values and LC
    flow words (zero-based leg indices) by ID; ``mapping`` is the candidate
    relation to the representative, ``None`` for the identity.
    """

    count = len(roles)
    permutation = (
        identity_permutation(count) if mapping is None else mapping.permutation
    )
    relabel = invert_permutation(permutation)
    conjugated = mapping is not None and mapping.conjugated
    helicity_keys = [
        tuple(helicities[identifier][index] for index in permutation)
        for identifier in resolved.helicity_ids
    ]
    color_keys = [
        flow_key(words[identifier], roles, relabel, reverse=conjugated)
        if resolved.color_accuracy == "lc"
        else identifier
        for identifier in resolved.color_ids
    ]
    return {
        (helicity, color): complex(cast(complex, value))
        for helicity, row in zip(helicity_keys, resolved.values[point], strict=True)
        for color, value in zip(color_keys, row, strict=True)
    }


def common_factor(
    member: Sequence[Components],
    representative: Sequence[Components],
    *,
    tolerance: float = TOLERANCE,
) -> complex | None:
    """The common ratio ``member / representative`` of all components.

    Both sequences hold one component table per point. Components that are
    zero on one side must be zero on the other; a point that vanishes
    entirely on either side gives no ratio.
    """

    factor: complex | None = None
    for member_point_values, representative_point_values in zip(
        member, representative, strict=True
    ):
        ratio = _point_factor(
            member_point_values, representative_point_values, tolerance
        )
        if ratio is None:
            return None
        if factor is None:
            factor = ratio
        elif abs(ratio - factor) > tolerance * abs(factor):
            return None
    return factor


def _point_factor(
    member: Components,
    representative: Components,
    tolerance: float,
) -> complex | None:
    member_scale = max(map(abs, member.values()), default=0.0)
    representative_scale = max(map(abs, representative.values()), default=0.0)
    if member_scale == 0.0 or representative_scale == 0.0:
        return None
    pivot = max(representative, key=lambda key: abs(representative[key]))
    ratio = member.get(pivot, 0j) / representative[pivot]
    if ratio == 0:
        return None
    for key in member.keys() | representative.keys():
        member_value = member.get(key, 0j)
        representative_value = representative.get(key, 0j)
        member_zero = abs(member_value) <= _ZERO * member_scale
        representative_zero = abs(representative_value) <= _ZERO * representative_scale
        if member_zero or representative_zero:
            if member_zero != representative_zero:
                return None
            continue
        allowed = tolerance * max(abs(member_value), _FLOOR * member_scale)
        if abs(member_value - ratio * representative_value) > allowed:
            return None
    return ratio


def accepted_factor(
    value: complex | None,
    *,
    only_unit_multiplier: bool,
    tolerance: float = TOLERANCE,
) -> Fraction | None:
    """The recordable rational factor of a numerical ratio, if any."""

    if value is None or abs(value.imag) > tolerance * abs(value):
        return None
    factor = rational_factor(value.real, tolerance=tolerance)
    if factor is None or (only_unit_multiplier and factor != 1):
        return None
    return factor


# --------------------------------------------------------------------------
# Artifact model and runtimes


@dataclass(frozen=True, slots=True)
class _Model:
    table: ParticleTable
    colors: Mapping[int, int]
    mass_names: Mapping[int, str]
    fixed: frozenset[str]
    defaults: Mapping[str, float]
    complex_externals: frozenset[str]


def _artifact_model(artifact: Path) -> _Model:
    manifest = load_manifest(artifact)
    records = [item for item in manifest.payloads if item.role == "compiled-model"]
    if len(records) != 1:
        raise ArtifactError("process grouping requires one compiled-model payload")
    ir = json.loads(
        confined_path(manifest.root, records[0].path).read_text(encoding="utf-8")
    )["ir"]
    parameters = {
        str(record["name"]): SimpleNamespace(**record) for record in ir["parameters"]
    }
    particles = [SimpleNamespace(**record) for record in ir["particles"]]
    return _Model(
        table=particle_table(particles, parameters),
        colors={int(record.pdg_code): int(record.color) for record in particles},
        mass_names={
            int(record.pdg_code): resolved_parameter_name(str(record.mass), parameters)
            for record in particles
        },
        fixed=frozenset(
            str(name) for record in particles for name in (record.mass, record.width)
        ),
        defaults={
            name: float((record.value or (0.0, 0.0))[0])
            for name, record in parameters.items()
        },
        complex_externals=frozenset(
            name
            for name, record in parameters.items()
            if record.nature == "external" and record.parameter_type == "complex"
        ),
    )


@dataclass(slots=True)
class _Reference:
    """A representative's own points and resolved components."""

    process: GroupingProcess
    accuracy: str
    points: tuple[Point, ...]
    components: tuple[Components, ...]


class _Probe:
    """One loaded process and the component view of its evaluations."""

    def __init__(
        self,
        artifact: Path,
        process: GroupingProcess,
        model: _Model,
        seed: int,
    ) -> None:
        from ..api import Runtime

        self.process = process
        self._model = model
        self._seed = seed
        self.runtime = Runtime.load(
            artifact, process=process.process_id, mute_warnings=True
        )
        physics = self.runtime.physics
        if physics.helicity_coverage != "complete":
            raise _Unanalysable("helicity coverage is selected")
        if physics.color_coverage == "selected":
            raise _Unanalysable("colour coverage is selected")
        pdgs = tuple(particle.pdg_id for particle in physics.external_particles)
        if pdgs != process.pdgs:
            raise _Unanalysable("runtime legs differ from the process map")
        self.accuracy = physics.color_accuracy
        self.parameters = physics.model_parameters
        # Masses are never varied: only derived ones need the runtime state.
        derived = {item.name for item in self.parameters if item.kind == "derived"}
        self._slots = (
            self._parameter_slots()
            if any(model.mass_names[pdg] in derived for pdg in pdgs)
            else None
        )
        self._helicities = {item.id: item.values for item in physics.helicities}
        index_of_label = {
            particle.label: particle.index for particle in physics.external_particles
        }
        self._words = {
            item.id: tuple(index_of_label[label] for label in item.word)
            for item in physics.color_flows
        }
        self._roles = color_roles(pdgs, process.initial_count, model.colors)

    def _parameter_slots(self) -> dict[str, int] | None:
        """Index of each real parameter in the runtime's current values.

        Derived and complex external parameters occupy a real and an
        imaginary slot. ``None`` when the runtime does not expose them.
        """

        values = self._current_values()
        if values is None:
            return None
        slots: dict[str, int] = {}
        defaults: list[float] = []
        for parameter in self.parameters:
            slots[parameter.name] = len(defaults)
            defaults.append(parameter.default_real)
            if (
                parameter.kind == "derived"
                or parameter.name in self._model.complex_externals
            ):
                defaults.append(parameter.default_imaginary)
        if len(defaults) != len(values) or not all(
            math.isclose(value, default, rel_tol=1e-9, abs_tol=1e-12)
            for value, default in zip(values, defaults, strict=True)
        ):
            raise _Unanalysable("runtime model-parameter layout is not recognised")
        return slots

    def _current_values(self) -> tuple[float, ...] | None:
        native = getattr(getattr(self.runtime, "_backend", None), "_runtime", None)
        if native is None:
            return None
        from ..runtime._native_selection import exact_runtime_state_payload

        values = exact_runtime_state_payload(native)["model_parameter_values"]
        return tuple(float(value) for value in cast(Sequence[float], values))

    def apply(self, point: int) -> None:
        values = parameter_variation(
            self.parameters, point=point, seed=self._seed, fixed=self._model.fixed
        )
        if values:
            self.runtime.set_model_parameters(values)

    def masses(self) -> tuple[float, ...]:
        """Current mass of every leg, derived ones read back from the runtime."""

        current = self._current_values() if self._slots is not None else None
        if current is not None and self._slots is not None:
            known = {name: current[slot] for name, slot in self._slots.items()}
        else:
            known = {
                parameter.name: parameter.default_real for parameter in self.parameters
            }
        masses = []
        for pdg in self.process.pdgs:
            name = self._model.mass_names[pdg]
            if name == "ZERO":
                masses.append(0.0)
                continue
            for candidate in (name, f"particle.{abs(pdg)}.mass"):
                if candidate in known:
                    masses.append(abs(known[candidate]))
                    break
            else:
                masses.append(abs(self._model.defaults.get(name, 0.0)))
        return tuple(masses)

    def reference(self) -> _Reference:
        points = []
        components = []
        count = self.process.initial_count
        for point in range(POINTS):
            self.apply(point)
            masses = self.masses()
            momenta = on_shell_point(
                masses[:count], masses[count:], seed=_point_seed(self._seed, point)
            )
            if momenta is None:
                raise _Unanalysable("no on-shell phase space for the leg masses")
            resolved = self.runtime.evaluate_resolved([momenta])
            points.append(momenta)
            components.append(self.components(resolved, 0, None))
        return _Reference(self.process, self.accuracy, tuple(points), tuple(components))

    def components(
        self,
        resolved: ResolvedEvaluation,
        point: int,
        mapping: LegMapping | None,
    ) -> Components:
        return resolved_components(
            resolved,
            point,
            helicities=self._helicities,
            words=self._words,
            roles=self._roles,
            mapping=mapping,
        )

    def relation(
        self,
        reference: _Reference,
        mappings: Sequence[LegMapping],
        *,
        only_unit_multiplier: bool,
    ) -> tuple[LegMapping, Fraction] | None:
        """The first mapping relating this process to ``reference``.

        Each point is evaluated for all mappings still consistent with the
        previous points, in one batch.
        """

        alive = list(mappings)
        evaluated: dict[LegMapping, list[Components]] = {
            mapping: [] for mapping in alive
        }
        for point in range(POINTS):
            if not alive:
                return None
            self.apply(point)
            resolved = self.runtime.evaluate_resolved(
                [
                    member_point(reference.points[point], mapping.permutation)
                    for mapping in alive
                ]
            )
            survivors = []
            for index, mapping in enumerate(alive):
                evaluated[mapping].append(self.components(resolved, index, mapping))
                if (
                    common_factor(evaluated[mapping], reference.components[: point + 1])
                    is not None
                ):
                    survivors.append(mapping)
            alive = survivors
        for mapping in alive:
            factor = accepted_factor(
                common_factor(evaluated[mapping], reference.components),
                only_unit_multiplier=only_unit_multiplier,
            )
            if factor is not None:
                return mapping, factor
        return None


# --------------------------------------------------------------------------
# Analysis and storage


def analyse_process_groups(
    artifact: str | Path,
    group_map: ProcessGroupMap,
    *,
    only_unit_multiplier: bool = False,
    consider_charge_conjugation: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Merge numerically equal representatives of ``group_map`` in place.

    Representatives are visited in canonical order and compared with the
    earlier ones that kept their group; the canonically smaller one stays the
    representative of a merged group.
    """

    root = Path(artifact).expanduser().resolve(strict=True)
    model = _artifact_model(root)
    processes = sorted(
        (member.process for member in group_map.representatives if member.generated),
        key=lambda process: representative_key(process, model.table),
    )
    kept: list[_Reference] = []
    relations: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for process in processes:
        try:
            probe = _Probe(root, process, model, seed)
            key = coarse_key(process, model.table)
            found = None
            for reference in kept:
                if (
                    reference.accuracy != probe.accuracy
                    or coarse_key(reference.process, model.table) != key
                ):
                    continue
                mappings = candidate_mappings(
                    process,
                    reference.process,
                    model.table,
                    consider_charge_conjugation=consider_charge_conjugation,
                )
                result = probe.relation(
                    reference,
                    mappings,
                    only_unit_multiplier=only_unit_multiplier,
                )
                if result is not None:
                    found = reference, *result
                    break
            if found is None:
                kept.append(probe.reference())
                continue
        except (_Unanalysable, PyAmpliColError) as error:
            _LOGGER.warning("process grouping skips %s: %s", process.process_id, error)
            skipped.append({"process_id": process.process_id, "reason": str(error)})
            continue
        reference, mapping, factor = found
        relation = GroupMember.identity(process).related(
            mapping, factor=factor, basis=GroupingBasis.NUMERICAL
        )
        group_map.merge(reference.process.process_id, relation, model.table)
        relations.append(
            {
                "process_id": process.process_id,
                "process": process.expression,
                "representative": reference.process.process_id,
                "factor": str(factor),
                "permutation": list(mapping.permutation),
                "conjugated": mapping.conjugated,
            }
        )
    return {
        "processes_analysed": len(processes) - len(skipped),
        "new_relations": relations,
        "skipped": skipped,
    }


def _analysis_record(
    *,
    only_unit_multiplier: bool,
    consider_charge_conjugation: bool,
    seed: int,
) -> dict[str, object]:
    return {
        "variant": str(GroupingAnalysis.NUMERICAL),
        "consider_charge_conjugation": consider_charge_conjugation,
        "only_unit_multiplier": only_unit_multiplier,
        "points": POINTS,
        "parameter_variation": PARAMETER_VARIATION,
        "tolerance": TOLERANCE,
        "seed": seed,
    }


def update_artifact_process_groups(
    artifact: str | Path,
    *,
    analysis: GroupingAnalysis | str,
    only_unit_multiplier: bool,
    consider_charge_conjugation: bool,
) -> dict[str, object]:
    """Analyse an artifact's representatives and rewrite its group map.

    ``"none"`` only writes the trivial map when the artifact has none. The
    numerical analysis merges new relations into the stored map; the artifact
    identity is unchanged. Returns a summary of the update.
    """

    analysis = GroupingAnalysis(analysis)
    root = Path(artifact).expanduser().resolve(strict=True)
    group_map = read_process_groups(root)
    changed = group_map is None
    if group_map is None:
        group_map = initial_process_groups(
            (
                str(process["id"]),
                str(process["expression"]),
                cast(Sequence[int], process["external_pdgs"]),
            )
            for process in load_manifest(root).processes
        )
    summary: dict[str, object] = {
        "artifact": str(root),
        "analysis": str(analysis),
        "groups_before": len(group_map.groups),
        "processes_analysed": 0,
        "new_relations": [],
        "skipped": [],
    }
    if analysis is GroupingAnalysis.NUMERICAL:
        summary.update(
            analyse_process_groups(
                root,
                group_map,
                only_unit_multiplier=only_unit_multiplier,
                consider_charge_conjugation=consider_charge_conjugation,
            )
        )
        record = _analysis_record(
            only_unit_multiplier=only_unit_multiplier,
            consider_charge_conjugation=consider_charge_conjugation,
            seed=DEFAULT_SEED,
        )
        if record not in group_map.analyses:
            group_map.analyses.append(record)
            changed = True
        changed = changed or bool(summary["new_relations"])
    if changed:
        write_process_groups(root, group_map)
    summary["groups_after"] = len(group_map.groups)
    return summary


__all__ = [
    "DEFAULT_SEED",
    "PARAMETER_VARIATION",
    "POINTS",
    "TOLERANCE",
    "accepted_factor",
    "analyse_process_groups",
    "color_roles",
    "common_factor",
    "flow_key",
    "member_point",
    "on_shell_point",
    "parameter_variation",
    "resolved_components",
    "update_artifact_process_groups",
]
