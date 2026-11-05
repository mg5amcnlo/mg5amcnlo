# SPDX-License-Identifier: 0BSD
"""Isolated, unreduced reference amplitudes for declared Born correlations.

The caller authenticates the correlation declaration and its generation policy.
This executor intentionally does not change any ordinary execution path. It
accepts only materialized, unreplayed compiled plans with complete source slots.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from typing import Any, cast

from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.api.protocols import Momenta
from pyamplicol.runtime._normalization_exact import exact_normalization
from pyamplicol.runtime.symbolica_exact import (
    SymbolicaExactExecutor,
    _assign_stage_outputs,
    _canonical_amplitude_outputs,
    _complex_mul,
    _complex_zero,
    _ComplexDecimal,
    _crossing_phase,
    _decimal,
    _fill_momenta,
    _json_integer,
    _pack_stage_inputs,
    _prepare_points,
    _runtime_state,
    _source_wavefunction,
    _validated_source_ir,
    _working_precision,
)


@dataclass(frozen=True, slots=True)
class CoherentAmplitudeGroup:
    """One physical-helicity/color-basis coefficient, before any metric."""

    group_id: int
    helicity_id: str
    helicities: tuple[int, ...]
    color_word: tuple[int, ...]
    color_sector_id: int


@dataclass(frozen=True, slots=True)
class CoherentAmplitudeBatch:
    """Complex amplitudes in ``(point, physical group, real/imag)`` order.

    Root colour/coupling coefficients are already included; folded helicity
    and squared-colour weights are not. Distinct raw groups contributing to
    the same physical helicity/colour sector have been summed coherently.
    ``normalization_factor`` belongs to the final bilinear, not each amplitude.
    ``spin_correlated_legs`` are one-based public legs replaced in this call;
    their entries in each group's helicity vector are placeholder labels only.
    """

    values: tuple[tuple[_ComplexDecimal, ...], ...]
    groups: tuple[CoherentAmplitudeGroup, ...]
    normalization_factor: Decimal
    precision: int
    spin_correlated_legs: tuple[int, ...]


_StageMemo = dict[int, tuple[tuple[_ComplexDecimal, ...], tuple[_ComplexDecimal, ...]]]


def _identical_stage_inputs(
    left: tuple[_ComplexDecimal, ...], right: tuple[_ComplexDecimal, ...]
) -> bool:
    """Conservative exact equality, including signed zero and decimal precision."""
    return len(left) == len(right) and all(
        a.as_tuple() == b.as_tuple()
        for x, y in zip(left, right, strict=True)
        for a, b in zip(x, y, strict=True)
    )


def _record(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"correlated {context} must be an object")
    return value


def _records(value: object, context: str) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ArtifactError(f"correlated {context} must be a sequence")
    return tuple(_record(item, context) for item in value)


def _complex_component(value: object) -> _ComplexDecimal:
    if isinstance(value, bool):
        raise EvaluationError("spin-vector components must be numeric scalars")
    if isinstance(value, complex):
        return (
            _decimal(value.real, "spin-vector real component"),
            _decimal(value.imag, "spin-vector imaginary component"),
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if len(value) != 2:
            raise EvaluationError(
                "a complex spin component must be a (real, imaginary) pair"
            )
        return (
            _decimal(value[0], "spin-vector real component"),
            _decimal(value[1], "spin-vector imaginary component"),
        )
    return (_decimal(value, "spin-vector component"), Decimal(0))


def _is_spin_vector(value: Sequence[object]) -> bool:
    """Distinguish four complex scalars from a per-point vector batch."""

    def scalar(item: object) -> bool:
        return not isinstance(item, Sequence) or isinstance(item, str | bytes)

    return len(value) == 4 and all(
        scalar(item)
        or (
            isinstance(item, Sequence)
            and len(item) == 2
            and all(scalar(part) for part in item)
        )
        for item in value
    )


def _vector(value: object) -> tuple[_ComplexDecimal, ...]:
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) != 4
    ):
        raise EvaluationError("a spin vector must contain four complex components")
    return tuple(_complex_component(component) for component in value)


def _prepare_spin_vectors(
    spin_vectors: Mapping[int, object] | None,
    *,
    allowed_legs: Sequence[int],
    point_count: int,
) -> tuple[dict[int, tuple[_ComplexDecimal, ...]], ...]:
    """Copy caller data; no vectors or mutable input buffers survive the call."""

    if spin_vectors is None:
        spin_vectors = {}
    if not isinstance(spin_vectors, Mapping):
        raise EvaluationError("spin_vectors must map public leg labels to vectors")
    result: tuple[dict[int, tuple[_ComplexDecimal, ...]], ...] = tuple(
        {} for _ in range(point_count)
    )
    for leg, raw in spin_vectors.items():
        if isinstance(leg, bool) or not isinstance(leg, int) or leg not in allowed_legs:
            raise EvaluationError(
                f"spin-vector leg {leg!r} was not declared at generation"
            )
        if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
            raise EvaluationError(
                "spin vectors must be complex four-vectors or a batch"
            )
        # A four-scalar vector broadcasts; a sequence of vectors is point-specific.
        broadcast = _is_spin_vector(raw)
        if broadcast:
            prepared = (_vector(raw),) * point_count
        else:
            if len(raw) != point_count:
                raise EvaluationError(
                    f"spin-vector batch has {len(raw)} points, expected {point_count}"
                )
            prepared = tuple(_vector(item) for item in raw)
        for point, wave in zip(result, prepared, strict=True):
            point[leg] = wave
    return result


def _fill_correlated_sources(
    state: list[_ComplexDecimal],
    point: Sequence[tuple[Decimal, Decimal, Decimal, Decimal]],
    schema: Mapping[str, object],
    model_parameters: Sequence[Decimal],
    spin_vectors: Mapping[int, tuple[_ComplexDecimal, ...]],
    *,
    scalar: type[Decimal] | None = None,
) -> None:
    scalar = Decimal if scalar is None else scalar
    source_fill = _record(schema.get("source_fill"), "source fill")
    seen: set[int] = set()
    for source in _records(source_fill.get("sources"), "sources"):
        leg = _json_integer(source["leg_label"])
        slot = _record(source.get("value_slot"), "source value slot")
        start = _json_integer(slot["component_start"])
        stop = _json_integer(slot["component_stop"])
        if not 0 <= start < stop <= len(state):
            raise ArtifactError("correlated source value slot is out of range")
        if leg in spin_vectors:
            source_ir, _identity, crossing = _validated_source_ir(source)
            if source_ir["wavefunction_family"] != "vector" or (
                _json_integer(source_ir["component_dimension"]) != 4
                or stop - start != 4
            ):
                raise CompatibilityError(
                    "spin correlation requires a full four-component vector source"
                )
            # Literal input, including its temporal component. In particular,
            # v=p remains a nonzero leaf, so a Ward test exercises the amplitude.
            # The public vector is not negated, projected, normalized or conjugated;
            # SourceIR's ordinary amplitude crossing phase is applied once.
            phase = cast(
                _ComplexDecimal,
                tuple(scalar(value) for value in _crossing_phase(crossing)),
            )
            wave = tuple(
                _complex_mul((scalar(component[0]), scalar(component[1])), phase)
                for component in spin_vectors[leg]
            )
            seen.add(leg)
        else:
            wave = _source_wavefunction(
                source, point, schema, model_parameters, scalar=scalar
            )
        if len(wave) != stop - start:
            raise ArtifactError(
                "correlated source wavefunction does not match its value slot"
            )
        state[start:stop] = wave
    if seen != set(spin_vectors):
        raise ArtifactError("a declared spin-vector leg has no retained source")


class CorrelatedExactExecutor(SymbolicaExactExecutor):
    """Reference executor for generation-declared, uncompressed correlations.

    Each call owns its vector data. Passing ``None`` or ``{}`` restores ordinary
    helicity sources; no global patching or cross-instance source state is used.
    """

    _scalar: type[Decimal] | None = None

    def __init__(
        self,
        artifact: Path,
        process_id: str,
        native_runtime: Any,
        *,
        spin_correlated_legs: Sequence[int] = (),
        coherent_groups: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        super().__init__(artifact, process_id, native_runtime)
        self._initialize_correlated_plan(spin_correlated_legs, coherent_groups)

    def _initialize_correlated_plan(
        self,
        spin_correlated_legs: Sequence[int],
        coherent_groups: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        particles = _records(
            self._physics.get("external_particles"), "external particles"
        )
        external_count = len(particles)
        if self._permutation not in (None, tuple(range(external_count))):
            raise CompatibilityError(
                "correlated reference execution does not support process aliases"
            )
        if any(
            getattr(self, name, None) is not None
            for name in (
                "_lc_replay",
                "_color_replay",
                "_helicity_plan",
                "_helicity_sum_execution",
            )
        ):
            raise CompatibilityError(
                "correlated reference execution requires an unreplayed, "
                "unquotiented plan"
            )
        legs = tuple(spin_correlated_legs)
        if any(
            isinstance(leg, bool)
            or not isinstance(leg, int)
            or not 1 <= leg <= external_count
            for leg in legs
        ) or len(set(legs)) != len(legs):
            raise EvaluationError(
                "declared spin-correlated legs must be unique public leg labels"
            )
        self._spin_correlated_legs = tuple(sorted(legs))
        schema = _record(self._execution.get("runtime_schema"), "runtime schema")
        amplitude = _record(schema.get("amplitude_stage"), "amplitude stage")
        if (
            amplitude.get("color_topology_replay") is not None
            or self._execution.get("lc_topology_replay") is not None
        ):
            raise CompatibilityError(
                "correlated reference execution cannot use color replay"
            )
        helicity_records = _records(self._physics.get("helicities"), "helicities")
        helicity_by_values: dict[tuple[int, ...], str] = {}
        computed_helicities: set[str] = set()
        for record in helicity_records:
            identifier = str(record["id"])
            if record.get("representative_id", identifier) != identifier:
                raise CompatibilityError(
                    "correlated reference execution cannot use helicity aliases"
                )
            computed = record.get("computed")
            structural_zero = record.get("structural_zero") is True
            if (computed is True and structural_zero) or (
                computed is not True and not (computed is False and structural_zero)
            ):
                raise CompatibilityError(
                    "correlated reference execution cannot use zero-pruned "
                    "or incomplete helicities"
                )
            expected_coefficient = 1 if computed is True else 0
            if (
                _decimal(record.get("coefficient", 1), "helicity coefficient")
                != expected_coefficient
            ):
                raise CompatibilityError(
                    "correlated reference execution requires unit computed-helicity "
                    "weights and zero structural-zero weights"
                )
            values = tuple(
                _json_integer(value)
                for value in cast(Sequence[object], record["values"])
            )
            if (
                len(values) != external_count
                or values in helicity_by_values
                or identifier in helicity_by_values.values()
            ):
                raise ArtifactError(
                    "correlated helicity metadata is incomplete or repeated"
                )
            helicity_by_values[values] = identifier
            if computed is True:
                computed_helicities.add(identifier)
            # The caller authenticates generation without source-dependent
            # zero pruning. A remaining uncomputed structural zero is therefore
            # intrinsic (e.g. massless-fermion chirality), even for literal
            # vector replacements. Retain its public selector as a zero state.
        roots = _records(amplitude.get("roots"), "amplitude roots")
        root_indices: dict[int, list[int]] = {}
        outputs: set[int] = set()
        for root in roots:
            group_id = _json_integer(root["coherent_group_id"])
            index = _json_integer(root["output_index"])
            if index in outputs:
                raise ArtifactError("correlated amplitude output index is repeated")
            outputs.add(index)
            root_indices.setdefault(group_id, []).append(index)
            if any(
                _decimal(root.get(key, 1), key) != 1
                for key in ("helicity_weight", "all_sector_weight")
            ):
                raise CompatibilityError(
                    "correlated reference execution cannot use folded amplitude weights"
                )
        if outputs != set(range(len(roots))):
            raise ArtifactError("correlated amplitude output mapping is incomplete")
        groups: list[CoherentAmplitudeGroup] = []
        seen_groups: set[int] = set()
        physical_groups: dict[tuple[tuple[int, ...], int], CoherentAmplitudeGroup] = {}
        physical_roots: dict[int, list[int]] = {}
        descriptors = (
            amplitude.get("coherent_groups")
            if coherent_groups is None
            else coherent_groups
        )
        for descriptor in _records(descriptors, "coherent groups"):
            group_id = _json_integer(descriptor["group_id"])
            values = tuple(
                _json_integer(value)
                for value in cast(Sequence[object], descriptor["helicities"])
            )
            if (
                group_id in seen_groups
                or group_id not in root_indices
                or values not in helicity_by_values
                or helicity_by_values[values] not in computed_helicities
            ):
                raise ArtifactError(
                    "correlated coherent-group mapping is incomplete or repeated"
                )
            seen_groups.add(group_id)
            word = tuple(
                _json_integer(value)
                for value in cast(Sequence[object], descriptor["color_word"])
            )
            sector = _json_integer(descriptor["color_sector_id"])
            key = (values, sector)
            physical = physical_groups.get(key)
            if physical is None:
                physical = CoherentAmplitudeGroup(
                    group_id=group_id,
                    helicity_id=helicity_by_values[values],
                    helicities=values,
                    color_word=word,
                    color_sector_id=sector,
                )
                physical_groups[key] = physical
                groups.append(physical)
            elif physical.color_word != word:
                raise ArtifactError(
                    "correlated coherent groups disagree on a physical color sector"
                )
            # Generation may partition one physical coefficient by basis_key
            # or root.color_weight. Compiled root outputs already contain those
            # weights, so combine their complex values without another factor.
            physical_roots.setdefault(physical.group_id, []).extend(
                root_indices[group_id]
            )
        if (
            seen_groups != set(root_indices)
            or {group.helicity_id for group in groups} != computed_helicities
        ):
            raise ArtifactError(
                "correlated coherent groups do not cover "
                "the retained physical helicities"
            )
        color_basis: dict[str, set[tuple[int, tuple[int, ...]]]] = {}
        for group in groups:
            basis = color_basis.setdefault(group.helicity_id, set())
            color = (group.color_sector_id, group.color_word)
            basis.add(color)
        if color_basis and any(
            basis != next(iter(color_basis.values())) for basis in color_basis.values()
        ):
            raise ArtifactError(
                "correlated coherent groups have incomplete helicity/color coverage"
            )
        self._coherent_groups = tuple(groups)
        self._known_helicity_ids = frozenset(helicity_by_values.values())
        self._coherent_root_indices = {
            key: tuple(value) for key, value in physical_roots.items()
        }
        self._raw_amplitude_count = len(roots)
        sources = _records(
            _record(schema.get("source_fill"), "source fill").get("sources"), "sources"
        )
        for leg in self._spin_correlated_legs:
            matching = [source for source in sources if source.get("leg_label") == leg]
            if not matching:
                raise ArtifactError(f"spin-correlated leg {leg} has no retained source")
            for source in matching:
                ir, _identity, _crossing = _validated_source_ir(source)
                slot = _record(source.get("value_slot"), "source value slot")
                if (
                    ir["wavefunction_family"] != "vector"
                    or _json_integer(ir["component_dimension"]) != 4
                    or _json_integer(slot["component_stop"])
                    - _json_integer(slot["component_start"])
                    != 4
                ):
                    raise CompatibilityError(
                        "spin correlation requires a full four-component vector source"
                    )

    def _selected_groups(
        self, helicities: Sequence[str] | None, active_legs: tuple[int, ...]
    ) -> tuple[CoherentAmplitudeGroup, ...]:
        known_helicities = self._known_helicity_ids
        selected = known_helicities if helicities is None else set(helicities)
        if not selected <= known_helicities:
            raise EvaluationError(
                "unknown correlated helicity selector: "
                f"{sorted(selected - known_helicities)!r}"
            )
        # Keep one placeholder helicity for each replaced leg, consistently
        # across the color basis. Spectator helicities remain incoherently summed.
        representatives: dict[tuple[int, ...], tuple[int, ...]] = {}
        for group in self._coherent_groups:
            if group.helicity_id in selected:
                spectator = tuple(
                    value
                    for leg, value in enumerate(group.helicities, 1)
                    if leg not in active_legs
                )
                representatives[spectator] = min(
                    representatives.get(spectator, group.helicities), group.helicities
                )
        return tuple(
            group
            for group in self._coherent_groups
            if group.helicity_id in selected
            and group.helicities
            == representatives[
                tuple(
                    value
                    for leg, value in enumerate(group.helicities, 1)
                    if leg not in active_legs
                )
            ]
        )

    def _coherent_values(
        self,
        raw: tuple[_ComplexDecimal, ...],
        groups: tuple[CoherentAmplitudeGroup, ...],
    ) -> tuple[_ComplexDecimal, ...]:
        if len(raw) != self._raw_amplitude_count:
            raise ArtifactError(
                "correlated evaluator returned the wrong amplitude count"
            )
        return tuple(
            (
                sum(
                    (
                        raw[index][0]
                        for index in self._coherent_root_indices[group.group_id]
                    ),
                    Decimal(0),
                ),
                sum(
                    (
                        raw[index][1]
                        for index in self._coherent_root_indices[group.group_id]
                    ),
                    Decimal(0),
                ),
            )
            for group in groups
        )

    def coherent_amplitudes(
        self,
        momenta: Momenta,
        *,
        spin_vectors: Mapping[int, object] | None = None,
        precision: int = 40,
        helicities: Sequence[str] | None = None,
    ) -> CoherentAmplitudeBatch:
        if (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision < 1
        ):
            raise EvaluationError(
                "precision must be a positive number of decimal digits"
            )
        points = _prepare_points(momenta, self._physics, None)
        vectors = _prepare_spin_vectors(
            spin_vectors,
            allowed_legs=self._spin_correlated_legs,
            point_count=len(points),
        )
        active_legs = tuple(sorted(vectors[0]))
        groups = self._selected_groups(helicities, active_legs)
        working_precision = _working_precision(precision)
        payload = _runtime_state(self._native_runtime)
        parameters = tuple(
            _decimal(value, "runtime model parameter")
            for value in payload["model_parameter_values"]
        )
        self._load_evaluators()
        with localcontext() as context:
            context.prec = working_precision
            context.rounding = ROUND_HALF_EVEN
            parameters = self._derive_model_parameters(parameters, working_precision)
            schema = _record(self._execution.get("runtime_schema"), "runtime schema")
            normalization = exact_normalization(
                self._physics,
                parameters,
                working_precision,
                cast(Any, schema.get("model_parameters", ())),
                scalar=self._scalar,
            )
            result = []
            for point, overrides in zip(points, vectors, strict=True):
                raw = self._evaluate_correlated_point(
                    point, parameters, working_precision, overrides
                )
                result.append(self._coherent_values(raw, groups))
        with localcontext() as context:
            context.prec = precision
            context.rounding = ROUND_HALF_EVEN
            values = tuple(
                tuple((+real, +imag) for real, imag in point) for point in result
            )
            normalization = +normalization
        return CoherentAmplitudeBatch(
            values, groups, normalization, precision, active_legs
        )

    def iter_coherent_amplitudes_many(
        self,
        momenta: Momenta,
        *,
        spin_vector_sets: Sequence[Mapping[int, object]],
        precision: int = 40,
        helicities: Sequence[str] | None = None,
    ) -> Iterator[tuple[int, int, CoherentAmplitudeBatch]]:
        """Stream ``(point index, spin-set index, one-point amplitudes)``.

        Common inputs and model parameters are prepared once. A stage retains
        at most its previous exact input/output for the current point, not an
        unbounded response tensor or a persistent numerical cache. Decimal
        contexts never remain active while this iterator yields to its caller.
        """
        if type(precision) is not int or precision < 1:
            raise EvaluationError(
                "precision must be a positive number of decimal digits"
            )
        points = _prepare_points(momenta, self._physics, None)
        vector_sets = tuple(
            _prepare_spin_vectors(
                vectors,
                allowed_legs=self._spin_correlated_legs,
                point_count=len(points),
            )
            for vectors in spin_vector_sets
        )
        active_sets = tuple(tuple(sorted(vectors[0])) for vectors in vector_sets)
        selected = {
            legs: self._selected_groups(helicities, legs) for legs in set(active_sets)
        }
        working_precision = _working_precision(precision)
        payload = _runtime_state(self._native_runtime)
        parameters = tuple(
            _decimal(value, "runtime model parameter")
            for value in payload["model_parameter_values"]
        )
        self._load_evaluators()
        with localcontext() as context:
            context.prec = working_precision
            context.rounding = ROUND_HALF_EVEN
            parameters = self._derive_model_parameters(parameters, working_precision)
            schema = _record(self._execution.get("runtime_schema"), "runtime schema")
            normalization = exact_normalization(
                self._physics,
                parameters,
                working_precision,
                cast(Any, schema.get("model_parameters", ())),
                scalar=self._scalar,
            )
        with localcontext() as context:
            context.prec = precision
            context.rounding = ROUND_HALF_EVEN
            normalization = +normalization
        for point_index, point in enumerate(points):
            memo: _StageMemo = {}
            for spin_index, (vectors, legs) in enumerate(
                zip(vector_sets, active_sets, strict=True)
            ):
                groups = selected[legs]
                with localcontext() as context:
                    context.prec = working_precision
                    context.rounding = ROUND_HALF_EVEN
                    raw = self._evaluate_correlated_point(
                        point,
                        parameters,
                        working_precision,
                        vectors[point_index],
                        stage_memo=memo,
                    )
                    values = self._coherent_values(raw, groups)
                with localcontext() as context:
                    context.prec = precision
                    context.rounding = ROUND_HALF_EVEN
                    values = tuple((+real, +imag) for real, imag in values)
                yield (
                    point_index,
                    spin_index,
                    CoherentAmplitudeBatch(
                        (values,), groups, normalization, precision, legs
                    ),
                )

    def _evaluate_stage(
        self, evaluator: Any, inputs: tuple[_ComplexDecimal, ...], precision: int
    ) -> tuple[_ComplexDecimal, ...]:
        """Evaluate one immutable stage; a narrow hook for diagnostic adapters."""
        return cast(tuple[_ComplexDecimal, ...], evaluator.evaluate(inputs, precision))

    def _evaluate_stage_cached(
        self,
        evaluator: Any,
        inputs: tuple[_ComplexDecimal, ...],
        precision: int,
        memo: _StageMemo | None,
        stage_index: int,
    ) -> tuple[_ComplexDecimal, ...]:
        previous = None if memo is None else memo.get(stage_index)
        if previous is not None and _identical_stage_inputs(previous[0], inputs):
            return previous[1]
        outputs = self._evaluate_stage(evaluator, inputs, precision)
        if memo is not None:
            memo[stage_index] = (inputs, outputs)
        return outputs

    def clear(self) -> None:
        """Release loaded evaluator states while retaining the source plan."""
        self._stage_evaluators = None
        self._amplitude_evaluator = None

    def _evaluate_correlated_point(
        self,
        point: tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...],
        model_parameters: tuple[Decimal, ...],
        precision: int,
        spin_vectors: Mapping[int, tuple[_ComplexDecimal, ...]],
        *,
        stage_memo: _StageMemo | None = None,
    ) -> tuple[_ComplexDecimal, ...]:
        scalar = Decimal if self._scalar is None else self._scalar
        if scalar is not Decimal:
            point = tuple(
                cast(
                    tuple[Decimal, Decimal, Decimal, Decimal],
                    tuple(scalar(value) for value in vector),
                )
                for vector in point
            )
        schema = _record(self._execution.get("runtime_schema"), "runtime schema")
        layout = _record(schema.get("parameter_layout"), "parameter layout")
        model_start = _json_integer(layout["value_component_count"]) + _json_integer(
            layout["momentum_parameter_count"]
        )
        state = [
            _complex_zero()
            for _ in range(
                max(
                    _json_integer(layout["parameter_count_if_flattened"]),
                    model_start + len(model_parameters),
                )
            )
        ]
        _fill_correlated_sources(
            state, point, schema, model_parameters, spin_vectors, scalar=self._scalar
        )
        _fill_momenta(state, point, schema)
        for index, value in enumerate(model_parameters):
            state[model_start + index] = (value, Decimal(0))
        compiled = _record(self._execution.get("compiled"), "compiled evaluators")
        stage_set = _record(compiled.get("stage_evaluators"), "stage evaluators")
        stages = _records(stage_set.get("stages"), "stage evaluators")
        assert self._stage_evaluators is not None
        for stage_index, (stage, evaluator) in enumerate(
            zip(stages, self._stage_evaluators, strict=True)
        ):
            outputs = self._evaluate_stage_cached(
                evaluator,
                _pack_stage_inputs(state, stage),
                precision,
                stage_memo,
                stage_index,
            )
            _assign_stage_outputs(state, outputs, stage)
        amplitude = _record(stage_set.get("amplitude_stage"), "amplitude evaluator")
        assert self._amplitude_evaluator is not None
        return _canonical_amplitude_outputs(
            self._evaluate_stage_cached(
                self._amplitude_evaluator,
                _pack_stage_inputs(state, amplitude),
                precision,
                stage_memo,
                len(stages),
            ),
            amplitude,
        )


class CorrelatedDoubleDoubleExecutor(CorrelatedExactExecutor):
    """The same retained plan, with explicitly owned DoubleFloat arithmetic."""

    _scalar: type[Decimal]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from pyamplicol.runtime._double_double import DoubleDoubleArithmetic

        self._double_double = DoubleDoubleArithmetic()
        self._scalar = self._double_double.scalar
        super().__init__(*args, **kwargs)

    def _derive_model_parameters(
        self,
        parameters: tuple[Decimal, ...],
        precision: int,
        *,
        arithmetic: str = "arbitrary",
    ) -> tuple[Decimal, ...]:
        return tuple(
            self._scalar(value)
            for value in super()._derive_model_parameters(
                tuple(self._scalar(value) for value in parameters),
                precision,
                arithmetic="double-double",
            )
        )

    def _evaluate_stage(
        self, evaluator: Any, inputs: tuple[_ComplexDecimal, ...], precision: int
    ) -> tuple[_ComplexDecimal, ...]:
        from pyamplicol.runtime.symbolica_exact import _upcast_complex_inputs

        # Bypass the arbitrary-precision policy, not the evaluator itself.
        # Each retained stage (including chunks) executes at DoubleFloat precision.
        outputs = evaluator._evaluate_prepared(_upcast_complex_inputs(inputs, 80), 32)
        return tuple((self._scalar(real), self._scalar(imag)) for real, imag in outputs)
