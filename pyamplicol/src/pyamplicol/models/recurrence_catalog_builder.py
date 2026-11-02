# SPDX-License-Identifier: 0BSD
"""Build the process-independent recurrence semantic catalog.

The builder deliberately consumes only model contracts and prepared-kernel
records.  It does not inspect a process and it does not classify a model by
name or Python type.  Any semantic relation that is not represented by those
contracts is rejected rather than inferred from a familiar particle inventory.

Prepared source-fill kernels are not part of the pre-existing eager kernel
catalog.  Recurrence companions therefore bind source semantics to generic
Rusticol source-fill templates, while vertex, propagator, closure, and model
parameter callables retain their prepared-kernel bindings.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from .base import (
    Model,
    RecurrenceQuantumFlowContract,
    Vertex,
    _runtime_particle_parameter_name,
)
from .contact_decomposition import HEFT_CONTACT_ORBIT_EVALUATOR_CLASS
from .lc_color_port_wiring import compile_lc_color_port_wirings
from .prepared_catalog import (
    PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF,
    PreparedClosureBinding,
    PreparedKernelCatalog,
    PreparedKernelCatalogError,
    PreparedKernelInput,
    PreparedParticleState,
    PreparedPropagatorBinding,
    PreparedVertexBinding,
)
from .recurrence_template import (
    ClosureTemplateV1,
    ColorContractionTemplateV1,
    ContactOrbitCertificateV1,
    ContactOrbitStepV1,
    CurrentStateTemplateV1,
    EvaluatorBindingV1,
    ExactComplexRationalV1,
    FlavourFlowOperationV1,
    LCColorSourceSeedV1,
    LCColorTransitionWitnessV1,
    ParameterTemplateV1,
    PropagatorTemplateV1,
    QuantumFlowTemplateV1,
    RecurrenceRuntimeHelicityContractV1,
    RecurrenceRuntimeHelicityVariantV1,
    RecurrenceTemplateCatalog,
    RecurrenceTemplateError,
    SourceTemplateV1,
    SymmetryProofV1,
    TransitionTemplateV1,
)

_PREPARED_CATALOG_ABI = "pyamplicol-prepared-kernel-catalog-v1"
_CURRENT_WORD_REFLECTION_PROOF = "canonical-current-word-reversal-v1"
_SUPPORTED_COLOR_RULES = frozenset(
    {
        "singlet",
        "color-identity",
        "fundamental-generator",
        "adjoint-structure-constant",
        "adjoint-structure-constant-product",
    }
)


def build_recurrence_template_catalog(
    model: Model,
    prepared_catalog: PreparedKernelCatalog,
    *,
    compiled_model_digest: str,
    prepared_kernel_pack_digest: str,
) -> RecurrenceTemplateCatalog:
    """Project one model and its exact prepared kernels into recurrence v1.

    The construction is deterministic and process independent.  Callback
    results that affect recurrence semantics are sampled twice and compared in
    canonical form; a stateful or nondeterministic model therefore fails
    closed.  All prepared callables are independently signature-checked before
    their IDs are admitted into the recurrence catalog.
    """

    _require_digest(compiled_model_digest, "compiled model digest")
    _require_digest(prepared_kernel_pack_digest, "prepared kernel pack digest")
    _validate_prepared_catalog_identity(model, prepared_catalog)
    kernels = _validated_kernels(prepared_catalog)
    recurrence_catalog = _recurrence_prepared_catalog(model, prepared_catalog)

    prepared_states = _collect_prepared_states(recurrence_catalog)
    parameters = _build_parameters(model, recurrence_catalog, prepared_states)
    parameter_ids = {parameter.name: parameter.template_id for parameter in parameters}
    current_states = _build_current_states(model, prepared_states, parameter_ids)
    state_ids = {
        (state.particle_id, state.chirality): state.template_id
        for state in current_states
    }
    state_by_id = {state.template_id: state for state in current_states}

    evaluator_builders: list[_EvaluatorRequest] = []
    sources, source_evaluator_bindings = _build_sources(
        model,
        prepared_states,
        state_ids,
        parameter_ids,
    )
    recurrence_vertex_bindings = _canonical_recurrence_vertex_bindings(
        recurrence_catalog.vertex_bindings
    )
    contact_orbit_certificates, contact_orbit_steps, contact_orbit_steps_by_binding = (
        _build_contact_orbit_records(recurrence_vertex_bindings)
    )
    (
        transition_quantum_flows,
        transitions,
        transition_colors,
        transition_proofs,
    ) = _build_transitions(
        model,
        recurrence_vertex_bindings,
        state_ids,
        parameter_ids,
        kernels,
        evaluator_builders,
        contact_orbit_steps_by_binding,
    )
    propagators, propagator_proofs = _build_propagators(
        model,
        recurrence_catalog.propagator_bindings,
        state_ids,
        parameter_ids,
        kernels,
        evaluator_builders,
    )
    (
        closure_quantum_flows,
        prepared_closures,
        prepared_closure_colors,
    ) = _build_closures(
        model,
        recurrence_catalog.closure_bindings,
        state_ids,
        parameter_ids,
        kernels,
        evaluator_builders,
    )
    direct_closures, direct_closure_colors, direct_closure_bindings = (
        _build_direct_closures(
            model,
            state_ids,
            state_by_id,
        )
    )
    closures = (*prepared_closures, *direct_closures)
    quantum_flows_by_id = {
        flow.template_id: flow
        for flow in (*transition_quantum_flows, *closure_quantum_flows)
    }
    closure_colors = (*prepared_closure_colors, *direct_closure_colors)
    _add_model_parameter_evaluator(
        recurrence_catalog,
        parameters,
        kernels,
        evaluator_builders,
    )
    evaluator_bindings = (
        *source_evaluator_bindings,
        *direct_closure_bindings,
        *_coalesce_evaluator_requests(evaluator_builders),
    )
    runtime_helicity_contracts = _build_runtime_helicity_contracts(
        model,
        recurrence_catalog,
        current_states,
        sources,
    )

    colors_by_id = {
        color.template_id: color for color in (*transition_colors, *closure_colors)
    }
    return RecurrenceTemplateCatalog.create(
        compiled_model_digest=compiled_model_digest,
        prepared_kernel_pack_digest=prepared_kernel_pack_digest,
        parameters=parameters,
        current_states=current_states,
        sources=sources,
        quantum_flows=tuple(quantum_flows_by_id.values()),
        contact_orbit_certificates=contact_orbit_certificates,
        contact_orbit_steps=contact_orbit_steps,
        transitions=transitions,
        propagators=propagators,
        closures=closures,
        color_contractions=tuple(colors_by_id.values()),
        symmetry_proofs=(*transition_proofs, *propagator_proofs),
        runtime_helicity_contracts=runtime_helicity_contracts,
        evaluator_bindings=evaluator_bindings,
    )


class _EvaluatorRequest:
    __slots__ = (
        "contract_kind",
        "input_state_template_ids",
        "kernel",
        "output_state_template_id",
        "semantic_template_id",
    )

    def __init__(
        self,
        *,
        kernel: Any,
        contract_kind: str,
        input_state_template_ids: tuple[str, ...],
        output_state_template_id: str | None,
        semantic_template_id: str,
    ) -> None:
        self.kernel = kernel
        self.contract_kind = contract_kind
        self.input_state_template_ids = input_state_template_ids
        self.output_state_template_id = output_state_template_id
        self.semantic_template_id = semantic_template_id


def _canonical_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RecurrenceTemplateError(
            "recurrence model callback returned noncanonical metadata"
        ) from exc


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _expression_digest(expression: str) -> str:
    if not isinstance(expression, str) or not expression:
        raise PreparedKernelCatalogError(
            "prepared kernel exact expressions must be nonempty strings"
        )
    return hashlib.sha256(expression.encode("utf-8")).hexdigest()


def _require_digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecurrenceTemplateError(f"{context} must be a lowercase SHA-256")
    return value


def _token(prefix: str, payload: object) -> str:
    return f"{prefix}:{_digest(payload)[:24]}"


def _exact_pair(value: object, context: str) -> ExactComplexRationalV1:
    if isinstance(value, complex):
        pair = (float(value.real), float(value.imag))
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        ):
            raise RecurrenceTemplateError(
                f"{context} must contain two real binary64 components"
            )
        pair = (float(value[0]), float(value[1]))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        pair = (float(value), 0.0)
    else:
        raise RecurrenceTemplateError(
            f"{context} must be a real or complex binary64 value"
        )
    if not all(math.isfinite(component) for component in pair):
        raise RecurrenceTemplateError(f"{context} must be finite")
    return ExactComplexRationalV1.from_binary64(*pair)


def _multiply_exact(
    left: ExactComplexRationalV1,
    right: ExactComplexRationalV1,
) -> ExactComplexRationalV1:
    real = left.real * right.real - left.imag * right.imag
    imag = left.real * right.imag + left.imag * right.real
    return ExactComplexRationalV1.from_fractions(real, imag)


def _validate_prepared_catalog_identity(
    model: Model,
    prepared_catalog: PreparedKernelCatalog,
) -> None:
    model_name = getattr(model, "name", None)
    if not isinstance(model_name, str) or not model_name:
        raise RecurrenceTemplateError("recurrence model name must be nonempty")
    if getattr(prepared_catalog, "model_name", None) != model_name:
        raise PreparedKernelCatalogError(
            "prepared kernel catalog model identity does not match the recurrence "
            f"model: {getattr(prepared_catalog, 'model_name', None)!r} != "
            f"{model_name!r}"
        )
    gaps = tuple(getattr(prepared_catalog, "unsupported_variants", ()))
    if gaps:
        details = "; ".join(
            f"{gap.contract_kind}:{gap.context}:{gap.reason}" for gap in gaps
        )
        raise PreparedKernelCatalogError(
            "prepared kernel catalog has unsupported recurrence semantics: " + details
        )


def _validated_kernels(
    prepared_catalog: PreparedKernelCatalog,
) -> Mapping[int, Any]:
    kernels = tuple(getattr(prepared_catalog, "kernels", ()))
    expected_ids = tuple(range(len(kernels)))
    actual_ids = tuple(getattr(kernel, "kernel_id", None) for kernel in kernels)
    if actual_ids != expected_ids:
        raise PreparedKernelCatalogError(
            "prepared kernel IDs must be contiguous before recurrence catalog build"
        )
    by_id: dict[int, Any] = {}
    signatures: list[str] = []
    for kernel in kernels:
        payload = {
            "abi": _PREPARED_CATALOG_ABI,
            "contract_kind": kernel.contract_kind,
            "inputs": [item.to_dict() for item in kernel.inputs],
            "outputs": list(kernel.exact_expressions),
            "output_layout": list(kernel.output_layout),
        }
        expected = _digest(payload)
        if kernel.canonical_signature != expected:
            raise PreparedKernelCatalogError(
                f"prepared kernel {kernel.kernel_id} has a stale canonical signature"
            )
        if len(kernel.exact_expressions) != len(kernel.output_layout):
            raise PreparedKernelCatalogError(
                f"prepared kernel {kernel.kernel_id} has misaligned outputs"
            )
        by_id[int(kernel.kernel_id)] = kernel
        signatures.append(str(kernel.canonical_signature))
    if tuple(signatures) != tuple(sorted(signatures)) or len(set(signatures)) != len(
        signatures
    ):
        raise PreparedKernelCatalogError(
            "prepared kernel signatures must be sorted and unique"
        )
    return by_id


def _collect_prepared_states(
    prepared_catalog: PreparedKernelCatalog,
) -> tuple[PreparedParticleState, ...]:
    states: dict[tuple[int, int], PreparedParticleState] = {}
    bindings: Sequence[Any] = (
        *prepared_catalog.vertex_bindings,
        *prepared_catalog.propagator_bindings,
        *prepared_catalog.closure_bindings,
    )
    for binding in bindings:
        candidates = tuple(
            state
            for name in ("left_state", "right_state", "result_state", "state")
            for state in (getattr(binding, name, None),)
            if state is not None
        )
        for state in candidates:
            key = (int(state.particle_id), int(state.chirality))
            previous = states.setdefault(key, state)
            if previous != state:
                raise PreparedKernelCatalogError(
                    "prepared bindings disagree on current-state semantics for "
                    f"particle={key[0]}, chirality={key[1]}"
                )
    return tuple(states[key] for key in sorted(states))


def _recurrence_prepared_catalog(
    model: Model,
    prepared_catalog: PreparedKernelCatalog,
) -> PreparedKernelCatalog:
    """Return the process-independent subset admitted by recurrence semantics.

    Prepared catalogs intentionally retain every exact model kernel useful to
    eager execution, including UFO ghost-interaction kernels.  Tree-level
    recurrence execution has no ghost source/current contract in v1.  Filter
    such bindings from the recurrence companion using the typed model source
    contract, without changing the prepared catalog or recognizing a model by
    name, particle ID, or Python implementation type.

    A future model that gives ghosts explicit recurrence semantics can extend
    the public source/runtime template ABI.  Until then, a process whose only
    construction needs an excluded binding will fail during process lowering
    because the semantic transition is absent; no unsupported identity is
    silently introduced.
    """

    states = _collect_prepared_states(prepared_catalog)
    admitted = {
        (state.particle_id, state.chirality)
        for state in states
        if _state_is_admitted_by_recurrence(model, state)
    }

    def state_key(state: PreparedParticleState) -> tuple[int, int]:
        return int(state.particle_id), int(state.chirality)

    vertex_bindings = tuple(
        binding
        for binding in prepared_catalog.vertex_bindings
        if all(
            state_key(state) in admitted
            for state in (
                binding.left_state,
                binding.right_state,
                binding.result_state,
            )
        )
    )
    propagator_bindings = tuple(
        binding
        for binding in prepared_catalog.propagator_bindings
        if state_key(binding.state) in admitted
    )
    closure_bindings = tuple(
        binding
        for binding in prepared_catalog.closure_bindings
        if all(
            state_key(state) in admitted
            for state in (
                binding.left_state,
                binding.right_state,
                binding.result_state,
            )
        )
    )
    return PreparedKernelCatalog(
        model_name=prepared_catalog.model_name,
        kernels=prepared_catalog.kernels,
        vertex_bindings=vertex_bindings,
        propagator_bindings=propagator_bindings,
        closure_bindings=closure_bindings,
        model_parameter_kernel_id=prepared_catalog.model_parameter_kernel_id,
        unsupported_variants=prepared_catalog.unsupported_variants,
    )


def _state_is_admitted_by_recurrence(
    model: Model,
    state: PreparedParticleState,
) -> bool:
    if model.auxiliary_kind(state.particle_id) is not None:
        return True
    wavefunction_family = _stable_callback(
        f"wavefunction family for particle {state.particle_id}",
        lambda: model.source_wavefunction_kind(state.particle_id),
        serializer=str,
    )
    if wavefunction_family == "ghost":
        return False
    if (
        _effective_auxiliary_kind(
            model,
            state.particle_id,
            state.chirality,
        )
        is not None
    ):
        return True
    source_ir = _stable_callback(
        f"source contract for particle {state.particle_id}",
        lambda: model._source_ir(state.particle_id),
        serializer=_canonical_source_ir,
    )
    if source_ir.statistics == "ghost" or source_ir.wavefunction_family == "ghost":
        return False
    if source_ir.statistics not in {"boson", "fermion", "auxiliary"}:
        raise RecurrenceTemplateError(
            "recurrence-template-v1 cannot classify source statistics "
            f"{source_ir.statistics!r} for particle={state.particle_id}"
        )
    # Require the corresponding finalization contract now.  This keeps missing
    # physical-state semantics distinct from deliberately unsupported ghosts.
    _stable_callback(
        f"propagator contract for particle {state.particle_id}",
        lambda: model._propagator_ir(state.particle_id, state.chirality),
        serializer=lambda value: _canonical_json(value.to_json_dict()),
    )
    return True


def _build_parameters(
    model: Model,
    prepared_catalog: PreparedKernelCatalog,
    prepared_states: Sequence[PreparedParticleState],
) -> tuple[ParameterTemplateV1, ...]:
    defaults: dict[str, ExactComplexRationalV1] = {}
    kinds: dict[str, str] = {}
    value_types: dict[str, str] = {}
    prepared_parameter_ids: dict[str, int] = {}

    for kernel in prepared_catalog.kernels:
        for item in kernel.inputs:
            if item.model_parameter_name is None:
                continue
            if item.model_parameter_index is None:
                raise RecurrenceTemplateError(
                    "prepared model-parameter input lacks its stable index"
                )
            name = str(item.model_parameter_name)
            index = int(item.model_parameter_index)
            previous = prepared_parameter_ids.setdefault(name, index)
            if previous != index:
                raise RecurrenceTemplateError(
                    f"prepared model parameter {name!r} has conflicting indices"
                )
    names_by_prepared_id: dict[int, str] = {}
    for name, index in prepared_parameter_ids.items():
        previous = names_by_prepared_id.setdefault(index, name)
        if previous != name:
            raise RecurrenceTemplateError(
                f"prepared parameter index {index} names both {previous!r} and {name!r}"
            )

    def add_default(name: object, value: object, *, kind: str) -> None:
        parameter_name = str(name)
        exact = _exact_pair(value, f"model parameter {parameter_name!r} default")
        previous = defaults.setdefault(parameter_name, exact)
        if previous != exact:
            raise RecurrenceTemplateError(
                f"model parameter {parameter_name!r} has conflicting defaults"
            )
        kinds.setdefault(parameter_name, kind)
        value_types.setdefault(
            parameter_name,
            "real" if exact.imag_numerator == 0 else "complex",
        )

    defaults_provider = getattr(model, "runtime_parameter_defaults", None)
    if callable(defaults_provider):
        first = dict(defaults_provider())
        second = dict(defaults_provider())
        if _canonical_binary64_mapping(first) != _canonical_binary64_mapping(second):
            raise RecurrenceTemplateError(
                "runtime_parameter_defaults is nondeterministic"
            )
        for name, value in first.items():
            add_default(name, value, kind="external")
            type_provider = getattr(model, "runtime_parameter_type", None)
            if callable(type_provider):
                declared = str(type_provider(str(name))).lower()
                if declared not in {"real", "complex"}:
                    raise RecurrenceTemplateError(
                        f"unsupported parameter type {declared!r} for {name!r}"
                    )
                value_types[str(name)] = declared

    normalization_provider = getattr(
        model, "runtime_normalization_parameter_defaults", None
    )
    if callable(normalization_provider):
        first = dict(normalization_provider())
        second = dict(normalization_provider())
        if _canonical_binary64_mapping(first) != _canonical_binary64_mapping(second):
            raise RecurrenceTemplateError(
                "runtime_normalization_parameter_defaults is nondeterministic"
            )
        for name, value in first.items():
            add_default(name, value, kind="external")

    definitions_provider = getattr(model, "runtime_derived_parameter_definitions", None)
    definitions = (
        {str(name): str(value) for name, value in definitions_provider().items()}
        if callable(definitions_provider)
        else {}
    )
    if callable(definitions_provider):
        repeated = {
            str(name): str(value) for name, value in definitions_provider().items()
        }
        if definitions != repeated:
            raise RecurrenceTemplateError(
                "runtime_derived_parameter_definitions is nondeterministic"
            )

    required_names = {
        str(item.model_parameter_name)
        for kernel in prepared_catalog.kernels
        for item in kernel.inputs
        if item.model_parameter_name is not None
    }
    required_names.update(
        str(name)
        for binding in prepared_catalog.propagator_bindings
        for name in binding.model_parameters
    )
    for state in prepared_states:
        if (
            _effective_auxiliary_kind(model, state.particle_id, state.chirality)
            is not None
        ):
            continue
        source_ir = _stable_callback(
            f"source contract for particle {state.particle_id}",
            lambda state=state: model._source_ir(state.particle_id),
            serializer=lambda value: _canonical_source_ir(value),
        )
        for name in (source_ir.mass_parameter, source_ir.width_parameter):
            if name is not None:
                required_names.add(str(name))
        if (
            source_ir.mass_parameter is not None
            and source_ir.mass_parameter not in defaults
            and source_ir.mass_parameter not in definitions
        ):
            add_default(
                source_ir.mass_parameter,
                model.mass(state.particle_id),
                kind="external",
            )
        if (
            source_ir.width_parameter is not None
            and source_ir.width_parameter not in defaults
            and source_ir.width_parameter not in definitions
        ):
            add_default(
                source_ir.width_parameter,
                model.width(state.particle_id),
                kind="external",
            )

        # Match the model-generic fallback naming contract used by the
        # prepared-kernel builder when a model does not expose named particle
        # parameters itself.
        particle = model.particle(state.particle_id)
        fallback_values = {
            f"particle.{int(particle.pdg)}.mass": model.mass(state.particle_id),
            f"particle.{int(particle.pdg)}.width": model.width(state.particle_id),
        }
        for name, value in fallback_values.items():
            if (
                name in required_names
                and name not in defaults
                and name not in definitions
            ):
                add_default(name, value, kind="external")

    derived_defaults_provider = getattr(
        model, "runtime_derived_parameter_defaults", None
    )
    derived_defaults = (
        dict(derived_defaults_provider()) if callable(derived_defaults_provider) else {}
    )
    for name, value in derived_defaults.items():
        add_default(name, value, kind="derived")

    missing = sorted(required_names - defaults.keys() - definitions.keys())
    if missing:
        raise RecurrenceTemplateError(
            "recurrence catalog cannot resolve defaults or exact definitions for "
            "prepared model parameters: " + ", ".join(missing)
        )

    all_names = tuple(sorted(defaults.keys() | definitions.keys()))
    # Prepared kernels only retain model-parameter descriptors that survive
    # exact expression simplification.  A restricted mass such as UFO ``Me``
    # can therefore be absent from every kernel input while still being
    # required by SourceIR at runtime.  Recurrence owns a parameter workspace
    # with one slot per catalog parameter, so bind every otherwise-unbound
    # required parameter to a deterministic free slot.  Existing kernel input
    # indices remain authoritative.
    occupied_prepared_ids = set(prepared_parameter_ids.values())
    free_prepared_ids = iter(
        index for index in range(len(all_names)) if index not in occupied_prepared_ids
    )
    for name in sorted(required_names):
        if name in prepared_parameter_ids:
            continue
        try:
            prepared_parameter_ids[name] = next(free_prepared_ids)
        except StopIteration as exc:  # pragma: no cover - guarded by cardinality
            raise RecurrenceTemplateError(
                "recurrence catalog has no free prepared slot for required "
                f"model parameter {name!r}"
            ) from exc

    ids = {name: _token("parameter", {"name": name}) for name in all_names}
    external_ids = tuple(
        sorted(ids[name] for name in all_names if name not in definitions)
    )
    records: list[ParameterTemplateV1] = []
    for name in all_names:
        derived = name in definitions
        default = defaults.get(name)
        declared = value_types.get(
            name,
            "complex"
            if default is not None and default.imag_numerator != 0
            else "real",
        )
        if declared == "real" and default is not None and default.imag_numerator != 0:
            raise RecurrenceTemplateError(
                f"real model parameter {name!r} has a complex default"
            )
        records.append(
            ParameterTemplateV1(
                template_id=ids[name],
                name=name,
                parameter_kind="derived" if derived else "external",
                value_type=declared,  # type: ignore[arg-type]
                mutable=not derived,
                default_value=None if derived else default,
                exact_expression_digest=(
                    _expression_digest(definitions[name]) if derived else None
                ),
                # A superset is conservative and avoids parsing model syntax as
                # proof. Rust may use the evaluator's actual input layout.
                dependency_parameter_ids=external_ids if derived else (),
                prepared_parameter_id=prepared_parameter_ids.get(name),
            )
        )
    return tuple(records)


def _canonical_binary64_mapping(values: Mapping[object, object]) -> str:
    return _canonical_json(
        {
            str(name): _exact_pair(value, f"parameter {name!r}").to_dict()
            for name, value in values.items()
        }
    )


def _build_current_states(
    model: Model,
    prepared_states: Sequence[PreparedParticleState],
    parameter_ids: Mapping[str, str],
) -> tuple[CurrentStateTemplateV1, ...]:
    records: list[CurrentStateTemplateV1] = []
    for state in prepared_states:
        auxiliary_kind = _effective_auxiliary_kind(
            model, state.particle_id, state.chirality
        )
        if auxiliary_kind is None:
            source_ir = _stable_callback(
                f"source contract for particle {state.particle_id}",
                lambda state=state: model._source_ir(state.particle_id),
                serializer=lambda value: _canonical_source_ir(value),
            )
            identity = source_ir.identity
            basis = model._current_basis(state.particle_id, state.chirality)
            statistics = source_ir.statistics
            mass_parameter = source_ir.mass_parameter
            width_parameter = source_ir.width_parameter
        else:
            identity = _stable_callback(
                f"auxiliary identity for particle {state.particle_id}",
                lambda state=state: model._particle_identity_ir(state.particle_id),
                serializer=lambda value: _canonical_json(value.to_json_dict()),
            )
            basis = model._current_basis(state.particle_id, state.chirality)
            statistics = "fermion" if model.is_fermion(state.particle_id) else "boson"
            propagator = model._propagator_ir(state.particle_id, state.chirality)
            mass_parameter = propagator.mass_parameter
            width_parameter = propagator.width_parameter
        if (
            identity.canonical_id != state.identity
            or identity.orientation != state.orientation
            or basis != state.basis
            or int(model.current_dimension(state.particle_id, state.chirality))
            != state.dimension
        ):
            raise PreparedKernelCatalogError(
                "prepared current-state metadata does not match the live model "
                f"contract for particle={state.particle_id}, "
                f"chirality={state.chirality}"
            )
        if statistics == "ghost":
            raise RecurrenceTemplateError(
                f"recurrence-template-v1 cannot encode ghost current {state.identity!r}"
            )
        public_statistics = "fermion" if statistics == "fermion" else "boson"
        mass_parameter = _runtime_particle_parameter_name(
            mass_parameter,
            particle_pdg=int(model.particle(state.particle_id).pdg),
            kind="mass",
            available_names=parameter_ids,
        )
        mass_id = _parameter_reference(
            mass_parameter, parameter_ids, "current mass parameter"
        )
        width_id = _parameter_reference(
            width_parameter, parameter_ids, "current width parameter"
        )
        records.append(
            CurrentStateTemplateV1(
                template_id=_state_template_id(state),
                particle_id=state.particle_id,
                anti_particle_id=identity.anti_pdg_label,
                species_id=identity.species_id,
                orientation=state.orientation,
                statistics=public_statistics,
                color_representation=int(model.color_rep(state.particle_id)),
                basis=state.basis,
                tensor_ordering=tuple(
                    f"{state.basis}:c{component}"
                    for component in range(state.dimension)
                ),
                dimension=state.dimension,
                chirality=state.chirality,
                lc_color_shape_kind=model.recurrence_lc_color_shape_contract(
                    state.particle_id,
                    state.chirality,
                ),
                auxiliary_kind=(
                    None if auxiliary_kind is None else str(auxiliary_kind)
                ),
                mass_parameter_id=mass_id,
                width_parameter_id=width_id,
            )
        )
    return tuple(records)


def _state_template_id(state: PreparedParticleState) -> str:
    return _token(
        "state",
        {
            "particle_id": state.particle_id,
            "identity": state.identity,
            "orientation": state.orientation,
            "basis": state.basis,
            "chirality": state.chirality,
            "dimension": state.dimension,
        },
    )


def _parameter_reference(
    name: str | None,
    parameter_ids: Mapping[str, str],
    context: str,
) -> str | None:
    if name is None:
        return None
    try:
        return parameter_ids[str(name)]
    except KeyError as exc:
        raise RecurrenceTemplateError(
            f"{context} {name!r} is absent from the recurrence parameter catalog"
        ) from exc


def _build_sources(
    model: Model,
    prepared_states: Sequence[PreparedParticleState],
    state_ids: Mapping[tuple[int, int], str],
    parameter_ids: Mapping[str, str],
) -> tuple[tuple[SourceTemplateV1, ...], tuple[EvaluatorBindingV1, ...]]:
    records: list[SourceTemplateV1] = []
    bindings: list[EvaluatorBindingV1] = []
    for prepared_state in prepared_states:
        if (
            _effective_auxiliary_kind(
                model,
                prepared_state.particle_id,
                prepared_state.chirality,
            )
            is not None
        ):
            continue
        source_ir = model._source_ir(prepared_state.particle_id)
        mass_parameter = _runtime_particle_parameter_name(
            source_ir.mass_parameter,
            particle_pdg=int(model.particle(prepared_state.particle_id).pdg),
            kind="mass",
            available_names=parameter_ids,
        )
        states = tuple(
            state
            for state in source_ir.states
            if int(state.chirality) == prepared_state.chirality
        )
        if not states:
            # Full internal fermion states may coexist with chiral external
            # source states. They are recurrence currents, not source-fill
            # entry points.
            continue
        for state in states:
            if not isinstance(state.spin_state, int) or isinstance(
                state.spin_state, bool
            ):
                raise RecurrenceTemplateError(
                    "recurrence-template-v1 cannot encode structured source spin "
                    f"state {state.spin_state!r}"
                )
            state_key = (
                prepared_state.particle_id,
                prepared_state.chirality,
            )
            flavour_flow = (int(prepared_state.particle_id),)
            quantum_number_flow = _stable_callback(
                f"source quantum-number flow for particle {prepared_state.particle_id}",
                lambda prepared_state=prepared_state: model.quantum_number_flow(
                    prepared_state.particle_id
                ),
                serializer=lambda value: _canonical_json(list(value)),
            )
            canonical_quantum_number_flow = tuple(
                (str(name), str(expression)) for name, expression in quantum_number_flow
            )
            mass_parameter_id = _parameter_reference(
                mass_parameter,
                parameter_ids,
                "source mass parameter",
            )
            width_parameter_id = _parameter_reference(
                source_ir.width_parameter,
                parameter_ids,
                "source width parameter",
            )
            lc_color_seed = _source_color_seed(
                model,
                prepared_state.particle_id,
                prepared_state.chirality,
                state_template_id=state_ids[state_key],
            )
            source_semantics = {
                "abi": "rusticol-source-fill-v1",
                "state": state_ids[state_key],
                "statistics": source_ir.statistics,
                "wavefunction_family": source_ir.wavefunction_family,
                "basis": source_ir.basis,
                "dimension": prepared_state.dimension,
                "crossing": _canonical_crossing(source_ir.crossing),
                "helicity": int(state.helicity),
                "chirality": int(state.chirality),
                "spin_state": int(state.spin_state),
                "flavour_flow": list(flavour_flow),
                "quantum_number_flow": [
                    list(item) for item in canonical_quantum_number_flow
                ],
                "lc_color_seed": lc_color_seed.to_dict(),
                "mass_parameter_id": mass_parameter_id,
                "width_parameter_id": width_parameter_id,
            }
            callable_signature = _digest(source_semantics)
            runtime_template = (
                f"rusticol.source-fill.{source_ir.wavefunction_family}.v1:"
                f"{callable_signature[:24]}"
            )
            template_id = _token(
                "source",
                source_semantics,
            )
            resolver_key = _token("evaluator", source_semantics)
            record = SourceTemplateV1(
                template_id=template_id,
                state_template_id=state_ids[state_key],
                crossing=_canonical_crossing(source_ir.crossing),
                wavefunction_family=source_ir.wavefunction_family,
                helicity=int(state.helicity),
                spin_state=int(state.spin_state),
                flavour_flow=flavour_flow,
                quantum_number_flow=canonical_quantum_number_flow,
                lc_color_seed=lc_color_seed,
                wavefunction_expression_digest=callable_signature,
                evaluator_resolver_key=resolver_key,
                mass_parameter_id=mass_parameter_id,
                width_parameter_id=width_parameter_id,
            )
            records.append(record)
            input_layout = (
                "momentum:energy",
                "momentum:px",
                "momentum:py",
                "momentum:pz",
                *(() if mass_parameter_id is None else (mass_parameter_id,)),
                *(() if width_parameter_id is None else (width_parameter_id,)),
            )
            output_layout = tuple(
                f"source-component:{component}"
                for component in range(prepared_state.dimension)
            )
            bindings.append(
                EvaluatorBindingV1(
                    resolver_key=resolver_key,
                    prepared_kernel_id=None,
                    callable_kind="rusticol-template",
                    runtime_template=runtime_template,
                    contract_kind="source",
                    callable_signature=callable_signature,
                    input_state_template_ids=(),
                    output_state_template_id=record.state_template_id,
                    input_layout=input_layout,
                    output_layout=output_layout,
                    exact_expression_digests=tuple(
                        _digest({"source": source_semantics, "component": component})
                        for component in range(prepared_state.dimension)
                    ),
                    semantic_template_ids=(record.template_id,),
                )
            )
    return tuple(records), tuple(bindings)


def _build_runtime_helicity_contracts(
    model: Model,
    prepared_catalog: PreparedKernelCatalog,
    current_states: Sequence[CurrentStateTemplateV1],
    sources: Sequence[SourceTemplateV1],
) -> tuple[RecurrenceRuntimeHelicityContractV1, ...]:
    """Certify source families that can execute through one full current state.

    Incomplete families are deliberately absent.  Topology replay continues to
    consume their concrete source states, while all-flow union calls the catalog
    preflight and fails before lowering instead of guessing an embedding.
    """

    states_by_id = {state.template_id: state for state in current_states}
    states_by_particle_chirality = {
        (state.particle_id, state.chirality): state for state in current_states
    }
    source_families: dict[int, list[SourceTemplateV1]] = {}
    for source in sources:
        state = states_by_id[source.state_template_id]
        source_families.setdefault(state.particle_id, []).append(source)

    contracts: list[RecurrenceRuntimeHelicityContractV1] = []
    for particle_id in sorted(source_families):
        family = tuple(
            sorted(source_families[particle_id], key=lambda item: item.template_id)
        )
        source_states = tuple(states_by_id[item.state_template_id] for item in family)
        unique_state_ids = {state.template_id for state in source_states}
        full_state = states_by_particle_chirality.get((particle_id, 0))
        if full_state is None and len(unique_state_ids) == 1:
            full_state = source_states[0]
        if full_state is None:
            continue

        variants: list[RecurrenceRuntimeHelicityVariantV1] = []
        proof_inputs: list[object] = []
        complete = True
        for source, source_state in zip(family, source_states, strict=True):
            if source_state.template_id == full_state.template_id:
                embedding = tuple(range(full_state.dimension))
                projection = tuple(range(full_state.dimension))
                ordering_proof: object = {
                    "kind": "identity",
                    "state_digest": full_state.semantic_digest,
                }
            else:
                try:
                    ordering = model.recurrence_current_embedding_contract(
                        source_state.particle_id,
                        source_state.chirality,
                    )
                except NotImplementedError:
                    complete = False
                    break
                embedding = tuple(ordering.input_embedding)
                projection = tuple(ordering.source_projection)
                if (
                    len(embedding) != full_state.dimension
                    or len(projection) != source_state.dimension
                    or ordering.source_dimension != source_state.dimension
                    or ordering.full_dimension != full_state.dimension
                    or ordering.source_basis != source_state.basis
                    or ordering.full_basis != full_state.basis
                ):
                    complete = False
                    break
                ordering_proof = ordering.to_json_dict()
            if not _prepared_full_state_family_is_complete(
                prepared_catalog,
                particle_id=particle_id,
                source_state=source_state,
                full_state_template=full_state,
                states_by_particle_chirality=states_by_particle_chirality,
            ):
                complete = False
                break
            factors = tuple(
                ExactComplexRationalV1.zero()
                if component is None
                else ExactComplexRationalV1.one()
                for component in embedding
            )
            proof_payload = {
                "abi": "pyamplicol-recurrence-runtime-helicity-variant-v1",
                "full_state": full_state.semantic_digest,
                "ordering": ordering_proof,
                "source": source.semantic_digest,
                "source_state": source_state.semantic_digest,
            }
            proof_digest = _digest(proof_payload)
            proof_inputs.append(proof_payload)
            variants.append(
                RecurrenceRuntimeHelicityVariantV1(
                    source_template_id=source.template_id,
                    source_state_template_id=source_state.template_id,
                    embedding_source_components=embedding,
                    embedding_factors=factors,
                    projection_full_components=projection,
                    proof_digest=proof_digest,
                )
            )
        if not complete or len(variants) != len(family):
            continue
        proof_payload = {
            "abi": "pyamplicol-recurrence-runtime-helicity-contract-v1",
            "full_state": full_state.semantic_digest,
            "variants": proof_inputs,
        }
        contracts.append(
            RecurrenceRuntimeHelicityContractV1(
                template_id=_token("runtime-helicity", proof_payload),
                full_state_template_id=full_state.template_id,
                variants=tuple(variants),
                proof_algorithm=(
                    "prepared-current-ordering-runtime-helicity-embedding-v1"
                ),
                proof_digest=_digest(proof_payload),
            )
        )
    return tuple(contracts)


def _prepared_full_state_family_is_complete(
    prepared_catalog: PreparedKernelCatalog,
    *,
    particle_id: int,
    source_state: CurrentStateTemplateV1,
    full_state_template: CurrentStateTemplateV1,
    states_by_particle_chirality: Mapping[tuple[int, int], CurrentStateTemplateV1],
) -> bool:
    if source_state.template_id == full_state_template.template_id:
        return True
    full_prepared_state = next(
        (
            binding.state
            for binding in prepared_catalog.propagator_bindings
            if binding.state.particle_id == particle_id
            and binding.state.chirality == full_state_template.chirality
        ),
        None,
    )
    if full_prepared_state is None:
        return False

    def full_state(state: PreparedParticleState) -> PreparedParticleState | None:
        template = states_by_particle_chirality.get((state.particle_id, 0))
        if template is None:
            return state if state.chirality == 0 else None
        return next(
            (
                candidate.state
                for candidate in prepared_catalog.propagator_bindings
                if candidate.state.particle_id == state.particle_id
                and candidate.state.chirality == template.chirality
            ),
            None,
        )

    vertex_inventory = {
        (
            binding.key.kind,
            binding.key.particles,
            full_state(binding.left_state),
            full_state(binding.right_state),
            full_state(binding.result_state),
        )
        for binding in prepared_catalog.vertex_bindings
        if all(
            full_state(state) == state
            for state in (binding.left_state, binding.right_state, binding.result_state)
        )
    }
    closure_inventory = {
        (
            binding.key.kind,
            binding.key.particles,
            full_state(binding.left_state),
            full_state(binding.right_state),
            full_state(binding.result_state),
        )
        for binding in prepared_catalog.closure_bindings
        if all(
            full_state(state) == state
            for state in (binding.left_state, binding.right_state, binding.result_state)
        )
    }
    for bindings, inventory in (
        (prepared_catalog.vertex_bindings, vertex_inventory),
        (prepared_catalog.closure_bindings, closure_inventory),
    ):
        for binding in bindings:
            states = (binding.left_state, binding.right_state, binding.result_state)
            if not any(
                state.particle_id == particle_id
                and state.chirality == source_state.chirality
                for state in states
            ):
                continue
            mapped = tuple(full_state(state) for state in states)
            if any(state is None for state in mapped):
                return False
            key = (binding.key.kind, binding.key.particles, *mapped)
            if key not in inventory:
                return False
    return True


def _source_color_seed(
    model: Model,
    particle_id: int,
    chirality: int,
    *,
    state_template_id: str,
) -> LCColorSourceSeedV1:
    model_type = type(model)
    mro = model_type.__mro__
    shape_owner = next(
        owner for owner in mro if "recurrence_lc_color_shape_contract" in owner.__dict__
    )
    seed_owner = next(
        owner
        for owner in mro
        if "recurrence_lc_source_color_contract" in owner.__dict__
    )
    if mro.index(seed_owner) > mro.index(shape_owner):
        raise RecurrenceTemplateError(
            f"{model_type.__name__}.recurrence_lc_color_shape_contract overrides "
            "the LC source shape without declaring a matching source-color seed "
            "contract"
        )

    def evaluate() -> Any:
        return model.recurrence_lc_source_color_contract(particle_id, chirality)

    contract = _stable_callback(
        f"LC source-color seed for particle {particle_id} chirality {chirality}",
        evaluate,
        serializer=lambda value: _canonical_json(
            {
                "operation": value.operation,
                "output_shape_kind": value.output_shape_kind,
                "component_kind": value.component_kind,
                "component_role": value.component_role,
                "proof_digest": value.proof_digest,
                "provenance": list(value.provenance),
            }
        ),
    )
    expected_shape = model.recurrence_lc_color_shape_contract(particle_id, chirality)
    if contract.output_shape_kind != expected_shape:
        raise RecurrenceTemplateError(
            "LC source-color seed output shape does not match its current-state "
            f"contract: {contract.output_shape_kind!r} != {expected_shape!r}"
        )
    payload = {
        "abi": "pyamplicol-recurrence-lc-source-seed-v1",
        "state_template_id": state_template_id,
        "particle_id": int(particle_id),
        "chirality": int(chirality),
        "operation": contract.operation,
        "output_shape_kind": contract.output_shape_kind,
        "component_kind": contract.component_kind,
        "component_role": contract.component_role,
        "provenance": [list(item) for item in contract.provenance],
    }
    return LCColorSourceSeedV1(
        operation=contract.operation,
        output_shape_kind=contract.output_shape_kind,
        component_kind=contract.component_kind,
        component_role=contract.component_role,
        proof_digest=contract.proof_digest or _digest(payload),
        provenance=contract.provenance,
    )


def _canonical_crossing(crossing: Any) -> str:
    payload = {
        "momentum_transform": crossing.momentum_transform,
        "helicity_factor": int(crossing.helicity_factor),
        "chirality_factor": int(crossing.chirality_factor),
        "spin_state_factor": int(crossing.spin_state_factor),
        "phase": _exact_pair(crossing.phase, "source crossing phase").to_dict(),
    }
    return _canonical_json(payload)


def _effective_auxiliary_kind(
    model: Model,
    particle_id: int,
    chirality: int,
) -> str | None:
    declared = model.auxiliary_kind(particle_id)
    if declared is not None:
        return str(declared)
    particle = model.particle(particle_id)
    if int(particle.spin) >= 0:
        return None
    try:
        propagator = model._propagator_ir(particle_id, chirality)
    except Exception as exc:
        raise RecurrenceTemplateError(
            "recurrence-template-v1 cannot encode a negative-spin state without "
            f"an explicit auxiliary/propagator contract: particle={particle_id}, "
            f"chirality={chirality}: {exc}"
        ) from exc
    if not propagator.applies_propagator and propagator.auxiliary_policy:
        return str(propagator.auxiliary_policy)
    raise RecurrenceTemplateError(
        "recurrence-template-v1 cannot encode a negative-spin state without an "
        f"explicit auxiliary contract: particle={particle_id}, "
        f"chirality={chirality}"
    )


def _canonical_source_ir(source_ir: Any) -> str:
    payload = source_ir.to_json_dict()
    crossing = dict(payload["crossing"])
    crossing["phase"] = _exact_pair(
        crossing["phase"], "source crossing phase"
    ).to_dict()
    payload["crossing"] = crossing
    return _canonical_json(payload)


def _build_contact_orbit_records(
    bindings: Sequence[PreparedVertexBinding],
) -> tuple[
    tuple[ContactOrbitCertificateV1, ...],
    tuple[ContactOrbitStepV1, ...],
    Mapping[object, tuple[ContactOrbitStepV1, ...]],
]:
    certificates: dict[str, ContactOrbitCertificateV1] = {}
    certificate_id_by_term: dict[int, str] = {}
    steps: dict[str, ContactOrbitStepV1] = {}
    steps_by_binding: dict[object, tuple[ContactOrbitStepV1, ...]] = {}
    for binding in bindings:
        for certificate in binding.contact_orbit_certificates:
            if certificate.reconstruction_factor != "1":
                raise RecurrenceTemplateError(
                    "recurrence v1 supports only exact unit contact-orbit "
                    "reconstruction"
                )
            certificate_id = _token("contact-orbit-certificate", certificate.to_dict())
            previous_id = certificate_id_by_term.setdefault(
                certificate.term_id, certificate_id
            )
            if previous_id != certificate_id:
                raise RecurrenceTemplateError(
                    "contact-orbit term has conflicting certificates"
                )
            certificate_record = ContactOrbitCertificateV1(
                template_id=certificate_id,
                algorithm=certificate.algorithm,
                algorithm_version=certificate.algorithm_version,
                term_id=certificate.term_id,
                vertex=certificate.vertex,
                particles=certificate.particles,
                evaluator_class=certificate.evaluator_class,
                physical_leg_equivalence_classes=(
                    certificate.physical_leg_equivalence_classes
                ),
                reconstruction_factor=ExactComplexRationalV1.one(),
            )
            previous_certificate = certificates.setdefault(
                certificate_id, certificate_record
            )
            if previous_certificate != certificate_record:
                raise RecurrenceTemplateError(
                    "contact-orbit certificate identity collision"
                )
    for binding in bindings:
        binding_steps: list[ContactOrbitStepV1] = []
        for step in binding.contact_orbit_steps:
            if step.reconstruction_factor != "1":
                raise RecurrenceTemplateError(
                    "recurrence v1 supports only exact unit contact-orbit "
                    "reconstruction"
                )
            certificate_id = certificate_id_by_term.get(step.term_id)
            if certificate_id is None:
                raise PreparedKernelCatalogError(
                    "prepared contact-orbit step has no certificate"
                )
            step_id = _token(
                "contact-orbit-step",
                {
                    "certificate_template_id": certificate_id,
                    "step": step.to_dict(),
                },
            )
            step_record = ContactOrbitStepV1(
                template_id=step_id,
                certificate_template_id=certificate_id,
                stage=step.stage,
                result_leg=step.result_leg,
                left_covered_legs=step.left_covered_legs,
                right_covered_legs=step.right_covered_legs,
                source_particle_legs=step.source_particle_legs,
                reconstruction_factor=ExactComplexRationalV1.one(),
            )
            previous_step = steps.setdefault(step_id, step_record)
            if previous_step != step_record:
                raise RecurrenceTemplateError("contact-orbit step identity collision")
            binding_steps.append(step_record)
        if binding_steps:
            steps_by_binding[binding.key] = tuple(
                sorted(set(binding_steps), key=lambda item: item.template_id)
            )
    return (
        tuple(sorted(certificates.values(), key=lambda item: item.template_id)),
        tuple(sorted(steps.values(), key=lambda item: item.template_id)),
        steps_by_binding,
    )


def _canonical_recurrence_vertex_bindings(
    bindings: Sequence[PreparedVertexBinding],
) -> tuple[PreparedVertexBinding, ...]:
    """Drop only compiler-certified mirrored scalar-HEFT tree orientations.

    Recurrence construction indexes an unordered pair of current states and
    reorders the concrete parents into a transition's evaluator convention.
    Consequently the two compiler-emitted input orientations of one HEFT tree
    node are the same recurrence transition.  Keeping both is harmless for the
    generic DAG backends, which consume the complete oriented kernel catalog,
    but doubles the contact contribution in recurrence modes because colored
    mirror currents do not share a dynamic-color identity.

    The filtering is deliberately limited to the exact HEFT evaluator class
    authenticated by the compiled-model contract.  Every discarded binding
    must have a retained, state-exact mirror with the source-leg lineage
    reversed; an unfamiliar or incomplete family fails closed.
    """

    heft_bindings: list[
        tuple[
            PreparedVertexBinding,
            object,
            object,
        ]
    ] = []
    retained: list[PreparedVertexBinding] = []
    for binding in bindings:
        heft_certificates = tuple(
            certificate
            for certificate in binding.contact_orbit_certificates
            if certificate.evaluator_class == HEFT_CONTACT_ORBIT_EVALUATOR_CLASS
        )
        if not heft_certificates:
            retained.append(binding)
            continue
        if (
            len(heft_certificates) != 1
            or len(binding.contact_orbit_certificates) != 1
            or len(binding.contact_orbit_steps) != 1
        ):
            raise PreparedKernelCatalogError(
                "a scalar-HEFT recurrence binding must carry exactly one "
                "certificate and one tree step"
            )
        certificate = heft_certificates[0]
        step = binding.contact_orbit_steps[0]
        if step.term_id != certificate.term_id:
            raise PreparedKernelCatalogError(
                "a scalar-HEFT recurrence step belongs to another certificate"
            )
        heft_bindings.append((binding, certificate, step))

    for binding, certificate, step in heft_bindings:
        left_key = (step.left_covered_legs, step.source_particle_legs[0])
        right_key = (step.right_covered_legs, step.source_particle_legs[1])
        if left_key <= right_key:
            retained.append(binding)
            continue

        mirrors = tuple(
            candidate
            for candidate, candidate_certificate, candidate_step in heft_bindings
            if candidate_certificate == certificate
            and candidate_step.stage == step.stage
            and candidate_step.result_leg == step.result_leg
            and candidate_step.left_covered_legs == step.right_covered_legs
            and candidate_step.right_covered_legs == step.left_covered_legs
            and candidate_step.source_particle_legs
            == (
                step.source_particle_legs[1],
                step.source_particle_legs[0],
                step.source_particle_legs[2],
            )
            and candidate.left_state == binding.right_state
            and candidate.right_state == binding.left_state
            and candidate.result_state == binding.result_state
            and candidate.key.coupling == binding.key.coupling
            and candidate.output_factor_source == binding.output_factor_source
        )
        if len(mirrors) != 1:
            raise PreparedKernelCatalogError(
                "a mirrored scalar-HEFT recurrence binding has no unique "
                "compiler-certified canonical orientation"
            )
        mirror_step = mirrors[0].contact_orbit_steps[0]
        mirror_left_key = (
            mirror_step.left_covered_legs,
            mirror_step.source_particle_legs[0],
        )
        mirror_right_key = (
            mirror_step.right_covered_legs,
            mirror_step.source_particle_legs[1],
        )
        if mirror_left_key > mirror_right_key:
            raise PreparedKernelCatalogError(
                "a mirrored scalar-HEFT recurrence binding did not resolve to "
                "the canonical orientation"
            )

    return tuple(sorted(retained, key=lambda item: item.key))


def _singleton_contact_orbit_step_groups(
    steps: Sequence[ContactOrbitStepV1],
) -> tuple[tuple[ContactOrbitStepV1, ...], ...]:
    """Split certified semantic lineage while preserving one numerical kernel.

    A prepared vertex binding can reuse one exact evaluator for several proven
    contact-orbit orientations.  Recurrence transitions, however, must name
    one such orientation apiece so native owner selection never suppresses an
    unrelated member of a numerically fused binding.  Uncertified bindings
    retain their single ordinary transition.
    """

    if not steps:
        return ((),)
    return tuple((step,) for step in sorted(steps, key=lambda item: item.template_id))


def _build_transitions(
    model: Model,
    bindings: Sequence[PreparedVertexBinding],
    state_ids: Mapping[tuple[int, int], str],
    parameter_ids: Mapping[str, str],
    kernels: Mapping[int, Any],
    evaluator_requests: list[_EvaluatorRequest],
    contact_orbit_steps_by_binding: Mapping[object, tuple[ContactOrbitStepV1, ...]],
) -> tuple[
    tuple[QuantumFlowTemplateV1, ...],
    tuple[TransitionTemplateV1, ...],
    tuple[ColorContractionTemplateV1, ...],
    tuple[SymmetryProofV1, ...],
]:
    candidates: list[
        tuple[
            str,
            tuple[int, int, str],
            TransitionTemplateV1,
            QuantumFlowTemplateV1,
            ColorContractionTemplateV1,
            Any,
            tuple[str, ...],
            str,
            tuple[ExactComplexRationalV1, str] | None,
        ]
    ] = []
    for binding in sorted(bindings, key=lambda item: item.key):
        contact_orbit_step_groups = _singleton_contact_orbit_step_groups(
            contact_orbit_steps_by_binding.get(binding.key, ())
        )
        vertex = Vertex(
            binding.key.kind,
            binding.key.particles,
            binding.key.coupling,
        )
        _validate_vertex_binding_against_model(model, binding, vertex)
        kernel = _kernel(kernels, binding.kernel_id, "vertex")
        if kernel.contract_kind != "vertex":
            raise PreparedKernelCatalogError(
                f"prepared vertex binding references a {kernel.contract_kind!r} kernel"
            )
        concrete_state_ids = (
            state_ids[(binding.left_state.particle_id, binding.left_state.chirality)],
            state_ids[(binding.right_state.particle_id, binding.right_state.chirality)],
        )
        result_state_id = state_ids[
            (binding.result_state.particle_id, binding.result_state.chirality)
        ]
        _validate_binary_kernel_layout(
            kernel,
            binding,
            output_dimension=binding.result_state.dimension,
            context=f"vertex kind {binding.key.kind}",
        )
        color = _color_template(
            model,
            vertex,
            closure=False,
            proof_subject=kernel.canonical_signature,
        )
        reflection_claim = _current_word_reflection_claim(
            model,
            binding,
            vertex,
            color,
        )
        flow_variants = _probe_quantum_flows(model, vertex, binding)
        if not flow_variants:
            raise RecurrenceTemplateError(
                "prepared vertex binding is not admitted by the live quantum-flow "
                f"contract: kind={binding.key.kind}, key={binding.key!r}"
            )
        coupling_parameters = _kernel_parameter_ids(kernel, parameter_ids)
        coupling_orders = _canonical_coupling_orders(
            model.vertex_coupling_orders(vertex),
            f"vertex kind {vertex.kind} coupling orders",
        )
        for variant in flow_variants:
            flow = _quantum_flow_template(
                binding,
                variant,
                state_ids,
                coupling_orders,
            )
            for contact_orbit_steps in contact_orbit_step_groups:
                transition_identity = {
                    "binding": _vertex_binding_payload(binding),
                    "flow": flow.semantic_digest,
                    "color": color.semantic_digest,
                    "kernel": kernel.canonical_signature,
                }
                if contact_orbit_steps:
                    transition_identity["contact_orbit_steps"] = [
                        step.semantic_digest for step in contact_orbit_steps
                    ]
                transition_id = _token("transition", transition_identity)
                resolver_key = _resolver_key(
                    kernel,
                    "vertex",
                    concrete_state_ids,
                    result_state_id,
                )
                transition = TransitionTemplateV1(
                    template_id=transition_id,
                    input_state_template_ids=concrete_state_ids,
                    result_state_template_id=result_state_id,
                    quantum_flow_template_id=flow.template_id,
                    evaluator_resolver_key=resolver_key,
                    canonical_input_order=tuple(binding.canonical_input_order),
                    momentum_convention=("incoming-left", "incoming-right"),
                    coupling_parameter_ids=coupling_parameters,
                    coupling_orders=coupling_orders,
                    color_contraction_template_id=color.template_id,
                    binding_coupling=_exact_pair(
                        binding.key.coupling,
                        f"vertex kind {vertex.kind} binding coupling",
                    ),
                    exact_factor=_exact_pair(
                        binding.equivalence_factor,
                        f"vertex kind {vertex.kind} equivalence factor",
                    ),
                    output_factor_source=binding.output_factor_source,
                    equivalence_class=binding.equivalence_class,
                    input_exchange_factor=(
                        None
                        if binding.input_exchange_factor is None
                        else _exact_pair(
                            binding.input_exchange_factor,
                            f"vertex kind {vertex.kind} input-exchange factor",
                        )
                    ),
                    output_projection=f"{binding.result_state.basis}:chirality={binding.result_state.chirality}",
                    contact_orbit_step_template_ids=tuple(
                        step.template_id for step in contact_orbit_steps
                    ),
                    contact_orbit_step_semantic_digests=tuple(
                        step.semantic_digest for step in contact_orbit_steps
                    ),
                )
                candidates.append(
                    (
                        _canonical_transition_alias_key(
                            binding=binding,
                            kernel=kernel,
                            flow=flow,
                            color=color,
                            transition=transition,
                        ),
                        _transition_alias_preference(model, binding, transition),
                        transition,
                        flow,
                        color,
                        kernel,
                        concrete_state_ids,
                        result_state_id,
                        reflection_claim,
                    )
                )

    # Oriented model catalogs may contain exact mirrored aliases.  Collapse only
    # rows whose complete prepared callable, quantum-flow, color, coupling, and
    # sign contracts agree after the certified evaluator input permutation.  In
    # particular, never aggregate alias coefficients: doing so would double a
    # single physical recurrence contribution.
    selected: dict[
        str,
        tuple[
            tuple[int, int, str],
            TransitionTemplateV1,
            QuantumFlowTemplateV1,
            ColorContractionTemplateV1,
            Any,
            tuple[str, ...],
            str,
            tuple[ExactComplexRationalV1, str] | None,
        ],
    ] = {}
    for (
        alias_key,
        preference,
        transition,
        flow,
        color,
        kernel,
        concrete_state_ids,
        result_state_id,
        reflection_claim,
    ) in candidates:
        candidate = (
            preference,
            transition,
            flow,
            color,
            kernel,
            concrete_state_ids,
            result_state_id,
            reflection_claim,
        )
        existing = selected.get(alias_key)
        if existing is None or preference < existing[0]:
            selected[alias_key] = candidate

    flows_by_id: dict[str, QuantumFlowTemplateV1] = {}
    colors: dict[str, ColorContractionTemplateV1] = {}
    transitions: list[TransitionTemplateV1] = []
    proofs: list[SymmetryProofV1] = []
    for alias_key in sorted(selected):
        (
            _,
            transition,
            flow,
            color,
            kernel,
            concrete_state_ids,
            result_state_id,
            reflection_claim,
        ) = selected[alias_key]
        flows_by_id.setdefault(flow.template_id, flow)
        colors.setdefault(color.template_id, color)
        transitions.append(transition)
        if reflection_claim is not None:
            proofs.append(
                _current_word_reflection_proof(
                    transition=transition,
                    kernel=kernel,
                    phase=reflection_claim[0],
                    certificate_witness_digest=reflection_claim[1],
                )
            )
        evaluator_requests.append(
            _EvaluatorRequest(
                kernel=kernel,
                contract_kind="vertex",
                input_state_template_ids=concrete_state_ids,
                output_state_template_id=result_state_id,
                semantic_template_id=transition.template_id,
            )
        )
    return (
        tuple(flows_by_id.values()),
        tuple(transitions),
        tuple(colors.values()),
        tuple(proofs),
    )


def _current_word_reflection_claim(
    model: Model,
    binding: PreparedVertexBinding,
    vertex: Vertex,
    color: ColorContractionTemplateV1,
) -> tuple[ExactComplexRationalV1, str] | None:
    if color.input_representations != (8, 8) or color.output_representation != 8:
        return None
    context = f"adjoint-current reflection phase for vertex kind {vertex.kind}"

    def phase_payload(value: object) -> dict[str, str] | None:
        if value is None:
            return None
        return _exact_pair(value, context).to_dict()

    callback_value = _stable_callback(
        context,
        lambda: model.adjoint_current_reflection_phase(vertex),
        serializer=lambda candidate: _canonical_json(phase_payload(candidate)),
    )
    callback_phase = (
        None if callback_value is None else _exact_pair(callback_value, context)
    )
    prepared_phase = (
        None
        if binding.input_exchange_factor is None
        else _exact_pair(
            binding.input_exchange_factor,
            f"prepared input-exchange factor for vertex kind {vertex.kind}",
        )
    )
    if prepared_phase is not None and binding.left_state != binding.right_state:
        prepared_phase = None
    color_claim = _adjoint_color_word_reflection_claim(color)
    color_phase = None if color_claim is None else color_claim[0]
    phases = tuple(
        candidate
        for candidate in (callback_phase, prepared_phase, color_phase)
        if candidate is not None
    )
    if not phases:
        return None
    if any(candidate != phases[0] for candidate in phases[1:]):
        raise RecurrenceTemplateError(
            f"{context} has conflicting prepared, model, or color certificates"
        )
    phase = phases[0]
    if phase == ExactComplexRationalV1.zero():
        raise RecurrenceTemplateError(f"{context} must be nonzero")
    certificate_payload = {
        "vertex": {
            "kind": int(vertex.kind),
            "particles": [int(particle_id) for particle_id in vertex.particles],
            "coupling": _exact_pair(
                vertex.coupling,
                f"vertex kind {vertex.kind} reflection coupling",
            ).to_dict(),
        },
        "exact_phase": phase.to_dict(),
        "prepared_input_exchange": (
            None
            if prepared_phase is None
            else {
                "equivalence_class": str(binding.equivalence_class),
                "canonical_input_order": [
                    int(index) for index in binding.canonical_input_order
                ],
                "factor": prepared_phase.to_dict(),
                "left_state": {
                    "particle_id": int(binding.left_state.particle_id),
                    "identity": str(binding.left_state.identity),
                    **binding.left_state.contract_dict(),
                },
                "right_state": {
                    "particle_id": int(binding.right_state.particle_id),
                    "identity": str(binding.right_state.identity),
                    **binding.right_state.contract_dict(),
                },
            }
        ),
        "model_callback": (
            None
            if callback_phase is None
            else {
                "contract": "adjoint_current_reflection_phase",
                "factor": callback_phase.to_dict(),
            }
        ),
        "color_word_reversal": (
            None
            if color_claim is None
            else {
                "factor": color_phase.to_dict(),
                "witness_digest": color_claim[1],
            }
        ),
    }
    return phase, _digest(certificate_payload)


def _adjoint_color_word_reflection_claim(
    color: ColorContractionTemplateV1,
) -> tuple[ExactComplexRationalV1, str] | None:
    """Prove reversal from paired exact ordered-word color witnesses."""

    if (
        color.input_representations != (8, 8)
        or color.output_representation != 8
        or len(color.transition_witnesses) != 2
    ):
        return None
    by_permutation = {
        witness.input_permutation: witness for witness in color.transition_witnesses
    }
    if set(by_permutation) != {(0, 1), (1, 0)}:
        return None
    forward = by_permutation[(0, 1)]
    reverse = by_permutation[(1, 0)]

    def swap_port(port: tuple[int, int]) -> tuple[int, int]:
        return (1 - port[0], port[1])

    swapped_pairings = tuple(
        tuple(swap_port(port) for port in pairing)
        for pairing in forward.input_port_pairings
    )
    swapped_bindings = tuple(swap_port(port) for port in forward.result_port_bindings)
    swapped_reverse_mask = ((forward.reverse_parent_mask & 1) << 1) | (
        (forward.reverse_parent_mask & 2) >> 1
    )
    if (
        reverse.input_shape_kinds != forward.input_shape_kinds
        or reverse.reverse_parent_mask != swapped_reverse_mask
        or reverse.component_operation != "concatenate-join"
        or forward.component_operation != "concatenate-join"
        or reverse.result_component_kind != forward.result_component_kind
        or reverse.result_component_role != forward.result_component_role
        or reverse.result_shape_kind != forward.result_shape_kind
        or reverse.input_port_pairings != swapped_pairings
        or reverse.result_port_bindings != swapped_bindings
    ):
        return None

    one = ExactComplexRationalV1.from_binary64(1.0)
    minus_one = ExactComplexRationalV1.from_binary64(-1.0)
    if reverse.exact_factor == forward.exact_factor:
        phase = one
    elif reverse.exact_factor == _multiply_exact(forward.exact_factor, minus_one):
        phase = minus_one
    else:
        return None
    witness = {
        "proof_algorithm": "paired-adjoint-color-word-reversal-v1",
        "color_semantic_digest": color.semantic_digest,
        "forward": forward.to_dict(),
        "reverse": reverse.to_dict(),
        "exact_phase": phase.to_dict(),
    }
    return phase, _digest(witness)


def _current_word_reflection_proof(
    *,
    transition: TransitionTemplateV1,
    kernel: Any,
    phase: ExactComplexRationalV1,
    certificate_witness_digest: str,
) -> SymmetryProofV1:
    expression_digests = tuple(
        _expression_digest(expression) for expression in kernel.exact_expressions
    )
    witness = {
        "proof_algorithm": _CURRENT_WORD_REFLECTION_PROOF,
        "subject_template_id": transition.template_id,
        "input_permutation": [1, 0],
        "exact_phase": phase.to_dict(),
        "prepared_kernel": {
            "contract_kind": kernel.contract_kind,
            "canonical_signature": kernel.canonical_signature,
            "expression_digests": list(expression_digests),
        },
        "certificate": {
            "prepared_input_exchange_or_model_callback": True,
            "witness_digest": certificate_witness_digest,
        },
    }
    return SymmetryProofV1(
        template_id=_token("proof", witness),
        proof_algorithm=_CURRENT_WORD_REFLECTION_PROOF,
        subject_template_ids=(transition.template_id,),
        input_permutation=(1, 0),
        exact_phase=phase,
        expression_digests=expression_digests,
        witness_digest=_digest(witness),
    )


def _canonical_transition_alias_key(
    *,
    binding: PreparedVertexBinding,
    kernel: Any,
    flow: QuantumFlowTemplateV1,
    color: ColorContractionTemplateV1,
    transition: TransitionTemplateV1,
) -> str:
    """Return an exact, orientation-independent transition proof key."""

    order = tuple(int(value) for value in binding.canonical_input_order)
    inverse_order = tuple(order.index(index) for index in (0, 1))
    contribution_factor = _multiply_exact(
        transition.exact_factor,
        color.exact_coefficient,
    )

    def reordered(values: Sequence[Any]) -> list[Any]:
        return [values[index] for index in order]

    color_witnesses = []
    for witness in color.transition_witnesses:
        reverse_parent_mask = 0
        for concrete_index in range(2):
            if witness.reverse_parent_mask & (1 << concrete_index):
                reverse_parent_mask |= 1 << inverse_order[concrete_index]
        canonical_witness_order = tuple(
            inverse_order[index] for index in witness.input_permutation
        )
        if witness.component_operation == "inherit-left":
            operation = f"inherit-input-{canonical_witness_order[0]}"
            operation_order: list[int] = []
        elif witness.component_operation == "inherit-right":
            operation = f"inherit-input-{canonical_witness_order[1]}"
            operation_order = []
        elif witness.component_operation == "empty":
            operation = witness.component_operation
            operation_order = []
        elif witness.component_operation == "concatenate-keep":
            operation = witness.component_operation
            operation_order = list(canonical_witness_order)
        else:
            operation = witness.component_operation
            operation_order = list(canonical_witness_order)
        color_witnesses.append(
            {
                "input_shape_kinds": reordered(witness.input_shape_kinds),
                "input_permutation": operation_order,
                "reverse_parent_mask": reverse_parent_mask,
                "component_operation": operation,
                "result_component_kind": witness.result_component_kind,
                "result_component_role": witness.result_component_role,
                "result_shape_kind": witness.result_shape_kind,
                # Transition orientation and LC witness signs are one exact
                # contribution coefficient.  Keeping them as independent key
                # fields prevents certified mirrored aliases such as
                # (tensor, vector) and (vector, tensor) from comparing equal,
                # even when their products are identical.
                "exact_contribution_factor": _multiply_exact(
                    contribution_factor,
                    witness.exact_factor,
                ).to_dict(),
                "input_port_pairings": [
                    [
                        [inverse_order[left[0]], left[1]],
                        [inverse_order[right[0]], right[1]],
                    ]
                    for left, right in witness.input_port_pairings
                ],
                "result_port_bindings": [
                    [inverse_order[parent], port]
                    for parent, port in witness.result_port_bindings
                ],
            }
        )
    color_witnesses.sort(key=_canonical_json)

    payload = {
        "prepared_kernel": {
            # `_validate_vertex_binding_against_model` has already required a
            # stable, verified model-owned permutation/sign certificate.
            # Mirrored oriented kernels may have different callable digests
            # even though that certificate proves the same canonical
            # evaluation class.  Keying on the digest would retain both rows
            # and double the recurrence contribution.
            "verified_equivalence_class": transition.equivalence_class,
            "contract_kind": str(kernel.contract_kind),
        },
        "input_state_template_ids": reordered(transition.input_state_template_ids),
        "result_state_template_id": transition.result_state_template_id,
        "quantum_flow": {
            "input_state_template_ids": reordered(flow.input_state_template_ids),
            "input_spin_states": reordered(flow.input_spin_states),
            "input_flavour_flows": [
                list(value) for value in reordered(flow.input_flavour_flows)
            ],
            "input_quantum_number_flows": [
                [list(item) for item in value]
                for value in reordered(flow.input_quantum_number_flows)
            ],
            **_canonical_flavour_flow_payload(flow, color, order),
            "quantum_number_flow_operation": flow.quantum_number_flow_operation,
            "coupling_orders": [list(item) for item in flow.coupling_orders],
            "result_state_template_id": flow.result_state_template_id,
            "result_spin_state": flow.result_spin_state,
            "result_quantum_number_flow": [
                list(item) for item in flow.result_quantum_number_flow
            ],
            "exact_coupling": flow.exact_coupling.to_dict(),
        },
        "color": {
            "rule_kind": color.rule_kind,
            "input_representations": reordered(color.input_representations),
            "output_representation": color.output_representation,
            "ordered_open_string_arity": color.ordered_open_string_arity,
            "nc_polynomial": [
                [power, coefficient.to_dict()]
                for power, coefficient in color.nc_polynomial
            ],
            "transition_witnesses": color_witnesses,
        },
        "momentum_semantics": _canonical_momentum_semantics(
            transition.momentum_convention,
            order,
        ),
        "coupling_parameter_ids": list(transition.coupling_parameter_ids),
        "contact_orbit_step_template_ids": list(
            getattr(transition, "contact_orbit_step_template_ids", ())
        ),
        "contact_orbit_step_semantic_digests": list(
            getattr(transition, "contact_orbit_step_semantic_digests", ())
        ),
        "coupling_orders": [list(item) for item in transition.coupling_orders],
        "binding_coupling": transition.binding_coupling.to_dict(),
        "output_factor_source": transition.output_factor_source,
        "equivalence_class": transition.equivalence_class,
        "input_exchange_factor": (
            None
            if transition.input_exchange_factor is None
            else transition.input_exchange_factor.to_dict()
        ),
        "output_projection": transition.output_projection,
    }
    return _canonical_json(payload)


def _canonical_flavour_flow_payload(
    flow: QuantumFlowTemplateV1,
    color: ColorContractionTemplateV1,
    canonical_input_order: tuple[int, int],
) -> dict[str, object]:
    """Express flavour ancestry in certified canonical input coordinates."""

    inverse_order = tuple(canonical_input_order.index(index) for index in (0, 1))
    inputs = tuple(flow.input_flavour_flows[index] for index in canonical_input_order)
    operation = flow.flavour_flow_operation
    if operation == "constant-result":
        return {
            "flavour_flow_operation": operation,
            "result_flavour_flow": list(flow.result_flavour_flow),
        }

    if not flow.result_flavour_flow:
        raise RecurrenceTemplateError(
            "recurrence quantum-flow result ancestry cannot be empty"
        )
    result_particle = flow.result_flavour_flow[-1]
    if operation in {"append-left-result", "append-right-result"}:
        concrete_parent = 0 if operation == "append-left-result" else 1
        canonical_parent = inverse_order[concrete_parent]
        ancestry = inputs[canonical_parent]
        result = (
            ancestry
            if ancestry and ancestry[-1] == result_particle
            else (*ancestry, result_particle)
        )
        return {
            "flavour_flow_operation": f"append-input-{canonical_parent}-result",
            "result_flavour_flow": list(result),
        }

    if operation != "concat-left-right-result":
        raise RecurrenceTemplateError(
            f"unsupported recurrence flavour operation {operation!r}"
        )

    parent_orders: set[tuple[int, int]] = set()
    for witness in color.transition_witnesses:
        concrete_order: list[int] = []
        for left, right in witness.input_port_pairings:
            concrete_order.extend((left[0], right[0]))
        concrete_order.extend(parent for parent, _ in witness.result_port_bindings)
        unique_order = tuple(dict.fromkeys(concrete_order))
        if len(unique_order) == 2 and set(unique_order) == {0, 1}:
            parent_orders.add(tuple(inverse_order[index] for index in unique_order))
    canonical_parent_order = (
        next(iter(parent_orders)) if len(parent_orders) == 1 else (0, 1)
    )
    result = (
        *(value for parent in canonical_parent_order for value in inputs[parent]),
        result_particle,
    )
    return {
        "flavour_flow_operation": (
            "concat-inputs-"
            + "-".join(str(index) for index in canonical_parent_order)
            + "-result"
        ),
        "result_flavour_flow": list(result),
    }


def _canonical_momentum_semantics(
    conventions: Sequence[str],
    canonical_input_order: tuple[int, int],
) -> list[str]:
    result: list[str] = []
    for concrete_index in canonical_input_order:
        value = str(conventions[concrete_index])
        side = "left" if concrete_index == 0 else "right"
        suffix = f"-{side}"
        result.append(value[: -len(suffix)] if value.endswith(suffix) else value)
    return result


def _transition_alias_preference(
    model: Model,
    binding: PreparedVertexBinding,
    transition: TransitionTemplateV1,
) -> tuple[int, int, str]:
    canonical_order_penalty = int(tuple(binding.canonical_input_order) != (0, 1))
    left_colored = abs(int(model.color_rep(binding.left_state.particle_id))) != 1
    right_singlet = abs(int(model.color_rep(binding.right_state.particle_id))) == 1
    return (
        0 if left_colored and right_singlet else 1,
        canonical_order_penalty,
        transition.template_id,
    )


def _quantum_flow_template(
    binding: PreparedVertexBinding | PreparedClosureBinding,
    variant: Mapping[str, Any],
    state_ids: Mapping[tuple[int, int], str],
    coupling_orders: tuple[tuple[str, int], ...],
) -> QuantumFlowTemplateV1:
    input_state_ids = (
        state_ids[(binding.left_state.particle_id, binding.left_state.chirality)],
        state_ids[(binding.right_state.particle_id, binding.right_state.chirality)],
    )
    result_state_id = state_ids[
        (binding.result_state.particle_id, binding.result_state.chirality)
    ]
    predicate = {
        "coupling_orders": [list(item) for item in coupling_orders],
        "flow": variant,
        "input_state_template_ids": list(input_state_ids),
        "result_state_template_id": result_state_id,
    }
    return QuantumFlowTemplateV1(
        template_id=_token("quantum-flow", predicate),
        input_state_template_ids=input_state_ids,
        input_spin_states=variant["input_spin_states"],
        input_flavour_flows=variant["input_flavour_flows"],
        input_quantum_number_flows=variant["input_quantum_number_flows"],
        flavour_flow_operation=variant["flavour_flow_operation"],
        quantum_number_flow_operation=variant["quantum_number_flow_operation"],
        coupling_orders=coupling_orders,
        result_state_template_id=result_state_id,
        result_spin_state=variant["result_spin_state"],
        result_flavour_flow=variant["result_flavour_flow"],
        result_quantum_number_flow=variant["result_quantum_number_flow"],
        exact_coupling=ExactComplexRationalV1.from_dict(variant["coupling"]),
        predicate_digest=_digest(predicate),
    )


def _validate_vertex_binding_against_model(
    model: Model,
    binding: PreparedVertexBinding,
    vertex: Vertex,
) -> None:
    equivalence = _stable_callback(
        f"vertex equivalence for kind {vertex.kind}",
        lambda: model.vertex_evaluation_equivalence(vertex.kind),
        serializer=lambda value: _canonical_json(value.to_json_dict()),
    )
    expected = (
        tuple(equivalence.input_order),
        _exact_pair(equivalence.factor, "vertex equivalence factor"),
        None
        if equivalence.input_exchange_factor is None
        else _exact_pair(
            equivalence.input_exchange_factor,
            "vertex input-exchange factor",
        ),
        str(equivalence.class_id),
        bool(equivalence.verified),
    )
    actual = (
        tuple(binding.canonical_input_order),
        _exact_pair(binding.equivalence_factor, "prepared equivalence factor"),
        None
        if binding.input_exchange_factor is None
        else _exact_pair(
            binding.input_exchange_factor,
            "prepared input-exchange factor",
        ),
        str(binding.equivalence_class),
        True,
    )
    if expected != actual:
        raise PreparedKernelCatalogError(
            f"prepared vertex binding for kind {vertex.kind} has stale "
            "permutation/sign proof metadata"
        )
    expected_contact = _stable_callback(
        f"contact-orbit contract for kind {vertex.kind}",
        lambda: model.vertex_contact_orbit_contracts(vertex.kind),
        serializer=lambda value: _canonical_json(
            [[item.to_dict() for item in group] for group in value]
        ),
    )
    actual_contact = (
        binding.contact_orbit_certificates,
        binding.contact_orbit_steps,
    )
    if expected_contact != actual_contact:
        raise PreparedKernelCatalogError(
            f"prepared vertex binding for kind {vertex.kind} has stale "
            "contact-orbit proof metadata"
        )


def _probe_quantum_flows(
    model: Model,
    vertex: Vertex,
    binding: PreparedVertexBinding | PreparedClosureBinding,
) -> tuple[dict[str, Any], ...]:
    contract = _recurrence_quantum_flow_contract(model, vertex, binding)
    static_result_quantum_flow = tuple(
        (str(name), str(expression))
        for name, expression in _stable_callback(
            f"quantum-number flow for result particle "
            f"{binding.result_state.particle_id}",
            lambda: model.quantum_number_flow(binding.result_state.particle_id),
            serializer=lambda value: _canonical_json(list(value)),
        )
    )
    left_spins = _spin_states_for_chirality(
        model, binding.left_state.particle_id, binding.left_state.chirality
    )
    right_spins = _spin_states_for_chirality(
        model, binding.right_state.particle_id, binding.right_state.chirality
    )
    variants: dict[str, dict[str, Any]] = {}
    for left_spin in left_spins:
        for right_spin in right_spins:
            baseline_left = _flow_probe_index(model, binding.left_state, left_spin)
            baseline_right = _flow_probe_index(model, binding.right_state, right_spin)

            def evaluate(
                left: SimpleNamespace,
                right: SimpleNamespace,
            ) -> tuple[dict[str, Any], ...]:
                try:
                    flows = model.allowed_quantum_flows(vertex, left, right)
                except Exception as exc:
                    raise RecurrenceTemplateError(
                        "cannot evaluate model quantum-flow callback for "
                        f"vertex kind {vertex.kind}: {exc}"
                    ) from exc
                rows: list[dict[str, Any]] = []
                for flow in flows:
                    if int(flow.chirality) != binding.result_state.chirality:
                        continue
                    if not isinstance(flow.spin_state, int) or isinstance(
                        flow.spin_state, bool
                    ):
                        raise RecurrenceTemplateError(
                            "recurrence-template-v1 cannot encode structured "
                            f"quantum-flow spin state {flow.spin_state!r}"
                        )
                    rows.append(
                        {
                            "input_spin_states": (
                                int(left.spin_state),
                                int(right.spin_state),
                            ),
                            "input_flavour_flows": (
                                tuple(int(value) for value in left.flavour_flow),
                                tuple(int(value) for value in right.flavour_flow),
                            ),
                            "input_quantum_number_flows": (
                                tuple(
                                    (str(name), str(expression))
                                    for name, expression in left.quantum_number_flow
                                ),
                                tuple(
                                    (str(name), str(expression))
                                    for name, expression in right.quantum_number_flow
                                ),
                            ),
                            "result_chirality": int(flow.chirality),
                            "result_spin_state": int(flow.spin_state),
                            "result_flavour_flow": tuple(
                                int(value) for value in flow.flavour_flow
                            ),
                            "result_quantum_number_flow": tuple(
                                (str(name), str(expression))
                                for name, expression in flow.quantum_number_flow
                            ),
                            "coupling": _exact_pair(
                                flow.coupling,
                                f"vertex kind {vertex.kind} quantum-flow coupling",
                            ).to_dict(),
                        }
                    )
                return tuple(sorted(rows, key=_canonical_json))

            baseline = _stable_quantum_flow_probe(
                evaluate,
                baseline_left,
                baseline_right,
                vertex_kind=vertex.kind,
            )
            probe_pairs = (
                (
                    _flow_probe_index(
                        model,
                        binding.left_state,
                        left_spin,
                        flavour_flow=(1_870_001, 1_870_002),
                        quantum_number_flow=(("__RECURRENCE_LEFT__", "17"),),
                        coupling_orders=(("QCD", 2),),
                    ),
                    _flow_probe_index(
                        model,
                        binding.right_state,
                        right_spin,
                        flavour_flow=(1_871_001, 1_871_002),
                        quantum_number_flow=(("__RECURRENCE_RIGHT__", "19"),),
                        coupling_orders=(("QED", 1),),
                    ),
                ),
                (
                    _flow_probe_index(
                        model,
                        binding.left_state,
                        left_spin,
                        flavour_flow=(1_872_001, 1_872_002, 1_872_003),
                        quantum_number_flow=(("electric_charge", "23/7"),),
                        coupling_orders=(("__RECURRENCE_LEFT__", 1),),
                    ),
                    _flow_probe_index(
                        model,
                        binding.right_state,
                        right_spin,
                        flavour_flow=(1_873_001, 1_873_002, 1_873_003),
                        quantum_number_flow=(("color_charge", "29/11"),),
                        coupling_orders=(("__RECURRENCE_RIGHT__", 2),),
                    ),
                ),
            )
            probes = tuple(
                _stable_quantum_flow_probe(
                    evaluate,
                    left,
                    right,
                    vertex_kind=vertex.kind,
                )
                for left, right in probe_pairs
            )
            baseline_by_key = _quantum_flow_rows_by_invariant_key(
                baseline, vertex_kind=vertex.kind
            )
            probe_maps = tuple(
                _quantum_flow_rows_by_invariant_key(rows, vertex_kind=vertex.kind)
                for rows in probes
            )
            if any(set(rows) != set(baseline_by_key) for rows in probe_maps):
                raise RecurrenceTemplateError(
                    "recurrence-template-v1 requires quantum-flow admission to be "
                    "independent of accumulated flavour, quantum-number, and "
                    "coupling-order flows; "
                    f"vertex kind {vertex.kind} violates that contract"
                )
            for key, row in baseline_by_key.items():
                for left, right, rows in (
                    (baseline_left, baseline_right, baseline_by_key),
                    *(
                        (left, right, rows)
                        for (left, right), rows in zip(
                            probe_pairs, probe_maps, strict=True
                        )
                    ),
                ):
                    expected = _apply_flavour_flow_operation(
                        contract.flavour_flow_operation,
                        tuple(int(value) for value in left.flavour_flow),
                        tuple(int(value) for value in right.flavour_flow),
                        binding.result_state.particle_id,
                    )
                    candidate = rows[key]
                    if tuple(candidate["result_flavour_flow"]) != expected:
                        raise RecurrenceTemplateError(
                            "quantum-flow callback contradicts its declared "
                            f"flavour operation for vertex kind {vertex.kind}"
                        )
                    if (
                        tuple(candidate["result_quantum_number_flow"])
                        != static_result_quantum_flow
                    ):
                        raise RecurrenceTemplateError(
                            "quantum-flow callback contradicts its declared "
                            "particle-static quantum-number operation for vertex "
                            f"kind {vertex.kind}"
                        )
                enriched = dict(row)
                enriched["flavour_flow_operation"] = contract.flavour_flow_operation
                enriched["quantum_number_flow_operation"] = (
                    contract.quantum_number_flow_operation
                )
                variants.setdefault(_canonical_json(enriched), enriched)
    return tuple(variants[key] for key in sorted(variants))


def _stable_quantum_flow_probe(
    evaluate: Any,
    left: SimpleNamespace,
    right: SimpleNamespace,
    *,
    vertex_kind: int,
) -> tuple[dict[str, Any], ...]:
    first = evaluate(left, right)
    second = evaluate(left, right)
    if _canonical_json(first) != _canonical_json(second):
        raise RecurrenceTemplateError(
            f"allowed_quantum_flows is nondeterministic for vertex kind {vertex_kind}"
        )
    return first


def _quantum_flow_rows_by_invariant_key(
    rows: Sequence[Mapping[str, Any]],
    *,
    vertex_kind: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key_payload = {
            name: value
            for name, value in row.items()
            if name
            not in {
                "input_flavour_flows",
                "input_quantum_number_flows",
                "result_flavour_flow",
                "result_quantum_number_flow",
            }
        }
        key = _canonical_json(key_payload)
        if key in result:
            raise RecurrenceTemplateError(
                "recurrence-template-v1 cannot distinguish multiple quantum-flow "
                f"branches for vertex kind {vertex_kind}"
            )
        result[key] = dict(row)
    return result


def _recurrence_quantum_flow_contract(
    model: Model,
    vertex: Vertex,
    binding: PreparedVertexBinding | PreparedClosureBinding,
) -> RecurrenceQuantumFlowContract:
    model_type = type(model)
    mro = model_type.__mro__
    allowed_owner = next(
        owner for owner in mro if "allowed_quantum_flows" in owner.__dict__
    )
    combine_owner = next(
        owner for owner in mro if "combine_flavour_flow" in owner.__dict__
    )
    contract_owner = next(
        owner for owner in mro if "recurrence_quantum_flow_contract" in owner.__dict__
    )
    if mro.index(contract_owner) > mro.index(allowed_owner):
        raise RecurrenceTemplateError(
            f"{model_type.__name__}.allowed_quantum_flows overrides the callback "
            "without declaring a matching recurrence quantum-flow contract"
        )
    if mro.index(contract_owner) > mro.index(combine_owner):
        raise RecurrenceTemplateError(
            f"{model_type.__name__}.combine_flavour_flow overrides the callback "
            "without declaring a matching recurrence quantum-flow contract"
        )
    try:
        contract = _stable_callback(
            f"recurrence quantum-flow contract for vertex kind {vertex.kind}",
            lambda: model.recurrence_quantum_flow_contract(
                vertex,
                binding.left_state.particle_id,
                binding.right_state.particle_id,
            ),
            serializer=lambda value: _canonical_json(value.to_json_dict()),
        )
    except Exception as exc:
        raise RecurrenceTemplateError(
            "model does not provide a recurrence quantum-flow contract for "
            f"vertex kind {vertex.kind}: {exc}"
        ) from exc
    if not isinstance(contract, RecurrenceQuantumFlowContract):
        raise RecurrenceTemplateError(
            "model recurrence quantum-flow contract must be a "
            f"RecurrenceQuantumFlowContract for vertex kind {vertex.kind}"
        )
    return contract


def _apply_flavour_flow_operation(
    operation: FlavourFlowOperationV1 | str,
    left: tuple[int, ...],
    right: tuple[int, ...],
    result_particle: int,
) -> tuple[int, ...]:
    if operation == "constant-result":
        return (int(result_particle),)
    if operation == "append-left-result":
        return (
            left if left and left[-1] == result_particle else (*left, result_particle)
        )
    if operation == "append-right-result":
        return (
            right
            if right and right[-1] == result_particle
            else (*right, result_particle)
        )
    if operation == "concat-left-right-result":
        return (*left, *right, result_particle)
    raise AssertionError(f"unknown flavour-flow operation {operation!r}")


def _spin_states_for_chirality(
    model: Model,
    particle_id: int,
    chirality: int,
) -> tuple[int, ...]:
    if _effective_auxiliary_kind(model, particle_id, chirality) is not None:
        result = model.result_spin_state(particle_id, chirality)
        if not isinstance(result, int) or isinstance(result, bool):
            raise RecurrenceTemplateError(
                "recurrence-template-v1 cannot encode structured auxiliary "
                "result spin state"
            )
        return (int(result),)
    values: set[int] = set()
    for state in model.source_spin_states(particle_id):
        if int(state.chirality) != chirality:
            continue
        if not isinstance(state.spin_state, int) or isinstance(state.spin_state, bool):
            raise RecurrenceTemplateError(
                "recurrence-template-v1 cannot encode structured source spin state "
                f"{state.spin_state!r}"
            )
        values.add(int(state.spin_state))
    result = model.result_spin_state(particle_id, chirality)
    if not isinstance(result, int) or isinstance(result, bool):
        raise RecurrenceTemplateError(
            "recurrence-template-v1 cannot encode structured result spin state"
        )
    values.add(int(result))
    return tuple(sorted(values))


def _flow_probe_index(
    model: Model,
    state: PreparedParticleState,
    spin_state: int,
    *,
    flavour_flow: tuple[int, ...] | None = None,
    quantum_number_flow: tuple[tuple[str, str], ...] | None = None,
    coupling_orders: tuple[tuple[str, int], ...] = (),
) -> SimpleNamespace:
    quantum_flow = (
        _stable_callback(
            f"quantum-number flow for particle {state.particle_id}",
            lambda: model.quantum_number_flow(state.particle_id),
            serializer=lambda value: _canonical_json(list(value)),
        )
        if quantum_number_flow is None
        else quantum_number_flow
    )
    return SimpleNamespace(
        particle_id=state.particle_id,
        pdg=state.particle_id,
        chirality=state.chirality,
        spin_state=spin_state,
        flavour_flow=(state.particle_id,) if flavour_flow is None else flavour_flow,
        quantum_number_flow=quantum_flow,
        coupling_orders=coupling_orders,
    )


def _canonical_coupling_orders(
    value: Sequence[tuple[str, int]],
    context: str,
) -> tuple[tuple[str, int], ...]:
    result = tuple(sorted((str(name).upper(), int(power)) for name, power in value))
    if len({name for name, _ in result}) != len(result) or any(
        power < 0 for _, power in result
    ):
        raise RecurrenceTemplateError(f"{context} is not canonical")
    return result


def _color_template(
    model: Model,
    vertex: Vertex,
    *,
    closure: bool,
    proof_subject: str,
) -> ColorContractionTemplateV1:
    model_type = type(model)
    mro = model_type.__mro__
    structure_owner = next(
        owner for owner in mro if "vertex_color_structure" in owner.__dict__
    )
    contract_owner = next(
        owner
        for owner in mro
        if "recurrence_lc_color_transition_contract" in owner.__dict__
    )
    if mro.index(contract_owner) > mro.index(structure_owner):
        raise RecurrenceTemplateError(
            f"{model_type.__name__}.vertex_color_structure overrides the color "
            "tensor callback without declaring a matching recurrence LC color contract"
        )

    def evaluate() -> tuple[
        str,
        tuple[int, ...],
        ExactComplexRationalV1,
        tuple[Any, ...],
    ]:
        contract = model.recurrence_lc_color_transition_contract(
            vertex,
            closure=closure,
        )
        structure = str(contract.rule_kind)
        representations = tuple(
            int(model.color_rep(particle)) for particle in vertex.particles
        )
        if structure not in _SUPPORTED_COLOR_RULES:
            raise RecurrenceTemplateError(
                f"recurrence-template-v1 cannot encode color rule {structure!r} "
                f"for vertex kind {vertex.kind}"
            )
        coefficient = _exact_pair(
            model.vertex_color_weight(vertex, color_accuracy="lc"),
            f"vertex kind {vertex.kind} LC color coefficient",
        )
        return structure, representations, coefficient, tuple(contract.witnesses)

    first = evaluate()
    second = evaluate()
    if first != second:
        raise RecurrenceTemplateError(
            f"color callback is nondeterministic for vertex kind {vertex.kind}"
        )
    structure, representations, coefficient, witness_contracts = first
    output_representation = None if closure else representations[2]
    input_shapes = tuple(
        model.recurrence_lc_color_shape_contract(particle)
        for particle in vertex.particles[:2]
    )
    result_shape = (
        None
        if closure
        else model.recurrence_lc_color_shape_contract(vertex.particles[2])
    )
    port_wirings = compile_lc_color_port_wirings(
        "direct-closure" if closure else structure,  # type: ignore[arg-type]
        representations[:2] if closure else representations,
        (
            representations[:2]
            if closure
            else (
                *representations[:2],
                -representations[2]
                if abs(representations[2]) == 3
                else representations[2],
            )
        ),
    )
    if len(port_wirings) != len(witness_contracts):
        raise RecurrenceTemplateError(
            f"vertex kind {vertex.kind} exposes {len(witness_contracts)} LC witness "
            f"terms but its certified port compiler produced {len(port_wirings)}"
        )
    witnesses_list: list[LCColorTransitionWitnessV1] = []
    for witness, wiring in zip(witness_contracts, port_wirings, strict=True):
        exact_factor = _exact_pair(
            witness.exact_factor,
            f"vertex kind {vertex.kind} LC color witness factor",
        )
        expected_port_factor = (
            ExactComplexRationalV1.one()
            if wiring.exact_factor == 1
            else ExactComplexRationalV1(
                real_numerator=-1,
                real_denominator=1,
                imag_numerator=0,
                imag_denominator=1,
            )
        )
        if exact_factor != expected_port_factor:
            raise RecurrenceTemplateError(
                f"vertex kind {vertex.kind} LC witness factor does not match its "
                "certified port-wiring term"
            )
        port_payload = {
            "input_pairings": [
                [
                    [
                        pairing.fundamental.parent_index,
                        pairing.fundamental.local_port_index,
                    ],
                    [
                        pairing.antifundamental.parent_index,
                        pairing.antifundamental.local_port_index,
                    ],
                ]
                for pairing in wiring.input_pairings
            ],
            "result_port_bindings": [
                [port.parent_index, port.local_port_index]
                for port in wiring.result_port_bindings
            ],
            "term_index": wiring.term_index,
        }
        model_proof_digest = witness.proof_digest or _digest(
            {
                "proof_subject": proof_subject,
                "rule_kind": structure,
                "representations": representations,
                "closure": closure,
                "input_shapes": input_shapes,
                "result_shape": result_shape,
                "input_permutation": witness.input_permutation,
                "reverse_parent_mask": witness.reverse_parent_mask,
                "operation": witness.component_operation,
                "result_component_kind": witness.result_component_kind,
                "result_component_role": witness.result_component_role,
                "factor": exact_factor.to_dict(),
            }
        )
        witnesses_list.append(
            LCColorTransitionWitnessV1(
                input_shape_kinds=input_shapes,  # type: ignore[arg-type]
                input_permutation=witness.input_permutation,
                reverse_parent_mask=witness.reverse_parent_mask,
                component_operation=witness.component_operation,
                result_component_kind=witness.result_component_kind,
                result_component_role=witness.result_component_role,
                result_shape_kind=result_shape,
                exact_factor=exact_factor,
                proof_digest=_digest(
                    {
                        "model_proof_digest": model_proof_digest,
                        "port_wiring": port_payload,
                    }
                ),
                provenance=witness.provenance,
                input_port_pairings=tuple(
                    (
                        (
                            pairing.fundamental.parent_index,
                            pairing.fundamental.local_port_index,
                        ),
                        (
                            pairing.antifundamental.parent_index,
                            pairing.antifundamental.local_port_index,
                        ),
                    )
                    for pairing in wiring.input_pairings
                ),
                result_port_bindings=tuple(
                    (port.parent_index, port.local_port_index)
                    for port in wiring.result_port_bindings
                ),
            )
        )
    witnesses = tuple(witnesses_list)
    payload = {
        "rule_kind": structure,
        "input_representations": representations[:2],
        "output_representation": output_representation,
        "coefficient": coefficient.to_dict(),
        "transition_witnesses": [witness.to_dict() for witness in witnesses],
    }
    return ColorContractionTemplateV1(
        template_id=_token("color", payload),
        rule_kind=structure,
        input_representations=representations[:2],
        output_representation=output_representation,
        ordered_open_string_arity=(
            1 if any(abs(value) == 3 for value in representations) else 0
        ),
        exact_coefficient=coefficient,
        nc_polynomial=((0, ExactComplexRationalV1.one()),),
        expression_digest=_digest(payload),
        transition_witnesses=witnesses,
    )


def _infer_model_defined_color_rule(representations: tuple[int, ...]) -> str:
    absolute = tuple(abs(value) for value in representations)
    if all(value == 1 for value in absolute):
        return "singlet"
    if sorted(absolute) == [3, 3, 8]:
        return "fundamental-generator"
    if sorted(absolute) == [1, 3, 3]:
        return "color-identity"
    if absolute == (8, 8, 8):
        return "adjoint-structure-constant"
    raise RecurrenceTemplateError(
        "model-defined color semantics are not reconstructible from generic "
        f"representations {representations!r}; the prepared recurrence companion "
        "must provide an explicit certified color rule"
    )


def _build_propagators(
    model: Model,
    bindings: Sequence[PreparedPropagatorBinding],
    state_ids: Mapping[tuple[int, int], str],
    parameter_ids: Mapping[str, str],
    kernels: Mapping[int, Any],
    evaluator_requests: list[_EvaluatorRequest],
) -> tuple[tuple[PropagatorTemplateV1, ...], tuple[SymmetryProofV1, ...]]:
    records: list[PropagatorTemplateV1] = []
    proofs: list[SymmetryProofV1] = []
    for binding in sorted(bindings, key=lambda item: item.key):
        metadata = _stable_callback(
            f"propagator contract for {binding.key!r}",
            lambda binding=binding: model._propagator_ir(
                binding.key.particle_id, binding.key.chirality
            ),
            serializer=lambda value: _canonical_json(value.to_json_dict()),
        )
        if (
            bool(metadata.applies_propagator) != binding.applies_propagator
            or str(metadata.kind) != binding.propagator_kind
            or str(metadata.mass_class) != binding.mass_class
            or metadata.gauge != binding.gauge
            or metadata.identity.canonical_id != binding.state.identity
            or metadata.basis != binding.state.basis
        ):
            raise PreparedKernelCatalogError(
                f"prepared propagator binding {binding.key!r} is stale"
            )
        state_id = state_ids[(binding.state.particle_id, binding.state.chirality)]
        template_id = _token(
            "propagator",
            {
                "state": state_id,
                "metadata": metadata.to_json_dict(),
                "kernel_id": binding.kernel_id,
            },
        )
        kernel = None
        resolver_key = None
        numerator_digest = None
        denominator_digest = None
        proof_id = None
        if binding.applies_propagator:
            if binding.kernel_id is None:
                raise PreparedKernelCatalogError(
                    f"active propagator {binding.key!r} has no prepared kernel"
                )
            kernel = _kernel(kernels, binding.kernel_id, "propagator")
            if kernel.contract_kind != "propagator":
                raise PreparedKernelCatalogError(
                    f"prepared propagator binding references a "
                    f"{kernel.contract_kind!r} kernel"
                )
            _validate_propagator_kernel_layout(kernel, binding)
            if metadata.numerator is None or metadata.denominator is None:
                raise RecurrenceTemplateError(
                    f"active propagator {binding.key!r} lacks exact formula metadata"
                )
            numerator_digest = _expression_digest(str(metadata.numerator))
            denominator_digest = _expression_digest(str(metadata.denominator))
            resolver_key = _resolver_key(kernel, "propagator", (state_id,), state_id)
            if PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF in kernel.proof_classes:
                proof_id = _token(
                    "proof",
                    {
                        "algorithm": PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF,
                        "subject": template_id,
                        "kernel": kernel.canonical_signature,
                    },
                )
                expression_digests = tuple(
                    _expression_digest(value) for value in kernel.exact_expressions
                )
                proofs.append(
                    SymmetryProofV1(
                        template_id=proof_id,
                        proof_algorithm=PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF,
                        subject_template_ids=(template_id,),
                        input_permutation=(0,),
                        exact_phase=ExactComplexRationalV1.one(),
                        expression_digests=expression_digests,
                        witness_digest=_digest(
                            {
                                "kernel": kernel.canonical_signature,
                                "proof_class": (
                                    PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF
                                ),
                            }
                        ),
                    )
                )
        record = PropagatorTemplateV1(
            template_id=template_id,
            state_template_id=state_id,
            applies_propagator=binding.applies_propagator,
            evaluator_resolver_key=resolver_key,
            numerator_expression_digest=numerator_digest,
            denominator_expression_digest=denominator_digest,
            mass_parameter_id=_parameter_reference(
                metadata.mass_parameter,
                parameter_ids,
                "propagator mass parameter",
            ),
            width_parameter_id=_parameter_reference(
                metadata.width_parameter,
                parameter_ids,
                "propagator width parameter",
            ),
            gauge=metadata.gauge,
            linearity_proof_template_id=proof_id,
        )
        records.append(record)
        if kernel is not None:
            evaluator_requests.append(
                _EvaluatorRequest(
                    kernel=kernel,
                    contract_kind="propagator",
                    input_state_template_ids=(state_id,),
                    output_state_template_id=state_id,
                    semantic_template_id=record.template_id,
                )
            )
    return tuple(records), tuple(proofs)


def _build_closures(
    model: Model,
    bindings: Sequence[PreparedClosureBinding],
    state_ids: Mapping[tuple[int, int], str],
    parameter_ids: Mapping[str, str],
    kernels: Mapping[int, Any],
    evaluator_requests: list[_EvaluatorRequest],
) -> tuple[
    tuple[QuantumFlowTemplateV1, ...],
    tuple[ClosureTemplateV1, ...],
    tuple[ColorContractionTemplateV1, ...],
]:
    flows_by_id: dict[str, QuantumFlowTemplateV1] = {}
    records: list[ClosureTemplateV1] = []
    colors: dict[str, ColorContractionTemplateV1] = {}
    for binding in sorted(bindings, key=lambda item: item.key):
        vertex = Vertex(
            binding.key.kind,
            binding.key.particles,
            binding.key.coupling,
        )
        kernel = _kernel(kernels, binding.kernel_id, "closure")
        if kernel.contract_kind != "closure":
            raise PreparedKernelCatalogError(
                f"prepared closure binding references a {kernel.contract_kind!r} kernel"
            )
        _validate_binary_kernel_layout(
            kernel,
            binding,
            output_dimension=1,
            context=f"closure kind {vertex.kind}",
        )
        contraction = model.closure_contraction_ir(
            binding.result_state.particle_id,
            binding.result_state.chirality,
        )
        if contraction is None or contraction.name != binding.projection:
            raise PreparedKernelCatalogError(
                f"prepared closure binding for kind {vertex.kind} does not match "
                "the live contraction contract"
            )
        color = _color_template(
            model,
            vertex,
            closure=True,
            proof_subject=kernel.canonical_signature,
        )
        colors.setdefault(color.template_id, color)
        coupling_orders = _canonical_coupling_orders(
            model.vertex_coupling_orders(vertex),
            f"closure kind {vertex.kind} coupling orders",
        )
        flow_variants = _probe_quantum_flows(model, vertex, binding)
        if not flow_variants:
            raise RecurrenceTemplateError(
                "prepared closure binding is not admitted by the live quantum-flow "
                f"contract: kind={binding.key.kind}, key={binding.key!r}"
            )
        eligible_flows = tuple(
            _quantum_flow_template(
                binding,
                variant,
                state_ids,
                coupling_orders,
            )
            for variant in flow_variants
        )
        for flow in eligible_flows:
            flows_by_id.setdefault(flow.template_id, flow)
        input_state_ids = (
            state_ids[(binding.left_state.particle_id, binding.left_state.chirality)],
            state_ids[(binding.right_state.particle_id, binding.right_state.chirality)],
        )
        resolver_key = _resolver_key(kernel, "closure", input_state_ids, None)
        template_id = _token(
            "closure",
            {
                "binding": _closure_binding_payload(binding),
                "kernel": kernel.canonical_signature,
                "color": color.semantic_digest,
                "projection": contraction.to_json_dict(),
            },
        )
        record = ClosureTemplateV1(
            template_id=template_id,
            input_state_template_ids=input_state_ids,
            result_state_template_id=state_ids[
                (
                    binding.result_state.particle_id,
                    binding.result_state.chirality,
                )
            ],
            evaluator_resolver_key=resolver_key,
            canonical_input_order=tuple(binding.canonical_input_order),
            coupling_parameter_ids=_kernel_parameter_ids(kernel, parameter_ids),
            coupling_orders=coupling_orders,
            eligible_quantum_flow_template_ids=tuple(
                sorted(flow.template_id for flow in eligible_flows)
            ),
            color_contraction_template_id=color.template_id,
            binding_coupling=_exact_pair(
                binding.key.coupling,
                f"closure kind {vertex.kind} binding coupling",
            ),
            exact_factor=_exact_pair(
                binding.equivalence_factor,
                f"closure kind {vertex.kind} equivalence factor",
            ),
            output_factor_source=binding.output_factor_source,
            equivalence_class=binding.equivalence_class,
            input_exchange_factor=(
                None
                if binding.input_exchange_factor is None
                else _exact_pair(
                    binding.input_exchange_factor,
                    f"closure kind {vertex.kind} input-exchange factor",
                )
            ),
            projection=binding.projection,
        )
        records.append(record)
        evaluator_requests.append(
            _EvaluatorRequest(
                kernel=kernel,
                contract_kind="closure",
                input_state_template_ids=input_state_ids,
                output_state_template_id=None,
                semantic_template_id=record.template_id,
            )
        )
    return (
        tuple(flows_by_id.values()),
        tuple(records),
        tuple(colors.values()),
    )


def _build_direct_closures(
    model: Model,
    state_ids: Mapping[tuple[int, int], str],
    state_by_id: Mapping[str, CurrentStateTemplateV1],
) -> tuple[
    tuple[ClosureTemplateV1, ...],
    tuple[ColorContractionTemplateV1, ...],
    tuple[EvaluatorBindingV1, ...],
]:
    """Bind exact model current-pair contractions to Rusticol templates."""

    direct = getattr(model, "_direct_contraction_ir_by_state", {})
    if not isinstance(direct, Mapping):
        raise RecurrenceTemplateError(
            "model direct-contraction inventory is not a mapping"
        )
    records: list[ClosureTemplateV1] = []
    colors: dict[str, ColorContractionTemplateV1] = {}
    evaluator_bindings: list[EvaluatorBindingV1] = []
    for raw_key in sorted(direct):
        if not isinstance(raw_key, tuple) or len(raw_key) != 4:
            raise RecurrenceTemplateError(
                f"direct-contraction key {raw_key!r} is not a four-state tuple"
            )
        key = tuple(int(value) for value in raw_key)
        left_key = key[0], key[1]
        right_key = key[2], key[3]
        if left_key not in state_ids or right_key not in state_ids:
            # Ghost-only and otherwise unadmitted states are intentionally not
            # part of recurrence v1.
            continue
        contraction = _stable_callback(
            f"direct contraction for states {key!r}",
            lambda key=key: model.direct_contraction_ir(key[0], key[2], key[1], key[3]),
            serializer=lambda value: (
                "null" if value is None else _canonical_json(value.to_json_dict())
            ),
        )
        if contraction is None:
            continue
        left_state_id = state_ids[left_key]
        right_state_id = state_ids[right_key]
        left_state = state_by_id[left_state_id]
        right_state = state_by_id[right_state_id]
        if (
            contraction.left_basis != left_state.basis
            or contraction.right_basis != right_state.basis
        ):
            raise RecurrenceTemplateError(
                f"direct contraction {contraction.name!r} basis does not match "
                f"states {key!r}"
            )
        if (
            left_state.dimension != right_state.dimension
            or len(contraction.coefficients) != left_state.dimension
        ):
            raise RecurrenceTemplateError(
                f"direct contraction {contraction.name!r} component count does "
                f"not match states {key!r}"
            )
        if contraction.chirality_relation == "equal" and key[1] != key[3]:
            raise RecurrenceTemplateError(
                f"direct contraction {contraction.name!r} violates equal chirality"
            )
        if contraction.chirality_relation == "opposite" and key[1] != -key[3]:
            raise RecurrenceTemplateError(
                f"direct contraction {contraction.name!r} violates opposite chirality"
            )
        exact_coefficients = tuple(
            _exact_pair(value, f"direct contraction {contraction.name!r} coefficient")
            for value in contraction.coefficients
        )
        callable_payload = {
            "abi": "rusticol-closure-reduce-v1",
            "left_basis": left_state.basis,
            "right_basis": right_state.basis,
            "dimension": left_state.dimension,
            "coefficients": [value.to_dict() for value in exact_coefficients],
            "chirality_relation": contraction.chirality_relation,
            "metric_signature": contraction.metric_signature,
        }
        callable_signature = _digest(callable_payload)
        runtime_template = f"rusticol.closure-reduce.v1:{callable_signature[:24]}"
        input_state_ids = left_state_id, right_state_id
        resolver_key = _token(
            "evaluator",
            {
                "runtime_template": runtime_template,
                "input_states": input_state_ids,
            },
        )
        color = _direct_closure_color_template(model, key[0], key[2])
        colors.setdefault(color.template_id, color)
        closure_payload = {
            "states": input_state_ids,
            "projection": contraction.to_json_dict(),
            "callable": callable_signature,
            "color": color.semantic_digest,
        }
        record = ClosureTemplateV1(
            template_id=_token("closure", closure_payload),
            input_state_template_ids=input_state_ids,
            result_state_template_id=None,
            evaluator_resolver_key=resolver_key,
            canonical_input_order=(0, 1),
            coupling_parameter_ids=(),
            coupling_orders=(),
            eligible_quantum_flow_template_ids=(),
            color_contraction_template_id=color.template_id,
            binding_coupling=ExactComplexRationalV1.one(),
            exact_factor=ExactComplexRationalV1.one(),
            output_factor_source="none",
            equivalence_class="direct-contraction",
            input_exchange_factor=None,
            projection=contraction.name,
            component_coefficients=exact_coefficients,
            chirality_relation=contraction.chirality_relation,
            metric_signature=contraction.metric_signature,
        )
        records.append(record)
        evaluator_bindings.append(
            EvaluatorBindingV1(
                resolver_key=resolver_key,
                prepared_kernel_id=None,
                callable_kind="rusticol-template",
                runtime_template=runtime_template,
                contract_kind="closure",
                callable_signature=callable_signature,
                input_state_template_ids=input_state_ids,
                output_state_template_id=None,
                input_layout=tuple(
                    f"left-current:{component}"
                    for component in range(left_state.dimension)
                )
                + tuple(
                    f"right-current:{component}"
                    for component in range(right_state.dimension)
                ),
                output_layout=("closure:amplitude",),
                exact_expression_digests=(_digest(callable_payload),),
                semantic_template_ids=(record.template_id,),
            )
        )
    # Model contraction inventories expose both orientations of a direct
    # bilinear. Once exact LC port wiring is attached, both orientations can
    # close the same physical current pair. They are aliases, not two terms in
    # the amplitude. Coalesce only when the complete process-independent
    # callable and contraction contracts agree after forgetting parent order.
    selected: dict[
        str,
        tuple[str, ClosureTemplateV1, EvaluatorBindingV1],
    ] = {}
    for record, binding in zip(records, evaluator_bindings, strict=True):
        alias_key = _canonical_json(
            {
                "callable_signature": binding.callable_signature,
                "chirality_relation": record.chirality_relation,
                "component_coefficients": [
                    value.to_dict() for value in record.component_coefficients
                ],
                "input_state_template_ids": sorted(record.input_state_template_ids),
                "metric_signature": record.metric_signature,
                "projection": record.projection,
            }
        )
        preference = _canonical_json(
            {
                "input_state_template_ids": record.input_state_template_ids,
                "template_id": record.template_id,
            }
        )
        existing = selected.get(alias_key)
        if existing is None or preference < existing[0]:
            selected[alias_key] = (preference, record, binding)

    selected_records = tuple(selected[key][1] for key in sorted(selected))
    selected_bindings = tuple(selected[key][2] for key in sorted(selected))
    selected_color_ids = {
        record.color_contraction_template_id for record in selected_records
    }
    selected_colors = tuple(
        color for color_id, color in colors.items() if color_id in selected_color_ids
    )
    return selected_records, selected_colors, selected_bindings


def _direct_closure_color_template(
    model: Model,
    left_particle_id: int,
    right_particle_id: int,
) -> ColorContractionTemplateV1:
    representations = (
        int(model.color_rep(left_particle_id)),
        int(model.color_rep(right_particle_id)),
    )
    if abs(representations[0]) != abs(representations[1]):
        raise RecurrenceTemplateError(
            "direct current contraction joins incompatible color "
            f"representations {representations!r}"
        )
    payload = {
        "rule_kind": "direct-pairing",
        "input_representations": representations,
    }
    if all(abs(value) == 1 for value in representations):
        closure_component_kind = None
    elif set(representations) == {3, -3}:
        closure_component_kind = "open-string"
    elif representations == (8, 8):
        closure_component_kind = "trace"
    else:
        raise RecurrenceTemplateError(
            "direct current contraction has unsupported LC closure components "
            f"{representations!r}"
        )
    (port_wiring,) = compile_lc_color_port_wirings(
        "direct-closure",
        representations,
        representations,
    )
    port_payload = {
        "input_pairings": [
            [
                [
                    pairing.fundamental.parent_index,
                    pairing.fundamental.local_port_index,
                ],
                [
                    pairing.antifundamental.parent_index,
                    pairing.antifundamental.local_port_index,
                ],
            ]
            for pairing in port_wiring.input_pairings
        ],
        "result_port_bindings": [],
    }
    witness = LCColorTransitionWitnessV1(
        input_shape_kinds=(
            model.recurrence_lc_color_shape_contract(left_particle_id),
            model.recurrence_lc_color_shape_contract(right_particle_id),
        ),
        input_permutation=(0, 1),
        reverse_parent_mask=0,
        component_operation="close",
        result_component_kind=closure_component_kind,
        result_component_role="none",
        result_shape_kind=None,
        exact_factor=ExactComplexRationalV1.one(),
        proof_digest=_digest(
            {**payload, "operation": "close", "port_wiring": port_payload}
        ),
        input_port_pairings=tuple(
            (
                (
                    pairing.fundamental.parent_index,
                    pairing.fundamental.local_port_index,
                ),
                (
                    pairing.antifundamental.parent_index,
                    pairing.antifundamental.local_port_index,
                ),
            )
            for pairing in port_wiring.input_pairings
        ),
        result_port_bindings=(),
    )
    payload["transition_witnesses"] = [witness.to_dict()]
    return ColorContractionTemplateV1(
        template_id=_token("color", payload),
        rule_kind="direct-pairing",
        input_representations=representations,
        output_representation=None,
        ordered_open_string_arity=(1 if abs(representations[0]) == 3 else 0),
        exact_coefficient=ExactComplexRationalV1.one(),
        nc_polynomial=((0, ExactComplexRationalV1.one()),),
        expression_digest=_digest(payload),
        transition_witnesses=(witness,),
    )


def _add_model_parameter_evaluator(
    prepared_catalog: PreparedKernelCatalog,
    parameters: Sequence[ParameterTemplateV1],
    kernels: Mapping[int, Any],
    evaluator_requests: list[_EvaluatorRequest],
) -> None:
    kernel_id = prepared_catalog.model_parameter_kernel_id
    derived = {
        parameter.name: parameter
        for parameter in parameters
        if parameter.parameter_kind == "derived"
    }
    if kernel_id is None:
        if derived:
            raise PreparedKernelCatalogError(
                "derived recurrence parameters have no prepared model-parameter kernel"
            )
        return
    kernel = _kernel(kernels, kernel_id, "model-parameter")
    if kernel.contract_kind != "model-parameter":
        raise PreparedKernelCatalogError(
            "prepared model-parameter binding references the wrong kernel kind"
        )
    output_names = tuple(
        value.removeprefix("model-parameter:") for value in kernel.output_layout
    )
    if set(output_names) != set(derived):
        raise PreparedKernelCatalogError(
            "prepared model-parameter outputs do not match derived parameters"
        )
    for name in output_names:
        evaluator_requests.append(
            _EvaluatorRequest(
                kernel=kernel,
                contract_kind="model-parameter",
                input_state_template_ids=(),
                output_state_template_id=None,
                semantic_template_id=derived[name].template_id,
            )
        )


def _coalesce_evaluator_requests(
    requests: Sequence[_EvaluatorRequest],
) -> tuple[EvaluatorBindingV1, ...]:
    groups: dict[
        tuple[int, str, tuple[str, ...], str | None],
        list[_EvaluatorRequest],
    ] = {}
    for request in requests:
        key = (
            int(request.kernel.kernel_id),
            request.contract_kind,
            request.input_state_template_ids,
            request.output_state_template_id,
        )
        groups.setdefault(key, []).append(request)
    result: list[EvaluatorBindingV1] = []
    for key in sorted(groups, key=_canonical_json):
        group = groups[key]
        first = group[0]
        kernel = first.kernel
        input_layout = _kernel_input_layout(kernel.inputs)
        result.append(
            EvaluatorBindingV1(
                resolver_key=_resolver_key(
                    kernel,
                    first.contract_kind,
                    first.input_state_template_ids,
                    first.output_state_template_id,
                ),
                prepared_kernel_id=int(kernel.kernel_id),
                contract_kind=first.contract_kind,  # type: ignore[arg-type]
                callable_signature=kernel.canonical_signature,
                input_state_template_ids=first.input_state_template_ids,
                output_state_template_id=first.output_state_template_id,
                input_layout=input_layout,
                output_layout=tuple(str(value) for value in kernel.output_layout),
                exact_expression_digests=tuple(
                    _expression_digest(value) for value in kernel.exact_expressions
                ),
                semantic_template_ids=tuple(
                    sorted({item.semantic_template_id for item in group})
                ),
            )
        )
    return tuple(result)


def _kernel_input_layout(inputs: Sequence[PreparedKernelInput]) -> tuple[str, ...]:
    return tuple(
        _canonical_json(
            {
                "role": item.role,
                "component": item.component,
                "symbol": item.symbol,
                "model_parameter_name": item.model_parameter_name,
                "model_parameter_index": item.model_parameter_index,
            }
        )
        for item in inputs
    )


def _kernel_parameter_ids(
    kernel: Any,
    parameter_ids: Mapping[str, str],
) -> tuple[str, ...]:
    names = tuple(
        sorted(
            {
                str(item.model_parameter_name)
                for item in kernel.inputs
                if item.model_parameter_name is not None
            }
        )
    )
    missing = tuple(name for name in names if name not in parameter_ids)
    if missing:
        raise RecurrenceTemplateError(
            "prepared kernel references absent recurrence parameters: "
            + ", ".join(missing)
        )
    return tuple(sorted(parameter_ids[name] for name in names))


def _kernel(
    kernels: Mapping[int, Any],
    kernel_id: int,
    context: str,
) -> Any:
    try:
        return kernels[int(kernel_id)]
    except KeyError as exc:
        raise PreparedKernelCatalogError(
            f"prepared {context} binding references unknown kernel {kernel_id}"
        ) from exc


def _validate_binary_kernel_layout(
    kernel: Any,
    binding: PreparedVertexBinding | PreparedClosureBinding,
    *,
    output_dimension: int,
    context: str,
) -> None:
    canonical_states = tuple(
        (binding.left_state, binding.right_state)[index]
        for index in binding.canonical_input_order
    )
    expected = {
        "left-current": canonical_states[0].dimension,
        "right-current": canonical_states[1].dimension,
    }
    for role, dimension in expected.items():
        components = tuple(
            item.component for item in kernel.inputs if item.role == role
        )
        if len(set(components)) != len(components) or any(
            component < 0 or component >= dimension for component in components
        ):
            raise PreparedKernelCatalogError(
                f"prepared {context} {role} layout does not match its state dimension"
            )
    if len(kernel.output_layout) != output_dimension:
        raise PreparedKernelCatalogError(
            f"prepared {context} output layout has dimension "
            f"{len(kernel.output_layout)}, expected {output_dimension}"
        )


def _validate_propagator_kernel_layout(
    kernel: Any,
    binding: PreparedPropagatorBinding,
) -> None:
    components = tuple(
        item.component for item in kernel.inputs if item.role == "current"
    )
    if len(set(components)) != len(components) or any(
        component < 0 or component >= binding.state.dimension
        for component in components
    ):
        raise PreparedKernelCatalogError(
            f"prepared propagator {binding.key!r} current layout does not match "
            "its state dimension"
        )
    if len(kernel.output_layout) != binding.state.dimension:
        raise PreparedKernelCatalogError(
            f"prepared propagator {binding.key!r} output dimension is stale"
        )


def _resolver_key(
    kernel: Any,
    contract_kind: str,
    input_states: tuple[str, ...],
    output_state: str | None,
) -> str:
    return _token(
        "evaluator",
        {
            "kernel_id": int(kernel.kernel_id),
            "signature": kernel.canonical_signature,
            "contract_kind": contract_kind,
            "input_states": input_states,
            "output_state": output_state,
        },
    )


def _vertex_binding_payload(binding: PreparedVertexBinding) -> dict[str, object]:
    return {
        "kind": binding.key.kind,
        "particles": list(binding.key.particles),
        "chiralities": [
            binding.key.left_chirality,
            binding.key.right_chirality,
            binding.key.result_chirality,
        ],
        "coupling": _exact_pair(binding.key.coupling, "vertex coupling").to_dict(),
        "canonical_input_order": list(binding.canonical_input_order),
        "equivalence_factor": _exact_pair(
            binding.equivalence_factor, "vertex equivalence factor"
        ).to_dict(),
        "output_factor_source": binding.output_factor_source,
        "contact_orbit_certificates": [
            item.to_dict() for item in sorted(binding.contact_orbit_certificates)
        ],
        "contact_orbit_steps": [
            item.to_dict() for item in sorted(binding.contact_orbit_steps)
        ],
    }


def _closure_binding_payload(binding: PreparedClosureBinding) -> dict[str, object]:
    return {
        "kind": binding.key.kind,
        "particles": list(binding.key.particles),
        "chiralities": [binding.key.left_chirality, binding.key.right_chirality],
        "coupling": _exact_pair(binding.key.coupling, "closure coupling").to_dict(),
        "canonical_input_order": list(binding.canonical_input_order),
        "equivalence_factor": _exact_pair(
            binding.equivalence_factor, "closure equivalence factor"
        ).to_dict(),
        "projection": binding.projection,
        "output_factor_source": binding.output_factor_source,
    }


def _stable_callback(
    context: str,
    callback: Any,
    *,
    serializer: Any,
) -> Any:
    try:
        first = callback()
        first_payload = serializer(first)
        second = callback()
        second_payload = serializer(second)
    except (PreparedKernelCatalogError, RecurrenceTemplateError):
        raise
    except Exception as exc:
        raise RecurrenceTemplateError(f"cannot project {context}: {exc}") from exc
    if first_payload != second_payload:
        raise RecurrenceTemplateError(f"{context} is nondeterministic")
    return first


__all__ = ["build_recurrence_template_catalog"]
