# SPDX-License-Identifier: 0BSD
"""Model-specific construction of process-independent prepared kernel catalogs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, cast

from .._internal.physics.symbols import symbols
from . import compiler_symbolica as _sym
from ._physics_ir import ContractionIR, PropagatorIR
from .base import Model, Vertex, VertexEvaluationEquivalence
from .contact_decomposition import (
    CompiledContactOrbitCertificate,
    CompiledContactOrbitStep,
)
from .expressions import _as_expression, _exact_complex_expression
from .external import CompiledUFOModel
from .prepared_catalog import (
    _CATALOG_ABI,
    PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF,
    PREPARED_INDEPENDENT_BLOCK_PROOF,
    ClosureKernelKey,
    PreparedClosureBinding,
    PreparedContractKind,
    PreparedKernelCatalog,
    PreparedKernelCatalogError,
    PreparedKernelInput,
    PreparedKernelSpec,
    PreparedOutputFactorSource,
    PreparedParticleState,
    PreparedPropagatorBinding,
    PreparedVertexBinding,
    PropagatorKernelKey,
    VertexKernelKey,
)
from .prepared_catalog_helpers import (
    canonical_expressions as _canonical_expressions,
)
from .prepared_catalog_helpers import (
    canonical_json as _canonical_json,
)
from .prepared_catalog_helpers import (
    component_descriptors as _component_descriptors,
)
from .prepared_catalog_helpers import (
    coupling_descriptor as _coupling_descriptor,
)
from .prepared_catalog_helpers import (
    formal_components as _formal_components,
)
from .prepared_catalog_helpers import (
    formal_model_parameter as _formal_model_parameter,
)
from .prepared_catalog_helpers import (
    model_parameter_descriptor as _model_parameter_descriptor,
)
from .prepared_catalog_helpers import (
    proves_homogeneous_complex_linearity as _proves_homogeneous_complex_linearity,
)
from .prepared_catalog_helpers import (
    proves_independent_current_block_eligibility as _proves_block_eligibility,
)
from .prepared_catalog_helpers import (
    used_input_descriptors as _used_input_descriptors,
)

_MAX_DISCOVERED_CURRENT_STATES = 4096


@dataclass(frozen=True, slots=True)
class _Candidate:
    contract_kind: PreparedContractKind
    canonical_signature: str
    exact_expressions: tuple[str, ...]
    inputs: tuple[PreparedKernelInput, ...]
    output_layout: tuple[str, ...]
    proof_class: str | None

    def spec(self, kernel_id: int, proof_classes: Sequence[str]) -> PreparedKernelSpec:
        certified = set(proof_classes)
        if _proves_block_eligibility(
            self.contract_kind,
            self.exact_expressions,
            self.inputs,
        ):
            certified.add(PREPARED_INDEPENDENT_BLOCK_PROOF)
        return PreparedKernelSpec(
            kernel_id=kernel_id,
            contract_kind=self.contract_kind,
            canonical_signature=self.canonical_signature,
            exact_expressions=self.exact_expressions,
            inputs=self.inputs,
            output_layout=self.output_layout,
            proof_classes=tuple(sorted(certified)),
        )


@dataclass(frozen=True, slots=True)
class _VertexCandidateResult:
    candidate: _Candidate
    key: VertexKernelKey
    equivalence: VertexEvaluationEquivalence
    left_state: PreparedParticleState
    right_state: PreparedParticleState
    result_state: PreparedParticleState
    output_factor_source: PreparedOutputFactorSource = "none"
    contact_orbit_certificates: tuple[CompiledContactOrbitCertificate, ...] = ()
    contact_orbit_steps: tuple[CompiledContactOrbitStep, ...] = ()


class _RuntimeParameterizedModel:
    """Model-local copy of the compiled-stage parameter overlay."""

    def __init__(self, base: Model, parameters: Mapping[str, Any]) -> None:
        self._base_model = base
        self._runtime_parameters = parameters
        self.name = base.name
        self.particles = base.particles
        self.vertices = base.vertices

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_model, name)

    def mass(self, pdg: int) -> Any:
        particle = self._base_model.particle(pdg)
        name = f"particle.{int(particle.pdg)}.mass"
        return self._runtime_parameters.get(name, self._base_model.mass(pdg))

    def width(self, pdg: int) -> Any:
        particle = self._base_model.particle(pdg)
        name = f"particle.{int(particle.pdg)}.width"
        return self._runtime_parameters.get(name, self._base_model.width(pdg))

    def propagator_lowering_rule(self, particle_id: int, chirality: int = 0) -> Any:
        return self._base_model.propagator_lowering_rule(particle_id, chirality)

    def propagator_component_expression(
        self,
        particle_id: int,
        value: Sequence[Any],
        momentum: Sequence[Any],
        *,
        chirality: int = 0,
        propagator: PropagatorIR | None = None,
    ) -> tuple[Any, ...]:
        return cast(
            tuple[Any, ...],
            type(self._base_model).propagator_component_expression(
                self,
                particle_id,
                value,
                momentum,
                chirality=chirality,
                propagator=propagator,
            ),
        )


def build_prepared_kernel_catalog(model: Model) -> PreparedKernelCatalog:
    """Build the complete process-independent exact catalog for ``model``."""

    _sym._ensure_symbolica()
    parameter_names = _model_parameter_names(model)
    parameter_indices = {name: index for index, name in enumerate(parameter_names)}
    candidates: dict[str, _Candidate] = {}
    proof_classes: dict[str, set[str]] = {}
    vertex_pending: list[_VertexCandidateResult] = []
    closure_pending: list[tuple[_VertexCandidateResult, str]] = []
    propagator_pending: list[
        tuple[
            PropagatorKernelKey, PreparedParticleState, PropagatorIR, _Candidate | None
        ]
    ] = []

    vertices = _unique_vertices(model)
    chiralities_by_particle: dict[int, set[int]] = {}
    for vertex in vertices:
        for particle_id in vertex.particles:
            chiralities_by_particle.setdefault(int(particle_id), set()).update(
                _state_chiralities(model, int(particle_id))
            )
    visited_inputs: set[tuple[object, ...]] = set()
    discovered = True
    while discovered:
        discovered = False
        for vertex in vertices:
            left_chiralities = tuple(
                sorted(chiralities_by_particle[vertex.particles[0]])
            )
            right_chiralities = tuple(
                sorted(chiralities_by_particle[vertex.particles[1]])
            )
            for left_chirality in left_chiralities:
                for right_chirality in right_chiralities:
                    input_key = (
                        vertex.kind,
                        vertex.particles,
                        vertex.coupling,
                        left_chirality,
                        right_chirality,
                    )
                    if input_key in visited_inputs:
                        continue
                    visited_inputs.add(input_key)
                    flows = _reachable_flows(
                        model,
                        vertex,
                        left_chirality=left_chirality,
                        right_chirality=right_chirality,
                    )
                    for result_chirality in flows:
                        try:
                            result = _vertex_candidate(
                                model,
                                vertex,
                                left_chirality=left_chirality,
                                right_chirality=right_chirality,
                                result_chirality=result_chirality,
                                parameter_indices=parameter_indices,
                                contract_kind="vertex",
                            )
                        except PreparedKernelCatalogError as error:
                            raise PreparedKernelCatalogError(
                                "prepared catalog cannot lower admitted vertex "
                                "orientation "
                                f"kind={vertex.kind}, particles={vertex.particles}, "
                                "chiralities="
                                f"({left_chirality},{right_chirality},"
                                f"{result_chirality}): {error}"
                            ) from error
                        _register_candidate(candidates, proof_classes, result.candidate)
                        vertex_pending.append(result)
                        result_states = chiralities_by_particle.setdefault(
                            int(vertex.particles[2]), set()
                        )
                        if result_chirality not in result_states:
                            result_states.add(result_chirality)
                            discovered = True
                            if (
                                sum(map(len, chiralities_by_particle.values()))
                                > _MAX_DISCOVERED_CURRENT_STATES
                            ):
                                raise PreparedKernelCatalogError(
                                    "prepared current-state discovery did not "
                                    "converge within the safety bound"
                                )
                        closure_ir = model.closure_contraction_ir(
                            vertex.particles[2], result_chirality
                        )
                        if (
                            model.vertex_closure_allowed(vertex)
                            and closure_ir is not None
                        ):
                            closure = _closure_candidate(result, closure_ir)
                            _register_candidate(
                                candidates,
                                proof_classes,
                                closure.candidate,
                            )
                            closure_pending.append((closure, closure_ir.name))

    propagator_states = _required_propagator_states(model, vertex_pending)
    for state in propagator_states:
        particle_id = state.particle_id
        chirality = state.chirality
        try:
            metadata = model._propagator_ir(particle_id, chirality)
        except (NotImplementedError, ValueError) as error:
            raise PreparedKernelCatalogError(
                "prepared catalog cannot lower admitted propagator "
                f"particle={particle_id}, chirality={chirality}: {error}"
            ) from error
        candidate = (
            _propagator_candidate(
                model,
                state,
                metadata,
                parameter_indices=parameter_indices,
            )
            if metadata.applies_propagator
            else None
        )
        if candidate is not None:
            _register_candidate(candidates, proof_classes, candidate)
        propagator_pending.append(
            (
                PropagatorKernelKey(particle_id, chirality),
                state,
                metadata,
                candidate,
            )
        )

    model_parameter_candidate = _model_parameter_candidate(
        model,
        parameter_indices=parameter_indices,
    )
    if model_parameter_candidate is not None:
        _register_candidate(candidates, proof_classes, model_parameter_candidate)

    ordered_signatures = tuple(sorted(candidates))
    kernel_ids = {
        signature: kernel_id for kernel_id, signature in enumerate(ordered_signatures)
    }
    kernels = tuple(
        candidates[signature].spec(
            kernel_ids[signature],
            proof_classes.get(signature, ()),
        )
        for signature in ordered_signatures
    )
    vertex_bindings = tuple(
        sorted(
            (
                PreparedVertexBinding(
                    key=result.key,
                    kernel_id=kernel_ids[result.candidate.canonical_signature],
                    canonical_input_order=result.equivalence.input_order,
                    equivalence_class=result.equivalence.class_id,
                    equivalence_factor=result.equivalence.factor,
                    input_exchange_factor=(result.equivalence.input_exchange_factor),
                    left_state=result.left_state,
                    right_state=result.right_state,
                    result_state=result.result_state,
                    output_factor_source=result.output_factor_source,
                    contact_orbit_certificates=result.contact_orbit_certificates,
                    contact_orbit_steps=result.contact_orbit_steps,
                )
                for result in vertex_pending
            ),
            key=lambda binding: binding.key,
        )
    )
    propagator_bindings = tuple(
        PreparedPropagatorBinding(
            key=key,
            kernel_id=(
                None if candidate is None else kernel_ids[candidate.canonical_signature]
            ),
            state=state,
            applies_propagator=metadata.applies_propagator,
            propagator_kind=metadata.kind,
            mass_class=metadata.mass_class,
            gauge=metadata.gauge,
            model_parameters=tuple(
                descriptor.model_parameter_name
                for descriptor in (() if candidate is None else candidate.inputs)
                if descriptor.model_parameter_name is not None
            ),
        )
        for key, state, metadata, candidate in sorted(
            propagator_pending,
            key=lambda item: item[0],
        )
    )
    closure_bindings = tuple(
        sorted(
            (
                PreparedClosureBinding(
                    key=ClosureKernelKey(
                        result.key.kind,
                        result.key.particles,
                        result.key.left_chirality,
                        result.key.right_chirality,
                        result.key.coupling,
                    ),
                    kernel_id=kernel_ids[result.candidate.canonical_signature],
                    canonical_input_order=result.equivalence.input_order,
                    equivalence_class=result.equivalence.class_id,
                    equivalence_factor=result.equivalence.factor,
                    input_exchange_factor=(result.equivalence.input_exchange_factor),
                    left_state=result.left_state,
                    right_state=result.right_state,
                    result_state=result.result_state,
                    projection=projection,
                    output_factor_source=result.output_factor_source,
                )
                for result, projection in closure_pending
            ),
            key=lambda binding: binding.key,
        )
    )
    missing_vertex_kinds = sorted(
        {vertex.kind for vertex in _unique_vertices(model)}
        - {binding.key.kind for binding in vertex_bindings}
    )
    if missing_vertex_kinds:
        raise PreparedKernelCatalogError(
            "prepared catalog has no constructive orientation for model vertex "
            f"kinds {missing_vertex_kinds}"
        )
    return PreparedKernelCatalog(
        model_name=str(model.name),
        kernels=kernels,
        vertex_bindings=vertex_bindings,
        propagator_bindings=propagator_bindings,
        closure_bindings=closure_bindings,
        model_parameter_kernel_id=(
            None
            if model_parameter_candidate is None
            else kernel_ids[model_parameter_candidate.canonical_signature]
        ),
        unsupported_variants=(),
    )


def _unique_vertices(model: Model) -> tuple[Vertex, ...]:
    by_key = {
        (
            int(vertex.kind),
            tuple(int(particle) for particle in vertex.particles),
            tuple(float(value) for value in vertex.coupling),
        ): vertex
        for vertex in model.iter_vertices(color_accuracy="full")
    }
    return tuple(by_key[key] for key in sorted(by_key))


def _oriented_particle_ids(model: Model) -> tuple[int, ...]:
    particle_ids = {
        int(particle_id)
        for particle_id in model.particles
        if model.source_wavefunction_kind(int(particle_id)) != "ghost"
    }
    particle_ids.update(
        model.anti_particle(particle_id) for particle_id in tuple(particle_ids)
    )
    return tuple(sorted(particle_ids))


def _required_propagator_states(
    model: Model,
    vertex_candidates: Sequence[_VertexCandidateResult],
) -> tuple[PreparedParticleState, ...]:
    """Return every source or transition-reachable current state.

    Source spin states do not necessarily span the internal current basis.  A
    massless fermion, for example, has chiral external wavefunctions but may be
    produced as a full Dirac current by a transition involving a massive
    fermion.  Prepared propagators must therefore cover vertex states as well
    as source states.
    """

    states: dict[tuple[int, int], PreparedParticleState] = {}

    def add(state: PreparedParticleState) -> None:
        if model.source_wavefunction_kind(int(state.particle_id)) == "ghost":
            return
        key = (int(state.particle_id), int(state.chirality))
        previous = states.setdefault(key, state)
        if previous != state:
            raise PreparedKernelCatalogError(
                "prepared vertex and source contracts disagree on current-state "
                f"semantics for particle={key[0]}, chirality={key[1]}"
            )

    for particle_id in _oriented_particle_ids(model):
        for chirality in _state_chiralities(model, particle_id):
            add(_particle_state(model, particle_id, chirality))
    for candidate in vertex_candidates:
        add(candidate.left_state)
        add(candidate.right_state)
        add(candidate.result_state)
    return tuple(states[key] for key in sorted(states))


def _state_chiralities(model: Model, particle_id: int) -> tuple[int, ...]:
    if model.auxiliary_kind(particle_id) is not None:
        return (0,)
    if model.particle(particle_id).spin < 0:
        return (0,)
    chiralities = {
        int(state.chirality) for state in model.source_spin_states(particle_id)
    }
    source_chiralities = tuple(
        sorted(chirality for chirality in chiralities if chirality)
    )
    if len(source_chiralities) < 2:
        return tuple(sorted(chiralities))
    try:
        contracts = tuple(
            model.recurrence_current_embedding_contract(particle_id, chirality)
            for chirality in source_chiralities
        )
    except NotImplementedError:
        return tuple(sorted(chiralities))
    full_contracts = {
        (contract.full_dimension, contract.full_basis) for contract in contracts
    }
    expected_full = (
        int(model.current_dimension(particle_id, 0)),
        model._current_basis(particle_id, 0),
    )
    if full_contracts != {expected_full}:
        raise PreparedKernelCatalogError(
            "recurrence current-embedding contracts disagree on the full state "
            f"for particle={particle_id}: {sorted(full_contracts)!r}"
        )
    chiralities.add(0)
    return tuple(sorted(chiralities))


def _particle_state(
    model: Model,
    particle_id: int,
    chirality: int,
) -> PreparedParticleState:
    metadata = model._particle_identity_ir(particle_id)
    return PreparedParticleState(
        particle_id=int(particle_id),
        identity=metadata.canonical_id,
        orientation=metadata.orientation,
        basis=model._current_basis(particle_id, chirality),
        chirality=int(chirality),
        dimension=int(model.current_dimension(particle_id, chirality)),
    )


def _reachable_flows(
    model: Model,
    vertex: Vertex,
    *,
    left_chirality: int,
    right_chirality: int,
) -> tuple[int, ...]:
    left = SimpleNamespace(
        particle_id=vertex.particles[0],
        pdg=vertex.particles[0],
        chirality=left_chirality,
        flavour_flow=(vertex.particles[0],),
        coupling_orders=(),
    )
    right = SimpleNamespace(
        particle_id=vertex.particles[1],
        pdg=vertex.particles[1],
        chirality=right_chirality,
        flavour_flow=(vertex.particles[1],),
        coupling_orders=(),
    )
    try:
        return tuple(
            sorted(
                {
                    int(flow.chirality)
                    for flow in model.allowed_quantum_flows(vertex, left, right)
                }
            )
        )
    except Exception as error:
        raise PreparedKernelCatalogError(
            "cannot enumerate prepared vertex flows for "
            f"model {model.name!r}, kind {vertex.kind}, particles "
            f"{vertex.particles}, chiralities "
            f"({left_chirality}, {right_chirality}): {error}"
        ) from error


def _vertex_candidate(
    model: Model,
    vertex: Vertex,
    *,
    left_chirality: int,
    right_chirality: int,
    result_chirality: int,
    parameter_indices: Mapping[str, int],
    contract_kind: Literal["vertex", "closure"],
) -> _VertexCandidateResult:
    equivalence = model.vertex_evaluation_equivalence(vertex.kind)
    contact_orbit_certificates, contact_orbit_steps = (
        model.vertex_contact_orbit_contracts(vertex.kind)
    )
    if not equivalence.verified:
        raise PreparedKernelCatalogError(
            f"prepared vertex kind {vertex.kind} has an unverified equivalence"
        )
    concrete_states = (
        _particle_state(model, vertex.particles[0], left_chirality),
        _particle_state(model, vertex.particles[1], right_chirality),
    )
    canonical_states = tuple(
        concrete_states[index] for index in equivalence.input_order
    )
    canonical_left = _formal_components(
        model,
        "left-current",
        canonical_states[0].dimension,
    )
    canonical_right = _formal_components(
        model,
        "right-current",
        canonical_states[1].dimension,
    )
    canonical_left_momentum = _formal_components(model, "left-momentum", 4)
    canonical_right_momentum = _formal_components(model, "right-momentum", 4)
    canonical_values = (canonical_left, canonical_right)
    canonical_momenta = (canonical_left_momentum, canonical_right_momentum)
    inverse_order = tuple(equivalence.input_order.index(index) for index in (0, 1))
    concrete_values = tuple(canonical_values[index] for index in inverse_order)
    concrete_momenta = tuple(canonical_momenta[index] for index in inverse_order)
    coupling_symbols = _formal_components(model, "coupling", 2)
    parameter_symbols = {
        name: _formal_model_parameter(model, name, parameter_indices[name])
        for name in _vertex_runtime_parameter_names(model, vertex.kind)
    }
    try:
        expressions, coupling_inputs = _vertex_expressions(
            model,
            vertex,
            left=concrete_values[0],
            right=concrete_values[1],
            left_chirality=left_chirality,
            right_chirality=right_chirality,
            result_chirality=result_chirality,
            left_momentum=concrete_momenta[0],
            right_momentum=concrete_momenta[1],
            coupling_symbols=coupling_symbols,
            parameter_symbols=parameter_symbols,
        )
    except Exception as error:
        raise PreparedKernelCatalogError(
            "cannot construct prepared vertex expression for "
            f"model {model.name!r}, kind {vertex.kind}, particles "
            f"{vertex.particles}, chiralities "
            f"({left_chirality}, {right_chirality}, {result_chirality}): {error}"
        ) from error
    factor = _exact_complex_expression(complex(*equivalence.factor))
    normalized = tuple(
        _as_expression(expression) / factor for expression in expressions
    )
    available_descriptors = (
        *_component_descriptors(model, "left-current", canonical_left),
        *_component_descriptors(model, "right-current", canonical_right),
        *_component_descriptors(model, "left-momentum", canonical_left_momentum),
        *_component_descriptors(model, "right-momentum", canonical_right_momentum),
        *(
            _coupling_descriptor(model, coupling_symbols[index], index)
            for index in coupling_inputs
        ),
        *(
            _model_parameter_descriptor(
                model,
                parameter_symbols[name],
                name,
                parameter_indices[name],
            )
            for name in sorted(parameter_symbols)
        ),
    )
    descriptors = _used_input_descriptors(normalized, available_descriptors)
    normalized, descriptors, output_factor_source = _factor_linear_coupling_output(
        normalized,
        descriptors,
        available_descriptors=available_descriptors,
        coupling_symbols=coupling_symbols,
        context=f"vertex kind {vertex.kind}",
    )
    exact = _canonical_expressions(normalized, context=f"vertex kind {vertex.kind}")
    if all(expression == "0" for expression in exact):
        raise PreparedKernelCatalogError(
            f"reachable prepared vertex kind {vertex.kind} lowered to zero"
        )
    output_state = _particle_state(
        model,
        vertex.particles[2],
        result_chirality,
    )
    payload = {
        "abi": _CATALOG_ABI,
        "contract_kind": contract_kind,
        "inputs": [descriptor.to_dict() for descriptor in descriptors],
        "outputs": list(exact),
        "output_layout": [
            f"{output_state.basis}:c{index}" for index in range(len(exact))
        ],
        "canonical_input_states": [state.contract_dict() for state in canonical_states],
        "result_state": output_state.contract_dict(),
    }
    candidate = _candidate_from_payload(
        contract_kind,
        payload,
        exact,
        descriptors,
        tuple(payload["output_layout"]),
        proof_class=equivalence.class_id,
    )
    return _VertexCandidateResult(
        candidate=candidate,
        key=VertexKernelKey(
            kind=int(vertex.kind),
            particles=tuple(int(value) for value in vertex.particles),
            left_chirality=int(left_chirality),
            right_chirality=int(right_chirality),
            result_chirality=int(result_chirality),
            coupling=tuple(float(value) for value in vertex.coupling),
        ),
        equivalence=equivalence,
        left_state=concrete_states[0],
        right_state=concrete_states[1],
        result_state=output_state,
        output_factor_source=output_factor_source,
        contact_orbit_certificates=contact_orbit_certificates,
        contact_orbit_steps=contact_orbit_steps,
    )


def _factor_linear_coupling_output(
    expressions: tuple[Any, ...],
    descriptors: tuple[PreparedKernelInput, ...],
    *,
    available_descriptors: tuple[PreparedKernelInput, ...],
    coupling_symbols: tuple[Any, ...],
    context: str,
) -> tuple[
    tuple[Any, ...],
    tuple[PreparedKernelInput, ...],
    PreparedOutputFactorSource,
]:
    """Factor one proof-certified real coupling component out of a callable."""

    coupling_inputs = tuple(
        descriptor
        for descriptor in descriptors
        if descriptor.role in ("coupling-real", "coupling-imag")
    )
    if len(coupling_inputs) != 1:
        return expressions, descriptors, "none"
    descriptor = coupling_inputs[0]
    component = 0 if descriptor.role == "coupling-real" else 1
    symbol = coupling_symbols[component]
    symbol_name = symbol.to_canonical_string()
    bases = tuple(
        _as_expression(expression).replace(symbol, 1) for expression in expressions
    )
    if any(
        symbol_name
        in {
            item.to_canonical_string()
            for item in _as_expression(base).get_all_symbols(False)
        }
        for base in bases
    ):
        return expressions, descriptors, "none"
    residuals = tuple(
        (_as_expression(expression) - symbol * _as_expression(base)).expand()
        for expression, base in zip(expressions, bases, strict=True)
    )
    if any(
        residual != "0"
        for residual in _canonical_expressions(
            residuals,
            context=f"{context} coupling-factor proof",
        )
    ):
        return expressions, descriptors, "none"
    factored_descriptors = _used_input_descriptors(bases, available_descriptors)
    return (
        bases,
        factored_descriptors,
        cast(PreparedOutputFactorSource, descriptor.role),
    )


def _vertex_expressions(
    model: Model,
    vertex: Vertex,
    *,
    left: Sequence[Any],
    right: Sequence[Any],
    left_chirality: int,
    right_chirality: int,
    result_chirality: int,
    left_momentum: Sequence[Any],
    right_momentum: Sequence[Any],
    coupling_symbols: tuple[Any, ...],
    parameter_symbols: Mapping[str, Any],
) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    if isinstance(model, CompiledUFOModel):
        kernel = model._kernel(vertex.kind)
        components = model._projected_kernel_components(
            kernel,
            left,
            right,
            left_chirality=left_chirality,
            right_chirality=right_chirality,
            result_chirality=result_chirality,
            left_momentum=left_momentum,
            right_momentum=right_momentum,
            runtime_parameter_values=parameter_symbols,
        )
        coupling = model._resolved_kernel_coupling_expression(
            kernel,
            parameter_symbols,
        )
        return tuple(component * coupling for component in components), ()

    rule = model.vertex_lowering_rule(vertex.kind)
    if not rule.full_tensor_network_ready or rule.backend == "unimplemented":
        raise PreparedKernelCatalogError(
            f"vertex kind {vertex.kind} has no complete local lowering contract"
        )
    coupling: tuple[Any, Any]
    coupling_inputs: tuple[int, ...]
    if rule.coupling_mode == "vertex":
        coupling = (coupling_symbols[0], coupling_symbols[1])
        coupling_inputs = (0, 1)
    else:
        coupling = vertex.coupling
        coupling_inputs = ()
    expressions = model.vertex_component_expression(
        vertex.kind,
        left,
        right,
        result_particle_id=vertex.particles[2],
        result_chirality=result_chirality,
        left_chirality=left_chirality,
        right_chirality=right_chirality,
        coupling=coupling,
        left_momentum=left_momentum,
        right_momentum=right_momentum,
    )
    if coupling_inputs:
        actual = model.vertex_component_expression(
            vertex.kind,
            left,
            right,
            result_particle_id=vertex.particles[2],
            result_chirality=result_chirality,
            left_chirality=left_chirality,
            right_chirality=right_chirality,
            coupling=vertex.coupling,
            left_momentum=left_momentum,
            right_momentum=right_momentum,
        )
        substituted = tuple(
            _as_expression(expression)
            .replace(coupling_symbols[0], vertex.coupling[0])
            .replace(coupling_symbols[1], vertex.coupling[1])
            for expression in expressions
        )
        # Equal exact values may retain different float/integer atom spellings
        # after substitution (for example 1.0*i versus i next to sqrt(2)).
        # Prove equality by cancellation, not by comparing those spellings.
        differences = tuple(
            _as_expression(formal) - _as_expression(concrete)
            for formal, concrete in zip(substituted, actual, strict=True)
        )
        if any(
            value != "0"
            for value in _canonical_expressions(
                differences, context="formal coupling difference"
            )
        ):
            coupling = (coupling_symbols[0], vertex.coupling[1])
            coupling_inputs = (0,)
            expressions = model.vertex_component_expression(
                vertex.kind,
                left,
                right,
                result_particle_id=vertex.particles[2],
                result_chirality=result_chirality,
                left_chirality=left_chirality,
                right_chirality=right_chirality,
                coupling=coupling,
                left_momentum=left_momentum,
                right_momentum=right_momentum,
            )
            branch_substituted = tuple(
                _as_expression(expression).replace(
                    coupling_symbols[0], vertex.coupling[0]
                )
                for expression in expressions
            )
            branch_differences = tuple(
                _as_expression(formal) - _as_expression(concrete)
                for formal, concrete in zip(branch_substituted, actual, strict=True)
            )
            if any(
                value != "0"
                for value in _canonical_expressions(
                    branch_differences, context="branch coupling difference"
                )
            ):
                raise PreparedKernelCatalogError(
                    "vertex coupling control flow cannot be represented by the "
                    "prepared scalar coupling inputs"
                )
    return tuple(expressions), coupling_inputs


def _closure_candidate(
    vertex: _VertexCandidateResult,
    contraction: ContractionIR,
) -> _VertexCandidateResult:
    source = tuple(
        _sym.E(expression) for expression in vertex.candidate.exact_expressions
    )
    if len(source) != len(contraction.coefficients):
        raise PreparedKernelCatalogError(
            f"closure projection {contraction.name!r} has dimension "
            f"{len(contraction.coefficients)}, expected {len(source)}"
        )
    projected = sum(
        (
            _exact_complex_expression(complex(*coefficient)) * expression
            for coefficient, expression in zip(
                contraction.coefficients,
                source,
                strict=True,
            )
        ),
        _sym.E("0"),
    )
    exact = _canonical_expressions((projected,), context="vertex closure")
    payload = {
        "abi": _CATALOG_ABI,
        "contract_kind": "closure",
        "inputs": [descriptor.to_dict() for descriptor in vertex.candidate.inputs],
        "outputs": list(exact),
        "output_layout": ["scalar:c0"],
        "projection": contraction.to_json_dict(),
    }
    candidate = _candidate_from_payload(
        "closure",
        payload,
        exact,
        vertex.candidate.inputs,
        ("scalar:c0",),
        proof_class=vertex.equivalence.class_id,
    )
    return _VertexCandidateResult(
        candidate=candidate,
        key=vertex.key,
        equivalence=vertex.equivalence,
        left_state=vertex.left_state,
        right_state=vertex.right_state,
        result_state=vertex.result_state,
        output_factor_source=vertex.output_factor_source,
        contact_orbit_certificates=vertex.contact_orbit_certificates,
        contact_orbit_steps=vertex.contact_orbit_steps,
    )


def _propagator_candidate(
    model: Model,
    state: PreparedParticleState,
    metadata: PropagatorIR,
    *,
    parameter_indices: Mapping[str, int],
) -> _Candidate:
    current = _formal_components(model, "current", state.dimension)
    momentum = _formal_components(model, "momentum", 4)
    names = _particle_runtime_parameter_names(model, state.particle_id)
    parameter_symbols = {
        name: _formal_model_parameter(model, name, parameter_indices[name])
        for name in names
    }
    parameterized: Any = (
        model.with_runtime_parameters(parameter_symbols)
        if hasattr(model, "with_runtime_parameters")
        else _RuntimeParameterizedModel(model, parameter_symbols)
    )
    try:
        expressions = parameterized.propagator_component_expression(
            state.particle_id,
            current,
            momentum,
            chirality=state.chirality,
            propagator=metadata,
        )
    except Exception as error:
        raise PreparedKernelCatalogError(
            "cannot construct prepared propagator expression for "
            f"model {model.name!r}, particle {state.particle_id}, chirality "
            f"{state.chirality}: {error}"
        ) from error
    exact = _canonical_expressions(
        expressions,
        context=f"propagator {state.particle_id}/{state.chirality}",
    )
    descriptors = _used_input_descriptors(
        tuple(_as_expression(value) for value in expressions),
        (
            *_component_descriptors(model, "current", current),
            *_component_descriptors(model, "momentum", momentum),
            *(
                _model_parameter_descriptor(
                    model,
                    parameter_symbols[name],
                    name,
                    parameter_indices[name],
                )
                for name in sorted(parameter_symbols)
            ),
        ),
    )
    payload = {
        "abi": _CATALOG_ABI,
        "contract_kind": "propagator",
        "inputs": [descriptor.to_dict() for descriptor in descriptors],
        "outputs": list(exact),
        "output_layout": [f"{state.basis}:c{index}" for index in range(len(exact))],
        "state": state.contract_dict(),
        "propagator": {
            "kind": metadata.kind,
            "mass_class": metadata.mass_class,
            "gauge": metadata.gauge,
            "kernel": metadata.kernel,
            "custom_source": metadata.custom_source,
            "goldstone_policy": metadata.goldstone_policy,
        },
    }
    return _candidate_from_payload(
        "propagator",
        payload,
        exact,
        descriptors,
        tuple(payload["output_layout"]),
        proof_class=(
            PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF
            if _proves_homogeneous_complex_linearity(expressions, current)
            else None
        ),
    )


def _model_parameter_candidate(
    model: Model,
    *,
    parameter_indices: Mapping[str, int],
) -> _Candidate | None:
    provider = getattr(model, "runtime_derived_parameter_definitions", None)
    if not callable(provider):
        return None
    definitions = {
        str(name): str(expression) for name, expression in provider().items()
    }
    if not definitions:
        return None
    defaults_provider = getattr(model, "runtime_parameter_defaults", None)
    defaults = (
        {str(name) for name in defaults_provider()}
        if callable(defaults_provider)
        else set()
    )
    formal = {
        name: _formal_model_parameter(model, name, parameter_indices[name])
        for name in sorted(defaults)
    }
    model_symbols = symbols.model(model.name)
    outputs: list[Any] = []
    for name in sorted(definitions):
        expression = model_symbols.expression(definitions[name])
        for parameter_name, replacement in formal.items():
            expression = expression.replace(
                model_symbols.symbol(parameter_name),
                replacement,
            )
        outputs.append(expression)
    exact = _canonical_expressions(outputs, context="model parameter derivation")
    descriptors = _used_input_descriptors(
        outputs,
        tuple(
            _model_parameter_descriptor(
                model,
                formal[name],
                name,
                parameter_indices[name],
            )
            for name in sorted(formal)
        ),
    )
    output_layout = tuple(f"model-parameter:{name}" for name in sorted(definitions))
    payload = {
        "abi": _CATALOG_ABI,
        "contract_kind": "model-parameter",
        "inputs": [descriptor.to_dict() for descriptor in descriptors],
        "outputs": list(exact),
        "output_layout": list(output_layout),
    }
    return _candidate_from_payload(
        "model-parameter",
        payload,
        exact,
        descriptors,
        output_layout,
        proof_class=None,
    )


def _model_parameter_names(model: Model) -> tuple[str, ...]:
    names: set[str] = set()
    defaults_provider = getattr(model, "runtime_parameter_defaults", None)
    if callable(defaults_provider):
        names.update(str(name) for name in defaults_provider())
    definitions_provider = getattr(model, "runtime_derived_parameter_definitions", None)
    if callable(definitions_provider):
        names.update(str(name) for name in definitions_provider())
    for particle_id in _oriented_particle_ids(model):
        names.update(_particle_runtime_parameter_names(model, particle_id))
    return tuple(sorted(names))


def _vertex_runtime_parameter_names(model: Model, kind: int) -> tuple[str, ...]:
    provider = getattr(model, "runtime_parameter_names_for_vertex", None)
    if not callable(provider):
        return ()
    return tuple(sorted(str(name) for name in provider(int(kind))))


def _particle_runtime_parameter_names(
    model: Model, particle_id: int
) -> tuple[str, ...]:
    provider = getattr(model, "runtime_parameter_names_for_particle", None)
    if callable(provider):
        return tuple(sorted(str(name) for name in provider(int(particle_id))))
    particle = model.particle(particle_id)
    names: list[str] = []
    if float(particle.mass) != 0.0:
        names.append(f"particle.{int(particle.pdg)}.mass")
    if float(particle.width) != 0.0:
        names.append(f"particle.{int(particle.pdg)}.width")
    return tuple(names)


def _candidate_from_payload(
    contract_kind: PreparedContractKind,
    payload: Mapping[str, object],
    exact_expressions: tuple[str, ...],
    inputs: tuple[PreparedKernelInput, ...],
    output_layout: tuple[str, ...],
    *,
    proof_class: str | None,
) -> _Candidate:
    declared_symbols = {descriptor.symbol for descriptor in inputs}
    used_symbols = {
        symbol.to_canonical_string()
        for expression in exact_expressions
        for symbol in _sym.E(expression).get_all_symbols(False)
        if not symbol.is_constant()
    }
    unbound_symbols = sorted(used_symbols - declared_symbols)
    if unbound_symbols:
        raise PreparedKernelCatalogError(
            f"prepared {contract_kind} contract has undeclared inputs: "
            f"{', '.join(unbound_symbols)}"
        )
    semantic_payload = {
        "abi": payload.get("abi"),
        "contract_kind": contract_kind,
        "inputs": [descriptor.to_dict() for descriptor in inputs],
        "outputs": list(exact_expressions),
        "output_layout": list(output_layout),
    }
    canonical = _canonical_json(semantic_payload)
    return _Candidate(
        contract_kind=contract_kind,
        canonical_signature=hashlib.sha256(canonical).hexdigest(),
        exact_expressions=exact_expressions,
        inputs=inputs,
        output_layout=output_layout,
        proof_class=proof_class,
    )


def _register_candidate(
    candidates: dict[str, _Candidate],
    proof_classes: dict[str, set[str]],
    candidate: _Candidate,
) -> None:
    existing = candidates.setdefault(candidate.canonical_signature, candidate)
    if (
        existing.contract_kind,
        existing.exact_expressions,
        existing.inputs,
        existing.output_layout,
    ) != (
        candidate.contract_kind,
        candidate.exact_expressions,
        candidate.inputs,
        candidate.output_layout,
    ):
        raise PreparedKernelCatalogError(
            f"prepared kernel SHA-256 collision for {candidate.canonical_signature}"
        )
    if candidate.proof_class:
        proof_classes.setdefault(candidate.canonical_signature, set()).add(
            candidate.proof_class
        )
