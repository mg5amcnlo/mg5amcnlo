# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyamplicol.models._physics_ir import ContractionIR
from pyamplicol.models.base import (
    Model,
    Particle,
    PropagatorLoweringRule,
    QuantumFlow,
    RecurrenceLCColorTransitionContract,
    RecurrenceLCColorWitnessContract,
    RecurrenceQuantumFlowContract,
    Vertex,
    VertexEvaluationEquivalence,
)
from pyamplicol.models.builtin.model import BuiltinSMModel
from pyamplicol.models.contact_decomposition import (
    CONTACT_ORBIT_ALGORITHM,
    CONTACT_ORBIT_ALGORITHM_VERSION,
    CONTACT_ORBIT_EVALUATOR_CLASS,
    HEFT_CONTACT_ORBIT_EVALUATOR_CLASS,
    CompiledContactOrbitCertificate,
    CompiledContactOrbitStep,
)
from pyamplicol.models.prepared_catalog import (
    PreparedKernelCatalog,
    PreparedKernelCatalogError,
    PreparedKernelInput,
    PreparedKernelSpec,
    PreparedParticleState,
    PreparedPropagatorBinding,
    PreparedVertexBinding,
    PropagatorKernelKey,
    VertexKernelKey,
    build_prepared_kernel_catalog,
)
from pyamplicol.models.recurrence_catalog_builder import (
    _canonical_recurrence_vertex_bindings,
    _canonical_transition_alias_key,
    _singleton_contact_orbit_step_groups,
    build_recurrence_template_catalog,
)
from pyamplicol.models.recurrence_template import (
    ExactComplexRationalV1,
    RecurrenceTemplateCatalog,
    RecurrenceTemplateError,
)

_MODEL_DIGEST = "a" * 64
_PACK_DIGEST = "b" * 64
_PREPARED_ABI = "pyamplicol-prepared-kernel-catalog-v1"


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _signature(
    contract_kind: str,
    inputs: tuple[PreparedKernelInput, ...],
    expressions: tuple[str, ...],
    output_layout: tuple[str, ...],
) -> str:
    payload = {
        "abi": _PREPARED_ABI,
        "contract_kind": contract_kind,
        "inputs": [item.to_dict() for item in inputs],
        "outputs": list(expressions),
        "output_layout": list(output_layout),
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


class _ParameterModel(Model):
    def __init__(self) -> None:
        super().__init__(name="generic-parameter-model")

    def runtime_parameter_defaults(self):
        return {"alpha": (0.1, 0.0)}

    def runtime_parameter_type(self, name):
        assert name == "alpha"
        return "real"

    def runtime_derived_parameter_definitions(self):
        return {"beta": "2*alpha"}

    def runtime_derived_parameter_defaults(self):
        return {"beta": complex(0.2, 0.0)}

    def runtime_normalization_parameter_defaults(self):
        return {"normalization.scale": 1.0}


def _parameter_catalog(*, parameter_index: int = 0) -> PreparedKernelCatalog:
    inputs = (
        PreparedKernelInput(
            role="model-parameter",
            component=0,
            symbol="alpha",
            model_parameter_name="alpha",
            model_parameter_index=parameter_index,
        ),
    )
    expressions = ("2*alpha",)
    output_layout = ("model-parameter:beta",)
    kernel = PreparedKernelSpec(
        kernel_id=0,
        contract_kind="model-parameter",
        canonical_signature=_signature(
            "model-parameter", inputs, expressions, output_layout
        ),
        exact_expressions=expressions,
        inputs=inputs,
        output_layout=output_layout,
    )
    return PreparedKernelCatalog(
        model_name="generic-parameter-model",
        kernels=(kernel,),
        vertex_bindings=(),
        propagator_bindings=(),
        closure_bindings=(),
        model_parameter_kernel_id=0,
    )


def test_parameter_catalog_is_deterministic_and_binary64_exact() -> None:
    model = _ParameterModel()
    first = build_recurrence_template_catalog(
        model,
        _parameter_catalog(),
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )
    second = build_recurrence_template_catalog(
        model,
        _parameter_catalog(),
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    assert first == second
    assert first.canonical_json == second.canonical_json
    alpha = next(item for item in first.parameters if item.name == "alpha")
    assert alpha.default_value == ExactComplexRationalV1.from_binary64(0.1)
    assert alpha.prepared_parameter_id == 0
    beta = next(item for item in first.parameters if item.name == "beta")
    assert beta.parameter_kind == "derived"
    assert beta.default_value is None
    assert beta.prepared_parameter_id is None
    assert len(first.evaluator_bindings) == 1
    assert first.evaluator_bindings[0].semantic_template_ids == (beta.template_id,)


def test_catalog_round_trip_preserves_builder_output() -> None:
    catalog = build_recurrence_template_catalog(
        _ParameterModel(),
        _parameter_catalog(),
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )
    loaded = RecurrenceTemplateCatalog.from_dict(json.loads(catalog.canonical_json))
    assert loaded == catalog


def test_direct_closure_mirror_aliases_are_not_double_counted() -> None:
    model = BuiltinSMModel()
    catalog = build_recurrence_template_catalog(
        model,
        build_prepared_kernel_catalog(model),
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    bindings = {
        semantic_id: binding
        for binding in catalog.evaluator_bindings
        for semantic_id in binding.semantic_template_ids
    }
    aliases: dict[tuple[str, tuple[str, ...]], int] = {}
    for closure in catalog.closures:
        if closure.equivalence_class != "direct-contraction":
            continue
        binding = bindings[closure.template_id]
        key = (
            binding.callable_signature,
            tuple(sorted(closure.input_state_template_ids)),
        )
        aliases[key] = aliases.get(key, 0) + 1

    assert aliases
    assert set(aliases.values()) == {1}


def test_builtin_massive_sources_and_currents_bind_fallback_mass_only() -> None:
    model = BuiltinSMModel()
    catalog = build_recurrence_template_catalog(
        model,
        build_prepared_kernel_catalog(model),
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    top_mass = next(
        parameter
        for parameter in catalog.parameters
        if parameter.name == "particle.6.mass"
    )
    assert top_mass.prepared_parameter_id is not None
    states_by_id = {state.template_id: state for state in catalog.current_states}

    for particle_id in (6, -6):
        currents = tuple(
            state
            for state in catalog.current_states
            if state.particle_id == particle_id
        )
        sources = tuple(
            source
            for source in catalog.sources
            if states_by_id[source.state_template_id].particle_id == particle_id
        )
        assert currents and sources
        assert {state.mass_parameter_id for state in currents} == {top_mass.template_id}
        assert {source.mass_parameter_id for source in sources} == {
            top_mass.template_id
        }
        assert {state.width_parameter_id for state in currents} == {None}
        assert {source.width_parameter_id for source in sources} == {None}

    for particle_id in (1, -1, 21):
        currents = tuple(
            state
            for state in catalog.current_states
            if state.particle_id == particle_id
        )
        sources = tuple(
            source
            for source in catalog.sources
            if states_by_id[source.state_template_id].particle_id == particle_id
        )
        assert currents and sources
        assert {state.mass_parameter_id for state in currents} == {None}
        assert {source.mass_parameter_id for source in sources} == {None}


def test_verified_auxiliary_transition_mirrors_are_not_double_counted() -> None:
    model = BuiltinSMModel()
    catalog = build_recurrence_template_catalog(
        model,
        build_prepared_kernel_catalog(model),
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    transitions = tuple(
        transition
        for transition in catalog.transitions
        if transition.equivalence_class == "builtin-sm:tensor-vector-to-vector"
    )

    # Three exact quantum-flow variants remain.  Their mirrored (g, aux) and
    # (aux, g) model bindings are one certified contribution each, not six
    # independently accumulated recurrence rows.
    assert len(transitions) == 3
    assert {transition.canonical_input_order for transition in transitions} == {(0, 1)}


def test_verified_fermion_pair_transition_mirrors_are_not_double_counted() -> None:
    model = BuiltinSMModel()
    catalog = build_recurrence_template_catalog(
        model,
        build_prepared_kernel_catalog(model),
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )
    states = {state.template_id: state for state in catalog.current_states}

    transitions = tuple(
        transition
        for transition in catalog.transitions
        if transition.equivalence_class == "builtin-sm:fermion-pair-to-vector"
        and {
            states[state_id].particle_id
            for state_id in transition.input_state_template_ids
        }
        == {-11, 11}
        and states[transition.result_state_template_id].particle_id in {22, 23}
        and all(
            states[state_id].dimension == 2
            for state_id in transition.input_state_template_ids
        )
    )

    # One photon and one Z transition survive for each physical chiral pair.
    # The mirrored (fermion, antifermion) and (antifermion, fermion) model
    # orientations are certified evaluator aliases, not independent diagrams.
    assert len(transitions) == 4
    assert {transition.canonical_input_order for transition in transitions} == {(0, 1)}
    assert {
        (
            tuple(
                states[state_id].chirality
                for state_id in transition.input_state_template_ids
            ),
            states[transition.result_state_template_id].particle_id,
        )
        for transition in transitions
    } == {
        ((1, -1), 22),
        ((1, -1), 23),
        ((-1, 1), 22),
        ((-1, 1), 23),
    }


@pytest.mark.parametrize(
    ("direct_kind", "mirrored_kind", "equivalence_class"),
    (
        (10, 23, "builtin-sm:fermion-vector-to-fermion"),
        (11, 24, "builtin-sm:antifermion-vector-to-antifermion"),
    ),
)
def test_verified_fermion_vector_transition_mirrors_share_one_orientation(
    direct_kind: int,
    mirrored_kind: int,
    equivalence_class: str,
) -> None:
    """Keep the two built-in electroweak input orientations as one diagram."""

    model = BuiltinSMModel()

    direct = model.vertex_evaluation_equivalence(direct_kind)
    mirrored = model.vertex_evaluation_equivalence(mirrored_kind)

    assert direct.class_id == mirrored.class_id == equivalence_class
    assert direct.input_order == (0, 1)
    assert mirrored.input_order == (1, 0)
    assert direct.factor == mirrored.factor == (1.0, 0.0)


def test_verified_scalar_vector_transition_mirrors_are_not_double_counted() -> None:
    model = BuiltinSMModel()
    catalog = build_recurrence_template_catalog(
        model,
        build_prepared_kernel_catalog(model),
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )
    states = {state.template_id: state for state in catalog.current_states}

    transitions = tuple(
        transition
        for transition in catalog.transitions
        if transition.equivalence_class == "builtin-sm:scalar-vector-to-vector"
        and tuple(
            states[state_id].particle_id
            for state_id in transition.input_state_template_ids
        )
        == (25, 23)
        and states[transition.result_state_template_id].particle_id == 23
    )

    # One H/Z -> Z transition survives for each physical Z spin state.
    # The mirrored (H, Z) and (Z, H) model orientations are evaluator aliases,
    # not two independently accumulated Higgsstrahlung contributions.
    assert len(transitions) == 3
    assert {transition.canonical_input_order for transition in transitions} == {(0, 1)}
    assert {
        next(
            flow.input_spin_states
            for flow in catalog.quantum_flows
            if flow.template_id == transition.quantum_flow_template_id
        )
        for transition in transitions
    } == {(0, -1), (0, 0), (0, 1)}


def test_concatenate_keep_alias_retains_canonical_parent_order() -> None:
    one = ExactComplexRationalV1.one()

    def concrete_pair(
        canonical: tuple[object, object],
        order: tuple[int, int],
    ) -> tuple[object, object]:
        concrete: list[object] = [None, None]
        for canonical_index, concrete_index in enumerate(order):
            concrete[concrete_index] = canonical[canonical_index]
        return concrete[0], concrete[1]

    def alias_key(order: tuple[int, int]) -> str:
        input_states = concrete_pair(("left", "right"), order)
        input_spins = concrete_pair((-1, 1), order)
        input_flavours = concrete_pair(((1,), (2,)), order)
        input_quantum = concrete_pair(
            ((("charge", "left"),), (("charge", "right"),)),
            order,
        )
        representations = concrete_pair((3, -3), order)
        shape_kinds = concrete_pair(("open-left", "open-right"), order)
        momentum = concrete_pair(("incoming-left", "incoming-right"), order)
        return _canonical_transition_alias_key(
            binding=SimpleNamespace(canonical_input_order=order),
            kernel=SimpleNamespace(
                canonical_signature="callable",
                contract_kind="vertex",
            ),
            flow=SimpleNamespace(
                input_state_template_ids=input_states,
                input_spin_states=input_spins,
                input_flavour_flows=input_flavours,
                input_quantum_number_flows=input_quantum,
                flavour_flow_operation="constant-result",
                result_flavour_flow=(3,),
                quantum_number_flow_operation="constant-result",
                coupling_orders=(("QCD", 1),),
                result_state_template_id="result",
                result_spin_state=0,
                result_quantum_number_flow=(("charge", "result"),),
                exact_coupling=one,
            ),
            color=SimpleNamespace(
                rule_kind="ordered-open-strings",
                input_representations=representations,
                output_representation=8,
                ordered_open_string_arity=2,
                nc_polynomial=((0, one),),
                exact_coefficient=one,
                transition_witnesses=(
                    SimpleNamespace(
                        input_shape_kinds=shape_kinds,
                        input_permutation=(0, 1),
                        reverse_parent_mask=0,
                        component_operation="concatenate-keep",
                        result_component_kind="open-string",
                        result_component_role="none",
                        result_shape_kind="open-string",
                        exact_factor=one,
                        input_port_pairings=(),
                        result_port_bindings=(),
                    ),
                ),
            ),
            transition=SimpleNamespace(
                input_state_template_ids=input_states,
                result_state_template_id="result",
                momentum_convention=momentum,
                coupling_parameter_ids=(),
                contact_orbit_step_template_ids=(),
                contact_orbit_step_semantic_digests=(),
                coupling_orders=(("QCD", 1),),
                binding_coupling=one,
                exact_factor=one,
                output_factor_source="transition",
                equivalence_class="test",
                input_exchange_factor=None,
                output_projection=None,
            ),
        )

    assert alias_key((0, 1)) != alias_key((1, 0))


class _ComplexParameterModel(_ParameterModel):
    def runtime_parameter_defaults(self):
        return {"alpha": (0.1, 0.2)}

    def runtime_parameter_type(self, name):
        assert name == "alpha"
        return "complex"

    def runtime_derived_parameter_defaults(self):
        return {"beta": complex(0.2, 0.4)}


def test_complex_parameter_retains_authoritative_prepared_kernel_index() -> None:
    catalog = build_recurrence_template_catalog(
        _ComplexParameterModel(),
        _parameter_catalog(parameter_index=17),
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    alpha = next(item for item in catalog.parameters if item.name == "alpha")
    assert alpha.value_type == "complex"
    assert alpha.prepared_parameter_id == 17
    assert alpha.default_value == ExactComplexRationalV1.from_binary64(0.1, 0.2)


def test_model_identity_mismatch_is_rejected() -> None:
    catalog = replace(_parameter_catalog(), model_name="different-model")
    with pytest.raises(PreparedKernelCatalogError, match="model identity"):
        build_recurrence_template_catalog(
            _ParameterModel(),
            catalog,
            compiled_model_digest=_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PACK_DIGEST,
        )


def test_stale_prepared_kernel_signature_is_rejected() -> None:
    catalog = _parameter_catalog()
    stale = replace(catalog.kernels[0], exact_expressions=("3*alpha",))
    mutated = replace(catalog, kernels=(stale,))
    with pytest.raises(PreparedKernelCatalogError, match="stale canonical signature"):
        build_recurrence_template_catalog(
            _ParameterModel(),
            mutated,
            compiled_model_digest=_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PACK_DIGEST,
        )


class _ScalarModel(Model):
    def __init__(self) -> None:
        particle = Particle(
            pdg=101,
            anti_pdg=101,
            spin=1,
            dimension=1,
            color_rep=1,
        )
        super().__init__(
            name="generic-scalar-model",
            particles={101: particle},
            vertices=(Vertex(0, (101, 101, 101)),),
        )
        self.source_kernel_id = -1

    def color_rep(self, pdg):
        return self.particle(pdg).color_rep

    def is_fermion(self, pdg):
        del pdg
        return False

    def is_chiral_eligible(self, pdg):
        del pdg
        return False

    def is_fundamental_colored_fermion(self, pdg):
        del pdg
        return False

    def is_massless_adjoint_vector(self, pdg):
        del pdg
        return False

    def quantum_number_flow(self, particle_id):
        del particle_id
        return (("generic-charge", "0"),)

    def recurrence_quantum_flow_contract(
        self, vertex, left_particle_id, right_particle_id
    ):
        return self._standard_recurrence_quantum_flow_contract(
            vertex, left_particle_id, right_particle_id
        )

    def recurrence_lc_color_shape_contract(self, particle_id, chirality=0):
        return self._standard_recurrence_lc_color_shape_contract(particle_id, chirality)

    def recurrence_lc_source_color_contract(self, particle_id, chirality=0):
        return self._standard_recurrence_lc_source_color_contract(
            particle_id, chirality
        )

    def recurrence_lc_color_transition_contract(self, vertex, *, closure):
        return self._standard_recurrence_lc_color_transition_contract(
            vertex, closure=closure
        )

    def vertex_evaluation_equivalence(self, kind):
        assert kind == 0
        return VertexEvaluationEquivalence(class_id="generic-scalar-exact-identity-v1")

    def vertex_coupling_orders(self, vertex):
        assert vertex.kind == 0
        return (("GENERIC", 1),)

    def vertex_color_structure(self, vertex):
        assert vertex.kind == 0
        return "singlet"

    def vertex_color_weight(self, vertex, *, color_accuracy):
        assert vertex.kind == 0
        assert color_accuracy == "lc"
        return (1.0, 0.0)

    def propagator_lowering_rule(self, particle_id, chirality=0):
        assert particle_id == 101
        assert chirality == 0
        return PropagatorLoweringRule(
            particle_id=particle_id,
            chirality=chirality,
            backend="identity",
            full_tensor_network_ready=True,
            applies_propagator=False,
            kernel="generic-scalar-identity",
            kind="identity",
            mass_class="not-applicable",
            auxiliary_policy="external-synthetic-scalar",
        )

    def recurrence_source_kernel_id(self, particle_id, chirality, helicity, spin_state):
        assert (particle_id, chirality, helicity, spin_state) == (101, 0, 0, 0)
        return self.source_kernel_id


def _kernel_namespace(
    *,
    contract_kind: str,
    inputs: tuple[PreparedKernelInput, ...],
    expressions: tuple[str, ...],
    output_layout: tuple[str, ...],
):
    return SimpleNamespace(
        kernel_id=-1,
        contract_kind=contract_kind,
        canonical_signature=_signature(
            contract_kind, inputs, expressions, output_layout
        ),
        exact_expressions=expressions,
        inputs=inputs,
        output_layout=output_layout,
        proof_classes=(),
    )


def _scalar_catalog(
    model: _ScalarModel,
    *,
    vertex_expression: str = "left0*right0",
):
    source_inputs = (PreparedKernelInput(role="momentum", component=0, symbol="p0"),)
    source = _kernel_namespace(
        contract_kind="source",
        inputs=source_inputs,
        expressions=("1",),
        output_layout=("scalar:c0",),
    )
    vertex_inputs = (
        PreparedKernelInput(role="left-current", component=0, symbol="left0"),
        PreparedKernelInput(role="right-current", component=0, symbol="right0"),
    )
    vertex = _kernel_namespace(
        contract_kind="vertex",
        inputs=vertex_inputs,
        expressions=(vertex_expression,),
        output_layout=("scalar:c0",),
    )
    ordered = sorted((source, vertex), key=lambda item: item.canonical_signature)
    for kernel_id, kernel in enumerate(ordered):
        kernel.kernel_id = kernel_id
    model.source_kernel_id = source.kernel_id

    state = PreparedParticleState(
        particle_id=101,
        identity=model._particle_identity_ir(101).canonical_id,
        orientation="self-conjugate",
        basis="scalar",
        chirality=0,
        dimension=1,
    )
    equivalence = model.vertex_evaluation_equivalence(0)
    vertex_binding = PreparedVertexBinding(
        key=VertexKernelKey(0, (101, 101, 101), 0, 0, 0, (1.0, 0.0)),
        kernel_id=vertex.kernel_id,
        canonical_input_order=equivalence.input_order,
        equivalence_class=equivalence.class_id,
        equivalence_factor=equivalence.factor,
        input_exchange_factor=equivalence.input_exchange_factor,
        left_state=state,
        right_state=state,
        result_state=state,
    )
    propagator_binding = PreparedPropagatorBinding(
        key=PropagatorKernelKey(101, 0),
        kernel_id=None,
        state=state,
        applies_propagator=False,
        propagator_kind="identity",
        mass_class="not-applicable",
        gauge=None,
        model_parameters=(),
    )
    return SimpleNamespace(
        model_name=model.name,
        kernels=tuple(ordered),
        vertex_bindings=(vertex_binding,),
        propagator_bindings=(propagator_binding,),
        closure_bindings=(),
        model_parameter_kernel_id=None,
        unsupported_variants=(),
    )


def _scalar_contact_orbit_contract() -> tuple[
    tuple[CompiledContactOrbitCertificate, ...],
    tuple[CompiledContactOrbitStep, ...],
]:
    certificate = CompiledContactOrbitCertificate(
        algorithm=CONTACT_ORBIT_ALGORITHM,
        algorithm_version=CONTACT_ORBIT_ALGORITHM_VERSION,
        term_id=0,
        vertex="V_scalar_contact",
        particles=("s", "s", "s", "s"),
        color_expression="1",
        lorentz_expression="1",
        coupling_expression="i*lam",
        evaluator_class=CONTACT_ORBIT_EVALUATOR_CLASS,
        physical_leg_equivalence_classes=(0, 0, 0, 0),
        reconstruction_factor="1",
    )
    steps = tuple(
        sorted(
            (
                CompiledContactOrbitStep(
                    algorithm=CONTACT_ORBIT_ALGORITHM,
                    algorithm_version=CONTACT_ORBIT_ALGORITHM_VERSION,
                    term_id=0,
                    stage="partial",
                    result_leg=0,
                    left_covered_legs=(1,),
                    right_covered_legs=(2,),
                    source_particle_legs=(1, 2, -1),
                    reconstruction_factor="1",
                ),
                CompiledContactOrbitStep(
                    algorithm=CONTACT_ORBIT_ALGORITHM,
                    algorithm_version=CONTACT_ORBIT_ALGORITHM_VERSION,
                    term_id=0,
                    stage="partial",
                    result_leg=0,
                    left_covered_legs=(2,),
                    right_covered_legs=(1,),
                    source_particle_legs=(2, 1, -1),
                    reconstruction_factor="1",
                ),
                CompiledContactOrbitStep(
                    algorithm=CONTACT_ORBIT_ALGORITHM,
                    algorithm_version=CONTACT_ORBIT_ALGORITHM_VERSION,
                    term_id=0,
                    stage="final",
                    result_leg=0,
                    left_covered_legs=(1, 2),
                    right_covered_legs=(3,),
                    source_particle_legs=(-1, 3, 0),
                    reconstruction_factor="1",
                ),
                CompiledContactOrbitStep(
                    algorithm=CONTACT_ORBIT_ALGORITHM,
                    algorithm_version=CONTACT_ORBIT_ALGORITHM_VERSION,
                    term_id=0,
                    stage="final",
                    result_leg=0,
                    left_covered_legs=(3,),
                    right_covered_legs=(1, 2),
                    source_particle_legs=(3, -1, 0),
                    reconstruction_factor="1",
                ),
            )
        )
    )
    return (certificate,), steps


def _heft_mirror_bindings() -> tuple[
    PreparedVertexBinding, PreparedVertexBinding
]:
    certificate = CompiledContactOrbitCertificate(
        algorithm=CONTACT_ORBIT_ALGORITHM,
        algorithm_version=CONTACT_ORBIT_ALGORITHM_VERSION,
        term_id=91,
        vertex="V_HEFT_HGGG",
        particles=("g", "g", "g", "H"),
        color_expression="f(1,2,3)",
        lorentz_expression="heft-lorentz",
        coupling_expression="heft-coupling",
        evaluator_class=HEFT_CONTACT_ORBIT_EVALUATOR_CLASS,
        physical_leg_equivalence_classes=(0, 0, 0, 1),
        reconstruction_factor="1",
    )
    canonical_step = CompiledContactOrbitStep(
        algorithm=CONTACT_ORBIT_ALGORITHM,
        algorithm_version=CONTACT_ORBIT_ALGORITHM_VERSION,
        term_id=91,
        stage="partial",
        result_leg=0,
        left_covered_legs=(2,),
        right_covered_legs=(3,),
        source_particle_legs=(2, 3, -1),
        reconstruction_factor="1",
    )
    mirrored_step = replace(
        canonical_step,
        left_covered_legs=(3,),
        right_covered_legs=(2,),
        source_particle_legs=(3, 2, -1),
    )
    gluon = PreparedParticleState(21, "g", "self", "vector", 0, 4)
    higgs = PreparedParticleState(25, "H", "self", "scalar", 0, 1)
    auxiliary = PreparedParticleState(
        9_050_001,
        "heft-aux",
        "self",
        "auxiliary",
        0,
        20,
    )
    canonical = PreparedVertexBinding(
        key=VertexKernelKey(100, (21, 25, 9_050_001), 0, 0, 0, (1.0, 0.0)),
        kernel_id=10,
        canonical_input_order=(1, 0),
        equivalence_class="heft-identity",
        equivalence_factor=(1.0, 0.0),
        input_exchange_factor=None,
        left_state=gluon,
        right_state=higgs,
        result_state=auxiliary,
        contact_orbit_certificates=(certificate,),
        contact_orbit_steps=(canonical_step,),
    )
    mirrored = PreparedVertexBinding(
        key=VertexKernelKey(101, (25, 21, 9_050_001), 0, 0, 0, (1.0, 0.0)),
        kernel_id=11,
        canonical_input_order=(0, 1),
        equivalence_class="heft-identity",
        equivalence_factor=(1.0, 0.0),
        input_exchange_factor=None,
        left_state=higgs,
        right_state=gluon,
        result_state=auxiliary,
        contact_orbit_certificates=(certificate,),
        contact_orbit_steps=(mirrored_step,),
    )
    return canonical, mirrored


def test_heft_recurrence_bindings_keep_one_certified_input_orientation() -> None:
    canonical, mirrored = _heft_mirror_bindings()

    assert _canonical_recurrence_vertex_bindings((mirrored, canonical)) == (
        canonical,
    )


def test_heft_recurrence_binding_without_its_mirror_fails_closed() -> None:
    _canonical, mirrored = _heft_mirror_bindings()

    with pytest.raises(
        PreparedKernelCatalogError,
        match="no unique compiler-certified canonical orientation",
    ):
        _canonical_recurrence_vertex_bindings((mirrored,))


class _OrbitScalarModel(_ScalarModel):
    def vertex_contact_orbit_contracts(self, kind):
        assert kind == 0
        return _scalar_contact_orbit_contract()


class _ReverseOrbitScalarModel(_ScalarModel):
    def vertex_contact_orbit_contracts(self, kind):
        assert kind == 0
        certificates, steps = _scalar_contact_orbit_contract()
        return certificates, tuple(reversed(steps))


def _prepared_orbit_scalar_catalog(model, *, reverse_steps: bool = False):
    certificates, steps = _scalar_contact_orbit_contract()
    prepared = _scalar_catalog(model)
    binding = replace(
        prepared.vertex_bindings[0],
        contact_orbit_certificates=certificates,
        contact_orbit_steps=steps,
    )
    if reverse_steps:
        # Exercise the builder's canonicalization independently of the
        # prepared-catalog constructor's normal sorted-input invariant.
        object.__setattr__(binding, "contact_orbit_steps", tuple(reversed(steps)))
    prepared.vertex_bindings = (binding,)
    return prepared


def test_certified_contact_orbit_transitions_are_singleton_and_share_evaluator(
) -> None:
    model = _OrbitScalarModel()
    _, steps = _scalar_contact_orbit_contract()
    prepared = _prepared_orbit_scalar_catalog(model)

    first = build_recurrence_template_catalog(
        model,
        prepared,  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )
    second = build_recurrence_template_catalog(
        model,
        prepared,  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    assert first == second
    assert first.catalog_digest == second.catalog_digest
    assert len(first.transitions) == len(steps)
    assert len({item.template_id for item in first.transitions}) == len(steps)
    assert all(
        len(item.contact_orbit_step_template_ids) == 1
        and len(item.contact_orbit_step_semantic_digests) == 1
        for item in first.transitions
    )
    assert {
        item.contact_orbit_step_template_ids[0] for item in first.transitions
    } == {item.template_id for item in first.contact_orbit_steps}
    vertex_evaluators = tuple(
        item
        for item in first.evaluator_bindings
        if item.contract_kind == "vertex"
    )
    assert len(vertex_evaluators) == 1
    assert vertex_evaluators[0].semantic_template_ids == tuple(
        sorted(item.template_id for item in first.transitions)
    )
    assert {
        item.evaluator_resolver_key for item in first.transitions
    } == {vertex_evaluators[0].resolver_key}


def test_contact_orbit_singleton_expansion_ignores_step_input_order() -> None:
    model = _OrbitScalarModel()
    prepared = _prepared_orbit_scalar_catalog(model)
    catalog = build_recurrence_template_catalog(
        model,
        prepared,  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )
    steps = catalog.contact_orbit_steps
    assert _singleton_contact_orbit_step_groups(steps) == (
        _singleton_contact_orbit_step_groups(tuple(reversed(steps)))
    )

    reverse_model = _ReverseOrbitScalarModel()
    reverse_catalog = build_recurrence_template_catalog(
        reverse_model,
        _prepared_orbit_scalar_catalog(reverse_model, reverse_steps=True),
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )
    assert reverse_catalog == catalog
    assert reverse_catalog.catalog_digest == catalog.catalog_digest


def test_contact_orbit_expansion_preserves_uncertified_catalog_identity() -> None:
    ordinary_model = _ScalarModel()
    ordinary = build_recurrence_template_catalog(
        ordinary_model,
        _scalar_catalog(ordinary_model),
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )
    assert ordinary.catalog_digest == (
        "67c63dda28fc341b090f1b4477b5665d1717465e4282bff364409031176baab3"
    )
    assert tuple(item.template_id for item in ordinary.transitions) == (
        "transition:b0dde38995251bc098e9e1d0",
    )
    ordinary_vertex = next(
        item for item in ordinary.evaluator_bindings if item.contract_kind == "vertex"
    )
    assert (
        ordinary_vertex.resolver_key,
        ordinary_vertex.prepared_kernel_id,
        ordinary_vertex.callable_signature,
        ordinary_vertex.exact_expression_digests,
        ordinary_vertex.semantic_template_ids,
    ) == (
        "evaluator:947ab283dac851256e1b9d95",
        1,
        "fa1ddc519b10118458d46aba316aa6115fea6f481bb06b9e763aca4be7083ef8",
        ("ba1269aaec248bfeaab4026b5cd7694f7516905d4eb8cdbf0177842c5ea238de",),
        ("transition:b0dde38995251bc098e9e1d0",),
    )

    orbit_model = _OrbitScalarModel()
    orbit = build_recurrence_template_catalog(
        orbit_model,
        _prepared_orbit_scalar_catalog(orbit_model),
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )
    orbit_vertex = next(
        item for item in orbit.evaluator_bindings if item.contract_kind == "vertex"
    )
    assert (
        orbit_vertex.resolver_key,
        orbit_vertex.prepared_kernel_id,
        orbit_vertex.callable_signature,
        orbit_vertex.input_state_template_ids,
        orbit_vertex.output_state_template_id,
        orbit_vertex.input_layout,
        orbit_vertex.output_layout,
        orbit_vertex.exact_expression_digests,
        orbit_vertex.callable_kind,
        orbit_vertex.runtime_template,
    ) == (
        ordinary_vertex.resolver_key,
        ordinary_vertex.prepared_kernel_id,
        ordinary_vertex.callable_signature,
        ordinary_vertex.input_state_template_ids,
        ordinary_vertex.output_state_template_id,
        ordinary_vertex.input_layout,
        ordinary_vertex.output_layout,
        ordinary_vertex.exact_expression_digests,
        ordinary_vertex.callable_kind,
        ordinary_vertex.runtime_template,
    )
    assert orbit_vertex.semantic_template_ids == tuple(
        sorted(item.template_id for item in orbit.transitions)
    )


def test_model_generic_scalar_catalog_covers_source_flow_color_and_propagator() -> None:
    model = _ScalarModel()
    catalog = build_recurrence_template_catalog(
        model,
        _scalar_catalog(model),  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    assert len(catalog.current_states) == 1
    assert len(catalog.sources) == 1
    assert len(catalog.quantum_flows) == 1
    assert len(catalog.transitions) == 1
    assert len(catalog.propagators) == 1
    assert not catalog.propagators[0].applies_propagator
    assert catalog.color_contractions[0].rule_kind == "singlet"
    assert len(catalog.runtime_helicity_contracts) == 1
    assert (
        catalog.runtime_helicity_contracts[0].full_state_template_id
        == catalog.current_states[0].template_id
    )
    catalog.require_complete_runtime_helicity_contracts()
    assert {item.contract_kind for item in catalog.evaluator_bindings} == {
        "source",
        "vertex",
    }
    assert "built-in" not in catalog.canonical_json
    assert "ufo" not in catalog.canonical_json.lower()
    assert not catalog.symmetry_proofs


class _AdjointReflectionScalarModel(_ScalarModel):
    def color_rep(self, pdg):
        assert pdg == 101
        return 8

    def vertex_color_structure(self, vertex):
        assert vertex.kind == 0
        return "adjoint-structure-constant"

    def recurrence_lc_color_transition_contract(self, vertex, *, closure):
        assert vertex.kind == 0
        assert not closure
        return RecurrenceLCColorTransitionContract(
            "adjoint-structure-constant",
            (
                RecurrenceLCColorWitnessContract(
                    input_permutation=(0, 1),
                    reverse_parent_mask=0,
                    component_operation="concatenate-join",
                    result_component_kind="adjoint-segment",
                    result_component_role="active",
                ),
                RecurrenceLCColorWitnessContract(
                    input_permutation=(1, 0),
                    reverse_parent_mask=0,
                    component_operation="concatenate-join",
                    result_component_kind="adjoint-segment",
                    result_component_role="active",
                    exact_factor=(-1.0, 0.0),
                ),
            ),
        )


class _ReflectionScalarModel(_AdjointReflectionScalarModel):
    def __init__(self, phase: tuple[float, float]) -> None:
        super().__init__()
        self._reflection_phase = phase

    def adjoint_current_reflection_phase(self, vertex):
        assert vertex.kind == 0
        return self._reflection_phase


def test_transition_reflection_proof_authenticates_kernel_and_model_callback() -> None:
    model = _ReflectionScalarModel((-1.0, 0.0))
    catalog = build_recurrence_template_catalog(
        model,
        _scalar_catalog(model),  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    proof = next(
        item
        for item in catalog.symmetry_proofs
        if item.proof_algorithm == "canonical-current-word-reversal-v1"
    )
    assert proof.subject_template_ids == (catalog.transitions[0].template_id,)
    assert proof.input_permutation == (1, 0)
    assert proof.exact_phase == ExactComplexRationalV1.from_binary64(-1.0)
    assert proof.expression_digests == (hashlib.sha256(b"left0*right0").hexdigest(),)

    changed_callback_model = _ReflectionScalarModel((1.0, 0.0))
    with pytest.raises(RecurrenceTemplateError, match="conflicting"):
        build_recurrence_template_catalog(
            changed_callback_model,
            _scalar_catalog(changed_callback_model),  # type: ignore[arg-type]
            compiled_model_digest=_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PACK_DIGEST,
        )

    changed_kernel_model = _ReflectionScalarModel((-1.0, 0.0))
    changed_kernel = build_recurrence_template_catalog(
        changed_kernel_model,
        _scalar_catalog(  # type: ignore[arg-type]
            changed_kernel_model,
            vertex_expression="right0*left0",
        ),
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    ).symmetry_proofs[0]
    assert changed_kernel.expression_digests != proof.expression_digests
    assert changed_kernel.witness_digest != proof.witness_digest


class _PreparedExchangeReflectionModel(_AdjointReflectionScalarModel):
    def vertex_evaluation_equivalence(self, kind):
        assert kind == 0
        return VertexEvaluationEquivalence(
            class_id="scalar-exchange-proof",
            input_exchange_factor=(-1.0, 0.0),
        )


def test_transition_reflection_proof_accepts_exact_color_witness_pair() -> None:
    model = _AdjointReflectionScalarModel()
    catalog = build_recurrence_template_catalog(
        model,
        _scalar_catalog(model),  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    proof = next(
        item
        for item in catalog.symmetry_proofs
        if item.proof_algorithm == "canonical-current-word-reversal-v1"
    )
    assert proof.subject_template_ids == (catalog.transitions[0].template_id,)
    assert proof.input_permutation == (1, 0)
    assert proof.exact_phase == ExactComplexRationalV1.from_binary64(-1.0)


def test_transition_reflection_proof_accepts_prepared_input_exchange() -> None:
    model = _PreparedExchangeReflectionModel()
    catalog = build_recurrence_template_catalog(
        model,
        _scalar_catalog(model),  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    proof = next(
        item
        for item in catalog.symmetry_proofs
        if item.proof_algorithm == "canonical-current-word-reversal-v1"
    )
    assert proof.subject_template_ids == (catalog.transitions[0].template_id,)
    assert proof.input_permutation == (1, 0)
    assert proof.exact_phase == ExactComplexRationalV1.from_binary64(-1.0)


class _ConflictingPreparedExchangeReflectionModel(_PreparedExchangeReflectionModel):
    def adjoint_current_reflection_phase(self, vertex):
        assert vertex.kind == 0
        return (1.0, 0.0)


def test_prepared_and_callback_reflection_disagreement_fails_closed() -> None:
    model = _ConflictingPreparedExchangeReflectionModel()
    with pytest.raises(
        RecurrenceTemplateError,
        match="conflicting",
    ):
        build_recurrence_template_catalog(
            model,
            _scalar_catalog(model),  # type: ignore[arg-type]
            compiled_model_digest=_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PACK_DIGEST,
        )


class _NondeterministicReflectionModel(_AdjointReflectionScalarModel):
    def __init__(self) -> None:
        super().__init__()
        self._reflection_calls = 0

    def adjoint_current_reflection_phase(self, vertex):
        assert vertex.kind == 0
        self._reflection_calls += 1
        return ((-1.0 if self._reflection_calls % 2 else 1.0), 0.0)


def test_nondeterministic_transition_reflection_callback_fails_closed() -> None:
    model = _NondeterministicReflectionModel()
    with pytest.raises(RecurrenceTemplateError, match="nondeterministic"):
        build_recurrence_template_catalog(
            model,
            _scalar_catalog(model),  # type: ignore[arg-type]
            compiled_model_digest=_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PACK_DIGEST,
        )


def test_source_fill_uses_a_generic_runtime_template() -> None:
    model = _ScalarModel()
    model.recurrence_source_kernel_id = None  # type: ignore[method-assign]
    catalog = build_recurrence_template_catalog(
        model,
        _scalar_catalog(model),  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    source = next(
        binding
        for binding in catalog.evaluator_bindings
        if binding.contract_kind == "source"
    )
    assert source.callable_kind == "rusticol-template"
    assert source.prepared_kernel_id is None
    assert source.runtime_template is not None
    assert source.runtime_template.startswith("rusticol.source-fill.scalar.v1:")


class _GhostFilteringModel(_ScalarModel):
    def __init__(self) -> None:
        super().__init__()
        ghost = Particle(
            pdg=909,
            anti_pdg=-909,
            spin=-1,
            dimension=1,
            color_rep=8,
        )
        anti_ghost = replace(ghost, pdg=-909, anti_pdg=909)
        self.particles = {**self.particles, 909: ghost, -909: anti_ghost}
        self.vertices = (*self.vertices, Vertex(1, (909, -909, 101)))

    def source_wavefunction_kind(self, particle_id):
        if abs(int(particle_id)) == 909:
            return "ghost"
        return super().source_wavefunction_kind(particle_id)


def test_ghost_only_bindings_are_excluded_from_recurrence_semantics() -> None:
    model = _GhostFilteringModel()
    prepared = _scalar_catalog(model)
    physical = prepared.vertex_bindings[0]
    ghost_left = PreparedParticleState(
        particle_id=909,
        identity=model._particle_identity_ir(909).canonical_id,
        orientation=model._particle_identity_ir(909).orientation,
        basis=model._current_basis(909, 0),
        chirality=0,
        dimension=1,
    )
    ghost_right = PreparedParticleState(
        particle_id=-909,
        identity=model._particle_identity_ir(-909).canonical_id,
        orientation=model._particle_identity_ir(-909).orientation,
        basis=model._current_basis(-909, 0),
        chirality=0,
        dimension=1,
    )
    ghost_binding = PreparedVertexBinding(
        key=VertexKernelKey(1, (909, -909, 101), 0, 0, 0, (1.0, 0.0)),
        kernel_id=physical.kernel_id,
        canonical_input_order=(0, 1),
        equivalence_class="ghost-interaction",
        equivalence_factor=(1.0, 0.0),
        input_exchange_factor=None,
        left_state=ghost_left,
        right_state=ghost_right,
        result_state=physical.result_state,
    )
    prepared.vertex_bindings = (physical, ghost_binding)

    catalog = build_recurrence_template_catalog(
        model,
        prepared,  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    assert {state.particle_id for state in catalog.current_states} == {101}
    assert len(catalog.transitions) == 1
    assert all("ghost" not in item.contract_kind for item in catalog.evaluator_bindings)


def test_direct_contraction_uses_exact_runtime_closure_template() -> None:
    model = _ScalarModel()
    model._direct_contraction_ir_by_state = {
        (101, 0, 101, 0): ContractionIR(
            name="generic-scalar-pairing",
            left_basis="scalar",
            right_basis="scalar",
            coefficients=((0.5, 0.0),),
        )
    }

    catalog = build_recurrence_template_catalog(
        model,
        _scalar_catalog(model),  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    assert len(catalog.closures) == 1
    closure = catalog.closures[0]
    assert closure.component_coefficients == (
        ExactComplexRationalV1.from_binary64(0.5),
    )
    evaluator = next(
        binding
        for binding in catalog.evaluator_bindings
        if binding.resolver_key == closure.evaluator_resolver_key
    )
    assert evaluator.callable_kind == "rusticol-template"
    assert evaluator.runtime_template is not None
    assert evaluator.runtime_template.startswith("rusticol.closure-reduce.v1:")


def test_mutated_vertex_equivalence_factor_is_rejected() -> None:
    model = _ScalarModel()
    catalog = _scalar_catalog(model)
    binding = replace(catalog.vertex_bindings[0], equivalence_factor=(-1.0, 0.0))
    catalog.vertex_bindings = (binding,)
    with pytest.raises(PreparedKernelCatalogError, match=r"stale.*proof metadata"):
        build_recurrence_template_catalog(
            model,
            catalog,  # type: ignore[arg-type]
            compiled_model_digest=_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PACK_DIGEST,
        )


class _NondeterministicFlowModel(_ScalarModel):
    def __init__(self) -> None:
        super().__init__()
        self._flow_calls = 0

    def allowed_quantum_flows(self, vertex, left_index, right_index):
        self._flow_calls += 1
        coupling = (1.0 if self._flow_calls % 2 else 2.0, 0.0)
        return (
            QuantumFlow(
                chirality=0,
                spin_state=0,
                flavour_flow=(101,),
                quantum_number_flow=(("generic-charge", "0"),),
                coupling=coupling,
            ),
        )

    def recurrence_quantum_flow_contract(
        self, vertex, left_particle_id, right_particle_id
    ):
        return super().recurrence_quantum_flow_contract(
            vertex, left_particle_id, right_particle_id
        )


def test_nondeterministic_quantum_flow_callback_fails_closed() -> None:
    model = _NondeterministicFlowModel()
    with pytest.raises(RecurrenceTemplateError, match="nondeterministic"):
        build_recurrence_template_catalog(
            model,
            _scalar_catalog(model),  # type: ignore[arg-type]
            compiled_model_digest=_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PACK_DIGEST,
        )


class _UndeclaredDynamicFlowModel(_ScalarModel):
    def allowed_quantum_flows(self, vertex, left_index, right_index):
        return super().allowed_quantum_flows(vertex, left_index, right_index)


def test_quantum_flow_override_requires_an_explicit_matching_contract() -> None:
    model = _UndeclaredDynamicFlowModel()
    with pytest.raises(
        RecurrenceTemplateError,
        match="overrides the callback without declaring a matching recurrence",
    ):
        build_recurrence_template_catalog(
            model,
            _scalar_catalog(model),  # type: ignore[arg-type]
            compiled_model_digest=_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PACK_DIGEST,
        )


class _UndeclaredFlavourCombinationModel(_ScalarModel):
    def combine_flavour_flow(self, result_particle, left_index, right_index):
        return (
            *left_index.flavour_flow,
            *right_index.flavour_flow,
            result_particle,
        )


def test_flavour_combination_override_requires_an_explicit_matching_contract() -> None:
    model = _UndeclaredFlavourCombinationModel()
    with pytest.raises(
        RecurrenceTemplateError,
        match="combine_flavour_flow overrides the callback without declaring "
        "a matching recurrence",
    ):
        build_recurrence_template_catalog(
            model,
            _scalar_catalog(model),  # type: ignore[arg-type]
            compiled_model_digest=_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PACK_DIGEST,
        )


def test_standard_flavour_flow_drops_closed_boson_ancestry_only() -> None:
    model = BuiltinSMModel()
    fermion_left = SimpleNamespace(particle_id=1, flavour_flow=(11,))
    fermion_right = SimpleNamespace(particle_id=-1, flavour_flow=(13,))
    boson = SimpleNamespace(particle_id=21, flavour_flow=(21,))

    assert model.combine_flavour_flow(21, fermion_left, fermion_right) == (21,)
    assert model.combine_flavour_flow(1, fermion_left, boson) == (11, 1)
    assert model.combine_flavour_flow(1, boson, fermion_right) == (13, 1)
    assert (
        model._standard_recurrence_quantum_flow_contract(
            Vertex(kind=0, particles=(1, -1, 21)),
            1,
            -1,
        ).flavour_flow_operation
        == "constant-result"
    )
    assert (
        model._standard_recurrence_quantum_flow_contract(
            Vertex(kind=0, particles=(1, 21, 1)),
            1,
            21,
        ).flavour_flow_operation
        == "append-left-result"
    )
    assert (
        model._standard_recurrence_quantum_flow_contract(
            Vertex(kind=0, particles=(21, -1, 1)),
            21,
            -1,
        ).flavour_flow_operation
        == "append-right-result"
    )


class _DeclaredFlavourCombinationModel(_UndeclaredFlavourCombinationModel):
    def recurrence_quantum_flow_contract(
        self,
        vertex,
        left_particle_id,
        right_particle_id,
    ):
        del vertex, left_particle_id, right_particle_id
        return RecurrenceQuantumFlowContract(
            flavour_flow_operation="concat-left-right-result",
            quantum_number_flow_operation="particle-static-result",
        )


def test_explicit_flavour_combination_contract_preserves_ancestry() -> None:
    model = _DeclaredFlavourCombinationModel()
    catalog = build_recurrence_template_catalog(
        model,
        _scalar_catalog(model),  # type: ignore[arg-type]
        compiled_model_digest=_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PACK_DIGEST,
    )

    assert {
        (
            flow.flavour_flow_operation,
            flow.input_flavour_flows,
            flow.result_flavour_flow,
        )
        for flow in catalog.quantum_flows
    } == {
        (
            "concat-left-right-result",
            ((101,), (101,)),
            (101, 101, 101),
        )
    }


class _DynamicQuantumNumberFlowModel(_ScalarModel):
    def allowed_quantum_flows(self, vertex, left_index, right_index):
        flow = super().allowed_quantum_flows(vertex, left_index, right_index)[0]
        return (
            replace(
                flow,
                quantum_number_flow=tuple(left_index.quantum_number_flow),
            ),
        )

    def recurrence_quantum_flow_contract(
        self, vertex, left_particle_id, right_particle_id
    ):
        return super().recurrence_quantum_flow_contract(
            vertex, left_particle_id, right_particle_id
        )


def test_dynamic_quantum_number_flow_contradicts_static_contract() -> None:
    model = _DynamicQuantumNumberFlowModel()
    with pytest.raises(
        RecurrenceTemplateError,
        match="particle-static quantum-number operation",
    ):
        build_recurrence_template_catalog(
            model,
            _scalar_catalog(model),  # type: ignore[arg-type]
            compiled_model_digest=_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PACK_DIGEST,
        )


class _UnsupportedColorModel(_ScalarModel):
    def vertex_color_structure(self, vertex):
        del vertex
        return "opaque-model-tensor"

    def recurrence_lc_color_transition_contract(self, vertex, *, closure):
        del vertex, closure
        return RecurrenceLCColorTransitionContract(
            "opaque-model-tensor",
            (
                RecurrenceLCColorWitnessContract(
                    input_permutation=(0, 1),
                    reverse_parent_mask=0,
                    component_operation="concatenate-keep",
                    result_component_kind=None,
                    result_component_role="none",
                ),
            ),
        )


class _UndeclaredColorOverrideModel(_ScalarModel):
    def vertex_color_structure(self, vertex):
        del vertex
        return "singlet"


def test_color_callback_override_requires_matching_recurrence_contract() -> None:
    model = _UndeclaredColorOverrideModel()
    with pytest.raises(
        RecurrenceTemplateError,
        match="without declaring a matching recurrence LC color contract",
    ):
        build_recurrence_template_catalog(
            model,
            _scalar_catalog(model),  # type: ignore[arg-type]
            compiled_model_digest=_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PACK_DIGEST,
        )


def test_unsupported_color_semantics_fail_closed() -> None:
    model = _UnsupportedColorModel()
    with pytest.raises(RecurrenceTemplateError, match="cannot encode color rule"):
        build_recurrence_template_catalog(
            model,
            _scalar_catalog(model),  # type: ignore[arg-type]
            compiled_model_digest=_MODEL_DIGEST,
            prepared_kernel_pack_digest=_PACK_DIGEST,
        )


def test_builtin_color_tensor_families_are_model_owned() -> None:
    model = BuiltinSMModel()
    expected = {
        (8, 8, 8): "adjoint-structure-constant",
        (8, 3, 3): "fundamental-generator",
        (3, 8, 3): "fundamental-generator",
        (-3, 3, 8): "fundamental-generator",
        (1, 3, 3): "color-identity",
        (3, -3, 1): "color-identity",
        (1, 1, 1): "singlet",
    }
    observed = {
        representations: model.vertex_color_structure(vertex)
        for vertex in model.vertices
        if (representations := tuple(model.color_rep(p) for p in vertex.particles))
        in expected
    }
    for representations, rule_kind in expected.items():
        assert observed[representations] == rule_kind


def test_builtin_fundamental_color_words_follow_kernel_orientation() -> None:
    model = BuiltinSMModel()
    expected = {
        4: ((1, 0), "open-string"),
        5: ((0, 1), "open-string"),
        6: ((0, 1), "open-string"),
        7: ((1, 0), "open-string"),
        9: ((1, 0), "adjoint-segment"),
    }
    observed: dict[int, tuple[tuple[int, int], str | None]] = {}
    for vertex in model.vertices:
        if vertex.kind not in expected or vertex.kind in observed:
            continue
        contract = model.recurrence_lc_color_transition_contract(
            vertex,
            closure=False,
        )
        assert contract.rule_kind == "fundamental-generator"
        assert len(contract.witnesses) == 1
        witness = contract.witnesses[0]
        assert witness.component_operation == "concatenate-join"
        observed[vertex.kind] = (
            witness.input_permutation,
            witness.result_component_kind,
        )
    assert observed == expected
