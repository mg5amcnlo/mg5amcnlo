# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import cast

import pytest

from pyamplicol.generation.on_the_fly_seed import (
    project_on_the_fly_process_seed_v1,
)
from pyamplicol.generation.recurrence_fermion_pairing import (
    build_recurrence_fermion_pairing_roots_v1,
)
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.base import Model
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.prepared_catalog import build_prepared_kernel_catalog
from pyamplicol.models.recurrence_catalog_builder import (
    build_recurrence_template_catalog,
)
from pyamplicol.models.recurrence_template import (
    CurrentStateTemplateV1,
    EvaluatorBindingV1,
    ExactComplexRationalV1,
    LCColorSourceSeedV1,
    ParameterTemplateV1,
    RecurrenceTemplateCatalog,
    SourceTemplateV1,
)
from pyamplicol.processes.ir import (
    CanonicalProcessIR,
    ColorEndpointSummary,
    ProcessLegIR,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _crossing() -> str:
    return json.dumps(
        {
            "chirality_factor": -1,
            "helicity_factor": -1,
            "momentum_transform": "negate-four-momentum",
            "phase": {
                "imag_denominator": "1",
                "imag_numerator": "0",
                "real_denominator": "1",
                "real_numerator": "1",
            },
            "spin_state_factor": -1,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _state(
    template_id: str,
    particle_id: int,
    *,
    family: str,
    role: str,
    orientation: str,
) -> CurrentStateTemplateV1:
    representation = {"adjoint": 8, "fundamental": 3, "antifundamental": -3}[role]
    shape = {
        "adjoint": "adjoint-segment",
        "fundamental": "fundamental-open-string",
        "antifundamental": "antifundamental-open-string",
    }[role]
    return CurrentStateTemplateV1(
        template_id=template_id,
        particle_id=particle_id,
        anti_particle_id=(particle_id if particle_id == 21 else -particle_id),
        species_id=("species:g" if particle_id == 21 else "species:q"),
        orientation=orientation,
        statistics="boson" if family == "vector" else "fermion",
        color_representation=representation,
        basis=family,
        tensor_ordering=(f"{family}:0", f"{family}:1"),
        dimension=2,
        chirality=0,
        lc_color_shape_kind=shape,
        auxiliary_kind=None,
        mass_parameter_id=None,
        width_parameter_id=None,
    )


def _sources(
    state: CurrentStateTemplateV1,
    family: str,
) -> tuple[SourceTemplateV1, ...]:
    return tuple(
        SourceTemplateV1(
            template_id=f"source:{state.template_id}:{helicity}",
            state_template_id=state.template_id,
            crossing=_crossing(),
            wavefunction_family=family,
            helicity=helicity,
            spin_state=helicity,
            flavour_flow=(state.particle_id,),
            quantum_number_flow=(),
            lc_color_seed=LCColorSourceSeedV1(
                operation="singleton",
                output_shape_kind=state.lc_color_shape_kind,
                component_kind=(
                    "adjoint-segment" if family == "vector" else "open-string"
                ),
                component_role="active",
                proof_digest=_sha(f"source-color:{state.template_id}"),
            ),
            wavefunction_expression_digest=_sha(
                f"source-expression:{state.template_id}:{helicity}"
            ),
            evaluator_resolver_key=f"resolver:{state.template_id}:{helicity}",
        )
        for helicity in (-1, 1)
    )


def _catalog(*, fermions: bool = False) -> RecurrenceTemplateCatalog:
    states = [
        _state(
            "state:g",
            21,
            family="vector",
            role="adjoint",
            orientation="self-conjugate",
        )
    ]
    if fermions:
        states.extend(
            (
                _state(
                    "state:q",
                    1,
                    family="fermion",
                    role="fundamental",
                    orientation="particle",
                ),
                _state(
                    "state:qbar",
                    -1,
                    family="fermion",
                    role="antifundamental",
                    orientation="antiparticle",
                ),
            )
        )
    parameter = ParameterTemplateV1(
        template_id="parameter:gs",
        name="gs",
        parameter_kind="external",
        value_type="real",
        mutable=True,
        default_value=ExactComplexRationalV1(1, 1, 0, 1),
        exact_expression_digest=None,
        dependency_parameter_ids=(),
        prepared_parameter_id=0,
    )
    sources = tuple(
        source
        for state in states
        for source in _sources(
            state, "vector" if state.particle_id == 21 else "fermion"
        )
    )
    state_by_id = {state.template_id: state for state in states}
    bindings = tuple(
        EvaluatorBindingV1(
            resolver_key=source.evaluator_resolver_key,
            prepared_kernel_id=None,
            callable_kind="rusticol-template",
            runtime_template=(
                "rusticol.source-fill."
                f"{source.wavefunction_family}.v1:"
                f"{source.wavefunction_expression_digest[:24]}"
            ),
            contract_kind="source",
            callable_signature=source.wavefunction_expression_digest,
            input_state_template_ids=(),
            output_state_template_id=source.state_template_id,
            input_layout=("momentum:energy",),
            output_layout=tuple(
                f"source-component:{index}"
                for index in range(state_by_id[source.state_template_id].dimension)
            ),
            exact_expression_digests=tuple(
                _sha(f"source-output:{source.template_id}:{index}")
                for index in range(state_by_id[source.state_template_id].dimension)
            ),
            semantic_template_ids=(source.template_id,),
        )
        for source in sources
    )
    return RecurrenceTemplateCatalog.create(
        compiled_model_digest=_sha("model"),
        prepared_kernel_pack_digest=_sha("pack"),
        parameters=(parameter,),
        current_states=tuple(states),
        sources=sources,
        evaluator_bindings=bindings,
    )


def _gluon_process(count: int) -> CanonicalProcessIR:
    return CanonicalProcessIR(
        process="g g > " + " ".join("g" for _ in range(count - 2)),
        key="gluon_process",
        color_accuracy="lc",
        legs=tuple(
            ProcessLegIR(
                label=index + 1,
                side="initial" if index < 2 else "final",
                particle="g",
                outgoing_particle="g",
                pdg=21,
                outgoing_pdg=21,
                statistics="boson",
                wavefunction_family="vector",
                color_role="adjoint",
                source_orientation="self-conjugate",
            )
            for index in range(count)
        ),
        color_endpoints=ColorEndpointSummary(0, 0, 0),
    )


def _many_fermion_process(pair_count: int) -> CanonicalProcessIR:
    legs = tuple(
        ProcessLegIR(
            label=index + 1,
            side="final",
            particle=("q" if index < pair_count else "q~"),
            outgoing_particle=("q" if index < pair_count else "q~"),
            pdg=(1 if index < pair_count else -1),
            outgoing_pdg=(1 if index < pair_count else -1),
            statistics="fermion",
            wavefunction_family="fermion",
            color_role=("fundamental" if index < pair_count else "antifundamental"),
            source_orientation=("particle" if index < pair_count else "antiparticle"),
        )
        for index in range(2 * pair_count)
    )
    return CanonicalProcessIR(
        process="all > " + " ".join(leg.particle for leg in legs),
        key=f"many_fermions_{pair_count}",
        color_accuracy="lc",
        legs=legs,
        color_endpoints=ColorEndpointSummary(pair_count, pair_count, pair_count),
    )


class _NormalizationModel:
    def coupling_order_hierarchies(self) -> dict[str, int]:
        return {"QCD": 1}

    def runtime_normalization_payload(self, _process: object) -> dict[str, object]:
        return {
            "average_factor": 1.0,
            "color_factor": 1.0,
            "couplings_in_stage_evaluators": True,
            "global_coupling_factor": 1.0,
            "identical_factor": 1.0,
        }


def _project(process: CanonicalProcessIR):
    return project_on_the_fly_process_seed_v1(
        process,
        _catalog(),
        cast(Model, _NormalizationModel()),
        coupling_order_policy="minimal",
        coupling_order_limits={"QCD": len(process.legs)},
    )


def test_source_only_projection_never_builds_a_dag_color_plan_or_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol.generation.recurrence_columnar as columnar
    import pyamplicol.generation.service as service

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("materialized process construction was called")

    monkeypatch.setattr(service, "compile_generic_dag", forbidden)
    monkeypatch.setattr(service, "build_color_plan", forbidden)
    monkeypatch.setattr(service, "_invoke_rust_recurrence_lowering_v2", forbidden)
    monkeypatch.setattr(columnar, "build_recurrence_builder_input_v1", forbidden)

    projected = _project(_gluon_process(4))
    assert projected.color_plan is None
    payload = projected.seed.to_json_dict()
    assert set(payload) == {
        "schema_version",
        "process_digest",
        "external_permutation",
        "external_sources",
        "parameter_projection",
        "coupling_order_policy",
        "coupling_hierarchies",
        "coupling_limits",
        "fermion_pairing",
        "normalization",
    }
    assert payload["external_permutation"] == [0, 1, 2, 3]
    assert payload["fermion_pairing"] is None
    assert all(len(row["states"]) == 2 for row in payload["external_sources"])
    encoded = projected.seed.to_json_bytes()
    for forbidden_name in (
        b"physical_sectors",
        b"public_flows",
        b"generic_dag",
        b"direct_plan",
        b"relation_discovery",
    ):
        assert forbidden_name not in encoded


@pytest.mark.parametrize("color_accuracy", ("nlc", "full"))
def test_contracted_projection_keeps_color_plan_outside_seed_v1(
    color_accuracy: str,
) -> None:
    projected = _project(
        replace(_gluon_process(4), color_accuracy=color_accuracy)
    )
    assert projected.color_plan is not None
    assert projected.color_plan.color_accuracy == color_accuracy
    assert projected.color_plan.sector_count == 6
    assert projected.color_plan.trace_reflections_folded is False

    payload = projected.seed.to_json_dict()
    assert "color_plan" not in payload
    assert "color_contraction" not in payload
    assert set(payload) == {
        "schema_version",
        "process_digest",
        "external_permutation",
        "external_sources",
        "parameter_projection",
        "coupling_order_policy",
        "coupling_hierarchies",
        "coupling_limits",
        "fermion_pairing",
        "normalization",
    }


def test_projection_is_deterministic_and_scales_with_external_sources_only() -> None:
    projected = tuple(_project(_gluon_process(count)).seed for count in (4, 6, 8))
    assert projected[0].to_json_bytes() == (
        _project(_gluon_process(4)).seed.to_json_bytes()
    )
    sizes = tuple(len(item.to_json_bytes()) for item in projected)
    assert sizes[1] - sizes[0] == sizes[2] - sizes[1]
    assert tuple(len(item.external_sources) for item in projected) == (4, 6, 8)
    assert tuple(
        sum(len(leg.source_states) for leg in item.external_sources)
        for item in projected
    ) == (8, 12, 16)


def test_projection_omits_limits_absent_from_template_coupling_dimension() -> None:
    projection = _project(_gluon_process(4)).seed

    assert projection.coupling_hierarchies == ()
    assert projection.coupling_limits == ()


def test_vector_execution_states_cross_only_for_initials() -> None:
    catalog = _catalog()
    projection = _project(_gluon_process(4)).seed
    payload = projection.to_json_dict()
    sources = tuple(sorted(catalog.sources, key=lambda row: row.template_id))

    for leg, encoded in zip(
        projection.external_sources,
        payload["external_sources"],
        strict=True,
    ):
        assert isinstance(encoded, dict)
        states = encoded["states"]
        assert isinstance(states, list)
        for state, row in zip(leg.source_states, states, strict=True):
            assert isinstance(row, dict)
            declared = sources[state.source_template_id]
            factor = -1 if leg.is_initial else 1
            assert row["public_helicity"] == declared.helicity * factor
            assert row["source_helicity"] == row["public_helicity"]
            assert row["spin_state"] == declared.spin_state * factor
            assert row["momentum_sign"] == (-1 if leg.is_initial else 1)


def test_builtin_incoming_fermions_keep_crossed_execution_contract() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > z")
    prepared = build_prepared_kernel_catalog(model)
    catalog = build_recurrence_template_catalog(
        model,
        prepared,
        compiled_model_digest="a" * 64,
        prepared_kernel_pack_digest="b" * 64,
    )
    projection = project_on_the_fly_process_seed_v1(
        process,
        catalog,
        model,
        coupling_order_policy="minimal",
        coupling_order_limits={"QED": 1},
    ).seed

    assert tuple(
        (row.name, row.minimum, row.maximum) for row in projection.coupling_limits
    ) == (("QED", 0, 1),)
    assert projection.process_digest == (
        "8395a16bc0f605c05c7c746d7b428b07f175fb869ebd84f856f8886fe139d82f"
    )
    assert projection.external_permutation == (0, 1, 2)
    assert tuple(leg.physical_pdg for leg in projection.external_sources[:2]) == (
        1,
        -1,
    )
    assert tuple(leg.outgoing_pdg for leg in projection.external_sources[:2]) == (
        -1,
        1,
    )
    source_rows = tuple(sorted(catalog.sources, key=lambda row: row.template_id))
    current_rows = tuple(
        sorted(catalog.current_states, key=lambda row: row.template_id)
    )
    current_by_name = {row.template_id: row for row in catalog.current_states}
    encoded_legs = projection.to_json_dict()["external_sources"]
    assert isinstance(encoded_legs, list)

    for leg, encoded in zip(
        projection.external_sources[:2],
        encoded_legs[:2],
        strict=True,
    ):
        assert leg.is_initial and leg.is_fermionic
        assert tuple(state.spin_state for state in leg.source_states) == (1, -1)
        assert isinstance(encoded, dict)
        encoded_states = encoded["states"]
        assert isinstance(encoded_states, list)
        for state, row in zip(leg.source_states, encoded_states, strict=True):
            assert isinstance(row, dict)
            source = source_rows[state.source_template_id]
            canonical_current = current_by_name[source.state_template_id]
            effective_current = current_rows[state.current_state_template_id]
            crossing = json.loads(source.crossing)
            assert row["public_helicity"] == (
                source.helicity * crossing["helicity_factor"]
            )
            assert row["source_helicity"] == row["public_helicity"]
            assert row["spin_state"] == (
                source.spin_state * crossing["spin_state_factor"]
            )
            assert row["chirality"] == effective_current.chirality
            assert row["chirality"] == (
                canonical_current.chirality * crossing["chirality_factor"]
            )
            assert row["momentum_sign"] == -1
            assert row["crossing_phase"] == crossing["phase"]


def test_compact_pairing_roots_do_not_enumerate_or_cap_runtime_pairings() -> None:
    process = _many_fermion_process(9)
    catalog = _catalog(fermions=True)
    roots = build_recurrence_fermion_pairing_roots_v1(
        process,
        catalog.current_states,
        quantum_flows=catalog.quantum_flows,
    )
    assert len(roots.endpoints) == 18
    assert len(roots.classes) == 1
    assert roots.classes[0].fundamental_source_slots == tuple(range(9))
    assert roots.classes[0].antifundamental_source_slots == tuple(range(9, 18))
    # The compact roots stay O(external legs); 9! concrete pairings are absent.
    assert not hasattr(roots, "rules")
