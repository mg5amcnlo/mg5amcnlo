# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import numpy as np
import pytest

import pyamplicol.generation.recurrence_template_columnar as template_columnar
from pyamplicol import _rusticol
from pyamplicol.generation.recurrence_columnar import (
    RecurrenceColumn,
)
from pyamplicol.generation.recurrence_template_columnar import (
    RecurrenceColumnarInputError,
    RecurrenceTemplateInputV1,
    build_recurrence_template_input_v1,
)
from pyamplicol.models.contact_decomposition import (
    CONTACT_ORBIT_ALGORITHM_VERSION,
)
from pyamplicol.models.recurrence_template import (
    ClosureTemplateV1,
    ColorContractionTemplateV1,
    ContactOrbitCertificateV1,
    ContactOrbitStepV1,
    CurrentStateTemplateV1,
    EvaluatorBindingV1,
    EvaluatorCallableKind,
    ExactComplexRationalV1,
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

_COMPILED_MODEL_DIGEST = "a" * 64
_PREPARED_PACK_DIGEST = "c" * 64
_EXPRESSION_A = "1" * 64
_EXPRESSION_B = "2" * 64
_EXPRESSION_C = "3" * 64
_PREDICATE = "4" * 64
_WITNESS = "5" * 64
_CALLABLE_A = "6" * 64
_CALLABLE_B = "7" * 64
_CALLABLE_C = "8" * 64
_CALLABLE_D = "9" * 64
_CALLABLE_E = "b" * 64


def _parameter_templates() -> tuple[ParameterTemplateV1, ...]:
    return (
        ParameterTemplateV1(
            template_id="parameter:coupling",
            name="model.coupling",
            parameter_kind="external",
            value_type="complex",
            mutable=True,
            default_value=ExactComplexRationalV1.from_fractions(
                Fraction(1, 3), Fraction(-1, 7)
            ),
            exact_expression_digest=None,
            dependency_parameter_ids=(),
        ),
        ParameterTemplateV1(
            template_id="parameter:mass",
            name="particle.mass",
            parameter_kind="external",
            value_type="real",
            mutable=True,
            default_value=ExactComplexRationalV1.from_binary64(172.5),
            exact_expression_digest=None,
            dependency_parameter_ids=(),
        ),
    )


def _state_templates() -> tuple[CurrentStateTemplateV1, ...]:
    return (
        CurrentStateTemplateV1(
            template_id="state:adjoint",
            particle_id=9000021,
            anti_particle_id=9000021,
            species_id="example:species:adjoint",
            orientation="self-conjugate",
            statistics="boson",
            color_representation=8,
            basis="lorentz-vector",
            tensor_ordering=("mu0", "mu1", "mu2", "mu3"),
            dimension=4,
            chirality=0,
            lc_color_shape_kind="adjoint-segment",
            auxiliary_kind=None,
            mass_parameter_id=None,
            width_parameter_id=None,
        ),
        CurrentStateTemplateV1(
            template_id="state:matter",
            particle_id=9000001,
            anti_particle_id=-9000001,
            species_id="example:species:matter",
            orientation="particle",
            statistics="fermion",
            color_representation=3,
            basis="weyl-chiral",
            tensor_ordering=("spin0", "spin1"),
            dimension=2,
            chirality=1,
            lc_color_shape_kind="fundamental-open-string",
            auxiliary_kind=None,
            mass_parameter_id="parameter:mass",
            width_parameter_id=None,
        ),
    )


def _color_template() -> ColorContractionTemplateV1:
    return ColorContractionTemplateV1(
        template_id="color:matter-adjoint",
        rule_kind="ordered-open-string-append",
        input_representations=(3, 8),
        output_representation=3,
        ordered_open_string_arity=1,
        exact_coefficient=ExactComplexRationalV1.one(),
        nc_polynomial=((0, ExactComplexRationalV1.one()),),
        expression_digest=_EXPRESSION_A,
        transition_witnesses=(
            LCColorTransitionWitnessV1(
                input_shape_kinds=(
                    "fundamental-open-string",
                    "adjoint-segment",
                ),
                input_permutation=(0, 1),
                reverse_parent_mask=0,
                component_operation="concatenate-join",
                result_component_kind="open-string",
                result_component_role="active",
                result_shape_kind="fundamental-open-string",
                exact_factor=ExactComplexRationalV1.one(),
                proof_digest=_WITNESS,
                input_port_pairings=(((0, 0), (1, 1)),),
                result_port_bindings=((1, 0),),
            ),
        ),
    )


def _proof() -> SymmetryProofV1:
    return SymmetryProofV1(
        template_id="proof:matter-linearity",
        proof_algorithm="prepared-kernel-homogeneous-complex-linear-current-v1",
        subject_template_ids=("state:matter",),
        input_permutation=(0,),
        exact_phase=ExactComplexRationalV1.one(),
        expression_digests=(_EXPRESSION_B,),
        witness_digest=_WITNESS,
    )


def _source_templates() -> tuple[SourceTemplateV1, ...]:
    return (
        SourceTemplateV1(
            template_id="source:adjoint:+1",
            state_template_id="state:adjoint",
            crossing="identity",
            wavefunction_family="vector",
            helicity=1,
            spin_state=1,
            flavour_flow=(21,),
            quantum_number_flow=(("electric_charge", "0"),),
            lc_color_seed=LCColorSourceSeedV1(
                operation="singleton",
                output_shape_kind="adjoint-segment",
                component_kind="adjoint-segment",
                component_role="active",
                proof_digest=_EXPRESSION_C,
            ),
            wavefunction_expression_digest=_EXPRESSION_B,
            evaluator_resolver_key="evaluator:source:adjoint",
        ),
        SourceTemplateV1(
            template_id="source:matter:+1",
            state_template_id="state:matter",
            crossing="identity",
            wavefunction_family="fermion",
            helicity=1,
            spin_state=1,
            flavour_flow=(1,),
            quantum_number_flow=(("electric_charge", "-1/3"),),
            lc_color_seed=LCColorSourceSeedV1(
                operation="singleton",
                output_shape_kind="fundamental-open-string",
                component_kind="open-string",
                component_role="active",
                proof_digest=_PREDICATE,
            ),
            wavefunction_expression_digest=_EXPRESSION_A,
            evaluator_resolver_key="evaluator:source:matter",
            mass_parameter_id="parameter:mass",
        ),
    )


def _flow_template() -> QuantumFlowTemplateV1:
    return QuantumFlowTemplateV1(
        template_id="flow:matter-adjoint-to-matter",
        input_state_template_ids=("state:matter", "state:adjoint"),
        input_spin_states=(1, 1),
        input_flavour_flows=((1,), (21,)),
        input_quantum_number_flows=(
            (("electric_charge", "-1/3"),),
            (("electric_charge", "0"),),
        ),
        flavour_flow_operation="append-left-result",
        quantum_number_flow_operation="particle-static-result",
        coupling_orders=(("QCD", 1),),
        result_state_template_id="state:matter",
        result_spin_state=1,
        result_flavour_flow=(1, 9000001),
        result_quantum_number_flow=(("electric_charge", "-1/3"),),
        exact_coupling=ExactComplexRationalV1.one(),
        predicate_digest=_PREDICATE,
    )


def _transition() -> TransitionTemplateV1:
    return TransitionTemplateV1(
        template_id="transition:matter-adjoint-to-matter",
        input_state_template_ids=("state:matter", "state:adjoint"),
        result_state_template_id="state:matter",
        quantum_flow_template_id="flow:matter-adjoint-to-matter",
        evaluator_resolver_key="evaluator:vertex:matter-adjoint",
        canonical_input_order=(0, 1),
        momentum_convention=("incoming-left", "incoming-right"),
        coupling_parameter_ids=("parameter:coupling",),
        coupling_orders=(("QCD", 1),),
        color_contraction_template_id="color:matter-adjoint",
        binding_coupling=ExactComplexRationalV1.one(),
        exact_factor=ExactComplexRationalV1.one(),
        output_factor_source="none",
        equivalence_class="ordered-matter-adjoint",
        input_exchange_factor=None,
        output_projection="weyl-chiral:+1",
    )


def _contact_orbit_records() -> tuple[
    tuple[ContactOrbitCertificateV1, ...],
    tuple[ContactOrbitStepV1, ...],
]:
    certificates = (
        ContactOrbitCertificateV1(
            template_id="contact-orbit-certificate:scalar:0",
            algorithm="compiler-certified-contact-orbit",
            algorithm_version=CONTACT_ORBIT_ALGORITHM_VERSION,
            term_id=0,
            vertex="V_scalar_0",
            particles=("s", "s", "s", "s"),
            evaluator_class=("constant-scalar-literal-singlet-self-conjugate-boson"),
            physical_leg_equivalence_classes=(0, 0, 0, 0),
            reconstruction_factor=ExactComplexRationalV1.one(),
        ),
        ContactOrbitCertificateV1(
            template_id="contact-orbit-certificate:scalar:1",
            algorithm="compiler-certified-contact-orbit",
            algorithm_version=CONTACT_ORBIT_ALGORITHM_VERSION,
            term_id=1,
            vertex="V_scalar_1",
            particles=("s", "s", "t", "u"),
            evaluator_class=("constant-scalar-literal-singlet-self-conjugate-boson"),
            physical_leg_equivalence_classes=(0, 0, 1, 2),
            reconstruction_factor=ExactComplexRationalV1.one(),
        ),
    )
    steps = (
        ContactOrbitStepV1(
            template_id="contact-orbit-step:scalar:partial",
            certificate_template_id=certificates[0].template_id,
            stage="partial",
            result_leg=2,
            left_covered_legs=(0,),
            right_covered_legs=(1,),
            source_particle_legs=(0, 1, -1),
            reconstruction_factor=ExactComplexRationalV1.one(),
        ),
        ContactOrbitStepV1(
            template_id="contact-orbit-step:scalar:final",
            certificate_template_id=certificates[1].template_id,
            stage="final",
            result_leg=3,
            left_covered_legs=(0, 1),
            right_covered_legs=(2,),
            source_particle_legs=(-1, 2, 3),
            reconstruction_factor=ExactComplexRationalV1.one(),
        ),
    )
    return certificates, steps


def _contact_orbit_catalog(
    *, reverse_records: bool = False
) -> RecurrenceTemplateCatalog:
    certificates, steps = _contact_orbit_records()
    bound_steps = tuple(sorted(steps, key=lambda item: item.template_id))
    transition = replace(
        _transition(),
        contact_orbit_step_template_ids=tuple(step.template_id for step in bound_steps),
        contact_orbit_step_semantic_digests=tuple(
            step.semantic_digest for step in bound_steps
        ),
        semantic_digest="",
    )
    if reverse_records:
        certificates = tuple(reversed(certificates))
        steps = tuple(reversed(steps))
    return _catalog(
        contact_orbit_certificates=certificates,
        contact_orbit_steps=steps,
        transitions=(transition,),
    )


def _propagator() -> PropagatorTemplateV1:
    return PropagatorTemplateV1(
        template_id="propagator:matter",
        state_template_id="state:matter",
        applies_propagator=True,
        evaluator_resolver_key="evaluator:propagator:matter",
        numerator_expression_digest=_EXPRESSION_B,
        denominator_expression_digest=_EXPRESSION_C,
        mass_parameter_id="parameter:mass",
        width_parameter_id=None,
        gauge=None,
        linearity_proof_template_id="proof:matter-linearity",
    )


def _identity_propagator() -> PropagatorTemplateV1:
    return PropagatorTemplateV1(
        template_id="propagator:adjoint:identity",
        state_template_id="state:adjoint",
        applies_propagator=False,
        evaluator_resolver_key=None,
        numerator_expression_digest=None,
        denominator_expression_digest=None,
        mass_parameter_id=None,
        width_parameter_id=None,
        gauge=None,
        linearity_proof_template_id=None,
    )


def _closure() -> ClosureTemplateV1:
    return ClosureTemplateV1(
        template_id="closure:matter-pair",
        input_state_template_ids=("state:matter", "state:adjoint"),
        result_state_template_id="state:matter",
        evaluator_resolver_key="evaluator:closure:matter-pair",
        canonical_input_order=(0, 1),
        coupling_parameter_ids=(),
        coupling_orders=(("QCD", 1),),
        eligible_quantum_flow_template_ids=("flow:matter-adjoint-to-matter",),
        color_contraction_template_id="color:matter-adjoint",
        binding_coupling=ExactComplexRationalV1.one(),
        exact_factor=ExactComplexRationalV1.one(),
        output_factor_source="none",
        equivalence_class="ordered-matter-pair",
        input_exchange_factor=None,
        projection="scalar",
    )


def _state_input_layout(*state_ids: str) -> tuple[str, ...]:
    dimensions = {"state:adjoint": 4, "state:matter": 2}
    return tuple(
        f"state:{state_id}:{component}"
        for state_id in state_ids
        for component in range(dimensions[state_id])
    )


def _evaluator_bindings() -> tuple[EvaluatorBindingV1, ...]:
    return (
        EvaluatorBindingV1(
            resolver_key="evaluator:closure:matter-pair",
            prepared_kernel_id=4,
            contract_kind="closure",
            callable_signature=_CALLABLE_E,
            input_state_template_ids=("state:matter", "state:adjoint"),
            output_state_template_id=None,
            input_layout=_state_input_layout("state:matter", "state:adjoint"),
            output_layout=("scalar",),
            exact_expression_digests=(_EXPRESSION_C,),
            semantic_template_ids=("closure:matter-pair",),
        ),
        EvaluatorBindingV1(
            resolver_key="evaluator:propagator:matter",
            prepared_kernel_id=3,
            contract_kind="propagator",
            callable_signature=_CALLABLE_D,
            input_state_template_ids=("state:matter",),
            output_state_template_id="state:matter",
            input_layout=_state_input_layout("state:matter"),
            output_layout=("spin0", "spin1"),
            exact_expression_digests=(_EXPRESSION_A, _EXPRESSION_B),
            semantic_template_ids=("propagator:matter",),
        ),
        EvaluatorBindingV1(
            resolver_key="evaluator:source:adjoint",
            prepared_kernel_id=1,
            contract_kind="source",
            callable_signature=_CALLABLE_B,
            input_state_template_ids=(),
            output_state_template_id="state:adjoint",
            input_layout=("momentum:E",),
            output_layout=("mu0", "mu1", "mu2", "mu3"),
            exact_expression_digests=(
                _EXPRESSION_A,
                _EXPRESSION_A,
                _EXPRESSION_A,
                _EXPRESSION_A,
            ),
            semantic_template_ids=("source:adjoint:+1",),
        ),
        EvaluatorBindingV1(
            resolver_key="evaluator:source:matter",
            prepared_kernel_id=0,
            contract_kind="source",
            callable_signature=_CALLABLE_A,
            input_state_template_ids=(),
            output_state_template_id="state:matter",
            input_layout=("momentum:E", "parameter:mass"),
            output_layout=("spin0", "spin1"),
            exact_expression_digests=(_EXPRESSION_A, _EXPRESSION_B),
            semantic_template_ids=("source:matter:+1",),
        ),
        EvaluatorBindingV1(
            resolver_key="evaluator:vertex:matter-adjoint",
            prepared_kernel_id=2,
            contract_kind="vertex",
            callable_signature=_CALLABLE_C,
            input_state_template_ids=("state:matter", "state:adjoint"),
            output_state_template_id="state:matter",
            input_layout=(
                *_state_input_layout("state:matter", "state:adjoint"),
                "parameter:coupling:real",
                "parameter:coupling:imag",
            ),
            output_layout=("spin0", "spin1"),
            exact_expression_digests=(_EXPRESSION_B, _EXPRESSION_C),
            semantic_template_ids=("transition:matter-adjoint-to-matter",),
        ),
    )


def _catalog(**overrides: object) -> RecurrenceTemplateCatalog:
    values: dict[str, object] = {
        "compiled_model_digest": _COMPILED_MODEL_DIGEST,
        "prepared_kernel_pack_digest": _PREPARED_PACK_DIGEST,
        "parameters": _parameter_templates(),
        "current_states": _state_templates(),
        "sources": _source_templates(),
        "quantum_flows": (_flow_template(),),
        "transitions": (_transition(),),
        "propagators": (_identity_propagator(), _propagator()),
        "closures": (_closure(),),
        "color_contractions": (_color_template(),),
        "symmetry_proofs": (_proof(),),
        "evaluator_bindings": _evaluator_bindings(),
    }
    values.update(overrides)
    return RecurrenceTemplateCatalog.create(**values)  # type: ignore[arg-type]


def _mutated_projected_input(
    table_name: str,
    column_name: str,
    value: int,
    *,
    row: int = 0,
    catalog: RecurrenceTemplateCatalog | None = None,
) -> RecurrenceTemplateInputV1:
    projected = build_recurrence_template_input_v1(
        _catalog() if catalog is None else catalog
    )
    tables = list(projected.tables)
    table_index = next(
        index for index, table in enumerate(tables) if table.name == table_name
    )
    table = tables[table_index]
    columns = list(table.columns)
    column_index = next(
        index for index, column in enumerate(columns) if column.name == column_name
    )
    values = np.array(columns[column_index].values, copy=True, order="C")
    values[row] = value
    values.flags.writeable = False
    columns[column_index] = RecurrenceColumn(
        name=columns[column_index].name,
        values=values,
    )
    tables[table_index] = replace(table, columns=tuple(columns))
    return RecurrenceTemplateInputV1(
        abi=projected.abi,
        catalog_digest=projected.catalog_digest,
        compiled_model_digest=projected.compiled_model_digest,
        prepared_kernel_pack_digest=projected.prepared_kernel_pack_digest,
        tables=tuple(tables),
    )


def test_private_table_adopts_owned_columns_and_copies_views() -> None:
    owned = np.arange(4, dtype=np.dtype("<u4"))
    adopted = template_columnar._table("owned", {"value": owned})

    assert adopted.column("value") is owned
    assert not owned.flags.writeable

    backing = np.arange(8, dtype=np.dtype("<u4"))
    view = backing[::2]
    copied = template_columnar._table("view", {"value": view})
    frozen = copied.column("value")

    assert frozen is not view
    assert frozen.flags.owndata
    assert frozen.flags.c_contiguous
    assert not frozen.flags.writeable
    assert np.array_equal(frozen, view)


def test_binary64_conversion_is_exact_and_distinguishes_sub_ulp_terms() -> None:
    tenth = ExactComplexRationalV1.from_binary64(0.1)
    assert tenth.real == Fraction(3602879701896397, 36028797018963968)
    assert tenth.imag == 0

    one = ExactComplexRationalV1.one()
    one_plus_sub_ulp = ExactComplexRationalV1.from_fractions(
        Fraction(1, 1) + Fraction(1, 2**54)
    )
    assert one != one_plus_sub_ulp
    assert one_plus_sub_ulp.real == Fraction(2**54 + 1, 2**54)


@pytest.mark.parametrize(
    "arguments",
    [
        (2, 4, 0, 1),
        (0, 2, 0, 1),
        (1, -2, 0, 1),
        (1, 2, 0, 0),
    ],
)
def test_exact_rational_rejects_noncanonical_fractions(
    arguments: tuple[int, int, int, int],
) -> None:
    with pytest.raises(RecurrenceTemplateError):
        ExactComplexRationalV1(*arguments)


def test_exact_rational_json_requires_canonical_decimal_integers() -> None:
    payload = ExactComplexRationalV1.one().to_dict()
    payload["real_numerator"] = "+1"
    with pytest.raises(RecurrenceTemplateError, match=r"canonically|decimal"):
        ExactComplexRationalV1.from_dict(payload)


def test_catalog_round_trip_is_canonical_and_content_addressed() -> None:
    catalog = _catalog()
    loaded = RecurrenceTemplateCatalog.from_dict(json.loads(catalog.canonical_json))

    assert loaded == catalog
    assert loaded.canonical_json == catalog.canonical_json
    assert loaded.catalog_digest == catalog.catalog_digest
    assert catalog.current_states[0].template_id == "state:adjoint"
    with pytest.raises(FrozenInstanceError):
        catalog.current_states[0].dimension = 3  # type: ignore[misc]


def test_contact_orbit_records_round_trip_and_ignore_input_row_order() -> None:
    catalog = _contact_orbit_catalog()
    reordered = _contact_orbit_catalog(reverse_records=True)
    restored = RecurrenceTemplateCatalog.from_dict(catalog.to_dict())

    assert restored == catalog
    assert reordered == catalog
    assert reordered.canonical_json == catalog.canonical_json
    assert reordered.catalog_digest == catalog.catalog_digest
    assert restored.transitions[0].contact_orbit_step_template_ids == (
        "contact-orbit-step:scalar:final",
        "contact-orbit-step:scalar:partial",
    )
    assert restored.transitions[0].contact_orbit_step_semantic_digests == (
        restored.contact_orbit_steps[0].semantic_digest,
        restored.contact_orbit_steps[1].semantic_digest,
    )


def test_contact_orbit_semantic_tampering_fails_closed() -> None:
    certificate, steps = _contact_orbit_records()
    with pytest.raises(RecurrenceTemplateError, match="stale semantic digest"):
        replace(certificate[0], term_id=17)

    with pytest.raises(RecurrenceTemplateError, match="not canonical"):
        replace(
            certificate[0],
            physical_leg_equivalence_classes=(0, 1, 2, 3),
            semantic_digest="",
        )

    with pytest.raises(RecurrenceTemplateError, match="partial step lineage"):
        replace(steps[0], source_particle_legs=(1, 0, -1), semantic_digest="")
    with pytest.raises(RecurrenceTemplateError, match="final step lineage"):
        replace(steps[1], left_covered_legs=(0,), semantic_digest="")


def test_contact_orbit_columnar_contract_rejects_transition_digest_tampering() -> None:
    catalog = _contact_orbit_catalog()
    projected = build_recurrence_template_input_v1(catalog)
    tables = {table.name: table for table in projected.tables}

    assert tables["contact_orbit_certificates"].row_count == 2
    assert tables["contact_orbit_steps"].row_count == 2
    assert {
        "contact_orbit_step_sequence_id",
        "contact_orbit_step_semantic_digest_sequence_id",
    } <= {column.name for column in tables["transitions"].columns}
    result = _rusticol._validate_recurrence_template_input_v1(
        projected,
        [0, 1, 2, 3, 4],
    )
    assert result["counts"]["contact_orbit_certificates"] == 2
    assert result["counts"]["contact_orbit_steps"] == 2

    transition_table = tables["transitions"]
    sequence_id = int(
        transition_table.column("contact_orbit_step_semantic_digest_sequence_id")[0]
    )
    sequence_ranges = tables["u32_sequence_ranges"]
    start = int(sequence_ranges.column("start")[sequence_id])
    different_valid_digest_id = int(transition_table.column("semantic_digest_id")[0])
    mismatched = _mutated_projected_input(
        "u32_sequence_values",
        "value",
        different_valid_digest_id,
        row=start,
        catalog=catalog,
    )
    with pytest.raises(ValueError, match="step digest does not match"):
        _rusticol._validate_recurrence_template_input_v1(
            mismatched,
            [0, 1, 2, 3, 4],
        )


def test_contact_orbit_native_certificate_contract_rejects_tampering() -> None:
    catalog = _contact_orbit_catalog()
    projected = build_recurrence_template_input_v1(catalog)
    tables = {table.name: table for table in projected.tables}
    certificates = tables["contact_orbit_certificates"]
    cases = (
        (
            _rusticol.CompatibilityError,
            "algorithm_string_id",
            int(certificates.column("evaluator_class_string_id")[0]),
            "contact-orbit certificate algorithm",
        ),
        (
            _rusticol.CompatibilityError,
            "evaluator_class_string_id",
            int(certificates.column("algorithm_string_id")[0]),
            "contact-orbit evaluator class",
        ),
        (
            ValueError,
            "reconstruction_factor_id",
            int(tables["parameters"].column("default_factor_id")[0]),
            "reconstruction factor is not one",
        ),
    )
    for error_type, column_name, value, message in cases:
        tampered = _mutated_projected_input(
            "contact_orbit_certificates",
            column_name,
            value,
            catalog=catalog,
        )
        with pytest.raises(error_type, match=message):
            _rusticol._validate_recurrence_template_input_v1(
                tampered,
                [0, 1, 2, 3, 4],
            )


def test_contact_orbit_native_certificate_rows_reject_reordering() -> None:
    projected = build_recurrence_template_input_v1(_contact_orbit_catalog())
    tables = list(projected.tables)
    table_index = next(
        index
        for index, table in enumerate(tables)
        if table.name == "contact_orbit_certificates"
    )
    table = tables[table_index]
    reversed_columns = []
    for column in table.columns:
        values = np.ascontiguousarray(column.values[::-1])
        values.flags.writeable = False
        reversed_columns.append(RecurrenceColumn(name=column.name, values=values))
    columns = tuple(reversed_columns)
    tables[table_index] = replace(table, columns=columns)
    reordered = replace(projected, tables=tuple(tables))

    with pytest.raises(
        ValueError,
        match="contact-orbit certificate row 0 has noncanonical id 1",
    ):
        _rusticol._validate_recurrence_template_input_v1(
            reordered,
            [0, 1, 2, 3, 4],
        )


def test_contact_orbit_native_step_lineage_tampering_fails_closed() -> None:
    projected = build_recurrence_template_input_v1(_contact_orbit_catalog())
    tables = list(projected.tables)
    table_index = next(
        index
        for index, table in enumerate(tables)
        if table.name == "contact_orbit_steps"
    )
    table = tables[table_index]
    columns = list(table.columns)
    stage_index = next(
        index for index, column in enumerate(columns) if column.name == "stage"
    )
    values = np.array(columns[stage_index].values, copy=True, order="C")
    values[0] = 0
    values.flags.writeable = False
    columns[stage_index] = RecurrenceColumn(name="stage", values=values)
    tables[table_index] = replace(table, columns=tuple(columns))
    tampered = replace(projected, tables=tuple(tables))

    with pytest.raises(ValueError, match="partial step 0 has inconsistent lineage"):
        _rusticol._validate_recurrence_template_input_v1(
            tampered,
            [0, 1, 2, 3, 4],
        )


def test_extended_recurrence_records_round_trip_exactly() -> None:
    catalog = _catalog()
    restored = RecurrenceTemplateCatalog.from_dict(catalog.to_dict())

    assert restored.quantum_flows[0].input_flavour_flows == ((1,), (21,))
    assert restored.quantum_flows[0].input_quantum_number_flows == (
        (("electric_charge", "-1/3"),),
        (("electric_charge", "0"),),
    )
    assert restored.quantum_flows[0].exact_coupling == ExactComplexRationalV1.one()
    assert restored.quantum_flows[0].flavour_flow_operation == "append-left-result"
    assert restored.quantum_flows[0].result_spin_state == 1
    assert restored.sources[0].flavour_flow == (21,)
    assert restored.sources[0].quantum_number_flow == (("electric_charge", "0"),)
    assert restored.transitions[0].equivalence_class == "ordered-matter-adjoint"
    assert restored.transitions[0].output_factor_source == "none"
    assert restored.closures[0].equivalence_class == "ordered-matter-pair"
    assert restored.closures[0].input_exchange_factor is None
    assert restored.closures[0].eligible_quantum_flow_template_ids == (
        "flow:matter-adjoint-to-matter",
    )

    payload = json.loads(catalog.canonical_json)
    flow = payload["quantum_flows"][0]
    assert flow["input_flavour_flows"] == [[1], [21]]
    assert flow["flavour_flow_operation"] == "append-left-result"
    assert flow["input_quantum_number_flows"][0] == [["electric_charge", "-1/3"]]


@pytest.mark.parametrize(
    ("record_name", "field", "value"),
    [
        ("quantum_flows", "exact_coupling", ExactComplexRationalV1.zero()),
        ("transitions", "binding_coupling", ExactComplexRationalV1.zero()),
        ("transitions", "output_factor_source", "coupling-real"),
        ("transitions", "equivalence_class", "another-proof"),
        (
            "transitions",
            "input_exchange_factor",
            ExactComplexRationalV1.from_fractions(-1),
        ),
        ("closures", "binding_coupling", ExactComplexRationalV1.zero()),
        ("closures", "output_factor_source", "coupling-imag"),
        ("closures", "equivalence_class", "another-closure-proof"),
        (
            "closures",
            "input_exchange_factor",
            ExactComplexRationalV1.from_fractions(-1),
        ),
    ],
)
def test_extended_fields_affect_semantic_and_catalog_digests(
    record_name: str,
    field: str,
    value: object,
) -> None:
    catalog = _catalog()
    records = getattr(catalog, record_name)
    updated = replace(records[0], **{field: value, "semantic_digest": ""})
    changed = _catalog(**{record_name: (updated,)})

    assert updated.semantic_digest != records[0].semantic_digest
    assert changed.catalog_digest != catalog.catalog_digest


def test_model_wide_columnar_projection_preserves_typed_contracts() -> None:
    projected = build_recurrence_template_input_v1(_catalog())
    tables = {table.name: table for table in projected.tables}

    assert projected.canonical_digest == projected.digest
    assert tables["flavour_flow_ranges"].row_count == 3
    assert tables["quantum_number_flow_ranges"].row_count == 2
    assert tables["quantum_flows"].row_count == 1
    assert tables["transitions"].row_count == 1
    assert tables["closures"].row_count == 1
    assert tables["lc_color_transition_witnesses"].row_count == 1
    assert "exact_coupling_factor_id" in {
        column.name for column in tables["quantum_flows"].columns
    }
    assert "binding_coupling_factor_id" in {
        column.name for column in tables["transitions"].columns
    }
    assert {"flavour_flow_id", "quantum_number_flow_id"} <= {
        column.name for column in tables["sources"].columns
    }
    assert "result_spin_state" in {
        column.name for column in tables["quantum_flows"].columns
    }
    assert "flavour_flow_operation_string_id" in {
        column.name for column in tables["quantum_flows"].columns
    }
    assert "eligible_quantum_flow_sequence_id" in {
        column.name for column in tables["closures"].columns
    }
    assert "lc_color_shape_string_id" in {
        column.name for column in tables["current_states"].columns
    }
    assert {"witness_start", "witness_count"} <= {
        column.name for column in tables["color_contractions"].columns
    }


def _runtime_helicity_catalog() -> RecurrenceTemplateCatalog:
    states = _state_templates()
    full_matter = replace(
        states[1],
        template_id="state:matter-full",
        basis="dirac",
        tensor_ordering=("spin0", "spin1", "spin2", "spin3"),
        dimension=4,
        chirality=0,
        semantic_digest="",
    )
    one = ExactComplexRationalV1.one()
    zero = ExactComplexRationalV1.zero()
    contracts = (
        RecurrenceRuntimeHelicityContractV1(
            template_id="runtime-helicity:adjoint",
            full_state_template_id="state:adjoint",
            variants=(
                RecurrenceRuntimeHelicityVariantV1(
                    source_template_id="source:adjoint:+1",
                    source_state_template_id="state:adjoint",
                    embedding_source_components=(0, 1, 2, 3),
                    embedding_factors=(one, one, one, one),
                    projection_full_components=(0, 1, 2, 3),
                    proof_digest=_EXPRESSION_A,
                ),
            ),
            proof_algorithm="prepared-current-ordering-runtime-helicity-embedding-v1",
            proof_digest=_EXPRESSION_B,
        ),
        RecurrenceRuntimeHelicityContractV1(
            template_id="runtime-helicity:matter",
            full_state_template_id="state:matter-full",
            variants=(
                RecurrenceRuntimeHelicityVariantV1(
                    source_template_id="source:matter:+1",
                    source_state_template_id="state:matter",
                    embedding_source_components=(0, 1, None, None),
                    embedding_factors=(one, one, zero, zero),
                    projection_full_components=(0, 1),
                    proof_digest=_EXPRESSION_C,
                ),
            ),
            proof_algorithm="prepared-current-ordering-runtime-helicity-embedding-v1",
            proof_digest=_PREDICATE,
        ),
    )
    return _catalog(
        current_states=(*states, full_matter),
        runtime_helicity_contracts=contracts,
    )


def test_runtime_helicity_contract_round_trips_and_flattens() -> None:
    catalog = _runtime_helicity_catalog()
    restored = RecurrenceTemplateCatalog.from_dict(catalog.to_dict())
    restored.require_complete_runtime_helicity_contracts()

    assert restored == catalog
    tables = {
        table.name: table
        for table in build_recurrence_template_input_v1(catalog).tables
    }
    assert tables["runtime_helicity_contracts"].row_count == 2
    assert tables["runtime_helicity_variants"].row_count == 2
    assert tables["runtime_helicity_embeddings"].row_count == 8
    assert tables["runtime_helicity_projections"].row_count == 6


def test_prepared_pack_digest_rebind_preserves_runtime_helicity_contracts() -> None:
    from pyamplicol.models.prepared_compile import (
        _rebind_recurrence_template_pack_digest,
    )

    catalog = _runtime_helicity_catalog()
    rebound = _rebind_recurrence_template_pack_digest(catalog, "f" * 64)

    assert rebound.header.prepared_kernel_pack_digest == "f" * 64
    assert rebound.runtime_helicity_contracts == catalog.runtime_helicity_contracts


def test_prepared_pack_digest_rebind_preserves_contact_orbit_records() -> None:
    from pyamplicol.models.prepared_compile import (
        _rebind_recurrence_template_pack_digest,
    )

    catalog = _contact_orbit_catalog()
    rebound = _rebind_recurrence_template_pack_digest(catalog, "f" * 64)

    assert rebound.header.prepared_kernel_pack_digest == "f" * 64
    assert rebound.contact_orbit_certificates == catalog.contact_orbit_certificates
    assert rebound.contact_orbit_steps == catalog.contact_orbit_steps
    assert rebound.transitions == catalog.transitions


def test_runtime_helicity_preflight_reports_uncertified_source() -> None:
    with pytest.raises(
        RecurrenceTemplateError,
        match=r"recurrence all-flow-union preflight.*source:adjoint",
    ):
        _catalog().require_complete_runtime_helicity_contracts()


def test_runtime_helicity_embedding_must_be_exactly_invertible() -> None:
    one = ExactComplexRationalV1.one()
    zero = ExactComplexRationalV1.zero()
    with pytest.raises(RecurrenceTemplateError, match="projection must invert"):
        RecurrenceRuntimeHelicityVariantV1(
            source_template_id="source:matter:+1",
            source_state_template_id="state:matter",
            embedding_source_components=(0, 1, None, None),
            embedding_factors=(one, one, zero, zero),
            projection_full_components=(1, 0),
            proof_digest=_EXPRESSION_A,
        )


def test_native_builder_exposes_only_the_direct_arena_v2_entry_point() -> None:
    assert callable(_rusticol._lower_recurrence_direct_v2)
    assert not hasattr(_rusticol, "_validate_recurrence_builder_input_v1")


@pytest.mark.parametrize("numerator", (1 << 127, -(1 << 127)))
def test_model_wide_columnar_projection_rejects_i128_overflow(numerator: int) -> None:
    transition = replace(
        _transition(),
        binding_coupling=ExactComplexRationalV1.from_fractions(numerator),
        semantic_digest="",
    )
    catalog = _catalog(transitions=(transition,))

    with pytest.raises(RecurrenceColumnarInputError, match=r"cannot cross.*i128"):
        build_recurrence_template_input_v1(catalog)


def test_model_wide_columnar_projection_rejects_fixed_width_overflow() -> None:
    states = list(_state_templates())
    states[0] = replace(states[0], particle_id=1 << 31, semantic_digest="")
    catalog = _catalog(current_states=tuple(states))

    with pytest.raises(
        RecurrenceColumnarInputError,
        match=r"current_states\.particle_id row 0.*does not fit <i4",
    ):
        build_recurrence_template_input_v1(catalog)


def test_runtime_template_evaluator_binding_round_trips() -> None:
    bindings = list(_evaluator_bindings())
    source_index = next(
        index
        for index, binding in enumerate(bindings)
        if binding.contract_kind == "source"
    )
    bindings[source_index] = replace(
        bindings[source_index],
        prepared_kernel_id=None,
        callable_kind="rusticol-template",
        runtime_template=(
            f"rusticol.source-fill.vector.v1:"
            f"{bindings[source_index].callable_signature[:24]}"
        ),
        semantic_digest="",
    )

    catalog = _catalog(evaluator_bindings=tuple(bindings))
    restored = RecurrenceTemplateCatalog.from_dict(catalog.to_dict())
    binding = restored.evaluator_bindings[source_index]
    assert binding.callable_kind == "rusticol-template"
    assert binding.runtime_template == (
        f"rusticol.source-fill.vector.v1:{binding.callable_signature[:24]}"
    )
    assert binding.prepared_kernel_id is None


def test_runtime_template_evaluator_binding_rejects_unregistered_contract() -> None:
    binding = next(
        item for item in _evaluator_bindings() if item.contract_kind == "source"
    )

    with pytest.raises(RecurrenceTemplateError, match="authenticated"):
        replace(
            binding,
            prepared_kernel_id=None,
            callable_kind="rusticol-template",
            runtime_template="rusticol.source-fill.vector.v1:stale",
            semantic_digest="",
        )


@pytest.mark.parametrize(
    ("prepared_kernel_id", "callable_kind", "runtime_template"),
    [
        (None, "prepared-kernel", None),
        (1, "rusticol-template", "rusticol.source-fill.vector.v1"),
        (None, "rusticol-template", None),
        (1, "prepared-kernel", "rusticol.source-fill.vector.v1"),
    ],
)
def test_evaluator_binding_requires_exactly_one_callable_lane(
    prepared_kernel_id: int | None,
    callable_kind: EvaluatorCallableKind,
    runtime_template: str | None,
) -> None:
    binding = _evaluator_bindings()[0]
    with pytest.raises(RecurrenceTemplateError):
        replace(
            binding,
            prepared_kernel_id=prepared_kernel_id,
            callable_kind=callable_kind,
            runtime_template=runtime_template,
            semantic_digest="",
        )


def test_catalog_creation_sorts_records_deterministically() -> None:
    forward = _catalog()
    reverse = _catalog(
        parameters=tuple(reversed(_parameter_templates())),
        current_states=tuple(reversed(_state_templates())),
        sources=tuple(reversed(_source_templates())),
        evaluator_bindings=tuple(reversed(_evaluator_bindings())),
    )
    assert reverse.canonical_json == forward.canonical_json
    assert reverse.catalog_digest == forward.catalog_digest


def test_record_rejects_stale_semantic_digest() -> None:
    source = _source_templates()[0]
    with pytest.raises(RecurrenceTemplateError, match="stale semantic digest"):
        replace(source, helicity=-1)


def test_concrete_source_and_quantum_spins_reject_dynamic_union_sentinel() -> None:
    sentinel = -(1 << 31)
    with pytest.raises(
        RecurrenceTemplateError,
        match="source spin_state uses the reserved dynamic-union sentinel",
    ):
        replace(_source_templates()[0], spin_state=sentinel, semantic_digest="")
    with pytest.raises(
        RecurrenceTemplateError,
        match="quantum-flow spin states uses the reserved dynamic-union sentinel",
    ):
        replace(
            _flow_template(),
            input_spin_states=(sentinel, 1),
            semantic_digest="",
        )
    with pytest.raises(
        RecurrenceTemplateError,
        match="quantum-flow result spin state uses the reserved dynamic-union sentinel",
    ):
        replace(
            _flow_template(),
            result_spin_state=sentinel,
            semantic_digest="",
        )


def test_catalog_rejects_stale_catalog_digest() -> None:
    catalog = _catalog()
    stale_header = replace(catalog.header, catalog_digest="f" * 64)
    with pytest.raises(RecurrenceTemplateError, match=r"stale.*catalog digest"):
        replace(catalog, header=stale_header)


def test_catalog_rejects_duplicate_semantic_identity() -> None:
    first = _parameter_templates()[0]
    duplicate = replace(
        first,
        name="another.parameter",
        semantic_digest="",
    )
    with pytest.raises(RecurrenceTemplateError, match="duplicate semantic identity"):
        _catalog(parameters=(first, duplicate, _parameter_templates()[1]))


def test_catalog_rejects_indirect_parameter_dependency_cycle() -> None:
    parameters = _parameter_templates()
    cyclic = (
        replace(
            parameters[0],
            dependency_parameter_ids=(parameters[1].template_id,),
            semantic_digest="",
        ),
        replace(
            parameters[1],
            dependency_parameter_ids=(parameters[0].template_id,),
            semantic_digest="",
        ),
    )

    with pytest.raises(
        RecurrenceTemplateError,
        match="parameter dependency graph contains a cycle",
    ):
        _catalog(parameters=cyclic)


def test_catalog_rejects_duplicate_evaluator_resolver_key() -> None:
    first = _evaluator_bindings()[0]
    duplicate = replace(
        first,
        prepared_kernel_id=99,
        callable_signature="c" * 64,
        semantic_digest="",
    )
    with pytest.raises(RecurrenceTemplateError, match="resolver keys must be unique"):
        _catalog(evaluator_bindings=(*_evaluator_bindings(), duplicate))


def test_unknown_proof_algorithm_fails_closed() -> None:
    with pytest.raises(RecurrenceTemplateError, match=r"unsupported.*algorithm"):
        replace(
            _proof(),
            proof_algorithm="trust-the-model-name-v1",
            semantic_digest="",
        )


def test_catalog_rejects_unknown_state_reference() -> None:
    transition = replace(
        _transition(),
        result_state_template_id="state:missing",
        semantic_digest="",
    )
    with pytest.raises(RecurrenceTemplateError, match="unknown 'state:missing'"):
        _catalog(transitions=(transition,))


def test_catalog_rejects_transition_quantum_flow_coupling_mismatch() -> None:
    transition = replace(
        _transition(),
        coupling_orders=(),
        semantic_digest="",
    )
    with pytest.raises(
        RecurrenceTemplateError,
        match="transition and quantum-flow state/coupling contracts",
    ):
        _catalog(transitions=(transition,))


def test_catalog_rejects_prepared_closure_without_quantum_flow_witness() -> None:
    closure = replace(
        _closure(),
        eligible_quantum_flow_template_ids=(),
        semantic_digest="",
    )
    with pytest.raises(
        RecurrenceTemplateError,
        match="prepared closure must carry at least one eligible quantum-flow",
    ):
        _catalog(closures=(closure,))


@pytest.mark.parametrize(
    ("table_name", "column_name", "value", "message"),
    [
        (
            "quantum_flows",
            "flavour_flow_operation_string_id",
            0,
            "unsupported flavour operation",
        ),
        (
            "transitions",
            "coupling_order_set_id",
            0,
            "state/coupling contracts do not match",
        ),
        (
            "closures",
            "eligible_quantum_flow_sequence_id",
            0,
            "prepared closure 0 has no eligible quantum-flow witness",
        ),
        (
            "closures",
            "result_state_template_id",
            0,
            "different result-state contracts",
        ),
        (
            "evaluator_bindings",
            "callable_kind",
            1,
            "direct Rusticol closure 0 carries prepared quantum-flow witnesses",
        ),
        (
            "color_contractions",
            "witness_count",
            0,
            "LC color transition witnesses",
        ),
        (
            "lc_color_transition_witnesses",
            "component_operation",
            6,
            "LC color witness operation 6 is not supported",
        ),
        (
            "lc_color_transition_witnesses",
            "result_component_kind",
            255,
            "LC color join witness requires component kind and role",
        ),
    ],
)
def test_native_catalog_validation_rejects_mutated_dynamic_contracts(
    table_name: str,
    column_name: str,
    value: int,
    message: str,
) -> None:
    projected = _mutated_projected_input(table_name, column_name, value)
    with pytest.raises(ValueError, match=message):
        _rusticol._validate_recurrence_template_input_v1(
            projected,
            [0, 1, 2, 3, 4],
        )


def test_catalog_rejects_closure_quantum_flow_state_mismatch() -> None:
    closure = replace(
        _closure(),
        input_state_template_ids=("state:matter", "state:matter"),
        semantic_digest="",
    )
    with pytest.raises(
        RecurrenceTemplateError,
        match="closure and eligible quantum-flow input/result/coupling contracts",
    ):
        _catalog(closures=(closure,))


def test_catalog_rejects_closure_quantum_flow_result_state_mismatch() -> None:
    closure = replace(
        _closure(),
        result_state_template_id="state:adjoint",
        semantic_digest="",
    )
    with pytest.raises(
        RecurrenceTemplateError,
        match="closure and eligible quantum-flow input/result/coupling contracts",
    ):
        _catalog(closures=(closure,))


def test_catalog_rejects_closure_quantum_flow_coupling_mismatch() -> None:
    closure = replace(
        _closure(),
        coupling_orders=(),
        semantic_digest="",
    )
    with pytest.raises(
        RecurrenceTemplateError,
        match="closure and eligible quantum-flow input/result/coupling contracts",
    ):
        _catalog(closures=(closure,))


def test_catalog_rejects_malformed_evaluator_state_contract() -> None:
    bindings = list(_evaluator_bindings())
    index = next(
        index
        for index, binding in enumerate(bindings)
        if binding.resolver_key == "evaluator:vertex:matter-adjoint"
    )
    bindings[index] = replace(
        bindings[index],
        input_state_template_ids=("state:adjoint", "state:matter"),
        semantic_digest="",
    )
    with pytest.raises(RecurrenceTemplateError, match="input states do not match"):
        _catalog(evaluator_bindings=tuple(bindings))


def test_catalog_rejects_malformed_evaluator_output_dimension() -> None:
    bindings = list(_evaluator_bindings())
    index = next(
        index
        for index, binding in enumerate(bindings)
        if binding.resolver_key == "evaluator:source:matter"
    )
    bindings[index] = replace(
        bindings[index],
        output_layout=("spin0",),
        exact_expression_digests=(_EXPRESSION_A,),
        semantic_digest="",
    )
    with pytest.raises(RecurrenceTemplateError, match="output layout"):
        _catalog(evaluator_bindings=tuple(bindings))


def test_catalog_loader_rejects_unknown_fields() -> None:
    payload = _catalog().to_dict()
    payload["model_name"] = "forbidden-model-specific-dispatch"
    with pytest.raises(RecurrenceTemplateError, match="unknown=model_name"):
        RecurrenceTemplateCatalog.from_dict(payload)


def test_catalog_loader_rejects_stale_nested_record_digest() -> None:
    payload = _catalog().to_dict()
    payload["sources"][0]["helicity"] = -1  # type: ignore[index]
    with pytest.raises(RecurrenceTemplateError, match="stale semantic digest"):
        RecurrenceTemplateCatalog.from_dict(payload)


def test_contract_is_model_generic() -> None:
    catalog = _catalog()
    serialized = catalog.canonical_json
    assert "built-in-sm" not in serialized
    assert "ufo" not in serialized.lower()
    assert "example:species:matter" in serialized
