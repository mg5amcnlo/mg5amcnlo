# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol.generation.dag_algorithms import (
    infer_minimal_coupling_order_limits,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_equivalence import (
    _derive_current_value_equivalences,
    assign_recursive_current_evaluation_reuse,
)
from pyamplicol.generation.dag_types import CurrentNode
from pyamplicol.models import BuiltinSMModel, CompiledUFOModel, compile_model_source
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.compiler_contacts import _four_point_contact_color_split
from pyamplicol.models.compiler_entry import (
    _annotate_oriented_kernel_evaluation_equivalence,
    _term_supports_transverse_massless_yang_mills,
)
from pyamplicol.models.compiler_tensor_ordering import (
    compile_tensor_ordering_metadata,
)
from pyamplicol.models.contracts import (
    CompiledModelIR,
    CompiledOrientedKernel,
    CompiledParticleRecord,
)
from pyamplicol.models.external_symmetries import (
    _tensor_product_signature,
    derive_external_symmetry_certificates,
)
from pyamplicol.processes.model import build_model_process_ir

MODEL_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)

_EXTERNAL_SM_TOPOLOGY_LADDER = (
    "d d~ > z",
    "u d~ > w+",
    "d d~ > z g",
    "d d~ > e- e+",
    "u d~ > e+ ve",
    "d d~ > z z",
    "d d~ > u u~",
    "d d~ > d d~",
    "g g > g g",
    "g g > t t~",
    "d d~ > t t~",
    "d d~ > z g g",
    "g g > t t~ g",
    "g g > g g g",
    "d d~ > t t~ g g",
    "g g > t t~ g g g",
    "d d~ > t t~ g g g",
    "d d~ > z z z",
    "d d~ > e+ e- z h",
    "d d~ > t t~ z h",
    "d d~ > e+ e- e+ e-",
    "d d~ > u u~ s s~",
)


def _replace_model_with_recompiled_tensor_metadata(
    model: CompiledModelIR,
    *,
    vertex_terms,
    oriented_kernels,
) -> CompiledModelIR:
    terms, kernels, orderings, current_orderings = compile_tensor_ordering_metadata(
        vertex_terms,
        model.particles,
        oriented_kernels,
        model.parameters,
        model.propagators,
    )
    return replace(
        model,
        vertex_terms=terms,
        oriented_kernels=kernels,
        tensor_orderings=orderings,
        current_orderings=current_orderings,
    )


@pytest.fixture(scope="module")
def external_sm():
    compiled = compile_model_source(
        MODEL_ROOT / "sm.json",
        restriction=str((MODEL_ROOT / "restrict_default.json").resolve()),
        use_cache=False,
    )
    return compiled, CompiledUFOModel(compiled)


@pytest.fixture(scope="module")
def relabelled_external_sm(tmp_path_factory: pytest.TempPathFactory):
    raw = json.loads((MODEL_ROOT / "sm.json").read_text(encoding="utf-8"))
    relabelled = deepcopy(raw)
    absolute_ids = sorted(
        {abs(int(particle["pdg_code"])) for particle in relabelled["particles"]}
    )
    replacements = {
        particle_id: 700_000 + index
        for index, particle_id in enumerate(absolute_ids, start=1)
    }
    for particle in relabelled["particles"]:
        pdg = int(particle["pdg_code"])
        particle["pdg_code"] = (
            replacements[abs(pdg)] if pdg >= 0 else -replacements[abs(pdg)]
        )

    model_root = tmp_path_factory.mktemp("relabelled-external-sm")
    model_path = model_root / "sm.json"
    restriction_path = model_root / "restrict_default.json"
    model_path.write_text(
        json.dumps(relabelled, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    restriction_path.write_bytes((MODEL_ROOT / "restrict_default.json").read_bytes())
    compiled = compile_model_source(
        model_path,
        restriction=str(restriction_path.resolve()),
        use_cache=False,
    )
    return compiled, CompiledUFOModel(compiled)


@pytest.fixture(scope="module")
def reordered_external_sm(tmp_path_factory: pytest.TempPathFactory):
    raw = json.loads((MODEL_ROOT / "sm.json").read_text(encoding="utf-8"))
    reordered = deepcopy(raw)
    for field in (
        "orders",
        "particles",
        "couplings",
        "lorentz_structures",
        "propagators",
        "functions",
        "form_factors",
        "vertex_rules",
    ):
        reordered[field] = list(reversed(reordered[field]))
    for vertex in reordered["vertex_rules"]:
        colors = vertex["color_structures"]
        lorentz = vertex["lorentz_structures"]
        coupling_matrix = vertex["couplings"]
        color_order = tuple(reversed(range(len(colors))))
        lorentz_order = tuple(reversed(range(len(lorentz))))
        vertex["color_structures"] = [colors[index] for index in color_order]
        vertex["lorentz_structures"] = [lorentz[index] for index in lorentz_order]
        vertex["couplings"] = [
            [coupling_matrix[row][column] for column in lorentz_order]
            for row in color_order
        ]

    model_root = tmp_path_factory.mktemp("reordered-external-sm")
    model_path = model_root / "sm.json"
    restriction_path = model_root / "restrict_default.json"
    model_path.write_text(
        json.dumps(reordered, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    restriction_path.write_bytes((MODEL_ROOT / "restrict_default.json").read_bytes())
    compiled = compile_model_source(
        model_path,
        restriction=str(restriction_path.resolve()),
        use_cache=False,
    )
    return compiled, CompiledUFOModel(compiled)


@pytest.fixture(scope="module")
def renamed_relabelled_reordered_external_sm(
    tmp_path_factory: pytest.TempPathFactory,
):
    raw = json.loads((MODEL_ROOT / "sm.json").read_text(encoding="utf-8"))
    transformed = deepcopy(raw)
    original_names = sorted(str(particle["name"]) for particle in raw["particles"])
    name_map = {
        name: f"state_{index:03d}"
        for index, name in enumerate(original_names, start=1)
    }
    absolute_ids = sorted(
        {abs(int(particle["pdg_code"])) for particle in transformed["particles"]}
    )
    pdg_map = {
        particle_id: 800_000 + index
        for index, particle_id in enumerate(absolute_ids, start=1)
    }
    for particle in transformed["particles"]:
        particle["name"] = name_map[str(particle["name"])]
        particle["antiname"] = name_map[str(particle["antiname"])]
        pdg = int(particle["pdg_code"])
        particle["pdg_code"] = pdg_map[abs(pdg)] if pdg >= 0 else -pdg_map[abs(pdg)]
    for propagator in transformed["propagators"]:
        propagator["particle"] = name_map[str(propagator["particle"])]
    for vertex in transformed["vertex_rules"]:
        vertex["particles"] = [name_map[str(name)] for name in vertex["particles"]]
        colors = vertex["color_structures"]
        lorentz = vertex["lorentz_structures"]
        coupling_matrix = vertex["couplings"]
        color_order = tuple(reversed(range(len(colors))))
        lorentz_order = tuple(reversed(range(len(lorentz))))
        vertex["color_structures"] = [colors[index] for index in color_order]
        vertex["lorentz_structures"] = [lorentz[index] for index in lorentz_order]
        vertex["couplings"] = [
            [coupling_matrix[row][column] for column in lorentz_order]
            for row in color_order
        ]
    for field in (
        "orders",
        "particles",
        "couplings",
        "lorentz_structures",
        "propagators",
        "functions",
        "form_factors",
        "vertex_rules",
    ):
        transformed[field] = list(reversed(transformed[field]))

    model_root = tmp_path_factory.mktemp("renamed-relabelled-reordered-external-sm")
    model_path = model_root / "sm.json"
    restriction_path = model_root / "restrict_default.json"
    model_path.write_text(
        json.dumps(transformed, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    restriction_path.write_bytes((MODEL_ROOT / "restrict_default.json").read_bytes())
    compiled = compile_model_source(
        model_path,
        restriction=str(restriction_path.resolve()),
        use_cache=False,
    )
    return compiled, CompiledUFOModel(compiled), name_map


@pytest.fixture(scope="module")
def color_dummy_relabelled_external_sm(tmp_path_factory: pytest.TempPathFactory):
    raw = json.loads((MODEL_ROOT / "sm.json").read_text(encoding="utf-8"))
    relabelled = deepcopy(raw)
    vertex = next(
        item for item in relabelled["vertex_rules"] if item["name"] == "V_37"
    )
    vertex["color_structures"] = [
        "*".join(reversed(source.replace("-1", "-97").split("*")))
        for source in vertex["color_structures"]
    ]

    model_root = tmp_path_factory.mktemp("color-dummy-relabelled-external-sm")
    model_path = model_root / "sm.json"
    restriction_path = model_root / "restrict_default.json"
    model_path.write_text(
        json.dumps(relabelled, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    restriction_path.write_bytes((MODEL_ROOT / "restrict_default.json").read_bytes())
    compiled = compile_model_source(
        model_path,
        restriction=str(restriction_path.resolve()),
        use_cache=False,
    )
    return compiled, CompiledUFOModel(compiled)


def _production_dag_signature(compiled, model, process: str) -> tuple[object, ...]:
    process_ir = build_model_process_ir(process, compiled.ir)
    limits = infer_minimal_coupling_order_limits(process_ir, model=model)
    dag = compile_generic_dag(
        process_ir,
        model=model,
        max_coupling_orders=limits,
    )
    return (
        limits,
        len(dag.currents),
        len(dag.interactions),
        len(dag.amplitude_roots),
        len(dag.color_plan.sectors),
        dag.interaction_evaluation_count,
    )


def _rename_process(process: str, name_map: dict[str, str]) -> str:
    folded_map = {
        name.casefold(): replacement for name, replacement in name_map.items()
    }
    return " ".join(
        name_map.get(token, folded_map.get(token.casefold(), token))
        for token in process.split()
    )


def test_external_sm_symmetries_are_proven_from_compiled_tensors(external_sm) -> None:
    compiled, model = external_sm
    certificates = model._symmetry_certificates

    assert certificates.yang_mills_adjoint_names == frozenset({"g"})
    assert len(certificates.yang_mills_kernel_kinds) == 4
    assert certificates.yang_mills_kernel_kinds <= certificates.parity_kernel_kinds
    reflection_phases = dict(certificates.adjoint_current_reflection_phases)
    assert reflection_phases
    assert set(reflection_phases.values()) == {(-1.0, 0.0)}
    parity_digests = dict(certificates.parity_kernel_digests)
    yang_mills_digests = dict(certificates.yang_mills_kernel_digests)
    adjoint_digests = dict(certificates.yang_mills_adjoint_digests)
    reflection_digests = dict(
        certificates.adjoint_current_reflection_digests
    )
    assert set(parity_digests) == certificates.parity_kernel_kinds
    assert set(yang_mills_digests) == certificates.yang_mills_kernel_kinds
    assert set(adjoint_digests) == certificates.yang_mills_adjoint_names
    assert set(reflection_digests) == set(reflection_phases)
    assert all(
        len(digest) == 64
        for digest in (
            *parity_digests.values(),
            *yang_mills_digests.values(),
            *adjoint_digests.values(),
            *reflection_digests.values(),
        )
    )
    assert all(
        reflection_digests[kind] == parity_digests[kind]
        for kind in reflection_digests
    )

    pure_adjoint = build_model_process_ir("g g > g g", compiled.ir)
    qcd_vertices = tuple(
        vertex
        for vertex in model.vertices
        if vertex.kind in certificates.parity_kernel_kinds
    )
    yang_mills_vertices = tuple(
        vertex
        for vertex in model.vertices
        if vertex.kind in certificates.yang_mills_kernel_kinds
    )

    assert model.global_helicity_flip_equivalence_is_proven(qcd_vertices)
    assert model.pure_massless_adjoint_helicity_zero_rule_is_proven(
        pure_adjoint,
        yang_mills_vertices,
    )
    assert model.lc_trace_reflection_equivalence_is_proven(pure_adjoint)

    cubic_kernel = next(
        kernel
        for kernel in compiled.ir.oriented_kernels
        if kernel.particles == ("g", "g", "g")
    )
    assert cubic_kernel.evaluation_input_exchange_factor == (-1.0, 0.0)
    auxiliary_input = next(
        kernel
        for kernel in compiled.ir.oriented_kernels
        if kernel.particles[:2] == ("g", "g")
        and kernel.kind in certificates.yang_mills_kernel_kinds
        and kernel.kind != cubic_kernel.kind
    )
    assert auxiliary_input.evaluation_input_exchange_factor == (-1.0, 0.0)
    auxiliary_outputs = tuple(
        kernel
        for kernel in compiled.ir.oriented_kernels
        if kernel.kind in certificates.yang_mills_kernel_kinds
        and kernel.kind not in {cubic_kernel.kind, auxiliary_input.kind}
    )
    assert len(auxiliary_outputs) == 2
    assert len({kernel.evaluation_class for kernel in auxiliary_outputs}) == 1
    assert {kernel.evaluation_input_order for kernel in auxiliary_outputs} == {
        (0, 1),
        (1, 0),
    }
    assert {kernel.evaluation_factor for kernel in auxiliary_outputs} == {
        (1.0, 0.0),
        (-1.0, 0.0),
    }


def test_external_kernel_reuse_includes_exact_colour_normalization(external_sm) -> None:
    from pyamplicol._internal.physics.symbols import symbols

    compiled, _model = external_sm
    cubic = next(
        kernel
        for kernel in compiled.ir.oriented_kernels
        if kernel.particles == ("g", "g", "g")
    )
    different_normalization = replace(
        cubic, lc_color_normalization_power=cubic.lc_color_normalization_power + 1
    )
    annotated = _annotate_oriented_kernel_evaluation_equivalence(
        (cubic, different_normalization),
        compiled.ir.particles,
        compiled.ir.vertex_terms,
        symbols.model(compiled.ir.name),
    )

    assert annotated[0].evaluation_class != annotated[1].evaluation_class
    assert all(kernel.evaluation_equivalence_verified for kernel in annotated)


def test_kernel_equivalence_ignores_mirrored_component_factorization() -> None:
    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.models import compiler_symbolica as sym

    sym._ensure_symbolica()
    registry = symbols.model("factorized-kernel-proof")
    particles = tuple(
        CompiledParticleRecord(
            name=name,
            antiname=name,
            pdg_code=index + 1,
            spin=3,
            color=1,
            mass="ZERO",
            width="ZERO",
            charge=0.0,
            quantum_numbers=(("electric_charge", "0"),),
            ghost_number=0,
            propagating=True,
            goldstoneboson=False,
            propagator=None,
        )
        for index, name in enumerate(("a", "b", "out"))
    )
    g, h, imaginary = (sym.E(value) for value in ("g", "h", "\U0001d456"))
    kernels = []
    for kind, mirrored in enumerate((False, True)):
        vector_side, spinor_side = ("right", "left") if mirrored else ("left", "right")
        a, b, d, e = (
            registry.kernel_component(kind, vector_side, index) for index in range(4)
        )
        x, y = (
            registry.kernel_component(kind, spinor_side, index) for index in range(2)
        )
        common = (a - imaginary * b) * x + (d + e) * y
        # Released Symbolica 3 can retain these different factorizations for
        # mirrored Z-antifermion kernels.  Both must certify the same operation.
        expression = (
            common * g + (a * x - imaginary * b * x + d * y + e * y) * h
            if mirrored
            else common * (g + h)
        )
        kernels.append(
            CompiledOrientedKernel(
                kind=kind,
                term_id=0,
                vertex="factorized-mirror",
                particles=("b", "a", "out") if mirrored else ("a", "b", "out"),
                source_particle_legs=(1, 0, 2) if mirrored else (0, 1, 2),
                component_expressions=(expression.to_canonical_string(), "0", "0", "0"),
                coupling_expression="1",
                coupling_orders=(),
                runtime_parameters=(),
                color_source="1",
                color_expression="1",
            )
        )

    direct, mirrored = _annotate_oriented_kernel_evaluation_equivalence(
        kernels, particles, (), registry
    )

    assert direct.evaluation_class == mirrored.evaluation_class
    assert direct.evaluation_input_order == mirrored.evaluation_input_order[::-1]
    assert direct.evaluation_factor == mirrored.evaluation_factor == (1.0, 0.0)
    assert direct.component_expressions == kernels[0].component_expressions
    assert mirrored.component_expressions == kernels[1].component_expressions


def test_tensor_product_signature_alpha_normalizes_contracted_indices() -> None:
    left = (
        "spenso::f(ufo_c_dummy_1_adjoint,ufo_c_1,ufo_c_2)"
        "*spenso::f(ufo_c_3,ufo_c_4,ufo_c_dummy_1_adjoint)"
    )
    right = (
        "spenso::f(ufo_c_3,ufo_c_4,ufo_c_dummy_97_adjoint)"
        "*spenso::f(ufo_c_dummy_97_adjoint,ufo_c_1,ufo_c_2)"
    )
    assert _tensor_product_signature(left, "1") == _tensor_product_signature(
        right,
        "1",
    )


def test_tensor_product_signature_distinguishes_contraction_graphs() -> None:
    chain = (
        "spenso::f(ufo_c_1,ufo_c_dummy_2_adjoint,ufo_c_dummy_7_adjoint)"
        "*spenso::f(ufo_c_dummy_2_adjoint,ufo_c_2,ufo_c_3)"
        "*spenso::f(ufo_c_dummy_7_adjoint,ufo_c_4,ufo_c_5)"
    )
    relabelled_chain = (
        "spenso::f(ufo_c_dummy_41_adjoint,ufo_c_4,ufo_c_5)"
        "*spenso::f(ufo_c_1,ufo_c_dummy_99_adjoint,ufo_c_dummy_41_adjoint)"
        "*spenso::f(ufo_c_dummy_99_adjoint,ufo_c_2,ufo_c_3)"
    )
    different = (
        "spenso::f(ufo_c_1,ufo_c_dummy_2_adjoint,ufo_c_dummy_7_adjoint)"
        "*spenso::f(ufo_c_dummy_2_adjoint,ufo_c_dummy_7_adjoint,ufo_c_3)"
        "*spenso::f(ufo_c_2,ufo_c_4,ufo_c_5)"
    )
    signature = _tensor_product_signature(chain, "1")
    assert signature == _tensor_product_signature(relabelled_chain, "1")
    assert signature != _tensor_product_signature(different, "1")


def test_external_sm_color_dummy_relabeling_preserves_yang_mills_reuse(
    external_sm,
    color_dummy_relabelled_external_sm,
) -> None:
    compiled, model = external_sm
    relabelled, relabelled_model = color_dummy_relabelled_external_sm

    assert relabelled_model._symmetry_certificates == model._symmetry_certificates
    assert _production_dag_signature(
        relabelled,
        relabelled_model,
        "g g > g g",
    ) == _production_dag_signature(compiled, model, "g g > g g")


def test_external_sm_four_gluon_contacts_keep_proven_lowering(external_sm) -> None:
    compiled, _model = external_sm
    contact_terms = tuple(
        term
        for term in compiled.ir.vertex_terms
        if term.particles == ("g", "g", "g", "g")
    )

    assert len(contact_terms) == 3
    assert all(
        _four_point_contact_color_split(term, result_leg) is not None
        for term in contact_terms
        for result_leg in range(4)
    )
    contact_term_ids = {term.id for term in contact_terms}
    assert any(
        contact_term_ids <= set(kernel.term_ids)
        for kernel in compiled.ir.oriented_kernels
    )


def test_external_sm_derives_fundamental_fierz_auxiliary(external_sm) -> None:
    compiled, model = external_sm
    auxiliaries = tuple(
        particle
        for particle in compiled.ir.particles
        if particle.auxiliary_kind == "u1-subtraction-color-flow-vector"
    )

    assert len(auxiliaries) == 1
    auxiliary = auxiliaries[0]
    assert auxiliary.name == "__pyamplicol_u1_subtraction_g"
    assert (auxiliary.spin, auxiliary.color, auxiliary.component_dimension) == (
        3,
        1,
        4,
    )
    synthetic = tuple(
        kernel
        for kernel in compiled.ir.oriented_kernels
        if kernel.vertex.endswith("::u1-subtraction")
    )
    assert len(synthetic) == 30
    assert all(
        kernel.color_projection_structure == "color-identity"
        and auxiliary.name in kernel.particles
        for kernel in synthetic
    )
    assert {
        kernel.color_projection_coefficient
        for kernel in synthetic
        if kernel.particles[2] == auxiliary.name
    } == {(1.0 / 3.0, 0.0)}
    assert {
        kernel.color_projection_coefficient
        for kernel in synthetic
        if kernel.particles[2] != auxiliary.name
    } == {(1.0, 0.0)}

    propagator = model._propagator_ir(auxiliary.pdg_code)
    assert propagator.kind == "vector"
    assert propagator.mass_class == "massless"
    assert propagator.gauge == "feynman"
    assert propagator.applies_propagator is True


@pytest.mark.parametrize(
    ("process", "topology", "forbidden_order"),
    (
        ("g g > t t~", (36, 44, 36, 32), (2, 1)),
        ("d d~ > z g g", (117, 242, 210, 48), (5, 4)),
    ),
)
def test_external_sm_recovers_builtin_lc_current_reuse(
    external_sm,
    process: str,
    topology: tuple[int, int, int, int],
    forbidden_order: tuple[int, int],
) -> None:
    compiled, external_model = external_sm
    builtin_dag = compile_generic_dag(
        build_process_ir(process),
        model=BuiltinSMModel(),
    )
    external_dag = compile_generic_dag(
        build_model_process_ir(process, compiled.ir),
        model=external_model,
    )

    for dag in (builtin_dag, external_dag):
        assert (
            len(dag.currents),
            len(dag.interactions),
            dag.interaction_evaluation_count,
            len(dag.amplitude_roots),
        ) == topology
        assert all(
            current.index.ordered_external_labels != forbidden_order
            for current in dag.currents
        )


@pytest.mark.parametrize("process", _EXTERNAL_SM_TOPOLOGY_LADDER)
def test_external_sm_matches_builtin_production_dag_topology(
    external_sm,
    process: str,
) -> None:
    """External SM must recover every built-in production DAG reduction."""

    compiled, external_model = external_sm
    builtin_model = BuiltinSMModel()
    builtin_process = build_process_ir(process)
    external_process = build_model_process_ir(process, compiled.ir)
    builtin_limits = infer_minimal_coupling_order_limits(
        builtin_process,
        model=builtin_model,
    )
    external_limits = infer_minimal_coupling_order_limits(
        external_process,
        model=external_model,
    )

    assert external_limits == builtin_limits

    builtin_dag = compile_generic_dag(
        builtin_process,
        model=builtin_model,
        max_coupling_orders=builtin_limits,
    )
    external_dag = compile_generic_dag(
        external_process,
        model=external_model,
        max_coupling_orders=external_limits,
    )

    def topology(dag) -> tuple[int, int, int, int]:
        return (
            len(dag.currents),
            len(dag.interactions),
            len(dag.amplitude_roots),
            len(dag.color_plan.sectors),
        )

    assert topology(external_dag) == topology(builtin_dag)
    # The external compiler may prove additional exact kernel equivalences,
    # but it may never lose a reuse relation available to the built-in model.
    assert (
        external_dag.interaction_evaluation_count
        <= builtin_dag.interaction_evaluation_count
    )


def test_route_safe_mixed_line_dag_stays_below_legacy_kernel_count(
    external_sm,
) -> None:
    """Contracted-color SM DAGs remain smaller than the legacy kernel set."""

    compiled, external_model = external_sm
    builtin_model = BuiltinSMModel()
    builtin_process = build_process_ir(
        "d d~ > t t~ g g g",
        color_accuracy="full",
    )
    external_process = build_model_process_ir(
        "d d~ > t t~ g g g",
        compiled.ir,
        color_accuracy="full",
    )
    coupling_limits = {"QCD": 5, "QED": 0}
    builtin_dag = compile_generic_dag(
        builtin_process,
        model=builtin_model,
        max_coupling_orders=coupling_limits,
    )
    external_dag = compile_generic_dag(
        external_process,
        model=external_model,
        max_coupling_orders=coupling_limits,
    )

    assert (
        len(builtin_dag.currents),
        len(builtin_dag.interactions),
        len(builtin_dag.amplitude_roots),
    ) == (14_088, 36_024, 3_072)
    assert (
        len(external_dag.currents),
        len(external_dag.interactions),
        len(external_dag.amplitude_roots),
    ) == (14_088, 36_024, 3_072)

    # Original AmpliCol evaluates 652 post-filter kernels for each of the 64
    # nonzero helicities. Ordered labels and helicity ancestry are physical
    # runtime routes and therefore may not be merged merely by a reflection
    # proof. Exact structural pruning still keeps both model implementations
    # below the legacy kernel count.
    legacy_kernel_evaluations = 652 * 64
    assert builtin_dag.interaction_evaluation_count == 36_024
    assert external_dag.interaction_evaluation_count == 36_024
    assert builtin_dag.interaction_evaluation_count < legacy_kernel_evaluations
    assert external_dag.interaction_evaluation_count < legacy_kernel_evaluations
    assert assign_recursive_current_evaluation_reuse(
        builtin_dag,
        builtin_model,
    ) == builtin_dag
    assert assign_recursive_current_evaluation_reuse(
        external_dag,
        external_model,
    ) == external_dag


def test_recursive_reuse_keeps_reflected_runtime_routes_distinct(
    external_sm,
) -> None:
    compiled, external_model = external_sm
    cases = (
        (
            BuiltinSMModel(),
            build_process_ir("g g > g g g", color_accuracy="nlc"),
        ),
        (
            external_model,
            build_model_process_ir(
                "g g > g g g",
                compiled.ir,
                color_accuracy="nlc",
            ),
        ),
    )

    for model, process in cases:
        dag = compile_generic_dag(
            process,
            model=model,
            max_coupling_orders={"QCD": 3, "QED": 0},
        )
        equivalences = _derive_current_value_equivalences(dag, model)
        currents_by_contract: dict[tuple[object, ...], list[CurrentNode]] = defaultdict(
            list
        )
        for current in dag.currents:
            index = current.index
            if len(index.ordered_external_labels) < 2:
                continue
            contract = (
                int(index.particle_id),
                int(index.external_mask),
                index.external_labels,
                int(index.helicity_ancestry),
                int(index.chirality),
                index.spin_state,
                index.flavour_flow,
                index.quantum_number_flow,
                int(index.momentum_mask),
                index.coupling_orders,
                index.auxiliary_kind,
                int(current.dimension),
            )
            currents_by_contract[contract].append(current)

        checked = 0
        for currents in currents_by_contract.values():
            representatives_by_order: dict[tuple[int, ...], set[int]] = defaultdict(set)
            for current in currents:
                representatives_by_order[current.index.ordered_external_labels].add(
                    equivalences[current.id].representative_id
                )
            for ordering, representatives in representatives_by_order.items():
                reversed_representatives = representatives_by_order.get(
                    tuple(reversed(ordering))
                )
                if reversed_representatives is None:
                    continue
                checked += 1
                assert representatives.isdisjoint(reversed_representatives)
        assert checked > 0


def test_recursive_current_reuse_keeps_mixed_line_routes_strict() -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("d d~ > t t~ g g", color_accuracy="full"),
        model=model,
    )
    equivalences = _derive_current_value_equivalences(dag, model)

    # This production-shaped DAG has no exact same-route current aliases.
    # Coefficient-deformation rejection is covered independently by the
    # synthetic projective-vector tests.
    assert all(
        equivalences[current.id].representative_id == current.id
        for current in dag.currents
        if not current.is_source
    )


def test_external_sm_tensor_ordering_ids_are_source_inventory_invariant(
    external_sm,
    relabelled_external_sm,
    reordered_external_sm,
) -> None:
    compiled, _model = external_sm
    relabelled, _relabelled_model = relabelled_external_sm
    reordered, _reordered_model = reordered_external_sm

    assert relabelled.ir.tensor_orderings == compiled.ir.tensor_orderings
    assert reordered.ir.tensor_orderings == compiled.ir.tensor_orderings
    assert {
        (
            kernel.input_ordering_ids,
            kernel.output_ordering_id,
        )
        for kernel in relabelled.ir.oriented_kernels
    } == {
        (
            kernel.input_ordering_ids,
            kernel.output_ordering_id,
        )
        for kernel in compiled.ir.oriented_kernels
    }
    assert {
        (
            kernel.input_ordering_ids,
            kernel.output_ordering_id,
        )
        for kernel in reordered.ir.oriented_kernels
    } == {
        (
            kernel.input_ordering_ids,
            kernel.output_ordering_id,
        )
        for kernel in compiled.ir.oriented_kernels
    }


@pytest.mark.parametrize("process", _EXTERNAL_SM_TOPOLOGY_LADDER)
def test_external_sm_topology_is_pdg_relabel_invariant(
    external_sm,
    relabelled_external_sm,
    process: str,
) -> None:
    compiled, model = external_sm
    relabelled, relabelled_model = relabelled_external_sm

    assert _production_dag_signature(
        relabelled,
        relabelled_model,
        process,
    ) == _production_dag_signature(compiled, model, process)


@pytest.mark.parametrize("process", _EXTERNAL_SM_TOPOLOGY_LADDER)
def test_external_sm_topology_is_ufo_inventory_order_invariant(
    external_sm,
    reordered_external_sm,
    process: str,
) -> None:
    compiled, model = external_sm
    reordered, reordered_model = reordered_external_sm

    assert _production_dag_signature(
        reordered,
        reordered_model,
        process,
    ) == _production_dag_signature(compiled, model, process)


@pytest.mark.parametrize("process", _EXTERNAL_SM_TOPOLOGY_LADDER)
def test_external_sm_topology_survives_combined_identity_and_inventory_changes(
    external_sm,
    renamed_relabelled_reordered_external_sm,
    process: str,
) -> None:
    compiled, model = external_sm
    transformed, transformed_model, name_map = (
        renamed_relabelled_reordered_external_sm
    )

    assert _production_dag_signature(
        transformed,
        transformed_model,
        _rename_process(process, name_map),
    ) == _production_dag_signature(compiled, model, process)


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_lc_current_reflection_reuse_does_not_prune_contracted_color_modes(
    external_sm,
    accuracy: str,
) -> None:
    compiled, external_model = external_sm
    cases = (
        (
            build_process_ir("d d~ > z g", color_accuracy=accuracy),
            BuiltinSMModel(),
        ),
        (
            build_model_process_ir(
                "d d~ > z g",
                compiled.ir,
                color_accuracy=accuracy,
            ),
            external_model,
        ),
    )

    for process, model in cases:
        limits = infer_minimal_coupling_order_limits(process, model=model)
        dag = compile_generic_dag(
            process,
            model=model,
            max_coupling_orders=limits,
        )
        assert (
            len(dag.currents),
            len(dag.interactions),
            len(dag.amplitude_roots),
        ) == (31, 34, 12)


def test_deformed_adjoint_kernel_disables_local_current_reuse(external_sm) -> None:
    compiled, model = external_sm
    reflection_kinds = dict(
        model._symmetry_certificates.adjoint_current_reflection_phases
    )
    target = next(
        kernel
        for kernel in compiled.ir.oriented_kernels
        if kernel.kind in reflection_kinds
    )
    deformed_kernels = tuple(
        replace(
            kernel,
            component_expressions=("1", *kernel.component_expressions[1:]),
        )
        if kernel.kind == target.kind
        else kernel
        for kernel in compiled.ir.oriented_kernels
    )
    certificates = derive_external_symmetry_certificates(
        replace(compiled.ir, oriented_kernels=deformed_kernels)
    )

    assert certificates.adjoint_current_reflection_phases == ()


def test_custom_adjoint_propagator_disables_global_symmetries(external_sm) -> None:
    compiled, model = external_sm
    original_yang_mills_kinds = model._symmetry_certificates.yang_mills_kernel_kinds
    propagators = tuple(
        replace(propagator, custom=True)
        if propagator.particle == "g"
        else propagator
        for propagator in compiled.ir.propagators
    )

    certificates = derive_external_symmetry_certificates(
        replace(compiled.ir, propagators=propagators)
    )

    assert original_yang_mills_kinds
    assert not (original_yang_mills_kinds & certificates.parity_kernel_kinds)
    assert certificates.yang_mills_kernel_kinds == frozenset()
    assert certificates.yang_mills_adjoint_names == frozenset()
    assert certificates.adjoint_current_reflection_phases == ()
    assert certificates.yang_mills_kernel_digests == ()
    assert certificates.yang_mills_adjoint_digests == ()
    assert certificates.adjoint_current_reflection_digests == ()


def test_transverse_yang_mills_lowering_requires_standard_adjoint_propagator(
    external_sm,
) -> None:
    compiled, _model = external_sm
    term = next(
        term
        for term in compiled.ir.vertex_terms
        if term.valence == 3 and set(term.particles) == {"g"}
    )
    particles = {particle.name: particle for particle in compiled.ir.particles}
    parameters = {parameter.name: parameter for parameter in compiled.ir.parameters}
    propagators = {
        propagator.name: propagator for propagator in compiled.ir.propagators
    }

    assert _term_supports_transverse_massless_yang_mills(
        term,
        particles,
        parameters,
        propagators,
    )

    custom = {
        name: replace(propagator, custom=True)
        if propagator.particle == "g"
        else propagator
        for name, propagator in propagators.items()
    }
    assert not _term_supports_transverse_massless_yang_mills(
        term,
        particles,
        parameters,
        custom,
    )


def test_chiral_gauge_current_does_not_receive_parity_certificate(external_sm) -> None:
    compiled, _model = external_sm
    vectorlike = next(term for term in compiled.ir.vertex_terms if term.id == 75)
    chiral = next(
        term
        for term in compiled.ir.vertex_terms
        if "projm" in term.lorentz_expression.casefold()
    )
    deformed_terms = tuple(
        replace(vectorlike, lorentz_expression=chiral.lorentz_expression)
        if term.id == vectorlike.id
        else term
        for term in compiled.ir.vertex_terms
    )
    deformed_terms = tuple(
        replace(term, index_bindings=()) if term.id == vectorlike.id else term
        for term in deformed_terms
    )
    certificates = derive_external_symmetry_certificates(
        _replace_model_with_recompiled_tensor_metadata(
            compiled.ir,
            vertex_terms=deformed_terms,
            oriented_kernels=compiled.ir.oriented_kernels,
        )
    )
    affected_kinds = {
        kernel.kind
        for kernel in compiled.ir.oriented_kernels
        if vectorlike.id in kernel.term_ids
    }

    assert affected_kinds
    assert not (affected_kinds & certificates.parity_kernel_kinds)


def test_deformed_quartic_coupling_disables_yang_mills_theorems(external_sm) -> None:
    compiled, _model = external_sm
    quartic_terms = {
        term.coupling for term in compiled.ir.vertex_terms if term.id in {36, 37, 38}
    }
    assert len(quartic_terms) == 1
    quartic_name = next(iter(quartic_terms))
    deformed_couplings = tuple(
        replace(coupling, expression=f"2*({coupling.expression})")
        if coupling.name == quartic_name
        else coupling
        for coupling in compiled.ir.couplings
    )
    certificates = derive_external_symmetry_certificates(
        replace(compiled.ir, couplings=deformed_couplings)
    )

    assert certificates.yang_mills_adjoint_names == frozenset()
    assert certificates.yang_mills_kernel_kinds == frozenset()


@pytest.mark.parametrize("field", ("color_expression", "lorentz_expression"))
def test_rescaled_cubic_tensor_disables_yang_mills_theorems(
    external_sm,
    field: str,
) -> None:
    compiled, _model = external_sm
    cubic = next(
        term for term in compiled.ir.vertex_terms if term.particles == ("g", "g", "g")
    )
    deformed_terms = tuple(
        replace(
            term,
            **{field: f"2*({getattr(term, field)})"},
        )
        if term.id == cubic.id
        else term
        for term in compiled.ir.vertex_terms
    )

    certificates = derive_external_symmetry_certificates(
        replace(compiled.ir, vertex_terms=deformed_terms)
    )

    assert certificates.yang_mills_adjoint_names == frozenset()
    assert certificates.yang_mills_kernel_kinds == frozenset()


def test_duplicate_cubic_term_disables_yang_mills_theorems(external_sm) -> None:
    compiled, _model = external_sm
    cubic = next(
        term for term in compiled.ir.vertex_terms if term.particles == ("g", "g", "g")
    )
    duplicate = replace(
        cubic,
        id=max(term.id for term in compiled.ir.vertex_terms) + 1,
        vertex=f"{cubic.vertex}_duplicate",
    )

    certificates = derive_external_symmetry_certificates(
        replace(compiled.ir, vertex_terms=(*compiled.ir.vertex_terms, duplicate))
    )

    assert certificates.yang_mills_adjoint_names == frozenset()
    assert certificates.yang_mills_kernel_kinds == frozenset()


def test_reachable_non_yang_mills_kernel_disables_global_adjoint_theorems(
    external_sm,
) -> None:
    compiled, _model = external_sm
    original_certificates = derive_external_symmetry_certificates(compiled.ir)
    representative = next(
        kernel
        for kernel in compiled.ir.oriented_kernels
        if kernel.kind in original_certificates.yang_mills_kernel_kinds
        and kernel.particles == ("g", "g", "g")
    )
    source_term = next(
        term
        for term in compiled.ir.vertex_terms
        if term.id in (representative.term_ids or (representative.term_id,))
    )
    extra_term_id = max(term.id for term in compiled.ir.vertex_terms) + 1
    extra_kind = max(kernel.kind for kernel in compiled.ir.oriented_kernels) + 1
    extra_term = replace(
        source_term,
        id=extra_term_id,
        vertex="V_non_yang_mills_adjoint",
        lorentz_name="L_non_yang_mills_adjoint",
        lorentz_source="1",
        lorentz_expression="1",
        index_bindings=(),
    )
    extra_kernel = replace(
        representative,
        kind=extra_kind,
        term_id=extra_term_id,
        term_ids=(extra_term_id,),
        vertex=extra_term.vertex,
        evaluation_class="",
        evaluation_equivalence_verified=False,
        evaluation_input_exchange_factor=None,
    )

    certificates = derive_external_symmetry_certificates(
        _replace_model_with_recompiled_tensor_metadata(
            compiled.ir,
            vertex_terms=(*compiled.ir.vertex_terms, extra_term),
            oriented_kernels=(*compiled.ir.oriented_kernels, extra_kernel),
        )
    )

    assert certificates.yang_mills_kernel_kinds
    assert certificates.yang_mills_adjoint_names == frozenset()
