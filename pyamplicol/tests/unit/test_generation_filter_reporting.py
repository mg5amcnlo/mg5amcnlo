# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import replace

import pytest

import pyamplicol.generation.service as service_module
from pyamplicol.api import ProcessRequest
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    RunConfig,
)
from pyamplicol.generation.helicity_replay import (
    HELICITY_RECURRENCE_CONTRACT_VERSION,
)
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def test_generation_reports_structural_reduction_and_helicity_recurrence() -> None:
    model = BuiltinSMModel()
    backend = service_module.GenerationBackend(GenerationConfig(), None)
    process_ir = build_process_ir("d d~ > z", color_accuracy="lc")
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    expanded = service_module._ExpandedProcess(
        request=ProcessRequest.parse("d d~ > z", name="ddbar_z"),
        process_ir=process_ir,
        aliases=(
            {
                "id": "ddbar_z_alias",
                "expression": "d d~ > z",
                "external_pdgs": [1, -1, 23],
                "external_permutation": [0, 1, 2],
            },
        ),
    )

    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(expanded, dag, coverage),
        model,
        index=0,
        phase=PhaseHandle("test", None, 1),
    )

    assert set(prepared.filters) == {
        "dynamic_color_projection",
        "structural_helicity_reduction",
        "helicity_recurrence",
        "lc_flow_layout",
        "relation_discovery",
    }
    dynamic_projection = prepared.filters["dynamic_color_projection"]
    assert isinstance(dynamic_projection, dict)
    assert dynamic_projection["equality_check_status"].startswith("passed-")
    structural = prepared.filters["structural_helicity_reduction"]
    assert isinstance(structural, dict)
    assert structural["mode"] == "proven global-helicity-flip equivalence"
    recurrence = prepared.filters["helicity_recurrence"]
    assert isinstance(recurrence, dict)
    assert recurrence["contract_version"] == HELICITY_RECURRENCE_CONTRACT_VERSION
    assert recurrence["residual_current_count"] == 0
    assert len(prepared.validation_points) == 1
    assert [point.seed for point in prepared.validation_points] == [12345]

    metadata_only_backend = service_module.GenerationBackend(
        GenerationConfig(
            validation=GenerationValidationConfig(enabled=False, samples=25)
        ),
        None,
    )
    metadata_only = metadata_only_backend._prepare_warmup_process(
        service_module._DagProcess(expanded, dag, coverage),
        model,
        index=0,
        phase=PhaseHandle("test-disabled", None, 1),
    )
    assert len(metadata_only.validation_points) == 1


def test_opt_in_relation_discovery_is_scoped_in_generation_filters() -> None:
    model = BuiltinSMModel()
    backend = service_module.GenerationBackend(
        GenerationConfig(
            relation_discovery=GenerationRelationDiscoveryConfig(
                mode="diagnostic",
                precision_digits=80,
                probe_count=2,
                verification_probe_count=2,
                seed=31,
            )
        ),
        None,
    )
    process_ir = build_process_ir("d d~ > z", color_accuracy="lc")
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    assert "relation_discovery" not in coverage

    expanded = service_module._ExpandedProcess(
        request=ProcessRequest.parse("d d~ > z", name="ddbar_z"),
        process_ir=process_ir,
        aliases=(),
    )
    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(expanded, dag, coverage),
        model,
        index=0,
        phase=PhaseHandle("test-discovery", None, 1),
    )
    discovery = prepared.filters["relation_discovery"]
    assert isinstance(discovery, dict)
    assert discovery["abi"] == "pyamplicol-artifact-numerical-current-reuse-v1"
    assert discovery["requested_mode"] == "diagnostic"
    assert discovery["effective_mode"] == "diagnostic"
    assert discovery["execution_mode"] == "compiled"
    assert discovery["applied_relation_count"] == 0
    assert discovery["relation_correctness"]["state"] == ("no-applied-relations")
    assert discovery["warning"]["required"] is False
    assert discovery == prepared.coverage["relation_discovery"]

    lanes = discovery["lanes"]
    assert isinstance(lanes, dict)
    assert discovery["lane_count"] == len(lanes)
    assert "primary" in lanes
    primary = lanes["primary"]
    assert primary["state"] == "no_certified_numerical_relation"
    assert primary["scope"] == {
        "execution_mode": "compiled",
        "color_accuracy": "lc",
        "representation": "generic-dag",
    }
    candidate = primary["candidate_capture"]
    verification = primary["verification_capture"]
    assert candidate["precision_digits"] == 80
    assert candidate["point_count"] == 2
    assert verification["point_count"] == 2
    assert set(candidate["kinematic_sha256s"]).isdisjoint(
        verification["kinematic_sha256s"]
    )


def test_numerical_current_reuse_opt_out_skips_every_warmup_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BuiltinSMModel()
    backend = service_module.GenerationBackend(
        GenerationConfig(
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off")
        ),
        None,
    )

    def forbidden_capture(*args, **kwargs):
        del args, kwargs
        raise AssertionError("opt-out must not run numerical current capture")

    monkeypatch.setattr(
        service_module,
        "run_generic_dag_numerical_current_warmup",
        forbidden_capture,
    )
    process_ir = build_process_ir("d d~ > z", color_accuracy="lc")
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    expanded = service_module._ExpandedProcess(
        request=ProcessRequest.parse("d d~ > z", name="ddbar_z"),
        process_ir=process_ir,
        aliases=(),
    )
    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(expanded, dag, coverage),
        model,
        index=0,
        phase=PhaseHandle("test-opt-out", None, 1),
    )

    discovery = prepared.filters["relation_discovery"]
    assert discovery["requested_mode"] == "off"
    assert discovery["effective_mode"] == "off"
    assert discovery["applied_relation_count"] == 0
    assert discovery["warning"]["required"] is False
    assert discovery["lanes"]
    for report in discovery["lanes"].values():
        assert report["state"] == "disabled-by-user"
        assert report["candidate_capture"] is None
        assert report["verification_capture"] is None


def test_full_color_opt_out_skips_inapplicable_projection_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BuiltinSMModel()
    backend = service_module.GenerationBackend(
        RunConfig(
            action="generate",
            color=ColorConfig(accuracy="full"),
            generation=GenerationConfig(
                relation_discovery=GenerationRelationDiscoveryConfig(mode="off")
            ),
            evaluator=EvaluatorConfig(execution_mode="compiled"),
        ),
        None,
    )
    process_ir = build_process_ir("d d~ > z", color_accuracy="full")
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    current = dag.currents[0]
    dag = replace(
        dag,
        currents=(
            replace(
                current,
                index=replace(current.index, helicity_ancestry=10**4300),
            ),
            *dag.currents[1:],
        ),
    )
    monkeypatch.setattr(
        service_module,
        "build_helicity_recurrence_plan",
        lambda *_args, **_kwargs: None,
    )
    expanded = service_module._ExpandedProcess(
        request=ProcessRequest.parse("d d~ > z", name="ddbar_z_full"),
        process_ir=process_ir,
        aliases=(),
    )

    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(expanded, dag, coverage),
        model,
        index=0,
        phase=PhaseHandle("test-full-opt-out", None, 1),
    )

    projection = prepared.filters["dynamic_color_projection"]
    assert projection["equality_check_status"] == "not-applicable-color-accuracy"
    assert projection["color_accuracy"] == "full"
    assert not any(key.endswith("_sha256") for key in projection)
    assert "old_to_new_current_ids" not in projection


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (None, False),
        ({}, False),
        ({"warning": {"required": False}}, False),
        ({"warning": {"required": True}}, True),
    ),
)
def test_numerical_current_warning_aggregation(value, expected) -> None:
    assert service_module._numerical_current_warning_required(value) is expected


def test_certified_numerical_current_reuse_applies_on_final_materialized_dag() -> None:
    model = BuiltinSMModel()
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full"),
        generation=GenerationConfig(
            relation_discovery=GenerationRelationDiscoveryConfig(
                mode="certified-reuse",
                precision_digits=80,
                probe_count=2,
                verification_probe_count=2,
            )
        ),
        evaluator=EvaluatorConfig(execution_mode="compiled"),
    )
    backend = service_module.GenerationBackend(config, None)
    process_ir = build_process_ir("g g > g g", color_accuracy="full")
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    expanded = service_module._ExpandedProcess(
        request=ProcessRequest.parse("g g > g g", name="gg_gg"),
        process_ir=process_ir,
        aliases=(),
    )

    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(expanded, dag, coverage),
        model,
        index=0,
        phase=PhaseHandle("test-applied", None, 1),
    )

    discovery = prepared.filters["relation_discovery"]
    assert discovery["requested_mode"] == "certified-reuse"
    assert discovery["certified_relation_count"] > 0
    assert discovery["applied_relation_count"] > 0
    assert discovery["warning"]["required"] is True
    assert discovery["relation_correctness"] == {
        "abi": "pyamplicol-numerical-current-relation-correctness-v1",
        "state": "member-scoped-v1",
        "applied_relation_count": discovery["applied_relation_count"],
    }
    assert all(
        "opposite" not in report["application_scope"]["allowed_relation_kinds"]
        for report in discovery["lanes"].values()
    )
    assert prepared.dag.helicity_materialization is not None
    assert any(
        report["application_validation"]["status"] == "verified"
        for report in discovery["lanes"].values()
    )


def test_n3_charged_current_all_flow_does_not_apply_generic_relations() -> None:
    model = BuiltinSMModel()
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="all-flow-union"),
        generation=GenerationConfig(
            relation_discovery=GenerationRelationDiscoveryConfig(
                mode="certified-reuse",
                precision_digits=80,
                probe_count=2,
                verification_probe_count=2,
            )
        ),
        evaluator=EvaluatorConfig(execution_mode="compiled"),
    )
    backend = service_module.GenerationBackend(config, None)
    process_ir = build_process_ir("u d~ > e+ ve g", color_accuracy="lc")
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    expanded = service_module._ExpandedProcess(
        request=ProcessRequest.parse("u d~ > e+ ve g", name="udbar_ep_ve_g"),
        process_ir=process_ir,
        aliases=(),
    )

    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(expanded, dag, coverage),
        model,
        index=0,
        phase=PhaseHandle("test-all-flow-contained", None, 1),
    )

    discovery = prepared.filters["relation_discovery"]
    assert prepared.dag.helicity_recurrence is not None
    assert prepared.dag.helicity_recurrence.physical_helicity_count > 1
    assert discovery["applied_relation_count"] == 0
    assert discovery["relation_correctness"]["state"] == ("no-applied-relations")
    for report in discovery["lanes"].values():
        scope = report["application_scope"]
        assert scope["reason"] == "multi-helicity-all-flow-selector-domain"
        assert scope["allowed_relation_kinds"] == []


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
@pytest.mark.parametrize("color_accuracy", ("lc", "nlc", "full"))
def test_relation_discovery_scope_covers_every_dag_mode_and_color(
    execution_mode,
    color_accuracy,
) -> None:
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=color_accuracy),
        generation=GenerationConfig(
            relation_discovery=GenerationRelationDiscoveryConfig(
                mode="diagnostic",
                precision_digits=80,
                probe_count=2,
                verification_probe_count=2,
            )
        ),
        evaluator=EvaluatorConfig(execution_mode=execution_mode),
    )
    backend = service_module.GenerationBackend(config, None)
    process_ir = build_process_ir(
        "d d~ > z",
        color_accuracy=color_accuracy,
    )
    model = BuiltinSMModel()
    dag, coverage = backend._compile_concrete_process(
        process_ir,
        model,
    )
    expanded = service_module._ExpandedProcess(
        request=ProcessRequest.parse("d d~ > z", name="ddbar_z"),
        process_ir=process_ir,
        aliases=(),
    )
    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(expanded, dag, coverage),
        model,
        index=0,
        phase=PhaseHandle("test-scope", None, 1),
    )

    aggregate = prepared.filters["relation_discovery"]
    assert aggregate["execution_mode"] == execution_mode
    discovery = aggregate["lanes"]["primary"]
    assert isinstance(discovery, dict)
    assert discovery["scope"] == {
        "execution_mode": execution_mode,
        "color_accuracy": color_accuracy,
        "representation": "generic-dag",
    }
