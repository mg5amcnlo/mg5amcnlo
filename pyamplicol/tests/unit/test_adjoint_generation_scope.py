# SPDX-License-Identifier: 0BSD
"""Non-native checks for the certified DDM generation scope and seed boundary."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

from pyamplicol.config import ColorConfig, EvaluatorConfig, ProcessConfig, RunConfig
from pyamplicol.generation import service as generation_service
from pyamplicol.generation.on_the_fly_seed import project_on_the_fly_process_seed_v1
from pyamplicol.generation.service import GenerationBackend
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.adjoint_color_certificate import (
    adjoint_tree_color_basis_is_proven,
)
from pyamplicol.models.base import Model
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.contracts import CompiledOrientedKernel
from pyamplicol.models.external_symmetries import _yang_mills_sector_is_closed
from tests.unit.test_on_the_fly_seed_projection import (
    _catalog,
    _gluon_process,
    _NormalizationModel,
)


class _UncertifiedAdjointModel(BuiltinSMModel):
    """An unknown extension must not inherit the pinned model's certificate."""


def test_pinned_builtin_certificate_does_not_leak_into_unknown_subclasses():
    process = build_process_ir("g g > g g", color_accuracy="full")
    model = BuiltinSMModel()
    unknown = _UncertifiedAdjointModel()
    assert model.lc_trace_reflection_equivalence_is_proven(process)
    assert unknown.lc_trace_reflection_equivalence_is_proven(process)
    assert adjoint_tree_color_basis_is_proven(model, process)
    assert not adjoint_tree_color_basis_is_proven(unknown, process)


def test_adjoint_certificate_dispatch_preserves_external_model_hook_and_limits():
    process = build_process_ir("g g > z g", color_accuracy="full")
    limits = {"HIG": 1}
    observed = []

    class CertifiedModel(_UncertifiedAdjointModel):
        def adjoint_tree_color_basis_is_proven(
            self, selected_process, *, max_coupling_orders=None
        ):
            observed.append((selected_process, max_coupling_orders))
            return True

    assert adjoint_tree_color_basis_is_proven(
        CertifiedModel(), process, max_coupling_orders=limits
    )
    assert observed == [(process, limits)]


def _backend(lane: str, basis: str = "adjoint", accuracy: str = "full"):
    return GenerationBackend(
        RunConfig(
            action="generate",
            process=ProcessConfig(
                coupling_order_policy="explicit",
                selected_source_helicities={"1": -1},
            ),
            color=ColorConfig(
                accuracy=accuracy,
                contraction="symmetric-group-fft",
                fft_basis=basis,
            ),
            evaluator=EvaluatorConfig(execution_mode=lane),
        ),
        None,
    )


@pytest.mark.parametrize("lane", ("recurrence", "on-the-fly"))
@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_builtin_certified_gluons_use_ddm_in_planning_and_construction(lane, accuracy):
    backend = _backend(lane, accuracy=accuracy)
    model = BuiltinSMModel()
    process = build_process_ir("g g > g g g", color_accuracy=accuracy)
    assert backend._fft_color_basis(process, model) == "adjoint"
    assert (
        backend._plan_concrete_process(process, model=model)["color_sector_count"] == 6
    )
    prepared = backend._prepare_process_construction(process, model)
    assert prepared.complete_color_plan.basis == "adjoint"
    assert prepared.restricted_color_plan.basis == "adjoint"
    assert prepared.complete_color_plan.sector_count == 6
    assert not prepared.complete_color_plan.trace_reflections_folded


@pytest.mark.parametrize("lane", ("recurrence", "on-the-fly"))
def test_adjoint_scope_requires_model_proof_not_external_adjoint_roles(lane):
    process = build_process_ir("g g > g g", color_accuracy="full")
    backend = _backend(lane)
    for model in (None, _UncertifiedAdjointModel()):
        assert backend._fft_color_basis(process, model) == "trace"
        planned = backend._plan_concrete_process(process, model=model)
        selected = planned["fft_basis_selection"]
        assert selected["requested"] == "adjoint"
        assert selected["name"] == "trace"
        assert "No DDM amplitude identity" in selected["reason"]


@pytest.mark.parametrize("lane", ("recurrence", "on-the-fly"))
@pytest.mark.parametrize(
    "process,name",
    (
        ("d d~ > g g", "fundamental-chain"),
        ("d d~ > d d~ g g", "fundamental-chain-products"),
        ("g g > g g z", "trace"),
        ("e- e+ > mu- mu+", "singlet"),
    ),
)
def test_general_adjoint_selection_preserves_other_exact_bases(lane, process, name):
    backend = _backend(lane)
    model = BuiltinSMModel()
    process_ir = build_process_ir(process, color_accuracy="full")
    assert backend._fft_color_basis(process_ir, model) == "trace"
    planned = backend._plan_concrete_process(process_ir, model=model)
    assert planned["fft_basis_selection"]["name"] == name
    prepared = backend._prepare_process_construction(process_ir, model)
    assert prepared.complete_color_plan.basis == "trace"
    assert backend._fft_basis_metadata(prepared.complete_color_plan) == {
        "fft_basis_selection": planned["fft_basis_selection"]
    }


def test_trace_scope_does_not_require_the_new_adjoint_certificate():
    assert (
        _backend("recurrence", basis="trace")._fft_color_basis(
            build_process_ir("d d~ > g g", color_accuracy="full"), None
        )
        == "trace"
    )


@pytest.mark.parametrize("lane", ("recurrence", "on-the-fly"))
def test_automatic_default_selects_certified_ddm_and_direct_opt_out_does_not(lane):
    process = build_process_ir("g g > g g", color_accuracy="full")
    model = BuiltinSMModel()
    for contraction, expected in (("auto", "adjoint"), ("direct", "trace")):
        backend = GenerationBackend(
            RunConfig(
                action="generate",
                color=ColorConfig(accuracy="full", contraction=contraction),
                evaluator=EvaluatorConfig(execution_mode=lane),
            ),
            None,
        )
        assert backend._fft_color_basis(process, model) == expected
        planned = backend._plan_concrete_process(process, model=model)
        assert ("fft_basis_selection" in planned) is (contraction == "auto")


def test_automatic_adjoint_selection_respects_representation_limits():
    model = BuiltinSMModel()
    large = build_process_ir("g g > " + " ".join(["g"] * 11), color_accuracy="full")
    backend = GenerationBackend(
        RunConfig(action="generate", color=ColorConfig(accuracy="full")), None
    )
    assert backend._fft_color_basis(large, model) == "trace"
    # A valid trace ordering need not preserve the two DDM anchors.
    process = build_process_ir("g g > g g", color_accuracy="full")
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full"),
        process=ProcessConfig(reference_color_order=(1, 2, 4, 3)),
    )
    automatic = GenerationBackend(config, None)
    planned = automatic._plan_concrete_process(process, model=model)
    assert planned["fft_basis_selection"]["actual_basis"] == "trace"
    assert "ordering" in planned["fft_basis_selection"]["reason"]
    forced = GenerationBackend(
        replace(config, color=replace(config.color, contraction="symmetric-group-fft")),
        None,
    )
    with pytest.raises(ValueError, match="preserve both anchors"):
        forced._plan_concrete_process(process, model=model)


def test_structural_preflight_never_materializes_the_full_trace_domain(monkeypatch):
    original = generation_service.build_color_plan
    probes = []

    def bounded_probe(process, **kwargs):
        # Twelve gluons would otherwise build 11! temporary trace sectors
        # before even selecting the much smaller DDM construction domain.
        assert kwargs["max_sectors"] == 1
        result = original(process, **kwargs)
        assert len(result.sectors) <= 1
        probes.append(result)
        return result

    monkeypatch.setattr(generation_service, "build_color_plan", bounded_probe)
    expressions = (
        "g g > " + " ".join(["g"] * 10),
        "d d~ > z g g",
        "d d > z d d~",  # unbalanced all-outgoing colour endpoints
    )
    candidates = tuple(
        build_process_ir(expression, color_accuracy="full")
        for expression in expressions
    )
    selected, rejected = generation_service._select_color_ready_processes(
        candidates, color_accuracy="full"
    )
    assert selected == candidates[:2]
    assert len(rejected) == 1
    assert rejected[0].startswith(expressions[-1] + ":")
    assert probes[0].truncated  # truncation is not structural rejection


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_adjoint_otf_projection_changes_only_the_separate_colour_plan(accuracy):
    process = replace(_gluon_process(5), color_accuracy=accuracy)
    projections = {
        basis: project_on_the_fly_process_seed_v1(
            process,
            _catalog(),
            cast(Model, _NormalizationModel()),
            coupling_order_policy="minimal",
            coupling_order_limits={"QCD": 3},
            color_basis=basis,
        )
        for basis in ("trace", "adjoint")
    }
    trace = projections["trace"]
    adjoint = projections["adjoint"]
    assert trace.color_plan is not None and adjoint.color_plan is not None
    assert trace.color_plan.sector_count == 24
    assert adjoint.color_plan.sector_count == 6
    assert adjoint.color_plan.basis == "adjoint"
    assert adjoint.seed.to_json_bytes() == trace.seed.to_json_bytes()
    assert adjoint.runtime_normalization == trace.runtime_normalization
    assert all(len(leg.source_states) == 2 for leg in adjoint.seed.external_sources)


def test_adjoint_otf_projection_canonicalizes_a_valid_cyclic_reference():
    projection = project_on_the_fly_process_seed_v1(
        replace(_gluon_process(4), color_accuracy="full"),
        _catalog(),
        cast(Model, _NormalizationModel()),
        coupling_order_policy="minimal",
        coupling_order_limits={},
        color_basis="adjoint",
        reference_color_order=(3, 2, 4, 1),
    )
    assert projection.color_plan is not None
    assert projection.color_plan.sectors[0].trace_labels == (1, 3, 2, 4)


def test_existing_yang_mills_certificate_excludes_reachable_singlet_exchange():
    # This closure check is part of the existing model certificate. Merely
    # recognizing ggg and gggg is insufficient when gg can also produce H.
    def kernels(*rows):
        return cast(
            tuple[CompiledOrientedKernel, ...],
            tuple(
                SimpleNamespace(kind=kind, particles=particles)
                for kind, particles in rows
            ),
        )

    standard = kernels(
        (1, ("g", "g", "g")), (2, ("g", "g", "aux")), (3, ("aux", "g", "g"))
    )
    certified = frozenset({1, 2, 3})
    assert _yang_mills_sector_is_closed(
        "g", standard, yang_mills_kernel_kinds=certified
    )
    assert not _yang_mills_sector_is_closed(
        "g",
        (*standard, *kernels((4, ("g", "g", "H")))),
        yang_mills_kernel_kinds=certified,
    )
    # A quark pair cannot be produced from purely gluonic tree currents.
    assert _yang_mills_sector_is_closed(
        "g",
        (*standard, *kernels((4, ("q", "q~", "g")))),
        yang_mills_kernel_kinds=certified,
    )
