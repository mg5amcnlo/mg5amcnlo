# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields
from typing import cast

import pytest

from pyamplicol.api.models import (
    CompiledModelCapabilities,
    CompiledModelInfo,
    CompiledModelSource,
    ModelCompilationIssue,
    ModelCompilationPhase,
    _compiled_model_info,
)


@dataclass
class _IssueRecord:
    severity: str
    code: str
    message: str
    context: str = ""


def _set_attribute(record: object, name: str, value: object) -> None:
    setattr(record, name, value)


def _info_from_plain_records() -> tuple[
    CompiledModelInfo,
    dict[str, object],
    dict[str, object],
    list[object],
    dict[str, object],
    dict[str, object],
]:
    restriction = {"kind": "file", "sha256": "restriction-digest"}
    options = {
        "restriction": restriction,
        "simplify": False,
        "private_future_option": {"mutable": []},
    }
    source: dict[str, object] = {
        "kind": "ufo",
        "source_name": "toy_ufo",
        "digest": "source-digest",
        "options": options,
    }
    spins = [1, 2, 3]
    colors = [-3, 1, 3, 8]
    accuracies = ["lc", "nlc", "full"]
    capabilities: dict[str, object] = {
        "particle_count": 17,
        "parameter_count": 6,
        "vertex_count": 23,
        "compiled_propagator_count": 2,
        "form_factor_count": 0,
        "max_vertex_valence": 4,
        "spins": spins,
        "color_representations": colors,
        "color_accuracy_modes": accuracies,
        "has_custom_propagators": True,
        "verified_kernel_evaluation_class_count": 99,
        "private_future_detail": {"mutable": []},
    }
    warning = _IssueRecord(
        severity="warning",
        code="experimental-feature",
        message="feature is experimental",
        context="X",
    )
    error: dict[str, object] = {
        "severity": "error",
        "code": "unsupported-feature",
        "message": "feature is unsupported",
        "context": "Y",
    }
    issues: list[object] = [warning, error]
    phases: dict[str, object] = {
        "model_loading": 0.125,
        "preflight": 0.25,
        "total": 0.5,
    }
    alpha_default = [0.118, 0.0]
    parameter_defaults: dict[str, object] = {
        "zeta": (2.0, -3.0),
        "alpha": alpha_default,
    }
    info = _compiled_model_info(
        name="toy",
        schema_version=4,
        model_compiler_version=9,
        source=source,
        capabilities=capabilities,
        parameter_defaults=parameter_defaults,
        issues=issues,
        phase_timings=phases,
        conversion_seconds=0.5,
    )
    return info, source, capabilities, issues, phases, parameter_defaults


def test_plain_compiler_records_convert_to_typed_public_views() -> None:
    info, _, _, _, _, _ = _info_from_plain_records()

    assert info == CompiledModelInfo(
        name="toy",
        schema_version=4,
        model_compiler_version=9,
        source=CompiledModelSource(
            kind="ufo",
            name="toy_ufo",
            digest="source-digest",
            restriction_digest="restriction-digest",
            simplify=False,
        ),
        capabilities=CompiledModelCapabilities(
            particle_count=17,
            parameter_count=6,
            vertex_count=23,
            propagator_count=2,
            form_factor_count=0,
            maximum_valence=4,
            spins=(1, 2, 3),
            color_representations=(-3, 1, 3, 8),
            supported_color_accuracies=("lc", "nlc", "full"),
            has_custom_propagators=True,
        ),
        parameters=info.parameters,
        issues=(
            ModelCompilationIssue(
                severity="warning",
                code="experimental-feature",
                message="feature is experimental",
                context="X",
            ),
            ModelCompilationIssue(
                severity="error",
                code="unsupported-feature",
                message="feature is unsupported",
                context="Y",
            ),
        ),
        compilation_phases=(
            ModelCompilationPhase(name="model_loading", seconds=0.125),
            ModelCompilationPhase(name="preflight", seconds=0.25),
            ModelCompilationPhase(name="total", seconds=0.5),
        ),
        conversion_seconds=0.5,
    )
    assert [parameter.name for parameter in info.parameters] == ["alpha", "zeta"]
    assert info.parameters[0].kind == "external"
    assert info.parameters[0].default_real == pytest.approx(0.118)
    assert info.parameters[1].default_imaginary == pytest.approx(-3.0)
    assert info.supported is False
    assert info.phases is info.compilation_phases
    assert info.capabilities.max_vertex_valence == 4
    assert info.capabilities.color_accuracy_modes == ("lc", "nlc", "full")
    assert not hasattr(
        info.capabilities,
        "verified_kernel_evaluation_class_count",
    )
    assert not hasattr(info.capabilities, "private_future_detail")


def test_public_views_are_deeply_immutable_and_detached_from_inputs() -> None:
    info, source, capabilities, issues, phases, defaults = _info_from_plain_records()

    source_options = cast(dict[str, object], source["options"])
    restriction = cast(dict[str, object], source_options["restriction"])
    restriction["sha256"] = "changed"
    cast(list[int], capabilities["spins"]).append(5)
    cast(list[str], capabilities["color_accuracy_modes"]).clear()
    cast(_IssueRecord, issues[0]).message = "changed"
    cast(dict[str, object], issues[1])["context"] = "changed"
    phases["total"] = 99.0
    cast(list[float], defaults["alpha"])[0] = 1.0

    assert info.source.restriction_digest == "restriction-digest"
    assert info.capabilities.spins == (1, 2, 3)
    assert info.capabilities.supported_color_accuracies == ("lc", "nlc", "full")
    assert info.issues[0].message == "feature is experimental"
    assert info.issues[1].context == "Y"
    assert info.compilation_phases[-1].seconds == pytest.approx(0.5)
    assert info.parameters[0].default_real == pytest.approx(0.118)

    records = (
        info,
        info.source,
        info.capabilities,
        *info.parameters,
        *info.issues,
        *info.compilation_phases,
    )
    assert all(not hasattr(record, "__dict__") for record in records)
    assert all(isinstance(hash(record), int) for record in records)
    assert isinstance(info.parameters, tuple)
    assert isinstance(info.issues, tuple)
    assert isinstance(info.compilation_phases, tuple)
    assert isinstance(info.capabilities.spins, tuple)
    assert isinstance(info.capabilities.color_representations, tuple)
    assert isinstance(info.capabilities.supported_color_accuracies, tuple)
    assert all(
        not isinstance(getattr(info.capabilities, field.name), dict)
        for field in fields(info.capabilities)
    )

    with pytest.raises(FrozenInstanceError):
        _set_attribute(info, "name", "changed")
    with pytest.raises(FrozenInstanceError):
        _set_attribute(info.source, "digest", "changed")
    with pytest.raises(FrozenInstanceError):
        _set_attribute(info.capabilities, "spins", ())
    with pytest.raises(FrozenInstanceError):
        _set_attribute(info.issues[0], "message", "changed")
    with pytest.raises(FrozenInstanceError):
        _set_attribute(info.compilation_phases[0], "seconds", 1.0)
