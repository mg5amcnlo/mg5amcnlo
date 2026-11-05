# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path

import pytest

import pyamplicol.api.models as api_models
import pyamplicol.models.loading as model_loading
from pyamplicol._internal.versions import COMPILED_MODEL_SCHEMA_VERSION
from pyamplicol.api import (
    ColorFlow,
    CompiledModel,
    HelicityConfiguration,
    ModelError,
    ModelSource,
    PhysicsReduction,
    ProcessAlias,
    ProcessPhysics,
    ProcessRequest,
    ProcessSet,
    ReductionGroup,
    ResolvedEvaluation,
)
from pyamplicol.config import ModelConfig


def test_model_source_kind_resolution_does_not_import_symbolica(tmp_path: Path) -> None:
    sys.modules.pop("symbolica", None)
    ufo = tmp_path / "ufo"
    ufo.mkdir()
    serialized = tmp_path / "model.json"
    serialized.write_text("{}", encoding="utf-8")
    compiled = tmp_path / "sm.pyamplicol-model.json"
    compiled.write_text("{}", encoding="utf-8")

    assert ModelSource.built_in_sm().kind == "built-in-sm"
    assert ModelSource.built_in_sm_heft().kind == "built-in-sm-heft"
    assert ModelSource.from_path(ufo).kind == "ufo"
    assert ModelSource.from_path(serialized).kind == "json"
    assert ModelSource.from_path(compiled).kind == "compiled"
    assert "symbolica" not in sys.modules


def test_model_source_validates_restriction_files(tmp_path: Path) -> None:
    ufo = tmp_path / "ufo"
    ufo.mkdir()
    restriction = ufo / "restrict_default.dat"
    restriction.write_text("BLOCK MASS\n", encoding="ascii")

    source = ModelSource.from_path(ufo, restriction="restrict_default.dat")

    assert source.restriction == restriction
    with pytest.raises(ModelError, match="does not exist"):
        ModelSource.from_path(ufo, restriction="restrict_missing.dat")


def test_model_source_preserves_named_restrictions_from_typed_config(
    tmp_path: Path,
) -> None:
    ufo = tmp_path / "ufo"
    ufo.mkdir()

    source = ModelSource.from_config(
        ModelConfig(
            source=str(ufo),
            restriction="no_widths",
            simplify=False,
        )
    )

    assert source.kind == "ufo"
    assert source.path == ufo
    assert source.restriction == "no_widths"
    assert source.simplify is False


def test_model_source_compile_forwards_named_restriction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ufo = tmp_path / "ufo"
    ufo.mkdir()
    source = ModelSource.from_path(ufo, restriction="no_widths")
    captured: dict[str, object] = {}
    payload = object()
    expected = object()

    def compile_source(value: object, **kwargs: object) -> object:
        captured["source"] = value
        captured.update(kwargs)
        return payload

    def wrap_compiled_model(value: object) -> object:
        captured["payload"] = value
        return expected

    monkeypatch.setattr(model_loading, "compile_model_source", compile_source)
    monkeypatch.setattr(api_models, "_compiled_model_from_payload", wrap_compiled_model)

    result = source.compile(use_cache=False)

    assert result is expected
    assert captured["payload"] is payload
    assert captured["source"] == ufo
    assert captured["restriction"] == "no_widths"
    assert captured["simplify"] is True
    assert captured["use_cache"] is False


def test_model_source_rejects_external_options_for_builtin_model() -> None:
    with pytest.raises(ModelError, match="external models"):
        ModelSource.from_config(
            ModelConfig(source="built-in-sm", restriction="no_widths")
        )
    with pytest.raises(ModelError, match="cannot be disabled"):
        ModelSource.from_config(ModelConfig(source="built-in-sm", simplify=False))


@pytest.mark.parametrize(
    "selector",
    ("built-in-sm-heft", "builtin_sm_heft", "sm-heft", "sm_heft"),
)
def test_model_source_selects_packaged_sm_heft(selector: str) -> None:
    source = ModelSource.from_config(ModelConfig(source=selector))

    assert source == ModelSource.built_in_sm_heft()
    with pytest.raises(ModelError, match="restrictions cannot be applied"):
        ModelSource.from_config(ModelConfig(source=selector, restriction="none"))
    with pytest.raises(ModelError, match="cannot be disabled"):
        ModelSource.from_config(ModelConfig(source=selector, simplify=False))


def test_public_compiled_model_uses_the_canonical_schema() -> None:
    compiled = ModelSource.built_in_sm().compile(use_cache=False)

    assert isinstance(compiled, CompiledModel)
    assert compiled.schema_version == COMPILED_MODEL_SCHEMA_VERSION == 9


def test_process_requests_normalize_and_sets_require_unique_names() -> None:
    request = ProcessRequest.parse("  d   d~   >  z g ", name="dd_zg")
    assert request.expression == "d d~ > z g"
    assert request.name == "dd_zg"
    processes = ProcessSet(
        (request,),
        aliases=(
            ProcessAlias(
                name="physical_alias",
                process_name="dd_zg",
                particle_permutation=(0, 1, 3, 2),
            ),
        ),
    )
    assert processes.aliases[0].process_name == "dd_zg"
    assert processes.aliases[0].particle_permutation == (0, 1, 3, 2)
    assert not hasattr(processes.aliases[0], "helicity_ids")
    assert not hasattr(processes.aliases[0], "color_flow_ids")
    with pytest.raises(ValueError, match="unique"):
        ProcessSet((request, request))
    with pytest.raises(ValueError, match="exactly one arrow"):
        ProcessRequest.parse("d > z > g")


def test_resolved_total_reproduces_fully_summed_values() -> None:
    resolved = ResolvedEvaluation(
        values=(
            ((1.0 + 0j, 2.0 + 0j), (3.0 + 0j, 4.0 + 0j)),
            ((5.0 + 0j, 6.0 + 0j), (7.0 + 0j, 8.0 + 0j)),
        ),
        helicity_ids=("h0", "h1"),
        color_ids=("c0", "c1"),
        color_accuracy="lc",
    )
    assert resolved.total() == (10.0 + 0j, 26.0 + 0j)


def test_resolved_decimal_total_does_not_use_ambient_context_precision() -> None:
    large = Decimal("10000000000000000000000000000")
    tiny = Decimal("0.0000000000000000000000000001")
    resolved = ResolvedEvaluation(
        values=(((large, tiny),),),
        helicity_ids=("h0",),
        color_ids=("c0", "c1"),
        color_accuracy="lc",
    )

    assert resolved.total() == (
        Decimal("10000000000000000000000000000.0000000000000000000000000001"),
    )


def test_process_physics_metadata_is_deeply_immutable() -> None:
    helicity = HelicityConfiguration("h0", 0, (1, -1), True, False, "h0", 1.0)
    flow = ColorFlow("c0", 0, (1, 2), True, "c0", 1.0)
    physics = ProcessPhysics(
        process_id="ddbar_z",
        process="d d~ > z",
        color_accuracy="lc",
        helicity_coverage="all",
        color_coverage="all",
        color_kind="flow",
        structural_zero_helicity_count=0,
        external_particles=(),
        helicities=(helicity,),
        color_flows=(flow,),
        contracted_color_components=(),
        reduction=PhysicsReduction(
            "lc-diagonal",
            (ReductionGroup("g0", "h0", "c0", ("h0",), ("c0",)),),
        ),
        model_parameters=(),
        selector_capabilities=("helicity", "color_flow"),
    )
    assert physics.helicity_ids == ("h0",)
    assert physics.color_ids == ("c0",)
    with pytest.raises(FrozenInstanceError):
        flow.id = "changed"  # type: ignore[misc]
