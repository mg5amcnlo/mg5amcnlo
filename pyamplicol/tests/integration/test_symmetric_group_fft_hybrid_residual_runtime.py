# SPDX-License-Identifier: 0BSD
"""Generated-runtime acceptance for a real hybrid FFT/direct-residual plan.

Run this deliberately bounded acceptance under the repository memory watchdog::

    PYAMPLICOL_RUN_FFT_HYBRID_RESIDUAL_RUNTIME_ACCEPTANCE=1 \
      .venv/bin/python tools/ci/memory_watchdog.py --limit-gib 30 -- \
      .venv/bin/python -m pytest -q \
      tests/integration/test_symmetric_group_fft_hybrid_residual_runtime.py

The tiny model gives a second adjoint vector a coupling to only one quark
flavour.  For ``d d~ > u u~ g x`` this leaves both a complete S2 orbit and an
incomplete fermion-line channel, so the generated recurrence artifact must use
FFT kernels together with nonzero direct residual/cross rows.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyamplicol import CompiledModel, Generator, ModelSource, Runtime
from pyamplicol.config import (
    Action,
    ColorAccuracy,
    ColorConfig,
    ColorContraction,
    EvaluatorConfig,
    EvaluatorExecutionMode,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RelationDiscoveryMode,
    RunConfig,
)
from pyamplicol.generation import recurrence_color as recurrence_color_codec
from pyamplicol.generation.phase_space import massive_rambo_final_state

_ACCEPTANCE_ENV = "PYAMPLICOL_RUN_FFT_HYBRID_RESIDUAL_RUNTIME_ACCEPTANCE"
_PROCESS = "d d~ > u u~ g x"
_HELICITIES = (-1, 1, -1, 1, -1, 1)
_SM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)

pytestmark = pytest.mark.skipif(
    os.environ.get(_ACCEPTANCE_ENV) != "1",
    reason=f"set {_ACCEPTANCE_ENV}=1 and run under the 30 GiB watchdog",
)


def _require_native_fft_runtime() -> None:
    assert importlib.util.find_spec("symbolica") is not None, (
        "the hybrid FFT runtime acceptance requires Symbolica"
    )
    assert importlib.util.find_spec("pyamplicol._rusticol") is not None, (
        "the hybrid FFT runtime acceptance requires the native Rusticol extension"
    )
    rusticol = importlib.import_module("pyamplicol._rusticol")
    assert hasattr(rusticol, "_lower_recurrence_direct_v2"), (
        "the installed Rusticol extension lacks recurrence Direct-Arena v2"
    )


def _write_flavour_selective_adjoint_model(root: Path) -> tuple[Path, Path]:
    raw = json.loads((_SM_ROOT / "sm.json").read_text(encoding="utf-8"))

    particles = {item["name"]: item for item in raw["particles"]}
    propagators = {item["name"]: item for item in raw["propagators"]}
    vertices = {item["name"]: item for item in raw["vertex_rules"]}
    lorentz = {item["name"]: item for item in raw["lorentz_structures"]}
    couplings = {item["name"]: item for item in raw["couplings"]}

    adjoint = copy.deepcopy(particles["g"])
    adjoint.update(
        {
            "name": "x",
            "antiname": "x",
            "pdg_code": 9_900_021,
            "texname": "x",
            "antitexname": "x",
            "propagator": "x_propFeynman",
        }
    )
    adjoint_propagator = copy.deepcopy(propagators["g_propFeynman"])
    adjoint_propagator.update({"name": "x_propFeynman", "particle": "x"})
    selective_vertex = copy.deepcopy(vertices["V_74"])
    selective_vertex.update({"name": "V_fft_hybrid_ddx", "particles": ["d~", "d", "x"]})

    raw["name"] = "fft-hybrid-residual"
    raw["particles"] = [
        copy.deepcopy(particles[name]) for name in ("d", "d~", "u", "u~", "g")
    ] + [adjoint]
    raw["propagators"] = [
        copy.deepcopy(propagators[name])
        for name in (
            "d_propFeynman",
            "d~_propFeynman",
            "u_propFeynman",
            "u~_propFeynman",
            "g_propFeynman",
        )
    ] + [adjoint_propagator]
    raw["vertex_rules"] = [
        copy.deepcopy(vertices["V_74"]),
        copy.deepcopy(vertices["V_135"]),
        selective_vertex,
    ]
    raw["lorentz_structures"] = [copy.deepcopy(lorentz["FFV1"])]
    raw["couplings"] = [copy.deepcopy(couplings["GC_11"])]
    raw["form_factors"] = []

    model_root = root / "model"
    model_root.mkdir()
    model_path = model_root / "fft_hybrid_residual.json"
    model_path.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    restriction_path = model_root / "restrict_default.json"
    restriction_path.write_bytes((_SM_ROOT / restriction_path.name).read_bytes())
    return model_path, restriction_path


def _generation_config(contraction: ColorContraction) -> RunConfig:
    return RunConfig(
        action=Action.GENERATE,
        process=ProcessConfig(
            selected_source_helicities={
                str(label): helicity
                for label, helicity in enumerate(_HELICITIES, start=1)
            }
        ),
        color=ColorConfig(
            accuracy=ColorAccuracy.FULL,
            contraction=contraction,
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
            relation_discovery=GenerationRelationDiscoveryConfig(
                mode=RelationDiscoveryMode.OFF
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=EvaluatorExecutionMode.RECURRENCE,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=1),
        ),
    )


@pytest.fixture(scope="module")
def prepared_hybrid_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[CompiledModel]:
    _require_native_fft_runtime()
    root = tmp_path_factory.mktemp("fft-hybrid-residual-model")
    model_path, restriction_path = _write_flavour_selective_adjoint_model(root)
    model = ModelSource.from_path(
        model_path,
        restriction=restriction_path,
    ).compile(
        cache_dir=root / "model-cache",
        use_cache=True,
        prepared_output=root / "fft-hybrid-residual-jit-o1.pyamplicol-model",
        evaluator=_generation_config(ColorContraction.DIRECT).evaluator,
    )
    assert model.is_prepared
    yield model


def _assert_hybrid_payload(artifact: Path) -> None:
    execution_paths = tuple((artifact / "processes").glob("*/execution.json"))
    assert len(execution_paths) == 1
    execution_path = execution_paths[0]
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    assert execution["kind"] == "pyamplicol-runtime-recurrence-execution"
    color = execution["runtime_metadata"]["color_contraction"]
    provenance = color["fft_provenance"]

    assert color["storage"] == "convolution-kernels"
    assert color["factorization"]["kind"] == "symmetric-group-fourier"
    assert provenance["degree"] == 2
    assert provenance["covered_local_group_count"] > 0
    assert provenance["residual_group_count"] > 0
    assert provenance["residual_entry_count"] > 0

    residual_count = provenance["residual_group_count"]
    residual_triangle_count = residual_count * (residual_count + 1) // 2
    assert provenance["residual_entry_count"] > residual_triangle_count

    degree = provenance["degree"]
    group_order = 1
    for value in range(2, degree + 1):
        group_order *= value
    channel_count = provenance["channel_count"]
    kernel_entry_count = channel_count * (channel_count + 1) // 2 * group_order

    payload = (execution_path.parent / color["path"]).read_bytes()
    header = recurrence_color_codec._HEADER.unpack_from(payload)
    assert header[0] == recurrence_color_codec.RECURRENCE_COLOR_CONTRACTION_MAGIC
    assert header[15] == color["entry_count"]
    residual_rows = tuple(
        recurrence_color_codec._ENTRY.unpack_from(
            payload,
            recurrence_color_codec._HEADER.size
            + row_index * recurrence_color_codec._ENTRY.size,
        )
        for row_index in range(kernel_entry_count, header[15])
    )
    assert len(residual_rows) == provenance["residual_entry_count"]
    covered_count = provenance["covered_local_group_count"]
    assert any(
        left < covered_count <= right and (weight_re != 0.0 or weight_im != 0.0)
        for left, right, weight_re, weight_im, _symmetry, _exact_id in residual_rows
    )


def _point() -> tuple[tuple[tuple[float, ...], ...], ...]:
    return (
        (
            (500.0, 0.0, 0.0, 500.0),
            (500.0, 0.0, 0.0, -500.0),
            *tuple(
                tuple(float(component) for component in momentum)
                for momentum in massive_rambo_final_state(
                    4,
                    sqrt_s=1000.0,
                    masses=(0.0, 0.0, 0.0, 0.0),
                    seed=101,
                )
            ),
        ),
    )


def test_generated_recurrence_hybrid_fft_residual_matches_direct(
    tmp_path: Path,
    prepared_hybrid_model: CompiledModel,
) -> None:
    artifacts = {
        contraction: tmp_path / contraction.value
        for contraction in (
            ColorContraction.DIRECT,
            ColorContraction.SYMMETRIC_GROUP_FFT,
        )
    }
    for contraction, artifact in artifacts.items():
        Generator(_generation_config(contraction)).generate(
            _PROCESS,
            artifact,
            model=prepared_hybrid_model,
        )

    fft_artifact = artifacts[ColorContraction.SYMMETRIC_GROUP_FFT]
    _assert_hybrid_payload(fft_artifact)

    direct = Runtime.load(artifacts[ColorContraction.DIRECT])
    fft = Runtime.load(fft_artifact)
    try:
        direct_value = tuple(complex(value) for value in direct.evaluate(_point()))
        fft_value = tuple(complex(value) for value in fft.evaluate(_point()))
        assert len(direct_value) == len(fft_value) == 1
        assert abs(direct_value[0]) > 0.0
        assert fft_value == pytest.approx(direct_value, rel=1.0e-10, abs=1.0e-18)
    finally:
        direct.clear()
        fft.clear()
