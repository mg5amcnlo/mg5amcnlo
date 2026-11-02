# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.resources
import importlib.util
import io
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest

from pyamplicol import CompiledModel, Generator, ModelSource, ProcessSet
from pyamplicol.config import (
    ColorConfig,
    CppConfig,
    EvaluatorBackend,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    JITConfig,
    RunConfig,
)

_PARAMETER_ID = "aS"
_EAGER_PARAMETER_ID = "normalization.alpha_ew"
_EAGER_PROCESS = "d d~ > z"
_EAGER_NAME = "ddbar_z_eager"
_MIXED_PROCESSES = (
    "d d~ > z g",
    "d d~ > z g g",
)
_MIXED_NAMES = ("ddbar_zg", "ddbar_zgg")
_METADATA_KEYS = (
    "process",
    "process_key",
    "color_accuracy",
    "external_particles",
    "helicities",
    "colors",
    "shape",
)
_LANGUAGES = ("python", "c", "cpp", "fortran", "rust")
_NATIVE_LANGUAGES = _LANGUAGES[1:]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EAGER_DIRECT_RUNTIME_CAPABILITIES = [
    "eager-direct-arena-v1",
    "rusticol.eager-runtime-layout.complex-f64.v1",
]


def _source_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_paths = [str((_PROJECT_ROOT / "src").resolve())]
    source_paths.extend(
        str(Path(entry).resolve())
        for entry in sys.path
        if entry and str(Path(entry).resolve()) not in source_paths
    )
    environment["PYTHONPATH"] = os.pathsep.join(source_paths)
    return environment


@dataclass(frozen=True, slots=True)
class _NativeTools:
    make: str
    cc: str
    cxx: str
    fortran: str
    rustc: str
    rusticol_config: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _BuiltBundle:
    artifact: Path
    process_ids: tuple[str, ...]
    python_driver: Path
    c_driver: Path
    cpp_driver: Path
    fortran_driver: Path
    rust_driver: Path


def _required_native_tests() -> bool:
    return os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1"


def _unavailable(reason: str) -> None:
    if _required_native_tests():
        pytest.fail(reason)
    pytest.skip(reason)


def _tool_from_environment(variable: str, default: str) -> str | None:
    configured = os.environ.get(variable)
    if configured:
        command = configured.split()[0]
        located = shutil.which(command)
        return located or (command if Path(command).is_file() else None)
    return shutil.which(default)


def _rusticol_config() -> tuple[str, ...] | None:
    configured = os.environ.get("RUSTICOL_CONFIG")
    if configured:
        command = shlex.split(configured)
        if command:
            executable = shutil.which(command[0])
            if executable or Path(command[0]).expanduser().is_file():
                return (executable or str(Path(command[0]).expanduser()), *command[1:])
    sibling = Path(sys.executable).parent / "rusticol-config"
    if sibling.is_file():
        return (str(sibling),)
    discovered = shutil.which("rusticol-config")
    if discovered:
        return (discovered,)
    if importlib.util.find_spec("rusticol_config") is not None:
        return (sys.executable, "-m", "rusticol_config")
    return None


@pytest.fixture(scope="module")
def native_tools() -> _NativeTools:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    discovered = {
        "make": shutil.which("make"),
        "cc": _tool_from_environment("CC", "cc"),
        "cxx": _tool_from_environment("CXX", "c++"),
        "fortran": _tool_from_environment("FC", "gfortran"),
        "rustc": _tool_from_environment("RUSTC", "rustc"),
        "rusticol_config": _rusticol_config(),
    }
    missing = tuple(name for name, value in discovered.items() if value is None)
    if missing:
        _unavailable("cross-language tests require " + ", ".join(missing))
    tools = _NativeTools(**discovered)  # type: ignore[arg-type]
    checked = subprocess.run(
        [*tools.rusticol_config, "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if checked.returncode != 0:
        _unavailable(
            "the installed Rusticol SDK is unavailable: " + checked.stderr.strip()
        )
    return tools


def _generation_config(
    accuracy: str,
    execution_mode: Literal["compiled", "eager"] = "compiled",
    *,
    evaluator: EvaluatorConfig | None = None,
) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy),
        generation=GenerationConfig(workers=1, emit_api_bundle=True),
        evaluator=evaluator
        or EvaluatorConfig(
            execution_mode=execution_mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=0 if execution_mode == "eager" else 3),
        ),
    )


def _manifest_process_ids(artifact: Path) -> tuple[str, ...]:
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    return tuple(str(process["id"]) for process in manifest["processes"])


def _assert_complete_runtime_coverage(
    artifact: Path, process_ids: tuple[str, ...], accuracy: str
) -> None:
    for process_id in process_ids:
        physics = json.loads(
            (artifact / "processes" / process_id / "physics.json").read_text(
                encoding="utf-8"
            )
        )
        assert physics["color_accuracy"] == accuracy
        assert physics["coverage"]["helicities"] == "complete"
        expected_color_coverage = "complete" if accuracy == "lc" else "contracted"
        assert physics["coverage"]["color"] == expected_color_coverage
        assert physics["selectors"]["helicity"] is True
        assert physics["selectors"]["color_flow"] is (accuracy == "lc")
        assert physics["selectors"]["contracted_color"] is False


def _assert_one_root_bundle(artifact: Path, process_ids: tuple[str, ...]) -> None:
    api_directories = tuple(path for path in artifact.rglob("API") if path.is_dir())
    assert api_directories == (artifact / "API",)
    for process_id in process_ids:
        assert not (artifact / "processes" / process_id / "API").exists()

    rows = {
        line.split("\t", 1)[0]
        for line in (artifact / "API/validation_points.dat")
        .read_text(encoding="ascii")
        .splitlines()[1:]
        if line and not line.startswith("#")
    }
    assert rows == set(process_ids)

    c_source = (artifact / "API/c/check_standalone.c").read_text(encoding="utf-8")
    assert "#include <rusticol.h>" in c_source
    assert "rusticol_runtime_load" in c_source
    assert "rusticol_runtime_evaluate_f64" in c_source
    assert "rusticol_runtime_evaluate_resolved_f64" in c_source

    rust_source = (artifact / "API/rust/check_standalone.rs").read_text(
        encoding="utf-8"
    )
    assert "Runtime::load" in rust_source
    assert "unsafe" not in rust_source
    assert 'extern "C"' not in rust_source
    assert "rusticol_runtime_" not in rust_source


def _build_bundle(
    artifact: Path,
    expressions: tuple[str, ...],
    names: tuple[str, ...],
    accuracy: str,
    model: CompiledModel | None,
    tools: _NativeTools,
    execution_mode: Literal["compiled", "eager"] = "compiled",
    evaluator: EvaluatorConfig | None = None,
) -> _BuiltBundle:
    processes = ProcessSet.from_expressions(expressions, names=names)
    Generator(
        _generation_config(
            accuracy,
            execution_mode,
            evaluator=evaluator,
        )
    ).generate(
        processes,
        artifact,
        model=model,
    )
    process_ids = _manifest_process_ids(artifact)
    assert process_ids == names
    _assert_complete_runtime_coverage(artifact, process_ids, accuracy)
    _assert_one_root_bundle(artifact, process_ids)

    environment = _source_environment()
    environment.update(
        {
            "CC": tools.cc,
            "CXX": tools.cxx,
            "FC": tools.fortran,
            "RUSTC": tools.rustc,
            "RUSTICOL_CONFIG": shlex.join(tools.rusticol_config),
        }
    )
    for language in _NATIVE_LANGUAGES:
        make_arguments = [
            tools.make,
            "-C",
            str(artifact / "API" / language),
        ]
        if language == "rust":
            make_arguments.append(f"RUSTC={tools.rustc}")
        make_arguments.append("all")
        completed = subprocess.run(
            make_arguments,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, (
            f"{language} API driver failed to build\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    return _BuiltBundle(
        artifact=artifact,
        process_ids=process_ids,
        python_driver=artifact / "API/python/check_standalone.py",
        c_driver=(
            artifact.parent
            / ".pyamplicol-api-build"
            / artifact.name
            / "c/check_standalone"
        ),
        cpp_driver=(
            artifact.parent
            / ".pyamplicol-api-build"
            / artifact.name
            / "cpp/check_standalone"
        ),
        fortran_driver=(
            artifact.parent
            / ".pyamplicol-api-build"
            / artifact.name
            / "fortran/check_standalone"
        ),
        rust_driver=(
            artifact.parent
            / ".pyamplicol-api-build"
            / artifact.name
            / "rust/check_standalone"
        ),
    )


def test_safe_rust_demo_compiles_and_runs_with_installed_or_source_sdk(
    native_tools: _NativeTools,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [*native_tools.rusticol_config, "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    sdk = json.loads(completed.stdout)
    package = importlib.resources.files("pyamplicol")
    if not isinstance(package, os.PathLike):
        _unavailable("the Rust SDK integration test requires an unpacked installation")
    package_path = Path(os.fspath(package))
    artifact = package_path / "assets" / "selftest" / sdk["target"] / "artifact"
    source = package_path / "assets" / "api_templates" / "rust" / "check_standalone.rs"
    if not artifact.is_dir() or not source.is_file():
        _unavailable("the target-specific Rust SDK self-test fixture is unavailable")

    binary = tmp_path / "check_standalone_rust"
    environment = os.environ.copy()
    environment["RUSTICOL_RUST_SOURCE"] = str(sdk["rust_source"])
    compiled = subprocess.run(
        [
            native_tools.rustc,
            "--edition=2021",
            "-Dwarnings",
            "-C",
            "opt-level=2",
            str(source),
            "-o",
            str(binary),
            *map(str, sdk["rust_flags"]),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, (
        f"safe Rust API driver failed to compile\n"
        f"stdout:\n{compiled.stdout}\nstderr:\n{compiled.stderr}"
    )

    expected = json.loads(
        (artifact.parent / "expected.json").read_text(encoding="utf-8")
    )
    process_id = str(expected["process_id"])

    def run(*arguments: str) -> dict[str, Any]:
        result = subprocess.run(
            [str(binary), "--json", "--process", process_id, *arguments],
            cwd=artifact,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    baseline = run()
    model_card = tmp_path / "model-parameters.json"
    model_card.write_text(
        json.dumps({"normalization.alpha_ew": [0.008, 0.0]}) + "\n",
        encoding="utf-8",
    )
    card = run("--model-parameters", str(model_card))
    direct = run("--set-parameter", "normalization.alpha_ew", "0.008", "0")

    for payload in (baseline, card, direct):
        assert payload["language"] == "rust"
        assert payload["available"] is True
        assert payload["process_key"] == process_id
        assert payload["resolved_sum"] == pytest.approx(
            payload["compatibility_total"], rel=1.0e-12, abs=1.0e-15
        )
        assert sum(payload["values"]) == pytest.approx(
            payload["compatibility_total"][0], rel=1.0e-12, abs=1.0e-15
        )
    assert card["values"] == pytest.approx(direct["values"])
    assert card["compatibility_total"] != pytest.approx(
        baseline["compatibility_total"], rel=1.0e-10
    )


@pytest.fixture(scope="module")
def external_sm_model(tmp_path_factory: pytest.TempPathFactory) -> CompiledModel:
    resource = importlib.resources.files("pyamplicol").joinpath(
        "assets/models/json/sm/sm.json"
    )
    cache = tmp_path_factory.mktemp("multilanguage-model-cache")
    with importlib.resources.as_file(resource) as source:
        return ModelSource.from_path(source, restriction="default").compile(
            cache_dir=cache,
            use_cache=True,
        )


@pytest.fixture(scope="module")
def generated_bundles(
    tmp_path_factory: pytest.TempPathFactory,
    native_tools: _NativeTools,
    external_sm_model: CompiledModel,
) -> dict[str, _BuiltBundle]:
    root = tmp_path_factory.mktemp("multilanguage-api")
    return {
        "mixed": _build_bundle(
            root / "mixed-lc",
            _MIXED_PROCESSES,
            _MIXED_NAMES,
            "lc",
            external_sm_model,
            native_tools,
        ),
        "single": _build_bundle(
            root / "single-lc",
            (_MIXED_PROCESSES[0],),
            (_MIXED_NAMES[0],),
            "lc",
            external_sm_model,
            native_tools,
        ),
        "nlc": _build_bundle(
            root / "single-nlc",
            (_MIXED_PROCESSES[0],),
            (_MIXED_NAMES[0],),
            "nlc",
            external_sm_model,
            native_tools,
        ),
        "full": _build_bundle(
            root / "single-full",
            (_MIXED_PROCESSES[0],),
            (_MIXED_NAMES[0],),
            "full",
            external_sm_model,
            native_tools,
        ),
    }


@pytest.fixture(scope="module")
def generated_eager_bundle(
    tmp_path_factory: pytest.TempPathFactory,
    native_tools: _NativeTools,
) -> _BuiltBundle:
    root = tmp_path_factory.mktemp("multilanguage-eager-api")
    return _build_bundle(
        root / "single-eager-lc",
        (_EAGER_PROCESS,),
        (_EAGER_NAME,),
        "lc",
        None,
        native_tools,
        execution_mode="eager",
    )


@pytest.fixture(scope="module")
def generated_native_eager_bundles(
    tmp_path_factory: pytest.TempPathFactory,
    native_tools: _NativeTools,
) -> dict[str, _BuiltBundle]:
    root = tmp_path_factory.mktemp("multilanguage-native-eager-api")
    bundles: dict[str, _BuiltBundle] = {}
    for backend in (EvaluatorBackend.ASM, EvaluatorBackend.CPP):
        evaluator = EvaluatorConfig(
            backend=backend,
            execution_mode="eager",
            optimization=EvaluatorOptimizationConfig(cores=1),
            cpp=CppConfig(compiler=native_tools.cxx),
        )
        prepared = ModelSource.built_in_sm().compile(
            use_cache=True,
            prepared_output=root / f"built-in-sm-{backend}.pyamplicol-model",
            evaluator=evaluator,
        )
        bundles[str(backend)] = _build_bundle(
            root / f"single-eager-{backend}",
            (_EAGER_PROCESS,),
            (f"ddbar_z_eager_{backend}",),
            "lc",
            prepared,
            native_tools,
            execution_mode="eager",
            evaluator=evaluator,
        )
    return bundles


def _driver_command(bundle: _BuiltBundle, language: str) -> list[str]:
    if language == "python":
        return [sys.executable, str(bundle.python_driver)]
    if language == "c":
        return [str(bundle.c_driver)]
    if language == "cpp":
        return [str(bundle.cpp_driver)]
    if language == "fortran":
        return [str(bundle.fortran_driver)]
    if language == "rust":
        return [str(bundle.rust_driver)]
    raise AssertionError(language)


def _run_python_driver_in_process(
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    driver = Path(command[1])
    namespace = runpy.run_path(str(driver), run_name="_pyamplicol_api_driver")
    main = namespace.get("main")
    assert callable(main)
    output = io.StringIO()
    with (
        patch.object(sys, "argv", [str(driver), *command[2:]]),
        redirect_stdout(output),
    ):
        returncode = main()
    assert isinstance(returncode, int)
    return subprocess.CompletedProcess(command, returncode, output.getvalue(), "")


def _run_driver(
    bundle: _BuiltBundle,
    language: str,
    *,
    process: str | None,
    model_card: Path | None = None,
    kinematics: Path | None = None,
    override: float | None = None,
    parameter_id: str = _PARAMETER_ID,
    precision: int = 16,
) -> dict[str, Any]:
    command = [*_driver_command(bundle, language), "--json"]
    if process is not None:
        command.extend(("--process", process))
    if model_card is not None:
        command.extend(("--model-parameters", str(model_card)))
    if kinematics is not None:
        command.extend(("--kinematics", str(kinematics)))
    if override is not None:
        command.extend(("--set-parameter", parameter_id, f"{override:.17g}", "0"))
    if precision != 16:
        command.extend(("--precision", str(precision)))
    if language == "python":
        # Generation owns the restricted Symbolica session, and importing the
        # public Python API initializes Symbolica even for binary64 evaluation.
        # Exercise the same generated entry point in that session. Independent
        # Python subprocess coverage lives in the serialized deployment smoke.
        completed = _run_python_driver_in_process(command)
    else:
        completed = subprocess.run(
            command,
            cwd=bundle.artifact,
            env=_source_environment(),
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert completed.returncode == 0, (
        f"{language} API driver failed\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        pytest.fail(
            f"{language} driver did not emit one JSON object: {error}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    assert payload["language"] == language
    assert payload["available"] is True
    assert payload["precision"] == precision
    return payload


@pytest.mark.parametrize("precision", (16, 32, 64))
def test_python_driver_reuses_the_generation_session_at_every_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, precision: int
) -> None:
    driver = tmp_path / "check_standalone.py"
    bundle = _BuiltBundle(tmp_path, ("test",), driver, driver, driver, driver, driver)
    commands: list[list[str]] = []

    def in_process(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        payload = {"language": "python", "available": True, "precision": precision}
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    def no_child(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Python API parity must reuse the generation session")

    monkeypatch.setattr(
        sys.modules[__name__], "_run_python_driver_in_process", in_process
    )
    monkeypatch.setattr(subprocess, "run", no_child)
    assert _run_driver(bundle, "python", process="test", precision=precision)[
        "precision"
    ] == precision
    assert len(commands) == 1
    assert commands[0][:2] == [sys.executable, str(driver)]
    assert ("--precision" in commands[0]) == (precision != 16)


def _assert_numeric_sequence(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    assert actual == pytest.approx(expected, rel=1.0e-12, abs=1.0e-15)


def _assert_language_payloads(payloads: dict[str, dict[str, Any]]) -> None:
    reference = payloads["python"]
    for language, payload in payloads.items():
        for key in _METADATA_KEYS:
            assert payload[key] == reference[key], (language, key)
        _assert_numeric_sequence(payload["values"], reference["values"])
        _assert_numeric_sequence(payload["resolved_sum"], reference["resolved_sum"])
        _assert_numeric_sequence(
            payload["compatibility_total"], reference["compatibility_total"]
        )
        _assert_numeric_sequence(
            payload["resolved_sum"], payload["compatibility_total"]
        )
        assert len(payload["values"]) == (
            payload["shape"][0] * payload["shape"][1] * payload["shape"][2]
        )


def _all_languages(
    bundle: _BuiltBundle,
    *,
    process: str | None,
    model_card: Path | None = None,
    kinematics: Path | None = None,
    override: float | None = None,
    parameter_id: str = _PARAMETER_ID,
) -> dict[str, dict[str, Any]]:
    return {
        language: _run_driver(
            bundle,
            language,
            process=process,
            model_card=model_card,
            kinematics=kinematics,
            override=override,
            parameter_id=parameter_id,
        )
        for language in _LANGUAGES
    }


def _validation_point(bundle: _BuiltBundle, process_id: str) -> list[list[float]]:
    lines = (bundle.artifact / "API/validation_points.dat").read_text(
        encoding="ascii"
    ).splitlines()
    assert lines[0] == "RUSTICOL_VALIDATION_POINTS_V1"
    for line in lines[1:]:
        fields = line.split("\t")
        if not fields or fields[0] != process_id:
            continue
        external_count = int(fields[1])
        components = tuple(float(value) for value in fields[2:])
        assert len(components) == external_count * 4
        return [
            list(components[offset : offset + 4])
            for offset in range(0, len(components), 4)
        ]
    raise AssertionError(f"no validation point for {process_id!r}")


def _reorder_point(
    point: list[list[float]], permutation: tuple[int, ...]
) -> list[list[float]]:
    reordered: list[list[float] | None] = [None] * len(point)
    for representative_index, public_index in enumerate(permutation):
        reordered[public_index] = point[representative_index]
    assert all(vector is not None for vector in reordered)
    return [vector for vector in reordered if vector is not None]


def _assert_public_metadata_permutation(
    representative: dict[str, Any],
    public: dict[str, Any],
    permutation: tuple[int, ...],
) -> None:
    expected_particles: list[dict[str, Any] | None] = [None] * len(permutation)
    for representative_index, public_index in enumerate(permutation):
        expected_particles[public_index] = {
            **representative["external_particles"][representative_index],
            "index": public_index,
        }
    assert public["external_particles"] == expected_particles

    for representative_helicity, public_helicity in zip(
        representative["helicities"], public["helicities"], strict=True
    ):
        expected_values: list[int | None] = [None] * len(permutation)
        for representative_index, public_index in enumerate(permutation):
            expected_values[public_index] = representative_helicity["helicities"][
                representative_index
            ]
        assert public_helicity["helicities"] == expected_values

    def remap_word(word: list[int]) -> list[int]:
        return [permutation[label - 1] + 1 for label in word]

    assert [color["word"] for color in public["colors"]] == [
        remap_word(color["word"]) for color in representative["colors"]
    ]


def test_all_generated_clients_load_and_evaluate_the_same_eager_artifact(
    generated_eager_bundle: _BuiltBundle,
    tmp_path: Path,
) -> None:
    bundle = generated_eager_bundle
    manifest = json.loads(
        (bundle.artifact / "artifact.json").read_text(encoding="utf-8")
    )
    assert manifest["default_process_id"] == bundle.process_ids[0]
    assert (
        manifest["runtime"]["required_runtime_capabilities"]
        == _EAGER_DIRECT_RUNTIME_CAPABILITIES
    )
    for process_id in bundle.process_ids:
        process = next(
            item for item in manifest["processes"] if item["id"] == process_id
        )
        assert (
            process["required_runtime_capabilities"]
            == _EAGER_DIRECT_RUNTIME_CAPABILITIES
        )
        execution = json.loads(
            (bundle.artifact / "processes" / process_id / "execution.json").read_text(
                encoding="utf-8"
            )
        )
        assert execution["kind"] == "pyamplicol-runtime-eager-execution"

    process_id = bundle.process_ids[0]
    model_card = tmp_path / "eager-model-parameters.json"
    updated_parameter = 0.008
    model_card.write_text(
        json.dumps({_EAGER_PARAMETER_ID: [updated_parameter, 0.0]}) + "\n",
        encoding="utf-8",
    )
    baseline = _all_languages(bundle, process=process_id)
    card = _all_languages(bundle, process=process_id, model_card=model_card)
    direct = _all_languages(
        bundle,
        process=process_id,
        override=updated_parameter,
        parameter_id=_EAGER_PARAMETER_ID,
    )
    for payloads in (baseline, card, direct):
        _assert_language_payloads(payloads)
    _assert_numeric_sequence(card["python"]["values"], direct["python"]["values"])
    assert card["python"]["compatibility_total"] != pytest.approx(
        baseline["python"]["compatibility_total"], rel=1.0e-10
    )


def test_all_generated_clients_support_native_eager_kernel_backends(
    generated_native_eager_bundles: dict[str, _BuiltBundle],
) -> None:
    for backend, bundle in generated_native_eager_bundles.items():
        manifest = json.loads(
            (bundle.artifact / "artifact.json").read_text(encoding="utf-8")
        )
        assert manifest["extensions"]["eager_prepared_pack"]["backend"] == backend
        assert (
            manifest["runtime"]["required_runtime_capabilities"]
            == _EAGER_DIRECT_RUNTIME_CAPABILITIES
        )
        for process_id in bundle.process_ids:
            process = next(
                item for item in manifest["processes"] if item["id"] == process_id
            )
            assert (
                process["required_runtime_capabilities"]
                == _EAGER_DIRECT_RUNTIME_CAPABILITIES
            )
        _assert_language_payloads(_all_languages(bundle, process=bundle.process_ids[0]))


def test_mixed_lc_bundle_agrees_for_all_processes_and_parameter_updates(
    generated_bundles: dict[str, _BuiltBundle],
    tmp_path: Path,
) -> None:
    bundle = generated_bundles["mixed"]
    model_card = tmp_path / "model-parameters.json"
    model_card.write_text(
        json.dumps({_PARAMETER_ID: [0.109, 0.0]}) + "\n",
        encoding="utf-8",
    )
    assert "alpha_s" not in model_card.read_text(encoding="utf-8").replace(
        _PARAMETER_ID, ""
    )

    for process_id in bundle.process_ids:
        baseline = _all_languages(bundle, process=process_id)
        card = _all_languages(bundle, process=process_id, model_card=model_card)
        direct = _all_languages(bundle, process=process_id, override=0.127)
        for payloads in (baseline, card, direct):
            _assert_language_payloads(payloads)

        baseline_total = baseline["python"]["compatibility_total"][0]
        card_total = card["python"]["compatibility_total"][0]
        direct_total = direct["python"]["compatibility_total"][0]
        assert card_total != pytest.approx(baseline_total, rel=1.0e-10)
        assert direct_total != pytest.approx(baseline_total, rel=1.0e-10)


@pytest.mark.parametrize(
    ("expression", "stable_id"),
    tuple(zip(_MIXED_PROCESSES, _MIXED_NAMES, strict=True)),
)
def test_all_languages_select_processes_by_concrete_expression(
    generated_bundles: dict[str, _BuiltBundle],
    expression: str,
    stable_id: str,
) -> None:
    bundle = generated_bundles["mixed"]
    selector = "  ".join(expression.split())
    payloads = _all_languages(bundle, process=selector)

    _assert_language_payloads(payloads)
    assert payloads["python"]["process"].casefold() == expression.casefold()
    assert payloads["python"]["process_key"] == stable_id


def test_all_languages_infer_public_order_and_accept_custom_kinematics(
    generated_bundles: dict[str, _BuiltBundle],
    tmp_path: Path,
) -> None:
    bundle = generated_bundles["mixed"]
    representative_id = _MIXED_NAMES[1]
    requested_process = "d d~ > g z g"
    permutation = (0, 1, 3, 2, 4)

    representative = _all_languages(bundle, process=representative_id)
    public_default = _all_languages(bundle, process=requested_process)
    for payloads in (representative, public_default):
        _assert_language_payloads(payloads)
    for language in _LANGUAGES:
        public = public_default[language]
        source = representative[language]
        assert public["process"].casefold() == requested_process.casefold()
        assert public["process_key"] == representative_id
        _assert_public_metadata_permutation(source, public, permutation)
        _assert_numeric_sequence(public["values"], source["values"])
        _assert_numeric_sequence(
            public["compatibility_total"], source["compatibility_total"]
        )

    representative_point = _validation_point(bundle, representative_id)
    public_point = _reorder_point(representative_point, permutation)
    numeric_path = tmp_path / "public-numeric-kinematics.json"
    numeric_path.write_text(json.dumps(public_point) + "\n", encoding="utf-8")
    string_path = tmp_path / "public-string-kinematics.json"
    string_point = [[format(value, ".17g") for value in leg] for leg in public_point]
    string_path.write_text(json.dumps([string_point]) + "\n", encoding="utf-8")

    numeric = _all_languages(
        bundle,
        process=requested_process,
        kinematics=numeric_path,
    )
    strings = _all_languages(
        bundle,
        process=requested_process,
        kinematics=string_path,
    )
    for payloads in (numeric, strings):
        _assert_language_payloads(payloads)
        _assert_numeric_sequence(
            payloads["python"]["values"], public_default["python"]["values"]
        )

    model_card = tmp_path / "ufo-model-parameters.json"
    model_card.write_text(
        json.dumps({_PARAMETER_ID: [0.109, 0.0]}) + "\n",
        encoding="utf-8",
    )
    overridden = _all_languages(
        bundle,
        process=requested_process,
        model_card=model_card,
        override=0.127,
    )
    direct = _all_languages(bundle, process=requested_process, override=0.127)
    for payloads in (overridden, direct):
        _assert_language_payloads(payloads)
    _assert_numeric_sequence(
        overridden["python"]["values"], direct["python"]["values"]
    )

    high_precision_a = tmp_path / "public-high-precision-a.json"
    high_precision_b = tmp_path / "public-high-precision-b.json"
    perturbed = [list(leg) for leg in string_point]
    perturbed[0][0] = str(Decimal(perturbed[0][0]) + Decimal("1e-25"))
    high_precision_a.write_text(json.dumps(string_point) + "\n", encoding="utf-8")
    high_precision_b.write_text(json.dumps(perturbed) + "\n", encoding="utf-8")
    exact_a = _run_driver(
        bundle,
        "python",
        process=requested_process,
        kinematics=high_precision_a,
        precision=50,
    )
    exact_b = _run_driver(
        bundle,
        "python",
        process=requested_process,
        kinematics=high_precision_b,
        precision=50,
    )
    assert exact_a["compatibility_total"] != exact_b["compatibility_total"]


def test_single_process_bundle_uses_default_process_without_selector(
    generated_bundles: dict[str, _BuiltBundle],
) -> None:
    bundle = generated_bundles["single"]
    payloads = _all_languages(bundle, process=None)
    _assert_language_payloads(payloads)
    assert payloads["python"]["process_key"] == bundle.process_ids[0]


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_native_contracted_color_is_one_component_per_helicity(
    generated_bundles: dict[str, _BuiltBundle],
    accuracy: str,
) -> None:
    bundle = generated_bundles[accuracy]
    payloads = _all_languages(bundle, process=bundle.process_ids[0])
    _assert_language_payloads(payloads)
    payload = payloads["python"]
    assert payload["color_accuracy"] == accuracy
    assert payload["shape"][2] == 1
    assert len(payload["colors"]) == 1
    assert payload["colors"][0]["kind"] == "contracted-color"
    assert payload["colors"][0]["word"] == []
