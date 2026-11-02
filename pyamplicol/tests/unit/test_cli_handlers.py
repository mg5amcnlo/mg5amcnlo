# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import errno
import io
import json
import os
import pty
import sys
import termios
import threading
import types
from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol.api import BenchmarkResult, BenchmarkStatistics
from pyamplicol.cli import run_cli
from pyamplicol.cli.handlers import _load_process_output, _process_set
from pyamplicol.config import (
    BenchmarkConfig,
    ConfigurationError,
    ProcessConfig,
    ProcessEntry,
    RunConfig,
)
from pyamplicol.reporting import (
    CallbackProgressSink,
    ProgressEnd,
    ProgressSink,
    ProgressStart,
    ProgressUpdate,
    get_logger,
)

ROOT = Path(__file__).resolve().parents[2]


class _Services:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.config: RunConfig | None = None

    def generate(self, config: RunConfig, progress: ProgressSink) -> object:
        del progress
        if self.fail:
            raise ConfigurationError("generation rejected")
        self.config = config
        return {
            "action": config.action,
            "workers": config.generation.workers,
            "requests": tuple(entry.expression for entry in config.process.entries),
        }

    def evaluate(self, config: RunConfig, progress: ProgressSink) -> object:
        raise AssertionError((config, progress))

    benchmark = evaluate
    inspect = evaluate
    model_inspect = evaluate
    model_compile = evaluate
    model_processes = evaluate
    group_processes = evaluate


class _ProfileServices(_Services):
    def benchmark(self, config: RunConfig, progress: ProgressSink) -> object:
        self.config = config
        progress.emit(ProgressStart("runtime-benchmark", "Profiling runtime", 2))
        progress.emit(ProgressUpdate("runtime-benchmark", 1, 2, "sampled"))
        progress.emit(ProgressUpdate("runtime-benchmark", 2, 2, "sampled"))
        progress.emit(ProgressEnd("runtime-benchmark"))
        benchmark = BenchmarkConfig(
            target_runtime=config.benchmark.target_runtime,
            batch_size=config.benchmark.batch_size,
            minimum_samples=config.benchmark.minimum_samples,
        )
        uncertainty = BenchmarkStatistics(1.0e-7, 5.0e-8, 0.05)
        return BenchmarkResult(
            requested_config=benchmark,
            effective_config=benchmark,
            sample_count=2,
            wall_time_per_point=1.0e-6,
            evaluator_time_per_point=8.0e-7,
            uncertainty=uncertainty,
            environment={
                "elapsed_seconds": benchmark.target_runtime,
                "platform": "test",
                "wall_time_source": "runtime_evaluate_wall_time",
                "evaluator_time_source": "runtime_profile_core_evaluator_call_time",
            },
            repetitions_per_sample=3,
            evaluator_uncertainty=uncertainty,
            process_id="d_dbar_to_z_g",
            process_expression="d d~ > z g",
        )


class _InterruptedProfileServices(_Services):
    def benchmark(self, config: RunConfig, progress: ProgressSink) -> object:
        del config, progress
        raise KeyboardInterrupt


class _PartialProfileServices(_ProfileServices):
    def benchmark(self, config: RunConfig, progress: ProgressSink) -> object:
        result = super().benchmark(config, progress)
        assert isinstance(result, BenchmarkResult)
        return replace(result, interrupted=True)


class _TtyCliServices(_ProfileServices):
    def generate(self, config: RunConfig, progress: ProgressSink) -> object:
        self.config = config
        logger = get_logger("tty")
        logger.info("generate before")
        progress.emit(ProgressStart("generation", "Generating processes", 1))
        progress.emit(
            ProgressStart(
                "generation:phase",
                "Constructing a deliberately width-sensitive DAG",
                2,
                parent_task_id="generation",
            )
        )
        progress.emit(ProgressUpdate("generation:phase", 1, 2, "first pass"))
        logger.warning("generate during")
        progress.emit(ProgressUpdate("generation:phase", 2, 2, "second pass"))
        progress.emit(ProgressEnd("generation:phase"))
        progress.emit(ProgressUpdate("generation", 1, 1))
        progress.emit(ProgressEnd("generation"))
        logger.info("generate after")
        return {"action": "generate"}

    def benchmark(self, config: RunConfig, progress: ProgressSink) -> object:
        logger = get_logger("tty")
        logger.info("profile before")
        progress.emit(ProgressStart("profile-load", "Loading profile artifact", 1))
        logger.warning("profile during load")
        progress.emit(ProgressUpdate("profile-load", 1, 1, "loaded"))
        progress.emit(ProgressEnd("profile-load"))
        logger.info("profile between dashboards")
        result = super().benchmark(config, progress)
        logger.info("profile after")
        return result


class _TerminalScreen:
    """Small ANSI emulator for the cursor/erase operations used by the dashboard."""

    def __init__(self, *, columns: int, rows: int = 40) -> None:
        self.columns = columns
        self.rows = [[" "] * columns for _ in range(rows)]
        self.row = 0
        self.column = 0
        self.frames: list[tuple[str, ...]] = []

    def feed(self, payload: bytes) -> None:
        text = payload.decode("utf-8", errors="replace")
        index = 0
        while index < len(text):
            character = text[index]
            if character == "\x1b" and index + 1 < len(text):
                index = self._escape(text, index)
                continue
            if character == "\r":
                self.column = 0
            elif character == "\n":
                self._move_down()
            elif character not in {"\x00", "\x08"} and character >= " ":
                self._write(character)
            index += 1

    def visible_lines(self) -> tuple[str, ...]:
        return tuple(line for row in self.rows if (line := "".join(row).rstrip()))

    def _escape(self, text: str, index: int) -> int:
        if text[index + 1] != "[":
            return index + 2
        final = index + 2
        while final < len(text) and not "@" <= text[final] <= "~":
            final += 1
        if final == len(text):
            return final
        parameters = text[index + 2 : final]
        values = [int(value) if value else 0 for value in parameters.split(";")]
        count = values[0] if values and values[0] else 1
        command = text[final]
        if command in {"F", "A"}:
            self.row = max(self.row - count, 0)
            if command == "F":
                self.column = 0
        elif command in {"E", "B"}:
            for _ in range(count):
                self._move_down()
            if command == "E":
                self.column = 0
        elif command == "K":
            start = 0 if values and values[0] == 2 else self.column
            self.rows[self.row][start:] = [" "] * (self.columns - start)
        elif command == "J":
            self.frames.append(self.visible_lines())
            self.rows[self.row][self.column :] = [" "] * (self.columns - self.column)
            for row in range(self.row + 1, len(self.rows)):
                self.rows[row] = [" "] * self.columns
        return final + 1

    def _write(self, character: str) -> None:
        if self.column >= self.columns:
            self._move_down()
            self.column = 0
        self.rows[self.row][self.column] = character
        self.column += 1

    def _move_down(self) -> None:
        self.row += 1
        if self.row < len(self.rows):
            return
        self.rows.pop(0)
        self.rows.append([" "] * self.columns)
        self.row = len(self.rows) - 1


def _run_cli_in_pty(
    arguments: tuple[str, ...],
    *,
    services: _TtyCliServices,
    columns: int = 52,
) -> tuple[int, bytes]:
    master, slave = pty.openpty()
    termios.tcsetwinsize(slave, (40, columns))
    chunks: list[bytes] = []

    def read_master() -> None:
        while True:
            try:
                chunk = os.read(master, 65536)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    return
                raise
            if not chunk:
                return
            chunks.append(chunk)

    reader = threading.Thread(target=read_master, daemon=True)
    reader.start()
    stderr = os.fdopen(os.dup(slave), "w", encoding="utf-8", buffering=1)
    os.close(slave)
    try:
        status = run_cli(
            arguments,
            services=services,
            stdout=io.StringIO(),
            stderr=stderr,
        )
    finally:
        stderr.close()
    reader.join(timeout=2.0)
    os.close(master)
    assert not reader.is_alive()
    return status, b"".join(chunks)


def _assert_stable_tty_output(
    payload: bytes,
    *,
    expected_statuses: tuple[str, ...],
    dashboard_text: str,
    columns: int = 52,
) -> None:
    assert b"\x00" not in payload
    assert b"\x1b[" in payload
    assert dashboard_text.encode() in payload

    terminal = _TerminalScreen(columns=columns)
    terminal.feed(payload)
    assert terminal.visible_lines() == expected_statuses
    assert any(
        any(dashboard_text in line for line in frame) for frame in terminal.frames
    )
    for frame in (*terminal.frames, terminal.visible_lines()):
        assert all(len(line) < columns for line in frame)
        for status in expected_statuses:
            assert sum(status in line for line in frame) <= 1


def test_generate_tty_dashboard_preserves_status_lines_and_terminal_width(
    tmp_path: Path,
) -> None:
    status, payload = _run_cli_in_pty(
        (
            "generate",
            "d d~ > z g",
            str(tmp_path / "artifact"),
            "--json",
            "--progress",
            "tty",
            "--color",
            "never",
        ),
        services=_TtyCliServices(),
    )

    assert status == 0
    _assert_stable_tty_output(
        payload,
        expected_statuses=(
            "INFO pyamplicol.tty: generate before",
            "WARNING pyamplicol.tty: generate during",
            "INFO pyamplicol.tty: generate after",
        ),
        dashboard_text="Generating processes",
    )


def test_profile_tty_dashboard_closes_between_lifecycle_roots(
    tmp_path: Path,
) -> None:
    status, payload = _run_cli_in_pty(
        (
            "profile",
            str(tmp_path / "artifact"),
            "--json",
            "--progress",
            "tty",
            "--color",
            "never",
        ),
        services=_TtyCliServices(),
    )

    assert status == 0
    _assert_stable_tty_output(
        payload,
        expected_statuses=(
            "INFO pyamplicol.tty: profile before",
            "WARNING pyamplicol.tty: profile during load",
            "INFO pyamplicol.tty: profile between dashboards",
            "INFO pyamplicol.tty: profile after",
        ),
        dashboard_text="Profiling runtime",
    )


def test_typed_config_entries_preserve_complete_process_set_behavior() -> None:
    config = RunConfig(
        action="generate",
        process=ProcessConfig(
            entries=(
                ProcessEntry("d d~ > z g", "ddbar_zg"),
                ProcessEntry("u u~ > z g"),
            )
        ),
    )

    processes = _process_set(config)

    assert tuple(request.expression for request in processes.requests) == (
        "d d~ > z g",
        "u u~ > z g",
    )
    assert processes.requests[0].name == "ddbar_zg"
    assert processes.requests[1].name == "u_ubar_to_z_g"


def test_process_output_loading_emits_visible_timed_progress(tmp_path: Path) -> None:
    events: list[object] = []
    progress = CallbackProgressSink(events.append)

    result = _load_process_output(tmp_path / "artifact", progress, lambda: "loaded")

    assert result == "loaded"
    assert isinstance(events[0], ProgressStart)
    assert str((tmp_path / "artifact").resolve()) in events[0].description
    assert isinstance(events[-1], ProgressEnd)
    assert events[-1].success is True
    assert events[-1].elapsed_seconds is not None


def test_cli_dispatches_protocol_and_keeps_json_on_stdout(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    services = _Services()
    sys.modules.pop("symbolica", None)
    status = run_cli(
        (
            "generate",
            "d d~ > z g",
            str(tmp_path / "artifact"),
            "--workers",
            "3",
            "--json",
            "--progress",
            "off",
        ),
        services=services,
        stdout=stdout,
        stderr=stderr,
    )
    assert status == 0
    assert json.loads(stdout.getvalue()) == {
        "action": "generate",
        "requests": ["d d~ > z g"],
        "workers": 3,
    }
    assert stderr.getvalue() == ""
    assert services.config is not None
    assert "symbolica" not in sys.modules


def test_group_processes_command_calls_the_public_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, dict[str, object]]] = []

    def group_processes(artifact: Path, **options: object) -> dict[str, object]:
        calls.append((artifact, options))
        return {"artifact": str(artifact), "merged": 0}

    module = types.ModuleType("pyamplicol.api.process_groups")
    module.group_processes = group_processes  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyamplicol.api.process_groups", module)
    stdout = io.StringIO()
    status = run_cli(
        (
            "group-processes",
            str(tmp_path / "artifact"),
            "--only-unit-multiplier",
            "--json",
            "--progress",
            "off",
        ),
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert status == 0
    assert calls == [
        (
            (tmp_path / "artifact").resolve(),
            {
                "analysis": "numerical",
                "only_unit_multiplier": True,
                "consider_charge_conjugation": True,
            },
        )
    ]
    assert json.loads(stdout.getvalue())["merged"] == 0


def test_cli_defaults_to_a_colored_table_for_structured_results(
    tmp_path: Path,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    status = run_cli(
        (
            "generate",
            "d d~ > z g",
            str(tmp_path / "artifact"),
            "--workers",
            "3",
            "--color",
            "always",
            "--progress",
            "off",
        ),
        services=_Services(),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0
    output = stdout.getvalue()
    assert "pyAmpliCol Result" in output
    assert "workers" in output
    assert "\x1b[" in output
    with pytest.raises(json.JSONDecodeError):
        json.loads(output)
    assert stderr.getvalue() == ""


def test_cli_failures_write_only_diagnostics_to_stderr(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    status = run_cli(
        (
            "generate",
            "d d~ > z g",
            str(tmp_path / "artifact"),
            "--json",
        ),
        services=_Services(fail=True),
        stdout=stdout,
        stderr=stderr,
    )
    assert status == 2
    assert stdout.getvalue() == ""
    assert "generation rejected" in stderr.getvalue()


def test_profile_json_is_stdout_clean_and_uses_benchmark_service(
    tmp_path: Path,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    services = _ProfileServices()

    status = run_cli(
        (
            "profile",
            str(tmp_path / "artifact"),
            "--process",
            "d d~ > z g",
            "--target-runtime",
            "0.1",
            "--batch-size",
            "4",
            "--minimum-samples",
            "2",
            "--json",
            "--progress",
            "log",
        ),
        services=services,
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0
    payload = json.loads(stdout.getvalue())
    assert payload["process_id"] == "d_dbar_to_z_g"
    assert payload["repetitions_per_sample"] == 3
    assert "Runtime Profile" not in stdout.getvalue()
    assert "\x1b[" not in stdout.getvalue()
    assert "Profiling runtime" in stderr.getvalue()
    assert services.config is not None
    assert services.config.action == "benchmark"
    assert services.config.evaluation.process == "d d~ > z g"


def test_profile_interrupted_before_sampling_exits_without_traceback(
    tmp_path: Path,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = run_cli(
        ("profile", str(tmp_path / "artifact"), "--progress", "off"),
        services=_InterruptedProfileServices(),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 130
    assert stdout.getvalue() == ""
    assert "interrupted before a complete result was available" in stderr.getvalue()


def test_partial_profile_prints_result_and_exits_as_interrupted(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = run_cli(
        (
            "profile",
            str(tmp_path / "artifact"),
            "--json",
            "--progress",
            "off",
        ),
        services=_PartialProfileServices(),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 130
    assert json.loads(stdout.getvalue())["interrupted"] is True
    assert stderr.getvalue() == ""


def test_inspect_cli_lists_artifact_processes_as_json() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    artifact = ROOT / "src/pyamplicol/assets/selftest/portable-64le/artifact"

    status = run_cli(
        ("inspect", str(artifact), "--json", "--progress", "off"),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0
    payload = json.loads(stdout.getvalue())
    assert payload["kind"] == "pyamplicol-artifact-inspection"
    assert payload["default_process_id"] == "d_dbar_to_z"
    assert payload["processes"][0]["expression"] == "d d~ > z"
    assert stderr.getvalue() == ""


def _inspection_with_alias() -> object:
    from pyamplicol.artifacts import inspect_artifact
    from pyamplicol.artifacts.inspection import ArtifactAliasInspection

    artifact = ROOT / "src/pyamplicol/assets/selftest/portable-64le/artifact"
    inspection = inspect_artifact(artifact)
    representative = inspection.processes[0]
    alias = ArtifactAliasInspection(
        id="stable-alias",
        expression="alias exact expression",
        representative_id=representative.id,
        external_pdgs=representative.external_pdgs,
    )
    return replace(
        inspection,
        processes=(replace(representative, aliases=(alias,)),),
    )


@pytest.mark.parametrize(
    "selector",
    ("d_dbar_to_z", "stable-alias", "d d~ > z"),
)
def test_inspect_process_defaults_to_compact_metadata_without_runtime(
    selector: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol.api as api_module
    import pyamplicol.artifacts as artifact_module

    inspection = _inspection_with_alias()

    class ForbiddenRuntime:
        @classmethod
        def load(cls, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("compact inspection loaded Runtime.physics")

    monkeypatch.setattr(artifact_module, "inspect_artifact", lambda _path: inspection)
    monkeypatch.setattr(api_module, "Runtime", ForbiddenRuntime)
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = run_cli(
        (
            "inspect",
            str(tmp_path / "artifact"),
            "--process",
            selector,
            "--json",
            "--progress",
            "off",
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0
    payload = json.loads(stdout.getvalue())
    assert payload["kind"] == "pyamplicol-artifact-inspection"
    assert [process["id"] for process in payload["processes"]] == ["d_dbar_to_z"]
    assert stderr.getvalue() == ""


# Rusticol resolves full-physics selectors, including side permutations and
# non-generated process-group members absent from the compact inventory.
@pytest.mark.parametrize("selector", ("stable-alias", "u u~ > a a"))
def test_inspect_full_physics_is_the_only_runtime_materialization_path(
    selector: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol.api as api_module
    import pyamplicol.artifacts as artifact_module

    inspection = _inspection_with_alias()
    loads: list[tuple[Path, str | None]] = []
    dense_physics = {"kind": "dense-legacy-physics"}

    class DenseRuntime:
        physics = dense_physics

        @classmethod
        def load(
            cls,
            artifact: Path,
            *,
            process: str | None = None,
        ) -> DenseRuntime:
            loads.append((artifact, process))
            return cls()

    monkeypatch.setattr(artifact_module, "inspect_artifact", lambda _path: inspection)
    monkeypatch.setattr(api_module, "Runtime", DenseRuntime)
    artifact = tmp_path / "artifact"
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = run_cli(
        (
            "inspect",
            str(artifact),
            "--process",
            selector,
            "--full-physics",
            "--json",
            "--progress",
            "off",
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0
    assert json.loads(stdout.getvalue()) == dense_physics
    assert loads == [(artifact.resolve(), selector)]
    assert stderr.getvalue() == ""


def test_inspect_process_keeps_only_the_selected_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol.artifacts as artifact_module
    from pyamplicol.artifacts.inspection import (
        ArtifactProcessGroupInspection,
        ArtifactProcessGroupMemberInspection,
        ArtifactProcessGroupsInspection,
    )

    def group(identifier: str, process: str) -> ArtifactProcessGroupInspection:
        member = ArtifactProcessGroupMemberInspection(
            id=identifier,
            process=process,
            pdgs=(1, -1, 23),
            initial_count=2,
            factor="1",
            permutation=(0, 1, 2),
            conjugated=False,
            basis="identity",
            generated=True,
        )
        return ArtifactProcessGroupInspection(identifier, process, (member,))

    inspection = replace(
        _inspection_with_alias(),
        process_groups=ArtifactProcessGroupsInspection(
            stored=True,
            groups=(group("d_dbar_to_z", "d d~ > z"), group("u_ubar_to_z", "u u~ > z")),
        ),
    )
    monkeypatch.setattr(artifact_module, "inspect_artifact", lambda _path: inspection)
    stdout = io.StringIO()

    status = run_cli(
        (
            "inspect",
            str(tmp_path / "artifact"),
            "--process",
            "stable-alias",
            "--json",
            "--progress",
            "off",
        ),
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert status == 0
    groups = json.loads(stdout.getvalue())["process_groups"]["groups"]
    assert [group["representative_id"] for group in groups] == ["d_dbar_to_z"]


def test_inspect_process_names_the_group_of_a_routed_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol.artifacts as artifact_module
    from pyamplicol.artifacts.inspection import (
        ArtifactProcessGroupInspection,
        ArtifactProcessGroupMemberInspection,
        ArtifactProcessGroupsInspection,
    )

    def member(
        identifier: str, process: str, *, generated: bool
    ) -> ArtifactProcessGroupMemberInspection:
        return ArtifactProcessGroupMemberInspection(
            id=identifier,
            process=process,
            pdgs=(1, -1, 23),
            initial_count=2,
            factor="1",
            permutation=(0, 1, 2),
            conjugated=False,
            basis="flavor-blind" if not generated else "identity",
            generated=generated,
        )

    inspection = replace(
        _inspection_with_alias(),
        process_groups=ArtifactProcessGroupsInspection(
            stored=True,
            groups=(
                ArtifactProcessGroupInspection(
                    "d_dbar_to_z",
                    "d d~ > z",
                    (
                        member("d_dbar_to_z", "d d~ > z", generated=True),
                        member("u_ubar_to_z", "u u~ > z", generated=False),
                    ),
                ),
            ),
        ),
    )
    monkeypatch.setattr(artifact_module, "inspect_artifact", lambda _path: inspection)

    for selector in ("u_ubar_to_z", "u u~ > z", "s s~ > z"):
        stderr = io.StringIO()
        status = run_cli(
            (
                "inspect",
                str(tmp_path / "artifact"),
                "--process",
                selector,
                "--progress",
                "off",
            ),
            stdout=io.StringIO(),
            stderr=stderr,
        )

        assert status == 2
        message = " ".join(stderr.getvalue().split())
        if selector == "s s~ > z":
            assert "did not match a stable representative ID" in message
            continue
        assert (
            "inspect process u_ubar_to_z (u u~ > z) is a routed member of process "
            "group d_dbar_to_z (d d~ > z) with no runtime of its own; --process "
            "d_dbar_to_z lists the group, and --full-physics shows the member's "
            "relation and physics"
        ) in message


def test_inspect_full_physics_adds_the_process_group_of_a_grouped_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol.api as api_module
    import pyamplicol.artifacts.inspection as inspection_module
    from pyamplicol.api import (
        ColorFlow,
        HelicityConfiguration,
        PhysicsReduction,
        ProcessPhysics,
        ReductionGroup,
    )
    from pyamplicol.artifacts.inspection import ArtifactProcessGroupSelection

    physics = ProcessPhysics(
        process_id="u_ubar_to_z",
        process="u u~ > z",
        color_accuracy="lc",
        helicity_coverage="complete",
        color_coverage="complete",
        color_kind="flow",
        structural_zero_helicity_count=0,
        external_particles=(),
        helicities=(
            HelicityConfiguration("h0", 0, (1, -1, 0), True, False, "h0", 1.0),
        ),
        color_flows=(ColorFlow("c0", 0, (1, 2), True, "c0", 1.0),),
        contracted_color_components=(),
        reduction=PhysicsReduction(
            "lc-diagonal", (ReductionGroup("g0", "h0", "c0", ("h0",), ("c0",)),)
        ),
        model_parameters=(),
        selector_capabilities=("helicity", "color_flow"),
    )
    selection = ArtifactProcessGroupSelection(
        process_id="u_ubar_to_z",
        role="routed member",
        representative_id="d_dbar_to_z",
        representative="d d~ > z",
        factor="1",
        permutation=(1, 0, 2),
        conjugated=False,
        basis="flavor-blind",
        warning=None,
        group_size=2,
        routed_count=1,
        conjugated_count=0,
    )
    requests: list[tuple[Path, str, tuple[int, ...] | None]] = []

    class MemberRuntime:
        external_permutation = (1, 0, 2)

        def __init__(self) -> None:
            self.physics = physics

        @classmethod
        def load(cls, _artifact: Path, *, process: str | None = None) -> object:
            return cls()

    def select(
        artifact: Path,
        process_id: str,
        *,
        permutation: tuple[int, ...] | None = None,
    ) -> ArtifactProcessGroupSelection:
        requests.append((artifact, process_id, permutation))
        return selection

    monkeypatch.setattr(api_module, "Runtime", MemberRuntime)
    monkeypatch.setattr(inspection_module, "process_group_selection", select)
    artifact = tmp_path / "artifact"
    stdout = io.StringIO()

    status = run_cli(
        (
            "inspect",
            str(artifact),
            "--process",
            "u~ u > z",
            "--full-physics",
            "--json",
            "--progress",
            "off",
        ),
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert status == 0
    payload = json.loads(stdout.getvalue())
    assert payload["process_id"] == "u_ubar_to_z"
    assert payload["helicities"][0]["id"] == "h0"
    assert payload["process_group"]["representative_id"] == "d_dbar_to_z"
    assert payload["process_group"]["permutation"] == [1, 0, 2]
    assert requests == [(artifact.resolve(), "u_ubar_to_z", (1, 0, 2))]
