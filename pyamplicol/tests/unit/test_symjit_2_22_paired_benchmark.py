# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "developer" / "symjit_2_22_paired_benchmark.py"
SPEC = importlib.util.spec_from_file_location(
    "symjit_2_22_paired_benchmark",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
paired = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = paired
SPEC.loader.exec_module(paired)


def _expected_identity_arguments(
    candidate_revision: str = "c" * 40,
) -> tuple[str, ...]:
    return (
        "--expected-candidate-source-revision",
        candidate_revision,
        "--expected-baseline-native-build-inputs-sha256",
        "a" * 64,
        "--expected-candidate-native-build-inputs-sha256",
        "b" * 64,
        "--expected-baseline-prepared-model-sha256",
        "c" * 64,
        "--expected-candidate-prepared-model-sha256",
        "d" * 64,
    )


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(path: Path, files: dict[str, str]) -> tuple[Path, str]:
    path.mkdir()
    _git(path, "init", "--quiet")
    for relative, content in files.items():
        destination = path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    _git(path, "add", "--all")
    _git(
        path,
        "-c",
        "user.name=paired benchmark test",
        "-c",
        "user.email=paired-benchmark-test@invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    return path.resolve(), _git(path, "rev-parse", "HEAD")


def _fake_driver_source() -> str:
    return r"""#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def canonical(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def addressed(value):
    result = dict(value)
    result["content_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--result-json", type=Path, required=True)
parser.add_argument("--paired-profile-coordination-dir", type=Path, required=True)
parser.add_argument("--paired-profile-role", required=True)
parser.add_argument("--paired-profile-session-id", required=True)
parser.add_argument("--lc-flow-layout", required=True)
arguments, _ = parser.parse_known_args()
source_root = Path(os.environ["PYAMPLICOL_BENCHMARK_SOURCE_ROOT"])
revision = subprocess.run(
    ("git", "rev-parse", "HEAD"),
    cwd=source_root,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
role_root = (
    arguments.paired_profile_coordination_dir
    / arguments.paired_profile_session_id
    / arguments.lc_flow_layout
    / arguments.paired_profile_role
)
plan = [
    {
        "schedule_index": 0,
        "round": 0,
        "mode": "compiled",
        "batch_size": 1,
    }
]
write(
    role_root / "ready.json",
    addressed(
        {
            "kind": "pyamplicol-paired-profile-ready",
            "schema_version": 1,
            "session_id": arguments.paired_profile_session_id,
            "role": arguments.paired_profile_role,
            "layout": arguments.lc_flow_layout,
            "source_revision": revision,
            "ready_at_utc": now(),
            "profile_schedule_plan": plan,
            "profile_schedule_plan_sha256": hashlib.sha256(
                canonical(plan)
            ).hexdigest(),
        }
    ),
)
token_path = role_root / "token-000000.json"
deadline = time.monotonic() + 15.0
while not token_path.is_file():
    if time.monotonic() >= deadline:
        raise RuntimeError("fake driver timed out waiting for token")
    time.sleep(0.01)
token = json.loads(token_path.read_text(encoding="utf-8"))
started = now()
time.sleep(0.01)
finished = now()
write(
    role_root / "completion-000000.json",
    addressed(
        {
            "kind": "pyamplicol-paired-profile-completion",
            "schema_version": 1,
            "session_id": arguments.paired_profile_session_id,
            "role": arguments.paired_profile_role,
            "layout": arguments.lc_flow_layout,
            "token_sha256": token["content_sha256"],
            "schedule_index": 0,
            "round": 0,
            "mode": "compiled",
            "batch_size": 1,
            "worker_started_at_utc": started,
            "worker_finished_at_utc": finished,
            "recorded_at_utc": now(),
            "worker_invocation_sha256": "1" * 64,
            "worker_result_record_sha256": "2" * 64,
        }
    ),
)
write(arguments.result_json, addressed({"role": arguments.paired_profile_role}))
"""


def _campaign_record_fixture() -> dict[str, object]:
    harness = paired._address(
        {
            "kind": paired.HARNESS_IDENTITY_KIND,
            "schema_version": paired.HARNESS_IDENTITY_SCHEMA,
            "candidate_relative_path": "tools/developer/fixture.py",
            "head_blob_sha256": "1" * 64,
            "working_file_sha256": "1" * 64,
            "head_blob_equals_working_file": True,
        }
    )
    orchestrator = paired._address(
        {
            "kind": paired.HARNESS_IDENTITY_KIND,
            "schema_version": paired.HARNESS_IDENTITY_SCHEMA,
            "candidate_relative_path": paired.PAIRED_DRIVER_RELATIVE_PATH,
            "head_blob_sha256": "2" * 64,
            "working_file_sha256": "2" * 64,
            "head_blob_equals_working_file": True,
        }
    )
    return paired._address(
        {
            "kind": paired.CAMPAIGN_KIND,
            "schema_version": paired.CAMPAIGN_SCHEMA,
            "complete": True,
            "session_id": "binding-fixture",
            "orchestrator": orchestrator,
            "harness": harness,
            "roles": {
                role: {
                    "source_revision": character * 40,
                    "python_resolved_target": f"/fixture/{role}/python",
                    "python_sha256": character * 64,
                    "prepared_model": f"/fixture/{role}/model.pack",
                    "prepared_model_sha256": character.upper() * 64,
                }
                for role, character in (("baseline", "a"), ("candidate", "b"))
            },
            "configuration": {
                "authoritative_modes": ["compiled", "recurrence"],
                "batch_sizes": [1, 128, 1024],
            },
            "layouts": {
                layout: {
                    "layout": layout,
                    "captures": {
                        role: {"sha256": character * 64}
                        for role, character in (
                            ("baseline", "c"),
                            ("candidate", "d"),
                        )
                    },
                }
                for layout in paired.LAYOUTS
            },
        }
    )


def test_parser_defaults_are_gate_compatible_and_eager_opt_out_is_explicit() -> None:
    required = (
        "--baseline-python",
        "baseline-python",
        "--baseline-source-root",
        "baseline-source",
        "--baseline-prepared-model",
        "baseline-model",
        "--candidate-python",
        "candidate-python",
        "--candidate-source-root",
        "candidate-source",
        "--candidate-prepared-model",
        "candidate-model",
        *_expected_identity_arguments(),
        "--output-root",
        "output",
        "--result-json",
        "result.json",
        "--session-id",
        "session",
    )

    defaults = paired.parser().parse_args(required)
    diagnostic = paired.parser().parse_args((*required, "--no-eager"))

    assert defaults.include_eager is True
    assert defaults.warmup_runs == paired.MINIMUM_WARMUPS == 2
    assert defaults.subprocess_samples == paired.MINIMUM_SUBPROCESS_SAMPLES == 7
    assert (
        defaults.generation_timeout
        == paired.MINIMUM_GENERATION_TIMEOUT_SECONDS
        == 10_800.0
    )
    assert (
        defaults.coordination_timeout
        == paired.MINIMUM_COORDINATION_TIMEOUT_SECONDS
        == 43_200.0
    )
    assert diagnostic.include_eager is False

    defaults.generation_timeout = 3_600.0
    with pytest.raises(paired.CampaignError, match="10800 seconds"):
        paired.run(defaults)


def test_campaign_rejects_more_than_exactly_two_warmups() -> None:
    arguments = paired.parser().parse_args(
        (
            "--baseline-python",
            "baseline-python",
            "--baseline-source-root",
            "baseline-source",
            "--baseline-prepared-model",
            "baseline-model",
            "--candidate-python",
            "candidate-python",
            "--candidate-source-root",
            "candidate-source",
            "--candidate-prepared-model",
            "candidate-model",
            *_expected_identity_arguments(),
            "--output-root",
            "output",
            "--result-json",
            "result.json",
            "--session-id",
            "session",
            "--warmup-runs",
            "3",
        )
    )

    with pytest.raises(paired.CampaignError, match="exactly two warmups"):
        paired.run(arguments)


def test_campaign_rejects_eight_outer_subprocess_pairs() -> None:
    arguments = paired.parser().parse_args(
        (
            "--baseline-python",
            "baseline-python",
            "--baseline-source-root",
            "baseline-source",
            "--baseline-prepared-model",
            "baseline-model",
            "--candidate-python",
            "candidate-python",
            "--candidate-source-root",
            "candidate-source",
            "--candidate-prepared-model",
            "candidate-model",
            *_expected_identity_arguments(),
            "--output-root",
            "output",
            "--result-json",
            "result.json",
            "--session-id",
            "session",
            "--subprocess-samples",
            "8",
        )
    )

    with pytest.raises(
        paired.CampaignError,
        match="exactly seven independent subprocess pairs",
    ):
        paired.run(arguments)


@pytest.mark.skipif(os.name != "posix", reason="executable symlinks require POSIX")
def test_executable_file_preserves_lexical_venv_symlink(tmp_path: Path) -> None:
    target = tmp_path / "python-real"
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    link = tmp_path / "venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)

    lexical, resolved = paired._executable_file(link, label="fixture Python")

    assert lexical == link.absolute()
    assert lexical.is_symlink()
    assert resolved == target.resolve()


def test_head_bound_harness_identity_records_head_and_working_bytes(
    tmp_path: Path,
) -> None:
    source = "print('authenticated harness')\n"
    repository, _ = _repository(
        tmp_path / "repository",
        {"tools/developer/harness.py": source},
    )

    resolved, identity = paired._head_bound_file_identity(
        repository / "tools" / "developer" / "harness.py",
        source_root=repository,
        label="benchmark harness",
    )

    expected_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert resolved == repository / "tools" / "developer" / "harness.py"
    assert identity == {
        "kind": paired.HARNESS_IDENTITY_KIND,
        "schema_version": paired.HARNESS_IDENTITY_SCHEMA,
        "candidate_relative_path": "tools/developer/harness.py",
        "head_blob_sha256": expected_sha256,
        "working_file_sha256": expected_sha256,
        "head_blob_equals_working_file": True,
        "content_sha256": identity["content_sha256"],
    }
    unsigned = dict(identity)
    content_sha256 = unsigned.pop("content_sha256")
    assert content_sha256 == paired._sha256(unsigned)


def test_source_revision_rejects_nested_repository_root(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path / "repository", {"tracked": "value\n"})
    nested = repository / "nested"
    nested.mkdir()

    with pytest.raises(
        paired.CampaignError,
        match="not the exact Git repository root",
    ):
        paired._source_revision(nested, role="candidate")


def test_head_bound_file_rejects_untracked_and_modified_harness(
    tmp_path: Path,
) -> None:
    repository, _ = _repository(
        tmp_path / "repository",
        {"tracked.py": "print('tracked')\n"},
    )
    tracked = repository / "tracked.py"
    untracked = repository / "untracked.py"
    untracked.write_text("print('untracked')\n", encoding="utf-8")

    with pytest.raises(paired.CampaignError, match="is not tracked"):
        paired._head_bound_file(
            untracked,
            source_root=repository,
            label="benchmark harness",
        )

    tracked.write_text("print('modified')\n", encoding="utf-8")
    with pytest.raises(paired.CampaignError, match="not byte-identical"):
        paired._head_bound_file(
            tracked,
            source_root=repository,
            label="benchmark harness",
        )


@pytest.mark.parametrize(
    "field_path",
    [
        ("harness", "candidate_relative_path"),
        ("orchestrator", "working_file_sha256"),
        ("roles", "baseline", "source_revision"),
        ("roles", "candidate", "source_revision"),
        ("roles", "baseline", "python_resolved_target"),
        ("roles", "baseline", "python_sha256"),
        ("roles", "candidate", "python_resolved_target"),
        ("roles", "candidate", "python_sha256"),
        ("roles", "baseline", "prepared_model"),
        ("roles", "baseline", "prepared_model_sha256"),
        ("roles", "candidate", "prepared_model"),
        ("roles", "candidate", "prepared_model_sha256"),
        ("layouts", "topology-replay", "captures", "baseline", "sha256"),
        ("layouts", "topology-replay", "captures", "candidate", "sha256"),
        ("layouts", "all-flow-union", "captures", "baseline", "sha256"),
        ("layouts", "all-flow-union", "captures", "candidate", "sha256"),
        ("session_id",),
        ("configuration", "batch_sizes"),
        ("layouts", "topology-replay", "layout"),
        ("layouts", "all-flow-union", "layout"),
    ],
)
def test_outer_campaign_address_binds_campaign_inputs_and_captures(
    tmp_path: Path,
    field_path: tuple[str, ...],
) -> None:
    tampered = copy.deepcopy(_campaign_record_fixture())
    owner = tampered
    for field in field_path[:-1]:
        child = owner[field]
        assert isinstance(child, dict)
        owner = child
    owner[field_path[-1]] = "tampered"
    path = tmp_path / "campaign.json"
    path.write_bytes(paired._canonical_json_bytes(tampered) + b"\n")

    with pytest.raises(paired.CampaignError, match="is not content-addressed"):
        paired._read_addressed(path, label="campaign")


def test_nested_harness_tamper_is_rejected_even_if_outer_address_is_rewritten(
    tmp_path: Path,
) -> None:
    tampered = copy.deepcopy(_campaign_record_fixture())
    harness = tampered["harness"]
    assert isinstance(harness, dict)
    harness["working_file_sha256"] = "e" * 64
    unsigned = dict(tampered)
    unsigned.pop("content_sha256")
    tampered["content_sha256"] = paired._sha256(unsigned)
    path = tmp_path / "campaign.json"
    path.write_bytes(paired._canonical_json_bytes(tampered) + b"\n")

    with pytest.raises(
        paired.CampaignError,
        match=r"campaign\.harness is not content-addressed",
    ):
        paired._read_addressed(path, label="campaign")


def test_group_signal_falls_back_to_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []

    class Process:
        pid = 12345

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def send_signal(signal_number: int) -> None:
            observed.append(signal_number)

    def inaccessible_group(_group: int, _signal: int) -> None:
        raise PermissionError

    monkeypatch.setattr(paired.os, "killpg", inaccessible_group)
    paired._signal_capture_group(Process(), signal.SIGTERM)

    assert observed == [signal.SIGTERM]


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "killpg"),
    reason="process-group cleanup requires POSIX",
)
def test_terminate_cleans_up_capture_child(tmp_path: Path) -> None:
    child_ready = tmp_path / "child-ready"
    child_stopped = tmp_path / "child-stopped"
    child_pid_path = tmp_path / "child-pid"
    driver = tmp_path / "driver.py"
    driver.write_text(
        """\
import subprocess
import sys
import time
from pathlib import Path

ready, stopped, pid_path = map(Path, sys.argv[1:])
child_source = '''\
import signal
import sys
import time
from pathlib import Path

ready = Path(sys.argv[1])
stopped = Path(sys.argv[2])

def stop(_signal, _frame):
    stopped.write_text("stopped", encoding="utf-8")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
ready.write_text("ready", encoding="utf-8")
while True:
    time.sleep(1)
'''
child = subprocess.Popen(
    [sys.executable, "-c", child_source, str(ready), str(stopped)]
)
while not ready.is_file():
    time.sleep(0.01)
pid_path.write_text(str(child.pid), encoding="ascii")
while True:
    time.sleep(1)
""",
        encoding="utf-8",
    )
    log_path = tmp_path / "driver.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = paired._start_capture(
            (
                sys.executable,
                str(driver),
                str(child_ready),
                str(child_stopped),
                str(child_pid_path),
            ),
            environment=os.environ,
            log=log,
        )
        deadline = time.monotonic() + 5.0
        child_pid: int | None = None
        while child_pid is None:
            if process.poll() is not None:
                pytest.fail(f"driver exited early with status {process.returncode}")
            if time.monotonic() >= deadline:
                pytest.fail("timed out waiting for capture child")
            raw_pid = (
                child_pid_path.read_text(encoding="ascii")
                if child_pid_path.is_file()
                else ""
            )
            if raw_pid.isdecimal():
                child_pid = int(raw_pid)
            time.sleep(0.01)
        paired._terminate({"driver": process}, grace_seconds=2.0)

    deadline = time.monotonic() + 2.0
    while not child_stopped.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    child_alive = True
    while child_alive and time.monotonic() < deadline:
        status = subprocess.run(
            ("ps", "-o", "stat=", "-p", str(child_pid)),
            check=False,
            capture_output=True,
            text=True,
        )
        child_alive = (
            status.returncode == 0
            and bool(status.stdout.strip())
            and not status.stdout.lstrip().startswith("Z")
        )
        if child_alive:
            time.sleep(0.01)
    if child_alive:
        os.kill(child_pid, signal.SIGKILL)

    assert process.poll() is not None
    assert child_stopped.read_text(encoding="utf-8") == "stopped"
    assert not child_alive


@pytest.mark.skipif(os.name != "posix", reason="venv symlinks require POSIX")
def test_fake_driver_campaign_preserves_pairing_and_executable_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_root, baseline_revision = _repository(
        tmp_path / "baseline",
        {"identity": "baseline\n"},
    )
    candidate_root, candidate_revision = _repository(
        tmp_path / "candidate",
        {
            "identity": "candidate\n",
            "fake_driver.py": _fake_driver_source(),
            paired.PAIRED_DRIVER_RELATIVE_PATH: SCRIPT.read_text(encoding="utf-8"),
        },
    )
    interpreter_target = Path(sys.executable).resolve()
    python_paths: dict[str, Path] = {}
    for role in paired.ROLES:
        python_path = tmp_path / f"{role}-venv" / "bin" / "python"
        python_path.parent.mkdir(parents=True)
        python_path.symlink_to(interpreter_target)
        python_paths[role] = python_path
    prepared_models: dict[str, Path] = {}
    for role in paired.ROLES:
        prepared_model = tmp_path / f"{role}.pack"
        prepared_model.write_bytes(role.encode("ascii"))
        prepared_models[role] = prepared_model

    monkeypatch.setattr(paired, "BASELINE_REVISION", baseline_revision)
    monkeypatch.setattr(paired, "LAYOUTS", ("topology-replay",))
    monkeypatch.setattr(paired, "AUTHORITATIVE_MODES", ("compiled",))
    monkeypatch.setattr(paired, "BATCH_SIZES", (1,))
    monkeypatch.setattr(paired, "MINIMUM_SUBPROCESS_SAMPLES", 1)
    monkeypatch.setattr(paired, "MINIMUM_WARMUPS", 1)
    monkeypatch.setattr(paired, "MINIMUM_TARGET_RUNTIME_SECONDS", 0.001)
    monkeypatch.setattr(paired, "MINIMUM_GENERATION_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(paired, "MINIMUM_COORDINATION_TIMEOUT_SECONDS", 15.0)
    output_root = tmp_path / "campaign-output"
    result_json = tmp_path / "campaign.json"
    arguments = argparse.Namespace(
        baseline_python=python_paths["baseline"],
        baseline_source_root=baseline_root,
        baseline_prepared_model=prepared_models["baseline"],
        candidate_python=python_paths["candidate"],
        candidate_source_root=candidate_root,
        candidate_prepared_model=prepared_models["candidate"],
        expected_candidate_source_revision=candidate_revision,
        expected_baseline_native_build_inputs_sha256="a" * 64,
        expected_candidate_native_build_inputs_sha256="b" * 64,
        expected_baseline_prepared_model_sha256=hashlib.sha256(
            b"baseline"
        ).hexdigest(),
        expected_candidate_prepared_model_sha256=hashlib.sha256(
            b"candidate"
        ).hexdigest(),
        harness=candidate_root / "fake_driver.py",
        output_root=output_root,
        result_json=result_json,
        session_id="fake-e2e",
        include_eager=False,
        process_expression="d d~ > z + 6*g",
        target_runtime=0.001,
        minimum_samples=1,
        subprocess_samples=1,
        warmup_runs=1,
        validation_samples=1,
        point_tile_size=1,
        generation_timeout=5.0,
        profile_timeout=5.0,
        coordination_timeout=15.0,
        topology_color_flow="1",
        topology_helicity="1",
        union_color_flow="1",
        union_helicity="1",
    )

    result = paired.run(arguments)

    assert result["complete"] is True
    assert result_json.is_file()
    assert paired._read_addressed(result_json, label="campaign") == result
    assert candidate_revision == result["roles"]["candidate"]["source_revision"]
    harness = result["harness"]
    assert harness["candidate_relative_path"] == "fake_driver.py"
    assert harness["head_blob_sha256"] == harness["working_file_sha256"]
    assert harness["head_blob_equals_working_file"] is True
    paired._validate_addressed_mapping(
        harness,
        label="benchmark harness identity",
    )
    for role in paired.ROLES:
        identity = result["roles"][role]
        assert identity["python"] == str(python_paths[role].absolute())
        assert identity["python_resolved_target"] == str(interpreter_target)
    layout = result["layouts"]["topology-replay"]
    assert len(layout["pairs"]) == 1
    assert layout["pairs"][0]["role_order"] == ["baseline", "candidate"]
    assert result["configuration"]["watchdog"] == {
        "required": True,
        "report_kind": paired.WATCHDOG_REPORT_KIND,
        "report_schema_version": paired.WATCHDOG_REPORT_SCHEMA,
        "limit_bytes": paired.WATCHDOG_LIMIT_BYTES,
        "scope": paired.WATCHDOG_SCOPE,
        "binding": paired.WATCHDOG_BINDING,
    }
