#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run the guarded recurrence-generation baseline/candidate ladder.

This developer-only orchestrator deliberately delegates artifact generation
and runtime validation to ``recurrence_z6g_benchmark.py``.  Every scheduled
baseline/candidate capture is a separate process-tree invocation of
``memory_watchdog.py --limit-gib 30`` with unique output, temporary, bytecode,
and XDG cache roots.

The default generation ladder covers n=2 through n=9 for both LC layouts with
three repetitions.  Baseline-first and candidate-first pairs alternate.  Add
``--runtime-n 6 --runtime-n 7`` to perform the approved runtime validation
cells with the harness defaults made explicit here: batches 1/128/1024,
seven subprocesses, seven native blocks, two warm-ups, and a five-second
measurement target.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_PARENT = ROOT / ".artifacts" / "recurrence-generation-opt"
CAMPAIGN_PARENT = ALLOWED_OUTPUT_PARENT / "benchmark-campaigns"
HARNESS_RELATIVE_PATH = Path("tools/developer/recurrence_z6g_benchmark.py")
WATCHDOG_RELATIVE_PATH = Path("tools/ci/memory_watchdog.py")
HARNESS_SOURCE_CHECKOUT_ENV = "PYAMPLICOL_RECURRENCE_Z6G_SOURCE_CHECKOUT"

CAMPAIGN_KIND = "pyamplicol-recurrence-generation-ab-ladder"
CAMPAIGN_SCHEMA_VERSION = 1
HARNESS_KIND = "pyamplicol-recurrence-z6g-benchmark"
HARNESS_SCHEMA_VERSION = 6
EXECUTION_KIND = "pyamplicol-runtime-recurrence-execution"
EXECUTION_SCHEMA_VERSION = 3
RUNTIME_ACCEPTANCE_KIND = "pyamplicol-recurrence-runtime-acceptance"
RUNTIME_ACCEPTANCE_SCHEMA_VERSION = 1

LAYOUTS = ("topology-replay", "all-flow-union")
DEFAULT_MULTIPLICITIES = tuple(range(2, 10))
DEFAULT_BATCH_SIZES = (1, 128, 1024)
WATCHDOG_LIMIT_GIB = 30.0
GIB = 1024**3

_CAMPAIGN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WATCHDOG_FINISHED_RE = re.compile(
    r"memory-watchdog: command finished"
    r" exit=(?P<exit>\d+)"
    r" peak_rss=(?P<rss>\d+(?:[.]\d+)?) GiB"
    r" peak_physical_footprint="
    r"(?P<physical>unavailable|\d+(?:[.]\d+)? GiB)"
    r" peak_guard=(?P<guard>\d+(?:[.]\d+)?) GiB"
    r" peak_processes=(?P<processes>\d+)"
)
_WATCHDOG_LIMIT_RE = re.compile(
    r"memory-watchdog: memory limit exceeded"
    r" reason=(?P<reason>[^ ]+)"
    r" observed=(?P<observed>\d+(?:[.]\d+)?) GiB"
    r" limit=(?P<limit>\d+(?:[.]\d+)?) GiB"
    r" rss=(?P<rss>\d+(?:[.]\d+)?) GiB"
    r" physical_footprint="
    r"(?P<physical>unavailable|\d+(?:[.]\d+)? GiB)"
    r" processes=(?P<processes>\d+);"
)
_HARNESS_GENERATION_TIMEOUT_RE = re.compile(
    r"^recurrence-z6g-benchmark: recurrence generation worker exceeded "
    r"(?P<seconds>\d+(?:[.]\d+)?) seconds$",
    re.MULTILINE,
)


class LadderError(RuntimeError):
    """Raised when a campaign or one of its captured results is invalid."""


@dataclass(frozen=True, slots=True)
class Variant:
    """One independently installed baseline or candidate environment."""

    name: str
    checkout: Path
    python: Path
    pythonpath: Path | None = None
    prepared_model: Path | None = None


@dataclass(frozen=True, slots=True)
class SampleSpec:
    """One outer watchdog-guarded harness capture."""

    sequence_index: int
    pair_index: int
    order_in_pair: int
    repetition: int
    multiplicity: int
    layout: str
    variant_name: str
    runtime_enabled: bool

    @property
    def process_expression(self) -> str:
        return process_expression(self.multiplicity)

    @property
    def sample_id(self) -> str:
        layout = self.layout.replace("-", "_")
        return (
            f"{self.sequence_index:04d}-p{self.pair_index:03d}"
            f"-r{self.repetition + 1:02d}-{self.variant_name}"
            f"-n{self.multiplicity:02d}-{layout}"
        )


@dataclass(frozen=True, slots=True)
class RunnerSettings:
    """Stable harness settings shared by every campaign sample."""

    validation_samples: int = 10
    point_tile_size: int = 1024
    jit_optimization_level: int = 2
    profile_timeout_seconds: float = 300.0
    minimum_samples: int = 7
    subprocess_samples: int = 7
    warmup_runs: int = 2
    target_runtime_seconds: float = 5.0
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES
    allow_diagnostic_incomplete_success: bool = False


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    """Minimal outcome of an outer watchdog process."""

    exit_code: int | None
    timed_out: bool
    wall_seconds: float
    error: str | None


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def process_expression(multiplicity: int) -> str:
    """Return exactly ``d d~ > Z + (n-1)*g`` in CLI process syntax."""

    if isinstance(multiplicity, bool) or multiplicity < 1:
        raise LadderError("process multiplicity n must be a positive integer")
    return " ".join(("d", "d~", ">", "Z", *(("g",) * (multiplicity - 1))))


def generation_timeout_seconds(multiplicity: int, layout: str) -> float:
    """Return the approved per-cell generation timeout."""

    if layout not in LAYOUTS:
        raise LadderError(f"unsupported LC flow layout: {layout}")
    if multiplicity < 1:
        raise LadderError("process multiplicity n must be positive")
    if multiplicity <= 4:
        return 5.0 * 60.0
    if multiplicity <= 7:
        return 15.0 * 60.0
    if multiplicity == 8:
        return (1.0 if layout == "topology-replay" else 2.0) * 60.0 * 60.0
    if multiplicity == 9:
        return (2.0 if layout == "topology-replay" else 6.0) * 60.0 * 60.0
    raise LadderError(
        "the approved timeout policy covers only process multiplicities n=1..9"
    )


def _outer_timeout_seconds(
    spec: SampleSpec,
    settings: RunnerSettings,
) -> float:
    generation = generation_timeout_seconds(spec.multiplicity, spec.layout)
    if not spec.runtime_enabled:
        return generation + 120.0
    profile_budget = (
        settings.profile_timeout_seconds
        * settings.subprocess_samples
        * len(settings.batch_sizes)
    )
    return generation + profile_budget + 300.0


def build_schedule(
    multiplicities: Sequence[int],
    layouts: Sequence[str],
    repetitions: int,
    runtime_multiplicities: frozenset[int],
) -> list[SampleSpec]:
    """Build deterministic pairs with alternating A/B order."""

    if repetitions <= 0:
        raise LadderError("repetitions must be positive")
    normalized_multiplicities = tuple(dict.fromkeys(multiplicities))
    normalized_layouts = tuple(dict.fromkeys(layouts))
    if not normalized_multiplicities:
        raise LadderError("at least one process multiplicity is required")
    if not normalized_layouts:
        raise LadderError("at least one flow layout is required")
    for multiplicity in normalized_multiplicities:
        generation_timeout_seconds(multiplicity, "topology-replay")
    for layout in normalized_layouts:
        if layout not in LAYOUTS:
            raise LadderError(f"unsupported LC flow layout: {layout}")
    if not runtime_multiplicities.issubset(normalized_multiplicities):
        raise LadderError("--runtime-n values must also be selected with --n")

    schedule: list[SampleSpec] = []
    pair_index = 0
    sequence_index = 0
    for multiplicity in normalized_multiplicities:
        for layout in normalized_layouts:
            for repetition in range(repetitions):
                order = (
                    ("baseline", "candidate")
                    if pair_index % 2 == 0
                    else ("candidate", "baseline")
                )
                for order_in_pair, variant_name in enumerate(order):
                    schedule.append(
                        SampleSpec(
                            sequence_index=sequence_index,
                            pair_index=pair_index,
                            order_in_pair=order_in_pair,
                            repetition=repetition,
                            multiplicity=multiplicity,
                            layout=layout,
                            variant_name=variant_name,
                            runtime_enabled=(multiplicity in runtime_multiplicities),
                        )
                    )
                    sequence_index += 1
                pair_index += 1
    return schedule


def build_sample_command(
    spec: SampleSpec,
    variant: Variant,
    sample_root: Path,
    settings: RunnerSettings,
) -> list[str]:
    """Construct the exact watchdog and harness command for one sample."""

    # The guard is measurement infrastructure, not part of either measured
    # runtime.  Use the audited watchdog from the campaign checkout for both
    # variants so a frozen baseline remains pristine and benefits from
    # fail-closed platform fixes made after that baseline was captured.
    watchdog = ROOT / WATCHDOG_RELATIVE_PATH
    harness = ROOT / HARNESS_RELATIVE_PATH
    harness_output = sample_root / "harness"
    result_json = sample_root / "harness-result.json"
    generation_timeout = generation_timeout_seconds(
        spec.multiplicity,
        spec.layout,
    )
    harness_command = [
        str(variant.python),
        str(harness),
        "--output-root",
        str(harness_output),
        "--result-json",
        str(result_json),
        "--gluon-count",
        str(max(1, spec.multiplicity - 1)),
        "--process-expression",
        spec.process_expression,
        "--validation-samples",
        str(settings.validation_samples),
        "--point-tile-size",
        str(settings.point_tile_size),
        "--jit-optimization-level",
        str(settings.jit_optimization_level),
        "--mode",
        "recurrence",
        "--lc-flow-layout",
        spec.layout,
        "--generation-timeout",
        f"{generation_timeout:g}",
        "--profile-timeout",
        f"{settings.profile_timeout_seconds:g}",
        "--minimum-samples",
        str(settings.minimum_samples),
        "--subprocess-samples",
        str(settings.subprocess_samples),
        "--warmup-runs",
        str(settings.warmup_runs),
        "--target-runtime",
        f"{settings.target_runtime_seconds:g}",
    ]
    if settings.allow_diagnostic_incomplete_success:
        harness_command.append("--allow-diagnostic-incomplete-success")
    if variant.prepared_model is not None:
        harness_command.extend(("--prepared-model", str(variant.prepared_model)))
    if spec.runtime_enabled:
        for batch_size in settings.batch_sizes:
            harness_command.extend(("--batch-size", str(batch_size)))
    else:
        harness_command.append("--generation-only")

    return [
        str(variant.python),
        str(watchdog),
        "--limit-gib",
        f"{WATCHDOG_LIMIT_GIB:g}",
        "--",
        *harness_command,
    ]


def _sample_environment(
    variant: Variant,
    sample_root: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    cache_root = sample_root / "cold-cache"
    temporary_root = sample_root / "tmp"
    pycache_root = sample_root / "pycache"
    matplotlib_root = cache_root / "matplotlib"
    for path in (cache_root, temporary_root, pycache_root, matplotlib_root):
        path.mkdir(parents=True, exist_ok=False)

    overrides = {
        HARNESS_SOURCE_CHECKOUT_ENV: str(variant.checkout),
        "MPLCONFIGDIR": str(matplotlib_root),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": str(pycache_root),
        "SYMBOLICA_HIDE_BANNER": "1",
        "TEMP": str(temporary_root),
        "TMP": str(temporary_root),
        "TMPDIR": str(temporary_root),
        "XDG_CACHE_HOME": str(cache_root),
    }
    environment = os.environ.copy()
    environment.update(overrides)
    if variant.pythonpath is None:
        environment.pop("PYTHONPATH", None)
    else:
        overrides["PYTHONPATH"] = str(variant.pythonpath)
        environment["PYTHONPATH"] = str(variant.pythonpath)
    return environment, overrides


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()
    process.wait()


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> ProcessOutcome:
    started = time.perf_counter()
    process: subprocess.Popen[bytes] | None = None
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=dict(environment),
                stdout=stdout,
                stderr=stderr,
                start_new_session=(os.name == "posix"),
            )
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process)
                return ProcessOutcome(
                    exit_code=None,
                    timed_out=True,
                    wall_seconds=time.perf_counter() - started,
                    error=f"outer sample timeout after {timeout_seconds:g} seconds",
                )
    except OSError as error:
        if process is not None:
            _terminate_process_tree(process)
        return ProcessOutcome(
            exit_code=None,
            timed_out=False,
            wall_seconds=time.perf_counter() - started,
            error=f"{type(error).__name__}: {error}",
        )
    return ProcessOutcome(
        exit_code=exit_code,
        timed_out=False,
        wall_seconds=time.perf_counter() - started,
        error=None,
    )


def _gib_value(text: str) -> dict[str, float | int]:
    gib = float(text.removesuffix(" GiB"))
    return {
        "gib": gib,
        "bytes_rounded_from_watchdog": round(gib * GIB),
    }


def parse_watchdog_text(text: str) -> dict[str, Any]:
    """Extract the watchdog's terminal process-tree memory observation."""

    finished = list(_WATCHDOG_FINISHED_RE.finditer(text))
    exceeded = list(_WATCHDOG_LIMIT_RE.finditer(text))
    if finished:
        match = finished[-1]
        physical = match.group("physical")
        return {
            "terminal_record": "command-finished",
            "limit_gib": WATCHDOG_LIMIT_GIB,
            "limit_exceeded": False,
            "child_exit_code": int(match.group("exit")),
            "peak_rss": _gib_value(match.group("rss")),
            "peak_physical_footprint": (
                None if physical == "unavailable" else _gib_value(physical)
            ),
            "peak_guard": _gib_value(match.group("guard")),
            "peak_process_count": int(match.group("processes")),
        }
    if exceeded:
        match = exceeded[-1]
        physical = match.group("physical")
        return {
            "terminal_record": "memory-limit-exceeded",
            "limit_gib": float(match.group("limit")),
            "limit_exceeded": True,
            "reason": match.group("reason"),
            "observed_guard": _gib_value(match.group("observed")),
            "rss_at_limit": _gib_value(match.group("rss")),
            "physical_footprint_at_limit": (
                None if physical == "unavailable" else _gib_value(physical)
            ),
            "process_count_at_limit": int(match.group("processes")),
        }
    return {
        "terminal_record": None,
        "limit_gib": WATCHDOG_LIMIT_GIB,
        "limit_exceeded": None,
    }


def parse_watchdog_log(path: Path) -> dict[str, Any]:
    """Parse at most the final MiB of a potentially large stderr log."""

    try:
        with path.open("rb") as stream:
            size = path.stat().st_size
            stream.seek(max(0, size - 1024 * 1024))
            raw = stream.read()
    except OSError as error:
        raise LadderError(f"cannot read watchdog stderr log: {path}") from error
    return parse_watchdog_text(raw.decode("utf-8", errors="replace"))


def parse_harness_generation_timeout(path: Path) -> dict[str, float] | None:
    """Return an exact inner generation-timeout record, if present."""

    try:
        with path.open("rb") as stream:
            size = path.stat().st_size
            stream.seek(max(0, size - 1024 * 1024))
            raw = stream.read()
    except OSError as error:
        raise LadderError(f"cannot read harness stderr log: {path}") from error
    text = raw.decode("utf-8", errors="replace")
    matches = tuple(_HARNESS_GENERATION_TIMEOUT_RE.finditer(text))
    if not matches:
        return None
    if len(matches) != 1:
        raise LadderError("harness stderr contains multiple generation timeouts")
    return {"configured_seconds": float(matches[0].group("seconds"))}


def _json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LadderError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise LadderError(f"{description} must be a JSON object: {path}")
    return value


def _required_mapping(
    value: object,
    *,
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LadderError(f"{description} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise LadderError(f"cannot hash result file: {path}") from error
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (float, int))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _verify_artifact_payload_inventory(
    artifact: Path,
    artifact_identity: Mapping[str, Any],
) -> dict[str, int]:
    """Authenticate every payload used by a generation-only capture."""

    identity_path = artifact_identity.get("path")
    if not isinstance(identity_path, str) or Path(identity_path).resolve() != artifact:
        raise LadderError("generation artifact identity path does not match")
    manifest_identity = _required_mapping(
        artifact_identity.get("manifest"),
        description="generation artifact manifest identity",
    )
    manifest_path = artifact / "artifact.json"
    recorded_manifest_path = manifest_identity.get("resolved_path")
    if (
        not isinstance(recorded_manifest_path, str)
        or Path(recorded_manifest_path).resolve() != manifest_path
        or manifest_identity.get("sha256") != _sha256_file(manifest_path)
        or manifest_identity.get("size_bytes") != manifest_path.stat().st_size
    ):
        raise LadderError("generation artifact manifest identity does not match")
    manifest = _json_object(
        manifest_path,
        description="generation artifact manifest",
    )
    if manifest.get("artifact_id") != artifact_identity.get("artifact_id"):
        raise LadderError("generation artifact ID does not match its manifest")
    raw_payloads = manifest.get("payloads")
    identity_payloads = artifact_identity.get("payloads")
    if not isinstance(raw_payloads, list) or not isinstance(identity_payloads, list):
        raise LadderError("generation artifact payload inventories do not match")
    identity_by_path: dict[str, Mapping[str, Any]] = {}
    for raw_identity_entry in identity_payloads:
        identity_entry = _required_mapping(
            raw_identity_entry,
            description="generation artifact identity payload entry",
        )
        identity_relative = identity_entry.get("path")
        if (
            not isinstance(identity_relative, str)
            or not identity_relative
            or identity_relative in identity_by_path
        ):
            raise LadderError("generation artifact identity payload entry is invalid")
        identity_by_path[identity_relative] = identity_entry
    if len(identity_by_path) != len(raw_payloads):
        raise LadderError("generation artifact payload inventories do not match")
    root = artifact.resolve()
    total_bytes = 0
    seen_paths: set[str] = set()
    for raw_entry in raw_payloads:
        entry = _required_mapping(
            raw_entry,
            description="generation artifact payload entry",
        )
        relative = entry.get("path")
        expected_sha256 = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen_paths
            or not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise LadderError("generation artifact payload entry is invalid")
        seen_paths.add(relative)
        identity_entry = identity_by_path.get(relative)
        if identity_entry is None or any(
            identity_entry.get(key) != entry.get(key)
            for key in ("path", "role", "sha256", "size_bytes")
        ):
            raise LadderError("generation artifact payload inventories do not match")
        if identity_entry.get("process_id") != entry.get("process_id"):
            raise LadderError("generation artifact payload inventories do not match")
        try:
            payload_path = (artifact / relative).resolve(strict=True)
            payload_path.relative_to(root)
        except (OSError, ValueError) as error:
            raise LadderError(
                f"generation artifact payload escapes or is missing: {relative!r}"
            ) from error
        if (
            not payload_path.is_file()
            or payload_path.stat().st_size != expected_size
            or _sha256_file(payload_path) != expected_sha256
        ):
            raise LadderError(
                f"generation artifact payload identity differs: {relative!r}"
            )
        total_bytes += expected_size
    return {
        "payload_count": len(raw_payloads),
        "payload_bytes": total_bytes,
    }


def _generation_only_acceptance(
    result: Mapping[str, Any],
    recurrence: Mapping[str, Any],
    *,
    artifact: Path,
) -> dict[str, Any] | None:
    """Validate a recurrence generation capture independently of runtime lanes."""

    configuration = _required_mapping(
        result.get("configuration"),
        description="benchmark configuration",
    )
    if configuration.get("generation_only") is not True:
        return None
    if result.get("profiles") != {}:
        raise LadderError("generation-only benchmark unexpectedly contains profiles")
    if recurrence.get("mode") != "recurrence":
        raise LadderError("generation-only capture is not recurrence mode")
    if recurrence.get("generation_reused") is not False:
        raise LadderError("generation-only capture did not use a cold generation")
    if not _positive_finite_number(recurrence.get("generation_wall_seconds")):
        raise LadderError("generation-only capture has invalid wall timing")

    signature = _required_mapping(
        recurrence.get("semantic_generation_signature"),
        description="semantic generation signature",
    )
    signature_sha256 = recurrence.get("semantic_generation_signature_sha256")
    validation = _required_mapping(
        signature.get("validation"),
        description="generation validation contract",
    )
    configured_samples = configuration.get("validation_samples")
    if (
        signature.get("kind") != "pyamplicol-benchmark-generation-signature"
        or signature.get("schema_version") != 1
        or signature.get("mode") != "recurrence"
        or signature.get("lc_flow_layout") != configuration.get("lc_flow_layout")
        or signature_sha256 != _canonical_sha256(signature)
        or validation.get("enabled") is not True
        or validation.get("post_build_validation") is not True
        or isinstance(configured_samples, bool)
        or not isinstance(configured_samples, int)
        or configured_samples <= 0
        or validation.get("samples") != configured_samples
    ):
        raise LadderError("generation validation signature is incomplete")

    semantic_identity = _required_mapping(
        recurrence.get("artifact_semantic_identity"),
        description="artifact semantic identity",
    )
    coverage = _required_mapping(
        semantic_identity.get("coverage"),
        description="artifact semantic coverage",
    )
    reduction_coverage = _required_mapping(
        semantic_identity.get("execution_reduction_coverage"),
        description="artifact reduction coverage",
    )
    if (
        recurrence.get("artifact_semantic_identity_sha256")
        != _canonical_sha256(semantic_identity)
        or coverage.get("complete_physical_axes") is not True
        or coverage.get("color") != "complete"
        or coverage.get("helicities") != "complete"
        or reduction_coverage.get("complete") is not True
        or reduction_coverage.get("errors") != []
    ):
        raise LadderError("generation artifact semantic coverage is incomplete")

    worker_evidence = _required_mapping(
        recurrence.get("generation_worker_result_evidence"),
        description="generation worker evidence",
    )
    evidence_without_digest = dict(worker_evidence)
    evidence_sha256 = evidence_without_digest.pop("content_sha256", None)
    worker_payload = _required_mapping(
        worker_evidence.get("payload"),
        description="generation worker evidence payload",
    )
    if (
        worker_evidence.get("kind") != "pyamplicol-preserved-worker-result-evidence"
        or worker_evidence.get("schema_version") != 1
        or evidence_sha256 != _canonical_sha256(evidence_without_digest)
        or worker_payload.get("mode") != "recurrence"
        or worker_payload.get("generation_reused") is not False
        or worker_payload.get("generation_wall_seconds")
        != recurrence.get("generation_wall_seconds")
    ):
        raise LadderError("generation worker evidence failed authentication")

    artifact_identity = _required_mapping(
        recurrence.get("artifact_identity"),
        description="generation artifact identity",
    )
    inventory = _verify_artifact_payload_inventory(artifact, artifact_identity)
    return {
        "kind": "pyamplicol-recurrence-generation-only-acceptance",
        "schema_version": 1,
        "passes": True,
        "post_build_validation_samples": configured_samples,
        "semantic_generation_signature_sha256": signature_sha256,
        "artifact_semantic_identity_sha256": recurrence.get(
            "artifact_semantic_identity_sha256"
        ),
        **inventory,
    }


def _recurrence_runtime_acceptance(
    result: Mapping[str, Any],
    recurrence: Mapping[str, Any],
    *,
    artifact: Path,
) -> dict[str, Any] | None:
    """Authenticate the harness's recurrence-only runtime acceptance record."""

    configuration = _required_mapping(
        result.get("configuration"),
        description="benchmark configuration",
    )
    raw_acceptance = result.get("recurrence_runtime_acceptance")
    if raw_acceptance is None:
        return None
    acceptance = _required_mapping(
        raw_acceptance,
        description="recurrence runtime acceptance",
    )
    without_digest = dict(acceptance)
    content_sha256 = without_digest.pop("content_sha256", None)
    if (
        acceptance.get("kind") != RUNTIME_ACCEPTANCE_KIND
        or acceptance.get("schema_version") != RUNTIME_ACCEPTANCE_SCHEMA_VERSION
        or content_sha256 != _canonical_sha256(without_digest)
    ):
        raise LadderError("recurrence runtime acceptance address is invalid")
    if (
        acceptance.get("accepted") is not True
        or acceptance.get("status") != "accepted"
        or acceptance.get("errors") != []
    ):
        return None

    required = _required_mapping(
        acceptance.get("required"),
        description="recurrence runtime requirements",
    )
    required_exact = {
        "modes": ["recurrence"],
        "generation_only": False,
        "batch_sizes": list(DEFAULT_BATCH_SIZES),
        "target_runtime_seconds": 5.0,
        "warmup_runs": 2,
        "minimum_internal_samples": 7,
        "subprocess_samples_minimum": 7,
        "subprocess_samples_maximum": 21,
        "process_family": "d d~ > Z + (n-1)*g",
        "accepted_n_values": [6, 7],
        "generation_specialized_axes_allowed": False,
        "complete_physical_axes_required": True,
    }
    if any(required.get(key) != value for key, value in required_exact.items()):
        raise LadderError("recurrence runtime requirements are incomplete")

    modes = configuration.get("modes")
    batches = configuration.get("batch_sizes")
    minimum_samples = configuration.get("minimum_samples")
    subprocess_samples = configuration.get("subprocess_samples")
    gluon_count = configuration.get("gluon_count")
    normalized_process = (
        " ".join(result["process"].split()).casefold()
        if isinstance(result.get("process"), str)
        else None
    )
    expected_process = (
        "d d~ > z" + " g" * gluon_count
        if (
            not isinstance(gluon_count, bool)
            and isinstance(gluon_count, int)
            and gluon_count in (5, 6)
        )
        else None
    )
    if (
        modes != ["recurrence"]
        or configuration.get("generation_only") is not False
        or batches != list(DEFAULT_BATCH_SIZES)
        or configuration.get("target_runtime_seconds") != 5.0
        or configuration.get("warmup_runs") != 2
        or isinstance(minimum_samples, bool)
        or not isinstance(minimum_samples, int)
        or minimum_samples < 7
        or isinstance(subprocess_samples, bool)
        or not isinstance(subprocess_samples, int)
        or not 7 <= subprocess_samples <= 21
        or expected_process is None
        or normalized_process != expected_process
        or configuration.get("lc_flow_layout") not in LAYOUTS
        or configuration.get("specialize_flow_at_generation") is not False
    ):
        raise LadderError("recurrence runtime configuration is not authoritative")

    contracts = _required_mapping(
        acceptance.get("contracts"),
        description="recurrence runtime contracts",
    )
    measurement = _required_mapping(
        contracts.get("measurement"),
        description="recurrence runtime measurement contract",
    )
    artifact_semantics = _required_mapping(
        contracts.get("artifact_semantics"),
        description="recurrence runtime artifact semantic contract",
    )
    lane_validation = _required_mapping(
        contracts.get("lane_validation"),
        description="recurrence runtime lane validation",
    )
    worker_rss = _required_mapping(
        contracts.get("worker_rss"),
        description="recurrence runtime worker RSS contract",
    )
    expected_worker_count = subprocess_samples * len(DEFAULT_BATCH_SIZES)
    cold_peak = worker_rss.get("maximum_cold_load_observed_lower_bound_bytes")
    post_peak = worker_rss.get("maximum_post_profile_observed_lower_bound_bytes")
    if (
        measurement.get("passes") is not True
        or artifact_semantics.get("passes") is not True
        or lane_validation.get("passes") is not True
        or lane_validation.get("lane_validation_passes") is not True
        or lane_validation.get("selectors_match") is not True
        or lane_validation.get("fixtures_match") is not True
        or lane_validation.get("pairwise_validation_passes") is not None
        or not isinstance(lane_validation.get("summary_sha256"), str)
        or _SHA256_RE.fullmatch(lane_validation["summary_sha256"]) is None
        or worker_rss.get("passes") is not True
        or worker_rss.get("errors") != []
        or worker_rss.get("worker_count") != expected_worker_count
        or isinstance(cold_peak, bool)
        or not isinstance(cold_peak, int)
        or cold_peak < 0
        or isinstance(post_peak, bool)
        or not isinstance(post_peak, int)
        or post_peak < cold_peak
    ):
        raise LadderError("recurrence runtime contracts are incomplete")

    profiles = _required_mapping(
        result.get("profiles"),
        description="benchmark profiles",
    )
    if set(profiles) != {"recurrence"}:
        raise LadderError("recurrence runtime profile inventory is invalid")
    recurrence_profile = _required_mapping(
        profiles.get("recurrence"),
        description="recurrence runtime profile",
    )
    profile_schedule = _required_mapping(
        result.get("profile_schedule"),
        description="recurrence runtime profile schedule",
    )
    bindings = _required_mapping(
        acceptance.get("bindings"),
        description="recurrence runtime bindings",
    )
    source = _required_mapping(
        result.get("source"),
        description="benchmark source identity",
    )
    runtime_provenance = _required_mapping(
        result.get("runtime_provenance"),
        description="benchmark runtime provenance",
    )
    selector_contract = _required_mapping(
        bindings.get("selector_contract"),
        description="recurrence runtime selector contract",
    )
    validation_fixture = _required_mapping(
        bindings.get("validation_fixture"),
        description="recurrence runtime validation fixture",
    )
    observed_configuration = {
        "modes": configuration.get("modes"),
        "generation_only": configuration.get("generation_only"),
        "batch_sizes": configuration.get("batch_sizes"),
        "target_runtime_seconds": configuration.get("target_runtime_seconds"),
        "warmup_runs": configuration.get("warmup_runs"),
        "minimum_internal_samples": configuration.get("minimum_samples"),
        "subprocess_samples": configuration.get("subprocess_samples"),
        "gluon_count": configuration.get("gluon_count"),
        "process_expression_override": result.get("process"),
        "lc_flow_layout": configuration.get("lc_flow_layout"),
        "color_flow_request": configuration.get("color_flow_request"),
        "helicity_request": configuration.get("helicity_request"),
        "validation_samples": configuration.get("validation_samples"),
        "point_tile_size": configuration.get("point_tile_size"),
        "jit_optimization_level": configuration.get("jit_optimization_level"),
        "specialize_flow_at_generation": configuration.get(
            "specialize_flow_at_generation"
        ),
        "prepared_model_path": configuration.get("prepared_model_path"),
    }
    bound_configuration = _required_mapping(
        bindings.get("configuration"),
        description="recurrence runtime bound configuration",
    )
    profile_selector = _required_mapping(
        recurrence_profile.get("selector_contract"),
        description="recurrence profile selector contract",
    )
    validation = _required_mapping(
        recurrence_profile.get("validation"),
        description="recurrence profile validation",
    )
    fixture = _required_mapping(
        validation.get("fixture"),
        description="recurrence profile validation fixture",
    )
    fixture_file = _required_mapping(
        fixture.get("file"),
        description="recurrence profile validation fixture file",
    )
    expected_fixture = {
        "point_count": fixture.get("point_count"),
        "points_sha256": fixture.get("points_sha256"),
        "file_sha256": fixture_file.get("sha256"),
    }
    if (
        bindings.get("source_identity_sha256") != _canonical_sha256(source)
        or bindings.get("runtime_provenance_sha256")
        != _canonical_sha256(runtime_provenance)
        or bindings.get("profile_schedule_sha256")
        != _canonical_sha256(profile_schedule)
        or bindings.get("recurrence_profile_sha256")
        != _canonical_sha256(recurrence_profile)
        or bindings.get("selector_contract_sha256")
        != _canonical_sha256(selector_contract)
        or dict(selector_contract) != dict(profile_selector)
        or bindings.get("validation_fixture_sha256")
        != _canonical_sha256(validation_fixture)
        or dict(validation_fixture) != expected_fixture
        or bindings.get("process_id") != recurrence_profile.get("process_id")
        or bindings.get("process_expression") != normalized_process
        or bindings.get("lc_flow_layout") != configuration.get("lc_flow_layout")
        or dict(bound_configuration) != observed_configuration
        or bindings.get("configuration_sha256")
        != _canonical_sha256(bound_configuration)
    ):
        raise LadderError("recurrence runtime bindings failed authentication")

    artifact_identity = _required_mapping(
        recurrence.get("artifact_identity"),
        description="recurrence runtime artifact identity",
    )
    artifact_id = artifact_identity.get("artifact_id")
    semantic_sha256 = recurrence.get("artifact_semantic_identity_sha256")
    if (
        recurrence.get("mode") != "recurrence"
        or recurrence.get("generation_reused") is not False
        or bindings.get("generation_artifact_id") != artifact_id
        or bindings.get("generation_artifact_semantic_identity_sha256")
        != semantic_sha256
        or recurrence_profile.get("artifact_semantic_identity_sha256")
        != semantic_sha256
    ):
        raise LadderError("recurrence runtime generation binding is invalid")
    inventory = _verify_artifact_payload_inventory(artifact, artifact_identity)
    return {
        "kind": RUNTIME_ACCEPTANCE_KIND,
        "schema_version": RUNTIME_ACCEPTANCE_SCHEMA_VERSION,
        "passes": True,
        "content_sha256": content_sha256,
        "configuration_sha256": bindings.get("configuration_sha256"),
        "profile_schedule_sha256": bindings.get("profile_schedule_sha256"),
        "recurrence_profile_sha256": bindings.get("recurrence_profile_sha256"),
        "selector_contract_sha256": bindings.get("selector_contract_sha256"),
        "validation_fixture_sha256": bindings.get("validation_fixture_sha256"),
        "worker_count": expected_worker_count,
        "maximum_cold_load_observed_lower_bound_bytes": cold_peak,
        "maximum_post_profile_observed_lower_bound_bytes": post_peak,
        **inventory,
    }


def _runtime_profile_summary(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    profile = _required_mapping(value, description="recurrence runtime profile")
    raw_measurements = profile.get("profiles")
    if not isinstance(raw_measurements, list):
        raise LadderError("recurrence runtime profile has no measurement list")
    measurements: list[dict[str, Any]] = []
    for raw_measurement in raw_measurements:
        measurement = _required_mapping(
            raw_measurement,
            description="runtime batch measurement",
        )
        raw_samples = measurement.get("subprocess_samples")
        if not isinstance(raw_samples, list):
            raise LadderError("runtime batch measurement has no subprocess samples")
        subprocess_samples = []
        for raw_sample in raw_samples:
            sample = _required_mapping(
                raw_sample,
                description="runtime subprocess sample",
            )
            process_record = sample.get("worker_process_record")
            worker_wall = (
                process_record.get("wall_seconds")
                if isinstance(process_record, Mapping)
                else None
            )
            subprocess_samples.append(
                {
                    "schedule_index": sample.get("schedule_index"),
                    "round": sample.get("round"),
                    "wall_seconds_per_point": sample.get("wall_seconds_per_point"),
                    "internal_sample_count": sample.get("internal_sample_count"),
                    "repetitions_per_sample": sample.get("repetitions_per_sample"),
                    "evaluation_count": sample.get("evaluation_count"),
                    "evaluated_point_count": sample.get("evaluated_point_count"),
                    "interrupted": sample.get("interrupted"),
                    "worker_wall_seconds": worker_wall,
                    "peak_rss_after_cold_load": sample.get("peak_rss_after_cold_load"),
                    "peak_rss_after_profile": sample.get("peak_rss_after_profile"),
                }
            )
        measurements.append(
            {
                "batch_size": measurement.get("batch_size"),
                "sample_count": measurement.get("sample_count"),
                "wall_seconds_per_point": measurement.get("wall_seconds_per_point"),
                "wall_seconds_per_point_median": measurement.get(
                    "wall_seconds_per_point_median"
                ),
                "wall_seconds_per_point_mad": measurement.get(
                    "wall_seconds_per_point_mad"
                ),
                "interrupted": measurement.get("interrupted"),
                "subprocess_samples": subprocess_samples,
            }
        )
    return {
        "process_id": profile.get("process_id"),
        "process_expression": profile.get("process_expression"),
        "measurements": measurements,
    }


def parse_harness_result(path: Path) -> dict[str, Any]:
    """Extract compact generation/native/runtime evidence from a harness result."""

    result = _json_object(path, description="recurrence benchmark result")
    if (
        result.get("kind") != HARNESS_KIND
        or result.get("schema_version") != HARNESS_SCHEMA_VERSION
    ):
        raise LadderError(
            "recurrence benchmark result kind/schema does not match this runner"
        )
    generation = _required_mapping(
        result.get("generation"),
        description="benchmark generation",
    )
    recurrence = _required_mapping(
        generation.get("recurrence"),
        description="recurrence generation record",
    )
    artifact_raw = recurrence.get("artifact")
    if not isinstance(artifact_raw, str):
        raise LadderError("recurrence generation record has no artifact path")
    artifact = Path(artifact_raw).expanduser().resolve(strict=True)
    execution_paths = sorted(artifact.glob("processes/*/execution.json"))
    if len(execution_paths) != 1:
        raise LadderError(
            "exact-process recurrence artifact must have one execution manifest"
        )
    execution = _json_object(
        execution_paths[0],
        description="recurrence execution manifest",
    )
    if (
        execution.get("kind") != EXECUTION_KIND
        or execution.get("schema_version") != EXECUTION_SCHEMA_VERSION
    ):
        raise LadderError("recurrence execution manifest kind/schema mismatch")
    plan = _required_mapping(
        execution.get("plan"),
        description="recurrence execution plan",
    )
    inspection = _required_mapping(
        plan.get("inspection_summary"),
        description="native recurrence inspection summary",
    )
    native_timings = _required_mapping(
        inspection.get("generation_timings_seconds"),
        description="native generation timings",
    )
    phases = _required_mapping(
        recurrence.get("phase_timings_seconds"),
        description="generation phase timings",
    )
    profiles = _required_mapping(
        result.get("profiles"),
        description="benchmark profiles",
    )
    artifact_identity = _required_mapping(
        recurrence.get("artifact_identity"),
        description="artifact identity",
    )
    process_record = recurrence.get("worker_process_record")
    worker_wall_seconds = (
        process_record.get("wall_seconds")
        if isinstance(process_record, Mapping)
        else None
    )
    provenance = result.get("provenance")
    driver_wall_seconds = (
        provenance.get("wall_seconds") if isinstance(provenance, Mapping) else None
    )
    generation_only_acceptance = _generation_only_acceptance(
        result,
        recurrence,
        artifact=artifact,
    )
    recurrence_runtime_acceptance = _recurrence_runtime_acceptance(
        result,
        recurrence,
        artifact=artifact,
    )
    return {
        "harness_result_sha256": _sha256_file(path),
        "process": result.get("process"),
        "layout": (
            result.get("configuration", {}).get("lc_flow_layout")
            if isinstance(result.get("configuration"), Mapping)
            else None
        ),
        "source": result.get("source"),
        "runtime_provenance": result.get("runtime_provenance"),
        "artifact": str(artifact),
        "artifact_id": artifact_identity.get("artifact_id"),
        "artifact_semantic_identity_sha256": recurrence.get(
            "artifact_semantic_identity_sha256"
        ),
        "generation_wall_seconds": recurrence.get("generation_wall_seconds"),
        "generation_worker_wall_seconds": worker_wall_seconds,
        "harness_driver_wall_seconds": driver_wall_seconds,
        "generation_peak_rss": recurrence.get("peak_rss"),
        "generation_phase_timings_seconds": dict(phases),
        "generation_phase_total_seconds": recurrence.get("phase_total_seconds"),
        "native_generation_timings_seconds": dict(native_timings),
        "native_inspection_summary": dict(inspection),
        "runtime_profile": _runtime_profile_summary(profiles.get("recurrence")),
        "generation_only_acceptance": generation_only_acceptance,
        "recurrence_runtime_acceptance": recurrence_runtime_acceptance,
        "complete": result.get("complete"),
        "passes": result.get("passes"),
    }


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                value,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        raise LadderError(f"cannot write campaign result: {path}") from error


def _variant_payload(variant: Variant) -> dict[str, str | None]:
    return {
        "checkout": str(variant.checkout),
        "python": str(variant.python),
        "pythonpath": (None if variant.pythonpath is None else str(variant.pythonpath)),
        "prepared_model": (
            None if variant.prepared_model is None else str(variant.prepared_model)
        ),
    }


def _measurement_driver_payload() -> dict[str, Any]:
    files = {}
    for relative in (
        Path(__file__).resolve().relative_to(ROOT),
        HARNESS_RELATIVE_PATH,
        WATCHDOG_RELATIVE_PATH,
    ):
        path = ROOT / relative
        files[relative.as_posix()] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "root": str(ROOT.resolve()),
        "files": files,
        "source_checkout_override_environment": HARNESS_SOURCE_CHECKOUT_ENV,
        "shared_driver_policy": (
            "one-content-hashed-driver-with-variant-source-runtime-binding-v1"
        ),
    }


def _sample_plan_payload(
    spec: SampleSpec,
    *,
    variant: Variant,
    campaign_root: Path,
    settings: RunnerSettings,
) -> dict[str, Any]:
    sample_root = campaign_root / "samples" / spec.sample_id
    return {
        "sample_id": spec.sample_id,
        "sequence_index": spec.sequence_index,
        "pair_index": spec.pair_index,
        "order_in_pair": spec.order_in_pair,
        "repetition": spec.repetition + 1,
        "variant": spec.variant_name,
        "multiplicity": spec.multiplicity,
        "process_expression": spec.process_expression,
        "layout": spec.layout,
        "runtime_enabled": spec.runtime_enabled,
        "generation_timeout_seconds": generation_timeout_seconds(
            spec.multiplicity,
            spec.layout,
        ),
        "outer_timeout_seconds": _outer_timeout_seconds(spec, settings),
        "sample_root": str(sample_root),
        "command": build_sample_command(
            spec,
            variant,
            sample_root,
            settings,
        ),
    }


def _execute_sample(
    spec: SampleSpec,
    *,
    variant: Variant,
    campaign_root: Path,
    settings: RunnerSettings,
) -> dict[str, Any]:
    sample_root = campaign_root / "samples" / spec.sample_id
    sample_root.mkdir(parents=True, exist_ok=False)
    environment, environment_overrides = _sample_environment(
        variant,
        sample_root,
    )
    command = build_sample_command(spec, variant, sample_root, settings)
    stdout_path = sample_root / "stdout.log"
    stderr_path = sample_root / "stderr.log"
    result_path = sample_root / "harness-result.json"
    outer_timeout = _outer_timeout_seconds(spec, settings)
    started_at = _utc_now()
    outcome = _run_process(
        command,
        cwd=variant.checkout,
        environment=environment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=outer_timeout,
    )
    finished_at = _utc_now()
    watchdog = parse_watchdog_log(stderr_path)
    harness_generation_timeout = parse_harness_generation_timeout(stderr_path)
    harness_summary: dict[str, Any] | None = None
    result_error: str | None = None
    if result_path.is_file():
        try:
            harness_summary = parse_harness_result(result_path)
        except LadderError as error:
            result_error = str(error)
    elif outcome.exit_code == 0 and not outcome.timed_out:
        result_error = "watchdog exited successfully without a harness result"

    status = _sample_status(
        outcome,
        watchdog=watchdog,
        result_error=result_error,
        harness_summary=harness_summary,
        harness_generation_timeout=harness_generation_timeout,
        allow_diagnostic_incomplete_success=(
            settings.allow_diagnostic_incomplete_success
        ),
    )
    return {
        **_sample_plan_payload(
            spec,
            variant=variant,
            campaign_root=campaign_root,
            settings=settings,
        ),
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "wall_seconds": outcome.wall_seconds,
        "exit_code": outcome.exit_code,
        "timed_out": outcome.timed_out,
        "process_error": outcome.error,
        "result_error": result_error,
        "environment_overrides": environment_overrides,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "harness_result": str(result_path),
        "watchdog": watchdog,
        "harness_generation_timeout": harness_generation_timeout,
        "telemetry": harness_summary,
    }


def _sample_status(
    outcome: ProcessOutcome,
    *,
    watchdog: Mapping[str, object],
    result_error: str | None,
    harness_summary: Mapping[str, object] | None,
    harness_generation_timeout: Mapping[str, object] | None = None,
    allow_diagnostic_incomplete_success: bool,
) -> str:
    """Classify authoritative success separately from censored diagnostics."""

    if outcome.timed_out:
        return "censored" if allow_diagnostic_incomplete_success else "timeout"
    if outcome.error is not None:
        return "launch-error"
    if harness_generation_timeout is not None:
        if watchdog.get("terminal_record") != "command-finished":
            return "invalid-watchdog-evidence"
        return "censored" if allow_diagnostic_incomplete_success else "timeout"
    if outcome.exit_code != 0:
        return "failed"
    if result_error is not None or harness_summary is None:
        return "invalid-result"
    if watchdog.get("terminal_record") != "command-finished":
        return "invalid-watchdog-evidence"
    complete = harness_summary.get("complete")
    passes = harness_summary.get("passes")
    if complete is True and passes is True:
        return "passed"
    generation_acceptance = harness_summary.get("generation_only_acceptance")
    if (
        isinstance(generation_acceptance, Mapping)
        and generation_acceptance.get("passes") is True
    ):
        return "passed"
    runtime_acceptance = harness_summary.get("recurrence_runtime_acceptance")
    if (
        isinstance(runtime_acceptance, Mapping)
        and runtime_acceptance.get("passes") is True
    ):
        return "passed"
    if allow_diagnostic_incomplete_success and complete is not True and passes is None:
        return "censored"
    return "failed-validation"


def _resolve_existing_path(path: Path, *, description: str) -> Path:
    try:
        result = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise LadderError(f"{description} does not exist: {path}") from error
    return result


def _resolve_python(path: Path, *, description: str) -> Path:
    """Validate while preserving a virtual-environment interpreter symlink."""

    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if not absolute.exists():
        raise LadderError(f"{description} does not exist: {path}")
    if not absolute.is_file():
        raise LadderError(f"{description} is not a regular file")
    return absolute.absolute()


def _resolve_variant(
    name: str,
    *,
    checkout: Path,
    python: Path,
    pythonpath: Path | None,
    prepared_model: Path | None,
) -> Variant:
    resolved_checkout = _resolve_existing_path(
        checkout,
        description=f"{name} checkout",
    )
    if not resolved_checkout.is_dir():
        raise LadderError(f"{name} checkout is not a directory")
    resolved_python = _resolve_python(
        python,
        description=f"{name} Python",
    )
    resolved_pythonpath = (
        None
        if pythonpath is None
        else _resolve_existing_path(
            pythonpath,
            description=f"{name} Python path",
        )
    )
    resolved_model = (
        None
        if prepared_model is None
        else _resolve_existing_path(
            prepared_model,
            description=f"{name} prepared model",
        )
    )
    for relative, description in (
        (WATCHDOG_RELATIVE_PATH, "memory watchdog"),
        (HARNESS_RELATIVE_PATH, "recurrence benchmark harness"),
    ):
        path = resolved_checkout / relative
        if not path.is_file():
            raise LadderError(f"{name} checkout has no {description}: {path}")
    return Variant(
        name=name,
        checkout=resolved_checkout,
        python=resolved_python,
        pythonpath=resolved_pythonpath,
        prepared_model=resolved_model,
    )


def _campaign_root(output_root: Path, campaign_id: str) -> Path:
    if _CAMPAIGN_ID_RE.fullmatch(campaign_id) is None or campaign_id in {".", ".."}:
        raise LadderError(
            "campaign ID must contain only letters, digits, '.', '_', and '-'"
        )
    allowed_parent = ALLOWED_OUTPUT_PARENT.resolve()
    requested_parent = output_root.expanduser().resolve()
    try:
        requested_parent.relative_to(allowed_parent)
    except ValueError as error:
        raise LadderError(
            f"campaign output must remain under {ALLOWED_OUTPUT_PARENT}"
        ) from error
    return requested_parent / campaign_id


def _default_campaign_id() -> str:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-pid{os.getpid()}"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline-python", type=Path, required=True)
    result.add_argument("--candidate-python", type=Path, required=True)
    result.add_argument("--baseline-checkout", type=Path, default=ROOT)
    result.add_argument("--candidate-checkout", type=Path, default=ROOT)
    result.add_argument("--baseline-pythonpath", type=Path)
    result.add_argument("--candidate-pythonpath", type=Path)
    result.add_argument("--baseline-prepared-model", type=Path)
    result.add_argument("--candidate-prepared-model", type=Path)
    result.add_argument(
        "--output-root",
        type=Path,
        default=CAMPAIGN_PARENT,
        help=f"campaign parent below {CAMPAIGN_PARENT}",
    )
    result.add_argument("--campaign-id", default=None)
    result.add_argument(
        "--n",
        dest="multiplicities",
        action="append",
        type=_positive_int,
        help="process multiplicity; repeat (default: n=2 through n=9)",
    )
    result.add_argument(
        "--layout",
        dest="layouts",
        action="append",
        choices=LAYOUTS,
        help="LC flow layout; repeat (default: both layouts)",
    )
    result.add_argument(
        "--repetitions",
        type=_positive_int,
        default=3,
        help="paired baseline/candidate repetitions per cell (default: 3)",
    )
    result.add_argument(
        "--runtime-n",
        action="append",
        type=_positive_int,
        default=None,
        help=(
            "enable runtime profiling for this selected n; repeat "
            "(approved validation: 6 and 7)"
        ),
    )
    result.add_argument(
        "--batch-size",
        action="append",
        type=_positive_int,
        default=None,
    )
    result.add_argument(
        "--validation-samples",
        type=_positive_int,
        default=10,
    )
    result.add_argument(
        "--point-tile-size",
        type=_positive_int,
        default=1024,
    )
    result.add_argument(
        "--jit-optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=2,
    )
    result.add_argument(
        "--profile-timeout",
        type=_positive_float,
        default=300.0,
    )
    result.add_argument(
        "--minimum-samples",
        type=_positive_int,
        default=7,
    )
    result.add_argument(
        "--subprocess-samples",
        type=_positive_int,
        default=7,
    )
    result.add_argument(
        "--warmup-runs",
        type=_nonnegative_int,
        default=2,
    )
    result.add_argument(
        "--target-runtime",
        type=_positive_float,
        default=5.0,
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="write the complete campaign plan without starting subprocesses",
    )
    result.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="stop after the first failed or timed-out sample",
    )
    result.add_argument(
        "--allow-diagnostic-incomplete-success",
        action="store_true",
        help=(
            "retain explicitly censored diagnostic captures without counting "
            "them as passing acceptance samples"
        ),
    )
    return result


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    multiplicities = tuple(
        dict.fromkeys(
            DEFAULT_MULTIPLICITIES
            if arguments.multiplicities is None
            else arguments.multiplicities
        )
    )
    layouts = tuple(
        dict.fromkeys(LAYOUTS if arguments.layouts is None else arguments.layouts)
    )
    runtime_multiplicities = frozenset(arguments.runtime_n or ())
    batch_sizes = tuple(
        DEFAULT_BATCH_SIZES
        if arguments.batch_size is None
        else dict.fromkeys(arguments.batch_size)
    )
    settings = RunnerSettings(
        validation_samples=arguments.validation_samples,
        point_tile_size=arguments.point_tile_size,
        jit_optimization_level=arguments.jit_optimization_level,
        profile_timeout_seconds=arguments.profile_timeout,
        minimum_samples=arguments.minimum_samples,
        subprocess_samples=arguments.subprocess_samples,
        warmup_runs=arguments.warmup_runs,
        target_runtime_seconds=arguments.target_runtime,
        batch_sizes=batch_sizes,
        allow_diagnostic_incomplete_success=(
            arguments.allow_diagnostic_incomplete_success
        ),
    )
    variants = {
        "baseline": _resolve_variant(
            "baseline",
            checkout=arguments.baseline_checkout,
            python=arguments.baseline_python,
            pythonpath=arguments.baseline_pythonpath,
            prepared_model=arguments.baseline_prepared_model,
        ),
        "candidate": _resolve_variant(
            "candidate",
            checkout=arguments.candidate_checkout,
            python=arguments.candidate_python,
            pythonpath=arguments.candidate_pythonpath,
            prepared_model=arguments.candidate_prepared_model,
        ),
    }
    schedule = build_schedule(
        multiplicities,
        layouts,
        arguments.repetitions,
        runtime_multiplicities,
    )
    campaign_id = arguments.campaign_id or _default_campaign_id()
    campaign_root = _campaign_root(arguments.output_root, campaign_id)
    try:
        campaign_root.mkdir(parents=True, exist_ok=False)
        (campaign_root / "samples").mkdir()
    except FileExistsError as error:
        raise LadderError(
            f"campaign directory already exists; choose a new ID: {campaign_root}"
        ) from error
    campaign_path = campaign_root / "campaign.json"
    plan_payloads = [
        _sample_plan_payload(
            spec,
            variant=variants[spec.variant_name],
            campaign_root=campaign_root,
            settings=settings,
        )
        for spec in schedule
    ]
    started_at = _utc_now()
    campaign: dict[str, Any] = {
        "kind": CAMPAIGN_KIND,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "status": "planned" if arguments.dry_run else "running",
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "campaign_root": str(campaign_root),
        "configuration": {
            "multiplicities": list(multiplicities),
            "layouts": list(layouts),
            "repetitions": arguments.repetitions,
            "runtime_multiplicities": sorted(runtime_multiplicities),
            "watchdog_limit_gib": WATCHDOG_LIMIT_GIB,
            "validation_samples": settings.validation_samples,
            "point_tile_size": settings.point_tile_size,
            "jit_optimization_level": settings.jit_optimization_level,
            "profile_timeout_seconds": settings.profile_timeout_seconds,
            "minimum_samples": settings.minimum_samples,
            "subprocess_samples": settings.subprocess_samples,
            "warmup_runs": settings.warmup_runs,
            "target_runtime_seconds": settings.target_runtime_seconds,
            "batch_sizes": list(settings.batch_sizes),
            "stop_on_failure": arguments.stop_on_failure,
            "allow_diagnostic_incomplete_success": (
                settings.allow_diagnostic_incomplete_success
            ),
            "cold_cache_policy": "unique-roots-per-outer-sample-v1",
            "ordering_policy": "alternating-baseline-candidate-pairs-v1",
            "watchdog_policy": "outer-process-tree-guard-covers-all-workers-v1",
            "measurement_driver": _measurement_driver_payload(),
        },
        "variants": {
            name: _variant_payload(variant) for name, variant in variants.items()
        },
        "schedule": plan_payloads,
        "samples": [],
        "summary": None,
    }
    _atomic_write_json(campaign_path, campaign)
    if arguments.dry_run:
        campaign["finished_at_utc"] = _utc_now()
        _atomic_write_json(campaign_path, campaign)
        return campaign

    campaign_started = time.perf_counter()
    for spec in schedule:
        sample = _execute_sample(
            spec,
            variant=variants[spec.variant_name],
            campaign_root=campaign_root,
            settings=settings,
        )
        campaign["samples"].append(sample)
        _atomic_write_json(campaign_path, campaign)
        acceptable_statuses = (
            {"passed", "censored"}
            if settings.allow_diagnostic_incomplete_success
            else {"passed"}
        )
        if arguments.stop_on_failure and sample["status"] not in acceptable_statuses:
            break

    statuses = [
        sample["status"]
        for sample in campaign["samples"]
        if isinstance(sample, Mapping)
    ]
    completed = len(campaign["samples"])
    scheduled = len(schedule)
    passed = sum(status == "passed" for status in statuses)
    censored = sum(status == "censored" for status in statuses)
    failed = completed - passed - censored
    campaign["finished_at_utc"] = _utc_now()
    if completed != scheduled or failed != 0:
        campaign["status"] = "failed"
    elif censored:
        campaign["status"] = "completed-with-censoring"
    else:
        campaign["status"] = "passed"
    campaign["summary"] = {
        "scheduled_sample_count": scheduled,
        "completed_sample_count": completed,
        "passed_sample_count": passed,
        "censored_sample_count": censored,
        "failed_sample_count": failed,
        "campaign_wall_seconds": time.perf_counter() - campaign_started,
    }
    _atomic_write_json(campaign_path, campaign)
    return campaign


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parser().parse_args(argv)
        result = run(arguments)
    except LadderError as error:
        print(f"recurrence-generation-ab-ladder: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "campaign_root": result["campaign_root"],
                "status": result["status"],
                "summary": result["summary"],
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return (
        0
        if result["status"] in {"planned", "passed", "completed-with-censoring"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
