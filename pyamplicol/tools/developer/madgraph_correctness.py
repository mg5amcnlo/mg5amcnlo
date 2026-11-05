# SPDX-License-Identifier: 0BSD
"""Reusable MadGraph standalone ``smatrix`` correctness driver.

This module owns only standalone generation, process-card binding, compilation
of the tiny Fortran driver, and deterministic evaluation.  Model-specific
authentication and performance calibration remain with their callers.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_DRIVER_OUTPUT_RE = re.compile(
    r"^PYAMPLICOL_MG_(VALUE|POINTS|SECONDS|CHECKSUM)\s+(\S+)\s*$",
    re.MULTILINE,
)
_IMPORT_FAILURE_RE = re.compile(
    r"Traceback \(most recent call last\)|UFOImportError|"
    r"Error detected in .*import model|Failed to load.*model",
    re.IGNORECASE,
)
_PARAMETER_VALUE_RE = re.compile(
    r"^(?P<prefix>.*\s)(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[EeDd][+-]?\d+)?)(?P<trailing>\s*)$"
)


class MadGraphAdapterError(RuntimeError):
    """The MadGraph standalone correctness contract could not be satisfied."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    args: tuple[str, ...]
    cwd: Path
    elapsed_seconds: float
    returncode: int
    stdout: str
    stderr: str

    def record(self) -> dict[str, object]:
        return {
            "args": list(self.args),
            "cwd": os.fspath(self.cwd.resolve(strict=False)),
            "elapsed_seconds": self.elapsed_seconds,
            "returncode": self.returncode,
        }


class CommandExecutor(Protocol):
    def run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path,
    ) -> CommandResult:
        raise NotImplementedError


class SubprocessExecutor:
    """Small command seam used by unit tests and worker supervision."""

    def run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path,
    ) -> CommandResult:
        rendered = tuple(os.fspath(item) for item in args)
        started = time.perf_counter()
        completed = subprocess.run(
            rendered,
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
        )
        return CommandResult(
            args=rendered,
            cwd=cwd,
            elapsed_seconds=time.perf_counter() - started,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


@dataclass(frozen=True, slots=True)
class DriverResult:
    value: float
    points: int
    seconds: float
    checksum: float
    command: CommandResult


@dataclass(frozen=True, slots=True)
class StandaloneProcess:
    installation: Path
    launcher: Path
    artifact: Path
    standalone: Path
    subprocess: Path
    executable: Path
    log_path: Path
    command_card: str
    generation: CommandResult
    compilation: CommandResult
    driver_sha256: str


_DRIVER_SOURCE = """\
program pyamplicol_madgraph_driver
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  include 'nexternal.inc'
  integer :: i, ios, momentum_unit, repetitions, warmup_calls
  integer(kind=int64) :: clock_start, clock_end, clock_rate
  real(kind=8) :: p(0:3,nexternal), answer, reference, checksum, seconds
  character(len=4096) :: momenta_path, param_card_path, raw_argument

  call get_command_argument(1, momenta_path)
  call get_command_argument(2, param_card_path)
  call get_command_argument(3, raw_argument)
  read(raw_argument, *, iostat=ios) repetitions
  if (ios /= 0 .or. repetitions < 1) error stop 11
  call get_command_argument(4, raw_argument)
  read(raw_argument, *, iostat=ios) warmup_calls
  if (ios /= 0 .or. warmup_calls < 20) error stop 12

  open(newunit=momentum_unit, file=trim(momenta_path), status='old', &
       action='read', iostat=ios)
  if (ios /= 0) error stop 13
  do i = 1, nexternal
    read(momentum_unit, *, iostat=ios) p(0,i), p(1,i), p(2,i), p(3,i)
    if (ios /= 0) error stop 14
  end do
  close(momentum_unit)

  call setpara(trim(param_card_path))
  do i = 1, warmup_calls
    call smatrix(p, answer)
  end do
  call smatrix(p, reference)

  checksum = 0.0d0
  call system_clock(clock_start, clock_rate)
  do i = 1, repetitions
    call smatrix(p, answer)
    if (answer /= reference) error stop 16
    checksum = checksum + answer
  end do
  call system_clock(clock_end)
  if (clock_rate <= 0) error stop 15
  seconds = real(clock_end - clock_start, kind=8) / real(clock_rate, kind=8)

  write(*,'(A,1X,ES25.17E3)') 'PYAMPLICOL_MG_VALUE', reference
  write(*,'(A,1X,I0)') 'PYAMPLICOL_MG_POINTS', repetitions
  write(*,'(A,1X,ES25.17E3)') 'PYAMPLICOL_MG_SECONDS', seconds
  write(*,'(A,1X,ES25.17E3)') 'PYAMPLICOL_MG_CHECKSUM', checksum
end program pyamplicol_madgraph_driver
"""
MADGRAPH_DRIVER_SOURCE_SHA256 = hashlib.sha256(
    _DRIVER_SOURCE.encode("utf-8")
).hexdigest()


def validate_installation(path: Path) -> tuple[Path, Path]:
    installation = path.expanduser().resolve(strict=True)
    if not installation.is_dir():
        raise MadGraphAdapterError(
            f"MadGraph installation is not a directory: {installation}"
        )
    launcher = installation / "bin" / "mg5_aMC"
    if (
        not launcher.is_file()
        or launcher.is_symlink()
        or not os.access(launcher, os.X_OK)
    ):
        raise MadGraphAdapterError(
            "MadGraph installation must contain a regular executable bin/mg5_aMC"
        )
    return installation, launcher


def madgraph_command_card(
    process: str,
    *,
    model_import: str = "sm",
    coupling_orders: Mapping[str, int] | None = None,
) -> str:
    if "\n" in process or "\r" in process or not process.strip():
        raise ValueError("MadGraph process must be one non-empty line")
    if (
        "\n" in model_import
        or "\r" in model_import
        or not model_import.strip()
        or any(character.isspace() for character in model_import)
    ):
        raise ValueError("MadGraph model import must be one non-empty token")
    orders = coupling_orders or {}
    if any(
        not isinstance(name, str)
        or not name
        or not name.isidentifier()
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for name, value in orders.items()
    ):
        raise ValueError("MadGraph coupling orders must be non-negative integers")
    order_suffix = "".join(f" {name}={orders[name]}" for name in sorted(orders))
    return "\n".join(
        (
            f"import model {model_import}",
            f"generate {process}{order_suffix}",
            "output standalone standalone -f",
            "launch -f",
            "",
        )
    )


def reject_failed_generation(
    result: CommandResult,
    standalone: Path,
    *,
    expected_model_import: str,
) -> None:
    output = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        raise MadGraphAdapterError(
            f"MadGraph exited with {result.returncode}; see madgraph.log"
        )
    if _IMPORT_FAILURE_RE.search(output):
        raise MadGraphAdapterError(
            "MadGraph reported a UFO import failure; refusing a possible model fallback"
        )
    process_card = standalone / "Cards" / "proc_card_mg5.dat"
    if not process_card.is_file():
        raise MadGraphAdapterError("MadGraph produced no standalone process card")
    recorded = process_card.read_text(encoding="utf-8", errors="replace")
    recorded = re.sub(r"\\\r?\n", "", recorded)
    import_lines = re.findall(r"(?m)^import model\s+(\S+)(?:\s+.*)?$", recorded)
    if not import_lines or set(import_lines) != {expected_model_import}:
        raise MadGraphAdapterError(
            "standalone process card is not bound exclusively to the requested UFO"
        )


def discover_subprocess(standalone: Path) -> Path:
    matches = tuple(
        sorted(
            path.parent
            for path in (standalone / "SubProcesses").glob("P*/matrix.f")
            if path.is_file()
        )
    )
    if len(matches) != 1:
        raise MadGraphAdapterError(
            "MadGraph authority requires exactly one generated subprocess; "
            f"found {len(matches)}"
        )
    subprocess_dir = matches[0]
    required = (
        subprocess_dir / "matrix.o",
        standalone / "lib" / "libdhelas.a",
        standalone / "lib" / "libmodel.a",
        standalone / "Cards" / "param_card.dat",
    )
    missing = tuple(path for path in required if not path.is_file())
    if missing:
        raise MadGraphAdapterError(
            "MadGraph launch did not compile the standalone tree: "
            + ", ".join(os.fspath(path) for path in missing)
        )
    return subprocess_dir


def fortran_compiler(standalone: Path) -> str:
    make_options = standalone / "Source" / "make_opts"
    source = make_options.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?m)^DEFAULT_F_COMPILER\s*=\s*(\S+)\s*$", source)
    configured = match.group(1) if match is not None else "gfortran"
    compiler = shutil.which(configured)
    if compiler is None:
        raise MadGraphAdapterError(
            f"MadGraph's configured Fortran compiler is unavailable: {configured}"
        )
    return compiler


def parse_driver_output(result: CommandResult, expected_points: int) -> DriverResult:
    if result.returncode != 0:
        raise MadGraphAdapterError(
            f"MadGraph driver exited with {result.returncode}; see madgraph.log"
        )
    fields: dict[str, str] = {}
    for field, raw in _DRIVER_OUTPUT_RE.findall(result.stdout + "\n" + result.stderr):
        if field in fields:
            raise MadGraphAdapterError(f"MadGraph driver repeated {field}")
        fields[field] = raw
    if set(fields) != {"VALUE", "POINTS", "SECONDS", "CHECKSUM"}:
        raise MadGraphAdapterError(
            "MadGraph driver output is incomplete: " + ", ".join(sorted(fields))
        )
    try:
        value = float(fields["VALUE"].replace("D", "E").replace("d", "e"))
        points = int(fields["POINTS"])
        seconds = float(fields["SECONDS"].replace("D", "E").replace("d", "e"))
        checksum = float(fields["CHECKSUM"].replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise MadGraphAdapterError(
            "MadGraph driver emitted malformed numbers"
        ) from error
    if (
        points != expected_points
        or not math.isfinite(value)
        or value < 0.0
        or not math.isfinite(seconds)
        or seconds < 0.0
        or not math.isfinite(checksum)
    ):
        raise MadGraphAdapterError(
            "MadGraph driver emitted invalid timing or value data"
        )
    expected_checksum = points * value
    unit_roundoff = math.ulp(1.0) / 2.0
    accumulated_roundoff = points * unit_roundoff
    checksum_rel_tol = max(
        5.0e-13,
        accumulated_roundoff / (1.0 - accumulated_roundoff),
    )
    if not math.isclose(
        checksum,
        expected_checksum,
        rel_tol=checksum_rel_tol,
        abs_tol=1.0e-300,
    ):
        raise MadGraphAdapterError("MadGraph repeated evaluations changed their value")
    return DriverResult(value, points, seconds, checksum, result)


def momenta_rows(
    points: object,
) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(points, tuple) or len(points) != 1:
        raise MadGraphAdapterError("MadGraph authority requires one validation point")
    raw_point = points[0]
    if not isinstance(raw_point, tuple):
        raise MadGraphAdapterError("MadGraph validation point is malformed")
    rows: list[tuple[float, float, float, float]] = []
    for raw_row in raw_point:
        if not isinstance(raw_row, tuple) or len(raw_row) != 4:
            raise MadGraphAdapterError("MadGraph validation momentum is malformed")
        row = tuple(float(component) for component in raw_row)
        if any(not math.isfinite(component) for component in row):
            raise MadGraphAdapterError("MadGraph validation momentum is not finite")
        rows.append(row)  # type: ignore[arg-type]
    return tuple(rows)


def set_parameter_card_values(
    path: Path,
    values: Mapping[str, float],
) -> tuple[str, ...]:
    """Set named real LHA inputs in an existing MadGraph parameter card."""

    if not values or any(
        not isinstance(name, str)
        or not name.isidentifier()
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for name, value in values.items()
    ):
        raise ValueError("parameter-card values must be finite named real numbers")
    try:
        lines = path.read_text(encoding="ascii").splitlines(keepends=True)
    except OSError as error:
        raise MadGraphAdapterError(
            f"cannot read MadGraph parameter card: {path}"
        ) from error

    observed: set[str] = set()
    rewritten: list[str] = []
    for line in lines:
        body, marker, comment = line.partition("#")
        name = (
            comment.strip().split(maxsplit=1)[0] if marker and comment.strip() else ""
        )
        if name not in values:
            rewritten.append(line)
            continue
        if name in observed:
            raise MadGraphAdapterError(
                f"MadGraph parameter card repeats external parameter {name}"
            )
        newline = "\n" if line.endswith("\n") else ""
        content = body[:-1] if newline and body.endswith("\n") else body
        match = _PARAMETER_VALUE_RE.fullmatch(content)
        if match is None:
            raise MadGraphAdapterError(
                f"MadGraph parameter card has a malformed value for {name}"
            )
        rendered = f"{float(values[name]):.14e}"
        rewritten.append(
            match.group("prefix")
            + rendered
            + match.group("trailing")
            + marker
            + comment.removesuffix("\n")
            + newline
        )
        observed.add(name)

    missing = sorted(set(values).difference(observed))
    if missing:
        raise MadGraphAdapterError(
            f"MadGraph parameter card lacks required external parameters: {missing!r}"
        )
    try:
        path.write_text("".join(rewritten), encoding="ascii")
    except OSError as error:
        raise MadGraphAdapterError(
            f"cannot update MadGraph parameter card: {path}"
        ) from error
    return tuple(sorted(observed))


class StandaloneMadGraphRunner:
    """Generate and drive standalone tree matrix elements."""

    def __init__(self, *, executor: CommandExecutor | None = None) -> None:
        self.executor = SubprocessExecutor() if executor is None else executor

    @staticmethod
    def _run_logged(
        executor: CommandExecutor,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path,
        log_path: Path,
    ) -> CommandResult:
        result = executor.run(args, cwd=cwd)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write("$ " + " ".join(os.fspath(item) for item in args) + "\n")
            stream.write(result.stdout)
            if result.stdout and not result.stdout.endswith("\n"):
                stream.write("\n")
            stream.write(result.stderr)
            if result.stderr and not result.stderr.endswith("\n"):
                stream.write("\n")
        return result

    def _compile_driver(
        self,
        standalone: Path,
        subprocess_dir: Path,
        *,
        log_path: Path,
    ) -> tuple[Path, CommandResult, str]:
        source_path = subprocess_dir / "pyamplicol_madgraph_driver.f90"
        executable = subprocess_dir / "pyamplicol_madgraph_driver"
        source_path.write_text(_DRIVER_SOURCE, encoding="utf-8")
        compiler = fortran_compiler(standalone)
        command = (
            compiler,
            "-O3",
            "-I.",
            "-I../../Source",
            "-I../../Source/MODEL",
            source_path.name,
            "matrix.o",
            "-L../../lib",
            "-ldhelas",
            "-lmodel",
            "-o",
            executable.name,
        )
        result = self._run_logged(
            self.executor,
            command,
            cwd=subprocess_dir,
            log_path=log_path,
        )
        if result.returncode != 0 or not executable.is_file():
            raise MadGraphAdapterError(
                "custom MadGraph Fortran driver failed to compile; see madgraph.log"
            )
        return executable, result, MADGRAPH_DRIVER_SOURCE_SHA256

    def _run_driver(
        self,
        executable: Path,
        *,
        subprocess_dir: Path,
        momenta_path: Path,
        points: int,
        warmup_calls: int,
        log_path: Path,
    ) -> DriverResult:
        result = self._run_logged(
            self.executor,
            (
                executable,
                momenta_path,
                "../../Cards/param_card.dat",
                str(points),
                str(warmup_calls),
            ),
            cwd=subprocess_dir,
            log_path=log_path,
        )
        return parse_driver_output(result, points)

    def generate(
        self,
        *,
        installation: Path,
        artifact: Path,
        process: str,
        model_import: str,
        coupling_orders: Mapping[str, int] | None = None,
    ) -> StandaloneProcess:
        installation, launcher = validate_installation(installation)
        artifact = artifact.expanduser().resolve(strict=False)
        artifact.mkdir(parents=True, exist_ok=True)
        log_path = artifact / "madgraph.log"
        standalone = artifact / "standalone"
        command_card = madgraph_command_card(
            process,
            model_import=model_import,
            coupling_orders=coupling_orders,
        )
        command_card_path = artifact / "madgraph_command_card.dat"
        command_card_path.write_text(command_card, encoding="utf-8")
        generation = self._run_logged(
            self.executor,
            (launcher, command_card_path),
            cwd=artifact,
            log_path=log_path,
        )
        reject_failed_generation(
            generation,
            standalone,
            expected_model_import=model_import,
        )
        subprocess_dir = discover_subprocess(standalone)
        executable, compilation, driver_sha256 = self._compile_driver(
            standalone,
            subprocess_dir,
            log_path=log_path,
        )
        return StandaloneProcess(
            installation=installation,
            launcher=launcher,
            artifact=artifact,
            standalone=standalone,
            subprocess=subprocess_dir,
            executable=executable,
            log_path=log_path,
            command_card=command_card,
            generation=generation,
            compilation=compilation,
            driver_sha256=driver_sha256,
        )

    def evaluate(
        self,
        standalone: StandaloneProcess,
        momenta: Sequence[Sequence[float]],
        *,
        repetitions: int = 1,
        warmup_calls: int = 20,
    ) -> DriverResult:
        rows = tuple(tuple(float(component) for component in row) for row in momenta)
        if any(
            len(row) != 4 or any(not math.isfinite(component) for component in row)
            for row in rows
        ):
            raise MadGraphAdapterError("MadGraph momenta must be finite four-vectors")
        momenta_path = standalone.subprocess / "pyamplicol_momenta.dat"
        momenta_path.write_text(
            "".join(
                " ".join(format(component, ".17e") for component in row) + "\n"
                for row in rows
            ),
            encoding="ascii",
        )
        return self._run_driver(
            standalone.executable,
            subprocess_dir=standalone.subprocess,
            momenta_path=momenta_path,
            points=repetitions,
            warmup_calls=warmup_calls,
            log_path=standalone.log_path,
        )


__all__ = [
    "MADGRAPH_DRIVER_SOURCE_SHA256",
    "CommandExecutor",
    "CommandResult",
    "DriverResult",
    "MadGraphAdapterError",
    "StandaloneMadGraphRunner",
    "StandaloneProcess",
    "SubprocessExecutor",
    "madgraph_command_card",
    "momenta_rows",
    "parse_driver_output",
    "reject_failed_generation",
    "set_parameter_card_values",
    "validate_installation",
]
