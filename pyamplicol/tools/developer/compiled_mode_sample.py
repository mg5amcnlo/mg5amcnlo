#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Emit one version-independent eager/compiled runtime timing sample as JSON.

The helper is executed by the interpreter being measured.  It deliberately
does not use :class:`pyamplicol.benchmarking.BenchmarkBackend`, whose timing
policy can differ between the frozen and current installations.  Every
headline block calls the native ``_benchmark_f64_wall_time`` entry point, whose
timer starts after it has packed the batch. A separate private warmed-Arena
profile uses the same borrowed input, preallocated-output boundary, selector
arguments, and repetition count solely for attribution. Frozen baseline
runtimes are measured only through an explicit ``frozen-pre-arena`` protocol
that invokes their historical ``profile_repeated`` operation. Automatic
capability detection and fallback are forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RESULT_KIND = "pyamplicol-compiled-mode-native-sample"
SCHEMA_VERSION = 5
INSTALLATION_IDENTITY_KIND = "pyamplicol-installed-distribution-identity"
INSTALLATION_IDENTITY_SCHEMA_VERSION = 2
NATIVE_WALL_TIME_SOURCE = "runtime_core_repeated_wall_time"
NATIVE_WALL_TIME_SAMPLE_PASS = "runtime._benchmark_f64_wall_time"
ARENA_PROFILE_ATTRIBUTION_SAMPLE_PASS = "runtime._profile_arena_repeated"
FROZEN_BASELINE_PROFILE_ATTRIBUTION_SAMPLE_PASS = "runtime.profile_repeated"
ARENA_PROFILE_BOUNDARY = "warmed-direct-arena-borrowed-input-preallocated-output-v1"
FROZEN_BASELINE_PROFILE_BOUNDARY = "materialized-native-profile-v1"
ARENA_PHASE_TIMING_SCOPE = "coarse-arena-boundary-only-v1"
FROZEN_BASELINE_PHASE_TIMING_SCOPE = "profiled-evaluator-phases-v1"
ARENA_PROFILE_PROTOCOL = "arena"
FROZEN_BASELINE_PROFILE_PROTOCOL = "frozen-pre-arena"
PROFILE_PROTOCOLS = (ARENA_PROFILE_PROTOCOL, FROZEN_BASELINE_PROFILE_PROTOCOL)
PAIRED_TIMING_SAMPLE_CONTRACT = "paired_unprofiled_headline_profiled_attribution_v1"
MAX_CALIBRATION_BLOCKS = 4
MAX_REPETITIONS = 1_000_000_000
MAX_CORRECTNESS_POINTS = 8
MIN_CORRECTNESS_POINTS = 8
CORRECTNESS_POINT_DERIVATION_KIND = (
    "pyamplicol-compiled-mode-correctness-point-derivation"
)
CORRECTNESS_POINT_DERIVATION_SCHEMA_VERSION = 1
CORRECTNESS_POINT_DERIVATION_CONTRACT = (
    "authenticated-first-validation-point-massive-rambo-v1"
)
CORRECTNESS_SEED_START = 20260726
MAX_CORRECTNESS_SEED_ATTEMPTS = 256
MIN_SCALED_PAIR_INVARIANT = 1.0e-10
KINEMATIC_RELATIVE_TOLERANCE = 1.0e-10
MASS_SQUARED_ROUNDOFF_ULPS = 512.0
_HASH_CHUNK_BYTES = 1024 * 1024

FourMomentum = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ValidationMomentaSource:
    """Authenticated artifact validation momenta and path-free provenance."""

    points: tuple[tuple[FourMomentum, ...], ...]
    identity: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ProfileOperation:
    operation: Any
    sample_pass: str
    boundary: str
    borrowed_flat_input: bool
    preallocated_output: bool
    phase_timing_scope: str
    evaluator_timing_available: bool


class SampleError(RuntimeError):
    """Raised when a trustworthy paired native sample cannot be produced."""


EXECUTION_MODES = ("eager", "compiled")
DEPENDENCY_PACKAGES = ("numpy", "symbolica", "ufo_model_loader")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _at_least_five(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 5:
        raise argparse.ArgumentTypeError("must be at least five")
    return parsed


def _finite_positive(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise SampleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SampleError(f"{label} must be positive and finite")
    return result


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise SampleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SampleError(f"{label} must be finite and non-negative")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _path_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(resolved),
    }


def _package_origin(package: str) -> Path | None:
    specification = importlib.util.find_spec(package)
    origin = None if specification is None else specification.origin
    if origin is None:
        return None
    try:
        return Path(origin).resolve(strict=True)
    except OSError as error:
        raise SampleError(
            f"cannot resolve {package} package origin: {origin}"
        ) from error


def _activate_dependency_site(path: Path | None) -> dict[str, object] | None:
    """Load exact dependencies without allowing them to shadow the measured wheel."""

    if path is None:
        return None
    try:
        dependency_site = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise SampleError(f"cannot resolve dependency site: {path}") from error
    if not dependency_site.is_dir():
        raise SampleError(f"dependency site is not a directory: {dependency_site}")
    pyamplicol_before = _package_origin("pyamplicol")
    if pyamplicol_before is None:
        raise SampleError("the selected interpreter has no pyamplicol package")
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(dependency_site))
        for package in DEPENDENCY_PACKAGES:
            importlib.import_module(package)
    except ImportError as error:
        raise SampleError(
            f"dependency site does not provide required package {error.name!r}"
        ) from error
    finally:
        sys.path[:] = original_path
    if str(dependency_site) not in sys.path:
        sys.path.append(str(dependency_site))
    pyamplicol_after = _package_origin("pyamplicol")
    if pyamplicol_after != pyamplicol_before:
        raise SampleError(
            "dependency-site activation changed the selected pyamplicol package "
            f"origin: {pyamplicol_before} -> {pyamplicol_after}"
        )
    distribution = importlib.metadata.distribution("pyamplicol")
    distribution_init = Path(
        str(distribution.locate_file("pyamplicol/__init__.py"))
    ).resolve(strict=True)
    if distribution_init != pyamplicol_before:
        raise SampleError(
            "selected pyamplicol distribution and import package disagree: "
            f"{distribution_init} != {pyamplicol_before}"
        )
    dependencies: dict[str, object] = {}
    for package in DEPENDENCY_PACKAGES:
        origin = _package_origin(package)
        if origin is None:
            raise SampleError(f"required dependency {package!r} has no import origin")
        try:
            origin.relative_to(dependency_site)
        except ValueError as error:
            raise SampleError(
                f"required dependency {package!r} was not loaded from "
                f"{dependency_site}: {origin}"
            ) from error
        dependencies[package] = _path_identity(origin)
    return {
        "path": str(path),
        "resolved_path": str(dependency_site),
        "pyamplicol_origin_before": _path_identity(pyamplicol_before),
        "pyamplicol_origin_after": _path_identity(pyamplicol_after),
        "dependencies": dependencies,
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _installed_distribution_identity() -> dict[str, object]:
    """Return a content identity for the installed generator/runtime wheel."""

    try:
        distribution = importlib.metadata.distribution("pyamplicol")
    except importlib.metadata.PackageNotFoundError as error:
        raise SampleError(
            "the selected interpreter has no pyamplicol installation"
        ) from error
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    native_modules: list[dict[str, object]] = []
    build_info_files: list[dict[str, object]] = []
    for entry in sorted(distribution.files or (), key=str):
        relative = str(entry)
        path = Path(str(distribution.locate_file(entry)))
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        size = resolved.stat().st_size
        entry_bytes = relative.encode("utf-8")
        digest.update(len(entry_bytes).to_bytes(8, "big"))
        digest.update(entry_bytes)
        digest.update(size.to_bytes(8, "big"))
        with resolved.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
        file_count += 1
        size_bytes += size
        name = resolved.name
        identity = {
            "relative_path": relative,
            **_path_identity(resolved),
        }
        if name.startswith("_rusticol.") and resolved.suffix in {
            ".so",
            ".pyd",
            ".dylib",
        }:
            native_modules.append(identity)
        if relative.endswith("pyamplicol/_build_info.json"):
            try:
                payload = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise SampleError(
                    f"cannot read installed build provenance: {resolved}"
                ) from error
            if not isinstance(payload, dict):
                raise SampleError(
                    "installed pyamplicol build provenance is not an object"
                )
            build_info_files.append({**identity, "payload": payload})
    if len(native_modules) != 1:
        raise SampleError(
            "the selected pyamplicol installation must contain exactly one "
            f"native Rusticol module, found {len(native_modules)}"
        )
    return {
        "kind": INSTALLATION_IDENTITY_KIND,
        "schema_version": INSTALLATION_IDENTITY_SCHEMA_VERSION,
        "package_version": distribution.version,
        "distribution_content": {
            "algorithm": "sha256-relative-path-size-content-v1",
            "sha256": digest.hexdigest(),
            "file_count": file_count,
            "size_bytes": size_bytes,
        },
        "native_modules": native_modules,
        "build_info_files": build_info_files,
    }


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SampleError("native profile contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(entry) for key, entry in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(entry) for entry in value]
    raise SampleError(
        f"native profile contains a non-JSON value of type {type(value).__name__}"
    )


def _statistics(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise SampleError("timing statistics require at least one sample")
    mean = float(statistics.fmean(values))
    deviation = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    error = deviation / math.sqrt(len(values))
    return {
        "standard_deviation": deviation,
        "standard_error": error,
        "relative_standard_error": error / mean if mean > 0.0 else 0.0,
    }


def _component(values: Sequence[float]) -> dict[str, object]:
    return {
        "mean_seconds_per_point": float(statistics.fmean(values)),
        "uncertainty": _statistics(values),
        "sample_count": len(values),
        "samples_seconds_per_point": list(values),
    }


def _benchmark_batch(points: Sequence[object], batch_size: int) -> tuple[object, ...]:
    source = tuple(points)
    if not source:
        raise SampleError("artifact has no deterministic validation point")
    return tuple(source[index % len(source)] for index in range(batch_size))


def _sum_momenta(momenta: Sequence[FourMomentum]) -> FourMomentum:
    return tuple(
        float(math.fsum(momentum[axis] for momentum in momenta)) for axis in range(4)
    )  # type: ignore[return-value]


def _minkowski_square(momentum: FourMomentum) -> float:
    return (
        momentum[0] * momentum[0]
        - momentum[1] * momentum[1]
        - momentum[2] * momentum[2]
        - momentum[3] * momentum[3]
    )


def _minkowski_dot(left: FourMomentum, right: FourMomentum) -> float:
    return (
        left[0] * right[0]
        - left[1] * right[1]
        - left[2] * right[2]
        - left[3] * right[3]
    )


def _kinematic_scale(*values: float) -> float:
    return max(1.0, *(abs(value) for value in values))


def _kinematically_close(left: float, right: float, *, scale: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=KINEMATIC_RELATIVE_TOLERANCE * max(1.0, scale),
    )


def _validated_mass_squared(momentum: FourMomentum, *, label: str) -> float:
    if momentum[0] <= 0.0:
        raise SampleError(f"{label} energy must be positive")
    spatial_squared = math.fsum(component * component for component in momentum[1:])
    mass_squared = momentum[0] * momentum[0] - spatial_squared
    roundoff_limit = (
        MASS_SQUARED_ROUNDOFF_ULPS
        * sys.float_info.epsilon
        * _kinematic_scale(momentum[0] * momentum[0], spatial_squared)
    )
    if mass_squared < -roundoff_limit:
        raise SampleError(f"{label} has a spacelike four-momentum")
    if abs(mass_squared) <= roundoff_limit:
        return 0.0
    return mass_squared


def _inferred_mass(momentum: FourMomentum, *, label: str) -> float:
    return math.sqrt(_validated_mass_squared(momentum, label=label))


def _boost_to_rest(
    momentum: FourMomentum,
    beta: tuple[float, float, float],
) -> FourMomentum:
    beta_squared = math.fsum(component * component for component in beta)
    if beta_squared == 0.0:
        return momentum
    if not 0.0 < beta_squared < 1.0:
        raise SampleError("RAMBO generated an invalid center-of-mass boost")
    gamma = 1.0 / math.sqrt(1.0 - beta_squared)
    beta_dot_momentum = math.fsum(beta[axis] * momentum[axis + 1] for axis in range(3))
    spatial_factor = (
        gamma - 1.0
    ) * beta_dot_momentum / beta_squared + gamma * momentum[0]
    return (
        gamma * (momentum[0] + beta_dot_momentum),
        momentum[1] + spatial_factor * beta[0],
        momentum[2] + spatial_factor * beta[1],
        momentum[3] + spatial_factor * beta[2],
    )


def _massive_rambo_final_state(
    multiplicity: int,
    *,
    sqrt_s: float,
    masses: Sequence[float],
    seed: int,
    numpy_module: Any,
) -> tuple[FourMomentum, ...]:
    """Standalone RAMBO using only the activated dependency-site NumPy."""

    if multiplicity < 2 or multiplicity != len(masses):
        raise SampleError("correctness RAMBO requires at least two final particles")
    rng = numpy_module.random.default_rng(seed)
    raw: list[FourMomentum] = []
    for _ in range(multiplicity):
        random_values = rng.random(4)
        cosine = 2.0 * float(random_values[0]) - 1.0
        sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
        phi = 2.0 * math.pi * float(random_values[1])
        logarithm_argument = max(
            float(random_values[2]) * float(random_values[3]),
            sys.float_info.min,
        )
        energy = -math.log(logarithm_argument)
        raw.append(
            (
                energy,
                energy * sine * math.cos(phi),
                energy * sine * math.sin(phi),
                energy * cosine,
            )
        )
    total = _sum_momenta(raw)
    invariant = _minkowski_square(total)
    if not math.isfinite(invariant) or invariant <= 0.0 or total[0] <= 0.0:
        raise SampleError("RAMBO generated a non-timelike total momentum")
    massless_scale = sqrt_s / math.sqrt(invariant)
    beta_to_rest = (
        -total[1] / total[0],
        -total[2] / total[0],
        -total[3] / total[0],
    )
    massless = tuple(
        tuple(massless_scale * component for component in boosted)
        for boosted in (_boost_to_rest(momentum, beta_to_rest) for momentum in raw)
    )
    spatial_norms = tuple(
        math.sqrt(math.fsum(component * component for component in momentum[1:]))
        for momentum in massless
    )

    def energy_sum(scale: float) -> float:
        return math.fsum(
            math.sqrt(mass * mass + (scale * spatial_norm) ** 2)
            for mass, spatial_norm in zip(masses, spatial_norms, strict=True)
        )

    low = 0.0
    high = 1.0
    while energy_sum(high) < sqrt_s:
        high *= 2.0
        if not math.isfinite(high):
            raise SampleError("RAMBO massive spatial scale did not converge")
    for _ in range(128):
        middle = 0.5 * (low + high)
        if energy_sum(middle) > sqrt_s:
            high = middle
        else:
            low = middle
    spatial_scale = 0.5 * (low + high)
    return tuple(
        (
            math.sqrt(mass * mass + (spatial_scale * spatial_norm) ** 2),
            spatial_scale * momentum[1],
            spatial_scale * momentum[2],
            spatial_scale * momentum[3],
        )
        for momentum, mass, spatial_norm in zip(
            massless,
            masses,
            spatial_norms,
            strict=True,
        )
    )


def _validated_source_kinematics(
    point: Sequence[FourMomentum],
) -> tuple[tuple[FourMomentum, FourMomentum], float, tuple[float, ...]]:
    if len(point) < 4:
        raise SampleError(
            "correctness derivation requires two incoming and at least two "
            "final-state particles"
        )
    incoming = (point[0], point[1])
    outgoing = tuple(point[2:])
    initial_total = _sum_momenta(incoming)
    final_total = _sum_momenta(outgoing)
    sqrt_s = initial_total[0]
    if not math.isfinite(sqrt_s) or sqrt_s <= 0.0:
        raise SampleError("validation point has an invalid center-of-mass energy")
    for axis in range(1, 4):
        if not _kinematically_close(
            initial_total[axis],
            0.0,
            scale=sqrt_s,
        ):
            raise SampleError(
                "validation point initial state is not in the center-of-mass frame"
            )
    for axis, (initial, final) in enumerate(
        zip(initial_total, final_total, strict=True)
    ):
        if not _kinematically_close(initial, final, scale=sqrt_s):
            raise SampleError(
                "validation point violates four-momentum conservation "
                f"on component {axis}"
            )
    masses = tuple(
        _inferred_mass(momentum, label=f"validation particle {index}")
        for index, momentum in enumerate(point)
    )
    final_masses = masses[2:]
    threshold = math.fsum(final_masses)
    if not threshold < sqrt_s:
        raise SampleError(
            "validation point is not strictly above its final-state mass threshold"
        )
    return incoming, sqrt_s, masses


def _validated_generated_point(
    point: Sequence[FourMomentum],
    *,
    sqrt_s: float,
    masses: Sequence[float],
) -> tuple[str, ...]:
    if len(point) != len(masses):
        raise SampleError("generated correctness point has the wrong multiplicity")
    incoming_total = _sum_momenta(point[:2])
    outgoing_total = _sum_momenta(point[2:])
    for axis, (initial, final) in enumerate(
        zip(incoming_total, outgoing_total, strict=True)
    ):
        if not _kinematically_close(initial, final, scale=sqrt_s):
            raise SampleError(
                "generated correctness point violates four-momentum conservation "
                f"on component {axis}"
            )
    for index, (momentum, expected_mass) in enumerate(zip(point, masses, strict=True)):
        observed_mass_squared = _validated_mass_squared(
            momentum,
            label=f"generated correctness particle {index}",
        )
        expected_mass_squared = expected_mass * expected_mass
        on_shell_tolerance = (
            4.0
            * MASS_SQUARED_ROUNDOFF_ULPS
            * sys.float_info.epsilon
            * _kinematic_scale(
                momentum[0] * momentum[0],
                math.fsum(component * component for component in momentum[1:]),
                expected_mass_squared,
            )
        )
        if abs(observed_mass_squared - expected_mass_squared) > on_shell_tolerance:
            raise SampleError(f"generated correctness particle {index} is off shell")
    invariant_values: list[str] = []
    squared_energy = sqrt_s * sqrt_s
    for left in range(len(point)):
        for right in range(left + 1, len(point)):
            scaled = 2.0 * _minkowski_dot(point[left], point[right]) / squared_energy
            if ((left < 2 and right >= 2) or (left >= 2 and right >= 2)) and (
                not math.isfinite(scaled) or scaled <= MIN_SCALED_PAIR_INVARIANT
            ):
                raise SampleError(
                    "generated correctness point violates the pair-invariant safety cut"
                )
            invariant_values.append(float(scaled).hex())
    return tuple(invariant_values)


def _numpy_derivation_identity(numpy_module: Any) -> dict[str, object]:
    version = getattr(numpy_module, "__version__", None)
    origin_value = getattr(numpy_module, "__file__", None)
    if not isinstance(version, str) or not version:
        raise SampleError("activated NumPy has no version identity")
    if not isinstance(origin_value, str) or not origin_value:
        raise SampleError("activated NumPy has no module origin")
    try:
        origin = Path(origin_value).resolve(strict=True)
    except OSError as error:
        raise SampleError("cannot resolve activated NumPy module origin") from error
    if not origin.is_file():
        raise SampleError("activated NumPy module origin is not a file")
    bit_generator = numpy_module.random.default_rng(
        CORRECTNESS_SEED_START
    ).bit_generator
    bit_generator_name = type(bit_generator).__name__
    if bit_generator_name != "PCG64":
        raise SampleError(
            "correctness derivation contract requires NumPy default_rng PCG64, "
            f"got {bit_generator_name!r}"
        )
    return {
        "module": "numpy",
        "version": version,
        "origin_file_name": origin.name,
        "origin_size_bytes": origin.stat().st_size,
        "origin_sha256": _sha256_file(origin),
        "random_generator": "default_rng",
        "bit_generator": bit_generator_name,
    }


def _correctness_batch(
    source: ValidationMomentaSource,
) -> tuple[tuple[tuple[FourMomentum, ...], ...], dict[str, object]]:
    """Derive eight deterministic, distinct and invariant-varied RAMBO events."""

    if not source.points:
        raise SampleError("artifact has no authenticated validation point")
    incoming, sqrt_s, masses = _validated_source_kinematics(source.points[0])
    numpy_module = importlib.import_module("numpy")
    numpy_identity = _numpy_derivation_identity(numpy_module)
    selected_points: list[tuple[FourMomentum, ...]] = []
    selected_seeds: list[int] = []
    point_sha256: list[str] = []
    invariant_sha256: list[str] = []
    observed_points: set[str] = set()
    observed_invariants: set[str] = set()
    attempts = 0
    for offset in range(MAX_CORRECTNESS_SEED_ATTEMPTS):
        attempts = offset + 1
        seed = CORRECTNESS_SEED_START + offset
        try:
            outgoing = _massive_rambo_final_state(
                len(masses) - 2,
                sqrt_s=sqrt_s,
                masses=masses[2:],
                seed=seed,
                numpy_module=numpy_module,
            )
            point = (*incoming, *outgoing)
            invariant_values = _validated_generated_point(
                point,
                sqrt_s=sqrt_s,
                masses=masses,
            )
        except (ArithmeticError, SampleError, ValueError):
            continue
        point_digest = _canonical_sha256(point)
        invariant_digest = _canonical_sha256(invariant_values)
        if point_digest in observed_points or invariant_digest in observed_invariants:
            continue
        observed_points.add(point_digest)
        observed_invariants.add(invariant_digest)
        selected_points.append(point)
        selected_seeds.append(seed)
        point_sha256.append(point_digest)
        invariant_sha256.append(invariant_digest)
        if len(selected_points) == MAX_CORRECTNESS_POINTS:
            break
    if len(selected_points) != MIN_CORRECTNESS_POINTS:
        raise SampleError(
            "could not derive exactly "
            f"{MIN_CORRECTNESS_POINTS} safe, distinct, invariant-varied "
            "correctness points within "
            f"{MAX_CORRECTNESS_SEED_ATTEMPTS} deterministic seeds"
        )
    batch = tuple(selected_points)
    derivation: dict[str, object] = {
        "kind": CORRECTNESS_POINT_DERIVATION_KIND,
        "schema_version": CORRECTNESS_POINT_DERIVATION_SCHEMA_VERSION,
        "contract": CORRECTNESS_POINT_DERIVATION_CONTRACT,
        "source_validation": dict(source.identity),
        "numpy": numpy_identity,
        "sqrt_s_f64_hex": sqrt_s.hex(),
        "external_mass_f64_hex": [mass.hex() for mass in masses],
        "final_state_mass_f64_hex": [mass.hex() for mass in masses[2:]],
        "seed_start": CORRECTNESS_SEED_START,
        "seed_attempt_limit": MAX_CORRECTNESS_SEED_ATTEMPTS,
        "attempt_count": attempts,
        "selected_seeds": selected_seeds,
        "point_sha256": point_sha256,
        "invariant_sha256": invariant_sha256,
        "batch_sha256": _canonical_sha256(batch),
    }
    derivation["sha256"] = _canonical_sha256(derivation)
    return batch, derivation


def _load_runtime(
    artifact: Path,
    *,
    process: str,
) -> tuple[object, dict[str, object]]:
    try:
        distribution = importlib.metadata.distribution("pyamplicol")
    except importlib.metadata.PackageNotFoundError as error:
        raise SampleError(
            "the selected interpreter has no pyamplicol installation"
        ) from error
    native_paths = [
        Path(str(distribution.locate_file(entry)))
        for entry in distribution.files or ()
        if Path(str(entry)).name.startswith("_rusticol.")
        and Path(str(entry)).suffix in {".so", ".pyd", ".dylib"}
    ]
    if len(native_paths) != 1:
        raise SampleError(
            "the selected pyamplicol installation must contain exactly one "
            f"native Rusticol module, found {len(native_paths)}"
        )
    native_path = native_paths[0].resolve(strict=True)
    specification = importlib.util.spec_from_file_location("_rusticol", native_path)
    if specification is None or specification.loader is None:
        raise SampleError(f"cannot load native Rusticol module: {native_path}")
    native_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(native_module)
    runtime_type = getattr(native_module, "Runtime", None)
    loader = getattr(runtime_type, "load", None)
    if not callable(loader):
        raise SampleError("native Rusticol module has no Runtime.load operation")
    runtime = loader(
        artifact,
        process=process,
        model_parameters=None,
        mute_warnings=False,
    )
    result: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": _path_identity(Path(sys.executable)),
        "package_version": distribution.version,
        "native_module": _path_identity(native_path),
    }
    build_info_entries = [
        Path(str(distribution.locate_file(entry)))
        for entry in distribution.files or ()
        if str(entry).endswith("pyamplicol/_build_info.json")
    ]
    if len(build_info_entries) == 1:
        build_info_path = build_info_entries[0].resolve(strict=True)
        build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
        if not isinstance(build_info, dict):
            raise SampleError("installed pyamplicol build provenance is not an object")
        result["build_info"] = {
            **_path_identity(build_info_path),
            "payload": build_info,
        }
        native_digest_operation = getattr(
            native_module,
            "native_build_inputs_sha256",
            None,
        )
        if callable(native_digest_operation):
            native_digest = native_digest_operation()
            result["native_build_inputs_sha256"] = native_digest
            expected_digest = build_info.get("native_build_inputs_sha256")
            if isinstance(expected_digest, str) and native_digest != expected_digest:
                raise SampleError(
                    "installed native Rusticol module does not match build provenance"
                )
    native_version_operation = getattr(native_module, "package_version", None)
    if callable(native_version_operation):
        result["native_package_version"] = native_version_operation()
    return runtime, result


def _load_public_runtime(artifact: Path, *, process: str) -> object:
    """Load the wheel's public facade for correctness-only exact evaluation."""

    package = importlib.import_module("pyamplicol")
    runtime_type = getattr(package, "Runtime", None)
    loader = getattr(runtime_type, "load", None)
    if not callable(loader):
        raise SampleError("selected pyamplicol package has no public Runtime.load")
    return loader(
        artifact,
        process=process,
        model_parameters=None,
        mute_warnings=False,
    )


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SampleError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise SampleError(f"{label} must be a JSON object: {path}")
    return payload


def _artifact_process(
    manifest: Mapping[str, Any],
    *,
    process: str,
) -> Mapping[str, Any]:
    records = manifest.get("processes")
    if not isinstance(records, list):
        raise SampleError("artifact manifest has no process records")
    normalized = " ".join(process.split())
    matches = [
        record
        for record in records
        if isinstance(record, Mapping)
        and (
            record.get("id") == process
            or (
                isinstance(record.get("expression"), str)
                and " ".join(str(record["expression"]).split()) == normalized
            )
        )
    ]
    if len(matches) != 1:
        raise SampleError(
            f"artifact must contain exactly one process matching {process!r}"
        )
    return matches[0]


def _validation_momenta(
    artifact: Path,
    *,
    process: str,
) -> ValidationMomentaSource:
    try:
        artifact_root = artifact.resolve(strict=True)
    except OSError as error:
        raise SampleError(f"cannot resolve artifact root: {artifact}") from error
    manifest = _json_object(artifact_root / "artifact.json", label="artifact manifest")
    process_record = _artifact_process(manifest, process=process)
    process_id = process_record.get("id")
    if not isinstance(process_id, str) or not process_id:
        raise SampleError("artifact process has no stable ID")
    process_expression = process_record.get("expression")
    if not isinstance(process_expression, str) or not process_expression.strip():
        raise SampleError("artifact process has no stable expression")
    expected_pdgs = process_record.get("external_pdgs")
    if (
        not isinstance(expected_pdgs, list)
        or len(expected_pdgs) < 4
        or any(
            isinstance(pdg, bool) or not isinstance(pdg, int) for pdg in expected_pdgs
        )
    ):
        raise SampleError(
            "artifact process must declare two incoming and at least two "
            "final-state integer PDGs"
        )
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise SampleError("artifact manifest has no payload inventory")
    candidates = [
        payload
        for payload in payloads
        if isinstance(payload, Mapping)
        and payload.get("role") == "validation-momenta"
        and payload.get("process_id") == process_id
    ]
    if len(candidates) != 1 or not isinstance(candidates[0].get("path"), str):
        raise SampleError(
            f"artifact process {process_id!r} has no unique validation momentum payload"
        )
    payload_record = candidates[0]
    payload_path_value = payload_record["path"]
    assert isinstance(payload_path_value, str)
    declared_sha256 = payload_record.get("sha256")
    declared_size = payload_record.get("size_bytes")
    if (
        not isinstance(declared_sha256, str)
        or len(declared_sha256) != 64
        or any(character not in "0123456789abcdef" for character in declared_sha256)
        or isinstance(declared_size, bool)
        or not isinstance(declared_size, int)
        or declared_size < 0
    ):
        raise SampleError("validation momentum payload has no valid digest identity")
    try:
        validation_path = (artifact_root / payload_path_value).resolve(strict=True)
        validation_path.relative_to(artifact_root)
    except (OSError, ValueError) as error:
        raise SampleError("validation momentum payload path is invalid") from error
    if not validation_path.is_file():
        raise SampleError("validation momentum payload is not a regular file")
    observed_size = validation_path.stat().st_size
    observed_sha256 = _sha256_file(validation_path)
    if observed_size != declared_size or observed_sha256 != declared_sha256:
        raise SampleError(
            "validation momentum payload does not match its manifest digest"
        )
    validation = _json_object(
        validation_path,
        label="validation momenta",
    )
    if (
        validation.get("kind") != "pyamplicol-rusticol-validation-momenta"
        or validation.get("schema_version") != 1
        or validation.get("process_id") != process_id
        or not isinstance(validation.get("process"), str)
        or " ".join(str(validation["process"]).split())
        != " ".join(process_expression.split())
    ):
        raise SampleError("validation momentum payload identity is invalid")
    points = validation.get("points")
    if validation.get("available") is not True or not isinstance(points, list):
        raise SampleError("artifact has no deterministic validation momenta")
    result: list[tuple[FourMomentum, ...]] = []
    for point in points:
        if not isinstance(point, list):
            raise SampleError("validation momentum point is invalid")
        vectors: list[tuple[float, float, float, float]] = []
        pdgs: list[int] = []
        for particle in point:
            if not isinstance(particle, Mapping):
                raise SampleError("validation momentum particle is invalid")
            raw_momentum = particle.get("momentum")
            if not isinstance(raw_momentum, list) or len(raw_momentum) != 4:
                raise SampleError("validation four-momentum is invalid")
            raw_pdg = particle.get("pdg")
            if isinstance(raw_pdg, bool) or not isinstance(raw_pdg, int):
                raise SampleError("validation momentum PDG is invalid")
            try:
                vector = tuple(float(component) for component in raw_momentum)
            except (TypeError, ValueError) as error:
                raise SampleError("validation momentum particle is invalid") from error
            if len(vector) != 4 or not all(math.isfinite(value) for value in vector):
                raise SampleError("validation four-momentum is non-finite")
            vectors.append((vector[0], vector[1], vector[2], vector[3]))
            pdgs.append(raw_pdg)
        if pdgs != expected_pdgs:
            raise SampleError("validation momentum PDG order does not match artifact")
        result.append(tuple(vectors))
    if not result:
        raise SampleError("artifact validation momentum payload is empty")
    source_point_sha256 = _canonical_sha256(result[0])
    return ValidationMomentaSource(
        points=tuple(result),
        identity={
            "file_sha256": observed_sha256,
            "canonical_sha256": _canonical_sha256(validation),
            "size_bytes": observed_size,
            "process_id": process_id,
            "process_expression": " ".join(process_expression.split()),
            "external_pdgs": list(expected_pdgs),
            "point_count": len(result),
            "selected_point_index": 0,
            "selected_point_sha256": source_point_sha256,
        },
    )


def _resolve_color_flows(
    artifact: Path,
    *,
    process: str,
    requested: Sequence[str],
) -> tuple[str, ...]:
    if not requested:
        return ()
    manifest = _json_object(artifact / "artifact.json", label="artifact manifest")
    process_record = _artifact_process(manifest, process=process)
    physics_path = process_record.get("physics_path")
    if not isinstance(physics_path, str) or not physics_path:
        raise SampleError("artifact process has no runtime physics payload")
    physics = _json_object(
        artifact / physics_path,
        label="runtime physics",
    )
    components = physics.get("color_components")
    if not isinstance(components, list):
        raise SampleError("runtime physics has no color component inventory")
    available = tuple(
        str(component["id"])
        for component in components
        if isinstance(component, Mapping) and isinstance(component.get("id"), str)
    )
    resolved: list[str] = []
    for value in requested:
        if value in available:
            resolved.append(value)
            continue
        try:
            ordinal = int(value, 10)
        except ValueError:
            resolved.append(value)
            continue
        if str(ordinal) != value.strip() or ordinal < 1 or ordinal > len(available):
            raise SampleError(
                f"color-flow ordinal {value!r} is out of range; "
                f"choose 1..{len(available)} or a stable color component ID"
            )
        resolved.append(available[ordinal - 1])
    return tuple(resolved)


def _resolve_helicities(
    artifact: Path,
    *,
    process: str,
    requested: Sequence[str],
) -> tuple[str, ...]:
    if not requested:
        return ()
    manifest = _json_object(artifact / "artifact.json", label="artifact manifest")
    process_record = _artifact_process(manifest, process=process)
    physics_path = process_record.get("physics_path")
    if not isinstance(physics_path, str) or not physics_path:
        raise SampleError("artifact process has no runtime physics payload")
    physics = _json_object(artifact / physics_path, label="runtime physics")
    helicities = physics.get("helicities")
    if not isinstance(helicities, list):
        raise SampleError("runtime physics has no helicity inventory")
    available = tuple(
        str(helicity["id"])
        for helicity in helicities
        if isinstance(helicity, Mapping)
        and isinstance(helicity.get("id"), str)
        and helicity.get("structural_zero") is not True
    )
    resolved: list[str] = []
    for value in requested:
        if value in available:
            resolved.append(value)
            continue
        try:
            ordinal = int(value, 10)
        except ValueError:
            resolved.append(value)
            continue
        if str(ordinal) != value.strip() or ordinal < 1 or ordinal > len(available):
            raise SampleError(
                f"non-structural-zero helicity ordinal {value!r} is out of range; "
                f"choose 1..{len(available)} or a stable helicity ID"
            )
        resolved.append(available[ordinal - 1])
    return tuple(resolved)


def _execution_mode(runtime: object) -> str:
    metadata_operation = getattr(runtime, "metadata_json", None)
    if not callable(metadata_operation):
        raise SampleError("native runtime has no metadata operation")
    try:
        metadata_payload = json.loads(metadata_operation())
    except (TypeError, json.JSONDecodeError) as error:
        raise SampleError("native runtime metadata is invalid") from error
    if not isinstance(metadata_payload, Mapping):
        raise SampleError("native runtime metadata is not an object")
    mode = metadata_payload.get("execution_mode")
    if not isinstance(mode, str):
        raise SampleError("native runtime metadata has no execution mode")
    return mode


def _calibrate_repetitions(
    timer: Any,
    batch: tuple[object, ...],
    *,
    target_seconds: float,
    sample_count: int,
    selector_arguments: Mapping[str, object],
) -> tuple[int, dict[str, object]]:
    target_per_block = target_seconds / sample_count
    repetitions = 1
    observed = _finite_positive(
        timer(batch, repetitions, **selector_arguments),
        label="native wall calibration duration",
    )
    blocks = [(repetitions, observed)]
    for _ in range(MAX_CALIBRATION_BLOCKS):
        estimate = math.ceil(repetitions * target_per_block / observed)
        candidate = min(max(estimate, 1), MAX_REPETITIONS)
        if candidate == repetitions:
            break
        repetitions = candidate
        observed = _finite_positive(
            timer(batch, repetitions, **selector_arguments),
            label="native wall calibration duration",
        )
        blocks.append((repetitions, observed))
        ratio = observed / target_per_block
        if 0.75 <= ratio <= 1.5:
            break
    return repetitions, {
        "target_seconds_per_block": target_per_block,
        "block_count": len(blocks),
        "evaluation_count": sum(block_repetitions for block_repetitions, _ in blocks),
        "blocks": [
            {
                "repetitions": block_repetitions,
                "duration_seconds": duration,
            }
            for block_repetitions, duration in blocks
        ],
    }


def _profile_components(
    profile: Mapping[str, object],
    *,
    evaluated_points: int,
    execution_mode: str,
    evaluator_timing_available: bool,
) -> tuple[float, float | None]:
    observed_mode = profile.get("execution_mode")
    if observed_mode not in (None, execution_mode):
        raise SampleError(
            "native repeated profile used execution mode "
            f"{observed_mode!r}, expected {execution_mode!r}"
        )
    wall = _finite_positive(
        profile.get("wall_time_s"),
        label="native repeated profile wall_time_s",
    )
    if not evaluator_timing_available:
        if (
            profile.get("phase_timing_scope") != ARENA_PHASE_TIMING_SCOPE
            or profile.get("evaluator_timing_available") is not False
        ):
            raise SampleError(
                "warmed Arena profile did not authenticate coarse-boundary-only timing"
            )
        return wall / evaluated_points, None
    stage = _finite_nonnegative(
        profile.get("stage_evaluator_call_time_s"),
        label="native repeated profile stage_evaluator_call_time_s",
    )
    if execution_mode == "eager":
        evaluator = stage
    else:
        amplitude = _finite_nonnegative(
            profile.get("amplitude_evaluator_call_time_s"),
            label="native repeated profile amplitude_evaluator_call_time_s",
        )
        evaluator = stage + amplitude
    return wall / evaluated_points, evaluator / evaluated_points


def _profile_operation(runtime: object, *, protocol: str) -> ProfileOperation:
    arena_profiler = getattr(runtime, "_profile_arena_repeated", None)
    if protocol == ARENA_PROFILE_PROTOCOL:
        if not callable(arena_profiler):
            raise SampleError("Arena profile protocol requires _profile_arena_repeated")
        return ProfileOperation(
            operation=arena_profiler,
            sample_pass=ARENA_PROFILE_ATTRIBUTION_SAMPLE_PASS,
            boundary=ARENA_PROFILE_BOUNDARY,
            borrowed_flat_input=True,
            preallocated_output=True,
            phase_timing_scope=ARENA_PHASE_TIMING_SCOPE,
            evaluator_timing_available=False,
        )
    if protocol != FROZEN_BASELINE_PROFILE_PROTOCOL:
        raise SampleError(f"unsupported profile protocol: {protocol!r}")
    if callable(arena_profiler):
        raise SampleError(
            "frozen pre-Arena profile protocol refuses an Arena-capable runtime"
        )
    legacy_profiler = getattr(runtime, "profile_repeated", None)
    if callable(legacy_profiler):
        return ProfileOperation(
            operation=legacy_profiler,
            sample_pass=FROZEN_BASELINE_PROFILE_ATTRIBUTION_SAMPLE_PASS,
            boundary=FROZEN_BASELINE_PROFILE_BOUNDARY,
            borrowed_flat_input=False,
            preallocated_output=False,
            phase_timing_scope=FROZEN_BASELINE_PHASE_TIMING_SCOPE,
            evaluator_timing_available=True,
        )
    raise SampleError("frozen pre-Arena profile protocol requires profile_repeated")


def _complex_pair(value: object, *, label: str) -> list[float]:
    if isinstance(value, bool):
        raise SampleError(f"{label} must be numeric")
    try:
        converted = complex(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SampleError(f"{label} must be numeric") from error
    if not math.isfinite(converted.real) or not math.isfinite(converted.imag):
        raise SampleError(f"{label} must be finite")
    return [float(converted.real), float(converted.imag)]


def _resolved_values_json(value: object, *, label: str) -> object:
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _resolved_values_json(entry, label=f"{label}[{index}]")
            for index, entry in enumerate(value)
        ]
    return _complex_pair(value, label=label)


def _warmed_numerical_result(
    evaluator: Any,
    resolved_evaluator: Any,
    validation_batch: tuple[object, ...],
    *,
    selector_arguments: Mapping[str, object],
) -> dict[str, object]:
    selector_values: dict[str, list[str]] = {}
    for key in ("helicities", "color_flows"):
        raw_selectors = selector_arguments.get(key)
        if raw_selectors is None:
            selector_values[key] = []
        elif (
            isinstance(raw_selectors, Sequence)
            and not isinstance(
                raw_selectors,
                (str, bytes, bytearray),
            )
            and all(isinstance(value, str) for value in raw_selectors)
        ):
            selector_values[key] = list(raw_selectors)
        else:
            raise SampleError(f"native warmed evaluation has invalid {key}")
    raw_values = evaluator(validation_batch, **selector_arguments)
    if not isinstance(raw_values, Sequence) or isinstance(
        raw_values,
        (str, bytes, bytearray),
    ):
        raise SampleError("native warmed evaluation did not return a value sequence")
    values: list[float] = []
    for raw_value in raw_values:
        real, imaginary = _complex_pair(
            raw_value,
            label="native warmed evaluation value",
        )
        if not math.isclose(imaginary, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
            raise SampleError("native warmed evaluation returned a complex total")
        values.append(real)
    if len(values) != len(validation_batch):
        raise SampleError(
            "native warmed evaluation returned an unexpected number of values"
        )
    resolved = resolved_evaluator(validation_batch, **selector_arguments)
    resolved_total_operation = getattr(resolved, "total", None)
    if not callable(resolved_total_operation):
        raise SampleError("native resolved evaluation has no total operation")
    raw_resolved_totals = resolved_total_operation()
    if not isinstance(raw_resolved_totals, Sequence) or isinstance(
        raw_resolved_totals,
        (str, bytes, bytearray),
    ):
        raise SampleError("native resolved evaluation returned invalid totals")
    resolved_totals = [
        _complex_pair(value, label="native resolved total")
        for value in raw_resolved_totals
    ]
    if len(resolved_totals) != len(values):
        raise SampleError("native resolved total count differs from evaluate()")
    for value, (resolved_real, resolved_imag) in zip(
        values,
        resolved_totals,
        strict=True,
    ):
        if not math.isclose(
            value,
            resolved_real,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ) or not math.isclose(resolved_imag, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
            raise SampleError(
                "native evaluate() result disagrees with evaluate_resolved().total()"
            )
    resolved_values = getattr(resolved, "values", None)
    resolved_shape = getattr(resolved, "shape", None)
    helicity_ids = getattr(resolved, "helicity_ids", None)
    color_ids = getattr(resolved, "color_ids", None)
    if (
        not isinstance(resolved_shape, Sequence)
        or isinstance(resolved_shape, (str, bytes, bytearray))
        or not all(isinstance(value, int) and value >= 0 for value in resolved_shape)
        or not isinstance(helicity_ids, Sequence)
        or isinstance(helicity_ids, (str, bytes, bytearray))
        or not all(isinstance(value, str) for value in helicity_ids)
        or not isinstance(color_ids, Sequence)
        or isinstance(color_ids, (str, bytes, bytearray))
        or not all(isinstance(value, str) for value in color_ids)
    ):
        raise SampleError("native resolved evaluation metadata is invalid")
    return {
        "precision": selector_arguments.get("precision"),
        "point_count": len(validation_batch),
        "distinct_point_count": len(validation_batch),
        "point_sha256": [_canonical_sha256(point) for point in validation_batch],
        "batch_sha256": _canonical_sha256(validation_batch),
        "values_f64": values,
        "values_f64_hex": [value.hex() for value in values],
        "helicities": selector_values["helicities"],
        "color_flows": selector_values["color_flows"],
        "resolved": {
            "shape": list(resolved_shape),
            "helicity_ids": list(helicity_ids),
            "color_ids": list(color_ids),
            "totals_complex": resolved_totals,
            "values_complex": _resolved_values_json(
                resolved_values,
                label="native resolved value",
            ),
        },
    }


def sample(arguments: argparse.Namespace) -> dict[str, object]:
    dependency_site_identity = _activate_dependency_site(
        getattr(arguments, "dependency_site", None)
    )
    artifact = arguments.artifact.expanduser().resolve(strict=True)
    load_started = time.perf_counter()
    runtime, runtime_identity = _load_runtime(
        artifact,
        process=arguments.process,
    )
    cold_load_seconds = time.perf_counter() - load_started
    execution_mode = getattr(arguments, "execution_mode", "compiled")
    if execution_mode not in EXECUTION_MODES:
        raise SampleError(f"unsupported execution mode: {execution_mode!r}")
    observed_execution_mode = _execution_mode(runtime)
    if observed_execution_mode != execution_mode:
        raise SampleError(
            "regression sampling requires a "
            f"{execution_mode} runtime, got {observed_execution_mode!r}"
        )
    runtime_identity["dependency_site"] = dependency_site_identity
    timer = getattr(runtime, "_benchmark_f64_wall_time", None)
    profile_protocol = getattr(arguments, "profile_protocol", None)
    profile_operation = _profile_operation(runtime, protocol=profile_protocol)
    profiler = profile_operation.operation
    evaluate_once = getattr(runtime, "evaluate", None)
    evaluate_resolved_once = getattr(runtime, "evaluate_resolved", None)
    if not callable(timer):
        raise SampleError("native unprofiled wall timer is unavailable")
    if not callable(evaluate_once):
        raise SampleError("native f64 evaluator is unavailable")
    if not callable(evaluate_resolved_once):
        raise SampleError("native resolved f64 evaluator is unavailable")
    validation_source = _validation_momenta(artifact, process=arguments.process)
    batch = _benchmark_batch(validation_source.points, arguments.batch_size)
    correctness_batch, correctness_point_derivation = _correctness_batch(
        validation_source
    )
    resolved_helicities = _resolve_helicities(
        artifact,
        process=arguments.process,
        requested=arguments.helicity,
    )
    helicities = resolved_helicities or None
    resolved_color_flows = _resolve_color_flows(
        artifact,
        process=arguments.process,
        requested=arguments.color_flow,
    )
    color_flows = resolved_color_flows or None
    selector_arguments: dict[str, object] = {
        "helicities": helicities,
        "color_flows": color_flows,
        "precision": 16,
    }
    profile_arguments = {**selector_arguments, "include_values": False}

    for _ in range(arguments.warmup_runs):
        _finite_positive(
            timer(batch, 1, **selector_arguments),
            label="native wall warmup duration",
        )
        warmup_profile = profiler(batch, 1, **profile_arguments)
        if not isinstance(warmup_profile, Mapping):
            raise SampleError("native repeated warmup profile is not a mapping")

    repetitions, calibration = _calibrate_repetitions(
        timer,
        batch,
        target_seconds=arguments.target_runtime,
        sample_count=arguments.minimum_samples,
        selector_arguments=selector_arguments,
    )
    evaluated_points_per_block = repetitions * len(batch)
    headline_samples: list[float] = []
    profiled_wall_samples: list[float] = []
    evaluator_samples: list[float] = []
    raw_profiles: list[object] = []
    headline_durations: list[float] = []
    attribution_elapsed = 0.0
    for _ in range(arguments.minimum_samples):
        duration = _finite_positive(
            timer(batch, repetitions, **selector_arguments),
            label="native headline wall duration",
        )
        headline_durations.append(duration)
        headline_samples.append(duration / evaluated_points_per_block)

        profile_started = time.perf_counter()
        raw_profile = profiler(batch, repetitions, **profile_arguments)
        attribution_elapsed += time.perf_counter() - profile_started
        if not isinstance(raw_profile, Mapping):
            raise SampleError("native repeated profile is not a mapping")
        profile_wall, evaluator_time = _profile_components(
            raw_profile,
            evaluated_points=evaluated_points_per_block,
            execution_mode=execution_mode,
            evaluator_timing_available=profile_operation.evaluator_timing_available,
        )
        profiled_wall_samples.append(profile_wall)
        if evaluator_time is not None:
            evaluator_samples.append(evaluator_time)
        raw_profiles.append(_json_value(raw_profile))

    numerical_result = _warmed_numerical_result(
        evaluate_once,
        evaluate_resolved_once,
        correctness_batch,
        selector_arguments=selector_arguments,
    )
    precision32_result = None
    if getattr(arguments, "include_precision32", False):
        public_runtime = _load_public_runtime(
            artifact,
            process=arguments.process,
        )
        public_evaluate = getattr(public_runtime, "evaluate", None)
        public_evaluate_resolved = getattr(public_runtime, "evaluate_resolved", None)
        if not callable(public_evaluate) or not callable(public_evaluate_resolved):
            raise SampleError(
                "public runtime has no precision-32 evaluation operations"
            )
        precision32_result = _warmed_numerical_result(
            public_evaluate,
            public_evaluate_resolved,
            correctness_batch,
            selector_arguments={**selector_arguments, "precision": 32},
        )
    sample_count = len(headline_samples)
    measured_evaluations = sample_count * repetitions
    measured_points = measured_evaluations * len(batch)
    headline_mean = float(statistics.fmean(headline_samples))
    evaluator_mean = (
        float(statistics.fmean(evaluator_samples)) if evaluator_samples else None
    )
    timing_breakdown = {
        "sample_count": sample_count,
        "execution_mode": execution_mode,
        "wall_time": _component(profiled_wall_samples),
        "evaluator_call_time": (
            _component(evaluator_samples) if evaluator_samples else None
        ),
        "raw_profile_samples": raw_profiles,
    }
    return {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "wall_time_per_point": headline_mean,
        "wall_time_samples_seconds_per_point": headline_samples,
        "sample_count": sample_count,
        "repetitions_per_sample": repetitions,
        "interrupted": False,
        "uncertainty": _statistics(headline_samples),
        "evaluator_time_per_point": evaluator_mean,
        "evaluator_uncertainty": (
            _statistics(evaluator_samples) if evaluator_samples else None
        ),
        "correctness_point_derivation": correctness_point_derivation,
        "warmed_numerical_result": numerical_result,
        "precision32_numerical_result": precision32_result,
        "cold_load_seconds": cold_load_seconds,
        "timing_breakdown": timing_breakdown,
        "environment": {
            "wall_time_source": NATIVE_WALL_TIME_SOURCE,
            "wall_time_sample_pass": NATIVE_WALL_TIME_SAMPLE_PASS,
            "evaluator_time_sample_pass": profile_operation.sample_pass,
            "timing_breakdown_sample_pass": profile_operation.sample_pass,
            "profile_attribution_boundary": profile_operation.boundary,
            "profile_attribution_borrowed_flat_input": (
                profile_operation.borrowed_flat_input
            ),
            "profile_attribution_preallocated_output": (
                profile_operation.preallocated_output
            ),
            "profile_attribution_phase_timing_scope": (
                profile_operation.phase_timing_scope
            ),
            "profile_attribution_evaluator_timing_available": (
                profile_operation.evaluator_timing_available
            ),
            "profile_protocol": profile_protocol,
            "timing_sample_contract": PAIRED_TIMING_SAMPLE_CONTRACT,
            "profile_attribution_paired_with_headline": True,
            "profile_attribution_identical_batch": True,
            "profile_attribution_identical_repetitions": True,
            "execution_mode": execution_mode,
            "batch_size": len(batch),
            "precision": 16,
            "interrupted": False,
            "completed_sample_count": sample_count,
            "planned_sample_count": arguments.minimum_samples,
            "measured_evaluation_count": measured_evaluations,
            "measured_point_count": measured_points,
            "native_profile_sample_count": len(raw_profiles),
            "native_profile_sample_limit": arguments.minimum_samples,
            "native_profile_repetitions_per_sample": repetitions,
            "native_profile_points_per_sample": evaluated_points_per_block,
            "native_profile_calls_per_block": 1.0,
            "profile_attribution_evaluation_count": measured_evaluations,
            "profile_attribution_point_count": measured_points,
            "elapsed_seconds": sum(headline_durations),
            "headline_block_durations_seconds": headline_durations,
            "profile_attribution_elapsed_seconds": attribution_elapsed,
            "batch_sha256": _canonical_sha256(batch),
            "helicities": list(helicities or ()),
            "color_flows": list(color_flows or ()),
            "calibration": calibration,
            "runtime_identity": runtime_identity,
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("artifact", type=Path)
    result.add_argument("--process", required=True)
    result.add_argument(
        "--execution-mode",
        choices=EXECUTION_MODES,
        default="compiled",
    )
    result.add_argument(
        "--profile-protocol",
        choices=PROFILE_PROTOCOLS,
        required=True,
        help=(
            "require the current Arena profiler or explicitly measure the "
            "frozen pre-Arena baseline; no automatic fallback is allowed"
        ),
    )
    result.add_argument(
        "--dependency-site",
        type=Path,
        help=(
            "append a dependency-only site after the measured installation; "
            "the helper fails if this changes the pyamplicol package origin"
        ),
    )
    result.add_argument("--target-runtime", type=_positive_float, required=True)
    result.add_argument("--batch-size", type=_positive_int, required=True)
    result.add_argument("--minimum-samples", type=_at_least_five, required=True)
    result.add_argument("--warmup-runs", type=_positive_int, required=True)
    result.add_argument(
        "--include-precision32",
        action="store_true",
        help="capture one correctness-only precision-32 numerical result",
    )
    result.add_argument("--helicity", action="append", default=[])
    result.add_argument("--color-flow", action="append", default=[])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    if tuple(sys.argv[1:] if argv is None else argv) == ("--installation-identity",):
        try:
            identity = _installed_distribution_identity()
        except (OSError, SampleError, ValueError) as error:
            print(f"compiled-mode-sample: {error}", file=sys.stderr)
            return 2
        print(json.dumps(identity, indent=2, sort_keys=True, allow_nan=False))
        return 0
    arguments = parser().parse_args(argv)
    try:
        result = sample(arguments)
    except (OSError, SampleError, ValueError) as error:
        print(f"compiled-mode-sample: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
