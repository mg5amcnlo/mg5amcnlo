#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Evaluate a generated process through pyAmpliCol's installed Python API."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any


def _load_pyamplicol() -> Any:
    try:
        return importlib.import_module("pyamplicol")
    except ModuleNotFoundError as error:
        if error.name != "pyamplicol":
            raise
        previous = os.environ.get("_PYAMPLICOL_API_BOOTSTRAP")
        if previous is not None:
            raise SystemExit(
                "pyamplicol remained unavailable after retrying with " + previous
            ) from error
        script = Path(__file__).resolve()
        layouts = (
            (Path("bin/python"), Path("bin/rusticol-config")),
            (Path("Scripts/python.exe"), Path("Scripts/rusticol-config.exe")),
        )
        for parent in script.parents:
            for python_relative, config_relative in layouts:
                python = parent / ".venv" / python_relative
                config = parent / ".venv" / config_relative
                if (
                    not python.is_file()
                    or not os.access(python, os.X_OK)
                    or not config.is_file()
                    or not os.access(config, os.X_OK)
                ):
                    continue
                environment = os.environ.copy()
                environment["_PYAMPLICOL_API_BOOTSTRAP"] = str(python)
                os.execve(
                    python,
                    (str(python), str(script), *sys.argv[1:]),
                    environment,
                )
        raise SystemExit(
            "pyamplicol is not importable; activate its environment or place this "
            "artifact below a checkout containing .venv from `just dev-install`"
        ) from error


pyamplicol = _load_pyamplicol()


def _validation_point(path: Path, process_id: str, precision: int):
    if not path.is_file():
        return None
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines or lines[0] != "RUSTICOL_VALIDATION_POINTS_V1":
        raise SystemExit("unsupported validation_points.dat format")
    for line in lines[1:]:
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 2 or fields[0] != process_id:
            continue
        external_count = int(fields[1])
        components = fields[2:]
        if len(components) != external_count * 4:
            raise SystemExit(f"invalid validation row for {process_id!r}")
        convert = float if precision == 16 else Decimal
        return [
            [convert(value) for value in components[index : index + 4]]
            for index in range(0, len(components), 4)
        ]
    return None


def _reorder_point(
    point: list[list[Any]], permutation: tuple[int, ...]
) -> list[list[Any]]:
    if len(point) != len(permutation) or sorted(permutation) != list(range(len(point))):
        raise SystemExit("runtime returned an invalid external permutation")
    reordered: list[list[Any] | None] = [None] * len(point)
    for representative_index, public_index in enumerate(permutation):
        reordered[public_index] = point[representative_index]
    if any(vector is None for vector in reordered):
        raise SystemExit("runtime returned an incomplete external permutation")
    return [vector for vector in reordered if vector is not None]


def _kinematics(path: str, precision: int, external_count: int) -> list[list[Any]]:
    source = Path(path).expanduser().resolve(strict=True)

    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_float=Decimal if precision != 16 else float,
            parse_int=Decimal if precision != 16 else int,
            parse_constant=invalid_constant,
        )
    except (json.JSONDecodeError, OSError, ValueError) as error:
        raise SystemExit(f"could not read kinematics JSON {source}: {error}") from error
    if (
        isinstance(payload, list)
        and len(payload) == 1
        and isinstance(payload[0], list)
        and payload[0]
        and isinstance(payload[0][0], list)
    ):
        payload = payload[0]
    if not isinstance(payload, list) or len(payload) != external_count:
        raise SystemExit(
            f"kinematics JSON must contain one [{external_count}][4] point"
        )
    convert = float if precision == 16 else Decimal
    point: list[list[Any]] = []
    for leg_index, raw_leg in enumerate(payload):
        if not isinstance(raw_leg, list) or len(raw_leg) != 4:
            raise SystemExit(
                f"kinematics leg {leg_index} must contain [E, px, py, pz]"
            )
        leg: list[Any] = []
        for component_index, raw in enumerate(raw_leg):
            if isinstance(raw, bool) or not isinstance(raw, (int, float, Decimal, str)):
                raise SystemExit(
                    f"kinematics leg {leg_index} component {component_index} "
                    "must be a number or decimal string"
                )
            try:
                value = convert(raw)
            except (TypeError, ValueError) as error:
                raise SystemExit(
                    f"kinematics leg {leg_index} component {component_index} "
                    "is not a decimal value"
                ) from error
            finite = (
                math.isfinite(value)
                if isinstance(value, float)
                else value.is_finite()
            )
            if not finite:
                raise SystemExit("kinematics components must be finite")
            leg.append(value)
        point.append(leg)
    return point


def _repeat_point(point: list[list[Any]], count: int) -> list[list[list[Any]]]:
    return [point for _ in range(count)]


def _flatten(values: Any) -> list[Any]:
    if hasattr(values, "reshape"):
        return values.reshape(-1).tolist()
    output: list[Any] = []

    def visit(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for entry in value:
                visit(entry)
        else:
            output.append(value)

    visit(values)
    return output


def _real(value: Any) -> Any:
    if isinstance(value, Decimal):
        return value
    converted = complex(value)
    if abs(converted.imag) > 1.0e-14 * max(1.0, abs(converted.real)):
        raise SystemExit("matrix-element output unexpectedly has an imaginary part")
    return converted.real


def _json_scalar(value: Any, precision: int) -> float | str:
    real = _real(value)
    return float(real) if precision == 16 else str(real)


def _parameter_component(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit("model-parameter components must be JSON numbers")
    component = float(value)
    if not math.isfinite(component):
        raise SystemExit("model-parameter components must be finite")
    return component


def _parameter_value(value: Any) -> complex | float:
    if isinstance(value, list):
        if len(value) != 2:
            raise SystemExit("model-parameter arrays must contain [real, imaginary]")
        return complex(
            _parameter_component(value[0]),
            _parameter_component(value[1]),
        )
    return _parameter_component(value)


def _model_parameters(path: str | None) -> dict[str, complex | float] | None:
    if path is None:
        return None
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("model-parameter JSON must be an object")
    return {str(name): _parameter_value(value) for name, value in payload.items()}


def _physics_payload(physics: pyamplicol.ProcessPhysics) -> dict[str, Any]:
    colors = [
        {"id": item.id, "kind": "lc-flow", "word": list(item.word)}
        for item in physics.color_flows
    ]
    colors.extend(
        {"id": item.id, "kind": "contracted-color", "word": []}
        for item in physics.contracted_color_components
    )
    return {
        "process": physics.process,
        "process_key": physics.process_id,
        "color_accuracy": physics.color_accuracy,
        "external_particles": [
            {"index": item.index, "pdg": item.pdg_id}
            for item in physics.external_particles
        ],
        "helicities": [
            {"id": item.id, "helicities": list(item.values)}
            for item in physics.helicities
        ],
        "colors": colors,
    }


def _profile(
    runtime: pyamplicol.Runtime,
    point: list[list[Any]],
    precision: int,
    target_runtime: float,
    batch_size: int,
) -> dict[str, float | int]:
    samples: list[float] = []
    elapsed = 0.0
    while len(samples) < 8 or elapsed < target_runtime:
        batch = _repeat_point(point, batch_size)
        started = time.perf_counter()
        runtime.evaluate(batch, precision=precision)
        wall = time.perf_counter() - started
        elapsed += wall
        samples.append(wall / batch_size)
    return {
        "samples": len(samples) * batch_size,
        "wall_us_per_point": statistics.mean(samples) * 1.0e6,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolved pyAmpliCol Python API example"
    )
    parser.add_argument(
        "--process",
        help="stable process/alias ID or exact concrete process expression",
    )
    parser.add_argument("--model-parameters")
    parser.add_argument(
        "--kinematics",
        help="JSON point in the external-particle order supplied by --process",
    )
    parser.add_argument(
        "--set-parameter",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "REAL", "IMAG"),
    )
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--target-runtime", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[2]
    runtime = pyamplicol.Runtime.load(
        root,
        process=args.process,
        model_parameters=_model_parameters(args.model_parameters),
    )
    if args.set_parameter:
        runtime.set_model_parameters(
            {
                name: complex(float(real), float(imaginary))
                for name, real, imaginary in args.set_parameter
            }
        )
    physics = runtime.physics
    metadata = _physics_payload(physics)
    if args.kinematics is None:
        point = _validation_point(
            root / "API" / "validation_points.dat",
            runtime.representative_process_key,
            args.precision,
        )
        if point is not None:
            point = _reorder_point(point, runtime.external_permutation)
    else:
        point = _kinematics(
            args.kinematics,
            args.precision,
            len(physics.external_particles),
        )
    if point is None:
        payload = {
            "language": "python",
            "available": False,
            "diagnostic": "no bundled validation point is available",
            **metadata,
        }
        if args.json:
            print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        else:
            print(f"process: {physics.process}")
            print("no bundled validation point is available; metadata load succeeded")
        return 0

    batch = _repeat_point(point, 1)
    resolved = runtime.evaluate_resolved(batch, precision=args.precision)
    total = runtime.evaluate(batch, precision=args.precision)
    values = _flatten(resolved.values)
    explicit_total = _flatten(resolved.total())
    compatibility_total = _flatten(total)
    payload: dict[str, Any] = {
        "language": "python",
        "available": True,
        "precision": args.precision,
        **metadata,
        "shape": list(resolved.shape),
        "values": [_json_scalar(value, args.precision) for value in values],
        "resolved_sum": [
            _json_scalar(value, args.precision) for value in explicit_total
        ],
        "compatibility_total": [
            _json_scalar(value, args.precision) for value in compatibility_total
        ],
    }
    if args.profile:
        payload["profile"] = _profile(
            runtime,
            point,
            args.precision,
            max(args.target_runtime, 0.0),
            max(args.batch_size, 1),
        )
    if args.json:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        return 0

    print(f"process: {physics.process} [{physics.process_id}]")
    print(f"resolved shape: {resolved.shape}")
    color_items = metadata["colors"]
    color_count = len(color_items)
    for helicity_index, helicity in enumerate(metadata["helicities"]):
        for color_index, color in enumerate(color_items):
            offset = helicity_index * color_count + color_index
            print(f"  {helicity['id']}  {color['id']}  {values[offset]}")
    print(f"explicit resolved sum: {explicit_total[0]}")
    print(f"compatibility total:   {compatibility_total[0]}")
    if not math.isclose(
        float(_real(explicit_total[0])),
        float(_real(compatibility_total[0])),
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise SystemExit("resolved components do not reproduce the compatibility total")
    if args.profile:
        profile = payload["profile"]
        assert isinstance(profile, dict)
        print(f"timing: {profile['wall_us_per_point']:.6g} us/point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
