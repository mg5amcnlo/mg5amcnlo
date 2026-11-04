# SPDX-License-Identifier: 0BSD
"""Evaluate totals and resolved components through the public Runtime facade."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pyamplicol import Runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("momenta", type=Path)
    parser.add_argument("--process")
    parser.add_argument("--parameters", type=Path)
    parser.add_argument(
        "--set-parameter",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="apply a direct real or complex UFO-parameter override",
    )
    parser.add_argument("--helicity", action="append")
    parser.add_argument("--color-flow", action="append")
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_complex(value: complex) -> dict[str, float]:
    return {"real": value.real, "imag": value.imag}


def _flatten(values: Any) -> list[complex]:
    if isinstance(values, (list, tuple)):
        return [item for value in values for item in _flatten(value)]
    return [complex(values)]


def _parameter_overrides(values: list[str]) -> dict[str, complex]:
    result: dict[str, complex] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name or not raw:
            raise SystemExit(f"invalid --set-parameter value: {value!r}")
        try:
            result[name] = complex(raw)
        except ValueError as error:
            raise SystemExit(f"invalid parameter value {raw!r}") from error
    return result


def main() -> int:
    args = _parser().parse_args()
    parameters = _read_json(args.parameters) if args.parameters else None
    momenta = _read_json(args.momenta)
    runtime = Runtime.load(
        args.artifact,
        process=args.process,
        model_parameters=parameters,
    )
    if overrides := _parameter_overrides(args.set_parameter):
        runtime.set_model_parameters(overrides)
    selectors = {
        "helicities": args.helicity,
        "color_flows": args.color_flow,
    }
    compatibility_totals = runtime.evaluate(momenta, **selectors)
    resolved = runtime.evaluate_resolved(momenta, **selectors)
    resolved_totals = resolved.total()
    if len(compatibility_totals) != len(resolved_totals) or any(
        not math.isclose(total.real, check.real, rel_tol=1e-12, abs_tol=1e-15)
        or not math.isclose(total.imag, check.imag, rel_tol=1e-12, abs_tol=1e-15)
        for total, check in zip(compatibility_totals, resolved_totals, strict=True)
    ):
        raise RuntimeError("resolved components do not reproduce summed values")

    print(
        json.dumps(
            {
                "color_accuracy": resolved.color_accuracy,
                "helicity_ids": resolved.helicity_ids,
                "color_flow_ids": resolved.color_flow_ids,
                "shape": list(resolved.shape),
                "values": [_json_complex(value) for value in _flatten(resolved.values)],
                "resolved_sum": [_json_complex(value) for value in resolved_totals],
                "compatibility_total": [
                    _json_complex(value) for value in compatibility_totals
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
