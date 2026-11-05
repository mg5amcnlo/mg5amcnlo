#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Compare compiled and eager artifacts in an isolated Symbolica process."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ComparisonError(RuntimeError):
    """Raised when comparison inputs or results are malformed."""


def _validation_momenta(
    artifact: Path, process_id: str
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    path = artifact / "processes" / process_id / "validation-momenta.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        points = payload["points"]
        return tuple(
            tuple(
                tuple(float(component) for component in particle["momentum"])
                for particle in point
            )
            for point in points
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ComparisonError(f"invalid validation momenta at {path}") from error


def _relative_difference(left: complex, right: complex) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


def _complex_comparison(left: object, right: object) -> dict[str, object]:
    left_value = complex(left)
    right_value = complex(right)
    absolute = abs(left_value - right_value)
    relative = _relative_difference(left_value, right_value)
    return {
        "left": [left_value.real, left_value.imag],
        "right": [right_value.real, right_value.imag],
        "absolute_difference": absolute,
        "relative_difference": relative,
        "passes": absolute <= 1.0e-15 or relative <= 1.0e-12,
    }


def _resolved_comparison(left: Any, right: Any) -> dict[str, object]:
    left_helicities = tuple(left.helicity_ids)
    right_helicities = tuple(right.helicity_ids)
    left_colors = tuple(left.color_ids)
    right_colors = tuple(right.color_ids)
    left_values = tuple(left.values)
    right_values = tuple(right.values)
    identifiers_match = (
        left_helicities == right_helicities and left_colors == right_colors
    )
    shape_matches = len(left_values) == len(right_values)
    comparisons: list[dict[str, object]] = []
    if shape_matches:
        for left_point, right_point in zip(left_values, right_values, strict=True):
            left_rows = tuple(left_point)
            right_rows = tuple(right_point)
            if len(left_rows) != len(right_rows):
                shape_matches = False
                break
            for left_row, right_row in zip(left_rows, right_rows, strict=True):
                left_components = tuple(left_row)
                right_components = tuple(right_row)
                if len(left_components) != len(right_components):
                    shape_matches = False
                    break
                comparisons.extend(
                    _complex_comparison(left_value, right_value)
                    for left_value, right_value in zip(
                        left_components,
                        right_components,
                        strict=True,
                    )
                )
            if not shape_matches:
                break
    maximum_absolute = max(
        (float(item["absolute_difference"]) for item in comparisons),
        default=0.0,
    )
    maximum_relative = max(
        (float(item["relative_difference"]) for item in comparisons),
        default=0.0,
    )
    return {
        "helicity_ids_match": left_helicities == right_helicities,
        "color_ids_match": left_colors == right_colors,
        "shape_matches": shape_matches,
        "point_count": len(left_values),
        "component_count": len(comparisons),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "passes": identifiers_match
        and shape_matches
        and bool(comparisons)
        and all(bool(item["passes"]) for item in comparisons),
    }


def compare_artifacts(
    eager_artifact: Path,
    compiled_artifact: Path,
    process_id: str,
    eager_selectors: Mapping[str, str],
    *,
    compiled_selectors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare f64 and exact resolved results from two artifacts."""

    from pyamplicol import Runtime

    momenta = _validation_momenta(eager_artifact, process_id)

    def selector_kwargs(selectors: Mapping[str, str]) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        if "color_flow" in selectors:
            kwargs["color_flows"] = (selectors["color_flow"],)
        if "helicity" in selectors:
            kwargs["helicities"] = (selectors["helicity"],)
        return kwargs

    eager_kwargs = selector_kwargs(eager_selectors)
    compiled_kwargs = selector_kwargs(
        eager_selectors if compiled_selectors is None else compiled_selectors
    )
    eager_runtime = Runtime.load(eager_artifact)
    compiled_runtime = Runtime.load(compiled_artifact)
    eager_total = eager_runtime.evaluate(momenta, **eager_kwargs)
    compiled_total = compiled_runtime.evaluate(momenta, **compiled_kwargs)
    total_comparisons = tuple(
        _complex_comparison(eager_value, compiled_value)
        for eager_value, compiled_value in zip(
            eager_total,
            compiled_total,
            strict=True,
        )
    )
    eager_resolved = eager_runtime.evaluate_resolved(momenta, **eager_kwargs)
    compiled_resolved = compiled_runtime.evaluate_resolved(
        momenta, **compiled_kwargs
    )
    resolved = _resolved_comparison(eager_resolved, compiled_resolved)
    eager_exact = eager_runtime.evaluate_resolved(
        momenta, precision=32, **eager_kwargs
    )
    compiled_exact = compiled_runtime.evaluate_resolved(
        momenta,
        precision=32,
        **compiled_kwargs,
    )
    exact = _resolved_comparison(eager_exact, compiled_exact)
    eager_reduction = tuple(
        _complex_comparison(total, resolved_total)
        for total, resolved_total in zip(
            eager_total,
            eager_resolved.total(),
            strict=True,
        )
    )
    compiled_reduction = tuple(
        _complex_comparison(total, resolved_total)
        for total, resolved_total in zip(
            compiled_total,
            compiled_resolved.total(),
            strict=True,
        )
    )
    return {
        "total": total_comparisons,
        "resolved_f64": resolved,
        "resolved_precision32": exact,
        "eager_resolved_sum": eager_reduction,
        "compiled_resolved_sum": compiled_reduction,
        "passes": bool(total_comparisons)
        and all(bool(item["passes"]) for item in total_comparisons)
        and bool(resolved["passes"])
        and bool(exact["passes"])
        and all(bool(item["passes"]) for item in eager_reduction)
        and all(bool(item["passes"]) for item in compiled_reduction),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--eager-artifact", type=Path, required=True)
    result.add_argument("--compiled-artifact", type=Path, required=True)
    result.add_argument("--process", required=True)
    result.add_argument("--color-flow")
    result.add_argument("--helicity")
    result.add_argument(
        "--compiled-selectors",
        choices=("same", "none"),
        default="same",
        help="reuse eager selectors or leave a specialized artifact unselected",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    selectors = {
        name: value
        for name, value in (
            ("color_flow", arguments.color_flow),
            ("helicity", arguments.helicity),
        )
        if value is not None
    }
    compiled_selectors = None if arguments.compiled_selectors == "same" else {}
    try:
        result = compare_artifacts(
            arguments.eager_artifact,
            arguments.compiled_artifact,
            arguments.process,
            selectors,
            compiled_selectors=compiled_selectors,
        )
    except (ComparisonError, OSError, ValueError) as error:
        print(f"eager-artifact-compare: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
