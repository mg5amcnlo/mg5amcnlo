# SPDX-License-Identifier: 0BSD
"""Compare scalar HEFT matrix elements with MadGraph standalone ``smatrix``.

This developer command prepares the authenticated upstream UFO locally for
MadGraph and evaluates pyAmpliCol's independently packaged ``built-in-sm-heft``
model. The UFO and generated artifacts remain under ignored developer
directories.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyamplicol import (  # noqa: E402
    Generator,
    ModelSource,
    ProcessRequest,
    ProcessSet,
    Runtime,
)
from pyamplicol.config import (  # noqa: E402
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    OutputConfig,
    ProcessConfig,
    RunConfig,
)
from tools.developer.heft_ufo import (  # noqa: E402
    DEFAULT_HEFT_UFO_ROOT,
    prepare_heft_ufo,
)
from tools.developer.madgraph_correctness import (  # noqa: E402
    StandaloneMadGraphRunner,
    set_parameter_card_values,
)

DEFAULT_OUTPUT_ROOT = ROOT / ".artifacts" / "heft-madgraph-acceptance"
RELATIVE_TOLERANCE = 1.0e-10
ABSOLUTE_TOLERANCE = 1.0e-300
VALIDATION_SEED = 101
_HEFT_INPUTS = ("aEWM1", "Gf", "aS", "MZ", "MT", "MH")


@dataclass(frozen=True, slots=True)
class ProcessSpec:
    process_id: str
    expression: str


PROCESSES = (
    ProcessSpec("gg_h", "g g > H"),
    ProcessSpec("gg_hg", "g g > H g"),
    ProcessSpec("gg_hgg", "g g > H g g"),
    ProcessSpec("gg_hggg", "g g > H g g g"),
)


class HEFTAcceptanceError(RuntimeError):
    """The scalar HEFT correctness gate could not be established."""


def _validation_momenta(
    artifact: Path,
    process_id: str,
) -> tuple[tuple[float, float, float, float], ...]:
    path = artifact / "processes" / process_id / "validation-momenta.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("kind") != "pyamplicol-rusticol-validation-momenta"
            or payload.get("process_id") != process_id
            or len(payload.get("points", ())) != 1
        ):
            raise ValueError("wrong validation metadata")
        point = payload["points"][0]
        rows = tuple(
            tuple(float(component) for component in particle["momentum"])
            for particle in point
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise HEFTAcceptanceError(
            f"invalid validation point for seed-{VALIDATION_SEED} process set "
            f"member {process_id}"
        ) from error
    if any(
        len(row) != 4 or any(not math.isfinite(component) for component in row)
        for row in rows
    ):
        raise HEFTAcceptanceError(f"non-finite validation momentum for {process_id}")
    return rows  # type: ignore[return-value]


def _pyamplicol_value(
    artifact: Path,
    process_id: str,
    momenta: tuple[tuple[float, float, float, float], ...],
) -> float:
    value = complex(
        Runtime.load(artifact, process=process_id).evaluate(
            (momenta,),
            precision=16,
        )[0]
    )
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise HEFTAcceptanceError(
            f"pyAmpliCol returned a non-finite {process_id} value"
        )
    if abs(value.imag) > max(ABSOLUTE_TOLERANCE, abs(value.real) * 1.0e-12):
        raise HEFTAcceptanceError(
            f"pyAmpliCol returned a non-real {process_id} matrix element"
        )
    if value.real == 0.0:
        raise HEFTAcceptanceError(f"pyAmpliCol returned zero for {process_id}")
    return value.real


def _madgraph_version(installation: Path) -> str | None:
    path = installation / "VERSION"
    return (
        path.read_text(encoding="utf-8", errors="replace").strip()
        if path.is_file()
        else None
    )


def run_acceptance(
    madgraph: Path,
    *,
    model_root: Path = DEFAULT_HEFT_UFO_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    runner: StandaloneMadGraphRunner | None = None,
) -> dict[str, object]:
    """Run the four-process scalar HEFT acceptance gate."""

    model_root = prepare_heft_ufo(model_root)
    output_root = output_root.expanduser().resolve(strict=False)
    if output_root.exists():
        raise HEFTAcceptanceError(
            f"acceptance output already exists; choose a clean root: {output_root}"
        )
    output_root.mkdir(parents=True)

    compiled_model = ModelSource.built_in_sm_heft().compile(
        use_cache=False,
        require_supported=True,
    )
    parameters = {parameter.name: parameter for parameter in compiled_model.parameters}
    missing_inputs = sorted(set(_HEFT_INPUTS).difference(parameters))
    if missing_inputs:
        raise HEFTAcceptanceError(
            f"packaged scalar HEFT model lacks inputs: {missing_inputs!r}"
        )
    madgraph_inputs = {}
    for name in _HEFT_INPUTS:
        parameter = parameters[name]
        if parameter.default_imaginary != 0.0:
            raise HEFTAcceptanceError(
                f"packaged scalar HEFT input {name} is unexpectedly complex"
            )
        madgraph_inputs[name] = parameter.default_real
    process_set = ProcessSet(
        requests=tuple(
            ProcessRequest.parse(spec.expression, name=spec.process_id)
            for spec in PROCESSES
        )
    )
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full"),
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"HIG": 1},
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=True,
                samples=1,
                seed=VALIDATION_SEED,
                relative_tolerance=1.0e-12,
                absolute_tolerance=ABSOLUTE_TOLERANCE,
                post_build_validation=True,
            ),
        ),
        evaluator=EvaluatorConfig(
            backend="jit",
            execution_mode="compiled",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
        output=OutputConfig(format="json", color="never", progress="off"),
    )
    artifact = output_root / "pyamplicol"
    Generator(config).generate(
        process_set,
        artifact,
        model=compiled_model,
        mode="error",
    )

    madgraph = madgraph.expanduser().resolve(strict=True)
    standalone_runner = StandaloneMadGraphRunner() if runner is None else runner
    results: dict[str, dict[str, object]] = {}
    for spec in PROCESSES:
        momenta = _validation_momenta(artifact, spec.process_id)
        pyamplicol_value = _pyamplicol_value(artifact, spec.process_id, momenta)
        standalone = standalone_runner.generate(
            installation=madgraph,
            artifact=output_root / "madgraph" / spec.process_id,
            process=spec.expression,
            model_import=os.fspath(model_root),
            coupling_orders={"HIG": 1},
        )
        set_parameter_card_values(
            standalone.standalone / "Cards" / "param_card.dat",
            madgraph_inputs,
        )
        madgraph_value = standalone_runner.evaluate(
            standalone,
            momenta,
            repetitions=1,
            warmup_calls=20,
        ).value
        if not math.isfinite(madgraph_value) or madgraph_value == 0.0:
            raise HEFTAcceptanceError(
                f"MadGraph returned a non-finite or zero {spec.process_id} value"
            )
        absolute_deviation = abs(pyamplicol_value - madgraph_value)
        relative_deviation = absolute_deviation / abs(madgraph_value)
        if not math.isclose(
            pyamplicol_value,
            madgraph_value,
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        ):
            raise HEFTAcceptanceError(
                f"{spec.process_id} differs from MadGraph: "
                f"relative deviation {relative_deviation:.17g}"
            )
        results[spec.process_id] = {
            "expression": spec.expression,
            "pyamplicol": pyamplicol_value,
            "madgraph": madgraph_value,
            "relative_deviation": relative_deviation,
        }

    return {
        "kind": "pyamplicol-heft-madgraph-acceptance",
        "model": "built-in-sm-heft",
        "effective_coupling_momentum_dependent": False,
        "madgraph_version": _madgraph_version(madgraph),
        "precision": 16,
        "validation_seed": VALIDATION_SEED,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "processes": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "madgraph",
        type=Path,
        help="MadGraph5_aMC installation containing bin/mg5_aMC",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_HEFT_UFO_ROOT,
        help="ignored destination for the authenticated prepared UFO",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="clean ignored destination for generated artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = run_acceptance(
            arguments.madgraph,
            model_root=arguments.model_root,
            output_root=arguments.output_root,
        )
    except (HEFTAcceptanceError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"heft-madgraph-acceptance: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
