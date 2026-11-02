# SPDX-License-Identifier: 0BSD
"""Warm and evaluate one LC flow of the packaged OTF ``p p > Z j j`` example."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from colorama import Fore, Style, just_fix_windows_console
from prettytable import HRuleStyle, PrettyTable

from pyamplicol import ColorFlow, Runtime, WarmUpResult
from pyamplicol.reporting import ProgressSink, close_progress_sink, progress_sink

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = WORKSPACE / "artifacts/otf_pp_zjj"
DEFAULT_MOMENTA = WORKSPACE / "data/pp_zjj_momenta.json"
DEFAULT_PROCESS = "d d~ > g z g"

FourMomentum = tuple[float, float, float, float]
PhaseSpacePoint = tuple[FourMomentum, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--momenta", type=Path, default=DEFAULT_MOMENTA)
    parser.add_argument("--process", default=DEFAULT_PROCESS)
    parser.add_argument(
        "--n-cores",
        type=int,
        default=None,
        metavar="N",
        help="warm-up construction workers (default: process output setting)",
    )
    parser.add_argument(
        "--color-flow",
        type=int,
        default=1,
        metavar="N",
        help="one-based physical LC flow ordinal (default: 1)",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="colored summary output (default: auto)",
    )
    return parser


def _one_point(path: Path) -> PhaseSpacePoint:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(
            "the explicit OTF warm-up example requires exactly one phase-space point"
        )
    raw_point = payload[0]
    if not isinstance(raw_point, list) or not raw_point:
        raise ValueError("the phase-space point must contain external momenta")
    point: list[FourMomentum] = []
    for raw_momentum in raw_point:
        if not isinstance(raw_momentum, list) or len(raw_momentum) != 4:
            raise ValueError("each external momentum must contain four components")
        if not all(
            isinstance(component, (int, float)) and not isinstance(component, bool)
            for component in raw_momentum
        ):
            raise ValueError("momentum components must be real numbers")
        components = tuple(float(component) for component in raw_momentum)
        point.append((components[0], components[1], components[2], components[3]))
    return tuple(point)


def _warm_and_evaluate(
    runtime: Runtime,
    point: PhaseSpacePoint,
    flow: ColorFlow,
    progress: ProgressSink,
    n_cores: int | None = None,
) -> tuple[WarmUpResult, complex]:
    # The structural warm-up contract is deliberately one point in native f64.
    # Omitting ``helicities`` requests the complete physical helicity sum.
    one_point = (point,)
    warmed = runtime.warm_up(
        one_point,
        precision=16,
        color_flows=(flow,),
        progress=progress,
        n_cores=n_cores,
    )
    values = runtime.evaluate(one_point, precision=16, color_flows=(flow,))
    if len(values) != 1:
        raise RuntimeError("one-point evaluation returned an unexpected result shape")
    return warmed, complex(values[0])


def _paint(text: str, color: str, *, enabled: bool) -> str:
    return f"{color}{text}{Style.RESET_ALL}" if enabled else text


def _bytes_text(value: int | None) -> str:
    if value is None:
        return "unavailable"
    units = ("B", "KiB", "MiB", "GiB")
    scaled = float(value)
    unit = units[0]
    for unit in units:
        if abs(scaled) < 1024.0 or unit == units[-1]:
            break
        scaled /= 1024.0
    return f"{scaled:.2f} {unit}"


def _value_text(value: complex) -> str:
    if value.imag == 0.0:
        return f"{value.real:.15e}"
    return f"{value.real:.15e}{value.imag:+.15e}j"


def _summary(
    *,
    runtime: Runtime,
    flow: ColorFlow,
    physical_helicity_count: int,
    warmed: WarmUpResult,
    value: complex,
    color: bool,
) -> str:
    table = PrettyTable(("field", "value"))
    table.title = _paint("OTF LC: one flow, helicity sum", Fore.CYAN, enabled=color)
    table.align["field"] = "l"
    table.align["value"] = "l"
    table.hrules = HRuleStyle.FRAME
    status = "already retained" if warmed.already_warm else "warmed now"
    rows = (
        ("process", runtime.physics.process, Fore.CYAN),
        ("precision", "f64 (16 digits)", Fore.GREEN),
        ("selected LC flow", f"{flow.index + 1}: {flow.id}", Fore.MAGENTA),
        ("helicities", f"sum of {physical_helicity_count}", Fore.MAGENTA),
        ("warm-up status", status, Fore.GREEN),
        (
            "queries",
            f"{warmed.warmed_query_count} new / {warmed.query_count} selected",
            Fore.YELLOW,
        ),
        ("warm-up wall", f"{warmed.elapsed_seconds:.3f} s", Fore.YELLOW),
        ("current RSS", _bytes_text(warmed.current_rss_bytes), Fore.YELLOW),
        ("peak RSS", _bytes_text(warmed.peak_rss_bytes), Fore.YELLOW),
        ("matrix element", _value_text(value), Fore.GREEN),
    )
    for field, text, shade in rows:
        table.add_row((field, _paint(text, shade, enabled=color)))
    return table.get_string()


def _uses_color(mode: str, stream: object) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    point = _one_point(args.momenta)
    runtime = Runtime.load(args.artifact, process=args.process)
    physics = runtime.physics
    if runtime.execution_mode != "on-the-fly" or physics.color_accuracy != "lc":
        raise RuntimeError("this example requires an on-the-fly LC artifact")
    if args.color_flow < 1 or args.color_flow > len(physics.color_flows):
        raise ValueError(
            f"--color-flow must be in 1..{len(physics.color_flows)} for this process"
        )
    flow = physics.color_flows[args.color_flow - 1]

    summary_color = _uses_color(args.color, sys.stdout)
    progress_color = _uses_color(args.color, sys.stderr)
    if summary_color or progress_color:
        just_fix_windows_console()
    sink = progress_sink(
        "auto",
        stream=sys.stderr,
        color=progress_color,
    )
    try:
        warmed, value = _warm_and_evaluate(runtime, point, flow, sink, args.n_cores)
    finally:
        close_progress_sink(sink)

    print(
        _summary(
            runtime=runtime,
            flow=flow,
            physical_helicity_count=len(physics.helicities),
            warmed=warmed,
            value=value,
            color=summary_color,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
