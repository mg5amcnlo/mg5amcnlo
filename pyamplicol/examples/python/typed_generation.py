# SPDX-License-Identifier: 0BSD
"""Plan or generate the external-JSON pp -> Zjj example with typed services."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyamplicol import Generator, ModelSource, ProcessRequest, ProcessSet
from pyamplicol.config import resolve_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "process",
        nargs="*",
        default=("p p > Z j j",),
        help="quoted process expressions (default: 'p p > Z j j')",
    )
    parser.add_argument(
        "--model",
        default="models/json/sm/sm.json",
        help="built-in-sm, a trusted UFO directory, or a model JSON path",
    )
    parser.add_argument("--restriction")
    parser.add_argument(
        "--multiparticle",
        action="append",
        default=["p=d,d~,g", "j=d,d~,g"],
        metavar="NAME=ITEM[,ITEM...]",
    )
    parser.add_argument("--flavor-scheme", type=int, default=2)
    parser.add_argument("--max-quark-lines", type=int, default=2)
    parser.add_argument("--color-accuracy", choices=("lc", "nlc", "full"), default="lc")
    parser.add_argument(
        "--mode",
        choices=("error", "append", "replace"),
        default="error",
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _multiparticles(values: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for value in values:
        name, separator, members = value.partition("=")
        entries = [member.strip() for member in members.split(",") if member.strip()]
        if not separator or not name or not entries:
            raise SystemExit(f"invalid --multiparticle value: {value!r}")
        result[name] = entries
    return result


def main() -> int:
    args = _parser().parse_args()
    model_config: dict[str, object] = {"source": args.model}
    if args.restriction is not None:
        model_config["restriction"] = args.restriction
    card = {
        "schema_version": 1,
        "action": "generate",
        "model": model_config,
        "process": {
            "entries": [{"expression": expression} for expression in args.process],
            "multiparticles": _multiparticles(args.multiparticle),
            "flavor_scheme": args.flavor_scheme,
            "max_quark_lines": args.max_quark_lines,
        },
        "color": {"accuracy": args.color_accuracy},
        "generation": {"output": args.output, "mode": args.mode},
        # Raw JSON/UFO model IR has no prepared recurrence-kernel bundle.
        # Match the primary card by building process-local compiled JIT evaluators.
        "evaluator": {
            "backend": "jit",
            "execution_mode": "compiled",
            "jit": {"compress": True},
        },
    }
    resolution = resolve_config(card)
    config = resolution.effective
    processes = ProcessSet(
        tuple(
            ProcessRequest.parse(entry.expression, name=entry.name)
            for entry in config.process.entries
        )
    )
    model_source = (
        ModelSource.built_in_sm()
        if args.model == "built-in-sm"
        else ModelSource.from_path(
            args.model,
            restriction=args.restriction,
            simplify=config.model.simplify,
        )
    )
    model = model_source.compile(use_cache=config.model.cache)

    generator = Generator(resolution)
    plan = generator.plan(processes, model=model)
    for process in plan.concrete_processes:
        print(f"{process.name}: {process.expression}")
    if args.plan_only:
        return 0

    result = generator.generate(
        processes,
        config.generation.output or args.output,
        model=model,
        mode=config.generation.mode,
    )
    print(result.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
