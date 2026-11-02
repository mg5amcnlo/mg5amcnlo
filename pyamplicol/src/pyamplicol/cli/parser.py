# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from pyamplicol.config import (
    ACTIONS,
    Action,
    ClampRequest,
    ColorContraction,
    ColorFFTBasis,
    ConfigResolution,
    ConfigurationError,
    EvaluatorExecutionMode,
    ProcessGroupAnalysis,
    resolve_config,
)

from .licensing import LicenseKind, LicenseRequestInvocation
from .utilities import UtilityInvocation, parse_utility

_LICENSE_ACTIONS: dict[str, LicenseKind] = {
    "request-symbolica-trial-license": "trial",
    "request-symbolica-hobbyist-license": "hobbyist",
}
_MODEL_ACTIONS: dict[str, Action] = {
    "inspect": Action.MODEL_INSPECT,
    "compile": Action.MODEL_COMPILE,
    "processes": Action.MODEL_PROCESSES,
}
_UTILITY_COMMANDS = frozenset({"config", "examples", "doctor", "self-test"})
_DIRECT_COMMANDS = frozenset(
    {
        *(action.value for action in ACTIONS if not action.value.startswith("model-")),
        "profile",
        "model",
        *_LICENSE_ACTIONS,
    }
)
_REMOVED_FLAT_MODEL_COMMANDS = frozenset(
    action.value for action in _MODEL_ACTIONS.values()
)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(entry) for key, entry in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(entry) for entry in value)
    return value


def _auto_int(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected 'auto' or an integer") from exc


def _nullable_int(value: str) -> int | None:
    if value.lower() in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer or 'null'") from exc


def _key_list(value: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=ITEM[,ITEM...]")
    key, raw_entries = value.split("=", 1)
    entries = tuple(entry.strip() for entry in raw_entries.split(",") if entry.strip())
    if not key.strip() or not entries:
        raise argparse.ArgumentTypeError("expected NAME=ITEM[,ITEM...]")
    return key.strip(), entries


def _key_int(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=INTEGER")
    key, raw_entry = value.split("=", 1)
    try:
        entry = int(raw_entry)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME=INTEGER") from exc
    if not key.strip():
        raise argparse.ArgumentTypeError("expected NAME=INTEGER")
    return key.strip(), entry


@dataclass(frozen=True, slots=True)
class CliInvocation:
    action: Action | None
    card: Path | None
    dedicated: Mapping[str, object]
    overrides: tuple[str, ...]
    dry_run: bool = False
    full_physics: bool = False
    correlators: Path | None = None

    def __post_init__(self) -> None:
        frozen = _freeze(self.dedicated)
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "dedicated", frozen)
        object.__setattr__(self, "overrides", tuple(self.overrides))

    def resolve(
        self,
        *,
        clamps: Sequence[ClampRequest] = (),
    ) -> ConfigResolution:
        if any(
            str(override).partition("=")[0].strip() == "output.format"
            and str(override).partition("=")[2].strip().strip("\"'").casefold()
            == "json"
            for override in self.overrides
        ):
            raise ConfigurationError(
                "CLI JSON output is selected only with --json; "
                "remove the output.format override"
            )
        return resolve_config(
            self.card,
            action=self.action,
            dedicated=self.dedicated,
            overrides=self.overrides,
            clamps=clamps,
        )


def _common_parent(*, include_card: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    if include_card:
        parser.add_argument(
            "--card",
            type=Path,
            default=argparse.SUPPRESS,
            help="Read defaults from a TOML run card.",
        )
    parser.add_argument(
        "--set",
        dest="_overrides",
        action="append",
        default=argparse.SUPPRESS,
        metavar="PATH=VALUE",
        help="Override one schema field; repeat to apply overrides in order.",
    )
    parser.add_argument(
        "--json",
        dest="_json_output",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit machine-readable JSON instead of human-facing tables.",
    )
    parser.add_argument(
        "--color",
        dest="output.color",
        choices=("auto", "always", "never"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--progress",
        dest="output.progress",
        choices=("auto", "tty", "log", "off"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-level",
        dest="output.log_level",
        choices=("debug", "info", "warning", "error"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--symbolica-suggestion",
        dest="symbolica.suggest_license",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    return parser


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", dest="model.source", default=argparse.SUPPRESS)
    parser.add_argument(
        "--restriction", dest="model.restriction", default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--simplify",
        dest="model.simplify",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model-cache",
        dest="model.cache",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model-cache-dir",
        dest="model.cache_dir",
        type=Path,
        default=argparse.SUPPRESS,
    )


def _add_process_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--name",
        dest="_process_names",
        action="append",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--multiparticle",
        dest="_multiparticles",
        type=_key_list,
        action="append",
        default=argparse.SUPPRESS,
        metavar="NAME=ITEM[,ITEM...]",
    )
    parser.add_argument(
        "--flavor-scheme",
        dest="process.flavor_scheme",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-quark-lines",
        dest="process.max_quark_lines",
        type=_nullable_int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--coupling-order-policy",
        dest="process.coupling_order_policy",
        choices=("minimal", "explicit"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-coupling-order",
        dest="_max_coupling_orders",
        type=_key_int,
        action="append",
        default=argparse.SUPPRESS,
        metavar="NAME=INTEGER",
    )


def _add_process_grouping_options(
    parser: argparse.ArgumentParser,
    *,
    generation: bool,
) -> None:
    if generation:
        parser.add_argument(
            "--flavor-blind",
            dest="process.flavor_blind",
            action="store_true",
            default=argparse.SUPPRESS,
            help=(
                "generate one representative per flavour-relabelling class and "
                "record the others as its group members (assumed, not verified)"
            ),
        )
    parser.add_argument(
        "--consider-charge-conjugation",
        dest="process.consider_charge_conjugation",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="also relate processes through charge conjugation (default: on)",
    )
    parser.add_argument(
        "--only-unit-multiplier",
        dest="process.only_unit_multiplier",
        action="store_true",
        default=argparse.SUPPRESS,
        help="accept only numerically found relations with factor 1",
    )
    parser.add_argument(
        "--analyze-process-groups",
        dest="process.analyze_process_groups",
        choices=tuple(ProcessGroupAnalysis),
        default=argparse.SUPPRESS if generation else ProcessGroupAnalysis.NUMERICAL,
        help=(
            "search for subprocesses with equal matrix elements up to a "
            f"rational factor (default: {'none' if generation else 'numerical'})"
        ),
    )


def _add_color_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--color-accuracy",
        dest="color.accuracy",
        choices=("lc", "nlc", "full"),
        default=argparse.SUPPRESS,
    )
    contraction = parser.add_mutually_exclusive_group()
    contraction.add_argument(
        "--color-contraction",
        dest="color.contraction",
        choices=tuple(ColorContraction),
        default=argparse.SUPPRESS,
        help=(
            "Default: auto, using adaptive FFT for supported NLC/full recurrence "
            "or on-the-fly plans and direct otherwise. Select direct to opt out."
        ),
    )
    contraction.add_argument(
        "--fft",
        dest="_fft_basis",
        choices=tuple(ColorFFTBasis),
        default=argparse.SUPPRESS,
        help=(
            "Force symmetric-group FFT contraction. 'adjoint' selects certified "
            "DDM where applicable, otherwise exact fundamental chains or an "
            "explicit trace fallback; requires --color-accuracy nlc or full."
        ),
    )
    parser.add_argument(
        "--lc-flow-layout",
        dest="color.lc_flow_layout",
        choices=("topology-replay", "all-flow-union"),
        default=argparse.SUPPRESS,
    )


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        dest="generation.mode",
        choices=("error", "append", "replace"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--force",
        dest="generation.mode",
        action="store_const",
        const="replace",
        default=argparse.SUPPRESS,
        help="atomically replace an existing output artifact",
    )
    parser.add_argument(
        "--workers",
        dest="generation.workers",
        type=_auto_int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--emit-api-bundle",
        dest="generation.emit_api_bundle",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--validation",
        dest="generation.validation.enabled",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--validation-samples",
        dest="generation.validation.samples",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--validation-seed",
        dest="generation.validation.seed",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--relative-tolerance",
        dest="generation.validation.relative_tolerance",
        type=float,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--absolute-tolerance",
        dest="generation.validation.absolute_tolerance",
        type=float,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--post-build-validation",
        dest="generation.validation.post_build_validation",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help=(
            "re-open and numerically smoke-test the written artifact (off by default)"
        ),
    )
    numerical_reuse = parser.add_mutually_exclusive_group()
    numerical_reuse.add_argument(
        "--numerical-current-reuse",
        dest="generation.relation_discovery.mode",
        action="store_const",
        const="certified-reuse",
        default=argparse.SUPPRESS,
        help=(
            "search for and apply independently certified numerical current "
            "relations (off by default: the high-precision probes are costly at "
            "generation time)"
        ),
    )
    numerical_reuse.add_argument(
        "--no-numerical-current-reuse",
        dest="generation.relation_discovery.mode",
        action="store_const",
        const="off",
        default=argparse.SUPPRESS,
        help="disable numerical current-relation discovery and reuse (the default)",
    )


def _add_evaluator_options(
    parser: argparse.ArgumentParser,
    *,
    include_runtime_controls: bool = True,
) -> None:
    parser.add_argument(
        "--backend",
        dest="evaluator.backend",
        choices=("jit", "asm", "cpp"),
        default=argparse.SUPPRESS,
    )
    if include_runtime_controls:
        parser.add_argument(
            "--execution-mode",
            dest="evaluator.execution_mode",
            choices=tuple(mode.value for mode in EvaluatorExecutionMode),
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--batch-size",
            dest="evaluator.batch_size",
            type=int,
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--output-chunk-size",
            dest="evaluator.output_chunk_size",
            type=_nullable_int,
            default=argparse.SUPPRESS,
        )
    parser.add_argument(
        "--horner-iterations",
        dest="evaluator.optimization.horner_iterations",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cpe-iterations",
        dest="evaluator.optimization.cpe_iterations",
        type=_nullable_int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cores",
        dest="evaluator.optimization.cores",
        type=_auto_int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-horner-variables",
        dest="evaluator.optimization.max_horner_variables",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-common-pair-cache-entries",
        dest="evaluator.optimization.max_common_pair_cache_entries",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-common-pair-distance",
        dest="evaluator.optimization.max_common_pair_distance",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--collect-factors",
        dest="evaluator.optimization.collect_factors",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jit-optimization-level",
        dest="evaluator.jit.optimization_level",
        type=int,
        choices=(0, 1, 2, 3),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jit-compress",
        dest="evaluator.jit.compress",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cpp-optimization",
        dest="evaluator.cpp.optimization",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cpp-compiler",
        dest="evaluator.cpp.compiler",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cpp-native-arch",
        dest="evaluator.cpp.native_arch",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cpp-extra-flag",
        dest="evaluator.cpp.extra_flags",
        action="append",
        default=argparse.SUPPRESS,
    )


def _add_evaluation_options(
    parser: argparse.ArgumentParser, *, include_artifact: bool = True
) -> None:
    if include_artifact:
        parser.add_argument("artifact", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--process",
        dest="evaluation.process",
        default=argparse.SUPPRESS,
        help="stable process/alias ID or exact concrete process expression",
    )
    parser.add_argument(
        "--precision",
        dest="evaluation.precision",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resolved",
        dest="evaluation.resolved",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--helicity",
        dest="evaluation.helicity_ids",
        action="append",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--color-flow",
        dest="evaluation.color_flow_ids",
        action="append",
        default=argparse.SUPPRESS,
        metavar="ID_OR_NUMBER",
        help="stable color-flow ID or one-based ordinal shown by inspect",
    )
    parser.add_argument(
        "--model-parameters",
        dest="evaluation.model_parameters",
        type=Path,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--momenta",
        dest="evaluation.momenta",
        type=Path,
        default=argparse.SUPPRESS,
    )


def _add_profile_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("target", type=Path, nargs="?", default=None, metavar="OUTPUT")
    parser.add_argument(
        "--process",
        dest="evaluation.process",
        default=argparse.SUPPRESS,
        metavar="PROCESS",
        help="select a stable process/alias ID or exact process expression",
    )
    parser.add_argument(
        "--target-runtime",
        dest="benchmark.target_runtime",
        type=float,
        default=argparse.SUPPRESS,
        metavar="SECONDS",
        help=(
            "target cumulative Rusticol timing duration; Ctrl-C reports "
            "statistics from complete samples collected so far"
        ),
    )
    parser.add_argument(
        "--batch-size",
        dest="benchmark.batch_size",
        type=int,
        default=argparse.SUPPRESS,
        metavar="POINTS",
        help="number of phase-space points evaluated by each runtime call",
    )
    parser.add_argument(
        "--precision",
        dest="benchmark.precision",
        type=int,
        default=argparse.SUPPRESS,
        metavar="DIGITS",
        help="decimal digits used by the Python runtime evaluator",
    )
    parser.add_argument(
        "--warmup-runs",
        dest="benchmark.warmup_runs",
        type=int,
        default=argparse.SUPPRESS,
        metavar="COUNT",
    )
    parser.add_argument(
        "--minimum-samples",
        dest="benchmark.minimum_samples",
        type=int,
        default=argparse.SUPPRESS,
        metavar="COUNT",
        help="minimum number of independent timed blocks",
    )
    parser.add_argument(
        "--helicity",
        dest="benchmark.helicity_ids",
        action="append",
        default=argparse.SUPPRESS,
        metavar="ID",
        help=(
            "stable helicity ID; omitting both selector options chooses the "
            "artifact layout's optimized profiling workload"
        ),
    )
    parser.add_argument(
        "--color-flow",
        dest="benchmark.color_flow_ids",
        action="append",
        default=argparse.SUPPRESS,
        metavar="ID_OR_NUMBER",
        help=(
            "stable color-flow ID or one-based ordinal shown by inspect; "
            "omitting both selector options chooses the artifact layout's "
            "optimized profiling workload"
        ),
    )
    parser.add_argument(
        "--momenta",
        dest="evaluation.momenta",
        type=Path,
        default=argparse.SUPPRESS,
        metavar="PATH",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyamplicol",
        description=(
            "Generate, inspect, evaluate, and profile pyAmpliCol process artifacts."
        ),
    )
    subparsers = parser.add_subparsers(dest="_action", required=True)
    common = _common_parent()

    generate = subparsers.add_parser(
        "generate",
        parents=[common],
        help="Generate a process artifact.",
    )
    generate.add_argument("process", nargs="?", default=None)
    generate.add_argument("output", type=Path, nargs="?", default=None)
    generate.add_argument(
        "--process",
        dest="_extra_processes",
        action="append",
        default=argparse.SUPPRESS,
    )
    generate.add_argument(
        "--dry-run",
        dest="_dry_run",
        action="store_true",
        help="Plan and validate generation without writing an artifact.",
    )
    _add_model_options(generate)
    _add_process_options(generate)
    _add_process_grouping_options(generate, generation=True)
    _add_color_options(generate)
    _add_generation_options(generate)
    generate.add_argument(
        "--correlators",
        dest="_correlators",
        type=Path,
        default=argparse.SUPPRESS,
        help="JSON declaration of colour IDs and allowed joint spin-correlation legs",
    )
    _add_evaluator_options(generate)

    evaluate = subparsers.add_parser(
        "evaluate",
        parents=[common],
        help="Evaluate momenta with a generated artifact.",
    )
    _add_evaluation_options(evaluate)

    profile = subparsers.add_parser(
        "profile",
        parents=[common],
        help="Profile a generated artifact.",
    )
    _add_profile_options(profile)

    benchmark = subparsers.add_parser(
        "benchmark",
        parents=[common],
        help="Compatibility alias for the profile command.",
    )
    _add_profile_options(benchmark)

    inspect = subparsers.add_parser(
        "inspect",
        parents=[common],
        help="Summarize a generated artifact and its processes.",
    )
    inspect.add_argument("artifact", type=Path, nargs="?", default=None)
    inspect.add_argument(
        "--process",
        dest="evaluation.process",
        default=argparse.SUPPRESS,
        help=(
            "show compact metadata for one stable process/alias ID or exact "
            "concrete process expression"
        ),
    )
    inspect.add_argument(
        "--full-physics",
        dest="_full_physics",
        action="store_true",
        help=(
            "load the selected runtime and materialize complete helicity/color "
            "physics (potentially very large at high multiplicity)"
        ),
    )

    group_processes = subparsers.add_parser(
        "group-processes",
        parents=[common],
        help=(
            "Find generated subprocesses with equal matrix elements and update "
            "the artifact's process-groups.json in place."
        ),
    )
    group_processes.add_argument("artifact", type=Path, nargs="?", default=None)
    _add_process_grouping_options(group_processes, generation=False)

    model = subparsers.add_parser(
        "model",
        help="Inspect, compile, or enumerate processes for a model.",
    )
    model_commands = model.add_subparsers(dest="_model_action", required=True)

    model_inspect = model_commands.add_parser(
        "inspect",
        parents=[common],
        help="Inspect a built-in, UFO, JSON, or compiled model.",
    )
    model_inspect.add_argument("source", nargs="?", default=None)
    _add_model_options(model_inspect)

    model_compile = model_commands.add_parser(
        "compile",
        parents=[common],
        help=(
            "Compile portable model IR or prepare one evaluator kernel bundle "
            "shared by eager, recurrence, and on-the-fly execution."
        ),
    )
    model_compile.add_argument("source", nargs="?", default=None)
    model_compile.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "IR output path, or a .pyamplicol-model path for a self-contained "
            "prepared bundle"
        ),
    )
    _add_model_options(model_compile)
    _add_generation_options(model_compile)
    _add_evaluator_options(model_compile, include_runtime_controls=False)

    model_processes = model_commands.add_parser(
        "processes",
        parents=[common],
        help="Expand concrete processes using a model's particle catalog.",
    )
    model_processes.add_argument("process", nargs="?", default=None)
    model_processes.add_argument(
        "--process",
        dest="_extra_processes",
        action="append",
        default=argparse.SUPPRESS,
    )
    _add_model_options(model_processes)
    _add_process_options(model_processes)

    for name, help_text in (
        ("config", "Create or resolve run configuration."),
        ("examples", "List, copy, or run packaged examples."),
        ("doctor", "Diagnose the installed pyAmpliCol environment."),
        ("self-test", "Run the installed-package self-test."),
    ):
        subparsers.add_parser(name, help=help_text)

    trial = subparsers.add_parser(
        "request-symbolica-trial-license",
        help="Request a Symbolica trial license.",
    )
    trial.add_argument("--name")
    trial.add_argument("--email")
    trial.add_argument("--organization")
    trial.add_argument("--yes", dest="assume_yes", action="store_true")
    trial.add_argument("--json", dest="_json_output", action="store_true")
    trial.add_argument(
        "--color",
        dest="_output_color",
        choices=("auto", "always", "never"),
        default="auto",
    )

    hobbyist = subparsers.add_parser(
        "request-symbolica-hobbyist-license",
        help="Request a Symbolica hobbyist license.",
    )
    hobbyist.add_argument("--name")
    hobbyist.add_argument("--email")
    hobbyist.add_argument("--yes", dest="assume_yes", action="store_true")
    hobbyist.add_argument("--json", dest="_json_output", action="store_true")
    hobbyist.add_argument(
        "--color",
        dest="_output_color",
        choices=("auto", "always", "never"),
        default="auto",
    )
    return parser


def build_card_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyamplicol", parents=[_common_parent(include_card=False)]
    )
    parser.add_argument("card", type=Path)
    return parser


def _is_prepared_model_output(value: object) -> bool:
    if value is None or value == argparse.SUPPRESS:
        return False
    try:
        return str(os.fspath(value)).lower().endswith(".pyamplicol-model")
    except TypeError:
        return False


def _namespace_to_invocation(
    namespace: argparse.Namespace,
) -> CliInvocation | LicenseRequestInvocation:
    raw = vars(namespace).copy()
    action_value = raw.pop("_action", None)
    model_action = raw.pop("_model_action", None)
    if action_value == "model":
        if model_action not in _MODEL_ACTIONS:
            raise ValueError("model requires an inspect, compile, or processes command")
        action_value = _MODEL_ACTIONS[model_action].value
    if action_value == "profile":
        action_value = Action.BENCHMARK.value
    if action_value in _LICENSE_ACTIONS:
        return LicenseRequestInvocation(
            kind=_LICENSE_ACTIONS[action_value],
            name=raw.pop("name", None),
            email=raw.pop("email", None),
            organization=raw.pop("organization", None),
            assume_yes=bool(raw.pop("assume_yes", False)),
            output_format="json" if bool(raw.pop("_json_output", False)) else "human",
            output_color=str(raw.pop("_output_color", "auto")),
        )
    action = Action(action_value) if action_value is not None else None
    if action is Action.MODEL_COMPILE:
        evaluator_arguments = tuple(
            sorted(key for key in raw if key.startswith("evaluator."))
        )
        evaluator_overrides = tuple(
            value
            for value in raw.get("_overrides", ())
            if str(value).partition("=")[0].strip().startswith("evaluator.")
        )
        has_evaluator_options = evaluator_arguments or evaluator_overrides
        if has_evaluator_options and not _is_prepared_model_output(raw.get("output")):
            raise ConfigurationError(
                "model compile evaluator options require an output ending with "
                "'.pyamplicol-model'; IR-only model outputs do not compile kernels"
            )
    card_value = raw.pop("card", None)
    card = (
        Path(os.fspath(card_value)).expanduser().resolve(strict=False)
        if card_value is not None
        else None
    )
    overrides = tuple(raw.pop("_overrides", ()))
    raw["output.format"] = "json" if bool(raw.pop("_json_output", False)) else "human"
    dry_run = bool(raw.pop("_dry_run", False))
    full_physics = bool(raw.pop("_full_physics", False))
    correlators = raw.pop("_correlators", None)

    positional_process = raw.pop("process", None)
    extra_processes = tuple(raw.pop("_extra_processes", ()))
    process_names = tuple(raw.pop("_process_names", ()))
    if positional_process == argparse.SUPPRESS:
        positional_process = None
    process_expressions = tuple(
        entry for entry in (positional_process, *extra_processes) if entry is not None
    )
    if process_names and len(process_names) != len(process_expressions):
        raise ConfigurationError(
            "direct process names must be empty or aligned with process expressions"
        )
    if process_expressions:
        raw["process.entries"] = tuple(
            {
                "expression": expression,
                **({"name": process_names[index]} if process_names else {}),
            }
            for index, expression in enumerate(process_expressions)
        )

    positional_output = raw.pop("output", None)
    if positional_output != argparse.SUPPRESS and positional_output is not None:
        raw["generation.output"] = positional_output
    positional_artifact = raw.pop("artifact", None)
    if positional_artifact != argparse.SUPPRESS and positional_artifact is not None:
        raw["evaluation.artifact"] = positional_artifact
    positional_target = raw.pop("target", None)
    if positional_target != argparse.SUPPRESS and positional_target is not None:
        raw["evaluation.artifact"] = positional_target
    positional_source = raw.pop("source", None)
    if positional_source != argparse.SUPPRESS and positional_source is not None:
        raw["model.source"] = positional_source

    multiparticles = raw.pop("_multiparticles", None)
    if multiparticles is not None:
        raw["process.multiparticles"] = dict(multiparticles)
    coupling_orders = raw.pop("_max_coupling_orders", None)
    if coupling_orders is not None:
        raw["process.max_coupling_orders"] = dict(coupling_orders)
    fft_basis = raw.pop("_fft_basis", None)
    if fft_basis is not None:
        raw["color.contraction"] = ColorContraction.SYMMETRIC_GROUP_FFT
        raw["color.fft_basis"] = fft_basis
    return CliInvocation(
        action=action,
        card=card,
        dedicated=raw,
        overrides=overrides,
        dry_run=dry_run,
        full_physics=full_physics,
        correlators=correlators,
    )


def _normalize_direct_arguments(
    arguments: Sequence[str],
    command_index: int,
) -> list[str]:
    command = arguments[command_index]
    before = list(arguments[:command_index])
    after = list(arguments[command_index + 1 :])
    if command != "model":
        return [command, *before, *after]

    model_index = next(
        (index for index, argument in enumerate(after) if argument in _MODEL_ACTIONS),
        None,
    )
    if model_index is None:
        return [command, *before, *after]
    return [
        command,
        after[model_index],
        *before,
        *after[:model_index],
        *after[model_index + 1 :],
    ]


def parse_cli(
    argv: Sequence[str] | None = None,
) -> CliInvocation | LicenseRequestInvocation | UtilityInvocation:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in _UTILITY_COMMANDS:
        return parse_utility(arguments)
    if any(argument in _REMOVED_FLAT_MODEL_COMMANDS for argument in arguments):
        return _namespace_to_invocation(build_parser().parse_args(arguments))
    command_index = next(
        (
            index
            for index, argument in enumerate(arguments)
            if argument in _DIRECT_COMMANDS
        ),
        None,
    )
    if command_index is not None:
        normalized = _normalize_direct_arguments(arguments, command_index)
        return _namespace_to_invocation(build_parser().parse_args(normalized))
    if arguments == ["--help"]:
        return _namespace_to_invocation(build_parser().parse_args(arguments))
    return _namespace_to_invocation(build_card_parser().parse_args(arguments))


__all__ = [
    "CliInvocation",
    "UtilityInvocation",
    "build_card_parser",
    "build_parser",
    "parse_cli",
]
