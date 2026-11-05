# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyamplicol.cli import build_parser, parse_cli
from pyamplicol.config import (
    ColorContraction,
    ColorFFTBasis,
    ConfigurationError,
    EvaluatorExecutionMode,
    LCFlowLayout,
    ProcessEntry,
    ProcessGroupAnalysis,
    RelationDiscoveryMode,
)


def test_direct_command_flags_then_ordered_set_overrides(tmp_path: Path) -> None:
    invocation = parse_cli(
        (
            "generate",
            "d d~ > z g",
            str(tmp_path / "artifact"),
            "--workers",
            "4",
            "--color-accuracy",
            "nlc",
            "--set",
            "generation.workers=2",
            "--set",
            "generation.workers=1",
        )
    )
    config = invocation.resolve().effective
    assert config.action == "generate"
    assert config.process.entries == (ProcessEntry("d d~ > z g"),)
    assert config.generation.output == (tmp_path / "artifact").resolve()
    assert config.generation.workers == 1
    assert config.color.accuracy == "nlc"


def test_generate_accepts_eager_execution_mode_override() -> None:
    config = parse_cli(("generate", "--execution-mode", "eager")).resolve().effective
    assert config.evaluator.execution_mode is EvaluatorExecutionMode.EAGER


def test_generate_accepts_on_the_fly_execution_mode_override() -> None:
    config = (
        parse_cli(("generate", "--execution-mode", "on-the-fly")).resolve().effective
    )
    assert config.evaluator.execution_mode is EvaluatorExecutionMode.ON_THE_FLY


def test_generate_defaults_to_recurrence_execution() -> None:
    config = parse_cli(("generate",)).resolve().effective
    assert config.evaluator.execution_mode is EvaluatorExecutionMode.RECURRENCE


def test_umami_grouping_is_not_a_user_option() -> None:
    config = parse_cli(("generate",)).resolve().effective
    assert config.generation.emit_api_bundle
    assert not hasattr(config.generation, "umami_grouping")
    with pytest.raises(SystemExit):
        parse_cli(("generate", "--umami-grouping", "flavour_blind_observables"))
    with pytest.raises(ConfigurationError, match="umami_grouping"):
        parse_cli(("generate", "--set", "generation.umami_grouping=exact")).resolve()


def test_generate_accepts_symmetric_group_color_contraction() -> None:
    config = (
        parse_cli(
            (
                "generate",
                "--color-accuracy",
                "full",
                "--color-contraction",
                "symmetric-group-fft",
            )
        )
        .resolve()
        .effective
    )
    assert config.color.contraction is ColorContraction.SYMMETRIC_GROUP_FFT
    assert config.color.fft_basis is ColorFFTBasis.ADJOINT


@pytest.mark.parametrize("accuracy", ("lc", "nlc", "full"))
@pytest.mark.parametrize("execution_mode", tuple(EvaluatorExecutionMode))
def test_automatic_fft_default_is_limited_to_supported_lanes(
    accuracy: str, execution_mode: EvaluatorExecutionMode
) -> None:
    resolution = parse_cli(
        (
            "generate",
            "--color-accuracy",
            accuracy,
            "--execution-mode",
            execution_mode.value,
        )
    ).resolve()
    assert resolution.requested.color.contraction is ColorContraction.AUTO
    assert resolution.requested.color.fft_basis is ColorFFTBasis.ADJOINT
    supported = accuracy != "lc" and execution_mode in {
        EvaluatorExecutionMode.RECURRENCE,
        EvaluatorExecutionMode.ON_THE_FLY,
    }
    assert resolution.effective.color.contraction is (
        ColorContraction.AUTO if supported else ColorContraction.DIRECT
    )
    assert resolution.effective.color.fft_basis is (
        ColorFFTBasis.ADJOINT if supported else ColorFFTBasis.TRACE
    )


def test_direct_cli_opt_out_overrides_adaptive_fft_default() -> None:
    config = (
        parse_cli(
            ("generate", "--color-accuracy", "full", "--color-contraction", "direct")
        )
        .resolve()
        .effective
    )
    assert config.color.contraction is ColorContraction.DIRECT
    assert config.color.fft_basis is ColorFFTBasis.TRACE


@pytest.mark.parametrize("fft_basis", tuple(ColorFFTBasis))
@pytest.mark.parametrize("accuracy", ("nlc", "full"))
@pytest.mark.parametrize("execution_mode", ("recurrence", "on-the-fly"))
def test_fft_shorthand_selects_contraction_and_basis(
    fft_basis: ColorFFTBasis, accuracy: str, execution_mode: str
) -> None:
    config = (
        parse_cli(
            (
                "generate",
                "--fft",
                fft_basis.value,
                "--color-accuracy",
                accuracy,
                "--execution-mode",
                execution_mode,
            )
        )
        .resolve()
        .effective
    )

    assert config.color.contraction is ColorContraction.SYMMETRIC_GROUP_FFT
    assert config.color.fft_basis is fft_basis
    assert config.color.accuracy == accuracy
    assert config.evaluator.execution_mode == execution_mode


@pytest.mark.parametrize("fft_basis", tuple(ColorFFTBasis))
def test_fft_shorthand_requires_explicit_contracted_accuracy(
    fft_basis: ColorFFTBasis,
) -> None:
    with pytest.raises(ConfigurationError, match=r"requires color\.accuracy"):
        parse_cli(("generate", "--fft", fft_basis.value)).resolve()


def test_fft_shorthand_obeys_card_and_ordered_override_precedence(
    tmp_path: Path,
) -> None:
    card = tmp_path / "fft.toml"
    card.write_text(
        'action = "generate"\n[color]\naccuracy = "full"\n'
        'contraction = "direct"\nfft_basis = "trace"\n',
        encoding="utf-8",
    )
    config = (
        parse_cli(("generate", "--card", str(card), "--fft", "adjoint"))
        .resolve()
        .effective
    )
    assert config.color.contraction is ColorContraction.SYMMETRIC_GROUP_FFT
    assert config.color.fft_basis is ColorFFTBasis.ADJOINT

    overridden = (
        parse_cli(
            (
                "generate",
                "--card",
                str(card),
                "--fft",
                "adjoint",
                "--set",
                "color.fft_basis=trace",
            )
        )
        .resolve()
        .effective
    )
    assert overridden.color.fft_basis is ColorFFTBasis.TRACE


@pytest.mark.parametrize(
    "arguments",
    (
        ("--fft", "invalid"),
        ("--fft", "adjoint", "--color-contraction", "direct"),
    ),
)
def test_fft_shorthand_rejects_invalid_or_conflicting_flags(
    arguments: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit):
        parse_cli(("generate", *arguments))


def test_profile_help_explains_layout_aware_selector_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(("profile", "--help"))

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert help_text.count("omitting both selector options") == 2
    assert "--helicity ID" in help_text
    assert "--color-flow ID_OR_NUMBER" in help_text


def test_generate_validation_samples_default_and_override() -> None:
    default = parse_cli(("generate",)).resolve().effective
    explicit = parse_cli(("generate", "--validation-samples", "10")).resolve().effective

    assert default.generation.validation.samples == 2
    assert explicit.generation.validation.samples == 10


def test_generate_post_build_validation_is_opt_in() -> None:
    default = parse_cli(("generate",)).resolve().effective
    explicit = parse_cli(("generate", "--post-build-validation")).resolve().effective

    assert not default.generation.validation.post_build_validation
    assert explicit.generation.validation.post_build_validation


def test_generate_defaults_to_no_numerical_current_reuse() -> None:
    config = parse_cli(("generate",)).resolve().effective
    assert config.generation.relation_discovery.mode is RelationDiscoveryMode.OFF


@pytest.mark.parametrize(
    ("flag", "expected"),
    (
        ("--numerical-current-reuse", RelationDiscoveryMode.CERTIFIED_REUSE),
        ("--no-numerical-current-reuse", RelationDiscoveryMode.OFF),
    ),
)
def test_generate_accepts_numerical_current_reuse_override(
    flag: str,
    expected: RelationDiscoveryMode,
) -> None:
    config = parse_cli(("generate", flag)).resolve().effective
    assert config.generation.relation_discovery.mode is expected


def test_direct_numerical_current_reuse_flag_overrides_card(
    tmp_path: Path,
) -> None:
    card = tmp_path / "generate.toml"
    card.write_text(
        "\n".join(
            (
                'action = "generate"',
                "[generation.relation_discovery]",
                'mode = "off"',
            )
        ),
        encoding="utf-8",
    )
    config = (
        parse_cli(
            (
                "generate",
                "--card",
                str(card),
                "--numerical-current-reuse",
            )
        )
        .resolve()
        .effective
    )
    assert (
        config.generation.relation_discovery.mode
        is RelationDiscoveryMode.CERTIFIED_REUSE
    )


@pytest.mark.parametrize(
    ("flag", "expected"),
    (("--jit-compress", True), ("--no-jit-compress", False)),
)
@pytest.mark.parametrize("mode", ("recurrence", "compiled", "eager", "on-the-fly"))
def test_generate_accepts_jit_compression_override(
    flag: str,
    expected: bool,
    mode: str,
) -> None:
    config = parse_cli(("generate", "--execution-mode", mode, flag)).resolve().effective
    assert config.evaluator.jit.compress is expected


@pytest.mark.parametrize("mode", ("recurrence", "compiled", "eager", "on-the-fly"))
def test_generation_modes_select_jit_compression_default(mode: str) -> None:
    config = parse_cli(("generate", "--execution-mode", mode)).resolve().effective
    assert config.evaluator.jit.compress is (mode == "compiled")


def test_generate_accepts_recurrence_execution_mode_override() -> None:
    config = (
        parse_cli(("generate", "--execution-mode", "recurrence")).resolve().effective
    )
    assert config.evaluator.execution_mode is EvaluatorExecutionMode.RECURRENCE


def test_generate_accepts_recurrence_schema_overrides() -> None:
    config = (
        parse_cli(
            (
                "generate",
                "--set",
                "evaluator.execution_mode=recurrence",
                "--set",
                "evaluator.recurrence.point_tile_size=2048",
                "--set",
                "evaluator.recurrence.workspace_mib=384",
            )
        )
        .resolve()
        .effective
    )
    assert config.evaluator.execution_mode is EvaluatorExecutionMode.RECURRENCE
    assert config.evaluator.recurrence.point_tile_size == 2048
    assert config.evaluator.recurrence.workspace_mib == 384


def test_generate_accepts_lc_flow_layout_override() -> None:
    resolution = parse_cli(("generate", "--lc-flow-layout", "all-flow-union")).resolve()
    assert resolution.requested.color.lc_flow_layout is LCFlowLayout.ALL_FLOW_UNION
    assert resolution.effective.color.lc_flow_layout is LCFlowLayout.ALL_FLOW_UNION


def test_generate_force_is_an_alias_for_atomic_replace(tmp_path: Path) -> None:
    config = (
        parse_cli(
            (
                "generate",
                "d d~ > z g",
                str(tmp_path / "artifact"),
                "--force",
            )
        )
        .resolve()
        .effective
    )
    assert str(config.generation.mode) == "replace"


def test_card_can_be_invoked_as_the_first_argument(tmp_path: Path) -> None:
    card = tmp_path / "evaluate.toml"
    card.write_text(
        'action = "evaluate"\n[evaluation]\nartifact = "artifact"\n',
        encoding="utf-8",
    )
    invocation = parse_cli((str(card), "--json"))
    config = invocation.resolve().effective
    assert invocation.action is None
    assert config.action == "evaluate"
    assert config.evaluation.artifact == (tmp_path / "artifact").resolve()
    assert config.output.format == "json"


def test_all_contract_actions_have_subcommands() -> None:
    arguments = {
        "generate": ("generate",),
        "evaluate": ("evaluate",),
        "benchmark": ("benchmark",),
        "inspect": ("inspect",),
        "model-inspect": ("model", "inspect"),
        "model-compile": ("model", "compile"),
        "model-processes": ("model", "processes"),
        "group-processes": ("group-processes",),
    }
    for action, argv in arguments.items():
        assert parse_cli(argv).action == action

    assert parse_cli(("profile",)).action == "benchmark"


def test_generate_process_grouping_flags_keep_flavor_scheme() -> None:
    default = parse_cli(("generate",)).resolve().effective.process
    assert not default.flavor_blind
    assert default.consider_charge_conjugation
    assert not default.only_unit_multiplier
    assert default.analyze_process_groups is ProcessGroupAnalysis.NONE

    process = (
        parse_cli(
            (
                "generate",
                "--flavor-scheme",
                "4",
                "--flavor-blind",
                "--no-consider-charge-conjugation",
                "--only-unit-multiplier",
                "--analyze-process-groups",
                "numerical",
            )
        )
        .resolve()
        .effective.process
    )
    assert process.flavor_scheme == 4
    assert process.flavor_blind
    assert not process.consider_charge_conjugation
    assert process.only_unit_multiplier
    assert process.analyze_process_groups is ProcessGroupAnalysis.NUMERICAL


def test_group_processes_defaults_to_numerical_analysis(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    config = parse_cli(("group-processes", str(artifact))).resolve().effective
    assert config.action == "group-processes"
    assert config.evaluation.artifact == artifact.resolve()
    assert config.process.analyze_process_groups is ProcessGroupAnalysis.NUMERICAL
    assert config.process.consider_charge_conjugation

    explicit = (
        parse_cli(
            (
                "group-processes",
                str(artifact),
                "--analyze-process-groups",
                "none",
                "--only-unit-multiplier",
                "--no-consider-charge-conjugation",
            )
        )
        .resolve()
        .effective.process
    )
    assert explicit.analyze_process_groups is ProcessGroupAnalysis.NONE
    assert explicit.only_unit_multiplier
    assert not explicit.consider_charge_conjugation
    # The choice "none" is not the null literal of --set.
    overridden = parse_cli(
        (
            "group-processes",
            str(artifact),
            "--set",
            "process.analyze_process_groups=none",
        )
    ).resolve()
    overridden_process = overridden.effective.process
    assert overridden_process.analyze_process_groups is ProcessGroupAnalysis.NONE
    with pytest.raises(SystemExit):
        parse_cli(("group-processes", str(artifact), "--flavor-blind"))


def test_inspect_full_physics_is_an_explicit_transient_opt_in(
    tmp_path: Path,
) -> None:
    default = parse_cli(
        ("inspect", str(tmp_path / "artifact"), "--process", "process-id")
    )
    full = parse_cli(
        (
            "inspect",
            str(tmp_path / "artifact"),
            "--process",
            "process-id",
            "--full-physics",
        )
    )

    assert default.full_physics is False
    assert full.full_physics is True
    assert full.resolve().effective.evaluation.process == "process-id"


@pytest.mark.parametrize(
    "command",
    ("model-inspect", "model-compile", "model-processes"),
)
def test_flat_model_commands_are_not_cli_compatibility_aliases(command: str) -> None:
    with pytest.raises(SystemExit):
        parse_cli((command,))


def test_model_processes_accepts_process_requests() -> None:
    invocation = parse_cli(
        (
            "model",
            "processes",
            "d d~ > z g",
            "--process",
            "u u~ > z g",
        )
    )
    config = invocation.resolve().effective
    assert config.action == "model-processes"
    assert config.process.entries == (
        ProcessEntry("d d~ > z g"),
        ProcessEntry("u u~ > z g"),
    )


def test_direct_process_names_are_assembled_into_typed_entries() -> None:
    config = (
        parse_cli(
            (
                "model",
                "processes",
                "d d~ > z g",
                "--process",
                "u u~ > z g",
                "--name",
                "ddbar_zg",
                "--name",
                "uubar_zg",
            )
        )
        .resolve()
        .effective
    )
    assert config.process.entries == (
        ProcessEntry("d d~ > z g", "ddbar_zg"),
        ProcessEntry("u u~ > z g", "uubar_zg"),
    )


def test_direct_process_names_must_align_with_expressions() -> None:
    with pytest.raises(ConfigurationError, match="aligned with process expressions"):
        parse_cli(
            (
                "model",
                "processes",
                "d d~ > z g",
                "--name",
                "one",
                "--name",
                "two",
            )
        )


def test_model_command_accepts_common_flags_before_the_command_group() -> None:
    invocation = parse_cli(("--json", "model", "inspect"))
    assert invocation.resolve().effective.output.format == "json"


def test_top_level_help_lists_nested_model_and_utility_commands() -> None:
    help_text = build_parser().format_help()
    for command in (
        "model",
        "config",
        "examples",
        "doctor",
        "self-test",
        "request-symbolica-trial-license",
        "request-symbolica-hobbyist-license",
    ):
        assert command in help_text
    assert "model-inspect" not in help_text
    assert "model-compile" not in help_text
    assert "model-processes" not in help_text


def test_model_help_does_not_import_symbolica(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sys.modules.pop("symbolica", None)
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(("model", "--help"))
    assert exit_info.value.code == 0
    assert "{inspect,compile,processes}" in capsys.readouterr().out
    assert "symbolica" not in sys.modules


def test_global_output_flags_are_accepted_before_subcommand(tmp_path: Path) -> None:
    invocation = parse_cli(
        (
            "--json",
            "generate",
            "u u~ > g g",
            str(tmp_path / "artifact"),
        )
    )
    assert invocation.resolve().effective.output.format == "json"


def test_json_output_is_selected_only_by_the_json_flag(tmp_path: Path) -> None:
    card = tmp_path / "json-card.toml"
    card.write_text(
        'schema_version = 1\naction = "generate"\n'
        '[output]\nformat = "json"\n'
        '[process]\nentries = [{ expression = "d d~ > z" }]\n',
        encoding="utf-8",
    )

    assert parse_cli((str(card),)).resolve().effective.output.format == "human"
    assert parse_cli((str(card), "--json")).resolve().effective.output.format == "json"

    with pytest.raises(ConfigurationError, match="selected only with --json"):
        parse_cli((str(card), "--set", "output.format=json")).resolve()
    assert (
        parse_cli((str(card), "--set", "output.format=human"))
        .resolve()
        .effective.output.format
        == "human"
    )

    with pytest.raises(SystemExit):
        parse_cli(("generate", "d d~ > z", "--format", "json"))


def test_set_order_is_preserved_across_subcommand_position(tmp_path: Path) -> None:
    invocation = parse_cli(
        (
            "--set",
            "generation.workers=5",
            "generate",
            "u u~ > g g",
            str(tmp_path / "artifact"),
            "--set",
            "generation.workers=2",
        )
    )
    assert invocation.overrides == (
        "generation.workers=5",
        "generation.workers=2",
    )
    assert invocation.resolve().effective.generation.workers == 2


@pytest.mark.parametrize(
    "arguments",
    (
        ("generate", "--color-coverage", "all"),
        ("generate", "--color-flow", "flow:2,4,1"),
        ("generate", "--zero-current-filter"),
        ("generate", "--no-zero-current-filter"),
        ("generate", "--current-merging"),
        ("generate", "--no-current-merging"),
    ),
)
def test_removed_generation_flags_are_rejected(arguments: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit):
        parse_cli(arguments)


def test_runtime_color_flow_flags_remain_available() -> None:
    evaluation = parse_cli(("evaluate", "--color-flow", "flow:2,4,1"))
    benchmark = parse_cli(
        (
            "benchmark",
            "--process",
            "d d~ > z g",
            "--precision",
            "80",
            "--color-flow",
            "flow:2,4,1",
        )
    )

    assert evaluation.resolve().effective.evaluation.color_flow_ids == ("flow:2,4,1",)
    assert benchmark.resolve().effective.evaluation.process == "d d~ > z g"
    assert benchmark.resolve().effective.benchmark.precision == 80
    assert benchmark.resolve().effective.benchmark.color_flow_ids == ("flow:2,4,1",)


def test_profile_is_primary_cli_spelling_for_typed_benchmark_config(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    profile = (
        parse_cli(
            (
                "profile",
                str(artifact),
                "--process",
                "d d~ > z g",
                "--target-runtime",
                "0.25",
                "--batch-size",
                "64",
                "--minimum-samples",
                "8",
            )
        )
        .resolve()
        .effective
    )
    benchmark = (
        parse_cli(
            (
                "benchmark",
                str(artifact),
                "--process",
                "d d~ > z g",
                "--target-runtime",
                "0.25",
                "--batch-size",
                "64",
                "--minimum-samples",
                "8",
            )
        )
        .resolve()
        .effective
    )

    assert profile == benchmark
    assert profile.action == "benchmark"
    assert profile.evaluation.artifact == artifact.resolve()
    assert profile.evaluation.process == "d d~ > z g"
    assert profile.benchmark.batch_size == 64
    assert profile.benchmark.minimum_samples == 8
