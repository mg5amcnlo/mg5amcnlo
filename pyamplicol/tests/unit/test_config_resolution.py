# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from pyamplicol.config import (
    ClampRequest,
    ColorContraction,
    ColorFFTBasis,
    ConfigurationError,
    EvaluatorExecutionMode,
    LCFlowLayout,
    ProcessEntry,
    config_to_dict,
    config_to_toml,
    load_config,
    parse_override,
    resolution_to_dict,
    resolve_config,
)


def test_resolution_precedence_paths_and_clamp_record(tmp_path: Path) -> None:
    card = tmp_path / "run.toml"
    card.write_text(
        """
schema_version = 1
action = "generate"

[model]
source = "model.json"

[process]
entries = [{ expression = "d d~ > z g", name = "ddbar_zg" }]

[generation]
output = "card-output"
workers = 6
""".strip(),
        encoding="utf-8",
    )

    resolution = resolve_config(
        card,
        dedicated={"generation.workers": 4, "generation.output": "cli-output"},
        overrides=("generation.workers=3", "generation.workers=2"),
        clamps=(
            ClampRequest(
                "generation.workers",
                1,
                "unlicensed Symbolica core limit",
            ),
        ),
    )

    assert resolution.requested.generation.workers == 2
    assert resolution.effective.generation.workers == 1
    assert resolution.requested.generation.output == (tmp_path / "cli-output").resolve()
    assert resolution.requested.model.source == str((tmp_path / "model.json").resolve())
    assert resolution.was_clamped
    assert resolution.clamps[0].requested == 2
    assert resolution.clamps[0].effective == 1


@pytest.mark.parametrize(
    "selector",
    ("built-in-sm-heft", "builtin_sm_heft", "sm-heft", "sm_heft"),
)
def test_sm_heft_model_selectors_are_not_resolved_as_paths(
    selector: str,
    tmp_path: Path,
) -> None:
    resolution = resolve_config(
        {"action": "generate", "model": {"source": selector}},
        base_dir=tmp_path,
    )

    assert resolution.effective.model.source == selector


def test_dynamic_map_override_is_schema_aware(tmp_path: Path) -> None:
    resolution = resolve_config(
        {
            "action": "generate",
            "process": {
                "entries": [{"expression": "u u~ > g g"}],
                "multiparticles": {"j": ["u", "d"]},
            },
        },
        base_dir=tmp_path,
        overrides=(
            'process.multiparticles.j=["u", "d", "g"]',
            "process.max_coupling_orders.QCD=2",
        ),
    )
    assert resolution.effective.process.multiparticles["j"] == ("u", "d", "g")
    assert resolution.effective.process.max_coupling_orders["QCD"] == 2


def test_selected_source_helicity_overrides_accept_signed_values() -> None:
    resolution = resolve_config(
        {"action": "generate"},
        overrides=(
            "process.selected_source_helicities.1=-1",
            "process.selected_source_helicities.2=1",
        ),
    )
    assert resolution.effective.process.selected_source_helicities == {"1": -1, "2": 1}


def test_nullable_and_bare_enum_overrides() -> None:
    assert parse_override("model.cache_dir=null").value is None
    assert parse_override("color.accuracy=nlc").value == "nlc"
    with pytest.raises(ConfigurationError, match="must be true or false"):
        parse_override("model.cache=1")


def test_eager_evaluator_card_and_dotted_overrides_round_trip() -> None:
    pytest.importorskip("tomli_w")
    config = resolve_config(
        {
            "action": "generate",
            "evaluator": {
                "execution_mode": "eager",
                "eager": {"point_tile_size": 2048, "workspace_mib": 384},
            },
        },
        overrides=("evaluator.eager.workspace_mib=512",),
    ).effective

    assert config.evaluator.execution_mode is EvaluatorExecutionMode.EAGER
    assert config.evaluator.eager.point_tile_size == 2048
    assert config.evaluator.eager.workspace_mib == 512
    plain = config_to_dict(config)
    assert plain["evaluator"]["eager"] == {  # type: ignore[index]
        "point_tile_size": 2048,
        "workspace_mib": 512,
    }
    assert resolve_config(plain).effective == config
    serialized = config_to_toml(config)
    assert 'execution_mode = "eager"' in serialized
    assert "[evaluator.eager]" in serialized
    assert resolve_config(tomllib.loads(serialized)).effective == config


def test_jit_compression_card_and_dotted_override_round_trip() -> None:
    pytest.importorskip("tomli_w")
    config = resolve_config(
        {
            "action": "generate",
            "evaluator": {"jit": {"compress": False}},
        },
        overrides=("evaluator.jit.compress=true",),
    ).effective

    assert config.evaluator.jit.compress is True
    plain = config_to_dict(config)
    assert plain["evaluator"]["jit"]["compress"] is True  # type: ignore[index]
    assert resolve_config(plain).effective == config
    serialized = config_to_toml(config)
    assert "compress = true" in serialized
    assert resolve_config(tomllib.loads(serialized)).effective == config


@pytest.mark.parametrize("mode", ("compiled", "recurrence", "on-the-fly", "eager"))
@pytest.mark.parametrize("level", (0, 1, 2, 3))
def test_jit_compression_auto_resolves_only_for_compiled_o2(
    mode: str, level: int
) -> None:
    resolution = resolve_config(
        {
            "action": "generate",
            "evaluator": {"execution_mode": mode, "jit": {"optimization_level": level}},
        }
    )

    assert resolution.requested.evaluator.jit.compress == "auto"
    assert resolution.effective.evaluator.jit.compress is (
        mode == "compiled" and level == 2
    )
    assert config_to_dict(resolution.effective)["evaluator"]["jit"]["compress"] is (  # type: ignore[index]
        mode == "compiled" and level == 2
    )
    # A mode override must resolve the original auto intent, not the old default.
    compiled = resolve_config(
        config_to_dict(resolution.requested),
        overrides=('evaluator.execution_mode="compiled"',),
    )
    assert compiled.effective.evaluator.jit.compress is (level == 2)


@pytest.mark.parametrize("compress", (True, False))
def test_compiled_o2_compression_explicit_card_and_override_win(compress: bool) -> None:
    card = {
        "action": "generate",
        "evaluator": {"execution_mode": "compiled", "jit": {"compress": compress}},
    }
    assert resolve_config(card).effective.evaluator.jit.compress is compress
    override = f"evaluator.jit.compress={str(not compress).lower()}"
    assert (
        resolve_config(card, overrides=(override,)).effective.evaluator.jit.compress
        is not compress
    )


@pytest.mark.parametrize("backend", ("cpp", "asm"))
def test_non_jit_backends_do_not_enable_compression(backend: str) -> None:
    config = resolve_config(
        {
            "action": "generate",
            "evaluator": {"execution_mode": "compiled", "backend": backend},
        }
    ).effective
    assert config.evaluator.jit.compress is False


def test_recurrence_evaluator_card_and_dotted_overrides_round_trip() -> None:
    pytest.importorskip("tomli_w")
    config = resolve_config(
        {
            "action": "generate",
            "evaluator": {
                "execution_mode": "recurrence",
                "recurrence": {"point_tile_size": 2048, "workspace_mib": 384},
            },
        },
        overrides=("evaluator.recurrence.workspace_mib=512",),
    ).effective

    assert config.evaluator.execution_mode is EvaluatorExecutionMode.RECURRENCE
    assert config.evaluator.recurrence.point_tile_size == 2048
    assert config.evaluator.recurrence.workspace_mib == 512
    plain = config_to_dict(config)
    assert plain["evaluator"]["recurrence"] == {  # type: ignore[index]
        "point_tile_size": 2048,
        "workspace_mib": 512,
    }
    assert resolve_config(plain).effective == config
    serialized = config_to_toml(config)
    assert 'execution_mode = "recurrence"' in serialized
    assert "[evaluator.recurrence]" in serialized
    assert resolve_config(tomllib.loads(serialized)).effective == config


def test_lc_flow_layout_card_and_dotted_overrides_round_trip() -> None:
    pytest.importorskip("tomli_w")
    config = resolve_config(
        {
            "action": "generate",
            "color": {"accuracy": "lc", "lc_flow_layout": "topology-replay"},
        },
        overrides=("color.lc_flow_layout=all-flow-union",),
    ).effective

    assert config.color.lc_flow_layout is LCFlowLayout.ALL_FLOW_UNION
    plain = config_to_dict(config)
    assert plain["color"] == {  # type: ignore[index]
        "accuracy": "lc",
        "contraction": "direct",
        "fft_basis": "trace",
        "lc_flow_layout": "all-flow-union",
    }
    assert resolve_config(plain).effective == config
    serialized = config_to_toml(config)
    assert 'lc_flow_layout = "all-flow-union"' in serialized
    assert resolve_config(tomllib.loads(serialized)).effective == config


def test_automatic_fft_card_round_trip_and_direct_override() -> None:
    card = {"action": "generate", "color": {"accuracy": "full"}}
    automatic = resolve_config(card).effective
    assert automatic.color.contraction is ColorContraction.AUTO
    assert automatic.color.fft_basis is ColorFFTBasis.ADJOINT
    assert resolve_config(config_to_dict(automatic)).effective == automatic
    direct = resolve_config(card, overrides=("color.contraction=direct",)).effective
    assert direct.color.contraction is ColorContraction.DIRECT
    assert direct.color.fft_basis is ColorFFTBasis.TRACE


@pytest.mark.parametrize("fft_basis", tuple(ColorFFTBasis))
def test_symmetric_group_fft_card_and_dotted_override_round_trip(
    fft_basis: ColorFFTBasis,
) -> None:
    pytest.importorskip("tomli_w")
    config = resolve_config(
        {
            "action": "generate",
            "color": {"accuracy": "full", "contraction": "direct"},
            "evaluator": {"execution_mode": "recurrence"},
        },
        overrides=(
            "color.contraction=symmetric-group-fft",
            f"color.fft_basis={fft_basis}",
        ),
    ).effective

    assert config.color.contraction is ColorContraction.SYMMETRIC_GROUP_FFT
    assert config.color.fft_basis is fft_basis
    plain = config_to_dict(config)
    assert plain["color"] == {  # type: ignore[index]
        "accuracy": "full",
        "contraction": "symmetric-group-fft",
        "fft_basis": fft_basis.value,
        "lc_flow_layout": "topology-replay",
    }
    assert resolve_config(plain).effective == config
    serialized = config_to_toml(config)
    assert 'contraction = "symmetric-group-fft"' in serialized
    assert f'fft_basis = "{fft_basis}"' in serialized
    assert resolve_config(tomllib.loads(serialized)).effective == config


def test_packaged_adjoint_fft_card_selects_certified_pure_gluon_example() -> None:
    card = Path(__file__).resolve().parents[2] / "examples/builtin_sm_adjoint_fft.toml"
    config = load_config(card)
    assert config.color.contraction is ColorContraction.SYMMETRIC_GROUP_FFT
    assert config.color.fft_basis is ColorFFTBasis.ADJOINT
    assert config.color.accuracy == "full"
    assert config.evaluator.execution_mode is EvaluatorExecutionMode.RECURRENCE
    assert config.process.entries[0].expression == "g g > g g g"


@pytest.mark.parametrize(
    "override",
    (
        "evaluator.execution_mode=streaming",
        "evaluator.eager.point_tile_size=0",
        "evaluator.eager.workspace_mib=-1",
    ),
)
def test_invalid_eager_evaluator_overrides_are_rejected(override: str) -> None:
    with pytest.raises(ConfigurationError, match="evaluator"):
        resolve_config({"action": "generate"}, overrides=(override,))


@pytest.mark.parametrize(
    "override",
    (
        "evaluator.recurrence.point_tile_size=0",
        "evaluator.recurrence.workspace_mib=-1",
    ),
)
def test_invalid_recurrence_evaluator_overrides_are_rejected(override: str) -> None:
    with pytest.raises(ConfigurationError, match=r"evaluator\.recurrence"):
        resolve_config({"action": "generate"}, overrides=(override,))


def test_unknown_card_and_override_fields_are_errors(tmp_path: Path) -> None:
    card = tmp_path / "bad.toml"
    card.write_text('action = "generate"\n[generator]\nworkers = 2\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown configuration field"):
        load_config(card)
    with pytest.raises(ConfigurationError, match="optimization_level"):
        parse_override("evaluator.jit.optimisation_level=2")


def test_removed_parallel_process_fields_suggest_typed_entries() -> None:
    with pytest.raises(ConfigurationError, match=r"process\.entries"):
        resolve_config(
            {
                "action": "generate",
                "process": {"requests": ["d d~ > z"], "names": ["ddbar_z"]},
            }
        )


def test_process_entry_override_and_optional_name() -> None:
    config = resolve_config(
        {"action": "generate"},
        overrides=(
            'process.entries=[{ expression = "d d~ > z", name = "ddbar_z" }, '
            '{ expression = "u u~ > z" }]',
        ),
    ).effective
    assert config.process.entries == (
        ProcessEntry("d d~ > z", "ddbar_z"),
        ProcessEntry("u u~ > z"),
    )


def test_card_validation_aggregates_entry_field_and_unknown_path_errors() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        resolve_config(
            {
                "action": "generate",
                "process": {
                    "entries": [
                        {"expresion": "d d~ > z", "name": "1-invalid"},
                        7,
                    ]
                },
                "generation": {"workers": False},
                "evaluator": {"batch_size": "many"},
                "outpt": {"format": "json"},
            }
        )

    message = str(exc_info.value)
    assert "configuration errors" in message
    assert "process.entries[0].expresion" in message
    assert "process entry expression" in message
    assert "process entry name" in message
    assert "process.entries[1] must be a table" in message
    assert "generation.workers" in message
    assert "evaluator.batch_size" in message
    assert "output" in message


@pytest.mark.parametrize(
    "path",
    (
        "color.coverage",
        "color.flow_ids",
        "generation.validation.zero_current_filter",
        "generation.validation.current_merging",
    ),
)
def test_removed_generation_dotted_paths_are_errors(path: str) -> None:
    with pytest.raises(ConfigurationError, match="unknown configuration field"):
        parse_override(f"{path}=true")


def test_config_serialization_uses_plain_paths_and_lists(tmp_path: Path) -> None:
    config = resolve_config(
        {"action": "evaluate", "evaluation": {"artifact": "artifact"}},
        base_dir=tmp_path,
    ).effective
    plain = config_to_dict(config)
    assert plain["evaluation"]["artifact"] == str((tmp_path / "artifact").resolve())  # type: ignore[index]
    assert plain["benchmark"]["helicity_ids"] == []  # type: ignore[index]


def test_resolution_serialization_records_requested_effective_and_reason() -> None:
    resolution = resolve_config(
        {"action": "generate", "generation": {"workers": 4}},
        clamps=(ClampRequest("generation.workers", 1, "restricted mode"),),
    )
    plain = resolution_to_dict(resolution)
    assert plain["requested"]["generation"]["workers"] == 4  # type: ignore[index]
    assert plain["effective"]["generation"]["workers"] == 1  # type: ignore[index]
    assert plain["adjustments"] == [
        {
            "path": "generation.workers",
            "requested": 4,
            "effective": 1,
            "reason": "restricted mode",
        }
    ]


def test_config_toml_round_trip_omits_nulls_without_changing_defaults() -> None:
    pytest.importorskip("tomli_w")
    original = resolve_config(
        {
            "action": "generate",
            "process": {"entries": [{"expression": "d d~ > z"}]},
            "generation": {"workers": 2},
        }
    ).effective
    serialized = config_to_toml(original)
    assert "null" not in serialized
    restored = resolve_config(tomllib.loads(serialized)).effective
    assert restored == original
