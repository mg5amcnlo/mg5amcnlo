# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import argparse
import subprocess
from types import SimpleNamespace

import pytest

from tools.developer import eager_artifact_compare as artifact_compare
from tools.developer import eager_benchmark_matrix as matrix


def test_failed_command_reports_stdout_and_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedProcess:
        returncode = 134
        pid = 123

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            assert timeout == 10.0
            return "native stdout detail", "watchdog stderr detail"

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FailedProcess())

    with pytest.raises(matrix.MatrixError) as failure:
        matrix._run_json(
            ("python", "command.py"),
            timeout=10.0,
            memory_limit_gib=30.0,
            environment={},
        )

    message = str(failure.value)
    assert "stdout (tail):\nnative stdout detail" in message
    assert "stderr (tail):\nwatchdog stderr detail" in message


def test_smoke_suite_selects_the_two_bounded_cases() -> None:
    selected = matrix._selected_cases("smoke", ())

    assert [case.key for case in selected] == ["dd_z_3g", "dd_3q_1g"]
    assert all(case.smoke for case in selected)


def test_suite_defaults_keep_scalar_batch_for_milestone_only() -> None:
    smoke = matrix.parser().parse_args(
        ["--builtin-pack", "pack", "--output-root", "output"]
    )
    milestone = matrix.parser().parse_args(
        [
            "--suite",
            "milestone",
            "--builtin-pack",
            "pack",
            "--output-root",
            "output",
        ]
    )

    assert smoke.batch_sizes is None
    assert milestone.batch_sizes is None
    assert matrix.SMOKE_BATCH_SIZES == (128, 1024)
    assert matrix.DEFAULT_BATCH_SIZES == (1, 128, 1024)


def test_explicit_case_selection_preserves_request_order() -> None:
    selected = matrix._selected_cases("milestone", ("dd_tt_3g", "dd_z_3g"))

    assert [case.key for case in selected] == ["dd_tt_3g", "dd_z_3g"]


def test_unknown_case_fails_closed() -> None:
    with pytest.raises(matrix.MatrixError, match="unknown process case"):
        matrix._selected_cases("smoke", ("not-a-case",))


def test_lc_workloads_choose_computed_nonzero_selectors() -> None:
    physics = {
        "color_components": [
            {"id": "flow:zero", "computed": False},
            {
                "id": "flow:chosen",
                "computed": True,
                "word": [2, 1],
            },
        ],
        "helicities": [
            {"id": "h:zero", "computed": True, "structural_zero": True},
            {
                "id": "h:chosen",
                "computed": True,
                "structural_zero": False,
                "values": [-1, 1],
            },
        ],
        "external_particles": [{"label": 1}, {"label": 2}],
    }

    assert matrix._workloads("lc", physics) == (
        {
            "name": "single-flow-helicity-sum",
            "selectors": {"color_flow": "flow:chosen"},
            "compiled_specialization": {
                "reference_color_order": (2, 1),
                "selected_color_sector_ids": (0,),
            },
        },
        {
            "name": "all-flow-single-helicity",
            "selectors": {"helicity": "h:chosen"},
            "compiled_specialization": {
                "selected_source_helicities": {1: -1, 2: 1},
            },
        },
    )


def test_contracted_color_has_one_summed_workload() -> None:
    assert matrix._workloads("nlc", {}) == ({"name": "summed", "selectors": {}},)
    assert matrix._workloads("full", {}) == ({"name": "summed", "selectors": {}},)


def test_generation_command_keeps_compiled_and_eager_modes_explicit() -> None:
    command = matrix._generation_command(
        matrix.Path("python"),
        process="d d~ > z g",
        artifact=matrix.Path("artifact"),
        model=matrix.Path("model.pyamplicol-model"),
        color="full",
        execution_mode="eager",
    )

    assert command[0] == "python"
    assert command[command.index("--execution-mode") + 1] == "eager"
    assert command[command.index("--color-accuracy") + 1] == "full"
    assert command[command.index("--jit-optimization-level") + 1] == "3"
    assert "--no-post-build-validation" in command


def test_generation_command_encodes_private_compiled_specialization() -> None:
    command = matrix._generation_command(
        matrix.Path("python"),
        process="d d~ > z g",
        artifact=matrix.Path("artifact"),
        model="built-in-sm",
        color="lc",
        execution_mode="compiled",
        process_overrides={
            "reference_color_order": (2, 1, 3, 4),
            "selected_color_sector_ids": (0,),
            "selected_source_helicities": {1: -1, 2: 1},
        },
    )

    overrides = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--set"
    ]
    assert overrides == [
        "process.reference_color_order=[2,1,3,4]",
        "process.selected_color_sector_ids=[0]",
        'process.selected_source_helicities={"1"=-1,"2"=1}',
    ]


def test_profile_command_encodes_both_lc_selector_axes() -> None:
    command = matrix._profile_command(
        matrix.Path("python"),
        artifact=matrix.Path("artifact"),
        process_id="process",
        batch_size=1024,
        target_runtime=5.0,
        minimum_samples=5,
        selectors={"color_flow": "flow:1,2", "helicity": "h:-1,+1"},
    )

    assert command[command.index("--batch-size") + 1] == "1024"
    assert command[command.index("--color-flow") + 1] == "flow:1,2"
    assert command[command.index("--helicity") + 1] == "h:-1,+1"


def test_comparison_command_isolates_exact_evaluation() -> None:
    command = matrix._comparison_command(
        matrix.Path("python"),
        eager_artifact=matrix.Path("eager"),
        compiled_artifact=matrix.Path("compiled"),
        process_id="process",
        selectors={"color_flow": "flow:1,2", "helicity": "h:-1,+1"},
        compiled_selectors="none",
    )

    assert command[:3] == (
        "python",
        "-m",
        "tools.developer.eager_artifact_compare",
    )
    assert command[command.index("--compiled-selectors") + 1] == "none"
    assert command[command.index("--color-flow") + 1] == "flow:1,2"
    assert command[command.index("--helicity") + 1] == "h:-1,+1"


def test_ufo_model_requires_source_and_pack(tmp_path: matrix.Path) -> None:
    arguments = argparse.Namespace(
        models=("ufo-sm",),
        builtin_pack=tmp_path / "builtin.pyamplicol-model",
        ufo_source=None,
        ufo_pack=None,
    )

    with pytest.raises(matrix.MatrixError, match="requires both"):
        matrix._model_specs(arguments)


def test_relative_difference_handles_zero_values() -> None:
    assert artifact_compare._relative_difference(0j, 0j) == 0.0
    assert (
        artifact_compare._relative_difference(1 + 0j, 1 + 1.0e-13j)
        < 1.1e-13
    )


def test_resolved_comparison_checks_every_component_and_identifier() -> None:
    baseline = SimpleNamespace(
        helicity_ids=("h0", "h1"),
        color_ids=("c0", "c1"),
        values=((((1 + 0j), (2 + 0j)), ((3 + 0j), (4 + 0j))),),
    )
    matching = SimpleNamespace(
        helicity_ids=("h0", "h1"),
        color_ids=("c0", "c1"),
        values=((((1 + 1.0e-14j), (2 + 0j)), ((3 + 0j), (4 + 0j))),),
    )
    different = SimpleNamespace(
        helicity_ids=("h0", "h1"),
        color_ids=("c0", "different"),
        values=baseline.values,
    )

    comparison = artifact_compare._resolved_comparison(baseline, matching)
    assert comparison["component_count"] == 4
    assert comparison["passes"] is True
    assert artifact_compare._resolved_comparison(baseline, different)["passes"] is False


def test_generation_assessment_separates_hard_gates_and_soft_targets() -> None:
    records = [
        {
            "case": {"key": "process"},
            "model": "built-in",
            "color": color,
            "compiled_over_eager_core_generation": ratio,
        }
        for color, ratio in (("lc", 2.0), ("nlc", 20.0), ("full", 30.0))
    ]

    hard, soft = matrix._generation_assessment(records)

    assert hard == {
        "nlc_full_each_at_least_7x": True,
        "lc_no_generation_regression": True,
        "per_process_geometric_mean_at_least_7x": True,
    }
    assert soft == {
        "nlc_full_each_at_least_10x": True,
        "per_process_geometric_mean_at_least_10x": True,
    }


def test_generation_assessment_rejects_partial_matrix_and_reports_soft_miss() -> None:
    records = [
        {
            "case": {"key": "process"},
            "model": "built-in",
            "color": "nlc",
            "compiled_over_eager_core_generation": 9.99,
        }
    ]

    hard, soft = matrix._generation_assessment(records)

    assert hard == {
        "nlc_full_each_at_least_7x": True,
        "lc_no_generation_regression": False,
        "per_process_geometric_mean_at_least_7x": False,
    }
    assert soft == {
        "nlc_full_each_at_least_10x": False,
        "per_process_geometric_mean_at_least_10x": False,
    }


def test_reusable_runtime_gate_is_hard_only_for_n4_and_n5() -> None:
    def record(n_final: int, ratio: float) -> dict[str, object]:
        return {
            "case": {"n_final": n_final},
            "workloads": [
                {
                    "profiles": [
                        {
                            "batch_size": batch_size,
                            "compiled_reusable_over_specialized_wall": ratio,
                        }
                        for batch_size in (128, 1024)
                    ]
                }
            ],
        }

    assert matrix._reusable_runtime_gate([record(4, 1.24), record(5, 1.25)])
    assert not matrix._reusable_runtime_gate([record(4, 1.251)])
    assert matrix._reusable_runtime_gate([record(5, 1.20), record(7, 4.0)])


def test_core_generation_excludes_model_loading_and_process_expansion() -> None:
    assert (
        matrix._core_generation_seconds(
            {
                "model-loading": 100.0,
                "process-expansion": 50.0,
                "dag": 2.0,
                "jit": 3.0,
            }
        )
        == 5.0
    )


def test_watchdog_peak_parser_uses_last_guarded_command() -> None:
    stderr = "\n".join(
        (
            "memory-watchdog: command finished exit=0 peak_rss=1.250 GiB",
            "memory-watchdog: command finished exit=0 peak_rss=2.500 GiB",
        )
    )

    assert matrix._watchdog_peak_gib(stderr) == 2.5
    assert matrix._watchdog_peak_gib("no watchdog result") is None


def test_eager_topology_accepts_unreported_plan_v3_maximum_fanout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(
        execution_mode="eager",
        maximum_fanout=None,
        **{
            field: index
            for index, field in enumerate(matrix._TOPOLOGY_FIELDS, start=1)
            if field != "maximum_fanout"
        },
    )
    monkeypatch.setattr(
        "pyamplicol.artifacts.inspect_artifact",
        lambda _artifact: SimpleNamespace(processes=(process,)),
    )

    topology = matrix._eager_topology(matrix.Path("artifact"))

    assert topology == {
        field: index
        for index, field in enumerate(matrix._TOPOLOGY_FIELDS, start=1)
        if field != "maximum_fanout"
    }


def test_topology_gate_compares_equivalent_models_per_process_and_color() -> None:
    topology = {field: index for index, field in enumerate(matrix._TOPOLOGY_FIELDS)}
    records = [
        {
            "case": {"key": "process"},
            "color": "nlc",
            "model": model,
            "eager_topology": topology,
        }
        for model in ("built-in", "ufo-sm")
    ]

    assert matrix._topology_gate(records, require_ufo=True)
    assert matrix._topology_gate(records[:1], require_ufo=False)


def test_topology_gate_rejects_missing_or_different_ufo_plan() -> None:
    topology = {field: index for index, field in enumerate(matrix._TOPOLOGY_FIELDS)}
    built_in = {
        "case": {"key": "process"},
        "color": "full",
        "model": "built-in",
        "eager_topology": topology,
    }
    ufo = {
        **built_in,
        "model": "ufo-sm",
        "eager_topology": {**topology, "invocation_count": 999},
    }

    assert not matrix._topology_gate((built_in,), require_ufo=True)
    assert not matrix._topology_gate((built_in, ufo), require_ufo=True)
