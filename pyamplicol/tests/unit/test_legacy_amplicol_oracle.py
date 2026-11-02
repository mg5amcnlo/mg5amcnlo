# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import struct
import subprocess
import sys
import tomllib
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "developer" / "legacy_amplicol.py"
REPORT = ROOT / "tests" / "fixtures" / "reference" / "legacy-fortran-v1.json"
PHYSICS = ROOT / "tests" / "fixtures" / "reference" / "physics-v1.json"
_ACTIVE_HELICITY = (-1, 1, -1, -1, 1)


def _module():
    spec = importlib.util.spec_from_file_location("legacy_amplicol_oracle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_path_import_works_in_an_isolated_interpreter(tmp_path: Path) -> None:
    source = "\n".join(
        (
            "import importlib.util, json, pathlib, sys",
            f"path = pathlib.Path({str(SCRIPT)!r})",
            "spec = importlib.util.spec_from_file_location('isolated_legacy', path)",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules[spec.name] = module",
            "spec.loader.exec_module(module)",
            "print(json.dumps([str(module.ROOT), module.ProcessEntry.__name__]))",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", source],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [str(ROOT), "ProcessEntry"]
    assert completed.stderr == ""


def test_facade_reexports_extracted_oracle_helpers() -> None:
    module = _module()
    package = module._ORACLE_PACKAGE
    model = importlib.import_module(f"{package}.model")
    processes = importlib.import_module(f"{package}.processes")
    probe = importlib.import_module(f"{package}.probe")
    evidence = importlib.import_module(f"{package}.evidence")

    assert module.ROOT == model.ROOT == ROOT
    assert module.LegacyOracleError is model.LegacyOracleError
    assert module.ProcessEntry is model.ProcessEntry
    assert module.LcRowPartition is model.LcRowPartition
    assert module.CompilerProvenance is model.CompilerProvenance
    assert module.ProbeResult is model.ProbeResult
    assert module.SelectedFlowProbeResult is model.SelectedFlowProbeResult
    assert module.parse_process_file is processes.parse_process_file
    assert module.select_process_entry is processes.select_process_entry
    assert (
        module.select_generated_process_entry
        is processes.select_generated_process_entry
    )
    assert module.source_mapped_color_order is processes.source_mapped_color_order
    assert (
        module.validate_selected_flow_quark_line_scope
        is processes.validate_selected_flow_quark_line_scope
    )
    assert (
        module.validate_direct_color_probe_quark_line_scope
        is processes.validate_direct_color_probe_quark_line_scope
    )
    assert module._permutation is processes._permutation
    assert module._parse_probe_output is probe._parse_probe_output
    assert (
        module._parse_selected_flow_probe_output
        is probe._parse_selected_flow_probe_output
    )
    assert module._binary64_input_sha256 is probe._binary64_input_sha256
    assert module._resolved_single_lc_row is probe._resolved_single_lc_row
    assert module._reference_fixture_v2_module is evidence._reference_fixture_v2_module
    assert module._semantic_replay_command is evidence._semantic_replay_command


def test_fortran_physics_fixture_is_not_bound_to_checkout_revision() -> None:
    module = _module()
    fixture = {
        "dependencies": [
            {
                "id": "dependency:legacy-amplicol",
                "name": "Pinned legacy Fortran AmpliCol",
                "revision": "historical-reference-revision",
            }
        ]
    }

    assert module._fortran_dependency_ids(fixture) == (
        "dependency:legacy-amplicol",
    )


def test_direct_script_cli_preserves_help_and_error_exit_codes(
    tmp_path: Path,
) -> None:
    help_result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    invalid_result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--jobs", "not-an-integer"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    missing_checkout_result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--repository",
            str(tmp_path / "missing-checkout"),
            "--no-build",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert help_result.returncode == 0
    assert help_result.stdout.startswith("usage: legacy_amplicol.py")
    assert help_result.stderr == ""
    assert invalid_result.returncode == 2
    assert invalid_result.stdout == ""
    assert "invalid int value: 'not-an-integer'" in invalid_result.stderr
    assert missing_checkout_result.returncode == 2
    assert missing_checkout_result.stdout == ""
    assert missing_checkout_result.stderr.startswith(
        "error: legacy AmpliCol checkout is absent:"
    )
    assert "Traceback" not in missing_checkout_result.stderr


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _refresh_evidence_record_hash(module: Any, record: dict[str, Any]) -> None:
    reference = module._reference_fixture_v2_module()
    record["evidence_record_sha256"] = reference.evidence_record_sha256(record)


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _v2_fixture(revision: str) -> dict[str, Any]:
    points = []
    for index, point_class in enumerate(("generic", "generic", "generic", "stress")):
        generic_final_states = (
            (
                ["250", "250", "0", "0"],
                ["375", "-125", "353.5533905932738", "0"],
                ["375", "-125", "-353.5533905932738", "0"],
            ),
            (
                ["250", "0", "250", "0"],
                ["375", "0", "-125", "353.5533905932738"],
                ["375", "0", "-125", "-353.5533905932738"],
            ),
            (
                ["250", "0", "0", "250"],
                ["375", "353.5533905932738", "0", "-125"],
                ["375", "-353.5533905932738", "0", "-125"],
            ),
        )
        final_states = (
            (
                ["499.999", "0", "0", "499.999"],
                ["500", "0", "0", "-500"],
                ["0.001", "0", "0", "0.001"],
            )
            if point_class == "stress"
            else generic_final_states[index]
        )
        points.append(
            {
                "id": f"point:{point_class}-{index}",
                "process_id": "process:dd-zgg",
                "class": point_class,
                "algorithm": {
                    "name": "deterministic-test-point",
                    "version": "1",
                    "rng": "PCG64" if point_class == "generic" else None,
                    "seed": index if point_class == "generic" else None,
                },
                "sqrt_s": "1000",
                "momenta": [
                    ["500", "0", "0", "500"],
                    ["500", "0", "0", "-500"],
                    *final_states,
                ],
                "masses": ["0", "0", "0", "0", "0"],
                "arithmetic_precision_bits": 53,
                "round_trip_decimal_digits": 17,
                "certified_decimal_digits": 12,
                "stress_metric": (
                    {
                        "kind": "minimum-final-energy-fraction",
                        "value": "0.000001",
                    }
                    if point_class == "stress"
                    else None
                ),
            }
        )

    domains = ((-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1))
    helicities = []
    for index, values in enumerate(product(*domains)):
        identifier = "h:" + ",".join(f"{value:+d}" for value in values)
        active = values == _ACTIVE_HELICITY
        helicities.append(
            {
                "id": identifier,
                "index": index,
                "values": list(values),
                "computed": active,
                "structural_zero": not active,
                "representative_id": identifier,
                "coefficient": "1" if active else "0",
            }
        )
    lc_colors = [
        {
            "kind": "lc-flow",
            "id": "flow:2,4,5,1",
            "index": 0,
            "word": [2, 4, 5, 1],
            "computed": True,
            "representative_id": "flow:2,4,5,1",
            "coefficient": "1",
        },
        {
            "kind": "lc-flow",
            "id": "flow:2,5,4,1",
            "index": 1,
            "word": [2, 5, 4, 1],
            "computed": True,
            "representative_id": "flow:2,5,4,1",
            "coefficient": "1",
        },
    ]
    contracted_color = [
        {
            "kind": "contracted-color",
            "id": "color:contracted",
            "index": 0,
            "description": "fully contracted color",
        }
    ]
    normalization = {
        "average_factor": "36",
        "color_factor": "27",
        "identical_factor": "2",
        "global_coupling_factor": "1",
        "quark_line_partner_factor": "1",
        "couplings_in_stage_evaluators": True,
    }

    def case(color_accuracy: str, total: str) -> dict[str, Any]:
        case_id = f"case:dd-zgg-{color_accuracy}"
        lc = color_accuracy == "lc"
        active_helicity_id = "h:" + ",".join(
            f"{value:+d}" for value in _ACTIVE_HELICITY
        )
        reduction_colors = (
            ("flow:2,4,5,1", "flow:2,5,4,1") if lc else ("color:contracted",)
        )
        values = [
            (["1.25", "0.75"] if lc else [total])
            if tuple(helicity["values"]) == _ACTIVE_HELICITY
            else (["0", "0"] if lc else ["0"])
            for helicity in helicities
        ]
        return {
            "id": case_id,
            "case_kind": "substantive",
            "model_id": "model:builtin-sm",
            "process_id": "process:dd-zgg",
            "color_accuracy": color_accuracy,
            "point_policy": "standard",
            "point_ids": [point["id"] for point in points],
            "coverage": {
                "helicities": "complete",
                "color": "complete" if lc else "contracted",
                "color_kind": "physical-lc-flows" if lc else "contracted-color",
                "helicity_count": len(helicities),
                "color_component_count": 2 if lc else 1,
                "structural_zero_helicity_count": len(helicities) - 1,
            },
            "selectors": {
                "helicity": True,
                "color_flow": lc,
                "omitted_helicity": "all-components",
                "omitted_color": "all-components" if lc else "contracted-component",
            },
            "normalization": normalization,
            "topology": {
                "currents": 17,
                "interactions": 31,
                "roots": 4,
                "reduction_groups": len(reduction_colors),
            },
            "artifact_physics_sha256": _sha(f"physics:{case_id}"),
            "artifact_execution_sha256": _sha(f"execution:{case_id}"),
            "physics_case_sha256": "0" * 64,
            "axes": {
                "helicities": helicities,
                "colors": lc_colors if lc else contracted_color,
            },
            "reduction": {
                "kind": "lc-diagonal" if lc else "contracted-color",
                "cell_semantics": (
                    "sum-all-contributing-groups" if lc else "fully-contracted-color"
                ),
                "groups": [
                    {
                        "id": f"reduction:{index}",
                        "representative_helicity_id": active_helicity_id,
                        "representative_color_id": color_id,
                        "physical_helicity_ids": [active_helicity_id],
                        "physical_color_ids": [color_id],
                    }
                    for index, color_id in enumerate(reduction_colors)
                ],
                "plan_sha256": "0" * 64,
            },
            "observations": [
                {
                    "point_id": point["id"],
                    "arithmetic_precision_bits": 53,
                    "round_trip_decimal_digits": 17,
                    "certified_decimal_digits": 10,
                    "values": values,
                    "total": total,
                    "evidence_refs": [f"evidence:fortran:{case_id}:{point['id']}"],
                }
                for point in points
            ],
        }

    return {
        "fixture_schema_version": 2,
        "kind": "pyamplicol-reference-physics",
        "fixture_id": "fixture:fortran-v2-test",
        "provenance": {
            "source_repository": "https://github.com/mg5amcnlo/pyamplicol",
            "source_revision": "1" * 40,
            "source_tree_sha256": _sha("source-tree"),
            "captured_at": "2026-07-16T12:00:00Z",
            "capture_command": ["pyamplicol", "capture-reference"],
            "working_tree_clean": True,
            "memory_watchdog_gb": 30,
        },
        "dependencies": [
            {
                "id": "dependency:legacy-amplicol",
                "name": "legacy Fortran AmpliCol",
                "version": "pinned-git",
                "revision": revision,
                "content_sha256": _sha("legacy-amplicol"),
                "serialization_abi": None,
                "license": "GPL-3.0-or-later",
            }
        ],
        "models": [
            {
                "id": "model:builtin-sm",
                "name": "built-in-sm",
                "source_kind": "built-in-sm",
                "content_sha256": _sha("model"),
                "compiled_model_sha256": _sha("compiled-model"),
                "compiled_schema_version": 6,
                "restriction": None,
                "dependency_ids": [],
                "parameter_defaults": {},
            }
        ],
        "processes": [
            {
                "id": "process:dd-zgg",
                "expression": "d d~ > z g g",
                "external_pdgs": [1, -1, 23, 21, 21],
                "external_labels": [1, 2, 3, 4, 5],
                "external_leg_ids": [
                    "leg:incoming-d",
                    "leg:incoming-dbar",
                    "leg:outgoing-z",
                    "leg:outgoing-g-1",
                    "leg:outgoing-g-2",
                ],
                "external_spins": [2, 2, 3, 3, 3],
                "external_colors": [3, -3, 1, 8, 8],
                "external_masses": ["0", "0", "0", "0", "0"],
                "external_helicity_domains": [list(domain) for domain in domains],
                "initial_state_count": 2,
                "alias_of": None,
                "final_state_permutation": None,
            }
        ],
        "points": points,
        "cases": [case("lc", "2"), case("nlc", "1.5"), case("full", "1.4")],
        "evidence_sets": ["oracle:legacy-fortran-amplicol"],
    }


def test_process_file_parser_preserves_fortran_external_order(tmp_path: Path) -> None:
    module = _module()
    process_file = tmp_path / "processes.txt"
    process_file.write_text(
        """4 1
1 -1 21 23


1

1   1   1   1 4 2 3
1   1   1 -1 21 23   2 3 1 4   1.0
""",
        encoding="utf-8",
    )

    entries = module.parse_process_file(process_file)

    assert entries == (
        module.ProcessEntry(
            group=1,
            integral=1,
            process_pdgs=(1, -1, 21, 23),
            color_order=(2, 3, 1, 4),
        ),
    )
    assert module.select_process_entry(entries, "d d~ > z g") == entries[0]
    assert module._permutation((1, -1, 23, 21), entries[0].process_pdgs) == (
        0,
        1,
        3,
        2,
    )


def test_pinned_branch_checkout_rejects_tracked_edits_but_allows_build_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    repository = tmp_path / "legacy"
    repository.mkdir()
    tracked = repository / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    _git(repository, "init")
    _git(repository, "config", "user.name", "Fixture Author")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "base")
    revision = _git(repository, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(module, "expected_revision", lambda: revision)

    module.validate_checkout(repository)

    (repository / "amplicol_color_probe").write_text("build output\n", encoding="utf-8")
    module.validate_checkout(repository)

    tracked.write_text("unexpected edit\n", encoding="utf-8")
    with pytest.raises(module.LegacyOracleError, match="contains tracked edits"):
        module.validate_checkout(repository)


def test_process_selection_requires_the_declared_fortran_row() -> None:
    module = _module()
    entries = (
        module.ProcessEntry(1, 1, (1, -1, 21, 23), (1, 2, 3, 4)),
        module.ProcessEntry(2, 1, (1, -1, 23, 21), (1, 2, 4, 3)),
    )

    with pytest.raises(module.LegacyOracleError, match="expected declared row"):
        module.select_process_entry(entries[:1], "d d~ > z g")

    entries = (
        module.expected_process_entry("d d~ > z g"),
        entries[1],
    )
    assert module.select_process_entry(entries, "d d~ > z g") == entries[0]
    assert module.matching_process_entries(entries, "d d~ > z g") == (
        entries[1],
        entries[0],
    )


def test_process_selection_rejects_ambiguous_pdg_rows() -> None:
    module = _module()
    expected = module.expected_process_entry("d d~ > z g")

    with pytest.raises(module.LegacyOracleError, match="ambiguous duplicate"):
        module.select_process_entry((expected, expected), "d d~ > z g")

    candidates = (
        module.ProcessEntry(7, 1, expected.process_pdgs, expected.color_order),
        module.ProcessEntry(8, 1, expected.process_pdgs, expected.color_order),
    )
    with pytest.raises(module.LegacyOracleError, match="ambiguous PDG-only"):
        module.select_process_entry(candidates, "d d~ > z g")


@pytest.mark.parametrize(
    ("process", "row_pdgs", "raw_color_order"),
    (
        ("u d~ > w+ g", (2, -1, 21, 24), (2, 3, 1, 4)),
        ("u d~ > e+ ve", (2, -1, -11, 12), (2, 1, 3, 4)),
        (
            "d d~ > z g g g",
            (1, -1, 21, 21, 21, 23),
            (2, 3, 4, 5, 1, 6),
        ),
        ("d d~ > z z", (1, -1, 23, 23), (2, 1, 3, 4)),
        ("d d~ > z z z", (1, -1, 23, 23, 23), (2, 1, 3, 4, 5)),
        ("d d~ > e+ e-", (1, -1, -11, 11), (2, 1, 3, 4)),
        ("d d~ > t t~", (1, -1, 6, -6), (3, 1, 2, 4)),
        (
            "d d~ > t t~ z h",
            (1, -1, 6, -6, 23, 25),
            (3, 1, 2, 4, 5, 6),
        ),
        ("g g > g g g", (21, 21, 21, 21, 21), (1, 2, 3, 4, 5)),
        ("g g > t t~ g", (21, 21, 21, 6, -6), (4, 1, 2, 3, 5)),
    ),
)
def test_generated_process_selection_covers_matrix_reference_families(
    process: str,
    row_pdgs: tuple[int, ...],
    raw_color_order: tuple[int, ...],
) -> None:
    module = _module()
    row = module.ProcessEntry(7, 3, row_pdgs, raw_color_order)
    wanted = module.process_pdgs(process)

    selected, matches = module.select_generated_process_entry(
        (row,),
        generated_process=process,
        wanted_pdgs=wanted,
    )

    assert selected == row
    assert matches == (row,)
    mapped = module.source_mapped_color_order(row, source_pdgs=wanted)
    assert sorted(mapped) == list(range(1, len(wanted) + 1))


def test_generated_process_selection_prefers_exact_external_order() -> None:
    module = _module()
    reordered = module.ProcessEntry(1, 1, (1, -1, -6, 6), (4, 1, 2, 3))
    exact = module.ProcessEntry(2, 1, (1, -1, 6, -6), (3, 1, 2, 4))

    selected, matches = module.select_generated_process_entry(
        (reordered, exact),
        generated_process="d d~ > t t~",
        wanted_pdgs=(1, -1, 6, -6),
    )

    assert selected == exact
    assert matches == (exact, reordered)
    assert (
        module.GENERATED_PROCESS_ROW_SELECTION_POLICY
        == "exact-external-pdg-order-then-process-file-order-v1"
    )


def test_source_mapped_color_order_preserves_raw_fortran_order_separately() -> None:
    module = _module()
    row = module.ProcessEntry(
        1,
        1,
        (2, -1, 21, 24),
        (2, 3, 1, 4),
    )

    mapped = module.source_mapped_color_order(
        row,
        source_pdgs=(2, -1, 24, 21),
    )

    assert row.color_order == (2, 3, 1, 4)
    assert mapped == (2, 4, 1, 3)


def test_source_mapped_color_order_rejects_invalid_fortran_positions() -> None:
    module = _module()
    row = module.ProcessEntry(1, 1, (1, -1, 23), (2, 1, 1))

    with pytest.raises(module.LegacyOracleError, match="must be a permutation"):
        module.source_mapped_color_order(row, source_pdgs=(1, -1, 23))


def test_process_contract_pins_row_and_color_order_multiplicities() -> None:
    module = _module()

    assert module.expected_process_match_count("d d~ > z g g") == 1
    assert module.expected_color_order_count("d d~ > z g g") == 2
    assert module.expected_process_match_count("g g > g g") == 3
    assert module.expected_color_order_count("g g > g g") == 6
    assert module.expected_process_match_count("d d~ > d d~") == 2
    assert module.expected_color_order_count("d d~ > d d~") == 2


def test_oracle_rejects_processes_beyond_two_quark_lines() -> None:
    module = _module()

    assert (
        module._validate_supported_quark_line_scope(
            (1, -1, 2, -2),
            context="two-line fixture",
        )
        == 2
    )
    with pytest.raises(module.LegacyOracleError, match="3 quark lines exceed"):
        module._validate_supported_quark_line_scope(
            (1, -1, 2, -2, 3, -3),
            context="three-line fixture",
        )


def test_selected_flow_scope_does_not_inherit_the_all_flow_line_cap() -> None:
    module = _module()
    pdgs = (1, -1, 2, -2, 3, -3)
    row = module.ProcessEntry(1, 1, pdgs, (1, 2, 3, 4, 5, 6))

    assert (
        module.validate_selected_flow_quark_line_scope(
            pdgs,
            context="three-line selected-flow campaign",
        )
        == 3
    )
    selected, matches = module.select_generated_process_entry(
        (row,),
        generated_process="d d~ > u u~ s s~",
        wanted_pdgs=pdgs,
    )
    assert selected == row
    assert matches == (row,)

    with pytest.raises(module.LegacyOracleError, match="3 quark lines exceed"):
        module._validate_supported_quark_line_scope(
            pdgs,
            context="three-line all-flow fixture",
        )


def test_direct_color_probe_supports_three_but_not_four_quark_lines() -> None:
    module = _module()

    assert (
        module.validate_direct_color_probe_quark_line_scope(
            (1, -1, 2, -2, 3, -3),
            color_accuracy="full",
            context="three-line direct probe",
        )
        == 3
    )
    with pytest.raises(module.LegacyOracleError, match="4 quark lines exceed"):
        module.validate_direct_color_probe_quark_line_scope(
            (1, -1, 2, -2, 3, -3, 4, -4),
            color_accuracy="lc",
            context="four-line direct probe",
        )
    assert (
        module.validate_direct_color_probe_quark_line_scope(
            (1, -1, 2, -2, 3, -3, 21, 21, 21, 21, 21),
            color_accuracy="lc",
            context="large leading-colour three-line direct probe",
        )
        == 3
    )
    with pytest.raises(module.LegacyOracleError, match=r"15120.*color flows exceed"):
        module.validate_direct_color_probe_quark_line_scope(
            (1, -1, 2, -2, 3, -3, 21, 21, 21, 21, 21),
            color_accuracy="nlc",
            context="oversized three-line direct probe",
        )
    with pytest.raises(module.LegacyOracleError, match="color singlets"):
        module.validate_direct_color_probe_quark_line_scope(
            (1, -1, 2, -2, 3, -3, 22),
            color_accuracy="lc",
            context="three-line photon probe",
        )


def test_public_legacy_checkout_uses_noninteractive_https() -> None:
    module = _module()

    assert module.checkout_url() == ("https://github.com/rikkert-frederix/AmpliCol.git")
    assert module.checkout_branch() == "amplicol_with_patches"
    assert module.expected_revision() == "dcb3efbcbe89c51969818cddf48113fba19ad763"


def test_compiler_provenance_records_build_inputs_and_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    repository = tmp_path / "legacy"
    repository.mkdir()
    executable = repository / "amplicol_color_probe"
    executable.write_bytes(b"compiled probe")
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, capture=True):
        del cwd, capture
        commands.append(command)
        if command[0] == "make":
            output = "FFLAGS = -ffast-math -O3\nFC = gfortran\n"
        elif command[-1] == "--version":
            output = "GNU Fortran (GCC) 14.2.0\nCopyright\n"
        elif command[-1] == "-dumpmachine":
            output = "aarch64-apple-darwin24\n"
        else:
            raise AssertionError(command)
        return module.subprocess.CompletedProcess(command, 0, output, "")

    class FakeShutil:
        @staticmethod
        def which(name: str) -> str:
            return f"/toolchain/{name}"

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "shutil", FakeShutil)

    assert module._compiler_provenance(repository) == module.CompilerProvenance(
        identity="gfortran",
        version="GNU Fortran (GCC) 14.2.0",
        flags=("-ffast-math", "-O3"),
        target="aarch64-apple-darwin24",
        executable_sha256=hashlib.sha256(b"compiled probe").hexdigest(),
    )
    assert commands[0] == [
        "make",
        "-f",
        "makefile",
        "PDF_BACKEND=internal",
        "-pn",
        "amplicol_color_probe",
    ]


def test_run_wrapper_uses_the_facade_subprocess_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    observed: dict[str, object] = {}

    class FakeSubprocess:
        @staticmethod
        def run(command, **options):
            observed.update(command=tuple(command), options=options)
            return subprocess.CompletedProcess(command, 0, "ok\n", "")

    monkeypatch.setattr(module, "subprocess", FakeSubprocess)

    completed = module._run(["oracle", "--version"], cwd=tmp_path)

    assert completed.stdout == "ok\n"
    assert observed == {
        "command": ("oracle", "--version"),
        "options": {
            "cwd": tmp_path,
            "text": True,
            "capture_output": True,
        },
    }


def test_color_probe_runs_outside_the_legacy_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    repository = tmp_path / "legacy"
    repository.mkdir()
    process_file = tmp_path / "processes.txt"
    process_file.write_text("fixture\n", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_run(command, *, cwd, capture=True):
        observed.update(command=tuple(command), cwd=cwd, capture=capture)
        output = """AMPICOL_COLOR_PROBE_CURRENTS 1
AMPICOL_COLOR_PROBE_VERTICES 1
AMPICOL_COLOR_PROBE_AMPLITUDES 1
AMPICOL_COLOR_PROBE_COLOR_ORDERS 1
AMPICOL_COLOR_PROBE_COMPONENTS 1.0 1.0 1.0
AMPICOL_COLOR_PROBE_VALUE lc 1 1 1.0
"""
        return module.subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(module, "_run", fake_run)
    entry = module.ProcessEntry(1, 1, (1, -1, 23), (1, 2, 3))

    result = module.run_color_probe(
        repository,
        process_file=process_file,
        entry=entry,
        source_pdgs=(1, -1, 23),
        momenta=(
            (50.0, 0.0, 0.0, 50.0),
            (50.0, 0.0, 0.0, -50.0),
            (100.0, 0.0, 0.0, 0.0),
        ),
        color_accuracy="lc",
    )

    assert result.value == 1.0
    assert observed["cwd"] != repository
    assert observed["command"][0] == str(
        (repository / "amplicol_color_probe").resolve()
    )


def _selected_flow_probe_output(
    *,
    pdgs: tuple[int, ...] = (1, -1, 2, -2, 3, -3),
    color_order: tuple[int, ...] = (2, 1, 3, 4, 5, 6),
    helicities: tuple[int, ...] | None = None,
) -> str:
    lines = [
        "AMPICOL_SELECTED_FLOW_PROBE_VALUE 3 1 6.3359823001718900E-10",
        "AMPICOL_SELECTED_FLOW_PROBE_PDGS "
        f"{len(pdgs)} {' '.join(str(value) for value in pdgs)}",
        "AMPICOL_SELECTED_FLOW_PROBE_COLOR_ORDER "
        f"{len(color_order)} {' '.join(str(value) for value in color_order)}",
        "AMPICOL_SELECTED_FLOW_PROBE_AMPLITUDES 4",
        "AMPICOL_SELECTED_FLOW_PROBE_COLOR_FACTOR 27",
        "AMPICOL_SELECTED_FLOW_PROBE_IDENTICAL_FACTOR 36",
        "AMPICOL_SELECTED_FLOW_PROBE_SINGLET_VERTICES 0",
        "AMPICOL_SELECTED_FLOW_PROBE_NORMALIZATION 1.3425000000000000E-01",
    ]
    if helicities is not None:
        lines.extend(
            (
                "AMPICOL_SELECTED_FLOW_PROBE_HELICITIES "
                f"{len(helicities)} {' '.join(str(value) for value in helicities)}",
                "AMPICOL_SELECTED_FLOW_PROBE_FIXED_HELICITY_VALUE "
                "2.5000000000000000E+00",
            )
        )
    return "\n".join(lines)


def test_selected_flow_probe_parser_records_generated_row_metadata() -> None:
    module = _module()

    result = module._parse_selected_flow_probe_output(_selected_flow_probe_output())

    assert result == module.SelectedFlowProbeResult(
        value=6.33598230017189e-10,
        group=3,
        integral=1,
        process_pdgs=(1, -1, 2, -2, 3, -3),
        color_order=(2, 1, 3, 4, 5, 6),
        amplitudes=4,
        color_factor=27,
        identical_factor=36,
        singlet_vertices=0,
        normalization=0.13425,
        value_decimal=Decimal("6.3359823001718900E-10"),
        normalization_decimal=Decimal("1.3425000000000000E-01"),
    )


def test_selected_flow_probe_parser_records_fixed_physical_helicity() -> None:
    module = _module()
    helicities = (-1, 1, -1, 1, -1, 1)

    result = module._parse_selected_flow_probe_output(
        _selected_flow_probe_output(helicities=helicities)
    )

    assert result.helicities == helicities
    assert result.fixed_helicity_value == 2.5
    assert result.fixed_helicity_value_decimal == Decimal("2.5000000000000000E+00")


def test_selected_flow_library_probe_supports_three_quark_lines_and_indexed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    repository = tmp_path / "legacy"
    repository.mkdir()
    source_pdgs = (1, -1, 2, -2, 3, -3)
    entry = module.ProcessEntry(
        group=3,
        integral=1,
        process_pdgs=(1, -1, 2, -2, 3, -3),
        color_order=(2, 1, 3, 4, 5, 6),
    )
    momenta = tuple(
        (float(index), float(index + 1), float(index + 2), float(index + 3))
        for index in range(0, 24, 4)
    )
    observed: dict[str, object] = {}

    def fake_run(command, *, cwd, capture=True):
        momentum_path = Path(command[4])
        observed.update(
            command=tuple(command),
            cwd=cwd,
            capture=capture,
            momenta=momentum_path.read_text(encoding="utf-8"),
        )
        return module.subprocess.CompletedProcess(
            command,
            0,
            _selected_flow_probe_output(helicities=(-1, 1, -1, 1, -1, 1)),
            "",
        )

    monkeypatch.setattr(module, "_run", fake_run)

    result = module.run_selected_flow_library_probe(
        repository,
        entry=entry,
        source_pdgs=source_pdgs,
        momenta=momenta,
        helicities=(-1, 1, -1, 1, -1, 1),
        points=7,
    )

    assert result.value == pytest.approx(6.33598230017189e-10)
    assert result.process_pdgs == entry.process_pdgs
    assert result.color_order == entry.color_order
    assert observed["cwd"] == repository
    assert observed["command"][1:4] == ("7", "3", "1")
    assert observed["command"][5:] == ("-1", "1", "-1", "1", "-1", "1")
    assert observed["momenta"] == "\n".join(
        " ".join(format(component, ".17g") for component in vector)
        for vector in momenta
    ) + "\n"


def test_selected_flow_library_probe_accepts_canonicalized_row_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    repository = tmp_path / "legacy"
    repository.mkdir()
    entry = module.ProcessEntry(
        group=3,
        integral=1,
        process_pdgs=(1, -1, 2, -2, 3, -3),
        color_order=(2, 1, 3, 4, 5, 6),
    )

    monkeypatch.setattr(
        module,
        "_run",
        lambda command, *, cwd, capture=True: module.subprocess.CompletedProcess(
            command,
            0,
            _selected_flow_probe_output(
                pdgs=(2, -2, -3, 3, -1, 1),
                color_order=(1, 3, 4, 5, 6, 2),
            ),
            "",
        ),
    )

    result = module.run_selected_flow_library_probe(
        repository,
        entry=entry,
        source_pdgs=entry.process_pdgs,
        momenta=((1.0, 2.0, 3.0, 4.0),) * 6,
    )

    assert result.process_pdgs == (2, -2, -3, 3, -1, 1)
    assert result.color_order == (1, 3, 4, 5, 6, 2)


def test_selected_flow_library_probe_maps_source_helicity_to_generated_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    repository = tmp_path / "legacy"
    repository.mkdir()
    source_pdgs = (1, -1, 23, 21)
    entry = module.ProcessEntry(
        group=3,
        integral=1,
        process_pdgs=(1, -1, 21, 23),
        color_order=(2, 3, 1, 4),
    )
    observed: dict[str, tuple[str, ...]] = {}

    def fake_run(command, *, cwd, capture=True):
        del cwd, capture
        observed["command"] = tuple(command)
        return module.subprocess.CompletedProcess(
            command,
            0,
            _selected_flow_probe_output(
                pdgs=entry.process_pdgs,
                color_order=entry.color_order,
                helicities=(-1, 1, -1, 0),
            ),
            "",
        )

    monkeypatch.setattr(module, "_run", fake_run)

    result = module.run_selected_flow_library_probe(
        repository,
        entry=entry,
        source_pdgs=source_pdgs,
        momenta=((1.0, 2.0, 3.0, 4.0),) * 4,
        helicities=(-1, 1, 0, -1),
    )

    assert result.helicities == (-1, 1, -1, 0)
    assert observed["command"][5:] == ("-1", "1", "-1", "0")


def test_selected_flow_library_probe_rejects_generated_pdg_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    repository = tmp_path / "legacy"
    repository.mkdir()
    entry = module.ProcessEntry(
        group=3,
        integral=1,
        process_pdgs=(1, -1, 2, -2, 3, -3),
        color_order=(2, 1, 3, 4, 5, 6),
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, *, cwd, capture=True: module.subprocess.CompletedProcess(
            command,
            0,
            _selected_flow_probe_output(pdgs=(1, -1, 2, -2, 4, -4)),
            "",
        ),
    )

    with pytest.raises(module.LegacyOracleError, match="PDG multiset"):
        module.run_selected_flow_library_probe(
            repository,
            entry=entry,
            source_pdgs=entry.process_pdgs,
            momenta=((1.0, 2.0, 3.0, 4.0),) * 6,
        )


def test_binary64_input_hash_covers_exact_post_permutation_values() -> None:
    module = _module()
    source_pdgs = (1, -1, 23, 21)
    target_pdgs = (1, -1, 21, 23)
    momenta = (
        (1.0, 2.0, 3.0, 4.0),
        (5.0, 6.0, 7.0, 8.0),
        (9.0, 10.0, 11.0, 12.0),
        (13.0, 14.0, 15.0, 16.0),
    )
    transmitted = (*momenta[0], *momenta[1], *momenta[3], *momenta[2])

    assert (
        module._binary64_input_sha256(
            source_pdgs,
            target_pdgs,
            momenta,
        )
        == hashlib.sha256(struct.pack(">16d", *transmitted)).hexdigest()
    )
    assert (
        module._binary64_input_sha256(
            source_pdgs,
            target_pdgs,
            momenta,
        )
        != hashlib.sha256(
            struct.pack(">16d", *(value for row in momenta for value in row))
        ).hexdigest()
    )


def test_probe_output_parser_records_values_and_topology() -> None:
    module = _module()
    result = module._parse_probe_output(
        """AMPICOL_COLOR_PROBE_CURRENTS 7
AMPICOL_COLOR_PROBE_VERTICES 4
AMPICOL_COLOR_PROBE_AMPLITUDES 1
AMPICOL_COLOR_PROBE_COLOR_ORDERS 1
AMPICOL_COLOR_PROBE_COMPONENTS 4.0E-1 3.5E-1 3.4E-1
AMPICOL_COLOR_PROBE_VALUE full 1 1 3.4E-1
AMPICOL_COLOR_PROBE_FLOW_VALUE 1 2.0E-1
AMPICOL_COLOR_PROBE_FLOW_PERM 1 2 4 1
AMPICOL_COLOR_PROBE_FLOW_VALUE 2 1.4E-1
AMPICOL_COLOR_PROBE_FLOW_PERM 2 2 1 4
AMPICOL_COLOR_PROBE_FLOW_SUM 3.4E-1
"""
    )

    assert result.value == pytest.approx(0.34)
    assert result.value_decimal == Decimal("0.34")
    assert result.components == pytest.approx((0.4, 0.35, 0.34))
    assert result.component_decimals == (
        Decimal("0.4"),
        Decimal("0.35"),
        Decimal("0.34"),
    )
    assert result.currents == 7
    assert result.vertices == 4
    assert result.amplitudes == 1
    assert result.color_orders == 1
    assert result.lc_row_partitions == (
        module.LcRowPartition(1, 0.2, (2, 4, 1)),
        module.LcRowPartition(2, 0.14, (2, 1, 4)),
    )
    assert result.lc_partition_sum == pytest.approx(0.34)
    assert result.lc_partition_sum_decimal == Decimal("0.34")


def test_probe_output_parser_rejects_duplicate_singleton_records() -> None:
    module = _module()
    output = """AMPICOL_COLOR_PROBE_CURRENTS 1
AMPICOL_COLOR_PROBE_VERTICES 1
AMPICOL_COLOR_PROBE_AMPLITUDES 1
AMPICOL_COLOR_PROBE_COLOR_ORDERS 1
AMPICOL_COLOR_PROBE_COMPONENTS 1.0 1.0 1.0
AMPICOL_COLOR_PROBE_VALUE lc 1 1 1.0
AMPICOL_COLOR_PROBE_VALUE lc 1 1 2.0
"""

    with pytest.raises(module.LegacyOracleError, match="exactly one"):
        module._parse_probe_output(output)


def test_probe_output_parser_normalizes_malformed_numbers() -> None:
    module = _module()
    output = """AMPICOL_COLOR_PROBE_CURRENTS 1
AMPICOL_COLOR_PROBE_VERTICES 1
AMPICOL_COLOR_PROBE_AMPLITUDES 1
AMPICOL_COLOR_PROBE_COLOR_ORDERS 1
AMPICOL_COLOR_PROBE_COMPONENTS 1.0 1.0 1.0
AMPICOL_COLOR_PROBE_VALUE lc 1 1 .
"""

    with pytest.raises(module.LegacyOracleError, match="malformed"):
        module._parse_probe_output(output)


def test_cli_entrypoint_normalizes_expected_input_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    monkeypatch.setattr(
        module,
        "main",
        lambda argv=None: (_ for _ in ()).throw(ValueError("bad fixture")),
    )

    assert module._entrypoint([]) == 2
    assert capsys.readouterr().err == "error: bad fixture\n"


def test_lc_row_partition_resolves_only_a_single_physical_flow() -> None:
    module = _module()
    probe = module.ProbeResult(
        value=0.34,
        components=(0.34, 0.34, 0.34),
        currents=7,
        vertices=4,
        amplitudes=1,
        color_orders=1,
        lc_row_partitions=(module.LcRowPartition(1, 0.34, (2, 3, 1)),),
        lc_partition_sum=0.34,
    )
    color = {"id": "flow:2,3,1", "word": [2, 3, 1]}

    assert module._resolved_single_lc_row(
        probe,
        colors=(color,),
        source_to_row_permutation=(0, 1, 2),
        context="single flow",
    ) == ["0.34"]
    with pytest.raises(module.LegacyOracleError, match="cannot resolve multiple"):
        module._resolved_single_lc_row(
            probe,
            colors=(color, {"id": "flow:3,2,1", "word": [3, 2, 1]}),
            source_to_row_permutation=(0, 1, 2),
            context="multi flow",
        )


def _capture_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    structural_zero: float = 0.0,
    module: Any | None = None,
) -> tuple[object, dict[str, Any], list[tuple[str, tuple[int, ...] | None]], int]:
    module = _module() if module is None else module
    tmp_path.mkdir(parents=True, exist_ok=True)
    repository = tmp_path / "legacy"
    repository.mkdir()
    (repository / "amplicol_color_probe").write_text("probe", encoding="utf-8")
    fixture_path = tmp_path / "physics-v2.json"
    fixture = _v2_fixture(module.expected_revision())
    reference = module._reference_fixture_v2_module()
    for case in fixture["cases"]:
        case["reduction"]["plan_sha256"] = reference.reduction_plan_sha256(
            case["reduction"]
        )
        case["physics_case_sha256"] = reference.physics_case_sha256(
            fixture,
            case["id"],
        )
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    process_generations = 0

    def fake_run(command, *, cwd, capture=True):
        nonlocal process_generations
        process_generations += 1
        return module.subprocess.CompletedProcess(command, 0, "", "")

    entry = module.ProcessEntry(
        group=1,
        integral=1,
        process_pdgs=(1, -1, 21, 21, 23),
        color_order=(2, 3, 4, 1, 5),
    )
    probe_calls: list[tuple[str, tuple[int, ...] | None]] = []

    def fake_probe(
        repository,
        *,
        process_file,
        entry,
        source_pdgs,
        momenta,
        color_accuracy,
        helicities=None,
    ):
        del repository, process_file, entry, source_pdgs, momenta
        selected = None if helicities is None else tuple(helicities)
        probe_calls.append((color_accuracy, selected))
        totals = {"lc": 2.0, "nlc": 1.5, "full": 1.4}
        value = totals[color_accuracy]
        if selected is not None and selected != _ACTIVE_HELICITY:
            value = structural_zero
        if color_accuracy == "lc" and selected is not None:
            first = structural_zero if selected != _ACTIVE_HELICITY else 1.25
            second = 0.0 if selected != _ACTIVE_HELICITY else 0.75
            # These deliberately do not map to the physical fixture flow words.
            # Multi-flow evidence must use only their aggregate probe value.
            lc_row_partitions = (
                module.LcRowPartition(1, first, (1, 2, 3, 4)),
                module.LcRowPartition(2, second, (4, 3, 2, 1)),
            )
            lc_partition_sum = first + second
        else:
            lc_row_partitions = ()
            lc_partition_sum = None
        return module.ProbeResult(
            value=value,
            components=(value, value, value),
            currents=17,
            vertices=31,
            amplitudes=4,
            color_orders=2,
            lc_row_partitions=lc_row_partitions,
            lc_partition_sum=lc_partition_sum,
        )

    monkeypatch.setattr(module, "validate_checkout", lambda repository: None)
    monkeypatch.setattr(
        module,
        "_compiler_provenance",
        lambda repository: module.CompilerProvenance(
            identity="gfortran",
            version="GNU Fortran (GCC) 14.2.0",
            flags=("-ffast-math", "-O3"),
            target="aarch64-apple-darwin24",
            executable_sha256=_sha("probe"),
        ),
    )
    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "parse_process_file", lambda path: (entry,))
    monkeypatch.setattr(module, "run_color_probe", fake_probe)
    report = module.verify_fixture(
        repository,
        fixture_path,
        jobs=1,
        build=False,
    )
    return module, report, probe_calls, process_generations


def test_v2_evidence_is_independent_of_temporary_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, first, _, _ = _capture_v2(tmp_path / "first", monkeypatch)
    _, second, _, _ = _capture_v2(tmp_path / "second", monkeypatch)

    assert first == second


def test_v2_oracle_rejects_changed_process_row_multiplicity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "expected_process_match_count", lambda process: 2)

    with pytest.raises(module.LegacyOracleError, match="row multiplicity"):
        _capture_v2(tmp_path, monkeypatch, module=module)


def test_v2_oracle_rejects_changed_color_order_multiplicity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "expected_color_order_count", lambda process: 1)

    with pytest.raises(module.LegacyOracleError, match="color-order count"):
        _capture_v2(tmp_path, monkeypatch, module=module)


def test_v2_oracle_uses_helicity_aggregates_for_multi_flow_lc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, report, probe_calls, process_generations = _capture_v2(
        tmp_path,
        monkeypatch,
    )

    assert report["evidence_schema_version"] == 2
    assert report["captured_at"] == "2026-07-16T12:00:00Z"
    assert report["oracle"]["revision"] == module.expected_revision()
    assert report["oracle"]["validation_profile"] == "binary64"
    assert report["oracle"]["tolerance_ceiling"] == {
        "relative": "0.0000000001",
        "absolute": "0.000000000001",
    }
    assert report["dependency_ids"] == ["dependency:legacy-amplicol"]
    assert process_generations == 1
    assert len(probe_calls) == 396
    assert len(report["records"]) == 12
    assert {record["case_id"] for record in report["records"]} == {
        "case:dd-zgg-lc",
        "case:dd-zgg-nlc",
        "case:dd-zgg-full",
    }
    for record in report["records"]:
        expected_total = {"lc": "2", "nlc": "1.5", "full": "1.4"}[
            record["case_id"].rsplit("-", 1)[1]
        ]
        assert record["id"] == (
            f"evidence:fortran:{record['case_id']}:{record['point_id']}"
        )
        assert record["arithmetic_precision_bits"] == 53
        assert record["round_trip_decimal_digits"] == 17
        assert record["certified_decimal_digits"] == 10
        assert record["arithmetic"] == "binary64"
        assert len(record["helicity_ids"]) == 32
        active_index = record["helicity_ids"].index("h:-1,+1,-1,-1,+1")
        if record["case_id"].endswith("-lc"):
            assert record["coverage"] == "helicity-aggregate"
            assert record["color_ids"] == ["flow:2,4,5,1", "flow:2,5,4,1"]
            assert record["observed_values"] is None
            assert record["observed_helicity_totals"][active_index] == "2"
            assert all(
                value == "0"
                for index, value in enumerate(record["observed_helicity_totals"])
                if index != active_index
            )
        else:
            assert record["coverage"] == "resolved"
            assert record["color_ids"] == ["color:contracted"]
            assert record["observed_values"][active_index] == [expected_total]
            assert all(
                all(value == "0" for value in row)
                for index, row in enumerate(record["observed_values"])
                if index != active_index
            )
            assert record["observed_helicity_totals"] is None
        assert record["observed_total"] == expected_total
        assert record["process_identity"] == {
            "expression": "d d~ > z g g",
            "ordered_external_pdgs": [1, -1, 21, 21, 23],
            "ordered_external_leg_ids": [
                "leg:incoming-d",
                "leg:incoming-dbar",
                "leg:outgoing-g-1",
                "leg:outgoing-g-2",
                "leg:outgoing-z",
            ],
            "source_to_row_permutation": [0, 1, 3, 4, 2],
            "row_id": "group:1:integral:1",
            "color_order_count": 2,
            "ordered_color_legs": [
                "leg:incoming-dbar",
                "leg:outgoing-g-1",
                "leg:outgoing-g-2",
                "leg:incoming-d",
                "leg:outgoing-z",
            ],
        }
        assert record["command"][0] == "amplicol_color_probe"
        assert all("/tmp/" not in argument for argument in record["command"])
        command = dict(
            argument.split("=", 1) for argument in record["command"] if "=" in argument
        )
        assert command["build_target"] == "amplicol_color_probe"
        assert command["compiler_identity"] == "gfortran"
        assert command["compiler_version"] == "GNU Fortran (GCC) 14.2.0"
        assert json.loads(command["compiler_flags"]) == ["-ffast-math", "-O3"]
        assert command["compiler_target"] == "aarch64-apple-darwin24"
        assert command["oracle_executable_sha256"] == _sha("probe")
        assert len(command["binary64_input_sha256"]) == 64
        assert command["binary64_input_sha256"] != record["input_sha256"]
        assert command["pdg_match_count"] == "1"
        assert len(record["oracle_output_sha256"]) == 64
        assert record["tolerances"] == {
            "relative": "0.0000000001",
            "absolute": "0.000000000001",
        }


def test_v2_multiflow_aggregate_report_validates_with_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, report, _, _ = _capture_v2(tmp_path, monkeypatch)
    fixture = _v2_fixture(module.expected_revision())
    reference = module._reference_fixture_v2_module()
    for case in fixture["cases"]:
        case["reduction"]["plan_sha256"] = reference.reduction_plan_sha256(
            case["reduction"]
        )
        case["physics_case_sha256"] = reference.physics_case_sha256(
            fixture,
            case["id"],
        )

    parsed = reference.parse_reference_fixture(fixture, [report])

    assert len(parsed.cases) == 3
    assert {
        record.coverage
        for record in parsed.evidence_sets[0].records
        if record.case_id.endswith("-lc")
    } == {"helicity-aggregate"}


def test_v2_oracle_canonicalizes_bounded_structural_zero_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, report, _, _ = _capture_v2(
        tmp_path,
        monkeypatch,
        structural_zero=1.0e-16,
    )

    fixture = _v2_fixture(module.expected_revision())
    cases = {case["id"]: case for case in fixture["cases"]}
    for record in report["records"]:
        case = cases[record["case_id"]]
        for index, helicity in enumerate(case["axes"]["helicities"]):
            if not helicity["structural_zero"]:
                continue
            if record["coverage"] == "helicity-aggregate":
                assert record["observed_helicity_totals"][index] == "0"
            else:
                assert all(value == "0" for value in record["observed_values"][index])


def test_v2_oracle_rejects_structural_zero_residue_above_tolerance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    with pytest.raises(
        module.LegacyOracleError,
        match="did not preserve structural zero",
    ):
        _capture_v2(
            tmp_path,
            monkeypatch,
            structural_zero=1.0e-9,
            module=module,
        )


def test_v2_report_replay_is_numeric_not_compiler_byte_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, tracked, _, _ = _capture_v2(tmp_path, monkeypatch)
    fresh = copy.deepcopy(tracked)
    record = fresh["records"][0]
    record["observed_total"] = module._canonical_decimal(
        Decimal(record["observed_total"]) * Decimal("1.00000000001")
    )
    record["oracle_output_sha256"] = _sha("fresh-compiler-output")
    replacements = {
        "compiler_identity": "other-gfortran",
        "compiler_version": "GNU Fortran (GCC) 15.1.0",
        "compiler_flags": '["-O2"]',
        "compiler_target": "x86_64-unknown-linux-gnu",
        "oracle_executable_sha256": _sha("fresh-executable"),
    }
    for index, argument in enumerate(record["command"]):
        key = argument.partition("=")[0]
        if key in replacements:
            record["command"][index] = f"{key}={replacements[key]}"
    _refresh_evidence_record_hash(module, record)

    module._assert_v2_reports_semantically_equal(tracked, fresh)

    record["observed_total"] = module._canonical_decimal(
        Decimal(tracked["records"][0]["observed_total"]) * Decimal("1.001")
    )
    _refresh_evidence_record_hash(module, record)
    with pytest.raises(module.LegacyOracleError, match="outside the tracked"):
        module._assert_v2_reports_semantically_equal(tracked, fresh)


def test_v2_report_replay_requires_exact_identity_and_structural_zeros(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, tracked, _, _ = _capture_v2(tmp_path, monkeypatch)
    fresh = copy.deepcopy(tracked)
    fresh["records"][0]["process_identity"]["row_id"] = "group:99:integral:1"
    _refresh_evidence_record_hash(module, fresh["records"][0])
    with pytest.raises(module.LegacyOracleError, match="identity differs"):
        module._assert_v2_reports_semantically_equal(tracked, fresh)

    fresh = copy.deepcopy(tracked)
    command = fresh["records"][0]["command"]
    binary64_index = next(
        index
        for index, argument in enumerate(command)
        if argument.startswith("binary64_input_sha256=")
    )
    command[binary64_index] = f"binary64_input_sha256={_sha('different-input')}"
    _refresh_evidence_record_hash(module, fresh["records"][0])
    with pytest.raises(module.LegacyOracleError, match="identity differs"):
        module._assert_v2_reports_semantically_equal(tracked, fresh)

    fresh = copy.deepcopy(tracked)
    zero_row = next(
        row
        for row in fresh["records"][0]["observed_values"]
        if all(value == "0" for value in row)
    )
    zero_row[0] = "0.0000000000000001"
    _refresh_evidence_record_hash(module, fresh["records"][0])
    with pytest.raises(module.LegacyOracleError, match="exact zero"):
        module._assert_v2_reports_semantically_equal(tracked, fresh)


def test_fixture_version_one_dispatch_preserves_legacy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    repository = tmp_path / "legacy"
    repository.mkdir()
    (repository / "amplicol_color_probe").write_text("probe", encoding="utf-8")
    fixture_path = tmp_path / "physics-v1.json"
    fixture_path.write_text('{"fixture_schema_version": 1}', encoding="utf-8")
    sentinel = {"schema_version": 1, "unchanged": True}
    monkeypatch.setattr(module, "validate_checkout", lambda repository: None)
    monkeypatch.setattr(
        module,
        "_verify_fixture_v1",
        lambda repository, fixture_path, fixture: sentinel,
    )

    assert (
        module.verify_fixture(
            repository,
            fixture_path,
            jobs=1,
            build=False,
        )
        is sentinel
    )


def test_tracked_fortran_report_is_pinned_and_matches_physics_fixture() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    physics = json.loads(PHYSICS.read_text(encoding="utf-8"))
    with (ROOT / "dependencies" / "contributor-lock.toml").open("rb") as stream:
        lock = tomllib.load(stream)

    assert report["schema_version"] == 1
    assert report["oracle"] == "legacy-fortran-amplicol-color-probe"
    assert report["revision"] == lock["legacy_amplicol"]["reference_fixture_revision"]
    assert report["fixture_sha256"] == hashlib.sha256(PHYSICS.read_bytes()).hexdigest()
    assert report["patches"] == []
    assert report["tolerances"] == {"relative": 1.0e-8, "absolute": 1.0e-15}
    assert set(report["cases"]) == {
        "builtin_sm_ddbar_z_lc",
        "builtin_sm_ddbar_z_nlc",
        "builtin_sm_ddbar_z_full",
        "builtin_sm_ddbar_zg_lc",
        "builtin_sm_ddbar_zg_nlc",
        "builtin_sm_ddbar_zg_full",
    }
    for name, observation in report["cases"].items():
        assert "source_to_row_permutation" in observation["fortran_process_entry"]
        assert observation["total"] == pytest.approx(
            physics["cases"][name]["total"], rel=1.0e-12, abs=1.0e-15
        )
        assert (
            observation["max_resolved_relative_difference"]
            < report["tolerances"]["relative"]
        )
