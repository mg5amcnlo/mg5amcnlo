# SPDX-License-Identifier: 0BSD
"""Process-group members that were not generated route to their representative."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, ProcessSet, Runtime
from pyamplicol.artifacts.process_groups import (
    initial_process_groups,
    write_process_groups,
)
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)
from pyamplicol.generation.process_grouping import (
    GroupingBasis,
    GroupMember,
)
from pyamplicol.models.builtin.validation import generic_validation_point

_MODES = ("compiled", "eager", "recurrence")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Representative, member, member relation (representative leg -> member leg),
# exact factor, and charge conjugation. Each member is also generated in a
# reference artifact under its own expression.
_GROUPS = (
    ("d d~ > g g", "u u~ > g g", (0, 1, 2, 3), Fraction(1), False),
    ("d d~ > a g", "u~ u > a g", (1, 0, 2, 3), Fraction(4), False),
    ("u g > u g", "u~ g > u~ g", (0, 1, 2, 3), Fraction(1), True),
    ("d d~ > a a", "u u~ > a a", (0, 1, 2, 3), Fraction(16), False),
)
_REFERENCES = ("u u~ > g g", "u u~ > a g", "u~ g > u~ g", "u u~ > a a")


def _require_native_runtime() -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        pytest.skip("Symbolica is unavailable")


def _generation_config(execution_mode: str, *, api_bundle: bool = False) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc"),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=api_bundle,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=execution_mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


def _member_id(expression: str) -> str:
    initial, final = expression.split(">")

    def name(token: str) -> str:
        return token.replace("~", "bar")

    return (
        "_".join(name(token) for token in initial.split())
        + "_to_"
        + "_".join(name(token) for token in final.split())
        + "_member"
    )


def _attach_process_groups(
    artifact: Path,
    members: Mapping[str, Sequence[GroupMember]],
) -> None:
    """Declare non-generated members on the stored representatives."""

    raw = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    write_process_groups(
        artifact,
        initial_process_groups(
            (
                (
                    str(process["id"]),
                    str(process["expression"]),
                    process["external_pdgs"],
                )
                for process in raw["processes"]
            ),
            members,
        ),
    )


def _process_ids(artifact: Path) -> dict[str, str]:
    raw = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    return {
        str(process["expression"]): str(process["id"]) for process in raw["processes"]
    }


def _attach_members(artifact: Path) -> None:
    ids = _process_ids(artifact)
    members: dict[str, list[GroupMember]] = {}
    for representative, member, permutation, factor, conjugated in _GROUPS:
        if representative not in ids:
            continue
        pdgs = tuple(int(particle.pdg) for particle in generic_validation_point(member))
        members.setdefault(ids[representative], []).append(
            GroupMember(
                process_id=_member_id(member),
                expression=member,
                pdgs=pdgs,
                initial_count=len(member.split(">")[0].split()),
                factor=factor,
                permutation=permutation,
                conjugated=conjugated,
                basis=GroupingBasis.FLAVOR_BLIND,
                generated=False,
            )
        )
    _attach_process_groups(artifact, members)


@pytest.fixture(scope="module")
def group_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Path]]:
    _require_native_runtime()
    root = tmp_path_factory.mktemp("process-group-routing")
    representatives = tuple(group[0] for group in _GROUPS)
    artifacts = {mode: root / mode for mode in _MODES}
    artifacts["reference"] = root / "reference"
    artifacts["single"] = root / "single"
    try:
        for mode in _MODES:
            Generator(_generation_config(mode)).generate(
                ProcessSet.from_expressions(representatives), artifacts[mode]
            )
            _attach_members(artifacts[mode])
        Generator(_generation_config("recurrence")).generate(
            ProcessSet.from_expressions(_REFERENCES), artifacts["reference"]
        )
        Generator(_generation_config("recurrence")).generate(
            "u g > u g", artifacts["single"]
        )
        _attach_members(artifacts["single"])
        yield artifacts
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _point(expression: str) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        tuple(float(component) for component in particle.momentum)  # type: ignore[misc]
        for particle in generic_validation_point(expression)
    )


def _reordered(
    point: Sequence[Sequence[float]], permutation: Sequence[int]
) -> tuple[tuple[float, ...], ...]:
    """Momenta of ``point`` at ``target[i] = point[permutation[i]]``."""

    return tuple(tuple(point[index]) for index in permutation)


def _flatten(values: object) -> tuple[complex, ...]:
    flattened: list[complex] = []

    def visit(value: object) -> None:
        if isinstance(value, tuple | list):
            for item in value:
                visit(item)
        else:
            flattened.append(complex(value))  # type: ignore[arg-type]

    visit(values)
    return tuple(flattened)


def _assert_close(actual: object, expected: object) -> None:
    assert _flatten(actual) == pytest.approx(
        _flatten(expected), rel=1.0e-12, abs=1.0e-15
    )


def _flow_values(runtime: Runtime, point: object, **selectors: object) -> dict:
    resolved = runtime.evaluate_resolved((point,), **selectors)  # type: ignore[arg-type]
    return {
        (helicity, color): resolved.values[0][h_index][c_index]
        for h_index, helicity in enumerate(resolved.helicity_ids)
        for c_index, color in enumerate(resolved.color_ids)
    }


@pytest.mark.parametrize("execution_mode", _MODES)
def test_flavour_relabelled_member_matches_its_generated_process(
    group_artifacts: dict[str, Path], execution_mode: str
) -> None:
    artifact = group_artifacts[execution_mode]
    representative_id = _process_ids(artifact)["d d~ > g g"]
    reference = Runtime.load(group_artifacts["reference"], process="u u~ > g g")
    for selector in (_member_id("u u~ > g g"), "u   U~ > g g"):
        member = Runtime.load(artifact, process=selector)
        assert member.physics.process_id == _member_id("u u~ > g g")
        assert member.physics.process == "u u~ > g g"
        assert [p.pdg_id for p in member.physics.external_particles] == [2, -2, 21, 21]
        assert [p.name for p in member.physics.external_particles] == [
            "u",
            "u~",
            "g",
            "g",
        ]
        assert member.representative_process_key == representative_id
        assert member.external_permutation == (0, 1, 2, 3)
        assert member.process_group_factor == 1
        assert member.charge_conjugated is False
        assert member.physics.color_ids == reference.physics.color_ids
        point = _point("u u~ > g g")
        _assert_close(member.evaluate((point,) * 3), reference.evaluate((point,) * 3))
        assert (
            _flow_values(member, point).keys() == _flow_values(reference, point).keys()
        )
        member_flows = _flow_values(member, point)
        reference_flows = _flow_values(reference, point)
        _assert_close(
            [member_flows[key] for key in sorted(member_flows)],
            [reference_flows[key] for key in sorted(member_flows)],
        )

    representative = Runtime.load(artifact, process=representative_id)
    assert representative.process_group_factor == 1
    assert representative.physics.process_id == representative_id
    assert (
        representative.inspect()["runtime_metadata"]["process_group_member_of"] is None
    )


@pytest.mark.parametrize("execution_mode", _MODES)
def test_swapped_member_in_another_order_applies_permutation_and_factor(
    group_artifacts: dict[str, Path], execution_mode: str
) -> None:
    artifact = group_artifacts[execution_mode]
    requested = "u~ u > g a"
    member = Runtime.load(artifact, process=requested)
    reference = Runtime.load(group_artifacts["reference"], process="u u~ > a g")
    # Representative d d~ a g -> member u~ u a g -> requested u~ u g a.
    assert member.external_permutation == (1, 0, 3, 2)
    assert member.process_group_factor == 4
    assert member.physics.process_id == _member_id("u~ u > a g")
    assert member.physics.process == requested
    assert [p.pdg_id for p in member.physics.external_particles] == [-2, 2, 21, 22]
    assert (
        member.inspect()["runtime_metadata"]["process_group_member_of"]
        == (_process_ids(artifact)["d d~ > a g"])
    )

    # Reference legs u u~ a g are requested legs (1, 0, 3, 2).
    reference_point = _point("u u~ > a g")
    requested_point = _reordered(reference_point, (1, 0, 3, 2))
    _assert_close(
        member.evaluate((requested_point,) * 2),
        reference.evaluate((reference_point,) * 2),
    )
    member_by_helicity: dict[tuple[int, ...], complex] = {}
    resolved = member.evaluate_resolved((requested_point,))
    for helicity, values in zip(
        member.physics.helicities, resolved.values[0], strict=True
    ):
        vector = helicity.values
        member_by_helicity[(vector[1], vector[0], vector[3], vector[2])] = sum(
            complex(value) for value in values
        )
    reference_resolved = reference.evaluate_resolved((reference_point,))
    reference_by_helicity = {
        helicity.values: sum(complex(value) for value in values)
        for helicity, values in zip(
            reference.physics.helicities, reference_resolved.values[0], strict=True
        )
    }
    assert member_by_helicity.keys() == reference_by_helicity.keys()
    _assert_close(
        [member_by_helicity[key] for key in sorted(member_by_helicity)],
        [reference_by_helicity[key] for key in sorted(member_by_helicity)],
    )
    selected = next(
        helicity for helicity in member.physics.helicities if helicity.computed
    )
    vector = selected.values
    reference_helicity = next(
        helicity.id
        for helicity in reference.physics.helicities
        if helicity.values == (vector[1], vector[0], vector[3], vector[2])
    )
    _assert_close(
        member.evaluate((requested_point,), helicities=(selected.id,)),
        reference.evaluate((reference_point,), helicities=(reference_helicity,)),
    )
    _assert_close(
        member.evaluate((requested_point,), precision=32),
        reference.evaluate((reference_point,)),
    )


@pytest.mark.parametrize("execution_mode", _MODES)
def test_charge_conjugated_member_reverses_lc_flows(
    group_artifacts: dict[str, Path], execution_mode: str
) -> None:
    artifact = group_artifacts[execution_mode]
    member = Runtime.load(artifact, process=_member_id("u~ g > u~ g"))
    reference = Runtime.load(group_artifacts["reference"], process="u~ g > u~ g")
    representative = Runtime.load(artifact, process="u g > u g")
    assert member.charge_conjugated is True
    assert member.process_group_factor == 1
    assert member.physics.process == "u~ g > u~ g"
    assert [p.pdg_id for p in member.physics.external_particles] == [-2, 21, -2, 21]
    assert sorted(member.physics.color_ids) == sorted(reference.physics.color_ids)
    for representative_flow, member_flow in zip(
        representative.physics.color_flows, member.physics.color_flows, strict=True
    ):
        assert member_flow.word == tuple(reversed(representative_flow.word))

    point = _point("u~ g > u~ g")
    member_flows = _flow_values(member, point)
    reference_flows = _flow_values(reference, point)
    assert member_flows.keys() == reference_flows.keys()
    _assert_close(
        [member_flows[key] for key in sorted(member_flows)],
        [reference_flows[key] for key in sorted(member_flows)],
    )
    for flow in member.physics.color_ids:
        selected = member.evaluate_resolved((point,), color_flows=(flow,))
        assert selected.color_ids == (flow,)
        _assert_close(
            member.evaluate((point,), color_flows=(flow,)),
            reference.evaluate((point,), color_flows=(flow,)),
        )
    exact = member.evaluate_resolved((point,), precision=32)
    assert sorted(exact.color_ids) == sorted(reference.physics.color_ids)
    exact_flows = {
        (helicity, color): exact.values[0][h_index][c_index]
        for h_index, helicity in enumerate(exact.helicity_ids)
        for c_index, color in enumerate(exact.color_ids)
    }
    _assert_close(
        [exact_flows[key] for key in sorted(exact_flows)],
        [reference_flows[key] for key in sorted(exact_flows)],
    )


@pytest.mark.parametrize("execution_mode", _MODES)
def test_member_with_non_unit_factor_matches_its_generated_process(
    group_artifacts: dict[str, Path], execution_mode: str
) -> None:
    member = Runtime.load(group_artifacts[execution_mode], process="u u~ > a a")
    reference = Runtime.load(group_artifacts["reference"], process="u u~ > a a")
    point = _point("u u~ > a a")
    assert member.process_group_factor == 16
    _assert_close(member.evaluate((point,) * 2), reference.evaluate((point,) * 2))
    _assert_close(
        member.evaluate_resolved((point,)).values,
        reference.evaluate_resolved((point,)).values,
    )


def test_single_process_recurrence_artifact_routes_members(
    group_artifacts: dict[str, Path],
) -> None:
    artifact = group_artifacts["single"]
    default = Runtime.load(artifact)
    assert default.physics.process == "u g > u g"
    assert default.charge_conjugated is False
    member = Runtime.load(artifact, process="u~ g > u~ g")
    reference = Runtime.load(group_artifacts["reference"], process="u~ g > u~ g")
    point = _point("u~ g > u~ g")
    assert member.charge_conjugated is True
    _assert_close(member.evaluate((point,)), reference.evaluate((point,)))
    with pytest.raises(Exception, match="unknown process"):
        Runtime.load(artifact, process="c~ g > c~ g")


def test_on_the_fly_conjugated_member_translates_compact_selectors(
    group_artifacts: dict[str, Path], tmp_path: Path
) -> None:
    rusticol = importlib.import_module("pyamplicol._rusticol")
    if not hasattr(rusticol, "_build_on_the_fly_process_seeds_v1"):
        pytest.skip("the installed Rusticol extension lacks on-the-fly generation")
    artifact = tmp_path / "otf"
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared:
        Generator(_generation_config("on-the-fly")).generate(
            "u g > u g", artifact, model=ModelSource.from_path(prepared)
        )
    _attach_members(artifact)
    member = Runtime.load(artifact, process="u~ g > u~ g")
    reference = Runtime.load(group_artifacts["reference"], process="u~ g > u~ g")
    assert member.execution_mode == "on-the-fly"
    assert member.charge_conjugated is True
    assert sorted(member.physics.color_ids) == sorted(reference.physics.color_ids)
    point = _point("u~ g > u~ g")
    member_flows = _flow_values(member, point)
    reference_flows = _flow_values(reference, point)
    assert member_flows.keys() == reference_flows.keys()
    _assert_close(
        [member_flows[key] for key in sorted(member_flows)],
        [reference_flows[key] for key in sorted(member_flows)],
    )
    flows = reference.physics.color_ids
    _assert_close(
        member.evaluate((point,) * len(flows), color_flow_by_point=flows),
        reference.evaluate((point,) * len(flows), color_flow_by_point=flows),
    )
    for flow in flows:
        selected = member.evaluate_resolved((point,), color_flows=(flow,))
        assert selected.color_ids == (flow,)
        _assert_close(
            member.evaluate((point,), color_flows=(flow,)),
            reference.evaluate((point,), color_flows=(flow,)),
        )


def test_c_api_selects_a_process_group_member(tmp_path: Path) -> None:
    _require_native_runtime()
    tools = {
        "make": shutil.which("make"),
        "cc": shutil.which(os.environ.get("CC", "cc")),
        "rusticol-config": shutil.which("rusticol-config")
        or (
            str(Path(sys.executable).parent / "rusticol-config")
            if (Path(sys.executable).parent / "rusticol-config").is_file()
            else None
        ),
    }
    missing = [name for name, value in tools.items() if value is None]
    if missing:
        pytest.skip("C API check requires " + ", ".join(missing))
    artifact = tmp_path / "artifact"
    Generator(_generation_config("compiled", api_bundle=True)).generate(
        "d d~ > a g", artifact
    )
    _attach_members(artifact)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(_PROJECT_ROOT / "src"), environment.get("PYTHONPATH", "")]
    )
    environment["CC"] = str(tools["cc"])
    environment["RUSTICOL_CONFIG"] = shlex.join([str(tools["rusticol-config"])])
    built = subprocess.run(
        [str(tools["make"]), "-C", str(artifact / "API/c"), "all"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    driver = tmp_path / ".pyamplicol-api-build/artifact/c/check_standalone"

    def run(process: str) -> dict:
        completed = subprocess.run(
            [str(driver), "--json", "--process", process],
            cwd=artifact,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return json.loads(completed.stdout)

    representative = run(_process_ids(artifact)["d d~ > a g"])
    member = run("u~ u > g a")
    assert member["process_key"] == _member_id("u~ u > a g")
    assert [particle["pdg"] for particle in member["external_particles"]] == [
        -2,
        2,
        21,
        22,
    ]
    # Both drivers evaluate the bundled representative point in their order.
    assert member["compatibility_total"] == pytest.approx(
        [4 * value for value in representative["compatibility_total"]], rel=1.0e-12
    )
    python = Runtime.load(artifact, process="u~ u > g a")
    assert python.process_group_factor == Fraction(4)

    source = tmp_path / "relation.c"
    source.write_text(
        """#include <inttypes.h>
#include <stdio.h>
#include <rusticol.h>
int main(int argc, char **argv) {
    RusticolRuntimeHandle *runtime = NULL;
    uint64_t numerator = 0, denominator = 0;
    int32_t conjugated = -1;
    if (argc != 3
        || rusticol_runtime_load(argv[1], argv[2], NULL, &runtime) != 0
        || rusticol_runtime_process_group_relation(
               runtime, &numerator, &denominator, &conjugated) != 0) {
        return 1;
    }
    printf("%" PRIu64 "/%" PRIu64 " %d\\n", numerator, denominator, conjugated);
    return rusticol_runtime_free(runtime);
}
""",
        encoding="utf-8",
    )
    flags = [
        argument
        for option in ("--cflags", "--libs")
        for argument in shlex.split(
            subprocess.run(
                [str(tools["rusticol-config"]), option],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    ]
    program = tmp_path / "relation"
    subprocess.run(
        [str(tools["cc"]), "-std=c11", str(source), *flags, "-o", str(program)],
        env=environment,
        check=True,
        timeout=120,
    )
    for process, expected in (
        ("u~ u > g a", "4/1 0"),
        (_process_ids(artifact)["d d~ > a g"], "1/1 0"),
    ):
        completed = subprocess.run(
            [str(program), str(artifact), process],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.stdout.strip() == expected
