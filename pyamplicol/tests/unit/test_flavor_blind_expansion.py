# SPDX-License-Identifier: 0BSD
"""Charge-filtered multiset expansion and flavour-blind process collapsing."""

from __future__ import annotations

import itertools
import logging
from fractions import Fraction
from pathlib import Path
from typing import cast

import pytest

import pyamplicol.generation.service as generation_service
import pyamplicol.licensing as licensing
from pyamplicol import Generator, ModelSource, ProcessSet
from pyamplicol.api.errors import GenerationError
from pyamplicol.api.models import CompiledModel as CompiledModelHandle
from pyamplicol.api.models import _compiled_model_payload
from pyamplicol.api.requests import ProcessAlias
from pyamplicol.color.plan import build_color_plan
from pyamplicol.config import (
    Action,
    EvaluatorConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation.dag_algorithms import has_model_supported_amplitudes
from pyamplicol.generation.process_grouping import (
    GroupingBasis,
    GroupingProcess,
    GroupMember,
    flavor_blind_mapping,
    particle_table,
)
from pyamplicol.generation.process_grouping_symmetry import VertexSymmetryTable
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.licensing import SymbolicaLicenseState
from pyamplicol.models.external import CompiledUFOModel
from pyamplicol.models.loading import CompiledModel
from pyamplicol.processes.conservation import conserved_charges
from pyamplicol.processes.model import (
    ModelParticleCatalog,
    build_model_process_ir,
    expand_model_processes,
)

MODEL_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)


@pytest.fixture(scope="module")
def ufo_sm_handle() -> CompiledModelHandle:
    return ModelSource(
        kind="json",
        path=MODEL_ROOT / "sm.json",
        restriction=MODEL_ROOT / "restrict_default.json",
    ).compile(use_cache=False)


@pytest.fixture(scope="module")
def ufo_sm(ufo_sm_handle: CompiledModelHandle) -> CompiledModel:
    return cast(CompiledModel, _compiled_model_payload(ufo_sm_handle))


def _restricted_license(**_kwargs: object) -> SymbolicaLicenseState:
    return SymbolicaLicenseState(licensed=False, restricted=True)


def _side_class(process: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    initial, final = process.split(">")
    return tuple(sorted(initial.split())), tuple(sorted(final.split()))


def test_conserved_charges_follow_the_model_vertices(ufo_sm: CompiledModel) -> None:
    charges = conserved_charges(ufo_sm.ir)

    # Electric charge, baryon and lepton numbers per generation (diagonal CKM).
    assert charges.balanced(("u", "d~"), ("W+",))
    assert charges.balanced(("u", "d~"), ("c", "s~"))
    assert charges.balanced(("e-", "mu+"), ("ve", "vm~"))
    assert not charges.balanced(("u", "u"), ("d", "d"))
    assert not charges.balanced(("u", "u~"), ("c", "u~"))
    assert not charges.balanced(("e-", "e+"), ("mu-", "e+"))
    assert charges.charges["u~"] == tuple(-value for value in charges.charges["u"])
    assert "g" not in charges.charges and "Z" not in charges.charges


def test_multiset_expansion_keeps_exactly_the_processes_with_amplitudes(
    ufo_sm: CompiledModel,
) -> None:
    catalog = ModelParticleCatalog(
        ufo_sm.ir.name, ufo_sm.ir.particles, ufo_sm.ir.parameters
    )
    model = CompiledUFOModel(ufo_sm)
    jets = catalog.default_multiparticles()["j"]

    def supported(process: str) -> bool:
        process_ir = build_model_process_ir(process, ufo_sm.ir)
        selected, _rejected = generation_service._select_color_ready_processes(
            (process_ir,), color_accuracy="lc", basis="trace"
        )
        return bool(selected) and has_model_supported_amplitudes(
            process_ir,
            model=model,
            color_plan=build_color_plan(
                process_ir,
                color_accuracy="lc",
                fold_trace_reflections=(
                    model.lc_trace_reflection_equivalence_is_proven(process_ir)
                ),
            ),
        )

    # The former expansion: ordered tuples, one process per side-permutation
    # class, filtered by amplitudes.
    first_of_class: dict[tuple[tuple[str, ...], tuple[str, ...]], str] = {}
    for initial in itertools.product(jets, repeat=2):
        for final in itertools.product(jets, repeat=2):
            process = f"{' '.join(initial)} > {' '.join(final)}"
            first_of_class.setdefault(_side_class(process), process)
    expected = [process for process in first_of_class.values() if supported(process)]

    expanded = expand_model_processes(
        "p p > j j", catalog, charges=conserved_charges(ufo_sm.ir)
    )

    assert len({_side_class(process) for process in expanded}) == len(expanded)
    assert [process for process in expanded if supported(process)] == expected
    # For dijets the charge prefilter alone already removes every process
    # without amplitudes.
    assert list(expanded) == expected


def _plan_names(**process: object) -> dict[str, str]:
    plan = Generator(
        RunConfig(
            action=Action.GENERATE,
            process=ProcessConfig(flavor_scheme=5, **process),  # type: ignore[arg-type]
            evaluator=EvaluatorConfig(execution_mode="compiled"),
        )
    ).plan("p p > j j j j")
    return {request.name: request.expression for request in plan.concrete_processes}


def test_flavor_blind_plan_keeps_one_process_per_flavour_class(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)

    every = _plan_names()
    without_conjugation = _plan_names(
        flavor_blind=True, consider_charge_conjugation=False
    )
    with caplog.at_level(logging.INFO, logger="pyamplicol.generation"):
        with_conjugation = _plan_names(flavor_blind=True)
    assert (
        "Flavor-blind grouping: 486 subprocesses -> 23 generated representatives; "
        "463 routed members (140 by charge conjugation); 0 relabelling warnings"
    ) in caplog.messages

    assert len(every) == 486
    assert len(without_conjugation) == 33
    assert len(with_conjugation) == 23
    # Representatives keep the IDs and expressions they have without the flag.
    for representatives in (without_conjugation, with_conjugation):
        assert all(
            every[name] == expression for name, expression in representatives.items()
        )
    assert set(with_conjugation.values()) <= set(without_conjugation.values())


def _expanded(
    processes: ProcessSet | str,
    *,
    model: CompiledModelHandle | None = None,
    **process: object,
) -> tuple[generation_service._ExpandedProcess, ...]:
    backend = generation_service.GenerationBackend(
        RunConfig(
            action=Action.GENERATE,
            process=ProcessConfig(flavor_blind=True, **process),  # type: ignore[arg-type]
            evaluator=EvaluatorConfig(execution_mode="compiled"),
        ),
        None,
    )
    if isinstance(processes, str):
        processes = ProcessSet.from_expressions((processes,))
    resolved = backend._resolve_model_for_plan(
        model or backend._configured_model_source()
    )
    return backend._expand_process_set(
        processes, resolved, PhaseHandle("test", None, 1)
    )


def _members(
    entries: tuple[generation_service._ExpandedProcess, ...],
) -> dict[str, tuple[str, GroupMember]]:
    return {
        member.expression: (entry.request.expression, member)
        for entry in entries
        for member in entry.group_members
    }


def test_flavor_blind_members_record_the_assumed_relation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)

    members = _members(_expanded("p p > j j", flavor_scheme=4))

    representative, member = members["u u~ > g g"]
    assert representative == "d d~ > g g"
    assert member.basis is GroupingBasis.FLAVOR_BLIND
    assert member.factor == Fraction(1)
    assert not member.generated and not member.conjugated
    assert member.permutation == (0, 1, 2, 3)
    representative, member = members["u~ g > u~ g"]
    assert representative == "d g > d g"
    assert member.conjugated
    # Pure QCD (the minimal coupling orders of jets) relabels and conjugates
    # quarks exactly.
    assert all(member.warning is None for _, member in members.values())


def test_flavor_blind_warns_when_couplings_distinguish_the_flavours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)

    z_jet = _members(_expanded("p p > z j", flavor_scheme=4))
    assert "relabelling" in z_jet["u u~ > g z"][1].warning
    assert "chiral" in z_jet["d~ g > d~ z"][1].warning

    dijet = _members(
        _expanded("p p > j j", flavor_scheme=4, coupling_order_policy="explicit")
    )
    assert "relabelling" in dijet["u u~ > g g"][1].warning
    # Exchanging whole generations is a symmetry of the built-in SM.
    assert dijet["s c > s c"][0] == "d u > d u"
    assert dijet["s c > s c"][1].warning is None


def test_flavor_blind_keeps_processes_with_explicit_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)
    processes = ProcessSet.from_expressions(
        ("d d~ > g g", "u u~ > g g", "s s~ > g g"),
        names=("dd", "uu", "ss"),
        aliases=(ProcessAlias("uu_swapped", "uu", (1, 0, 2, 3)),),
    )

    entries = _expanded(processes)

    assert [entry.request.name for entry in entries] == ["dd", "uu"]
    assert [member.process_id for member in entries[0].group_members] == ["ss"]


def test_ufo_flavor_blind_conjugation_moves_the_charged_leptons(
    monkeypatch: pytest.MonkeyPatch,
    ufo_sm_handle: CompiledModelHandle,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)

    members = _members(_expanded("p p > e+ e- j", model=ufo_sm_handle))

    # C turns the member's e- into the representative's e+:
    # ME(g d~ > e+ e- d~)(p) = ME(g d > e+ e- d)(p0, p1, p3, p2, p4).
    representative, member = members["g d~ > e+ e- d~"]
    assert representative == "g d > e+ e- d"
    assert member.conjugated
    assert member.permutation == (0, 1, 3, 2, 4)
    assert "chiral" in member.warning
    assert "relabelling" in members["g u > e+ e- u"][1].warning


def test_ufo_request_without_charge_conserving_candidates_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    ufo_sm_handle: CompiledModelHandle,
) -> None:
    monkeypatch.setattr(licensing, "detect_symbolica_license", _restricted_license)

    with pytest.raises(GenerationError, match="conserving the additive charges"):
        _expanded("p p > e+ mu-", model=ufo_sm_handle)


def test_vertex_symmetry_of_ufo_relabellings_and_conjugation(
    ufo_sm: CompiledModel,
) -> None:
    table = particle_table(
        ufo_sm.ir.particles,
        {parameter.name: parameter for parameter in ufo_sm.ir.parameters},
    )
    symmetry = VertexSymmetryTable.from_compiled_ir(ufo_sm.ir)
    pdgs = {particle.name: particle.pdg_code for particle in ufo_sm.ir.particles}

    def warning(member: str, representative: str, **limits: int) -> str | None:
        processes = []
        for expression in (member, representative):
            initial, final = expression.split(">")
            processes.append(
                GroupingProcess(
                    process_id=expression,
                    expression=expression,
                    pdgs=tuple(
                        pdgs[name] for name in (*initial.split(), *final.split())
                    ),
                    initial_count=len(initial.split()),
                )
            )
        mapping = flavor_blind_mapping(
            processes[0], processes[1], table, consider_charge_conjugation=True
        )
        return symmetry.relation_warning(
            processes[0], processes[1], mapping, table, limits=limits
        )

    assert warning("u u~ > g g", "d d~ > g g", QCD=2, QED=0) is None
    assert warning("u~ g > u~ g", "u g > u g", QCD=2, QED=0) is None
    assert "relabelling" in warning("u u~ > g g", "d d~ > g g", QCD=2, QED=2)
    assert "[d~ d Z]" in warning("d~ g > d~ Z", "d g > d Z", QCD=1, QED=1)
