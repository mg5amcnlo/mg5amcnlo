# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import pyamplicol.generation.recurrence_schedule_sharing as schedule_sharing
from pyamplicol.generation.artifact_writer import (
    _recurrence_binding_native_schedule_semantic_digest,
)
from pyamplicol.generation.recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceBuilderLogicalInputV1,
    RecurrenceCouplingLimitV1,
    RecurrenceExternalLegV1,
    RecurrenceLCOpenStringV1,
    RecurrenceNormalizationV1,
    RecurrenceParameterProjectionV1,
    RecurrencePhysicalLCSectorV1,
    RecurrencePublicLCFlowV1,
    RecurrenceReplayPartitionV1,
    RecurrenceReplayTargetV1,
    RecurrenceSemanticDigestV1,
    RecurrenceSemanticTemplateReferenceV1,
    RecurrenceSourceStateV1,
)
from pyamplicol.generation.recurrence_fermion_pairing import (
    NO_FERMION_LINE,
    ExternalFermionEndpointRowV1,
    FermionPairingCatalogV1,
    FermionPairingClassRowV1,
    FermionPairingRuleRowV1,
)
from pyamplicol.generation.recurrence_schedule_sharing import (
    RECURRENCE_PROCESS_BINDING_MAGIC,
    RecurrenceProcessExecutorPack,
    RecurrenceProcessRemap,
    RecurrenceScheduleLoweringCache,
    RecurrenceScheduleSharingError,
    encode_recurrence_process_binding,
    exact_recurrence_process_bijection,
    intern_recurrence_schedules,
    recurrence_helicity_selector_schedule_digest,
    recurrence_native_schedule_semantic_digest,
    recurrence_schedule_semantic_digest,
)
from pyamplicol.models.recurrence_template import (
    ExactComplexRationalV1 as ModelExactComplexRationalV1,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_helicity_selector_schedule_identity_binds_dispatch_sidecar() -> None:
    base = _digest("base schedule")
    first_dispatch = _digest("dispatch one")
    second_dispatch = _digest("dispatch two")

    first = recurrence_helicity_selector_schedule_digest(base, first_dispatch)

    assert first == recurrence_helicity_selector_schedule_digest(
        base,
        first_dispatch,
    )
    assert first != recurrence_helicity_selector_schedule_digest(
        base,
        second_dispatch,
    )
    assert first != base


def _identity_remap(
    *,
    source_count: int = 2,
    flow_count: int = 1,
    sector_count: int = 1,
) -> RecurrenceProcessRemap:
    return RecurrenceProcessRemap(
        source_slots=tuple(range(source_count)),
        source_momentum_signs=(1,) * source_count,
        source_helicity_signs=(1,) * source_count,
        source_state_offsets=tuple(range(source_count + 1)),
        source_state_indices=(0,) * source_count,
        public_flow_ids=tuple(range(flow_count)),
        physical_sector_ids=tuple(range(sector_count)),
        state_template_count=3,
        source_template_count=3,
        direct_executor_count=4,
        parameter_slot_count=1,
    ).with_digest(_digest("identity remap"))


def _executor_pack() -> RecurrenceProcessExecutorPack:
    descriptor = struct.pack("<IIBBHI", 16, 0, 0, 0, 0, 1)
    return RecurrenceProcessExecutorPack(
        compiled_model_digest=_digest("compiled model"),
        recurrence_template_catalog_digest=_digest("recurrence templates"),
        prepared_kernel_pack_digest=_digest("prepared pack"),
        direct_template_catalog_digest=_digest("direct templates"),
        runtime_layout_digest=_digest("runtime layout"),
        backend="jit",
        target_triple="symjit-storage-v3-portable",
        portable=True,
        cpu_features=(),
        catalog_executor_count=4,
        executor_ids=(0,),
        descriptor_payloads=(descriptor,),
    )


@dataclass(frozen=True)
class _Leg:
    physical_pdg: int
    support_mask: int


@dataclass(frozen=True)
class _Logical:
    process_id: str
    process_support_mask: int
    external_legs: tuple[_Leg, ...]
    proof_digest: str


def _schedule_key(logical: _Logical) -> str:
    return recurrence_schedule_semantic_digest(
        cast(RecurrenceBuilderLogicalInputV1, logical),
        prepared_kernel_pack_digest=_digest("pack"),
        direct_template_catalog_digest=_digest("templates"),
        point_tile_size=1024,
        workspace_mib=256,
    )


def test_prelower_identity_normalizes_only_process_ownership() -> None:
    first = _Logical("subprocess-a", 1, (_Leg(1, 1), _Leg(-1, 1)), _digest("p"))
    alias = _Logical("subprocess-b", 8, (_Leg(1, 8), _Leg(-1, 8)), _digest("p"))
    different = _Logical("subprocess-c", 2, (_Leg(2, 2), _Leg(-2, 2)), _digest("p"))

    assert _schedule_key(first) == _schedule_key(alias)
    assert _schedule_key(first) != _schedule_key(different)
    assert recurrence_schedule_semantic_digest(
        cast(RecurrenceBuilderLogicalInputV1, first),
        prepared_kernel_pack_digest=_digest("different pack"),
        direct_template_catalog_digest=_digest("templates"),
        point_tile_size=1024,
        workspace_mib=256,
    ) != _schedule_key(first)


def test_relation_discovery_only_changes_request_cache_identity() -> None:
    logical = cast(
        RecurrenceBuilderLogicalInputV1,
        _Logical("subprocess-a", 1, (_Leg(1, 1), _Leg(-1, 1)), _digest("p")),
    )

    def schedule_digest(
        relation_discovery: Mapping[str, object] | None = None,
    ) -> str:
        return recurrence_schedule_semantic_digest(
            logical,
            prepared_kernel_pack_digest=_digest("pack"),
            direct_template_catalog_digest=_digest("templates"),
            point_tile_size=1024,
            workspace_mib=256,
            relation_discovery=relation_discovery,
        )

    baseline = schedule_digest()
    native = recurrence_native_schedule_semantic_digest(
        logical,
        prepared_kernel_pack_digest=_digest("pack"),
        direct_template_catalog_digest=_digest("templates"),
        point_tile_size=1024,
        workspace_mib=256,
    )
    assert schedule_digest(None) == baseline
    assert native == baseline
    contract = {
        "mode": "diagnostic",
        "precision_digits": 96,
        "probe_count": 4,
        "verification_probe_count": 4,
        "relative_tolerance": 1.0e-70,
        "absolute_tolerance": 1.0e-80,
        "seed": 7,
        "color_accuracy": "lc",
        "probe_process_id": "subprocess-a",
    }
    diagnostic = schedule_digest(contract)
    certified = schedule_digest(
        {**contract, "mode": "certified-reuse"},
    )
    assert diagnostic != baseline
    assert certified != diagnostic
    assert native != diagnostic
    assert native != certified
    for field, replacement in (
        ("precision_digits", 112),
        ("probe_count", 5),
        ("verification_probe_count", 5),
        ("relative_tolerance", 1.0e-60),
        ("absolute_tolerance", 1.0e-75),
        ("seed", 8),
        ("color_accuracy", "nlc"),
        ("probe_process_id", "subprocess-b"),
    ):
        assert schedule_digest({**contract, field: replacement}) != diagnostic


def test_combined_schedule_digests_match_public_wrappers() -> None:
    logical = cast(
        RecurrenceBuilderLogicalInputV1,
        _Logical("subprocess-a", 1, (_Leg(1, 1), _Leg(-1, 1)), _digest("p")),
    )
    kwargs = {
        "prepared_kernel_pack_digest": _digest("pack"),
        "direct_template_catalog_digest": _digest("templates"),
        "point_tile_size": 1024,
        "workspace_mib": 256,
    }
    relation_discovery = {
        "mode": "certified-reuse",
        "precision_digits": 96,
        "probe_count": 4,
    }

    native, request, sharing_domain = (
        schedule_sharing._recurrence_schedule_semantic_digests(
            logical,
            **kwargs,
            relation_discovery=relation_discovery,
        )
    )
    baseline_native, baseline_request, baseline_domain = (
        schedule_sharing._recurrence_schedule_semantic_digests(
            logical,
            **kwargs,
        )
    )

    assert native == recurrence_native_schedule_semantic_digest(logical, **kwargs)
    assert request == recurrence_schedule_semantic_digest(
        logical,
        **kwargs,
        relation_discovery=relation_discovery,
    )
    assert baseline_native == baseline_request == native
    assert sharing_domain != baseline_domain
    _, _, other_probe_domain = schedule_sharing._recurrence_schedule_semantic_digests(
        logical,
        **kwargs,
        relation_discovery={
            **relation_discovery,
            "probe_process_id": "different-process",
        },
    )
    assert other_probe_domain != sharing_domain


def test_combined_schedule_digests_traverse_logical_input_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logical = cast(
        RecurrenceBuilderLogicalInputV1,
        _Logical("subprocess-a", 1, (_Leg(1, 1), _Leg(-1, 1)), _digest("p")),
    )
    original_schedule_plain = schedule_sharing._schedule_plain
    logical_traversals = 0

    def counting_schedule_plain(value: object) -> object:
        nonlocal logical_traversals
        if value is logical:
            logical_traversals += 1
        return original_schedule_plain(value)

    monkeypatch.setattr(schedule_sharing, "_schedule_plain", counting_schedule_plain)
    schedule_sharing._recurrence_schedule_semantic_digests(
        logical,
        prepared_kernel_pack_digest=_digest("pack"),
        direct_template_catalog_digest=_digest("templates"),
        point_tile_size=1024,
        workspace_mib=256,
        relation_discovery={"mode": "diagnostic"},
    )

    assert logical_traversals == 1


def test_identical_process_schedules_are_lowered_once() -> None:
    cache = RecurrenceScheduleLoweringCache[str]()
    calls = 0

    def lower() -> str:
        nonlocal calls
        calls += 1
        return "root schedule"

    digest = _digest("schedule")
    assert cache.lower_once(digest, lower) == "root schedule"
    assert cache.lower_once(digest, lower) == "root schedule"
    assert calls == 1


def _pairing(
    process_id: str,
    *,
    fundamental: int,
    antifundamental: int,
    source_count: int,
) -> FermionPairingCatalogV1:
    endpoints = (
        ExternalFermionEndpointRowV1(
            endpoint_id=0,
            source_slot=antifundamental,
            public_label=antifundamental + 1,
            species_class_id=0,
            species_id="model:d",
            particle_orientation="antiparticle",
            color_orientation="antifundamental",
            state_template_ids=("dbar",),
            anti_state_template_ids=("d",),
            basis_ids=("dirac",),
            color_representations=(-3,),
            contract_digest=_digest(f"{process_id}:anti"),
        ),
        ExternalFermionEndpointRowV1(
            endpoint_id=1,
            source_slot=fundamental,
            public_label=fundamental + 1,
            species_class_id=0,
            species_id="model:d",
            particle_orientation="particle",
            color_orientation="fundamental",
            state_template_ids=("d",),
            anti_state_template_ids=("dbar",),
            basis_ids=("dirac",),
            color_representations=(3,),
            contract_digest=_digest(f"{process_id}:fund"),
        ),
    )
    pairing_class = FermionPairingClassRowV1(
        class_id=0,
        species_class_id=0,
        species_id="model:d",
        fundamental_source_slots=(fundamental,),
        antifundamental_source_slots=(antifundamental,),
        reference_pairings=((fundamental, antifundamental),),
        pairing_count=1,
        proof_digest=_digest(f"{process_id}:class"),
    )
    lineage = [NO_FERMION_LINE] * source_count
    lineage[fundamental] = 0
    lineage[antifundamental] = 0
    rule = FermionPairingRuleRowV1(
        rule_id=0,
        class_pairing_indices=((0, 0),),
        endpoint_pairings=((fundamental, antifundamental),),
        source_slot_permutation=tuple(range(source_count)),
        lineage_by_source_slot=tuple(lineage),
        fermion_parity=1,
        exact_factor=ModelExactComplexRationalV1.one(),
        multiplicity=1,
        proof_algorithm="canonical-external-fermion-pairing-v1",
        proof_digest=_digest(f"{process_id}:rule"),
    )
    return FermionPairingCatalogV1(
        process_key=process_id,
        source_count=source_count,
        endpoints=endpoints,
        pairing_classes=(pairing_class,),
        rules=(rule,),
        topology_digest=_digest(f"{process_id}:pairing-topology"),
        semantic_digest=_digest(f"{process_id}:pairing-semantic"),
    )


def _source_state(
    *,
    state_template_id: int,
    public_helicity: int,
    momentum_sign: int,
) -> RecurrenceSourceStateV1:
    return RecurrenceSourceStateV1(
        state_index=0,
        public_helicity=public_helicity,
        chirality=0,
        spin_state=public_helicity,
        current_state_template_id=state_template_id,
        source_template_id=state_template_id,
        momentum_sign=momentum_sign,
        crossing_phase=ExactComplexRationalV1(1),
    )


def _crossed_logical(*, target: bool) -> RecurrenceBuilderLogicalInputV1:
    process_id = "d_g_to_d_g" if target else "d_dbar_to_g_g"
    if target:
        outgoing = (-1, 21, 1, 21)
        physical = (1, 21, 1, 21)
        initial = (True, True, False, False)
        helicities = (-1, -1, 1, 1)
        momentum_signs = (-1, -1, 1, 1)
        state_ids = (0, 2, 1, 2)
        source_map = (0, 2, 1, 3)
    else:
        outgoing = (-1, 1, 21, 21)
        physical = (1, -1, 21, 21)
        initial = (True, True, False, False)
        helicities = (-1, 1, -1, 1)
        momentum_signs = (-1, -1, 1, 1)
        state_ids = (0, 1, 2, 2)
        source_map = (0, 1, 2, 3)
    external = tuple(
        RecurrenceExternalLegV1(
            source_slot=slot,
            public_label=slot + 1,
            physical_pdg=physical[slot],
            outgoing_pdg=outgoing[slot],
            is_initial=initial[slot],
            is_fermionic=abs(outgoing[slot]) == 1,
            source_states=(
                _source_state(
                    state_template_id=state_ids[slot],
                    public_helicity=helicities[slot],
                    momentum_sign=momentum_signs[slot],
                ),
            ),
            momentum_mask=1 << slot,
            support_mask=2 if target else 1,
        )
        for slot in range(4)
    )
    root_word = (1, 2, 3, 0)
    word = tuple(source_map[slot] for slot in root_word)
    open_string = RecurrenceLCOpenStringV1(
        fundamental_source_slot=source_map[1],
        antifundamental_source_slot=source_map[0],
        adjoint_source_slots=(source_map[2], source_map[3]),
    )
    sector = RecurrencePhysicalLCSectorV1(
        sector_id=0,
        public_id=f"{process_id}:sector",
        kind="open-lines",
        closure_source_slot=source_map[0],
        closure_proof_algorithm="canonical-lc-closure-anchor-v2",
        closure_proof_digest=_digest(f"{process_id}:closure"),
        open_strings=(open_string,),
        word_source_slots=word,
        support_mask=2 if target else 1,
    )
    flow = RecurrencePublicLCFlowV1(
        flow_id=0,
        public_id="flow:" + ",".join(str(slot + 1) for slot in word),
        construction_sector_id=0,
        word_source_slots=word,
        source_slot_permutation=(0, 1, 2, 3),
    )
    replay_target = RecurrenceReplayTargetV1(
        sector_id=0,
        external_permutation=(0, 1, 2, 3),
        source_slot_permutation=(0, 1, 2, 3),
    )
    replay = RecurrenceReplayPartitionV1(
        representative_sector_id=0,
        materialized_sector_id=0,
        proof_algorithm="exact-crossing-replay-v1",
        proof_digest=_digest(f"{process_id}:replay"),
        targets=(replay_target,),
    )
    references = tuple(
        RecurrenceSemanticTemplateReferenceV1(
            kind=kind,
            template_id=index,
            semantic_digest=_digest(f"{kind}:{index}"),
        )
        for kind in ("current-state", "source")
        for index in range(3)
    )
    semantics = (
        *(
            RecurrenceSemanticDigestV1(role, _digest(f"{process_id}:{role}"))
            for role in (
                "process",
                "color-plan",
                "fermion-pairing-semantic",
                "fermion-pairing-topology",
                "closure-reconstruction",
            )
        ),
        RecurrenceSemanticDigestV1("model-catalog", _digest("model")),
        RecurrenceSemanticDigestV1("prepared-catalog", _digest("prepared")),
        RecurrenceSemanticDigestV1(
            "helicity-equivalence:global-flip-v1",
            _digest(f"{process_id}:global-flip"),
        ),
    )
    return RecurrenceBuilderLogicalInputV1(
        process_id=process_id,
        layout="topology-replay",
        semantic_digests=semantics,
        external_legs=external,
        physical_sectors=(sector,),
        public_flows=(flow,),
        semantic_template_references=references,
        normalization=RecurrenceNormalizationV1(
            ExactComplexRationalV1(2 if target else 1),
            "binding-local-normalization",
            _digest(f"{process_id}:normalization"),
        ),
        fermion_pairing_catalog=_pairing(
            process_id,
            fundamental=source_map[1],
            antifundamental=source_map[0],
            source_count=4,
        ),
        replay_partitions=(replay,),
        coupling_limits=(RecurrenceCouplingLimitV1("QCD", 2, 2),),
        parameter_projection=(RecurrenceParameterProjectionV1(0, "alpha_s", 0, 0),),
        process_support_mask=2 if target else 1,
    )


def test_crossed_processes_share_only_through_an_exact_bijection() -> None:
    root = _crossed_logical(target=False)
    target = _crossed_logical(target=True)
    remap = exact_recurrence_process_bijection(
        root,
        target,
        direct_executor_count=4,
        parameter_slot_count=1,
    )
    assert remap is not None
    assert remap.source_slots == (0, 2, 1, 3)
    assert remap.source_momentum_signs == (1, -1, -1, 1)
    assert remap.source_helicity_signs == (1, 1, 1, 1)
    assert remap.source_state_offsets == (0, 1, 2, 3, 4)
    assert remap.source_state_indices == (0, 0, 0, 0)
    assert remap.public_flow_ids == (0,)
    assert remap.physical_sector_ids == (0,)

    cache = RecurrenceScheduleLoweringCache[str]()
    calls = 0

    def lower(value: str) -> str:
        nonlocal calls
        calls += 1
        return value

    first = cache.lower_process(
        root,
        schedule_digest=_digest("root request"),
        sharing_domain_digest=_digest("shared domain"),
        native_schedule_semantic_digest=_digest("root native semantics"),
        direct_executor_count=4,
        parameter_slot_count=1,
        lower=lambda: lower("root"),
    )
    second = cache.lower_process(
        target,
        schedule_digest=_digest("target request"),
        sharing_domain_digest=_digest("shared domain"),
        native_schedule_semantic_digest=_digest("target native semantics"),
        direct_executor_count=4,
        parameter_slot_count=1,
        lower=lambda: lower("target"),
    )
    assert calls == 1
    assert first.output == second.output == "root"
    assert second.schedule_digest == first.schedule_digest
    assert second.native_schedule_semantic_digest == (
        first.native_schedule_semantic_digest
    )
    assert second.remap == remap


def test_isomorphic_processes_do_not_share_different_sharing_domains() -> None:
    root = _crossed_logical(target=False)
    target = _crossed_logical(target=True)
    cache = RecurrenceScheduleLoweringCache[str]()
    calls = 0

    def lower(value: str) -> str:
        nonlocal calls
        calls += 1
        return value

    first = cache.lower_process(
        root,
        schedule_digest=_digest("root probe context"),
        sharing_domain_digest=_digest("root probe domain"),
        native_schedule_semantic_digest=_digest("root probe native semantics"),
        direct_executor_count=4,
        parameter_slot_count=1,
        lower=lambda: lower("root evidence"),
    )
    second = cache.lower_process(
        target,
        schedule_digest=_digest("target probe context"),
        sharing_domain_digest=_digest("target probe domain"),
        native_schedule_semantic_digest=_digest("target probe native semantics"),
        direct_executor_count=4,
        parameter_slot_count=1,
        lower=lambda: lower("target evidence"),
    )

    assert calls == 2
    assert first.output == "root evidence"
    assert second.output == "target evidence"
    assert first.schedule_digest != second.schedule_digest


def test_cross_process_aliasing_fails_closed_on_contract_changes() -> None:
    root = _crossed_logical(target=False)
    target = _crossed_logical(target=True)

    changed_parameter = replace(
        target,
        parameter_projection=(
            replace(target.parameter_projection[0], prepared_parameter_id=None),
        ),
    )
    assert (
        exact_recurrence_process_bijection(
            root,
            changed_parameter,
            direct_executor_count=4,
            parameter_slot_count=1,
        )
        is None
    )

    changed_flow = replace(
        target,
        public_flows=(
            replace(
                target.public_flows[0],
                reduction_weight=ExactComplexRationalV1(2),
            ),
        ),
    )
    assert (
        exact_recurrence_process_bijection(
            root,
            changed_flow,
            direct_executor_count=4,
            parameter_slot_count=1,
        )
        is None
    )

    changed_templates = list(target.semantic_template_references)
    changed_templates[0] = replace(
        changed_templates[0],
        semantic_digest=_digest("different kernel contract"),
    )
    assert (
        exact_recurrence_process_bijection(
            root,
            replace(target, semantic_template_references=tuple(changed_templates)),
            direct_executor_count=4,
            parameter_slot_count=1,
        )
        is None
    )


def _process(
    root: Path,
    *,
    process_id: str,
    schedule_digest: str,
    native_schedule_digest: str | None = None,
    payload: bytes,
    support_mask: int,
) -> SimpleNamespace:
    path = root / f"{schedule_digest}-{process_id}.pacbin"
    path.write_bytes(payload)
    return SimpleNamespace(
        process_id=process_id,
        recurrence_schedule_path=path,
        recurrence_schedule_digest=schedule_digest,
        recurrence_native_schedule_semantic_digest=(
            schedule_digest
            if native_schedule_digest is None
            else native_schedule_digest
        ),
        recurrence_schedule_size_bytes=len(payload),
        recurrence_schedule_sha256=hashlib.sha256(payload).hexdigest(),
        recurrence_schedule_member_count=1,
        recurrence_schedule_unpacked_size_bytes=len(payload),
        recurrence_schedule_index_sha256=_digest(f"index-{schedule_digest}"),
        builder_input_sha256=_digest(f"binding-{process_id}"),
        process_digest=_digest(f"process-{process_id}"),
        process_support_mask=support_mask,
        recurrence_process_remap=_identity_remap(),
        recurrence_process_executor_pack=_executor_pack(),
    )


def test_bounded_process_set_interns_before_publication(tmp_path: Path) -> None:
    shared = _digest("shared schedule")
    distinct = _digest("distinct schedule")
    plan = intern_recurrence_schedules(
        (
            _process(
                tmp_path,
                process_id="u_ubar_to_g_g",
                schedule_digest=shared,
                payload=b"shared",
                support_mask=1,
            ),
            _process(
                tmp_path,
                process_id="d_dbar_to_g_g",
                schedule_digest=shared,
                payload=b"shared",
                support_mask=2,
            ),
            _process(
                tmp_path,
                process_id="g_g_to_g_g",
                schedule_digest=distinct,
                payload=b"distinct",
                support_mask=4,
            ),
        )
    )

    assert len(plan.bindings) == 3
    assert len(plan.schedules) == 2
    assert plan.to_mapping()["runtime_ownership"] == (
        "root-schedule-plus-process-binding"
    )
    assert plan.to_mapping()["interning_phase"] == "before-direct-lowering"
    assert plan.binding("u_ubar_to_g_g").artifact_path.endswith(
        "/recurrence-binding.bin"
    )
    assert plan.binding("u_ubar_to_g_g").native_schedule_semantic_digest == shared
    assert plan.binding("u_ubar_to_g_g").process_digest == _digest(
        "process-u_ubar_to_g_g"
    )


def test_package_identity_is_distinct_from_native_schedule_semantics(
    tmp_path: Path,
) -> None:
    request_digest = _digest("certified request")
    native_digest = _digest("relation-independent source schedule")
    plan = intern_recurrence_schedules(
        (
            _process(
                tmp_path,
                process_id="u_ubar_to_g_g",
                schedule_digest=request_digest,
                native_schedule_digest=native_digest,
                payload=b"schedule",
                support_mask=1,
            ),
        )
    )

    binding = plan.binding("u_ubar_to_g_g")
    assert plan.schedules[0].digest == request_digest
    assert binding.schedule_digest == request_digest
    assert binding.native_schedule_semantic_digest == native_digest
    assert binding.to_mapping()["native_schedule_semantic_digest"] == native_digest


def test_shared_request_digest_rejects_different_native_semantics(
    tmp_path: Path,
) -> None:
    request_digest = _digest("shared certified request")
    with pytest.raises(
        RecurrenceScheduleSharingError,
        match="maps to different native semantics",
    ):
        intern_recurrence_schedules(
            (
                _process(
                    tmp_path,
                    process_id="first",
                    schedule_digest=request_digest,
                    native_schedule_digest=_digest("first source schedule"),
                    payload=b"shared",
                    support_mask=1,
                ),
                _process(
                    tmp_path,
                    process_id="second",
                    schedule_digest=request_digest,
                    native_schedule_digest=_digest("second source schedule"),
                    payload=b"shared",
                    support_mask=2,
                ),
            )
        )


def test_binding_requires_native_schedule_semantic_digest() -> None:
    schedule_digest = _digest("native schedule identity")
    with pytest.raises(ValueError, match="native schedule semantic digest"):
        _recurrence_binding_native_schedule_semantic_digest(
            {"schedule_digest": schedule_digest},
            process_id="missing",
        )
    with pytest.raises(ValueError, match="native schedule semantic digest"):
        _recurrence_binding_native_schedule_semantic_digest(
            {
                "schedule_digest": schedule_digest,
                "native_schedule_semantic_digest": None,
            },
            process_id="malformed",
        )


def test_binding_payload_is_compact_and_process_owned() -> None:
    schedule = _digest("schedule")
    process = _digest("canonical process")
    semantic = _digest("process")
    payload = encode_recurrence_process_binding(
        process_id="u_ubar_to_g_g",
        schedule_digest=schedule,
        process_digest=process,
        process_semantic_digest=semantic,
        process_support_mask=1 << 70,
        remap=_identity_remap(),
        executor_pack=_executor_pack(),
    )
    (
        version,
        fixed_size,
        process_len,
        word_count,
        target_len,
        feature_count,
        descriptor_count,
        catalog_count,
    ) = struct.unpack_from("<8I", payload, 8)
    assert payload[:8] == RECURRENCE_PROCESS_BINDING_MAGIC
    assert payload == encode_recurrence_process_binding(
        process_id="u_ubar_to_g_g",
        schedule_digest=schedule,
        process_digest=process,
        process_semantic_digest=semantic,
        process_support_mask=1 << 70,
        remap=_identity_remap(),
        executor_pack=_executor_pack(),
    )
    assert version == 4
    assert fixed_size == 344
    assert word_count == 2
    assert target_len == len(b"symjit-storage-v3-portable")
    assert feature_count == 0
    assert descriptor_count == 1
    assert catalog_count == 4
    assert payload[88:120] == bytes.fromhex(schedule)
    assert payload[120:152] == bytes.fromhex(semantic)
    assert payload[152:184] == bytes.fromhex(_digest("compiled model"))
    assert payload[280:312] == bytes.fromhex(_digest("runtime layout"))
    assert payload[312:344] == bytes.fromhex(process)
    assert payload[344 : 344 + process_len] == b"u_ubar_to_g_g"
    assert payload.endswith(_executor_pack().descriptor_payloads[0])
    assert len(payload) < 512


def test_same_schedule_digest_cannot_name_different_payloads(
    tmp_path: Path,
) -> None:
    digest = _digest("schedule")
    with pytest.raises(
        RecurrenceScheduleSharingError,
        match="different payloads",
    ):
        intern_recurrence_schedules(
            (
                _process(
                    tmp_path,
                    process_id="first",
                    schedule_digest=digest,
                    payload=b"one",
                    support_mask=1,
                ),
                _process(
                    tmp_path,
                    process_id="second",
                    schedule_digest=digest,
                    payload=b"two",
                    support_mask=2,
                ),
            )
        )


def test_process_support_bits_are_independent() -> None:
    first = SimpleNamespace(
        process_id="first",
        recurrence_schedule_path=Path(__file__),
        recurrence_schedule_digest=_digest("schedule"),
        recurrence_native_schedule_semantic_digest=_digest("schedule"),
        recurrence_schedule_size_bytes=Path(__file__).stat().st_size,
        recurrence_schedule_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        recurrence_schedule_member_count=1,
        recurrence_schedule_unpacked_size_bytes=Path(__file__).stat().st_size,
        recurrence_schedule_index_sha256=_digest("index"),
        builder_input_sha256=_digest("first"),
        process_digest=_digest("first process"),
        process_support_mask=1,
        recurrence_process_remap=_identity_remap(),
        recurrence_process_executor_pack=_executor_pack(),
    )
    second = SimpleNamespace(**{**vars(first), "process_id": "second"})
    with pytest.raises(RecurrenceScheduleSharingError, match="support mask"):
        intern_recurrence_schedules((first, second))


@pytest.mark.skip(
    reason=(
        "full p p > j j j j generation belongs to the 30 GiB guarded "
        "post-build process-set acceptance run"
    )
)
def test_full_pp_to_four_jets_has_fewer_schedules_than_subprocesses() -> None:
    """Acceptance placeholder retained until the native split build is available."""
