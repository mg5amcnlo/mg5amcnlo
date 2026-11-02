# SPDX-License-Identifier: 0BSD
"""Topology grouping and replay safety for color sectors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from ..processes.ir import CanonicalProcessIR
from .plan_build import _ordered_open_line_blocks
from .plan_types import (
    ColorTopologyReplayCertificate,
    GenericColorPlan,
    LCColorSector,
    LCColorSectorReplayPartition,
    LCColorSectorTopologyGroup,
    LCColorTopologyReplayPlan,
)

if TYPE_CHECKING:
    from ..models.base import Model

_LC_REPLAY_PROOF_ALGORITHM = "canonical-model-contract-label-equivariance-v1"
_COLOR_REPLAY_PROOF_ALGORITHM = (
    "canonical-model-contract-color-label-equivariance-v2"
)


def _sector_topology_groups(
    process: CanonicalProcessIR,
    sectors: tuple[LCColorSector, ...],
) -> tuple[LCColorSectorTopologyGroup, ...]:
    by_signature: dict[tuple[object, ...], list[LCColorSector]] = {}
    for sector in sectors:
        by_signature.setdefault(
            _sector_topology_signature(process, sector),
            [],
        ).append(sector)

    groups: list[LCColorSectorTopologyGroup] = []
    for signature, sector_group in by_signature.items():
        representative = sector_group[0]
        representative_labels = _sector_topology_labels(representative)
        permutations: list[tuple[tuple[int, int], ...]] = []
        for sector in sector_group:
            sector_labels = _sector_topology_labels(sector)
            if len(sector_labels) != len(representative_labels):
                raise ValueError("isomorphic colour sectors have mismatched labels")
            permutations.append(
                tuple(zip(representative_labels, sector_labels, strict=True))
            )
        groups.append(
            LCColorSectorTopologyGroup(
                signature=signature,
                representative_sector_id=representative.id,
                sector_ids=tuple(sector.id for sector in sector_group),
                label_permutations=tuple(permutations),
            )
        )
    return tuple(groups)


def _sector_topology_signature(
    process: CanonicalProcessIR,
    sector: LCColorSector,
) -> tuple[object, ...]:
    pdg_by_label = {
        leg.label: leg.outgoing_pdg
        for leg in process.legs
        if leg.outgoing_pdg is not None
    }
    if sector.kind == "open-lines":
        line_by_coloured = {
            line.coloured_labels: line for line in sector.open_color_lines
        }
        ordered_blocks = (
            _ordered_open_line_blocks(sector.word_labels, sector.open_color_lines)
            if sector.word_labels
            else None
        )
        ordered_lines = (
            tuple(line_by_coloured[block] for block in ordered_blocks)
            if ordered_blocks is not None
            else sector.open_color_lines
        )
        return (
            sector.kind,
            tuple(
                (
                    pdg_by_label[line.fundamental_label],
                    tuple(pdg_by_label[label] for label in line.adjoint_labels),
                    pdg_by_label[line.antifundamental_label],
                    tuple(pdg_by_label[label] for label in line.singlet_labels),
                )
                for line in ordered_lines
            ),
        )
    if sector.kind == "single-trace":
        return (
            sector.kind,
            tuple(pdg_by_label[label] for label in sector.trace_labels),
            tuple(pdg_by_label[label] for label in sector.singlet_labels),
        )
    return (
        sector.kind,
        tuple(pdg_by_label[label] for label in sector.singlet_labels),
    )


def _sector_topology_labels(sector: LCColorSector) -> tuple[int, ...]:
    if sector.kind == "open-lines":
        line_by_coloured = {
            line.coloured_labels: line for line in sector.open_color_lines
        }
        ordered_blocks = (
            _ordered_open_line_blocks(sector.word_labels, sector.open_color_lines)
            if sector.word_labels
            else None
        )
        ordered_lines = (
            tuple(line_by_coloured[block] for block in ordered_blocks)
            if ordered_blocks is not None
            else sector.open_color_lines
        )
        return tuple(label for line in ordered_lines for label in line.line_labels)
    if sector.kind == "single-trace":
        return (*sector.trace_labels, *sector.singlet_labels)
    return sector.singlet_labels


def _jsonable_signature(signature: tuple[object, ...]) -> list[object]:
    def convert(value: object) -> object:
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        return value

    return [convert(item) for item in signature]


def lc_topology_replay_safe_groups(
    color_plan: GenericColorPlan,
) -> tuple[LCColorSectorTopologyGroup, ...]:
    """Return LC topology groups safe for physical-input replay.

    Runtime replay currently evaluates one representative sector with physical
    external momenta in the user's process ordering.  Therefore every replay
    permutation must preserve the initial-state label set.  Pure single-trace
    sectors are also kept out of this fast path for now because their trace
    symmetries need dedicated colour-convention validation.
    """

    if color_plan.color_accuracy != "lc":
        return ()
    return tuple(
        group
        for group in color_plan.topology_groups
        if _lc_topology_group_replay_safe(color_plan, group)
    )


def color_topology_replay_partitions(
    color_plan: GenericColorPlan,
) -> tuple[LCColorSectorReplayPartition, ...]:
    """Partition topology groups into exact replay-safe representatives.

    The selected-sector ``lc_topology_replay_safe_groups`` helper only accepts
    groups whose full topology orbit preserves the initial-state label set.
    Complete artifacts can be more general: they split one topology group into
    several initial-label-safe blocks and materialize one representative
    sidecar per block.  The partition is purely structural and is valid for
    every requested colour accuracy; mode-specific proof and runtime code must
    still authenticate the color reduction before executing it.

    In particular, a full-color seven-gluon plan has 720 physical trace
    sectors but exactly six 120-sector partitions.  Those are the same six
    crossing templates generated by the original AmpliCol library.  Keeping
    this structural primitive color-generic lets recurrence, compiled, and
    eager lowering share one fail-closed orbit certificate.
    """

    if color_plan.color_accuracy not in {"lc", "nlc", "full"}:
        return ()
    initial_labels = {leg.label for leg in color_plan.process.initial_legs}
    if not initial_labels:
        return ()
    partitions: list[LCColorSectorReplayPartition] = []
    for group in color_plan.topology_groups:
        sectors = tuple(
            sector
            for sector_id in group.sector_ids
            if (sector := color_plan.sector(sector_id)) is not None
        )
        if len(sectors) != len(group.sector_ids):
            continue
        if any(sector.kind not in {"open-lines", "single-trace"} for sector in sectors):
            continue
        base_maps = {
            int(sector_id): {
                int(representative_label): int(sector_label)
                for representative_label, sector_label in permutation
            }
            for sector_id, permutation in zip(
                group.sector_ids,
                group.label_permutations,
                strict=True,
            )
        }
        sector_ids_by_initial_preimage: dict[tuple[int, ...], list[int]] = {}
        complete_initial_maps = True
        for sector_id in group.sector_ids:
            sector_map = base_maps[int(sector_id)]
            inverse_sector_map = {
                sector_label: representative_label
                for representative_label, sector_label in sector_map.items()
            }
            if not initial_labels.issubset(inverse_sector_map):
                complete_initial_maps = False
                break
            initial_preimage = tuple(
                sorted(inverse_sector_map[label] for label in initial_labels)
            )
            sector_ids_by_initial_preimage.setdefault(initial_preimage, []).append(
                int(sector_id)
            )
        if not complete_initial_maps:
            continue
        partition_sector_ids = sorted(
            sector_ids_by_initial_preimage.values(),
            key=lambda sector_ids: min(sector_ids),
        )
        for grouped_sector_ids in partition_sector_ids:
            representative = min(grouped_sector_ids)
            representative_map = base_maps[representative]
            inverse_representative_map = {
                sector_label: representative_label
                for representative_label, sector_label in representative_map.items()
            }
            active_sector_ids: list[int] = []
            relative_permutations: list[tuple[tuple[int, int], ...]] = []
            replay_weights: list[float] = []
            grouped_sector_id_set = set(grouped_sector_ids)
            for sector_id in group.sector_ids:
                if int(sector_id) not in grouped_sector_id_set:
                    continue
                sector = color_plan.sector(int(sector_id))
                if sector is None:
                    continue
                sector_map = base_maps[int(sector_id)]
                relative_map = {
                    representative_label: sector_map[
                        inverse_representative_map[representative_label]
                    ]
                    for representative_label in sorted(representative_map.values())
                }
                if {relative_map[label] for label in initial_labels} != initial_labels:
                    raise RuntimeError(
                        "internal LC replay partitioning error: sectors grouped "
                        "by initial-label preimage do not preserve the initial set"
                    )
                active_sector_ids.append(int(sector_id))
                relative_permutations.append(tuple(sorted(relative_map.items())))
                replay_weights.append(
                    _lc_topology_replay_sector_weight(color_plan, sector)
                )
            if not active_sector_ids:
                raise RuntimeError(
                    "internal LC replay partitioning error: representative sector "
                    f"{representative} did not produce a non-empty replay block"
                )
            partitions.append(
                LCColorSectorReplayPartition(
                    representative_sector_id=representative,
                    active_sector_ids=tuple(active_sector_ids),
                    label_permutations=tuple(relative_permutations),
                    replay_weights=tuple(replay_weights),
                )
            )
    return tuple(partitions)


def lc_topology_replay_partitions(
    color_plan: GenericColorPlan,
) -> tuple[LCColorSectorReplayPartition, ...]:
    """Return color-generic partitions only for the existing LC runtime."""

    if color_plan.color_accuracy != "lc":
        return ()
    return color_topology_replay_partitions(color_plan)


def build_color_topology_replay_certificate(
    color_plan: GenericColorPlan,
    model: Model,
) -> ColorTopologyReplayCertificate | None:
    """Prove replay orbits without selecting a runtime generation mode.

    The proof binds the requested color accuracy in addition to the canonical
    model, source, sector, and relabeling contracts.  A consumer may only use
    these representatives after separately proving that its color reduction
    accepts replayed physical-sector amplitudes.
    """

    if (
        color_plan.color_accuracy not in {"lc", "nlc", "full"}
        or color_plan.truncated
        or len(color_plan.sectors) < 2
    ):
        return None
    physical_sector_ids = tuple(sorted(int(sector.id) for sector in color_plan.sectors))
    try:
        model_contract_digest = _canonical_contract_digest(
            _canonical_model_replay_contract(model)
        )
    except Exception:
        return None
    residual: set[int] = set(physical_sector_ids)
    proven: list[LCColorSectorReplayPartition] = []
    diagnostics: list[str] = []
    for candidate in color_topology_replay_partitions(color_plan):
        if len(candidate.active_sector_ids) < 2:
            continue
        try:
            proof = _prove_topology_replay_partition(
                color_plan,
                candidate,
                model,
                model_contract_digest=model_contract_digest,
                proof_algorithm=_COLOR_REPLAY_PROOF_ALGORITHM,
                bind_color_accuracy=True,
            )
        except Exception as exc:
            proof = None
            diagnostics.append(
                "color replay partition "
                f"{candidate.representative_sector_id} proof failed closed: {exc}"
            )
        if proof is None:
            diagnostics.append(
                "color replay partition "
                f"{candidate.representative_sector_id} remains materialized"
            )
            continue
        proof_digest, replay_signs = proof
        partition = replace(
            candidate,
            materialized_sector_id=candidate.representative_sector_id,
            replay_signs=replay_signs,
            proof_algorithm=_COLOR_REPLAY_PROOF_ALGORITHM,
            proof_digest=proof_digest,
        )
        proven.append(partition)
        residual.difference_update(partition.active_sector_ids)
    if not proven:
        return None
    return ColorTopologyReplayCertificate(
        color_accuracy=color_plan.color_accuracy,
        physical_sector_ids=physical_sector_ids,
        partitions=tuple(proven),
        residual_sector_ids=tuple(sorted(residual)),
        diagnostics=tuple(diagnostics),
    )


def build_lc_topology_replay_plan(
    color_plan: GenericColorPlan,
    model: Model,
) -> LCColorTopologyReplayPlan | None:
    """Build independently proof-gated replay classes for a complete LC plan.

    Candidate topology partitions are cheap structural orbits.  This second
    pass certifies each orbit against canonical source/kernel contracts and
    exact relabelled colour structures.  A failed orbit becomes residual
    materialization without disabling any independently proven orbit.
    """

    if (
        color_plan.color_accuracy != "lc"
        or color_plan.truncated
        or len(color_plan.sectors) < 2
    ):
        return None
    physical_sector_ids = tuple(sorted(int(sector.id) for sector in color_plan.sectors))
    try:
        model_contract_digest = _canonical_contract_digest(
            _canonical_model_replay_contract(model)
        )
    except Exception:
        return None
    residual: set[int] = set(physical_sector_ids)
    proven: list[LCColorSectorReplayPartition] = []
    diagnostics: list[str] = []
    for candidate in lc_topology_replay_partitions(color_plan):
        if len(candidate.active_sector_ids) < 2:
            continue
        try:
            proof = _prove_lc_topology_replay_partition(
                color_plan,
                candidate,
                model,
                model_contract_digest=model_contract_digest,
            )
        except Exception as exc:
            proof = None
            diagnostics.append(
                "LC replay partition "
                f"{candidate.representative_sector_id} proof failed closed: {exc}"
            )
        if proof is None:
            diagnostics.append(
                "LC replay partition "
                f"{candidate.representative_sector_id} remains materialized"
            )
            continue
        proof_digest, replay_signs = proof
        partition = replace(
            candidate,
            materialized_sector_id=candidate.representative_sector_id,
            replay_signs=replay_signs,
            proof_algorithm=_LC_REPLAY_PROOF_ALGORITHM,
            proof_digest=proof_digest,
        )
        proven.append(partition)
        residual.difference_update(partition.active_sector_ids)
    if not proven:
        return None
    return LCColorTopologyReplayPlan(
        physical_sector_ids=physical_sector_ids,
        partitions=tuple(proven),
        residual_sector_ids=tuple(sorted(residual)),
        diagnostics=tuple(diagnostics),
    )


def _prove_lc_topology_replay_partition(
    color_plan: GenericColorPlan,
    partition: LCColorSectorReplayPartition,
    model: Model,
    *,
    model_contract_digest: str,
) -> tuple[str, tuple[int, ...]] | None:
    """Preserve the accepted LC proof and digest contract."""

    return _prove_topology_replay_partition(
        color_plan,
        partition,
        model,
        model_contract_digest=model_contract_digest,
        proof_algorithm=_LC_REPLAY_PROOF_ALGORITHM,
        bind_color_accuracy=False,
    )


def _prove_topology_replay_partition(
    color_plan: GenericColorPlan,
    partition: LCColorSectorReplayPartition,
    model: Model,
    *,
    model_contract_digest: str,
    proof_algorithm: str,
    bind_color_accuracy: bool,
) -> tuple[str, tuple[int, ...]] | None:
    """Certify label equivariance for one local replay partition.

    External UFO tensors have already been canonized by the model compiler;
    their exact oriented-kernel expressions and proof digests are included in
    the contract below.  Built-in models contribute the same model-generic
    source/lowering/evaluation contracts.  No process name or SM particle list
    participates in this proof.
    """

    representative = color_plan.sector(partition.representative_sector_id)
    if representative is None:
        return None
    labels = tuple(sorted(int(leg.label) for leg in color_plan.process.legs))
    initial_labels = frozenset(
        int(leg.label) for leg in color_plan.process.initial_legs
    )
    leg_by_label = {int(leg.label): leg for leg in color_plan.process.legs}
    source_contracts = {
        label: _canonical_source_contract(model, leg_by_label[label])
        for label in labels
    }
    replay_signs: list[int] = []
    mapping_contracts: list[dict[str, object]] = []
    for sector_id, permutation, weight in zip(
        partition.active_sector_ids,
        partition.label_permutations,
        partition.weights,
        strict=True,
    ):
        sector = color_plan.sector(sector_id)
        if sector is None:
            return None
        explicit_mapping = {int(left): int(right) for left, right in permutation}
        if set(explicit_mapping) != set(explicit_mapping.values()):
            return None
        if not set(explicit_mapping).issubset(labels):
            return None
        mapping = {
            label: explicit_mapping.get(label, label)
            for label in labels
        }
        if set(mapping.values()) != set(labels):
            return None
        if {mapping[label] for label in initial_labels} != set(initial_labels):
            return None
        if _remapped_sector_contract(representative, mapping) != _sector_contract(
            sector
        ):
            return None
        for source_label, target_label in mapping.items():
            if source_contracts[source_label] != source_contracts[target_label]:
                return None
        sign = _external_fermion_permutation_sign(
            labels,
            mapping,
            source_contracts,
        )
        replay_signs.append(sign)
        mapping_contracts.append(
            {
                "sector_id": int(sector_id),
                "weight": float(weight),
                "sign": sign,
                "label_permutation": [
                    list(item) for item in sorted(explicit_mapping.items())
                ],
                "sector_contract": _sector_contract(sector),
            }
        )
    proof_payload = {
        "algorithm": proof_algorithm,
        "process": {
            "initial_labels": sorted(initial_labels),
            "external_source_contracts": [
                [label, source_contracts[label]] for label in labels
            ],
        },
        "representative_sector_id": partition.representative_sector_id,
        "representative_sector_contract": _sector_contract(representative),
        "mappings": mapping_contracts,
        "model_contract_digest": model_contract_digest,
    }
    if bind_color_accuracy:
        proof_payload["color_accuracy"] = color_plan.color_accuracy
    return _canonical_contract_digest(proof_payload), tuple(replay_signs)


def _canonical_contract_digest(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _canonical_source_contract(model: Model, leg: object) -> dict[str, object]:
    outgoing_pdg = getattr(leg, "outgoing_pdg", None)
    if outgoing_pdg is None:
        raise ValueError("LC replay source has no outgoing particle identity")
    source = model._source_ir(int(outgoing_pdg))
    return {
        "role": "initial" if bool(getattr(leg, "is_initial", False)) else "final",
        "source": source.to_json_dict(),
    }


def _canonical_model_replay_contract(model: Model) -> dict[str, object]:
    oriented_kernels: list[object] = []
    compiled = getattr(model, "compiled", None)
    ir = getattr(compiled, "ir", None)
    for kernel in getattr(ir, "oriented_kernels", ()):
        serializer = getattr(kernel, "to_dict", None)
        if callable(serializer):
            oriented_kernels.append(serializer())
    vertices = []
    seen_kinds: set[int] = set()
    for vertex in sorted(
        model.vertices,
        key=lambda item: (int(item.kind), tuple(item.particles), tuple(item.coupling)),
    ):
        kind = int(vertex.kind)
        if kind in seen_kinds:
            continue
        seen_kinds.add(kind)
        vertices.append(
            {
                "kind": kind,
                "lowering": model.vertex_lowering_rule(kind).to_json_dict(),
                "evaluation_equivalence": (
                    model.vertex_evaluation_equivalence(kind).to_json_dict()
                ),
            }
        )
    certificates = getattr(model, "_symmetry_certificates", None)
    certificate_contract = None
    if certificates is not None:
        certificate_contract = {
            name: _json_contract_value(getattr(certificates, name))
            for name in getattr(certificates, "__dataclass_fields__", {})
        }
    return {
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "vertices": vertices,
        "oriented_kernels": oriented_kernels,
        "external_symmetry_certificates": certificate_contract,
    }


def _json_contract_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_contract_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, frozenset | set):
        return sorted(_json_contract_value(item) for item in value)
    if isinstance(value, tuple | list):
        return [_json_contract_value(item) for item in value]
    return value


def _sector_contract(sector: LCColorSector) -> dict[str, object]:
    return {
        "kind": sector.kind,
        "open_color_lines": sorted(
            (
                int(line.fundamental_label),
                int(line.antifundamental_label),
                tuple(int(label) for label in line.adjoint_labels),
                tuple(int(label) for label in line.singlet_labels),
            )
            for line in sector.open_color_lines
        ),
        "trace_labels": tuple(int(label) for label in sector.trace_labels),
        "singlet_labels": tuple(int(label) for label in sector.singlet_labels),
        "word_labels": tuple(int(label) for label in sector.word_labels),
    }


def _remapped_sector_contract(
    sector: LCColorSector,
    mapping: Mapping[int, int],
) -> dict[str, object]:
    def mapped(labels: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(mapping[int(label)] for label in labels)

    return {
        "kind": sector.kind,
        "open_color_lines": sorted(
            (
                mapping[int(line.fundamental_label)],
                mapping[int(line.antifundamental_label)],
                mapped(line.adjoint_labels),
                mapped(line.singlet_labels),
            )
            for line in sector.open_color_lines
        ),
        "trace_labels": mapped(sector.trace_labels),
        "singlet_labels": mapped(sector.singlet_labels),
        "word_labels": mapped(sector.word_labels),
    }


def _external_fermion_permutation_sign(
    labels: tuple[int, ...],
    mapping: Mapping[int, int],
    source_contracts: Mapping[int, Mapping[str, object]],
) -> int:
    classes: dict[tuple[str, str], list[int]] = {}
    for label in labels:
        contract = source_contracts[label]
        source = contract["source"]
        if not isinstance(source, Mapping) or source.get("statistics") != "fermion":
            continue
        identity = source.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("fermion source identity contract is missing")
        classes.setdefault(
            (str(contract["role"]), str(identity["canonical_id"])),
            [],
        ).append(label)
    sign = 1
    for class_labels in classes.values():
        ordered = sorted(class_labels)
        positions = {label: index for index, label in enumerate(ordered)}
        permutation = [positions[mapping[label]] for label in ordered]
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(len(permutation))
            for right in range(left + 1, len(permutation))
        )
        if inversions % 2:
            sign = -sign
    return sign


def _lc_topology_replay_sector_weight(
    color_plan: GenericColorPlan,
    sector: LCColorSector,
) -> float:
    """Return the LC multiplicity represented by a materialized sector.

    A model-proven pure single-trace LC plan may fold trace reflections during
    colour-sector enumeration.  Replaying such a materialized representative
    with weight two preserves the full ordering sum.
    """

    if (
        color_plan.color_accuracy == "lc"
        and color_plan.trace_reflections_folded
        and sector.kind == "single-trace"
        and len(sector.trace_labels) > 2
    ):
        return 2.0
    return 1.0


def lc_line_pairing_representative_ids(
    color_plan: GenericColorPlan,
) -> tuple[int, ...]:
    """Return one sector per LC open-line pairing/allocation.

    This is a generic colour-flow pruning helper.  It keeps distinct
    fundamental-antifundamental pairings and distinct adjoint/singlet
    attachments, but drops
    duplicate sectors that only permute complete open-line blocks in the colour
    word.  It deliberately does not group sectors with different flavour
    pairings or different particles assigned to a line.
    """

    if color_plan.color_accuracy != "lc":
        return ()
    representatives: list[int] = []
    seen: set[tuple[object, ...]] = set()
    pdg_by_label = {
        leg.label: leg.outgoing_pdg
        for leg in color_plan.process.legs
        if leg.outgoing_pdg is not None
    }
    for sector in color_plan.sectors:
        signature = _sector_line_pairing_signature(pdg_by_label, sector)
        if signature in seen:
            continue
        seen.add(signature)
        representatives.append(int(sector.id))
    return tuple(representatives)


def _sector_line_pairing_signature(
    pdg_by_label: dict[int, int | None],
    sector: LCColorSector,
) -> tuple[object, ...]:
    if sector.kind == "open-lines":
        return (
            sector.kind,
            tuple(
                sorted(
                    (
                        line.fundamental_label,
                        pdg_by_label.get(line.fundamental_label),
                        line.antifundamental_label,
                        pdg_by_label.get(line.antifundamental_label),
                        tuple(
                            (label, pdg_by_label.get(label))
                            for label in line.adjoint_labels
                        ),
                        tuple(
                            (label, pdg_by_label.get(label))
                            for label in line.singlet_labels
                        ),
                    )
                    for line in sector.open_color_lines
                )
            ),
        )
    if sector.kind == "single-trace":
        canonical_trace = min(sector.trace_labels, tuple(reversed(sector.trace_labels)))
        return (
            sector.kind,
            tuple((label, pdg_by_label.get(label)) for label in canonical_trace),
            tuple((label, pdg_by_label.get(label)) for label in sector.singlet_labels),
        )
    return (
        sector.kind,
        tuple((label, pdg_by_label.get(label)) for label in sector.singlet_labels),
    )


def _lc_topology_group_replay_safe(
    color_plan: GenericColorPlan,
    group: LCColorSectorTopologyGroup,
) -> bool:
    sectors = tuple(
        sector
        for sector_id in group.sector_ids
        if (sector := color_plan.sector(sector_id)) is not None
    )
    if len(sectors) != len(group.sector_ids):
        return False
    if any(sector.kind != "open-lines" for sector in sectors):
        return False
    initial_labels = {leg.label for leg in color_plan.process.initial_legs}
    if not initial_labels:
        return False
    for permutation in group.label_permutations:
        mapping = {
            int(representative_label): int(sector_label)
            for representative_label, sector_label in permutation
        }
        if {mapping.get(label) for label in initial_labels} != initial_labels:
            return False
    return True
