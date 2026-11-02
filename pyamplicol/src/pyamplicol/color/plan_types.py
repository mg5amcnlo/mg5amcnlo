# SPDX-License-Identifier: 0BSD
"""Frozen color-sector and color-plan records."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from itertools import permutations
from typing import Literal

from ..processes.ir import CanonicalProcessIR

ColorAccuracy = Literal["lc", "nlc", "full"]
ColorSectorKind = Literal["singlet", "open-lines", "single-trace"]


def _canonical_open_string_product_key(
    open_strings: Iterable[tuple[int, Iterable[int], int, Iterable[int]]],
) -> tuple[tuple[int, tuple[int, ...], int, tuple[int, ...]], ...]:
    """Return the unordered identity of a product of complete open strings."""

    return tuple(
        sorted(
            (
                int(fundamental),
                tuple(int(label) for label in adjoints),
                int(antifundamental),
                tuple(int(label) for label in singlets),
            )
            for fundamental, adjoints, antifundamental, singlets in open_strings
        )
    )


@dataclass(frozen=True)
class LCOpenColorLine:
    """One leading-colour open line in all-outgoing conventions."""

    fundamental_label: int
    antifundamental_label: int
    adjoint_labels: tuple[int, ...]
    singlet_labels: tuple[int, ...] = ()

    @property
    def coloured_labels(self) -> tuple[int, ...]:
        return (
            self.fundamental_label,
            *self.adjoint_labels,
            self.antifundamental_label,
        )

    @property
    def line_labels(self) -> tuple[int, ...]:
        return (*self.coloured_labels, *self.singlet_labels)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "fundamental_label": self.fundamental_label,
            "antifundamental_label": self.antifundamental_label,
            "adjoint_labels": list(self.adjoint_labels),
            "singlet_labels": list(self.singlet_labels),
            "line_labels": list(self.line_labels),
        }


@dataclass(frozen=True)
class LCColorSector:
    """One colour-flow sector produced during generation warmup."""

    id: int
    kind: ColorSectorKind
    open_color_lines: tuple[LCOpenColorLine, ...] = ()
    trace_labels: tuple[int, ...] = ()
    singlet_labels: tuple[int, ...] = ()
    word_labels: tuple[int, ...] = ()

    @property
    def coloured_label_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.kind == "open-lines":
            return tuple(line.coloured_labels for line in self.open_color_lines)
        if self.kind == "single-trace":
            return (self.trace_labels,)
        return ()

    @property
    def line_label_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.kind == "open-lines":
            return tuple(line.coloured_labels for line in self.open_color_lines)
        if self.kind == "single-trace":
            return ((*self.trace_labels, *self.singlet_labels),)
        if self.kind == "singlet":
            return (self.singlet_labels,)
        return ()

    @cached_property
    def color_words(self) -> tuple[tuple[int, ...], ...]:
        if self.word_labels:
            return (self.word_labels,)
        if self.kind == "open-lines":
            return (
                tuple(
                    label
                    for line in self.open_color_lines
                    for label in line.coloured_labels
                ),
            )
        if self.kind == "single-trace":
            return (self.trace_labels,)
        if self.kind == "singlet":
            return ((),)
        return ()

    @cached_property
    def admissible_traversal_words(self) -> tuple[tuple[int, ...], ...]:
        """Colour words accepted while constructing currents for this sector.

        The sector itself has one physical colour word.  During current
        construction, however, complete open-line blocks can be traversed
        in different intermediate orders.  This lets the generic recursion
        reproduce AmpliCol's colour-ordered current closures without naming a
        process family.
        """

        from .plan_build import _ordered_open_line_blocks

        if self.kind != "open-lines" or len(self.open_color_lines) < 2:
            return self.color_words
        primary = self.color_words[0]
        blocks = _ordered_open_line_blocks(primary, self.open_color_lines)
        if blocks is None or len(blocks) < 2:
            return self.color_words
        words: list[tuple[int, ...]] = [primary]
        seen = {primary}
        for block_permutation in permutations(blocks):
            word = tuple(label for block in block_permutation for label in block)
            if word in seen:
                continue
            seen.add(word)
            words.append(word)
        return tuple(words)

    def canonical_closure_traversal_word(
        self,
        process: CanonicalProcessIR,
    ) -> tuple[int, ...]:
        """Return one deterministic block-equivalent traversal for closure.

        Intermediate currents may use every admissible complete-open-line
        block permutation, but the final amplitude must close on one cyclic
        traversal.  AmpliCol's representation convention starts with the
        complete open-line block containing the minimum canonical coloured
        source slot, preserves the public cyclic block order, and closes on the
        terminal endpoint of the preceding block.  Colour-singlet sources are
        skipped when selecting that slot.  This uses only source ordering, the
        physical colour word, and declared open-line blocks; particle flavour,
        mass, process identity, and multiplicity do not enter.
        """

        blocks = self._canonical_closure_blocks(process)
        return tuple(label for block in blocks for label in block)

    def _canonical_closure_blocks(
        self,
        process: CanonicalProcessIR,
    ) -> tuple[tuple[int, ...], ...]:
        """Reconstruct open-line blocks with the canonical source block first."""

        primary = self.color_words[0]
        if self.kind != "open-lines" or not self.open_color_lines:
            return (primary,)
        from .plan_build import _ordered_open_line_blocks

        blocks = _ordered_open_line_blocks(primary, self.open_color_lines)
        if blocks is None or len(blocks) != len(self.open_color_lines):
            raise ValueError(
                f"LC sector {self.id} colour word is not a complete open-line "
                "block traversal"
            )
        known_labels = {int(leg.label) for leg in process.legs}
        coloured_labels = tuple(label for block in blocks for label in block)
        unknown_labels = sorted(set(coloured_labels) - known_labels)
        if unknown_labels:
            raise ValueError(
                f"LC sector {self.id} closure line references unknown external "
                f"labels {unknown_labels}"
            )
        if not coloured_labels or not process.legs:
            return blocks
        coloured_label_set = set(coloured_labels)
        distinguished_label = next(
            (
                int(leg.label)
                for leg in process.legs
                if int(leg.label) in coloured_label_set
            ),
            None,
        )
        if distinguished_label is None:
            return blocks
        positions = tuple(
            index for index, block in enumerate(blocks) if distinguished_label in block
        )
        if not positions:
            return blocks
        if len(positions) != 1:
            raise ValueError(
                f"LC sector {self.id} distinguished source label "
                f"{distinguished_label} belongs to {len(positions)} open-line blocks"
            )
        first_position = positions[0]
        return (*blocks[first_position:], *blocks[:first_position])

    def canonical_closure_sink_label(self, process: CanonicalProcessIR) -> int | None:
        """Return the terminal label of the canonical closure traversal."""

        word = self.canonical_closure_traversal_word(process)
        return None if not word else int(word[-1])

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "open_color_lines": [line.to_json_dict() for line in self.open_color_lines],
            "trace_labels": list(self.trace_labels),
            "singlet_labels": list(self.singlet_labels),
            "word_labels": list(self.word_labels),
            "coloured_label_groups": [
                list(group) for group in self.coloured_label_groups
            ],
            "line_label_groups": [list(group) for group in self.line_label_groups],
            "color_words": [list(word) for word in self.color_words],
            "admissible_traversal_words": [
                list(word) for word in self.admissible_traversal_words
            ],
        }


@dataclass(frozen=True)
class LCColorSectorTopologyGroup:
    """Colour sectors that can share one compiled current topology.

    The group key is built from model/process data carried by the external
    labels, not from a process-family name.  Runtime reuse still evaluates each
    sector with its own external-label permutation, so this is a generation-time
    sharing plan rather than a physics approximation.
    """

    signature: tuple[object, ...]
    representative_sector_id: int
    sector_ids: tuple[int, ...]
    label_permutations: tuple[tuple[tuple[int, int], ...], ...]

    def to_json_dict(self) -> dict[str, object]:
        from .plan_replay import _jsonable_signature

        return {
            "signature": _jsonable_signature(self.signature),
            "representative_sector_id": self.representative_sector_id,
            "sector_ids": list(self.sector_ids),
            "label_permutations": [
                [[left, right] for left, right in permutation]
                for permutation in self.label_permutations
            ],
        }


@dataclass(frozen=True)
class LCColorSectorReplayPartition:
    """Initial-label-safe replay block inside one LC topology group."""

    representative_sector_id: int
    active_sector_ids: tuple[int, ...]
    label_permutations: tuple[tuple[tuple[int, int], ...], ...]
    replay_weights: tuple[float, ...] = ()
    replay_signs: tuple[int, ...] = ()
    materialized_sector_id: int | None = None
    proof_algorithm: str | None = None
    proof_digest: str | None = None

    def __post_init__(self) -> None:
        if self.materialized_sector_id is None:
            object.__setattr__(
                self,
                "materialized_sector_id",
                int(self.representative_sector_id),
            )
        count = len(self.active_sector_ids)
        if len(self.label_permutations) != count:
            raise ValueError(
                "LC replay partition permutations do not match active sectors"
            )
        if self.replay_weights and len(self.replay_weights) != count:
            raise ValueError("LC replay partition weights do not match active sectors")
        if self.replay_signs and len(self.replay_signs) != count:
            raise ValueError("LC replay partition signs do not match active sectors")
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in self.weights):
            raise ValueError("LC replay partition weights must be positive and finite")
        if any(sign not in {-1, 1} for sign in self.signs):
            raise ValueError("LC replay partition signs must be -1 or 1")
        if self.representative_sector_id not in self.active_sector_ids:
            raise ValueError("LC replay partition does not contain its representative")
        if self.materialized_sector_id != self.representative_sector_id:
            raise ValueError(
                "LC replay currently materializes the partition representative"
            )
        if (self.proof_algorithm is None) != (self.proof_digest is None):
            raise ValueError(
                "LC replay proof algorithm and digest must be present together"
            )
        if self.proof_digest is not None and (
            len(self.proof_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in self.proof_digest
            )
        ):
            raise ValueError("LC replay proof digest must be a lowercase SHA-256")

    @property
    def weights(self) -> tuple[float, ...]:
        return self.replay_weights or (1.0,) * len(self.active_sector_ids)

    @property
    def signs(self) -> tuple[int, ...]:
        return self.replay_signs or (1,) * len(self.active_sector_ids)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "representative_sector_id": self.representative_sector_id,
            "materialized_sector_id": self.materialized_sector_id,
            "active_sector_ids": list(self.active_sector_ids),
            "label_permutations": [
                [[left, right] for left, right in permutation]
                for permutation in self.label_permutations
            ],
            "replay_weights": list(self.weights),
            "replay_signs": list(self.signs),
            "proof": (
                None
                if self.proof_digest is None
                else {
                    "status": "proven",
                    "algorithm": self.proof_algorithm,
                    "digest": self.proof_digest,
                }
            ),
        }

    def to_runtime_manifest(self) -> dict[str, object]:
        """Return the additive compiled-runtime replay group contract."""

        return {
            "representative_sector_id": self.representative_sector_id,
            "materialized_sector_id": self.materialized_sector_id,
            "active_sector_ids": list(self.active_sector_ids),
            "proof": {
                "status": "proven",
                "algorithm": self.proof_algorithm,
                "digest": self.proof_digest,
            },
            "sector_permutations": [
                {
                    "sector_id": sector_id,
                    "weight": weight,
                    "sign": sign,
                    "factor": [weight * sign, 0.0],
                    "label_permutation": [
                        {
                            "representative_label": representative_label,
                            "sector_label": sector_label,
                        }
                        for representative_label, sector_label in permutation
                    ],
                }
                for sector_id, permutation, weight, sign in zip(
                    self.active_sector_ids,
                    self.label_permutations,
                    self.weights,
                    self.signs,
                    strict=True,
                )
            ],
        }


@dataclass(frozen=True)
class ColorTopologyReplayCertificate:
    """Color-accuracy-bound proof for external-label topology replay.

    This record is deliberately runtime-neutral.  Recurrence, compiled, and
    eager lowering may consume the same proven partitions, but each mode must
    separately authenticate how representative amplitudes enter its color
    reduction.
    """

    color_accuracy: ColorAccuracy
    physical_sector_ids: tuple[int, ...]
    partitions: tuple[LCColorSectorReplayPartition, ...]
    residual_sector_ids: tuple[int, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.color_accuracy not in {"lc", "nlc", "full"}:
            raise ValueError(
                f"unsupported topology replay accuracy {self.color_accuracy!r}"
            )
        physical = tuple(sorted(int(value) for value in self.physical_sector_ids))
        residual = tuple(sorted(int(value) for value in self.residual_sector_ids))
        object.__setattr__(self, "physical_sector_ids", physical)
        object.__setattr__(self, "residual_sector_ids", residual)
        if len(set(physical)) != len(physical):
            raise ValueError("topology replay physical sectors contain duplicates")
        if len(set(residual)) != len(residual):
            raise ValueError("topology replay residual sectors contain duplicates")
        if any(
            partition.proof_algorithm is None or partition.proof_digest is None
            for partition in self.partitions
        ):
            raise ValueError(
                "topology replay certificate contains an unproven partition"
            )
        replayed = tuple(
            sector_id
            for partition in self.partitions
            for sector_id in partition.active_sector_ids
        )
        if len(set(replayed)) != len(replayed):
            raise ValueError("topology replay certificate partitions overlap")
        if set(replayed) & set(residual):
            raise ValueError("topology replay sectors overlap residual coverage")
        if set(replayed) | set(residual) != set(physical):
            raise ValueError(
                "topology replay certificate does not cover every physical sector"
            )

    @property
    def replayed_sector_count(self) -> int:
        return sum(len(partition.active_sector_ids) for partition in self.partitions)

    @property
    def materialized_sector_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    *(
                        int(partition.materialized_sector_id)
                        for partition in self.partitions
                    ),
                    *self.residual_sector_ids,
                }
            )
        )

    @property
    def optimized(self) -> bool:
        return any(
            len(partition.active_sector_ids) > 1 for partition in self.partitions
        )

    def representative_for(self, sector_id: int) -> int:
        sector_id = int(sector_id)
        for partition in self.partitions:
            if sector_id in partition.active_sector_ids:
                return int(partition.materialized_sector_id)
        if sector_id in self.residual_sector_ids:
            return sector_id
        raise KeyError(f"topology replay certificate has no sector {sector_id}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "pyamplicol-color-topology-replay-certificate",
            "mode": "external-label-permutation",
            "color_accuracy": self.color_accuracy,
            "physical_sector_ids": list(self.physical_sector_ids),
            "materialized_sector_ids": list(self.materialized_sector_ids),
            "residual_sector_ids": list(self.residual_sector_ids),
            "replayed_sector_count": self.replayed_sector_count,
            "partitions": [partition.to_json_dict() for partition in self.partitions],
            "diagnostics": list(self.diagnostics),
        }

    def to_runtime_manifest(self) -> dict[str, object]:
        """Return the color-generic representative-lane runtime contract."""

        return {
            "enabled": self.optimized,
            "mode": "external-label-permutation",
            "contract_version": 3,
            "color_accuracy": self.color_accuracy,
            "physical_sector_count": len(self.physical_sector_ids),
            "replayed_sector_count": self.replayed_sector_count,
            "materialized_sector_ids": list(self.materialized_sector_ids),
            "residual_sector_ids": list(self.residual_sector_ids),
            "groups": [
                partition.to_runtime_manifest() for partition in self.partitions
            ],
        }


@dataclass(frozen=True)
class LCColorTopologyReplayPlan:
    """Proof-gated replay classes plus independently materialized residuals."""

    physical_sector_ids: tuple[int, ...]
    partitions: tuple[LCColorSectorReplayPartition, ...]
    residual_sector_ids: tuple[int, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        physical = tuple(sorted(int(value) for value in self.physical_sector_ids))
        residual = tuple(sorted(int(value) for value in self.residual_sector_ids))
        object.__setattr__(self, "physical_sector_ids", physical)
        object.__setattr__(self, "residual_sector_ids", residual)
        if len(set(physical)) != len(physical):
            raise ValueError("LC replay physical sectors contain duplicates")
        if len(set(residual)) != len(residual):
            raise ValueError("LC replay residual sectors contain duplicates")
        if any(
            partition.proof_algorithm is None or partition.proof_digest is None
            for partition in self.partitions
        ):
            raise ValueError("LC replay plan contains an unproven partition")
        replayed = tuple(
            sector_id
            for partition in self.partitions
            for sector_id in partition.active_sector_ids
        )
        if len(set(replayed)) != len(replayed):
            raise ValueError("LC replay partitions overlap")
        if set(replayed) & set(residual):
            raise ValueError("LC replay sectors overlap residual coverage")
        if set(replayed) | set(residual) != set(physical):
            raise ValueError("LC replay plan does not cover every physical sector")

    @property
    def replayed_sector_count(self) -> int:
        return sum(len(partition.active_sector_ids) for partition in self.partitions)

    @property
    def materialized_sector_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    *(
                        int(partition.materialized_sector_id)
                        for partition in self.partitions
                    ),
                    *self.residual_sector_ids,
                }
            )
        )

    @property
    def optimized(self) -> bool:
        return any(
            len(partition.active_sector_ids) > 1 for partition in self.partitions
        )

    def representative_for(self, sector_id: int) -> int:
        sector_id = int(sector_id)
        for partition in self.partitions:
            if sector_id in partition.active_sector_ids:
                return int(partition.materialized_sector_id)
        if sector_id in self.residual_sector_ids:
            return sector_id
        raise KeyError(f"LC replay plan has no physical sector {sector_id}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "pyamplicol-lc-topology-replay-plan",
            "mode": "external-label-permutation",
            "physical_sector_ids": list(self.physical_sector_ids),
            "materialized_sector_ids": list(self.materialized_sector_ids),
            "residual_sector_ids": list(self.residual_sector_ids),
            "replayed_sector_count": self.replayed_sector_count,
            "partitions": [partition.to_json_dict() for partition in self.partitions],
            "diagnostics": list(self.diagnostics),
        }

    def to_runtime_manifest(self) -> dict[str, object]:
        return {
            "enabled": self.optimized,
            "mode": "external-label-permutation",
            "contract_version": 2,
            "physical_sector_count": len(self.physical_sector_ids),
            "replayed_sector_count": self.replayed_sector_count,
            "materialized_sector_ids": list(self.materialized_sector_ids),
            "residual_sector_ids": list(self.residual_sector_ids),
            "groups": [
                partition.to_runtime_manifest() for partition in self.partitions
            ],
        }


@dataclass(frozen=True)
class GenericColorPlan:
    """Colour-flow planning payload shared by future Python/Rust runtimes."""

    process: CanonicalProcessIR
    color_accuracy: str
    sectors: tuple[LCColorSector, ...]
    diagnostics: tuple[str, ...] = ()
    truncated: bool = False
    trace_reflections_folded: bool = False
    # The ordered words select primitive amplitudes in either basis.  Only the
    # associated colour tensors/metric differ; adjoint means DDM half-ladders.
    basis: Literal["trace", "adjoint"] = "trace"

    @property
    def sector_count(self) -> int:
        return len(self.sectors)

    @property
    def ready_for_leading_colour(self) -> bool:
        return self.color_accuracy == "lc" and bool(self.sectors) and not self.truncated

    @property
    def ready_for_requested_colour(self) -> bool:
        return bool(self.sectors) and not self.truncated

    @property
    def coloured_labels(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    *self.process.fundamental_labels,
                    *self.process.antifundamental_labels,
                    *self.process.adjoint_labels,
                }
            )
        )

    @cached_property
    def _sectors_by_id(self) -> dict[int, LCColorSector]:
        return {int(sector.id): sector for sector in self.sectors}

    def sector(self, color_sector: int) -> LCColorSector | None:
        return self._sectors_by_id.get(int(color_sector))

    @cached_property
    def topology_groups(self) -> tuple[LCColorSectorTopologyGroup, ...]:
        from .plan_replay import _sector_topology_groups

        return _sector_topology_groups(self.process, self.sectors)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "process": self.process.to_json_dict(),
            "color_accuracy": self.color_accuracy,
            **({"basis": self.basis} if self.basis != "trace" else {}),
            "sector_count": self.sector_count,
            "truncated": self.truncated,
            "trace_reflections_folded": self.trace_reflections_folded,
            "ready_for_leading_colour": self.ready_for_leading_colour,
            "ready_for_requested_colour": self.ready_for_requested_colour,
            "coloured_labels": list(self.coloured_labels),
            "diagnostics": list(self.diagnostics),
            "sectors": [sector.to_json_dict() for sector in self.sectors],
            "topology_groups": [group.to_json_dict() for group in self.topology_groups],
        }
