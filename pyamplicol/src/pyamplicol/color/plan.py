# SPDX-License-Identifier: 0BSD
"""Compatibility facade for color-sector planning."""

from __future__ import annotations

from .plan_build import build_color_plan
from .plan_replay import (
    build_color_topology_replay_certificate,
    build_lc_topology_replay_plan,
    color_topology_replay_partitions,
    lc_line_pairing_representative_ids,
    lc_topology_replay_partitions,
    lc_topology_replay_safe_groups,
)
from .plan_types import (
    ColorAccuracy,
    ColorSectorKind,
    ColorTopologyReplayCertificate,
    GenericColorPlan,
    LCColorSector,
    LCColorSectorReplayPartition,
    LCColorSectorTopologyGroup,
    LCColorTopologyReplayPlan,
    LCOpenColorLine,
)

__all__ = [
    "ColorAccuracy",
    "ColorSectorKind",
    "ColorTopologyReplayCertificate",
    "GenericColorPlan",
    "LCColorSector",
    "LCColorSectorReplayPartition",
    "LCColorSectorTopologyGroup",
    "LCColorTopologyReplayPlan",
    "LCOpenColorLine",
    "build_color_plan",
    "build_color_topology_replay_certificate",
    "build_lc_topology_replay_plan",
    "color_topology_replay_partitions",
    "lc_line_pairing_representative_ids",
    "lc_topology_replay_partitions",
    "lc_topology_replay_safe_groups",
]
