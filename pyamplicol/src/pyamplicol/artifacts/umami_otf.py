# SPDX-License-Identifier: 0BSD
"""Explicit SDK selector tables from a compact on-the-fly source seed.

This does not construct recurrence schedules or discover amplitude relations.
The expanded tables are SDK payloads only; the runtime seed remains compact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import product
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..generation.on_the_fly_seed import OnTheFlyProcessSeedProjectionV1
    from ..processes.ir import CanonicalProcessIR
    from .umami import UmamiProcessInput


def build_umami_on_the_fly_physics(
    public_metadata: Mapping[str, Any],
    process: CanonicalProcessIR,
    seed: OnTheFlyProcessSeedProjectionV1,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    """Enumerate available axes, conservatively without additional proofs."""
    from ..color import build_color_plan

    sources = {source.public_label: source for source in seed.external_sources}
    # Match the native selector's seed-domain order exactly. Its mixed-radix
    # enumeration, like product(), varies the last public leg fastest.
    domains = [
        tuple(state.public_helicity for state in sources[leg.label].source_states)
        for leg in process.legs
    ]
    helicities = []
    for index, values in enumerate(product(*domains)):
        identifier = "h:" + ",".join(f"{value:+d}" for value in values)
        helicities.append(
            {
                "id": identifier,
                "index": index,
                "values": list(values),
                "computed": True,
                "representative_id": identifier,
                "coefficient": 1.0,
                "structural_zero": False,
            }
        )
    sectors: tuple[Mapping[str, Any], ...] = ()
    if process.color_accuracy == "lc":
        plan = build_color_plan(process, color_accuracy="lc")
        sectors = tuple(sector.to_json_dict() for sector in plan.sectors)
        colors = []
        for index, sector in enumerate(plan.sectors):
            word = tuple(sector.word_labels or sector.color_words[0])
            identifier = "flow:" + (
                ",".join(str(label) for label in word) if word else "singlet"
            )
            colors.append(
                {
                    "kind": "lc-flow",
                    "id": identifier,
                    "index": index,
                    "word": list(word),
                    "computed": True,
                    "representative_id": identifier,
                    "coefficient": 1.0,
                }
            )
    else:
        colors = [{"kind": "contracted-color", "id": "color:contracted", "index": 0}]
    return (
        {
            **public_metadata,
            "kind": "pyamplicol-resolved-physics",
            "helicities": helicities,
            "color_components": colors,
            "coverage": {
                "helicities": "complete",
                "color": "complete" if process.color_accuracy == "lc" else "contracted",
            },
            "extensions": {},
        },
        sectors,
    )


def restore_umami_on_the_fly_input(
    process: UmamiProcessInput, documents: Sequence[Mapping[str, Any]]
) -> UmamiProcessInput:
    """Recover an old compact process's explicit SDK axes during append.

    The SDK retains every physical selector and its colour tags in its member
    records, even when integration orbits have been compressed. Recover those
    records, not just the exported representatives or their multiplicities.
    Another provider may also load this runtime for flavour-blind members only.
    """
    matches = [
        (document, runtime)
        for document in documents
        for runtime in document["runtime_processes"]
        if runtime["id"] == process.process_id
        and any(
            member["process_id"] == process.process_id
            for channel in document["channels"]
            for entry in channel["processes"]
            for member in entry["members"]
        )
    ]
    if len(matches) != 1:
        raise ValueError("UMAMI append requires the previous OTF process metadata")
    document, runtime = matches[0]
    if (
        tuple(runtime["pdgs"]) != process.external_pdgs
        or runtime["color_accuracy"] != process.color_accuracy
    ):
        raise ValueError("previous UMAMI OTF process identity disagrees with append")
    helicity_values = {
        "h:" + ",".join(f"{value:+d}" for value in values): list(values)
        for values in document["helicities"]
    }
    helicities = [
        {
            "id": identifier,
            "index": index,
            "values": helicity_values[identifier],
            "computed": True,
            "representative_id": identifier,
            "coefficient": 1.0,
            "structural_zero": False,
        }
        for index, identifier in enumerate(runtime["helicity_ids"])
    ]
    colors = []
    flows = {}
    for channel in document["channels"]:
        for entry in channel["processes"]:
            for member in entry["members"]:
                if member["process_id"] != process.process_id:
                    continue
                identifier = member["color_id"]
                if process.color_accuracy != "lc":
                    colors.append(
                        {
                            "kind": "contracted-color",
                            "id": "color:contracted",
                            "index": 0,
                        }
                    )
                    continue
                if not isinstance(identifier, str) or not identifier.startswith(
                    "flow:"
                ):
                    raise ValueError("previous UMAMI OTF colour selector is invalid")
                suffix = identifier.removeprefix("flow:")
                word = (
                    [] if suffix == "singlet" else [int(v) for v in suffix.split(",")]
                )
                tags = document["color_flows"][member["color_flows"]]
                if len(tags) != len(process.external_pdgs) or any(
                    len(pair) != 2 or any(not isinstance(v, int) or v < 0 for v in pair)
                    for pair in tags
                ):
                    raise ValueError("previous UMAMI OTF colour tags are invalid")
                flows[identifier] = tags
                colors.append(
                    {
                        "kind": "lc-flow",
                        "id": identifier,
                        "index": len(colors),
                        "word": word,
                        "computed": True,
                        "representative_id": identifier,
                        "coefficient": 1.0,
                    }
                )
    if not helicities or not colors or len({c["id"] for c in colors}) != len(colors):
        raise ValueError("previous UMAMI OTF selectors are empty or repeated")
    return replace(
        process,
        physics={
            **process.physics,
            "kind": "pyamplicol-resolved-physics",
            "helicities": helicities,
            "color_components": colors,
            "coverage": dict(runtime["coverage"]),
            "extensions": {},
        },
        color_flows=flows,
    )
