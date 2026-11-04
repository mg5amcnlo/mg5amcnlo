# SPDX-License-Identifier: 0BSD
"""Small C data emitter shared by generated UMAMI providers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _string(value: object) -> str:
    return "NULL" if value is None else json.dumps(str(value), ensure_ascii=True)


def _number(value: object) -> str:
    return format(float(value), ".17g")


def _boolean(value: object) -> str:
    return "true" if value else "false"


def provider_header(data: Mapping[str, Any]) -> str:
    """Render the private ABI defined by ``umami_provider.h``."""
    provider = data["provider"]
    lines = [
        "/* SPDX-License-Identifier: 0BSD */",
        "/* Generated from the same records as metadata.json. */",
        '#include "umami_provider.h"',
        f"#define UMAMI_HAS_ALPHA_S {int(provider['supports_alpha_s'])}",
        f"#define UMAMI_HAS_COLOR_FLOW {int(provider['color_accuracy'] == 'lc')}",
        f"#define UMAMI_ARTIFACT_ID {_string(provider.get('artifact_id'))}",
        "static const double umami_masses[] = {"
        + ", ".join(_number(v) for v in provider["masses"])
        + "};",
    ]
    process_rows = []
    for index, process in enumerate(data["runtime_processes"]):
        prefix = f"umami_p{index}"
        lines.append(
            f"static const int {prefix}_pdgs[] = {{"
            + ", ".join(str(v) for v in process["pdgs"])
            + "};"
        )
        helicities = process["helicity_ids"]
        parameters = process["parameters"]
        if helicities:
            lines.append(
                f"static const char *const {prefix}_helicities[] = {{"
                + ", ".join(_string(v) for v in helicities)
                + "};"
            )
        if parameters:
            lines.append(f"static const UmamiParameter {prefix}_parameters[] = {{")
            for p in parameters:
                lines.append(
                    "    {"
                    + ", ".join(
                        (
                            _string(p["name"]),
                            _number(p["default_real"]),
                            _number(p.get("default_imaginary", 0.0)),
                            _boolean(p["mutable"]),
                            _boolean(
                                p.get(
                                    "is_complex", p.get("default_imaginary", 0.0) != 0.0
                                )
                            ),
                        )
                    )
                    + "},"
                )
            lines.append("};")
        process_rows.append(
            "    {"
            + ", ".join(
                (
                    _string(process["id"]),
                    f"{prefix}_pdgs",
                    _string(process["alpha_s_parameter"]),
                    _string(process["color_accuracy"]),
                    str(len(helicities)),
                    f"{prefix}_helicities" if helicities else "NULL",
                    str(len(parameters)),
                    f"{prefix}_parameters" if parameters else "NULL",
                )
            )
            + "},"
        )
    lines.extend(
        ["static const UmamiProcess umami_processes[] = {", *process_rows, "};"]
    )
    channel_rows = []
    for channel_index, channel in enumerate(data["channels"]):
        flavour_rows = []
        for flavour_index, flavour in enumerate(channel["processes"]):
            prefix = f"umami_c{channel_index}_f{flavour_index}"
            runtime = flavour["runtime"]
            lines.append(
                f"static const size_t {prefix}_permutation[] = {{"
                + ", ".join(str(v) for v in runtime["permutation"])
                + "};"
            )
            exported = runtime["helicity_export"]
            to_runtime = [-1] * len(data["helicities"])
            for runtime_index, export_index in enumerate(exported):
                to_runtime[export_index] = runtime_index
            lines.append(
                f"static const int {prefix}_helicity_runtime[] = {{"
                + ", ".join(str(v) for v in to_runtime or [-1])
                + "};"
            )
            lines.append(
                f"static const size_t {prefix}_helicity_export[] = {{"
                + ", ".join(str(v) for v in exported or [0])
                + "};"
            )
            lines.append(
                f"static const UmamiContribution {prefix}_contributions[] = {{"
            )
            lines.append(
                "    {"
                + ", ".join(
                    (
                        str(runtime["process_index"]),
                        _string(runtime["color_id"]),
                        str(runtime["color_index"]),
                        f"{prefix}_permutation",
                        _number(runtime["factor"]),
                        f"{prefix}_helicity_runtime",
                        f"{prefix}_helicity_export",
                    )
                )
                + "},"
            )
            lines.append("};")
            flavour_rows.append(f"    {{1, {prefix}_contributions}},")
        lines.extend(
            [
                f"static const UmamiFlavour umami_c{channel_index}_flavours[] = {{",
                *flavour_rows,
                "};",
            ]
        )
        channel_rows.append(
            f"    {{{len(flavour_rows)}, umami_c{channel_index}_flavours}},"
        )
    lines.extend(
        ["static const UmamiChannel umami_channels[] = {", *channel_rows, "};"]
    )
    lines.extend(
        [
            "static const UmamiProvider umami_provider = {",
            "    "
            + ", ".join(
                (
                    _string(provider["id"]),
                    str(provider["particle_count"]),
                    str(provider["incoming_count"]),
                    str(provider["helicity_count"]),
                    str(provider["color_count"]),
                    str(len(data["helicities"])),
                )
            )
            + ",",
            f"    umami_masses, {len(process_rows)}, umami_processes,",
            f"    {len(channel_rows)}, umami_channels, "
            + _boolean(provider["supports_alpha_s"]),
            "};",
            "",
        ]
    )
    return "\n".join(lines)
