# SPDX-License-Identifier: 0BSD
"""UMAMI providers of colour-accuracy ratios between two artifacts.

The ratio is the target matrix element (e.g. full colour) over the leading-colour
reference summed over all colour flows, both at the helicity of the event. It
reweights events of a leading-colour UMAMI provider generated from the same
processes, with the channel, flavour and helicity indices of that provider.
"""

from __future__ import annotations

import importlib.resources
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .umami_c import _boolean, _string

_TEMPLATES = {
    "umami_reweight.c": "umami_reweight.c",
    "reweight_provider.h": "reweight_provider.h",
    "umami.h": "umami.h",
    "umami.exports": "umami.exports",
    "umami.exports.macos": "umami.exports.macos",
    "UMAMI_LICENSE": "UMAMI_LICENSE",
    "Makefile.reweight": "Makefile",
}


_IDENTIFIER = re.compile(r"[A-Za-z0-9_]+")


def _identifier(value: object) -> str:
    # Provider ids end up in file names and in the makefile.
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid UMAMI provider id {value!r}")
    return value


def _integer(value: object) -> int:
    # Metadata values are written into generated C sources.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected an integer in the UMAMI metadata, got {value!r}")
    return value


def load_umami_metadata(artifact: Path) -> list[dict[str, Any]]:
    """The metadata documents of all UMAMI providers of an artifact."""
    root = (Path(artifact) / "API" / "umami").resolve()
    index = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if index.get("kind") != "pyamplicol-umami-provider-index":
        return [index]
    documents = []
    for entry in index["providers"]:
        path = (root / entry["metadata"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"UMAMI metadata path {entry['metadata']!r} leaves {root}")
        documents.append(json.loads(path.read_text(encoding="utf-8")))
    return documents


def _process_row(process: Mapping[str, Any], prefix: str, lines: list[str]) -> str:
    lines.append(
        f"static const int {prefix}_pdgs[] = {{"
        + ", ".join(str(_integer(v)) for v in process["pdgs"])
        + "};"
    )
    return (
        "    {"
        + ", ".join(
            (
                _string(process["id"]),
                f"{prefix}_pdgs",
                _string(process["alpha_s_parameter"]),
                str(len(process["helicity_ids"])),
            )
        )
        + "},"
    )


def _int_array(name: str, ctype: str, values: Sequence[int]) -> str:
    return (
        f"static const {ctype} {name}[] = {{"
        + ", ".join(str(_integer(v)) for v in values or [-1])
        + "};"
    )


def reweight_header(
    reference: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    reference_path: str,
    target_path: str,
) -> str:
    """Render the tables of one reweight provider, see ``reweight_provider.h``."""
    target_processes: list[Mapping[str, Any]] = []
    target_ids: list[str | None] = []
    target_lookup = {}
    for document in targets:
        for process in document["runtime_processes"]:
            target_lookup.setdefault(
                tuple(process["pdgs"]), (process, document["provider"]["artifact_id"])
            )
    reference_processes = reference["runtime_processes"]
    lines = [
        "/* SPDX-License-Identifier: 0BSD */",
        "/* Generated from the UMAMI metadata of the two artifacts. */",
        '#include "reweight_provider.h"',
    ]
    rows = [
        _process_row(process, f"reweight_r{i}", lines)
        for i, process in enumerate(reference_processes)
    ]
    lines.extend(["static const ReweightProcess reweight_references[] = {", *rows, "};"])

    exported_count = len(reference["helicities"])
    channel_rows = []
    for c, channel in enumerate(reference["channels"]):
        flavour_rows = []
        for f, entry in enumerate(channel["processes"]):
            runtime = entry["runtime"]
            if "helicity_export" not in runtime:
                raise ValueError("the reference UMAMI metadata has no helicity export")
            process_index = _integer(runtime["process_index"])
            process = reference_processes[process_index]
            match = target_lookup.get(tuple(process["pdgs"]))
            if match is None:
                raise ValueError(
                    f"no process of the target artifact matches {process['id']!r}"
                )
            target, target_id = match
            if target not in target_processes:
                target_processes.append(target)
                target_ids.append(target_id)
            target_index = target_processes.index(target)
            reference_helicity = [-1] * exported_count
            target_helicity = [-1] * exported_count
            for runtime_index, exported in enumerate(runtime["helicity_export"]):
                reference_helicity[exported] = runtime_index
                identifier = process["helicity_ids"][runtime_index]
                if identifier in target["helicity_ids"]:
                    target_helicity[exported] = target["helicity_ids"].index(identifier)
            prefix = f"reweight_c{c}_f{f}"
            lines.append(
                _int_array(f"{prefix}_permutation", "size_t", runtime["permutation"])
            )
            lines.append(_int_array(f"{prefix}_reference", "int", reference_helicity))
            lines.append(_int_array(f"{prefix}_target", "int", target_helicity))
            flavour_rows.append(
                f"    {{{process_index}, {target_index}, "
                f"{prefix}_permutation, {prefix}_reference, {prefix}_target}},"
            )
        lines.extend(
            [
                f"static const ReweightFlavour reweight_c{c}_flavours[] = {{",
                *flavour_rows,
                "};",
            ]
        )
        channel_rows.append(f"    {{{len(flavour_rows)}, reweight_c{c}_flavours}},")
    if len(set(target_ids)) > 1:
        raise ValueError("the target processes come from different artifacts")
    rows = [
        _process_row(process, f"reweight_t{i}", lines)
        for i, process in enumerate(target_processes)
    ]
    lines.extend(["static const ReweightProcess reweight_targets[] = {", *rows, "};"])
    lines.extend(
        ["static const ReweightChannel reweight_channels[] = {", *channel_rows, "};"]
    )
    supports_alpha_s = all(
        p["alpha_s_parameter"] is not None
        for p in [*reference_processes, *target_processes]
    )
    lines.extend(
        [
            f"#define UMAMI_HAS_ALPHA_S {int(supports_alpha_s)}",
            "static const ReweightProvider reweight_provider = {",
            f"    {_string(reference['provider']['id'])}, "
            f"{_integer(reference['provider']['particle_count'])}, {exported_count},",
            f"    {_string(reference_path)}, {_string(target_path)},",
            f"    {_string(reference['provider'].get('artifact_id'))}, "
            f"{_string(target_ids[0] if target_ids else None)},",
            f"    {len(reference_processes)}, reweight_references,",
            f"    {len(target_processes)}, reweight_targets,",
            f"    {len(channel_rows)}, reweight_channels, {_boolean(supports_alpha_s)}",
            "};",
            "",
        ]
    )
    return "\n".join(lines)


def write_reweight_provider(
    reference_artifact: str | os.PathLike[str],
    target_artifact: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> list[str]:
    """Write the sources of the reweight providers to ``output``.

    One provider is written per UMAMI provider of the leading-colour reference
    artifact; ``make -C output`` builds ``build/libreweight_<id>.so``. The
    libraries are initialized with the absolute path of ``output``, the artifacts
    are found relative to it. Returns the provider ids.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    references = load_umami_metadata(Path(reference_artifact))
    targets = load_umami_metadata(Path(target_artifact))
    if any(d["provider"]["color_accuracy"] != "lc" for d in references):
        raise ValueError("the reference artifact must have leading-colour accuracy")
    reference_path = os.path.relpath(Path(reference_artifact).resolve(), output.resolve())
    target_path = os.path.relpath(Path(target_artifact).resolve(), output.resolve())
    templates = importlib.resources.files("pyamplicol.assets.api_templates") / "umami"
    for source, destination in _TEMPLATES.items():
        (output / destination).write_bytes((templates / source).read_bytes())
    ids = []
    for reference in references:
        identifier = _identifier(reference["provider"]["id"])
        (output / f"reweight_{identifier}.h").write_text(
            reweight_header(reference, targets, reference_path, target_path),
            encoding="utf-8",
        )
        ids.append(identifier)
    (output / "providers.mk").write_text(
        "UMAMI_PROVIDERS := " + " ".join(ids) + "\n", encoding="utf-8"
    )
    return ids


__all__ = ["load_umami_metadata", "reweight_header", "write_reweight_provider"]
