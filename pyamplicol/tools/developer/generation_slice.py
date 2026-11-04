#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Developer-only generation of deliberately specialized process artifacts.

Production generation always targets complete physical coverage. Performance
campaigns occasionally need an artifact specialized to one LC flow or one
source-helicity configuration so it can be compared with an equally
specialized reference implementation. This module keeps those unstable
topology controls out of the public TOML, CLI, and Python API.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyamplicol as _pyamplicol
from pyamplicol.api.requests import (
    ModelSource,
    ProcessRequest,
    ProcessSet,
)
from pyamplicol.api.results import GenerationResult
from pyamplicol.api.services import _generation_resource_resolution
from pyamplicol.config import ConfigResolution, GenerationConfig, RunConfig
from pyamplicol.generation.service import GenerationBackend, _ProcessSelection
from pyamplicol.reporting import ProgressSink


def _positive_unique(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in result
    ):
        raise ValueError(f"{name} must contain positive integer labels")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique labels")
    return result


def _nonnegative_unique(values: Iterable[int], name: str) -> frozenset[int]:
    result = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in result
    ):
        raise ValueError(f"{name} must contain non-negative integer IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique IDs")
    return frozenset(result)


@dataclass(frozen=True, slots=True)
class GenerationSlice:
    """Unstable developer selection for one specialized generation run."""

    reference_color_order: tuple[int, ...] = ()
    selected_color_sector_ids: tuple[int, ...] = ()
    selected_source_helicities: Mapping[int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_color_order",
            _positive_unique(
                self.reference_color_order,
                "reference_color_order",
            ),
        )
        selected_ids = _nonnegative_unique(
            self.selected_color_sector_ids,
            "selected_color_sector_ids",
        )
        object.__setattr__(
            self,
            "selected_color_sector_ids",
            tuple(sorted(selected_ids)),
        )
        raw_helicities = self.selected_source_helicities or {}
        normalized_helicities: dict[int, int] = {}
        for label, helicity in raw_helicities.items():
            if isinstance(label, bool) or not isinstance(label, int) or label < 1:
                raise ValueError(
                    "selected_source_helicities keys must be positive integer labels"
                )
            if isinstance(helicity, bool) or not isinstance(helicity, int):
                raise ValueError(
                    "selected_source_helicities values must be integer helicities"
                )
            normalized_helicities[label] = helicity
        object.__setattr__(
            self,
            "selected_source_helicities",
            normalized_helicities or None,
        )

    def _selection(self) -> _ProcessSelection:
        return _ProcessSelection(
            reference_color_order=self.reference_color_order or None,
            selected_color_sector_ids=(
                frozenset(self.selected_color_sector_ids)
                if self.selected_color_sector_ids
                else None
            ),
            selected_source_helicities=self.selected_source_helicities,
        )


def _process_set(
    processes: ProcessSet | ProcessRequest | str | Iterable[ProcessRequest | str],
) -> ProcessSet:
    if isinstance(processes, ProcessSet):
        return processes
    if isinstance(processes, ProcessRequest):
        return ProcessSet((processes,))
    if isinstance(processes, str):
        return ProcessSet((ProcessRequest.parse(processes),))
    return ProcessSet(
        tuple(
            process
            if isinstance(process, ProcessRequest)
            else ProcessRequest.parse(process)
            for process in processes
        )
    )


def generate_slice(
    processes: ProcessSet | ProcessRequest | str | Iterable[ProcessRequest | str],
    output: os.PathLike[str] | str,
    *,
    selection: GenerationSlice,
    model: ModelSource | _pyamplicol.CompiledModel | None = None,
    mode: Literal["error", "append", "replace"] = "error",
    config: GenerationConfig | RunConfig | ConfigResolution | None = None,
    progress: ProgressSink | None = None,
) -> GenerationResult:
    """Generate one benchmark-only artifact with an explicit topology slice."""

    if not isinstance(selection, GenerationSlice):
        raise TypeError("selection must be a GenerationSlice")
    resolved = _generation_resource_resolution(config)
    backend = GenerationBackend(
        resolved,
        progress,
        process_selection=selection._selection(),
    )
    return backend.generate(
        _process_set(processes),
        Path(os.fspath(output)).expanduser().resolve(strict=False),
        model=model,
        mode=mode,
    )


__all__ = ["GenerationSlice", "generate_slice"]
