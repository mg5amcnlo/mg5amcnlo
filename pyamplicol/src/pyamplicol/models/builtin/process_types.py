# SPDX-License-Identifier: 0BSD
"""Built-in-SM process enumeration and selection records.

These records describe the historical built-in process catalogue.  They are
not part of the model-neutral process syntax or external-model API.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...processes.core_syntax import OrderTuple, ProcessTuple


@dataclass(frozen=True)
class BuiltinParsedProcess:
    initial_state: ProcessTuple
    jet_count: int
    rest: ProcessTuple
    leptons: ProcessTuple = ()


@dataclass(frozen=True)
class BuiltinProcessOptions:
    flavour_scheme: int = 5
    include_3qqbar: bool = False
    include_cc: bool = False
    include_resonance: bool = False
    serial: bool = True


@dataclass(frozen=True)
class BuiltinSubprocessRecord:
    process: ProcessTuple
    color_order: OrderTuple
    multichannel_partners: tuple[int, ...] = ()
    identical_factor: float = 1.0


@dataclass(frozen=True)
class BuiltinPhaseSpaceGroup:
    group_id: int
    phase_space_order: OrderTuple
    records: tuple[BuiltinSubprocessRecord, ...]


@dataclass(frozen=True)
class BuiltinProcessEnumeration:
    request: BuiltinParsedProcess
    options: BuiltinProcessOptions
    unique_processes: tuple[ProcessTuple, ...]
    groups: tuple[BuiltinPhaseSpaceGroup, ...]

    @property
    def n_external(self) -> int:
        if self.unique_processes:
            return len(self.unique_processes[0])
        if self.groups and self.groups[0].records:
            return len(self.groups[0].records[0].process)
        return 0

    @property
    def n_records(self) -> int:
        return sum(len(group.records) for group in self.groups)


@dataclass(frozen=True)
class BuiltinProcessSetEntry:
    key: str
    process: str
    enumeration: BuiltinProcessEnumeration


@dataclass(frozen=True)
class BuiltinProcessSelectionRecord:
    source: str
    process: str
    key: str
    status: str
    reason: str
    quark_lines: int
    charge3: int
    family: int


@dataclass(frozen=True)
class BuiltinProcessSelectionReport:
    request: str
    options: BuiltinProcessOptions
    entries: tuple[BuiltinProcessSetEntry, ...]
    records: tuple[BuiltinProcessSelectionRecord, ...]
    candidate_count: int
    evaluated_count: int
    selected_count: int
    duplicate_count: int
    rejected_count: int
    rejection_counts: tuple[tuple[str, int], ...]
    stage_timings: tuple[tuple[str, float], ...]
    elapsed_s: float
    prefilter_enabled: bool

    @property
    def selected_records(self) -> tuple[BuiltinProcessSelectionRecord, ...]:
        return tuple(record for record in self.records if record.status == "selected")

    @property
    def duplicate_records(self) -> tuple[BuiltinProcessSelectionRecord, ...]:
        return tuple(record for record in self.records if record.status == "duplicate")


@dataclass(frozen=True)
class BuiltinProcessSetEnumeration:
    request: str
    options: BuiltinProcessOptions
    entries: tuple[BuiltinProcessSetEntry, ...]
    selection_report: BuiltinProcessSelectionReport | None = None

    @property
    def default_key(self) -> str:
        if not self.entries:
            raise ValueError("process set is empty")
        return self.entries[0].key


__all__ = [
    "BuiltinParsedProcess",
    "BuiltinPhaseSpaceGroup",
    "BuiltinProcessEnumeration",
    "BuiltinProcessOptions",
    "BuiltinProcessSelectionRecord",
    "BuiltinProcessSelectionReport",
    "BuiltinProcessSetEntry",
    "BuiltinProcessSetEnumeration",
    "BuiltinSubprocessRecord",
]
