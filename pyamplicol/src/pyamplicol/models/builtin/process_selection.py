# SPDX-License-Identifier: 0BSD
"""Legacy built-in-SM process-set selection and compatibility reporting."""

from __future__ import annotations

import itertools
import time
from collections import Counter
from collections.abc import Sequence

from ...processes.core_syntax import (
    ProcessTuple,
    _tokenize_side,
    canonical_process_key,
    expand_process_variants,
)
from .process_catalog import (
    ANTI_PARTICLE,
    ANTIQUARKS,
    QUARKS,
    SORT_PARTICLES,
    concrete_processes_from_inclusive_enumeration,
    process_uses_inclusive_labels,
    request_allows_charged_current,
    selection_metadata,
)
from .process_enumeration import ProcessEnumerator
from .process_types import (
    BuiltinPhaseSpaceGroup as PhaseSpaceGroup,
)
from .process_types import (
    BuiltinProcessEnumeration as ProcessEnumeration,
)
from .process_types import (
    BuiltinProcessOptions as ProcessOptions,
)
from .process_types import (
    BuiltinProcessSelectionRecord as ProcessSelectionRecord,
)
from .process_types import (
    BuiltinProcessSelectionReport as ProcessSelectionReport,
)
from .process_types import (
    BuiltinProcessSetEntry as ProcessSetEntry,
)
from .process_types import (
    BuiltinProcessSetEnumeration as ProcessSetEnumeration,
)
from .process_types import (
    BuiltinSubprocessRecord as SubprocessRecord,
)


def enumerate_process_set(
    process_string: str,
    options: ProcessOptions | None = None,
) -> ProcessSetEnumeration:
    active_options = options or ProcessOptions()
    entries: list[ProcessSetEntry] = []
    seen: set[str] = set()
    for process in expand_process_variants(process_string):
        inclusive_enumeration: ProcessEnumeration | None = None
        if process_uses_inclusive_labels(process):
            inclusive_enumeration = ProcessEnumerator(active_options).enumerate(process)
            concrete_processes = concrete_processes_from_inclusive_enumeration(
                inclusive_enumeration
            )
        else:
            concrete_processes = (process,)
        for concrete_process in concrete_processes:
            key = canonical_process_key(concrete_process)
            if key in seen:
                continue
            seen.add(key)
            enumeration = ProcessEnumerator(active_options).enumerate(concrete_process)
            if enumeration.n_records == 0:
                continue
            entries.append(
                ProcessSetEntry(
                    key=key,
                    process=concrete_process,
                    enumeration=enumeration,
                )
            )
    if not entries:
        raise ValueError(f"no valid processes found for {process_string!r}")
    return ProcessSetEnumeration(
        request=process_string,
        options=active_options,
        entries=tuple(entries),
    )


def enumerate_generic_process_set(
    process_string: str,
    options: ProcessOptions | None = None,
    *,
    max_quark_pairs: int | None = None,
    use_prefilter: bool = True,
) -> ProcessSetEnumeration:
    """Expand process-set syntax for generic DAG generation.

    The generic compiler only needs concrete subprocess keys and parser
    metadata.  It should not pay the legacy phase-space/color-order enumeration
    cost for an already concrete partonic process; colour sectors and closures
    are discovered later by the model-driven DAG.  Inclusive ``p``/``j``
    requests still use the legacy enumerator once to obtain concrete children.
    """

    active_options = options or ProcessOptions()
    report = build_generic_process_selection_report(
        process_string,
        active_options,
        max_quark_pairs=max_quark_pairs,
        use_prefilter=use_prefilter,
    )
    if not report.entries:
        raise ValueError(f"no valid processes found for {process_string!r}")
    return ProcessSetEnumeration(
        request=process_string,
        options=active_options,
        entries=report.entries,
        selection_report=report,
    )


def build_generic_process_selection_report(
    process_string: str,
    options: ProcessOptions | None = None,
    *,
    max_quark_pairs: int | None = None,
    use_prefilter: bool = True,
) -> ProcessSelectionReport:
    """Return selected generic subprocesses plus enumeration diagnostics."""

    started = time.perf_counter()
    active_options = options or ProcessOptions()
    entries: list[ProcessSetEntry] = []
    seen: set[str] = set()
    records: list[ProcessSelectionRecord] = []
    rejection_counts: Counter[str] = Counter()
    candidate_count = 0
    evaluated_count = 0
    stage_timings: list[tuple[str, float]] = []

    stage_started = time.perf_counter()
    variants = expand_process_variants(process_string)
    stage_timings.append(("expand", time.perf_counter() - stage_started))

    stage_started = time.perf_counter()
    for process in variants:
        if use_prefilter:
            result = _generic_selection_records_from_request(
                process,
                active_options,
                max_quark_pairs=max_quark_pairs,
                seen=seen,
            )
        else:
            result = _legacy_selection_records_from_request(
                process,
                active_options,
                max_quark_pairs=max_quark_pairs,
                seen=seen,
            )
        candidate_count += result[0]
        evaluated_count += result[1]
        rejection_counts.update(result[2])
        records.extend(result[3])
    stage_timings.append(("prefilter", time.perf_counter() - stage_started))

    stage_started = time.perf_counter()
    for record in records:
        if record.status != "selected":
            continue
        enumeration = _lightweight_concrete_process_enumeration(
            record.process,
            active_options,
        )
        if enumeration.n_records == 0:
            rejection_counts["model-reachability"] += 1
            continue
        entries.append(
            ProcessSetEntry(
                key=record.key,
                process=record.process,
                enumeration=enumeration,
            )
        )
    stage_timings.append(("canonicalize", time.perf_counter() - stage_started))
    duplicate_count = sum(1 for record in records if record.status == "duplicate")
    selected_count = len(entries)
    rejected_count = candidate_count - selected_count - duplicate_count
    return ProcessSelectionReport(
        request=process_string,
        options=active_options,
        entries=tuple(entries),
        records=tuple(records),
        candidate_count=candidate_count,
        evaluated_count=evaluated_count,
        selected_count=selected_count,
        duplicate_count=duplicate_count,
        rejected_count=max(rejected_count, 0),
        rejection_counts=tuple(sorted(rejection_counts.items())),
        stage_timings=tuple(stage_timings),
        elapsed_s=time.perf_counter() - started,
        prefilter_enabled=use_prefilter,
    )


def _generic_selection_records_from_request(
    process: str,
    options: ProcessOptions,
    *,
    max_quark_pairs: int | None,
    seen: set[str],
) -> tuple[int, int, Counter[str], list[ProcessSelectionRecord]]:
    if process_uses_inclusive_labels(process):
        return _prefilter_inclusive_request(
            process,
            options,
            max_quark_pairs=max_quark_pairs,
            seen=seen,
        )
    return _classify_concrete_request(
        process,
        options,
        max_quark_pairs=max_quark_pairs,
        seen=seen,
        source=process,
        symmetric_initial=False,
    )


def _legacy_selection_records_from_request(
    process: str,
    options: ProcessOptions,
    *,
    max_quark_pairs: int | None,
    seen: set[str],
) -> tuple[int, int, Counter[str], list[ProcessSelectionRecord]]:
    if process_uses_inclusive_labels(process):
        enumeration = ProcessEnumerator(options).enumerate(process)
        concrete_processes = concrete_processes_from_inclusive_enumeration(enumeration)
    else:
        concrete_processes = (process,)
    records: list[ProcessSelectionRecord] = []
    rejection_counts: Counter[str] = Counter()
    candidate_count = 0
    evaluated_count = 0
    for concrete_process in concrete_processes:
        result = _classify_concrete_request(
            concrete_process,
            options,
            max_quark_pairs=max_quark_pairs,
            seen=seen,
            source=process,
            symmetric_initial=False,
        )
        candidate_count += result[0]
        evaluated_count += result[1]
        rejection_counts.update(result[2])
        records.extend(result[3])
    return candidate_count, evaluated_count, rejection_counts, records


def _classify_concrete_request(
    process: str,
    options: ProcessOptions,
    *,
    max_quark_pairs: int | None,
    seen: set[str],
    source: str,
    symmetric_initial: bool,
) -> tuple[int, int, Counter[str], list[ProcessSelectionRecord]]:
    enumerator = ProcessEnumerator(options)
    request = enumerator.parse(process)
    physical_initial = tuple(ANTI_PARTICLE[p] for p in request.initial_state)
    final_state = tuple(request.rest)
    return _classify_physical_candidate(
        enumerator,
        physical_initial,
        final_state,
        source=source,
        allow_charged_current=request_allows_charged_current(request),
        max_quark_pairs=max_quark_pairs,
        seen=seen,
        symmetric_initial=symmetric_initial,
    )


def _prefilter_inclusive_request(
    process: str,
    options: ProcessOptions,
    *,
    max_quark_pairs: int | None,
    seen: set[str],
) -> tuple[int, int, Counter[str], list[ProcessSelectionRecord]]:
    enumerator = ProcessEnumerator(options)
    request = enumerator.parse(process)
    initial_options: list[tuple[str, ...]] = []
    parton_options = tuple(
        sorted(enumerator.massless_qcd, key=lambda p: SORT_PARTICLES[p])
    )
    for crossed_particle in request.initial_state:
        if crossed_particle in {"p", "j"}:
            initial_options.append(parton_options)
        else:
            initial_options.append((ANTI_PARTICLE[crossed_particle],))

    symmetric_initial = (
        len(request.initial_state) == 2
        and request.initial_state[0] == request.initial_state[1]
        and request.initial_state[0] in {"p", "j"}
    )
    final_candidates = tuple(
        _generic_final_candidate(request.rest, jets)
        for jets in itertools.combinations_with_replacement(
            parton_options,
            request.jet_count,
        )
    )
    final_by_charge: Counter[int] = Counter(
        final_charge for _, final_charge, _, _, _ in final_candidates
    )
    final_by_charge_family: Counter[tuple[int, int]] = Counter(
        (final_charge, final_family)
        for _, final_charge, final_family, _, _ in final_candidates
    )
    bucketed_finals: dict[
        tuple[int, int | None], list[tuple[ProcessTuple, int, int, int, int]]
    ] = {}
    for candidate in final_candidates:
        final_state, final_charge, final_family, _, _ = candidate
        del final_state
        bucketed_finals.setdefault((final_charge, None), []).append(candidate)
        bucketed_finals.setdefault((final_charge, final_family), []).append(candidate)

    allow_charged_current = request_allows_charged_current(request)
    records: list[ProcessSelectionRecord] = []
    rejection_counts: Counter[str] = Counter()
    candidate_count = 0
    evaluated_count = 0
    for initial_state in itertools.product(*initial_options):
        crossed_initial = tuple(ANTI_PARTICLE[p] for p in initial_state)
        initial_charge = _charge3_sum(crossed_initial)
        initial_family = _family_sum(crossed_initial)
        candidate_count += len(final_candidates)
        charge_key = -initial_charge
        charge_matches = final_by_charge[charge_key]
        rejection_counts["charge"] += len(final_candidates) - charge_matches
        if options.include_cc:
            candidate_bucket = bucketed_finals.get((charge_key, None), ())
        else:
            family_key = -initial_family
            family_matches = final_by_charge_family[(charge_key, family_key)]
            rejection_counts["fermion-family"] += charge_matches - family_matches
            candidate_bucket = bucketed_finals.get((charge_key, family_key), ())
        for final_state, _, _, _, _ in candidate_bucket:
            result = _classify_physical_candidate(
                enumerator,
                initial_state,
                final_state,
                source=process,
                allow_charged_current=allow_charged_current,
                max_quark_pairs=max_quark_pairs,
                seen=seen,
                symmetric_initial=symmetric_initial,
                prechecked_charge_family=True,
            )
            candidate_count += result[0] - 1
            evaluated_count += result[1]
            rejection_counts.update(result[2])
            records.extend(result[3])
    return candidate_count, evaluated_count, rejection_counts, records


def _classify_physical_candidate(
    enumerator: ProcessEnumerator,
    physical_initial: Sequence[str],
    final_state: Sequence[str],
    *,
    source: str,
    allow_charged_current: bool,
    max_quark_pairs: int | None,
    seen: set[str],
    symmetric_initial: bool,
    prechecked_charge_family: bool = False,
) -> tuple[int, int, Counter[str], list[ProcessSelectionRecord]]:
    display_process = _canonical_physical_process(
        physical_initial,
        final_state,
        symmetric_initial=symmetric_initial,
    )
    canonical_initial, _, canonical_final = display_process.partition(">")
    canonical_crossed_initial = tuple(
        ANTI_PARTICLE[p] for p in _tokenize_side(canonical_initial.strip())
    )
    canonical_final_state = tuple(_tokenize_side(canonical_final.strip()))
    all_outgoing = (*canonical_crossed_initial, *canonical_final_state)
    charge3 = _charge3_sum(all_outgoing)
    family = _family_sum(all_outgoing)
    quark_lines = _quark_pair_count(all_outgoing)
    records: list[ProcessSelectionRecord] = []
    rejection_counts: Counter[str] = Counter()
    evaluated_count = 1
    reason = ""
    if not prechecked_charge_family and charge3 != 0:
        reason = "charge"
    elif (
        not prechecked_charge_family
        and not enumerator.options.include_cc
        and family != 0
    ):
        reason = "fermion-family"
    elif max_quark_pairs is not None and quark_lines > int(max_quark_pairs):
        reason = "max-quark-lines"
    elif _single_impossible_colored_state(all_outgoing):
        reason = "single-coloured-state"
    elif not _valid_generic_process(
        enumerator,
        all_outgoing,
        allow_charged_current=allow_charged_current,
    ):
        reason = "model-reachability"

    key = canonical_process_key(display_process)
    dedupe_key = _process_dedupe_signature(
        physical_initial,
        final_state,
        symmetric_initial=symmetric_initial,
    )
    if reason:
        rejection_counts[reason] += 1
        return 1, evaluated_count, rejection_counts, []
    if dedupe_key in seen:
        records.append(
            ProcessSelectionRecord(
                source=source,
                process=display_process,
                key=key,
                status="duplicate",
                reason="canonical-duplicate",
                quark_lines=quark_lines,
                charge3=charge3,
                family=family,
            )
        )
        return 1, evaluated_count, rejection_counts, records
    seen.add(dedupe_key)
    records.append(
        ProcessSelectionRecord(
            source=source,
            process=display_process,
            key=key,
            status="selected",
            reason="selected",
            quark_lines=quark_lines,
            charge3=charge3,
            family=family,
        )
    )
    return 1, evaluated_count, rejection_counts, records


def _canonical_physical_process(
    initial_state: Sequence[str],
    final_state: Sequence[str],
    *,
    symmetric_initial: bool,
) -> str:
    initial = tuple(initial_state)
    if symmetric_initial:
        initial = tuple(sorted(initial, key=lambda particle: SORT_PARTICLES[particle]))
    final = tuple(final_state)
    return f"{' '.join(initial)} > {' '.join(final)}"


def _process_dedupe_signature(
    initial_state: Sequence[str],
    final_state: Sequence[str],
    *,
    symmetric_initial: bool,
) -> str:
    initial = tuple(initial_state)
    if symmetric_initial:
        initial = tuple(sorted(initial, key=lambda particle: SORT_PARTICLES[particle]))
    final = tuple(sorted(final_state, key=lambda particle: SORT_PARTICLES[particle]))
    return f"{' '.join(initial)} > {' '.join(final)}"


def _single_impossible_colored_state(process: Sequence[str]) -> bool:
    return (
        sum(1 for particle in process if selection_metadata(particle).is_colored) == 1
    )


def _valid_generic_process(
    enumerator: ProcessEnumerator,
    process: Sequence[str],
    *,
    allow_charged_current: bool,
) -> bool:
    if enumerator._valid_process(
        process,
        allow_charged_current=allow_charged_current,
    ):
        return True
    if any(selection_metadata(particle).is_colored for particle in process):
        return False
    if _charge3_sum(process) != 0:
        return False
    return enumerator.options.include_cc or _family_sum(process) == 0


def _quark_pair_count(process: Sequence[str]) -> int:
    quarks, antiquarks = _quark_counts(process)
    return min(quarks, antiquarks)


def _charge3_sum(process: Sequence[str]) -> int:
    return sum(selection_metadata(particle).charge3 for particle in process)


def _family_sum(process: Sequence[str]) -> int:
    return sum(selection_metadata(particle).family for particle in process)


def _quark_counts(process: Sequence[str]) -> tuple[int, int]:
    counts = Counter(process)
    quarks = sum(counts[item] for item in QUARKS if item in counts)
    antiquarks = sum(counts[item] for item in ANTIQUARKS if item in counts)
    return quarks, antiquarks


def _generic_final_candidate(
    rest: Sequence[str],
    jets: Sequence[str],
) -> tuple[ProcessTuple, int, int, int, int]:
    final_state = tuple(
        sorted(
            (*rest, *jets),
            key=lambda particle: SORT_PARTICLES[particle],
        )
    )
    quarks, antiquarks = _quark_counts(final_state)
    return (
        final_state,
        _charge3_sum(final_state),
        _family_sum(final_state),
        quarks,
        antiquarks,
    )


def _lightweight_concrete_process_enumeration(
    process: str,
    options: ProcessOptions,
) -> ProcessEnumeration:
    enumerator = ProcessEnumerator(options)
    request = enumerator.parse(process)
    all_outgoing = (*request.initial_state, *request.rest)
    if not _valid_generic_process(
        enumerator,
        all_outgoing,
        allow_charged_current=request_allows_charged_current(request),
    ):
        return ProcessEnumeration(
            request=request,
            options=options,
            unique_processes=(),
            groups=(),
        )
    order = tuple(range(len(all_outgoing)))
    return ProcessEnumeration(
        request=request,
        options=options,
        unique_processes=(
            tuple(sorted(all_outgoing, key=lambda item: SORT_PARTICLES[item])),
        ),
        groups=(
            PhaseSpaceGroup(
                group_id=1,
                phase_space_order=order,
                records=(
                    SubprocessRecord(
                        process=tuple(all_outgoing),
                        color_order=order,
                        multichannel_partners=(0,),
                        identical_factor=1.0,
                    ),
                ),
            ),
        ),
    )
