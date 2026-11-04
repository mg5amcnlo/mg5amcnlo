# SPDX-License-Identifier: 0BSD
"""Legacy built-in-SM concrete subprocess enumeration."""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import replace

from ...processes.core_syntax import (
    OrderTuple,
    ProcessTuple,
    _tokenize_side,
    expand_process_variants,
)
from .process_catalog import (
    ALL_COLOURED,
    ANTI_PARTICLE,
    ANTIQUARKS,
    CHARGES3,
    FAMILY,
    GLUONS,
    PDGS,
    QUARKS,
    SINGLETS,
    SORT_PARTICLES,
)
from .process_catalog import (
    request_allows_charged_current as _request_allows_charged_current,
)
from .process_types import BuiltinParsedProcess as ParsedProcess
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
    BuiltinSubprocessRecord as SubprocessRecord,
)


def _ordered_compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    if parts <= 0:
        return ((),) if total == 0 else ()
    if parts == 1:
        return ((total,),)
    return tuple(
        (first, *rest)
        for first in range(total + 1)
        for rest in _ordered_compositions(total - first, parts - 1)
    )


def _chunk_sequence(
    values: Sequence[int],
    chunk_lengths: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    chunks: list[tuple[int, ...]] = []
    start = 0
    for length in chunk_lengths:
        chunks.append(tuple(values[start : start + length]))
        start += length
    return tuple(chunks)


class ProcessEnumerator:
    """Structured port of the legacy process_list.py enumeration logic."""

    def __init__(self, options: ProcessOptions | None = None) -> None:
        self.options = options or ProcessOptions()
        self.flavour_scheme = self._flavour_scheme(self.options.flavour_scheme)
        self.massless_qcd = (
            self.flavour_scheme
            | frozenset(f"{q}~" for q in self.flavour_scheme)
            | GLUONS
        )
        self.proton = self.massless_qcd
        self.jet = self.massless_qcd
        self._phase_space_orders: dict[OrderTuple, list[SubprocessRecord]] = {}
        self._all_keys_sorted: list[OrderTuple] = []
        self._process_order_to_index: dict[tuple[ProcessTuple, OrderTuple], int] = {}

    def parse(self, process_string: str) -> ParsedProcess:
        variants = expand_process_variants(process_string)
        if len(variants) != 1:
            raise ValueError(
                "ProcessEnumerator.parse expects one concrete process; "
                "use enumerate_process_set for process sets or multiparticle expansion"
            )
        input_string = variants[0].lower().replace("bar", "~")
        parts = input_string.split(">")
        if len(parts) != 2:
            raise ValueError("invalid collision format; expected 'initial > final'")

        initial_state = _tokenize_side(parts[0].strip())
        if len(initial_state) != 2:
            raise ValueError("exactly two incoming particles are required")
        crossed_initial = tuple(
            ANTI_PARTICLE[p] if p not in {"p", "j"} else p for p in initial_state
        )

        final_state = _tokenize_side(parts[1].strip())
        jet_count = 0
        rest: list[str] = []
        for token in final_state:
            jet_match = re.fullmatch(r"(\d+)j", token)
            if token == "j":
                jet_count += 1
            elif jet_match:
                jet_count += int(jet_match.group(1))
            else:
                rest.append(token)
        leptons: list[str] = []
        if self.options.include_resonance:
            leptons = [p for p in rest if 11 <= abs(int(PDGS[p])) <= 16]
            for lepton in leptons:
                rest.remove(lepton)
            charge = sum(CHARGES3[p] for p in leptons)
            if charge == 0:
                rest.append("z")
            elif charge < 0:
                rest.append("w-")
            else:
                rest.append("w+")

        self._validate_particles([*crossed_initial, *rest])
        return ParsedProcess(crossed_initial, jet_count, tuple(rest), tuple(leptons))

    def enumerate(self, process_string: str) -> ProcessEnumeration:
        request = self.parse(process_string)
        unique_processes = self._generate_all_unique_processes(request)
        subprocesses = self._generate_all_processes(unique_processes, request)
        self._phase_space_orders = self._combine_results(
            self._process_subprocess(proc) for proc in subprocesses
        )
        self._all_keys_sorted = sorted(self._phase_space_orders)
        self._determine_multichannel_partners_and_symmetry_factor()
        self._check_consistency()

        groups = tuple(
            PhaseSpaceGroup(
                group_id=i + 1,
                phase_space_order=key,
                records=tuple(
                    sorted(self._phase_space_orders[key], key=self._sort_record)
                ),
            )
            for i, key in enumerate(self._all_keys_sorted)
        )
        return ProcessEnumeration(
            request=request,
            options=self.options,
            unique_processes=tuple(
                tuple(proc)
                for proc in sorted(
                    (
                        tuple(sorted(p, key=lambda x: SORT_PARTICLES[x]))
                        for p in unique_processes
                    ),
                    key=self._sort_process,
                )
            ),
            groups=groups,
        )

    def enumerate_color_complete(self, process_string: str) -> ProcessEnumeration:
        """Enumerate a reference-only colour-complete legacy process file.

        The production LC process list deliberately removes colour-order
        representatives related by legacy symmetries.  That is correct for
        ordinary integration and LC library timing, but NLC/full-colour
        generated-library validation needs raw amplitudes for every colour
        basis row used by AmpliCol's colour matrix.  This reference path keeps
        all model-compatible candidate colour words in a single phase-space
        group and avoids process-family assumptions.
        """

        request = self.parse(process_string)
        unique_processes = self._generate_all_unique_processes(request)
        subprocesses = self._generate_all_processes(unique_processes, request)
        records: list[SubprocessRecord] = []
        seen: set[tuple[ProcessTuple, OrderTuple]] = set()
        for proc in sorted(subprocesses, key=self._sort_process):
            for perm in self._candidate_color_orders(proc):
                order = tuple(perm)
                key = (proc, order)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    SubprocessRecord(
                        process=proc,
                        color_order=order,
                        multichannel_partners=(0,),
                        identical_factor=self._identical_particle_symmetry_factor(proc),
                    )
                )
        if not records:
            return ProcessEnumeration(
                request=request,
                options=self.options,
                unique_processes=(),
                groups=(),
            )
        return ProcessEnumeration(
            request=request,
            options=self.options,
            unique_processes=tuple(
                tuple(proc)
                for proc in sorted(
                    (
                        tuple(sorted(p, key=lambda x: SORT_PARTICLES[x]))
                        for p in unique_processes
                    ),
                    key=self._sort_process,
                )
            ),
            groups=(
                PhaseSpaceGroup(
                    group_id=1,
                    phase_space_order=tuple(range(len(records[0].process))),
                    records=tuple(records),
                ),
            ),
        )

    def _flavour_scheme(self, value: int) -> frozenset[str]:
        flavours = ("d", "u", "s", "c", "b", "t")
        if not 1 <= value <= 6:
            raise ValueError(f"unknown flavour scheme: {value}")
        return frozenset(flavours[:value])

    def _validate_particles(self, particles: Iterable[str]) -> None:
        unknown = sorted(set(particles).difference(PDGS).difference({"p", "j"}))
        if unknown:
            raise ValueError(f"unknown particle name(s): {unknown}")

    def _valid_color_order(self, proc: ProcessTuple, perm: OrderTuple) -> bool:
        found_quark = found_antiquark = found_singlet = found_gluon = found_first = (
            False
        )
        quark_idx = -1
        for idx in perm:
            if idx == 0:
                found_first = True
            particle = proc[idx]
            if particle in QUARKS:
                if found_quark or found_gluon:
                    return False
                found_quark = True
                found_antiquark = found_singlet = found_gluon = False
                quark_idx = idx
            elif particle in ANTIQUARKS:
                if found_antiquark or found_singlet or not found_quark:
                    return False
                if not found_first:
                    return False
                found_antiquark = True
                found_quark = found_singlet = found_gluon = False
                _ = quark_idx
            elif particle in GLUONS:
                if found_antiquark or found_singlet:
                    return False
                found_gluon = True
            else:
                if found_quark or found_gluon or not found_antiquark:
                    return False
                found_singlet = True
        return not (found_gluon and perm[0] != 0)

    def _unique_color_order(self, proc: ProcessTuple, perm: OrderTuple) -> bool:
        zero = perm.index(0)
        perm_mapped = perm[zero:] + perm[:zero]
        for particle in ALL_COLOURED:
            particle_positions = [
                i + 2 for i, part in enumerate(proc[2:]) if part == particle
            ]
            previous_position = 0
            for position in particle_positions:
                mapped_position = perm_mapped.index(position)
                if mapped_position < previous_position:
                    return False
                previous_position = mapped_position
        return True

    def _order_proc_perm(
        self, proc: ProcessTuple, perm: OrderTuple
    ) -> tuple[ProcessTuple, OrderTuple]:
        zero = perm.index(0)
        perm_mapped = list(perm[zero:] + perm[:zero])
        elements_to_order = [
            i for i in perm_mapped if proc[i] in self.massless_qcd and i > 1
        ]
        indices = [perm_mapped.index(x) for x in elements_to_order]
        for idx, value in zip(indices, sorted(elements_to_order), strict=True):
            perm_mapped[idx] = value
        perm_ordered = tuple(
            perm_mapped[len(perm) - zero :] + perm_mapped[: len(perm) - zero]
        )
        proc_ordered: list[str | None] = [None] * len(proc)
        for i, perm_index in enumerate(perm_ordered):
            proc_ordered[perm_index] = proc[perm[i]]
        return tuple(x for x in proc_ordered if x is not None), perm_ordered

    def _process_subprocess(
        self, proc: ProcessTuple
    ) -> dict[OrderTuple, list[SubprocessRecord]]:
        local: dict[OrderTuple, list[SubprocessRecord]] = {}
        for perm in self._candidate_color_orders(proc):
            order = tuple(perm)
            if not self._valid_color_order(proc, order):
                continue
            if not self._unique_color_order(proc, order):
                continue
            ordered_proc, ordered_perm = self._order_proc_perm(proc, order)
            zero = ordered_perm.index(0)
            perm_mapped = tuple(ordered_perm[zero:] + ordered_perm[:zero])
            local.setdefault(perm_mapped, []).append(
                SubprocessRecord(ordered_proc, ordered_perm)
            )
        return local

    def _candidate_color_orders(self, proc: ProcessTuple) -> Iterable[OrderTuple]:
        quark_indices = tuple(
            i for i, particle in enumerate(proc) if particle in QUARKS
        )
        anti_indices = tuple(
            i for i, particle in enumerate(proc) if particle in ANTIQUARKS
        )
        gluon_indices = tuple(
            i for i, particle in enumerate(proc) if particle in GLUONS
        )
        singlet_indices = tuple(
            i for i, particle in enumerate(proc) if particle in SINGLETS
        )
        if not quark_indices:
            for tail in itertools.permutations(
                index for index in gluon_indices if index != 0
            ):
                yield (0, *tail)
            return

        if singlet_indices:
            singlet_perms = tuple(itertools.permutations(singlet_indices))
        else:
            singlet_perms = ((),)
        gluon_compositions = _ordered_compositions(
            len(gluon_indices),
            len(quark_indices),
        )
        singlet_compositions = _ordered_compositions(
            len(singlet_indices),
            len(quark_indices),
        )
        for quark_order in itertools.permutations(quark_indices):
            for anti_order in itertools.permutations(anti_indices):
                for gluon_perm in itertools.permutations(gluon_indices):
                    for gluon_chunks in gluon_compositions:
                        gluon_by_line = _chunk_sequence(gluon_perm, gluon_chunks)
                        for singlet_perm in singlet_perms:
                            for singlet_chunks in singlet_compositions:
                                singlet_by_line = _chunk_sequence(
                                    singlet_perm,
                                    singlet_chunks,
                                )
                                order: list[int] = []
                                for line in range(len(quark_indices)):
                                    order.append(quark_order[line])
                                    order.extend(gluon_by_line[line])
                                    order.append(anti_order[line])
                                    order.extend(singlet_by_line[line])
                                yield tuple(order)

    def _valid_process(
        self,
        proc: Sequence[str],
        *,
        allow_charged_current: bool | None = None,
    ) -> bool:
        allow_cc = (
            self.options.include_cc
            if allow_charged_current is None
            else allow_charged_current
        )
        nq = self._count_matching(proc, QUARKS)
        naq = self._count_matching(proc, ANTIQUARKS)
        if nq != naq:
            return False
        if sum(CHARGES3[x] for x in proc) != 0:
            return False
        if not self.options.include_cc and sum(FAMILY[x] for x in proc) != 0:
            return False
        if not allow_cc and "w+" not in proc and "w-" not in proc:
            for q in QUARKS:
                if self._count_matching(proc, [q]) != self._count_matching(
                    proc, [f"{q}~"]
                ):
                    return False
        return not (nq == 0 and self._count_matching(proc, SINGLETS) > 0)

    def _compatible_unique_process(
        self, request: ParsedProcess, proc: Sequence[str]
    ) -> bool:
        mandatory = [p for p in request.initial_state if p not in {"p", "j"}]
        mandatory.extend(request.rest)
        proc_local = list(proc)
        try:
            for particle in mandatory:
                proc_local.remove(particle)
        except ValueError:
            return False
        return True

    def _compatible_process(self, request: ParsedProcess, proc: ProcessTuple) -> bool:
        proc_local = list(proc[2:])
        for i in (0, 1):
            if (
                request.initial_state[i] not in {"p", "j"}
                and proc[i] != request.initial_state[i]
            ):
                return False
        try:
            for particle in request.rest:
                proc_local.remove(particle)
        except ValueError:
            return False
        return True

    def _generate_all_unique_processes(
        self, request: ParsedProcess
    ) -> set[ProcessTuple]:
        processes: list[list[str]] = [[]]
        for part in request.initial_state:
            if part not in {"p", "j"} and part not in self.massless_qcd:
                raise ValueError(
                    "initial state should be a proton or massless QCD parton"
                )

        if request.jet_count == 0 and all(
            part not in {"p", "j"} for part in request.initial_state
        ):
            fixed_process = tuple(
                sorted(
                    (*request.initial_state, *request.rest),
                    key=lambda item: SORT_PARTICLES[item],
                )
            )
            if self._valid_process(
                fixed_process,
                allow_charged_current=_request_allows_charged_current(request),
            ) and self._compatible_unique_process(request, fixed_process):
                return {fixed_process}
            return set()

        qcd_rest = sum(1 for part in request.rest if part in self.massless_qcd)
        for _part_index in range(request.jet_count + 2 + qcd_rest):
            new_processes: list[list[str]] = []
            for candidate in processes:
                for particle in self.massless_qcd:
                    new_processes.append(sorted([*candidate, particle]))
            processes = new_processes

        for part in request.rest:
            if part in self.massless_qcd:
                continue
            for candidate in processes:
                candidate.append(part)

        return {
            tuple(proc)
            for proc in processes
            if self._valid_process(
                proc,
                allow_charged_current=_request_allows_charged_current(request),
            )
            and self._compatible_unique_process(request, proc)
        }

    def _generate_all_processes(
        self, unique_processes: set[ProcessTuple], request: ParsedProcess
    ) -> set[ProcessTuple]:
        processes: set[ProcessTuple] = set()
        for proc in unique_processes:
            for i, j in itertools.combinations(range(len(proc)), 2):
                if proc[i] in self.jet and proc[j] in self.jet:
                    remaining = [
                        entry for k, entry in enumerate(proc) if k not in (i, j)
                    ]
                    proc1 = (proc[i], proc[j], *remaining)
                    proc2 = (proc[j], proc[i], *remaining)
                    if self._compatible_process(request, proc1):
                        processes.add(proc1)
                    if self._compatible_process(request, proc2):
                        processes.add(proc2)
        return processes

    def _combine_results(
        self, results: Iterable[dict[OrderTuple, list[SubprocessRecord]]]
    ) -> dict[OrderTuple, list[SubprocessRecord]]:
        combined: dict[OrderTuple, list[SubprocessRecord]] = {}
        for result in results:
            for key, value in result.items():
                combined.setdefault(key, []).extend(value)
        return combined

    def _determine_multichannel_partners_and_symmetry_factor(self) -> None:
        self._build_process_index()
        for key in self._all_keys_sorted:
            for index, record in enumerate(tuple(self._phase_space_orders[key])):
                self._phase_space_orders[key][index] = self._record_with_partners(
                    record
                )

        for key in self._all_keys_sorted:
            for index, record in enumerate(tuple(self._phase_space_orders[key])):
                self._phase_space_orders[key][index] = replace(
                    record,
                    identical_factor=record.identical_factor
                    * self._identical_particle_symmetry_factor(record.process),
                )

    def _build_process_index(self) -> None:
        self._process_order_to_index = {}
        for i, key in enumerate(self._all_keys_sorted):
            for record in self._phase_space_orders[key]:
                self._process_order_to_index[(record.process, record.color_order)] = i

    def _record_with_partners(self, record: SubprocessRecord) -> SubprocessRecord:
        proc = record.process
        perm = record.color_order
        singlet_indices = [perm.index(i) for i, p in enumerate(proc) if p in SINGLETS]
        anti_quark_indices = tuple(
            perm.index(i) for i, p in enumerate(proc) if p in ANTIQUARKS
        )
        nsinglets = len(singlet_indices)
        if len(singlet_indices) > 1:
            singlet_perms = tuple(itertools.permutations(singlet_indices))
        elif len(singlet_indices) == 1:
            singlet_perms = (tuple(singlet_indices),)
        else:
            singlet_perms = ()

        iden = float(math.factorial(max(len(anti_quark_indices) - 1, 0)))
        possible: list[tuple[OrderTuple, ProcessTuple]] = []
        if not singlet_perms:
            possible.append((perm, proc))
        elif anti_quark_indices:
            for singlets in singlet_perms:
                for chunks in _ordered_compositions(
                    nsinglets,
                    len(anti_quark_indices),
                ):
                    starts = [0]
                    for chunk in chunks[:-1]:
                        starts.append(starts[-1] + chunk)
                    singlet_chunks = tuple(
                        singlets[start : start + chunk]
                        for start, chunk in zip(starts, chunks, strict=True)
                    )
                    order_list: list[int] = []
                    for i in range(len(perm)):
                        if i in anti_quark_indices:
                            anti_index = anti_quark_indices.index(i)
                            order_list.extend(
                                perm[p]
                                for p in (
                                    (anti_quark_indices[anti_index],)
                                    + singlet_chunks[anti_index]
                                )
                            )
                        elif i not in singlet_indices:
                            order_list.append(perm[i])
                    possible.append((tuple(order_list), proc))
        else:
            raise RuntimeError(
                "colour-singlet partners require at least one quark line"
            )

        partners: list[int] = []
        for possible_order, process in possible:
            partner = self._process_order_to_index.get((process, possible_order))
            if partner is None:
                raise RuntimeError(
                    "expected multichannel partner not found for "
                    f"{process} {possible_order}"
                )
            if partner in partners:
                raise RuntimeError(
                    f"duplicate multichannel partner for {process} {possible_order}"
                )
            partners.append(partner)
        return replace(
            record,
            multichannel_partners=tuple(sorted(partners)),
            identical_factor=1.0 / iden,
        )

    def _identical_particle_symmetry_factor(self, proc: ProcessTuple) -> float:
        factor = 1.0
        for particle in ALL_COLOURED:
            factor *= max(1, math.factorial(proc[2:].count(particle)))
        return factor

    def _check_consistency(self) -> None:
        all_processes: dict[ProcessTuple, float] = {}
        for key in self._all_keys_sorted:
            for record in self._phase_space_orders[key]:
                proc = (
                    record.process[0],
                    record.process[1],
                    *sorted(record.process[2:], key=lambda x: SORT_PARTICLES[x]),
                )
                all_processes[proc] = all_processes.get(proc, 0.0) + (
                    record.identical_factor / len(record.multichannel_partners)
                )
        for proc, count in all_processes.items():
            expected = self.expected_number_of_dual_amplitudes(proc)
            if abs(count - expected) > 1e-5:
                raise RuntimeError(
                    f"inconsistent number of dual amplitudes for {proc}: "
                    f"{count}, expected {expected}"
                )

    def expected_number_of_dual_amplitudes(self, proc: Sequence[str]) -> float:
        nq = self._count_matching(proc, QUARKS)
        ng = self._count_matching(proc, GLUONS)
        if nq == 0:
            return math.factorial(ng - 1)
        return math.factorial(ng) * math.comb(ng + nq - 1, nq - 1) * math.factorial(nq)

    def _count_matching(self, main: Sequence[str], check: Iterable[str]) -> int:
        counts = Counter(main)
        return sum(counts[item] for item in check if item in counts)


def enumerate_processes(
    process_string: str, options: ProcessOptions | None = None
) -> ProcessEnumeration:
    return ProcessEnumerator(options).enumerate(process_string)
