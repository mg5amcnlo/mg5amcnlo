#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect events from multiple LHE files, shuffle them, and write a new LHE file.

Key behavior
------------
- Events are copied verbatim.
- The output header is built from a separate template file.
- The <initrwgt> and <scalesfunctionalform> blocks are extracted from the
  first input file and inserted into the final <header> block.
- The <init> block is taken from the first input file.
- Template lines containing only <![CDATA[ or ]]> are removed.
- The template iseed line is rewritten from the requested seed.
- Output can be written uncompressed or as .gz.
- mode="auto" chooses between an in-memory shuffle path and a scalable
  disk-backed shuffle path.

This module can be imported and called via collect_events(...), or run as a CLI.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import mmap
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

PATT_LHE_OPEN = re.compile(rb"<\s*LesHouchesEvents\b[^>]*>", re.I)
PATT_LHE_CLOSE = re.compile(rb"</\s*LesHouchesEvents\s*>", re.I)
PATT_HEADER_OPEN = re.compile(rb"<\s*header\b[^>]*>", re.I)
PATT_HEADER_FULL = re.compile(rb"<\s*header\b[^>]*>(.*?)</\s*header\s*>", re.I | re.S)
PATT_INIT_OPEN = re.compile(rb"<\s*init\b[^>]*>", re.I)
PATT_INIT_CLOSE = re.compile(rb"</\s*init\s*>", re.I)
PATT_EVENT_OPEN = re.compile(rb"<\s*event\b", re.I)
PATT_EVENT_CLOSE = re.compile(rb"</\s*event\s*>", re.I)
PATT_XML_DECL = re.compile(rb"^\s*<\?xml[^>]*>\s*", re.I | re.S)
PATT_CDATA_LINE = re.compile(rb"(?m)^\s*(<!\[CDATA\[|\]\]>)\s*(?:\r?\n)?")
PATT_ISEED_LINE = re.compile(rb"(?m)^(?P<indent>\s*).*=\s*iseed\b.*(?:\r?\n)?")

INDEX_RECORD = struct.Struct(">Q I Q Q")

DEFAULT_MODE = "auto"
AUTO_MAX_FILES_FOR_MEMORY = 128
AUTO_MAX_INPUT_BYTES_FOR_MEMORY = 2048 * 1024 * 1024
EXTERNAL_RUN_RECORD_CAPACITY = 250_000

PathLike = Union[str, Path]

@dataclass(frozen=True)
class EventRef:
    file_idx: int
    start: int
    end: int

@dataclass
class IndexedLHE:
    path: Path
    open_tag: bytes
    special_header_blocks: bytes
    init_block: bytes
    events: List[EventRef]

@dataclass
class LHEPreamble:
    path: Path
    open_tag: bytes
    special_header_blocks: bytes
    init_block: bytes

# ============================
# Generic helpers
# ============================

def _wants_gzip_output(output_path: Path) -> bool:
    return output_path.suffix.lower() == ".gz"

@contextmanager
def _open_output_stream(
    output_path: Path,
    prefer_pigz: bool = True,
    gzip_level: int = 6,
    verbose: bool = False,
):
    gzip_level = max(0, min(9, int(gzip_level)))

    if not _wants_gzip_output(output_path):
        with open(output_path, "wb") as fout:
            yield fout
        return

    pigz_path = shutil.which("pigz") if prefer_pigz else None
    if pigz_path:
        cmd = [pigz_path, "-c", "-n", f"-{gzip_level}"]
        if verbose:
            print(f"      output compression: pigz ({' '.join(cmd[1:])})")
        with open(output_path, "wb") as raw_out:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=raw_out,
                stderr=subprocess.PIPE,
            )
            try:
                if proc.stdin is None:
                    raise RuntimeError("Failed to open pigz stdin pipe")
                yield proc.stdin
                proc.stdin.close()
                stderr = b""
                if proc.stderr is not None:
                    stderr = proc.stderr.read()
                ret = proc.wait()
                if ret != 0:
                    msg = stderr.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"pigz failed with exit code {ret}: {msg or 'no error message'}")
            except Exception:
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
                proc.kill()
                proc.wait()
                raise
            finally:
                if proc.stderr is not None:
                    proc.stderr.close()
        return

    if verbose:
        print(f"      output compression: gzip (python stdlib, level={gzip_level})")
    with open(output_path, "wb") as raw_out:
        with gzip.GzipFile(fileobj=raw_out, mode="wb", compresslevel=gzip_level, mtime=0) as fout:
            yield fout

def _ensure_trailing_newline(blob: bytes) -> bytes:
    return blob if blob.endswith(b"\n") else blob + b"\n"

def _extract_open_tag(blob: bytes) -> Optional[bytes]:
    m = PATT_LHE_OPEN.search(blob)
    if not m:
        return None
    return _ensure_trailing_newline(m.group(0))

def _extract_header_inner(blob: bytes) -> bytes:
    m = PATT_HEADER_FULL.search(blob)
    if not m:
        return b""
    inner = m.group(1).strip()
    return _ensure_trailing_newline(inner) if inner else b""

def _extract_tag_blocks(blob: bytes, tag: str) -> List[bytes]:
    tag_re = re.escape(tag.encode("utf-8"))
    full_re = re.compile(rb"<\s*" + tag_re + rb"\b[^>]*>.*?</\s*" + tag_re + rb"\s*>", re.I | re.S)
    self_re = re.compile(rb"<\s*" + tag_re + rb"\b[^>]*/\s*>", re.I)
    out = full_re.findall(blob)
    out.extend(self_re.findall(blob))
    return [_ensure_trailing_newline(x.strip()) for x in out if x.strip()]

def _extract_special_header_blocks(blob: bytes) -> bytes:
    """
    Extract only the special blocks that live between the opening
    <LesHouchesEvents ...> tag and the <header> block.
    """
    m_open = PATT_LHE_OPEN.search(blob)
    if not m_open:
        return b""

    m_header = PATT_HEADER_OPEN.search(blob, m_open.end())
    if not m_header:
        return b""

    region = bytes(blob[m_open.end():m_header.start()])
    pieces: List[bytes] = []
    for tag in ("initrwgt", "scalesfunctionalform", "MonteCarloMasses"):
        pieces.extend(_extract_tag_blocks(region, tag))
    return b"".join(pieces)

def _rewrite_iseed_line(blob: bytes, seed: Optional[int]) -> bytes:
    out_seed = 0 if seed is None else seed

    def repl(match: re.Match[bytes]) -> bytes:
        indent = match.group("indent")
        return indent + f"{out_seed}    = iseed       ! rnd seed (0=assigned automatically=default))\n".encode("utf-8")

    return PATT_ISEED_LINE.sub(repl, blob)

def _filter_template_header_xml_blocks(blob: bytes, keep_tags: Optional[Sequence[str]]) -> bytes:
    """
    Keep only the requested top-level XML blocks from the template header,
    while preserving raw non-XML text between them.

    This deliberately does *not* require the whole header fragment to be valid
    XML. Some MG header fragments contain raw text that ElementTree rejects.
    We therefore identify top-level blocks by tag boundaries in the raw bytes and
    copy kept blocks verbatim.
    """
    if not keep_tags:
        return blob

    keep = {tag.strip() for tag in keep_tags if tag and tag.strip()}
    if not keep:
        return blob

    tag_name = rb"(?:[A-Za-z_][\w.:-]*)"
    block_re = re.compile(
        rb"<\s*(?P<name>" + tag_name + rb")\b[^>]*"
        rb"(?:/\s*>|>.*?</\s*(?P=name)\s*>)",
        re.I | re.S,
    )

    out: List[bytes] = []
    pos = 0
    for m in block_re.finditer(blob):
        start, end = m.span()
        if start > pos:
            out.append(blob[pos:start])
        full_name = m.group("name").decode("ascii", "ignore")
        local_name = full_name.rsplit(":", 1)[-1]
        if local_name in keep:
            out.append(blob[start:end])
        pos = end

    if pos < len(blob):
        out.append(blob[pos:])

    return b"".join(out)

def _sanitize_header_template(blob: bytes, seed: Optional[int], template_header_keep_tags: Optional[Sequence[str]] = None) -> Tuple[Optional[bytes], bytes]:
    """
    Accepts either a full LHE file, a <header>...</header> block, or a raw header
    fragment. Returns:
      - optional <LesHouchesEvents ...> opening tag from the template
      - the header *inner* content to place inside the output <header>

    If template_header_keep_tags is given, only those XML blocks are retained
    from the template header content. Non-XML text outside the dropped blocks
    is preserved.
    """
    open_tag = _extract_open_tag(blob)

    header_inner = _extract_header_inner(blob)
    if not header_inner:
        cleaned = blob
        cleaned = PATT_XML_DECL.sub(b"", cleaned, count=1)
        cleaned = PATT_LHE_OPEN.sub(b"", cleaned, count=1)
        cleaned = PATT_LHE_CLOSE.sub(b"", cleaned, count=1)
        cleaned = re.sub(rb"<\s*init\b[^>]*>.*?</\s*init\s*>", b"", cleaned, count=1, flags=re.I | re.S)
        cleaned = re.sub(rb"<\s*header\b[^>]*>", b"", cleaned, count=1, flags=re.I)
        cleaned = re.sub(rb"</\s*header\s*>", b"", cleaned, count=1, flags=re.I)
        header_inner = cleaned

    header_inner = PATT_CDATA_LINE.sub(b"", header_inner)
    header_inner = _filter_template_header_xml_blocks(header_inner, template_header_keep_tags)
    header_inner = _rewrite_iseed_line(header_inner, seed)
    header_inner = header_inner.strip()
    return open_tag, (_ensure_trailing_newline(header_inner) if header_inner else b"")

# ============================
# LHE inspection / indexing
# ============================

def read_lhe_preamble(path: Path) -> LHEPreamble:
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            m_lhe = PATT_LHE_OPEN.search(mm)
            if not m_lhe:
                raise RuntimeError(f"{path} is missing <LesHouchesEvents>")
            open_tag = _ensure_trailing_newline(m_lhe.group(0))

            special_header_blocks = _extract_special_header_blocks(mm)

            m_init_open = PATT_INIT_OPEN.search(mm)
            m_init_close = PATT_INIT_CLOSE.search(mm)
            if not m_init_open or not m_init_close or m_init_close.start() < m_init_open.start():
                raise RuntimeError(f"{path} is missing a valid <init>...</init> block")
            init_block = bytes(mm[m_init_open.start():m_init_close.end()])
            init_block = _ensure_trailing_newline(init_block)
        finally:
            mm.close()

    return LHEPreamble(
        path=path,
        open_tag=open_tag,
        special_header_blocks=special_header_blocks,
        init_block=init_block,
    )

def iter_lhe_events(path: Path, file_idx: int) -> Iterator[EventRef]:
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            m_init_open = PATT_INIT_OPEN.search(mm)
            m_init_close = PATT_INIT_CLOSE.search(mm)
            if not m_init_open or not m_init_close or m_init_close.start() < m_init_open.start():
                raise RuntimeError(f"{path} is missing a valid <init>...</init> block")

            size = mm.size()
            pos = m_init_close.end()
            while True:
                mo = PATT_EVENT_OPEN.search(mm, pos)
                if not mo:
                    break
                s = mo.start()
                mc = PATT_EVENT_CLOSE.search(mm, s)
                if not mc:
                    raise RuntimeError(f"{path} has an <event> block without a closing </event>")
                e = mc.end()
                if e < size and mm[e:e + 1] == b"\n":
                    e += 1
                yield EventRef(file_idx=file_idx, start=s, end=e)
                pos = e
        finally:
            mm.close()

def index_lhe_file(path: Path, file_idx: int) -> IndexedLHE:
    preamble = read_lhe_preamble(path)
    events = list(iter_lhe_events(path, file_idx))
    return IndexedLHE(
        path=path,
        open_tag=preamble.open_tag,
        special_header_blocks=preamble.special_header_blocks,
        init_block=preamble.init_block,
        events=events,
    )

# ============================
# Header construction
# ============================

def build_output_header(
    header_template: Path,
    first_input_file: Union[IndexedLHE, LHEPreamble],
    seed: Optional[int],
    template_header_keep_tags: Optional[Sequence[str]] = None,
) -> Tuple[bytes, bytes]:
    template_bytes = header_template.read_bytes()
    template_open_tag, template_header_inner = _sanitize_header_template(
        template_bytes,
        seed,
        template_header_keep_tags=template_header_keep_tags,
    )

    open_tag = template_open_tag or first_input_file.open_tag or b'<LesHouchesEvents version="3.0">\n'

    header_parts: List[bytes] = []
    if template_header_inner:
        header_parts.append(_ensure_trailing_newline(template_header_inner))
    if first_input_file.special_header_blocks:
        header_parts.append(_ensure_trailing_newline(first_input_file.special_header_blocks))

    if header_parts:
        header_inner = b"".join(header_parts)
        header_block = b"<header>\n" + header_inner + b"</header>\n"
    else:
        header_block = b"<header/>\n"

    return open_tag, header_block

# ============================
# In-memory path
# ============================

def _split_chunks(arr: List[EventRef], k: int) -> List[List[EventRef]]:
    k = max(1, k)
    n = len(arr)
    base = n // k
    rem = n % k
    chunks: List[List[EventRef]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        end = start + size
        if size > 0:
            chunks.append(arr[start:end])
        start = end
    return chunks

def _copy_event_refs(input_paths: Sequence[str], refs: Sequence[EventRef], fout) -> None:
    handles = {}
    mmaps = {}
    try:
        for ref in refs:
            if ref.file_idx not in mmaps:
                fh = open(input_paths[ref.file_idx], "rb")
                handles[ref.file_idx] = fh
                mmaps[ref.file_idx] = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            fout.write(mmaps[ref.file_idx][ref.start:ref.end])
    finally:
        for mm in mmaps.values():
            mm.close()
        for fh in handles.values():
            fh.close()

def _write_part(input_paths: Sequence[str], refs: Sequence[EventRef], part_path: str) -> None:
    with open(part_path, "wb") as fout:
        _copy_event_refs(input_paths, refs, fout)

def write_randomized_events_memory(
    input_paths: Sequence[Path],
    refs: List[EventRef],
    output_path: Path,
    open_tag: bytes,
    header_block: bytes,
    init_block: bytes,
    seed: Optional[int],
    subset: Optional[int],
    workers: int,
    prefer_pigz: bool,
    gzip_level: int,
    verbose: bool,
) -> None:
    shuffled = list(refs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if subset is not None and subset < len(shuffled):
        shuffled = shuffled[:subset]

    str_paths = [str(p) for p in input_paths]

    if workers <= 1 or len(shuffled) == 0:
        with _open_output_stream(output_path, prefer_pigz=prefer_pigz, gzip_level=gzip_level, verbose=verbose) as fout:
            fout.write(open_tag)
            fout.write(header_block)
            fout.write(init_block)
            _copy_event_refs(str_paths, shuffled, fout)
            fout.write(b"</LesHouchesEvents>\n")
        return

    chunks = _split_chunks(shuffled, workers)
    part_paths = [output_path.with_suffix(output_path.suffix + f".part{i}") for i in range(len(chunks))]

    if verbose:
        print(f"      writing event chunks with {len(chunks)} thread worker(s)")

    try:
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(_write_part, str_paths, chunk, str(part)) for chunk, part in zip(chunks, part_paths)]
            for future in futures:
                future.result()

        with _open_output_stream(output_path, prefer_pigz=prefer_pigz, gzip_level=gzip_level, verbose=verbose) as fout:
            fout.write(open_tag)
            fout.write(header_block)
            fout.write(init_block)
            for part in part_paths:
                with open(part, "rb") as fp:
                    while True:
                        buf = fp.read(1024 * 1024)
                        if not buf:
                            break
                        fout.write(buf)
            fout.write(b"</LesHouchesEvents>\n")
    finally:
        for part in part_paths:
            try:
                part.unlink()
            except Exception:
                pass

# ============================
# Disk-backed scalable path
# ============================

def _pack_record(key: int, file_idx: int, start: int, end: int) -> bytes:
    return INDEX_RECORD.pack(key, file_idx, start, end)

def _unpack_record(blob: bytes) -> Tuple[int, int, int, int]:
    return INDEX_RECORD.unpack(blob)

def _flush_sorted_run(records: List[Tuple[int, int, int, int]], run_path: Path) -> None:
    records.sort()
    with open(run_path, "wb") as fout:
        for rec in records:
            fout.write(_pack_record(*rec))

def _iter_merged_run_records(run_paths: Sequence[Path]) -> Iterator[Tuple[int, int, int, int]]:
    files = [open(path, "rb") for path in run_paths]
    heap: List[Tuple[int, int, int, int, int]] = []
    try:
        for run_idx, fin in enumerate(files):
            blob = fin.read(INDEX_RECORD.size)
            if blob:
                key, file_idx, start, end = _unpack_record(blob)
                heapq.heappush(heap, (key, run_idx, file_idx, start, end))

        while heap:
            key, run_idx, file_idx, start, end = heapq.heappop(heap)
            yield key, file_idx, start, end
            blob = files[run_idx].read(INDEX_RECORD.size)
            if blob:
                next_key, next_file_idx, next_start, next_end = _unpack_record(blob)
                heapq.heappush(heap, (next_key, run_idx, next_file_idx, next_start, next_end))
    finally:
        for fin in files:
            fin.close()

def _copy_record_iter(
    input_paths: Sequence[str],
    records: Iterator[Tuple[int, int, int, int]],
    fout,
    limit: Optional[int] = None,
) -> int:
    handles = {}
    mmaps = {}
    written = 0
    try:
        for _, file_idx, start, end in records:
            if limit is not None and written >= limit:
                break
            if file_idx not in mmaps:
                fh = open(input_paths[file_idx], "rb")
                handles[file_idx] = fh
                mmaps[file_idx] = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            fout.write(mmaps[file_idx][start:end])
            written += 1
    finally:
        for mm in mmaps.values():
            mm.close()
        for fh in handles.values():
            fh.close()
    return written

def _build_external_shuffle_runs(
    input_paths: Sequence[Path],
    seed: Optional[int],
    subset: Optional[int],
    run_capacity: int,
    temp_dir: Path,
    verbose: bool,
) -> Tuple[List[Path], int]:
    rng = random.Random(seed)
    total_events = 0

    use_heap_sampling = subset is not None and subset >= 0 and subset <= run_capacity
    if use_heap_sampling:
        if verbose:
            print(f"      large-job path: using heap sampling for subset={subset}")
        heap: List[Tuple[int, int, int, int]] = []
        for file_idx, path in enumerate(input_paths):
            file_events = 0
            for ref in iter_lhe_events(path, file_idx):
                total_events += 1
                file_events += 1
                key = rng.getrandbits(64)
                item = (-key, ref.file_idx, ref.start, ref.end)
                if len(heap) < subset:
                    heapq.heappush(heap, item)
                elif subset > 0 and key < -heap[0][0]:
                    heapq.heapreplace(heap, item)
            if verbose:
                print(f"      {path.name}: {file_events} event(s)")

        selected = [(-neg_key, file_idx, start, end) for neg_key, file_idx, start, end in heap]
        selected.sort()
        run_path = temp_dir / "sampled_subset.run"
        _flush_sorted_run(selected, run_path)
        return [run_path], total_events

    if verbose:
        print(f"      large-job path: building external shuffle runs (run_capacity={run_capacity})")

    run_paths: List[Path] = []
    records: List[Tuple[int, int, int, int]] = []
    run_idx = 0

    for file_idx, path in enumerate(input_paths):
        file_events = 0
        for ref in iter_lhe_events(path, file_idx):
            total_events += 1
            file_events += 1
            records.append((rng.getrandbits(64), ref.file_idx, ref.start, ref.end))
            if len(records) >= run_capacity:
                run_path = temp_dir / f"shuffle_run_{run_idx:06d}.bin"
                _flush_sorted_run(records, run_path)
                run_paths.append(run_path)
                records = []
                run_idx += 1
        if verbose:
            print(f"      {path.name}: {file_events} event(s)")

    if records:
        run_path = temp_dir / f"shuffle_run_{run_idx:06d}.bin"
        _flush_sorted_run(records, run_path)
        run_paths.append(run_path)

    return run_paths, total_events

def write_randomized_events_external(
    input_paths: Sequence[Path],
    output_path: Path,
    open_tag: bytes,
    header_block: bytes,
    init_block: bytes,
    seed: Optional[int],
    subset: Optional[int],
    workers: int,
    run_capacity: int,
    verbose: bool,
    prefer_pigz: bool,
    gzip_level: int,
) -> int:
    if workers > 1 and verbose:
        print("      note: workers>1 is ignored in external mode; using a single writer")

    with tempfile.TemporaryDirectory(prefix="collect_events_") as tmp:
        temp_dir = Path(tmp)
        run_paths, total_events = _build_external_shuffle_runs(
            input_paths=input_paths,
            seed=seed,
            subset=subset,
            run_capacity=run_capacity,
            temp_dir=temp_dir,
            verbose=verbose,
        )

        target = total_events if subset is None else min(subset, total_events)
        records_iter = _iter_merged_run_records(run_paths) if run_paths else iter(())
        with _open_output_stream(output_path, prefer_pigz=prefer_pigz, gzip_level=gzip_level, verbose=verbose) as fout:
            fout.write(open_tag)
            fout.write(header_block)
            fout.write(init_block)
            _copy_record_iter([str(p) for p in input_paths], records_iter, fout, limit=target)
            fout.write(b"</LesHouchesEvents>\n")

    return total_events

# ============================
# Strategy selection / public API
# ============================

def _choose_mode(mode: str, input_paths: Sequence[Path]) -> str:
    mode = mode.lower()
    if mode not in {"auto", "memory", "external"}:
        raise RuntimeError(f"Unknown mode: {mode}")
    if mode != "auto":
        return mode

    total_bytes = sum(path.stat().st_size for path in input_paths)
    if len(input_paths) > AUTO_MAX_FILES_FOR_MEMORY:
        return "external"
    if total_bytes > AUTO_MAX_INPUT_BYTES_FOR_MEMORY:
        return "external"
    return "memory"

def collect_events(
    output: PathLike,
    header_template: PathLike,
    input_files: Sequence[PathLike],
    seed: Optional[int] = None,
    subset: Optional[int] = None,
    workers: int = 1,
    verbose: bool = False,
    mode: str = DEFAULT_MODE,
    external_run_capacity: int = EXTERNAL_RUN_RECORD_CAPACITY,
    prefer_pigz: bool = True,
    gzip_level: int = 6,
    template_header_keep_tags: Optional[Sequence[str]] = None,
) -> Path:
    """
    Collect LHE events from multiple files, shuffle them, and write a new LHE file.

    Parameters
    ----------
    output:
        Path to the output LHE file.
    header_template:
        Path to the separate template used to build the final <header> block.
    input_files:
        Sequence of input LHE files.
    seed:
        Shuffle seed. Also used to rewrite the iseed line in the template.
        If None, the template iseed line is rewritten to 0.
    subset:
        If given, keep only this many events after shuffling.
    workers:
        Number of parallel worker threads for event copying. Must be >= 1.
        Used only in memory mode.
    verbose:
        If True, print progress messages.
    mode:
        "auto", "memory", or "external".
    external_run_capacity:
        Number of shuffle-index records held in memory per external-sort run.
    prefer_pigz:
        If True and the output path ends in .gz, prefer pigz when available.
    gzip_level:
        Compression level used for .gz output (0-9).
    template_header_keep_tags:
        If given, keep only these XML blocks from the template header content.
        Non-XML text outside those blocks is preserved. Tag matching ignores
        namespaces and uses local tag names.

    Returns
    -------
    Path
        The resolved path of the written output file.
    """
    output_path = Path(output).resolve()
    header_template_path = Path(header_template).resolve()
    input_paths = [Path(x).resolve() for x in input_files]
    workers = max(1, workers)
    external_run_capacity = max(1, int(external_run_capacity))
    gzip_level = max(0, min(9, int(gzip_level)))

    if subset is not None and subset < 0:
        raise RuntimeError("subset must be >= 0")
    if not header_template_path.is_file():
        raise RuntimeError(f"Header template file not found: {header_template_path}")
    if not input_paths:
        raise RuntimeError("No input files provided.")
    for path in input_paths:
        if not path.is_file():
            raise RuntimeError(f"Input file not found: {path}")

    chosen_mode = _choose_mode(mode, input_paths)

    if verbose:
        total_bytes = sum(path.stat().st_size for path in input_paths)
        compression_note = "compressed" if _wants_gzip_output(output_path) else "uncompressed"
        print(
            f"[1/3] Preparing {len(input_paths)} input file(s) "
            f"(mode={chosen_mode}, requested={mode}, total_size={total_bytes} bytes, output={compression_note}) ..."
        )

    first_preamble = read_lhe_preamble(input_paths[0])

    if verbose:
        print("[2/3] Building output header and init block ...")

    open_tag, header_block = build_output_header(
        header_template=header_template_path,
        first_input_file=first_preamble,
        seed=seed,
        template_header_keep_tags=template_header_keep_tags,
    )
    init_block = first_preamble.init_block

    if chosen_mode == "memory":
        indexed_files: List[IndexedLHE] = []
        all_refs: List[EventRef] = []
        if verbose:
            print("[3/3] Indexing input files in memory ...")
        for i, path in enumerate(input_paths):
            item = index_lhe_file(path, i)
            indexed_files.append(item)
            all_refs.extend(item.events)
            if verbose:
                print(f"      {path.name}: {len(item.events)} event(s)")

        n_target = len(all_refs) if subset is None else min(subset, len(all_refs))
        if verbose:
            print(
                f"      writing {n_target} shuffled event(s) -> {output_path} "
                f"(workers={workers}, seed={seed})"
            )

        write_randomized_events_memory(
            input_paths=input_paths,
            refs=all_refs,
            output_path=output_path,
            open_tag=open_tag,
            header_block=header_block,
            init_block=init_block,
            seed=seed,
            subset=subset,
            workers=workers,
            prefer_pigz=prefer_pigz,
            gzip_level=gzip_level,
            verbose=verbose,
        )
    else:
        if verbose:
            print("[3/3] Using scalable disk-backed shuffle path ...")
        total_events = write_randomized_events_external(
            input_paths=input_paths,
            output_path=output_path,
            open_tag=open_tag,
            header_block=header_block,
            init_block=init_block,
            seed=seed,
            subset=subset,
            workers=workers,
            run_capacity=external_run_capacity,
            verbose=verbose,
            prefer_pigz=prefer_pigz,
            gzip_level=gzip_level,
        )
        n_target = total_events if subset is None else min(subset, total_events)
        if verbose:
            print(f"      wrote {n_target} shuffled event(s) -> {output_path} (seed={seed})")

    if verbose:
        print("Done.")

    return output_path

# ============================
# CLI
# ============================

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Collect LHE events from multiple files, shuffle them, and write a new LHE file. Events are copied verbatim."
    )
    ap.add_argument("-o", "--output", required=True, help="Output LHE file.")
    ap.add_argument("--header-template", required=True, help="Separate file used to build the output header.")
    ap.add_argument("--seed", type=int, default=None, help="Shuffle seed.")
    ap.add_argument("--subset", type=int, default=None, help="Keep only this many events after shuffling.")
    ap.add_argument("--workers", type=int, default=1, help="Parallel writer threads (>=1). Used only in memory mode.")
    ap.add_argument("--mode", choices=["auto", "memory", "external"], default=DEFAULT_MODE, help="Shuffle strategy.")
    ap.add_argument(
        "--external-run-capacity",
        type=int,
        default=EXTERNAL_RUN_RECORD_CAPACITY,
        help="Records per sorted run in external mode.",
    )
    ap.add_argument(
        "--no-pigz",
        action="store_true",
        help="Do not use pigz for .gz output; use Python gzip instead.",
    )
    ap.add_argument(
        "--gzip-level",
        type=int,
        default=6,
        help="Compression level for .gz output (0-9).",
    )
    ap.add_argument(
        "--template-header-tag",
        action="append",
        default=None,
        metavar="TAG",
        help=(
            "Keep only this XML block from the template header. "
            "Repeat to keep multiple block types. Non-XML text outside those blocks is preserved."
        ),
    )
    ap.add_argument("inputs", nargs="+", help="Input LHE file(s).")
    return ap.parse_args(argv)

def main(argv: Optional[Iterable[str]] = None) -> int:
    ns = parse_args(argv)
    collect_events(
        output=ns.output,
        header_template=ns.header_template,
        input_files=ns.inputs,
        seed=ns.seed,
        subset=ns.subset,
        workers=ns.workers,
        verbose=True,
        mode=ns.mode,
        external_run_capacity=ns.external_run_capacity,
        prefer_pigz=not ns.no_pigz,
        gzip_level=ns.gzip_level,
        template_header_keep_tags=ns.template_header_tag,
    )
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
