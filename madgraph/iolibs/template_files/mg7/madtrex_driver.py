#! /usr/bin/env python3
"""Standalone MadtRex reweighting driver.

Reweights the events of an MG7 process directory -- or of a gridpack, which
ships this file as bin/reweight together with the three *_reweight_card.py
modules it imports -- using only the madspace and madtrex packages: nothing
here imports MadGraph. mg7/madtrex.py is the MadGraph-side wrapper around
this module, which adds what does need MadGraph (regenerating the process
in another model for a "change model" reweight card).

Usage (from a process directory or gridpack, or see --process-dir):
    python madtrex_driver.py --param-rwgt-card param_reweight_card.dat
    bin/reweight --rwgt-card reweight_card.dat            # in a gridpack
"""

# annotations stay strings: some name madtrex classes (mtx.Driver), which
# Python < 3.14 would otherwise evaluate when this module is imported
from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Callable

# The madspace and madtrex installations sit next to each other in both
# layouts this file runs from: <MG>/{madspace,madtrex}/install, with this file
# in <MG>/madgraph/iolibs/template_files/mg7/, and <gridpack>/{madspace,
# madtrex}/install, with this file in <gridpack>/bin/. Use the first
# directory above this file that has either; failing that, whatever madspace/
# madtrex are already importable. Nothing here installs them.
_HERE = Path(__file__).resolve().parent


def _find_install_root() -> Path | None:
    for parent in _HERE.parents:
        if (parent / "madspace" / "install" / "madspace").is_dir() or (
            parent / "madtrex" / "install" / "madtrex"
        ).is_dir():
            return parent
    return None


def _ensure_gridpack_madtrex() -> None:
    """In a gridpack made with include_madtrex_source (this file is its
    bin/reweight, next to madtrex/install.py): build madtrex from those
    sources when the gridpack has no madtrex install, or rebuild it when the
    install was compiled against another madspace (e.g. a madspace the
    gridpack rebuilt on this machine). Elsewhere, nothing."""
    root = _HERE.parent
    installer = root / "madtrex" / "install.py"
    if _HERE.name != "bin" or not installer.is_file():
        return
    package = root / "madtrex" / "install" / "madtrex"
    madspace_package = root / "madspace" / "install" / "madspace"
    clean = False
    if not package.is_dir():
        reason = "this gridpack has no madtrex install"
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_madtrex_compat",
                                                      package / "_compat.py")
        if spec is None or not madspace_package.is_dir():
            return
        compat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compat)
        if not compat.build_problem(str(package), str(madspace_package)):
            return
        reason = "this gridpack's madtrex was built against another madspace"
        clean = True
    if not (madspace_package / "include").is_dir():
        raise RuntimeError(
            f"{reason}, and it cannot be built from the included sources: its "
            "madspace has no headers (a precompiled copy). Make the gridpack "
            "with include_madspace = false and include_madspace_source = true, "
            "so that madspace is built here too.")
    print(f"{reason}: building it from the included sources (madtrex/install.py)")
    command = [sys.executable, str(installer), "--yes"] + (["--clean"] if clean else [])
    if subprocess.run(command).returncode != 0:
        raise RuntimeError("madtrex installation failed: see the output above")


_ensure_gridpack_madtrex()
_INSTALL_ROOT = _find_install_root()
if _INSTALL_ROOT is not None:
    for _dir, _pkg in (
        (_INSTALL_ROOT / "madtrex" / "install", "madtrex"),
        (_INSTALL_ROOT / "madspace" / "install", "madspace"),
    ):
        if (_dir / _pkg).is_dir() and str(_dir) not in sys.path:
            sys.path.insert(0, str(_dir))

_MISSING_MESSAGE = (
    "MadtRex reweighting needs the madspace and madtrex packages (madtrex "
    "is installed separately, on top of madspace: python madtrex/install.py "
    "in MadGraph, and include_madtrex in the run_card for a gridpack), or "
    "both on PYTHONPATH. ({})"
)


class _MissingMadtrex:
    """Stands for the madtrex module when it is not installed: this module
    still imports (MadGraph imports it to choose and drive the reweighting),
    and using madtrex raises the ImportError saying how to install it."""

    def __init__(self, reason: str):
        self._reason = reason

    def __getattr__(self, name):
        raise ImportError(_MISSING_MESSAGE.format(self._reason))


try:
    import madspace as ms
except ImportError as _err:
    raise ImportError(_MISSING_MESSAGE.format(_err)) from _err
try:
    import madtrex as mtx
    # with the MadGraph directory on sys.path, the madtrex/ source directory
    # imports as an empty namespace package when madtrex is not installed
    if not hasattr(mtx, "Driver"):
        raise ImportError("madtrex is not installed (no madtrex.Driver in %s)"
                          % list(getattr(mtx, "__path__", [])))
except ImportError as _err:
    mtx = _MissingMadtrex(str(_err))


def _require_madtrex() -> None:
    """Raises the ImportError of a missing madtrex, before any work."""
    if isinstance(mtx, _MissingMadtrex):
        mtx.Driver  # raises

if __package__:
    from . import param_reweight_card as param_rwgt_card
    from . import pdf_reweight_card as pdf_rwgt_card
    from . import generic_reweight_card as generic_rwgt_card
else:
    if str(_HERE) not in sys.path:
        sys.path.append(str(_HERE))
    import param_reweight_card as param_rwgt_card
    import pdf_reweight_card as pdf_rwgt_card
    import generic_reweight_card as generic_rwgt_card

# Regenerates a process in another model; see run_parameter_reweighting()'s
# regenerate argument and mg7/madtrex.py's regenerate_process_in_model().
Regenerator = Callable[..., str]

logger = logging.getLogger("madgraph7")


def find_latest_run_dir(events_dir: str = "Events") -> str:
    """Returns the most recently modified Events/<run_name>_NN run directory."""
    run_dirs = glob.glob(os.path.join(events_dir, "*_[0-9][0-9]"))
    if not run_dirs:
        raise FileNotFoundError(f"no run directories found under {events_dir}")
    return max(run_dirs, key=os.path.getmtime)


def find_run_events(run_dir: str) -> str:
    """Returns run_dir's generated events file, preferring LHEF XML over the
    internal binary format (see launch.py's generate_events, which writes
    events.lhe.gz for output_format="lhe" and events.npy for "lhe_npy")."""
    for name in ("events.lhe", "events.lhe.gz", "events.npy"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"no events.lhe(.gz) or events.npy found in {run_dir}")


def prepare_events(events_path: str) -> str:
    """Returns a path MadtRex can read events_path from: the MadtRex readers
    don't handle gzip, so a gzipped LHE file (launch.py gzips events.lhe by
    default) is decompressed once into a sibling file without the .gz suffix,
    reused on later calls. Any other path is returned unchanged."""
    if not events_path.endswith(".gz"):
        return events_path
    plain_path = events_path[: -len(".gz")]
    if not os.path.isfile(plain_path) or os.path.getmtime(plain_path) < os.path.getmtime(events_path):
        logger.info("MadtRex: decompressing %s -> %s", events_path, plain_path)
        with gzip.open(events_path, "rb") as src, open(plain_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return plain_path


def read_init_node(events_path: str) -> "ms.LHEMeta | None":
    """Reads the <init> block launch.py's MadgraphProcess.write_lhe_header()
    writes as header.lhe next to events_path (or, for runs made before that
    existed, the init.lhe the old madevent.py wrote), if present, into an
    LHEMeta that
    Driver.load_events() can use to seed the reweighted sample's baseline
    cross section and error. Needed for events.npy input (compact_npy and
    lhe_npy), which -- unlike events.lhe -- carries no <init> block of its
    own; without this, MadtRex reweighting has no baseline cross section to
    normalise or propagate errors against (see reweightor::calc_xErrs() /
    calc_norm() in Rex/src/teaRex.cc). Returns None if neither file is
    found, in which case load_events() falls back to its own empty-LHEMeta
    default."""
    run_dir = os.path.dirname(events_path)
    for name in ("header.lhe", "init.lhe"):
        init_path = os.path.join(run_dir, name)
        if os.path.isfile(init_path):
            break
    else:
        return None
    with open(init_path) as f:
        match = re.search(r"<init>\s*\n(.*?)</init>", f.read(), re.S)
    if not match:
        return None
    lines = [line.split() for line in match.group(1).strip().splitlines()]
    header, proc_lines = lines[0], lines[1:]

    def process(fields: list[str]) -> ms.LHEProcess:
        proc = ms.LHEProcess(
            cross_section=float(fields[0]), cross_section_error=float(fields[1]),
            process_id=int(fields[3]),
        )
        proc.max_weight = float(fields[2])  # constructor kwarg is misspelled in the binding
        return proc

    return ms.LHEMeta(
        beam1_pdg_id=int(header[0]), beam2_pdg_id=int(header[1]),
        beam1_energy=float(header[2]), beam2_energy=float(header[3]),
        beam1_pdf_authors=int(header[4]), beam2_pdf_authors=int(header[5]),
        beam1_pdf_id=int(header[6]), beam2_pdf_id=int(header[7]),
        weight_mode=int(header[8]),
        processes=[process(fields) for fields in proc_lines],
    )


def _pdf_data_path() -> str | None:
    """Resolves the LHAPDF data directory: $LHAPDF_DATA_PATH if set, else
    lhapdf.paths()[0] (the configured lhapdf's own data dir, eg lhapdf6 in
    HEPTools) if the lhapdf Python module is importable, else None -- same
    convention as launch.py's own PDF_PATH."""
    if "LHAPDF_DATA_PATH" in os.environ:
        return os.environ["LHAPDF_DATA_PATH"]
    try:
        import lhapdf
        lhapdf.setVerbosity(0)
        return lhapdf.paths()[0]
    except ImportError:
        return None


def resolve_pdf_set_files(pdf_set_name: str) -> tuple[str, str]:
    """Resolves pdf_set_name (an LHAPDF set -- the only kind PdfDriver
    currently supports) to its (grid_file, alpha_s_grid_file) paths, per the
    same $LHAPDF_DATA_PATH/lhapdf.paths()[0] convention launch.py's own
    init_beam()/ensure_pdf_set() use for the run_card's own [beam] pdf.
    Unlike ensure_pdf_set(), this never attempts to download a missing set:
    standalone reweighting has no run_card.toml/MADGRAPH_LHAPDF_CONFIG to
    drive a download from, so it only resolves paths, raising a clear error
    if pdf_set_name isn't already present on disk."""
    data_path = _pdf_data_path()
    if data_path is None:
        raise RuntimeError(
            f"cannot resolve PDF set {pdf_set_name!r}: no LHAPDF_DATA_PATH is set and the "
            f"lhapdf Python module is not importable"
        )
    set_dir = os.path.join(data_path, pdf_set_name)
    if not os.path.isdir(set_dir):
        raise RuntimeError(
            f"PDF set {pdf_set_name!r} not found under {data_path!r}; install it (eg via "
            f"MadGraph's `install` command, or lhapdf's own `lhapdf install`) before "
            f"reweighting against it"
        )
    return (
        os.path.join(set_dir, f"{pdf_set_name}_0000.dat"),
        os.path.join(set_dir, f"{pdf_set_name}.info"),
    )


def default_output_path(events_path: str) -> str:
    """A path next to events_path for writing the reweighted events
    elsewhere (the runners themselves write in place by default)."""
    root, ext = os.path.splitext(events_path)
    if ext == ".npy":
        # write_weights() targets save_weights_binary() for binary input,
        # which treats its path argument as a prefix, writing one
        # "<prefix>_<weight_id>.npy" file per recorded weight -- so the
        # default must not carry the .npy extension itself.
        return f"{root}_reweighted"
    return f"{root}_reweighted{ext}"


# The [run]/[beam] run_card.toml defaults this module relies on, as declared
# by RunCardMG7 in madgraph/various/banner.py -- read with tomllib here so
# reweighting doesn't need MadGraph's card machinery. MadGraph always writes
# every entry out, so these only matter for hand-written cards.
_RUN_CARD_DEFAULTS = {
    "run": {"device": ["cpu"], "cpu_mode": "auto", "precision": "color32"},
    "beam": {
        "leptonic": False,
        "pdf1": "NNPDF40_lo_as_01180",
        "pdf2": "NNPDF40_lo_as_01180",
    },
}


def _run_card_path(process_directory: str) -> str:
    """The card the matrix elements' [run] settings are read from: in a
    gridpack, Cards/grid_run_card.toml, which (unlike the full run_card.toml
    shipped next to it) names the cpu backend the shipped libraries were
    actually built for rather than 'auto'; else Cards/run_card.toml."""
    grid_card = os.path.join(process_directory, "Cards", "grid_run_card.toml")
    if os.path.isfile(grid_card):
        return grid_card
    return os.path.join(process_directory, "Cards", "run_card.toml")


def _read_run_card(path: str) -> dict:
    """The [run] and [beam] sections of the run_card.toml at path, with
    _RUN_CARD_DEFAULTS filled in for missing entries."""
    with open(path, "rb") as f:
        return _fill_run_card(tomllib.load(f))


def _fill_run_card(card: dict) -> dict:
    return {
        section: {**defaults, **card.get(section, {})}
        for section, defaults in _RUN_CARD_DEFAULTS.items()
    }


_RUN_CARD_BLOCK_RE = re.compile(r"<MG7RunCard>(.*?)</MG7RunCard>", re.S)


def _read_events_header_block(events_path: str, block_re: re.Pattern) -> str | None:
    """The content of the first block_re match in the LHE header of the
    events of events_path -- header.lhe next to an events.npy, else the LHE
    file's own header -- CDATA-unwrapped or unescaped, or None."""
    if events_path.endswith(".npy"):
        header_path = os.path.join(os.path.dirname(events_path), "header.lhe")
    else:
        header_path = events_path
    if not os.path.isfile(header_path):
        return None
    opener = gzip.open if header_path.endswith(".gz") else open
    lines = []
    try:
        with opener(header_path, "rt", errors="replace") as f:
            for line in f:  # the header only: stop at <init> or the first event
                if "<init" in line or "<event" in line:
                    break
                lines.append(line)
    except OSError:
        return None
    match = block_re.search("".join(lines))
    if not match:
        return None
    content = match.group(1).strip()
    if content.startswith("<![CDATA[") and content.endswith("]]>"):
        return content[len("<![CDATA["):-len("]]>")]
    import html
    return html.unescape(content)


def _read_events_run_card(events_path: str) -> dict | None:
    """The run_card.toml the events of events_path were generated with --
    the <MG7RunCard> block launch.py (and a gridpack) write into the LHE
    header (see _read_events_header_block()) -- parsed as _read_run_card()
    does, or None when there is none. This, not the process directory's
    current Cards/run_card.toml (which may have been edited since, e.g. for
    another run), is what the sample was made with."""
    content = _read_events_header_block(events_path, _RUN_CARD_BLOCK_RE)
    if content is None:
        return None
    try:
        return _fill_run_card(tomllib.loads(content))
    except tomllib.TOMLDecodeError:
        return None


_SLHA_BLOCK_RE = re.compile(r"<slha>(.*?)</slha>", re.S)


def _read_events_param_card(events_path: str) -> str | None:
    """The param_card the events of events_path were generated with: the
    <slha> block of their LHE header (see _read_events_header_block()), as
    launch.py's write_lhe_header() and madevent write it -- with the widths
    "auto" resolved to the values the generation used. None without one."""
    content = _read_events_header_block(events_path, _SLHA_BLOCK_RE)
    if content is None or not content.strip():
        return None
    return content.strip() + "\n"


def _param_card_values(text: str) -> dict[tuple, float | None]:
    """The numbers of an SLHA param_card, keyed (block, ids...): "DECAY pid
    width" as ("decay", pid). Comments, case and formatting do not matter;
    branching ratios are left out (the matrix elements do not depend on
    them), a value that is no number ("auto") is None."""
    values: dict[tuple, float | None] = {}
    block = None
    for raw in text.splitlines():
        words = raw.split("#", 1)[0].split()
        if not words:
            continue
        head = words[0].lower()
        if head == "block":
            block = words[1].lower() if len(words) > 1 else None
            continue
        if head == "decay":
            block = None
            try:
                key = ("decay", int(words[1]))
            except (IndexError, ValueError):
                continue
            try:
                values[key] = float(words[2])
            except (IndexError, ValueError):
                values[key] = None
            continue
        if block is None:
            continue
        try:
            ids = tuple(int(word) for word in words[:-1])
        except ValueError:
            continue
        try:
            values[(block,) + ids] = float(words[-1])
        except ValueError:
            values[(block,) + ids] = None
    return values


def _param_card_differences(generation_text: str, other_text: str,
                            rel_tol: float = 1e-6) -> list[str]:
    """The parameters other_text gives another value than generation_text
    (numerically, to rel_tol; a value that is no number, eg an "auto"
    width, is not compared), as "BLOCK IDS: generation -> other"."""
    import math
    generation, other = _param_card_values(generation_text), _param_card_values(other_text)
    differences = []
    for key in sorted(set(generation) | set(other), key=str):
        a, b = generation.get(key, "missing"), other.get(key, "missing")
        if a is None or b is None:
            continue
        if isinstance(a, float) and isinstance(b, float) and math.isclose(a, b, rel_tol=rel_tol, abs_tol=0.0):
            continue
        name = " ".join(str(part) for part in key)
        differences.append(f"{name}: {a} -> {b}")
    return differences


def _generation_param_card(process_directory: str, events_path: str, param_card: str,
                           card_dir: str) -> str:
    """The param_card to normalise the events of events_path with: the one
    they were generated with, from their LHE header, written into card_dir;
    a warning lists the parameters in which param_card (default
    process_directory/Cards/param_card.dat) differs from it. Without one in
    the header, param_card, with a warning that the normalisation then
    cannot be checked against the generation."""
    current = param_card or os.path.join(process_directory, "Cards", "param_card.dat")
    text = _read_events_param_card(events_path)
    if text is None:
        logger.warning(
            "MadtRex: the events of %s carry no param_card (no <slha> block in their "
            "header): they are normalised with %s, which cannot be checked against the "
            "one they were generated with", events_path, current)
        return current
    path = os.path.join(card_dir, "generation_param_card.dat")
    with open(path, "w") as f:
        f.write(text)
    if os.path.isfile(current):
        with open(current) as f:
            differences = _param_card_differences(text, f.read())
        if differences:
            shown = differences[:10] + ([f"... {len(differences) - 10} more"]
                                        if len(differences) > 10 else [])
            logger.warning(
                "MadtRex: %s differs from the param_card the events of %s were generated "
                "with (%s); the reweighting starts from the latter, as the events were "
                "generated, not from %s", current, events_path, "; ".join(shown), current)
    return path


def _check_stored_matrix_elements(events_path: str, init_meta, process_directory: str,
                                  param_card: str, devices: list[str], fptype: str,
                                  sample: int = 32, rel_tol: float = 1e-6,
                                  card_label: str = "") -> None:
    """Re-evaluates the original matrix element (helicity-summed, with
    param_card, at fptype) for a sample of the events of events_path whose
    generation-time matrix elements are stored (STORED_MATRIX_ELEMENT_COLUMN)
    and warns if they differ by more than rel_tol: the stored ones are then
    not those of this process directory's libraries and card. A launch
    changing nothing normalised by the stored ones gives the ratio of the
    two as its weight over the event weight. rel_tol leaves room for an MLM
    run's stored matrix elements, which are the first evaluation rescaled to
    the event's alpha_s rather than evaluated again (about 1e-7 apart at the
    mixed fptype)."""
    import numpy as np
    if not events_path.endswith(".npy"):
        return
    events = np.load(events_path, mmap_mode="r", max_header_size=1 << 30)
    if STORED_MATRIX_ELEMENT_COLUMN not in (events.dtype.names or ()) or len(events) == 0:
        return
    picked = np.unique(np.linspace(0, len(events) - 1, min(sample, len(events))).astype(int))
    with tempfile.TemporaryDirectory() as tmp:
        sample_path = os.path.join(tmp, "events.npy")
        np.save(sample_path, np.ascontiguousarray(events[picked]))
        card = os.path.join(tmp, "check.rwgt")
        with open(card, "w") as f:
            f.write("launch --rwgt_name=check\n")
        try:
            driver = mtx.Driver()
            driver.set_helicity_reweighting(False)
            driver.set_device_priority(devices)
            driver.set_fptype_priority([fptype])
            driver.load_process_directory(process_directory, param_card)
            driver.load_param_reweighting(card)
            if init_meta is not None:
                driver.load_events(sample_path, init_meta)
            else:
                driver.load_events(sample_path)
            driver.warehouse().get().run()
            driver.write_weights(os.path.join(tmp, "check"))
            weights = np.load(os.path.join(tmp, "check_check.npy"))["weight"]
        except Exception as error:  # a check only: the reweighting itself reports
            logger.warning("MadtRex: could not check the stored matrix elements of %s "
                           "against %s: %s", events_path, process_directory, error)
            return
    original = np.asarray(events["weight"][picked])
    nonzero = original != 0
    deviation = np.abs(weights[nonzero] / original[nonzero] - 1)
    if deviation.size and deviation.max() > rel_tol:
        logger.warning(
            "MadtRex: the matrix elements stored with %s differ from the ones the "
            "libraries of %s give with %s (up to %.3g relative, %d of %d sampled events "
            "beyond %g): helicity-summed launches are normalised with the stored ones, "
            "the others with re-evaluated ones", events_path, process_directory,
            card_label or param_card, deviation.max(), int(np.sum(deviation > rel_tol)),
            deviation.size, rel_tol)


def _generation_run_card(process_directory: str, events_path: str | None,
                         card_path: str) -> tuple[dict | None, str]:
    """(run card, where it was read) for the sample of events_path: the run
    card in the events' header (see _read_events_run_card()), else the card
    at card_path in the process directory, else (None, '')."""
    if events_path is not None:
        card = _read_events_run_card(events_path)
        if card is not None:
            return card, "the header of " + events_path
    if os.path.isfile(card_path):
        try:
            return _read_run_card(card_path), card_path
        except tomllib.TOMLDecodeError:
            pass
    return None, ""


def reweighting_threads(process_directory: str, events_path: str | None = None) -> int:
    """The worker threads MadtRex reweights with unless the card says
    otherwise ("change nb_core"): as many as the event generation uses, the
    [run] cpu_thread_pool_size of the process directory's run card (else of
    the run card in the header of events_path), all available cores when it
    is -1 or 0 (as launch.py and madspace resolve it) or cannot be read."""
    card, _ = _generation_run_card(process_directory, None, _run_card_path(process_directory))
    if card is None and events_path is not None:
        card = _read_events_run_card(events_path)
    try:
        threads = int(card["run"].get("cpu_thread_pool_size", -1)) if card else -1
    except (TypeError, ValueError):
        threads = -1
    if threads > 0:
        return threads
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def resolve_auto_backend(build_path: str) -> str:
    """Ask the matrix-element Makefile to resolve the ``auto`` cpu_mode (as
    resolve_auto_backend() in launch.py, which this module can't import
    without MadGraph)."""
    try:
        result = subprocess.run(
            ["make", "-n", "BACKEND=auto", "detect-backend"],
            cwd=build_path,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Could not resolve cpu_mode='auto' in '{build_path}': {exc}"
        ) from exc
    match = re.search(r"^BACKEND=(\S+) \(was auto\)$", result.stdout, re.MULTILINE)
    if match is None or match.group(1) == "auto":
        raise RuntimeError(
            f"Could not resolve cpu_mode='auto' in '{build_path}': "
            f"make failed to report a backend.\n{result.stdout}{result.stderr}"
        )
    return match.group(1)


def _make(args: list[str], cwd: str) -> None:
    """Runs make with args in cwd, raising with its output on failure."""
    result = subprocess.run(
        ["make", *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if result.returncode != 0:
        output = result.stdout.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"'make {' '.join(args)}' failed in {cwd}:\n{output}"
        )


def _read_devices(process_directory: str) -> list[str]:
    """The matrix-element build backends (the BACKEND= / {device} values) of
    process_directory's run_card.toml: [run] device mapped through cpu_mode
    as launch.py's backend_of() does, with cpu_mode 'auto' resolved to the
    concrete SIMD backend so it matches the library names on disk."""
    run_card = _read_run_card(_run_card_path(process_directory))
    device_names = run_card["run"]["device"]
    if not isinstance(device_names, list):
        device_names = [device_names]
    cpu_mode = run_card["run"]["cpu_mode"]
    backends = []
    for name in device_names:
        device_type = name.split(":", 1)[0]
        backends.append(cpu_mode if device_type == "cpu" else device_type)
    if "auto" in backends:
        with open(os.path.join(process_directory, "SubProcesses", "subprocesses.json")) as f:
            first_proc_path = json.load(f)[0]["path"]
        if not os.path.isabs(first_proc_path):
            first_proc_path = os.path.join(process_directory, first_proc_path)
        auto_backend = resolve_auto_backend(first_proc_path)
        backends = [auto_backend if b == "auto" else b for b in backends]
    return list(dict.fromkeys(backends))


def _read_default_pdf_set(process_directory: str, events_path: str | None = None
                          ) -> tuple[str | None, str]:
    """(PDF set, where it was read): [beam] pdf1 of the run card the events
    of events_path were generated with -- the <MG7RunCard> of their LHE
    header (see _read_events_run_card()), else process_directory/Cards/
    run_card.toml. This is run_pdf_reweighting()'s baseline PDF set when the
    PDF reweight card names none: the PDF set the sample's own weights
    correspond to. The header comes first because the current run_card may
    have been edited since the sample was made (another run with another
    PDF), which would scale every weight by the wrong ratio.

    PdfDriver needs the same set on both beams: a run with pdf1 != pdf2, a
    leptonic run, or no run card at all gives (None, ...), and the card then
    has to name its baseline PDF_SET explicitly."""
    card, source = _generation_run_card(
        process_directory, events_path,
        os.path.join(process_directory, "Cards", "run_card.toml"))
    if card is None:
        return None, ""
    try:
        beam = card["beam"]
        if beam["leptonic"] or beam["pdf1"] != beam["pdf2"]:
            return None, source
        pdf_set = beam["pdf1"]
    except (KeyError, TypeError):
        return None, source
    return (pdf_set or None), source


# [run] precision -> FPTYPE, as launch.py's fptype_of()
_PRECISION_2_FPTYPE = {"all64": "d", "all32": "f", "color32": "m", "denom64": "v"}


def _read_run_fptype(process_directory: str, events_path: str | None = None) -> str | None:
    """The FPTYPE the sample of events_path was generated (and its matrix
    elements compiled) at: [run] precision of the run card in the events'
    LHE header (see _read_events_run_card()), else of the process
    directory's run card (Cards/grid_run_card.toml in a gridpack), mapped as
    launch.py's fptype_of() does. Used by run_parameter_reweighting() as

    - the default *target* fptype (param_rwgt_card.group_by_fptype()'s
      default_fptype) when a reweight card sets no "change fptype", so a
      plain reweight stays at the sample's own precision;
    - the *normaliser* fptype (Driver.set_normaliser_fptype_priority()),
      always: a launch is normalised against the sample's own configuration,
      never against the fptype it is reweighted to.

    None when there is no run card to read; callers then fall back to
    param_rwgt_card.DEFAULT_FPTYPE."""
    card, _ = _generation_run_card(process_directory, events_path,
                                   _run_card_path(process_directory))
    if card is None:
        return None
    try:
        precision = card["run"]["precision"]
    except (KeyError, TypeError):
        return None
    if not precision:
        return None
    precision = str(precision).lower()
    return _PRECISION_2_FPTYPE.get(precision, precision)


def _read_libtag(directory: str) -> str:
    """The madmatrix LIBTAG directory's libraries are built with: "" but for
    a process MadtRex regenerated in another model (mg7/madtrex.py's
    regenerate_process_in_model() records it in its marker file)."""
    marker = os.path.join(directory, ".madtrex_source.json")
    if not os.path.isfile(marker):
        return ""
    with open(marker) as f:
        return json.load(f).get("libtag", "")


def _check_distinct_libraries(directory: str, original_directory: str) -> None:
    """Raises unless no madmatrix library of directory (a process
    regenerated in another model) has the name of one of
    original_directory's: the dynamic loader identifies a library by its
    name, so one of the same name, loaded next to the original's for the
    normalisation, would silently be the original's (its parameters and
    all). See LIBTAG in madmatrix.mk."""
    def names(d):
        lib = os.path.join(d, "lib")
        return {n for n in (os.listdir(lib) if os.path.isdir(lib) else [])
                if n.startswith("libmadmatrix") and n.endswith(".so")}
    shared = names(directory) & names(original_directory)
    if shared:
        raise RuntimeError(
            f"the matrix element libraries of {directory} have the names of the original "
            f"process's ({', '.join(sorted(shared))}), so they cannot be loaded next to them "
            f"for the normalisation: remove {directory} to have it regenerated"
        )


def _compile_subprocesses(directory: str, devices: list[str], fptype: str) -> None:
    """Compiles whichever subprocesses under
    directory/SubProcesses/subprocesses.json aren't already compiled for
    fptype, for each of devices (BACKEND=<device> FPTYPE=<fptype>) --
    mirrors the existence-checked per-device compile in launch.py's
    MadgraphProcess.compile_matrix_elements, generalized to whichever fptype a reweight
    card's "set FPTYPE ..." (see reweight_card.py's group_by_fptype()) or a
    caller actually asked for. Safe (a fast no-op) to call for a
    device/fptype combination that's already built, so callers needn't
    track that themselves."""
    with open(os.path.join(directory, "SubProcesses", "subprocesses.json")) as f:
        subprocesses = json.load(f)
    for entry in subprocesses:
        for device in devices:
            expected_path = os.path.join(directory, entry["me_path"].format(device=device, fptype=fptype))
            if os.path.isfile(expected_path):
                continue
            source_path = os.path.join(directory, entry["path"])
            if not os.path.isfile(os.path.join(source_path, "makefile")) and not os.path.isfile(
                os.path.join(source_path, "Makefile")
            ):
                raise RuntimeError(
                    f"{expected_path} is not available and there is no source to compile "
                    f"it from in {source_path} (a gridpack ships the sources only with "
                    f"include_source = true): reweight at the precision the libraries "
                    f"were built with, or remake the gridpack"
                )
            logger.info(
                "MadtRex: compiling %s for device '%s', fptype '%s'",
                entry["path"], device, fptype,
            )
            # madmatrix.mk's own USEBUILDDIR=1 build-directory tag
            # (BACKEND+FPTYPE+HELINL+HRDCOD) refuses to build a *different*
            # tag while another tag's now-stale intermediate .o files are
            # still sitting in build.<device>/ from an earlier compile (eg
            # the original generation's own FPTYPE, or a previous reweight
            # card's) -- "Cannot build for tag=... as old builds exist for
            # other tags ...: Please run 'make clean' first". Harmless to
            # always do so before compiling here: this branch only runs
            # when expected_path doesn't exist yet, ie a real compile is
            # needed regardless, and the *already-built* .so for whatever
            # other tag was there previously lives in lib/, untouched by
            # 'clean' removing only build.<device>/'s own intermediates.
            # 'clean' here (madmatrix.mk) also recurses into the shared
            # directory/src/ common-library build (fixed to do so under
            # USEBUILDDIR=1 too -- it used to only clean src/ that way for
            # the non-USEBUILDDIR path), so this one call covers both.
            variables = [f"BACKEND={device}", f"FPTYPE={fptype}", "USEBUILDDIR=1"]
            libtag = _read_libtag(directory)
            if libtag:
                variables.append(f"LIBTAG={libtag}")
            _make(["clean"] + variables, cwd=source_path)
            _make(variables, cwd=source_path)


def _sanitize_dirname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_") or "model"


def _replace_atomically(new_path: str, target: str) -> None:
    """Moves new_path onto target, gzipping it first when target is a .gz
    file; new_path is next to target, so the final rename is atomic."""
    if target.endswith(".gz"):
        tmp = target + ".tmp"
        with open(new_path, "rb") as src, gzip.open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(new_path)
        os.replace(tmp, target)
    else:
        os.replace(new_path, target)


def _npy_weight_column(weight_id: str) -> str:
    return f"rwgt_{weight_id}"


def _merge_npy_weights(events_path: str, prefix: str, weight_ids: list[str],
                       weight_contexts: list[str], drop_columns=()) -> None:
    """Adds the weight files "<prefix>_<id>.npy" of the new weight_ids to
    events_path as "rwgt_<id>" columns, after its existing rwgt_ columns
    (the systematics' rwgt_<n>, a previous reweighting's), and lists them
    in a weight group of the header.lhe next to it, which npy_to_lhe and the
    MadtRex readers take the weight names from. The files are removed. The
    drop_columns of events_path are left out of the rewritten file."""
    import numpy as np
    from numpy.lib import format as npy_format

    events = np.load(events_path, mmap_mode="r", max_header_size=1 << 30)
    names = [name for name in events.dtype.names if name not in drop_columns]
    columns = {}
    for weight_id in weight_ids:
        column = _npy_weight_column(weight_id)
        if column in names:
            raise ValueError(
                f"{events_path} already has a weight {weight_id!r}: give the reweight "
                f"card's launches other names (launch --rwgt_name=...)"
            )
        data = np.load(f"{prefix}_{weight_id}.npy")
        columns[column] = data["weight"] if data.dtype.names else data
        if len(columns[column]) != len(events):
            raise ValueError(f"{prefix}_{weight_id}.npy does not have one weight per event")
    # the new columns go where madspace puts weight columns: after the last
    # rwgt_ column, else before the bias and the particles
    position = max((i + 1 for i, name in enumerate(names) if name.startswith("rwgt_")),
                   default=None)
    if position is None:
        position = next((i for i, name in enumerate(names)
                         if name == "bias" or name.startswith("part")), len(names))
    descr = [(name, events.dtype.fields[name][0]) for name in names]
    descr[position:position] = [(column, "<f8") for column in columns]
    tmp = events_path + ".tmp"
    out = npy_format.open_memmap(tmp, mode="w+", dtype=np.dtype(descr), shape=events.shape)
    chunk = 100000
    for start in range(0, len(events), chunk):
        stop = min(start + chunk, len(events))
        for name in names:
            out[name][start:stop] = events[name][start:stop]
        for column, data in columns.items():
            out[column][start:stop] = data[start:stop]
    out.flush()
    del out, events
    os.replace(tmp, events_path)
    for weight_id in weight_ids:
        os.remove(f"{prefix}_{weight_id}.npy")

    header = os.path.join(os.path.dirname(events_path), "header.lhe")
    if os.path.isfile(header):
        with open(header) as f:
            text = f.read()
        group = "<weightgroup name='tearex_reweighting' weight_name_strategy='includeIdInWeightName'>\n"
        group += "".join(f"<weight id='{weight_id}'>{context}</weight>\n"
                         for weight_id, context in zip(weight_ids, weight_contexts))
        group += "</weightgroup>\n"
        if "</initrwgt>" in text:
            text = text.replace("</initrwgt>", group + "</initrwgt>", 1)
        else:
            text = text.replace("</header>", f"<initrwgt>\n{group}</initrwgt>\n</header>", 1)
        with open(header + ".tmp", "w") as f:
            f.write(text)
        os.replace(header + ".tmp", header)


# the generation-time matrix elements MadGraph stores in events.npy for
# MadtRex's normalisation (launch.py's MadgraphProcess.store_matrix_elements)
STORED_MATRIX_ELEMENT_COLUMN = "matrix_element"


def write_weights_in_place(driver, result, source_path: str, events_path: str,
                           drop_stored_matrix_elements: bool = False) -> str:
    """Writes the weights driver (a Driver or PdfDriver, whose reweighted
    sample is result) computed into the event file they were computed for,
    and returns its path: source_path, the run's event file, as the other
    tools (MadSpin, the shower, npy_to_lhe) and the classic reweighting
    module use it. An LHE file is rewritten through a temporary file next
    to it (gzipped again when it was) and replaced atomically, the
    decompressed copy events_path read from removed; an npy file gets the
    new weights as rwgt_<id> columns (see _merge_npy_weights()). Either
    way the existing weights (systematics, earlier reweightings) are kept,
    and the new ones follow them. drop_stored_matrix_elements removes the
    stored matrix elements (STORED_MATRIX_ELEMENT_COLUMN) from an npy file in
    the same rewrite: for the last reweighting pass over it."""
    run_dir = os.path.dirname(os.path.abspath(source_path))
    if source_path.endswith(".npy"):
        prefix = os.path.join(run_dir, ".madtrex_weights")
        existing = set(_npy_existing_weight_ids(source_path))
        driver.write_weights(prefix)
        ids = list(result.weight_ids)
        contexts = list(getattr(result, "weight_context", [])) or [""] * len(ids)
        new = [(i, c) for i, c in zip(ids, contexts + [""] * len(ids)) if i not in existing]
        for weight_id in ids:
            if weight_id in existing and os.path.isfile(f"{prefix}_{weight_id}.npy"):
                os.remove(f"{prefix}_{weight_id}.npy")
        _merge_npy_weights(source_path, prefix, [i for i, _ in new], [c for _, c in new],
                           drop_columns=(STORED_MATRIX_ELEMENT_COLUMN,)
                           if drop_stored_matrix_elements else ())
        return source_path
    tmp = os.path.join(run_dir, ".madtrex_reweighted.lhe")
    driver.write_weights(tmp)
    _replace_atomically(tmp, source_path)
    if os.path.abspath(events_path) != os.path.abspath(source_path) and os.path.isfile(events_path):
        os.remove(events_path)  # the decompressed copy, stale now
    return source_path


def _npy_existing_weight_ids(events_path: str) -> list[str]:
    """The weight ids events_path already has columns for (rwgt_<id>)."""
    import numpy as np
    events = np.load(events_path, mmap_mode="r", max_header_size=1 << 30)
    return [name[len("rwgt_"):] for name in events.dtype.names if name.startswith("rwgt_")]


def _tag_output_path(output_path: str, tag: str) -> str:
    root, ext = os.path.splitext(output_path)
    return f"{root}_{tag}{ext}"


def run_parameter_reweighting(
    rwgt_path: str,
    events_path: str | None = None,
    output_path: str | None = None,
    process_directory: str = ".",
    param_card: str = "",
    regenerate: Regenerator | None = None,
    helicity_reweighting: bool | None = None,
    drop_stored_matrix_elements: bool = False,
) -> dict[tuple[str | None, str], mtx.Driver]:
    """Runs MadtRex reweighting end to end against an existing MadGraph7 run.

    Parses rwgt_path (see reweight_card.py) into groups of launches by which
    UFO model *and* which floating-point precision (fptype) they target (a
    bare card with no "change model"/"set FPTYPE" lines is one group
    targeting the original process_directory as-is, at its baseline
    precision -- see reweight_card.group_by_fptype()). Each group with at
    least one launch runs as its own independent reweighting pass, since
    each needs its own MatrixElementApi loaded (a different model needs
    different physics code entirely; a different fptype needs a
    differently-compiled library for the *same* code -- see
    Driver.set_fptype_priority()): any group with a "change model" is
    regenerated/compiled into its own subdirectory of process_directory
    first by regenerate (mg7/madtrex.py's regenerate_process_in_model();
    without one, such a card is an error) -- named "rw_me" if the card
    only ever requests that one model, else "rw_me_<model>" per distinct
    model (shared across that model's fptype groups, since regeneration
    itself doesn't depend on fptype) -- and every group, regenerated
    directory or not, has its own fptype compiled on demand (see
    _compile_subprocesses()) before it's loaded. By default (output_path
    None) every pass adds its weights to events_path itself, after the
    weights already there (see write_weights_in_place()), so the groups'
    weights accumulate in that one file. With output_path, each pass writes
    its own output file next to output_path instead: untagged if it is the
    card's only group, else suffixed "_<model>_<fptype>" (the original-model
    group, if not alone, uses "original" for <model>). events_path is
    auto-discovered from the latest run under process_directory/Events if
    not given.

    The card's preamble may also set any of reweight_card.DRIVER_OPTION_NAMES
    (helicity reweighting, warning verbosity, event-batch splitting, deferred-
    iteration flushing, streaming -- see reweight_card.DriverOptions), applied
    to every Driver this builds, regardless of group. helicity_reweighting,
    unless None, overrides the card's "set helicity_reweighting": True
    reweights each event at its own helicity (Driver's default), False uses
    helicity-summed matrix elements.

    Returns a dict of the Driver built for each group, keyed by its
    (model, fptype) (model is None for the original-model group), so
    callers can inspect driver.warehouse().get() afterwards.

    drop_stored_matrix_elements: the last group writing its weights into an
    npy events_path in place also removes the stored matrix elements from
    it (see write_weights_in_place()).

    Every launch is normalised by the matrix element the events were
    generated with: the original process directory's, with the param_card
    of the events' own header (see _generation_param_card(); a warning
    lists how process_directory's Cards/param_card.dat differs from it,
    another one says when the header has none), at the sample's fptype.
    The original model's launches start from that same card, unless
    param_card names another one. When the events store their matrix
    elements, a sample of them is first compared with re-evaluated ones
    (see _check_stored_matrix_elements())."""
    with tempfile.TemporaryDirectory() as card_dir:
        return _run_parameter_reweighting(
            rwgt_path, events_path, output_path, process_directory, param_card,
            regenerate, helicity_reweighting, drop_stored_matrix_elements, card_dir)


def _run_parameter_reweighting(
    rwgt_path, events_path, output_path, process_directory, param_card,
    regenerate, helicity_reweighting, drop_stored_matrix_elements, card_dir,
):
    """run_parameter_reweighting(), writing the generation's param_card
    into card_dir."""
    if events_path is None:
        events_path = find_run_events(
            find_latest_run_dir(os.path.join(process_directory, "Events"))
        )
    source_path = events_path
    events_path = prepare_events(source_path)
    in_place = output_path is None
    init_meta = read_init_node(events_path)

    with open(rwgt_path) as f:
        parsed = param_rwgt_card.expand_scans(param_rwgt_card.parse_reweight_card(f.read()))
    # Card-wide, not per-group -- see DriverOptions -- so resolved once here
    # from the still-ungrouped ParsedCard and applied to every Driver below.
    driver_options = parsed.driver_options
    # The sample's own, actual generation-time fptype -- see
    # _read_run_fptype()'s own doc comment. Used both as group_by_fptype()'s
    # default *target* fptype (a card that sets no FPTYPE at all should stay
    # at the sample's own precision, not drift to a hardcoded default) and,
    # unconditionally below, as every group.model_spec-is-None group's
    # *normaliser* fptype -- a launch must always be normalised against this
    # original configuration, never against whichever fptype it targets.
    original_fptype = (_read_run_fptype(process_directory, source_path)
                       or param_rwgt_card.DEFAULT_FPTYPE)
    # The param_card the events were generated with (from their header), not
    # whatever Cards/param_card.dat is now: the normaliser of every group,
    # and where the original model's launches start from unless param_card
    # names another one.
    generation_card = _generation_param_card(process_directory, source_path, param_card,
                                             card_dir)
    groups = [
        group for group in param_rwgt_card.group_by_fptype(parsed, default_fptype=original_fptype)
        if group.launches
    ]
    if not groups:
        raise ValueError(f"reweight card {rwgt_path!r} has no 'launch' blocks")
    multi = len(groups) > 1

    devices = _read_devices(process_directory)
    threads = reweighting_threads(process_directory, source_path)
    _compile_subprocesses(process_directory, devices, original_fptype)
    _check_stored_matrix_elements(
        events_path, init_meta, process_directory, generation_card, devices, original_fptype,
        card_label=("the param_card they were generated with"
                    if generation_card.startswith(card_dir) else ""))
    drivers: dict[tuple[str | None, str], mtx.Driver] = {}
    for group_index, group in enumerate(groups):
        model_tag = _sanitize_dirname(group.model_spec) if group.model_spec is not None else "original"
        tag = f"{model_tag}_{group.fptype}" if multi else None

        if group.model_spec is None:
            load_directory, group_param_card = process_directory, param_card or generation_card
        else:
            if regenerate is None:
                raise RuntimeError(
                    f"reweight card {rwgt_path!r} asks for 'change model {group.model_spec}', "
                    f"which needs a full MadGraph installation to regenerate the process in "
                    f"that model: run it through MadGraph (mg7/madtrex.py) rather than this "
                    f"standalone driver, or reweight within the process's own model"
                )
            rw_subdir = "rw_me" if not multi else f"rw_me_{model_tag}"
            load_directory = regenerate(
                process_directory, group.model_spec, group.process_definition, rw_subdir,
                fptype=group.fptype,
            )
            group_param_card = ""

        # Guarantees group.fptype is actually compiled for load_directory,
        # regardless of whether regenerate() just did so
        # (only true on the first group to regenerate a given rw_subdir --
        # see its own doc comment) or load_directory is process_directory
        # itself (never compiled for anything but launch.py's own
        # hardcoded default fptype at generation time).
        _compile_subprocesses(load_directory, devices, group.fptype)
        # The normaliser needs original_fptype's own compiled library too,
        # whenever it differs from the target fptype just compiled above
        # (see the driver.set_normaliser_fptype_priority() call below). A
        # "change model" group is normalised against the original process
        # directory's own matrix elements, at original_fptype, like every
        # other group (see Driver.set_normaliser_process_directory()).
        if group.model_spec is not None:
            _compile_subprocesses(process_directory, devices, original_fptype)
        elif original_fptype != group.fptype:
            _compile_subprocesses(load_directory, devices, original_fptype)

        if in_place:
            # each group adds its weights to the file the previous one wrote
            events_path = prepare_events(source_path)
        else:
            group_output_path = output_path if tag is None else _tag_output_path(output_path, tag)

        launches = group.launches
        if param_rwgt_card.has_parameter_names(launches):
            # "set MT 175": named by the param_card this group runs with
            card_path = group_param_card or os.path.join(load_directory, "Cards", "param_card.dat")
            with open(card_path) as f:
                launches = param_rwgt_card.resolve_parameter_names(launches, f.read())

        with tempfile.NamedTemporaryFile("w", suffix=".rwgt", delete=False) as fh:
            fh.write(param_rwgt_card.render_launches(launches))
            resolved_rwgt_path = fh.name
        try:
            logger.info(
                "MadtRex: reweighting %s against model '%s' at fptype '%s' with %s",
                events_path, group.model_spec or "<original>", group.fptype, rwgt_path,
            )
            driver = mtx.Driver()
            # threads and batch splitting by default as the generation, then
            # whatever the card sets
            driver.set_threads(threads)
            driver.set_min_batch(param_rwgt_card.DEFAULT_MIN_BATCH)
            driver_options.apply(driver)
            logger.info(
                "MadtRex: %d thread(s), subprocesses split into batches of at least "
                "%d events", driver.threads(), driver.min_batch())
            if helicity_reweighting is not None:
                driver.set_helicity_reweighting(helicity_reweighting)
            driver.set_device_priority(devices)
            driver.set_fptype_priority([group.fptype])
            # The normaliser is always the matrix element the events were
            # generated with: the original process directory's, with its
            # param_card, at the sample's own generation-time fptype
            # (original_fptype), never group.fptype -- see
            # Driver.set_normaliser_fptype_priority()'s own doc comment. A
            # no-op for the original model (normaliser_priority ==
            # fptype_priority, one api shared for both roles) whenever the
            # card doesn't actually change fptype, since group.fptype then
            # also resolves to original_fptype (see group_by_fptype()'s
            # default_fptype). A "change model" group's weights are the
            # other model's matrix element over that original one
            # (MadGraph's reweighting convention), never over the other
            # model's own.
            driver.set_normaliser_fptype_priority([original_fptype])
            if group.model_spec is not None:
                _check_distinct_libraries(load_directory, process_directory)
                driver.set_normaliser_process_directory(process_directory, generation_card)
            else:
                driver.set_normaliser_param_card(generation_card)
            driver.load_process_directory(load_directory, group_param_card)
            driver.load_param_reweighting(resolved_rwgt_path)
            if init_meta is not None:
                driver.load_events(events_path, init_meta)
            else:
                driver.load_events(events_path)
            driver.warehouse().get().run()
            if in_place:
                group_output_path = write_weights_in_place(
                    driver, driver.warehouse().get(), source_path, events_path,
                    drop_stored_matrix_elements=(drop_stored_matrix_elements
                                                 and group_index == len(groups) - 1))
            else:
                driver.write_weights(group_output_path)
        finally:
            os.remove(resolved_rwgt_path)
        logger.info("MadtRex: wrote %s", group_output_path)
        drivers[(group.model_spec, group.fptype)] = driver

    return drivers


def run_pdf_reweighting(
    rwgt_path: str,
    events_path: str | None = None,
    output_path: str | None = None,
    process_directory: str = ".",
    drop_stored_matrix_elements: bool = False,
) -> mtx.PdfDriver:
    """Runs MadtRex PDF-only reweighting end to end against an existing
    MadGraph7 run, via PdfDriver -- entirely independent of run_parameter_reweighting()
    (matrix-element/parameter reweighting): see pdf_reweight_card.py's module
    docstring for why the two are split into separate cards/flows (PDFs and
    matrix elements factorise), and main() for running both together against
    the same sample.

    Unlike run_parameter_reweighting(), no model regeneration or subprocess compilation
    is needed: a PDF reweighting hypothesis is just a different (PDF set,
    scale) pair fed to the same PdfSet/PdfDriver machinery, not a different
    physics library, so this is a single pass regardless of how many "launch"
    blocks or distinct PDF sets rwgt_path uses.

    Parses rwgt_path (see pdf_reweight_card.py), first expanding any
    scan:[...] MUF/MUR/MUF_FACTOR/MUR_FACTOR values into explicit per-point
    launches (pdf_reweight_card.expand_scans(), the same mechanism
    run_parameter_reweighting() uses for physics parameter scans), then resolving the
    result into a ResolvedPdfCard: a required baseline PDF set -- unlike
    run_parameter_reweighting()'s FPTYPE, which falls back to a hard-set default when
    unspecified, MadtRex does not
    currently record which PDF set an existing sample was generated with
    anywhere *on the sample itself*. If the card's preamble never sets
    PDF_SET/BEAM1_PDF_SET/BEAM2_PDF_SET at all, this looks it up instead,
    from process_directory's own run_card.toml [beam] pdf (see
    _read_default_pdf_set()) -- a real fact about the run, not a guessed
    default, available whenever process_directory still has (or ever had)
    the Cards/ that generated events_path, which covers both "reweighting
    immediately after generation" and any later run against that same
    process directory. Only once that lookup also comes up empty (no
    run_card.toml, or one predating [beam] pdf) does pdf_rwgt_card.resolve()
    raise, requiring PDF_SET explicitly -- eg for a bare/relocated sample,
    or to deliberately reweight against a baseline other than the one the
    run actually used. A further hook could still infer the baseline PDF
    set straight from the LHE file's own metadata instead of the run_card
    -- not implemented, since MadGraph7's LHE output carries no such field
    today. Every distinct PDF set name the card references (baseline and
    launches alike) is load_pdf_set()'d once, resolved to its LHAPDF
    grid/.info files via resolve_pdf_set_files().

    Beam energies are taken from events_path's sibling header.lhe/init.lhe
    (see read_init_node()) if present, else from its LHE <init> block directly
    (PdfDriver.load_events() sniffs this itself). events_path is auto-
    discovered from the latest run under process_directory/Events if not
    given. The weights are added to events_path itself unless output_path
    is given (see write_weights_in_place()).

    Returns the built PdfDriver, so callers can inspect driver.get()
    afterwards."""
    if events_path is None:
        events_path = find_run_events(
            find_latest_run_dir(os.path.join(process_directory, "Events"))
        )
    source_path = events_path
    events_path = prepare_events(source_path)
    init_meta = read_init_node(events_path)

    with open(rwgt_path) as f:
        parsed = pdf_rwgt_card.expand_scans(pdf_rwgt_card.parse_pdf_reweight_card(f.read()))
    baseline = parsed.baseline
    if baseline.pdf_set_name is None and baseline.beam1_pdf_set_name is None and baseline.beam2_pdf_set_name is None:
        default_pdf_set, source = _read_default_pdf_set(process_directory, source_path)
        if default_pdf_set is not None:
            logger.info(
                "MadtRex: PDF reweight card %s sets no baseline PDF_SET; using '%s', "
                "the PDF set the events were generated with (from %s)",
                rwgt_path, default_pdf_set, source,
            )
            baseline.pdf_set_name = default_pdf_set
    card = pdf_rwgt_card.resolve(parsed)
    if not card.launches:
        raise ValueError(f"PDF reweight card {rwgt_path!r} has no 'launch' blocks")

    driver = mtx.PdfDriver()
    if card.verbose is not None:
        driver.set_verbose(card.verbose)
    if card.alpha_s_power is not None:
        driver.set_alpha_s_power(card.alpha_s_power)
    if card.scale_variation_output is not None:
        driver.set_scale_variation_output(
            getattr(mtx.ScaleVariationOutput, card.scale_variation_output.upper())
        )

    def as_scale_choice(scale: pdf_rwgt_card.PdfScale) -> mtx.ScaleChoice:
        return mtx.ScaleChoice(scale.value, scale.absolute)

    loaded_pdf_sets: set[str] = set()

    def ensure_loaded(name: str) -> None:
        if name in loaded_pdf_sets:
            return
        grid_file, alpha_s_grid_file = resolve_pdf_set_files(name)
        driver.load_pdf_set(name, grid_file, alpha_s_grid_file)
        loaded_pdf_sets.add(name)

    for name in card.baseline_pdf_set_names:
        ensure_loaded(name)
    for launch in card.launches:
        for name in launch.pdf_set_names:
            ensure_loaded(name)

    driver.set_baseline(
        *card.baseline_pdf_set_names,
        as_scale_choice(card.baseline_muF_scale),
        as_scale_choice(card.baseline_muR_scale),
    )
    for i, launch in enumerate(card.launches):
        driver.add_hypothesis(
            launch.name or f"pdf_rwgt_{i + 1}",
            *launch.pdf_set_names,
            as_scale_choice(launch.muF_scale),
            as_scale_choice(launch.muR_scale),
        )

    logger.info("MadtRex: PDF-reweighting %s with %s", events_path, rwgt_path)
    if init_meta is not None:
        # PdfDriver.load_events() only actually consults meta's beam
        # energies for binary input -- for XML LHE it reads the file's own
        # <init> block instead (older MadGraph7 writers left a zeroed-out
        # stub there, not real beam energies). set_beam_energies() is the
        # documented way to override that, so use it explicitly whenever
        # init_meta (from header.lhe/init.lhe) is available, regardless of
        # events_path's format.
        driver.set_beam_energies(init_meta.beam1_energy, init_meta.beam2_energy)
        driver.load_events(events_path, init_meta)
    else:
        driver.load_events(events_path)
    driver.get().run()
    if output_path is None:
        output_path = write_weights_in_place(
            driver, driver.get(), source_path, events_path,
            drop_stored_matrix_elements=drop_stored_matrix_elements)
    else:
        driver.write_weights(output_path)
    logger.info("MadtRex: wrote %s", output_path)
    return driver


def _merge_reweighting_results(target, source) -> None:
    """Merges source's weight columns onto target, in place: appends every
    one of source's weight_ids after target's own, and correspondingly
    extends every target event's own wgts with the matching source event's.
    target and source must come from reweighting the exact same events, in
    the same order (true for run_generic_reweighting()'s own dispatch,
    since both passes load the same events_path) -- raises if their event
    counts differ, as a cheap sanity check of that assumption.

    Uses add_weight_id() (in-place push_back through the *existing*
    weight_ids shared_ptr) rather than reassigning target.weight_ids
    directly -- the bound weight_ids property's setter replaces the whole
    shared_ptr, which would desync every event's own (separately
    shared_ptr-aliased) weight_ids from it and corrupt the write. See
    madtrex::PdfDriver::keep_weights()'s own doc comment (pdfware.hpp) for
    the same gotcha encountered the other way round (dropping columns
    rather than adding them)."""
    target_events = target.events
    source_events = source.events
    if len(target_events) != len(source_events):
        raise ValueError(
            "cannot coalesce reweighting results: event counts differ "
            f"({len(target_events)} vs {len(source_events)}) -- were they reweighted "
            f"against the same events file?"
        )
    for name in source.weight_ids:
        target.add_weight_id(name)
    for t_ev, s_ev in zip(target_events, source_events):
        t_ev.wgts = t_ev.wgts + s_ev.wgts


def run_generic_reweighting(
    rwgt_path: str,
    events_path: str | None = None,
    output_path: str | None = None,
    process_directory: str = ".",
    param_card: str = "",
    regenerate: Regenerator | None = None,
    helicity_reweighting: bool | None = None,
    drop_stored_matrix_elements: bool = False,
) -> dict[str, object]:
    """Runs MadtRex's generic reweight card end to end: splits rwgt_path
    (see generic_reweight_card.py) into its "parameter"/"pdf" sections --
    both, or whichever one is present -- runs each through its own real
    routine (run_parameter_reweighting()/run_pdf_reweighting()) exactly as
    if it had been handed a standalone card of that kind, then, if *both*
    modes ran, coalesces the results back into a single sample per
    parameter-reweighting group: every parameter-reweighting output
    additionally gets the PDF-reweighting weights merged onto it (in place,
    then rewritten to that same group's own output path), so a caller only
    looking at that one "primary" output file sees every requested weight,
    from both modes, on one set of events -- not two separate files to
    reconcile themselves. If only one mode is present, nothing needs
    coalescing -- that mode's own output, written by its own routine, is
    already the whole answer.

    events_path/output_path/process_directory/param_card are all forwarded
    to whichever of run_parameter_reweighting()/run_pdf_reweighting()
    actually runs; param_card is unused if the card has no "parameter"
    section. Without output_path, each pass adds its weights to events_path
    in turn and nothing needs coalescing. With output_path, it anchors the
    *parameter* pass's own naming (its usual untagged-unless-multiple-groups
    scheme, same as run_parameter_reweighting() alone); the PDF pass's own
    intermediate output is separately tagged "_pdf" whenever a parameter
    section is also present, mirroring main()'s own two-flag convention, so
    the two never collide, and the PDF weights are merged onto the
    parameter output(s).

    Returns {"parameter": dict[(model, fptype), Driver] | None, "pdf":
    PdfDriver | None} -- one real entry per mode that actually ran
    (mirroring each routine's own native return value), None for a mode
    the card didn't request. Callers after the *merged* result should read
    the output file(s) written, not these driver objects -- which of them
    ends up with its in-memory weight_ids/events mutated by the merge (the
    parameter side's, when both modes ran) is an implementation detail, not
    part of the return contract."""
    with open(rwgt_path) as f:
        sections = generic_rwgt_card.split_by_mode(f.read())
    if not sections:
        raise ValueError(f"generic reweight card {rwgt_path!r} has no 'set mode' sections")

    if events_path is None:
        events_path = find_run_events(
            find_latest_run_dir(os.path.join(process_directory, "Events"))
        )
    # output_path None: each pass adds its weights to events_path in turn
    in_place = output_path is None

    result: dict[str, object] = {"parameter": None, "pdf": None}
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "parameter" in sections:
            param_path = os.path.join(tmp_dir, "parameter.rwgt")
            with open(param_path, "w") as f:
                f.write(sections["parameter"])
            result["parameter"] = run_parameter_reweighting(
                param_path,
                events_path=events_path,
                output_path=output_path,
                process_directory=process_directory,
                param_card=param_card,
                regenerate=regenerate,
                helicity_reweighting=helicity_reweighting,
                drop_stored_matrix_elements=(drop_stored_matrix_elements
                                             and "pdf" not in sections),
            )
        if "pdf" in sections:
            pdf_path = os.path.join(tmp_dir, "pdf.rwgt")
            with open(pdf_path, "w") as f:
                f.write(sections["pdf"])
            pdf_output_path = output_path
            if not in_place and "parameter" in sections:
                pdf_output_path = _tag_output_path(output_path, "pdf")
            result["pdf"] = run_pdf_reweighting(
                pdf_path,
                events_path=events_path,
                output_path=pdf_output_path,
                process_directory=process_directory,
                drop_stored_matrix_elements=drop_stored_matrix_elements,
            )

    param_drivers, pdf_driver = result["parameter"], result["pdf"]
    if not in_place and param_drivers is not None and pdf_driver is not None:
        multi = len(param_drivers) > 1
        for (model_spec, fptype), driver in param_drivers.items():
            model_tag = _sanitize_dirname(model_spec) if model_spec is not None else "original"
            tag = f"{model_tag}_{fptype}" if multi else None
            group_output_path = output_path if tag is None else _tag_output_path(output_path, tag)
            _merge_reweighting_results(driver.warehouse().get(), pdf_driver.get())
            driver.write_weights(group_output_path)
            logger.info(
                "MadtRex: coalesced PDF reweighting into parameter reweighting result for "
                "model '%s' at fptype '%s' -> %s",
                model_spec or "<original>", fptype, group_output_path,
            )

    return result


def run_reweighting(
    rwgt_path: str,
    events_path: str | None = None,
    card_kind: str | None = None,
    output_path: str | None = None,
    process_directory: str = ".",
    param_card: str = "",
    regenerate: Regenerator | None = None,
    helicity_reweighting: bool | None = None,
    drop_stored_matrix_elements: bool = False,
):
    """Runs the reweight card rwgt_path with the runner its card_kind needs:
    "param" -> run_parameter_reweighting(), "pdf" -> run_pdf_reweighting(),
    "generic" -> run_generic_reweighting(). card_kind defaults to
    generic_reweight_card.card_kind() of the card. Returns that runner's
    result."""
    _require_madtrex()
    if card_kind is None:
        with open(rwgt_path) as f:
            card_kind = generic_rwgt_card.card_kind(f.read())
    if card_kind not in generic_rwgt_card.CARD_KINDS:
        raise ValueError(
            f"unknown reweight card kind {card_kind!r}; expected one of "
            f"{list(generic_rwgt_card.CARD_KINDS)}"
        )
    common = dict(events_path=events_path, output_path=output_path,
                  process_directory=process_directory,
                  drop_stored_matrix_elements=drop_stored_matrix_elements)
    if card_kind == "pdf":
        return run_pdf_reweighting(rwgt_path, **common)
    runner = run_parameter_reweighting if card_kind == "param" else run_generic_reweighting
    return runner(rwgt_path, param_card=param_card, regenerate=regenerate,
                  helicity_reweighting=helicity_reweighting, **common)


def check_card_for_standalone(rwgt_path: str) -> None:
    """Raises when the reweight card at rwgt_path needs the classic
    reweighting module ("change backend classic", or a directive only the
    classic module understands): that module is part of MadGraph, which
    this standalone driver (a gridpack's bin/reweight) runs without.
    "change backend madtrex|auto" is fine."""
    problems = []
    with open(rwgt_path) as f:
        for number, raw_line in enumerate(f, 1):
            line = generic_rwgt_card.strip_comment(raw_line)
            words = line.split()
            if len(words) >= 2 and words[0].lower() == "change" and words[1].lower() == "backend":
                value = words[2].lower() if len(words) > 2 else ""
                if value == "classic":
                    problems.append(f"line {number}: {line.strip()!r}")
                elif value not in generic_rwgt_card.BACKENDS:
                    raise ValueError(
                        f"{rwgt_path} line {number}: unknown backend {value!r}; expected "
                        f"one of {sorted(generic_rwgt_card.BACKENDS)}"
                    )
                continue
            reason = generic_rwgt_card.classic_only_reason(line)
            if reason is not None:
                problems.append(f"line {number}: {line.strip()!r} ({reason})")
    if problems:
        raise SystemExit(
            f"{rwgt_path} needs the classic reweighting module, and classic reweighting "
            f"needs MadGraph (this standalone MadtRex driver runs without it): "
            + "; ".join(problems)
        )


def _default_process_directory() -> str:
    """The process directory this file belongs to when it is shipped inside
    one (a gridpack's bin/reweight: <gridpack>/SubProcesses/subprocesses.json
    next to <gridpack>/bin/), else the current directory."""
    root = _HERE.parent
    if (root / "SubProcesses" / "subprocesses.json").is_file() and (root / "Cards").is_dir():
        return str(root)
    return "."


def main(argv: list[str] | None = None, regenerate: Regenerator | None = None) -> None:
    """Command line entry point. regenerate is passed on to
    run_parameter_reweighting(); without it a "change model" card is an
    error (see mg7/madtrex.py for the MadGraph-side entry point)."""
    _require_madtrex()
    parser = argparse.ArgumentParser(
        description="Run MadtRex event reweighting against an existing MadGraph7 run. "
                     "Parameter (matrix-element) and PDF reweighting are independent "
                     "passes -- see run_parameter_reweighting()/run_pdf_reweighting(). Pass "
                     "--rwgt-card alone for a generic card that dispatches ('change mode "
                     "LO'/'change mode pdf') to one or both automatically and coalesces "
                     "the result -- see run_generic_reweighting() -- or --param-rwgt-card/"
                     "--pdf-rwgt-card directly (either, or both, for two separate outputs)."
    )
    parser.add_argument(
        "--rwgt-card", dest="generic_rwgt_path", default=None,
        help="generic reweight card dispatching to parameter and/or PDF reweighting (see "
             "generic_reweight_card.py), eg Cards/reweight_card.dat; mutually exclusive "
             "with --param-rwgt-card/--pdf-rwgt-card",
    )
    parser.add_argument(
        "--param-rwgt-card", dest="param_rwgt_path", default=None,
        help="parameter (SLHA/matrix-element) reweight card, eg "
             "Cards/param_reweight_card.dat",
    )
    parser.add_argument(
        "--pdf-rwgt-card", dest="pdf_rwgt_path", default=None,
        help="PDF reweight card, eg Cards/pdf_reweight_card.dat",
    )
    parser.add_argument(
        "--events", dest="events_path", default=None,
        help="events file to reweight (events.lhe or events.npy); "
             "defaults to the latest run under Events/",
    )
    parser.add_argument(
        "--output", dest="output_path", default=None,
        help="write the reweighted events to this path (or binary weight-file prefix for "
             "events.npy input) instead of adding the weights to --events itself, the "
             "default (--pdf-rwgt-card's own output, when both flags are given, is further "
             "tagged '_pdf' so the two don't collide)",
    )
    parser.add_argument(
        "--process-dir", dest="process_directory", default=None,
        help="process directory (or gridpack) containing SubProcesses/ and Cards/ "
             "(default: the gridpack this script is in, else .)",
    )
    parser.add_argument(
        "--helicity-reweighting", dest="helicity_reweighting",
        action=argparse.BooleanOptionalAction, default=None,
        help="reweight each event at its own helicity (--helicity-reweighting, the "
             "default) or with helicity-summed matrix elements "
             "(--no-helicity-reweighting); overrides the reweight card's 'change "
             "helicity'",
    )
    parser.add_argument(
        "--param-card", dest="param_card", default="",
        help="baseline param_card.dat (--param-rwgt-card/--rwgt-card only); defaults to "
             "<process-dir>/Cards/param_card.dat",
    )
    args = parser.parse_args(argv)
    if args.process_directory is None:
        args.process_directory = _default_process_directory()
    if not logger.handlers and not logging.getLogger().handlers:
        # standalone run: nothing has set up MadGraph's logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    if regenerate is None:
        # without MadGraph (see mg7/madtrex.py for the MadGraph side)
        for path in (args.generic_rwgt_path, args.param_rwgt_path, args.pdf_rwgt_path):
            if path is not None:
                check_card_for_standalone(path)
    if args.generic_rwgt_path is not None:
        if args.param_rwgt_path is not None or args.pdf_rwgt_path is not None:
            parser.error("--rwgt-card is mutually exclusive with --param-rwgt-card/--pdf-rwgt-card")
        run_generic_reweighting(
            args.generic_rwgt_path,
            events_path=args.events_path,
            output_path=args.output_path,
            process_directory=args.process_directory,
            param_card=args.param_card,
            regenerate=regenerate,
            helicity_reweighting=args.helicity_reweighting,
        )
        return
    if args.param_rwgt_path is None and args.pdf_rwgt_path is None:
        parser.error("at least one of --rwgt-card/--param-rwgt-card/--pdf-rwgt-card is required")
    if args.param_rwgt_path is not None:
        run_parameter_reweighting(
            args.param_rwgt_path,
            events_path=args.events_path,
            output_path=args.output_path,
            process_directory=args.process_directory,
            param_card=args.param_card,
            regenerate=regenerate,
            helicity_reweighting=args.helicity_reweighting,
        )
    if args.pdf_rwgt_path is not None:
        # An explicit --output is untagged by default (run_pdf_reweighting()
        # only tags "_pdf" onto its own auto-derived default); tag it
        # ourselves here so an explicit --output can't collide with
        # --param-rwgt-card's own write when both flags are given.
        pdf_output_path = args.output_path
        if pdf_output_path is not None and args.param_rwgt_path is not None:
            pdf_output_path = _tag_output_path(pdf_output_path, "pdf")
        run_pdf_reweighting(
            args.pdf_rwgt_path,
            events_path=args.events_path,
            output_path=pdf_output_path,
            process_directory=args.process_directory,
        )


if __name__ == "__main__":
    main()
