"""Runs, scans and the entry point of the mg7 launcher."""

import contextlib
import gc
import json
import logging
import os
import re
import shutil
import tempfile

from madgraph.mg7.common import logger
from madgraph.various.banner import RunCardMG7
from madgraph.mg7.model_widths import compute_auto_widths
from madgraph.mg7 import npy_to_lhe
from madgraph.mg7.options import load_mg5_options
from madgraph.mg7.postprocess import (
    _find_event_file,
    _find_or_convert_event_file,
    _off,
    _report_failure,
    run_lhe_postprocessing,
    run_selected_tools,
    selected_tools,
)
from madgraph.mg7.madgraph_process import MadgraphProcess

def compute_decay_widths(
    me_dir: str = ".", precision: float = 0.01, nb_core: int | None = None,
    extra_make_args=(),
) -> list[dict]:
    """Integrate the decays of an mg7 decay directory and return their widths.

    This is the madspace counterpart of madevent's ``survey decay``: only the
    survey runs (no unweighting, no events, no systematics), and it stops once
    every channel is below ``precision`` relative error, like the
    ``--precision_channel`` of ``compute_widths``.

    ``me_dir`` is an output directory of 1 -> n processes of one decaying
    particle. It is meant to run in a process of its own (``bin/generate_events
    --decay-widths=FILE``, which is how ``compute_widths`` calls it): the
    madspace objects write their status file from a destructor, and the
    caller removes the temporary directory as soon as the widths are back.
    The result holds one entry per decay and flavour assignment::

        {"incoming": 25, "outgoing": [5, -5], "width": 4.3e-3, "error": 1e-9}

    (signed pdg ids, width in GeV). The flavours of a subprocess (different
    matrix elements, e.g. the u u~ g and d d~ g of z > q q~ g) are integrated
    separately (MadgraphProcess.split_decay_flavors); the assignments of one
    flavour share its matrix element (they are the same function of the
    momenta, or they would not be grouped) and split its width evenly. A
    decay the decaying particle is too light for has no entry.
    """
    cwd = os.getcwd()
    os.chdir(me_dir)
    process = None
    try:
        card_path = os.path.join("Cards", "run_card.toml")
        card = RunCardMG7(card_path, consistency=False)
        card["generation"]["survey_target_precision"] = precision
        card["generation"]["survey_max_iters"] = 30
        # one survey result per decay, not per group of mergeable ones
        card["phasespace"]["merge_subprocesses"] = False
        card["run"]["verbosity"] = "silent"
        if nb_core:
            card["run"]["cpu_thread_pool_size"] = int(nb_core)
        card.write(card_path)

        # one survey result per flavour, not per subprocess
        process = MadgraphProcess(extra_make_args=extra_make_args,
                                  split_flavors=True)
        if not process.is_decay:
            raise ValueError("compute_decay_widths needs a decay (1 -> n) output")
        process.survey()

        channel_status = process.event_generator.channel_status()
        results = []
        offset = 0
        for subproc, phasespace in zip(process.subprocesses, process.phasespaces):
            statuses = channel_status[offset:offset + len(phasespace.channels)]
            offset += len(phasespace.channels)
            width = sum(status.mean for status in statuses)
            error = sum(status.error ** 2 for status in statuses) ** 0.5
            options = _decay_flavor_options(subproc.meta)
            for option in options:
                results.append({
                    "incoming": option[0],
                    "outgoing": list(option[1:]),
                    "width": width / len(options),
                    "error": error / len(options),
                })
        return results
    finally:
        if process is not None:
            # StatusFile finishes info.json from its destructor: run it before
            # the caller removes the directory (see init_event_dir)
            _release_channel_generators(process)
            del process
            gc.collect()
        os.chdir(cwd)


def _decay_flavor_options(meta: dict) -> list[list[int]]:
    """The flavour assignments of a subprocess of compute_decay_widths, which
    share its width evenly.

    That is right for the assignments of one flavour, which have one matrix
    element (d d~ and s s~ of a massless s, the three neutrinos). Different
    flavours have different matrix elements (the u u~ and d d~ of a
    flavour-grouped subprocess) and the survey of a subprocess only gives
    their sum: compute_decay_widths splits them first (split_flavors), and a
    subprocess still holding several is an error, not an even split that
    would get the branching ratios wrong.
    """
    flavors = meta["flavors"]
    if len(flavors) != 1:
        raise ValueError(
            "compute_decay_widths: the subprocess %s has %d flavours with "
            "different matrix elements, the survey cannot give the width of "
            "each" % (meta.get("path", "?"), len(flavors))
        )
    return [list(option) for option in flavors[0]["options"]]


def _release_channel_generators(process) -> None:
    """Drop the per-channel event/weight generators so their intermediate
    .npy files are deleted right away. madspace only frees them when the
    owning Python objects are collected, and process.event_generator /
    process.phasespaces sit in a reference cycle that plain refcounting
    never breaks -- only an explicit gc.collect() does, promptly, instead of
    leaving it to whenever the cyclic GC next runs on its own."""
    process.event_generator = None
    process.phasespaces = None
    gc.collect()
    if getattr(process, "auto_temp_path", False):
        shutil.rmtree(process.temp_path, ignore_errors=True)


REWEIGHT_CARD = os.path.join("Cards", "reweight_card.dat")


class ExistingRun:
    """The run directory Events/<run_name> of an earlier generation, with
    what run_reweighting() needs of a MadgraphProcess: run_path, events_path
    (the file the generation wrote: events.npy for the npy formats, else
    events.lhe[.gz]) and run_card["run"]["output_format"] (from the files)."""

    def __init__(self, run_name, events_dir="Events"):
        self.run_path = os.path.abspath(os.path.join(events_dir, run_name))
        if not os.path.isdir(self.run_path):
            runs = list_runs(events_dir)
            raise ValueError("no run '%s' in %s%s" % (
                run_name, events_dir,
                "; runs: %s" % ", ".join(runs) if runs else ""))
        npy = os.path.join(self.run_path, npy_to_lhe.EVENT_FILE)
        if os.path.isfile(npy):
            self.events_path = npy
            compact = os.path.isfile(os.path.join(self.run_path,
                                                  npy_to_lhe.COMPLETER_FILE))
            output_format = "compact_npy" if compact else "lhe_npy"
        else:
            self.events_path = _find_event_file(self.run_path)
            output_format = "lhe"
        if self.events_path is None:
            raise ValueError("no event file (events.npy, events.lhe[.gz]) in %s"
                             % self.run_path)
        self.run_card = {"run": {"output_format": output_format}}


def list_runs(events_dir="Events") -> list:
    """The runs of events_dir with an event file, sorted by name."""
    if not os.path.isdir(events_dir):
        return []
    return sorted(
        name for name in os.listdir(events_dir)
        if any(os.path.isfile(os.path.join(events_dir, name, f)) for f in
               (npy_to_lhe.EVENT_FILE, "events.lhe", "events.lhe.gz")))


def install_reweight_card(path) -> None:
    """``--reweight=CARD``: CARD becomes Cards/reweight_card.dat. A different
    card already there is kept as reweight_card.dat.bak."""
    if os.path.exists(REWEIGHT_CARD):
        if os.path.samefile(path, REWEIGHT_CARD):
            return
        with open(path) as new, open(REWEIGHT_CARD) as old:
            if new.read() == old.read():
                return
        shutil.copy(REWEIGHT_CARD, REWEIGHT_CARD + ".bak")
        logger.info("--reweight: %s saved as %s.bak", REWEIGHT_CARD, REWEIGHT_CARD)
    shutil.copy(path, REWEIGHT_CARD)
    logger.info("--reweight: using %s as %s", path, REWEIGHT_CARD)


# reweight switch values naming a backend (ON is auto)
_SWITCH_BACKENDS = {"classic": "classic", "density": "classic", "madtrex": "madtrex"}


def prepare_reweighting(switch, cli_backend=None, options=None):
    """The reweight_backend.Decision for this run, or None when it does not
    reweight: the reweight switch is off, or Cards/reweight_card.dat is
    missing or has no "launch". Chosen before the generation, so that a card
    that cannot run fails before the events are generated, and the event
    format the backend needs is known (force_lhe_output_if_needed).

    Precedence: cli_backend (--reweight-backend), then a switch value naming
    a backend (classic/density/madtrex; ON is auto), then the card's
    "change backend" line; auto otherwise."""
    from madgraph.various import reweight_backend as rb

    value = (switch or {}).get("reweight")
    if _off(value) or not os.path.exists(REWEIGHT_CARD):
        return None
    with open(REWEIGHT_CARD) as f:
        traits = rb.classify_card(f.read())
    if not traits.launches:
        logger.info("reweighting is on but %s has no 'launch': nothing to reweight",
                    REWEIGHT_CARD)
        return None
    plugin = value if value not in ("ON", "classic", "madtrex", "density") else None
    # a plugin is a classic reweighting module
    switch_backend = "classic" if plugin else _SWITCH_BACKENDS.get(value)
    requested = [("--reweight-backend", cli_backend),
                 ("the reweight switch (%s)" % value, switch_backend)]
    if value == "density":
        traits.density = True
    if options is None:
        options = load_mg5_options()
    env = rb.probe_environment(os.getcwd(), options)
    decision = rb.select_backend(requested, traits, env, plugin=plugin)
    for text in decision.warnings:
        logger.warning(text)
    return decision


@contextlib.contextmanager
def _stripped_reweight_card(path):
    """path without its "change backend" line for the duration (the classic
    module reads the card itself); restored afterwards."""
    from madgraph.various import reweight_backend as rb
    with open(path) as f:
        text = f.read()
    stripped = rb.strip_backend_line(text)
    if stripped == text:
        yield
        return
    with open(path, "w") as f:
        f.write(stripped)
    try:
        yield
    finally:
        with open(path, "w") as f:
            f.write(text)


# the <weightgroup name=...> each backend writes into the LHE header
REWEIGHT_WEIGHT_GROUPS = {"classic": "mg_reweighting", "madtrex": "tearex_reweighting"}


def write_reweighting_record(run_path, decision, card_path, events=None) -> str:
    """Record who reweighted the run, and why, in <run>/reweighting.json:
    backend, how it was chosen (source, reasons, warnings), the card kind,
    the event file the weights went into (events), the LHE weight group it
    wrote and the card itself. Returns the path."""
    with open(card_path) as f:
        card = f.read()
    record = {
        "backend": decision.backend,
        "source": decision.source,
        "reasons": [r.replace("{card}", os.path.basename(card_path))
                    for r in decision.reasons],
        "warnings": list(decision.warnings),
        "card_kind": decision.card_kind,
        "plugin": decision.plugin,
        "weight_group": REWEIGHT_WEIGHT_GROUPS[decision.backend],
        "events": events,
        "card": card,
    }
    path = os.path.join(run_path, "reweighting.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=1)
    return path


def drop_stored_matrix_elements(events_path) -> None:
    """Removes the "matrix_element" column (MadgraphProcess.
    store_matrix_elements) from the npy events at events_path, if they have
    one: it is only there for MadtRex's normalisation, not part of the
    run's output."""
    import numpy as np
    from numpy.lib import format as npy_format

    if not events_path.endswith(".npy") or not os.path.isfile(events_path):
        return
    events = np.load(events_path, mmap_mode="r", max_header_size=1 << 30)
    if "matrix_element" not in (events.dtype.names or ()):
        return
    names = [name for name in events.dtype.names if name != "matrix_element"]
    descr = [(name, events.dtype.fields[name][0]) for name in names]
    tmp = events_path + ".tmp"
    out = npy_format.open_memmap(tmp, mode="w+", dtype=np.dtype(descr), shape=events.shape)
    chunk = 100000
    for start in range(0, len(events), chunk):
        stop = min(start + chunk, len(events))
        for name in names:
            out[name][start:stop] = events[name][start:stop]
    out.flush()
    del out, events
    os.replace(tmp, events_path)


def run_reweighting(process, card_path=REWEIGHT_CARD, requested=None, decision=None):
    """Reweight the events of process with card_path, once, by the backend
    of decision (prepare_reweighting()), or of select_backend(requested)
    when no decision is given. The classic module runs through MG7RunCmd
    ("reweight -from_cards", on the LHE file); MadtRex through
    madtrex.run_reweighting(), on the events as written (lhe or npy)."""
    from madgraph.various import reweight_backend as rb

    log = logging.getLogger("madevent")
    if decision is None:
        with open(card_path) as f:
            traits = rb.classify_card(f.read())
        env = rb.probe_environment(os.getcwd(), load_mg5_options())
        decision = rb.select_backend(requested, traits, env)
    log.info(decision.describe(os.path.basename(card_path)))
    run_dir = process.run_path
    # both backends add their weights to the run's own event file (in place,
    # after the systematics weights): events.npy or events.lhe[.gz]

    if decision.backend == "madtrex":
        if process.run_card["run"]["output_format"] == "compact_npy":
            raise ValueError(
                "MadtRex reweighting requires run_card [run] output_format to be "
                "'lhe' or 'lhe_npy': 'compact_npy' discards the per-event flavor/"
                "helicity information MadtRex needs to match events to the "
                "matrix element library")
        from madgraph.iolibs.template_files.mg7.madtrex import (
            run_reweighting as madtrex_reweighting)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                card = os.path.join(tmp, "reweight_card.dat")
                with open(card_path) as f, open(card, "w") as out:
                    out.write(rb.strip_backend_line(f.read()))
                # the stored matrix elements leave events.npy with the
                # reweighting's last weight merge
                madtrex_reweighting(card, events_path=process.events_path,
                                    card_kind=decision.card_kind,
                                    process_directory=os.getcwd(),
                                    drop_stored_matrix_elements=True)
        finally:
            # and here only when that merge did not happen (a failure)
            drop_stored_matrix_elements(process.events_path)
        tool_copy = os.path.join(run_dir, "unweighted_events.lhe.gz")
        if not process.events_path.endswith(".npy") and os.path.exists(tool_copy):
            from madgraph.mg7.run_interface import sync_lhe_copy
            sync_lhe_copy(process.events_path, tool_copy)
        log.info("reweighting weights written to %s", process.events_path)
        write_reweighting_record(run_dir, decision, card_path, process.events_path)
        return decision

    lhe_path = _find_or_convert_event_file(run_dir, log)
    if lhe_path is None:
        raise RuntimeError("No LHE event file in %s: the classic reweighting "
                           "module needs one" % run_dir)
    run_name = os.path.basename(os.path.dirname(os.path.abspath(lhe_path)))
    from madgraph.mg7.run_interface import MG7RunCmd
    cmd = MG7RunCmd(os.getcwd(), load_mg5_options(), run_name, lhe_path)
    cmd.set_run_name(run_name, None, "parton")
    command = "reweight --backend=classic --mode=%s -from_cards" % (
        "density" if decision.card_kind == "density" else "ON")
    if decision.plugin:
        command += " --plugin=%s" % decision.plugin
    with _stripped_reweight_card(card_path):
        cmd.exec_cmd(command, postcmd=False, printcmd=False)
    try:
        cmd.store_result()
    except Exception as error:
        _report_failure(log, "store_result", error, run_dir)
    # the classic module's weights into the run's event file (lhe_path)
    cmd.sync_reweighted_events()
    log.info("reweighting weights written to %s", lhe_path)
    write_reweighting_record(run_dir, decision, card_path, lhe_path)
    return decision


def lhe_via_binary(process, reweight) -> bool:
    """Whether an output_format = "lhe" run that MadtRex reweights is
    generated as npy ("lhe_npy") and turned into the LHE file it asked for
    only after the reweighting (deliver_lhe_from_binary()): MadtRex reads
    and extends the npy events much faster than an LHE file, which it would
    decompress, parse and rewrite in full. Not when an LHE-only step runs
    before the reweighting: the run_card [postprocessing] time_of_flight or
    legacy systematics (see run_lhe_postprocessing())."""
    if getattr(reweight, "backend", None) != "madtrex":
        return False
    if process.gridpack_run_mode != "regular":
        return False
    if process.run_card["run"]["output_format"] != "lhe":
        return False
    cfg = process.run_card["postprocessing"]
    try:
        if float(cfg.get("time_of_flight", -1.0)) >= 0:
            return False
    except (TypeError, ValueError):
        return False
    if cfg.get("systematics") and not process.run_card["systematics"]["enable"]:
        return False
    return True


def deliver_lhe_from_binary(process) -> None:
    """The LHE file of a run lhe_via_binary() generated as npy: npy_to_lhe
    writes Events/<run>/events.lhe.gz from events.npy and header.lhe, with
    their weights (the systematics', MadtRex's), and it replaces them as the
    run's event file (process.events_path, and reweighting.json's events).
    When the conversion fails the npy events are kept, and reported."""
    log = logging.getLogger("madevent")
    npy_path = process.events_path
    process.lhe_from_binary = False
    try:
        lhe_path = npy_to_lhe.convert(npy_path)
    except Exception as error:
        _report_failure(log, "npy->LHE conversion", error, process.run_path)
        log.warning("the events of %s are kept as %s", os.path.basename(process.run_path),
                    npy_path)
        return
    os.remove(npy_path)
    os.remove(os.path.join(process.run_path, npy_to_lhe.HEADER_FILE))
    process.events_path = lhe_path
    record = os.path.join(process.run_path, "reweighting.json")
    if os.path.isfile(record):
        with open(record) as f:
            content = json.load(f)
        if content.get("events") == npy_path:
            content["events"] = lhe_path
            with open(record, "w") as f:
                json.dump(content, f, indent=1)


def run_single(switch=None, reweight=None) -> "MadgraphProcess":
    """Run a single generation and return the process (for its result).
    reweight is the reweight_backend.Decision of prepare_reweighting(), or
    None: reweighting runs once, after the run_card post-processing
    (systematics) and before the post-processing tools."""
    compute_auto_widths()
    process = MadgraphProcess()
    process.store_matrix_elements = getattr(reweight, "backend", None) == "madtrex"
    process.survey()
    process.train_madnis()
    process.lhe_from_binary = lhe_via_binary(process, reweight)
    if process.lhe_from_binary:
        logging.getLogger("madevent").info(
            "MadtRex reweights this run: the events are written as npy, and as "
            "the requested LHE file once reweighted")
    process.generate_events("lhe_npy" if process.lhe_from_binary else None)
    if process.gridpack_run_mode != "regular":
        return process
    # run_card-driven LHE post-processing (displaced vertex + systematics)
    run_lhe_postprocessing(process)
    # reweighting, after the systematics, by the backend chosen beforehand.
    # Like the post-processing tools, a failure is reported without losing
    # the rest of the run (MadSpin, shower, analyses): the event file is only
    # ever replaced atomically, so the events are kept, without the weights.
    if reweight is not None:
        try:
            run_reweighting(process, REWEIGHT_CARD, decision=reweight)
        except Exception as error:
            log = logging.getLogger("madevent")
            _report_failure(log, "reweighting", error, process.run_path)
            log.warning(
                "the events of %s are kept without the reweighting weights; to "
                "reweight them once the problem is fixed: 'launch %s -i' in "
                "MadGraph, then 'reweight %s'", os.path.basename(process.run_path),
                os.getcwd(), os.path.basename(process.run_path))
    # the LHE file of an "lhe" run MadtRex reweighted as npy, with the
    # reweighting weights, or without them when the reweighting failed
    if process.lhe_from_binary:
        deliver_lhe_from_binary(process)
    # run the post-processing tools (Pythia8/Delphes/MadSpin/analysis)
    # selected in the merged question above on the generated events.
    if switch:
        run_selected_tools(switch, process)
    return process


def detect_run_scan(run_card_path):
    """Return a banner.RunCardIterator if the run_card contains scan:[...]
    values, else None."""
    from madgraph.various import banner as banner_mod
    from madgraph.various import misc as _misc
    with _misc.TMP_variable(banner_mod.RunCard, 'allow_scan', True):
        rc = banner_mod.RunCard(run_card_path, consistency=False)
    if getattr(rc, 'scan_set', None):
        return banner_mod.RunCardIterator(run_card_path)
    return None


def detect_param_scan(param_card_path):
    """Return a ParamCardIterator if the param_card contains scan:[...] values,
    else None."""
    if not os.path.exists(param_card_path):
        return None
    from madgraph.models import check_param_card as param_card_mod
    it = param_card_mod.ParamCardIterator(param_card_path)
    for block in it.order:
        for param in block:
            if isinstance(param.value, str) and param.value.strip().lower().startswith('scan'):
                return it
    return None


def run_scan(iterator, card_path, switch=None, reweight=None) -> None:
    """Iterate over all scan points, running a full generation for each and
    accumulating the results, then write the scan summary. Works for both the
    run_card (RunCardIterator) and the param_card (ParamCardIterator); their
    interface (__iter__/write/store_entry/get_next_name/write_summary) is the
    same. The scan card is restored afterwards. reweight, if given, is
    forwarded to run_single() so every scan point is reweighted in turn."""
    import tomllib
    with open(os.path.join("Cards", "run_card.toml"), "rb") as f:
        run_name = tomllib.load(f).get("run", {}).get("run_name", "run")

    from madgraph.models import check_param_card as param_card_mod
    is_param_scan = isinstance(iterator, param_card_mod.ParamCardIterator)

    backup = card_path + ".scan_bak"
    shutil.copy(card_path, backup)
    try:
        for i, point in enumerate(iterator):
            point.write(card_path)
            logger.info("=== scan point %d ===", i + 1)
            process = run_single(switch, reweight)
            # use the run directory the process actually created, so the
            # per-point params.dat written by write_summary has a home
            name = os.path.basename(process.run_path)
            if is_param_scan:
                # pass the (possibly auto-width-updated) card so the summary
                # records the model-computed width for each 'auto' entry
                iterator.store_entry(name, process.get_result(),
                                     param_card_path=card_path)
            else:
                iterator.store_entry(name, process.get_result())
            _release_channel_generators(process)
        os.makedirs("Events", exist_ok=True)
        summary = os.path.join("Events", "scan_%s.txt" % run_name)
        iterator.write_summary(summary)
        logger.info("scan results written to %s", summary)
    finally:
        shutil.move(backup, card_path)


def run_generation(switch=None, reweight=None) -> None:
    """Run the generation, expanding a scan over the run_card or the param_card
    when one is present (scanning both simultaneously is not allowed).
    reweight (see prepare_reweighting()), if given, is forwarded to
    run_single()/run_scan() to reweight after each generation."""
    run_card_path = os.path.join("Cards", "run_card.toml")
    param_card_path = os.path.join("Cards", "param_card.dat")
    run_iter = detect_run_scan(run_card_path)
    param_iter = detect_param_scan(param_card_path)
    if run_iter and param_iter:
        raise RuntimeError(
            "Scanning simultaneously over the run_card and the param_card is "
            "not allowed. Please keep the scan:[...] entries in only one card.")
    if run_iter:
        run_scan(run_iter, run_card_path, switch, reweight)
    elif param_iter:
        run_scan(param_iter, param_card_path, switch, reweight)
    else:
        _release_channel_generators(run_single(switch, reweight))


def _lhe_needed_by(switch, card, reweight=None) -> "str | None":
    """Why the run needs an LHE event file, or None when it does not: every
    post-processing tool run_selected_tools drives, the classic reweighting
    module and the run_card [postprocessing] steps read the LHE file. MadtRex
    also reads the npy events. ``card`` is the raw TOML content of the
    run_card, ``reweight`` the prepare_reweighting() decision."""
    if reweight is not None and reweight.backend == "classic":
        return "reweighting (classic)"
    tools = selected_tools(switch) if switch else []
    if tools:
        return tools[0]
    cfg = card.get("postprocessing", {})
    try:
        if float(cfg.get("time_of_flight", -1.0)) >= 0:
            return "time_of_flight"
    except (TypeError, ValueError):
        pass
    # the legacy systematics.py path only runs when madspace did not compute
    # the weights itself (see run_lhe_postprocessing)
    if cfg.get("systematics") \
            and not card.get("systematics", {}).get("enable", True):
        return "systematics"
    return None


def force_lhe_output_if_needed(switch, reweight=None) -> None:
    """The events are written as npy by default; switch the run_card to the
    LHE format when a selected post-processing needs an LHE file. The card is
    rewritten rather than refused, so that asking for a shower in the launch
    question is enough -- the user does not have to know that it constrains
    output_format.

    The card is read as raw TOML and only its output_format line is edited: a
    run_card scan ("scan:[...]" values) is not a valid RunCardMG7 and must
    survive untouched for run_scan."""
    import tomllib
    path = os.path.join("Cards", "run_card.toml")
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        card = tomllib.load(f)
    output_format = card.get("run", {}).get(
        "output_format", RunCardMG7()["run"]["output_format"])
    if output_format == "lhe":
        return
    reason = _lhe_needed_by(switch, card, reweight)
    if reason is None:
        return
    with open(path) as f:
        text = f.read()
    text, count = re.subn(r'(?m)^(\s*output_format\s*=\s*)("[^"]*"|\'[^\']*\'|\S+)',
                          r'\g<1>"lhe"', text, count=1)
    if not count:
        text, count = re.subn(r'(?m)^\[run\][^\n]*\n', '\\g<0>output_format = "lhe"\n',
                              text, count=1)
    if not count:
        text += '\n[run]\noutput_format = "lhe"\n'
    with open(path, "w") as f:
        f.write(text)
    logging.getLogger("madevent").info(
        "output_format set to 'lhe' (required by %s).", reason)
