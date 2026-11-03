"""Post-processing tools run on the events, and their logging."""

import logging
import os
import sys
import time

from madgraph.mg7 import npy_to_lhe
from madgraph.mg7.options import lhapdf_paths, load_mg5_options

def _find_event_file(run_path):
    """Locate the LHE event file produced by generate_events (if any)."""
    for name in ("events.lhe", "events.lhe.gz"):
        path = os.path.join(run_path, name)
        if os.path.exists(path):
            return path
    return None


def _find_or_convert_event_file(run_path, log):
    """The run's LHE event file, converting its npy events when that is all
    the run wrote (see npy_to_lhe)."""
    lhe_path = _find_event_file(run_path)
    if lhe_path is None and npy_to_lhe.can_convert(run_path):
        log.info("No LHE event file in %s: converting the npy events.", run_path)
        try:
            lhe_path = npy_to_lhe.convert(run_path)
        except Exception as error:
            _report_failure(log, "npy->LHE conversion", error, run_path)
    return lhe_path


def _report_failure(log, what, error, directory=None):
    """Log a post-processing failure and write the full traceback to a file
    (whose path is printed) so the problem can be investigated."""
    import traceback
    if not directory or not os.path.isdir(directory):
        directory = os.getcwd()
    path = os.path.join(directory, "%s_crash.log" % what.replace(" ", "_"))
    try:
        with open(path, "w") as fsock:
            fsock.write(traceback.format_exc())
    except Exception:
        path = None
    log.warning("%s failed: %s", what, error)
    if path:
        log.warning("full traceback written to: %s", path)


def _off(value) -> bool:
    """True when a switch value means the tool was not selected."""
    return value in (None, "OFF", "Not Avail.", "Not Avail. (numpy missing)")


def _has_effective_handler(lg) -> bool:
    """True when ``lg`` (or an ancestor it propagates to) already has a handler,
    i.e. some other party is already displaying these records."""
    while lg:
        if lg.handlers:
            return True
        if not lg.propagate:
            return False
        lg = lg.parent
    return False


_TOOL_LOGGING_READY = False


def _setup_logging():
    """Make the reused madevent tool drivers' progress visible on screen.

    The drivers already narrate what they do through logger.info /
    update_status ("Running Pythia8 [arXiv:...]", "Splitting .lhe event
    file...", the live Idle/Running/Completed job counters, "Running
    MadSpin", ...), but the standalone mg7 launcher never attaches a handler
    to the 'madgraph'/'madevent' loggers, so all of it is swallowed. Attach a
    colored INFO console handler to them (idempotent)."""
    global _TOOL_LOGGING_READY
    if _TOOL_LOGGING_READY:
        return
    if _has_effective_handler(logging.getLogger("madgraph")):
        # Someone already configured logging for us -- in practice MG5, which
        # now runs the launch in its own process. Its configuration is
        # authoritative (it owns stdout_level and the tutorial logger); adding
        # a second handler here would print every line twice.
        _TOOL_LOGGING_READY = True
        return
    try:
        import madgraph.interface.coloring_logging  # registers ColorFormatter
        formatter = logging.ColorFormatter("%(message)s")
    except Exception:
        formatter = logging.Formatter("%(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler._mg7_tool_handler = True
    for name in ("madgraph", "madevent", "cmdprint", "madgraph7"):
        lg = logging.getLogger(name)
        if not any(getattr(h, "_mg7_tool_handler", False) for h in lg.handlers):
            lg.addHandler(handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    _TOOL_LOGGING_READY = True


def selected_tools(switch) -> list:
    """The post-processing drivers run_selected_tools runs for this switch."""
    # MadAnalysis5 hadron level analyses the shower/detector output, so it only
    # makes sense when a shower ran (mirrors madevent's card gating:
    # analysis == 'MadAnalysis5' and shower != 'OFF').
    ma5 = switch.get("analysis") == "MadAnalysis5"
    showered = not _off(switch.get("shower"))
    # reweighting is not one of them: run_single runs it (run_reweighting)
    return [t for t, on in (
        ("MadSpin", not _off(switch.get("madspin"))),
        ("MadAnalysis5 (parton level)", ma5),
        ("Pythia8 shower", switch.get("shower") == "Pythia8"),
        ("Delphes", switch.get("detector") == "Delphes"),
        ("MadAnalysis5 (hadron level)", ma5 and showered),
        ("Rivet", switch.get("analysis") == "Rivet"),
    ) if on]


def run_selected_tools(switch, process) -> None:
    """Run the optional post-processing programs selected in the merged
    question on the generated events.

    All the tools are driven through :class:`MG7RunCmd`, a thin adapter around
    the standard madevent run interface (``MadEventCmd``). This means the mg7
    output reuses the *exact* madevent tool drivers -- in particular Pythia8 is
    showered in parallel over the cluster/multicore backend, identically to a
    madevent run -- instead of re-implementing each tool here. The command
    sequence mirrors ``MadEventCmd.do_launch`` (MadSpin, MA5 parton, shower,
    Delphes, MA5 hadron, Rivet); each tool checks its own card, so a tool is
    invoked only when its switch is on. Reweighting, which comes first in
    that sequence, already ran (run_reweighting, from run_single).
    """
    log = logging.getLogger("madevent")

    tools = selected_tools(switch)
    if not tools:
        # A switch that is not off does not necessarily select a driver here:
        # "Not Avail." is not off, analysis = ExRoot has no mg7 driver, and a
        # shower switch set to something other than Pythia8 selects nothing
        # either. Without this a plain generate/output/launch announced a
        # post-processing step with nothing after the colon, and paid for
        # building the run interface (or an npy->LHE conversion) to do nothing.
        return

    lhe_path = _find_or_convert_event_file(process.run_path, log)
    if lhe_path is None:
        log.warning("No LHE event file in %s; cannot run %s.",
                    process.run_path, ", ".join(tools))
        return

    run_name = os.path.basename(os.path.dirname(os.path.abspath(lhe_path)))
    run_dir = os.path.dirname(os.path.abspath(lhe_path))

    ma5 = switch.get("analysis") == "MadAnalysis5"
    showered = not _off(switch.get("shower"))

    log.info("")
    log.info("Post-processing the generated events with: %s", ", ".join(tools))

    try:
        from madgraph.mg7.run_interface import MG7RunCmd
        cmd = MG7RunCmd(os.getcwd(), load_mg5_options(), run_name, lhe_path)
        cmd.set_run_name(run_name, None, "parton")
    except Exception as error:
        _report_failure(log, "post-processing setup", error, run_dir)
        return

    def run(tool, command):
        bar = "=" * 60
        log.info("")
        log.info(bar)
        log.info("  post-processing step: %s", tool)
        log.info(bar)
        start = time.time()
        try:
            cmd.exec_cmd(command, postcmd=False, printcmd=False)
        except Exception as error:
            _report_failure(log, tool, error, run_dir)
        else:
            log.info("  -> %s done (%.1fs)", tool, time.time() - start)

    # Same order as MadEventCmd.do_launch. The commands take no run name and so
    # act on cmd.run_name: decay_events (MadSpin) repoints cmd.run_name to the
    # "<run>_decayed_i" run, so the shower and the analyses that follow act on
    # the decayed events -- exactly as a madevent run does. Each driver still
    # guards on its own card.
    if not _off(switch.get("madspin")):
        run("MadSpin", "decay_events -from_cards")
    if ma5:
        run("MadAnalysis5 (parton)", "madanalysis5_parton --no_default")
    # Delphes has no step of its own, as in MadEventCmd.do_launch: do_pythia8
    # runs it on the shower output when Cards/delphes_card.dat exists. Calling
    # it here too ran the detector simulation a second time on the same HepMC.
    if switch.get("shower") == "Pythia8":
        run("Pythia8 shower", "shower --no_default")
    # hadron-level MA5 needs the shower/detector output: only run it if a shower
    # was requested (otherwise there is no hadron-level event file to analyse).
    if ma5 and showered:
        run("MadAnalysis5 (hadron)", "madanalysis5_hadron --no_default")
    if switch.get("analysis") == "Rivet":
        run("Rivet", "rivet --no_default")

    # Finalize the run exactly as MadEventCmd.do_launch does after the
    # shower/detector/analysis tools: store_result() processes the deferred
    # "to_store" actions -- in particular it gzips the Pythia8 HepMC to
    # <tag>_pythia8_events.hepmc.gz (the pythia8_card default is HEPMCoutput:file
    # = hepmc.gz). The deferred Rivet job reads exactly that .gz path, so this
    # must run before the Rivet/Contur postprocessor below.
    try:
        cmd.store_result()
    except Exception as error:
        _report_failure(log, "store_result", error, run_dir)

    if switch.get("analysis") == "Rivet":
        # do_rivet only *prepared* the run (wrote Events/<run>/run_rivet.sh) and
        # deferred execution to the postprocessor -- the rivet_card default has
        # run_rivet_later = True, so with --no_default it logged "Skipping Rivet
        # for now, passing it to postprocessor" and appended the run to
        # cmd.postprocessing_dirs. MadEventCmd.do_launch runs that postprocessor
        # at the end of a run; do the same here so the .yoda (and, when
        # run_contur = True, the Contur limits) are actually produced.
        bar = "=" * 60
        log.info("")
        log.info(bar)
        log.info("  post-processing step: Rivet/Contur postprocessing")
        log.info(bar)
        start = time.time()
        try:
            cmd.postprocessing()
        except Exception as error:
            _report_failure(log, "Rivet-Contur postprocessing", error, run_dir)
        else:
            log.info("  -> Rivet/Contur postprocessing done (%.1fs)",
                     time.time() - start)


def _add_time_of_flight(lhe_path, threshold, param_card_path, log):
    """Add invariant-lifetime (vtim) information to the LHE events, drawing each
    unstable particle's decay length from its width (in mm). Mirrors
    common_run_interface.CommonRunCmd.do_add_time_of_flight but reads the widths
    from the local param_card instead of the LHE banner."""
    import random
    try:
        import madgraph.various.lhe_parser as lhe_parser
    except ImportError:
        import internal.lhe_parser as lhe_parser
    from madgraph.models import check_param_card as param_card_mod
    from madgraph.various import misc as _misc
    import madgraph.iolibs.files as files

    need_zip = lhe_path.endswith('.gz')
    if need_zip:
        _misc.gunzip(lhe_path)
        lhe_path = lhe_path[:-3]

    param_card = param_card_mod.ParamCard(param_card_path)
    cst = 6.58211915e-25   # hbar in GeV s
    c = 299792458000       # speed of light in mm/s
    log.info('Adding time of flight information on %s', lhe_path)
    lhe = lhe_parser.EventFile(lhe_path)
    out = open('%s_2vertex.lhe' % lhe_path, 'w')
    out.write(lhe.banner)
    for event in lhe:
        for particle in event:
            # default=0 -> particles without a decay entry are treated as stable
            width = param_card['decay'].get((abs(particle.pid),), 0.).value
            if width:
                vtim = c * random.expovariate(width / cst)
                if vtim > threshold:
                    particle.vtim = vtim
        out.write(str(event))
    out.write('</LesHouchesEvents>\n')
    out.close()
    lhe.close()
    files.mv('%s_2vertex.lhe' % lhe_path, lhe_path)
    if need_zip:
        _misc.gzip(lhe_path)


def _lhapdf_config_path():
    """Path to lhapdf-config so systematics can import the python lhapdf
    module (required to compute PDF/scale variations)."""
    return lhapdf_paths().config


def _run_systematics(lhe_path, cfg, log):
    """Run systematics.py (scale/PDF variations) on the LHE file, reusing the
    exact worker madevent uses (systematics.call_systematics)."""
    try:
        import madgraph.various.systematics as systematics
    except ImportError:
        import internal.systematics as systematics

    def fmt(vals):
        return ','.join(str(v) for v in vals)

    opts = []
    if cfg.get('systematics_mur'):
        opts.append('--mur=%s' % fmt(cfg['systematics_mur']))
    if cfg.get('systematics_muf'):
        opts.append('--muf=%s' % fmt(cfg['systematics_muf']))
    if cfg.get('systematics_pdf'):
        opts.append('--pdf=%s' % fmt(cfg['systematics_pdf']))
    extra = cfg.get('systematics_str_options', '')
    if extra:
        opts.extend(extra.split())
    # The mg7 LHE has no <mgrwt> block, so systematics cannot read the per-event
    # LO reweighting info. For a process with a single alpha_s power (stored in
    # SubProcesses/proc_characteristics at output time) it can be reconstructed
    # from the events: pass that power as --lo_nqcd. -1 means the QCD power is
    # not uniform, so the reconstruction is not applicable.
    if not any(o.startswith('--lo_nqcd') for o in opts):
        try:
            from madgraph.various import banner as _banner_mod
        except ImportError:
            import internal.banner as _banner_mod
        pc_path = os.path.join('SubProcesses', 'proc_characteristics')
        if os.path.exists(pc_path):
            try:
                nqcd = int(_banner_mod.ProcCharacteristic(pc_path)['single_qcd_order'])
            except Exception:
                nqcd = -1
            if nqcd >= 0:
                opts.append('--lo_nqcd=%d' % nqcd)
    # tell systematics where to find lhapdf (so it can link the python module)
    if not any(o.startswith('--lhapdf_config') for o in opts):
        lhapdf_config = _lhapdf_config_path()
        if lhapdf_config:
            opts.append('--lhapdf_config=%s' % lhapdf_config)

    log.info('Running systematics on %s %s', lhe_path, ' '.join(opts))
    systematics.call_systematics([lhe_path, lhe_path] + opts,
                                 log=lambda x: log.info(str(x)))


def run_lhe_postprocessing(process) -> None:
    """Run the LHE-level post-processings configured in the run_card
    [postprocessing] section on the generated event file (displaced-vertex
    time-of-flight and systematics). Only applies when an LHE file exists."""
    lhe_path = _find_event_file(process.run_path)
    if lhe_path is None:
        return
    # process.run_card is a RunCardMG7; the section view exposes .get(key, def)
    try:
        cfg = process.run_card["postprocessing"]
    except Exception:
        return
    log = logging.getLogger('madevent')

    if cfg.get('systematics') and getattr(process, 'is_decay', False):
        # Scale and PDF variations are a beam quantity. A decay has neither, so
        # systematics can only fail here ("not supported for pdlabel=none") --
        # and it is not free: MadSpin reruns the launcher for every pool refill.
        log.info("decay mode: skipping systematics (no beams to vary)")
    elif cfg.get('systematics'):
        native = getattr(process, 'systematics', None)
        if native is not None and native.weight_count > 0:
            log.warning("[postprocessing] systematics is ignored: the scale/PDF "
                        "weights were already computed by madspace ([systematics] "
                        "section); set systematics.enable = false to use the legacy "
                        "systematics.py path instead.")
        else:
            try:
                _run_systematics(lhe_path, cfg, log)
            except Exception as error:
                _report_failure(log, "systematics computation", error,
                                os.path.dirname(lhe_path))

    tof = cfg.get('time_of_flight', -1.0)
    try:
        if tof is not None and float(tof) >= 0:
            _add_time_of_flight(lhe_path, float(tof), process.param_card_path, log)
    except Exception as error:
        _report_failure(log, "add_time_of_flight", error,
                        os.path.dirname(lhe_path))
