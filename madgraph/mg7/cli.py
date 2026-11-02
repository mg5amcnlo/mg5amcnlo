"""The MG7> command line interface and its card editing."""

import argparse
import json
import os
import re
import resource
import sys

from madgraph.mg7.common import logger
from madgraph.interface.extended_cmd import Cmd
from madgraph.various.banner import RunCardMG7
from madgraph.mg7 import npy_to_lhe
from madgraph.mg7.bootstrap import MG_ROOT as _MG_ROOT
from madgraph.mg7.driver import (
    ExistingRun,
    REWEIGHT_CARD,
    compute_decay_widths,
    force_lhe_output_if_needed,
    install_reweight_card,
    list_runs,
    prepare_reweighting,
    run_generation,
    run_reweighting,
)
from madgraph.mg7.model_widths import compute_auto_widths, load_process_model
from madgraph.mg7.options import lhapdf_index_paths, load_mg5_options
from madgraph.mg7.postprocess import _off, _setup_logging

class MG7Cmd(Cmd):
    """The mg7 run interface: the command object that drives a launch.

    It plays two roles, and they are the same object on purpose.

    1. It is the *mother* of the merged switch/card question (the thing
       ``AskRunEditCard`` calls back into for ``keep_cards``/``do_open``/
       ``compute_widths``).
    2. It is the child command interface MG5 registers via
       ``define_child_cmd_interface`` when you type ``launch`` on an mg7 output.

    Role 2 is what makes role 1 behave. ``Cmd.ask`` resolves a scripted answer
    through ``self.check_answer_in_input_file``, where ``self`` is the mother --
    so once MG5 hands us its ``inputfile`` and registers itself as our
    ``mother``, the question reads the very lines that follow ``launch`` in the
    user's command file, and any line it does not recognise goes back up to MG5
    through ``store_line`` instead of being silently consumed. This is exactly
    how a madevent output already behaves; before this class existed the mg7
    branch shelled out to ``bin/generate_events`` and had to guess which
    following lines belonged to the run, which quietly swallowed every ``set``.
    """

    prompt = 'MG7> '

    def __init__(self, me_dir='.', options=None, *args, **opts):
        super().__init__(*args, **opts)
        self.me_dir = os.path.abspath(me_dir) if me_dir != '.' else '.'
        # The rest of the launcher is written against the process directory as
        # the working directory, so do_generate_events chdirs there; but at
        # construction time MG5 has not moved, hence the explicit me_dir.
        self.options = load_mg5_options(
            self.me_dir if self.me_dir != '.' else None)
        if options:
            # MG5's own resolved tool paths win: they are the ones the user
            # configured in the session that is doing the launching.
            self.options.update({k: v for k, v in options.items()
                                 if k in self.options and v is not None})
            # and so does its error policy: with crash_on_error a failing
            # launch has to stop the script with a non-zero exit code, as a
            # madevent launch does, instead of returning to the prompt.
            if 'crash_on_error' in options:
                self.options['crash_on_error'] = options['crash_on_error']
        self.plugin_path = []
        self.proc_characteristics = {'grouped_matrix': False, 'limitations': []}
        self._model = None  # lazily imported by get_model()

    # ------------------------------------------------------------------
    # the contract the card question expects from its mother
    # ------------------------------------------------------------------
    def keep_cards(self, need_card=[], ignore=[]):
        from madgraph.interface.common_run_interface import CommonRunCmd
        return CommonRunCmd.keep_cards(self, need_card, ignore)

    def do_open(self, line):
        from madgraph.interface.common_run_interface import CommonRunCmd
        CommonRunCmd.do_open(self, line)

    def check_open(self, args):
        from madgraph.interface.common_run_interface import CommonRunCmd
        CommonRunCmd.check_open(self, args)

    def get_model(self):
        """The model this process was generated with (an ``import_ufo`` model).

        This is the hook the card question goes through for everything that
        needs the model rather than the card: 'update dependent' (which resets
        the masses/widths the model computes from the free parameters) and the
        generic auto-width handler both call ``mother_interface.get_model()``.
        Without it the launch question could only warn that it had failed to
        update the dependent parameters, and MadSpin/the shower then read
        whatever inconsistent values the param_card still held.

        None when the process recorded no model (an output made before
        SubProcesses/model.txt existed) -- the caller warns in that case.
        """
        if self._model is None:
            me_dir = self.me_dir if self.me_dir != '.' else None
            # False (not None) so a model that cannot be imported is not
            # retried at every question.
            self._model = load_process_model(me_dir) or False
        return self._model or None

    def do_compute_widths(self, line):
        # The interactive card editor delegates 'auto' width computation to
        # the mother interface. Reuse the runtime helper (madgraph subprocess
        # + the model stored at output time). ``line`` looks like
        # "<pdgs> --path=<param_card> [--nlo]"; we only need the card path.
        m = re.search(r'--path=(\S+)', line or "")
        path = m.group(1) if m else os.path.join("Cards", "param_card.dat")
        compute_auto_widths(path)
        # return an empty mapping: the caller iterates out.items() for the
        # small-width treatment, which mg7 does not apply.
        return {}

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    def do_generate_events(self, line):
        """generate_events [-f] [-n NAME] [--laststep=parton] [--reweight=CARD]
                           [--reweight-backend=classic|madtrex|auto]

        Run the mg7 generation and the selected post-processing tools.
        --reweight=CARD is a shortcut for switching reweighting on with CARD
        as Cards/reweight_card.dat; --reweight-backend chooses who reweights
        (see madgraph/various/reweight_backend.py)."""

        opts = self._parse_run_options(line)
        cwd = os.getcwd()
        target = self.me_dir if self.me_dir != '.' else cwd
        if opts['reweight_card']:
            opts['reweight_card'] = os.path.abspath(opts['reweight_card'])
        try:
            os.chdir(target)
            _setup_logging()
            if opts['name']:
                self._set_run_name(opts['name'])
            if opts['reweight_card']:
                install_reweight_card(opts['reweight_card'])
            switch = {}
            if not opts['force']:
                switch = ask_edit_cards(mother=self)
            switch = self._apply_laststep(switch, opts)
            if opts['reweight_card'] and _off(switch.get('reweight')):
                switch = dict(switch, reweight='ON')
            reweight = prepare_reweighting(switch, opts['reweight_backend'],
                                           self.options)
            force_lhe_output_if_needed(switch, reweight)
            _raise_open_file_limit()
            run_generation(switch, reweight)
        finally:
            os.chdir(cwd)

    # ``launch`` is the name MG5 users type; keep it working here too.
    do_launch = do_generate_events

    def do_reweight(self, line):
        """reweight RUN_NAME [--reweight=CARD] [--reweight-backend=classic|madtrex|auto]

        Reweight the events of an earlier run with Cards/reweight_card.dat
        (or CARD, which then becomes Cards/reweight_card.dat), as
        generate_events does after the generation: the weights are added to
        the run's event file, after the weights already there."""
        args = self.split_arg(line)
        run_name, card, backend = None, None, None
        for arg in args:
            if arg.startswith('--reweight='):
                card = os.path.abspath(arg.split('=', 1)[1])
            elif arg.startswith('--reweight-backend='):
                backend = arg.split('=', 1)[1]
            elif arg.startswith('-'):
                logger.warning("'%s' is not an option of reweight and is ignored", arg)
            else:
                run_name = arg
        cwd = os.getcwd()
        target = self.me_dir if self.me_dir != '.' else cwd
        try:
            os.chdir(target)
            _setup_logging()
            if run_name is None:
                runs = list_runs()
                raise self.InvalidCmd(
                    'reweight: give the run to reweight%s'
                    % (' (%s)' % ', '.join(runs) if runs else ': there is no run yet'))
            try:
                run = ExistingRun(run_name)
            except ValueError as error:
                raise self.InvalidCmd('reweight: %s' % error)
            if card:
                install_reweight_card(card)
            if not os.path.exists(REWEIGHT_CARD):
                raise self.InvalidCmd('reweight: %s is missing' % REWEIGHT_CARD)
            decision = prepare_reweighting({'reweight': 'ON'}, backend, self.options)
            if decision is None:
                return
            run_reweighting(run, REWEIGHT_CARD, decision=decision)
        finally:
            os.chdir(cwd)

    def complete_reweight(self, text, line, begidx, endidx):
        me_dir = self.me_dir if self.me_dir != '.' else os.getcwd()
        options = ['--reweight=', '--reweight-backend=']
        return self.list_completion(text, list_runs(os.path.join(me_dir, 'Events'))
                                    + options, line)

    def do_npy_to_lhe(self, line):
        """npy_to_lhe [RUN_NAME] [-o OUTPUT] [--seed N] [--no-gzip]

        Write the LHE file (events.lhe.gz) of a run that produced npy events,
        e.g. to shower it. Without a run name, the most recent such run."""
        import shlex
        me_dir = self.me_dir if self.me_dir != '.' else os.getcwd()
        try:
            npy_to_lhe.main(shlex.split(line), me_dir=me_dir)
        except SystemExit as error:
            if error.code not in (None, 0):
                logger.error(str(error.code))

    def do_quit(self, line):
        """Leave the mg7 run interface."""
        return super().do_quit(line)

    do_exit = do_quit

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _parse_run_options(self, line):
        """Understand the subset of MG5's `launch` options that mean something
        for an mg7 run, and say so out loud for the ones that do not."""
        out = {'force': False, 'name': '', 'laststep': '', 'reweight_card': None,
               'reweight_backend': None}
        args = self.split_arg(line)
        index = 0
        while index < len(args):
            arg = args[index]
            index += 1
            if arg in ('-f', '--force'):
                out['force'] = True
            elif arg in ('-n', '--name') and index < len(args):
                out['name'] = args[index]
                index += 1
            elif arg.startswith(('-n=', '--name=')):
                out['name'] = arg.split('=', 1)[1]
            elif arg in ('-s', '--laststep') and index < len(args):
                out['laststep'] = args[index]
                index += 1
            elif arg.startswith(('-s=', '--laststep=')):
                out['laststep'] = arg.split('=', 1)[1]
            elif arg == '--reweight' and index < len(args):
                out['reweight_card'] = args[index]
                index += 1
            elif arg.startswith('--reweight='):
                out['reweight_card'] = arg.split('=', 1)[1]
            elif arg == '--reweight-backend' and index < len(args):
                out['reweight_backend'] = args[index]
                index += 1
            elif arg.startswith('--reweight-backend='):
                out['reweight_backend'] = arg.split('=', 1)[1]
            elif arg.startswith('-'):
                logger.warning(
                    "'%s' is not supported by the mg7 output and is ignored; "
                    "the equivalent settings live in Cards/run_card.toml", arg)
            elif arg and not out['name']:
                out['name'] = arg
        return out

    def _set_run_name(self, name):
        """Honour `launch -n NAME`: the mg7 run directory is
        Events/<run.run_name>_NN, so the name is a run_card setting."""
        path = os.path.join("Cards", "run_card.toml")
        run_card = RunCardMG7(path, consistency=False)
        if run_card["run"]["run_name"] != name:
            run_card["run"]["run_name"] = name
            run_card.write(path)
            logger.info("run name set to '%s'", name)

    def _apply_laststep(self, switch, opts):
        """`--laststep=parton` means: generate the events, run nothing on them.

        The mg7 tool selection is the switch dict the question returns, so
        stopping at parton level is switching those entries off.
        """
        if opts['laststep'] not in ('parton', 'auto', ''):
            logger.warning(
                "--laststep=%s is not supported by the mg7 output; "
                "select the programs to run in the launch question instead",
                opts['laststep'])
            return switch
        if opts['laststep'] != 'parton' or not switch:
            return switch
        switch = dict(switch)
        for key in ('shower', 'detector', 'analysis', 'madspin', 'reweight'):
            if not _off(switch.get(key, 'OFF')):
                logger.info("--laststep=parton: not running %s", key)
                switch[key] = 'OFF'
        return switch


def build_selector_cmd(mother=None):
    """Build the (monkey-patched) mother command + merged switch/card selector
    used by the mg7 output.  Returns the selector *class* and a mother instance
    understood by AskRun/AskforEditCard.

    ``mother`` lets an in-process caller supply its own :class:`MG7Cmd` -- the
    one MG5 registered as its child -- so that the question resolves scripted
    answers against MG5's own command file (see :class:`MG7Cmd`). When it is
    omitted a standalone instance is built, which is what ``bin/generate_events``
    gets.
    """

    from madgraph.interface.madevent_interface import AskRunEditCard

    from madgraph.various import banner as _banner_mod
    from madgraph.various import misc as _misc

    class MG7Selector(AskRunEditCard):
        """Merged switch/card question for the mg7 output.

        The mg7 run_card is a TOML file, so define_paths/init_run/do_set are
        overridden *on this class* (not globally on AskforEditCard) to treat it
        as such. Scoping them here is important: MadSpin/reweight spawn their own
        legacy madevent runs in the same process, and a global patch would break
        their (run_card.dat based) card editing."""

        # param_card + the TOML run_card are always offered
        always_cards = ['param_card.dat', 'run_card.toml']
        optional_cards = []

        def define_paths(self, **opt):
            super().define_paths(**opt)
            self.paths["run"] = os.path.join(self.me_dir, "Cards", "run_card.toml")
            self.paths["run_card.toml"] = os.path.join(self.me_dir, "Cards", "run_card.toml")
            # the TOML run_card uses its own default file (concrete defaults
            # written at output time); this powers "set <param> default".
            self.paths["run_default"] = os.path.join(self.me_dir, "Cards", "run_card_default.toml")

        def init_run(self, cards):
            # Make sure the run_card is loaded as a RunCardMG7 (the generic
            # editor does not recognise run_card.toml), else "set <param>" fails.
            out = super().init_run(cards)
            if not isinstance(getattr(self, "run_card", None), RunCardMG7):
                toml_path = self.paths.get("run") or os.path.join(
                    self.me_dir, "Cards", "run_card.toml")
                if os.path.exists(toml_path):
                    try:
                        # allow_scan so a run_card holding scan:[...] values loads
                        with _misc.TMP_variable(_banner_mod.RunCard, "allow_scan", True):
                            self.run_card = RunCardMG7(toml_path, consistency="warning")
                        self.run_set = list(self.run_card.keys())
                    except Exception as err:
                        logger.warning("could not load %s: %s", toml_path, err)
            if isinstance(getattr(self, "run_card", None), RunCardMG7):
                self.run_card.allow_scan = True
            return getattr(self, "run_set", out)

        def check_card_consistency(self):
            # madevent settings given per beam ("set fixed_fac_scale1 ...")
            # that the mg7 card could not follow: said once, at the end, so
            # "set fixed_fac_scale1 T" then "set fixed_fac_scale2 T" stays
            # quiet. Collected first: lo_set_warnings also forgets them, which
            # must happen even if the generic checks below raise.
            warnings = []
            if isinstance(getattr(self, "run_card", None), RunCardMG7):
                warnings = self.run_card.lo_set_warnings()
            try:
                super().check_card_consistency()
            finally:
                for text in warnings:
                    logger.warning(text)

        def do_set(self, line, *args, **kwargs):
            # madevent-style shortcuts (lhc/lep/fixed_scale/no_parton_cut), cut
            # editing, energy units and arithmetic/mass expressions on the TOML
            # run_card, intercepted before delegating to the generic editor.
            targs = self.split_arg(line)
            run_card = getattr(self, "run_card", None)
            if isinstance(run_card, RunCardMG7) and targs:
                start = 1 if targs[0] == "run_card" else 0
                if len(targs) > start:
                    name = targs[start]
                    nlow = name.lower()
                    rest = " ".join(targs[start + 1:]).split("#")[0].strip()
                    masses = run_card.get_mass_shortcuts(getattr(self, "param_card", None))

                    if nlow in ("no_parton_cut", "nocut", "no_cut"):
                        run_card.remove_all_cut()
                        logger.info("removing all cuts from the run_card.toml")
                        self.modified_card.add("run")
                        return
                    # [histograms] is a list of observables, not a parameter,
                    # so "set histograms ..." can only mean the whole section.
                    if nlow in ("histograms", "histogram", "plots"):
                        value = rest.lower()
                        if value in ("off", "none", "no", "false", "0"):
                            run_card.remove_all_histograms()
                            logger.info(
                                "removing all histograms from the run_card.toml")
                            self.modified_card.add("run")
                            return
                        if value in ("default", "on"):
                            if run_card.restore_default_histograms(
                                    self.paths.get("run_default")):
                                logger.info(
                                    "restored the %d histograms written at "
                                    "output time",
                                    len(run_card["histograms"]))
                                self.modified_card.add("run")
                            else:
                                self.invalid_set(
                                    "no run_card_default.toml to restore the "
                                    "histograms from")
                            return
                        self.invalid_set(
                            "'set histograms %s': only OFF (remove them all) "
                            "and default (put the generated ones back) are "
                            "understood; edit the [histograms] section of "
                            "Cards/run_card.toml for anything else"
                            % (rest or "(no value)"))
                        return
                    if nlow in ("lhc", "lep", "ilc", "lcc") and rest:
                        try:
                            ecm = run_card.set_collider(nlow, rest, masses)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (nlow, rest, error))
                            return
                        logger.info("set %s collider: e_cm = %s GeV", nlow, ecm)
                        self.modified_card.add("run")
                        return
                    if nlow == "fixed_scale" and rest:
                        try:
                            val = run_card.set_fixed_scale(rest, masses)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (nlow, rest, error))
                            return
                        logger.info("set fixed scales to %s GeV", val)
                        self.modified_card.add("run")
                        return

                    # the generic editor would say "Ambiguous key 'pdf'"
                    # without naming the per-beam entries
                    if nlow == "pdf" and rest:
                        self.invalid_set("Ambiguous key 'pdf' — use beam.pdf (both beams), "
                                         "beam.pdf1, beam.pdf2 or systematics.pdf")
                        return
                    # custom function parameters are not run_card keys, so the
                    # generic editor rejects them
                    if rest and nlow.startswith("custom_functions.") and nlow.count(".") == 2:
                        # the parameter name is matched as written by the
                        # custom function: keep its case
                        try:
                            run_card.set(name, run_card.evaluate(rest, masses), user=True)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (name, rest, error))
                            return
                        logger.info("modify parameter %s of the run_card.toml to %s", name, rest)
                        self.modified_card.add("run")
                        return

                    if rest and run_card.is_cut_name(name):
                        try:
                            cut, bound, val = run_card.set_cut(name, rest, masses)
                        except _banner_mod.InvalidCmd as error:
                            # an unknown bound, a value that is not a number,
                            # or a unit on a cut that is not an energy
                            self.invalid_set("invalid 'set %s %s': %s" % (name, rest, error))
                            return
                        logger.info("modify cut %s.%s of the run_card.toml to %s", cut, bound, val)
                        self.modified_card.add("run")
                        return

                    # legacy madevent cut names ("set ptj 20", "set mmll 200",
                    # "set etaj -1", ...) -> the [cuts] entry the LO->MG7
                    # run_card conversion maps them to (_LO_CUT_MAP), with the
                    # same "no cut" sentinels (minimum <= 0, maximum < 0).
                    if rest and nlow in run_card._LO_CUT_MAP:
                        if rest.lower().startswith("scan"):
                            self.invalid_set("invalid 'set %s %s': the [cuts] entries of "
                                             "the mg7 run_card.toml cannot be scanned"
                                             % (nlow, rest))
                            return
                        try:
                            cut, bound, val = run_card.set_lo_cut(nlow, rest, masses)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (nlow, rest, error))
                            return
                        if val is None:
                            logger.info("set %s (mg7 cut %s.%s) of the run_card.toml: "
                                        "no cut", nlow, cut, bound)
                        else:
                            logger.info("set %s (mg7 cut %s.%s) of the run_card.toml to %s",
                                        nlow, cut, bound, val)
                        self.modified_card.add("run")
                        return

                    # madevent parameters that are not a plain rename: the
                    # value is translated ("set dynamical_scale_choice 3") or
                    # several of them make one mg7 entry (fixed_fac_scale1 +
                    # fixed_fac_scale2 -> beam.fixed_fact_scale). mg7 values
                    # ("set sde_strategy denominators") come back as None and
                    # go on to the generic editor.
                    if rest and nlow in run_card._LO_SET_PARAMS:
                        try:
                            done = run_card.set_lo_param(nlow, rest, masses)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (nlow, rest, error))
                            return
                        if done is not None:
                            changes, warnings = done
                            for key, val in changes.items():
                                if val is None:
                                    logger.info("set %s: removed %s from the run_card.toml",
                                                nlow, key)
                                else:
                                    logger.info("set %s (mg7 %s) of the run_card.toml to %s",
                                                nlow, key, val)
                            for text in warnings:
                                logger.warning(text)
                            if changes:
                                self.modified_card.add("run")
                            elif not warnings:
                                logger.info("set %s %s: nothing to change in the run_card.toml",
                                            nlow, rest)
                            return

                    # 'iseed' is the madevent spelling of the mg7 run.seed,
                    # but the two disagree on how to ask for a random seed:
                    # madevent uses 0, mg7 uses -1 (0 being a perfectly good
                    # fixed seed there). Translate here rather than in
                    # _LO_SCALAR_MAP, which also drives the LO -> MG7 run_card
                    # conversion and would silently pin those runs to seed 0.
                    if rest and nlow == 'iseed':
                        try:
                            value = int(run_card.evaluate(rest, masses, energy=False,
                                                          name=nlow))
                        except (TypeError, ValueError, _banner_mod.InvalidCmd):
                            value = None
                        if value is None:
                            self.invalid_set("invalid 'set iseed %s': not an integer" % rest)
                            return
                        seed = -1 if value == 0 else value
                        run_card.set('run.seed', seed, user=True)
                        logger.info("set iseed (mg7 run.seed) of the run_card.toml to %s%s",
                                    seed, " (random)" if seed == -1 else "")
                        self.modified_card.add("run")
                        return

                    # madevent beam settings (ebeam, lpp, pdlabel, lhaid, per
                    # beam or for both) onto the per-beam [beam] entries
                    if rest and nlow in run_card.lo_beam_params:
                        data_paths = lhapdf_index_paths() if nlow == "lhaid" else ()
                        try:
                            done = run_card.set_lo_beam_param(nlow, rest, masses, data_paths)
                        except _banner_mod.InvalidRunCard as error:
                            text = "'set %s %s': %s" % (name, rest, error)
                            # an unknown lhaid is reported, not rejected, in every mode
                            if nlow == "lhaid" and isinstance(error, _banner_mod.UnknownPDFSet):
                                logger.warning("ignoring " + text)
                            else:
                                self.invalid_set("invalid " + text)
                            return
                        logger.info("set %s %s: %s", nlow, rest, done)
                        self.modified_card.add("run")
                        return

                    # quantities the card stores per beam: set both beams
                    if rest and nlow in ("e_cm", "beam.e_cm", "beam.pdf"):
                        try:
                            value = rest if nlow == "beam.pdf" else run_card.evaluate(rest, masses)
                            run_card.set(nlow, value, user=True)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (nlow, rest, error))
                            return
                        logger.info("set %s of the run_card.toml to %s (both beams)",
                                    nlow, run_card["beam." + nlow.split(".")[-1]])
                        self.modified_card.add("run")
                        return

                    # legacy madevent run_card names -> mg7 "section.key", so a
                    # madevent-style launch script ("set nevents 500", "set
                    # use_syst F", "set bwcutoff 10", ...) edits the mg7
                    # run_card.toml verbatim. RunCardMG7._LO_SCALAR_MAP is the
                    # same rename table used by the LO->MG7 run_card conversion.
                    if rest and nlow in run_card._LO_SCALAR_MAP \
                            and nlow not in [k.lower() for k in run_card.keys()]:
                        target = run_card._LO_SCALAR_MAP[nlow]
                        try:
                            run_card.set(target, rest, user=True)
                        except _banner_mod.InvalidCmd as error:
                            self.invalid_set("invalid 'set %s %s': %s" % (nlow, rest, error))
                            return
                        logger.info("set %s (mg7 %s) of the run_card.toml to %s",
                                    nlow, target, run_card[target])
                        self.modified_card.add("run")
                        return

                    if rest and nlow in [k.lower() for k in run_card.keys()]:
                        current = run_card[nlow]
                        if isinstance(current, (int, float)) and not isinstance(current, bool):
                            try:
                                resolved = run_card.evaluate(
                                    rest, masses, name=nlow,
                                    energy=nlow in run_card.energy_parameter)
                            except _banner_mod.InvalidCmd as error:
                                self.invalid_set("invalid 'set %s %s': %s" % (name, rest, error))
                                return
                            if not isinstance(resolved, str):
                                prefix = "run_card " if start == 1 else ""
                                line = "%s%s %s" % (prefix, name, resolved)
            return super().do_set(line, *args, **kwargs)

    return MG7Selector, (mother if mother is not None else MG7Cmd())


def ask_edit_cards(mother=None) -> dict:
    """Single (MadDM-style) question letting the user both pick which programs
    to run after generation and edit the associated cards.  Returns the switch
    dict describing the selected tools.

    ``mother`` is the :class:`MG7Cmd` asking the question. Passing the instance
    MG5 registered as its child is what lets a scripted ``launch`` answer this
    question from MG5's own command file.
    """

    selector_class, mother = build_selector_cmd(mother)
    # path_msg is what makes Cmd.check_answer_in_input_file accept a bare path as
    # an answer (its "elif path:" branch), so that a scripted launch can hand the
    # question a card/banner path -- as the question itself advertises -- and have
    # it replace the corresponding card. Without it the path is rejected ("This
    # answer is not valid for current question") and the default is used instead.
    try:
        switch, question = mother.ask('', '0', [], path_msg='enter path',
                                      ask_class=selector_class,
                                      mode='auto', line_args=[], force=False,
                                      return_instance=True)
    except BaseException:
        # interrupted (e.g. ctrl-C) before the prune below could run: prune
        # here too, using the instance ask() stashed pre-interrupt, or the
        # eagerly-materialised tool cards get misread as "left ON" next time.
        instance = getattr(mother, '_last_ask_instance', None)
        if instance is not None and hasattr(instance, 'active_cards'):
            try:
                mother.keep_cards(instance.active_cards())
            except Exception as error:
                logger.debug('could not prune tool cards after aborted launch question: %s', error)
        raise
    switch = dict(switch)
    prune_unselected_tool_cards(mother, question, switch)
    return switch


def prune_unselected_tool_cards(mother, question, switch) -> None:
    """Hide the cards of the tools that were NOT selected in the question.

    The question materialises every candidate tool card (copying the
    *_default.dat) so that it can offer them all for edition; madevent's
    ask_run_configuration prunes them again afterwards (keep_cards on the
    selected cards), but the mg7 launcher had no such cleanup. The materialised
    cards therefore persisted, and a later re-launch of the same output saw them
    all present and defaulted every switch to ON (set_default_<tool> keys off
    card presence). Prune here too, using the question's own switch->card map, so
    a re-launch reflects the previous selection instead of turning everything on.
    """
    try:
        keep = list(question.always_cards)
        for spec in question.switch_cards:
            if spec['on'](switch):
                keep.append(spec['card'])
        mother.keep_cards(keep)
    except Exception as error:
        logger.debug('could not prune unselected tool cards: %s', error)


def _raise_open_file_limit() -> None:
    """Remove the soft limit on the number of open files: it can be quite low
    on some systems and the event-combination step opens one file per channel."""
    soft_lim, hard_lim = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_lim, hard_lim))
    except (ValueError, OSError) as error:
        logger.debug("could not raise the open-file limit: %s", error)


def _stdin_is_redirected() -> bool:
    """True when stdin is a pipe, a socket or a file, i.e. neither a terminal
    nor a character device such as /dev/null (batch jobs)."""
    import stat
    try:
        return not stat.S_ISCHR(os.fstat(sys.stdin.fileno()).st_mode)
    except (OSError, ValueError, AttributeError):
        return False


def _stdin_has_input() -> bool:
    """True when something is waiting on stdin. A forced (-f) run never reads
    stdin, so a closed or idle pipe (CI runners, agents) is harmless there,
    but piped answers would be silently dropped."""
    import select
    try:
        if not select.select([sys.stdin], [], [], 0)[0]:
            return False
        return bool(os.read(sys.stdin.fileno(), 1))
    except (OSError, ValueError, AttributeError):
        return False


def main() -> None:
    """``bin/generate_events`` entry point.

    It runs the same :class:`MG7Cmd` command MG5's in-process ``launch`` uses,
    so the two entry points cannot drift apart; the only difference is that
    nothing here supplies a ``mother``/``inputfile``, so the question falls back
    to reading this process's own stdin.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", action="store_false", dest="ask_edit_cards")
    parser.add_argument(
        "--decay-widths", metavar="JSON", default=None,
        help="only integrate the 1 -> n decays of this directory and write "
             "their widths (GeV) as JSON, see compute_decay_widths")
    parser.add_argument(
        "--precision", type=float, default=0.01,
        help="relative precision aimed at for every channel (--decay-widths)")
    parser.add_argument(
        "--nb-core", type=int, default=None,
        help="number of threads and compilation jobs (--decay-widths)")
    parser.add_argument(
        "--make-arg", action="append", default=[],
        help="extra argument of the matrix element build (--decay-widths)")
    parser.add_argument(
        "--reweight", dest="reweight_card", default=None,
        help="reweight the events after generation with this reweight card "
             "(it becomes Cards/reweight_card.dat and switches reweighting on)",
    )
    parser.add_argument(
        "--reweight-backend", dest="reweight_backend", default=None,
        choices=("classic", "madtrex", "auto"),
        help="who reweights: the classic module, MadtRex, or auto (the default: "
             "MadtRex when it can run the card)",
    )
    parser.add_argument(
        "--allow-stdin", action="store_true", default=False,
        help="read the answers to the questions piped or redirected into stdin, "
             "as an interactive session: invalid \"set\" commands then only warn")
    args = parser.parse_args()

    # The answers to the questions are read from stdin as in an interactive
    # session, where a command that cannot be applied (e.g. an invalid "set")
    # is only a warning: refuse that mode unless asked for. --decay-widths and
    # a forced run ask nothing; the latter is still refused when answers are
    # waiting on stdin, which it would silently drop.
    if not args.allow_stdin and not args.decay_widths and _stdin_is_redirected() \
            and (args.ask_edit_cards or _stdin_has_input()):
        sys.exit("Reading the answers to the questions from stdin is not supported: "
                 "write the line 'launch %s' and the answers to a file and run "
                 "'%s PATH' instead, so that an invalid command stops the run.\n"
                 "Use --allow-stdin to read stdin as an interactive session (invalid "
                 "commands then only warn)."
                 % (os.getcwd(), _MG_ROOT / 'bin' / 'madgraph'))

    if args.decay_widths:
        results = compute_decay_widths(
            ".", precision=args.precision, nb_core=args.nb_core,
            extra_make_args=args.make_arg)
        with open(args.decay_widths, "w") as f:
            json.dump(results, f)
        return

    line = "generate_events" if args.ask_edit_cards else "generate_events -f"
    if args.reweight_card:
        line += " --reweight=%s" % args.reweight_card
    if args.reweight_backend:
        line += " --reweight-backend=%s" % args.reweight_backend
    MG7Cmd().run_cmd(line)
