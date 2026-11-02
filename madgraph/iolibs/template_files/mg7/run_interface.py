"""Reuse the madevent post-processing tool drivers from the standard (mg7)
output.

The mg7 output is a lean C++/madspace directory: it has no Fortran ``Source``
to compile, its run card is ``run_card.toml`` (not ``run_card.dat``) and its
events live in ``Events/<run>/events.lhe``. The madevent interface
(:class:`~madgraph.interface.madevent_interface.MadEventCmd`) already knows how
to run Pythia8 (in parallel, over the cluster/multicore backend), MadSpin,
reweighting, Delphes, Rivet and MadAnalysis5 -- but it assumes the madevent
directory layout.

:class:`MG7RunCmd` is a thin adapter subclass that reconciles the two so that
the mg7 launcher can simply do ``cmd.exec_cmd('pythia8 --no_default run_01')``
and inherit the real (parallel) shower code, etc., instead of re-implementing
each tool.
"""

import contextlib
import json
import logging
import os
import shutil

import madgraph.interface.madevent_interface as madevent_interface
import madgraph.madevent.gen_crossxhtml as gen_crossxhtml
import madgraph.various.banner as banner_mod
import madgraph.various.lhe_parser as lhe_parser
from madgraph.various import misc

pjoin = os.path.join
logger = logging.getLogger('madevent')


def sync_lhe_copy(source, target):
    """Make the LHE file target hold the events of source, gzipped or not
    as each file's name says; replaced atomically. The mg7 run's event file
    (events.lhe[.gz]) and the copy the madevent tools read
    (unweighted_events.lhe.gz, see MG7RunCmd._prepare_run_dir) are kept the
    same this way after a reweighting wrote into one of them."""
    import gzip
    tmp = target + '.tmp'
    opener_in = gzip.open if source.endswith('.gz') else open
    opener_out = gzip.open if target.endswith('.gz') else open
    if source.endswith('.gz') == target.endswith('.gz'):
        shutil.copyfile(source, tmp)
    else:
        with opener_in(source, 'rb') as src, opener_out(tmp, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    os.replace(tmp, target)


@contextlib.contextmanager
def _quiet_setup():
    """Silence the INFO chatter of a madevent interface being constructed.

    Raises the loggers that carry it to WARNING for the duration, so the
    banner, the configuration lines and the shell-tool notes are dropped while
    anything that actually needs saying still gets through.
    """

    # three different loggers carry the setup chatter: the MADEVENT banner
    # (madevent.stdout), the "load configuration from ..." lines
    # (madgraph.stdout, from CommonRunCmd.set_configuration) and the shell-tool
    # notes such as "Using default gzip" (cmdprint.ext_program, from misc)
    names = ('madevent.stdout', 'madgraph.stdout', 'cmdprint.ext_program')
    previous = {}
    for name in names:
        lg = logging.getLogger(name)
        previous[name] = lg.level
        lg.setLevel(max(lg.level, logging.WARNING))
    try:
        yield
    finally:
        for name, level in previous.items():
            logging.getLogger(name).setLevel(level)


class MG7Results(gen_crossxhtml.AllResults):
    """The madevent results database, kept in memory only.

    The tool drivers book their runs here (``add_run``, ``add_detail``,
    ``current``), so :class:`MG7RunCmd` needs one. But madevent also writes it
    to disk: ``save`` pickles it to ``HTML/results.pkl`` and ``output`` renders
    ``crossx.html``, the status page of a madevent directory. The mg7 output
    has no ``HTML/`` directory, so ``save`` -- called by ``store_result`` at
    the end of every post-processing run -- failed with "I/O error (2): No such
    file or directory", and ``crossx.html`` was left behind at the process root
    still reading "Currently Running" and linking to pages that do not exist.
    Nothing reads either back (:meth:`MG7RunCmd.load_results_db` starts a fresh
    database every time), so neither is written.
    """

    def save(self):
        return

    def output(self):
        return


class MG7RunCmd(madevent_interface.MadEventCmd):
    """A :class:`MadEventCmd` pointed at an mg7 output directory.

    Only the pieces that are genuinely madevent-specific are overridden:

    * ``configure_directory`` -- the base version compiles the Fortran
      ``Source`` and links LHAPDF; none of that exists / is needed here.
    * ``set_run_name`` -- loads the run card from ``run_card.toml``
      (:class:`RunCardMG7`) and takes the banner straight from the mg7 LHE,
      instead of reading ``run_card.dat`` / juggling banner files.
    * ``do_treatcards`` -- writes Fortran ``.inc`` files; not applicable.
    * ``load_results_db`` -- an in-memory :class:`MG7Results`: there is no
      ``HTML/`` directory to keep it in.
    """

    def __init__(self, me_dir, options, run_name, lhe_path):
        self._mg7_run_name = run_name
        self._mg7_lhe_path = os.path.abspath(lhe_path)
        self._mg7_banner_cache = {}
        # Seed the full madevent default option set (run_mode, nb_core,
        # cluster_*, tool paths, ...) before layering the caller's options on
        # top: CommonRunCmd only fills these defaults when it receives an empty
        # options dict, and the tool drivers read e.g. options['run_mode'].
        merged = dict(self.options_configuration)
        merged.update(self.options_madgraph)
        merged.update(self.options_madevent)
        merged.update(options or {})
        compression = banner_mod.RunCardMG7(
            pjoin(me_dir, 'Cards', 'run_card.toml')).gzip_configuration()
        # Make the run directory look enough like a madevent run for the
        # results-db scan (and the tools' event-file lookup) to be happy.
        misc.configure_gzip(compression)
        self._prepare_run_dir()
        # MadEventCmd.__init__ announces itself with the MADEVENT welcome
        # banner and a run of "load configuration from ..." lines. That is
        # right for `./bin/madevent`, which is an interface someone just
        # started; this is an internal adapter built to drive a tool, in the
        # middle of a run that has already introduced itself. Keep the setup
        # quiet -- warnings and errors still come through.
        with _quiet_setup():
            super(MG7RunCmd, self).__init__(me_dir, merged, force_run=True)
        # MadEventCmd reloads its own compression configuration during setup.
        misc.configure_gzip(compression)

    # ------------------------------------------------------------------
    # directory / run-state adaptation
    # ------------------------------------------------------------------
    def _prepare_run_dir(self):
        """Provide the file names the madevent tools look for: a gzipped
        ``unweighted_events.lhe.gz`` next to the mg7 event file.

        The mg7 event file may already be gzipped (``events.lhe.gz``, the
        default) or plain (``events.lhe``). Compressing an already-gzipped
        source would produce a doubly-gzipped file: every consumer would
        decompress it once, get gzip bytes instead of LHE text, and read zero
        events. So copy when the source is already compressed and only gzip a
        plain source."""
        run_dir = os.path.dirname(self._mg7_lhe_path)
        gz = pjoin(run_dir, 'unweighted_events.lhe.gz')
        if not os.path.exists(gz) and os.path.exists(self._mg7_lhe_path):
            if self._mg7_lhe_path.endswith('.gz'):
                shutil.copyfile(self._mg7_lhe_path, gz)
            else:
                misc.gzip(self._mg7_lhe_path, stdout=gz, keep=True)

    def load_results_db(self):
        """Fresh results database without recreating old runs from banners: the
        mg7 LHE banner lacks the <MGGenerationInfo> block that madevent's
        recreate() expects. The run we post-process is registered explicitly in
        :meth:`set_run_name`."""
        model = self.find_model_name()
        self.results = MG7Results(model, self.process, self.me_dir,
                                  recreateold=False)
        self.last_mode = ''
        return self.results

    def _event_file_for_run(self, name):
        """The LHE of run ``name``; falls back to the original mg7 LHE. Lets a
        run created downstream (e.g. ``<run>_decayed_1`` from MadSpin) provide
        its own events/banner so the shower runs on the decayed events."""
        run_dir = pjoin(self.me_dir, 'Events', name)
        for fname in ('unweighted_events.lhe.gz', 'unweighted_events.lhe',
                      'events.lhe.gz', 'events.lhe'):
            path = pjoin(run_dir, fname)
            if os.path.exists(path):
                return path
        return self._mg7_lhe_path

    def _banner_for_run(self, name):
        """Banner (slha + MG7RunCard + init + proc card) of run ``name``,
        parsed once per run."""
        if name not in self._mg7_banner_cache:
            lhe = lhe_parser.EventFile(self._event_file_for_run(name))
            self._mg7_banner_cache[name] = banner_mod.Banner(lhe.banner)
        return self._mg7_banner_cache[name]

    def set_run_name(self, name, tag=None, level='parton', reload_card=False,
                     allow_new_tag=True):
        """mg7 flavour of :meth:`MadEventCmd.set_run_name`.

        Loads the run card from the TOML card and takes the banner from the
        run's own LHE, then registers the run with the results database so the
        standard tool drivers (which store their output there) work unchanged.
        """
        self.run_name = name
        self.run_card = banner_mod.RunCardMG7(
            pjoin(self.me_dir, 'Cards', 'run_card.toml'))
        self.run_tag = tag or 'tag_1'
        self.run_card['run_tag'] = self.run_tag

        if name not in self.results:
            self.results.add_run(name, self.run_card)
            self._register_generation_results(name)
        else:
            self.results.def_current(name, self.run_tag)

        self.banner = self._banner_for_run(name)
        # As MadEventCmd.set_run_name, return the tag of the run holding the
        # input of `level`: check_delphes looks for <tag>_pythia8_events.hepmc.gz,
        # and silently skips Delphes (--no_default) when it is None_pythia8_...
        # An mg7 run has one tag, so the shower output carries run_tag.
        if level == 'parton':
            return None
        return self.run_tag

    def _register_generation_results(self, name):
        """Book the cross section, its error and the number of events of the
        generated sample on the run just added to the results database.

        madevent fills them during survey/refine, which an mg7 run does not go
        through: the tools' "Results Summary" (printed after the Pythia8 shower,
        for instance) then read "Cross-section : 0 +- 0 pb, Nb of events : 0".
        They come from the info.json of the run (the numbers the mg7 run
        itself printed), else from the <init> block of its LHE. A run booked
        by a tool (MadSpin's <run>_decayed_i) is added with its own numbers
        and does not come here.
        """
        cross = error = nb_event = None
        try:
            with open(pjoin(self.me_dir, 'Events', name, 'info.json')) as fsock:
                status = json.load(fsock)['process']
            cross, error = float(status['mean']), float(status['error'])
            nb_event = int(status['count_unweighted'])
        except Exception:
            try:
                cross, error = self._banner_for_run(name).get_cross(witherror=True)
            except Exception:
                return
        self.results.add_detail('cross', cross)
        self.results.add_detail('error', error)
        if nb_event is not None:
            self.results.add_detail('nb_event', nb_event)

    def configure_directory(self, html_opening=True):
        """Lightweight replacement: the mg7 output has no Fortran ``Source`` to
        compile and no ``run_card.dat``. Set only what the tools consult."""
        if self.options.get('heptools_install_dir'):
            libdir = os.path.abspath(
                pjoin(self.options['heptools_install_dir'], 'lib'))
            for var in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
                os.environ[var] = libdir + ':' + os.environ.get(var, '')
        self.make_opts_var = {}
        if not hasattr(self, 'random'):
            try:
                self.random = int(self.run_card['iseed']) or 1234
            except Exception:
                self.random = 1234
        return

    def do_treatcards(self, line, mode=None, opt=None):
        """No Fortran include files to generate for the mg7 output."""
        return

    def get_model(self):
        """The model of the process, read from ``SubProcesses/model.txt``.

        The inherited version imports ``bin/internal/ufomodel``, the copy a
        madevent output carries; the mg7 output has none, it records which
        model it was generated with instead. Same hook, same object -- so the
        tools this class drives (MadSpin, reweight, the auto widths and the
        'update dependent' pass of their card questions) work here too.
        """
        from madgraph.iolibs.template_files.mg7.launch import load_process_model
        return load_process_model(self.me_dir)

    def do_reweight(self, line):
        """``reweight [--backend=classic|madtrex|auto] ...``: the backend
        option (also spelled ``--reweight-backend=``) chooses who reweights,
        together with the reweight switch and the card's ``change backend``
        (see madgraph/various/reweight_backend.py). MadtRex reweights this
        run's events (an MG7 output); the classic module is the inherited
        command, without the option."""
        from madgraph.various import reweight_backend as rb
        requested = None
        words = []
        for word in line.split():
            if word.startswith(('--backend=', '--reweight-backend=')):
                requested = word.split('=', 1)[1]
            else:
                words.append(word)
        line = ' '.join(words)
        card_path = pjoin(self.me_dir, 'Cards', 'reweight_card.dat')
        if '-from_cards' in words and not os.path.exists(card_path):
            return
        if requested == 'classic' or not os.path.exists(card_path):
            return self._classic_reweight(line)
        with open(card_path) as fsock:
            traits = rb.classify_card(fsock.read())
        plugin = [w.split('=', 1)[1] for w in words if w.startswith('--plugin=')]
        if '--mode=density' in words:
            traits.density = True
        decision = rb.select_backend(
            [('reweight --backend', requested)], traits,
            rb.probe_environment(self.me_dir, self.options),
            plugin=plugin[0] if plugin else None)
        if decision.backend == 'classic':
            return self._classic_reweight(line)
        logger.info(decision.describe())
        from madgraph.iolibs.template_files.mg7 import madtrex
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            card = pjoin(tmp, 'reweight_card.dat')
            with open(card_path) as fsock, open(card, 'w') as out:
                out.write(rb.strip_backend_line(fsock.read()))
            # in place: the weights go into the run's event file
            madtrex.run_reweighting(card, events_path=self._mg7_lhe_path,
                                    card_kind=decision.card_kind,
                                    process_directory=self.me_dir)
        tool_copy = self._tool_copy()
        if os.path.exists(tool_copy):
            sync_lhe_copy(self._mg7_lhe_path, tool_copy)

    def _tool_copy(self):
        return pjoin(os.path.dirname(self._mg7_lhe_path), 'unweighted_events.lhe.gz')

    def _classic_reweight(self, line):
        """The classic module, which writes its weights into the tools' copy
        of the events (unweighted_events.lhe.gz); the run's own event file
        gets them too, as with MadtRex."""
        out = super(MG7RunCmd, self).do_reweight(line)
        self.sync_reweighted_events()
        return out

    def sync_reweighted_events(self):
        """Copy the tools' copy of the events (unweighted_events.lhe[.gz])
        onto the run's event file when it is newer, i.e. when the classic
        reweighting module wrote into it. Safe to call again (e.g. after
        store_result(), which may gzip the copy)."""
        run_dir = os.path.dirname(self._mg7_lhe_path)
        copies = [pjoin(run_dir, name) for name in
                  ('unweighted_events.lhe.gz', 'unweighted_events.lhe')]
        copies = [path for path in copies if os.path.exists(path)]
        if not copies:
            return
        newest = max(copies, key=os.path.getmtime)
        if os.path.getmtime(newest) > os.path.getmtime(self._mg7_lhe_path):
            sync_lhe_copy(newest, self._mg7_lhe_path)

    def do_set(self, line, log=True):
        """Intercept the compiler options: the base handler patches
        ``Source/make_opts`` to switch Fortran/C++ compilers, which does not
        exist in the mg7 (C++/madspace) output. Everything else is delegated."""
        args = self.split_arg(line)
        if args and args[0] in ('fortran_compiler', 'cpp_compiler',
                                'f2py_compiler'):
            self.options[args[0]] = None if args[1] == 'None' else args[1]
            return
        return super(MG7RunCmd, self).do_set(line, log=log)
