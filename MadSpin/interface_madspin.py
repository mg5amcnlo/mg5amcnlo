################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
""" Command interface for MadSpin """
from __future__ import division
from __future__ import absolute_import
import collections
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import glob
from itertools import chain, filterfalse, product
pjoin = os.path.join
if '__main__' == __name__:
    import sys
    sys.path.append(pjoin(os.path.dirname(__file__), '..'))

import madgraph.interface.extended_cmd as extended_cmd
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.master_interface as master_interface
import madgraph.interface.madevent_interface as madevent_interface
import madgraph.interface.common_run_interface as common_run_interface
import madgraph.interface.reweight_interface as rwgt_interface
import madgraph.various.misc as misc
import madgraph.iolibs.files as files
import madgraph.iolibs.export_v4 as export_v4
import madgraph.various.banner as banner
import madgraph.various.lhe_parser as lhe_parser

import models.import_ufo as import_ufo
import models.check_param_card as check_param_card
import MadSpin.decay as madspin

logger = logging.getLogger('decay.stdout') # -> stdout
logger_stderr = logging.getLogger('decay.stderr') # ->stderr
cmd_logger = logging.getLogger('cmdprint2') # -> print

class MadSpinOptions(banner.ConfigFile):
    
    def default_setup(self):

        self.add_param("max_weight", -1)
        self.add_param('curr_dir', os.path.realpath(os.getcwd()))
        self.add_param('Nevents_for_max_weight', 0)
        self.add_param("max_weight_ps_point", 500)
        self.add_param('BW_cut', -1)
        self.add_param('nb_sigma', 0.)
        self.add_param('ms_dir', '')
        self.add_param('max_running_process', 100)
        self.add_param('onlyhelicity', False)
        self.add_param('spinmode', "madspin", allowed=['full', 'madspin', 'none', 'onshell', 'PA', 'madspin_v1', 'onshell_v1'])
        self.add_param('use_old_dir', False, comment='should be use only for faster debugging')
        self.add_param('run_card', '' , comment='define cut for decay_events (in onshell frame). Path to run_card to use')
        self.add_param('fixed_order', False, comment='to activate fixed order handling of counter-event')
        self.add_param('seed', 0, comment='control the seed of madspin')
        self.add_param('cross_section', {'__type__':0.}, comment="forcing normalization of cross-section after MS (for none/onshell)" )
        self.add_param('new_wgt', 'cross-section' ,allowed=['cross-section', 'BR'], comment="if not consistent number of particles, choose what to do for the weight. (BR: means local according to number of part, cross use the force cross-section")
        self.add_param('input_format', 'auto', allowed=['auto','lhe', 'hepmc', 'lhe_no_banner'])
        self.add_param('frame_id', 6)
        self.add_param('global_order_coupling', '')
        self.add_param('identical_particle_in_prod_and_decay', 'average')
        self.add_param('beampol', [0., 0.], comment='beam polarisation of each beam in percent, -100 .. 100, exactly as the run_card polbeam1/polbeam2 (0 is unpolarised). Taken from the run_card of the production when it has one.')
        self.add_param('density_debug', False, comment='Turn on check against full ME calculation')
        self.add_param('density_tolerance', 1E-4, comment='Tolerance for deviation between density and full ME')
        self.add_param('decay_event_mult', 1E0, comment='Produce more events than needed so that MadSpin does not have to regenerate decay events')
        self.add_param('nb_core', 0, comment='Number of cores for the MadSpin parallel unweighting (0 = use the global MG5 nb_core). nb_core>1 enables the process-parallel unweighting path.')
        self.add_param('density_keep_jacobian', True, comment='PA spinmode only: fold the offshell-reshuffling phase-space jacobian into the accept/reject weight (default) instead of applying the reshuffle as a post-acceptance kinematic dressing (False). Ignored by the madspin/full spinmodes, which always include that jacobian.')
        self.add_param('unweighting', 'auto',
                       allowed=['auto', 'joint', 'two_stage', 'sequential',
                                'sequential_global_retry',
                                'sequential_with_mass'],
                       comment="how the accept/reject is organised (density modes). "
                       "joint: one test over the virtualities and every decay at once, the historical scheme. "
                       "two_stage: unweight the set of virtualities first, then every decay against a single bound, redrawing only the decays on a rejection -- the production reshuffling and its density matrix are then evaluated once per accepted mass set instead of once per trial. "
                       "sequential: as two_stage but one test per decaying particle, redrawing only the particle that was rejected. "
                       "sequential_global_retry: as sequential, but a rejected decay redraws the virtualities too. "
                       "sequential_with_mass: one test per decaying particle with that particle's virtuality drawn *inside* its own accept/reject, so nothing is ever frozen and no stage has a conditional normalisation to divide out. Needs a per-particle mass draw, i.e. the PA spinmode; elsewhere it falls back to sequential. "
                       "two_stage, sequential and sequential_global_retry unweight the set of virtualities first; the first two then need a tabulated running-width factor, measured during the max-weight scan to ~0.5%, which is far inside the pole approximation these modes already assume; sequential_global_retry does without it at 2-3x the cost, and is meant as a cross-check rather than a default. "
                       "auto: sequential under PA/onshell, where it was the fastest scheme at every decay multiplicity measured; offshell joint up to two decaying particles and sequential from three, since offshell every mass set costs a production reshuffle and a production density and below three decays there are not enough of them to save to pay for it.")
        self.add_param('sequential_decay', 'auto',
                       comment='DEPRECATED, use unweighting: True maps to sequential, False to joint.')
        self.auto_set.add('sequential_decay')
        self.add_param('sequential_spin_order', '2 3 1', comment='spin order (MG5 2S+1 convention) deciding which particle is accept/rejected first in the sequential unweighting modes: default fermions, then vectors, then scalars (which can never be rejected).')
        self.add_param('sequential_debug', False, comment='the up-front-mass unweighting schemes (two_stage, sequential, sequential_global_retry): on every accepted chain, recompute the joint weight for the same production event, virtualities and decays and check that the product of the stage weights reproduces it (times the number of helicity states). Deterministic check of the decomposition itself -- the tabulated factor cancels out of it -- at roughly the cost of a joint trial per event. Debugging only.')

    ############################################################################
    ##  Special post-processing of the options                                ## 
    ############################################################################
    def post_set_ms_dir(self, value, change_userdefine, raiseerror, *opts):
        """ special handling for set ms_dir """
        
        self.__setitem__('curr_dir', value, change_userdefine=change_userdefine)
        
    ############################################################################
    def post_set_seed(self, value, change_userdefine, raiseerror):
        """ special handling for set seed """
        
        if not hasattr(random, 'mg_seedset'):
            random.seed(self['seed'])  
            random.mg_seedset = self['seed']  

    def post_set_beampol(self, value, change_userdefine, raiseerror, *opts):
        """Two values or none: one number is ambiguous (which beam?) and would
        otherwise only surface as an IndexError once the run reaches a matrix
        element, long after the card was read."""
        if value and len(value) != 2:
            raise banner.InvalidCmd(
                "beampol takes the polarisation of *both* beams, in percent: "
                "'set beampol [%s, 0]' for the first beam only. Got %s value(s)."
                % (value[0] if value else 0, len(value)))

    def beampol_me(self):
        """The beam polarisations in the convention the matrix elements use.

        The card and the run_card both speak percent (-100 .. 100, 0 for an
        unpolarised beam). ``/to_beampol/`` -- the v1 driver's msP/msF SMATRIX
        and GET_DENSITY -- is a verbatim copy of madevent's
        ``/to_polarization/`` reweighting, so it wants madevent's *internal*
        value, ``sign(1 + |polbeam|/100, polbeam)``: 1 unpolarised, +2 fully
        polarised along +1 helicity, -2 fully polarised along -1. Converting
        here rather than at parse time keeps the stored option equal to what the
        user typed, and keeps one convention in the cards and one in Fortran.
        """
        pol = self['beampol'] or [0., 0.]      # unset / [] is unpolarised
        out = []
        for value in (pol[0], pol[1]):
            value = float(value)
            out.append(1. if not value
                       else math.copysign(1 + abs(value) / 100., value))
        return tuple(out)

    def post_set_sequential_decay(self, value, change_userdefine, raiseerror, *opts):
        """Deprecated alias for 'unweighting'. True/False were the only values
        it ever had beyond 'auto', so they map onto the two modes that existed
        then."""
        if value in ('auto', None):
            mode = 'auto'
        elif value in (True, 'True', 'true', 1, '1'):
            mode = 'sequential'
        else:
            mode = 'joint'
        logger.warning("MadSpin: 'sequential_decay' is deprecated; "
                       "use 'set unweighting %s'", mode)
        self['unweighting'] = mode

    ############################################################################        
    def post_set_run_card(self, value, change_userdefine, raiseerror, *opts):
        """ special handling for set run_card """
        
        if value == 'default':
            self.run_card = None
        elif not value:
            self.run_card = None
        elif os.path.isfile(value):
            self.run_card = banner.RunCard(value)
        else:
            args = value.split()
            if  len(args) >1:
                if not hasattr(self, 'run_card'):
                    self.run_card =  banner.RunCardLO()
                    self.run_card.remove_all_cut()
                self.run_card[args[0]] = ' '.join(args[1:])
            else:
                raise Exception("wrong syntax for \"set run_card %s\"" % value)
            
        
    ############################################################################
    def post_fixed_order(self, value, change_userdefine, raiseerror):
        """ special handling for set fixed_order """
        
        if value:
            logger.warning('Fix order madspin fails to have the correct scale information. This can bias the results!')
            logger.warning('Not all functionalities of MadSpin handle this mode correctly (only onshell mode so far).')

    ############################################################################
    def post_identical_particle_in_prod_and_decay(self, value, change_userdefine, raiseerror):
        """ special handling for set fixed_order """
        if value not in ["crash", 'average', 'max', 'first']:
            raise Exception("value %s not supported for this parameter identical_in_prod_and_decay")

def _force_rmtree(path):
    """shutil.rmtree that also succeeds on read-only trees. Frozen concurrent
    gridpacks are chmod 555, so a plain rmtree raises PermissionError; make every
    directory/file writable first, then remove."""
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            try:
                os.chmod(root, 0o755)
            except OSError:
                pass
            for fname in files:
                try:
                    os.chmod(pjoin(root, fname), 0o644)
                except OSError:
                    pass
    shutil.rmtree(path, ignore_errors=True)


class _StridedEvents(object):
    """One parallel worker's disjoint, lock-free view of a shared decay-event
    file.

    Worker number ``offset`` (0 <= offset < stride) consumes the decay events
    at file positions ``offset, offset+stride, offset+2*stride, ...``; the
    events in between belong to the other workers, which each hold their own
    independent file handle over the same file. Because the stripes are
    disjoint, no decay event is ever consumed twice, so the statistics are
    unbiased and identical in distribution to the serial consumption.

    Only the attributes that :func:`MadSpinInterface.get_decay_from_file`
    actually reads are proxied (``cross`` for cross-section-weighted channel
    selection, ``name`` for reopening), so that hot function stays unchanged.
    On exhaustion this raises ``StopIteration`` exactly like an ``EventFile``,
    letting the caller trigger its (now shard-private) refill path.
    """

    def __init__(self, evtfile, offset, stride):
        self.f = evtfile
        self.stride = stride
        self._exhausted = False
        # advance to this worker's phase in the shared file
        for _ in range(offset):
            try:
                next(self.f)
            except StopIteration:
                self._exhausted = True
                break

    def __iter__(self):
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration
        ev = next(self.f)                     # this worker's event
        for _ in range(self.stride - 1):      # skip the other workers' events
            try:
                next(self.f)
            except StopIteration:
                self._exhausted = True
                break
        return ev
    next = __next__

    @property
    def cross(self):
        return self.f.cross

    @property
    def name(self):
        return self.f.name


class _ChainedEvents(object):
    """Reader over a decay pool that the unweighting wrote as several files
    (run_card ``nb_unweight_output``).

    The parent (max-weight estimation, serial unweighting) consumes it as a
    single stream. ``paths`` is kept public so that each parallel worker can
    instead open just *its* file: that is what removes the need to stride, i.e.
    to have every worker scan the whole pool to pick one event out of nb_core.
    Only the attributes get_decay_from_file reads are exposed (``cross`` for the
    cross-section weighted channel choice, ``name``).
    """

    def __init__(self, paths):
        self.paths = list(paths)
        self._idx = -1
        self._current = None
        self._first = lhe_parser.EventFile(self.paths[0])

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._current is None:
                self._idx += 1
                if self._idx >= len(self.paths):
                    raise StopIteration
                self._current = lhe_parser.EventFile(self.paths[self._idx])
            try:
                return next(self._current)
            except StopIteration:
                try:
                    self._current.close()
                except Exception:
                    pass
                self._current = None
    next = __next__

    @property
    def cross(self):
        return self._first.cross

    @property
    def name(self):
        return self.paths[0]


class _LimitedEvents(object):
    """Reader that yields at most ``limit`` events from ``evtfile`` then raises
    StopIteration, as if the file were that much shorter.

    Used to make a channel's *owner* worker (see MadSpinInterface._channel_owner)
    run its slice of the decay pool out ~10% before the other workers do. The
    owner is the only worker allowed to (re)generate that channel, so having it
    reach the refill point first means the pool is usually ready by the time the
    others need it -- they rarely have to block. Only the attributes
    get_decay_from_file reads (``cross``, ``name``) are proxied."""

    def __init__(self, evtfile, limit):
        self.f = evtfile
        self.limit = max(0, int(limit))
        self._n = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._n >= self.limit:
            raise StopIteration
        ev = next(self.f)
        self._n += 1
        return ev
    next = __next__

    @property
    def cross(self):
        return self.f.cross

    @property
    def name(self):
        return self.f.name


class MadSpinInterface(extended_cmd.Cmd):
    """Basic interface for madspin"""

    prompt = 'MadSpin>'
    debug_output = 'MS_debug'

    # Process-wide counter used to make each MadSpinInterface instance use
    # its own ``madspin_me`` output subdirectory. The actual fortran/f2py
    # artefacts (the ``.so`` extension and the ``liball_2me`` shared
    # library) are recompiled to that subdirectory each call, and
    # ``dlopen`` caches loaded libraries by absolute path: without a fresh
    # path the second MadSpin call in the same process (e.g. inline
    # MadSpin followed by ``decay_events``) keeps the *first* call's
    # matrix elements in memory and ``pdg2prefix`` ends up missing the
    # second card's decay channels (KeyError in ``get_pdir``).
    _ms_run_counter = 0


    @misc.mute_logger()
    def __init__(self, event_path=None, *completekey, **stdin):
        """initialize the interface with potentially an event_path"""

        cmd_logger.info('************************************************************')
        cmd_logger.info('*                                                          *')
        cmd_logger.info('*           W E L C O M E  to  M A D S P I N               *')
        cmd_logger.info('*                                                          *')
        cmd_logger.info('************************************************************')
        extended_cmd.Cmd.__init__(self, *completekey, **stdin)

        MadSpinInterface._ms_run_counter += 1
        self._ms_run_id = MadSpinInterface._ms_run_counter
        # First call keeps the historical ``madspin_me`` name (the
        # decay/import code still hard-codes that in a few places and the
        # vast majority of MadSpin uses only sees one run per process).
        # Subsequent calls use a unique suffix so dlopen sees a fresh
        # file path.
        if self._ms_run_id == 1:
            self.ms_me_subdir = 'madspin_me'
            self.ms_me_decay_subdir = 'madspin_decay'
        else:
            self.ms_me_subdir = 'madspin_me_%d' % self._ms_run_id
            self.ms_me_decay_subdir = 'madspin_decay_%d' % self._ms_run_id

        self.decay = madspin.decay_misc()
        self.model = None
        #self.mode = "madspin" # can be flat/bridge change the way the decay is done.
        #                      # note amc@nlo does not support bridge.
        
        self.options = MadSpinOptions()
        
        self.events_file = None
        self.decay_processes = {}
        self.list_branches = {}
        # the resolved '@' decay grouping, or None when the card declares none
        # (or declares one this run cannot honour). See _resolve_decay_groups.
        self._decay_groups = None
        self.to_decay={}
        self.mg5cmd = master_interface.MasterCmd()
        self.seed = None
        self.err_branching_ratio = 0
        self.me_run_name = "" # Events diretory name where to stotre the events (used by madevent) not use internally
        self.all_iden = {}
        
        if event_path:
            logger.info("Extracting the banner ...")
            self.do_import(event_path)
            
    
    def setup_for_pure_decay(self):
        """this is for spinmode=none -> simple decay
           We go here if they are no banner.
           -> this requires that a command import model appears in the card!
        """

        logger.info("Setup the code for pure decay mode")
        self.proc_option = []
        self.final_state_full = ''
        self.final_state_compact = ''
        self.prod_branches = ''
        self.final_state = set()

    def _load_f2py_matrix_module(self, sp_path, menum=2):
        """Load the freshly-compiled ``all_matrix<menum>py`` extension under
        ``sp_path``.

        Each MadSpin run compiles its matrix elements into its own
        ``madspin_me_<N>`` subdir, and (from the second call onwards)
        ``decay.compile()`` overrides the makefile's ``PROCNAME`` so the
        resulting Fortran shared library
        (``liball<PROCNAME>_<MENUM>me.{so,dylib}``) has a unique SONAME /
        install_name. The combination of a unique wrapper path *and* a
        unique dependent-library identity is what stops the dynamic
        loader from returning the first call's already-loaded matrix
        elements on the second call.

        Within a single run the production (madspin_me) and decay
        (madspin_decay) modules are both loaded into this process; they are
        built with distinct ``MENUM`` values (2 for production, 1 for decay)
        so the f2py module name (``all_matrix<MENUM>py``) and the dependent
        library (``liball_<MENUM>me``) differ — otherwise the two identically
        named Fortran extensions clash and segfault. ``menum`` selects which
        one to load.

        This helper just picks the loadable ``.so`` and loads it via
        ``importlib.util.spec_from_file_location`` to bypass the
        ``sys.modules`` cache (which would otherwise short-circuit
        ``__import__`` to the first call's module object).
        """
        import importlib.util
        import glob

        modname = 'all_matrix%dpy' % menum
        # The actual loadable file is the cpython-tagged ``.so``; on some
        # builds the unsuffixed ``all_matrix<menum>py.so`` is a 0-byte stub.
        # Pick the largest matching file so we always load real code.
        patterns = [
            '%s.cpython*.so' % modname,
            '%s.cpython*.dylib' % modname,
            '%s.so' % modname,
            '%s.dylib' % modname,
        ]
        candidates = []
        for pat in patterns:
            for hit in glob.glob(pjoin(sp_path, pat)):
                if os.path.getsize(hit) > 0:
                    candidates.append(hit)
        if not candidates:
            # Fall back to the historical ``__import__`` so we at least
            # produce a meaningful error if nothing got compiled.
            return __import__(modname)
        candidates.sort(key=os.path.getsize, reverse=True)
        so_path = candidates[0]

        # Load via spec_from_file_location to bypass the sys.modules cache
        # while keeping the module name as ``all_matrix<menum>py`` (the .so's
        # PyInit_all_matrix<menum>py init symbol is baked in at compile time).
        spec = importlib.util.spec_from_file_location(modname, so_path)
        if spec is None or spec.loader is None:
            return __import__(modname)
        mymod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mymod)
        return mymod

    def _log_lhe_timers(self):
        if not getattr(lhe_parser, "_ENABLE_LHE_TIMERS", False):
            return
        timers, counts = lhe_parser.get_lhe_timers()
        if not timers:
            print("LHE parser timing enabled but no samples were collected.")
            return
        print("LHE parser timing summary:")
        for key in sorted(timers):
            total = timers[key]
            count = counts.get(key, 1)
            print("  %s: %.6fs total over %d call(s) (avg %.6fs)" %
                  (key, total, count, total / max(1, count)))
        
     
    def do_import(self, inputfile):
        """import the event file"""
        
        args = self.split_arg(inputfile)
        if not args:
            return self.InvalidCmd, 'import requires arguments'
        elif args[0] == 'model':
            return self.import_model(args[1:])
        
        # change directory where to write the output
        self.options['curr_dir'] = os.path.realpath(os.path.dirname(inputfile))
        if os.path.basename(os.path.dirname(os.path.dirname(inputfile))) == 'Events':
            self.options['curr_dir'] = pjoin(self.options['curr_dir'], 
                                                      os.path.pardir, os.pardir)
        
        if not os.path.exists(inputfile):
            if inputfile.endswith('.gz'):
                if not os.path.exists(inputfile[:-3]):
                    misc.sprint(os.getcwd(), os.listdir('.'), inputfile, os.path.exists(inputfile), os.path.exists(inputfile[:-3]))
                    raise self.InvalidCmd('No such file or directory : %s' % inputfile)
                else: 
                    inputfile = inputfile[:-3]
            elif os.path.exists(inputfile + '.gz'):
                inputfile = inputfile + '.gz'
            else: 
                raise self.InvalidCmd('No such file or directory : %s' % inputfile)

        self.inputfile = inputfile
        if self.options['spinmode'] == 'none' and \
           (self.options['input_format'] not in ['lhe','auto'] or 
             (self.options['input_format'] == 'auto' and '.lhe'  not in inputfile[-7:])):  
            self.banner = banner.Banner()
            self.setup_for_pure_decay()
            return  
        
        if inputfile.endswith('.gz'):
            misc.gunzip(inputfile)
            inputfile = inputfile[:-3]
        # Read the banner of the inputfile
        self.events_file = open(os.path.realpath(inputfile))
        self.banner = banner.Banner(self.events_file)


        # Check the validity of the banner:
        if 'slha' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain model information')
        elif 'mg5proccard' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain generation information')

        
        if 'madspin' in self.banner:
            raise self.InvalidCmd('This event file was already decayed by MS. This is not possible to add to it a second decay')
        
        if 'mgruncard' in self.banner:
            run_card = self.banner.charge_card('run_card')
            if not self.options['Nevents_for_max_weight']:
                nevents = run_card['nevents']
                N_weight = max([75, int(3*nevents**(1/3))])
                self.options['Nevents_for_max_weight'] = N_weight
                N_sigma = max(4.5, math.log(nevents,7.7))
                self.options['nb_sigma'] = N_sigma
            if self.options['BW_cut'] == -1:
                self.options['BW_cut'] = float(self.banner.get_detail('run_card', 'bwcutoff'))
                if self.options['BW_cut'] > 25:
                    logger.critical("value of bwcutoff set to %s from the input file. This is much too large value for Madspin and the validity of the Narrow-width-Approximation. Please ensure that you overwrite that value via \"set BW_cut X\"  to a smaller value (like X=10)", self.options['BW_cut'])
            
            if isinstance(run_card, banner.RunCardLO):
                run_card.update_system_parameter_for_include()
                # The run_card of the production is the default source for both,
                # but an explicit "set frame_id"/"set beampol" in the MadSpin
                # card wins -- otherwise neither option could be set from the
                # card the rest of the MadSpin options live in.
                if 'frame_id' not in self.options.user_set:
                    self.options['frame_id'] = run_card['frame_id']
                if 'beampol' not in self.options.user_set:
                    self.options['beampol'] = [run_card['polbeam1'],
                                               run_card['polbeam2']]
            else:
                if 'frame_id' not in self.options.user_set:
                    self.options['frame_id'] = 6
                if 'beampol' not in self.options.user_set:
                    self.options['beampol'] = [0., 0.]

        else:
            if not self.options['Nevents_for_max_weight']:
                self.options['Nevents_for_max_weight'] = 75
                self.options['nb_sigma'] = 4.5
            if self.options['BW_cut'] == -1:
                self.options['BW_cut'] = 15.0
                
                
        # load information
        process = self.banner.get_detail('proc_card', 'generate')
        if not process:
            msg = 'Invalid proc_card information in the file (no generate line):\n %s' % self.banner['mg5proccard']
            raise Exception(msg)
        process, option = mg_interface.MadGraphCmd.split_process_line(process)
        self.proc_option = option
        
        logger.info("process: %s" % process)
        logger.info("options: %s" % option)

        if not hasattr(self,'multiparticles_ms'):
            for key, value in self.banner.get_detail('proc_card','multiparticles'):
                try:
                    self.do_define('%s = %s' % (key, value))
                except self.mg5cmd.InvalidCmd:  
                    pass
                
        # Read the final state of the production process:
        #     "_full" means with the complete decay chain syntax 
        #     "_compact" means without the decay chain syntax 
        self.final_state_full = process[process.find(">")+1:]
        self.final_state_compact, self.prod_branches=\
                 self.decay.get_final_state_compact(self.final_state_full)
                
        # Load the model
        complex_mass = False   
        has_cms = re.compile(r'''set\s+complex_mass_scheme\s*(True|T|1|true|$|;)''')
        for line in self.banner.proc_card:
            if line.startswith('set'):
                self.mg5cmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                if has_cms.search(line):
                    complex_mass = True
        
          
        info = self.banner.get('proc_card', 'full_model_line')
        if '-modelname' in info:
            mg_names = False
        else:
            mg_names = True
        model_name = self.banner.get('proc_card', 'model')
        if model_name:
            model_name = os.path.expanduser(model_name)
            self.load_model(model_name, mg_names, complex_mass)
        else:
            raise self.InvalidCmd('Only UFO model can be loaded in MadSpin.')
        # check particle which can be decayed:
        self.final_state = set()
        final_model = False
        for line in self.banner.proc_card:
            line = ' '.join(line.strip().split())
            if line.startswith('generate'):
                self.final_state.update(self.mg5cmd.get_final_part(line[8:]))
            elif line.startswith('add process'):
                self.final_state.update(self.mg5cmd.get_final_part(line[11:]))
            elif line.startswith('define'):
                try:
                    self.mg5cmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                except self.mg5cmd.InvalidCmd:
                    if final_model:
                        raise
                    else:
                        key = line.split()[1]
                        if key in self.multiparticles_ms:
                            del self.multiparticles_ms[key]            
            elif line.startswith('set') and not line.startswith('set gauge'):
                self.mg5cmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
            elif line.startswith('import model'):
                if model_name in line:
                    final_model = True
                    
                
    def import_model(self, args):
        """syntax: import model NAME CARD_PATH
            args didn't include import model"""
        
        bypass_check = False
        if '--bypass_check' in args:
            args.remove('--bypass_check')
            bypass_check = True
        if len(args) == 1:  
            logger.warning("""No param_card defined for the new model. We will use the default one but this might completely wrong.""")
        elif len(args) != 2:
            return self.InvalidCmd, 'import model requires two arguments'
        
        model_name = args[0]
        self.load_model(model_name, False, False)
        
        if len(args) == 2:
            card = args[1]
            if not os.path.exists(card):
                raise self.InvalidCmd('%s: no such file' % card)
        else:
            card = "madspin_param_card.dat"
            export_v4.UFO_model_to_mg4.create_param_card_static(self.model,
                                card, rule_card_path=None)

        

        #Check the param_card
        if not (bypass_check or self.options['input_format'] in ['hepmc', 'lhe_no_banner']):
            if not hasattr(self.banner, 'param_card'):
                self.banner.charge_card('slha')
            param_card = check_param_card.ParamCard(card)
            # checking that all parameter of the old param card are present in 
            #the new one with the same value
            try:
                diff = self.banner.param_card.create_diff(param_card)
            except Exception:
                raise self.InvalidCmd('''The two param_card seems very different. 
    So we prefer not to proceed. If you are sure about what you are doing, 
    you can use the command \'import model MODELNAME PARAM_CARD_PATH --bypass_check\'''')
            if diff:
                raise self.InvalidCmd('''Original param_card differs on some parameters:
    %s
    Due to those differences, we prefer not to proceed. If you are sure about what you are doing, 
    you can use the command \'import model MODELNAME PARAM_CARD_PATH --bypass_check\''''
    % diff.replace('\n','\n    '))
   
   
                
        #OK load the new param_card (but back up the old one)
        if 'slha' in self.banner:
            self.banner['slha_original'] = self.banner['slha']
        self.banner['slha'] = open(card).read()
        if hasattr(self.banner, 'param_card'):
            del self.banner.param_card
        self.banner.charge_card('slha')
                


    @extended_cmd.debug()
    def complete_import(self, text, line, begidx, endidx):
        "Complete the import command"
        
        args=self.split_arg(line[0:begidx])
        
        
        if len(args) == 1:
            base_dir = '.'
        else:
            base_dir = args[1]
        
        return self.path_completion(text, base_dir)
        
        # Directory continuation
        if os.path.sep in args[-1] + text:
            return self.path_completion(text,
                                    pjoin(*[a for a in args if \
                                                      a.endswith(os.path.sep)]))

    def do_decay(self, decaybranch):
        """add a process in the list of decayed particles"""
        
        #if self.model and not self.model['case_sensitive']:
        #    decaybranch = decaybranch.lower()

        if self.options['spinmode'] not in  ['full','madspin', 'madspin_v1'] and '{' in decaybranch:
            if self.options['spinmode'] == 'none':
                logger.warning("polarization option used with spinmode=none. The polarization definition will be done according to the rest-frame of the decaying particles (which is likely not what you expect).")
            else:
                logger.warning("polarization option used with spinmode=%s. This combination is not validated and is by construction using sub-optimal method which can likely lead to bias in some situation. Use at your own risk." % self.options['spinmode'])
        if "=" in decaybranch and self.options['spinmode'] in ['full', 'madspin_v1']:
            logger.warning("Note that coupling order restriction are not associated to specific Branching Ratio. The total cross-section might therefore use the wrong branching ratio.")
        decay_process, init_part = self.decay.reorder_branch(decaybranch)
        if init_part not in self.list_branches:
            self.list_branches[init_part] = []
        self.list_branches[init_part].append(decay_process)
        del decay_process, init_part    
        
    
    def check_set(self, args):
        """checking the validity of the set command"""
        
        if len(args) < 2:
            if args and '=' in args[0]:
                name, value = args[0].split('=')
                args[0]= name
                args.append(value)
            elif len(args) == 1 and args[0] in ['onlyhelicity']:
                args.append('True')
            else:
                raise self.InvalidCmd('set command requires at least two argument.')
        
        if args[1].strip() == '=':
            args.pop(1)
        
        valid = ['max_weight','seed','curr_dir', 'spinmode', 'run_card']
        if args[0] not in self.options and args[0] not in valid:
            raise self.InvalidCmd('Unknown options %s' % args[0])        
    
        if args[0] == 'max_weight':
            try:
                args[1] = float(args[1].replace('d','e'))
            except ValueError:
                raise self.InvalidCmd('second argument should be a real number.')
        
        elif args[0] == 'curr_dir':
            if not os.path.isdir(args[1]):
                raise self.InvalidCmd('second argument should be a path to a existing directory')
        elif args[0] == "spinmode":
            allowed = [mode.lower() for mode in self.options.allowed_value['spinmode']]
            if args[1].lower() not in allowed:
                raise self.InvalidCmd("spinmode can only take one of those values: full/madspin/onshell/none/PA/madspin_v1/onshell_v1")

        elif args[0] == "run_card":
            if self.options['spinmode'] in ["madspin_v1"]:
                raise self.InvalidCmd("edition of the run_card is not allowed within spinmode=madspin_v1")
            if "=" in args:
                args.remove("=")
            if len(args)==2 and "=" in args[1]:
                data = args.pop(1)
                arg, value = data.split("=")
                args.append(arg)
                args.append(value)
        elif args[0] == 'Nevents_for_max_weigth':
            args[0] = 'Nevents_for_max_weight'
        
    def do_set(self, line):
        """ add one of the options """
        
        args = self.split_arg(line)
        self.check_set(args)

        self.options[args[0]] = ' '.join(args[1:])
        # ConfigFile only fills user_set through its own set(); record it here
        # so options that are otherwise taken from the production run_card
        # (frame_id, beampol) can still be overridden from the MadSpin card.
        self.options.user_set.add(args[0].strip().lower())
        

    def complete_set(self,  text, line, begidx, endidx):
        

        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) == 1:
            opts = list(self.options.keys()) + ['seed']
            return self.list_completion(text, opts) 
        elif len(args) == 2:
            if args[1] == 'ms_dir':
                return self.path_completion(text, '.', only_dirs = True)
        elif args[1] == 'ms_dir':
            curr_path = pjoin(*[a for a in args \
                                                   if a.endswith(os.path.sep)])
            return self.path_completion(text, curr_path, only_dirs = True)
        elif args[1] == "spinmode":
            return self.list_completion(text, ["full", "madspin", "none", "onshell", "PA", "madspin_v1", "onshell_v1"], line)
         
    def help_set(self):
        """help the set command"""
        
        print('syntax: set OPTION VALUE')
        print('')
        print('-- assign to a given option a given value')
        print('   - set max_weight VALUE: pre-define the maximum_weight for the reweighting')
        print('   - set seed VALUE: fix the value of the seed to a given value.')
        print('       by default use the current time to set the seed. random number are')
        print('       generated by the python module random using the Mersenne Twister generator.')
        print('       It has a period of 2**19937-1.')
        print('   - set max_running_process VALUE: allow to limit the number of open file used by the code')
        print('       The number of running is raising like 2*VALUE')
        print('   - set spinmode=XXX: different approximation/implementation are available')
        print('        none: no spin correlation no offshell effects.')
        print('        onshell: spin correlation but no offshell effects.')
        print('        PA: spin correlation and offshell effects with a pure Breit-Wigner.')
        print('        madspin: spin correlation and offshell effects with offshell matrix-elements.')
        print('        full: resolves to madspin.')
        print('        madspin_v1: legacy MadSpin implementation (different reshuffling and no density matrix)')
        print('        onshell_v1: legacy MadSpin implementation (no density matrix) --for debugging only--')
    
    def do_define(self, line):
        """ """

        try:
            self.mg5cmd.exec_cmd('define %s' % line)
        except:
            #cleaning if the error is recover later
            key = line.split()[0]
            if hasattr(self, 'multiparticles_ms') and key in self.multiparticles_ms:
                del self.multiparticles_ms[key]
            raise
           
        self.multiparticles_ms = dict([(k,list(pdgs)) for k, pdgs in \
                                        self.mg5cmd._multiparticles.items()])
    
    
    def update_status(self, *args, **opts):
        """ """
        pass # function overwritten for MS launched by ME
    
    def complete_define(self, *args):
        """ """
        try:
            return self.mg5cmd.complete_define(*args)
        except Exception as error:
            misc.sprint(error)
            
    def complete_decay(self, *args):
        """ """
        try:
            return self.mg5cmd.complete_generate(*args)
        except Exception as error:
            misc.sprint(error)
            
    def check_launch(self, args):
        """check the validity of the launch command"""
        
        if not self.list_branches and not self.options['onlyhelicity']:
            raise self.InvalidCmd("Nothing to decay ... Please specify some decay")
        if not self.events_file:
            raise self.InvalidCmd("No events files defined.")
        
        # Validity check. Need lhe version 3 if matching is on
        if self.banner.get("run_card", "lhe_version") < 3 and \
            self.banner.get("run_card", "ickkw") > 0:
            raise Exception("MadSpin requires LHEF version 3 when running with matching/merging")

    def help_launch(self):
        """help for the launch command"""
        
        print('''Running Madspin on the loaded events, following the decays enter
        An example of a full run is the following:
        import ../mssm_events.lhe.gz
        define sq = ur ur~
        decay go > sq j
        launch
        ''')
        
        self.parser_launch.print_help()

    def parser_launch(self):
        usage = """launch [-n RUN_NAME]   
        """
        parser = misc.OptionParser(usage=usage)
        parser.add_option("-n", "--name",
                  default="",
                  help="When NOT run in standalone instruct MG5aMC where to store the events file")
        return parser
    
    def parse_launch(self, line):
        
        args = self.split_arg(line)
        return self.parser_launch().parse_args(args)
        

    # ``decay t > w+ b, w+ > l+ vl @1``: the @ suffix sorts the decay lines into
    # *groups* meant to be used together -- the semi-leptonic ttbar idiom, where
    # only the two charge assignments exist and no fully leptonic or fully
    # hadronic event is produced.
    #
    # The tag is kept inside the branch string rather than in a structure of its
    # own: ``list_branches`` is renamed, pruned and handed to MG5 from several
    # places, and index i of ``list_branches[name]`` is also the number of the
    # ``decay_<pdg>_<i>`` pool, so a parallel list would be one more thing to
    # keep in step. It is split off at the few points that need it.
    _DECAY_GROUP_TAG = re.compile(r'@\s*(\d+)\s*$')

    @classmethod
    def _split_group_tag(cls, branch):
        """``'t > w+ b, w+ > l+ vl @1'`` -> ``('t > w+ b, w+ > l+ vl', '1')``.

        The tag is returned as a string: it names a group, it is not an integer
        the code ever does arithmetic on. ``(branch, None)`` when untagged.
        """
        found = cls._DECAY_GROUP_TAG.search(branch)
        if not found:
            return branch, None
        return branch[:found.start()].rstrip(), found.group(1)

    def _decay_group_layout(self):
        """Sort the decay lines into groups.

        Returns ``(layout, reason)``. ``layout`` is None when the card declares
        no group at all (``reason`` None too) or when the tags cannot be honoured
        (``reason`` says why, for the warning). Otherwise::

            {'tags':  ['1', '2'],                       # first-appearance order
             'lines': {particle_name: {tag: [index into list_branches[name]]}}}

        An *untagged* line belongs to every group -- the natural way to write a
        third particle that decays the same way whatever the group is, and what
        madspin_v1 does (it prepends the untagged branches to each group).

        This is the half of the check that needs only the card. Whether each
        group covers every decaying particle the right number of times needs the
        production events too; see :meth:`_validate_decay_groups`.
        """
        tags = []
        parsed = {}
        for name, branches in self.list_branches.items():
            parsed[name] = []
            for branch in branches:
                stripped, tag = self._split_group_tag(branch)
                if '@' in stripped:
                    # an '@' that is not a trailing group tag: refuse rather than
                    # half-read it, since MG5 would take it as a process number
                    return None, ("the decay line %r carries an '@' that is not "
                                  "a group tag at the end of the line" % branch)
                parsed[name].append(tag)
                if tag is not None and tag not in tags:
                    tags.append(tag)
        if not tags:
            return None, None
        lines = {}
        for name, name_tags in parsed.items():
            lines[name] = dict((tag, []) for tag in tags)
            for i, tag in enumerate(name_tags):
                for target in (tags if tag is None else [tag]):
                    lines[name][target].append(i)
        return {'tags': tags, 'lines': lines}, None

    def _validate_decay_groups(self, layout, to_decay, nb_event, name2pdg):
        """Can this grouping be honoured for these production events?

        Supported shape -- rectangular: every group gives exactly ``n_part``
        lines for every decaying particle, ``n_part`` being how many of that
        particle each event carries. ``n_part == 1`` is the semi-leptonic ttbar
        idiom; ``n_part == 2`` is ``p p > t t~ t~ t``, where the group's two
        lines for a pdg are handed to its two particles by the existing
        positional rule.

        Anything else is refused rather than approximated: the group would not
        be a complete assignment, so neither its rate ``prod_k Gamma_k`` nor the
        branching ratio is defined.

        ``name2pdg`` maps a decay-line parent name to its pdg, or to None when
        the name is a multiparticle (refused: one name would own several pools).

        Returns ``(ok, reason)``.
        """
        for name, per_tag in layout['lines'].items():
            pdg = name2pdg(name)
            if pdg is None:
                return False, ("%s is a multiparticle: grouping needs one "
                               "decaying particle per decay line" % name)
            if pdg not in to_decay or not nb_event:
                continue            # never appears in the events; ignored anyway
            if to_decay[pdg] % nb_event:
                return False, ("the production events do not all carry the same "
                               "number of %s, and a branching ratio per group "
                               "cannot be defined then" % name)
            nb_part = to_decay[pdg] // nb_event
            for tag in layout['tags']:
                got = len(per_tag.get(tag, ()))
                if got != nb_part:
                    return False, (
                        "group @%s gives %d decay line(s) for %s but every event "
                        "carries %d of them -- each group must decay every "
                        "particle exactly once" % (tag, got, name, nb_part))
        return True, None

    def _warn_ignored_decay_groups(self, spinmode, reason=None):
        """Warn when the card carries @ grouping tags this run cannot honour.

        A tag that silently means nothing is the worst outcome: the
        decay-matrix-element generation appends an ``@`` process number of its
        own to every branch, so MG5 sees two of them, binds its own at the top
        level and absorbs the user's as the process number of the sub-decay.
        Nothing fails, the matrix element that comes out is the correct
        *ungrouped* one, and without this the card simply does not mean what it
        looks like it means.

        Returns the (particle, branch, tag) triples found, for the tests.
        """
        if spinmode == 'madspin_v1':
            return []                     # v1 implements the grouping itself
        tags = []
        for name, branches in self.list_branches.items():
            for branch in branches:
                stripped, tag = self._split_group_tag(branch)
                if tag is not None:
                    tags.append((name, branch, '@%s' % tag))
                elif '@' in stripped:
                    # not a trailing tag, so not a group -- but MG5 will still
                    # read it as a process number, so it is worth the same line
                    tags.append((name, branch, '@?'))
        if not tags:
            return []
        logger.warning(
            "The decay lines carry '@' grouping tags (%s) but this run cannot "
            "honour them: %s. The tags therefore change nothing (MG5 reads them "
            "as an ordinary process number) and the sample will contain EVERY "
            "combination of the channels listed, not only the tagged ones. "
            "Generate one group per MadSpin run and merge the outputs -- fixing "
            "the normalisation with 'set cross_section' if the automatic "
            "branching ratio is not what you want.",
            ', '.join(sorted(set(t[2] for t in tags))),
            reason or ("grouping is available in the density spin modes (PA, "
                       "onshell, madspin/full) and in madspin_v1, and "
                       "spinmode=%s is neither" % spinmode))
        return tags

    # ------------------------------------------------------------------
    # Resolved grouping: what the run-time draw and the branching ratio use
    # ------------------------------------------------------------------
    def _decay_group_pdgs(self):
        """The pdgs whose decay lines are grouped (empty when they are not)."""
        groups = getattr(self, '_decay_groups', None)
        return () if not groups else groups['lines']

    def _resolve_decay_groups(self, to_decay, nb_event, density_method):
        """Turn the card's '@' tags into the grouping the run will use, or warn
        and fall back to the ungrouped behaviour.

        Sets (and returns) ``self._decay_groups``::

            {'tags':  ['1', '2'],
             'lines': {pdg: {tag: [decay pool number]}},
             'prob':  None}                 # filled by _resolve_group_rates

        None when the card declares no group, or when this run cannot honour the
        one it declares. Must run before anything is generated: it decides how
        many pools each particle gets, and it forces the joint accept/reject.
        """
        self._decay_groups = None
        layout, reason = self._decay_group_layout()
        if layout is None and reason is None:
            return None                              # no tags at all

        spinmode = self.options['spinmode']
        if not reason and not density_method:
            reason = ("spinmode=%s does not fill one decay pool per channel, "
                      "which is what a group draws from" % spinmode)
        if not reason and self.options['fixed_order']:
            reason = ("fixed_order rides the counter-events along with the "
                      "decays, and a group would have to be drawn once per "
                      "event group")
        if not reason:
            ok, reason = self._validate_decay_groups(
                layout, to_decay, nb_event, self._decay_parent_pdg)
        if reason:
            self._warn_ignored_decay_groups(spinmode, reason)
            return None

        lines = {}
        for name, per_tag in layout['lines'].items():
            pdg = self._decay_parent_pdg(name)
            if pdg in to_decay:          # a species that never appears in the
                lines[pdg] = per_tag     # events is dropped anyway
        if not lines:
            return None
        self._decay_groups = {'tags': layout['tags'], 'lines': lines,
                              'prob': None}
        logger.info("MadSpin: %s decay groups (@%s); each event draws one group "
                    "and every particle takes that group's channel",
                    len(layout['tags']), ', @'.join(layout['tags']))
        return self._decay_groups

    def _decay_parent_pdg(self, name):
        """pdg of a decay line's parent, or None when the name stands for more
        than one particle (a multiparticle owning several pools at once is what
        the grouping cannot express)."""
        if name in self.mg5cmd._multiparticles:
            pdgs = self.mg5cmd._multiparticles[name]
            return pdgs[0] if len(pdgs) == 1 else None
        try:
            return self.mg5cmd._curr_model.get('name2pdg')[name]
        except KeyError:
            return None

    @staticmethod
    def _clamped_partial_width(pwidth, totwidth, pdg=None):
        """A measured partial width, reconciled with the total width of the
        param_card. Only ever applied to *one* channel's width (or to a sum over
        channels, which is still a width): a product of several is not
        comparable with a single total.

        The two regimes are deliberate, and are the long-standing behaviour this
        helper factors out of run_onshell rather than a new policy:

        - up to 1% above the total, the excess is Monte Carlo noise in the width
          measurement, so it is capped silently and the branching ratio stays at
          most 1;
        - more than 1% above, the param_card's total genuinely disagrees with
          what was generated. That is warned about and the *measured* value is
          used, because capping there would quietly reshape the normalisation to
          match a card that is wrong, and hide the disagreement instead of
          reporting it.
        """
        if pwidth > 1.01 * totwidth:
            logger.warning('partial width (%s) larger than total width (%s) '
                           '--from param_card-- for pdg %s',
                           pwidth, totwidth, pdg)
        elif pwidth > totwidth:
            return totwidth
        return pwidth

    @classmethod
    def _assignment_multiplicity(cls, branches):
        """How many *distinct* ways this multiset of decay lines can be dealt to
        that many identical parents: ``n! / prod_c m_c!``.

        The positional rule generates one of those assignments and multiplies
        the rate by their number. A plain ``n!`` is right only while the lines
        are all different -- with two identical ones it counts the same final
        state twice.
        """
        counts = collections.Counter(cls._split_group_tag(b)[0].strip()
                                     for b in branches)
        out = math.factorial(len(branches))
        for repeat in counts.values():
            out //= math.factorial(repeat)
        return out

    def _resolve_group_rates(self, gen_jobs, channel_widths):
        """Branching ratio of the grouped particles, and each group's share.

        A group is one complete assignment of channels to particles, so its rate
        is a product over the particles::

            br_g = prod_pdg  mult(g,pdg) * prod_{i in g} Gamma_i / Gamma_tot^n

        with ``mult`` from :meth:`_assignment_multiplicity`. The groups are
        alternatives, so ``br = sum_g br_g`` -- and the group to use is drawn
        with ``p_g = br_g / br``, which is exactly the unpolarised factorisation
        of its rate. Everything the polarisation adds on top is then carried by
        the accept/reject weight, the same way the per-channel cross-section
        draw already works for the ungrouped multi-channel case.
        """
        rates = []
        for tag in self._decay_groups['tags']:
            rate = 1.0
            for pdg, per_tag in self._decay_groups['lines'].items():
                if pdg not in gen_jobs:
                    continue
                indices = per_tag[tag]
                widths = channel_widths.get(pdg) or {}
                branches = self.list_branches[
                    self.model.get_particle(pdg).get_name()]
                for i in indices:
                    rate *= self._clamped_partial_width(
                        widths.get(i, 0.0), gen_jobs[pdg]['totwidth'], pdg)
                rate *= self._assignment_multiplicity(
                    [branches[i] for i in indices])
                rate /= gen_jobs[pdg]['totwidth'] ** len(indices)
            rates.append(rate)
        total = sum(rates)
        if total <= 0:
            raise Exception("MadSpin: every decay group has a vanishing rate; "
                            "check the decay lines and the param_card widths.")
        self._decay_groups['prob'] = [r / total for r in rates]
        logger.info("MadSpin: decay group rates %s (BR %.6g)",
                    ', '.join('@%s=%.4f' % (tag, p) for tag, p in
                              zip(self._decay_groups['tags'],
                                  self._decay_groups['prob'])), total)
        return total

    def _draw_decay_group(self):
        """The group this chain uses, or None when the card declares none.

        Drawn afresh on every trial -- the caller is the top of the draw, which
        the joint accept/reject re-enters on each rejection. That is what keeps
        the group part of what is being unweighted, rather than something a
        later stage could normalise away.
        """
        groups = getattr(self, '_decay_groups', None)
        if not groups:
            return None
        r = random.random()
        cumul = 0.0
        for tag, prob in zip(groups['tags'], groups['prob']):
            cumul += prob
            if r < cumul:
                return tag
        return groups['tags'][-1]

    @misc.mute_logger()
    def do_launch(self, line):
        """end of the configuration launched the code"""

        (options, args) = self.parse_launch(line)
        if getattr(lhe_parser, "_ENABLE_LHE_TIMERS", False):
            lhe_parser.reset_lhe_timers()
        
        if options.name:
            self.me_run_name = options.name # Only use by MG5aMC
        else:
            self.me_run_name = ''

        try:
            if 'noborn' in self.banner.get_detail('proc_card', 'generate'):
                process_LI = True
            else:
                process_LI = False
        except: #this exception is added because the test 'test_hepmc_decay' does not present a proc_card. Maybe there is a way to have this information under this format ?
            logger.warning("The proc_card has not been found. It is unknown whether the process is at tree-level or loop-induced")
            logger.warning("The process is now considered as tree-level")
            process_LI = False

        # the legacy modes 'madspin_v1' and 'onshell_v1' are not compatible with loop-induced processes
        if self.options['spinmode'] in ['madspin_v1', 'onshell_v1'] and process_LI:
            raise ValueError("The MadSpin modes 'madspin_v1' and 'onshell_v1' are are not compatible with loop-induced processes. Please choose a mode among 'none', 'PA', 'madspin' or 'onshell'.")

        if self.options['onlyhelicity']:
            self.options['spinmode'] = 'madspin_v1'

        spinmode = self.options['spinmode']
        if spinmode == 'full':
            spinmode = 'madspin'
            self.options['spinmode'] = spinmode

        logger.info("Running MadSpin in spinmode %s" % spinmode)
        # The density modes decide about the '@' grouping later, in run_onshell,
        # where the production events say how many of each particle an event
        # carries. These two never can, so say it now rather than after the
        # generation.
        if spinmode in ('none', 'onshell_v1'):
            self._warn_ignored_decay_groups(spinmode)

        if spinmode in ["none"]:
            out = self.run_bridge(line)
            self._log_lhe_timers()
            return out
        elif spinmode.startswith("onshell"):
            if spinmode == "onshell_v1":
                out = self.run_onshell(line)
            else:
                out = self.run_onshell(line, density_method=True)
            self._log_lhe_timers()
            return out
        elif spinmode == "PA":
            out = self.run_onshell(line, density_method=True)
            self._log_lhe_timers()
            return out
        elif spinmode == "madspin_v1":
            # legacy MadSpin / decay-chain path: fall through to decay_all_events below
            pass
        elif spinmode == "madspin":
            out = self.run_onshell(line, density_method=True)
            self._log_lhe_timers()
            return out
        elif spinmode == "bridge":
            raise Exception("Bridge mode not available.")
        else:
            raise Exception("spinmode %s not supported" % spinmode)
        
        if self.options['ms_dir'] and os.path.exists(pjoin(self.options['ms_dir'], 'madspin.pkl')):
            out = self.run_from_pickle()
            self._log_lhe_timers()
            return out
        
    
        args = self.split_arg(line)
        self.check_launch(args)
        for part in list(self.list_branches.keys()):
            if part in self.mg5cmd._multiparticles:
                if any(pid in self.final_state for pid in self.mg5cmd._multiparticles[part]):
                    break
            else:
                try:
                    pid = self.mg5cmd._curr_model.get('name2pdg')[part]
                except KeyError:
                    pid = self.mg5cmd._curr_model.get('name2pdg')[part.lower()]
                    self.list_branches[part.lower()] = self.list_branches[part]
                    del self.list_branches[part]
                    particle = self.mg5cmd._curr_model.get_particle(pid)
                    if particle.get('antiname').upper() in self.list_branches:
                        self.list_branches[particle.get('antiname').lower()] = \
                            self.list_branches[particle.get('antiname').upper()]
                        del self.list_branches[particle.get('antiname').upper()]
                if pid in self.final_state:
                    break
        else:
            if not self.options['onlyhelicity']:
                logger.info("Nothing to decay ...")
                return
        
        if self.options['BW_cut'] > 100:
            raise Exception("BW_cut parameter is much too large (>100) for narrow width approximation. Please set it up to a smaller value in your madspin_card.dat")

        model_line = self.banner.get('proc_card', 'full_model_line')

        if not self.options['seed']:
            self.options['seed'] = random.randint(0, int(30081*30081))
            #self.do_set('seed %s' % self.seed)
            logger.info('Will use seed %s' % self.options['seed'])
            self.history.insert(0, 'set seed %s' % self.options['seed'])

        if self.options['seed'] > 30081*30081: # can't use too big random number
            msg = 'Random seed too large ' + str(self.options['seed']) + ' > 30081*30081'
            raise Exception(msg)

        #self.options['seed'] = self.seed
        text = '%s\n' % '\n'.join([ line for line in self.history if line])
        self.banner.add_text('madspin' , text)
            
        time_me_generation = time.time()
        self.update_status('generating Madspin matrix element')
        generate_all = madspin.decay_all_events(self, self.banner, self.events_file, 
                                                    self.options)
        logger.info(f"Time for ME: {time.time()-time_me_generation:.2f} sec")        
        self.update_status('running MadSpin')
        generate_all.run()
                        
        self.branching_ratio = generate_all.branching_ratio
        self.cross = generate_all.cross
        self.error = generate_all.error
        self.efficiency = generate_all.efficiency
        try:
            self.err_branching_ratio = generate_all.err_branching_ratio
        except Exception:
            self.err_branching_ratio = 0
            
        evt_path = self.events_file.name
        try:
            self.events_file.close()
        except:
            pass
        misc.gzip(evt_path)
        decayed_evt_file=evt_path.replace('.lhe', '_decayed.lhe')
        misc.gzip(pjoin(self.options['curr_dir'],'decayed_events.lhe'),
                  stdout=decayed_evt_file)
        if not self.mother:
            logger.info("Decayed events have been written in %s.gz" % decayed_evt_file)

        # Now arxiv the shower card used if RunMaterial is present
        ms_card_path = pjoin(self.options['curr_dir'],'Cards','madspin_card.dat')
        run_dir = os.path.realpath(os.path.dirname(decayed_evt_file))
        if os.path.exists(ms_card_path):
            if os.path.exists(pjoin(run_dir,'RunMaterial.tar.gz')):
                misc.call(['tar','-xzpf','RunMaterial.tar.gz'], cwd=run_dir)
                base_path = pjoin(run_dir,'RunMaterial')
            else:
                base_path = pjoin(run_dir)

            evt_name = os.path.basename(decayed_evt_file).replace('.lhe', '')
            ms_card_to_copy = pjoin(base_path,'madspin_card_for_%s.dat'%evt_name)
            count = 0    
            while os.path.exists(ms_card_to_copy):
                count += 1
                ms_card_to_copy = pjoin(base_path,'madspin_card_for_%s_%d.dat'%\
                                                               (evt_name,count))
            files.cp(str(ms_card_path),str(ms_card_to_copy))
            
            if os.path.exists(pjoin(run_dir,'RunMaterial.tar.gz')):
                misc.call(['tar','-czpf','RunMaterial.tar.gz','RunMaterial'], 
                                                                    cwd=run_dir)
                shutil.rmtree(pjoin(run_dir,'RunMaterial'))
        self._log_lhe_timers()

    def run_from_pickle(self):
        import madgraph.iolibs.save_load_object as save_load_object
        
        generate_all = save_load_object.load_from_file(pjoin(self.options['ms_dir'], 'madspin.pkl'))
        
        #restore data passed to string to help pickle
        generate_all.all_decay = eval(generate_all.all_decay)
        for me in generate_all.all_ME:
            for d in generate_all.all_ME[me]['decays']:
                if isinstance(d['decay_struct'], str):
                    d['decay_struct'] = eval(d['decay_struct'])


        # Re-create information which are not save in the pickle.
        generate_all.evtfile = self.events_file
        generate_all.curr_event = madspin.Event(self.events_file, self.banner ) 
        generate_all.mgcmd = self.mg5cmd
        generate_all.mscmd = self 
        generate_all.pid2width = lambda pid: generate_all.banner.get('param_card', 'decay', abs(pid)).value
        generate_all.pid2mass = lambda pid: generate_all.banner.get('param_card', 'mass', abs(pid)).value
        if generate_all.path_me != self.options['ms_dir']:
            for decay in generate_all.all_ME.values():
                decay['path'] = decay['path'].replace(generate_all.path_me, self.options['ms_dir'])
                for decay2 in decay['decays']:
                    if decay2['path']: 
                        decay2['path'] = decay2['path'].replace(generate_all.path_me, self.options['ms_dir'])
            generate_all.path_me = self.options['ms_dir'] # directory can have been move
            generate_all.ms_dir = generate_all.path_me
        
        if not hasattr(self.banner, 'param_card'):
            self.banner.charge_card('slha')
        
        # Special treatment for the mssm. Convert the param_card to the correct
        # format
        if self.banner.get('model').startswith('mssm-') or self.banner.get('model')=='mssm':
            self.banner.param_card = check_param_card.convert_to_mg5card(\
                    self.banner.param_card, writting=False)
            
        for name, block in self.banner.param_card.items():
            if name.startswith('decay'):
                continue
                        
            orig_block = generate_all.banner.param_card[name]
            if block != orig_block:                
                raise Exception("""The directory %s is specific to a mass spectrum. 
                Your event file is not compatible with this one. (Different param_card: %s different)
                orig block:
                %s
                new block:
                %s""" \
                % (self.options['ms_dir'], name, orig_block, block))

        #replace init information
        generate_all.banner['init'] = self.banner['init']

        #replace run card if present in header (to make sure correct random seed is recorded in output file)
        if 'mgruncard' in self.banner:
            generate_all.banner['mgruncard'] = self.banner['mgruncard']   
        
        # NOW we have all the information available for RUNNING
        
        if self.options['seed']:
            #seed is specified need to use that one:
            open(pjoin(self.options['ms_dir'],'seeds.dat'),'w').write('%s\n'%self.options['seed'])
            #remove all ranmar_state
            for name in misc.glob(pjoin('*', 'SubProcesses','*','ranmar_state.dat'), 
                                                        self.options['ms_dir']):
                os.remove(name)    
        
        generate_all.ending_run()
        self.branching_ratio = generate_all.branching_ratio
        self.cross = generate_all.cross
        self.error = generate_all.error
        self.efficiency = generate_all.efficiency
        try:
            self.err_branching_ratio = generate_all.err_branching_ratio
        except Exception:
            # might not be define in some gridpack mode
            self.err_branching_ratio = 0 
        evt_path = self.events_file.name
        try:
            self.events_file.close()
        except:
            pass
        misc.gzip(evt_path)
        decayed_evt_file=evt_path.replace('.lhe', '_decayed.lhe')
        misc.gzip(pjoin(self.options['curr_dir'],'decayed_events.lhe'),
                  stdout=decayed_evt_file)
        if not self.mother:
            logger.info("Decayed events have been written in %s.gz" % decayed_evt_file)    
    
    

    def run_bridge(self, line):
        """Run the Bridge Algorithm"""
        
        # 1. Read the event file to check which decay to perform and the number
        #   of event to generate for each type of particle.
        # 2. Generate the events requested
        # 3. perform the merge of the events.
        #    if not enough events. re-generate the missing one.
        
        args = self.split_arg(line)


        asked_to_decay = set()
        for part in self.list_branches.keys():
            if part in self.mg5cmd._multiparticles:
                for pdg in self.mg5cmd._multiparticles[part]:
                    asked_to_decay.add(pdg)
            else:
                asked_to_decay.add(self.mg5cmd._curr_model.get('name2pdg')[part])

        #0. Define the path where to write the file
        self.path_me = os.path.realpath(self.options['curr_dir']) 
        if self.options['ms_dir']:
            self.path_me = os.path.realpath(self.options['ms_dir'])
            if not os.path.exists(self.path_me):
                os.mkdir(self.path_me) 
        else:
            # cleaning (force: previous run may have left read-only frozen gridpacks)
            for name in misc.glob("decay_*_*", self.path_me):
                _force_rmtree(name)

        if self.events_file:
            self.events_file.close()
            filename = self.events_file.name
        else:
            filename = self.inputfile

        if self.options['input_format'] == 'auto':
            if '.lhe' in filename :
                self.options['input_format']  = 'lhe'
            elif '.hepmc' in filename:
                self.options['input_format']  = 'hepmc'
            else:
                raise Exception("fail to recognized input format automatically")
                
        if self.options['input_format'] in ['lhe', 'lhe_no_banner']:
            orig_lhe = lhe_parser.EventFile(filename)
            if self.options['input_format'] == 'lhe_no_banner':
                orig_lhe.allow_empty_event = True
                
        elif self.options['input_format'] in ['hepmc']:
            import madgraph.various.hepmc_parser as hepmc_parser
            orig_lhe = hepmc_parser.HEPMC_EventFile(filename)
            orig_lhe.allow_empty_event = True
            logger.info("Parsing input event to know how many decay to generate. This can takes few minuts.")
        else:
            raise Exception
            
        to_decay = collections.defaultdict(int)
        nb_event = 0
 
        for event in orig_lhe:
            nb_event +=1
            for particle in event:
                if particle.status == 1 and particle.pdg in asked_to_decay:
                    # final state and tag as to decay
                    to_decay[particle.pdg] += 1
            if self.options['input_format'] == 'hepmc' and nb_event == 250:
                currpos = orig_lhe.tell()
                filesize = orig_lhe.getfilesize()
                for key in to_decay:
                    to_decay[key] *= 1.05 * filesize/ currpos 
                    # 1.05 to avoid accidental coincidence with nevents
                break

        # Handle the banner of the output file
        if not self.options['seed']:
            self.options['seed'] = random.randint(0, int(30081*30081))
            #self.do_set('seed %s' % self.seed)
            logger.info('Will use seed %s' % self.options['seed'])
            self.history.insert(0, 'set seed %s' % self.options['seed'])

        if self.options['seed'] > 30081*30081: # can't use too big random number
            msg = 'Random seed too large ' + str(self.options['seed']) + ' > 30081*30081'
            raise Exception(msg)

        #self.options['seed'] = self.options['seed']
        
        text = '%s\n' % '\n'.join([ line for line in self.history if line])
        self.banner.add_text('madspin' , text)


        # 2. Generate the events requested
        with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
            mg5 = self.mg5cmd
            if not self.model:
                modelpath = self.model.get('modelpath+restriction')
                mg5.exec_cmd("import model %s" % modelpath)      
            evt_decayfile = {} 
            for pdg, nb_needed in to_decay.items():
                #check if a splitting is needed
                if nb_needed == nb_event:
                    evt_decayfile[pdg] = self.generate_events(pdg, min(nb_needed,100000), mg5)
                elif nb_needed %  nb_event == 0:
                    nb_mult = nb_needed // nb_event
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches:
                        continue
                    elif len(self.list_branches[name]) == nb_mult:
                        evt_decayfile[pdg] = self.generate_events(pdg, min(nb_event,100000), mg5)
                    else:
                        evt_decayfile[pdg] = self.generate_events(pdg, min(nb_needed,100000), mg5, cumul=True)
                elif self.options['cross_section']:
                    #cross-section hard-coded -> allow 
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    
                    if name not in self.list_branches:
                        continue
                    else:
                        try:
                            evt_decayfile[pdg] = self.generate_events(pdg, min(nb_needed,100000), mg5, cumul=True)
                        except common_run_interface.ZeroResult:
                            logger.warning("Branching ratio is zero for this particle. Not decaying it")
                            del to_decay[pdg]                    
                else:
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches or len(self.list_branches[name]) == 0:
                        continue
                    #raise self.InvalidCmd("The bridge mode of MadSpin does not support event files where events do not *all* share the same set of final state particles to be decayed. One workaround is to force the final cross-section manually.")
                    if len(self.list_branches[name]) == 1:
                        evt_decayfile[pdg] = self.generate_events(pdg, min(nb_event,100000), mg5)
                    else:
                        evt_decayfile[pdg] = self.generate_events(pdg, min(nb_needed,100000), mg5, cumul=True)
                    
                     
        # Compute the branching ratio.
        if not self.options['cross_section']:
            br = 1
            multi_br = [ ]
            multi_totevt = 0
            for (pdg, event_files) in evt_decayfile.items():
                if not event_files:
                    continue
                totwidth = float(self.banner.get('param', 'decay', abs(pdg)).value)
                if to_decay[pdg] == nb_event:
                    # Exactly one particle of this type to decay by event
                    pwidth = sum([event_files[k].cross for k in event_files])
                    if pwidth > 1.01 * totwidth:
                        logger.critical("Branching ratio larger than one for %s " % pdg) 
                    br *= pwidth / totwidth
                elif to_decay[pdg] % nb_event == 0:
                    # More than one particle of this type to decay by event
                    # Need to check the number of event file to check if we have to 
                    # make separate type of decay or not.
                    nb_mult = to_decay[pdg] // nb_event
                    if nb_mult == len(event_files):
                        for k in event_files:
                            pwidth = event_files[k].cross
                            if pwidth > 1.01 * totwidth:
                                logger.critical("Branching ratio larger than one for %s " % pdg)                       
                            br *= pwidth / totwidth
                        br *= math.factorial(nb_mult)
                    else:
                        pwidth = sum(event_files[k].cross for k in event_files)
                        if pwidth > 1.01 * totwidth:
                            logger.critical("Branching ratio larger than one for %s " % pdg) 
                        br *= (pwidth / totwidth)**nb_mult
                else:
                    pwidth = sum([event_files[k].cross for k in event_files])        
                    multi_br.append(pwidth / totwidth) 
                    multi_totevt += to_decay[pdg] % nb_event
            if multi_br and multi_totevt % nb_event == 0:
                if all(misc.equal(br,multi_br[0], 2) for br in multi_br): 
                    logger.warning("not all event are decaying the same particle, this is only supported if each event have ONE decaying particle (not checked) and that all particles have the same BR")        
                else:
                    raise self.InvalidCmd("The bridge mode of MadSpin does not support event files where events do not *all* share the same set of final state particles to be decayed: [%s %s ] " %(multi_br, multi_totevt))
            elif multi_br:
                raise self.InvalidCmd("The bridge mode of MadSpin does not support event files where events do not *all* share the same set of final state particles to be decayed. (%s %s)" % (multi_br, multi_totevt))
        else:
            br = 1
        self.branching_ratio = br
        self.efficiency = 1
        try:
            self.cross, self.error = self.banner.get_cross(witherror=True)
        except:
            if self.options['input_format'] != 'lhe':
                self.cross, self.error = 0, 0
        self.cross *= br
        self.error *= br
        
        # modify the cross-section in the init block of the banner
        if not self.options['cross_section']:
            self.banner.scale_init_cross(self.branching_ratio)
        else:
            
            if self.options['input_format'] in ['lhe_no_banner','hepmc'] and 'init' not in self.banner:
                self.cross = sum(self.options['cross_section'].values())
                self.error = 0
                self.branching_ratio = 1
            else:  
                self.banner.modify_init_cross(self.options['cross_section'])
                new_cross, new_error =   self.banner.get_cross(witherror=True)
                self.branching_ratio = new_cross / self.cross
                self.cross = new_cross   
                self.error = new_error

        # 3. Merge the various file together.
        if self.options['input_format'] == 'hepmc':
            name = orig_lhe.name.replace('.hepmc', '_decayed.lhe')
            if not name.endswith('.gz'):
                name = '%s.gz' % name
            
            output_lhe = lhe_parser.EventFile(name, 'w')
        else:
            name = orig_lhe.name.replace('.lhe', '_decayed.lhe')
            if not name.endswith('.gz'):
                name = '%s.gz' % name
            output_lhe = lhe_parser.EventFile(name, 'w')
        try:
            self.banner.write(output_lhe, close_tag=False)
        except Exception:
            if self.options['input_format'] == 'lhe':
                raise
        
        # initialise object which store not use event due to wrong helicity
        bufferedEvents_decay = {}
        for pdg in evt_decayfile:
            bufferedEvents_decay[pdg] = [{}] * len(evt_decayfile[pdg])
        
        import time
        start = time.time()
        counter = 0
        orig_lhe.seek(0)

        for event in orig_lhe:
            if counter and counter % 100 == 0 and float(str(counter)[1:]) ==0:
                print("decaying event number %s [%s s]" % (counter, time.time()-start))
            counter +=1
            
            # use random order for particles to avoid systematics when more than 
            # one type of decay is asked.
            particles = [p for p in event if int(p.status) == 1.0]
            random.shuffle(particles)
            ids = [particle.pid for particle in particles]
            br = 1 #br for that particular events (for special/weighted case)
            hepmc_output = lhe_parser.Event() #for hepmc case: collect the decay particle
            for i,particle in enumerate(particles):
                # check if we need to decay the particle 
                if self.final_state and particle.pdg not in self.final_state:
                    continue # nothing to do for this particle
                if particle.pdg not in evt_decayfile:
                    continue # nothing to do for this particle
                
                # check how the decay need to be done
                nb_decay = len(evt_decayfile[particle.pdg])
                if nb_decay == 0:
                    continue #nothing to do for this particle
                if nb_decay == 1:
                    decay_file = evt_decayfile[particle.pdg][0]
                    decay_file_nb = 0
                elif ids.count(particle.pdg) == nb_decay:
                    decay_file = evt_decayfile[particle.pdg][ids[:i].count(particle.pdg)]
                    decay_file_nb = ids[:i].count(particle.pdg)
                else:
                    #need to select the file according to the associate cross-section
                    r = random.random()
                    tot = sum(evt_decayfile[particle.pdg][key].cross for key in evt_decayfile[particle.pdg])
                    r = r * tot
                    cumul = 0
                    for j,events in evt_decayfile[particle.pdg].items():
                        cumul += events.cross
                        if r <= cumul:
                            decay_file = events
                            decay_file_nb = j
                            break
                    else:
                        # security for numerical accuracy issue... (unlikely but better safe)
                        if (cumul-tot)/tot < 1e-5:
                            decay_file = events
                            decay_file_nb = j
                        else:
                            misc.sprint(j,cumul, events.cross, tot, (tot-cumul)/tot)
                            raise Exception
                
                if self.options['new_wgt'] == 'BR':
                    tot_width = float(self.banner.get('param', 'decay', abs(pdg)).value)
                    if tot_width:
                        br = decay_file.cross / tot_width
                # ok start the procedure
                if hasattr(particle,'helicity'):
                    helicity = particle.helicity
                else:
                    helicity = 9
                bufferedEvents = bufferedEvents_decay[particle.pdg][decay_file_nb]
                
                # now that we have the file to read. find the associate event
                # checks if we have one event in memory
                if helicity in bufferedEvents and bufferedEvents[helicity]:
                    decay = bufferedEvents[helicity].pop()
                else:
                    # read the event file up to completion
                    while 1:
                        try:
                            decay = next(decay_file)
                        except StopIteration:
                            # check how far we are
                            ratio = counter / nb_event 
                            needed = 1.05 * to_decay[particle.pdg] - counter
                            needed = min(100000, max(needed, 6000))
                            with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
                                new_file = self.generate_events(particle.pdg, needed, mg5, [decay_file_nb])
                            evt_decayfile[particle.pdg].update(new_file)
                            decay_file = evt_decayfile[particle.pdg][decay_file_nb]
                            continue

                        if helicity == decay[0].helicity or helicity==9 or \
                                            self.options["spinmode"] == "none":
                            break # use that event
                        # not valid event store it for later
                        if helicity not in bufferedEvents:
                            bufferedEvents[helicity] = [decay]
                        elif len(bufferedEvents[helicity]) < 200:
                            # only add to the buffering if the buffer is not too large
                            bufferedEvents[helicity].append(decay)
                # now that we have the event make the merge
                if self.options['input_format'] != 'hepmc':
                    particle.add_decay(decay)
                else:
                    if len(hepmc_output) == 0:
                        hepmc_output.append(lhe_parser.Particle(event=hepmc_output))
                        hepmc_output[0].color2 = 0
                        hepmc_output[0].status = -1
                        hepmc_output.nexternal+=1
                    decayed_particle = lhe_parser.Particle(particle, hepmc_output)
                    decayed_particle.mother1 = hepmc_output[0]
                    decayed_particle.mother2 = hepmc_output[0]
                    hepmc_output.append(decayed_particle)
                    hepmc_output.nexternal+=1
                    decayed_particle.add_decay(decay)
            # change the weight associate to the event
            if self.options['new_wgt'] == 'cross-section':
                event.wgt *= self.branching_ratio
                br = self.branching_ratio
            else:
                event.wgt *= br
                
            if self.options['input_format'] != 'hepmc':
                wgts = event.parse_reweight()
                for key in wgts:
                    wgts[key] *= br
                # all particle have been decay if needed
                output_lhe.write(str(event))
            else:
                hepmc_output.wgt = event.wgt
                hepmc_output.nexternal = len(hepmc_output) # the append does not update nexternal
                output_lhe.write(str(hepmc_output))
        else:
            if counter==0:
                raise Exception
        output_lhe.write('</LesHouchesEvents>\n')        
                    
    
    def load_model(self, name, use_mg_default, complex_mass=False):
        """load the model"""
        
        loop = False
        #if (name.startswith('loop_')):
        #    logger.info("The model in the banner is %s" % name)
        #    logger.info("Set the model to %s since only" % name[:5])
        #    logger.info("tree-level amplitudes are used for the decay ")
        #    name = name[5:]
        #    self.banner.proc_card.info['full_model_line'].replace('loop_','')

        logger.info('detected model: %s. Loading...' % name)
        model_path = name
        #base_model = import_ufo.import_model(model_path)

        # Import model
        base_model = import_ufo.import_model(name, decay=True,
                                               complex_mass_scheme=complex_mass)

        if use_mg_default:
            base_model.pass_particles_name_in_mg_default()
        
        self.model = base_model
        self.mg5cmd._curr_model = self.model
        self.mg5cmd.process_model()

    def generate_events(self, pdg, nb_event, mg5, restrict_file=None, cumul=False,
                        output_width=False, run_name='run_01'):
        """generate new events for this particle
           restrict_file allow to only generate a subset of the definition
           cumul allow to merge all the definition in one run (add process)
                 to generate events according to cross-section
           run_name allow a refill to write to its own run directory instead of
                 overwriting the pool that is currently being read
        """
        if not hasattr(self, 'me_int'):
            self.me_int = {}
            
        
        
        nb_event = int(nb_event) # in case of hepmc request the nb_event is not an integer
        # Gridpack-based decay generation is only used when we persist a gridpack
        # across runs (ms_dir). Building/packaging a gridpack and then generating
        # through run.sh is markedly slower (and uses the cores less densely)
        # than the direct MadEventCmdShell generation below, so the parallel
        # unweighting (nb_core>1) does NOT use it: the parent pre-generates the
        # whole decay pool here, once, with the fast native path.
        use_gridpack = bool(self.options['ms_dir'])
        if cumul:
            width = 0.
        else:
            width = 1.
        # channel number -> its own partial width. ``width`` above is the product
        # (or the sum, under cumul) that the branching ratio has always been
        # built from; the grouping needs each channel on its own, since a group
        # takes one channel per particle and its rate is the product over *its*
        # channels only.
        channel_widths = {}
        part = self.model.get_particle(pdg)
        if not part:
            return {}# this particle is not defined in the current model so ignore it
        name = part.get_name()
        out = {}
        time_gen_dec = time.time()
        logger.info("generate %s decay event for particle %s" % (int(nb_event), name))
        if name not in self.list_branches:
            return out
        for i,tagged_proc in enumerate(self.list_branches[name]):
            # the '@' grouping tag is MadSpin's own bookkeeping: MG5 would read
            # it as a process number (and the decay-ME generation appends one of
            # its own), so it never reaches the generation
            proc = self._split_group_tag(tagged_proc)[0]
            if restrict_file and i not in restrict_file:
                continue
            decay_dir = pjoin(self.path_me, "decay_%s_%s" %(str(pdg).replace("-","x"),i))
            if not os.path.exists(decay_dir):
                if cumul:
                    mg5.exec_cmd("generate %s" % proc)
                    for j,proc2 in enumerate(self.list_branches[name][1:]):
                        if restrict_file and j not in restrict_file:
                            raise Exception # Do not see how this can happen
                        mg5.exec_cmd("add process %s"
                                     % self._split_group_tag(proc2)[0])
                    mg5.exec_cmd("output %s -f" % decay_dir)
                else:
                    mg5.exec_cmd("generate %s" % proc)
                    mg5.exec_cmd("output %s -f" % decay_dir)
                
                options = dict(mg5.options)
                if use_gridpack:
                    # gridpack mode -> build the integration grid once here
                    if decay_dir in self.me_int:
                        me5_cmd = self.me_int[decay_dir]
                    else:
                        me5_cmd = madevent_interface.MadEventCmdShell(me_dir=os.path.realpath(\
                                                decay_dir), options=options)
                        me5_cmd.options["automatic_html_opening"] = False
                        me5_cmd.options["madanalysis5_path"] = None
                        me5_cmd.options["madanalysis_path"] = None
                        me5_cmd.allow_notification_center = False
                        try:
                            os.remove(pjoin(decay_dir, 'Cards', 'madanalysis5_parton_card_default.dat'))
                            os.remove(pjoin(decay_dir, 'Cards', 'madanalysis5_parton_card.dat'))
                        except Exception as error:
                            logger.debug(error)
                            pass 
                        self.me_int[decay_dir] = me5_cmd

                    if self.options["run_card"]:
                        run_card = self.run_card
                    else:
                        run_card = banner.RunCard(pjoin(decay_dir, "Cards", "run_card.dat"))                        
                    run_card["iseed"] = self.options['seed']
                    run_card['gridpack'] = True
                    run_card['systematics_program'] = 'False'
                    run_card['use_syst'] = False
                    run_card.__setitem__('allow_overshoot_events', True, change_userdefine=True)
                    run_card.__setitem__('refine_evt_by_job', 5000, change_userdefine=True)
                    run_card.write(pjoin(decay_dir, "Cards", "run_card.dat"))
                    param_card = self.banner['slha']
                    open(pjoin(decay_dir, "Cards", "param_card.dat"),"w").write(param_card)
                    self.options['seed'] += 1
                    self.seed = self.options['seed'] 
                    # actually creation
                    me5_cmd.exec_cmd("generate_events run_01 -f")
                    if output_width:
                        channel_widths[i] = me5_cmd.results.current['cross']
                        if cumul:
                            width += me5_cmd.results.current['cross']
                        else:
                            width *= me5_cmd.results.current['cross']
                    me5_cmd.exec_cmd("exit")
                    #remove pointless informat
                    if not os.path.exists(pjoin(decay_dir, 'run.sh')):
                        devnull = open('/dev/null','w')
                        misc.call(["rm", "Cards", "bin", 'Source', 'SubProcesses'], cwd=decay_dir,stdout=devnull, stderr=-2)
                        misc.call(['tar', '-xzpvf', 'run_01_gridpack.tar.gz'], cwd=decay_dir,stdout=devnull, stderr=-2)
                        devnull.close()
            # Now generate the events
            if not use_gridpack:
                if decay_dir in self.me_int:
                        me5_cmd = self.me_int[decay_dir]
                else:
                    # ``_gen_nb_core`` caps the cores this generation may use, so
                    # that several decay generations running at the same time
                    # (see _generate_decays) never oversubscribe the machine.
                    gen_options = dict(mg5.options)
                    if getattr(self, '_gen_nb_core', None):
                        gen_options['nb_core'] = self._gen_nb_core
                    me5_cmd = madevent_interface.MadEventCmdShell(me_dir=os.path.realpath(\
                                                    decay_dir), options=gen_options)
                    me5_cmd.options["automatic_html_opening"] = False
                    me5_cmd.options["automatic_html_opening"] = False
                    me5_cmd.options["madanalysis5_path"] = None
                    me5_cmd.options["madanalysis_path"] = None
                    me5_cmd.allow_notification_center = False
                    try:
                        os.remove(pjoin(decay_dir, 'Cards', 'madanalysis5_parton_card_default.dat'))
                        os.remove(pjoin(decay_dir, 'Cards', 'madanalysis5_parton_card.dat'))
                    except Exception as error:
                        logger.debug(error)
                        pass                 
                    self.me_int[decay_dir] = me5_cmd
                if self.options["run_card"]:
                    if hasattr(self, 'run_card'):
                        run_card = self.run_card
                    elif hasattr(self.options, 'run_card'):
                        run_card = self.options.run_card
                    else:
                        self.run_card = banner.RunCard(self.options["run_card"])
                        run_card = self.run_card 
                else:
                    run_card = banner.RunCard(pjoin(decay_dir, "Cards", "run_card.dat"))
                run_card["nevents"] = int(0.8*nb_event)
                run_card.__setitem__('allow_overshoot_events', True, change_userdefine=True)
                run_card.__setitem__('refine_evt_by_job', 5000, change_userdefine=True)
                # Handle the banner of the output file
                if not self.seed:
                    self.seed = random.randint(0, int(30081*30081))
                    self.do_set('seed %s' % self.seed)
                    logger.info('Will use seed %s' % self.seed)
                    self.history.insert(0, 'set seed %s' % self.seed)
                run_card["iseed"] = self.seed
                run_card["systematics_program"] = 'None'
                run_card['use_syst'] = False
                # Under the parallel unweighting, have the final unweighting hand
                # us one file per worker instead of a single pool: each worker
                # then reads only its own file. And do not gzip them -- they are
                # read back immediately, compressing them is pure overhead.
                nb_split = self._decay_pool_split()
                if nb_split > 1:
                    run_card.__setitem__('nb_unweight_output', nb_split,
                                         change_userdefine=True)
                    run_card.__setitem__('zip_unweighted_events', False,
                                         change_userdefine=True)
                run_card.write(pjoin(decay_dir, "Cards", "run_card.dat"))
                param_card = self.banner['slha']
                open(pjoin(decay_dir, "Cards", "param_card.dat"),"w").write(param_card)
                self.seed += 1
                me5_cmd.exec_cmd("generate_events %s -f" % run_name)
                if output_width:
                    channel_widths[i] = me5_cmd.results.current['cross']
                    if cumul:
                        width += me5_cmd.results.current['cross']
                    else:
                        width *= me5_cmd.results.current['cross']
                if run_card["nevents"] > 1.01 * me5_cmd.results.current['nb_event']:
                    logger.critical('The number of event generated is only %s/%s. This typically indicates that you need specify cut on the decay process.',me5_cmd.results.current['nb_event'], run_card["nevents"])
                    logger.critical('We strongly suggest that you cancel/discard this run.')
                me5_cmd.exec_cmd("exit")
                if nb_split > 1:
                    out[i] = _ChainedEvents(lhe_parser.EventFile.unweight_output_paths(
                        pjoin(decay_dir, "Events", run_name, 'unweighted_events.lhe'),
                        nb_split))
                else:
                    out[i] = lhe_parser.EventFile(pjoin(decay_dir, "Events", run_name, 'unweighted_events.lhe.gz'))
            else:
                if not self.seed:
                    if hasattr(self, 'mother'):
                        try:
                            self.seed = 100 + self.mother.run_card['iseed']
                        except:
                            self.seed = random.randint(0, int(30081*30081))
                self.seed += 1
                if self.seed > 30081*30081:
                    self.seed -= 30081*30081
                logger.info('Will use seed %s' % (self.seed))
                rc, log = self._run_gridpack(
                    [pjoin(decay_dir, 'run.sh'), str(int(1.2*nb_event)),
                     str(self.seed), '-p', str(self._resolve_nb_core())],
                    cwd=decay_dir)
                events_path = pjoin(decay_dir, 'events.lhe.gz')
                if not os.path.exists(events_path):
                    raise Exception(
                        "Gridpack decay generation failed (rc=%s): %s was "
                        "not produced by run.sh/gridrun.\n"
                        "--- last run.sh/gridrun output ---\n%s"
                        % (rc, events_path, log))
                out[i] = lhe_parser.EventFile(events_path)
            if cumul:
                break
        time_gen_dec = time.time()-time_gen_dec
        logger.info(f"Time for decay event generation = {time_gen_dec:.1f} sec")
        if not output_width:
            return out
        else:
            return out, width, channel_widths

    def run_onshell(self, line, density_method=False):
        """Run the onshell Algorithm"""
        
        # 1. Read the event file to check which decay to perform and the number
        #   of event to generate for each type of particle. (assume efficiency=1 for spin 0
        #   otherwise efficiency=2
        # 2. Generate the associated events
        # 3. generate the various matrix-element (production/decay/production+decay) 
        #    => no production+decay if density_method on True
        # 4. determine the maxwgt
        # 5. generate the decay (for each production event)
        # 6. perform the merge of the events.
        #    if not enough events. re-generate the missing one.
        
        # Spyros: this is not used - remove?
        args = self.split_arg(line)

        # First define an utility function for generating events when needed
        # Spyros what should be done here? 

        # Find which particles should be decayed
        asked_to_decay = set()
        for part in self.list_branches.keys():
            if part in self.mg5cmd._multiparticles:
                for pdg in self.mg5cmd._multiparticles[part]:
                    asked_to_decay.add(pdg)
            else:
                asked_to_decay.add(self.mg5cmd._curr_model.get('name2pdg')[part])

        # 0. Define the path where to write the file
        self.path_me = os.path.realpath(self.options['curr_dir']) 
        if self.options['ms_dir']:
            self.path_me = os.path.realpath(self.options['ms_dir'])
            if not os.path.exists(self.path_me):
                os.mkdir(self.path_me) 
        else:
            # cleaning (force: previous run may have left read-only frozen gridpacks)
            for name in misc.glob("decay_*_*", self.path_me):
                _force_rmtree(name)

        self.events_file.close()
        if self.events_file.name.endswith('.gz'):
            misc.gunzip(self.events_file.name)
        orig_lhe = lhe_parser.EventFile(self.events_file.name)
        if self.options['fixed_order']:
            orig_lhe.eventgroup = True

        # Dictionary with particle properties
        decay_dict = {}
        
        # 1. Open input event file and check which particles to decay
        # - count the number of particles to be decayed.
        to_decay = collections.defaultdict(int)	
        nb_event = 0
        for event in orig_lhe:
            if self.options['fixed_order']:
                event = event[0]
            nb_event +=1
            for particle in event:
                if particle.status == 1 and particle.pdg in asked_to_decay:
                    # final state and tag as to decay
                    to_decay[particle.pdg] += 1
                    # Properties of decaying particle
                    width = self.banner.get('param_card', 'decay', abs(particle.pdg)).value
                    mass = self.banner.get('param_card', 'mass', abs(particle.pdg)).value
                    color = self.model.get_particle(particle.pdg).get('color')
                    spin = self.model.get_particle(particle.pdg).get('spin')
                    decay_dict[particle.pdg] = [width, mass, color, spin]
        #print(f"to_decay = {to_decay}")
        # How many particles decay in one event -- the same multiplicity the
        # pool ladder counts. It decides which unweighting scheme 'auto' picks,
        # so it is resolved once here rather than per event: the modes have
        # different bounds, and a mode that changed event to event would be
        # testing against somebody else's.
        self._nb_decaying = sum(max(1, int(nb) // int(nb_event))
                                for nb in to_decay.values()) if nb_event else 0
                	
        with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
            mg5 = self.mg5cmd
            if not self.model:
                modelpath = self.model.get('modelpath+restriction')
                mg5.exec_cmd("import model %s" % modelpath)  
                self.model = mg5._curr_model

        # Handle the banner of the output file
        if not self.seed:
            self.seed = random.randint(0, int(30081*30081))
            self.do_set('seed %s' % self.seed)
            logger.info('Will use seed %s' % self.seed)
            self.history.insert(0, 'set seed %s' % self.seed)

        if self.seed > 30081*30081: # can't use too big random number
            msg = 'Random seed too large ' + str(self.seed) + ' > 30081*30081'
            raise Exception(msg)

        self.options['seed'] = self.seed
        #print(f"from run onshell seed = {self.seed}")
        
        text = '%s\n' % '\n'.join([ line for line in self.history if line])
        self.banner.add_text('madspin' , text)


        # 2. Generate the events requested
        nevents_for_max = self.options['Nevents_for_max_weight']
        if nevents_for_max == 0 :
            nevents_for_max = 75
        nevents_for_max *= self.options['max_weight_ps_point']
        # Security margin on the decays reserved for the maximum-weight scan.
        # The scan draws exactly nevents_for_max decays *per slot* (measured: no
        # restarts for tops), but the parallel scan stripes the pool across the
        # workers, and dividing the bare reservation leaves each worker's slice
        # only just big enough -- an uneven split then exhausts one worker and
        # forces a mid-scan refill. A 50% margin absorbs that unevenness.
        nevents_for_max = int(1.5 * nevents_for_max)
        
        with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
            mg5 = self.mg5cmd
            if not self.model:
                modelpath = self.model.get('modelpath+restriction')
                mg5.exec_cmd("import model %s" % modelpath)      
            evt_decayfile = {}
            br = 1.
            # pdg -> br_pdg for the "mixed final-state" case (events do not
            # all share the same set of decaying particles). Filled in the
            # else-branch below and consumed after the loop to compute the
            # per-pdg drop probability that equalizes BRs across productions.
            mixed_pdgs_br = {}
            # 1) Decide what has to be generated for each decaying particle.
            #    Nothing is generated yet: the generations are launched together
            #    below so they overlap instead of running one particle after the
            #    other.
            gen_jobs = collections.OrderedDict()
            # '@' grouping tags. Resolved before anything is generated: it
            # decides how many pools each particle gets, and it forces the joint
            # accept/reject, which _sequential_pool_ladder below already needs to
            # know about.
            self._resolve_decay_groups(to_decay, nb_event, density_method)
            # How many decay events one production event burns per decaying
            # particle. The joint accept/reject redraws the whole set on a
            # reject, so every pool is consumed at the same rate; the sequential
            # one redraws a single particle, so each pool is consumed at its own
            # slot's rate -- the ladder of _decay_pool_ladder.
            seq_ladder = self._sequential_pool_ladder(to_decay, nb_event,
                                                      density_method)
            for pdg, nb_needed in to_decay.items():
                # muliply by expected effeciency of generation
                spin = self.model.get_particle(pdg).get('spin')
                if pdg in seq_ladder:
                    efficiency = seq_ladder[pdg]
                elif spin == 1:
                    efficiency = 1.1
                else:
                    efficiency = 2.0

                totwidth = self.banner.get('param_card', 'decay', abs(pdg)).value

                #check if a splitting is needed
                if pdg in self._decay_group_pdgs():
                    # Grouped: one pool per decay *line*, whatever the
                    # multiplicity -- a group hands each of its lines to one
                    # particle, so a line is never mixed with another one in a
                    # single pool the way `simple`/`mult_cumul` mix them.
                    #
                    # Every pool is sized as if its group were drawn on every
                    # event. The exact size would be that times the group's
                    # share p_g, but p_g is only known once the partial widths
                    # have been measured, i.e. once this generation has run. So
                    # over-generate by at most a factor |groups| rather than
                    # under-generate and pay for refills; `decay_event_mult`
                    # scales it down for anyone who minds.
                    gen_jobs[pdg] = {'kind': 'grouped', 'totwidth': totwidth,
                        'nb_mult': nb_needed // nb_event, 'cumul': False,
                        'nb_gen': (int(efficiency*nb_event) + nevents_for_max)
                                  * self.options['decay_event_mult']}
                elif nb_needed == nb_event:
                    gen_jobs[pdg] = {'kind': 'simple', 'totwidth': totwidth,
                        'cumul': True,
                        'nb_gen': (int(efficiency*nb_needed) + nevents_for_max)
                                  * self.options['decay_event_mult']}
                elif nb_needed %  nb_event == 0:
                    nb_mult = nb_needed // nb_event
                    nb_cumul = (int(efficiency*nb_needed) + nevents_for_max*nb_mult) \
                                * self.options['decay_event_mult']
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches:
                        continue
                    elif len(self.list_branches[name]) == nb_mult:
                        gen_jobs[pdg] = {'kind': 'mult_split', 'totwidth': totwidth,
                            'nb_mult': nb_mult, 'cumul': False,
                            'nb_gen': nb_event * self.options['decay_event_mult']}
                    else:
                        gen_jobs[pdg] = {'kind': 'mult_cumul', 'totwidth': totwidth,
                            'nb_mult': nb_mult, 'cumul': True, 'nb_gen': nb_cumul}
                else:
                    # Mixed case: events do not all share the same final-state
                    # particles to be decayed. We collect this pdg here and, once
                    # the loop is done, equalize BRs via the legacy
                    # add_loose_decay mechanism (drop events sampled to the
                    # "fake" decay channel so the output stays unweighted).
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches or len(self.list_branches[name]) == 0:
                        continue
                    gen_jobs[pdg] = {'kind': 'mixed', 'totwidth': totwidth,
                        'cumul': True,
                        'nb_gen': (int(efficiency*nb_needed) + nevents_for_max)
                                  * self.options['decay_event_mult']}

            # 2) Generate every particle's decay events at the same time.
            gen_results = self._generate_decays(gen_jobs, mg5)

            # 3) Fold the measured partial widths into the branching ratio.
            channel_widths = {}
            for pdg, job in gen_jobs.items():
                evt_decayfile[pdg], pwidth, channel_widths[pdg] = gen_results[pdg]
                totwidth = job['totwidth']
                if job['kind'] == 'grouped':
                    continue        # done together, below: a group's rate is a
                                    # product over several particles at once
                if job['kind'] == 'mult_split':
                    # ``pwidth`` is the *product* of the channels' widths here,
                    # so measuring it against a single total width is a category
                    # error -- it fires as soon as totwidth > 1 GeV (two decay
                    # lines for a top) and the clamp below would then quietly
                    # wreck the branching ratio. Check each channel instead.
                    product = 1.0
                    for i in sorted(channel_widths[pdg]):
                        product *= self._clamped_partial_width(
                            channel_widths[pdg][i], totwidth, pdg)
                    br *= (product / totwidth**job['nb_mult']
                           * self._assignment_multiplicity(
                               self.list_branches[self.model.get_particle(pdg)
                                                  .get_name()]))
                    continue
                pwidth = self._clamped_partial_width(pwidth, totwidth, pdg)
                if job['kind'] == 'simple':
                    br *= pwidth / totwidth
                elif job['kind'] == 'mult_cumul':
                    br *= (pwidth / totwidth)**job['nb_mult']
                else:
                    mixed_pdgs_br[pdg] = pwidth / totwidth

            # 3b) The grouped particles' branching ratio, and with it the
            #     probability of each group. A group is one complete assignment
            #     of channels to particles, so its rate is the product of its
            #     own partial widths over every grouped particle -- and the
            #     groups are alternatives, so they add.
            if getattr(self, '_decay_groups', None):
                br *= self._resolve_group_rates(gen_jobs, channel_widths)

        # Equalize branching ratios across mixed productions (legacy
        # add_loose_decay mechanism): pick max_br as the global BR factor and
        # drop, per event, with probability 1 - br_pdg / max_br so the output
        # sample stays unweighted. The banner cross-section is corrected after
        # the decay loop (below) once the actual number of kept events is
        # known.
        drop_prob_per_pdg = {}
        if mixed_pdgs_br:
            max_mixed_br = max(mixed_pdgs_br.values())
            br *= max_mixed_br
            for pdg, pdg_br in mixed_pdgs_br.items():
                drop_prob_per_pdg[pdg] = 1.0 - pdg_br / max_mixed_br
            if any(d > 1e-9 for d in drop_prob_per_pdg.values()):
                logger.warning(
                    "Mixed-pdg production processes have different total BRs "
                    "(per-pdg BR=%s, max=%g). Equalizing by dropping events; "
                    "the output sample stays unweighted and the banner "
                    "cross-section reflects the effective BR.",
                    {k: '%.4g' % v for k, v in mixed_pdgs_br.items()},
                    max_mixed_br,
                )
        mixed_pdgs_set = set(drop_prob_per_pdg.keys())

        self.branching_ratio = br
        self.efficiency = 1
        self.cross, self.error = self.banner.get_cross(witherror=True)
        self.cross *= self.branching_ratio
        self.error *= self.branching_ratio
        

        density_pole_approximation = self.options['spinmode'] in ['PA', 'onshell']
        density_do_reshuffle = self.options['spinmode'] == 'PA'
        density_needs_reshuffle = (
            density_method
            and (not density_pole_approximation
                 or density_do_reshuffle)
        )

        # 3. generate the various matrix-elements
        time_me_generation = time.time()
        self.update_status('generating Madspin matrix element (density_method=%s)' % density_method)
        if density_method:
            self.generate_all = madspin.decay_all_events_density(self, self.banner, self.events_file,self.options)
        else:
            self.generate_all = madspin.decay_all_events_onshell(self, self.banner, self.events_file,self.options)

        self.generate_all.compile()
        self.all_me = self.generate_all.all_me
        self.all_f2py = {}
        self.all_amp = {}
        self.all_nhel = {}
        self.all_jamp = {}
        self.all_inter = {}
        self.all_density = {}
        self.all_matrix = {}
        time_me_generation = time.time() - time_me_generation
        logger.info(f"Time ME generation: {time_me_generation:.2f} sec")         
	
	    #4. determine the maxwgt
        #print(f"Spyros decay file: {evt_decayfile}")
        # Sequential mode tests one decaying particle at a time and so needs one
        # bound per slot of the ordering; it falls back to the joint bound if the
        # probe cannot produce them (nothing decays, too few events to spread).
        sequential = self._sequential_active(density_method)
        maxwgts = []
        if sequential:
            maxwgts = self.get_sequential_maxwgt(orig_lhe, evt_decayfile)
            if not maxwgts:
                logger.info("MadSpin: no per-particle maximum weight could be "
                            "estimated, using the joint accept/reject")
                sequential = False
        maxwgt = None
        if not sequential:
            maxwgt = self.get_maxwgt_for_onshell(orig_lhe, evt_decayfile, decay_dict)

        #5. generate the decay (for each production event)
        # The per-event unweighting loop is embarrassingly parallel (events are
        # independent). It is factored into ``_unweight_range`` so it can run
        # either in-process (nb_core==1, byte-for-byte the historical path) or
        # across ``nb_core`` forked worker processes. Parallelism is process
        # level, NOT threads: the matrix-element f2py extension carries global
        # Fortran COMMON-block state and is not thread-safe; after ``fork`` each
        # worker owns an independent address-space copy of it.
        orig_lhe.seek(0)
        base_out = orig_lhe.name.replace('.lhe', '_decayed.lhe')
        # nb_event for the decay-pool refill sizing is the banner-declared count
        # (historical behaviour), not the physical number of events on disk.
        nb_event = orig_lhe.get_banner().run_card['nevents']

        nb_core = self._resolve_nb_core()
        nb_core = max(1, min(nb_core, int(nb_event) if nb_event else 1))

        ctx = dict(
            maxwgt=maxwgt,
            maxwgts=maxwgts,
            sequential=sequential,
            decay_dict=decay_dict,
            drop_prob_per_pdg=drop_prob_per_pdg,
            mixed_pdgs_set=mixed_pdgs_set,
            density_method=density_method,
            density_pole_approximation=density_pole_approximation,
            density_needs_reshuffle=density_needs_reshuffle,
            branching_ratio=self.branching_ratio,
            base_seed=int(self.seed) if self.seed else random.randint(0, 30081*30081),
        )

        start = time.time()
        logger.info("Start generating decays")
        if nb_core == 1:
            output_lhe = lhe_parser.EventFile(base_out, 'w')
            if self.options['fixed_order']:
                output_lhe.eventgroup = True
                orig_lhe.eventgroup = True
            self.banner.scale_init_cross(self.branching_ratio)
            self.banner.write(output_lhe, close_tag=False)
            self.efficiency = 1.
            ctx['shard_nb_event'] = nb_event
            stats = self._unweight_range(orig_lhe, evt_decayfile, output_lhe, ctx)
            output_lhe.write('</LesHouchesEvents>\n')
            try:
                output_lhe.close()
            except Exception:
                pass
            self._apply_accounting(base_out, [stats])
        else:
            logger.info("MadSpin: unweighting %s events on %s cores", nb_event, nb_core)
            self._run_onshell_parallel(orig_lhe, nb_event, nb_core,
                                       evt_decayfile, base_out, ctx)
        logger.info(f"Time for decay = {time.time()-start:.2f} sec")

    def _resolve_nb_core(self):
        """Number of worker processes for the parallel unweighting / gridpack
        decay generation. A madspin-card ``set nb_core N`` takes precedence;
        otherwise fall back to the global MG5 ``nb_core``. Non-positive / unset /
        unparseable => serial (1)."""
        candidates = []
        try:
            candidates.append(self.options['nb_core'])
        except Exception:
            pass
        try:
            candidates.append(self.mg5cmd.options['nb_core'])
        except Exception:
            pass
        for source in candidates:
            try:
                n = int(source)
            except (TypeError, ValueError):
                continue
            if n >= 1:
                return n
        return 1

    def _gridpack_env(self):
        """Environment for run.sh / gridrun subprocesses. The gridpack scripts
        start with ``#!/usr/bin/env python3``, which otherwise resolves via PATH
        to whatever ``python3`` comes first -- often NOT the interpreter running
        MadSpin, and thus one missing modules that gridrun needs (e.g. ``six``,
        whose absence gridrun treats as fatal). Guarantee the same interpreter by
        putting a ``python3`` -> sys.executable shim first on PATH. Also expose
        ``six`` explicitly if this interpreter has it, in case it lives outside
        the default site-packages."""
        env = os.environ.copy()
        # 1. python3 shim so `env python3` == the MadSpin interpreter, regardless
        #    of whether dirname(sys.executable) even contains a bare `python3`.
        if not getattr(self, '_py3_shim_dir', None):
            import tempfile
            import atexit
            shim = tempfile.mkdtemp(prefix='ms_py3shim_')
            atexit.register(shutil.rmtree, shim, ignore_errors=True)
            link = pjoin(shim, 'python3')
            target = os.path.abspath(sys.executable)
            try:
                os.symlink(target, link)
            except (OSError, NotImplementedError, AttributeError):
                with open(link, 'w') as f:
                    f.write('#!/bin/sh\nexec "%s" "$@"\n' % target)
                os.chmod(link, 0o755)
            self._py3_shim_dir = shim
        env['PATH'] = self._py3_shim_dir + os.pathsep + env.get('PATH', '')
        # 2. belt-and-suspenders: if we can import six here, make sure the
        #    subprocess can find it too.
        try:
            import six as _six
            sixdir = os.path.dirname(os.path.abspath(_six.__file__))
            env['PYTHONPATH'] = sixdir + os.pathsep + env.get('PYTHONPATH', '')
        except Exception:
            pass
        return env

    def _run_gridpack(self, cmd, cwd):
        """Run a gridpack run.sh, capturing its output (the tail) so a generation
        failure can be reported with the actual gridrun output rather than an
        opaque missing-file error. The output is only echoed at DEBUG level, so
        the (very verbose) per-job gridrun progress does not spam the log."""
        import subprocess
        import collections as _collections
        proc = subprocess.Popen(cmd, cwd=cwd, env=self._gridpack_env(),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                bufsize=1, universal_newlines=True)
        tail = _collections.deque(maxlen=80)
        for line in proc.stdout:
            line = line.rstrip('\n')
            tail.append(line)
            logger.debug("[gridpack] %s", line)
        proc.wait()
        return proc.returncode, '\n'.join(tail)

    def _decay_pool_split(self):
        """In how many files the decay pool should be written: one per parallel
        unweighting worker (1 = single pool, historical behaviour)."""
        return self._resolve_nb_core()

    def _sequential_pool_ladder(self, to_decay, nb_event, density_method):
        """pdg -> how many decay events one production event burns on it, when
        the accept/reject is done one particle at a time. Empty when it is not.

        Each slot is redrawn until accepted, so it burns 1/eff_k events from its
        own pool, and eff_k drops along the ordering (_decay_pool_ladder). The
        pools are per pdg while the ladder is per slot: identical parents share
        a pool and take consecutive slots, so charge that pdg the largest of
        them.
        """
        if not self._sequential_active(density_method):
            return {}
        spins = {}
        for pdg in to_decay:
            try:
                spins[pdg] = self.model.get_particle(pdg).get('spin')
            except Exception:
                return {}
        preference = self._sequential_spin_order()
        def rank(pdg):
            spin = spins[pdg]
            return (preference.index(spin) if spin in preference
                    else len(preference), abs(pdg), pdg)
        ladder = {}
        position = 0
        for pdg in sorted(to_decay, key=rank):
            multiplicity = 1
            if nb_event:
                multiplicity = max(1, int(to_decay[pdg]) // int(nb_event))
            ladder[pdg] = self._decay_pool_ladder(position + multiplicity - 1,
                                                  spins[pdg])
            position += multiplicity
        return ladder

    def _log_once(self, key, message, *args):
        """Log a resolution message the first time only: these are decided per
        production event but say something about the run."""
        seen = getattr(self, '_logged_once', None)
        if seen is None:
            seen = self._logged_once = set()
        if key not in seen:
            seen.add(key)
            logger.info(message, *args)

    def _unweighting_mode(self, density_method=True):
        """Which accept/reject scheme this run uses: one of 'joint',
        'two_stage', 'sequential', 'sequential_global_retry',
        'sequential_with_mass'.

        All of them sample the same distribution; they differ in how the test is
        split and in what a rejection redraws.

          joint                    one test over the virtualities and every
                                   decay at once -- the historical scheme, and
                                   the only one available outside density mode.
          two_stage                the set of virtualities is unweighted first,
                                   then every decay against a single bound; a
                                   rejection redraws the decays only, so the
                                   production reshuffling and its density matrix
                                   are reused across the retries.
          sequential               as two_stage, but one test per decaying
                                   particle, redrawing only the particle that
                                   was rejected.
          sequential_global_retry  as sequential, but a rejected decay redraws
                                   the virtualities as well.
          sequential_with_mass     one test per decaying particle, with that
                                   particle's virtuality drawn *inside* its own
                                   accept/reject rather than up front.

        The first four share an up-front mass draw and so a mass-set stage;
        ``sequential_with_mass`` is the odd one out and not a variant of
        ``sequential``: the mass is drawn and redrawn together with that slot's
        angles, so nothing is ever frozen, no stage has a conditional
        normalisation to divide out, and the tabulated running-width factor the
        two-stage schemes need does not arise. It is the historical PA scheme.
        It needs a per-particle mass draw, i.e. the PA spinmode; the offshell
        spinmodes reshuffle the whole production onto the mass set at once, so
        there they fall back to ``sequential``.

        ``auto`` has two branches, one per spinmode family. They were measured
        over the number of decaying particles n on `p p > w+ j` (n=1),
        `p p > t t~` (2), `p p > t t~ z` (3) and `p p > t t~ t t~` (4), 50000
        events each -- see MADSPIN_SEQUENTIAL_PLAN.md section 12.

        **PA/onshell -> ``sequential``, at every n.** It was the fastest of the
        three at all four multiplicities, by 1.2x at n=1 rising to 3.8x at n=4.
        The joint test's cost grows as n x (trials per event), since one
        rejection throws every decay away, while the per-particle one's grows
        far more slowly; and the up-front mass draw evaluates the production
        reshuffling jacobian once per mass set instead of once per slot trial.
        Even at n=1, where the angle stage degenerates to the joint test, the
        mass stage still pays for itself: a mass set can be rejected before any
        decay is drawn.

        **madspin/full -> ``joint`` up to two decaying particles, then
        ``sequential``.** Offshell, each mass set costs a production reshuffle
        *and* an offshell production density, which the joint test pays per
        trial but which a staged scheme pays per mass set -- and below n=3 there
        are not enough decays to save to cover it. At n=1 it is worse than that:
        the mass-set weight carries ``Tr(rho_off)/|M_prod|^2_on``, and when the
        single decaying particle carries most of the production matrix
        element's virtuality dependence (`p p > w+ j`) that ratio spans orders
        of magnitude, no bound covers it, and the mass stage needs ~790 sets per
        accepted event. From n=3 the per-particle test wins by 2.2x and 4.3x.

        ``two_stage`` is not the fastest scheme at any measured point -- joint
        beats it at n<=2 and ``sequential`` at n>=3 -- so it is reachable but
        never chosen here. It stays useful as a cross-check, being the one
        staged scheme whose angle stage is a single joint test.

        ``fixed_order`` forces joint: its counter-events ride along with the
        decays and have not been thought through here.
        """
        if not density_method:
            return 'joint'
        asked = mode = self.options['unweighting']
        if mode == 'auto':
            nb_decaying = getattr(self, '_nb_decaying', 2)
            if self.options['spinmode'] in ['PA', 'onshell']:
                # fastest at every multiplicity measured; rho is fixed on shell
                # so the mass stage costs a reshuffling jacobian and nothing else
                mode = 'sequential'
            elif nb_decaying <= 2:
                # offshell a mass set costs a production reshuffle and a
                # production density, and there are not yet enough decays to
                # save to pay for it
                mode = 'joint'
            else:
                mode = 'sequential'
        if mode == 'joint':
            return self._announce_mode('joint', asked)
        if self.options['fixed_order']:
            self._log_once('fixed_order',
                           "MadSpin: fixed_order is on, keeping the joint "
                           "accept/reject (unweighting ignored)")
            return self._announce_mode('joint', asked)
        if getattr(self, '_decay_groups', None):
            # The per-particle test redraws one slot until it is accepted, which
            # divides E[w_k | group] out of the chain -- and that expectation
            # differs between groups, so the accepted group fractions would come
            # out distorted by its reciprocal. The joint test redraws the whole
            # set, group included, so the group stays part of what is being
            # unweighted. Lifting this needs a bound and a rate factor per
            # group; see doc/madspin_decay_groups.md.
            #
            # two_stage self-normalises the same way (it redraws the angle set
            # to acceptance), so it falls back too. Whether
            # sequential_global_retry could be exempted -- a rejection there
            # throws the whole chain away rather than renormalising -- depends
            # on whether the group is redrawn with it, and has not been checked.
            self._log_once('decay_groups',
                           "MadSpin: the decay lines are grouped ('@' tags), "
                           "keeping the joint accept/reject "
                           "(unweighting ignored)")
            return self._announce_mode('joint', asked)
        if self.options['spinmode'] not in ['PA', 'onshell', 'madspin', 'full']:
            self._log_once('spinmode',
                           "MadSpin: spinmode=%s keeps the joint accept/reject "
                           "(unweighting ignored)", self.options['spinmode'])
            return self._announce_mode('joint', asked)
        if (mode == 'sequential_with_mass'
                and self.options['spinmode'] not in ['PA', 'onshell']):
            self._log_once('with_mass_pa_only',
                           "MadSpin: unweighting=sequential_with_mass needs a "
                           "per-particle mass draw, which the offshell "
                           "spinmodes do not have (they reshuffle the whole "
                           "production onto the mass set at once); using "
                           "sequential instead")
            return self._announce_mode('sequential', asked)
        return self._announce_mode(mode, asked)

    def _announce_mode(self, mode, asked):
        """Say once which scheme the run uses. Worth a line because the card no
        longer answers it: 'auto' resolves on the process."""
        self._log_once('mode', "MadSpin: unweighting = %s (%s)", mode,
                       'auto, %s decaying particle(s)'
                       % getattr(self, '_nb_decaying', '?')
                       if asked == 'auto' else 'set explicitly')
        return mode

    def _sequential_active(self, density_method):
        """Whether any of the per-particle / two-stage schemes is in use, i.e.
        anything but the historical joint accept/reject."""
        return self._unweighting_mode(density_method) != 'joint'

    def _sequential_spin_order(self):
        """The spin order (MG5 2S+1 convention) driving which particle is
        accept/rejected first. Unlisted spins go last, in their natural slot
        order."""
        try:
            order = [int(x) for x in
                     str(self.options['sequential_spin_order']).replace(',', ' ').split()]
        except (ValueError, TypeError):
            order = []
        return order or [2, 3, 1]

    def _decay_slot_order(self, decaying_spins):
        """Order in which the slots are accept/rejected.

        Sorted by ``sequential_spin_order`` (default: fermions, then vectors,
        then scalars), ties broken by slot index so a run stays reproducible and
        independent of dict ordering. Only decides *which slot is filled next* --
        the tensor product itself must stay in slot order, see
        MADSPIN_SEQUENTIAL_PLAN.md."""
        pref = self._sequential_spin_order()
        def key(slot):
            spin = decaying_spins[slot]
            rank = pref.index(spin) if spin in pref else len(pref)
            return (rank, slot)
        return sorted(range(len(decaying_spins)), key=key)

    @staticmethod
    def _decay_pool_ladder(position, spin):
        """Expected number of decay events one production event burns on the
        slot sitting at ``position`` of the accept/reject ordering.

        Each slot is redrawn until accepted, so it burns 1/eff_k events from its
        own pool. The first slot sees a production density matrix traced over
        everything else -- close to unpolarised, so mild modulation and a high
        acceptance; each subsequent one sees a more conditioned, more polarised
        parent, hence a wider weight spread and a lower acceptance. Hence the
        ladder 1.5, 2, 2.5, 3, ...

        A spin-0 parent (MG5 spin==1) has a 1x1 decay density matrix, so its
        ratio is identically 1: it can never be rejected and burns exactly one
        event wherever it sits. Charging it the ladder would just generate
        decays nobody consumes."""
        if spin == 1:
            return 1.1
        return 1.5 + 0.5 * position

    @staticmethod
    def _decay_dir(path_me, pdg, decay_file_nb):
        return pjoin(path_me,
                     "decay_%s_%s" % (str(pdg).replace("-", "x"), decay_file_nb))

    def _regenerate_events(self, pdg, decay_file_nb, needed, run_name):
        """Produce ``needed`` extra decay events for a channel that has already
        been generated once, into a run of its own. Returns the reader over the
        events produced (already split per worker when running in parallel)."""
        decay_dir = self._decay_dir(self.path_me, pdg, decay_file_nb)
        # Never let the generation land on an existing run: madevent then asks
        # "do you wish to overwrite?", the non-interactive answer is 'n', and it
        # silently produces nothing -- which used to make every worker retry the
        # same doomed generation in turn.
        stale = pjoin(decay_dir, 'Events', run_name)
        if os.path.exists(stale):
            _force_rmtree(stale)
        # RunWeb is madevent's "this directory is busy" marker and makes
        # MadEventCmd refuse to start (AlreadyRunning). The generation that
        # created the pool does not always clean it up. The caller holds the
        # exclusive refill lock, so no other madevent can be running here: any
        # RunWeb we find is stale and removing it is safe.
        runweb = pjoin(decay_dir, 'RunWeb')
        if os.path.exists(runweb):
            logger.debug("removing stale RunWeb in %s before the refill", decay_dir)
            try:
                os.remove(runweb)
            except OSError:
                pass
        with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"],
                             [50, 50, 50, 50]):
            out = self.generate_events(pdg, needed, self.mg5cmd,
                                       [decay_file_nb], run_name=run_name)
        reader = out[decay_file_nb]
        if not os.path.exists(reader.name):
            raise Exception(
                "MadSpin: decay-event refill for pdg %s produced no events "
                "(expected %s)." % (pdg, reader.name))
        return reader

    @staticmethod
    def _reader_paths(reader):
        """Every file behind a decay-pool reader (one per worker when split)."""
        return list(getattr(reader, 'paths', None) or [reader.name])

    @staticmethod
    def _reader_from_paths(paths):
        """Rebuild a decay-pool reader from the paths marshalled across a fork."""
        if len(paths) > 1:
            return _ChainedEvents(paths)
        return lhe_parser.EventFile(paths[0])

    def _generate_decay_entry(self, pdg, job, nb_core, seed_offset, res_path):
        """Generate the decay events of one particle inside a forked process
        (see :meth:`_generate_decays`). The produced EventFile objects cannot
        cross the process boundary, so report their paths (and the partial
        width) through a small JSON file."""
        import json
        try:
            # cap the cores this generation may use, use a seed of our own, and
            # never reuse a MadEventCmdShell inherited through fork
            # multiprocessing reseeds Python's global RNG from OS entropy in
            # every forked child (BaseProcess._bootstrap), so a child that does
            # not re-seed it deterministically is NOT reproducible. That matters
            # here because the tail of the generation is pure Python
            # accept/reject on that RNG (combine_runs.copy_events and
            # EventFile.unweight): left unseeded, the very same iseed gives a
            # decay pool of a different size and content on every run, and the
            # decayed sample diverges with it. Key it on the same offset as the
            # madevent seed so the two particles keep distinct streams, with a
            # multiplier of its own so a generation child can never land on the
            # stream of an unweighting/max-weight worker (those use 7919).
            random.seed((int(self.seed) if self.seed else 0) + 104729 * seed_offset)
            self._gen_nb_core = nb_core
            self.seed = (int(self.seed) + 1000003 * seed_offset) % (30081 * 30081)
            self.options['seed'] = self.seed
            self.me_int = {}
            out, width, channel_widths = self.generate_events(
                pdg, job['nb_gen'], self.mg5cmd,
                cumul=job['cumul'], output_width=True)
            with open(res_path, 'w') as fp:
                # send back every file of each channel: when the pool is split
                # per worker (nb_unweight_output) reporting only the first one
                # would silently shrink the pool to a single slice.
                json.dump({'files': dict((str(k), self._reader_paths(v))
                                         for k, v in out.items()),
                           'width': width,
                           'channel_widths': dict((str(k), v) for k, v
                                                  in channel_widths.items())},
                          fp)
        except Exception as exc:
            import traceback
            try:
                with open(res_path, 'w') as fp:
                    json.dump({'error': str(exc), 'tb': traceback.format_exc()}, fp)
            except Exception:
                pass

    def _generate_decays(self, gen_jobs, mg5):
        """Generate the decay events for every decaying particle; returns
        ``{pdg: ({file_nb: EventFile}, partial_width, {file_nb: width})}``.

        The generations are independent (each particle has its own
        ``decay_<pdg>_<i>`` directory) and a single MadEvent generation only
        keeps every core busy for a short while before tailing off to a single
        job, so run them concurrently -- one forked process per particle -- and
        split the core budget between them so their *sum* never oversubscribes
        the machine.
        """
        if len(gen_jobs) <= 1:
            return dict((pdg, self.generate_events(pdg, job['nb_gen'], mg5,
                                                   cumul=job['cumul'],
                                                   output_width=True))
                        for pdg, job in gen_jobs.items())

        import multiprocessing as mp
        import json
        budget = self._resolve_nb_core()
        try:
            budget = min(budget, mp.cpu_count())
        except NotImplementedError:
            pass
        per = max(1, 2 * budget // len(gen_jobs))
        logger.info("MadSpin: generating the decay events of %s particles at "
                    "once, %s core(s) each", len(gen_jobs), per)

        mpctx = mp.get_context('fork')
        procs = []
        for offset, (pdg, job) in enumerate(gen_jobs.items()):
            res = pjoin(self.path_me, 'ms_gen_%s.json' % str(pdg).replace('-', 'x'))
            p = mpctx.Process(target=self._generate_decay_entry,
                              args=(pdg, job, per, offset + 1, res))
            p.start()
            procs.append((pdg, p, res))
        for pdg, p, res in procs:
            p.join()

        out = {}
        for pdg, p, res in procs:
            if not os.path.exists(res):
                raise Exception("MadSpin: the decay generation of pdg %s produced "
                                "no result (crashed?)." % pdg)
            with open(res) as fp:
                data = json.load(fp)
            os.remove(res)
            if 'error' in data:
                raise Exception("MadSpin: the decay generation of pdg %s failed:\n%s"
                                % (pdg, data.get('tb', data['error'])))
            out[pdg] = (dict((int(k), self._reader_from_paths(v))
                             for k, v in data['files'].items()),
                        data['width'],
                        dict((int(k), v) for k, v
                             in data.get('channel_widths', {}).items()))
        return out

    def _refill_pool_path(self, decay_dir, gen):
        """This worker's own file of the refill pool ``gen``. The refill asks the
        unweighting for one file per worker, so a worker never reads (nor even
        parses) the events that belong to the others."""
        base = pjoin(decay_dir, 'Events', 'ms_refill_%d' % gen,
                     'unweighted_events.lhe')
        paths = lhe_parser.EventFile.unweight_output_paths(base, self._shard_nb_core)
        path = paths[self._shard_tag] if len(paths) > 1 else paths[0]
        if not os.path.exists(path):
            raise Exception("MadSpin: refill pool %s is missing for worker %s"
                            % (path, self._shard_tag))
        return path

    @staticmethod
    def _count_lhe_events(path):
        """Number of ``<event`` records in an LHE file (plain or gzip). Used to
        size the owner's ~10% shortfall -- a cheap line scan, no parsing."""
        try:
            if path.endswith('.gz'):
                import gzip
                opener = lambda p: gzip.open(p, 'rt', errors='ignore')
            else:
                opener = lambda p: open(p, 'r', errors='ignore')
            n = 0
            with opener(path) as fh:
                for line in fh:
                    if '<event' in line:
                        n += 1
            return n
        except (IOError, OSError):
            return 0

    def _channel_owner(self, pdg, decay_file_nb):
        """The single worker that is the sole (re)generator of this decay
        channel. Channels are numbered in a fixed sorted order and dealt out to
        the workers round-robin, so the owner is a deterministic function of the
        channel alone -- never of run-time timing. Fixing the generator this way
        is what makes the refilled pool (and thus the decayed sample)
        reproducible: contrast the old "whoever ran out first generates", whose
        winner was a lock race."""
        keys = getattr(self, '_channel_keys', None) or []
        try:
            idx = keys.index((pdg, decay_file_nb))
        except ValueError:
            idx = 0
        return idx % max(1, int(self._shard_nb_core))

    def _open_refill_slice(self, decay_dir, gen, owner):
        """This worker's reader over refill pool ``gen``; the owner's slice is
        cut ~10% short (see _LimitedEvents) so the owner runs out first."""
        path = self._refill_pool_path(decay_dir, gen)
        reader = lhe_parser.EventFile(path)
        if self._shard_tag == owner:
            frac = float(getattr(self, '_owner_undersize', 0.10) or 0.0)
            if frac > 0:
                n = self._count_lhe_events(path)
                reader = _LimitedEvents(reader,
                                        int(math.floor((1.0 - frac) * n)))
        return reader

    @staticmethod
    def _published_gen(decay_dir):
        """Highest refill generation published on disk for this channel (0 if
        none). The single source of truth both the owner and the waiters read."""
        gen_file = pjoin(decay_dir, 'ms_refill.gen')
        if not os.path.exists(gen_file):
            return 0
        try:
            return int(open(gen_file).read().strip())
        except (ValueError, IOError):
            return 0

    def _owner_generate(self, pdg, decay_file_nb, target_gen, needed):
        """Generate this channel's pool up to ``target_gen`` under the channel
        lock, then publish the gen counter. Idempotent: if the gen is already on
        disk the while-loop is skipped.

        The madevent seed is keyed on the channel and the gen number with a base
        shared by every worker, so gen N gets the SAME seed no matter who
        generates it. The size follows the caller's efficiency-based ``needed``.
        In normal operation only the fixed owner calls this (its ``needed`` is
        deterministic, so the pool is reproducible); the deadlock fail-safe in
        _worker_refill may also call it from a non-owner, whose ``needed`` differs
        -- that (rare) refill is intentionally not guaranteed reproducible."""
        import fcntl
        decay_dir = self._decay_dir(self.path_me, pdg, decay_file_nb)
        gen_file = pjoin(decay_dir, 'ms_refill.gen')
        with open(pjoin(decay_dir, 'ms_refill.lock'), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                current = self._published_gen(decay_dir)
                while current < target_gen:
                    new_gen = current + 1
                    seed_base = getattr(self, '_refill_seed_base', None)
                    if seed_base is None:
                        seed_base = int(self.seed) if self.seed else 0
                    # sign-aware pdg term so a particle and its antiparticle
                    # channel do not collide onto the same iseed
                    det_seed = 1 + ((int(seed_base)
                                     + 1000003 * ((int(pdg) % 998244353) + 1)
                                     + 101 * (int(decay_file_nb) + 1)
                                     + 100003 * new_gen) % (30081 * 30081))
                    self.seed = det_seed
                    self.options['seed'] = det_seed
                    det_needed = min(200000, max(1000, int(math.ceil(
                        needed / float(self._shard_nb_core))))) \
                        * self._shard_nb_core
                    logger.info("MadSpin worker %s OWNS pdg %s: generating gen %s "
                                "(%s events, seed %s) for all %s workers",
                                self._shard_tag, pdg, new_gen, det_needed,
                                det_seed, self._shard_nb_core)
                    # The generation runs a full madevent IN THIS PROCESS and
                    # reseeds / consumes Python's global ``random`` (combine +
                    # unweight). Snapshot and restore it so the generation never
                    # perturbs this worker's accept/reject stream. Use a *fresh*
                    # MadEventCmdShell (never the fork-inherited one); it writes a
                    # run of its own and splits one file per worker.
                    rng_state = random.getstate()
                    self.me_int = {}
                    stag, self._shard_tag = self._shard_tag, None
                    try:
                        self._regenerate_events(pdg, decay_file_nb, det_needed,
                                                'ms_refill_%d' % new_gen)
                    finally:
                        self._shard_tag = stag
                        random.setstate(rng_state)
                    # publish only once every file is complete on disk
                    with open(gen_file, 'w') as fp:
                        fp.write('%d\n' % new_gen)
                    current = new_gen
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    # ---- cross-process worker status, for the deadlock fail-safe -------------
    # Each forked worker publishes a one-line status file others can read:
    #   'R'          running (making progress)
    #   'G'          generating a decay pool
    #   'W <id>'     blocked, waiting for worker <id> to generate a pool
    # so a blocked owner can walk the wait-for chain and spot a cycle.
    def _status_path(self, worker_id):
        return pjoin(self.path_me, 'ms_wstatus_%d' % worker_id)

    def _clear_worker_status(self, nb_core):
        """Remove stale per-worker status files before forking a phase, so a
        'D'(one) left by the previous phase's worker of the same id can't be
        misread as this phase's worker being done. Called by the parent."""
        for wid in range(int(nb_core)):
            try:
                os.remove(self._status_path(wid))
            except OSError:
                pass

    def _set_status(self, state, target=None):
        tag = getattr(self, '_shard_tag', None)
        if tag is None:
            return
        try:
            with open(self._status_path(tag), 'w') as f:
                f.write(state if target is None else '%s %d' % (state, target))
        except (IOError, OSError):
            pass

    def _read_worker_status(self, worker_id):
        """(state, [target]) tuple for ``worker_id``, or None if unreadable."""
        try:
            parts = open(self._status_path(worker_id)).read().split()
        except (IOError, OSError):
            return None
        if not parts:
            return None
        if parts[0] == 'W' and len(parts) >= 2:
            try:
                return ('W', int(parts[1]))
            except ValueError:
                return None
        return (parts[0],)

    def _wait_cycle_to_self(self, first_target):
        """True if the wait-for chain that starts at ``first_target`` leads back
        to this worker -- a deadlock cycle only I can break. A worker on the path
        that is running or generating (not 'W') means the chain is making
        progress, so there is no deadlock through it."""
        cur = first_target
        for _ in range(int(self._shard_nb_core) + 1):
            if cur == self._shard_tag:
                return True
            st = self._read_worker_status(cur)
            if not st or st[0] != 'W':
                return False
            cur = st[1]
        return False

    def _worker_refill(self, pdg, decay_file_nb, needed):
        """Owner-based decay-event refill for the forked workers. Returns the
        reader this worker should continue from.

        Each channel has one deterministic OWNER worker (:meth:`_channel_owner`).
        In normal operation only the owner (re)generates that channel's pool;
        any other worker that runs out BLOCKS until the owner has published the
        generation it needs, then opens its own slice. Because the generator is
        fixed rather than "whoever ran out first", the regenerated pool -- and
        hence the decayed sample -- is identical from one run to the next. The
        owner's slice is deliberately ~10% short (:meth:`_open_refill_slice`) so
        it reaches the refill point first and the others rarely wait.

        Deadlock fail-safe: while blocked, a worker publishes that it is waiting
        for the owner and walks the wait-for chain (:meth:`_wait_cycle_to_self`).
        If the chain loops back to itself -- a circular wait no owner can clear
        on its own -- the worker generates the channel itself to break it. That
        abandons the fixed-owner rule for this one refill, so its size (and thus
        the sample) is not guaranteed reproducible -- an accepted trade to avoid
        a hang. Generation runs on all cores."""
        decay_dir = self._decay_dir(self.path_me, pdg, decay_file_nb)
        key = (pdg, decay_file_nb)
        my_gen = self._pool_gen.get(key, 0)
        target_gen = my_gen + 1
        owner = self._channel_owner(pdg, decay_file_nb)

        if self._shard_tag == owner:
            self._set_status('G')
            try:
                self._owner_generate(pdg, decay_file_nb, target_gen, needed)
            finally:
                self._set_status('R')
            self._pool_gen[key] = target_gen
            return self._open_refill_slice(decay_dir, target_gen, owner)

        # Not my channel: wait for the owner, advertising who I wait for so the
        # deadlock detection can see me. A cycle back to myself has to persist a
        # few consecutive checks (statuses update asynchronously) before I act,
        # to avoid tripping on a transient state.
        timeout = float(os.environ.get('MADSPIN_REFILL_WAIT', '3600'))
        try:
            need_hits = int(os.environ.get('MADSPIN_DEADLOCK_HITS', '10'))
        except (TypeError, ValueError):
            need_hits = 10
        self._set_status('W', owner)
        waited = 0.0
        cycle_hits = 0
        try:
            while self._published_gen(decay_dir) < target_gen:
                reason = None
                if self._read_worker_status(owner) == ('D',):
                    # The owner has finished (or died) without producing this
                    # generation and never will -- e.g. in the max-weight scan an
                    # owner may exhaust its short probe slice before it ever runs
                    # its owned channel dry. I must generate it myself.
                    reason = ("owner %s is DONE but never produced" % owner)
                elif self._wait_cycle_to_self(owner):
                    cycle_hits += 1
                    if cycle_hits >= need_hits:
                        reason = ("deadlock wait-cycle (chain from owner %s "
                                  "loops back to me)" % owner)
                else:
                    cycle_hits = 0
                if reason is not None:
                    # Fail-safe: break the stall by generating the channel here,
                    # abandoning the fixed-owner rule. The size is this worker's
                    # own (efficiency-based), so this one refill's reproducibility
                    # is not guaranteed -- an accepted trade against a hang.
                    logger.warning(
                        "MadSpin worker %s: %s; generating gen %s of pdg %s "
                        "(decay file %s) myself. Seed reproducibility is NOT "
                        "guaranteed for this refill.", self._shard_tag, reason,
                        target_gen, pdg, decay_file_nb)
                    self._set_status('G')
                    self._owner_generate(pdg, decay_file_nb, target_gen, needed)
                    break
                time.sleep(0.1)
                waited += 0.1
                if waited > timeout:
                    raise Exception(
                        "MadSpin worker %s waited %.0fs for owner worker %s to "
                        "generate gen %s of channel pdg %s (decay file %s) and "
                        "gave up. Raise the owner undersize fraction "
                        "(MADSPIN_OWNER_UNDERSIZE) or lower nb_core."
                        % (self._shard_tag, waited, owner, target_gen, pdg,
                           decay_file_nb))
        finally:
            self._set_status('R')
        self._pool_gen[key] = target_gen
        return self._open_refill_slice(decay_dir, target_gen, owner)

    def _unweight_range(self, prod_source, evt_decayfile, output_lhe, ctx):
        """Decay + accept/reject over every production event in ``prod_source``,
        writing accepted events to the open ``output_lhe`` (no banner, no closing
        tag). Returns a small picklable stats dict.

        This is the body of the onshell unweighting loop, formerly inline in
        ``run_onshell``. It is called directly for nb_core==1 and once per shard
        inside each forked worker for nb_core>1. It only touches its arguments
        plus per-instance state that is private after ``fork`` (``self.efficiency``,
        ``self.branching_ratio``, the RNG, and the f2py module)."""
        maxwgt = ctx['maxwgt']
        maxwgts = ctx.get('maxwgts') or []
        sequential = ctx.get('sequential', False)
        decay_dict = ctx['decay_dict']
        drop_prob_per_pdg = ctx['drop_prob_per_pdg']
        mixed_pdgs_set = ctx['mixed_pdgs_set']
        density_method = ctx['density_method']
        density_pole_approximation = ctx['density_pole_approximation']
        density_needs_reshuffle = ctx['density_needs_reshuffle']
        nb_event = ctx['shard_nb_event']
        fixed_order = self.options['fixed_order']

        nb_try = 0
        nb_loose_skip = 0  # events dropped to equalize BRs (fake-decay path)
        sequential_stats = collections.defaultdict(int)
        curr_event = -1    # guard: an (over-sharded) empty range leaves it unset
        start = time.time()
        for curr_event, production in enumerate(prod_source):
            if fixed_order:
                production, counterevt = production[0], production[1:]
            if (curr_event and curr_event % 10 == 0
                    and float(str(curr_event)[1:]) == 0
                    and getattr(self, '_shard_tag', None) in (None, 0)):
                # only one worker prints progress -- the others would just
                # interleave the same lines
                if sequential and sequential_stats:
                    # per-particle unweighting cost: how many decay events each
                    # decaying particle burned per accepted event. Clearer than a
                    # single number, and it is the sum of these -- and reported
                    # once per event, i.e. gated on the slot-0 pass, not per slot.
                    positions = sorted(int(k.rsplit('_', 1)[1]) for k in
                                       sequential_stats if k.startswith('nb_try_'))
                    per = ' '.join(
                        'p%d=%.2f' % (k, sequential_stats['nb_try_%d' % k]
                                      / float(curr_event + 1)) for k in positions)
                    logger.info("decaying event number %s/%s. Decay events per "
                                "accepted event, per particle: %s [%s s]"
                                % (curr_event, nb_event, per, time.time()-start))
                elif self.efficiency:
                    logger.info("decaying event number %s/%s. Trials per event: "
                                "%.4g [%s s]" % (curr_event, nb_event, 1/self.efficiency,
                                                 time.time()-start))

            # BR-equalization: drop this event with probability
            # 1 - br_pdg / max_br when this production process has a smaller
            # total BR than the largest one in the mixed sample. Done before
            # any matrix-element work so dropped events are cheap.
            if drop_prob_per_pdg:
                evt_mixed_pdgs = [p.pid for p in production
                                  if int(p.status) == 1 and p.pid in mixed_pdgs_set]
                if len(evt_mixed_pdgs) == 1:
                    drop = drop_prob_per_pdg[evt_mixed_pdgs[0]]
                    if drop > 0 and random.random() < drop:
                        nb_loose_skip += 1
                        continue
                elif len(evt_mixed_pdgs) > 1:
                    raise self.InvalidCmd(
                        "BR equalization for events with more than one "
                        "mixed-pdg decaying particle is not implemented yet "
                        "(event %d has pdgs=%s). Please report this case." %
                        (curr_event, evt_mixed_pdgs))

            if sequential:
                # Accept/reject one decaying particle at a time. Every
                # production event yields a set (each slot is redrawn until it
                # is accepted), so there is no outer rejection loop here.
                seq_stats = collections.defaultdict(int)
                decays = self.sequential_accept_reject(
                                production, evt_decayfile, maxwgts,
                                nb_event - curr_event, stats=seq_stats,
                                decay_dict=decay_dict)
                if decays is None:
                    # nothing to decay in this production event
                    output_lhe.write_events(production)
                    continue
                for key, value in seq_stats.items():
                    sequential_stats[key] += value
                nb_try += sum(v for k, v in seq_stats.items()
                              if k.startswith('nb_try_'))
                full_evt = lhe_parser.Event(str(production))
                full_evt = full_evt.add_decays(decays)
                if density_needs_reshuffle:
                    # the decays already carry their sampled virtualities; this
                    # is the single production reshuffling of the chain, which
                    # sequential_accept_reject has already checked is possible
                    full_evt.reshuffle_production()
                self.efficiency = float(curr_event + 1) / nb_try if nb_try else 1.0
                full_evt.wgt *= self.branching_ratio
                wgts = full_evt.parse_reweight()
                for key in wgts:
                    wgts[key] *= self.branching_ratio
                output_lhe.write_events(full_evt)
                continue

            # Per-production-event cache reused across rejection retries.
            prod_density_cached = None

            while 1:
                nb_try += 1
                decays = self.get_decay_from_file(production, evt_decayfile, nb_event-curr_event)
                # In density mode do not do full event construction before accept/reject
                build_event = (not density_method) or self.options['fixed_order']

                # Offshell (madspin/full) density: the reshuffle of the chain
                # happens INSIDE get_onshell_evt_and_wgt (its jacobian is folded
                # into wgt there). It mutates the production event in place, so pass
                # a per-trial onshell copy -- otherwise a rejected trial's offshell
                # kinematics leak into the next trial's denominator ME and reshuffle
                # jacobian (mass_shuffle's chi telescopes, so the kinematics are
                # unchanged, but the jacobian and MEdenom_prod are not).
                offshell_density = density_method and not density_pole_approximation
                prod_trial = lhe_parser.Event(str(production)) if offshell_density else production

                if prod_density_cached is None or not density_pole_approximation:
                    full_evt, wgt, prod_density_cached = self.get_onshell_evt_and_wgt(
                        prod_trial, decays, decay_dict, build_event=build_event)
                else:
                    full_evt, wgt, _ = self.get_onshell_evt_and_wgt(
                        prod_trial, decays, decay_dict, prod_density_cached, build_event=build_event)
                jac = 1
                if (density_needs_reshuffle and not offshell_density
                        and self.options['density_keep_jacobian']):
                    # PA with explicit jacobian tracking: reshuffle BEFORE
                    # accept/reject so the reshuffling jacobian enters the weight
                    # (wgt*jac). Build on a fresh copy because this runs on every
                    # trial, including rejected ones (must not mutate the shared
                    # production event). The offshell/madspin path does NOT enter
                    # here: its reshuffle jacobian is already inside wgt.
                    full_evt = lhe_parser.Event(str(production))
                    full_evt = full_evt.add_decays(decays)
                    jac = full_evt.reshuffle_production()

                if random.random()*maxwgt < wgt*jac:
                    if offshell_density:
                        # prod_trial has already been reshuffled internally (its
                        # jacobian is in wgt); build the event to write out from the
                        # reshuffled copy, without reshuffling a second time. If
                        # get_onshell already built it (fixed_order / density_debug),
                        # reuse that event -- decays were consumed there.
                        if full_evt is None:
                            full_evt = lhe_parser.Event(str(prod_trial))
                            full_evt = full_evt.add_decays(decays)
                    elif (density_needs_reshuffle
                            and density_pole_approximation
                            and not self.options['density_keep_jacobian']):
                        # PA (default): reshuffle AFTER acceptance. The reshuffle is
                        # a kinematic dressing of the accepted event; the Breit-Wigner
                        # sampling jacobian is already folded into wgt, so the
                        # reshuffling jacobian must not re-enter the accept/reject
                        # test. For 2 -> 1 production no mass was sampled and
                        # reshuffle_production short-circuits (NWA-style no-op).
                        full_evt = lhe_parser.Event(str(production))
                        full_evt = full_evt.add_decays(decays)
                        jac = full_evt.reshuffle_production()
                    elif full_evt is None:
                        # No-reshuffle density mode still needs a concrete event to write out.
                        if density_method and density_pole_approximation:
                            full_evt = lhe_parser.Event(str(production))
                        else:
                            full_evt = production
                        full_evt = full_evt.add_decays(decays)
                    if self.options['fixed_order']:
                        full_evt = [full_evt] + [evt.add_decays(decays) for evt in counterevt]
                    break
                #else:
                #    misc.sprint('fail-> retry')
            # Efficiency = accepted / trials (+1 because current event is already accepted)
            self.efficiency = float(curr_event + 1) / nb_try
            #if density_method:
            #    full_evt.reshuffle_production()
            if self.options['fixed_order']:
                for evt in full_evt:
                    # change the weight associated to the event
                    evt.wgt *= self.branching_ratio
                    wgts = evt.parse_reweight()
                    for key in wgts:
                        wgts[key] *= self.branching_ratio
            else:
                # change the weight associated to the event
                full_evt.wgt *= self.branching_ratio
                wgts = full_evt.parse_reweight()
                for key in wgts:
                    wgts[key] *= self.branching_ratio

            output_lhe.write_events(full_evt)

        worker = getattr(self, '_shard_tag', None)
        if worker is None:
            logger.info("decay unweighting done. [%.1f s]" % (time.time()-start))
        else:
            logger.info("worker %s of %s done. [%.1f s]"
                        % (worker, getattr(self, '_shard_nb_core', '?'),
                           time.time()-start))
        n_processed = curr_event + 1
        return dict(n_processed=n_processed,
                    n_written=n_processed - nb_loose_skip,
                    nb_try=nb_try,
                    nb_loose_skip=nb_loose_skip,
                    sequential_stats=dict(sequential_stats))

    def _report_sequential_stats(self, stats_list, n_written):
        """Per-slot report of the sequential accept/reject.

        The per-slot acceptance is what the pool ladder is meant to predict, and
        the overflow count is the safety net: a weight above its slot's maximum
        means the bound was under-estimated, which biases the sample silently,
        so it is reported loudly rather than left in a debug line.
        """
        merged = collections.defaultdict(int)
        for stats in stats_list:
            for key, value in (stats.get('sequential_stats') or {}).items():
                merged[key] += value
        if not merged:
            return
        rejects = merged.get('nb_mass_reject', 0)
        restarts = merged.get('nb_production_restart', 0)
        exact_restarts = merged.get('nb_exact_restart', 0)
        angle_tries = merged.get('nb_angleset_try', 0)
        if angle_tries:
            # two_stage: one bound over all the angles
            logger.info("MadSpin sequential angle stage: %.2f angle sets per "
                        "accepted event (%d drawn, %d rejected)",
                        float(angle_tries) / n_written if n_written
                        else float('inf'), angle_tries,
                        merged.get('nb_angle_reject', 0))
        if rejects or exact_restarts or merged.get("nb_angle_reject", 0):
            # the offshell mass-set stage: how many virtuality sets (each one a
            # production reshuffling and a production density) are drawn per
            # accepted event
            # An angle-set rejection does not cost a mass set: the set is kept
            # and only the decays are drawn again, which is the whole point of
            # two_stage. Counting those here would inflate the
            # production-side work by the angle-stage rejections.
            drawn = rejects + restarts + exact_restarts + n_written
            logger.info("MadSpin sequential mass stage: %.2f mass sets per "
                        "accepted event (%d drawn, %d rejected%s)",
                        float(drawn) / n_written if n_written else float('inf'),
                        drawn, rejects,
                        ', %d dropped by a rejected decay' % exact_restarts
                        if exact_restarts else '')
        positions = sorted(int(k.rsplit('_', 1)[1]) for k in merged
                           if k.startswith('nb_try_'))
        for position in positions:
            tries = merged['nb_try_%d' % position]
            extra = []
            redraws = merged.get('nb_mass_redraw_%d' % position, 0)
            if redraws:
                extra.append('%d mass redraws' % redraws)
            infeasible = merged.get('nb_infeasible_%d' % position, 0)
            if infeasible:
                extra.append('%d decays that could not be reshuffled' % infeasible)
            overflows = merged.get('nb_overflow_%d' % position, 0)
            if overflows:
                extra.append('%d ABOVE the maximum weight' % overflows)
            logger.info(
                "MadSpin sequential slot %d: %.2f decay events per accepted one"
                " (%d drawn)%s", position,
                float(tries) / n_written if n_written else float('inf'), tries,
                (' [%s]' % ', '.join(extra)) if extra else '')
        if restarts:
            logger.info("MadSpin sequential: %d chains restarted on a mass set "
                        "the production could not reshuffle", restarts)
        checks = merged.get('nb_identity_check', 0)
        if checks:
            mean = merged.get('identity_ratio_sum', 0.0) / checks
            variance = (merged.get('identity_ratio_sqsum', 0.0) / checks
                        - mean * mean)
            spread = (math.sqrt(max(0.0, variance)) / abs(mean)
                      if mean else float('inf'))
            # density_tolerance, like density_debug: the two routes evaluate the
            # same matrix elements through different code, and the density
            # matrices are single precision, so agreement is bounded by float32
            # epsilon (~1.2e-7) and not by the physics
            if spread > self.options['density_tolerance']:
                logger.critical(
                    "MadSpin sequential: the weight identity FAILED on %d "
                    "accepted chains -- the chain weight is not proportional "
                    "to the joint weight (relative spread of the ratio %.3g, "
                    "mean %.10g). This scheme is not sampling the joint "
                    "distribution.", checks, spread, mean)
            else:
                logger.info("MadSpin sequential: weight identity verified on "
                            "%d accepted chains -- chain weight / joint weight "
                            "constant to %.3g (ratio %.10g)",
                            checks, spread, mean)
        total_overflow = sum(v for k, v in merged.items()
                             if k.startswith('nb_overflow_'))
        if total_overflow:
            logger.critical(
                "MadSpin sequential: %d weights exceeded their per-particle "
                "maximum. That bound is under-estimated and the sample is "
                "biased: raise nb_sigma or Nevents_for_max_weight, or set "
                "unweighting = joint.", total_overflow)

    def _apply_accounting(self, base_out, stats_list):
        """Post-loop accounting shared by the serial and parallel paths: the
        unweighting-efficiency log, the BR-equalization banner rewrite, and the
        gzip of input+output. ``base_out`` must already be a complete LHE file
        (banner + events + closing tag). Counter sums over ``stats_list`` are
        order-independent, so one shard or many gives the identical result a
        single serial stream would have."""
        n_processed = sum(s['n_processed'] for s in stats_list)
        n_written = sum(s['n_written'] for s in stats_list)
        nb_try = sum(s['nb_try'] for s in stats_list)
        nb_loose_skip = sum(s['nb_loose_skip'] for s in stats_list)

        eff = float(n_written) / nb_try if nb_try else 0.0
        logger.info(
            "MadSpin unweight efficiency: %.4f (%d written / %d trials, %.2f trials/event)",
            eff, n_written, nb_try, (1.0 / eff if eff else float("inf"))
        )
        self._report_sequential_stats(stats_list, n_written)
        if nb_loose_skip > 0:
            # Rewrite the banner with the corrected cross-section so it
            # matches the actual sum of kept-event weights. Each kept event
            # already has wgt = orig_wgt * max_br; we need the banner to read
            # σ * max_br * (n_written / n_processed) ≈ σ * <br>.
            br_correction = float(n_written) / n_processed if n_processed else 1.0
            self._rewrite_lhe_banner_cross(base_out, br_correction,
                                           n_written=n_written)
            self.branching_ratio *= br_correction
            self.cross *= br_correction
            self.error *= br_correction
            logger.info(
                "BR equalization: dropped %d/%d events (effective BR rescale = %.4g).",
                nb_loose_skip, n_processed, br_correction,
            )
            # Downstream sets nb_event = int(original_nb_event * efficiency)
            # so the kept-fraction needs to be communicated as the efficiency.
            self.efficiency = br_correction
        else:
            self.efficiency = 1 # to let me5 to write the correct number of events
        # Re-gzip the input events file (gunzipped at the start of this
        # routine) and the decayed output, matching the legacy MadSpin path
        # so downstream code (banners, crossx.html) finds the *.lhe.gz files
        # it expects.
        try:
            input_evt_path = self.events_file.name
            if input_evt_path.endswith('.lhe') and os.path.exists(input_evt_path):
                misc.gzip(input_evt_path)
        except Exception as exc:
            logger.warning('Could not re-gzip MadSpin input file %s: %s',
                           getattr(self.events_file, 'name', '?'), exc)
        try:
            if base_out.endswith('.lhe') and os.path.exists(base_out):
                misc.gzip(base_out)
        except Exception as exc:
            logger.warning('Could not gzip MadSpin decayed output %s: %s',
                           base_out, exc)
        logger.info('Done so far. output written in %s' % base_out)

    def _split_production(self, orig_lhe, nb_core, base_out):
        """Split the production event file into up to ``nb_core`` contiguous
        shard files (bannerless: the worker's EventFile tolerates a missing
        banner). Returns ``(paths, counts)``. In fixed_order mode each production
        item is an event-group, written back with its ``<eventgroup>`` wrapper so
        the shard round-trips."""
        fixed_order = self.options['fixed_order']
        orig_lhe.seek(0)
        if fixed_order:
            orig_lhe.eventgroup = True
        nb_event = sum(1 for _ in orig_lhe)
        orig_lhe.seek(0)
        if fixed_order:
            orig_lhe.eventgroup = True

        chunk = int(math.ceil(nb_event / float(nb_core))) if nb_event else 0
        paths, counts, shard_files = [], [], []
        for sid in range(nb_core):
            p = '%s.prodshard%d.lhe' % (base_out, sid)
            ef = lhe_parser.EventFile(p, 'w')
            if fixed_order:
                ef.eventgroup = True
            shard_files.append(ef)
            paths.append(p)
            counts.append(0)

        for idx, production in enumerate(orig_lhe):
            sid = min(idx // chunk, nb_core - 1) if chunk else 0
            shard_files[sid].write_events(production)
            counts[sid] += 1
        for ef in shard_files:
            try:
                ef.close()
            except Exception:
                pass

        # keep only non-empty shards (drops trailing shards when nb_core > nb_event)
        keep = [(p, c) for p, c in zip(paths, counts) if c > 0]
        for p, c in zip(paths, counts):
            if c == 0:
                try:
                    os.remove(p)
                except OSError:
                    pass
        return [p for p, _ in keep], [c for _, c in keep]

    def _reopen_decay_pool(self, evt_decayfile, shard_id, nb_core):
        """Return this worker's private view of the decay pools.

        The unweighting already wrote one file per worker
        (``nb_unweight_output``), so a worker simply opens *its* file: no two
        workers ever read the same event and none of them has to scan the whole
        pool. If the pool happens to be a single file (e.g. it was produced
        before this was in place) fall back to striding it, which is correct but
        makes every worker parse everything."""
        local = {}
        for pdg, channels in evt_decayfile.items():
            local[pdg] = {}
            for file_nb, evtfile in channels.items():
                paths = getattr(evtfile, 'paths', None)
                own_file = bool(paths and len(paths) == nb_core)
                if own_file:
                    reader = lhe_parser.EventFile(paths[shard_id])
                elif paths:
                    # split, but not into exactly nb_core files: stride the WHOLE
                    # chained pool. ``evtfile.name`` is only its first file, so
                    # striding that would strand every other file's events.
                    reader = _StridedEvents(_ChainedEvents(paths), shard_id, nb_core)
                else:
                    fresh = lhe_parser.EventFile(evtfile.name)
                    reader = _StridedEvents(fresh, shard_id, nb_core)
                # The worker that OWNS this channel reads its slice ~10% short so
                # it runs out (and, being the sole generator, regenerates) before
                # the others do -- keeping their wait for the refill minimal.
                # Only meaningful on the own-file fast path where the count of
                # this worker's slice is well defined.
                if own_file and self._channel_owner(pdg, file_nb) == shard_id:
                    frac = float(getattr(self, '_owner_undersize', 0.10) or 0.0)
                    if frac > 0:
                        n = self._count_lhe_events(paths[shard_id])
                        reader = _LimitedEvents(reader,
                                                int(math.floor((1.0 - frac) * n)))
                local[pdg][file_nb] = reader
        return local

    def _init_owner_refill(self, evt_decayfile, seed_base):
        """Per-worker set-up for the owner-based refill. Must run in every forked
        worker (unweighting and both max-weight scans) before its decay pools are
        opened, so :meth:`_channel_owner` sees the same channel ordering
        everywhere. Sets: the fixed sorted channel list, the shard-independent
        seed base for the deterministic generation seed, and the owner undersize
        fraction (``MADSPIN_OWNER_UNDERSIZE``, default 0.10). Also publishes the
        initial 'running' worker status for the deadlock detection."""
        self._channel_keys = sorted(
            (pdg, fnb) for pdg, chans in evt_decayfile.items() for fnb in chans)
        self._refill_seed_base = int(seed_base) if seed_base else 0
        try:
            self._owner_undersize = float(
                os.environ.get('MADSPIN_OWNER_UNDERSIZE', '0.10'))
        except (TypeError, ValueError):
            self._owner_undersize = 0.10
        self._set_status('R')

    def _unweight_shard_entry(self, shard_id, nb_core, shard_path, out_path,
                              evt_decayfile, ctx, stats_path):
        """Worker entry point (runs in a forked child process). Owns its RNG, its
        f2py COMMON blocks (independent address space after fork), and its output
        fragment. Writes a JSON stats file the parent reads back; on failure
        writes the traceback there instead of raising into the parent (which only
        sees exit codes)."""
        import json
        try:
            # distinct RNG streams per shard (channel selection + accept/reject)
            random.seed(ctx['base_seed'] + 7919 * (shard_id + 1))
            # marks this process as a forked worker: any decay-event refill must
            # go through the centralised, locked _worker_refill
            self._shard_tag = shard_id
            self._shard_nb_core = nb_core
            self._pool_gen = {}
            self.options['seed'] = (ctx['base_seed'] + 100003 * (shard_id + 1)) % (30081 * 30081)
            self.seed = self.options['seed']
            self.efficiency = 1.0
            self.branching_ratio = ctx['branching_ratio']
            self._init_owner_refill(evt_decayfile, ctx['base_seed'])

            prod = lhe_parser.EventFile(shard_path)
            if self.options['fixed_order']:
                prod.eventgroup = True
            local_pool = self._reopen_decay_pool(evt_decayfile, shard_id, nb_core)

            out = lhe_parser.EventFile(out_path, 'w')
            if self.options['fixed_order']:
                out.eventgroup = True
            stats = self._unweight_range(prod, local_pool, out, ctx)
            try:
                out.close()
            except Exception:
                pass
            with open(stats_path, 'w') as f:
                json.dump(stats, f)
        except Exception as exc:
            import traceback
            try:
                with open(stats_path, 'w') as f:
                    json.dump({'error': str(exc), 'tb': traceback.format_exc()}, f)
            except Exception:
                pass
        finally:
            # tell any worker still blocked on a channel I own that I am gone, so
            # it stops waiting for a generation that will never come and produces
            # it itself (deadlock fail-safe).
            self._set_status('D')

    def _run_onshell_parallel(self, orig_lhe, nb_event, nb_core, evt_decayfile,
                              base_out, ctx):
        """Parallel driver for the unweighting stage: split the production events
        into contiguous shards, fork one worker per shard, then merge the
        fragments under a single banner and apply the global accounting.

        Uses the ``fork`` start method so each worker inherits the fully set-up
        interface (compiled ME dir, model, banner) via copy-on-write memory --
        no pickling of ``self`` -- and gets its own address-space copy of the
        matrix-element COMMON blocks. Results come back through per-shard JSON
        files rather than a Queue to avoid any pickling of worker state."""
        import multiprocessing as mp
        import json

        shard_paths, shard_counts = self._split_production(orig_lhe, nb_core, base_out)
        nb_core = len(shard_paths)
        if nb_core == 0:
            # no events at all: emit a banner-only file
            output_lhe = lhe_parser.EventFile(base_out, 'w')
            self.banner.scale_init_cross(self.branching_ratio)
            self.banner.write(output_lhe, close_tag=False)
            output_lhe.write('</LesHouchesEvents>\n')
            try:
                output_lhe.close()
            except Exception:
                pass
            self._apply_accounting(base_out, [dict(n_processed=0, n_written=0,
                                                   nb_try=0, nb_loose_skip=0)])
            return

        self._clear_worker_status(nb_core)   # fresh status board for this phase
        mpctx = mp.get_context('fork')
        procs, frag_paths, stats_paths = [], [], []
        for sid in range(nb_core):
            frag = '%s.shard%d.lhe' % (base_out, sid)
            stp = '%s.shard%d.json' % (base_out, sid)
            cctx = dict(ctx)
            cctx['shard_nb_event'] = shard_counts[sid]
            p = mpctx.Process(
                target=self._unweight_shard_entry,
                args=(sid, nb_core, shard_paths[sid], frag, evt_decayfile,
                      cctx, stp))
            p.start()
            procs.append(p)
            frag_paths.append(frag)
            stats_paths.append(stp)
        for p in procs:
            p.join()

        # collect stats and surface worker failures
        stats_list = []
        for sid, stp in enumerate(stats_paths):
            if not os.path.exists(stp):
                raise Exception("MadSpin worker %s produced no result (crashed, exitcode=%s). "
                                "Re-run with nb_core=1 to reproduce/debug."
                                % (sid, procs[sid].exitcode))
            with open(stp) as f:
                s = json.load(f)
            if 'error' in s:
                raise Exception("MadSpin worker %s failed:\n%s"
                                % (sid, s.get('tb', s['error'])))
            stats_list.append(s)

        # merge: one banner + fragment bodies (in production order) + closing tag
        output_lhe = lhe_parser.EventFile(base_out, 'w')
        if self.options['fixed_order']:
            output_lhe.eventgroup = True
        self.banner.scale_init_cross(self.branching_ratio)
        self.banner.write(output_lhe, close_tag=False)
        for frag in frag_paths:
            if os.path.exists(frag):
                with open(frag) as fr:
                    for line in fr:
                        output_lhe.write(line)
        output_lhe.write('</LesHouchesEvents>\n')
        try:
            output_lhe.close()
        except Exception:
            pass

        for pth in shard_paths + frag_paths + stats_paths:
            try:
                os.remove(pth)
            except OSError:
                pass

        self._apply_accounting(base_out, stats_list)

    def _rewrite_lhe_banner_cross(self, path, ratio, n_written=None):
        """Rewrite an already-written LHE file, multiplying every <init> line
        cross-section / error / xmax by ``ratio`` and (optionally) replacing
        the ``Number of Events`` entry in the MGGenerationInfo block with
        ``n_written``. Mirrors decay_all_events.write_banner_information for
        the PA-mode (run_onshell) code path."""

        tmp_path = path + '.tmp_brfix'
        shutil.move(path, tmp_path)
        with open(tmp_path, 'r') as src, open(path, 'w') as dst:
            in_init = False
            in_mggen = False
            for line in src:
                stripped = line.strip()
                lstripped = stripped.lower()
                if lstripped.startswith('<init'):
                    in_init = True
                    dst.write(line)
                    continue
                if in_init:
                    if lstripped.startswith('</init'):
                        in_init = False
                        dst.write(line)
                        continue
                    parts = stripped.split()
                    if len(parts) == 4:
                        try:
                            xsec, xerr, xmax = (float(parts[0]), float(parts[1]), float(parts[2]))
                            pid = int(parts[3])
                            dst.write("   %+13.7e %+13.7e %+13.7e %i\n" %
                                      (ratio*xsec, ratio*xerr, ratio*xmax, pid))
                            continue
                        except ValueError:
                            pass
                    dst.write(line)
                    continue
                # MGGenerationInfo block: update Number of Events and any
                # ":" -separated numeric field with the BR correction ratio.
                if lstripped.startswith('<mggenerationinfo'):
                    in_mggen = True
                    dst.write(line)
                    continue
                if in_mggen:
                    if lstripped.startswith('</mggenerationinfo'):
                        in_mggen = False
                        dst.write(line)
                        continue
                    if 'Number of Events' in line and n_written is not None:
                        dst.write('#  Number of Events        :       %i\n' % n_written)
                        continue
                    if ':' in line:
                        head, tail = line.rsplit(':', 1)
                        try:
                            value = float(tail.strip())
                            dst.write('%s : %s\n' % (head, value * ratio))
                            continue
                        except ValueError:
                            pass
                    dst.write(line)
                    continue
                dst.write(line)
        os.remove(tmp_path)

    def get_decay_from_file(self,production, evt_decayfile, nb_remain):
        """return a dictionary PDG -> list of associated decay"""

        out = collections.defaultdict(list)
        for i, particle, decay in self._draw_all_decays(production, evt_decayfile,
                                                        nb_remain):
            out[particle.pdg].append(decay)
        return out

    def _draw_all_decays(self, production, evt_decayfile, nb_remain, group=None):
        """Yield (slot_index, particle, decay) for every decaying particle of the
        production event, in production order -- which is the order the density
        matrix slots are built in.

        ``group``: the '@' decay group this draw belongs to. Drawn here when the
        caller does not impose one, so that it is redrawn on every trial of the
        joint accept/reject along with the decays it selects."""
        particles = [p for p in production if int(p.status) == 1.0]
        ids = [particle.pid for particle in particles]
        if group is None:
            group = self._draw_decay_group()
        for i, particle in enumerate(particles):
            decay = self._draw_one_decay(particle, i, ids, evt_decayfile,
                                         nb_remain, group)
            if decay is not None:
                yield i, particle, decay

    def _draw_one_decay(self, particle, i, ids, evt_decayfile, nb_remain,
                        group=None):
        """Draw one decay event for ``particle`` -- the i-th final-state particle
        of the production event, ``ids`` being the pdgs of all of them -- and
        refill its pool if it runs out. Returns None when that particle does not
        decay.

        Factored out of get_decay_from_file so that the sequential accept/reject
        can redraw a single particle without touching the ones already accepted.

        ``group`` restricts the choice to the channels that group gives this
        particle; the rules below then apply to those, unchanged. That is the
        whole of the grouping at run time -- a group supplies exactly one channel
        per particle (or one per identical parent, which the positional rule then
        deals out), so restricting the candidates is all it takes.
        """
        # check if we need to decay the particle
        if particle.pdg not in evt_decayfile:
            return None # nothing to do for this particle
        channels = evt_decayfile[particle.pdg]
        if group is not None:
            keys = [k for k in self._decay_groups['lines']
                                    .get(particle.pdg, {}).get(group, ())
                    if k in channels]
        else:
            keys = sorted(channels)
        # check how the decay need to be done
        nb_decay = len(keys)
        if nb_decay == 0:
            return None #nothing to do for this particle
        # Determine the file to read in order to get the decay [decay_file]
        if nb_decay == 1:
            decay_file_nb = keys[0]
            decay_file = channels[decay_file_nb]
        elif ids.count(particle.pdg) == nb_decay:
            decay_file_nb = keys[ids[:i].count(particle.pdg)]
            decay_file = channels[decay_file_nb]
        else:
            #need to select the file according to the associate cross-section
            r = random.random()
            tot = sum(channels[key].cross for key in keys)
            r = r * tot
            cumul = 0
            for j in keys:
                events = channels[j]
                cumul += events.cross
                if r < cumul:
                    decay_file = events
                    decay_file_nb = j
                    break
                else:
                    continue
            else:
                raise Exception
        # So now we know which file to read. Do it and re-generate events for that 
        # file if needed.
        while 1:
            try:
                decay = next(decay_file)
                break
            except StopIteration:
                # Estimate refill size from remaining production events
                # efficiency and per-trial consumption if decaying particles
                # Take into account identical parents
                # Oversample by 10% to reduce refill frequency; cap to limit one refill cost.
                eff = max(self.efficiency, 1e-12)
                same_pdg = ids.count(particle.pdg)
                if nb_decay == 1:
                    burn = same_pdg
                elif nb_decay == same_pdg:
                    burn = 1.0
                else:
                    burn = max(1.0, float(same_pdg) / float(nb_decay))
                needed = int(math.ceil(1.10 * burn * nb_remain / eff))
                # Statistical-fluctuation security: the number of events we
                # actually get back fluctuates like sqrt(N), so ask for
                # sqrt(target) more than the bare target -- running short
                # would cost a whole extra refill.
                needed += int(math.ceil(math.sqrt(needed)))
                needed = min(200000, max(needed, 1000))
                if getattr(self, '_shard_tag', None) is not None:
                    # Parallel unweighting: generation is not fork-safe and must
                    # not run concurrently. Each channel has a fixed OWNER worker
                    # that generates a pool for everybody (nb_core * remaining /
                    # eff); the others block until it is ready. _worker_refill
                    # returns this worker's own reader over that pool (the owner's
                    # is ~10% short so it runs out -- and regenerates -- first).
                    evt_decayfile[particle.pdg][decay_file_nb] = \
                        self._worker_refill(
                            particle.pdg, decay_file_nb,
                            needed * self._shard_nb_core)
                else:
                    # serial: _regenerate_events already returns the reader
                    # over the events it produced
                    self._refill_nb = getattr(self, '_refill_nb', 0) + 1
                    evt_decayfile[particle.pdg][decay_file_nb] = \
                        self._regenerate_events(
                            particle.pdg, decay_file_nb, needed,
                            'ms_refill_%d' % self._refill_nb)
                decay_file = evt_decayfile[particle.pdg][decay_file_nb]
                continue
        return decay
        
    
    def get_maxwgt_for_onshell(self, orig_lhe, evt_decayfile, decay_dict):
        """determine the maximum weight for the onshell (or similar) strategy"""
        #print(f"decay_dict = {decay_dict} - length = {len(decay_dict)}")
        # event_decay is a dict pdg -> list of event file (contain the decay)
                
        if self.options['ms_dir'] and os.path.exists(pjoin(self.options['ms_dir'], 'max_wgt')):
            return float(open(pjoin(self.options['ms_dir'], 'max_wgt'),'r').read())

        nevents = self.options['Nevents_for_max_weight']
        if nevents == 0 :
            nevents = 75
        nb_ps_point = self.options['max_weight_ps_point']

        # Same even-split rounding as the sequential scan: round the probe events
        # up to a multiple of nb_core and reduce nb_ps_point to keep the sampling
        # budget, so the forked scan splits evenly across the workers.
        nb_core = self._resolve_nb_core()
        if nb_core > 1 and nevents % nb_core:
            budget = nevents * nb_ps_point
            nevents = int(math.ceil(nevents / float(nb_core))) * nb_core
            nb_ps_point = max(1, int(round(budget / float(nevents))))

        logger.info("Estimating the maximum weight")
        logger.info("*****************************")
        logger.info("Probing the first %s events with %s phase space points"
                    % (nevents, nb_ps_point))

        orig_lhe.seek(0)
        if self.options['fixed_order']:
            orig_lhe.eventgroup = True
        events = []
        for _ in range(nevents):
            try:
                events.append(next(orig_lhe))
            except StopIteration:
                break
        if not events:
            return 0.0

        # The probe events are independent, so the scan forks the same way as
        # the sequential one -- each worker owns a slice of the events and its
        # own view of the decay pools.
        nb_core = max(1, min(nb_core, len(events)))
        if nb_core == 1:
            all_maxwgt = self._joint_maxwgt_range(events, 0, len(events),
                                                  evt_decayfile, decay_dict,
                                                  nevents, nb_ps_point)
        else:
            logger.info("MadSpin: probing the maximum weight on %s cores", nb_core)
            all_maxwgt, _ = self._scan_maxwgt_parallel(
                orig_lhe, events, evt_decayfile, nb_core,
                self._joint_maxwgt_shard_entry, (decay_dict, nevents, nb_ps_point))

        base_max_weight = self._combine_maxwgt(all_maxwgt)
        if self.options['ms_dir']:
            open(pjoin(self.options['ms_dir'], 'max_wgt'),'w').write(str(base_max_weight))
        return base_max_weight

    def _scan_maxwgt_range(self, events, start, stop, evt_decayfile,
                           nevents, nb_ps_point):
        """Per-event probe data for ``events[start:stop]``, and the samples of
        the rate factor collected along the way.

        Returns ``(per_event, z_samples)``, or ``(None, {})`` as soon as a
        production event turns out to have nothing to decay (the caller then
        falls back to the joint bound).

        Under ``sequential_with_mass`` ``per_event`` is one max-weight vector
        per event, holding for each ordering position the largest w_k over
        ``nb_ps_point`` chains -- all the bound needs. The up-front-mass schemes
        keep every chain instead: their mass-set weight is only complete once
        Z_k is known, and Z_k is fitted from ``z_samples``, which this same
        probe produces. Taking the maximum there is deferred to the caller, over
        the completed weights.
        """
        self.efficiency = 1. / nb_ps_point
        upfront = self._sequential_upfront()
        t0 = time.time()
        per_event = []
        z_samples = collections.defaultdict(list)
        for i in range(start, stop):
            if (i - start) % 5 == 1 and getattr(self, '_shard_tag', None) in (None, 0):
                # only one worker prints scan progress
                logger.info("Event %s/%s :  %2fs" % (i, stop, time.time()-t0))
            base_event = events[i]
            # events left in *this* range (the worker's shard), not the global
            # nevents - i. It only feeds the decay-pool refill sizing, and the
            # refill already multiplies by nb_core to share the pool across
            # workers -- so passing the global count made a forked scan worker
            # refill nb_core times too many decays (900k instead of ~60k).
            nb_remain = stop - i
            best = None
            chains = []
            extra = {}
            for _ in range(nb_ps_point):
                probe = []
                out = self.sequential_accept_reject(base_event, evt_decayfile,
                                                    None, nb_remain, probe=probe,
                                                    probe_extra=extra)
                if out is None:
                    return None, {}
                if upfront:
                    chains.append([list(probe), list(extra['mass'])])
                    for key, mass, value in extra.pop('z', ()):
                        z_samples[key].append((mass, value))
                elif best is None:
                    best = list(probe)
                else:
                    best = [max(old, new) for old, new in zip(best, probe)]
            if upfront:
                if chains:
                    per_event.append({'keys': extra['keys'],
                                      'order': extra['order'],
                                      'chains': chains})
            elif best:
                per_event.append(best)
        return per_event, dict(z_samples)

    def _scan_maxwgt_shard_entry(self, shard_id, nb_core, events, start, stop,
                                 evt_decayfile, nevents, nb_ps_point, out_path):
        """Worker entry (forked child): scan its slice of the probe events and
        write the per-event vectors as JSON, mirroring _unweight_shard_entry --
        its own RNG stream, its own reopened decay pools, failures reported in
        the JSON rather than raised into the parent."""
        import json
        try:
            random.seed((int(self.seed) if self.seed else 0)
                        + 7919 * (shard_id + 1))
            self._shard_tag = shard_id
            self._shard_nb_core = nb_core
            self._pool_gen = {}
            self._init_owner_refill(evt_decayfile, self.seed)
            local_pool = self._reopen_decay_pool(evt_decayfile, shard_id, nb_core)
            per_event, z_samples = self._scan_maxwgt_range(
                            events, start, stop, local_pool, nevents, nb_ps_point)
            with open(out_path, 'w') as f:
                json.dump({'per_event': per_event, 'z_samples': z_samples}, f)
        except Exception as exc:
            import traceback
            try:
                with open(out_path, 'w') as f:
                    json.dump({'error': str(exc),
                               'tb': traceback.format_exc()}, f)
            except Exception:
                pass
        finally:
            self._set_status('D')   # release any worker blocked on a channel I own

    def _joint_maxwgt_range(self, events, start, stop, evt_decayfile, decay_dict,
                            nevents, nb_ps_point):
        """Per-event maximum weight for the *joint* accept/reject, over
        ``events[start:stop]``: for each production event, the largest
        full_me/(prod*dec)*jac seen over nb_ps_point decay draws. The serial body
        of get_maxwgt_for_onshell, factored so it can also run in a forked
        worker."""
        self.efficiency = 1. / nb_ps_point
        t0 = time.time()
        density_pole_approximation = self.options['spinmode'] in ['PA', 'onshell']
        density_do_reshuffle = self.options['spinmode'] == 'PA'
        density_needs_reshuffle = (
            self.generate_all.mode == 'density'
            and (not density_pole_approximation or density_do_reshuffle))
        per_event = []
        for i in range(start, stop):
            if (i - start) % 5 == 1 and getattr(self, '_shard_tag', None) in (None, 0):
                logger.info("Event %s/%s :  %2fs" % (i, stop, time.time()-t0))
            base_event = events[i]
            if self.options['fixed_order']:
                base_event = base_event[0]
            maxwgt = 0
            density_matrix_prod = None
            offshell_density = (self.generate_all.mode == 'density'
                                and not density_pole_approximation)
            for j in range(nb_ps_point):
                # stop - i (this worker's remaining events) sizes the pool refill
                decays = self.get_decay_from_file(base_event, evt_decayfile, stop - i)
                # offshell/madspin reshuffles the production event in place; use a
                # per-draw onshell copy so repeated draws don't compound and so the
                # reshuffle jacobian (now folded into wgt) is taken from the onshell
                # reference each draw.
                prod_draw = lhe_parser.Event(str(base_event)) if offshell_density else base_event
                if density_matrix_prod is None:
                    _, wgt, density_matrix_prod = self.get_onshell_evt_and_wgt(
                        prod_draw, decays, decay_dict, build_event=False)
                else:
                    wgt = self.get_onshell_evt_and_wgt(
                        prod_draw, decays, decay_dict, density_matrix_prod,
                        build_event=False)[1]
                jac = 1
                if (density_needs_reshuffle and not offshell_density
                        and self.options['density_keep_jacobian']):
                    # PA with explicit jacobian tracking: reshuffle to expose the
                    # jacobian in the max weight. Offshell/madspin does NOT enter
                    # here -- its reshuffle jacobian is already inside wgt.
                    full_evt = lhe_parser.Event(str(base_event))
                    full_evt = full_evt.add_decays(decays)
                    jac = full_evt.reshuffle_production()
                maxwgt = max(wgt*jac, maxwgt)
            per_event.append(float(getattr(maxwgt, 'real', maxwgt)))
        return per_event

    def _joint_maxwgt_shard_entry(self, shard_id, nb_core, events, start, stop,
                                  evt_decayfile, decay_dict, nevents, nb_ps_point,
                                  out_path):
        """Worker entry (forked child) for the joint max-weight scan. Mirrors
        _scan_maxwgt_shard_entry: own RNG stream, own reopened decay pools,
        failures reported in the JSON."""
        import json
        try:
            random.seed((int(self.seed) if self.seed else 0)
                        + 7919 * (shard_id + 1))
            self._shard_tag = shard_id
            self._shard_nb_core = nb_core
            self._pool_gen = {}
            self._init_owner_refill(evt_decayfile, self.seed)
            local_pool = self._reopen_decay_pool(evt_decayfile, shard_id, nb_core)
            per_event = self._joint_maxwgt_range(events, start, stop, local_pool,
                                                 decay_dict, nevents, nb_ps_point)
            with open(out_path, 'w') as f:
                json.dump({'per_event': per_event}, f)
        except Exception as exc:
            import traceback
            try:
                with open(out_path, 'w') as f:
                    json.dump({'error': str(exc),
                               'tb': traceback.format_exc()}, f)
            except Exception:
                pass
        finally:
            self._set_status('D')   # release any worker blocked on a channel I own

    def _scan_maxwgt_parallel(self, orig_lhe, events, evt_decayfile, nb_core,
                              shard_entry, extra):
        """Fork one worker per contiguous slice of the probe events; each runs
        ``shard_entry`` and returns its per-event data through a JSON file.
        Concatenating them is order independent -- _combine_maxwgt takes the
        max/spread over all events -- so the result matches the serial scan up to
        which decays each draw pulls. ``extra`` is the tuple of arguments the
        shard entry needs after ``evt_decayfile`` (the joint and sequential scans
        pass different ones). Used by both get_maxwgt_for_onshell and
        get_sequential_maxwgt."""
        import multiprocessing as mp
        import json
        base = '%s.maxwgt' % orig_lhe.name
        chunk = int(math.ceil(len(events) / float(nb_core)))
        # contiguous slices; the last cores get nothing if events < nb_core
        ranges = [(sid * chunk, min((sid + 1) * chunk, len(events)))
                  for sid in range(nb_core)]
        ranges = [(a, b) for (a, b) in ranges if a < b]
        # Keep the ORIGINAL nb_core as the pool-addressing count: the decay pool
        # was split into nb_core files, and each worker must address it with that
        # same count so it opens *its* file (paths[shard_id]). Reducing it to the
        # number of non-empty ranges made len(paths) != nb_core, which dropped
        # every worker onto the striding fallback -- reading only the first file.
        # Trailing empty shards are simply not launched (their files go unused by
        # the scan, which is fine -- the pool is generated uniformly).

        self._clear_worker_status(nb_core)   # fresh status board for this phase
        mpctx = mp.get_context('fork')
        procs, out_paths = [], []
        for sid, (start, stop) in enumerate(ranges):
            outp = '%s.shard%d.json' % (base, sid)
            p = mpctx.Process(
                target=shard_entry,
                args=(sid, nb_core, events, start, stop, evt_decayfile)
                     + tuple(extra) + (outp,))
            p.start()
            procs.append(p)
            out_paths.append(outp)
        for p in procs:
            p.join()

        per_event = []
        result = per_event
        z_samples = collections.defaultdict(list)
        for sid, outp in enumerate(out_paths):
            if not os.path.exists(outp):
                raise Exception("MadSpin max-weight worker %s produced no result "
                                "(crashed). Re-run with nb_core=1 to debug." % sid)
            with open(outp) as f:
                r = json.load(f)
            if 'error' in r:
                raise Exception("MadSpin max-weight worker %s failed:\n%s"
                                % (sid, r.get('tb', r['error'])))
            if r['per_event'] is None:
                result = None   # nothing to decay: fall back to the joint bound
            elif result is not None:
                result.extend(r['per_event'])
            for key, samples in (r.get('z_samples') or {}).items():
                # json turns the (mass, value) pairs into lists
                z_samples[key].extend((s[0], s[1]) for s in samples)
        for outp in out_paths:
            try:
                os.remove(outp)
            except OSError:
                pass
        return result, dict(z_samples)

    def get_sequential_maxwgt(self, orig_lhe, evt_decayfile):
        """One bound C_k per position of the decay ordering, for the sequential
        accept/reject. Returns [] when nothing decays.

        Same probe as the joint scan -- the first Nevents_for_max_weight
        production events, max_weight_ps_point sets of decays each, the largest
        weight per production event, then mean + nb_sigma*sd -- but applied to
        each slot's own weight.

        The probe draws every slot from the pool uniformly whereas the real
        chain draws slot k conditioned on the decays it has accepted. The
        support is the same, so this estimates the same bound; only the density
        with which the tail is explored differs, which is why the margins are
        kept and the accept/reject counts its overflows.
        """
        offshell = self._sequential_offshell()
        upfront = self._sequential_upfront()
        cache = None
        if self.options['ms_dir']:
            # a distinct name: the joint bound is a single float, this is a list.
            # The up-front-mass bounds come with the Z_k tables and depend on the
            # unweighting mode *and* on the spinmode family (the mass-set weight
            # is a different quantity offshell and under PA), so they get a name
            # (and a format) of their own -- a cache written for one cannot be
            # read back for the other.
            if upfront:
                mode = self._unweighting_mode()
                variant = '' if mode == 'sequential' else '_%s' % mode
                cache = pjoin(self.options['ms_dir'],
                              'max_wgt_sequential_%s%s'
                              % ('offshell' if offshell else 'pa', variant))
                cached = self._read_offshell_cache(cache)
                if cached is not None:
                    self._z_tables = cached['z_tables']
                    return cached['maxwgts']
            else:
                cache = pjoin(self.options['ms_dir'], 'max_wgt_sequential')
                if os.path.exists(cache):
                    return [float(x) for x in open(cache).read().split()]

        nevents = self.options['Nevents_for_max_weight']
        if nevents == 0:
            nevents = 75
        nb_ps_point = self.options['max_weight_ps_point']

        # Round the number of probe events up to a multiple of nb_core so the
        # parallel scan splits evenly -- every worker gets the same number of
        # events, no worker is the odd one out with an extra event whose pool
        # slice runs short. Reduce nb_ps_point to keep the total decays drawn
        # (nevents * nb_ps_point, the sampling budget) roughly unchanged.
        nb_core = self._resolve_nb_core()
        if nb_core > 1 and nevents % nb_core:
            budget = nevents * nb_ps_point
            nevents = int(math.ceil(nevents / float(nb_core))) * nb_core
            nb_ps_point = max(1, int(round(budget / float(nevents))))

        logger.info("Estimating the maximum weight of each decaying particle")
        logger.info("*****************************")
        logger.info("Probing the first %s events with %s phase space points"
                    % (nevents, nb_ps_point))
        # a non-joint unweighting never reaches here with fixed_order (it falls back to
        # the joint accept/reject), so the events are plain, not event-groups.
        orig_lhe.seek(0)
        events = []
        for _ in range(nevents):
            try:
                events.append(next(orig_lhe))
            except StopIteration:
                break
        if not events:
            return []

        # The probe events are independent, exactly like the unweighting, so the
        # scan forks the same way -- each worker owns a slice of the events and
        # its own view of the decay pools.
        nb_core = self._resolve_nb_core()
        nb_core = max(1, min(nb_core, len(events)))
        if nb_core == 1:
            per_event, z_samples = self._scan_maxwgt_range(
                events, 0, len(events), evt_decayfile, nevents, nb_ps_point)
        else:
            logger.info("MadSpin: probing the maximum weight on %s cores", nb_core)
            per_event, z_samples = self._scan_maxwgt_parallel(
                orig_lhe, events, evt_decayfile, nb_core,
                self._scan_maxwgt_shard_entry, (nevents, nb_ps_point))

        if per_event is None:
            return []   # a production event had nothing to decay
        if len(per_event) < 2:
            # _combine_maxwgt needs a spread to work with
            return []

        if upfront:
            self._z_tables = self._build_z_tables(z_samples)
            per_event = [self._complete_upfront_probe(event)
                         for event in per_event]

        maxwgts = [self._combine_maxwgt([event[slot] for event in per_event])
                   for slot in range(len(per_event[0]))]
        logger.info("Sequential maximum weights: %s",
                    ' '.join('%.4g' % w for w in maxwgts))
        if cache and upfront:
            import json
            with open(cache, 'w') as f:
                json.dump({'format': self._OFFSHELL_CACHE_FORMAT,
                           'maxwgts': maxwgts, 'z_tables': self._z_tables}, f)
        elif cache:
            open(cache, 'w').write(' '.join(repr(w) for w in maxwgts))
        return maxwgts

    # Bumped whenever the cache's *meaning* changes: another entry in the bound
    # vector, a different fit variable or degree, another key in a table. The
    # file name already separates the unweighting modes, and the offshell
    # spinmodes from PA/onshell; this separates one version of this code from
    # the next, which a name cannot.
    # 2: the mass-set weight is normalised by |M_prod|^2 on shell, so every
    #    bound in the vector changed scale.
    _OFFSHELL_CACHE_FORMAT = 2

    def _read_offshell_cache(self, path):
        """The cached offshell bounds and Z_k tables, or None if there is
        nothing usable there.

        A cache that does not match what this code writes is *ignored*, not
        repaired and not raised on: the scan that produced it is reproducible,
        so paying for it again is always an option, whereas a table read under
        the wrong schema would either crash deep inside the accept/reject or --
        worse -- silently weight the virtualities with somebody else's fit.
        Hence a format tag, and a structural check of every field the
        accept/reject will dereference.
        """
        if not path or not os.path.exists(path):
            return None
        import json
        try:
            with open(path) as f:
                cached = json.load(f)
            if cached.get('format') != self._OFFSHELL_CACHE_FORMAT:
                raise ValueError('format %s, expected %s'
                                 % (cached.get('format'),
                                    self._OFFSHELL_CACHE_FORMAT))
            maxwgts = [float(w) for w in cached['maxwgts']]
            if not maxwgts:
                raise ValueError('no bounds')
            tables = cached['z_tables']
            for key, table in tables.items():
                missing = {'pole', 'coeff', 'zero_below',
                           'range'} - set(table)
                if missing:
                    raise ValueError('slot %s is missing %s'
                                     % (key, ', '.join(sorted(missing))))
                if len(table['coeff']) != 3 or len(table['range']) != 2:
                    raise ValueError('slot %s has a malformed fit' % key)
        except Exception as error:
            logger.warning("MadSpin: ignoring the cached sequential maximum "
                           "weights in %s (%s); they will be measured again.",
                           path, error)
            return None
        return {'maxwgts': maxwgts, 'z_tables': tables}

    def _complete_upfront_probe(self, event):
        """The per-event maximum-weight vector of an up-front-mass probe, over
        the chains it recorded and with the Z_k factors the loop could not apply
        while they were still being measured: Z_k(m_k) into the mass-set weight,
        and -- under sequential_global_retry, where the mass stage pays it and the
        per-angle stage takes it back -- 1/Z_k into that slot's own weight.
        """
        keys, order = event['keys'], event['order']
        mode = self._unweighting_mode()
        joint_angles = mode == 'two_stage'
        # Both sequential_global_retry and two_stage test w_k/Z_hat_k rather
        # than w_k: the first because the mass stage has already paid Z_hat and
        # the two must cancel, the second because it flattens the virtuality
        # dependence out of the bound. The probe has to be completed the same
        # way or the bound and the weight it bounds are different quantities.
        exact = joint_angles or mode == 'sequential_global_retry'
        best = None
        for weights, masses in event['chains']:
            zhat = [self._zhat(key, mass) for key, mass in zip(keys, masses)]
            current = list(weights)
            for z in zhat:
                current[0] *= z
            if exact:
                for position, slot in enumerate(order):
                    current[position + 1] = (weights[position + 1] / zhat[slot]
                                             if zhat[slot] > 0 else 0.0)
            if joint_angles:
                # one bound over the product, so the vector is [C_mass, C_angles]
                product = 1.0
                for value in current[1:]:
                    product *= value
                current = [current[0], product]
            best = current if best is None else \
                   [max(old, new) for old, new in zip(best, current)]
        return best

    def _combine_maxwgt(self, all_maxwgt):
        """Turn the per-production-event maxima of a probe into the bound the
        accept/reject uses: mean + nb_sigma*sd with a safety margin, refined on
        the largest ones and never below the second largest seen.

        Shared by the joint bound and by each slot's bound in sequential mode.
        The sequential accept/reject has no way to carry a per-slot overweight
        forward (redraw-until-accept, not staged unweighting), so a weight above
        the bound biases the sample directly -- hence a 10% margin rather than
        the historical 5%.
        """
        margin = 1.10
        all_maxwgt.sort(reverse=True)
        assert all_maxwgt[0] >= all_maxwgt[1], "ERROR: "
        decay_tools=madspin.decay_misc()
        ave_weight, std_weight = decay_tools.get_mean_sd(all_maxwgt)
        base_max_weight = margin * (ave_weight+self.options['nb_sigma']*std_weight)

        for i in [20, 30, 40, 50]:
            if len(all_maxwgt) < i:
                break
            ave_weight, std_weight = decay_tools.get_mean_sd(all_maxwgt[:i])
            base_max_weight = max(base_max_weight, margin * (ave_weight+self.options['nb_sigma']*std_weight))

            if all_maxwgt[1] > base_max_weight:
                base_max_weight = margin * all_maxwgt[1]
        return base_max_weight

            
    def _density_basis(self, production, decays_key):
        """Helicity-basis bookkeeping for the production density matrix: which
        particles decay, where they sit (``position``, ``init_part``), their
        helicity bases (``helicities``, and the ``allowed_hel``/``ncomb``/
        ``dimension`` the Fortran side needs), plus the averaging and identical
        final-state symmetry factors.

        It depends only on the production event and on *which* pdgs decay --
        not on the decay events, and not on any sampled mass -- so it is
        computed once per production event and reused across every retry, and
        across every slot of the sequential accept/reject.
        """
        # Production averaging factor (spin/color initial state) from standalone
        iden_p = self.get_iden(production)

        # Symmetry factor for identical final states in production
        final_pdgs = [int(p.pid) for p in production if getattr(p, "status", None) == 1]
        counts_final = collections.Counter(final_pdgs)
        sym_factor_prod_ident = 1
        for n in counts_final.values():
            if n > 1:
                sym_factor_prod_ident *= math.factorial(n)

        # Find particles that should decay (status==1 and pid in decays keys)
        init_part = [part for pdg in decays_key for part in production
                     if part.pid == pdg and part.status == 1]
        nchanging = len(init_part)

        # Allowed helicities per spin
        hel_dict = {1: [0], 2: [1, -1], 3: [-1, 0, 1]}

        # Decaying-particle positions (+1 for Fortran), spins, helicities
        position = [i + 1 for pdg in decays_key
                    for i in range(len(production))
                    if production[i].pid == pdg and production[i].status == 1]
        decaying_pdg = [int(production[i - 1].pid) for i in position]
        decaying_spins = [self.model.get_particle(i).get('spin') for i in decaying_pdg]
        helicities = [hel_dict[i] for i in decaying_spins]

        allowed_hel_pairs, allowed_hel = self.get_allowed_hel(helicities)

        return {
            'decays_key': decays_key,
            'iden_p': iden_p,
            'sym_factor_prod_ident': sym_factor_prod_ident,
            'init_part': init_part,
            'nchanging': nchanging,
            'position': position,
            'helicities': helicities,
            'decaying_spins': decaying_spins,
            'allowed_hel': allowed_hel,
            'ncomb': len(allowed_hel_pairs),
            'dimension': math.prod(len(i) for i in helicities),
        }

    @staticmethod
    def _decaying_pdgs(production, evt_decayfile):
        """The pdgs that decay, in order of first appearance among the
        production's final-state particles.

        That is the order ``get_decay_from_file`` fills its dict in, hence the
        order ``_density_basis`` lays the density matrix slots out in. The
        sequential accept/reject needs it *before* drawing anything, to build
        the basis, so it is derived from the pools rather than from a draw. The
        "does this particle decay" test must stay identical to
        ``_draw_one_decay``'s.
        """
        out = []
        for particle in production:
            if int(particle.status) != 1:
                continue
            if particle.pdg not in evt_decayfile:
                continue
            if not len(evt_decayfile[particle.pdg]):
                continue
            if particle.pdg not in out:
                out.append(particle.pdg)
        return tuple(out)

    @staticmethod
    def _sequential_slots(production, decays_key):
        """Map each density matrix slot to the production final-state particle
        it belongs to.

        Returns (particles, slot_to_index): ``particles`` is the final state in
        production order (what ``_draw_one_decay`` indexes into), and
        ``slot_to_index[s]`` is the position in it of slot s's particle. The
        slot order mirrors ``_density_basis``'s ``init_part`` -- for pdg in
        decays_key, in production order -- which is the order the tensor product
        is built in. The decay *ordering* permutes which slot is filled next; it
        must never permute this.
        """
        particles = [p for p in production if int(p.status) == 1]
        slot_to_index = []
        for pdg in decays_key:
            for i, particle in enumerate(particles):
                if particle.pid == pdg:
                    slot_to_index.append(i)
        return particles, slot_to_index

    @staticmethod
    def _production_jacobian_for(production, slot_to_index, slot_masses):
        """J_k: the production reshuffling jacobian with the slots drawn so far
        carrying their sampled virtuality and the rest still at their nominal
        mass. Returns -1 (or 0) when that mass set cannot be reshuffled.

        ``slot_masses`` maps slot -> (new_mass, reshuffle_info).

        The masses are placed by slot identity rather than through
        ``add_decays``, which attaches a pdg's decays to its particles in
        production order: with a pdg owning several slots and only some of them
        drawn, that would hand a mass to the wrong particle. Resonance handling
        is left to reshuffle_production, which is why this is not a function of
        the top-level masses alone.
        """
        probe = lhe_parser.Event(str(production))
        finals = [p for p in probe if int(p.status) == 1]
        for slot, (new_mass, info) in slot_masses.items():
            particle = finals[slot_to_index[slot]]
            particle.new_mass = new_mass
            if info is not None:
                particle.reshuffle_info = info
        return probe.reshuffle_production(_allow_retry=False)

    def _slot_identity(self, hel):
        """The I/n a slot contributes while its decay has not been drawn yet.
        Depends only on the helicity list, so cache it per basis."""
        key = tuple(hel)
        try:
            cache = self._slot_identity_cache
        except AttributeError:
            cache = self._slot_identity_cache = {}
        if key not in cache:
            cache[key] = madspin.DensityMatrix.identity(1, list(hel), len(hel))
        return cache[key]

    def _partial_density_contraction(self, density_prod, helicities, slot_densities):
        """N_k: the production density matrix contracted with the normalised
        decay density matrix (Dhat = D/Tr D) of every slot drawn so far, the
        slots still to be drawn contributing I/n -- the average of a decay
        density matrix over its full phase space.

        ``slot_densities`` maps slot index -> DensityMatrix as get_density
        returns it (un-normalised); slots absent from it are the undrawn ones.

        The tensor product is built in *slot* order, which is what the
        production density matrix's helicity index follows. The accept/reject
        ordering only decides which slot gets filled next -- it must never
        permute the tensor. See MADSPIN_SEQUENTIAL_PLAN.md.
        """
        density_dec = None
        for slot, hel in enumerate(helicities):
            density = slot_densities.get(slot)
            if density is None:
                density = self._slot_identity(hel)
            else:
                density = density.normalized()
            if density_dec is None:
                density_dec = density
            else:
                density_dec = density_dec.tensor_product(density)
        return density_dec.scalar_multiplication(density_prod)

    def _decay_reshuffle_jacobian(self, decay):
        """jac_dec: the jacobian of mapping this decay onto the virtuality just
        sampled for its parent. 0 when that is kinematically impossible --
        ``t > b j j`` with a top below MW+Mb -- which is also what tells the
        caller to draw the mass again.

        Probed on a copy: the real reshuffling of the decay still happens once,
        with the production, exactly as it does today. The value is the same
        either way -- ``mass_shuffle`` boosts to the parent rest frame before it
        builds the jacobian, and ``reshuffle_decay``'s ``new_incoming`` only
        feeds the Lorentz map applied afterwards -- so this probe is what enters
        the accept/reject weight and the final ``reshuffle_production`` merely
        recomputes it."""
        probe = lhe_parser.Event(str(decay))
        probe[0].new_mass = decay[0].new_mass
        probe[0].reshuffle_info = decay[0].reshuffle_info
        try:
            return probe.reshuffle_decayevt()
        except Exception:
            return 0

    def _slot_density(self, decay, parent, hel, frame_boost=None):
        """The decay density matrix of one slot, in the lab frame of its parent
        (and then in the ``frame_id`` frame, when there is one -- see
        ``get_density``)."""
        rest_leg = None
        if frame_boost is not None:
            rest_leg = self._decay_frame_rest_leg(parent, frame_boost)
        boost = -1 * lhe_parser.FourMomentum(parent)
        boost.E *= -1
        decay.boost(boost)
        return self.get_density(decay, position=[1], allow_hel=hel,
                                ncomb=len(hel), dimension=len(hel),
                                frame_boost=frame_boost, frame_rest_leg=rest_leg)

    def _draw_mass_value(self, pdg, budget):
        """Sample one resonance virtuality from its Breit-Wigner, capped at the
        remaining ``budget`` (what is left of sqrt(shat)). Returns
        ``(mass, reshuffle_info, jac_bw)`` where jac_bw is the Breit-Wigner
        sampling jacobian (gap/pi)."""
        pole = self.banner.get('param', 'mass', abs(pdg)).value
        width = self.banner.get('param', 'decay', abs(pdg)).value
        if self.options['BW_cut'] < 0:
            bw_cut = 15
        else:
            bw_cut = self.options['BW_cut']
        min_mass = pole - bw_cut * width
        max_mass = min(pole + bw_cut * width, budget)
        mass = lhe_parser.Event.generate_random_mass(pole, width, min_mass, max_mass)
        info = (pole, width, min_mass, max_mass)
        gap = math.atan((pole**2-min_mass**2)/pole/width)
        gap += math.atan((max_mass**2-pole**2)/pole/width)
        return mass, info, gap/math.pi

    def _draw_offshell_mass(self, pdg, dec, budget):
        """Sample one resonance virtuality and store it on the decay event that
        carries it: ``dec[0]`` gets ``new_mass`` and ``reshuffle_info``. Returns
        the budget left and that draw's jacobian.

        ``budget`` is what is left of sqrt(shat) once the resonances drawn
        before this one are paid for, so the draw is order dependent and the
        caller owns that order (used by the PA per-slot draw).
        """
        mass, info, jac = self._draw_mass_value(pdg, budget)
        dec[0].new_mass = mass
        dec[0].reshuffle_info = info
        return budget - mass, jac

    def _upfront_production(self, production, order, particles, slot_to_index,
                            prod_static, offshell, draw_mass=True,
                            density_prod=None, frame_boost=None):
        """Set up the production for one chain attempt of an up-front-mass
        scheme: draw a virtuality for every decaying particle *before* the
        per-particle loop, and settle everything that depends on the mass set
        but not on the decay angles.

        Returns ``(rho, jac_prod, slot_mass, parents, frame_boost)`` or None if
        the mass set is one the production cannot be reshuffled onto (the caller
        redraws the whole set):
        - ``slot_mass[slot]`` = (mass, reshuffle_info, jac_bw), empty when
          ``draw_mass`` is False (onshell, and 2 -> 1 production under PA, have
          no virtuality to sample). ``draw_mass`` describes the *PA* draw only:
          offshell always samples, since its rho is defined at the reshuffled
          momenta and there is nothing to evaluate without a mass set;
        - ``jac_prod``        = the production reshuffling jacobian of that mass
          set;
        - ``parents[slot]``   = the production particle to boost that slot's
          decay to;
        - ``frame_boost``     = the momentum whose rest frame ``frame_id``
          picks, for the momenta rho was evaluated at, so that every decay
          contracted against it is taken in the same frame. Offshell that is the
          *reshuffled* production, which only exists here; under PA rho is the
          cached onshell one, so the caller's boost is handed straight back.

        The two spinmode families differ in what the up-front draw is *for*.

        Offshell (madspin/full): rho depends on the whole mass set, so a *copy*
        of the production is reshuffled here (leaving the shared event
        untouched) and the density is evaluated at those momenta. Fixing rho
        before the loop is what makes the per-particle decomposition possible at
        all -- see MADSPIN_SEQUENTIAL_PLAN.md section 10.

        PA: rho is evaluated at the *onshell* momenta and is already fixed per
        production event (cached on it), so there is nothing to gain there. What
        the up-front draw buys instead is ``jac_prod``: with
        ``density_keep_jacobian`` on, the per-slot scheme calls
        ``_production_jacobian_for`` -- an event copy and a reshuffle -- on every
        slot trial and telescopes the results, whereas here it is one call per
        mass set and the J_k/J_{k-1} ratios disappear. The production event
        itself is never reshuffled here: under PA that is a post-acceptance
        dressing (or the final ``reshuffle_production``), and the decays are
        boosted to the onshell parents.
        """
        budget = production.sqrts
        slot_mass = {}
        if offshell or draw_mass:
            for slot in order:
                pdg = particles[slot_to_index[slot]].pid
                mass, info, jac_bw = self._draw_mass_value(pdg, budget)
                slot_mass[slot] = (mass, info, jac_bw)
                budget -= mass

        if not offshell:
            # PA/onshell: onshell rho, onshell parents. Only the feasibility of
            # the mass set and its reshuffling jacobian are settled here.
            jac_prod = 1.0
            if slot_mass:
                jac_prod = self._production_jacobian_for(
                    production, slot_to_index,
                    {slot: (mass, info)
                     for slot, (mass, info, _) in slot_mass.items()})
                if jac_prod in (0, -1):
                    return None
            # rho is the cached onshell one, so the frame it was evaluated in is
            # the caller's and is handed straight back
            return (density_prod, jac_prod, slot_mass,
                    prod_static['init_part'], frame_boost)

        prod_off = lhe_parser.Event(str(production))
        finals = [p for p in prod_off if int(p.status) == 1]
        for slot, (mass, info, _) in slot_mass.items():
            part = finals[slot_to_index[slot]]
            part.new_mass = mass
            part.reshuffle_info = info
        # _allow_retry=False so the drawn masses stay put (a retry would resample
        # them and diverge from the masses we reshuffle each decay to); an
        # impossible set is reported as -1 and the caller restarts.
        jac_reshuffle = prod_off.reshuffle_production(_allow_retry=False)
        if jac_reshuffle in (0, -1):
            return None
        # the frame is derived from the *reshuffled* production, since that is
        # the event rho_off is evaluated at; the decays are contracted against
        # it, so they have to be boosted with this same momentum
        frame_boost = self._frame_boost(prod_off)
        rho_off = self.get_density(prod_off, prod_static['position'],
                                   prod_static['allowed_hel'],
                                   prod_static['ncomb'], prod_static['dimension'],
                                   frame_boost=frame_boost)
        parents = {slot: finals[slot_to_index[slot]] for slot in order}
        return rho_off, jac_reshuffle, slot_mass, parents, frame_boost

    def _sequential_offshell(self):
        """Whether the sequential accept/reject runs its offshell (madspin/full)
        branch: the production density is evaluated at reshuffled momenta, so the
        virtualities are drawn up front and rho is fixed per chain."""
        return self.options['spinmode'] not in ['PA', 'onshell']

    def _sequential_upfront(self, density_method=True):
        """Whether the chain draws every virtuality *before* the angles, i.e.
        whether there is a mass-set accept/reject in front of the angle stage.

        True for every scheme but ``sequential_with_mass``, which draws each
        slot's mass inside that slot's own accept/reject. What the up-front draw
        buys differs by spinmode: offshell it fixes rho for the chain (which is
        what makes the per-particle decomposition possible at all), while under
        PA rho is already fixed at the onshell momenta and what is frozen
        instead is the *production reshuffling jacobian* -- one reshuffle per
        mass set rather than one per slot trial. Either way the angle stage then
        redraws to acceptance and divides out its own normalisation, which is
        what the tabulated ``_zhat`` puts back.
        """
        return self._unweighting_mode(density_method) not in \
                    ('joint', 'sequential_with_mass')

    # ------------------------------------------------------------------
    # Z_k(m): the rate factor of one slot, in the up-front-mass schemes
    # ------------------------------------------------------------------
    # Those chains are unweighted in two stages -- the mass set first, then each
    # slot's decay angles -- and the per-angle stage redraws until it accepts.
    # That divides its own normalisation
    #
    #     Z_k(m) = Integral p_pool(Omega) w_k(Omega, m) dOmega
    #
    # out of the accepted sample, so without a compensating factor in the
    # mass-set weight the accepted virtualities follow the Breit-Wigner instead
    # of the physical one. Either way Z_k is a smooth function of that slot's
    # virtuality *alone*: the production event, the other slots' masses and the
    # angles already accepted all cancel out of it, which is what makes it
    # tabulable. See MADSPIN_SEQUENTIAL_PLAN.md sections 10 and 11.
    #
    # What sits inside the average is the spinmode's own per-angle weight:
    #
    #   offshell   Z_k(m) = Integral dPhi_off(m) |M_dec|^2
    #                       / Integral dPhi_on |M_dec|^2
    #                     = (m/M) Gamma_k(m) / Gamma_k(M)
    #              -- the decay reshuffling jacobian *and* the offshell/onshell
    #              rate ratio Tr(D^off)/|M_dec|^2_on, the running width;
    #
    #   PA         Z_k(m) = E_pool[ jac_dec(m, Omega) ]
    #              -- the decay reshuffling jacobian alone. PA evaluates its
    #              matrix elements on shell, so there is no offshell integrand
    #              to reweight; what the angle stage normalises away is purely
    #              the phase-space cost of mapping a pool decay onto the sampled
    #              virtuality. (With density_keep_jacobian off that jacobian is
    #              not in the weight at all, and Z_k degenerates to the fraction
    #              of the pool that can reach m.)
    #
    # Same machinery for both: bin in m, average, fit, multiply into the
    # mass-set weight. Note that ``sequential_with_mass`` needs none of this --
    # it redraws the mass with the angles, so no stage freezes a virtuality and
    # none of them has a conditional normalisation to divide out.
    #
    # A *wrong* Z_hat does not cancel: the per-angle stage divides out the true
    # Z_k whatever weight it is given (rescaling w_k by anything that does not
    # depend on the angles leaves its accepted distribution unchanged), so the
    # residual bias of the tabulated scheme is exactly Z_hat/Z. That is why
    # sequential_global_retry exists -- it stops the per-angle stage normalising at
    # all, and then Z_hat cancels identically and only sets the efficiency.

    @staticmethod
    def _z_slot_keys(particles, slot_to_index):
        """Table key of each slot: its pdg and which occurrence of that pdg it
        is. Slots of one pdg are consecutive and in production order, which is
        also how _draw_one_decay picks a decay file when there is one file per
        identical parent -- so the two agree on which slot draws from what."""
        keys = []
        seen = collections.defaultdict(int)
        for index in slot_to_index:
            pdg = particles[index].pid
            keys.append('%s_%s' % (pdg, seen[pdg]))
            seen[pdg] += 1
        return keys

    def _zhat(self, key, mass):
        """The tabulated Z_k at a sampled virtuality. 1 when no table has been
        built (the max-weight probe itself, which is what measures it, and every
        non-offshell mode)."""
        table = (getattr(self, '_z_tables', None) or {}).get(key)
        if not table:
            return 1.0
        if mass < table['zero_below']:
            return 0.0
        lo, hi = table['range']
        # held constant outside the probed range rather than extrapolated: the
        # fit is only constrained where the Breit-Wigner put samples
        u = math.log(min(max(mass, lo), hi) / table['pole'])
        c = table['coeff']
        return math.exp(c[0] + u * (c[1] + u * c[2]))

    @staticmethod
    def _weighted_polyfit2(xs, ys, ws):
        """Weighted least-squares quadratic y = c0 + c1 x + c2 x^2, by the normal
        equations. Returns None when the system is too close to degenerate to
        solve meaningfully. Small and self contained -- numpy is not imported
        here, and this runs once per slot at the end of the max-weight scan, so
        a 3x3 solve in plain python costs nothing worth optimising.

        The pivot tolerance is *relative* to the size of the matrix: the fit
        variable is ln(m/pole), which spans about +-0.13 over a Breit-Wigner
        window, so the moments of x^4 are ~1e-4 of the moments of x^0 and an
        absolute threshold would mean something different for every resonance.
        A relative one rejects exactly the case that matters -- every bin at the
        same virtuality, where the quadratic is not determined -- and accepts
        the normal conditioning of this fit (~3e4, which double precision
        handles with ten digits to spare).
        """
        n = len(xs)
        if n < 3:
            return None
        # symmetric 3x3 normal matrix, moments of x up to 4
        moment = [sum(w * x ** k for x, w in zip(xs, ws)) for k in range(5)]
        rhs = [sum(w * y * x ** k for x, y, w in zip(xs, ys, ws)) for k in range(3)]
        mat = [[moment[i + j] for j in range(3)] + [rhs[i]] for i in range(3)]
        tolerance = 1e-12 * max(abs(value) for row in mat for value in row[:3])
        for col in range(3):       # gaussian elimination with partial pivoting
            pivot = max(range(col, 3), key=lambda r: abs(mat[r][col]))
            if abs(mat[pivot][col]) <= tolerance:
                return None
            mat[col], mat[pivot] = mat[pivot], mat[col]
            for row in range(3):
                if row == col:
                    continue
                factor = mat[row][col] / mat[col][col]
                for k in range(col, 4):
                    mat[row][k] -= factor * mat[col][k]
        return [mat[i][3] / mat[i][i] for i in range(3)]

    def _build_z_tables(self, z_samples, nb_bin=20, min_per_bin=20):
        """Fit ln Z_k(m) from the (virtuality, rate factor) pairs the max-weight
        probe collected, one table per slot key.

        The samples are binned in m and *averaged* -- Z_k is an expectation, so
        the mean of the samples estimates it while their logarithm would not --
        then a quadratic in ln(m/pole) is fitted through the bin means, weighted
        by their counts. The fit is what smooths the tails, where the
        Breit-Wigner leaves few samples; the accepted lineshape is sensitive to
        the *slope* of ln Z, and a fractional error there survives as the same
        fractional error on the shift it corrects.

        Bins whose mean is zero sit below the decay threshold (no pool event can
        be reshuffled onto that virtuality): they are excluded from the fit and
        recorded as ``zero_below``, which then makes the mass-set stage reject
        those virtualities outright.
        """
        tables = {}
        for key, samples in sorted(z_samples.items()):
            if len(samples) < 4 * min_per_bin:
                logger.warning("MadSpin sequential: only %d probe samples for "
                               "slot %s, not tabulating its rate factor",
                               len(samples), key)
                continue
            pole = self.banner.get('param', 'mass',
                                   abs(int(key.split('_')[0]))).value
            samples = sorted(samples)
            lo, hi = samples[0][0], samples[-1][0]
            if not pole or hi <= lo:
                continue
            width = (hi - lo) / float(nb_bin)
            bins = [[0, 0.0, 0.0] for _ in range(nb_bin)]   # count, sum m, sum s
            for mass, value in samples:
                index = min(nb_bin - 1, int((mass - lo) / width))
                bins[index][0] += 1
                bins[index][1] += mass
                bins[index][2] += value
            zero_below = 0.0
            points = []
            for count, sum_mass, sum_value in bins:
                if count < min_per_bin:
                    continue
                mean_mass = sum_mass / count
                mean_value = sum_value / count
                if mean_value <= 0:
                    # entirely below threshold: everything up to here is dead
                    zero_below = max(zero_below, mean_mass)
                    points = []
                    continue
                points.append((math.log(mean_mass / pole),
                               math.log(mean_value), count))
            coeff = self._weighted_polyfit2([p[0] for p in points],
                                            [p[1] for p in points],
                                            [float(p[2]) for p in points])
            if coeff is None:
                logger.warning("MadSpin sequential: could not fit the "
                               "rate factor of slot %s (%d usable bins)",
                               key, len(points))
                continue
            fit = lambda u: math.exp(coeff[0] + u * (coeff[1] + u * coeff[2]))
            residual = max(abs(math.exp(y) / fit(u) - 1) for u, y, _ in points)
            # normalise to 1 at the pole: a constant multiplies every mass set
            # alike, so it is absorbed by the maximum weight, and it makes the
            # table readable against the running width it estimates
            coeff[0] = 0.0      # ``fit`` closes over coeff: normalised from here
            tables[key] = {'pole': pole, 'coeff': coeff,
                           'zero_below': zero_below,
                           'range': (max(lo, zero_below), hi)}
            logger.info("MadSpin sequential: slot %s rate factor "
                        "Z(%.5g)=%.3f  Z(%.5g)=1  Z(%.5g)=%.3f "
                        "(%d samples, %d bins, bin/fit deviation up to %.1f%%)",
                        key, lo, fit(math.log(max(lo, zero_below) / pole)),
                        pole, hi, fit(math.log(hi / pole)),
                        len(samples), len(points), 100 * residual)
        return tables

    def _check_weight_identity(self, production, decays, decay_dict, w_seq,
                               helicities, stats, offshell=True, keep_jac=True,
                               parents=None):
        """sequential_debug: the identity the whole decomposition rests on,
        checked on the accepted chain instead of inferred from a distribution.

            w_mass_raw * prod_k w_k_raw  ==  prod_i n_i * wgt_joint

        with w_mass_raw the mass-set weight *before* the tabulated Z_hat and
        w_k_raw each slot's weight before it is divided back out -- Z_hat
        cancels between the two stages, so this tests the decomposition itself
        and not the quality of the table. ``wgt_joint`` is recomputed here by the
        joint code, on copies, for the same production event, the same
        virtualities and the same decays.

        A statistical A/B can only bound a bias at the level its Monte Carlo
        error allows, and needs an error model to say even that. This is
        deterministic: any scheme whose per-chain weight product is not the
        joint weight is wrong, whatever a lineshape comparison happens to show.
        """
        prod_copy = lhe_parser.Event(str(production))
        decays_copy = collections.defaultdict(list)
        jac_bw = 1.0
        index = 0
        for pdg, decay_list in decays.items():
            for decay in decay_list:
                copy = lhe_parser.Event(str(decay))
                if not offshell and parents is not None:
                    # PA hands back its accepted decays already boosted to the
                    # lab frame -- _slot_density boosts them in place, and that
                    # is the frame add_decays wants -- while
                    # calculate_matrix_element_from_density does that boost
                    # itself. Undo it so the joint route starts where it
                    # expects to. (Offshell takes its density on a copy, so
                    # there the drawn decay is still in its rest frame.)
                    copy.boost(lhe_parser.FourMomentum(parents[index]))
                mass = getattr(decay[0], 'new_mass', None)
                if mass is not None:
                    copy[0].new_mass = mass
                    copy[0].reshuffle_info = decay[0].reshuffle_info
                decays_copy[pdg].append(copy)
                index += 1
        # the Breit-Wigner sampling jacobians: the joint path folds them in
        # itself when it draws the masses, and here the masses are given, so
        # they are recomputed from the same (pole, width, window) the draw used
        for pdg, decay_list in decays.items():
            for decay in decay_list:
                info = getattr(decay[0], 'reshuffle_info', None)
                if info is None:
                    continue        # no virtuality was sampled for this slot
                pole, width, min_mass, max_mass = info
                gap = math.atan((pole ** 2 - min_mass ** 2) / pole / width)
                gap += math.atan((max_mass ** 2 - pole ** 2) / pole / width)
                jac_bw *= gap / math.pi
        full_me, _, prod_diag, dec_diag, jac_reshuffle = \
            self.calculate_matrix_element_from_density(prod_copy, decays_copy,
                                                       decay_dict)
        w_joint = full_me / (prod_diag * dec_diag) * jac_reshuffle * jac_bw
        if not offshell and keep_jac:
            # PA's reshuffling jacobian is not inside
            # calculate_matrix_element_from_density: the pole approximation
            # leaves the momenta onshell there (it returns jac_reshuffle = 1)
            # and the joint path takes the jacobian from a reshuffle of the
            # *complete* event, done outside. Recomputing it that way is what
            # makes this an independent check of the two pieces the chain used
            # instead -- the mass stage's _production_jacobian_for and the
            # per-slot _decay_reshuffle_jacobian -- whose product it must equal.
            rebuilt = collections.defaultdict(list)
            for pdg, decay_list in decays.items():
                for decay in decay_list:
                    copy = lhe_parser.Event(str(decay))
                    copy[0].new_mass = decay[0].new_mass
                    copy[0].reshuffle_info = decay[0].reshuffle_info
                    rebuilt[pdg].append(copy)
            full_evt = lhe_parser.Event(str(production)).add_decays(rebuilt)
            jac_full = full_evt.reshuffle_production(_allow_retry=False)
            if jac_full in (0, -1):
                # the chain kept this mass set, so this should not happen; skip
                # the chain rather than compare against a meaningless number
                logger.debug('sequential_debug: the accepted mass set does not '
                             'reshuffle through the joint route, chain skipped')
                return
            w_joint *= jac_full
        nb_hel = 1
        for hel in helicities:
            nb_hel *= len(hel)
        if not w_joint:
            return
        # What must hold is *proportionality*, not equality: the chain weight and
        # the joint weight differ by a constant -- the number of helicity states,
        # and whatever normalisation the density path applies to the decay matrix
        # elements relative to calculate_matrix_element -- and any constant is
        # absorbed by the bounds. So accumulate the ratio and let the report look
        # at its spread: a constant ratio *is* the identity, whatever its value,
        # while a scheme that samples the wrong distribution has a ratio that
        # varies chain to chain.
        ratio = w_seq / (nb_hel * w_joint)
        stats['nb_identity_check'] += 1
        stats['identity_ratio_sum'] += ratio
        stats['identity_ratio_sqsum'] += ratio * ratio

    def sequential_accept_reject(self, production, evt_decayfile, maxwgts,
                                 nb_remain, stats=None, probe=None,
                                 probe_extra=None, decay_dict=None):
        """Accept/reject one decaying particle at a time, in density mode.

        Returns the accepted ``decays`` dict (pdg -> list of decay events, in
        the order add_decays expects), or None if the production event has
        nothing to decay.

        Exactness: slot k is accepted with probability w_k / C_k where

            w_k = (N_k / N_{k-1}) * jac_bw_k * jac_dec_k * (J_k / J_{k-1})

        with jac_bw_k the Breit-Wigner sampling jacobian of slot k's virtuality,
        jac_dec_k the jacobian of reshuffling slot k's decay onto it, and J_k the
        production reshuffling jacobian with the slots drawn so far offshell. The
        N and J factors telescope and the per-slot ones multiply out, so the
        product reproduces the joint weight (which gets jac_dec_k from the same
        reshuffle_production call that gives it J). On a reject only *that* slot
        is redrawn; the slots already accepted are kept. See
        MADSPIN_SEQUENTIAL_PLAN.md.

        Failure handling follows the scope of the failure. Under
        ``sequential_with_mass``, where slot k draws its own mass, a mass its
        decay products cannot accommodate is redrawn on the spot; a mass *set*
        the production cannot reshuffle is only knowable once every slot has a
        mass, so it trashes the whole set and restarts the chain. In the
        up-front schemes the mass set is fixed before the loop, so the same
        decay-side failure is a rejection of that decay instead.

        Every scheme but ``sequential_with_mass`` splits this in two: a mass-set
        accept/reject first, then the per-angle loop above. The mass stage
        carries everything that depends on the virtualities but not on the
        angles -- the Breit-Wigner sampling jacobians, the production
        reshuffling jacobian, and offshell also the offshell production trace --
        so those are evaluated once per mass set instead of once per slot trial,
        which is the whole point of drawing the masses first. The per-angle loop
        then redraws until it accepts and so divides out its own normalisation
        Z_k(m), a function of the sampled virtuality -- hence the tabulated
        ``_zhat`` factor in the mass-set weight, without which the accepted
        resonance lineshape is the Breit-Wigner one. Under
        ``sequential_global_retry`` a rejected decay trashes the mass set, the
        per-angle stage stops normalising, and Z_hat cancels from the chain
        (leaving it a pure efficiency preconditioner).

        ``probe``: when a list is given nothing is ever rejected and each slot's
        w_k is appended to it instead. That is how the max-weight scan measures
        the bounds -- on exactly the weights this loop will later test, since it
        is this same code computing them. The weights are recorded *before*
        ``_zhat`` is applied (the probe is what measures it, so it does not
        exist yet); ``probe_extra`` carries the virtualities and the rate-factor
        samples the scan needs to build it.
        """
        if getattr(self, '_decay_groups', None):
            # Loud on purpose: this loop redraws one slot until it is accepted,
            # which would divide E[w_k | group] out of the chain and distort the
            # group fractions. _sequential_active refuses the whole scheme when
            # the decays are grouped; if that guard is ever lifted it must come
            # with a bound and a rate factor per group, not with this loop as it
            # stands. See doc/madspin_decay_groups.md section 4.4.
            raise Exception("MadSpin: the per-particle accept/reject cannot "
                            "honour '@' decay groups; this should have fallen "
                            "back to the joint one.")
        decays_key = self._decaying_pdgs(production, evt_decayfile)
        if not decays_key:
            return None


        if not hasattr(self, 'f2py_module'):
            self.f2py_module = [0, 0] # first index is production, second is decay
            self.pdg2prefix = [0, 0]
            all_prefix = [0, 0]
            all_pdg = [0, 0]
            all_procid = [0, 0]

            self.create_and_initialise_f2py_modules(all_prefix, all_pdg, all_procid)

        prod_static = getattr(production, '_ms_density_static', None)
        if not prod_static or prod_static.get('decays_key') != decays_key:
            prod_static = self._density_basis(production, decays_key)
            production._ms_density_static = prod_static

        helicities = prod_static['helicities']
        init_part = prod_static['init_part']
        order = self._decay_slot_order(prod_static['decaying_spins'])
        particles, slot_to_index = self._sequential_slots(production, decays_key)
        ids = [p.pid for p in particles]

        # madspin/full evaluate the production density at reshuffled (offshell)
        # momenta that couple all decay masses, so rho is drawn per chain (after
        # the up-front reshuffle) rather than once at onshell. PA/onshell keep a
        # fixed onshell rho, cached on the production event.
        offshell = self._sequential_offshell()
        # Whether the virtualities are drawn before the angle loop. True for
        # every scheme but sequential_with_mass, which draws each slot's mass
        # inside that slot's own accept/reject.
        # sequential_global_retry: reject the mass set on a rejected decay
        # instead of redrawing that decay, so the per-angle stage never
        # normalises. See the Z_k discussion above _z_slot_keys.
        # two_stage: one bound over all the angles instead of one per particle,
        # the mass set paying for a rejection either way -- the joint
        # accept/reject with a mass-set stage in front of it. Exact for the same
        # reason sequential_global_retry is (nothing is redrawn in place, so no
        # stage normalises itself) and it keeps Z_hat only as a preconditioner,
        # since it cancels between the two stages.
        mode = self._unweighting_mode()
        upfront = mode not in ('joint', 'sequential_with_mass')
        joint_angles = upfront and mode == 'two_stage'
        exact = upfront and mode == 'sequential_global_retry'
        zkeys = self._z_slot_keys(particles, slot_to_index) if upfront else None
        # |M_prod|^2 on shell: the denominator the joint offshell weight divides
        # by (calculate_matrix_element_from_density evaluates it *before*
        # reshuffle_production and returns it as prod_diag). It depends on the
        # production event only -- which the chain never redraws -- so leaving it
        # out would not bias anything, but it would leave the absolute scale of
        # the production matrix element inside the mass-set weight while its
        # bound is a single number shared by every production event: the loud
        # kinematics would set the bound and then overflow it, the quiet ones
        # would pay for it in acceptance. Cached on the event under the name the
        # joint path already uses for the same quantity.
        # frame the helicity basis is defined in (run_card me_frame), shared by
        # the production density and by every decay contracted against it. The
        # offshell branch gets its own from _upfront_production, derived from
        # the reshuffled production rho is evaluated at.
        frame_boost = None
        me_prod_on = 1.0
        if offshell:
            me_prod_on = getattr(production, 'me_wgt', None)
            if not me_prod_on:
                me_prod_on = self.calculate_matrix_element(production)
                production.me_wgt = me_prod_on
            if not me_prod_on:
                # a production event with no matrix element cannot be normalised
                # to itself; leave the weight unscaled rather than divide by zero
                logger.debug('sequential: |M_prod|^2 = 0, mass weight unscaled')
                me_prod_on = 1.0
        density_prod = None
        if not offshell:
            # The frame has to travel with the cached rho: every decay density
            # is contracted against that rho, so it must be taken in the same
            # frame. Cached alongside it rather than recomputed, because the
            # max-weight probe calls this hundreds of times per production event
            # -- and recomputing it only on a cache *miss* would leave it None on
            # every call after the first, silently contracting lab-frame decay
            # densities against an me_frame production one.
            density_prod = getattr(production, '_ms_density_prod', None)
            if density_prod is None:
                frame_boost = self._frame_boost(production)
                density_prod = self.get_density(production, prod_static['position'],
                                                prod_static['allowed_hel'],
                                                prod_static['ncomb'],
                                                prod_static['dimension'],
                                                frame_boost=frame_boost)
                production._ms_density_prod = density_prod
                production._ms_frame_boost = frame_boost
            else:
                frame_boost = getattr(production, '_ms_frame_boost', None)

        # PA samples a virtuality per resonance; onshell does not. 2 -> 1
        # production has no recoil phase space for RAMBO to redistribute.
        nb_prod_final = sum(1 for p in production if int(p.status) == 1)
        draw_mass = (self.options['spinmode'] == 'PA' and nb_prod_final > 1)
        # Whether the production reshuffling jacobian enters the accept/reject
        # weight. Follows the joint path (interface_madspin.py, get_onshell PA
        # block): on by default -- when off the reshuffle is a post-acceptance
        # kinematic dressing and only the Breit-Wigner sampling jacobian is in
        # the weight. The feasibility of the mass set is still checked either
        # way, to trigger the whole-set restart.
        keep_jac = draw_mass and self.options['density_keep_jacobian']

        if stats is None:
            stats = collections.defaultdict(int)
        if probe is not None and probe_extra is None:
            probe_extra = {}

        while True:     # restart point: an impossible/rejected production mass set
            parents = init_part
            jac_prod = 1.0
            slot_mass = {}
            if upfront:
                # draw every virtuality, then settle whatever depends on the
                # mass set alone: offshell that is the production reshuffle and
                # rho, under PA the production reshuffling jacobian
                setup = self._upfront_production(production, order, particles,
                                                 slot_to_index, prod_static,
                                                 offshell, draw_mass=draw_mass,
                                                 density_prod=density_prod,
                                                 frame_boost=frame_boost)
                if setup is None:
                    stats['nb_production_restart'] += 1
                    continue
                density_prod, jac_prod, slot_mass, parents, frame_boost = setup

                # Mass-set accept/reject, before the per-angle loop. All the
                # factors that depend on the mass set but not the decay angles --
                # the production reshuffling jacobian, the Breit-Wigner sampling
                # jacobians, and offshell the production trace -- go here, so the
                # per-angle loop no longer carries them (that bundling made slot
                # 0's acceptance ~1/300 offshell, and cost PA one production
                # reshuffling per slot trial). See MADSPIN_SEQUENTIAL_PLAN.md
                # sections 10 and 11.
                if offshell:
                    # Tr(rho_off)/|M_prod|^2_on -- the offshell production matrix
                    # element over the onshell one, which is what the joint weight
                    # carries. Applied in probe mode too, unlike Z_hat: it is known
                    # before the scan, so the bound is measured on the same quantity
                    # the accept/reject will test. jac_prod is the jacobian of the
                    # reshuffle that produced rho_off.
                    w_mass = density_prod.trace().real / me_prod_on * jac_prod
                else:
                    # PA/onshell: rho is the onshell one and cancels between N_n
                    # and N_0, so nothing of the production matrix element rides
                    # here. jac_prod is the reshuffling jacobian of the whole
                    # mass set -- the factor the per-slot scheme re-evaluates on
                    # every trial and telescopes. Under density_keep_jacobian =
                    # False the reshuffle is a post-acceptance dressing and is
                    # not in the weight at all; only its feasibility was checked.
                    w_mass = jac_prod if keep_jac else 1.0
                for s in slot_mass:
                    w_mass *= slot_mass[s][2]
                # before Z_hat, which cancels between the two stages:
                # this is what the weight-identity check compares
                w_mass_raw = w_mass
                if probe is not None:
                    del probe[:]            # start this chain's probe vector
                    probe.append(float(w_mass))
                    probe_extra['keys'] = zkeys
                    probe_extra['order'] = list(order)
                    probe_extra['mass'] = [slot_mass[s][0] if s in slot_mass
                                           else 0.0
                                           for s in range(len(order))]
                    # not reset with the rest: a chain that ends up restarting
                    # still drew valid (virtuality, rate factor) pairs, and Z_k
                    # wants every one of them -- including the zeros of a
                    # virtuality below threshold, which is precisely where
                    # dropping them would bias the table upwards
                    probe_extra.setdefault('z', [])
                else:
                    # Z_k(m_k): what the per-angle stage will divide out again.
                    # Without it the accepted virtualities are Breit-Wigner
                    # distributed instead of physically distributed.
                    for s in slot_mass:
                        w_mass *= self._zhat(zkeys[s], slot_mass[s][0])
                if probe is None and maxwgts and slot_mass:
                    # no virtuality to unweight means w_mass is the constant 1
                    # (onshell, and 2 -> 1 production under PA): testing it
                    # against its bound would only throw chains away
                    if w_mass > maxwgts[0]:
                        stats['nb_overflow_mass'] += 1
                    if random.random() * maxwgts[0] >= w_mass:
                        stats['nb_mass_reject'] += 1
                        continue            # redraw the whole mass set

            # Angle stage. The mass set, the production density and its
            # reshuffling jacobian are all fixed above and are *reused* by
            # every pass of this loop -- which is the whole point of drawing the
            # virtualities first: the joint accept/reject pays a production
            # reshuffling (and, offshell, a production density matrix) on every
            # trial, because a rejection there redraws the masses too.
            #   two_stage: a rejected angle set is redrawn
            #     against the same mass set, so this loop is where the reuse
            #     happens -- and, redrawing to acceptance, it normalises itself,
            #     which is what the Z_hat factor in w_mass compensates.
            #   sequential_global_retry: a rejected *decay* costs the mass set, which
            #     makes Z_hat cancel and that scheme exact whatever the table
            #     says, at the price of the reuse.
            while True:
                slot_densities = {}
                slot_decays = {}
                slot_masses = {}
                n_prev = self._partial_density_contraction(density_prod, helicities, {})
                j_prev = 1.0
                budget = production.sqrts
                restart = False
                w_angles = 1.0      # joint_angles: the product tested once, below
                angle_dead = False  # a zero member: reject the set, stop drawing
                w_slots = 1.0       # product of the raw per-slot weights

                for position, slot in enumerate(order):
                    index = slot_to_index[slot]
                    particle = particles[index]
                    # the up-front schemes reserve maxwgts[0] for the mass set,
                    # so their per-slot bounds start at index 1
                    wpos = position + 1 if upfront else position
                    if joint_angles:
                        # every slot contributes to the single angle weight, tested
                        # once the last one has been drawn
                        maxwgt = None
                    elif maxwgts:
                        maxwgt = maxwgts[wpos] if wpos < len(maxwgts) \
                                               else maxwgts[-1]
                    else:
                        maxwgt = None
                    nb_infeasible = 0
                    while True:
                        stats['nb_try_%d' % position] += 1
                        decay = self._draw_one_decay(particle, index, ids,
                                                     evt_decayfile, nb_remain)

                        if upfront:
                            mass = slot_mass.get(slot)
                            me_on = 1.0
                            dcopy = None
                            jac_dec = 1.0
                            if offshell:
                                # madspin/full: offshell numerator over onshell
                                # denominator. The mass was drawn up front, so the
                                # decay is reshuffled to it.
                                me_on = self.calculate_matrix_element(decay)   # |M_dec|^2_on
                                decay[0].new_mass, decay[0].reshuffle_info = \
                                    mass[0], mass[1]
                                # The offshell density is taken on a copy: the drawn
                                # decay must stay in its onshell rest frame (only tagged
                                # with new_mass) so the final add_decays + a single
                                # reshuffle_production rebuild consistent kinematics.
                                # Reshuffling/boosting it in place leaves it on the
                                # offshell parent and add_decays then rejects it.
                                dcopy = lhe_parser.Event(str(decay))
                                dcopy[0].new_mass = mass[0]
                                dcopy[0].reshuffle_info = mass[1]
                                # jac_dec_k: the decay reshuffling jacobian. Joint
                                # madspin has it (calculate_matrix_element_from_density,
                                # 'jac *= dec.reshuffle_decayevt()'), so it belongs in
                                # the per-slot weight here -- it depends on this slot's
                                # decay only, hence no telescoping ratio.
                                jac_dec = dcopy.reshuffle_decayevt()
                            elif mass is not None:
                                # PA: the decay stays onshell, only *tagged* with
                                # the virtuality that add_decays and the final
                                # reshuffle_production will consume, exactly as
                                # the per-slot mass draw leaves it. Its
                                # reshuffling jacobian is probed on a copy, and
                                # it is the same factor the joint PA weight picks
                                # up inside its full-event reshuffle_production.
                                decay[0].new_mass, decay[0].reshuffle_info = \
                                    mass[0], mass[1]
                                jac_dec = self._decay_reshuffle_jacobian(decay)
                            if jac_dec in (0, -1):
                                # This decay cannot be mapped onto the sampled
                                # virtuality (its products do not fit). That is a
                                # zero-weight candidate, i.e. an ordinary rejection
                                # of *this decay*, not of the mass set: a zero is
                                # part of Z_k, so counting it as a rejection here is
                                # what makes the tabulated Z_k the exact correction
                                # near a threshold. It is recorded as such in the
                                # probe. The fail-safe covers a virtuality no decay
                                # in the pool can reach -- with the table in place
                                # the mass stage rejects those outright, since Z_k
                                # vanishes there.
                                stats['nb_infeasible_%d' % position] += 1
                                if probe is not None:
                                    probe_extra['z'].append(
                                        (zkeys[slot], mass[0], 0.0))
                                elif joint_angles:
                                    # A zero anywhere makes the whole angle set
                                    # weight zero, so the set is rejected: stop
                                    # drawing the remaining slots. Redrawing
                                    # just this slot instead would propose from
                                    # the *feasible* part of the pool, and the
                                    # normalisation the mass stage compensates
                                    # for would become Z_k/(1 - q_k(m)) -- a
                                    # different function of the virtuality than
                                    # the tabulated one.
                                    w_angles = 0.0
                                    angle_dead = True
                                    break
                                elif exact:
                                    # Same argument for the per-slot restart
                                    # scheme: a zero-weight decay has to reject
                                    # the mass set rather than be redrawn here,
                                    # or the feasible fraction 1 - q_k(m) is
                                    # divided out of the accepted mass sets --
                                    # the same class of bias as Z_k itself, and
                                    # it would survive precisely where Z_k is
                                    # supposed to vanish.
                                    stats['nb_exact_restart'] += 1
                                    restart = True
                                    break
                                nb_infeasible += 1
                                if nb_infeasible < 200:
                                    continue
                                stats['nb_production_restart'] += 1
                                restart = True
                                break
                            # the frame rho was evaluated in, whichever branch
                            # produced it: the decay densities are contracted
                            # against rho, so they have to be taken there too
                            if offshell:
                                # per-angle factor only: (N_k/N_{k-1}) * jac_dec_k
                                # * Tr(D_off)/|M_dec|^2_on. jac_bw and the
                                # *production* reshuffling jacobian are in w_mass.
                                density = self._slot_density(dcopy, parents[slot],
                                                             helicities[slot],
                                                             frame_boost=frame_boost)
                                rate = jac_dec * (density.trace().real / me_on)
                            else:
                                # PA/onshell: the matrix elements are on shell, so
                                # there is no offshell/onshell rate ratio -- the
                                # only angle-dependent factor left over the density
                                # ratio is the decay reshuffling jacobian. With
                                # density_keep_jacobian off, joint PA does not put
                                # it in its weight either (the reshuffle runs after
                                # acceptance), so neither does this; a decay that
                                # cannot reach the virtuality is still a zero, and
                                # Z_k then measures the feasible fraction.
                                density = self._slot_density(decay, parents[slot],
                                                             helicities[slot],
                                                             frame_boost=frame_boost)
                                rate = jac_dec if keep_jac else 1.0
                            slot_densities[slot] = density
                            n_k = self._partial_density_contraction(
                                            density_prod, helicities, slot_densities)
                            wgt = (n_k / n_prev).real * rate
                            wgt_raw = wgt        # before any Z_hat division
                            j_k, new_budget = j_prev, budget
                            # Z_hat_k(m_k), or 1 where there is no virtuality to
                            # condition on (onshell, 2 -> 1 production under PA)
                            zhat = self._zhat(zkeys[slot], mass[0]) \
                                   if mass is not None else 1.0
                            if probe is not None:
                                probe.append(float(wgt))
                                # E[rate | m] = E[w_k | m] = Z_k(m) -- the pool
                                # average of the density ratio is one at fixed m,
                                # so the two have the same expectation and rate
                                # carries no polarisation modulation, which makes
                                # it the tighter estimator of the two.
                                if mass is not None:
                                    probe_extra['z'].append(
                                        (zkeys[slot], mass[0], float(rate)))
                                accept = True
                            elif joint_angles:
                                # no test here: every slot feeds the single angle
                                # weight, tested once the last decay is drawn
                                w_angles *= wgt / zhat if zhat > 0 else 0.0
                                accept = True
                            elif maxwgt is None:
                                accept = True
                            else:
                                if exact:
                                    # the mass stage already paid Z_hat_k, so the
                                    # bound this is tested against is the one of
                                    # w_k/Z_hat_k -- flat in the virtuality, and the
                                    # two factors cancel over the chain
                                    wgt = wgt / zhat if zhat > 0 else 0.0
                                if wgt > maxwgt:
                                    stats['nb_overflow_%d' % position] += 1
                                    logger.debug('sequential: slot %s weight %s above'
                                                 ' its max %s', position, wgt, maxwgt)
                                accept = random.random() * maxwgt < wgt
                            if accept:
                                slot_decays[slot] = decay
                                n_prev = n_k
                                w_slots *= wgt_raw
                                break
                            slot_densities.pop(slot, None)
                            if exact:
                                # Redrawing this decay until it is accepted would
                                # normalise the per-angle stage and divide Z_k(m)
                                # out of the accepted mass sets. Rejecting the mass
                                # set instead keeps the chain acceptance
                                # proportional to the joint weight -- exact whatever
                                # Z_hat is, at the cost of throwing the slots
                                # already accepted away.
                                stats['nb_exact_restart'] += 1
                                restart = True
                                break
                            continue

                        jac_bw = 1.0        # Breit-Wigner *sampling* jacobian
                        jac_dec = 1.0       # decay *reshuffling* jacobian
                        new_budget = budget
                        if draw_mass:
                            # decay-side failure is local to this slot: redraw its
                            # mass, keep every slot already accepted
                            while True:
                                new_budget, jac_bw = self._draw_offshell_mass(
                                                    particle.pdg, decay, budget)
                                jac_dec = self._decay_reshuffle_jacobian(decay)
                                if jac_dec not in (0, -1):
                                    break
                                stats['nb_mass_redraw_%d' % position] += 1
                            slot_masses[slot] = (decay[0].new_mass,
                                                 getattr(decay[0], 'reshuffle_info', None))
                            if not keep_jac:
                                # joint PA bundles jac_dec with J_prod: both come out
                                # of the single reshuffle_production, which under
                                # density_keep_jacobian = False runs only after the
                                # event is accepted and so enters no weight. Mirror
                                # that here, or the two schemes stop matching.
                                jac_dec = 1.0

                        # The production reshuffling jacobian only enters the
                        # weight under density_keep_jacobian; then it is needed per
                        # trial. Otherwise its sole use is spotting a mass set the
                        # production cannot reshuffle, and that depends on the whole
                        # set, so it is checked once after the chain is complete --
                        # not here, where the reshuffle-on-a-copy dominated the cost.
                        j_k = j_prev
                        if keep_jac:
                            j_probe = self._production_jacobian_for(production,
                                                                   slot_to_index,
                                                                   slot_masses)
                            if j_probe in (0, -1):
                                stats['nb_production_restart'] += 1
                                restart = True
                                break
                            j_k = j_probe

                        # accepted slots reuse their stored (already normalised)
                        # density; only this slot's decay is evaluated here
                        slot_densities[slot] = self._slot_density(
                                        decay, init_part[slot], helicities[slot],
                                        frame_boost=frame_boost)
                        n_k = self._partial_density_contraction(density_prod, helicities,
                                                                slot_densities)
                        # jac_dec is this slot's own factor (it depends on this
                        # decay only), so unlike J it enters without a ratio -- the
                        # product over slots is what the joint reshuffle_production
                        # multiplies in.
                        wgt = (n_k / n_prev).real * jac_bw * jac_dec * (j_k / j_prev)
                        if probe is not None:
                            # python float: these are marshalled as JSON when the
                            # scan runs across forked workers
                            probe.append(float(wgt))
                            accept = True
                        else:
                            if wgt > maxwgt:
                                # the bound was under-estimated: this biases
                                # silently, so it has to be visible
                                stats['nb_overflow_%d' % position] += 1
                                logger.debug('sequential: slot %s weight %s above '
                                             'its max %s', position, wgt, maxwgt)
                            accept = random.random() * maxwgt < wgt
                        if accept:
                            slot_decays[slot] = decay
                            n_prev, j_prev, budget = n_k, j_k, new_budget
                            break
                        # rejected: this slot only, drop what it contributed
                        slot_densities.pop(slot, None)
                        slot_masses.pop(slot, None)
                    if restart or angle_dead:
                        break
                if joint_angles and not restart and probe is None and maxwgts:
                    # One bound over the product of every slot's weight, instead
                    # of one bound per slot: the per-slot bounds' product is
                    # looser than a single bound on the product, and this buys
                    # that back at the price of the early exit a per-slot test
                    # gets when the first particle is rejected.
                    stats['nb_angleset_try'] += 1
                    c_angles = maxwgts[1] if len(maxwgts) > 1 else maxwgts[-1]
                    if w_angles > c_angles:
                        stats['nb_overflow_angles'] += 1
                        logger.debug('sequential: angle weight %s above its max %s',
                                     w_angles, c_angles)
                    if random.random() * c_angles >= w_angles:
                        stats['nb_angle_reject'] += 1
                        # keep the mass set, and with it the reshuffled
                        # production and its density matrix: only the decays are
                        # drawn again. That reuse is the point of this scheme.
                        continue
                break
            if (not restart and not upfront and draw_mass and not keep_jac
                    and probe is None):
                # sequential_with_mass, jacobian off: the masses are only all
                # known once the chain is complete, so the feasibility of the
                # set is checked here -- one reshuffle for the whole chain
                # instead of one per trial. The up-front schemes settled it
                # before the angle loop.
                if self._production_jacobian_for(production, slot_to_index,
                                                 slot_masses) in (0, -1):
                    stats['nb_production_restart'] += 1
                    restart = True
            if restart and probe is not None:
                # a partial probe vector was appended for this chain; drop it so
                # the next attempt records a clean [w_mass, w_0, ...] vector
                del probe[:]
            if not restart:
                break

        # back to the pdg -> list layout add_decays consumes, in slot order
        decays = collections.defaultdict(list)
        for slot in range(len(order)):
            decays[particles[slot_to_index[slot]].pid].append(slot_decays[slot])
        if (upfront and probe is None and decay_dict
                and self.options['sequential_debug']):
            self._check_weight_identity(production, decays, decay_dict,
                                        w_mass_raw * w_slots, helicities, stats,
                                        offshell, keep_jac, parents)
        return decays

    def get_onshell_evt_and_wgt(self, production, decays, decay_dict, prod_density_cached=None, build_event=True):
        """ return the onshell wgt for the production event associated to the decays
            return also the full event with decay. 
            Carefull this modifies production event (pass to the full one)
            build_event: if False (density mode) compute weight without building event"""
        #print("\n\n\n\n\n======== debug get_onshell_evt_and_wgt =========")
        density_pole_approximation = self.options['spinmode'] in ['PA', 'onshell']
        density_do_reshuffle = self.options['spinmode'] == 'PA'
        decay_me = 1.0
        decay_me_debug = 1.0
        jac = 1.0
        tag, order = production.get_tag_and_order()
        try:
            info = self.generate_all.all_me[tag]
        except:
            misc.sprint(self.generate_all.all_me)
            misc.sprint(production)
            misc.sprint(decays)
            raise
        
        # Calculate decay ME
        if self.generate_all.mode == 'onshell':
            #print(f"len(decays) = {len(decays)}")
            for pdg in decays:
                for dec in decays[pdg]:
                    #print(f"dec = {dec}")
                    decay_me *= self.calculate_matrix_element(dec)
        else:
            if self.options['density_debug']:
                #print(f"len(decays) = {len(decays)}")
                for pdg in decays:
                    for dec in decays[pdg]:
                        #print(f"dec = {dec}")
                        decay_me_debug *= self.calculate_matrix_element(dec)

        # Calculate production*decay ME
        if self.generate_all.mode == 'onshell':
            full_event = lhe_parser.Event(str(production))
            full_event = full_event.add_decays(decays)
            #print(f"full_event = {full_event}")
            full_me = self.calculate_matrix_element(full_event)
            #print(f"full_me = {full_me}")
        else:
            #offshell mode
            full_dqrts = production.sqrts
            jac = 1
            # 2 -> 1 production: the single resonance virtuality is fully fixed
            # by sqrt(shat); there is no recoil phase space, so RAMBO cannot
            # redistribute a sampled Breit-Wigner mass. Keep the resonance
            # onshell at the production-determined mass (no reshuffling) instead
            # of building kinematically inconsistent momenta.
            nb_prod_final = sum(1 for p in production if int(p.status) == 1)
            if nb_prod_final > 1 and (not density_pole_approximation or
                    density_do_reshuffle):
                for pdg in decays:
                    for dec in decays[pdg]:
                        full_dqrts, jac_dec = self._draw_offshell_mass(
                                                    pdg, dec, full_dqrts)
                        jac *= jac_dec
            if prod_density_cached is None:
                full_me, prod_density_cached, prod_diag, dec_diag, jac_reshuffle = self.calculate_matrix_element_from_density(production, decays, decay_dict)
            else:
                full_me, _, prod_diag, dec_diag, jac_reshuffle = self.calculate_matrix_element_from_density(production, decays, decay_dict, prod_density_cached)
            # The internal reshuffle (offshell/madspin) is the reshuffle of the
            # chain; fold its jacobian into the weight here so the caller does not
            # reshuffle the already-offshell event a second time.
            jac *= jac_reshuffle
            #print(f"full_me from density = {full_me}")
   
            full_event = None
            if build_event or self.options['density_debug']:
                # Create full event from production and decays
                if density_pole_approximation:
                    full_event = lhe_parser.Event(str(production))
                else:
                    full_event = production          
                # CAUTION: the next line removes everything from decays dictionary
                full_event = full_event.add_decays(decays)
            
                #print(f"full event 2 = {full_event}")
                if self.options['density_debug']:
                    me1 = self.calculate_matrix_element(full_event)
                    #print(f"me1 = {me1} , me2 = {full_me} , ratio = {me1/full_me}")
                    if abs(1-me1/full_me) > self.options['density_tolerance']:
                        print(f"full = {me1} , density = {full_me} , ratio = {me1/full_me}")	    
                        print(full_event)
                        print(production)
                        print(decays)

                        print("ERROR matrix element from density does not match with full matrix element")	
                        raise RuntimeError("ERROR matrix element from density does not match with full matrix element")	 
    
        # Calculate production ME and cache it so that if we reject 
        # the decay the production ME will not be recalculated
        if hasattr(production, 'me_wgt'):
            production_me = production.me_wgt
        else:
            production_me = self.calculate_matrix_element(production) if self.generate_all.mode == 'onshell' \
                            else prod_diag
            production.me_wgt = production_me

        if self.generate_all.mode == 'density' and self.options['density_debug']:
            prod_me = self.calculate_matrix_element(production)
            #print(f"prod_diag = {prod_diag} , prod_me = {prod_me}")
            if abs(1-prod_diag/prod_me) > self.options['density_tolerance']:
                print(f"prod_me = {prod_me} , prod_diag = {prod_diag} , ratio = {prod_diag/prod_me}")	    
                raise RuntimeError("ERROR production matrix element from density does not match with diagonal")	     
            if abs(1-dec_diag/decay_me_debug) > self.options['density_tolerance']:
                print(f"decay_me = {decay_me_debug} , dec_diag = {dec_diag} , ratio = {dec_diag/decay_me_debug}")	    
                raise RuntimeError("ERROR decay matrix element from density does not match with diagonal")	   
        
        if self.generate_all.mode == 'density':
            decay_me = dec_diag

        #print(f"full_event = {full_event}")
        #print(f"full_me = {full_me}")
        #print(f"production_me = {production_me}")
        #print(f"decay_me = {decay_me}")
        #print(f"wgt = {full_me/(production_me*decay_me)}")
        
        return full_event, full_me/(production_me*decay_me)*jac, prod_density_cached


    def initialise_f2py_module(self, mymod, sp_path, prod_or_decay):
        """ Routine to initialise the fortran module with module.initialise(param_card_path).
            If one the process is at loop-induced level, it is also needed to call module.set_madloop_path(path_to_MadLoop5_resources)
        """
        if prod_or_decay == 'prod':
            folder_name = self.ms_me_subdir
        elif prod_or_decay == 'decay':
            folder_name = self.ms_me_decay_subdir
        else:
            raise ValueError("prod_or_decay only accepts values as 'prod' or 'decay'.")

        with misc.chdir(sp_path): #changed the search of the card to the subdirectories madspin_me and madspin_decay
            if (not os.path.exists(pjoin(self.path_me, folder_name, 'Cards', 'param_card.dat'))
                    and os.path.exists(pjoin(self.path_me, folder_name, 'param_card.dat'))):
                mymod.initialise(pjoin(self.path_me, 'param_card.dat'))
            else:
                mymod.initialise(pjoin(self.path_me, folder_name, 'Cards', 'param_card.dat'))
            # If the module is loop-induced, we also need to set the directory in which the MadLoop param card is present
            MadLoopCardPath = pjoin(self.path_me, folder_name, 'SubProcesses', 'MadLoop5_resources')
            if os.path.exists(MadLoopCardPath):
                MLCard = banner.MadLoopParam(pjoin(MadLoopCardPath, 'MadLoopParams.dat'))
                MLCard.set("HelicityFilterLevel", 0) # HelicityFilterLevel is set to 0 because the computation of density matrices loop-induced requires it.
                MLCard.set("MLStabThres", 0.001)

                # Every forked unweighting worker runs this lazily, so N workers
                # rewrite this one shared file while sibling workers' MadLoop
                # Fortran init is reading it. An in-place write lets a reader see
                # a truncated card: MLReductionLib stays all-zero and MadLoop
                # answers with STOP "No available loop reduction lib ...", which
                # (a Fortran STOP) kills the worker with exit code 0, bypassing
                # every Python handler. Write to a private temp file and rename,
                # so a concurrent reader sees either the old or the new card,
                # never a partial one.
                _ml_dat = pjoin(MadLoopCardPath, 'MadLoopParams.dat')
                _ml_tmp = '%s.tmp%d' % (_ml_dat, os.getpid())
                MLCard.write(_ml_tmp)
                os.replace(_ml_tmp, _ml_dat)
                mymod.set_madloop_path(MadLoopCardPath)

        # the beam polarisation is constant over a run, so it is pushed into
        # the library once per module rather than passed on every call
        self._set_f2py_beampol(mymod)


    def create_f2py_module(self, sp_path, prod_or_decay, all_prefix, all_pdg, all_procid):
        """ Load the density-matrix f2py extensions and build the pdg -> prefix
            map, once. Both the matrix-element evaluation and get_density / get_pdir
            need it, so the sequential accept/reject -- which calls get_density
            directly, without going through calculate_matrix_element_from_density --
            must be able to trigger the same setup.
            The setup is done independently for the production part and the decay part
            and are stored in self.f2py_module[0] (production), self.f2py_module[1] (decay)
        """
        if sys.path[0] != sp_path:
            sys.path.insert(0, sp_path)
        
        if prod_or_decay == "prod":
            i = 0
            menum = 2
        elif prod_or_decay == "decay":
            i = 1
            menum = 1
        else:
            raise ValueError("The only acceptable values of prod_or_decay are 'prod' and 'decay'")

        # production and decay are built with distinct MENUM (2 vs 1) so
        # their f2py modules / dependent libraries don't clash in-process
        # (see decay_all_events_onshell.compile).
        mymod = self._load_f2py_matrix_module(sp_path, menum=menum)
        self.f2py_module[i] = mymod

        all_prefix[i] = self.f2py_module[i].get_prefix()
        all_pdg[i], all_procid[i] = self.f2py_module[i].get_pdg_order()
        self.pdg2prefix[i] = {}
        for j, pdg in enumerate(all_pdg[i]):
            pdg = tuple([x for x in pdg if x != 0])
            self.pdg2prefix[i][pdg] = (str(all_prefix[i][j].decode()).strip(), j)
        

        if self.model_init_prod and prod_or_decay == 'prod':
            self.model_init_prod = False
            self.initialise_f2py_module(mymod, sp_path, prod_or_decay='prod')

        if self.model_init_decay and prod_or_decay == 'decay':
            self.model_init_decay = False
            self.initialise_f2py_module(mymod, sp_path, prod_or_decay='decay')


    def create_and_initialise_f2py_modules(self, all_prefix, all_pdg, all_procid):
        """ Routine to create the f2py modules and to initialise them. It separates production and decay.
            It also fills all_prefix, all_pdg, all_procid which are lists of 2 elements. 
            The first element is the value for the production part, the second element is for the decay part.
        """
        try:
            sp_path_prod = pjoin(self.path_me, self.ms_me_subdir, 'SubProcesses')
            self.create_f2py_module(sp_path_prod, 'prod', all_prefix, all_pdg, all_procid)
        except:
            logger.critical("Error while creating the f2py modules for the production part.")

        # legacy options 'onshell_v1' and 'madspin_v1' store both the production and the decay in a single folder
        if self.options['spinmode'] not in ['onshell_v1', 'madspin_v1']:
            try:
                sp_path_decay = pjoin(self.path_me, self.ms_me_decay_subdir, 'SubProcesses')
                self.create_f2py_module(sp_path_decay, 'decay', all_prefix, all_pdg, all_procid)
            except:
                logger.critical("Error while creating the f2py modules for the decay part.")


    def calculate_matrix_element_from_density(self, production, decays, decay_dict, prod_density_cached=None):
        """routine to return the matrix element from density matrices"""

        # ------------------------------------------------------------------
        # Load f2py module and build pdg2prefix map if needed (unchanged logic)
        # ------------------------------------------------------------------
        # Since we need to compute the density matrix for both the production and the decay, we need to import two fortran modules
         
        if not hasattr(self, 'f2py_module'):
            self.f2py_module = [0, 0] # first index is production, second is decay
            self.pdg2prefix = [0, 0]
            all_prefix = [0, 0]
            all_pdg = [0, 0]
            all_procid = [0, 0]

            self.create_and_initialise_f2py_modules(all_prefix, all_pdg, all_procid)


        # ------------------------------------------------------------------
        # Cache production-only metadata reused across rejection retries
        # ------------------------------------------------------------------
        decays_key = tuple(decays.keys())
        MEdenom_prod, MEdenom_decay = None, None
        # Reshuffling jacobian of the internal (offshell/madspin) reshuffle. This
        # is THE reshuffle of the chain: the caller must fold it into the weight
        # rather than reshuffling the already-offshell event a second time. It
        # stays 1.0 for the pole-approximation path (which reshuffles later, after
        # acceptance) and for 2 -> 1 production (no phase space to redistribute).
        jac_reshuffle = 1.0
        prod_static = getattr(production, '_ms_density_static', None)
        density_pole_approximation = self.options['spinmode'] in ['PA', 'onshell']
        density_do_reshuffle = self.options['spinmode'] == 'PA'
        if not density_pole_approximation or \
            (not prod_static or prod_static.get('decays_key') != decays_key):
            prod_static = self._density_basis(production, decays_key)
            production._ms_density_static = prod_static

            use_new_mass = (
                not density_pole_approximation or
                density_do_reshuffle
            )
            if use_new_mass:
                new_mass = {}
                reshuffle_info = {}
                for key in decays:
                    new_mass[key] = [getattr(dec[0], 'new_mass', dec[0].mass)
                                     for dec in decays[key]]
                    # Carry the Breit-Wigner sampling info (pole, width, min, max)
                    # alongside new_mass so reshuffle_production can re-sample this
                    # mass if needed. Without it the reshuffling retry crashes on
                    # p.reshuffle_info (e.g. complex-mass-scheme runs).
                    reshuffle_info[key] = [getattr(dec[0], 'reshuffle_info', None)
                                           for dec in decays[key]]

                for particle in production:
                    if particle.status == 1 and particle.pid in new_mass:
                        particle.new_mass = new_mass[particle.pid].pop(0)
                        info = reshuffle_info[particle.pid].pop(0)
                        if info is not None:
                            particle.reshuffle_info = info
            else:
                for particle in production:
                    if hasattr(particle, 'new_mass'):
                        del particle.new_mass
                    if hasattr(particle, 'reshuffle_info'):
                        del particle.reshuffle_info

            #VALENTIN: except for the mode "full", we should not compute the matrix element here
            MEdenom_prod, MEdenom_decay = None, None
            if not density_pole_approximation:
                # compute the denominator and then reshuffle the event before 
                # computing the numerator 
                MEdenom_prod = self.calculate_matrix_element(production)  
                MEdenom_decay = 1.0              
                for key in decays:
                    for dec in decays[key]:
                        MEdenom_decay *= self.calculate_matrix_element(dec)
                # now doing the reshuffling
                # doing the reshuffling for each part:
                jac = 1.0
                jac *= production.reshuffle_production()
                # reshuffle_production may have resampled a resonance's new_mass
                # (its jac==-1 retry) so the set of masses fits sqrt(shat). Push the
                # (possibly resampled) production masses back onto the matching decay
                # so each decay is reshuffled to the SAME mass as its production
                # resonance; otherwise the two diverge and rebuilding the full event
                # later hits an inconsistent-boost assertion. The i-th production
                # resonance of a given pid corresponds to decays[pid][i] (same
                # ordering used when new_mass was assigned above).
                resampled = {key: [] for key in decays}
                for particle in production:
                    if (particle.status == 1 and particle.pid in resampled
                            and hasattr(particle, 'new_mass')):
                        resampled[particle.pid].append(particle.new_mass)
                for key in decays:
                    for dec, mass in zip(decays[key], resampled[key]):
                        dec[0].new_mass = mass
                for key in decays:
                    for dec in decays[key]:
                        jac *= dec.reshuffle_decayevt()
                if jac == 0:
                    raise Exception
                # hand the reshuffling jacobian back to the caller (folded into
                # the accept/reject weight) instead of discarding it.
                jac_reshuffle = jac

        iden_p = prod_static['iden_p']
        sym_factor_prod_ident = prod_static['sym_factor_prod_ident']
        init_part = prod_static['init_part']
        position = prod_static['position']
        helicities = prod_static['helicities']
        allowed_hel = prod_static['allowed_hel']
        ncomb = prod_static['ncomb']
        dimension = prod_static['dimension']

        # ------------------------------------------------------------------
        # Normalization
        # ------------------------------------------------------------------
        dec_diag = 1.0
        prod_color = 1
        prod_denominators = 1
        # GET_INTER returns each density matrix with the standalone matrix
        # element's IDEN division already applied.  The density
        # contraction below predates applies its own normalization, 
        # so restore one IDEN factor per matrix before
        # using that denominator.
        density_iden_prod = iden_p * sym_factor_prod_ident
        density_iden_decay = 1

        # frame the helicity basis is defined in (run_card me_frame); shared by
        # the production and by every decay, otherwise the two sides of the
        # contraction below would not be in the same basis. None unless the
        # beams are polarised -- see the comment above _beampol.
        frame_boost = self._frame_boost(production)

        density_prod = self.get_density(production,
                                        position,
                                        allowed_hel,
                                        ncomb,
                                        dimension,
                                        frame_boost=frame_boost) \
            if prod_density_cached is None else prod_density_cached

        # ------------------------------------------------------------------
        # Symmetry factor:
        # For each parent-PDG group with N identical parents and decay-channel
        # multiplicities {n_k}, the factor that belongs to the denominator is:
        #   sym_group = (Π_k n_k!) / (N!)
        # and sym_factor_decay = Π_groups sym_group.
        # ------------------------------------------------------------------
        sym_factor_decay = 1.0

        # Canonical decay-channel signature: sorted final-state PDGs only.
        def _decay_signature(dec_evt):
            pdgs = []
            for p in dec_evt:
                if p.status == 1:
                    pdgs.append(int(p.pid))
            pdgs.sort()
            return tuple(pdgs)
        
        # ------------------------------------------------------------------
        # Build total decay density matrix as tensor product
        # ------------------------------------------------------------------
        decaying_idx = 0
        density_dec = None

        for pdg, decay_event_list in decays.items():
            N = len(decay_event_list)

            # decay symmetry for this PDG group
            if N > 1:
                # Fast path for N==2 avoids building multiplicity maps.
                if N == 2:
                    if _decay_signature(decay_event_list[0]) != _decay_signature(decay_event_list[1]):
                        sym_factor_decay *= 0.5
                else:
                    sig_counts = {}
                    for evt in decay_event_list:
                        sig = _decay_signature(evt)
                        sig_counts[sig] = sig_counts.get(sig, 0) + 1
                    sym = 1
                    for nk in sig_counts.values():
                        if nk > 1:
                            sym *= math.factorial(nk)
                    sym_factor_decay *= (sym / float(math.factorial(N)))

            # particle properties for this parent PDG
            width = decay_dict[pdg][0]
            mass = decay_dict[pdg][1]
            color = decay_dict[pdg][2]
            spin = decay_dict[pdg][3]

            for i_decay_event in range(N):
                current_decay_event = decay_event_list[i_decay_event]

                # boost to lab frame using corresponding production particle momentum
                part = init_part[decaying_idx + i_decay_event]
                boost = -1 * lhe_parser.FourMomentum(part)
                boost.E *= -1
                current_decay_event.boost(boost)

                density_dec_tmp = self.get_density(
                    current_decay_event,
                    position=[1],
                    allow_hel=helicities[decaying_idx + i_decay_event],
                    ncomb=len(helicities[decaying_idx + i_decay_event]),
                    dimension=len(helicities[decaying_idx + i_decay_event]),
                    frame_boost=frame_boost,
                    frame_rest_leg=None if frame_boost is None
                                   else self._decay_frame_rest_leg(part, frame_boost)
                )

                if density_dec is None:
                    density_dec = density_dec_tmp
                else:
                    density_dec = density_dec.tensor_product(density_dec_tmp)

                if MEdenom_decay is None:
                    dec_diag *= density_dec_tmp.trace().real
                density_iden_decay *= color * spin
                prod_color *= color
                D = complex(0, mass * width)
                prod_denominators *= (D * D.conjugate())
            

            decaying_idx += N

        # ------------------------------------------------------------------
        # Contract production and decay density matrices
        # ------------------------------------------------------------------
        me = density_dec.scalar_multiplication(density_prod)
        me *= density_iden_prod * density_iden_decay

        # ------------------------------------------------------------------
        # include production identical-final-state symmetry factor
        # ------------------------------------------------------------------
        denominator = iden_p * sym_factor_prod_ident * prod_color * prod_denominators * sym_factor_decay
        me = me.real / denominator

        #print(f"production = {production}")
        #print(f"decays = {decays}")
        if MEdenom_prod is None:
            prod_diag = density_prod.trace().real 
        else: 
            prod_diag = MEdenom_prod
        if MEdenom_decay is not None:
            dec_diag *= MEdenom_decay
        return me, density_prod, prod_diag, dec_diag, jac_reshuffle


    def get_density_matrix_indices(self, nhel_decay):
        #print("------")
        #print(f"get_density_matrix_indices , nhel_decay = {nhel_decay}")
        diag = [sum(range(nhel_decay, nhel_decay - i, -1)) for i in range(nhel_decay)]
        off_diag = [i for i in list(range(nhel_decay * (nhel_decay + 1) // 2)) if i not in diag]
        return diag, off_diag

    def get_density_matrix_element_from_label(matrix, label):
        if label in label_to_index:
            i, j = label_to_index[label]
            return matrix[i, j]
        else:
            raise ValueError(f"Label {label} is not valid for this matrix size: {matrix.shape}.")

    def get_allowed_hel(self, list_hels):
        # list_hels is a list of lists with all possible helicities of the decaying particles, e.g.
        # [[1,-1], [1,0,-1]] - we need to construct a list of lists with all possible helicities
        # [ [1,1] , [1,0], [1,-1], [-1,1], ... ] which should eventually be converted into a flat list
        # [ 1, 1, 1, 0, 1, -1, ... ]
        key = tuple(tuple(hels) for hels in list_hels)
        # Cache allowed helicities - they depend only on spins
        # avoid rebuilding allowed helicities per trial
        if not hasattr(self, '_allowed_hel_cache'):
            self._allowed_hel_cache = {}
        if key in self._allowed_hel_cache:
            return self._allowed_hel_cache[key]

        helicity_combinations = [list(l) for l in product(*list_hels)]
        concatenated_hel_list = list(chain.from_iterable(helicity_combinations))
        out = (helicity_combinations, concatenated_hel_list)
        self._allowed_hel_cache[key] = out
        return out  

    def _beampol(self):
        """The (pol1, pol2) actually in force, or None for unpolarised beams.

        |beampol| runs from 1 (unpolarised) to 2 (fully polarised), so anything
        at or below 1 means no polarisation -- the same test the matrix elements
        make, so that an out-of-range value cannot switch on the frame boost
        here while the Fortran ignores it.
        """
        pol = self.options.beampol_me()
        if abs(pol[0]) <= 1. and abs(pol[1]) <= 1.:
            return None
        return pol

    def _set_f2py_beampol(self, mymod):
        """Push the beam polarisation into the matrix-element library once per
        module. The value is constant over a run and ``get_density`` is on the
        hot path, so this is a setter rather than a per-call argument."""
        pol = self._beampol()
        if pol is None:
            # the library defaults to unpolarised (BLOCK DATA BEAMPOL_DEFAULT)
            return
        if not hasattr(mymod, 'py_set_beampol'):
            logger.warning('The matrix elements of this MadSpin run predate the '
                           'beam-polarisation support of the density modes; '
                           'beampol=%s will be ignored. Regenerate the process '
                           'directory to enable it.', list(pol))
            return
        mymod.py_set_beampol(pol[0], pol[1])

    def _frame_boost(self, event):
        """The 4-momentum whose rest frame ``frame_id`` selects for ``event``,
        or None when the frame machinery cannot change anything.

        ``frame_id`` is the bitmask the run_card builds as
        ``sum(2**n for n in me_frame)``, so external leg n (counted from 1, in
        the matrix element's own ordering) is selected by bit n -- the same
        convention ``mapid`` uncompresses with ``btest(id, i)``. The returned
        momentum is the sum of the selected legs, ready to be handed to
        ``Event.boost`` / ``_boost_momenta``, which negate the spatial part
        themselves (HELAS ``boostx``, exactly what ``boost_to_frame`` does in
        driver.f).
        """
        if self._beampol() is None:
            return None
        frame_id = int(self.options['frame_id'])
        if frame_id <= 0:
            return None
        _, orig_order, _, _ = self.get_pdir(event)
        momenta = event.get_momenta(orig_order)
        selected = [n for n in range(1, len(momenta) + 1) if frame_id >> n & 1]
        if not selected:
            return None
        pboost = lhe_parser.FourMomentum()
        for n in selected:
            pboost += lhe_parser.FourMomentum(momenta[n - 1])
        # A single selected leg has to end up exactly at rest: vxxxxx branches
        # on pp.eq.rZero and takes the frame z axis as quantisation axis there,
        # so a residual 1d-14 three-momentum left by the boost arithmetic would
        # silently pick a different polarisation state (see the same fix in
        # boost_to_frame, Template/LO/SubProcesses/genps.f).
        if len(selected) == 1:
            pboost.rest_leg = selected[0]
            mom = momenta[selected[0] - 1]
            pboost.rest_leg_mom = (mom[0], mom[1], mom[2], mom[3])
        else:
            pboost.rest_leg = None
            pboost.rest_leg_mom = None
        return pboost

    @staticmethod
    def _decay_frame_rest_leg(parent, frame_boost):
        """1 when ``frame_id`` selects exactly this resonance, so leg 1 of its
        decay matrix element has to be forced to zero three-momentum; None
        otherwise. Same rounding argument as in ``_frame_boost``."""
        rest_mom = getattr(frame_boost, 'rest_leg_mom', None)
        if rest_mom == (parent.E, parent.px, parent.py, parent.pz):
            return 1
        return None

    @staticmethod
    def _boost_momenta(momenta, pboost, rest_leg=-1):
        """``boost_to_frame``: every momentum of ``momenta`` into the rest frame
        of ``pboost``, as (E, px, py, pz) tuples.

        This works on the momenta rather than on the event, so a decay event
        stays where the rest of MadSpin needs it -- in the lab, which is what
        ``add_decays`` and the reshuffling assume. ``rest_leg`` (1-based, -1 to
        take it from ``pboost``) is the leg the frame is built from when it is a
        single one, forced exactly at rest.
        """
        neg = lhe_parser.FourMomentum(pboost.E, -pboost.px, -pboost.py, -pboost.pz)
        out = []
        for mom in momenta:
            new = lhe_parser.FourMomentum(mom).boost(neg)
            out.append((new.E, new.px, new.py, new.pz))
        if rest_leg == -1:
            rest_leg = getattr(pboost, 'rest_leg', None)
        if rest_leg is not None and rest_leg <= len(out):
            out[rest_leg - 1] = (out[rest_leg - 1][0], 0., 0., 0.)
        return out

    def get_density(self, event, position, allow_hel, ncomb, dimension,
                    frame_boost=None, frame_rest_leg=-1):
        """``frame_boost`` is the momentum whose rest frame ``frame_id`` picks
        (see ``_frame_boost``); the momenta are boosted there before the matrix
        element sees them, which is what defines the axis the initial-state
        helicities -- the ones ``beampol`` reweights -- are quantised along.

        The *same* momentum is used for the production and for every decay
        contracted against it. A decay event reaches this point already boosted
        into the lab (by its parent's momentum), so applying the frame boost to
        its momenta here composes the two in the right order and leaves both
        sides of the contraction in one helicity basis. ``frame_rest_leg``
        names the leg to force exactly at rest; the default takes it from
        ``frame_boost``, which is right for a production event, and the decay
        callers pass ``_decay_frame_rest_leg``'s answer instead.
        """

        orig_order = getattr(event, '_ms_orig_order_for_density', None)
        if orig_order is None:
            _, orig_order, _, _, tag = self.get_pdir(event)
            event._ms_orig_order_for_density = orig_order
        else: #in any case, we need tag to differentiate between production and decay
            tag, _ = event.get_tag_and_order()


        # Fast path: single-point momentum extraction without permutation construction.
        try:
            p = event.get_momenta(orig_order)
        except Exception:
            # Safety fallback for unusual event structures.
            all_p = event.get_all_momenta(orig_order)
            assert len(all_p) == 1, "Error: get_density can only be called for a single phase-space point"
            p = all_p[0]
        if frame_boost is not None:
            p = self._boost_momenta(p, frame_boost, rest_leg=frame_rest_leg)
        P = rwgt_interface.ReweightInterface.invert_momenta(p) 
        pdgs =list(orig_order[0])+list(orig_order[1])
        n_changing = len(position)
        if n_changing == 0:
            raise ValueError("Error in get_density: 'position' must contain at least one position index")
        if len(allow_hel) % n_changing != 0:
            raise ValueError("Error in get_density: inconsistent 'allow_hel' and 'position' lengths")
        
        # PY_GET_DENSITY(PDGS, PROCID, P, POS, ALLOW_HEL, ALPHAS, SCALE2)
        if self.all_me[tag]['type'] == 'production':
            # misc.sprint("Computation of the production density matrix")
            density_array = self.f2py_module[0].py_get_density(pdgs=pdgs, 
                                                                procid=-1, 
                                                                p=P, 
                                                                pos=position, 
                                                                allow_hel=allow_hel, 
                                                                alphas=event.aqcd,
                                                                scale2=event.scale**2)

        elif self.all_me[tag]['type'] == 'decay':
            # misc.sprint("Computation of the decay density matrix")
            density_array = self.f2py_module[1].py_get_density(pdgs=pdgs, 
                                                                procid=-1, 
                                                                p=P, 
                                                                pos=position, 
                                                                allow_hel=allow_hel, 
                                                                alphas=event.aqcd,
                                                                scale2=event.scale**2)
        else:
            raise ValueError("The key 'type' of sel.all_me can only take as values 'production' or 'decay'.")


        #print(f"density_array = {density_array}") 
        density_matrix = madspin.DensityMatrix(density_array, 
                                               n_changing, 
                                               allow_hel, 
                                               dimension)
        return density_matrix

   
    def get_inter_value(self,event,nhel):
        """routine to return all the possible inter for an event"""
        
        pdir,orig_order = self.get_pdir(event)
        	
        if pdir in self.all_amp:
            all_p = event.get_all_momenta(orig_order)
            for p in all_p:
#                print(pdir,'Momenta=',p)
                P = rwgt_interface.ReweightInterface.invert_momenta(p)
#               print("Momenta =",P,"\n")
                IC = [1]*len(p)
                amp = []
                jamp = []
                inter = []
           
                for i,hel in enumerate(nhel):
                    #print(f"hel = {hel}")		
                    amp.append(self.all_amp[pdir](P,hel,IC))
                    jamp.append(self.all_jamp[pdir](amp[i]))
                #print(f"len(jamp) = {len(jamp)}")
                for i in range(len(jamp)): 
                    for j in range(len(jamp)): 
                        inter.append(self.all_inter[pdir](jamp[i],jamp[j]))
                return inter
        else : 
            self.all_amp[pdir],self.all_jamp[pdir],self.all_inter[pdir],self.all_matrix[pdir]= self.get_mymod(pdir,'INTER')

        return self.get_inter_value(event,nhel) 


    def get_nhel(self,event,position):

        pdir,orig_order, prefix, pos, tag = self.get_pdir(event)
        if pdir in self.all_nhel:
            iden,NHEL = self.all_nhel[pdir]
            if position == -1:
                return iden
            nhel = rwgt_interface.ReweightInterface.invert_momenta(NHEL)
            groups = {} 
            nhel = sorted(nhel) 
            for item in nhel:
                a = item.copy()
                del a[position]
                t = tuple(a)
                groups.setdefault(t, []).append(item)
                grouped = list(groups.values())
            return grouped,iden
        else:
            #transer nhel information from fortran to wrapper
            getattr(self.f2py_module, '%sget_nhel_entry' % prefix.lower())()
            #transer now to python dictionary
            nhel = getattr(getattr(self.f2py_module, '%sprocess_nhel' % prefix.lower()), '%snhel' %prefix.lower())
            iden = getattr(self.f2py_module, 'get_idens')()[pos]
            self.all_nhel[pdir] = (iden, nhel)
            return self.get_nhel(event,position)


    def get_iden(self, event):
        # DEBUGGING REMOVE
        #print("---- DEBUG ---")
        #pdgs, allproc = self.f2py_module.get_pdg_order()
        #idens = self.f2py_module.get_idens()
        #
        #print("len(pdgs) =", len(pdgs))
        #print("len(idens) =", len(idens))
        #   
        #for i in range(len(idens)):
        #    print(i, pdgs[i], idens[i])
        #print("--- END")
        # END REMOVE

        # get_pdir returns (pdir, orig_order, prefix, pos, tag)
        _, _, _, pos, tag = self.get_pdir(event)

        if self.all_me[tag]['type'] == 'production':
            idens = self.f2py_module[0].get_idens()
        elif self.all_me[tag]['type'] == 'decay':
            idens = self.f2py_module[1].get_idens()
        else:
            raise ValueError("The key 'type' of self.all_me can only take as values 'production' or 'decay'.")

        #print(f"idens = {idens} , pos = {pos}")
        return idens[pos]
    

    def get_mymod(self,pdir,MODE): 
        
        all_prefix = self.f2py_module.get_prefix()
        tag = [t for t in self.all_me if self.all_me[t]['pdir'] == pdir][0]
        return 



    def get_pdir(self,event): 
        tag, order = event.get_tag_and_order()
#        print(order)
        try:
            orig_order = self.all_me[tag]['order']
        except Exception:
            # try to pass to full anti-particles for 1->N
            init, final = tag
            if len(init) == 2:
                raise
            init = (-init[0],)
            final = tuple(-i for i in final)
            tag = (init, final)
            orig_order = self.all_me[tag]['order']
        pdir = self.all_me[tag]['pdir']

        if self.all_me[tag]['type'] == 'production':
            prefix, pos = self.pdg2prefix[0][tuple(list(orig_order[0]) + list(orig_order[1]))]
        elif self.all_me[tag]['type'] == 'decay':
            prefix, pos = self.pdg2prefix[1][tuple(list(orig_order[0]) + list(orig_order[1]))]
        else:
            raise ValueError("The key 'type' of self.all_me can only take as values 'production' or 'decay'.")
        #misc.sprint(f"get_pdir: pdir = {pdir} , orig_order = {orig_order} , prefix = {prefix}")
        return pdir,orig_order, prefix, pos, tag

    # Two model_init are used, one for the production and one the decay (to support LO decay + NLO production or different models for each side)
    model_init_prod = True
    model_init_decay = True
    model_init = True
    def calculate_matrix_element(self, event):
        """routine to return the matrix element"""        
        
        tag, order = event.get_tag_and_order()
        try:
            orig_order = self.all_me[tag]['order']
        except Exception:
            # try to pass to full anti-particles for 1->N
            init, final = tag
            if len(init) == 2:
                raise
            init = (-init[0],)
            final = tuple(-i for i in final)
            tag = (init, final)
            orig_order = self.all_me[tag]['order']
        pdir = self.all_me[tag]['pdir']
        if pdir in self.all_f2py:
            all_p = event.get_all_momenta(orig_order)
            #print(f"len identical = {len(all_p)} , p = {all_p}")
            if self.options['identical_particle_in_prod_and_decay'] == "crash" and\
                len(all_p)> 1:
                raise Exception("Ambiguous particle in production and decay. crash as requested by 'identical_particle_in_prod_and_decay'")
            out = 0
            for p in all_p:
                #print(f"Before : {p}")
                p = rwgt_interface.ReweightInterface.invert_momenta(p)
                #print(f"After : {p}")
                #print(event[0])
                #print(event[0].color1)
                #print(event.aqcd)
                if event[0].color1 == 599 and event.aqcd==0:
                    new_value = self.all_f2py[pdir](p, 0.113, 0)
                else:
                    new_value = self.all_f2py[pdir](p, event.aqcd, event.scale, -1)
                #if the process is Loop-Induced, smatrixhel returns the tuple (value, returncode), we need to keep only the value
                if isinstance(new_value, tuple):
                    new_value = new_value[0]
                if self.options['identical_particle_in_prod_and_decay'] == "average":
                    out += new_value
                else:
                    if abs(out)< abs(new_value):
                        out = new_value
                if self.options['identical_particle_in_prod_and_decay'] == 'first':
                    return out
            if self.options['identical_particle_in_prod_and_decay'] == "average":
                return out/len(all_p)
            else:
                return out
        else:            
            # First time we see a new ``pdir`` for this MadSpin instance:
            # load the freshly-compiled f2py extension once and cache a
            # smatrixhel lambda per pdir. The .so / pdg2prefix only need
            # to be loaded the first time we hit this branch — subsequent
            # ``pdir``s reuse ``self.f2py_module`` so we don't re-trigger
            # the spec-from-file-location load (which is fine on Linux
            # but wasteful, and on macOS would re-walk the install_name
            # bookkeeping every time).

            # Valentin: we only pass here with the options 'onshell_v1' and 'madspin_v1' for which I kept only one madspin_me directory
            # it is not adapted to modes where the directories for production and decays are separated
            if not hasattr(self, 'f2py_module'):
                sp_path = pjoin(self.path_me, self.ms_me_subdir, 'SubProcesses')
                if sys.path[0] != sp_path:
                    sys.path.insert(0, sp_path)

                mymod = self._load_f2py_matrix_module(sp_path)
                self.f2py_module = mymod

                all_prefix = self.f2py_module.get_prefix()
                all_pdg, all_procid = self.f2py_module.get_pdg_order()
                self.pdg2prefix = {}
                for i, pdg in enumerate(all_pdg):
                    pdg = tuple([x for x in pdg if x != 0])
                    self.pdg2prefix[tuple(pdg)] = (str(all_prefix[i].decode()).strip(), i)

                if self.model_init:
                    self.model_init = False
                    with misc.chdir(sp_path):
                        if not os.path.exists(pjoin(self.path_me, 'Cards','param_card.dat')) and \
                                os.path.exists(pjoin(self.path_me,'param_card.dat')):
                            mymod.initialise(pjoin(self.path_me,'param_card.dat'))
                        else:
                            mymod.initialise(pjoin(self.path_me, 'Cards','param_card.dat'))
            mymod = self.f2py_module

            #if Rpath linking is not working the below code can be an alternative:
            #import ctypes
            #exts = ['so','dylib','dll']
            #for ext in exts:
            #    me_library = pjoin(self.path_me, self.ms_me_subdir, 'SubProcesses', pdir, 'libme%s.%s' % (pdir, ext))
            #    if os.path.exists(me_library):
            #        break
            # ctypes.CDLL(me_library)

            pdg = list(orig_order[0]) + list(orig_order[1])

            if self.options['spinmode'] in ['onshell_v1', 'madspin_v1']:
                self.all_f2py[pdir] = lambda *args : mymod.smatrixhel(pdg, 0, *args)
            else:
                if self.all_me[tag]['type'] == 'production':
                    self.all_f2py[pdir] = lambda *args : mymod[0].smatrixhel(pdg, 0, *args)
                elif self.all_me[tag]['type'] == 'decay':
                    self.all_f2py[pdir] = lambda *args : mymod[1].smatrixhel(pdg, 0, *args)
                else:
                    raise ValueError("The key 'type' of sel.all_me can only take as values 'production' or 'decay'.")

            return self.calculate_matrix_element(event)
        
        
    def generate_all_matrix_element(self):
        
        # 1. compute the production matrix element -----------------------------
        processes = [line[9:].strip() for line in self.banner.proc_card
                     if line.startswith('generate')]
        processes += [' '.join(line.split()[2:]) for line in self.banner.proc_card
                      if re.search(r'^\s*add\s+process', line)]
        # 2. compute the decay matrix-element
        decay_text = []
        processes_decay = []
        for decays in self.list_branches.values():
            for decay in  decays:
                if '=' not in decay:
                    decay += ' QCD=99'
                if ',' in decay:
                    decay_text.append('(%s)' % decay)
                else:
                    decay_text.append(decay)
                processes_decay.append(decay)            
        decay_text = ', '.join(decay_text)
        processes += []
        
        #handle NLO
        new_processes = []
        for proc in processes:
            # deal with @ syntax need to move it after the decay specification
            if '@' in proc:
                proc, proc_nb = proc.split('@')
                try:
                    proc_nb = int(proc_nb)
                except ValueError:
                    raise MadSpinError('MadSpin didn\'t allow order restriction after the @ comment: \"%s\" not valid' % proc_nb)
                proc_nb = '@ %i' % proc_nb 
                if self.options['global_order_coupling']:
                    proc_nb = '%s %s' % (proc_nb, self.options['global_order_coupling'])
            else:
                if self.options['global_order_coupling']:
                    proc_nb = '@0 %s ' % self.options['global_order_coupling']    
                
            rwgt_interface.ReweightInterface.get_LO_definition_from_NLO()        
        
        raise Exception

if __name__ == '__main__':
    
    a = MadSpinInterface()
    a.cmdloop()
    
    


        
