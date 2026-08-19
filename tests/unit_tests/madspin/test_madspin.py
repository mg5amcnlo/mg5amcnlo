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

"""Unit test library for the spin correlated decay routines
in the madspin directory"""

from __future__ import absolute_import
import sys
import os
import string
import shutil
pjoin = os.path.join

from subprocess import Popen, PIPE, STDOUT

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.interface.master_interface as Cmd
import madgraph.various.banner as banner

import copy
import array
import collections
import inspect
import math
import random

import madgraph.core.base_objects as MG
import madgraph.various.misc as misc
import MadSpin.decay as madspin
import madgraph.various.lhe_parser as lhe_parser
import MadSpin.interface_madspin as interface_madspin
import models.import_ufo as import_ufo


from madgraph import MG5DIR, MadGraph5Error


def _borrow_decision_helpers(namespace):
    """Add the small spinmode/scheme predicates that `_unweighting_mode` and
    `_sequential_upfront`/`_sequential_offshell` are built on to a stub class
    namespace. Call as ``_borrow_decision_helpers(locals())`` from the class
    body of any stub that borrows one of those.

    getattr_static keeps the staticmethod wrappers intact.

    The list is hand-kept, so a new call added inside the borrowed code hits a
    method the stub never borrowed. The `__getattr__` installed here turns that
    into an error naming the method (and where to add it) rather than a bare
    AttributeError; it still *is* an AttributeError, so `getattr(self, x, d)`
    and `hasattr` keep behaving as before.
    """
    for name in ('_auto_unweighting_mode', '_density_pole_approximation',
                 '_density_do_reshuffle', '_density_needs_reshuffle',
                 '_spinmode_has_density', '_is_upfront_scheme',
                 # _auto_unweighting_mode's polarisation branch is built on
                 # these two; a stub that does not seed
                 # _production_polarization_cache gets '{}' out of it, since
                 # there is no banner to read a proc_card from.
                 '_density_spinmode', '_production_polarization',
                 # _unweighting_mode consults these first: the interference
                 # mode and decay_output = weighted both force joint, so a stub
                 # that borrows the resolver needs them -- and _weighted_decay
                 # is built on the decay_output resolver
                 '_pure_interference', '_weighted_decay', '_decay_output',
                 '_announce_decay_output'):
        namespace[name] = inspect.getattr_static(
            interface_madspin.MadSpinInterface, name)

    def __getattr__(self, name):
        try:
            inspect.getattr_static(interface_madspin.MadSpinInterface, name)
        except AttributeError:
            raise AttributeError(name)
        raise AttributeError(
            '%s does not borrow MadSpinInterface.%s: add it to the list in'
            ' _borrow_decision_helpers, or define it on the stub'
            % (type(self).__name__, name))
    namespace.setdefault('__getattr__', __getattr__)
    return namespace


def _borrow_frame_helpers(namespace):
    """Add ``_frame_boost`` and everything its guard reaches through to a stub
    class namespace. Call as ``_borrow_frame_helpers(locals())`` from the class
    body of any stub that borrows ``_frame_boost``.

    The guard is ``_needs_frame_axis``, which asks about the beams, the
    production braces and the two polarisation-weight lists; going through this
    helper is what keeps widening it from meaning an edit in every stub. The
    caller still has to provide ``_production_polarization`` and ``options``,
    which are what the stub is actually choosing.
    """
    for name in ('_beampol', '_frame_boost', '_needs_frame_axis',
                 '_polarization_weight_labels',
                 '_polarization_weights_enabled'):
        namespace[name] = inspect.getattr_static(
            interface_madspin.MadSpinInterface, name)
    namespace['InvalidCmd'] = interface_madspin.MadSpinInterface.InvalidCmd
    return namespace
#
class TestBanner(unittest.TestCase):
    """Test class for the reading of the banner"""

    def test_extract_info(self):
        """Test that the banner is read properly"""

        path=pjoin(MG5DIR, 'tests', 'input_files', 'tt_banner.txt')
        inputfile = open(path, 'r')
        mybanner = banner.Banner(inputfile)
#        mybanner.ReadBannerFromFile()
        process=mybanner.get("generate")
        model=mybanner.get("model")
        self.assertEqual(process,"p p > t t~ @1")
        self.assertEqual(model,"sm")
        
    
    def test_get_final_state_particle(self):
        """test that we find the final state particles correctly"""

        cmd = Cmd.MasterCmd()
        cmd.do_import('sm')
        fct = lambda x: cmd.get_final_part(x)
        
        # 
        self.assertEqual(set([11, -11]), fct('p p > e+ e-'))
        self.assertEqual(set([11, 24]), fct('p p > w+ e-'))
        self.assertEqual(set([11, 24]), fct('p p > W+ e-'))
        self.assertEqual(set([1, 2, 3, 4, -1, 11, 21, -4, -3, -2]), fct('p p > W+ e-, w+ > j j'))
        self.assertEqual(fct('p p > t t~, (t > b w+, w+ > j j) ,t~ > b~ w-'), set([1, 2, 3, 4, -1, 21, -4, -3, -2,5,-5,-24]))
        self.assertEqual(fct('e+ e- > all all, all > e+ e-'), set([-11,11]))
        self.assertEqual(fct('e+ e- > j w+, j > e+ e-'), set([-11,11,24]))

    def test_get_proc_with_decay_LO(self):

        cmd = Cmd.MasterCmd()
        cmd.do_import('sm')
        
        # Note the ; at the end of the line is important!
        #1 simple case
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~', 't> w+b', cmd._curr_model)
        self.assertEqual(['generate p p > t t~, t> w+b  --no_warning=duplicate;'],[out])

        #2 with @0
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ @0', 't> w+b', cmd._curr_model)
        self.assertEqual(['generate p p > t t~ , t> w+b @0 --no_warning=duplicate;'],[out])

        #3 with @0 and --no_warning=duplicate
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ @0 --no_warning=duplicate', 't> w+b', cmd._curr_model)
        self.assertEqual(['generate p p > t t~ , t> w+b @0 --no_warning=duplicate;'],[out])

        #4 test with already present decay chain
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~, t > w+ b @0 --no_warning=duplicate', 't~ > w+b', cmd._curr_model)
        self.assertEqual(['generate p p > t t~, t~ > w+b, ( t > w+ b , t~ > w+b) @0  --no_warning=duplicate;'],[out])
        
        #4 test with already present decay chain
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~, t > w+ b, t~ > w- b~ @0 --no_warning=duplicate', 'w >  all all', cmd._curr_model)
        self.assertEqual(['generate p p > t t~, w >  all all, ( t > w+ b, w >  all all), ( t~ > w- b~ , w >  all all) @0 --no_warning=duplicate;'],[out])

        #6 case with noborn=QCD
        # This is technically not yet supported by MS, but it is nice that this functions supports it.
        out = madspin.decay_all_events.get_proc_with_decay('generate g g > h QED=1 [noborn=QCD]', 'h > b b~', cmd._curr_model)
        self.assertEqual(['add process g g > h QED=1 [sqrvirt=QCD], h > b b~  --no_warning=duplicate;'], 
                         [out]) 

        # simple case but failing initial implementation. Handle it now but raising a critical message [mute here]
        with misc.MuteLogger(['decay'], [60]):
            out = madspin.decay_all_events.get_proc_with_decay('p p > t t~', 't~ > w- b~  QCD=99, t > w+ b  QCD=99', cmd._curr_model)
            self.assertEqual(['add process p p > t t~, t~ > w- b~  QCD=99, t > w+ b  QCD=99  --no_warning=duplicate;'],[out])
        
        self.assertRaises(Exception, madspin.decay_all_events.get_proc_with_decay, 'generate p p > t t~, (t> w+ b, w+ > e+ ve)')

    def test_get_proc_with_decay_NLO(self):

        cmd = Cmd.MasterCmd()
        cmd.do_import('sm')
        
        #1 simple case
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ [QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~, t> w+b  --no_warning=duplicate',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #2 simple case with QED=1
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #3 simple case with options
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [QCD] --test', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate --test',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD QED=1, t> w+b  --no_warning=duplicate --test'],
                         out.split(';')[:-1])

        #4 case with LOonly
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [LOonly]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #5 case with LOonly=QCD
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [LOonly=QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #5 case with LOonly=QCD
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [LOonly=QCD,QED]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #5 case with LOonly=QCD
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [LOonly=QCD QED]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])
        
        
        #6 case with all=QCD
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [all=QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])       

        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [ all= QCD]', 't> w+b', cmd._curr_model)
         
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])       

        #6 case with virt=QCD, technically not valid but I like that the function can do it
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [virt=QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1 [virt=QCD], t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])       

          

class _ProcCardBanner(object):
    """Minimal banner stand-in exposing only the proc_card lines."""

    def __init__(self, proc_card):
        self.proc_card = proc_card


class _ProcCardInterface(object):
    """Minimal MadSpinInterface stand-in.

    generate_all_matrix_element only reads banner.proc_card, list_branches and
    options, so a full interface (which needs an event file and a model) is not
    required to reach the '@' order-restriction check.
    """

    def __init__(self, proc_card):
        self.banner = _ProcCardBanner(proc_card)
        self.list_branches = {}
        self.options = {'global_order_coupling': ''}


class TestOrderRestrictionError(unittest.TestCase):
    """An invalid order restriction after '@' must report what is wrong.

    interface_madspin.generate_all_matrix_element used to raise a bare
    'MadSpinError', a name that module never imports.  The int(proc_nb)
    ValueError therefore led to 'NameError: name MadSpinError is not defined'
    raised *during* the handling of that ValueError, and the intended
    diagnostic was lost.
    """

    bad_line = 'generate p p > t t~ @NLO'

    def test_interface_reports_the_invalid_order_restriction(self):
        """the interface copy of the check raises MadSpinError, not NameError"""

        cmd = _ProcCardInterface([self.bad_line])
        gen = interface_madspin.MadSpinInterface.generate_all_matrix_element

        self.assertRaises(madspin.MadSpinError, gen, cmd)

        try:
            gen(cmd)
        except madspin.MadSpinError as error:
            # the offending token has to be in the message, otherwise the user
            # cannot tell which process line to fix
            self.assertIn('NLO', str(error))
            self.assertIn('order restriction after the @ comment', str(error))
            # raised from inside 'except ValueError', so the int() failure is
            # the chained context; what matters is that the *raised* error is
            # the MadSpin one and not a NameError from the handler itself
            self.assertNotIsInstance(error, NameError)
            self.assertIsInstance(error, MadGraph5Error)

    def test_decay_path_reports_the_same_error(self):
        """the decay.py copy of the same check stays consistent with it"""

        # no '[' in the process, so the model is never looked at before the
        # order-restriction check
        self.assertRaises(madspin.MadSpinError,
                          madspin.decay_all_events.get_proc_with_decay,
                          self.bad_line, 't > w+ b', None)

    def test_valid_order_restriction_passes_the_check(self):
        """a numeric '@' tag is accepted: the guard does not fire"""

        cmd = _ProcCardInterface(['generate p p > t t~ @1'])
        gen = interface_madspin.MadSpinInterface.generate_all_matrix_element

        # the method is still unfinished further down, so it raises something
        # else; the point is that it is no longer the order-restriction error
        try:
            gen(cmd)
        except madspin.MadSpinError as error:
            self.fail('valid order restriction rejected: %s' % error)
        except Exception:
            pass


class TestDensity(unittest.TestCase):
    """Test class for the reading of the lhe input file"""

    def test_get_density_mapping(self):
        """test the utility function that return the order of the density matrix-elements"""

        fct = madspin.DensityMatrix.get_map_density_matrix

        # check one single fermion case first
        out = fct([-1,1], n_changing=1)

        ordered_keys = list(out.keys())
        ordered_keys.sort()
        self.assertEqual(ordered_keys, [(-1, -1), (-1, 1), (1, -1), (1, 1)])

        self.assertEqual(out[(-1, -1)], (True, 0))
        self.assertEqual(out[(-1,  1)], (True, 1))    
        self.assertEqual(out[( 1, -1)], (False,1))
        self.assertEqual(out[(1, 1)], (True, 2))

        # check one massive boson next
        out = fct([-1,0,1], n_changing=1)

        ordered_keys = list(out.keys())
        ordered_keys.sort()
        self.assertEqual(ordered_keys, [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)])

        self.assertEqual(out[(-1, -1)], (True, 0))
        self.assertEqual(out[(-1,  0)], (True, 1))    
        self.assertEqual(out[(-1, 1)],  (True ,2))
        self.assertEqual(out[(0, 0)], (True, 3))
        self.assertEqual(out[(0, 1)], (True,4))
        self.assertEqual(out[(1, 1)], (True, 5))

        self.assertEqual(out[(1,  0)], (False, 4))    
        self.assertEqual(out[(1, -1)], (False,2))


        # check a tt~ case
        out = fct([1,1,-1,1,1,-1,-1,-1], n_changing=2)
        ordered_keys = list(out.keys())
        ordered_keys.sort()
        self.assertEqual(len(ordered_keys), 16) 
        self.assertEqual(out[(1,1,1,1)], (True,0))
        self.assertEqual(out[(1,-1,1,1)], (True,1))
        self.assertEqual(out[(1,1,1,-1)], (True,2))
        self.assertEqual(out[(1,-1,1,-1)], (True,3))
        self.assertEqual(out[(-1,-1,1,1)], (True,4))
        self.assertEqual(out[(-1,1,1,-1)], (True,5))
        self.assertEqual(out[(-1,-1,1,-1)], (True,6))
        self.assertEqual(out[(1,1,-1,-1)], (True,7))
        self.assertEqual(out[(1,-1,-1,-1)], (True,8))
        self.assertEqual(out[(-1,-1,-1,-1)], (True,9))    

        nb_false = len([True for key in out if not out[key][0]])
        self.assertEqual(nb_false, 6)


class _StubOptions(dict):
    """A plain options dict plus the one method the interface calls on it."""
    beampol_me = interface_madspin.MadSpinOptions.beampol_me


class _PIModelStub(object):
    """The one thing _pure_interference asks the model for."""

    NAME2PDG = {'w+': 24, 'w-': -24, 't': 6, 't~': -6, 'z': 23}

    def get(self, key):
        assert key == 'name2pdg'
        return dict(self.NAME2PDG)


class _FrameStub(object):
    """Just enough of MadSpinInterface for the frame/beampol helpers: they only
    need the options and the matrix-element ordering of the event."""

    def __init__(self, frame_id, beampol, prodpol=None, vector=(), fermion=(),
                 pure_interference=''):
        self.options = interface_madspin.MadSpinOptions()
        self.options['frame_id'] = frame_id
        self.options['beampol'] = list(beampol)
        self.options['keep_weight_for_polarization_vector'] = list(vector)
        self.options['keep_weight_for_polarization_fermion'] = list(fermion)
        self.options['pure_interference'] = pure_interference
        self.model = _PIModelStub()
        # what _production_polarization would have parsed out of the banner's
        # proc_card: {} for a brace-free production process
        self._production_polarization_cache = prodpol if prodpol else {}

    def get_pdir(self, event):
        # the real one returns (pdir, orig_order, prefix, pos, tag) -- five
        # values.  _frame_boost used to unpack four, so it raised ValueError the
        # moment it was reached; keep the arity honest here so the tests can see
        # that again if it comes back.
        return None, None, None, None, None

    def _production_polarization(self):
        return self._production_polarization_cache

    _borrow_frame_helpers(locals())
    _pure_interference = interface_madspin.MadSpinInterface._pure_interference
    _parse_pol_side = interface_madspin.MadSpinInterface._parse_pol_side
    _POL_TOKENS = interface_madspin.MadSpinInterface._POL_TOKENS
    InvalidCmd = interface_madspin.MadSpinInterface.InvalidCmd
    _boost_momenta = staticmethod(interface_madspin.MadSpinInterface._boost_momenta)


class _MomentaEvent(object):
    """Stands in for the production event: get_density/_frame_boost only ever
    ask it for its momenta in the matrix element's ordering."""

    def __init__(self, momenta):
        self.momenta = momenta

    def get_momenta(self, orig_order):
        return self.momenta


class TestFrameBoost(unittest.TestCase):
    """me_frame / frame_id and beampol support in the density modes."""

    # 1 and 2 along +z/-z, 3 and 4 sharing the recoil
    MOMENTA = [(500., 0., 0., 500.),
               (200., 0., 0., -200.),
               (300., 100., 50., -80.),
               (400., -100., -50., 380.)]

    def _stub(self, frame_id, beampol=(80., 0.), prodpol=None,
              vector=(), fermion=(), pure_interference=''):
        return _FrameStub(frame_id, beampol, prodpol, vector, fermion,
                          pure_interference)

    def test_polbeam_to_beampol(self):
        """the card speaks percent, like the run_card polbeam1/polbeam2, and
        beampol_me lands it on madevent's /to_polarization/ convention -- not on
        the [0,1] left-handed fraction the eva PDF uses"""
        def fct(polbeam):
            options = interface_madspin.MadSpinOptions()
            options['beampol'] = [polbeam, 0.]
            return options.beampol_me()[0]
        self.assertEqual(fct(0), 1.)
        self.assertEqual(fct(100), 2.)
        self.assertEqual(fct(-100), -2.)
        self.assertEqual(fct(50), 1.5)
        self.assertEqual(fct(-50), -1.5)
        # an unpolarised beam stays exactly 1, whichever slot it is in
        options = interface_madspin.MadSpinOptions()
        options['beampol'] = [60., 0.]
        self.assertEqual(options.beampol_me(), (1.6, 1.))
        # the two branches of the matrix-element reweighting, as written in
        # matrix_standalone_msP_v4.inc: unpolarised leaves both helicities
        # alone, +-100%% keeps one and kills the other
        for polbeam, hel_plus, hel_minus in [(0, 1., 1.),
                                             (100, 2., 0.),
                                             (-100, 0., 2.),
                                             (50, 1.5, 0.5)]:
            pol = fct(polbeam)
            if abs(pol) <= 1:
                got = (1., 1.)
            elif pol > 0:
                got = (abs(pol), 2 - abs(pol))
            else:
                got = (2 - abs(pol), abs(pol))
            self.assertEqual(got, (hel_plus, hel_minus))

    def test_beampol_needs_both_beams(self):
        """One number is ambiguous -- which beam? -- and used to surface only as
        an IndexError once the run reached a matrix element. It is refused when
        the card is read instead."""
        options = interface_madspin.MadSpinOptions()
        for bad in ('50', '[50]'):
            self.assertRaises(Exception, options.__setitem__, 'beampol', bad)
        # unset stays legal and means unpolarised
        options['beampol'] = '[]'
        self.assertEqual(options.beampol_me(), (1., 1.))

    def test_frame_inert_without_polarisation(self):
        """with unpolarised beams and a brace-free production the contraction is
        a trace: a boost acts on it as a unitary change of basis and cancels
        between rho_prod and rho_dec, so there is nothing to do"""
        stub = self._stub(6, beampol=(0., 0.))
        self.assertIsNone(stub._frame_boost(_MomentaEvent(self.MOMENTA)))

    def test_frame_follows_a_production_polarisation_brace(self):
        """a brace on the production (`p p > w+{0} w-`) is applied as a
        *projection* on rho_prod, and a projection does not commute with the
        change of basis a boost induces -- so the frame has to be honoured even
        with unpolarised beams, or MadSpin would restrict a different helicity
        than the one MadEvent generated the events with"""
        stub = self._stub(6, beampol=(0., 0.), prodpol={24: (0,)})
        boost = stub._frame_boost(_MomentaEvent(self.MOMENTA))
        self.assertIsNotNone(boost)
        self.assertEqual((boost.E, boost.px, boost.py, boost.pz),
                         (700., 0., 0., 300.))

    def test_frame_follows_a_polarization_weight_request(self):
        """keep_weight_for_polarization_vector/_fermion apply the same
        set_hel_restriction projection as a production brace, only to build an
        extra <wgt> line rather than the nominal weight. They can be asked for
        on a production that carries no brace and with unpolarised beams, so
        neither of the other two clauses sees them -- and a projection taken on
        the lab axis restricts a different helicity than the one MG5 names.
        Each species list switches the frame on by itself."""
        for kwargs in [dict(vector=['0']), dict(fermion=['+']),
                       dict(vector=['T'], fermion=['-'])]:
            stub = self._stub(6, beampol=(0., 0.), **kwargs)
            boost = stub._frame_boost(_MomentaEvent(self.MOMENTA))
            self.assertIsNotNone(boost, kwargs)
            self.assertEqual((boost.E, boost.px, boost.py, boost.pz),
                             (700., 0., 0., 300.), kwargs)

    def test_frame_boost_matches_needs_frame_axis(self):
        """_frame_boost's guard *is* _needs_frame_axis: the boost is taken for
        each of the three triggers on its own and for nothing else. Pinned
        together so the two cannot drift apart again."""
        cases = [(dict(), False),
                 (dict(beampol=(80., 0.)), True),
                 (dict(prodpol={24: (0,)}), True),
                 (dict(vector=['0']), True),
                 (dict(fermion=['+']), True),
                 (dict(beampol=(80., 0.), vector=['T']), True)]
        for kwargs, wanted in cases:
            kwargs.setdefault('beampol', (0., 0.))
            stub = self._stub(6, **kwargs)
            self.assertEqual(stub._needs_frame_axis(), wanted, kwargs)
            self.assertEqual(stub._frame_boost(_MomentaEvent(self.MOMENTA))
                             is not None, wanted, kwargs)

    def test_frame_follows_the_pure_interference_mode(self):
        """The pure-interference cross restriction is a projection too, and it
        names two helicity SETS, which only mean something once the axis is
        fixed. Its production process is unpolarised by construction, so the
        production-brace test above finds nothing: the mode has to switch the
        frame on for itself or the interference would be taken between
        lab-quantised states while the events were generated in me_frame."""
        stub = self._stub(6, beampol=(0., 0.), pure_interference='w+ = 0 T')
        boost = stub._frame_boost(_MomentaEvent(self.MOMENTA))
        self.assertIsNotNone(boost)
        self.assertEqual((boost.E, boost.px, boost.py, boost.pz),
                         (700., 0., 0., 300.))
        # and it is still inert with the mode off, on the same stub
        self.assertIsNone(self._stub(6, beampol=(0., 0.))._frame_boost(
                                            _MomentaEvent(self.MOMENTA)))

    def test_frame_boost_unpacks_get_pdir(self):
        """get_pdir returns five values; unpacking four raised ValueError the
        first time the frame was actually used (which no unpolarised run ever
        did, so it went unnoticed)"""
        stub = self._stub(6)
        self.assertEqual(len(stub.get_pdir(None)), 5)
        self.assertIsNotNone(stub._frame_boost(_MomentaEvent(self.MOMENTA)))

    def test_frame_id_bitmask(self):
        """frame_id = sum(2**n over the selected legs), the convention mapid
        uncompresses with btest(id, i)"""
        # 6 = 2**1 + 2**2 -> the two initial legs
        boost = self._stub(6)._frame_boost(_MomentaEvent(self.MOMENTA))
        self.assertEqual((boost.E, boost.px, boost.py, boost.pz),
                         (700., 0., 0., 300.))
        # 24 = 2**3 + 2**4 -> the two final legs; same frame, by momentum
        # conservation
        boost = self._stub(24)._frame_boost(_MomentaEvent(self.MOMENTA))
        self.assertEqual((boost.E, boost.px, boost.py, boost.pz),
                         (700., 0., 0., 300.))
        # 8 = 2**3 -> leg 3 alone
        boost = self._stub(8)._frame_boost(_MomentaEvent(self.MOMENTA))
        self.assertEqual((boost.E, boost.px, boost.py, boost.pz),
                         (300., 100., 50., -80.))
        # a frame_id selecting nothing is not a frame
        self.assertIsNone(self._stub(1)._frame_boost(_MomentaEvent(self.MOMENTA)))
        self.assertIsNone(self._stub(0)._frame_boost(_MomentaEvent(self.MOMENTA)))

    def test_boost_to_partonic_cms(self):
        """frame_id = 6 on back-to-back beams is a pure z boost; check every leg
        against the closed form"""
        stub = self._stub(6)
        boost = stub._frame_boost(_MomentaEvent(self.MOMENTA))
        out = stub._boost_momenta(self.MOMENTA, boost)

        mass = math.sqrt(700.**2 - 300.**2)
        gamma, gammabeta = 700. / mass, 300. / mass
        for mom, new in zip(self.MOMENTA, out):
            E, px, py, pz = mom
            self.assertAlmostEqual(new[0], gamma * E - gammabeta * pz, places=9)
            self.assertAlmostEqual(new[1], px, places=9)
            self.assertAlmostEqual(new[2], py, places=9)
            self.assertAlmostEqual(new[3], gamma * pz - gammabeta * E, places=9)

        # the frame is defined by legs 1+2, so their sum is at rest in it
        self.assertAlmostEqual(out[0][3] + out[1][3], 0., places=9)
        # and the boost is an invariance of the masses
        for mom, new in zip(self.MOMENTA, out):
            m2 = mom[0]**2 - mom[1]**2 - mom[2]**2 - mom[3]**2
            n2 = new[0]**2 - new[1]**2 - new[2]**2 - new[3]**2
            self.assertAlmostEqual(m2, n2, delta=1e-6 * abs(mom[0])**2)

    def test_single_leg_frame_is_exactly_at_rest(self):
        """with one selected leg that leg has to come out at exactly zero
        three-momentum: vxxxxx branches on pp.eq.rZero and would otherwise pick
        its quantisation axis from the rounding noise"""
        stub = self._stub(8)
        boost = stub._frame_boost(_MomentaEvent(self.MOMENTA))
        out = stub._boost_momenta(self.MOMENTA, boost)
        self.assertEqual(out[2][1:], (0., 0., 0.))
        self.assertAlmostEqual(out[2][0], math.sqrt(300.**2 - 100.**2 - 50.**2 - 80.**2),
                               places=9)
        # two selected legs: no leg sits on that branch point, nothing is forced
        boost = self._stub(6)._frame_boost(_MomentaEvent(self.MOMENTA))
        self.assertIsNone(boost.rest_leg)

    def test_boost_of_a_system_already_at_rest(self):
        """A lepton-collider event arrives in the partonic CMS, so frame_id = 6
        asks for a boost with no spatial part. That is the identity, not an
        excuse to replace every momentum by the boost (HELAS boostx's
        qq.eq.rZero branch)."""
        momenta = [(250., 0., 0., 250.),
                   (250., 0., 0., -250.),
                   (250., 100., 50., -80.),
                   (250., -100., -50., 80.)]
        stub = _FrameStub(6, (1.8, 1.))
        boost = stub._frame_boost(_MomentaEvent(momenta))
        self.assertEqual((boost.E, boost.px, boost.py, boost.pz),
                         (500., 0., 0., 0.))
        self.assertEqual(stub._boost_momenta(momenta, boost), momenta)


class TestEvent(unittest.TestCase):
    """Test class for the reading of the lhe input file"""
    
    
    def test_madspin_event(self):
        """check the reading/writting of the events inside MadSpin"""
        
        inputfile = open(pjoin(MG5DIR, 'tests', 'input_files', 'madspin_event.lhe'))
        
        events = madspin.Event(inputfile)
        
        # First event
        event = events.get_next_event()
        self.assertEqual(event, 1)
        event = events
        self.assertEqual(event.string_event_compact(), """21 0.0 0 586.8395 586.84 0.7505772
21 0.0 0 -182.0876 182.0891 0.7488873
6 197.60403 48.42486 76.8186 277.8892 173
-6 -212.77359 -34.66934 359.4546 453.4437 173
21 15.169561 -13.75551 -31.52123 37.59628 0.7499895
""")
#        21 0.0 0.0 586.83954 586.84002    0.750577236977    
#21 0.0 0.0 -182.0876 182.08914    0.748887294316    
#6 197.60403 48.424858 76.818601 277.88922    173.00000459    
#-6 -212.77359 -34.669345 359.45458 453.44366    172.999981581    
#21 15.169561 -13.755513 -31.521232 37.59628    0.749989476383 
        self.assertEqual(event.get_tag(), (((21, 21), (-6, 6, 21)), [[21, 21], [6, -6, 21]]))   
        event.assign_scale_line("8 3 0.1 125 0.1 0.3")
        event.change_wgt(factor=0.4)
        
        self.assertEqual(event.string_event().split('\n'), """<event>
  8      3 +4.0000000e-02 1.25000000e+02 1.00000000e-01 3.00000000e-01
       21 -1    0    0  503  502 +0.00000000000e+00 +0.00000000000e+00 +5.86839540000e+02  5.86840020000e+02  7.50000000000e-01 0.0000e+00 0.0000e+00
       21 -1    0    0  501  503 +0.00000000000e+00 +0.00000000000e+00 -1.82087600000e+02  1.82089140000e+02  7.50000000000e-01 0.0000e+00 0.0000e+00
        6  1    1    2  504    0 +1.97604030000e+02 +4.84248580000e+01 +7.68186010000e+01  2.77889220000e+02  1.73000000000e+02 0.0000e+00 0.0000e+00
       -6  1    1    2    0  502 -2.12773590000e+02 -3.46693450000e+01 +3.59454580000e+02  4.53443660000e+02  1.73000000000e+02 0.0000e+00 0.0000e+00
       21  1    1    2  501  504 +1.51695610000e+01 -1.37555130000e+01 -3.15212320000e+01  3.75962800000e+01  7.50000000000e-01 0.0000e+00 0.0000e+00
#aMCatNLO 2  5  3  3  1 0.45933500E+02 0.45933500E+02 9  0  0 0.99999999E+00 0.69338413E+00 0.14872513E+01 0.00000000E+00 0.00000000E+00
  <rwgt>
   <wgt id='1001'>  +1.2946800e+02 </wgt>
   <wgt id='1002'>  +1.1581600e+02 </wgt>
   <wgt id='1003'>  +1.4560400e+02 </wgt>
   <wgt id='1004'>  +1.0034800e+02 </wgt>
   <wgt id='1005'>  +8.9768000e+01 </wgt>
   <wgt id='1006'>  +1.1285600e+02 </wgt>
   <wgt id='1007'>  +1.7120800e+02 </wgt>
   <wgt id='1008'>  +1.5316000e+02 </wgt>
   <wgt id='1009'>  +1.9254800e+02 </wgt>
</rwgt>
</event> 
""".split('\n'))
        
        # Second event
        event = events.get_next_event()    
        self.assertEqual(event, 1)
        event =events
        self.assertEqual(event.get_tag(), (((21, 21), (-6, 6, 21)), [[21, 21], [6, 21, -6]]))
                
        self.assertEqual(event.string_event().split('\n'), """<event>
  5     66 +3.2366351e+02 4.39615290e+02 7.54677160e-03 1.02860750e-01
       21 -1    0    0  503  502 +0.00000000000e+00 +0.00000000000e+00 +1.20582240000e+03  1.20582260000e+03  7.50000000000e-01 0.0000e+00 0.0000e+00
       21 -1    0    0  501  503 +0.00000000000e+00 +0.00000000000e+00 -5.46836110000e+01  5.46887540000e+01  7.50000000000e-01 0.0000e+00 0.0000e+00
        6  1    1    2  501    0 -4.03786550000e+01 -1.41924320000e+02 +3.66089980000e+02  4.30956860000e+02  1.73000000000e+02 0.0000e+00 0.0000e+00
       21  1    1    2  504  502 -2.46716450000e+01 +3.98371210000e+01 +2.49924260000e+02  2.54280130000e+02  7.50000000000e-01 0.0000e+00 0.0000e+00
       -6  1    1    2    0  504 +6.50503000000e+01 +1.02087200000e+02 +5.35124510000e+02  5.75274350000e+02  1.73000000000e+02 0.0000e+00 0.0000e+00
#aMCatNLO 2  5  4  4  4 0.40498390E+02 0.40498390E+02 9  0  0 0.99999997E+00 0.68201705E+00 0.15135239E+01 0.00000000E+00 0.00000000E+00
  <mgrwgt>
  some information
  <scale> even more infor
  </mgrwgt>
  <clustering>
  blabla
  </clustering>
  <rwgt>
   <wgt id='1001'> 0.32367e+03 </wgt>
   <wgt id='1002'> 0.28621e+03 </wgt>
   <wgt id='1003'> 0.36822e+03 </wgt>
   <wgt id='1004'> 0.24963e+03 </wgt>
   <wgt id='1005'> 0.22075e+03 </wgt>
   <wgt id='1006'> 0.28400e+03 </wgt>
   <wgt id='1007'> 0.43059e+03 </wgt>
   <wgt id='1008'> 0.38076e+03 </wgt>
   <wgt id='1009'> 0.48987e+03 </wgt>
  </rwgt>
</event> 
""".split('\n'))
        
        # Third event ! Not existing
        event = events.get_next_event()
        self.assertEqual(event, "no_event")
        



#class Testtopo(unittest.TestCase):
#    """Test the extraction of the topologies for the undecayed process"""
#
#    def test_topottx(self):
#
#        os.environ['GFORTRAN_UNBUFFERED_ALL']='y'
#        path_for_me=pjoin(MG5DIR, 'tests','unit_tests','madspin')
#        shutil.copyfile(pjoin(MG5DIR, 'tests','input_files','param_card_sm.dat'),\
#		pjoin(path_for_me,'param_card.dat'))
#        curr_dir=os.getcwd()
#        os.chdir('/tmp')
#        temp_dir=os.getcwd()
#        mgcmd=Cmd.MasterCmd()
#        process_prod=" g g > t t~ "
#        process_full=process_prod+", ( t > b w+ , w+ > mu+ vm ), "
#        process_full+="( t~ > b~ w- , w- > mu- vm~ ) "
#        decay_tools=madspin.decay_misc()
#        topo=decay_tools.generate_fortran_me([process_prod],"sm",0, mgcmd, path_for_me)
#        decay_tools.generate_fortran_me([process_full],"sm", 1,mgcmd, path_for_me)
#
#        prod_name=decay_tools.compile_fortran_me_production(path_for_me)
#	decay_name = decay_tools.compile_fortran_me_full(path_for_me)
#
#
#        topo_test={1: {'branchings': [{'index_propa': -1, 'type': 's',\
#                'index_d2': 3, 'index_d1': 4}], 'get_id': {}, 'get_momentum': {}, \
#                'get_mass2': {}}, 2: {'branchings': [{'index_propa': -1, 'type': 't', \
#                'index_d2': 3, 'index_d1': 1}, {'index_propa': -2, 'type': 't', 'index_d2': 4,\
#                 'index_d1': -1}], 'get_id': {}, 'get_momentum': {}, 'get_mass2': {}}, \
#                   3: {'branchings': [{'index_propa': -1, 'type': 't', 'index_d2': 4, \
#                'index_d1': 1}, {'index_propa': -2, 'type': 't', 'index_d2': 3, 'index_d1': -1}],\
#                 'get_id': {}, 'get_momentum': {}, 'get_mass2': {}}}
#        
#        self.assertEqual(topo,topo_test)
#  
#
#        p_string='0.5000000E+03  0.0000000E+00  0.0000000E+00  0.5000000E+03  \n'
#        p_string+='0.5000000E+03  0.0000000E+00  0.0000000E+00 -0.5000000E+03 \n'
#        p_string+='0.5000000E+03  0.1040730E+03  0.4173556E+03 -0.1872274E+03 \n'
#        p_string+='0.5000000E+03 -0.1040730E+03 -0.4173556E+03  0.1872274E+03 \n'        
#
#       
#        os.chdir(pjoin(path_for_me,'production_me','SubProcesses',prod_name))
#        executable_prod="./check"
#        external = Popen(executable_prod, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
# 
#        external.stdin.write(p_string)
#
#        info = int(external.stdout.readline())
#        nb_output = abs(info)+1
#
#
#        prod_values = ' '.join([external.stdout.readline() for i in range(nb_output)])
#
#        prod_values=prod_values.split()
#        prod_values_test=['0.59366146660637686', '7.5713552297679376', '12.386583104018380', '34.882849897228873']
#        self.assertEqual(prod_values,prod_values_test)               
#        external.terminate()
#
#
#        os.chdir(temp_dir)
#        
#        p_string='0.5000000E+03  0.0000000E+00  0.0000000E+00  0.5000000E+03 \n'
#        p_string+='0.5000000E+03  0.0000000E+00  0.0000000E+00 -0.5000000E+03 \n'
#        p_string+='0.8564677E+02 -0.8220633E+01  0.3615807E+02 -0.7706033E+02 \n'
#        p_string+='0.1814001E+03 -0.5785084E+02 -0.1718366E+03 -0.5610972E+01 \n'
#        p_string+='0.8283621E+02 -0.6589913E+02 -0.4988733E+02  0.5513262E+01 \n'
#        p_string+='0.3814391E+03  0.1901552E+03  0.2919968E+03 -0.1550888E+03 \n'
#        p_string+='0.5422284E+02 -0.3112810E+02 -0.7926714E+01  0.4368438E+02\n'
#        p_string+='0.2144550E+03 -0.2705652E+02 -0.9850424E+02  0.1885624E+03\n'
#
#        os.chdir(pjoin(path_for_me,'full_me','SubProcesses',decay_name))
#        executable_decay="./check"
#        external = Popen(executable_decay, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
#        external.stdin.write(p_string)
#
#        nb_output =1 
#        decay_value = ' '.join([external.stdout.readline() for i in range(nb_output)])
#
#        decay_value=decay_value.split()
#        decay_value_test=['3.8420345719455465E-017']
#        for i in range(len(decay_value)): 
#            self.assertAlmostEqual(eval(decay_value[i]),eval(decay_value_test[i]))
#        os.chdir(curr_dir)
#        external.terminate()
#        shutil.rmtree(pjoin(path_for_me,'production_me'))
#        shutil.rmtree(pjoin(path_for_me,'full_me'))
#        os.remove(pjoin(path_for_me,'param_card.dat'))
#        os.environ['GFORTRAN_UNBUFFERED_ALL']='n'


class _FakeDecayFile(object):
    """Minimal stand-in for a decay-pool EventFile: a list-backed iterator that
    also exposes .cross and .name, i.e. exactly the surface _StridedEvents wraps
    and get_decay_from_file reads."""
    def __init__(self, events, cross=1.0, name='fake'):
        self._it = iter(events)
        self._cross = cross
        self._name = name
    def __iter__(self):
        return self
    def __next__(self):
        return next(self._it)
    next = __next__
    @property
    def cross(self):
        return self._cross
    @property
    def name(self):
        return self._name


class TestStridedEvents(unittest.TestCase):
    """Unit tests for the parallel-MadSpin decay-pool striping helper.
    These are pure-Python (no matrix-element / physics), so they validate the
    lock-free disjoint-consumption invariant that the process-parallel
    unweighting relies on."""

    def _drain(self, strided):
        out = []
        while True:
            try:
                out.append(next(strided))
            except StopIteration:
                break
        return out

    def test_partition_is_disjoint_and_complete(self):
        """K workers striping over N events must together consume every event
        exactly once, with no overlap and no loss."""
        for n in (0, 1, 7, 100, 101):
            for k in (1, 2, 3, 5):
                events = list(range(n))
                collected = []
                for shard_id in range(k):
                    src = _FakeDecayFile(list(events))
                    collected.extend(self._drain(
                        interface_madspin._StridedEvents(src, shard_id, k)))
                self.assertEqual(sorted(collected), events,
                                 'n=%s k=%s partition wrong' % (n, k))

    def test_phase_offset(self):
        """Worker `offset` must yield events offset, offset+stride, ..."""
        events = list(range(20))
        for shard_id in range(4):
            src = _FakeDecayFile(list(events))
            got = self._drain(
                interface_madspin._StridedEvents(src, shard_id, 4))
            self.assertEqual(got, list(range(shard_id, 20, 4)))

    def test_single_worker_is_identity(self):
        """stride==1 must reproduce the full sequence unchanged."""
        events = list(range(13))
        src = _FakeDecayFile(list(events))
        got = self._drain(interface_madspin._StridedEvents(src, 0, 1))
        self.assertEqual(got, events)

    def test_cross_and_name_proxied(self):
        """Channel selection reads .cross; reopening reads .name."""
        src = _FakeDecayFile([1, 2, 3], cross=42.5, name='chan0')
        strided = interface_madspin._StridedEvents(src, 0, 2)
        self.assertEqual(strided.cross, 42.5)
        self.assertEqual(strided.name, 'chan0')

    def test_offset_beyond_end_is_empty(self):
        """A worker whose phase is past EOF yields nothing (over-sharding)."""
        src = _FakeDecayFile([0, 1])
        strided = interface_madspin._StridedEvents(src, 3, 4)
        self.assertEqual(self._drain(strided), [])



class TestDensityIdentity(unittest.TestCase):
    """DensityMatrix.identity / normalized: the primitives that let the
    accept/reject be done one decaying particle at a time.

    A particle whose decay has not been drawn yet contributes the average of its
    decay density matrix over the full decay phase space. Rotational invariance
    in the parent rest frame makes that average delta_{hh'}/n, so contracting
    the production density matrix against it must give exactly Tr(rho)/n."""

    # the helicity bases MadSpin builds (hel_dict, MG5 2S+1 convention)
    BASES = {1: [0], 2: [1, -1], 3: [-1, 0, 1]}

    def _random_density(self, hel, seed=0):
        """A hermitian-looking density matrix on that basis, in the packed
        upper-triangular storage the Fortran side produces."""
        import numpy as np
        rng = np.random.default_rng(seed)
        n = len(hel)
        arr = (rng.normal(size=n * (n + 1) // 2)
               + 1j * rng.normal(size=n * (n + 1) // 2)).astype('complex64')
        for i in range(n):  # a real diagonal, as a physical density matrix has
            arr[i * (2 * n - i + 1) // 2] = abs(arr[i * (2 * n - i + 1) // 2])
        return madspin.DensityMatrix(arr, 1, hel, n)

    def test_identity_is_flat_diagonal_of_unit_trace(self):
        """delta_{hh'}/n: unit trace, nothing off-diagonal."""
        import numpy as np
        for spin, hel in self.BASES.items():
            n = len(hel)
            I = madspin.DensityMatrix.identity(1, hel, n)
            self.assertAlmostEqual(I.trace().real, 1.0, places=6)
            self.assertEqual(int(np.count_nonzero(I._diag_mask)), n)
            self.assertTrue(np.allclose(I.values[~I._diag_mask], 0))
            self.assertTrue(np.allclose(I.values[I._diag_mask], 1.0 / n))

    def test_contraction_with_identity_is_the_trace(self):
        """The load-bearing property: <rho, I/n> == Tr(rho)/n."""
        import numpy as np
        for spin, hel in self.BASES.items():
            rho = self._random_density(hel, seed=spin)
            I = madspin.DensityMatrix.identity(1, hel, len(hel))
            self.assertTrue(np.allclose(I.scalar_multiplication(rho),
                                        rho.trace() / len(hel)))

    def test_identity_shares_the_cached_map(self):
        """Built like a real density matrix, so the scalar_multiplication fast
        path stays available instead of falling back to sorted alignment."""
        for spin, hel in self.BASES.items():
            rho = self._random_density(hel, seed=spin)
            I = madspin.DensityMatrix.identity(1, hel, len(hel))
            self.assertIsNotNone(I.map_density_matrix_ind)
            self.assertIs(I.map_density_matrix_ind, rho.map_density_matrix_ind)

    def test_normalized_has_unit_trace_and_keeps_direction(self):
        """Dhat = D/Tr(D) puts a drawn decay on the same footing as identity."""
        import numpy as np
        for spin, hel in self.BASES.items():
            rho = self._random_density(hel, seed=spin + 10)
            D = rho.normalized()
            self.assertAlmostEqual(D.trace().real, 1.0, places=5)
            # same matrix up to the scale
            self.assertTrue(np.allclose(D.values * rho.trace(), rho.values,
                                        rtol=1e-4, atol=1e-6))

    def test_scalar_parent_identity_equals_its_density(self):
        """A spin-0 parent has a 1x1 density matrix: normalized() and identity()
        coincide, so its accept/reject ratio is identically 1 -- it can never be
        rejected. This is why the pool ladder must not charge scalars."""
        import numpy as np
        hel = self.BASES[1]
        rho = self._random_density(hel, seed=3)
        I = madspin.DensityMatrix.identity(1, hel, 1)
        self.assertTrue(np.allclose(rho.normalized().values, I.values))


class TestSequentialOrdering(unittest.TestCase):
    """Ordering and pool sizing for the per-particle accept/reject.

    Spins are in the MG5 2S+1 convention: 1 = scalar, 2 = fermion, 3 = vector.
    """

    class _Stub(object):
        _sequential_spin_order = interface_madspin.MadSpinInterface._sequential_spin_order
        _decay_slot_order = interface_madspin.MadSpinInterface._decay_slot_order
        def __init__(self, order='2 3 1'):
            self.options = {'sequential_spin_order': order}

    def test_default_order_is_fermions_vectors_scalars(self):
        s = self._Stub()
        # slots: scalar, vector, fermion, fermion -> fermions, vector, scalar
        self.assertEqual(s._decay_slot_order([1, 3, 2, 2]), [2, 3, 1, 0])
        # scalar last even when it comes first in slot order
        self.assertEqual(s._decay_slot_order([1, 2]), [1, 0])

    def test_ties_broken_by_slot_index(self):
        """Same spin -> keep slot order, so a run is reproducible."""
        s = self._Stub()
        self.assertEqual(s._decay_slot_order([2, 2]), [0, 1])
        self.assertEqual(s._decay_slot_order([3, 3, 3]), [0, 1, 2])

    def test_order_is_configurable(self):
        """The hidden option allows A/B testing the ordering per process."""
        s = self._Stub('3 2 1')  # vectors first
        self.assertEqual(s._decay_slot_order([2, 3, 1]), [1, 0, 2])

    def test_unlisted_spin_goes_last_and_garbage_falls_back(self):
        self.assertEqual(self._Stub('2')._decay_slot_order([3, 2, 1]), [1, 0, 2])
        self.assertEqual(self._Stub('garbage')._decay_slot_order([2, 1, 3]), [0, 2, 1])
        self.assertEqual(self._Stub('')._sequential_spin_order(), [2, 3, 1])

    def test_ladder_grows_with_position(self):
        """1/eff_k: each slot sees a more polarised parent than the last."""
        ladder = interface_madspin.MadSpinInterface._decay_pool_ladder
        self.assertEqual([ladder(k, 2) for k in range(4)], [1.5, 2.0, 2.5, 3.0])
        self.assertEqual([ladder(k, 3) for k in range(4)], [1.5, 2.0, 2.5, 3.0])

    def test_scalar_is_never_charged_the_ladder(self):
        """A spin-0 parent can never be rejected (1x1 density matrix), so it
        burns exactly one decay event wherever it sits in the ordering."""
        ladder = interface_madspin.MadSpinInterface._decay_pool_ladder
        for position in range(4):
            self.assertEqual(ladder(position, 1), 1.1)


class TestDrawOneDecay(unittest.TestCase):
    """_draw_one_decay: drawing a single particle's decay, so the sequential
    accept/reject can redraw one particle without touching the others.

    get_decay_from_file is now a loop over it and must behave exactly as before.
    """

    class _Part(object):
        def __init__(self, pid):
            self.pid = pid
            self.pdg = pid
            self.status = 1

    class _Pool(object):
        """Stands in for an lhe_parser.EventFile of decay events."""
        def __init__(self, tag, n=50, cross=1.0):
            self.tag = tag
            self._it = iter(range(n))
            self.cross = cross
        def __next__(self):
            return '%s:%s' % (self.tag, next(self._it))

    class _Stub(object):
        get_decay_from_file = interface_madspin.MadSpinInterface.get_decay_from_file
        _draw_all_decays = interface_madspin.MadSpinInterface._draw_all_decays
        _draw_one_decay = interface_madspin.MadSpinInterface._draw_one_decay
        _draw_decay_group = interface_madspin.MadSpinInterface._draw_decay_group
        efficiency = 0.5

    def _setup(self):
        # t t~ t plus a gluon that never decays; two decay files for each pdg so
        # both the one-file-per-parent and the cross-section-weighted branches
        # are exercised.
        production = [self._Part(6), self._Part(-6), self._Part(6), self._Part(21)]
        evt_decayfile = {6: {0: self._Pool('t0'), 1: self._Pool('t1')},
                         -6: {0: self._Pool('tx0', cross=2.0),
                              1: self._Pool('tx1', cross=3.0)}}
        return production, evt_decayfile

    def test_non_decaying_particle_gives_none(self):
        production, evt_decayfile = self._setup()
        gluon = production[3]
        self.assertIsNone(self._Stub()._draw_one_decay(
            gluon, 3, [p.pid for p in production], evt_decayfile, 10))

    def test_empty_pool_dict_gives_none(self):
        production, evt_decayfile = self._setup()
        evt_decayfile[6] = {}
        self.assertIsNone(self._Stub()._draw_one_decay(
            production[0], 0, [p.pid for p in production], evt_decayfile, 10))

    def test_slots_come_in_production_order(self):
        """The density matrix slots are built in production order, so the draw
        must walk the particles in that same order."""
        production, evt_decayfile = self._setup()
        got = [(i, part.pid) for i, part, _ in
               self._Stub()._draw_all_decays(production, evt_decayfile, 10)]
        self.assertEqual(got, [(0, 6), (1, -6), (2, 6)])

    def test_identical_parents_read_their_own_file(self):
        """Two tops with two decay files: one file each, in order."""
        production, evt_decayfile = self._setup()
        out = self._Stub().get_decay_from_file(production, evt_decayfile, 10)
        self.assertEqual(out[6], ['t0:0', 't1:0'])

    def test_joint_path_is_unchanged(self):
        """get_decay_from_file must still consume the RNG in the same order and
        return the same draws as before the refactor."""
        import random
        for seed in range(50):
            random.seed(seed)
            production, evt_decayfile = self._setup()
            got = dict(self._Stub().get_decay_from_file(production, evt_decayfile, 10))
            random.seed(seed)
            production, evt_decayfile = self._setup()
            want = dict(self._reference(production, evt_decayfile))
            self.assertEqual(got, want)

    @staticmethod
    def _reference(production, evt_decayfile):
        """The implementation as it was before _draw_one_decay was extracted."""
        import random
        out = collections.defaultdict(list)
        particles = [p for p in production if int(p.status) == 1.0]
        ids = [particle.pid for particle in particles]
        for i, particle in enumerate(particles):
            if particle.pdg not in evt_decayfile:
                continue
            nb_decay = len(evt_decayfile[particle.pdg])
            if nb_decay == 0:
                continue
            if nb_decay == 1:
                decay_file = evt_decayfile[particle.pdg][0]
            elif ids.count(particle.pdg) == nb_decay:
                decay_file = evt_decayfile[particle.pdg][ids[:i].count(particle.pdg)]
            else:
                r = random.random()
                tot = sum(evt_decayfile[particle.pdg][k].cross
                          for k in evt_decayfile[particle.pdg])
                r = r * tot
                cumul = 0
                for j, events in evt_decayfile[particle.pdg].items():
                    cumul += events.cross
                    if r < cumul:
                        decay_file = events
                        break
                else:
                    raise Exception
            out[particle.pdg].append(next(decay_file))
        return out


class TestDrawOffshellMass(unittest.TestCase):
    """_draw_offshell_mass: one resonance virtuality, owned by the decay that
    carries it.

    The draw used to be a single pass over every decay, closing over a shared
    sqrt(shat) budget. The sequential accept/reject needs to draw one slot at a
    time and redraw a single mass on a reshuffling failure, so the budget is now
    passed in and out and the caller owns the order.
    """

    class _Val(object):
        def __init__(self, value):
            self.value = value

    class _Banner(object):
        def get(self, card, kind, pdg):
            return TestDrawOffshellMass._Val(173.0 if kind == 'mass' else 1.5)

    class _Dec(object):
        pass

    class _Stub(object):
        _draw_mass_value = interface_madspin.MadSpinInterface._draw_mass_value
        _draw_offshell_mass = interface_madspin.MadSpinInterface._draw_offshell_mass
        def __init__(self, bw_cut=-1):
            self.banner = TestDrawOffshellMass._Banner()
            self.options = {'BW_cut': bw_cut}

    def _reference(self, pdg, dec, budget, banner, options):
        """The inline draw block, mirroring the extracted _draw_mass_value: the
        Breit-Wigner sampling jacobian gap/pi with the atan argument divided by
        the width (not multiplied)."""
        pole = banner.get('param', 'mass', abs(pdg)).value
        width = banner.get('param', 'decay', abs(pdg)).value
        if options['BW_cut'] < 0:
            bw_cut = 15
        else:
            bw_cut = options['BW_cut']
        min_mass = pole - bw_cut * width
        max_mass = min(pole + bw_cut * width, budget)
        dec[0].new_mass = lhe_parser.Event.generate_random_mass(
                                    pole, width, min_mass, max_mass)
        dec[0].reshuffle_info = (pole, width, min_mass, max_mass)
        budget -= dec[0].new_mass
        gap = math.atan((pole ** 2 - min_mass ** 2) / pole / width)
        gap += math.atan((max_mass ** 2 - pole ** 2) / pole / width)
        return budget, gap / math.pi

    def test_identical_to_the_previous_inline_draw(self):
        """Same masses, same jacobians, same budget, same random sequence."""
        import random
        stub = self._Stub()
        for seed in range(50):
            random.seed(seed)
            budget, got = 500.0, []
            for _ in range(2):   # two resonances off one shared budget, as t t~
                dec = [self._Dec()]
                budget, jac = stub._draw_offshell_mass(6, dec, budget)
                got.append((dec[0].new_mass, jac, dec[0].reshuffle_info))
            random.seed(seed)
            ref_budget, want = 500.0, []
            for _ in range(2):
                dec = [self._Dec()]
                ref_budget, jac = self._reference(6, dec, ref_budget,
                                                  stub.banner, stub.options)
                want.append((dec[0].new_mass, jac, dec[0].reshuffle_info))
            self.assertEqual(got, want)
            self.assertEqual(budget, ref_budget)

    def test_mass_and_resample_info_land_on_the_decay(self):
        """The decay owns its mass: new_mass plus what is needed to redraw it."""
        import random
        random.seed(0)
        dec = [self._Dec()]
        _, jac = self._Stub()._draw_offshell_mass(6, dec, 500.0)
        self.assertTrue(hasattr(dec[0], 'new_mass'))
        pole, width, min_mass, max_mass = dec[0].reshuffle_info
        self.assertEqual((pole, width), (173.0, 1.5))
        self.assertTrue(min_mass <= dec[0].new_mass <= max_mass)
        self.assertTrue(jac > 0)

    def test_budget_shrinks_so_the_draw_is_order_dependent(self):
        """Each resonance eats into what the next one may take -- which is why
        the slot has to own the draw rather than inherit one pass over all."""
        import random
        random.seed(1)
        stub = self._Stub()
        dec = [self._Dec()]
        left, _ = stub._draw_offshell_mass(6, dec, 500.0)
        self.assertAlmostEqual(left, 500.0 - dec[0].new_mass)
        self.assertLess(left, 500.0)

    def test_bw_cut_option_is_honoured(self):
        """BW_cut < 0 means the default 15 widths; otherwise the option wins."""
        import random
        random.seed(2)
        dec = [self._Dec()]
        self._Stub(bw_cut=2)._draw_offshell_mass(6, dec, 500.0)
        pole, width, min_mass, max_mass = dec[0].reshuffle_info
        self.assertAlmostEqual(min_mass, 173.0 - 2 * 1.5)
        self.assertAlmostEqual(max_mass, 173.0 + 2 * 1.5)


class TestPartialDensityContraction(unittest.TestCase):
    """_partial_density_contraction: N_k, the production density matrix
    contracted with the decays drawn so far, the rest replaced by I/n.

    This is the heart of the per-particle accept/reject, so it is pinned against
    the two endpoints it has to reproduce and against the telescoping identity
    the method's exactness rests on.
    """

    class _Stub(object):
        _slot_identity = interface_madspin.MadSpinInterface._slot_identity
        _partial_density_contraction = \
            interface_madspin.MadSpinInterface._partial_density_contraction

    def _density(self, hel, seed):
        """A density matrix on a single-particle basis, packed as Fortran gives it."""
        import numpy as np
        rng = np.random.default_rng(seed)
        n = len(hel)
        arr = (rng.normal(size=n * (n + 1) // 2)
               + 1j * rng.normal(size=n * (n + 1) // 2)).astype('complex64')
        for i in range(n):
            arr[i * (2 * n - i + 1) // 2] = abs(arr[i * (2 * n - i + 1) // 2])
        return madspin.DensityMatrix(arr, 1, hel, n)

    def _production(self, hels, seed=7):
        """A production density matrix over the joint helicity index."""
        import numpy as np
        import itertools
        dim = 1
        for h in hels:
            dim *= len(h)
        allowed = []
        for combo in itertools.product(*hels):
            allowed.extend(combo)
        rng = np.random.default_rng(seed)
        arr = (rng.normal(size=dim * (dim + 1) // 2)
               + 1j * rng.normal(size=dim * (dim + 1) // 2)).astype('complex64')
        for i in range(dim):
            arr[i * (2 * dim - i + 1) // 2] = abs(arr[i * (2 * dim - i + 1) // 2])
        return madspin.DensityMatrix(arr, len(hels), allowed, dim), dim

    CASES = [[[1, -1], [1, -1]],            # t t~
             [[-1, 0, 1], [-1, 0, 1]],      # W W
             [[1, -1], [0]],                # fermion + scalar
             [[1, -1], [-1, 0, 1], [0]]]    # fermion + vector + scalar

    def test_nothing_drawn_is_the_trace(self):
        """N_0 = Tr(rho) / prod n_i: every slot contributes I/n."""
        import numpy as np
        for hels in self.CASES:
            rho, dim = self._production(hels)
            got = self._Stub()._partial_density_contraction(rho, hels, {})
            self.assertTrue(np.allclose(got, rho.trace() / dim))

    def test_everything_drawn_is_the_joint_contraction(self):
        """N_n = <rho, (x)_i Dhat_i>: the weight the joint accept/reject uses."""
        import numpy as np
        for hels in self.CASES:
            rho, _ = self._production(hels)
            densities = {i: self._density(h, 10 + i) for i, h in enumerate(hels)}
            got = self._Stub()._partial_density_contraction(rho, hels, densities)
            joint = None
            for i, h in enumerate(hels):
                d = densities[i].normalized()
                joint = d if joint is None else joint.tensor_product(d)
            self.assertTrue(np.allclose(got, joint.scalar_multiplication(rho)))

    def test_ratios_telescope_whatever_the_fill_order(self):
        """prod_k N_k/N_{k-1} == N_n/N_0 -- why the per-slot test targets the
        same distribution as the joint one, for any decay ordering."""
        import numpy as np
        hels = [[1, -1], [-1, 0, 1], [0]]
        rho, _ = self._production(hels, seed=3)
        densities = {i: self._density(h, 20 + i) for i, h in enumerate(hels)}
        stub = self._Stub()
        n_0 = stub._partial_density_contraction(rho, hels, {})
        n_n = stub._partial_density_contraction(rho, hels, densities)
        for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2]):
            filled, previous, product = {}, n_0, 1.0
            for slot in order:
                filled[slot] = densities[slot]
                current = stub._partial_density_contraction(rho, hels, filled)
                product *= current / previous
                previous = current
            self.assertTrue(np.allclose(product, n_n / n_0))

    def test_scalar_slot_cannot_be_rejected(self):
        """Filling a spin-0 slot leaves N_k untouched: its ratio is exactly 1."""
        import numpy as np
        hels = [[1, -1], [-1, 0, 1], [0]]
        rho, _ = self._production(hels, seed=3)
        densities = {i: self._density(h, 20 + i) for i, h in enumerate(hels)}
        stub = self._Stub()
        without = stub._partial_density_contraction(rho, hels, {0: densities[0],
                                                                1: densities[1]})
        with_scalar = stub._partial_density_contraction(rho, hels, densities)
        self.assertTrue(np.allclose(with_scalar / without, 1.0))


class TestDensityPolarizationRestriction(unittest.TestCase):
    """Restricting the production/decay density-matrix convolution to the
    polarisation written on the *production* process.

    W = sum_i sum_j rho_prod(i,j) rho_dec(i,j) becomes a sum over the (i,j)
    block the braces allow: a single-state brace ({0}, {+}/{R}, {-}/{L}) keeps
    only the diagonal rho(X,X) term, {T} keeps the -1/+1 block and drops the 0
    row and column, and no brace leaves the full double sum untouched.
    """

    FERMION = [1, -1]
    VECTOR = [-1, 0, 1]

    def _packed(self, hel, seed):
        """A hermitian density matrix on that basis, in the packed
        upper-triangular storage the Fortran side hands back."""
        import numpy as np
        rng = np.random.default_rng(seed)
        n = len(hel)
        arr = (rng.normal(size=n * (n + 1) // 2)
               + 1j * rng.normal(size=n * (n + 1) // 2)).astype('complex64')
        for i in range(n):
            arr[i * (2 * n - i + 1) // 2] = abs(arr[i * (2 * n - i + 1) // 2])
        return arr

    def _density(self, hel, seed, restriction=None):
        rho = madspin.DensityMatrix(self._packed(hel, seed), 1, hel, len(hel))
        if restriction is not None:
            rho.set_hel_restriction(restriction)
        return rho

    def _brute_force(self, dec, prod, allowed):
        """Sum_{(i,j) allowed} rho_dec(i,j) rho_prod(i,j), read off the labels
        rather than the masks -- an independent implementation of what
        scalar_multiplication must produce."""
        import numpy as np
        table = {tuple(int(x) for x in lab): val
                 for lab, val in zip(prod.helicities, prod.values)}
        total = 0j
        for lab, val in zip(dec.helicities, dec.values):
            lab = tuple(int(x) for x in lab)
            keep = True
            for k, ok in enumerate(allowed):
                if ok is None:
                    continue
                if lab[2 * k] not in ok or lab[2 * k + 1] not in ok:
                    keep = False
                    break
            if keep:
                total += complex(val) * complex(table[lab])
        return total

    # -- no braces: nothing may move ---------------------------------------

    def test_no_braces_leaves_the_full_double_sum(self):
        """The behaviour-neutrality requirement: an absent or empty restriction
        must not touch a single value."""
        import numpy as np
        for hel in (self.FERMION, self.VECTOR):
            prod = self._density(hel, seed=1)
            dec = self._density(hel, seed=2)
            reference = dec.scalar_multiplication(prod)
            for empty in (None, [None], [()], [[]]):
                other = self._density(hel, seed=1, restriction=empty)
                self.assertIsNone(other.hel_restriction)
                self.assertTrue(np.allclose(dec.scalar_multiplication(other),
                                            reference))
            self.assertTrue(np.allclose(prod.trace(), self._brute_force(
                prod, madspin.DensityMatrix.identity(1, hel, len(hel)),
                [None]) * len(hel)))

    # -- single state -------------------------------------------------------

    def test_single_state_keeps_only_the_diagonal_term(self):
        """{0}, {+}/{R}, {-}/{L}: one helicity X survives, so the double sum
        collapses onto rho_prod(X,X) rho_dec(X,X)."""
        import numpy as np
        cases = [(self.VECTOR, 0), (self.FERMION, 1), (self.FERMION, -1),
                 (self.VECTOR, 1), (self.VECTOR, -1)]
        for hel, x in cases:
            prod = self._density(hel, seed=11, restriction=[(x,)])
            dec = self._density(hel, seed=12)
            got = dec.scalar_multiplication(prod)
            # rho(X,X) rho(X,X): pick the two entries by label
            pi = {tuple(int(v) for v in l): c
                  for l, c in zip(prod.helicities, prod.values)}[(x, x)]
            di = {tuple(int(v) for v in l): c
                  for l, c in zip(dec.helicities, dec.values)}[(x, x)]
            self.assertTrue(np.allclose(got, complex(di) * complex(pi)))
            self.assertTrue(np.allclose(got, self._brute_force(dec, prod, [(x,)])))

    def test_transverse_drops_the_zero_row_and_column(self):
        """{T} = [-1,1]: the double sum survives but neither index may be 0."""
        import numpy as np
        prod = self._density(self.VECTOR, seed=21, restriction=[(-1, 1)])
        dec = self._density(self.VECTOR, seed=22)
        got = dec.scalar_multiplication(prod)
        self.assertTrue(np.allclose(got,
                                    self._brute_force(dec, prod, [(-1, 1)])))
        # strictly between the single-state and the unrestricted answers: four
        # terms out of nine, and the off-diagonal (-1,1)/(1,-1) pair kept
        mask = prod._restriction_row_mask(prod.hel_restriction)
        self.assertEqual(int(mask.sum()), 4)
        kept = set(tuple(int(v) for v in l)
                   for l, m in zip(prod.helicities, mask) if m)
        self.assertEqual(kept, {(-1, -1), (-1, 1), (1, -1), (1, 1)})

    # -- several decaying particles ----------------------------------------

    def test_multi_particle_mask_is_a_per_index_product(self):
        """t{0} t~{T}-like: the restriction is per decaying particle and the
        masks combine multiplicatively over the tensor-product structure."""
        import numpy as np
        import itertools
        hels = [self.VECTOR, self.VECTOR]
        dim = len(hels[0]) * len(hels[1])
        allowed_hel = [h for combo in itertools.product(*hels) for h in combo]
        prod = madspin.DensityMatrix(self._packed(list(range(dim)), 31),
                                     2, allowed_hel, dim)
        dec = self._density(hels[0], 32).tensor_product(
              self._density(hels[1], 33))

        for restriction in ([(0,), (-1, 1)], [None, (0,)], [(1,), None],
                            [(-1, 1), (-1, 1)]):
            prod.set_hel_restriction(restriction)
            got = dec.scalar_multiplication(prod)
            self.assertTrue(np.allclose(
                got, self._brute_force(dec, prod, restriction)))

        # 1 (the single (0,0) pair) x 4 (the transverse block) of 9 x 9 entries
        prod.set_hel_restriction([(0,), (-1, 1)])
        mask = prod._restriction_row_mask(prod.hel_restriction)
        self.assertEqual(int(mask.sum()), 4)
        self.assertEqual(len(mask), 81)

    def test_restriction_survives_the_tensor_product(self):
        """A restricted index keeps its restriction when the matrix is tensored
        with an unrestricted one -- the per-index masks simply concatenate."""
        left = self._density(self.FERMION, 41, restriction=[(1,)])
        right = self._density(self.VECTOR, 42)
        self.assertEqual(left.tensor_product(right).hel_restriction,
                         ((1,), None))
        self.assertEqual(right.tensor_product(left).hel_restriction,
                         (None, (1,)))

    # -- normalisation ------------------------------------------------------

    def test_trace_follows_the_restriction(self):
        """Tr rho normalises the convolution, so it has to be restricted too:
        it is the polarised production cross-section the events were generated
        with. Getting this wrong would make the accept/reject weight stop
        averaging to 1/n."""
        import numpy as np
        hel = self.VECTOR
        arr = self._packed(hel, 51)
        full = madspin.DensityMatrix(arr, 1, hel, 3)
        diag = [complex(arr[i * (2 * 3 - i + 1) // 2]) for i in range(3)]
        self.assertTrue(np.allclose(full.trace(), sum(diag)))
        # labels are ordered as hel: [-1, 0, 1]
        for x, expected in zip(hel, diag):
            rho = madspin.DensityMatrix(arr, 1, hel, 3).set_hel_restriction([(x,)])
            self.assertTrue(np.allclose(rho.trace(), expected))
        transverse = madspin.DensityMatrix(arr, 1, hel, 3)
        transverse.set_hel_restriction([(-1, 1)])
        self.assertTrue(np.allclose(transverse.trace(), diag[0] + diag[2]))

    def test_identity_contraction_still_gives_the_restricted_trace(self):
        """<rho, I/n> restricted = (restricted Tr rho)/n, so a slot whose decay
        has not been drawn yet contributes the same 1/n it always did and the
        sequential accept/reject normalisation is untouched."""
        import numpy as np
        for hel in (self.FERMION, self.VECTOR):
            for allowed in ([(hel[0],)], [tuple(hel[:2])]):
                rho = self._density(hel, 61, restriction=allowed)
                I = madspin.DensityMatrix.identity(1, hel, len(hel))
                self.assertTrue(np.allclose(I.scalar_multiplication(rho),
                                            rho.trace() / len(hel)))

    def test_mask_is_cached_per_basis(self):
        """Recomputed once per (basis, restriction), never per event."""
        hel = self.VECTOR
        a = self._density(hel, 71, restriction=[(0,)])
        b = self._density(hel, 72, restriction=[(0,)])
        self.assertIs(a._restriction_row_mask(a.hel_restriction),
                      b._restriction_row_mask(b.hel_restriction))
        self.assertIsNot(a._restriction_row_mask(a.hel_restriction),
                         a._restriction_row_mask(((-1, 1),)))

    def test_restricted_paths_are_bit_identical_to_masking(self):
        """The contractions gather the surviving rows by index instead of by
        boolean mask. That is a speed change only: the rows come out in
        increasing index order either way, so the sums must agree *exactly*,
        not merely to within a tolerance."""
        import numpy as np
        import itertools
        for hels in ([self.FERMION], [self.VECTOR],
                     [self.FERMION, self.VECTOR],
                     [self.VECTOR, self.FERMION, self.VECTOR]):
            dim = 1
            for h in hels:
                dim *= len(h)
            allowed_hel = [h for combo in itertools.product(*hels) for h in combo]
            prod = madspin.DensityMatrix(self._packed(list(range(dim)), 101),
                                         len(hels), allowed_hel, dim)
            dec = None
            for i, h in enumerate(hels):
                d = self._density(h, 102 + i)
                dec = d if dec is None else dec.tensor_product(d)
            choices = [[None] + [(x,) for x in h] + [tuple(h[:2])] for h in hels]
            for restriction in itertools.product(*choices):
                prod.set_hel_restriction(list(restriction))
                r = prod.hel_restriction
                if r is None:
                    continue
                mask = prod._restriction_row_mask(r)
                rows = prod._restriction_rows(r)[1]
                self.assertTrue(np.array_equal(rows, np.flatnonzero(mask)))
                # trace: same elements, same order
                self.assertEqual(prod.trace(),
                                 np.sum(prod.values[prod._diag_mask & mask]))
                # contraction, both the map fast path (dec is prod's basis for
                # one particle) and the sorted-alignment path
                for lhs, rhs in ((dec, prod), (prod, dec)):
                    m = lhs._restriction_row_mask(r)
                    if (lhs.map_density_matrix_ind is not None and
                            lhs.map_density_matrix_ind is rhs.map_density_matrix_ind):
                        want = np.sum(lhs.values[m] * rhs.values[m])
                    else:
                        lhs._ensure_sorted_view()
                        rhs._ensure_sorted_view()
                        a, b = lhs._sort_order, rhs._sort_order
                        aligned = m[a]
                        want = np.sum(lhs.values[a][aligned] * rhs.values[b][aligned])
                    self.assertEqual(lhs.scalar_multiplication(rhs), want)

    def test_contradicting_restrictions_are_refused(self):
        hel = self.FERMION
        a = self._density(hel, 81, restriction=[(1,)])
        b = self._density(hel, 82, restriction=[(-1,)])
        self.assertRaises(ValueError, a.scalar_multiplication, b)

    def test_sequential_contraction_sees_the_restriction(self):
        """_partial_density_contraction contracts through the production matrix,
        so attaching the restriction there is enough for the sequential
        accept/reject -- no call site has to pass it along."""
        import numpy as np
        import itertools
        hels = [self.FERMION, self.VECTOR]
        dim = len(hels[0]) * len(hels[1])
        allowed_hel = [h for combo in itertools.product(*hels) for h in combo]
        rho = madspin.DensityMatrix(self._packed(list(range(dim)), 91),
                                    2, allowed_hel, dim)
        rho.set_hel_restriction([(1,), (0,)])
        stub = TestPartialDensityContraction._Stub()
        got = stub._partial_density_contraction(rho, hels, {})
        # every slot is I/n, so this is the restricted trace over prod n_i
        self.assertTrue(np.allclose(got, rho.trace() / dim))


class TestPureInterferenceRestriction(unittest.TestCase):
    """Proof of concept for the pure-interference ("cross") restriction:
    one index restricted to the production-side polarisation P, the other to
    the decay-side one D, with P and D disjoint.

    These tests pin down the algebra section 13 of doc/madspin_sequential_plan.md
    argues from; the mode itself (syntax, signed unweighting, zero
    cross-section bookkeeping) is NOT implemented.
    """

    FERMION = TestDensityPolarizationRestriction.FERMION
    VECTOR = TestDensityPolarizationRestriction.VECTOR
    _packed = TestDensityPolarizationRestriction._packed
    _density = TestDensityPolarizationRestriction._density
    _brute_force = TestDensityPolarizationRestriction._brute_force

    def _cross_brute_force(self, dec, prod, spec):
        """Sum over the entries a per-particle cross restriction keeps, read
        off the labels. ``spec`` has one entry per particle: None, a flat set
        (symmetric), or a (P, D) pair (cross = P x D union D x P)."""
        table = {tuple(int(x) for x in lab): complex(val)
                 for lab, val in zip(prod.helicities, prod.values)}
        total = 0j
        for lab, val in zip(dec.helicities, dec.values):
            lab = tuple(int(x) for x in lab)
            keep = True
            for k, entry in enumerate(spec):
                if entry is None:
                    continue
                i, j = lab[2 * k], lab[2 * k + 1]
                if isinstance(entry[0], (list, tuple, set, frozenset)):
                    p, d = set(entry[0]), set(entry[1])
                    ok = (i in p and j in d) or (i in d and j in p)
                else:
                    ok = i in entry and j in entry
                if not ok:
                    keep = False
                    break
            if keep:
                total += complex(val) * table[lab]
        return total

    # -- normalisation of the new entry form -------------------------------

    def test_cross_entry_normalisation(self):
        norm = madspin.DensityMatrix.normalize_hel_restriction
        self.assertEqual(norm([[(0,), (-1, 1)]]), (((0,), (-1, 1)),))
        # order inside each side is canonicalised, the two sides are not swapped
        self.assertEqual(norm([[(1, -1), (0,)]]), (((-1, 1), (0,)),))
        # a cross pair with two identical sides is just the symmetric form
        self.assertEqual(norm([[(-1, 1), (1, -1)]]), ((-1, 1),))
        # an empty side would kill everything: fall back to unrestricted
        self.assertIsNone(norm([[(), (0,)]]))
        # the historical flat spelling is untouched
        self.assertEqual(norm([(0,)]), ((0,),))
        self.assertRaises(ValueError, norm, [[(0,), (1,), (-1,)]])

    # -- the mask itself ----------------------------------------------------

    def test_cross_mask_is_the_off_diagonal_block_and_its_transpose(self):
        """{0} against {T} on a vector: exactly the four entries (0,+-1) and
        (+-1,0). No diagonal entry survives."""
        prod = self._density(self.VECTOR, seed=101,
                             restriction=[[(0,), (-1, 1)]])
        mask = prod._restriction_row_mask(prod.hel_restriction)
        kept = set(tuple(int(v) for v in l)
                   for l, m in zip(prod.helicities, mask) if m)
        self.assertEqual(kept, {(0, -1), (0, 1), (-1, 0), (1, 0)})
        self.assertEqual(int(mask.sum()), 4)

    def test_cross_is_closed_under_transposition(self):
        """The crux: the kept set must be stable under (i,j) -> (j,i),
        otherwise the contraction is complex and there is no 'sign of the
        convolution' to speak of."""
        for hel, spec in ((self.VECTOR, [[(0,), (-1, 1)]]),
                          (self.VECTOR, [[(-1,), (0, 1)]]),
                          (self.FERMION, [[(1,), (-1,)]])):
            rho = self._density(hel, seed=102, restriction=spec)
            mask = rho._restriction_row_mask(rho.hel_restriction)
            kept = set(tuple(int(v) for v in l)
                       for l, m in zip(rho.helicities, mask) if m)
            self.assertEqual(kept, set((j, i) for i, j in kept))

    # -- the algebra --------------------------------------------------------

    def test_cross_contraction_is_real_and_is_twice_the_real_part(self):
        """sum over (P x D) alone is complex; adding its transpose gives
        2 Re[...], which is what the mask computes."""
        import numpy as np
        hel = self.VECTOR
        prod = self._density(hel, seed=111, restriction=[[(0,), (-1, 1)]])
        dec = self._density(hel, seed=112)
        got = dec.scalar_multiplication(prod)

        table = {tuple(int(x) for x in l): complex(v)
                 for l, v in zip(prod.helicities, prod.values)}
        half = sum(complex(v) * table[tuple(int(x) for x in l)]
                   for l, v in zip(dec.helicities, dec.values)
                   if int(l[0]) == 0 and int(l[1]) in (-1, 1))
        # the half-sum is genuinely complex: this is why P x D alone is not
        # a usable weight
        self.assertGreater(abs(half.imag), 1e-3 * abs(half))
        self.assertTrue(np.allclose(got, 2 * half.real, atol=1e-5))
        self.assertLess(abs(complex(got).imag), 1e-5 * abs(complex(got).real))
        self.assertTrue(np.allclose(
            got, self._cross_brute_force(dec, prod, [[(0,), (-1, 1)]]),
            atol=1e-5))

    def test_the_three_blocks_add_up_to_the_full_convolution(self):
        """<rho_prod, rho_dec> = PP + DD + interference, with P u D the whole
        basis. The interference block is what this restriction isolates."""
        import numpy as np
        hel = self.VECTOR
        dec = self._density(hel, seed=122)
        full = dec.scalar_multiplication(self._density(hel, seed=121))
        pp = dec.scalar_multiplication(
            self._density(hel, seed=121, restriction=[(0,)]))
        dd = dec.scalar_multiplication(
            self._density(hel, seed=121, restriction=[(-1, 1)]))
        inter = dec.scalar_multiplication(
            self._density(hel, seed=121, restriction=[[(0,), (-1, 1)]]))
        self.assertTrue(np.allclose(full, pp + dd + inter, atol=1e-4))

    # -- the two consequences the mode has to live with ---------------------

    def test_cross_restricted_trace_vanishes(self):
        """No diagonal entry survives, so the cross-restricted trace is 0.

        Physically: the interference term carries no cross-section. Practically:
        the accept/reject weight must NOT be normalised by this trace -- the
        denominator has to stay the (unrestricted) production matrix element
        the input events were generated with, which is what
        set_hel_restriction_trace is for."""
        for hel, spec in ((self.VECTOR, [[(0,), (-1, 1)]]),
                          (self.FERMION, [[(1,), (-1,)]])):
            rho = self._density(hel, seed=131, restriction=spec)
            # asked for explicitly, the theorem still holds
            self.assertEqual(complex(rho.trace(hel_restriction=spec)), 0j)

    def test_cross_restriction_leaves_the_normalising_trace_alone(self):
        """The contraction restriction and the normalisation restriction are
        two different objects (section 13.4).

        A cross restriction attached to the matrix restricts the *contraction*
        but must not reach trace(): with no trace restriction set -- the
        unpolarised production this mode requires -- trace() is the full,
        unrestricted trace, i.e. exactly the production matrix element the
        input events were generated with. So the weight is a zero-mean
        numerator over a strictly positive denominator, not 0/0."""
        for hel, spec in ((self.VECTOR, [[(0,), (-1, 1)]]),
                          (self.FERMION, [[(1,), (-1,)]])):
            plain = self._density(hel, seed=131)
            cross = self._density(hel, seed=131, restriction=spec)
            self.assertEqual(cross.hel_restriction_trace, None)
            self.assertAlmostEqual(complex(cross.trace()).real,
                                   complex(plain.trace()).real, places=5)
            self.assertNotEqual(complex(plain.trace()), 0j)

    def test_cross_numerator_over_a_polarised_production_trace(self):
        """The trace restriction, when there is one, is the *symmetric* P u D
        one -- what the production process' own braces impose -- and it does
        not resurrect the diagonal of the interference block.

        Zero numerator over a non-zero denominator: the weight is well defined
        and its mean over the decay phase space is zero."""
        hel, spec = self.VECTOR, [[(0,), (-1, 1)]]
        rho = self._density(hel, seed=151, restriction=spec)
        rho.set_hel_restriction_trace([(-1, 0, 1)])
        identity = madspin.DensityMatrix.identity(1, hel, len(hel))
        # numerator: the interference block against a not-yet-drawn decay slot
        self.assertEqual(complex(identity.scalar_multiplication(rho)), 0j)
        # denominator: the polarised production trace, which is not zero
        self.assertNotEqual(complex(rho.trace()), 0j)

    def test_symmetric_restriction_still_normalises_by_its_own_trace(self):
        """Nothing that existed before the interference mode moves: a symmetric
        restriction is returned untouched by _trace_restriction, so trace()
        keeps applying it and hel_restriction_trace is never consulted."""
        hel, spec = [(-1,), (0,), (1,)], [(-1, 1)]
        rho = self._density(self.VECTOR, seed=161, restriction=spec)
        self.assertEqual(rho._trace_restriction(), ((-1, 1),))
        self.assertEqual(complex(rho.trace()),
                         complex(rho.trace(hel_restriction=spec)))

    def test_cross_contraction_against_the_identity_vanishes(self):
        """A decay slot that has not been drawn yet contributes I/n, which is
        diagonal, so every partial contraction is identically zero.

        This is both the reason the interference integrates to zero over the
        decay phase space, and the reason the sequential (per-particle)
        accept/reject cannot be used in this mode: no prefix ever has a
        non-zero weight to unweight against."""
        for hel, spec in ((self.VECTOR, [[(0,), (-1, 1)]]),
                          (self.FERMION, [[(1,), (-1,)]])):
            rho = self._density(hel, seed=141, restriction=spec)
            identity = madspin.DensityMatrix.identity(1, hel, len(hel))
            self.assertEqual(complex(identity.scalar_multiplication(rho)), 0j)

    # -- several decaying particles ----------------------------------------

    def test_multi_particle_cross_stays_a_per_index_product(self):
        """Mixing a cross entry with a symmetric one and with None: the mask
        keeps its per-particle AND structure, and stays transposition-closed
        (hence real) because each factor separately is."""
        import numpy as np
        import itertools
        hels = [self.VECTOR, self.VECTOR]
        dim = len(hels[0]) * len(hels[1])
        allowed_hel = [h for combo in itertools.product(*hels) for h in combo]
        prod = madspin.DensityMatrix(self._packed(list(range(dim)), 151),
                                     2, allowed_hel, dim)
        dec = self._density(hels[0], 152).tensor_product(
              self._density(hels[1], 153))

        for spec in ([[(0,), (-1, 1)], None],
                     [[(0,), (-1, 1)], (-1, 1)],
                     [None, [(-1,), (1,)]],
                     [[(0,), (-1, 1)], [(-1,), (1,)]]):
            prod.set_hel_restriction(spec)
            got = prod.scalar_multiplication(dec)
            self.assertTrue(np.allclose(
                got, self._cross_brute_force(dec, prod, spec), atol=1e-4))
            self.assertLess(abs(complex(got).imag),
                            1e-4 * max(abs(complex(got).real), 1e-6))
            mask = prod._restriction_row_mask(prod.hel_restriction)
            kept = set(tuple(int(v) for v in l)
                       for l, m in zip(prod.helicities, mask) if m)
            # transposing the joint index swaps bra and ket of every particle
            self.assertEqual(kept, set(
                tuple(x for pair in zip(k[1::2], k[0::2]) for x in pair)
                for k in kept))

        # 4 (the (0,+-1)/(+-1,0) block) x 4 (the (-1,1)/(1,-1) block) of 81
        prod.set_hel_restriction([[(0,), (-1, 1)], [(-1,), (1,)]])
        mask = prod._restriction_row_mask(prod.hel_restriction)
        self.assertEqual(int(mask.sum()), 8)

    def test_cross_restriction_survives_the_tensor_product(self):
        left = self._density(self.FERMION, 161, restriction=[[(1,), (-1,)]])
        right = self._density(self.VECTOR, 162)
        self.assertEqual(left.tensor_product(right).hel_restriction,
                         (((1,), (-1,)), None))

    def test_cross_mask_is_cached_per_basis(self):
        a = self._density(self.VECTOR, 171, restriction=[[(0,), (-1, 1)]])
        b = self._density(self.VECTOR, 172, restriction=[[(0,), (-1, 1)]])
        self.assertIs(a._restriction_row_mask(a.hel_restriction),
                      b._restriction_row_mask(b.hel_restriction))
        self.assertIsNot(a._restriction_row_mask(a.hel_restriction),
                         a._restriction_row_mask(((0,),)))

    # -- behaviour neutrality ----------------------------------------------

    def test_symmetric_restrictions_are_untouched(self):
        """The whole point of the new entry form is that nothing symmetric
        moves: same normalisation, same mask objects, same answers."""
        import numpy as np
        for hel in (self.FERMION, self.VECTOR):
            dec = self._density(hel, seed=182)
            for spec in ([None], [(hel[0],)], [tuple(hel[:2])]):
                prod = self._density(hel, seed=181, restriction=spec)
                self.assertTrue(np.allclose(
                    dec.scalar_multiplication(prod),
                    self._brute_force(dec, prod, [
                        None if spec == [None] else spec[0]])))


class TestPureInterferenceMode(unittest.TestCase):
    """The mode itself: card syntax, validation, and the two restrictions it
    hands the density matrices."""

    class _Stub(object):
        """Just enough MadSpinInterface for the pure-interference helpers."""
        InvalidCmd = interface_madspin.MadSpinInterface.InvalidCmd
        _POL_TOKENS = interface_madspin.MadSpinInterface._POL_TOKENS
        _parse_pol_side = interface_madspin.MadSpinInterface._parse_pol_side
        _pure_interference = interface_madspin.MadSpinInterface._pure_interference
        _pure_interference_pdgs = \
            interface_madspin.MadSpinInterface._pure_interference_pdgs
        _apply_pure_interference = \
            interface_madspin.MadSpinInterface._apply_pure_interference
        _validate_pure_interference = \
            interface_madspin.MadSpinInterface._validate_pure_interference
        _pure_interference_unweighted = \
            interface_madspin.MadSpinInterface._pure_interference_unweighted
        _density_spinmode = interface_madspin.MadSpinInterface._density_spinmode
        _unweighting_mode = interface_madspin.MadSpinInterface._unweighting_mode
        _announce_mode = interface_madspin.MadSpinInterface._announce_mode
        _log_once = interface_madspin.MadSpinInterface._log_once
        # _unweighting_mode is built on the spinmode/scheme predicate family
        # (#346); borrowing it means borrowing them too
        _borrow_decision_helpers(locals())

        def __init__(self, spec='', spinmode='madspin', pol_map=None,
                     branches=('w+', 'w-'), unweighting='sequential',
                     pol_weights=False, output='auto'):
            self.options = interface_madspin.MadSpinOptions()
            self.options['pure_interference'] = spec
            self.options['decay_output'] = output
            self.options['spinmode'] = spinmode
            self.options['unweighting'] = unweighting
            self.options['fixed_order'] = False
            self.model = _PIModelStub()
            self.list_branches = dict((name, []) for name in branches)
            self._pol = pol_map or {}
            self._pol_weights = pol_weights

        def _production_polarization(self):
            return self._pol

        def _polarization_weights_enabled(self):
            return self._pol_weights

    # -- syntax ------------------------------------------------------------

    def test_parses_a_single_particle(self):
        got = self._Stub('w+ = 0 T')._pure_interference()
        self.assertEqual(got, {24: ((0,), (-1, 1))})

    def test_the_two_sides_are_ordered_production_then_decay(self):
        """'0 T' and 'T 0' are different specifications: the first index of the
        kept (i,j) block comes from the production-side set."""
        self.assertEqual(self._Stub('w+ = T 0')._pure_interference(),
                         {24: ((-1, 1), (0,))})

    def test_accepts_a_pdg_code_and_several_particles(self):
        got = self._Stub('t = + - ; -6 = L R',
                         branches=('t', 't~'))._pure_interference()
        self.assertEqual(got, {6: ((1,), (-1,)), -6: ((-1,), (1,))})

    def test_the_L_R_T_vocabulary_matches_the_braces(self):
        stub = self._Stub()
        self.assertEqual(stub._parse_pol_side('L', 'x'), (-1,))
        self.assertEqual(stub._parse_pol_side('R', 'x'), (1,))
        self.assertEqual(stub._parse_pol_side('T', 'x'), (-1, 1))
        self.assertEqual(stub._parse_pol_side('0', 'x'), (0,))
        # a side may name several states explicitly
        self.assertEqual(stub._parse_pol_side('0,+', 'x'), (0, 1))

    def test_unset_is_off(self):
        self.assertEqual(self._Stub('')._pure_interference(), {})
        self.assertEqual(self._Stub('   ')._pure_interference(), {})

    def test_overlapping_sides_are_refused(self):
        """An overlap puts a diagonal entry back in, so the block carries
        cross-section and stops being a pure interference term."""
        stub = self._Stub('w+ = T +')
        self.assertRaises(stub.InvalidCmd, stub._pure_interference)

    def test_malformed_entries_are_refused(self):
        for spec in ('w+ 0 T',        # no '='
                     'w+ = 0',        # one side only
                     'w+ = 0 T X',    # three sides
                     'w+ = 0 Q',      # not a polarisation
                     'nosuch = 0 T'): # not a particle
            stub = self._Stub(spec)
            self.assertRaises(stub.InvalidCmd, stub._pure_interference)

    # -- validation --------------------------------------------------------

    def test_a_non_density_spinmode_is_refused(self):
        stub = self._Stub('w+ = 0 T', spinmode='none')
        self.assertRaises(stub.InvalidCmd, stub._validate_pure_interference)

    def test_a_particle_that_is_never_decayed_is_refused(self):
        """Otherwise the mode is silently inert while the signed weights and
        the zeroed cross-section are still in force."""
        stub = self._Stub('t = + -', branches=('w+', 'w-'))
        self.assertRaises(stub.InvalidCmd, stub._validate_pure_interference)

    def test_a_braced_production_is_refused(self):
        """p p > w+{0} w- contains no transverse amplitude, so there is no
        0-T interference in the sample to project onto (section 13.5)."""
        stub = self._Stub('w+ = 0 T', pol_map={24: (0,)})
        self.assertRaises(stub.InvalidCmd, stub._validate_pure_interference)

    def test_an_unpolarised_production_is_accepted(self):
        self._Stub('w+ = 0 T')._validate_pure_interference()

    def test_a_production_brace_covering_both_sides_is_accepted(self):
        """A brace that keeps everything the mode names is not a problem: the
        amplitudes it needs are all there."""
        self._Stub('w+ = 0 T',
                   pol_map={24: (-1, 0, 1)})._validate_pure_interference()

    # -- the two restrictions ---------------------------------------------

    def test_builds_a_cross_restriction_and_an_unrestricted_trace(self):
        stub = self._Stub('w+ = 0 T')
        restriction, trace = stub._apply_pure_interference(
                                    [24, -24], [[-1, 0, 1], [-1, 0, 1]], None)
        self.assertEqual(restriction, (((0,), (-1, 1)), None))
        # unpolarised production -> the normalising trace stays the full one
        self.assertEqual(trace, None)

    def test_keeps_the_symmetric_restriction_as_the_trace_one(self):
        """With a production brace, the contraction moves to the interference
        block while the normalisation keeps using the polarised trace."""
        stub = self._Stub('w+ = 0 T')
        restriction, trace = stub._apply_pure_interference(
                        [24], [[-1, 0, 1]], ((-1, 0, 1),))
        self.assertEqual(restriction, (((0,), (-1, 1)),))
        self.assertEqual(trace, ((-1, 0, 1),))

    def test_is_inert_when_off(self):
        stub = self._Stub('')
        self.assertEqual(stub._apply_pure_interference([24], [[-1, 0, 1]], None),
                         (None, None))
        self.assertEqual(
            stub._apply_pure_interference([24], [[-1, 0, 1]], ((0,),)),
            (((0,),), None))

    def test_a_polarisation_outside_the_basis_is_refused(self):
        """{0} on a fermion: the density modes use [1,-1] for it."""
        stub = self._Stub('t = 0 T', branches=('t',))
        self.assertRaises(stub.InvalidCmd, stub._apply_pure_interference,
                          [6], [[1, -1]], None)

    def test_only_the_named_particle_is_crossed(self):
        """The mask keeps its per-particle structure: the partner keeps
        whatever it had."""
        stub = self._Stub('w+ = 0 T')
        restriction, _ = stub._apply_pure_interference(
                        [24, -24], [[-1, 0, 1], [-1, 0, 1]], (None, (0,)))
        self.assertEqual(restriction, (((0,), (-1, 1)), (0,)))

    # -- interaction with the rest of the machinery ------------------------

    def test_the_mode_forces_the_joint_accept_reject(self):
        """Every partial contraction against DensityMatrix.identity is zero in
        this mode, so no staged scheme has anything to unweight against."""
        for asked in ('sequential', 'two_stage', 'sequential_global_retry',
                      'auto'):
            stub = self._Stub('w+ = 0 T', unweighting=asked)
            self.assertEqual(stub._unweighting_mode(True), 'joint')

    def test_the_mode_does_not_touch_the_scheme_when_off(self):
        stub = self._Stub('', unweighting='sequential')
        self.assertEqual(stub._unweighting_mode(True), 'sequential')

    # -- diagonal blocks, and what the option may not be -------------------

    def test_identical_sides_name_the_diagonal_block(self):
        """Two equal sides are not an "overlap" but the diagonal block D_S:
        normalize_hel_restriction collapses (S, S) back to the symmetric S, so
        a mixed block such as (I, D-) is expressible from the card alone."""
        got = self._Stub('w+ = 0 T ; w- = 0 0')._pure_interference()
        self.assertEqual(got, {24: ((0,), (-1, 1)), -24: ((0,), (0,))})
        restriction, trace = self._Stub(
            'w+ = 0 T ; w- = 0 0')._apply_pure_interference(
                [24, -24], [[-1, 0, 1], [-1, 0, 1]], None)
        # the cross entry stays a pair; the equal one collapses to the plain
        # symmetric restriction, i.e. the diagonal block
        self.assertEqual(restriction, (((0,), (-1, 1)), (0,)))
        self.assertEqual(trace, None)

    def test_a_partial_overlap_is_still_refused(self):
        """'T +' is neither an interference block nor a diagonal one: it puts
        some diagonal entries into an off-diagonal block."""
        stub = self._Stub('w+ = T +')
        self.assertRaises(stub.InvalidCmd, stub._pure_interference)

    def test_only_diagonal_entries_are_refused(self):
        """Nothing interferes, so every piece of the mode -- the zeroed
        <init>, the signed weights, the separate trace restriction -- is wrong.
        """
        stub = self._Stub('w+ = 0 0')
        self.assertRaises(stub.InvalidCmd, stub._validate_pure_interference)

    def test_polarization_weights_are_refused_with_the_mode(self):
        """keep_weight_for_polarization_* writes nominal x (diagonal block /
        full contraction); in this mode the nominal weight is a signed
        interference weight and the diagonal blocks are exactly what the mode
        removes, so the product means nothing."""
        stub = self._Stub('w+ = 0 T', pol_weights=True)
        self.assertRaises(stub.InvalidCmd, stub._validate_pure_interference)
        # ... and without the mode the two are unrelated
        self._Stub('w+ = 0 T', pol_weights=False)._validate_pure_interference()

    # -- decay_output, which governs this mode's output shape ---------------

    def test_auto_gives_the_fully_weighted_output(self):
        """``decay_output = auto`` (the default) resolves to the fully
        weighted output here: it uses every production event and measures ~6x
        less variance per production event (section 13.17). That is what the
        mode has always done."""
        stub = self._Stub('w+ = 0 T')
        self.assertEqual(stub.options['decay_output'], 'auto')
        self.assertEqual(stub._decay_output(), 'weighted')
        self.assertFalse(stub._pure_interference_unweighted())

    def test_auto_resolves_the_other_way_without_the_mode(self):
        """The same 'auto' is the ordinary accept/reject outside the mode --
        that is the whole point of it being one option."""
        stub = self._Stub('')
        self.assertEqual(stub._decay_output(), 'unweighted')
        self.assertFalse(stub._pure_interference_unweighted())

    def test_the_output_option_selects_the_unweighted_variant(self):
        stub = self._Stub('w+ = 0 T', output='unweighted')
        self.assertTrue(stub._pure_interference_unweighted())
        stub._validate_pure_interference()

    def test_an_explicit_weighted_matches_what_auto_resolves_to(self):
        stub = self._Stub('w+ = 0 T', output='weighted')
        self.assertFalse(stub._pure_interference_unweighted())
        stub._validate_pure_interference()

    def test_the_unweighted_variant_is_inert_without_the_mode(self):
        """Outside the mode 'unweighted' is the ordinary accept/reject, not
        the unweighted-up-to-a-sign interference output."""
        stub = self._Stub('', output='unweighted')
        self.assertFalse(stub._pure_interference_unweighted())
        # and validation is a no-op
        stub._validate_pure_interference()

    def test_the_replaced_option_no_longer_exists(self):
        """``pure_interference_output`` was folded into ``decay_output``; the
        old spelling is gone rather than silently accepted."""
        options = interface_madspin.MadSpinOptions()
        self.assertNotIn('pure_interference_output', options)

    def test_the_output_option_rejects_an_unknown_value(self):
        """ConfigFile keeps the previous value and warns rather than raising,
        so the check is that an unknown spelling does not silently become the
        active one."""
        options = interface_madspin.MadSpinOptions()
        options['decay_output'] = 'unweighted'
        options['decay_output'] = 'signed'
        self.assertEqual(options['decay_output'], 'unweighted')


class TestPureInterferenceCardSyntax(unittest.TestCase):
    """The card spelling of a multi-particle pure_interference request.

    ``extended_cmd.Cmd.precmd`` splits every card line on ';' and dispatches
    the pieces as separate commands, so the one-line spelling silently loses
    everything after the first particle. Repeated ``set`` lines are therefore
    the only spelling that can work, and the ';' one has to fail loudly rather
    than produce a valid-looking single-particle sample.
    """

    def _interface(self):
        return interface_madspin.MadSpinInterface()

    def test_repeated_set_lines_accumulate(self):
        ms = self._interface()
        ms.exec_cmd('set pure_interference t = + -', precmd=True)
        ms.exec_cmd('set pure_interference t~ = + -', precmd=True)
        self.assertEqual(ms.options['pure_interference'], 't = + - ; t~ = + -')

    def test_a_single_set_line_is_unchanged(self):
        ms = self._interface()
        ms.exec_cmd('set pure_interference t = + -', precmd=True)
        self.assertEqual(ms.options['pure_interference'], 't = + -')

    def test_the_semicolon_spelling_fails_loudly(self):
        """The failure mode this replaces is silent: 't~ = + -' used to reach
        Cmd.default, log a generic 'not recognized' warning, and leave a valid
        single-particle sample behind."""
        ms = self._interface()
        self.assertRaises(
            ms.InvalidCmd,
            lambda: ms.exec_cmd('set pure_interference t = + - ; t~ = + -',
                                precmd=True))

    def test_an_orphan_entry_on_its_own_line_also_fails(self):
        ms = self._interface()
        self.assertRaises(ms.InvalidCmd,
                          lambda: ms.exec_cmd('t~ = + -', precmd=True))

    def test_an_ordinary_unknown_command_is_still_only_a_warning(self):
        """The loud failure is targeted at the entry shape; anything else keeps
        the historical behaviour."""
        ms = self._interface()
        ms.exec_cmd('not_a_command with args', precmd=True)

    def test_other_options_still_overwrite(self):
        ms = self._interface()
        ms.exec_cmd('set BW_cut 15', precmd=True)
        ms.exec_cmd('set BW_cut 25', precmd=True)
        self.assertEqual(ms.options['BW_cut'], 25)


class TestPureInterferenceNormalisation(unittest.TestCase):
    """``c = <W>``, the decay-side constant the fully weighted output divides
    by, and the helper that measures it (section 13.13)."""

    def _packed(self, hel, seed):
        import numpy as np
        rng = np.random.default_rng(seed)
        n = len(hel)
        arr = (rng.normal(size=n * (n + 1) // 2)
               + 1j * rng.normal(size=n * (n + 1) // 2)).astype('complex64')
        for i in range(n):
            arr[i * (2 * n - i + 1) // 2] = abs(arr[i * (2 * n - i + 1) // 2])
        return arr

    def _density(self, hel, seed):
        return madspin.DensityMatrix(self._packed(hel, seed), 1, hel, len(hel))

    def test_the_helper_lifts_the_cross_restriction_and_restores_it(self):
        hel = [-1, 0, 1]
        prod = self._density(hel, 3)
        dec = self._density(hel, 11)
        full = complex(dec.scalar_multiplication(prod))

        prod.set_hel_restriction([((0,), (-1, 1))])
        prod.set_hel_restriction_trace([None])
        restricted = complex(dec.scalar_multiplication(prod))
        self.assertNotAlmostEqual(abs(restricted), abs(full), places=6)

        got = interface_madspin.MadSpinInterface._pi_unrestricted_contraction(
            prod, dec)
        self.assertAlmostEqual(complex(got).real, full.real, places=4)
        self.assertAlmostEqual(complex(got).imag, full.imag, places=4)
        # and the matrix is left exactly as it was found
        self.assertEqual(prod.hel_restriction, (((0,), (-1, 1)),))
        self.assertAlmostEqual(
            complex(dec.scalar_multiplication(prod)).real, restricted.real,
            places=6)

    def test_the_helper_uses_the_trace_restriction_not_the_full_sum(self):
        """With a production brace on another leg the ordinary run's weight is
        normalised by the *braced* trace, so c must be the braced contraction
        -- not the unrestricted one."""
        hel = [-1, 1]
        prod = self._density(hel, 5)
        dec = self._density(hel, 9)
        prod.set_hel_restriction([((-1,), (1,))])
        prod.set_hel_restriction_trace([(-1, 1)])
        got = complex(interface_madspin.MadSpinInterface
                      ._pi_unrestricted_contraction(prod, dec))
        prod.set_hel_restriction([(-1, 1)])
        expect = complex(dec.scalar_multiplication(prod))
        self.assertAlmostEqual(got.real, expect.real, places=5)

    def test_c_against_the_identity_decay_matrix(self):
        """Substituting the decay-phase-space average I/n for rho_dec is what
        makes c a constant: the cross block then contracts to exactly zero
        while the unrestricted one gives trace(rho_prod)/n."""
        hel = [-1, 0, 1]
        prod = self._density(hel, 21)
        dec = madspin.DensityMatrix.identity(1, hel, len(hel))
        prod.set_hel_restriction([((0,), (-1, 1))])
        prod.set_hel_restriction_trace([None])
        self.assertAlmostEqual(
            abs(complex(dec.scalar_multiplication(prod))), 0.0, places=6)
        got = complex(interface_madspin.MadSpinInterface
                      ._pi_unrestricted_contraction(prod, dec))
        self.assertAlmostEqual(got.real, complex(prod.trace()).real / len(hel),
                               places=5)

    def test_finalize_turns_the_raw_moments_into_c_and_its_error(self):
        class _Stub(object):
            InvalidCmd = interface_madspin.MadSpinInterface.InvalidCmd
            _finalize_pi_c = interface_madspin.MadSpinInterface._finalize_pi_c
        stub = _Stub()
        values = [1.0, 2.0, 3.0, 4.0]
        stub._pi_c_stats = {'sum': sum(values),
                            'sumsq': sum(v * v for v in values),
                            'n': len(values)}
        stub._finalize_pi_c()
        self.assertAlmostEqual(stub._pi_c, 2.5)
        # sd of the population is sqrt(1.25); the error on the mean is /sqrt(n)
        self.assertAlmostEqual(stub._pi_c_err, math.sqrt(1.25 / 4.0))

    def test_finalize_refuses_an_empty_or_zero_measurement(self):
        class _Stub(object):
            InvalidCmd = interface_madspin.MadSpinInterface.InvalidCmd
            _finalize_pi_c = interface_madspin.MadSpinInterface._finalize_pi_c
        stub = _Stub()
        stub._pi_c_stats = {'sum': 0.0, 'sumsq': 0.0, 'n': 0}
        self.assertRaises(stub.InvalidCmd, stub._finalize_pi_c)
        stub._pi_c_stats = {'sum': 0.0, 'sumsq': 4.0, 'n': 2}
        self.assertRaises(stub.InvalidCmd, stub._finalize_pi_c)


class TestPureInterferenceUnweightedOutput(unittest.TestCase):
    """``pure_interference_output = unweighted``: ``<|W|>``, its plumbing, and
    the estimator identity that says the accept/reject bound cancels out of
    the weight (section 13.17)."""

    class _Stub(object):
        InvalidCmd = interface_madspin.MadSpinInterface.InvalidCmd
        _finalize_pi_absw = \
            interface_madspin.MadSpinInterface._finalize_pi_absw
        _write_pi_c_cache = \
            interface_madspin.MadSpinInterface._write_pi_c_cache
        _read_pi_c_cache = interface_madspin.MadSpinInterface._read_pi_c_cache

        def __init__(self, unweighted=False):
            self._unweighted = unweighted

        def _pure_interference_unweighted(self):
            return self._unweighted

    # -- <|W|> -------------------------------------------------------------

    def test_finalize_turns_the_raw_moments_into_absw_and_its_error(self):
        """Without per-event blocking it falls back to the trial-level error."""
        stub = self._Stub()
        values = [-1.0, 2.0, -3.0, 4.0]
        stub._pi_absw_stats = {'sum': sum(abs(v) for v in values),
                               'sumsq': sum(v * v for v in values),
                               'n': len(values)}
        stub._finalize_pi_absw()
        self.assertAlmostEqual(stub._pi_absw, 2.5)
        self.assertAlmostEqual(stub._pi_absw_err, math.sqrt(1.25 / 4.0))

    def test_the_absw_error_is_blocked_by_production_event(self):
        """The nb_ps_point draws of one production point share its |W| scale,
        so the trial-level spread is not an error on <|W|>. Measured on
        p p > t t~: 0.46% trial-level against 5.0% blocked, and the blocked
        one is the honest number -- the probe's <|W|> came out 9% away from
        the run's."""
        stub = self._Stub()
        # four production events, ten near-identical draws each: the trials
        # look extremely well determined, the production events do not
        per_event = [1.0, 2.0, 3.0, 4.0]
        trials = [m for m in per_event for _ in range(10)]
        stub._pi_absw_stats = {
            'sum': sum(trials), 'sumsq': sum(v * v for v in trials),
            'n': len(trials),
            'ev_sum': sum(per_event),
            'ev_sumsq': sum(v * v for v in per_event),
            'ev_n': len(per_event)}
        stub._finalize_pi_absw()
        self.assertAlmostEqual(stub._pi_absw, 2.5)
        # the blocked error: sd of [1,2,3,4] over sqrt(4)
        self.assertAlmostEqual(stub._pi_absw_err, math.sqrt(1.25 / 4.0))
        # ... and it is much larger than the trial-level one would have been
        naive = math.sqrt((sum(v * v for v in trials) / 40.0 - 6.25) / 40.0)
        self.assertGreater(stub._pi_absw_err, 3.0 * naive)

    def test_a_missing_absw_is_fatal_only_for_the_unweighted_output(self):
        empty = {'sum': 0.0, 'sumsq': 0.0, 'n': 0}
        weighted = self._Stub(unweighted=False)
        weighted._pi_absw_stats = dict(empty)
        weighted._finalize_pi_absw()          # a diagnostic there, so tolerated
        self.assertEqual(weighted._pi_absw, 0.0)
        unweighted = self._Stub(unweighted=True)
        unweighted._pi_absw_stats = dict(empty)
        self.assertRaises(unweighted.InvalidCmd, unweighted._finalize_pi_absw)

    # -- the ms_dir cache --------------------------------------------------

    def test_the_cache_round_trips_c_and_absw(self):
        import tempfile
        stub = self._Stub()
        stub._pi_c, stub._pi_c_err = 2.3e-10, 3.1e-13
        stub._pi_analytic_c = 2.25e-10
        stub._pi_absw, stub._pi_absw_err = 1.7e-11, 4.0e-14
        path = os.path.join(tempfile.mkdtemp(), 'pure_interference_c')
        stub._write_pi_c_cache(path)
        back = self._Stub(unweighted=True)
        self.assertTrue(back._read_pi_c_cache(path))
        self.assertAlmostEqual(back._pi_c, 2.3e-10)
        self.assertAlmostEqual(back._pi_absw, 1.7e-11)
        self.assertAlmostEqual(back._pi_absw_err, 4.0e-14)

    def test_a_cache_written_before_absw_is_rejected_when_it_is_needed(self):
        """A three-field file is one the fully weighted mode wrote. Reusing it
        for the unweighted output would run on a missing normalisation, so the
        reader says no and the caller re-scans."""
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), 'pure_interference_c')
        with open(path, 'w') as fsock:
            fsock.write('2.3e-10 3.1e-13 2.25e-10\n')
        self.assertFalse(self._Stub(unweighted=True)._read_pi_c_cache(path))
        # ... but the fully weighted mode does not need it and reads on
        weighted = self._Stub(unweighted=False)
        self.assertTrue(weighted._read_pi_c_cache(path))
        self.assertAlmostEqual(weighted._pi_c, 2.3e-10)

    # -- the estimator identity -------------------------------------------

    def _toy(self, seed, bound_factor, w0_rule):
        """A pure-python stand-in for the unweighting loop.

        ``W(p, u) = a_p cos(2 pi u)`` with ``u`` uniform, so ``<W> = 0`` for
        every production point (the defining property of the mode) while
        ``<|W|>`` varies with ``a_p`` -- the local size of the interference.
        The observable is ``O(u) = cos(2 pi u)``, so everything is closed form:

            <W O>  = <a> / 2        <|W|> = <a> * 2/pi

        ``w0_rule(absw, c)`` supplies the weight magnitude under test.
        """
        rng = random.Random(seed)
        a = [1.0 + 3.0 * (k % 7) / 6.0 for k in range(500)]   # a_p in [1, 4]
        c, ref, n = 0.5, 12.0, 400000
        absw = (sum(a) / len(a)) * 2.0 / math.pi
        bound = max(a) * bound_factor
        w0 = w0_rule(absw, c)
        total, n_file = 0.0, 0
        for _ in range(n):
            a_p = a[rng.randrange(len(a))]
            u = rng.random()
            obs = math.cos(2 * math.pi * u)
            W = a_p * obs
            if rng.random() * bound >= abs(W):
                continue
            n_file += 1
            total += math.copysign(w0, W) * obs
        return total / n_file, n_file, n

    def test_the_unweighted_weight_reproduces_the_interference_and_M_cancels(self):
        """The claim being tested, end to end on a toy:

            (1/N_file) sum w O  =  w0 <W O> / <|W|>

        so ``w0 = sigma*BR*<|W|>/c`` makes it the interference contribution
        ``sigma*BR*<W O>/c`` the fully weighted output writes -- with no
        ``M`` in it. Two bounds a factor 4 apart must therefore give the same
        physics from very different file sizes.
        """
        a = [1.0 + 3.0 * (k % 7) / 6.0 for k in range(500)]
        mean_a = sum(a) / len(a)
        target = 12.0 * (mean_a / 2.0) / 0.5          # ref * <W O> / c
        rule = lambda absw, c: 12.0 * absw / c        # noqa: E731
        tight, n_tight, n_read = self._toy(11, 1.0, rule)
        loose, n_loose, _ = self._toy(11, 4.0, rule)
        # the two runs keep very different numbers of events ...
        self.assertGreater(n_tight, 3.0 * n_loose)
        self.assertAlmostEqual(n_tight / float(n_read),
                               mean_a * 2.0 / math.pi / max(a), delta=0.01)
        # ... and agree with each other and with the analytic answer
        self.assertAlmostEqual(tight, target, delta=0.02 * target)
        self.assertAlmostEqual(loose, target, delta=0.03 * target)

    def test_a_mis_measured_absw_biases_the_result_by_exactly_that_factor(self):
        """Why the run does not normalise with the maximum-weight probe's
        <|W|>. Unlike c it is not a decay-side constant, so the probe's
        handful of production events knows it only to ~10% -- and the
        estimator is linear in it, so a 10% error is a 10% error on every
        physics number. Feeding the toy an inflated <|W|> shows the bias is
        exactly the ratio."""
        a = [1.0 + 3.0 * (k % 7) / 6.0 for k in range(500)]
        absw = (sum(a) / len(a)) * 2.0 / math.pi
        target = 12.0 * ((sum(a) / len(a)) / 2.0) / 0.5
        got, _, _ = self._toy(11, 1.0, lambda _absw, c: 12.0 * (1.1 * absw) / c)
        self.assertAlmostEqual(got / target, 1.1, delta=0.03)

    def test_the_realised_keep_rate_normalisation_needs_no_absw_estimate(self):
        """What the run actually writes. Taking <|W|> = (N_file/N_read) * M
        from the run itself makes N_file cancel out of the estimator, so the
        answer is right whatever the probe said and whatever M was. Two very
        different bounds, and a deliberately wrong probe value, all land on
        the same physics."""
        a = [1.0 + 3.0 * (k % 7) / 6.0 for k in range(500)]
        target = 12.0 * ((sum(a) / len(a)) / 2.0) / 0.5
        for bound_factor in (1.0, 4.0):
            # a first pass with a deliberately silly provisional magnitude ...
            _, n_file, n_read = self._toy(11, bound_factor,
                                          lambda absw, c: 1.0)
            # ... and the correction the run applies from its own keep rate
            absw_run = (n_file / float(n_read)) * max(a) * bound_factor
            got, n2, _ = self._toy(11, bound_factor,
                                   lambda absw, c: 12.0 * absw_run / c)
            self.assertEqual(n2, n_file)
            self.assertAlmostEqual(got, target, delta=0.02 * target)

    def test_the_design_notes_weight_would_be_wrong_per_file_event(self):
        """The superseded proposal ``w = +- sigma*BR*maxwgt/c`` normalises per
        event READ, not per event in the file, so a consumer dividing by
        N_file -- the only count an LHE file carries, and what IDWTUP = -4
        means -- is off by exactly ``M/<|W|>``, a bound-dependent factor."""
        a = [1.0 + 3.0 * (k % 7) / 6.0 for k in range(500)]
        target = 12.0 * ((sum(a) / len(a)) / 2.0) / 0.5
        got, _, _ = self._toy(11, 1.0, lambda absw, c: 12.0 * max(a) / c)
        absw = (sum(a) / len(a)) * 2.0 / math.pi
        self.assertAlmostEqual(got / target, max(a) / absw, delta=0.03)
        self.assertGreater(got / target, 1.5)


class TestWeightedDecayOutput(unittest.TestCase):
    """``decay_output = weighted``: the option that drops the accept/reject
    for an ORDINARY (non-interference) run and writes
    ``w = w_prod * BR * W / c`` instead (section 13.18)."""

    class _Stub(object):
        InvalidCmd = interface_madspin.MadSpinInterface.InvalidCmd
        _weighted_decay = interface_madspin.MadSpinInterface._weighted_decay
        _pure_interference_unweighted = \
            interface_madspin.MadSpinInterface._pure_interference_unweighted
        _validate_weighted_decay = \
            interface_madspin.MadSpinInterface._validate_weighted_decay
        _weighted_decay_note = \
            interface_madspin.MadSpinInterface._weighted_decay_note
        _read_lhe_init_cross = inspect.getattr_static(
            interface_madspin.MadSpinInterface, '_read_lhe_init_cross')
        _unweighting_mode = interface_madspin.MadSpinInterface._unweighting_mode
        _announce_mode = interface_madspin.MadSpinInterface._announce_mode
        _log_once = interface_madspin.MadSpinInterface._log_once
        _POL_TOKENS = interface_madspin.MadSpinInterface._POL_TOKENS
        _parse_pol_side = interface_madspin.MadSpinInterface._parse_pol_side
        _borrow_decision_helpers(locals())

        def __init__(self, output='auto', spinmode='onshell',
                     pure_interference='', unweighting='auto'):
            self.options = interface_madspin.MadSpinOptions()
            self.options['decay_output'] = output
            self.options['spinmode'] = spinmode
            self.options['pure_interference'] = pure_interference
            self.options['unweighting'] = unweighting
            self.options['fixed_order'] = False
            self.model = _PIModelStub()
            self._production_polarization_cache = {}

    # -- the predicate ------------------------------------------------------

    def test_off_by_default(self):
        """The shipped default is 'auto', which is 'unweighted' for an
        ordinary run -- what MadSpin has always done."""
        stub = self._Stub()
        self.assertEqual(stub.options['decay_output'], 'auto')
        self.assertEqual(stub._decay_output(), 'unweighted')
        self.assertFalse(stub._weighted_decay())

    def test_an_explicit_unweighted_is_the_same_as_the_default(self):
        stub = self._Stub(output='unweighted')
        self.assertFalse(stub._weighted_decay())

    def test_on_when_asked_for_in_a_density_mode(self):
        for spinmode in ('madspin', 'full', 'PA', 'onshell'):
            stub = self._Stub(output='weighted', spinmode=spinmode)
            self.assertTrue(stub._weighted_decay(), spinmode)

    def test_it_governs_the_interference_output_instead(self):
        """Under pure_interference the option does not step aside: it chooses
        that mode's output shape. The ordinary weighted path stays off -- the
        interference mode reaches the same 'keep every trial' code by its own
        route, with a signed W and a zeroed <init>."""
        stub = self._Stub(output='weighted', pure_interference='t = + -')
        self.assertEqual(stub._decay_output(), 'weighted')
        self.assertFalse(stub._weighted_decay())
        self.assertFalse(stub._pure_interference_unweighted())
        stub._validate_weighted_decay()          # no refusal, no step-aside

        stub = self._Stub(output='unweighted', pure_interference='t = + -')
        self.assertFalse(stub._weighted_decay())
        self.assertTrue(stub._pure_interference_unweighted())
        stub._validate_weighted_decay()

    def test_auto_follows_the_mode(self):
        """'auto' is 'weighted' under pure_interference and 'unweighted'
        otherwise, which is each mode's own historical default."""
        self.assertEqual(self._Stub()._decay_output(), 'unweighted')
        self.assertEqual(
            self._Stub(pure_interference='t = + -')._decay_output(),
            'weighted')

    def test_auto_never_raises_outside_the_density_modes(self):
        """'weighted' is refused there, so the default must not resolve to it
        -- an ordinary spinmode=none card has to keep working."""
        for spinmode in ('madspin_v1', 'onshell_v1', 'none'):
            stub = self._Stub(spinmode=spinmode)
            self.assertEqual(stub._decay_output(), 'unweighted')
            stub._validate_weighted_decay()

    def test_refused_outside_the_density_modes(self):
        for spinmode in ('madspin_v1', 'onshell_v1', 'none'):
            stub = self._Stub(output='weighted', spinmode=spinmode)
            self.assertFalse(stub._weighted_decay(), spinmode)
            self.assertRaises(stub.InvalidCmd, stub._validate_weighted_decay)

    def test_an_unweighted_card_validates_silently_everywhere(self):
        for spinmode in ('madspin', 'PA', 'onshell', 'madspin_v1', 'none'):
            self._Stub(spinmode=spinmode)._validate_weighted_decay()

    def test_the_option_rejects_an_unknown_value(self):
        options = interface_madspin.MadSpinOptions()
        options['decay_output'] = 'weighted'
        options['decay_output'] = 'signed'
        self.assertEqual(options['decay_output'], 'weighted')

    def test_the_option_accepts_its_three_spellings(self):
        options = interface_madspin.MadSpinOptions()
        for value in ('auto', 'unweighted', 'weighted'):
            options['decay_output'] = value
            self.assertEqual(options['decay_output'], value)

    # -- it forces the joint path ------------------------------------------

    def test_it_takes_the_joint_path(self):
        """There is no accept/reject to stage, and the joint branch is the one
        that carries the weighted path."""
        for unweighting in ('auto', 'sequential', 'two_stage'):
            stub = self._Stub(output='weighted', unweighting=unweighting)
            self.assertEqual(stub._unweighting_mode(True), 'joint', unweighting)

    def test_it_does_not_touch_the_scheme_when_off(self):
        stub = self._Stub(unweighting='sequential')
        self.assertNotEqual(stub._unweighting_mode(True), 'joint')

    # -- the banner note and its self-check ---------------------------------

    INIT = """<LesHouchesEvents version="3.0">
<header>
</header>
<init>
2212 2212 6.5e+03 6.5e+03 0 0 247000 247000 -4 1
2.0e+01 1.0e-02 2.0e+01 1
</init>
</LesHouchesEvents>
"""

    def _note(self, weights, br_correction=1.0):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), 'out.lhe')
        with open(path, 'w') as f:
            f.write(self.INIT)
        stub = self._Stub(output='weighted')
        stub._pi_c, stub._pi_c_err = 2.25e-10, 3.0e-13
        stub._pi_c_stats = {'n': 44016}
        stats = [{'sum_w': sum(weights),
                  'sum_w2': sum(w * w for w in weights),
                  'nb_pi_dead': 0}]
        return '\n'.join(stub._weighted_decay_note(
            path, stats, len(weights), br_correction))

    def test_the_note_compares_mean_w_against_the_reference_sigma_br(self):
        """Under IDWTUP = -4 the cross-section IS the mean of the weights, so
        mean(w) == sigma*BR is the check that c = <W> was measured right --
        the exact analogue of the interference mode's z test, with the target
        at 1 instead of 0."""
        note = self._note([19.0, 20.0, 21.0])
        self.assertIn('Reference sigma * BR   (pb)  : +2.00000000e+01', note)
        self.assertIn('mean(w), the sample XSECUP   : +2.00000000e+01', note)
        self.assertIn('mean(w) / reference          : 1.000000', note)
        self.assertIn('Events written               : 3', note)
        self.assertIn('Normalisation constant     c : +2.25000000e-10', note)

    def test_the_note_reports_a_mis_normalised_sample_as_a_ratio(self):
        note = self._note([22.0, 22.0, 22.0, 22.0])
        self.assertIn('mean(w) / reference          : 1.100000', note)

    def test_the_note_follows_a_br_equalization_rescale(self):
        """<init> is rescaled by the same pass, so the reference the note
        quotes has to be the post-rescale one."""
        note = self._note([10.0, 10.0], br_correction=0.5)
        self.assertIn('Reference sigma * BR   (pb)  : +1.00000000e+01', note)
        self.assertIn('mean(w) / reference          : 1.000000', note)


class TestUnweightRangeWeightPaths(unittest.TestCase):
    """The three shapes ``_unweight_range`` can write, driven through the real
    loop with the matrix element stubbed out.

    This exists because the interference and weighted paths all leave the
    ``while 1`` through the same ``if <no ordinary test> or random()*maxwgt <
    test`` line, and getting that condition wrong is silent: the 'unweighted'
    interference path, which has already made its own decision on |W|, once
    fell through to the signed test and had every negative weight rejected a
    second time. The z check caught it end to end, but only after a run.
    """

    EVENT = """<event>
 4  1 +%.7e 1.00000000e+02 7.54677100e-03 1.30800000e-01
       -1 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 +5.0e+02 5.0e+02 0.0e+00 0.0e+00 1.0
        1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 -5.0e+02 5.0e+02 0.0e+00 0.0e+00 1.0
       11  1    1    2    0    0 +1.0000000e+02 +0.0000000e+00 +0.0e+00 1.0e+02 0.0e+00 0.0e+00 1.0
      -11  1    1    2    0    0 -1.0000000e+02 +0.0000000e+00 +0.0e+00 9.0e+02 0.0e+00 0.0e+00 1.0
</event>"""

    class _Sink(object):
        def __init__(self):
            self.events = []

        def write_events(self, evt):
            self.events.append(evt)

    class _Stub(object):
        """Enough MadSpinInterface for the joint branch of _unweight_range."""
        InvalidCmd = interface_madspin.MadSpinInterface.InvalidCmd
        _unweight_range = interface_madspin.MadSpinInterface._unweight_range
        _dead_trial = interface_madspin.MadSpinInterface._dead_trial
        _POL_TOKENS = interface_madspin.MadSpinInterface._POL_TOKENS
        _parse_pol_side = interface_madspin.MadSpinInterface._parse_pol_side
        _pure_interference = \
            interface_madspin.MadSpinInterface._pure_interference

        def __init__(self, weights, pure_interference=''):
            self.options = interface_madspin.MadSpinOptions()
            self.options['pure_interference'] = pure_interference
            self.model = _PIModelStub()
            self.options['fixed_order'] = False
            self.options['density_keep_jacobian'] = False
            self.branching_ratio = 2.0
            self.efficiency = 1.0
            self._weights = list(weights)
            self._draw = 0

        # -- the pieces the loop calls out to -----------------------------
        def get_decay_from_file(self, production, evt_decayfile, nb_remain):
            return {}

        def get_onshell_evt_and_wgt(self, prod, decays, decay_dict,
                                    cached=None, build_event=False):
            wgt = self._weights[self._draw % len(self._weights)]
            self._draw += 1
            return None, wgt, 'density'

        def _add_polarization_weights(self, evt, ratios):
            pass

    def _ctx(self, **over):
        ctx = dict(maxwgt=1.0, maxwgts=[], sequential=False, decay_dict={},
                   drop_prob_per_pdg={}, mixed_pdgs_set=set(),
                   density_method=True, density_pole_approximation=True,
                   density_needs_reshuffle=False, shard_nb_event=10,
                   branching_ratio=2.0, base_seed=1,
                   pure_interference_c=0.5, pure_interference_absw=0.25,
                   pure_interference_unweighted=False, weighted_decay=False)
        ctx.update(over)
        return ctx

    def _run(self, stub, ctx, nb_events=4):
        source = [lhe_parser.Event(self.EVENT % 1.0) for _ in range(nb_events)]
        sink = self._Sink()
        stats = stub._unweight_range(source, {}, sink, ctx)
        return [e.wgt for e in sink.events], stats

    # ------------------------------------------------------------------

    def test_the_ordinary_path_still_accepts_and_rejects(self):
        """A weight of 0.5 against maxwgt 1.0 keeps roughly half the trials,
        and every written event carries w_prod * BR with no W on it."""
        random.seed(7)
        stub = self._Stub([0.5])
        wgts, stats = self._run(stub, self._ctx(), nb_events=50)
        self.assertEqual(len(wgts), 50)          # redraw-until-accept
        self.assertEqual(set(round(w, 9) for w in wgts), {2.0})
        self.assertGreater(stats['nb_try'], 60)  # ... and it did redraw

    def test_the_fully_weighted_interference_path_keeps_every_trial(self):
        random.seed(7)
        stub = self._Stub([0.5, -0.25], pure_interference='w+ = 0 T')
        wgts, stats = self._run(stub, self._ctx(), nb_events=4)
        self.assertEqual(stats['nb_try'], 4)     # one draw per event, no redraw
        # w = w_prod * BR * W / c, c = 0.5
        self.assertEqual([round(w, 9) for w in wgts],
                         [2.0, -1.0, 2.0, -1.0])

    def test_the_weighted_decay_path_keeps_every_trial(self):
        random.seed(7)
        stub = self._Stub([0.5, 0.25])
        wgts, stats = self._run(stub, self._ctx(weighted_decay=True),
                                nb_events=4)
        self.assertEqual(stats['nb_try'], 4)
        self.assertEqual([round(w, 9) for w in wgts], [2.0, 1.0, 2.0, 1.0])

    def test_the_weighted_decay_path_zeroes_a_failed_reshuffle(self):
        """W <= 0 outside the interference mode means jac <= 0, i.e. a mass
        set the production could not be reshuffled onto: weight 0, counted."""
        random.seed(7)
        stub = self._Stub([0.5, -0.5])
        wgts, stats = self._run(stub, self._ctx(weighted_decay=True),
                                nb_events=4)
        self.assertEqual([round(w, 9) for w in wgts], [2.0, 0.0, 2.0, 0.0])
        self.assertEqual(stats['nb_pi_dead'], 2)

    def test_the_unweighted_interference_path_does_not_test_twice(self):
        """THE regression. Every trial here has |W| = maxwgt, so the |W| test
        accepts all of them; a second, signed test would throw away every
        negative one. Both signs must survive, with one magnitude."""
        random.seed(7)
        stub = self._Stub([1.0, -1.0], pure_interference='w+ = 0 T')
        wgts, stats = self._run(
            stub, self._ctx(pure_interference_unweighted=True), nb_events=20)
        self.assertEqual(len(wgts), 20)
        # w = +- w_prod * BR * <|W|> / c = +- 2.0 * 0.25/0.5
        self.assertEqual(sorted(set(round(w, 9) for w in wgts)), [-1.0, 1.0])
        self.assertEqual(sum(1 for w in wgts if w > 0), 10)
        self.assertEqual(sum(1 for w in wgts if w < 0), 10)
        self.assertEqual(stats['nb_pi_reject'], 0)

    def test_the_unweighted_interference_path_drops_on_rejection(self):
        """|W| = maxwgt/4 keeps about a quarter, and writes nothing for the
        rest -- one draw, no redraw."""
        random.seed(11)
        stub = self._Stub([0.25, -0.25], pure_interference='w+ = 0 T')
        wgts, stats = self._run(
            stub, self._ctx(pure_interference_unweighted=True), nb_events=400)
        self.assertEqual(stats['nb_try'], 400)          # exactly one each
        self.assertEqual(len(wgts) + stats['nb_pi_reject'], 400)
        self.assertGreater(len(wgts), 60)
        self.assertLess(len(wgts), 140)
        self.assertEqual(sorted(set(round(abs(w), 9) for w in wgts)), [1.0])
        self.assertTrue(any(w > 0 for w in wgts))
        self.assertTrue(any(w < 0 for w in wgts))


class TestBannerEventWeightRescale(unittest.TestCase):
    """``_rewrite_lhe_banner_cross(event_scale=...)``: the second pass that
    replaces the provisional weight magnitude of the 'unweighted'
    pure-interference output by the one the run realised."""

    LHE = """<LesHouchesEvents version="3.0">
<header>
<MGGenerationInfo>
#  Number of Events        :       2
#  Integrated weight (pb)   : 7.0
</MGGenerationInfo>
</header>
<init>
2212 2212 6.5e+03 6.5e+03 0 0 247000 247000 -4 1
5.0e+02 2.8e-01 5.0e+02 1
</init>
<event>
 4 1 -3.0000000e+00 1.8e+02 7.5e-03 1.1e-01
       21 -1    0    0  503  502 +0.0 +0.0 +1.0 1.0 0.0 0.0e+00 1.0e+00
<rwgt>
<wgt id='r1'> -6.0000000e+00 </wgt>
</rwgt>
</event>
<event>
 4 1 +3.0000000e+00 1.8e+02 7.5e-03 1.1e-01
       21 -1    0    0  503  502 +0.0 +0.0 +1.0 1.0 0.0 0.0e+00 1.0e+00
</event>
</LesHouchesEvents>
"""

    class _Stub(object):
        _RWGT_LINE = interface_madspin.MadSpinInterface._RWGT_LINE
        _rewrite_lhe_banner_cross = \
            interface_madspin.MadSpinInterface._rewrite_lhe_banner_cross

    def _run(self, **kwargs):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), 'out.lhe')
        with open(path, 'w') as f:
            f.write(self.LHE)
        self._Stub()._rewrite_lhe_banner_cross(path, 0.0, **kwargs)
        return open(path).read()

    def _weights(self, text):
        out = []
        in_event = want = False
        for line in text.split('\n'):
            low = line.strip().lower()
            if low.startswith('<event'):
                in_event, want = True, True
                continue
            if low.startswith('</event'):
                in_event = False
                continue
            if in_event and want and len(line.split()) == 6:
                want = False
                out.append(float(line.split()[2]))
            match = interface_madspin.MadSpinInterface._RWGT_LINE.match(line)
            if match:
                out.append(float(match.group(2)))
        return out

    def test_without_event_scale_the_events_are_untouched(self):
        """Every other caller passes nothing, so no event may move."""
        text = self._run()
        self.assertEqual(self._weights(text), [-3.0, -6.0, 3.0])
        # the <init> block is still zeroed by ratio, as before
        self.assertIn('+0.0000000e+00 +0.0000000e+00 +0.0000000e+00 1', text)

    def test_event_scale_multiplies_xwgtup_and_the_rwgt_entries(self):
        text = self._run(event_scale=0.5)
        self.assertEqual(self._weights(text), [-1.5, -3.0, 1.5])

    def test_event_scale_leaves_the_particle_lines_and_the_banner_alone(self):
        """The particle lines have 13 fields, not 6, and nothing outside an
        <event> is an event at all."""
        text = self._run(event_scale=0.5)
        self.assertIn('21 -1    0    0  503  502 +0.0 +0.0 +1.0 1.0 0.0 '
                      '0.0e+00 1.0e+00', text)
        self.assertIn('2212 2212 6.5e+03 6.5e+03 0 0 247000 247000 -4 1', text)
        self.assertEqual(text.count('<event>'), 2)
        self.assertEqual(text.count('</event>'), 2)

    def test_the_note_block_still_goes_in_with_a_scale(self):
        text = self._run(event_scale=2.0, note=['#  hello'],
                         note_tag='MGPureInterference', n_written=7)
        self.assertIn('<MGPureInterference>\n#  hello\n'
                      '</MGPureInterference>', text)
        self.assertIn('#  Number of Events        :       7', text)
        self.assertEqual(self._weights(text), [-6.0, -12.0, 6.0])


class TestProductionPolarizationPlumbing(unittest.TestCase):
    """Reading the production polarisation and turning it into the basis /
    restriction the density matrices are built with."""

    class _Stub(object):
        """Just enough MadSpinInterface for the polarisation helpers."""
        InvalidCmd = interface_madspin.MadSpinInterface.InvalidCmd

        def __init__(self, pol_map=None, spinmode='madspin'):
            self._pol = pol_map or {}
            self.options = {'spinmode': spinmode}
            self.list_branches = {}

        def _production_polarization(self):
            return self._pol

        _density_spinmode = interface_madspin.MadSpinInterface._density_spinmode
        _apply_production_polarization = \
            interface_madspin.MadSpinInterface._apply_production_polarization
        _format_polarization_sequence = staticmethod(
            interface_madspin.MadSpinInterface._format_polarization_sequence)
        do_decay = interface_madspin.MadSpinInterface.do_decay
        # this test drives the real _unweighting_mode, which now consults the
        # interference mode and the spinmode family before resolving
        _pure_interference = \
            interface_madspin.MadSpinInterface._pure_interference
        _spinmode_has_density = \
            interface_madspin.MadSpinInterface._spinmode_has_density

    HEL = {1: [0], 2: [1, -1], 3: [-1, 0, 1]}

    def test_no_braces_leaves_the_basis_untouched(self):
        """Nothing may move for an unpolarised production process."""
        stub = self._Stub()
        hels = [self.HEL[2], self.HEL[3]]
        got, restriction = stub._apply_production_polarization([6, 24], hels)
        self.assertEqual(got, hels)
        self.assertIsNone(restriction)

    def test_single_state_puts_its_helicity_first(self):
        """GET_DENSITY matches the process NHEL table against the *first*
        ALLOW_HEL combination, and a polarised process has no NHEL row outside
        its polarisation -- so the allowed helicity has to lead, or the whole
        production density matrix comes back zero."""
        stub = self._Stub({24: ((0,),)})
        got, restriction = stub._apply_production_polarization(
                                        [24], [list(self.HEL[3])])
        self.assertEqual(got, [[0, -1, 1]])
        self.assertEqual(restriction, ((0,),))

        stub = self._Stub({6: ((-1,),)})
        got, restriction = stub._apply_production_polarization(
                                        [6], [list(self.HEL[2])])
        self.assertEqual(got, [[-1, 1]])
        self.assertEqual(restriction, ((-1,),))

    def test_transverse_keeps_two_states_and_drops_zero_from_the_front(self):
        stub = self._Stub({24: ((-1, 1),)})
        got, restriction = stub._apply_production_polarization(
                                        [24], [list(self.HEL[3])])
        self.assertEqual(got, [[-1, 1, 0]])
        self.assertEqual(restriction, ((-1, 1),))

    def test_restriction_is_per_particle(self):
        """t{0} t~{T}: slot 1 collapses onto its diagonal 0 entry, slot 2 keeps
        the -1/+1 block, and an unpolarised third particle keeps everything."""
        stub = self._Stub({24: ((0,),), -24: ((-1, 1),)})
        got, restriction = stub._apply_production_polarization(
                    [24, -24, 6], [list(self.HEL[3]), list(self.HEL[3]),
                                   list(self.HEL[2])])
        self.assertEqual(got, [[0, -1, 1], [-1, 1, 0], [1, -1]])
        self.assertEqual(restriction, ((0,), (-1, 1), None))

    def test_unsupported_polarization_is_refused(self):
        """{A}, {G}, ... have no place in the -1/0/+1 helicity basis the density
        matrices are built on."""
        stub = self._Stub({24: ((99,),)})
        self.assertRaises(stub.InvalidCmd,
                          stub._apply_production_polarization,
                          [24], [list(self.HEL[3])])
        # a longitudinal brace on a fermion cannot be honoured either
        stub = self._Stub({6: ((0,),)})
        self.assertRaises(stub.InvalidCmd,
                          stub._apply_production_polarization,
                          [6], [list(self.HEL[2])])

    def test_decay_side_polarization_is_an_explicit_error(self):
        """Polarisation on a 'decay' line is not what the density modes
        contract, so it must fail loudly rather than be silently ignored."""
        for spinmode in ('madspin', 'full', 'PA', 'onshell'):
            stub = self._Stub(spinmode=spinmode)
            try:
                stub.do_decay('t > w+{0} b')
            except stub.InvalidCmd as error:
                self.assertIn('not supported', str(error))
                self.assertIn(spinmode, str(error))
            else:
                self.fail('decay-side polarization accepted for %s' % spinmode)

    def test_density_spinmode_detection(self):
        for mode in ('madspin', 'full', 'PA', 'onshell'):
            self.assertTrue(self._Stub(spinmode=mode)._density_spinmode())
        for mode in ('none', 'madspin_v1', 'onshell_v1'):
            self.assertFalse(self._Stub(spinmode=mode)._density_spinmode())


class TestSamePdgProductionPolarization(unittest.TestCase):
    """'p p > w+{0} w+{T}': the same pdg twice with different braces.

    The density basis lays its slots out as 'for pdg in decays_key, in
    production-event order', so a pdg owns a contiguous block of slots whose
    k-th entry is the k-th such particle of the event; the k-th brace of that
    pdg in the process line goes to it.
    """

    Stub = TestProductionPolarizationPlumbing._Stub
    HEL = TestProductionPolarizationPlumbing.HEL

    class _Banner(object):
        def __init__(self, lines):
            self.proc_card = list(lines)

    class _PolStub(object):
        """_production_polarization on top of a banner, with the real MG5
        process parser replaced by a minimal one: the point under test is the
        bookkeeping, not MG5's brace syntax (which the parser owns)."""
        InvalidCmd = interface_madspin.MadSpinInterface.InvalidCmd
        _production_polarization = \
            interface_madspin.MadSpinInterface._production_polarization
        _format_polarization_sequence = staticmethod(
            interface_madspin.MadSpinInterface._format_polarization_sequence)

        POL = {'0': [0], 'T': [1, -1], '+': [1], '-': [-1], 'A': [99]}
        NAMES = {'w+': [24], 'w-': [-24], 'z': [23], 't': [6], 't~': [-6],
                 'p': [21, 2, -2], 'j': [21, 2, -2], 'V': [24, -24, 23]}

        class _Leg(dict):
            def get(self, key):
                return self[key]

        class _Proc(dict):
            def get(self, key):
                return self[key]

        def __init__(self, lines):
            self.banner = TestSamePdgProductionPolarization._Banner(lines)
            self.mg5cmd = self

        def extract_process(self, line):
            legs = []
            initial, final = line.split('>')
            for state, part in ([(False, p) for p in initial.split()] +
                                [(True, p) for p in final.split()]):
                pol = []
                if '{' in part:
                    part, brace = part.split('{')
                    pol = self.POL[brace.rstrip('}')]
                legs.append(self._Leg(ids=self.NAMES[part], state=state,
                                      polarization=pol))
            return self._Proc(legs=legs)

    def polarization(self, *lines):
        return self._PolStub(lines)._production_polarization()

    # ------------------------------------------------------------------
    # reading the process line
    # ------------------------------------------------------------------

    def test_same_pdg_two_braces_keeps_both_in_order(self):
        self.assertEqual(self.polarization('generate p p > w+{0} w+{T}'),
                         {24: ((0,), (-1, 1))})
        # ... and the order is the process line's, not sorted
        self.assertEqual(self.polarization('generate p p > w+{T} w+{0}'),
                         {24: ((-1, 1), (0,))})

    def test_uniform_polarisation_collapses_to_one_entry(self):
        """Same brace twice is not a positional case: one entry, broadcast to
        however many of that pdg the event holds."""
        self.assertEqual(self.polarization('generate p p > z{0} z{0}'),
                         {23: ((0,),)})

    def test_partially_polarised_same_pdg(self):
        """'z{0} z': the second Z has no brace and stays summed over."""
        self.assertEqual(self.polarization('generate p p > z{0} z'),
                         {23: ((0,), None)})

    def test_broadcast_survives_extra_subprocesses(self):
        """The multiplicity of a broadcast pdg does not have to match between
        subprocesses -- this is the common 'generate X; add process X j' case."""
        self.assertEqual(
            self.polarization('generate p p > w+{0} w-',
                              'add process p p > w+{0} w- j'),
            {24: ((0,),)})

    def test_same_sequence_in_several_subprocesses_is_fine(self):
        self.assertEqual(
            self.polarization('generate p p > w+{0} w+{T}',
                              'add process p p > w+{0} w+{T} j'),
            {24: ((0,), (-1, 1))})

    def test_subprocesses_that_disagree_are_refused(self):
        """Two lines with the same final state but different brace patterns
        produce indistinguishable events -- refuse rather than pick one."""
        for lines in (('generate p p > w+{0} w+{T}',
                       'add process p p > w+{T} w+{0}'),
                      ('generate p p > w+{0} w+{T}',
                       'add process p p > w+{0} w+{0}'),
                      ('generate p p > w+{0} w-',
                       'add process p p > w+ w-')):
            self.assertRaises(interface_madspin.MadSpinInterface.InvalidCmd,
                              self.polarization, *lines)

    def test_multiparticle_label_with_mixed_polarisation_is_refused(self):
        """'p p > V{0} V{T}' with V a multiparticle label: how many of a given
        pdg an event holds is not fixed by the line, so the n-th brace cannot
        be pinned to the n-th particle."""
        self.assertRaises(interface_madspin.MadSpinInterface.InvalidCmd,
                          self.polarization, 'generate p p > V{0} V{T}')
        # uniform braces inside a multiparticle label stay fine: no positional
        # matching is needed there
        self.assertEqual(self.polarization('generate p p > V{0} V{0}'),
                         {24: ((0,),), -24: ((0,),), 23: ((0,),)})

    def test_no_braces_gives_an_empty_map(self):
        self.assertEqual(self.polarization('generate p p > w+ w-'), {})

    # ------------------------------------------------------------------
    # turning it into the per-slot basis / restriction
    # ------------------------------------------------------------------

    def test_slots_of_one_pdg_take_the_braces_in_order(self):
        stub = self.Stub({24: ((0,), (-1, 1))})
        got, restriction = stub._apply_production_polarization(
                        [24, 24], [list(self.HEL[3]), list(self.HEL[3])])
        # the allowed helicity leads each basis: the first ALLOW_HEL
        # combination is (0, -1), which the polarised NHEL table does contain
        self.assertEqual(got, [[0, -1, 1], [-1, 1, 0]])
        self.assertEqual(restriction, ((0,), (-1, 1)))

    def test_the_other_order_gives_the_other_assignment(self):
        stub = self.Stub({24: ((-1, 1), (0,))})
        got, restriction = stub._apply_production_polarization(
                        [24, 24], [list(self.HEL[3]), list(self.HEL[3])])
        self.assertEqual(got, [[-1, 1, 0], [0, -1, 1]])
        self.assertEqual(restriction, ((-1, 1), (0,)))

    def test_a_single_entry_is_broadcast_to_every_slot(self):
        stub = self.Stub({24: ((0,),)})
        got, restriction = stub._apply_production_polarization(
                        [24, 24], [list(self.HEL[3]), list(self.HEL[3])])
        self.assertEqual(got, [[0, -1, 1], [0, -1, 1]])
        self.assertEqual(restriction, ((0,), (0,)))

    def test_unbraced_occurrence_stays_unrestricted(self):
        stub = self.Stub({23: ((0,), None)})
        got, restriction = stub._apply_production_polarization(
                        [23, 23], [list(self.HEL[3]), list(self.HEL[3])])
        self.assertEqual(got, [[0, -1, 1], [-1, 0, 1]])
        self.assertEqual(restriction, ((0,), None))

    def test_two_pdgs_each_with_their_own_sequence(self):
        """The per-pdg counter must not leak from one pdg block to the next."""
        stub = self.Stub({24: ((0,), (-1, 1)), 23: ((1,),)})
        got, restriction = stub._apply_production_polarization(
                        [24, 24, 23], [list(self.HEL[3])] * 3)
        self.assertEqual(got, [[0, -1, 1], [-1, 1, 0], [1, -1, 0]])
        self.assertEqual(restriction, ((0,), (-1, 1), (1,)))

    def test_length_mismatch_is_refused(self):
        """A positional sequence that does not match the number of that pdg in
        the event cannot be attached one by one."""
        stub = self.Stub({24: ((0,), (-1, 1))})
        self.assertRaises(stub.InvalidCmd,
                          stub._apply_production_polarization,
                          [24], [list(self.HEL[3])])
        self.assertRaises(stub.InvalidCmd,
                          stub._apply_production_polarization,
                          [24, 24, 24], [list(self.HEL[3])] * 3)


class TestKeepWeightForPolarization(unittest.TestCase):
    """keep_weight_for_polarization_vector / _fermion: one extra LHEF v3 weight
    per polarisation COMBINATION -- one per element of the cartesian product
    over the decaying particles, each drawing from the list of its own species --
    equal to nominal * (restricted convolution)/(full convolution).

    The restriction machinery itself is PR #349's; what is tested here is the
    product built out of the *card*, the id that names it slot by slot, and the
    fact that the nominal weight never moves.
    """

    MI = interface_madspin.MadSpinInterface

    FERMION = [1, -1]       # pdg 6,  spin 2
    VECTOR = [-1, 0, 1]     # pdg 23, spin 3
    SCALAR = [0]            # pdg 25, spin 1

    class _Stub(object):
        """Just enough MadSpinInterface for the polarisation-weight helpers."""
        InvalidCmd = interface_madspin.MadSpinInterface.InvalidCmd
        POLARIZATION_SPECIES = \
            interface_madspin.MadSpinInterface.POLARIZATION_SPECIES
        POLARIZATION_COMBINATION_WARN = \
            interface_madspin.MadSpinInterface.POLARIZATION_COMBINATION_WARN
        _polarization_weight_labels = \
            interface_madspin.MadSpinInterface._polarization_weight_labels
        _polarization_weights_enabled = \
            interface_madspin.MadSpinInterface._polarization_weights_enabled
        _polarization_weight_id = staticmethod(
            interface_madspin.MadSpinInterface._polarization_weight_id)
        _polarization_slot_choices = \
            interface_madspin.MadSpinInterface._polarization_slot_choices
        _polarization_combinations = \
            interface_madspin.MadSpinInterface._polarization_combinations
        _polarization_ratios = \
            interface_madspin.MadSpinInterface._polarization_ratios
        _polarization_slot_layout = staticmethod(
            interface_madspin.MadSpinInterface._polarization_slot_layout)
        _polarization_particle_name = \
            interface_madspin.MadSpinInterface._polarization_particle_name
        _declare_polarization_weights = \
            interface_madspin.MadSpinInterface._declare_polarization_weights
        _add_polarization_weights = \
            interface_madspin.MadSpinInterface._add_polarization_weights
        _slot_identity = interface_madspin.MadSpinInterface._slot_identity

        def __init__(self, vector=(), fermion=(), banner=None):
            self.options = {
                'keep_weight_for_polarization_vector': list(vector),
                'keep_weight_for_polarization_fermion': list(fermion)}
            self.banner = {} if banner is None else banner

    # ------------------------------------------------------------------
    # matrices
    # ------------------------------------------------------------------

    def _packed(self, dim, seed):
        """Hermitian matrix in the packed upper-triangular Fortran storage."""
        import numpy as np
        rng = np.random.default_rng(seed)
        arr = (rng.normal(size=dim * (dim + 1) // 2)
               + 1j * rng.normal(size=dim * (dim + 1) // 2)).astype('complex64')
        for i in range(dim):
            arr[i * (2 * dim - i + 1) // 2] = abs(arr[i * (2 * dim - i + 1) // 2])
        return arr

    def _joint(self, hels, seed, diagonal_only=False):
        """A density matrix over the joint helicity index of ``hels``."""
        import itertools
        import numpy as np
        dim = 1
        for h in hels:
            dim *= len(h)
        allowed = []
        for combo in itertools.product(*hels):
            allowed.extend(combo)
        arr = self._packed(dim, seed)
        rho = madspin.DensityMatrix(arr, len(hels), allowed, dim)
        if diagonal_only:
            # kill every interference entry: the only configuration in which the
            # polarisation sum rule can hold at all (see the sum-rule tests)
            rho.values = np.where(rho._diag_mask, rho.values, 0).astype('complex64')
        return rho

    def _brute_force(self, dec, prod, restriction):
        """sum_{(i,j) allowed} rho_dec(i,j) rho_prod(i,j), read off the helicity
        labels -- an implementation independent of the cached row masks."""
        table = {tuple(int(x) for x in lab): val
                 for lab, val in zip(prod.helicities, prod.values)}
        total = 0j
        for lab, val in zip(dec.helicities, dec.values):
            lab = tuple(int(x) for x in lab)
            keep = True
            for k, ok in enumerate(restriction or []):
                if ok is None:
                    continue
                if lab[2 * k] not in ok or lab[2 * k + 1] not in ok:
                    keep = False
                    break
            if keep:
                total += complex(val) * complex(table[lab])
        return total

    #: helicity basis and MG5 spin of the pdgs used below
    BY_PDG = {6: ([1, -1], 2), -6: ([1, -1], 2),
              23: ([-1, 0, 1], 3), 24: ([-1, 0, 1], 3),
              25: ([0], 1)}

    def _static(self, pdgs, base=None, helicities=None):
        """A ``prod_static`` stub for a slot layout given by its pdgs."""
        if helicities is None:
            helicities = [list(self.BY_PDG[p][0]) for p in pdgs]
        return {'helicities': [list(h) for h in helicities],
                'hel_restriction': base,
                'decaying_pdg': list(pdgs),
                'decaying_spins': [self.BY_PDG[p][1] for p in pdgs]}

    # ------------------------------------------------------------------
    # the option itself
    # ------------------------------------------------------------------

    def test_default_is_empty_and_changes_nothing(self):
        """The behaviour-neutrality requirement: unset options must not add a
        weight, must not touch the banner, and must not even build a mask."""
        options = interface_madspin.MadSpinOptions()
        self.assertEqual(options['keep_weight_for_polarization_vector'], [])
        self.assertEqual(options['keep_weight_for_polarization_fermion'], [])

        stub = self._Stub()
        self.assertFalse(stub._polarization_weights_enabled())
        self.assertEqual(stub._polarization_weight_labels('vector'), [])
        self.assertEqual(stub._polarization_weight_labels('fermion'), [])
        stub._declare_polarization_weights()
        self.assertEqual(stub.banner, {})

        prod = self._joint([self.VECTOR], 11)
        dec = self._joint([self.VECTOR], 12)
        static = self._static([23])
        self.assertIsNone(stub._polarization_ratios(prod, dec, static))
        self.assertNotIn('pol_weight_combinations', static)
        self.assertEqual(stub._polarization_combinations(static), [])

        event = self._event()
        before = str(event)
        stub._add_polarization_weights(event, None)
        stub._add_polarization_weights(event, {})
        self.assertEqual(str(event), before)
        self.assertNotIn('<rwgt>', str(event))

    def test_card_accepts_the_documented_spellings(self):
        options = interface_madspin.MadSpinOptions()
        options['keep_weight_for_polarization_vector'] = '[0, T, +, -]'
        self.assertEqual(options['keep_weight_for_polarization_vector'],
                         ['0', 'T', '+', '-'])
        # L/R alias -/+ exactly as MG5's braces do, and the canonical spelling
        # is what is stored (so the weight ids do not depend on the typing)
        options['keep_weight_for_polarization_fermion'] = 'L R t'
        self.assertEqual(options['keep_weight_for_polarization_fermion'],
                         ['-', '+', 'T'])
        # duplicates collapse rather than emitting the same weight twice
        options['keep_weight_for_polarization_fermion'] = '[+, R, +]'
        self.assertEqual(options['keep_weight_for_polarization_fermion'], ['+'])

    def test_card_refuses_a_non_polarisation(self):
        options = interface_madspin.MadSpinOptions()
        self.assertRaises(banner.InvalidCmd, options.__setitem__,
                          'keep_weight_for_polarization_vector', '[0, A]')
        self.assertRaises(banner.InvalidCmd, options.__setitem__,
                          'keep_weight_for_polarization_fermion', '[+, A]')

    def test_needs_frame_axis_covers_the_three_projections(self):
        """A helicity *projection* does not commute with a boost, so it only
        means what the user asked for on MG5's quantisation axis (frame_id). The
        polarisation weights are the same projection as a production brace, so
        the predicate that _frame_boost's guard tests has to be true for them.
        TestFrameBoost.test_frame_boost_matches_needs_frame_axis pins the guard
        itself onto this predicate."""
        class Frame(self._Stub):
            _needs_frame_axis = \
                interface_madspin.MadSpinInterface._needs_frame_axis
            # the fourth clause: the interference mode is a projection too.
            # Off by default here; the case below overrides it.
            def _pure_interference(self):
                return {}

            def __init__(self, beampol=None, braces=None, **kwargs):
                super(Frame, self).__init__(**kwargs)
                self._beam = beampol
                self._braces = braces or {}

            def _beampol(self):
                return self._beam

            def _production_polarization(self):
                return self._braces

        # nothing projected: the contraction is a trace and the lab will do
        self.assertFalse(Frame()._needs_frame_axis())
        # ... each of the three clauses on its own
        self.assertTrue(Frame(beampol=(2.0, 1.0))._needs_frame_axis())
        self.assertTrue(Frame(braces={23: ((0,),)})._needs_frame_axis())
        self.assertTrue(Frame(vector=['0'])._needs_frame_axis())
        class FrameInterference(Frame):
            def _pure_interference(self):
                return {6: ((0,), (-1, 1))}

        self.assertTrue(FrameInterference()._needs_frame_axis())
        self.assertTrue(Frame(fermion=['+'])._needs_frame_axis())

    def test_the_two_species_lists_are_independent(self):
        options = interface_madspin.MadSpinOptions()
        options['keep_weight_for_polarization_vector'] = '[0, T]'
        self.assertEqual(options['keep_weight_for_polarization_fermion'], [])
        options['keep_weight_for_polarization_fermion'] = '[+, -]'
        self.assertEqual(options['keep_weight_for_polarization_vector'],
                         ['0', 'T'])

    def test_the_same_list_on_both_species_drops_the_unphysical_entry(self):
        """A card that gives both species the same list is legal and each side
        is canonicalised on its own; the '0' is then unphysical for a fermion
        slot and is dropped when the combinations are built, so it does not emit
        a duplicate column."""
        options = interface_madspin.MadSpinOptions()
        options['keep_weight_for_polarization_vector'] = '[0, R, -]'
        options['keep_weight_for_polarization_fermion'] = '[0, R, -]'
        self.assertEqual(options['keep_weight_for_polarization_vector'],
                         ['0', '+', '-'])
        self.assertEqual(options['keep_weight_for_polarization_fermion'],
                         ['0', '+', '-'])
        stub = self._Stub(vector=['0', '+', '-'], fermion=['0', '+', '-'])
        self.assertEqual(
            [wid for wid, _ in stub._polarization_combinations(
                self._static([6]))],
            ['ms_pol_6:+', 'ms_pol_6:-'])

    def test_label_parsing(self):
        parse = interface_madspin.parse_polarization_label
        self.assertEqual(parse('0'), ('0', (0,)))
        self.assertEqual(parse('+'), ('+', (1,)))
        self.assertEqual(parse('R'), ('+', (1,)))
        self.assertEqual(parse('-'), ('-', (-1,)))
        self.assertEqual(parse('l'), ('-', (-1,)))
        self.assertEqual(parse('T'), ('T', (-1, 1)))
        self.assertEqual(parse('{T}'), ('T', (-1, 1)))
        self.assertIsNone(parse('A'))
        self.assertIsNone(parse(''))

    # ------------------------------------------------------------------
    # the product of the per-particle combinations
    # ------------------------------------------------------------------

    def test_ttz_is_the_full_two_by_two_by_four_product(self):
        """The headline case: 'p p > t t~ z' with the vector list [0, T, +, -]
        and the fermion list [+, -] is 2*2*4 = 16 weights, not 4."""
        stub = self._Stub(vector=['0', 'T', '+', '-'], fermion=['+', '-'])
        combos = stub._polarization_combinations(self._static([6, -6, 23]))
        self.assertEqual(len(combos), 16)
        self.assertEqual([wid for wid, _ in combos],
                         ['ms_pol_6:%s_-6:%s_23:%s' % (t, tb, z)
                          for t in '+-' for tb in '+-' for z in ['0', 'T', '+', '-']])
        # every slot really carries its own restriction
        got = dict(combos)
        self.assertEqual(got['ms_pol_6:+_-6:-_23:0'], ((1,), (-1,), (0,)))
        self.assertEqual(got['ms_pol_6:-_-6:-_23:T'], ((-1,), (-1,), (-1, 1)))

    def test_the_product_is_deterministic_and_slot_ordered(self):
        """The ids must be reproducible run to run: the slot order is the
        density basis one and the label order is the one the card lists, with
        the LAST slot varying fastest (itertools.product)."""
        first = self._Stub(vector=['T', '0'], fermion=['-', '+'])
        second = self._Stub(vector=['T', '0'], fermion=['-', '+'])
        a = [wid for wid, _ in first._polarization_combinations(
            self._static([6, 23]))]
        b = [wid for wid, _ in second._polarization_combinations(
            self._static([6, 23]))]
        self.assertEqual(a, b)
        self.assertEqual(a, ['ms_pol_6:-_23:T', 'ms_pol_6:-_23:0',
                             'ms_pol_6:+_23:T', 'ms_pol_6:+_23:0'])

    def test_the_id_names_every_slot_including_same_pdg_ones(self):
        """The id has one token per slot, in slot order, so two Zs are told
        apart by position -- which is what 'ms_pol_X' could not do."""
        stub = self._Stub(vector=['0', 'T'])
        combos = stub._polarization_combinations(self._static([23, 23]))
        self.assertEqual([wid for wid, _ in combos],
                         ['ms_pol_23:0_23:0', 'ms_pol_23:0_23:T',
                          'ms_pol_23:T_23:0', 'ms_pol_23:T_23:T'])
        self.assertEqual(dict(combos)['ms_pol_23:0_23:T'], ((0,), (-1, 1)))
        # and the id builder itself, on its own
        self.assertEqual(
            self.MI._polarization_weight_id([(6, '+'), (-6, None), (23, '0')]),
            'ms_pol_6:+_-6:*_23:0')

    def test_a_species_with_an_empty_list_does_not_multiply_the_count(self):
        """Only the vector list set: the tops stay summed over, contribute a
        single '*' entry each, and the product is the Z's four choices."""
        stub = self._Stub(vector=['0', 'T', '+', '-'])
        combos = stub._polarization_combinations(self._static([6, -6, 23]))
        self.assertEqual(len(combos), 4)
        self.assertEqual([wid for wid, _ in combos],
                         ['ms_pol_6:*_-6:*_23:%s' % z
                          for z in ['0', 'T', '+', '-']])
        self.assertEqual(dict(combos)['ms_pol_6:*_-6:*_23:0'],
                         (None, None, (0,)))
        # the mirror case
        stub = self._Stub(fermion=['+', '-'])
        combos = stub._polarization_combinations(self._static([6, -6, 23]))
        self.assertEqual(len(combos), 4)
        self.assertEqual(combos[0][0], 'ms_pol_6:+_-6:+_23:*')

    def test_a_scalar_slot_contributes_one_unrestricted_entry(self):
        """A spin-0 particle has a 1x1 density matrix and no polarisation: its
        slot is a single '*' rather than a factor in the product."""
        stub = self._Stub(vector=['0', 'T'], fermion=['+', '-'])
        combos = stub._polarization_combinations(self._static([25, 23]))
        self.assertEqual([wid for wid, _ in combos],
                         ['ms_pol_25:*_23:0', 'ms_pol_25:*_23:T'])
        self.assertEqual(dict(combos)['ms_pol_25:*_23:0'], (None, (0,)))
        # a scalar has no choices of its own, so a process of scalars only asks
        # for nothing at all -- rather than one weight equal to the nominal
        self.assertEqual(stub._polarization_combinations(self._static([25])), [])
        self.assertEqual(stub._polarization_combinations(
            self._static([25, 25])), [])

    def test_unphysical_labels_are_dropped_from_a_slot(self):
        """'0' is not a fermion helicity: it is dropped from the top's choices
        instead of being kept as an unrestricted duplicate of another weight.
        A slot left with nothing falls back to a single '*' entry."""
        stub = self._Stub(fermion=['0', '+', '-'])
        combos = stub._polarization_combinations(self._static([6]))
        self.assertEqual([wid for wid, _ in combos],
                         ['ms_pol_6:+', 'ms_pol_6:-'])
        # every label unphysical -> the slot is summed over, and since no slot
        # then carries a label there is nothing to emit
        stub = self._Stub(fermion=['0'])
        self.assertEqual(stub._polarization_slot_choices(self._static([6])),
                         [[(None, None)]])
        self.assertEqual(stub._polarization_combinations(self._static([6])), [])
        # ... but a *second* slot that does have choices keeps the '*' company
        stub = self._Stub(fermion=['0'], vector=['0', 'T'])
        self.assertEqual(
            [wid for wid, _ in stub._polarization_combinations(
                self._static([6, 23]))],
            ['ms_pol_6:*_23:0', 'ms_pol_6:*_23:T'])

    def test_duplicate_slot_restrictions_collapse(self):
        """Two labels that end up selecting the same helicities on one slot (a
        {+} production leg offered both 'T' and '+') must not produce two
        identical columns."""
        stub = self._Stub(vector=['T', '+', '0'])
        static = self._static([23], base=((1,),))
        self.assertEqual([wid for wid, _ in
                          stub._polarization_combinations(static)],
                         ['ms_pol_23:T'])

    def test_combinations_are_cached_on_the_production_static(self):
        stub = self._Stub(vector=['T'])
        static = self._static([23])
        first = stub._polarization_combinations(static)
        self.assertIs(first, stub._polarization_combinations(static))
        self.assertIs(first, static['pol_weight_combinations'])

    def test_slot_species_can_be_inferred_without_the_spins(self):
        """``prod_static`` from an older pickle may not carry decaying_spins;
        the basis length is enough to tell the three spins apart."""
        stub = self._Stub(vector=['0'], fermion=['+'])
        static = self._static([6, 23])
        del static['decaying_spins']
        self.assertEqual([wid for wid, _ in
                          stub._polarization_combinations(static)],
                         ['ms_pol_6:+_23:0'])

    # ------------------------------------------------------------------
    # interaction with the production polarisation (PR #349 / #353)
    # ------------------------------------------------------------------

    def test_production_braces_are_intersected_per_slot(self):
        """p p > t{+} t~ z: the nominal convolution is already restricted to a
        right-handed top, so the top slot keeps only the choices compatible with
        it and the other slots are unaffected."""
        stub = self._Stub(vector=['0', 'T', '+', '-'], fermion=['+', '-'])
        static = self._static([6, -6, 23], base=((1,), None, None))
        combos = stub._polarization_combinations(static)
        # the top has one surviving choice, the anti-top two, the Z four
        self.assertEqual(len(combos), 1 * 2 * 4)
        self.assertTrue(all(wid.startswith('ms_pol_6:+_') for wid, _ in combos))
        self.assertEqual(dict(combos)['ms_pol_6:+_-6:-_23:0'],
                         ((1,), (-1,), (0,)))

    def test_same_pdg_mixed_production_braces_are_intersected_per_slot(self):
        """p p > z{0} z{T} (#353): the two slots carry *different* production
        restrictions. One label applied to both slots at once was empty on
        every choice -- all four weights came back exactly 0. The product form
        keeps the three assignments that are compatible slot by slot."""
        stub = self._Stub(vector=['0', 'T', '+', '-'])
        static = self._static([23, 23], base=((0,), (-1, 1)),
                              helicities=[[0, -1, 1], [-1, 1, 0]])
        combos = stub._polarization_combinations(static)
        self.assertEqual([wid for wid, _ in combos],
                         ['ms_pol_23:0_23:T', 'ms_pol_23:0_23:+',
                          'ms_pol_23:0_23:-'])
        got = dict(combos)
        self.assertEqual(got['ms_pol_23:0_23:T'], ((0,), (-1, 1)))
        self.assertEqual(got['ms_pol_23:0_23:+'], ((0,), (1,)))
        self.assertEqual(got['ms_pol_23:0_23:-'], ((0,), (-1,)))
        # with an unbraced second Z the first one is still pinned
        static = self._static([23, 23], base=((0,), None),
                              helicities=[[0, -1, 1], [-1, 0, 1]])
        self.assertEqual([wid for wid, _ in
                          stub._polarization_combinations(static)],
                         ['ms_pol_23:0_23:0', 'ms_pol_23:0_23:T',
                          'ms_pol_23:0_23:+', 'ms_pol_23:0_23:-'])

    def test_the_denominator_is_the_restricted_convolution(self):
        """With production braces the ratio must be taken against the nominal --
        already restricted -- convolution, or it would not be the fraction of
        what is actually written out."""
        import numpy as np
        stub = self._Stub(vector=['0'])
        static = self._static([6, 23], base=((1,), None))
        hels = [self.FERMION, self.VECTOR]
        prod = self._joint(hels, 41)
        prod.set_hel_restriction(((1,), None))
        dec = self._joint(hels, 42)
        ratio = stub._polarization_ratios(prod, dec, static)['ms_pol_6:*_23:0']
        num = self._brute_force(dec, prod, ((1,), (0,)))
        den = self._brute_force(dec, prod, ((1,), None))
        self.assertTrue(np.allclose(ratio, (num / den).real, atol=1e-5))
        # and the production matrix comes back exactly as it went in
        self.assertEqual(prod.hel_restriction, ((1,), None))

    # ------------------------------------------------------------------
    # the ratio and the emitted weight
    # ------------------------------------------------------------------

    def test_ratio_matches_an_independent_contraction(self):
        import numpy as np
        stub = self._Stub(vector=['0', 'T', '+', '-'], fermion=['+', '-'])
        hels = [self.FERMION, self.VECTOR]
        static = self._static([6, 23])
        prod = self._joint(hels, 51)
        dec = self._joint(hels, 52)
        ratios = stub._polarization_ratios(prod, dec, static)
        self.assertEqual(len(ratios), 8)
        full = self._brute_force(dec, prod, None)
        for wid, restriction in stub._polarization_combinations(static):
            expected = (self._brute_force(dec, prod, restriction) / full).real
            self.assertTrue(np.allclose(ratios[wid], expected, atol=1e-5),
                            '%s: %s != %s' % (wid, ratios[wid], expected))
        # nothing was left attached to the production matrix
        self.assertIsNone(prod.hel_restriction)

    def test_nominal_contraction_is_untouched(self):
        """The nominal weight is what the accept/reject and the cross-section
        are built on: computing the extra weights must not perturb it."""
        import numpy as np
        hels = [self.FERMION, self.VECTOR]
        prod = self._joint(hels, 61)
        dec = self._joint(hels, 62)
        before = dec.scalar_multiplication(prod)
        self._Stub(vector=['0', 'T', '+', '-'],
                   fermion=['+', '-'])._polarization_ratios(
            prod, dec, self._static([6, 23]))
        self.assertTrue(np.allclose(dec.scalar_multiplication(prod), before))

    def test_joint_and_sequential_agree(self):
        """The joint path contracts the raw decay tensor, the sequential one the
        tensor of *normalised* per-slot densities (D/Tr D). Those differ by an
        overall scalar per slot, which cancels in the ratio -- so both paths must
        hand back the same polarisation weights for the same chain."""
        import numpy as np
        hels = [self.FERMION, self.VECTOR]
        static = self._static([6, 23])
        prod = self._joint(hels, 111)
        slots = {0: self._joint([self.FERMION], 112),
                 1: self._joint([self.VECTOR], 113)}
        joint_dec = slots[0].tensor_product(slots[1])
        seq_dec = interface_madspin.decay_density_tensor(
            interface_madspin.MadSpinInterface._slot_identity.__get__(
                TestPartialDensityContraction._Stub()), hels, slots)
        a = self._Stub(vector=['0', 'T', '+', '-'],
                       fermion=['+', '-'])._polarization_ratios(
            prod, joint_dec, dict(static))
        b = self._Stub(vector=['0', 'T', '+', '-'],
                       fermion=['+', '-'])._polarization_ratios(
            prod, seq_dec, dict(static))
        self.assertEqual(sorted(a), sorted(b))
        for wid in a:
            self.assertTrue(np.allclose(a[wid], b[wid], atol=1e-5),
                            '%s: %s != %s' % (wid, a[wid], b[wid]))

    def _event(self, wgt=3.5):
        text = """<event>
 4  1 +%.7e 1.00000000e+02 7.54677100e-03 1.30800000e-01
       -1 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 +5.0e+02 5.0e+02 0.0e+00 0.0e+00 1.0
        1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 -5.0e+02 5.0e+02 0.0e+00 0.0e+00 1.0
       11  1    1    2    0    0 +1.0000000e+02 +0.0000000e+00 +0.0e+00 1.0e+02 0.0e+00 0.0e+00 1.0
      -11  1    1    2    0    0 -1.0000000e+02 +0.0000000e+00 +0.0e+00 9.0e+02 0.0e+00 0.0e+00 1.0
</event>""" % wgt
        return lhe_parser.Event(text)

    def test_emitted_weight_is_nominal_times_the_ratio(self):
        """The value that lands in the <rwgt> block, and the fact that the
        nominal weight of the event is not modified."""
        import numpy as np
        stub = self._Stub(vector=['0'], fermion=['+'])
        event = self._event(wgt=3.5)
        stub._add_polarization_weights(event, {'ms_pol_6:+_23:0': 0.25,
                                               'ms_pol_6:+_23:T': 0.5})
        self.assertEqual(event.wgt, 3.5)
        wgts = event.parse_reweight()
        self.assertTrue(np.allclose(wgts['ms_pol_6:+_23:0'], 3.5 * 0.25))
        self.assertTrue(np.allclose(wgts['ms_pol_6:+_23:T'], 3.5 * 0.5))
        text = str(event)
        self.assertIn("<wgt id='ms_pol_6:+_23:0'>", text)
        self.assertIn("<wgt id='ms_pol_6:+_23:T'>", text)
        # round trip through the parser: the ':' and '*' of the id must survive
        again = lhe_parser.Event(text).parse_reweight()
        self.assertTrue(np.allclose(again['ms_pol_6:+_23:0'], 3.5 * 0.25))
        event = self._event(wgt=2.0)
        stub._add_polarization_weights(event, {'ms_pol_6:*_23:0': 0.5})
        self.assertTrue(np.allclose(
            lhe_parser.Event(str(event)).parse_reweight()['ms_pol_6:*_23:0'],
            1.0))

    def test_existing_event_weights_are_preserved(self):
        import numpy as np
        stub = self._Stub(vector=['0'])
        event = self._event(wgt=2.0)
        event.parse_reweight()['1001'] = 7.0
        stub._add_polarization_weights(event, {'ms_pol_23:0': 0.5})
        wgts = lhe_parser.Event(str(event)).parse_reweight()
        self.assertTrue(np.allclose(wgts['1001'], 7.0))
        self.assertTrue(np.allclose(wgts['ms_pol_23:0'], 1.0))

    # ------------------------------------------------------------------
    # the banner declaration
    # ------------------------------------------------------------------

    def test_slot_layout_matches_the_density_basis_order(self):
        """The banner has to enumerate the ids before a single event is decayed,
        so the slot layout is rebuilt from the topology. It must reproduce
        _decaying_pdgs (first appearance) then _density_basis (per pdg, in
        production order)."""
        layout = self.MI._polarization_slot_layout
        self.assertEqual(layout((6, 23, -6, 23), {6, -6, 23}), (6, 23, 23, -6))
        # a pdg with no decay events is not a slot
        self.assertEqual(layout((6, 23, -6, 21), {6, -6}), (6, -6))
        self.assertEqual(layout((21, 21), {6}), ())

    def test_weights_are_declared_in_the_banner(self):
        # a real Banner, not a dict: Banner.get is get_detail and knows about a
        # handful of card tags only, so 'initrwgt' has to be probed with `in`
        real = banner.Banner()
        stub = self._Stub(vector=['0', 'T'], fermion=['+', '-'], banner=real)
        real['initrwgt'] = "<weightgroup name='other'>\n</weightgroup>\n"
        stub._declare_polarization_weights([self._static([6, 23])])
        text = real['initrwgt']
        self.assertIn("<weightgroup name='madspin_polarization'>", text)
        for wid in ['ms_pol_6:+_23:0', 'ms_pol_6:+_23:T',
                    'ms_pol_6:-_23:0', 'ms_pol_6:-_23:T']:
            self.assertIn("<weight id='%s'>" % wid, text)
        self.assertIn("name='other'", text)
        # idempotent: run_onshell may be re-entered, the block must not double
        stub._declare_polarization_weights([self._static([6, 23])])
        self.assertEqual(text.count("ms_pol_6:+_23:0"),
                         real['initrwgt'].count("ms_pol_6:+_23:0"))

    def test_the_declaration_is_the_union_over_the_topologies(self):
        """'generate p p > t t~' + 'add process p p > t t~ z' put events with
        different slot layouts in the same file; each carries its own weights
        and the banner has to declare both sets."""
        real = banner.Banner()
        stub = self._Stub(vector=['0'], fermion=['+', '-'], banner=real)
        stub._declare_polarization_weights([self._static([6, -6]),
                                            self._static([6, -6, 23])])
        text = real['initrwgt']
        for wid in ['ms_pol_6:+_-6:-', 'ms_pol_6:+_-6:-_23:0']:
            self.assertIn("<weight id='%s'>" % wid, text)

    def test_weights_are_declared_without_a_pre_existing_block(self):
        real = banner.Banner()
        self.assertNotIn('initrwgt', real)
        stub = self._Stub(vector=['+'], banner=real)
        stub._declare_polarization_weights([self._static([23])])
        self.assertIn("<weight id='ms_pol_23:+'>", real['initrwgt'])

    def test_nothing_is_declared_when_nothing_is_requested(self):
        real = banner.Banner()
        self._Stub(banner=real)._declare_polarization_weights(
            [self._static([6, 23])])
        self.assertNotIn('initrwgt', real)
        # ... nor when the requested lists produce no combination at all
        real = banner.Banner()
        self._Stub(vector=['0'], banner=real)._declare_polarization_weights(
            [self._static([25])])
        self.assertNotIn('initrwgt', real)

    def test_a_large_product_warns(self):
        """Combinatorial growth: four decaying vectors with a 4-entry list is
        256 extra weights *and* 256 extra contractions per event. It is legal --
        the user asked for it -- but it is said out loud."""
        real = banner.Banner()
        stub = self._Stub(vector=['0', 'T', '+', '-'], banner=real)
        with self.assertLogs('decay.stdout', level='WARNING') as caught:
            stub._declare_polarization_weights(
                [self._static([23, 23, 23, 24])])
        self.assertTrue(any('256' in line for line in caught.output),
                        caught.output)
        # and below the threshold it stays quiet
        real = banner.Banner()
        stub = self._Stub(vector=['0', 'T'], banner=real)
        stub._declare_polarization_weights([self._static([23, 23])])
        self.assertIn("<weight id='ms_pol_23:0_23:T'>", real['initrwgt'])

    # ------------------------------------------------------------------
    # the sum rule
    # ------------------------------------------------------------------
    # sum_C w_C = w only when the combinations *partition* the (i,j) terms that
    # actually contribute. In the product form that needs two conditions (the
    # one-label-per-weight version needed a third, "only one particle may be
    # restricted", which the product removes):
    #   (a) every species list must partition its slots' helicity basis --
    #       [+, -] for a fermion, [0, +, -] or [0, T] for a vector. The default
    #       vector list [0, T, +, -] does NOT: T = {-1,+1} covers the same
    #       entries as + and - together, so the weights overlap;
    #   (b) the contraction must have no off-diagonal (interference) piece --
    #       the double sum's i != j terms belong to no single-state block. {T}
    #       is the exception that carries its own (-1,+1) block.
    # All of them are tested below, in both directions.

    def test_sum_rule_holds_for_a_partitioning_product(self):
        import numpy as np
        # a vector is partitioned by {+}/{-}/{0}, a fermion by {+}/{-} alone --
        # its '0' entry is unphysical and is dropped from its choices
        for pdgs, hels in (([23], [self.VECTOR]),
                           ([6], [self.FERMION]),
                           ([6, 23], [self.FERMION, self.VECTOR]),
                           ([6, -6], [self.FERMION, self.FERMION])):
            stub = self._Stub(vector=['+', '-', '0'], fermion=['+', '-', '0'])
            prod = self._joint(hels, 71)
            dec = self._joint(hels, 72, diagonal_only=True)
            ratios = stub._polarization_ratios(prod, dec, self._static(pdgs))
            self.assertTrue(np.allclose(sum(ratios.values()), 1.0, atol=1e-5),
                            '%s -> %s' % (pdgs, ratios))

    def test_sum_rule_holds_for_transverse_plus_longitudinal(self):
        """{T} and {0} are the other complete, non-overlapping decomposition of
        a vector -- and {T} keeps its own off-diagonal (-1,+1) block, so it is
        a genuinely different partition of the same nine terms."""
        import numpy as np
        stub = self._Stub(vector=['T', '0'])
        prod = self._joint([self.VECTOR], 81)
        dec = self._joint([self.VECTOR], 82, diagonal_only=True)
        ratios = stub._polarization_ratios(prod, dec, self._static([23]))
        self.assertTrue(np.allclose(sum(ratios.values()), 1.0, atol=1e-5))

    def test_the_product_restores_the_two_particle_sum_rule(self):
        """What the one-weight-per-label form could not do: with both tops
        restricted at once, (+,-) and (-,+) belonged to no weight and the sum
        fell short. The cartesian product has them as combinations of their
        own, so the sum rule comes back."""
        import numpy as np
        hels = [self.FERMION, self.FERMION]
        stub = self._Stub(fermion=['+', '-'])
        prod = self._joint(hels, 101)
        dec = self._joint(hels, 102, diagonal_only=True)
        ratios = stub._polarization_ratios(prod, dec, self._static([6, -6]))
        self.assertEqual(len(ratios), 4)
        self.assertTrue(np.allclose(sum(ratios.values()), 1.0, atol=1e-5))

    def test_sum_rule_fails_on_the_off_diagonal_terms(self):
        """Condition (b): with interference in the contraction the single-state
        blocks cover the diagonal only, so the sum falls short of 1. Pinned so
        the sum rule is not mistaken for an identity."""
        import numpy as np
        stub = self._Stub(vector=['+', '-', '0'])
        prod = self._joint([self.VECTOR], 91)
        dec = self._joint([self.VECTOR], 92)     # full, interference included
        ratios = stub._polarization_ratios(prod, dec, self._static([23]))
        self.assertFalse(np.allclose(sum(ratios.values()), 1.0, atol=1e-3))
        # what the sum *does* reproduce is the diagonal part of the double sum
        full = self._brute_force(dec, prod, None)
        diag = sum(complex(v) * complex(p)
                   for v, p, d in zip(dec.values, prod.values, dec._diag_mask)
                   if d)
        self.assertTrue(np.allclose(sum(ratios.values()),
                                    (diag / full).real, atol=1e-5))

    def test_sum_rule_fails_for_an_overlapping_list(self):
        """Condition (a): the [0, T, +, -] default is not a partition -- T is
        + and - together -- so its combinations double count and the sum
        overshoots. Pinned because it is the list the documentation suggests."""
        import numpy as np
        stub = self._Stub(vector=['0', 'T', '+', '-'])
        prod = self._joint([self.VECTOR], 71)
        dec = self._joint([self.VECTOR], 72, diagonal_only=True)
        ratios = stub._polarization_ratios(prod, dec, self._static([23]))
        self.assertEqual(len(ratios), 4)
        # T is exactly + and - together, so the transverse part is counted twice
        # and the sum overshoots by that fraction
        self.assertTrue(np.allclose(ratios['ms_pol_23:T'],
                                    ratios['ms_pol_23:+'] + ratios['ms_pol_23:-'],
                                    atol=1e-5), ratios)
        self.assertTrue(np.allclose(sum(ratios.values()),
                                    1.0 + ratios['ms_pol_23:T'], atol=1e-5),
                        ratios)
        self.assertFalse(np.allclose(sum(ratios.values()), 1.0, atol=1e-3))


class TestSequentialSlots(unittest.TestCase):
    """_decaying_pdgs / _sequential_slots: which density matrix slot belongs to
    which production particle.

    The sequential accept/reject must know this *before* drawing anything (it
    needs the basis to build N_0), and it must agree exactly with the slot order
    _density_basis lays out -- for pdg in decays_key, in production order -- or
    the tensor product and the production helicity index stop lining up.
    """

    class _Part(object):
        def __init__(self, pid, status=1):
            self.pid = pid
            self.pdg = pid
            self.status = status

    def _production(self):
        # t t~ t plus a gluon spectator, off an initial state
        return [self._Part(2, -1), self._Part(-2, -1), self._Part(6),
                self._Part(-6), self._Part(6), self._Part(21)]

    def _pools(self):
        # two files for the tops, one for the anti-top; 23 never appears
        return {6: {0: 'f', 1: 'f'}, -6: {0: 'f'}, 23: {0: 'f'}}

    def test_decays_key_is_first_appearance_order(self):
        """The order get_decay_from_file fills its dict in."""
        got = interface_madspin.MadSpinInterface._decaying_pdgs(
                        self._production(), self._pools())
        self.assertEqual(got, (6, -6))

    def test_pdg_without_a_pool_is_not_a_slot(self):
        """Same 'does this particle decay' test as _draw_one_decay."""
        production = self._production()
        interface = interface_madspin.MadSpinInterface
        self.assertEqual(interface._decaying_pdgs(production, {6: {}, -6: {0: 'f'}}),
                         (-6,))
        self.assertEqual(interface._decaying_pdgs(production, {}), ())

    def test_slots_match_the_basis_init_part(self):
        """The mapping must resolve to exactly the particles _density_basis
        puts in init_part, in the same order and as the same objects -- they are
        the parents the decays get boosted to."""
        production = self._production()
        interface = interface_madspin.MadSpinInterface
        decays_key = interface._decaying_pdgs(production, self._pools())
        particles, slots = interface._sequential_slots(production, decays_key)

        init_part = [part for pdg in decays_key for part in production
                     if part.pid == pdg and part.status == 1]
        self.assertEqual([particles[i].pid for i in slots],
                         [p.pid for p in init_part])
        for slot, index in enumerate(slots):
            self.assertIs(particles[index], init_part[slot])

    def test_slots_are_grouped_by_pdg_not_production_order(self):
        """t t~ t -> slots are (t, t, t~): grouped by pdg, production order
        inside a group. The indices therefore are not sorted."""
        production = self._production()
        interface = interface_madspin.MadSpinInterface
        decays_key = interface._decaying_pdgs(production, self._pools())
        particles, slots = interface._sequential_slots(production, decays_key)
        self.assertEqual(slots, [0, 2, 1])
        self.assertEqual([p.pid for p in particles], [6, -6, 6, 21])


class TestProductionJacobianForSlots(unittest.TestCase):
    """_production_jacobian_for: J_k, the production jacobian with the slots
    drawn so far offshell and the rest nominal. Each slot carries J_k/J_{k-1}.
    """

    EVENT = """ <event>
 12      1 +4.8368719e+02 1.76709900e+02 7.54677100e-03 1.17102600e-01
         2 -1    0    0  502    0 +0.0000000000e+00 +0.0000000000e+00 +1.6801959055e+02 1.6801959055e+02 0.0000000000e+00  0.0000e+00 1.0000e+00
        -2 -1    0    0    0  501 -0.0000000000e+00 -0.0000000000e+00 -3.6057100553e+02 3.6057100553e+02 0.0000000000e+00  0.0000e+00 -1.0000e+00
         6  2    1    2  502    0 -1.0742571918e+01 -3.4379861756e+01 -2.8025420328e+02 3.3131374285e+02 1.7300000000e+02  0.0000e+00 9.0000e+00
        -6  1    1    2    0  501 +1.0742571918e+01 +3.4379861756e+01 +8.7702788293e+01 1.9727685323e+02 1.7300000000e+02  0.0000e+00 9.0000e+00
         5  1    3    3  502    0 -6.3369583864e+00 +5.5362090397e+01 -7.6229914475e+01 9.4542096209e+01 4.7000000000e+00  0.0000e+00 -1.0000e+00
        24  1    3    3    0    0 -4.4056135319e+00 -8.9741952154e+01 -2.0402428881e+02 2.3677164665e+02 7.9761361725e+01  0.0000e+00 9.0000e+00
        </event>"""

    INFO = (173.0, 1.5, 150.0, 200.0)

    def _production(self):
        evt = lhe_parser.Event()
        evt.parse(self.EVENT)
        return evt

    def test_nothing_offshell_is_unit_jacobian(self):
        """J_0: no mass moved, so the reshuffling is the identity."""
        jac = interface_madspin.MadSpinInterface._production_jacobian_for(
                        self._production(), {0: 0}, {})
        self.assertAlmostEqual(jac, 1.0, places=6)

    def test_leaves_the_production_untouched(self):
        """J_k is needed at every slot; the reshuffling happens once, later."""
        production = self._production()
        before = str(production)
        interface_madspin.MadSpinInterface._production_jacobian_for(
                        production, {0: 0}, {0: (180.0, self.INFO)})
        self.assertEqual(str(production), before)

    def test_offshell_slot_changes_the_jacobian(self):
        """J_1 != J_0 once a virtuality is sampled -- the ratio is what the slot
        carries in its accept/reject weight."""
        interface = interface_madspin.MadSpinInterface
        j_0 = interface._production_jacobian_for(self._production(), {0: 0}, {})
        j_1 = interface._production_jacobian_for(self._production(), {0: 0},
                                                {0: (180.0, self.INFO)})
        self.assertNotAlmostEqual(j_1, j_0, places=3)
        self.assertTrue(0 < j_1 / j_0 < 2)

    def test_impossible_mass_set_is_reported(self):
        """The production-side kinematic failure: the caller trashes the set."""
        jac = interface_madspin.MadSpinInterface._production_jacobian_for(
                        self._production(), {0: 0},
                        {0: (1e6, (173.0, 1.5, 150.0, 2e6))})
        self.assertEqual(jac, -1)


class TestSequentialAcceptReject(unittest.TestCase):
    """sequential_accept_reject: accepting one decaying particle at a time must
    sample the *same* distribution as the joint accept/reject, i.e. p(decays)
    proportional to N_n.

    Driven with synthetic density matrices so no matrix element is needed. The
    decay pool has to be physical for the claim to hold: Dhat = (I + a n.sigma)/2
    for a spin-1/2 parent, and antipodal directions so the pool average is
    exactly I/2 -- the trace property the method rests on. (With a pool that
    violates it the method is genuinely biased; that is the caveat the opt-out
    flag exists for.)
    """

    POOL = 4
    HELS = [[1, -1], [1, -1]]      # t t~

    class _Part(object):
        def __init__(self, pid, status=1):
            self.pid = pid
            self.pdg = pid
            self.status = status

    class _Prod(list):
        sqrts = 1000.0

    def _fermion_decay(self, nhat, alpha=0.9):
        import numpy as np
        nx, ny, nz = nhat
        matrix = 0.5 * np.array([[1 + alpha * nz, alpha * (nx - 1j * ny)],
                                 [alpha * (nx + 1j * ny), 1 - alpha * nz]],
                                dtype=complex)
        arr = np.array([matrix[0, 0], matrix[0, 1], matrix[1, 1]],
                       dtype='complex64')
        return madspin.DensityMatrix(arr, 1, [1, -1], 2)

    def _pool(self, seed):
        import numpy as np
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(self.POOL // 2):
            vec = rng.normal(size=3)
            vec /= np.linalg.norm(vec)
            out.append(self._fermion_decay(vec))
            out.append(self._fermion_decay(-vec))
        return out

    def _production_density(self, seed=11, rank=3, dim=4):
        import numpy as np
        import itertools
        rng = np.random.default_rng(seed)
        matrix = np.zeros((dim, dim), dtype=complex)
        for _ in range(rank):
            vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
            matrix += np.outer(vec, vec.conj())
        arr = np.array([matrix[i, j] for i in range(dim) for j in range(i, dim)],
                       dtype='complex64')
        allowed = []
        for combo in itertools.product(*self.HELS):
            allowed.extend(combo)
        return madspin.DensityMatrix(arr, 2, allowed, dim)

    def _stub(self, rho, pools, unweighting='sequential_with_mass'):
        interface = interface_madspin.MadSpinInterface
        hels = self.HELS
        pool = self.POOL

        class Stub(object):
            _decaying_pdgs = staticmethod(interface._decaying_pdgs)
            _sequential_slots = staticmethod(interface._sequential_slots)
            _slot_identity = interface._slot_identity
            _partial_density_contraction = interface._partial_density_contraction
            _sequential_spin_order = interface._sequential_spin_order
            _decay_slot_order = interface._decay_slot_order
            sequential_accept_reject = interface.sequential_accept_reject
            _polarization_weight_labels = \
                interface._polarization_weight_labels
            _polarization_weights_enabled = \
                interface._polarization_weights_enabled
            # the zero-density / dead-trial guards live on the same loop
            _check_production_density = interface._check_production_density
            _raise_degenerate_weight = interface._raise_degenerate_weight
            _dead_trial = interface._dead_trial
            _scan_maxwgt_range = interface._scan_maxwgt_range
            _sequential_offshell = interface._sequential_offshell
            _sequential_upfront = interface._sequential_upfront
            _upfront_production = interface._upfront_production
            _z_slot_keys = staticmethod(interface._z_slot_keys)
            _zhat = interface._zhat
            _complete_upfront_probe = interface._complete_upfront_probe
            _unweighting_mode = interface._unweighting_mode
            _announce_mode = interface._announce_mode
            _log_once = interface._log_once
            _borrow_decision_helpers(locals())
            _borrow_frame_helpers(locals())
            _production_polarization = staticmethod(lambda: {})
            # pure-interference mode off: _frame_boost / _unweighting_mode both
            # ask, and neither stub carries the card option
            _pure_interference = staticmethod(lambda: {})
            def __init__(self):
                self.options = _StubOptions(
                               {'spinmode': 'onshell',
                                'sequential_spin_order': '2 3 1',
                                'unweighting': unweighting,
                                'sequential_debug': False,
                                'fixed_order': False,
                                'beampol': [0., 0.], 'frame_id': 6})
            def _density_basis(self, production, decays_key):
                particles, slots = interface._sequential_slots(production, decays_key)
                return {'decays_key': decays_key, 'helicities': hels,
                        'init_part': [particles[i] for i in slots],
                        'decaying_spins': [2, 2], 'position': [1, 2],
                        'allowed_hel': [], 'ncomb': 0, 'dimension': 4}
            def create_and_initialise_f2py_modules(self, all_prefix, all_pdg, all_procid):
                pass
            def get_density(self, *args, **opts):
                return rho
            def _draw_one_decay(self, particle, index, ids, evt_decayfile, nb_remain):
                import random
                return ('cand', self._slot_of[index], random.randrange(pool))
            def _slot_density(self, decay, parent, hel, frame_boost=None):
                return pools[decay[1]][decay[2]]
        return Stub()

    def test_pool_average_is_the_identity(self):
        """The property the method needs from the decay sample."""
        import numpy as np
        for seed in (100, 200):
            pool = self._pool(seed)
            average = sum(np.array([[d.values[0], d.values[1]],
                                    [np.conj(d.values[1]), d.values[2]]])
                          for d in pool) / self.POOL
            self.assertTrue(np.allclose(average, np.eye(2) / 2))

    def _target(self, stub, rho, pools):
        """p(decays) proportional to N_n, normalised -- what every scheme has to
        sample."""
        exact = {(a, b): stub._partial_density_contraction(
                            rho, self.HELS, {0: pools[0][a], 1: pools[1][b]}).real
                 for a in range(self.POOL) for b in range(self.POOL)}
        total = sum(exact.values())
        return {k: v / total for k, v in exact.items()}

    def _fixture(self, unweighting='sequential_with_mass'):
        rho = self._production_density()
        pools = {0: self._pool(100), 1: self._pool(200)}
        stub = self._stub(rho, pools, unweighting=unweighting)
        production = self._Prod([self._Part(2, -1), self._Part(-2, -1),
                                 self._Part(6), self._Part(-6)])
        _, slots = interface_madspin.MadSpinInterface._sequential_slots(
                                                    production, (6, -6))
        stub._slot_of = {index: slot for slot, index in enumerate(slots)}
        return stub, rho, pools, production, {6: {0: 'f'}, -6: {0: 'f'}}

    def test_reproduces_the_joint_distribution(self):
        """The whole claim: p(decays) proportional to N_n, i.e. the same target
        the joint accept/reject samples -- while only ever redrawing one
        particle at a time."""
        import random
        stub, rho, pools, production, evt_decayfile = self._fixture()
        exact = self._target(stub, rho, pools)
        self.assertTrue(all(v > 0 for v in exact.values()))

        random.seed(0)
        counts = collections.Counter()
        nb_run = 20000
        for _ in range(nb_run):
            decays = stub.sequential_accept_reject(production, evt_decayfile,
                                                   [4.0, 4.0], 10)
            counts[(decays[6][0][2], decays[-6][0][2])] += 1

        for combo, want in exact.items():
            got = counts[combo] / float(nb_run)
            self.assertLess(abs(got / want - 1), 0.15,
                            'combo %s: got %.4f, expected %.4f' % (combo, got, want))

    def test_every_scheme_samples_the_same_target(self):
        """The up-front-mass schemes must land on that same distribution. There
        is no virtuality here (onshell), so their mass stage is the degenerate
        one -- what is exercised is the plumbing the PA up-front draw shares
        with them: the shifted bound vector, the Z_hat divisions cancelling
        against a table that does not exist, and the three rejection policies.
        """
        import random
        for mode in ('sequential', 'two_stage', 'sequential_global_retry'):
            stub, rho, pools, production, evt_decayfile = self._fixture(mode)
            self.assertTrue(stub._sequential_upfront(True), mode)
            exact = self._target(stub, rho, pools)
            # maxwgts[0] bounds the (constant) mass weight; then either one
            # bound per slot -- C_k over w_k = N_k/N_{k-1} -- or, for two_stage,
            # a single one over their product N_n/N_0
            contract = stub._partial_density_contraction
            n_0 = contract(rho, self.HELS, {}).real
            n_1 = [contract(rho, self.HELS, {0: pools[0][a]}).real
                   for a in range(self.POOL)]
            n_2 = [[contract(rho, self.HELS,
                             {0: pools[0][a], 1: pools[1][b]}).real
                    for b in range(self.POOL)] for a in range(self.POOL)]
            c_0 = 1.01 * max(n_1[a] / n_0 for a in range(self.POOL))
            c_1 = 1.01 * max(n_2[a][b] / n_1[a]
                             for a in range(self.POOL)
                             for b in range(self.POOL))
            if mode == 'two_stage':
                maxwgts = [1.1, 1.01 * max(n_2[a][b] / n_0
                                           for a in range(self.POOL)
                                           for b in range(self.POOL))]
            else:
                maxwgts = [1.1, c_0, c_1]

            random.seed(0)
            counts = collections.Counter()
            nb_run = 20000
            for _ in range(nb_run):
                decays = stub.sequential_accept_reject(production, evt_decayfile,
                                                       maxwgts, 10)
                counts[(decays[6][0][2], decays[-6][0][2])] += 1
            for combo, want in exact.items():
                got = counts[combo] / float(nb_run)
                self.assertLess(abs(got / want - 1), 0.15,
                                '%s combo %s: got %.4f, expected %.4f'
                                % (mode, combo, got, want))


class TestPAUpFrontMass(unittest.TestCase):
    """The PA up-front mass draw and the rate factor it makes necessary.

    Freezing the virtualities before the angles buys the production reshuffling
    jacobian -- one evaluation per mass set instead of one per slot trial -- but
    it also makes the angle stage self-normalising: redrawing a slot until it
    accepts divides out

        Z_k(m) = E_pool[ w_k ] = E_pool[ jac_dec(m, Omega) ]

    (the density ratio N_k/N_{k-1} averages to one at fixed m, so only the decay
    reshuffling jacobian is left). Unless the mass stage pays that factor back,
    the accepted virtualities come out distributed as the Breit-Wigner prior
    rather than as the physical lineshape -- exactly the bias the offshell path
    was fixed for.

    Driven synthetically: a fake ``jac_dec(m, decay) = f(m) g(decay)`` with
    ``E_pool[g] = 1``, so ``Z_k(m) = f(m)`` exactly and the accepted mass
    distribution has a closed form to compare against. ``g`` is constant over
    each antipodal pair of the pool, which is what keeps ``E[g D] = E[g] E[D]``
    exact and hence the factorisation above.
    """

    POOL = 4
    HELS = TestSequentialAcceptReject.HELS
    MASSES = (160.0, 173.0, 190.0)
    POLE = 173.0
    ALPHA = 3.0                     # f(m) = (m/pole)**ALPHA

    def _f(self, mass):
        return (mass / self.POLE) ** self.ALPHA

    def _g(self, index):
        # constant over each antipodal pair, mean one over the pool
        return 0.5 if index < 2 else 1.5

    class _Decay(object):
        """The minimum a decay event needs to be here: a slot/pool identity and
        a first particle that can carry new_mass."""
        class _Head(object):
            pass
        def __init__(self, slot, index):
            self.slot = slot
            self.index = index
            self.head = self._Head()
        def __getitem__(self, position):
            assert position == 0
            return self.head

    def _stub(self, rho, pools, z_table=True, keep_jac=True,
              unweighting='sequential'):
        interface = interface_madspin.MadSpinInterface
        outer = self
        hels = self.HELS

        class Stub(object):
            _decaying_pdgs = staticmethod(interface._decaying_pdgs)
            _sequential_slots = staticmethod(interface._sequential_slots)
            _slot_identity = interface._slot_identity
            _partial_density_contraction = interface._partial_density_contraction
            _sequential_spin_order = interface._sequential_spin_order
            _decay_slot_order = interface._decay_slot_order
            sequential_accept_reject = interface.sequential_accept_reject
            _polarization_weight_labels = \
                interface._polarization_weight_labels
            _polarization_weights_enabled = \
                interface._polarization_weights_enabled
            # the zero-density / dead-trial guards live on the same loop
            _check_production_density = interface._check_production_density
            _raise_degenerate_weight = interface._raise_degenerate_weight
            _dead_trial = interface._dead_trial
            _upfront_production = interface._upfront_production
            _sequential_offshell = interface._sequential_offshell
            _sequential_upfront = interface._sequential_upfront
            _z_slot_keys = staticmethod(interface._z_slot_keys)
            _zhat = interface._zhat
            _draw_offshell_mass = interface._draw_offshell_mass
            _unweighting_mode = interface._unweighting_mode
            _announce_mode = interface._announce_mode
            _log_once = interface._log_once
            _borrow_decision_helpers(locals())
            _borrow_frame_helpers(locals())
            _production_polarization = staticmethod(lambda: {})
            # pure-interference mode off: _frame_boost / _unweighting_mode both
            # ask, and neither stub carries the card option
            _pure_interference = staticmethod(lambda: {})

            def __init__(self):
                # unpolarised beams and a brace-free production, so _frame_boost
                # short circuits to None: this class is about the mass stage,
                # and the frame machinery has its own tests
                self.options = _StubOptions(
                               {'spinmode': 'PA',
                                'sequential_spin_order': '2 3 1',
                                'unweighting': unweighting,
                                'density_keep_jacobian': keep_jac,
                                'sequential_debug': False,
                                'fixed_order': False,
                                'beampol': [0., 0.], 'frame_id': 6})
                # Z_k(m) = f(m) = exp(ALPHA * ln(m/pole)) is exactly the
                # log-quadratic the tabulation fits, so the "perfect table"
                # can be written down instead of measured
                self._z_tables = {}
                if z_table:
                    for key in ('6_0', '-6_0'):
                        self._z_tables[key] = {
                            'pole': outer.POLE,
                            'coeff': [0.0, outer.ALPHA, 0.0],
                            'zero_below': 0.0,
                            'range': (min(outer.MASSES), max(outer.MASSES))}

            def _density_basis(self, production, decays_key):
                particles, slots = interface._sequential_slots(production,
                                                               decays_key)
                return {'decays_key': decays_key, 'helicities': hels,
                        'init_part': [particles[i] for i in slots],
                        'decaying_spins': [2, 2], 'position': [1, 2],
                        'allowed_hel': [], 'ncomb': 0, 'dimension': 4}

            def create_and_initialise_f2py_modules(self, *args):
                pass

            def get_density(self, *args, **opts):
                return rho

            def _draw_mass_value(self, pdg, budget):
                """Discrete and flat, so the prior is uniform and any structure
                in the accepted virtualities comes from the weights."""
                import random
                mass = random.choice(outer.MASSES)
                return mass, (outer.POLE, 1.5, min(outer.MASSES),
                              max(outer.MASSES)), 1.0

            def _production_jacobian_for(self, production, slot_to_index,
                                         slot_masses):
                return 1.0

            def _decay_reshuffle_jacobian(self, decay):
                return outer._f(decay[0].new_mass) * outer._g(decay.index)

            def _draw_one_decay(self, particle, index, ids, evt_decayfile,
                                nb_remain):
                import random
                return TestPAUpFrontMass._Decay(self._slot_of[index],
                                                random.randrange(outer.POOL))

            def _slot_density(self, decay, parent, hel, frame_boost=None):
                return pools[decay.slot][decay.index]

        return Stub()

    def _fixture(self, **opts):
        base = TestSequentialAcceptReject()
        rho = base._production_density()
        pools = {0: base._pool(100), 1: base._pool(200)}
        stub = self._stub(rho, pools, **opts)
        production = base._Prod([base._Part(2, -1), base._Part(-2, -1),
                                 base._Part(6), base._Part(-6)])
        _, slots = interface_madspin.MadSpinInterface._sequential_slots(
                                                    production, (6, -6))
        stub._slot_of = {index: slot for slot, index in enumerate(slots)}
        return stub, rho, pools, production, {6: {0: 'f'}, -6: {0: 'f'}}

    def _bounds(self, stub, rho, pools):
        contract = stub._partial_density_contraction
        n_0 = contract(rho, self.HELS, {}).real
        n_1 = [contract(rho, self.HELS, {0: pools[0][a]}).real
               for a in range(self.POOL)]
        n_2 = [[contract(rho, self.HELS,
                         {0: pools[0][a], 1: pools[1][b]}).real
                for b in range(self.POOL)] for a in range(self.POOL)]
        top_jac = max(self._f(m) for m in self.MASSES) * \
                  max(self._g(d) for d in range(self.POOL))
        c_0 = 1.01 * top_jac * max(n_1[a] / n_0 for a in range(self.POOL))
        c_1 = 1.01 * top_jac * max(n_2[a][b] / n_1[a]
                                   for a in range(self.POOL)
                                   for b in range(self.POOL))
        c_mass = 1.01 * max(self._f(m) for m in self.MASSES) ** 2
        return [c_mass, c_0, c_1], n_0, n_1, n_2

    def _run(self, stub, production, evt_decayfile, maxwgts, nb_run=30000,
             seed=0):
        import random
        random.seed(seed)
        masses = collections.Counter()
        combos = collections.Counter()
        for _ in range(nb_run):
            decays = stub.sequential_accept_reject(production, evt_decayfile,
                                                   maxwgts, 10)
            masses[decays[6][0][0].new_mass] += 1
            combos[(decays[6][0].index, decays[-6][0].index)] += 1
        return masses, combos

    def test_the_rate_factor_is_the_decay_reshuffling_jacobian(self):
        """What the probe records for Z_k under PA: the jacobian of mapping the
        drawn decay onto the sampled virtuality, and nothing else. Offshell that
        slot also carries Tr(D^off)/|M|^2_on -- PA evaluates on shell, so there
        is no such ratio to carry."""
        import random
        stub, _, _, production, evt_decayfile = self._fixture()
        random.seed(3)
        probe, extra = [], {}
        stub.sequential_accept_reject(production, evt_decayfile, None, 10,
                                      probe=probe, probe_extra=extra)
        self.assertEqual(extra['keys'], ['6_0', '-6_0'])
        self.assertEqual(len(extra['z']), 2)
        for (key, mass, value), slot in zip(extra['z'], (0, 1)):
            self.assertIn(mass, self.MASSES)
            # f(m) g(d) for one of the four pool members
            self.assertTrue(any(abs(value - self._f(mass) * self._g(d)) < 1e-9
                                for d in range(self.POOL)),
                            'slot %s recorded %s at m=%s' % (slot, value, mass))

    def test_no_rate_factor_recorded_when_the_jacobian_is_not_in_the_weight(self):
        """density_keep_jacobian off: joint PA applies the reshuffle after
        acceptance, so it is in no weight, and the only mass dependence left in
        w_k is whether the decay can reach the virtuality at all."""
        import random
        stub, _, _, production, evt_decayfile = self._fixture(keep_jac=False)
        random.seed(3)
        probe, extra = [], {}
        stub.sequential_accept_reject(production, evt_decayfile, None, 10,
                                      probe=probe, probe_extra=extra)
        self.assertEqual([value for _, _, value in extra['z']], [1.0, 1.0])

    def test_the_accepted_virtualities_follow_the_rate_factor(self):
        """The closure: with Z_k in the mass weight the accepted virtualities
        are distributed as prior(m) * f(m), which is what the joint PA
        accept/reject produces."""
        stub, rho, pools, production, evt_decayfile = self._fixture()
        maxwgts, _, _, _ = self._bounds(stub, rho, pools)
        masses, _ = self._run(stub, production, evt_decayfile, maxwgts)

        total = sum(self._f(m) for m in self.MASSES)
        nb_run = sum(masses.values())
        for mass in self.MASSES:
            want = self._f(mass) / total
            got = masses[mass] / float(nb_run)
            self.assertLess(abs(got / want - 1), 0.05,
                            'm=%s: got %.4f, expected %.4f' % (mass, got, want))

    def test_without_the_rate_factor_the_virtualities_are_the_prior(self):
        """The bug the factor exists for: the angle stage divides Z_k out
        whatever the mass stage paid, so leaving it out leaves the accepted
        virtualities distributed as the Breit-Wigner prior -- here flat --
        instead of as the physical lineshape."""
        stub, rho, pools, production, evt_decayfile = self._fixture(
                                                            z_table=False)
        maxwgts, _, _, _ = self._bounds(stub, rho, pools)
        masses, _ = self._run(stub, production, evt_decayfile, maxwgts)

        nb_run = sum(masses.values())
        for mass in self.MASSES:
            got = masses[mass] / float(nb_run)
            self.assertLess(abs(got * len(self.MASSES) - 1), 0.05,
                            'm=%s: got %.4f, expected the flat prior %.4f'
                            % (mass, got, 1.0 / len(self.MASSES)))
        # and that is a real distortion, not a wash: the correct answer is far
        # outside the tolerance just used
        total = sum(self._f(m) for m in self.MASSES)
        self.assertGreater(abs(masses[self.MASSES[-1]] / float(nb_run)
                               / (self._f(self.MASSES[-1]) / total) - 1), 0.15)

    def test_the_decays_are_unaffected_by_the_table(self):
        """The angle stage normalises itself, so its accepted decays are the
        same whatever the mass stage pays -- which is exactly why an inaccurate
        Z_hat shows up in the virtualities and nowhere else. (Their target is
        N_n weighted by the decay's own share g of the reshuffling jacobian; the
        virtuality-dependent share f cancels here.)"""
        for z_table in (True, False):
            stub, rho, pools, production, evt_decayfile = self._fixture(
                                                            z_table=z_table)
            maxwgts, _, _, n_2 = self._bounds(stub, rho, pools)
            _, combos = self._run(stub, production, evt_decayfile, maxwgts)
            nb_run = sum(combos.values())
            weighted = [[n_2[a][b] * self._g(a) * self._g(b)
                         for b in range(self.POOL)] for a in range(self.POOL)]
            total = sum(sum(row) for row in weighted)
            for a in range(self.POOL):
                for b in range(self.POOL):
                    want = weighted[a][b] / total
                    got = combos[(a, b)] / float(nb_run)
                    self.assertLess(abs(got / want - 1), 0.15,
                                    'table=%s combo %s: got %.4f, expected %.4f'
                                    % (z_table, (a, b), got, want))

    def test_the_production_jacobian_is_evaluated_once_per_mass_set(self):
        """What the up-front draw buys under PA. The per-slot mass draw calls
        _production_jacobian_for -- an event copy and a reshuffle -- on every
        slot trial and telescopes the results; here it is one call per mass
        set."""
        import random
        for unweighting, expected in (('sequential', 'per mass set'),
                                      ('sequential_with_mass', 'per trial')):
            stub, rho, pools, production, evt_decayfile = self._fixture(
                                                    unweighting=unweighting)
            maxwgts, _, _, _ = self._bounds(stub, rho, pools)
            counts = collections.Counter()

            def _count(name, wrapped):
                def counted(*args, **opts):
                    counts[name] += 1
                    return wrapped(*args, **opts)
                return counted

            stub._production_jacobian_for = _count(
                            'jacobian', stub._production_jacobian_for)
            stub._draw_one_decay = _count('trial', stub._draw_one_decay)
            random.seed(7)
            if unweighting == 'sequential_with_mass':
                maxwgts = maxwgts[1:]
            for _ in range(200):
                stub.sequential_accept_reject(production, evt_decayfile,
                                              maxwgts, 10)
            if expected == 'per mass set':
                # one per mass set, and there are fewer mass sets than trials
                self.assertLess(counts['jacobian'], counts['trial'])
                self.assertGreaterEqual(counts['jacobian'], 200)
            else:
                self.assertEqual(counts['jacobian'], counts['trial'])


class TestSequentialPoolLadder(unittest.TestCase):
    """_sequential_pool_ladder / _sequential_active: how many decay events a
    production event burns per pdg once the accept/reject is per particle, and
    when that regime applies at all.
    """

    class _Part(object):
        def __init__(self, spin):
            self._spin = spin
        def get(self, key):
            return self._spin

    class _Model(object):
        def __init__(self, spins):
            self.spins = spins
        def get_particle(self, pdg):
            return TestSequentialPoolLadder._Part(self.spins[pdg])

    def _stub(self, spins, polarization=None, **options):
        interface = interface_madspin.MadSpinInterface
        class Stub(object):
            _sequential_pool_ladder = interface._sequential_pool_ladder
            _sequential_active = interface._sequential_active
            _sequential_upfront = interface._sequential_upfront
            _unweighting_mode = interface._unweighting_mode
            _pure_interference = staticmethod(lambda: {})
            _announce_mode = interface._announce_mode
            _log_once = interface._log_once
            _sequential_spin_order = interface._sequential_spin_order
            _decay_pool_ladder = staticmethod(interface._decay_pool_ladder)
            _borrow_decision_helpers(locals())
        stub = Stub()
        stub.model = self._Model(spins)
        stub.options = {'unweighting': 'sequential', 'fixed_order': False,
                        'spinmode': 'PA', 'sequential_spin_order': '2 3 1'}
        stub.options.update(options)
        # Seed _production_polarization's own cache rather than a banner: '{}'
        # is what it returns for a process line without braces, and a dict of
        # braces is what it returns for one with them. The parsing that fills
        # it is covered by its own tests.
        stub._production_polarization_cache = polarization or {}
        return stub

    NB = 1000

    def test_ladder_follows_the_ordering_and_spares_scalars(self):
        """t, t~ take the first two rungs; the higgs is last and stays at 1.1
        because it can never be rejected."""
        stub = self._stub({6: 2, -6: 2, 25: 1})
        got = stub._sequential_pool_ladder({6: self.NB, -6: self.NB, 25: self.NB},
                                           self.NB, True)
        self.assertEqual(got, {-6: 1.5, 6: 2.0, 25: 1.1})

    def test_identical_parents_share_a_pool_at_their_largest_rung(self):
        """Two tops occupy slots 0 and 1 but read the same pool, so it must be
        sized for the hungrier of the two; the vector follows at slot 2."""
        stub = self._stub({6: 2, 24: 3})
        got = stub._sequential_pool_ladder({6: 2 * self.NB, 24: self.NB},
                                           self.NB, True)
        self.assertEqual(got, {6: 2.0, 24: 2.5})

    def test_no_ladder_when_the_joint_test_is_used(self):
        """Empty dict -> the caller keeps the historical 1.1 / 2.0."""
        interface = interface_madspin.MadSpinInterface
        spins = {6: 2, -6: 2}
        pools = {6: self.NB, -6: self.NB}
        # opted out
        self.assertEqual(self._stub(spins, unweighting='joint')
                             ._sequential_pool_ladder(pools, self.NB, True), {})
        # not density mode
        self.assertEqual(self._stub(spins)
                             ._sequential_pool_ladder(pools, self.NB, False), {})
        # fixed_order keeps the joint test
        self.assertEqual(self._stub(spins, fixed_order=True)
                             ._sequential_pool_ladder(pools, self.NB, True), {})
        # a spinmode outside the supported set keeps the joint test
        self.assertEqual(self._stub(spins, spinmode='none')
                             ._sequential_pool_ladder(pools, self.NB, True), {})

    def test_sequential_active_gate(self):
        stub = self._stub({6: 2})
        self.assertTrue(stub._sequential_active(True))
        self.assertFalse(stub._sequential_active(False))  # not density mode
        # onshell is supported just like PA
        self.assertTrue(self._stub({6: 2}, spinmode='onshell')._sequential_active(True))

    def test_sequential_active_auto(self):
        """'auto' resolves per spinmode: per-particle under PA/onshell, and
        offshell only once there are enough decays to pay for the mass stage
        (the default _nb_decaying here is 2, so offshell still takes joint)."""
        for mode, expected in [('PA', True), ('onshell', True),
                               ('madspin', False), ('full', False),
                               ('none', False)]:
            stub = self._stub({6: 2}, unweighting='auto', spinmode=mode)
            self.assertEqual(stub._sequential_active(True), expected,
                             'auto + spinmode=%s' % mode)
        for mode in ('madspin', 'full'):
            stub = self._stub({6: 2}, unweighting='auto', spinmode=mode)
            stub._nb_decaying = 3
            self.assertTrue(stub._sequential_active(True), mode)
        # fixed_order still forces the joint test
        stub = self._stub({6: 2}, unweighting='auto', fixed_order=True)
        self.assertFalse(stub._sequential_active(True))
        self.assertEqual(stub._unweighting_mode(True), 'joint')

    def test_auto_picks_the_scheme_by_the_number_of_decays(self):
        """Offshell, a mass set costs a production reshuffle and a production
        density, so the staged schemes only pay off once there are enough decays
        to save: auto takes joint up to two decaying particles and sequential
        from three. See doc/madspin_sequential_plan.md section 12."""
        for nb, expected in [(1, 'joint'), (2, 'joint'),
                             (3, 'sequential'), (6, 'sequential')]:
            for spinmode in ('madspin', 'full'):
                stub = self._stub({6: 2}, unweighting='auto', spinmode=spinmode)
                stub._nb_decaying = nb
                self.assertEqual(stub._unweighting_mode(True), expected,
                                 '%s, %d decaying particles' % (spinmode, nb))

    # a brace on the production, as _production_polarization returns it:
    # 'p p > t{+} t~{+}' -> the (1,) helicity kept for both tops.
    POL = {6: ((1,),), -6: ((1,),)}

    def test_auto_is_per_particle_when_the_production_is_polarised(self):
        """A production brace restricts the convolution to a polarisation
        subspace, and the joint weight then sits far below the single bound the
        max-weight scan hands it. Measured offshell on `p p > t t~` with both
        tops decayed (500 events, so n=2 and the multiplicity rule alone would
        say joint): `t{+}t~{+}` 112 trials per accepted event under joint
        against 9.1 under sequential, `t{+}t~{-}` 162 against 8.4 -- while
        unpolarised joint is the better of the two at 3.3 against 6.1. At 50000
        events the joint column of the same three rises to 204-213, 5800-6300
        and 4.05. So the brace overrides the multiplicity rule, at every n."""
        for nb in (1, 2, 3, 6):
            for spinmode in ('madspin', 'full'):
                stub = self._stub({6: 2}, unweighting='auto', spinmode=spinmode,
                                  polarization=self.POL)
                stub._nb_decaying = nb
                self.assertEqual(stub._unweighting_mode(True), 'sequential',
                                 'polarised %s, %d decaying particles'
                                 % (spinmode, nb))

    def test_polarised_auto_leaves_the_unpolarised_resolution_alone(self):
        """The clause keys on the brace and nothing else: the same stub without
        one keeps the two-branch rule exactly."""
        for nb, expected in [(1, 'joint'), (2, 'joint'),
                             (3, 'sequential'), (6, 'sequential')]:
            stub = self._stub({6: 2}, unweighting='auto', spinmode='madspin')
            stub._nb_decaying = nb
            self.assertEqual(stub._unweighting_mode(True), expected,
                             'unpolarised, %d decaying particles' % nb)

    def test_polarised_production_does_not_override_an_explicit_joint(self):
        """Only 'auto' looks at the brace. A user who asks for joint gets joint,
        slow or not -- it is the cross-check the staged schemes are validated
        against, and losing it on polarised processes would leave them
        unvalidated exactly where they matter most."""
        for spinmode in ('madspin', 'full', 'PA', 'onshell'):
            stub = self._stub({6: 2}, unweighting='joint', spinmode=spinmode,
                              polarization=self.POL)
            self.assertEqual(stub._unweighting_mode(True), 'joint', spinmode)
            self.assertFalse(stub._sequential_active(True), spinmode)
        # and the other explicit schemes are untouched too
        for mode in ('two_stage', 'sequential', 'sequential_global_retry'):
            stub = self._stub({6: 2}, unweighting=mode, spinmode='madspin',
                              polarization=self.POL)
            self.assertEqual(stub._unweighting_mode(True), mode)

    def test_polarised_production_leaves_pa_and_onshell_where_they_were(self):
        """PA/onshell already resolve to sequential under auto at every
        multiplicity, so the brace has nothing to change there -- pinned so a
        later reshuffle of the branches cannot make the polarised case take a
        different path from the unpolarised one."""
        for spinmode in ('PA', 'onshell'):
            for nb in (1, 2, 3, 6):
                for pol in (None, self.POL):
                    stub = self._stub({6: 2}, unweighting='auto',
                                      spinmode=spinmode, polarization=pol)
                    stub._nb_decaying = nb
                    self.assertEqual(stub._unweighting_mode(True), 'sequential',
                                     '%s, %d decaying, pol=%s'
                                     % (spinmode, nb, bool(pol)))

    def test_polarised_auto_still_yields_to_the_joint_only_gates(self):
        """fixed_order and '@' grouping force joint whatever auto resolved to;
        the polarisation clause must not smuggle a staged scheme past them."""
        stub = self._stub({6: 2}, unweighting='auto', spinmode='madspin',
                          fixed_order=True, polarization=self.POL)
        stub._nb_decaying = 2
        self.assertEqual(stub._unweighting_mode(True), 'joint')
        stub = self._stub({6: 2}, unweighting='auto', spinmode='madspin',
                          polarization=self.POL)
        stub._nb_decaying = 2
        stub._decay_groups = {'1': {}, '2': {}}
        self.assertEqual(stub._unweighting_mode(True), 'joint')

    def test_auto_is_per_particle_under_pa_at_every_multiplicity(self):
        """PA/onshell keep rho fixed on shell, so their mass stage costs a
        reshuffling jacobian and nothing else -- sequential was the fastest of
        the three at every multiplicity measured, including one decaying
        particle, where the mass set can still be rejected before any decay is
        drawn."""
        for spinmode in ('PA', 'onshell'):
            for nb in (1, 2, 3, 6):
                stub = self._stub({6: 2}, unweighting='auto', spinmode=spinmode)
                stub._nb_decaying = nb
                self.assertEqual(stub._unweighting_mode(True), 'sequential',
                                 '%s, %d decaying particles' % (spinmode, nb))

    def test_auto_is_joint_for_a_single_decaying_particle_offshell(self):
        """One decaying particle offshell is the worst case for a mass stage:
        the mass-set weight carries Tr(rho_off)/|M_prod|^2_on, and when that one
        particle carries most of the production matrix element's virtuality
        dependence the ratio spans orders of magnitude (measured: ~790 mass sets
        per accepted event on p p > w+ j). auto must not go there."""
        for spinmode in ('madspin', 'full'):
            stub = self._stub({6: 1}, unweighting='auto', spinmode=spinmode)
            stub._nb_decaying = 1
            self.assertEqual(stub._unweighting_mode(True), 'joint', spinmode)
            self.assertFalse(stub._sequential_active(True))
        # asked for explicitly it is still honoured, for cross-checks
        stub = self._stub({6: 1}, unweighting='sequential', spinmode='madspin')
        stub._nb_decaying = 1
        self.assertEqual(stub._unweighting_mode(True), 'sequential')

    def test_grouped_decays_force_the_joint_test(self):
        """'@' grouping forces the joint accept/reject, whatever unweighting
        asks for. The per-particle and two_stage schemes redraw to acceptance,
        which divides E[w | group] out of the chain -- and that expectation
        differs between groups, so the accepted group fractions would come out
        distorted by its reciprocal. Pinned here because the gate was ported by
        hand onto the unweighting API it now lives in.
        """
        for mode in ('auto', 'sequential', 'two_stage',
                     'sequential_global_retry'):
            stub = self._stub({6: 2}, unweighting=mode, spinmode='madspin')
            # three decaying particles: offshell `auto` only leaves joint from
            # three up, so this is where grouping has something to override
            stub._nb_decaying = 3
            stub._decay_groups = None
            self.assertNotEqual(stub._unweighting_mode(True), 'joint', mode)
            stub._decay_groups = {'1': {}, '2': {}}
            self.assertEqual(stub._unweighting_mode(True), 'joint',
                             '%s with grouped decays' % mode)

    def test_up_front_mass_modes_are_available_under_pa(self):
        """PA has an up-front mass draw of its own now, so the three schemes
        that split the accept/reject there are honoured rather than downgraded
        to the per-slot mass draw."""
        for mode in ('two_stage', 'sequential', 'sequential_global_retry'):
            for spinmode in ('PA', 'onshell', 'madspin'):
                stub = self._stub({6: 2}, unweighting=mode, spinmode=spinmode)
                self.assertEqual(stub._unweighting_mode(True), mode,
                                 '%s under %s' % (mode, spinmode))
                self.assertTrue(stub._sequential_upfront(True))

    def test_with_mass_needs_a_per_particle_mass_draw(self):
        """sequential_with_mass draws each slot's virtuality inside that slot's
        accept/reject, which the offshell spinmodes cannot do -- they reshuffle
        the whole production onto the mass set at once."""
        stub = self._stub({6: 2}, unweighting='sequential_with_mass',
                          spinmode='PA')
        self.assertEqual(stub._unweighting_mode(True), 'sequential_with_mass')
        self.assertFalse(stub._sequential_upfront(True))
        for spinmode in ('madspin', 'full'):
            stub = self._stub({6: 2}, unweighting='sequential_with_mass',
                              spinmode=spinmode)
            self.assertEqual(stub._unweighting_mode(True), 'sequential')
            self.assertTrue(stub._sequential_upfront(True))

    def test_joint_is_not_an_up_front_scheme(self):
        """_sequential_upfront gates the mass stage, so it must be False
        wherever there is no sequential accept/reject at all."""
        stub = self._stub({6: 2}, unweighting='joint', spinmode='PA')
        self.assertFalse(stub._sequential_upfront(True))
        stub = self._stub({6: 2}, unweighting='sequential', spinmode='PA')
        self.assertFalse(stub._sequential_upfront(False))   # not density mode

    def test_madspin_option_defaults(self):
        """The shipped defaults: spinmode=madspin, jacobian in the weight,
        unweighting on auto, and the deprecated alias still understood."""
        options = interface_madspin.MadSpinOptions()
        self.assertEqual(options['spinmode'], 'madspin')
        self.assertEqual(options['density_keep_jacobian'], True)
        self.assertEqual(options['unweighting'], 'auto')
        for value in ('joint', 'two_stage', 'sequential',
                      'sequential_global_retry', 'sequential_with_mass'):
            options['unweighting'] = value
            self.assertEqual(options['unweighting'], value)

    def test_the_replaced_spellings_no_longer_exist(self):
        """``sequential_decay`` and the singular
        ``keep_weight_for_polarization`` were deprecated aliases that only ever
        translated at load time; they are gone rather than silently accepted."""
        options = interface_madspin.MadSpinOptions()
        self.assertNotIn('sequential_decay', options)
        self.assertNotIn('keep_weight_for_polarization', options)


class TestScanMaxwgtDecomposition(unittest.TestCase):
    """The parallel max-weight scan splits the probe events across workers and
    concatenates their per-event vectors. That is only valid if scanning a range
    of events, then another, gives exactly the per-event vectors of scanning the
    whole -- which is what _scan_maxwgt_parallel relies on. Tested here without
    fork, on the synthetic densities of TestSequentialAcceptReject.
    """

    def _fixture(self, unweighting='sequential_with_mass'):
        base = TestSequentialAcceptReject()
        stub, _, _, production, evt_decayfile = base._fixture(unweighting)
        return stub, [production] * 6, evt_decayfile

    def test_range_split_matches_the_whole(self):
        """scan[0:6] == scan[0:2] + scan[2:6], event for event, at fixed seed."""
        import random
        stub, events, evt_decayfile = self._fixture()

        random.seed(5)
        whole, _ = stub._scan_maxwgt_range(events, 0, 6, evt_decayfile, 6, 30)

        random.seed(5)
        first, _ = stub._scan_maxwgt_range(events, 0, 2, evt_decayfile, 6, 30)
        second, _ = stub._scan_maxwgt_range(events, 2, 6, evt_decayfile, 6, 30)

        self.assertEqual(len(whole), 6)
        self.assertEqual(first + second, whole)

    def test_one_vector_per_event_one_entry_per_slot(self):
        stub, events, evt_decayfile = self._fixture()
        import random
        random.seed(1)
        #Ignore the message "Error while creating the f2py modules for the production/decay part"
        per_event, _ = stub._scan_maxwgt_range(events, 0, 6, evt_decayfile, 6, 20)
        self.assertEqual(len(per_event), 6)
        for vec in per_event:
            self.assertEqual(len(vec), 2)          # two decaying particles
            self.assertTrue(all(w >= 0 for w in vec))

    def test_the_up_front_probe_keeps_its_chains(self):
        """The up-front-mass schemes cannot max online -- their mass-set weight
        is only complete once Z_k is known, and Z_k is fitted from this same
        probe -- so the scan hands every chain back and the bound is taken
        later, over the completed weights. One entry per chain, and one more
        entry per vector than there are slots (the mass set takes index 0)."""
        import random
        stub, events, evt_decayfile = self._fixture('sequential')
        random.seed(1)
        per_event, z_samples = stub._scan_maxwgt_range(events, 0, 6,
                                                       evt_decayfile, 6, 20)
        self.assertEqual(len(per_event), 6)
        for event in per_event:
            self.assertEqual(event['keys'], ['6_0', '-6_0'])
            self.assertEqual(len(event['chains']), 20)
            for weights, masses in event['chains']:
                self.assertEqual(len(weights), 3)   # mass set + two slots
                self.assertEqual(len(masses), 2)
        # onshell samples no virtuality, so there is nothing to tabulate and the
        # completed vector is the raw one
        self.assertEqual(z_samples, {})
        best = stub._complete_upfront_probe(per_event[0])
        self.assertEqual(best, [max(chain[0][slot]
                                    for chain in per_event[0]['chains'])
                                for slot in range(3)])


class TestOffshellRateFactor(unittest.TestCase):
    """Z_k(m): the normalisation the per-angle stage of the offshell sequential
    accept/reject divides out, and which therefore has to be put back into the
    mass-set weight.

    Z_k is the offshell decay rate at the sampled virtuality over the onshell
    one -- for a two-body decay of a spin-1/2 parent it is (m/M) Gamma(m)/Gamma(M),
    a smooth function of that slot's virtuality alone. These tests check that the
    tabulation recovers a known one from the samples the max-weight probe
    collects, that it interpolates and clips as advertised, and that it lands in
    the two weights the way sequential_exact needs.
    """

    class _Val(object):
        def __init__(self, value):
            self.value = value

    class _Banner(object):
        def get(self, card, kind, pdg):
            return TestOffshellRateFactor._Val(173.0)

    class _Stub(object):
        _unweighting_mode = interface_madspin.MadSpinInterface._unweighting_mode
        _pure_interference = staticmethod(lambda: {})
        _announce_mode = interface_madspin.MadSpinInterface._announce_mode
        _log_once = interface_madspin.MadSpinInterface._log_once
        _borrow_decision_helpers(locals())
        _build_z_tables = interface_madspin.MadSpinInterface._build_z_tables
        _weighted_polyfit2 = staticmethod(
                        interface_madspin.MadSpinInterface._weighted_polyfit2)
        _z_slot_keys = staticmethod(
                        interface_madspin.MadSpinInterface._z_slot_keys)
        _zhat = interface_madspin.MadSpinInterface._zhat
        _complete_upfront_probe = \
                        interface_madspin.MadSpinInterface._complete_upfront_probe
        def __init__(self, exact=False, joint_angles=False):
            self.banner = TestOffshellRateFactor._Banner()
            mode = 'sequential'
            if joint_angles:
                mode = 'two_stage'
            elif exact:
                mode = 'sequential_global_retry'
            self.options = {'unweighting': mode, 'fixed_order': False,
                            'spinmode': 'madspin'}
            self._z_tables = {}

    @staticmethod
    def _truth(mass):
        """A stand-in running width: (m/M) Gamma(m)/Gamma(M) for t > W b."""
        pole, mw = 173.0, 80.419
        def rate(m):
            x = (mw / m) ** 2
            return m ** 4 * (1 - x) ** 2 * (1 + 2 * x)
        return rate(mass) / rate(pole)

    def _samples(self, nb=20000, seed=3, spread=1.0, threshold=0.0):
        """(virtuality, rate factor) pairs as the probe records them: one draw
        each, the rate factor fluctuating around Z(m) with a large spread -- the
        table is an average, not a fit through clean points."""
        import random
        rng = random.Random(seed)
        out = []
        for _ in range(nb):
            mass = rng.uniform(150.0, 196.0)
            if mass < threshold:
                out.append((mass, 0.0))
                continue
            # lognormal noise, mean one: E[value | m] = Z(m)
            noise = math.exp(rng.gauss(0, spread) - 0.5 * spread ** 2)
            out.append((mass, self._truth(mass) * noise))
        return out

    def test_polyfit_recovers_a_quadratic(self):
        xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        ys = [0.5 - 2 * x + 3 * x ** 2 for x in xs]
        coeff = self._Stub()._weighted_polyfit2(xs, ys, [1.0] * len(xs))
        for got, want in zip(coeff, [0.5, -2.0, 3.0]):
            self.assertAlmostEqual(got, want, places=6)

    def test_polyfit_is_degenerate_below_three_points(self):
        stub = self._Stub()
        self.assertIsNone(stub._weighted_polyfit2([1.0, 2.0], [1.0, 2.0], [1, 1]))
        self.assertIsNone(stub._weighted_polyfit2([1.0] * 4, [1.0] * 4, [1] * 4))

    def test_tabulation_recovers_the_running_width(self):
        """The whole point: an average over noisy per-draw samples reproduces
        the underlying Z(m) across the Breit-Wigner range.

        The tolerance that matters is on the *slope* of ln Z -- a fractional
        error there survives as the same fractional error on the lineshape shift
        the factor corrects -- so it is checked directly, and much more tightly
        than the 10% the physics needs.
        """
        stub = self._Stub()
        stub._z_tables = stub._build_z_tables({'6_0': self._samples()})
        self.assertIn('6_0', stub._z_tables)
        for mass in (152.0, 160.0, 173.0, 185.0, 195.0):
            self.assertLess(abs(stub._zhat('6_0', mass) / self._truth(mass) - 1),
                            0.05, 'Z(%s) off' % mass)
        slope = ((math.log(stub._zhat('6_0', 190.0) / stub._zhat('6_0', 156.0)))
                 / (math.log(self._truth(190.0) / self._truth(156.0))))
        self.assertLess(abs(slope - 1), 0.05)

    def test_normalised_at_the_pole(self):
        stub = self._Stub()
        stub._z_tables = stub._build_z_tables({'6_0': self._samples()})
        self.assertAlmostEqual(stub._zhat('6_0', 173.0), 1.0, places=6)

    def test_held_constant_outside_the_probed_range(self):
        """Beyond the samples the fit is unconstrained, so it is frozen at the
        edge rather than extrapolated."""
        stub = self._Stub()
        stub._z_tables = stub._build_z_tables({'6_0': self._samples()})
        self.assertEqual(stub._zhat('6_0', 400.0), stub._zhat('6_0', 196.0))
        self.assertEqual(stub._zhat('6_0', 100.0), stub._zhat('6_0', 150.0))

    def test_threshold_is_recorded_as_a_hard_zero(self):
        """Virtualities no decay of the pool can be reshuffled onto give a rate
        factor of exactly zero, and the mass-set stage must reject them: Z there
        is 0, not the low edge of the fit."""
        stub = self._Stub()
        stub._z_tables = stub._build_z_tables(
                            {'6_0': self._samples(threshold=165.0)})
        self.assertEqual(stub._zhat('6_0', 160.0), 0.0)
        self.assertGreater(stub._zhat('6_0', 180.0), 0.0)

    def test_no_table_is_a_factor_one(self):
        """The max-weight probe itself runs before any table exists, and every
        non-offshell mode never builds one."""
        stub = self._Stub()
        self.assertEqual(stub._zhat('6_0', 160.0), 1.0)
        stub._z_tables = stub._build_z_tables({'6_0': self._samples(nb=10)})
        self.assertEqual(stub._z_tables, {})
        self.assertEqual(stub._zhat('6_0', 160.0), 1.0)

    def test_slot_keys_separate_identical_parents(self):
        Part = TestSequentialAcceptReject._Part
        particles = [Part(6), Part(-6), Part(6)]
        keys = self._Stub()._z_slot_keys(particles, [0, 2, 1])
        self.assertEqual(keys, ['6_0', '6_1', '-6_0'])

    def _probe_event(self):
        return {'keys': ['6_0', '-6_0'], 'order': [1, 0],
                'chains': [[[2.0, 3.0, 5.0], [160.0, 180.0]],
                           [[1.0, 7.0, 4.0], [173.0, 173.0]]]}

    def test_probe_completion_puts_z_in_the_mass_weight(self):
        """The probe records the mass-set weight before Z exists; completing it
        multiplies in one Z per slot, and leaves the per-angle weights alone."""
        stub = self._Stub()
        stub._z_tables = stub._build_z_tables({'6_0': self._samples(),
                                               '-6_0': self._samples(seed=9)})
        z0, z1 = stub._zhat('6_0', 160.0), stub._zhat('-6_0', 180.0)
        best = stub._complete_upfront_probe(self._probe_event())
        self.assertAlmostEqual(best[0], max(2.0 * z0 * z1, 1.0), places=10)
        self.assertEqual(best[1], 7.0)      # position 0 = slot 1
        self.assertEqual(best[2], 5.0)

    def test_probe_completion_divides_z_out_per_slot_when_exact(self):
        """Under sequential_exact the mass stage pays Z and the slot takes it
        back, so the bound each slot is tested against is the one of w_k/Z_k --
        mapped through the ordering, not the slot order."""
        stub = self._Stub(exact=True)
        stub._z_tables = stub._build_z_tables({'6_0': self._samples(),
                                               '-6_0': self._samples(seed=9)})
        z_by_slot = [stub._zhat('6_0', 160.0), stub._zhat('-6_0', 180.0)]
        best = stub._complete_upfront_probe(self._probe_event())
        # order [1, 0]: probe position 0 is slot 1, position 1 is slot 0
        self.assertAlmostEqual(best[1], max(3.0 / z_by_slot[1], 7.0), places=10)
        self.assertAlmostEqual(best[2], max(5.0 / z_by_slot[0], 4.0), places=10)

    def test_probe_completion_collapses_to_two_bounds_when_joint(self):
        """Variant B tests every angle against one bound, so the probe vector
        collapses to [C_mass, C_angles] and the second entry is the *product*
        over slots of w_k/Z_k -- maxed chain by chain, not per slot, since the
        chain is what is accepted or rejected."""
        stub = self._Stub(joint_angles=True)
        stub._z_tables = stub._build_z_tables({'6_0': self._samples(),
                                               '-6_0': self._samples(seed=9)})
        z = [stub._zhat('6_0', 160.0), stub._zhat('-6_0', 180.0)]
        best = stub._complete_upfront_probe(self._probe_event())
        self.assertEqual(len(best), 2)
        # chain 1: masses (160, 180), weights w_slot1 = 3.0, w_slot0 = 5.0
        # chain 2: masses (173, 173) where Z = 1, weights 7.0 and 4.0
        self.assertAlmostEqual(best[0], max(2.0 * z[0] * z[1], 1.0), places=10)
        self.assertAlmostEqual(best[1],
                               max((3.0 / z[1]) * (5.0 / z[0]), 7.0 * 4.0),
                               places=10)


class TestTwoStageMassDistribution(unittest.TestCase):
    """The bias the Z_k factor exists to remove, and the two cures, on a model
    of the offshell chain small enough to solve exactly.

    The chain is: draw a virtuality m from a prior, accept the mass set with
    probability w_mass(m) Z_hat(m) / C, then draw decay angles until one is
    accepted with probability w(m, angle) / C_ang. The target -- what the joint
    accept/reject samples -- is p(m) w_mass(m) E_angle[w(m, .)]. Because the
    per-angle stage redraws until it accepts, it divides its own normalisation
    Z(m) = E_angle[w(m, .)] out again, so with Z_hat = 1 the accepted
    virtualities come out proportional to p(m) w_mass(m) instead: the
    Breit-Wigner shape rather than the offshell one. This is the measured
    ttbar bias in miniature.
    """

    ANGLES = [0.3, 0.8, 1.6, 2.5]      # the decay "pool"
    MASSES = [160.0, 170.0, 180.0, 190.0]

    def _w(self, mass, angle):
        """Per-angle weight; its angle average rises steeply with the mass, as
        the offshell rate factor does."""
        return (mass / 170.0) ** 6 * angle

    def _w_mass(self, mass):
        return 1.0 + (mass - 160.0) / 100.0

    def _z(self, mass):
        return sum(self._w(mass, a) for a in self.ANGLES) / len(self.ANGLES)

    def _target(self):
        raw = {m: self._w_mass(m) * self._z(m) for m in self.MASSES}
        total = sum(raw.values())
        return {m: v / total for m, v in raw.items()}

    def _run(self, zhat, exact, nb=200000, seed=7):
        """The two-stage chain, mirroring sequential_accept_reject's offshell
        branch: mass-set accept/reject, then one slot redrawn to acceptance
        (or, under ``exact``, a rejected angle killing the mass set)."""
        import random
        rng = random.Random(seed)
        c_mass = max(self._w_mass(m) * zhat(m) for m in self.MASSES) * 1.01
        c_ang = max(self._w(m, a) / zhat(m)
                    for m in self.MASSES for a in self.ANGLES) * 1.01
        counts = collections.Counter()
        for _ in range(nb):
            while True:
                mass = rng.choice(self.MASSES)
                if rng.random() * c_mass >= self._w_mass(mass) * zhat(mass):
                    continue
                restart = False
                while True:
                    angle = rng.choice(self.ANGLES)
                    if rng.random() * c_ang < self._w(mass, angle) / zhat(mass):
                        break
                    if exact:
                        restart = True
                        break
                if not restart:
                    break
            counts[mass] += 1
        return {m: counts[m] / float(nb) for m in self.MASSES}

    def _assert_close(self, got, want, tolerance):
        for mass in self.MASSES:
            self.assertLess(abs(got[mass] / want[mass] - 1), tolerance,
                            'mass %s: got %.4f, want %.4f' % (mass, got[mass],
                                                              want[mass]))

    def test_without_the_factor_the_mass_distribution_is_biased(self):
        """The bug: the accepted virtualities follow p(m) w_mass(m), the shape
        the per-angle stage was supposed to reweight."""
        got = self._run(lambda m: 1.0, exact=False)
        prior = {m: self._w_mass(m) for m in self.MASSES}
        total = sum(prior.values())
        self._assert_close(got, {m: v / total for m, v in prior.items()}, 0.03)
        self.assertGreater(abs(got[190.0] / self._target()[190.0] - 1), 0.3)

    def test_the_exact_factor_restores_the_target(self):
        self._assert_close(self._run(self._z, exact=False), self._target(), 0.03)

    def test_a_wrong_factor_biases_by_exactly_its_error(self):
        """Why the tabulation has to be accurate: the per-angle stage divides
        out the true Z whatever weight it is given, so the residual bias is
        Z_hat/Z -- it does not cancel."""
        skew = lambda m: self._z(m) * (m / 170.0) ** 2
        got = self._run(skew, exact=False)
        want = {m: v * (m / 170.0) ** 2 for m, v in self._target().items()}
        total = sum(want.values())
        self._assert_close(got, {m: v / total for m, v in want.items()}, 0.03)

    def test_sequential_exact_is_right_for_any_factor(self):
        """And why the switch exists: rejecting the mass set instead of
        redrawing the angle stops the per-angle stage from normalising, so
        Z_hat cancels from the chain and only sets the efficiency."""
        for zhat in (lambda m: 1.0,
                     lambda m: self._z(m),
                     lambda m: self._z(m) * (m / 170.0) ** 2):
            self._assert_close(self._run(zhat, exact=True), self._target(), 0.03)


class TestUpfrontCache(unittest.TestCase):
    """The up-front-mass sequential bounds travel with the Z_k tables that
    complete them, so the cache holds two coupled objects and must not be read
    back under a schema it was not written with.  A mismatch is ignored rather
    than raised on: the scan is reproducible, so re-measuring is always
    available, whereas a table dereferenced under the wrong schema would either
    crash inside the accept/reject or silently weight the virtualities with the
    wrong fit.
    """

    class _Stub(object):
        _UPFRONT_CACHE_FORMAT = \
            interface_madspin.MadSpinInterface._UPFRONT_CACHE_FORMAT
        _read_upfront_cache = \
            interface_madspin.MadSpinInterface._read_upfront_cache

    def _write(self, payload):
        import json, tempfile
        handle, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(handle, 'w') as f:
            json.dump(payload, f)
        self.addCleanup(os.remove, path)
        return path

    def _good(self):
        return {'format': self._Stub._UPFRONT_CACHE_FORMAT,
                'maxwgts': [17.0, 2.3, 3.9],
                'z_tables': {'6_0': {'pole': 173.0, 'coeff': [0.0, 2.0, -1.0],
                                     'zero_below': 0.0, 'range': [150.0, 195.0]}}}

    def test_round_trip(self):
        got = self._Stub()._read_upfront_cache(self._write(self._good()))
        self.assertEqual(got['maxwgts'], [17.0, 2.3, 3.9])
        self.assertEqual(got['z_tables']['6_0']['pole'], 173.0)

    def test_missing_file_is_not_an_error(self):
        self.assertIsNone(self._Stub()._read_upfront_cache('/no/such/file'))
        self.assertIsNone(self._Stub()._read_upfront_cache(''))

    def test_every_malformed_shape_is_ignored(self):
        """Each of these would otherwise surface as a KeyError, an IndexError or
        a wrong weight somewhere inside the unweighting loop."""
        cases = {}
        cases['no format tag'] = {k: v for k, v in self._good().items()
                                  if k != 'format'}
        cases['older format'] = dict(self._good(), format=0)
        cases['no bounds'] = dict(self._good(), maxwgts=[])
        cases['no tables'] = {k: v for k, v in self._good().items()
                              if k != 'z_tables'}
        short = self._good()
        short['z_tables']['6_0'].pop('zero_below')
        cases['table missing a field'] = short
        degree = self._good()
        degree['z_tables']['6_0']['coeff'] = [0.0, 2.0, -1.0, 0.5]
        cases['a cubic fit'] = degree
        window = self._good()
        window['z_tables']['6_0']['range'] = [150.0]
        cases['a malformed range'] = window
        for name, payload in cases.items():
            self.assertIsNone(
                self._Stub()._read_upfront_cache(self._write(payload)), name)

    def test_garbage_is_ignored(self):
        import tempfile
        handle, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(handle, 'w') as f:
            f.write('not json at all')
        self.addCleanup(os.remove, path)
        self.assertIsNone(self._Stub()._read_upfront_cache(path))


class TestPolyfitConditioning(unittest.TestCase):
    """_weighted_polyfit2 solves the normal equations of a fit in
    u = ln(m/pole), which spans about +-0.13 over a Breit-Wigner window.  The
    moments therefore range over four orders of magnitude before any data is
    seen, which is why the degeneracy test has to be relative to the size of the
    matrix rather than an absolute floor.
    """

    def _fit(self, xs, ys, ws=None):
        return interface_madspin.MadSpinInterface._weighted_polyfit2(
                        xs, ys, ws or [1.0] * len(xs))

    def _window(self, pole=173.0, lo=150.7, hi=195.4, nb=20):
        return [math.log((lo + i * (hi - lo) / (nb - 1)) / pole)
                for i in range(nb)]

    def test_recovers_a_quadratic_on_the_real_fit_variable(self):
        """Exact data over the actual window, with the actual weights (bin
        counts in the thousands): the conditioning of this fit is ~3e4, so
        double precision must return the coefficients essentially exactly."""
        xs = self._window()
        ys = [0.3 + 2.0 * x - 1.0 * x ** 2 for x in xs]
        got = self._fit(xs, ys, [1000.0] * len(xs))
        for value, want in zip(got, [0.3, 2.0, -1.0]):
            self.assertAlmostEqual(value, want, places=8)

    def test_all_bins_at_one_virtuality_is_degenerate(self):
        """The case the tolerance exists for.  An absolute floor of 1e-30 would
        let this through -- the moments are O(n) -- and return a quadratic
        fitted to nothing."""
        self.assertIsNone(self._fit([0.1] * 6, [1.0, 2.0, 1.5, 1.2, 1.8, 1.4]))

    def test_a_needle_narrow_window_is_degenerate(self):
        """A resonance whose samples all land within rounding of each other:
        the moments are tiny in absolute terms but the matrix is still
        singular relative to itself."""
        xs = [1e-13 * i for i in range(6)]
        self.assertIsNone(self._fit(xs, [1.0, 2.0, 1.5, 1.2, 1.8, 1.4],
                                    [1000.0] * 6))

    def test_two_distinct_points_cannot_fix_a_quadratic(self):
        self.assertIsNone(self._fit([0.0, 0.0, 0.1, 0.1], [1.0, 1.0, 2.0, 2.0]))


class TestClampedPartialWidth(unittest.TestCase):
    """_clamped_partial_width reconciles a *measured* partial width with the
    param_card total. The two regimes are deliberate and long-standing, so they
    are pinned here: a review read the helper as under-clamping, when in fact
    capping the large-disagreement case is what would be wrong.
    """

    fct = staticmethod(
        interface_madspin.MadSpinInterface._clamped_partial_width)

    def test_below_the_total_is_untouched(self):
        self.assertEqual(self.fct(0.4, 1.0), 0.4)
        self.assertEqual(self.fct(1.0, 1.0), 1.0)

    def test_a_per_cent_over_is_monte_carlo_noise_and_is_capped(self):
        """Within 1% the excess is noise in the width measurement; capping keeps
        the branching ratio at most 1 and costs nothing."""
        self.assertEqual(self.fct(1.005, 1.0), 1.0)
        self.assertEqual(self.fct(1.01, 1.0), 1.0)

    def test_a_real_disagreement_is_reported_not_hidden(self):
        """Past 1% the param_card total genuinely disagrees with what was
        generated. The measured value is kept: capping would quietly reshape the
        normalisation to match a card that is wrong, and swallow the evidence.
        """
        self.assertEqual(self.fct(1.5, 1.0), 1.5)
        self.assertEqual(self.fct(20.0, 1.0), 20.0)
class TestDecayGroupTagWarning(unittest.TestCase):
    """The `@` grouping tags of the semi-leptonic ttbar idiom are implemented
    only in madspin_v1. Everywhere else MG5 reads them as an ordinary process
    number and they change nothing, so MadSpin must say so rather than let the
    card mean something it does not."""

    class _Stub(object):
        _DECAY_GROUP_TAG = interface_madspin.MadSpinInterface._DECAY_GROUP_TAG
        _split_group_tag = interface_madspin.MadSpinInterface._split_group_tag
        _warn_ignored_decay_groups = \
            interface_madspin.MadSpinInterface._warn_ignored_decay_groups

        def __init__(self, list_branches):
            self.list_branches = list_branches

    TAGGED = {'t': ['t > w+ b, w+ > l+ vl @1', 't > w+ b, w+ > j j @2'],
              't~': ['t~ > w- b~, w- > j j @1', 't~ > w- b~, w- > l- vl~ @2']}
    UNTAGGED = {'z': ['z > e+ e-', 'z > u u~']}

    def test_tags_are_reported_in_every_density_mode(self):
        for spinmode in ('madspin', 'full', 'PA', 'onshell', 'onshell_v1',
                         'none'):
            stub = self._Stub(self.TAGGED)
            found = stub._warn_ignored_decay_groups(spinmode)
            self.assertEqual(len(found), 4, spinmode)
            self.assertEqual(sorted(set(t[2] for t in found)), ['@1', '@2'])

    def test_madspin_v1_is_silent(self):
        """v1 implements the grouping, so there is nothing to warn about."""
        stub = self._Stub(self.TAGGED)
        self.assertEqual(stub._warn_ignored_decay_groups('madspin_v1'), [])

    def test_untagged_card_is_silent(self):
        stub = self._Stub(self.UNTAGGED)
        self.assertEqual(stub._warn_ignored_decay_groups('madspin'), [])

    def test_whitespace_between_at_and_number(self):
        stub = self._Stub({'t': ['t > w+ b, w+ > l+ vl @ 12']})
        self.assertEqual([t[2] for t in
                          stub._warn_ignored_decay_groups('madspin')], ['@12'])

    def test_a_coupling_restriction_is_not_a_group_tag(self):
        """`@` only tags a group when a number follows it; QED=1 and the like
        must not trigger the warning."""
        stub = self._Stub({'t': ['t > w+ b QED=1, w+ > l+ vl']})
        self.assertEqual(stub._warn_ignored_decay_groups('madspin'), [])


class TestDecayGroupLayout(unittest.TestCase):
    """`@` grouping tags: sorting the decay lines into groups, and deciding
    whether the grouping can be honoured for a given set of production events.

    Supported shape is rectangular -- every group decays every particle exactly
    once (or n times for n identical parents). Anything else is refused rather
    than approximated, because the group is then not a complete assignment and
    neither its rate nor its branching ratio is defined.
    """

    class _Stub(object):
        _DECAY_GROUP_TAG = interface_madspin.MadSpinInterface._DECAY_GROUP_TAG
        _split_group_tag = interface_madspin.MadSpinInterface._split_group_tag
        _decay_group_layout = \
            interface_madspin.MadSpinInterface._decay_group_layout
        _validate_decay_groups = \
            interface_madspin.MadSpinInterface._validate_decay_groups

        def __init__(self, list_branches):
            self.list_branches = collections.OrderedDict(list_branches)

    # the semi-leptonic ttbar idiom: one t and one t~ per event, two groups
    TTBAR = [('t',  ['t > w+ b, w+ > l+ vl @1', 't > w+ b, w+ > j j @2']),
             ('t~', ['t~ > w- b~, w- > j j @1', 't~ > w- b~, w- > l- vl~ @2'])]
    NAME2PDG = staticmethod(lambda name: {'t': 6, 't~': -6, 'z': 23,
                                          'w+': 24}.get(name))

    # ------------------------------------------------------------------ split
    def test_split_tag(self):
        self.assertEqual(
            interface_madspin.MadSpinInterface._split_group_tag(
                't > w+ b, w+ > l+ vl @1'),
            ('t > w+ b, w+ > l+ vl', '1'))

    def test_split_tag_tolerates_spacing(self):
        self.assertEqual(
            interface_madspin.MadSpinInterface._split_group_tag(
                't > w+ b @ 12  '),
            ('t > w+ b', '12'))

    def test_untagged_line_is_left_alone(self):
        for branch in ('z > e+ e-', 't > w+ b QED=1', 'z > mu+ mu- {T}'):
            self.assertEqual(
                interface_madspin.MadSpinInterface._split_group_tag(branch),
                (branch, None))

    # ----------------------------------------------------------------- layout
    def test_no_tag_is_not_a_grouping(self):
        layout, reason = self._Stub([('z', ['z > e+ e-', 'z > u u~'])]) \
                             ._decay_group_layout()
        self.assertIsNone(layout)
        self.assertIsNone(reason)

    def test_layout_of_the_ttbar_idiom(self):
        layout, reason = self._Stub(self.TTBAR)._decay_group_layout()
        self.assertIsNone(reason)
        self.assertEqual(layout['tags'], ['1', '2'])
        # index i is also the number of the decay_<pdg>_<i> pool
        self.assertEqual(layout['lines']['t'],  {'1': [0], '2': [1]})
        self.assertEqual(layout['lines']['t~'], {'1': [0], '2': [1]})

    def test_untagged_line_belongs_to_every_group(self):
        layout, reason = self._Stub(
            self.TTBAR + [('z', ['z > e+ e-'])])._decay_group_layout()
        self.assertIsNone(reason)
        self.assertEqual(layout['lines']['z'], {'1': [0], '2': [0]})

    def test_stray_at_is_refused_not_half_read(self):
        layout, reason = self._Stub(
            [('t', ['t > w+ b @1 QED=1'])])._decay_group_layout()
        self.assertIsNone(layout)
        self.assertIn("not a group tag", reason)

    # --------------------------------------------------------------- validate
    def _validate(self, branches, to_decay, nb_event=100):
        stub = self._Stub(branches)
        layout, reason = stub._decay_group_layout()
        self.assertIsNone(reason)
        self.assertIsNotNone(layout)
        return stub._validate_decay_groups(layout, to_decay, nb_event,
                                           self.NAME2PDG)

    def test_ttbar_is_supported(self):
        ok, reason = self._validate(self.TTBAR, {6: 100, -6: 100})
        self.assertTrue(ok, reason)

    def test_four_tops_two_lines_per_group_is_supported(self):
        """p p > t t t~ t~: a group hands its two lines for a pdg to the two
        particles of that pdg, by the positional rule."""
        branches = [
            ('t',  ['t > w+ b, w+ > l+ vl @1', 't > w+ b, w+ > j j @1',
                    't > w+ b, w+ > j j @2',   't > w+ b, w+ > j j @2']),
            ('t~', ['t~ > w- b~, w- > j j @1', 't~ > w- b~, w- > j j @1',
                    't~ > w- b~, w- > l- vl~ @2', 't~ > w- b~, w- > j j @2']),
        ]
        ok, reason = self._validate(branches, {6: 200, -6: 200})
        self.assertTrue(ok, reason)

    def test_group_missing_a_particle_is_refused(self):
        branches = [('t',  ['t > w+ b, w+ > l+ vl @1',
                            't > w+ b, w+ > j j @2']),
                    ('t~', ['t~ > w- b~, w- > j j @1'])]      # no @2 for t~
        ok, reason = self._validate(branches, {6: 100, -6: 100})
        self.assertFalse(ok)
        self.assertIn('@2', reason)
        self.assertIn('t~', reason)

    def test_wrong_line_count_for_the_multiplicity_is_refused(self):
        """two tops per event but only one line per group."""
        ok, reason = self._validate(self.TTBAR, {6: 200, -6: 200})
        self.assertFalse(ok)
        self.assertIn('1 decay line(s)', reason)

    def test_tagged_and_untagged_for_the_same_particle_is_refused(self):
        """the untagged line joins every group, so that group has two lines for
        a particle the event carries once."""
        branches = [('t',  ['t > w+ b, w+ > l+ vl @1',
                            't > w+ b, w+ > j j @2',
                            't > w+ b, w+ > ta+ vt']),
                    ('t~', ['t~ > w- b~, w- > j j @1',
                            't~ > w- b~, w- > l- vl~ @2'])]
        ok, reason = self._validate(branches, {6: 100, -6: 100})
        self.assertFalse(ok)
        self.assertIn('2 decay line(s)', reason)

    def test_mixed_final_states_are_refused(self):
        """not every event carries a t, so there is no branching ratio per
        group to normalise with."""
        ok, reason = self._validate(self.TTBAR, {6: 150, -6: 150})
        self.assertFalse(ok)
        self.assertIn('same number', reason)

    def test_multiparticle_parent_is_refused(self):
        branches = [('t',  ['t > w+ b, w+ > l+ vl @1',
                            't > w+ b, w+ > j j @2']),
                    ('vv', ['vv > e+ e- @1', 'vv > u u~ @2'])]
        ok, reason = self._validate(branches, {6: 100})   # 'vv' -> None
        self.assertFalse(ok)
        self.assertIn('multiparticle', reason)

    def test_a_particle_absent_from_the_events_is_ignored(self):
        """a decay line for a species that never appears is ignored anyway, so
        it must not make the grouping unusable."""
        branches = self.TTBAR + [('z', ['z > e+ e- @1', 'z > u u~ @2'])]
        ok, reason = self._validate(branches, {6: 100, -6: 100})
        self.assertTrue(ok, reason)


class TestDecayGroupDraw(unittest.TestCase):
    """The run-time half of the grouping: a group is drawn once per event, with
    the probability of its rate, and every particle then takes that group's
    channel."""

    class _Pool(object):
        def __init__(self, tag, n=200, cross=1.0):
            self.tag = tag
            self._it = iter(range(n))
            self.cross = cross
        def __next__(self):
            return '%s:%s' % (self.tag, next(self._it))

    class _Part(object):
        def __init__(self, pid):
            self.pid = pid
            self.pdg = pid
            self.status = 1

    class _Model(object):
        NAMES = {6: 't', -6: 't~'}
        def get_particle(self, pdg):
            name = self.NAMES[pdg]
            return type('P', (), {'get_name': staticmethod(lambda n=name: n)})()

    class _Stub(object):
        _DECAY_GROUP_TAG = interface_madspin.MadSpinInterface._DECAY_GROUP_TAG
        _split_group_tag = interface_madspin.MadSpinInterface._split_group_tag
        _assignment_multiplicity = \
            interface_madspin.MadSpinInterface._assignment_multiplicity
        _clamped_partial_width = staticmethod(
            interface_madspin.MadSpinInterface._clamped_partial_width)
        _resolve_group_rates = \
            interface_madspin.MadSpinInterface._resolve_group_rates
        _draw_decay_group = interface_madspin.MadSpinInterface._draw_decay_group
        _draw_one_decay = interface_madspin.MadSpinInterface._draw_one_decay
        _draw_all_decays = interface_madspin.MadSpinInterface._draw_all_decays
        get_decay_from_file = \
            interface_madspin.MadSpinInterface.get_decay_from_file
        efficiency = 0.5

    # ------------------------------------------------- assignment multiplicity
    def test_multiplicity_counts_distinct_assignments(self):
        mult = interface_madspin.MadSpinInterface._assignment_multiplicity
        self.assertEqual(mult(['t > w+ b, w+ > l+ vl']), 1)
        self.assertEqual(mult(['a > b c', 'a > d e']), 2)
        self.assertEqual(mult(['a > b c', 'a > b c']), 1)   # NOT 2
        self.assertEqual(mult(['a > b c', 'a > d e', 'a > f g']), 6)
        self.assertEqual(mult(['a > b c', 'a > b c', 'a > d e']), 3)

    def test_multiplicity_ignores_the_tag(self):
        mult = interface_madspin.MadSpinInterface._assignment_multiplicity
        self.assertEqual(mult(['a > b c @1', 'a > b c @2']), 1)

    # ------------------------------------------------------------- group rates
    def _ttbar_stub(self, g_lep=2.0, g_had=6.0, totwidth=9.0):
        stub = self._Stub()
        stub.model = self._Model()
        stub.list_branches = {
            't':  ['t > w+ b, w+ > l+ vl @1', 't > w+ b, w+ > j j @2'],
            't~': ['t~ > w- b~, w- > j j @1', 't~ > w- b~, w- > l- vl~ @2']}
        stub._decay_groups = {'tags': ['1', '2'],
                              'lines': {6:  {'1': [0], '2': [1]},
                                        -6: {'1': [0], '2': [1]}},
                              'prob': None}
        gen_jobs = {6: {'totwidth': totwidth}, -6: {'totwidth': totwidth}}
        channel_widths = {6:  {0: g_lep, 1: g_had},
                          -6: {0: g_had, 1: g_lep}}
        return stub, gen_jobs, channel_widths

    def test_semileptonic_branching_ratio(self):
        """the tt~ idiom: BR = 2 x BR_lep x BR_had, which is what madspin_v1
        writes for the same card."""
        stub, gen_jobs, widths = self._ttbar_stub()
        br = stub._resolve_group_rates(gen_jobs, widths)
        self.assertAlmostEqual(br, 2 * (2. / 9.) * (6. / 9.), places=12)
        self.assertEqual(stub._decay_groups['prob'], [0.5, 0.5])

    def test_group_probability_follows_the_rate(self):
        """an asymmetric card: group 1 is lep+had, group 2 had+had, so group 2
        is the more likely of the two in the ratio of their widths."""
        stub, gen_jobs, widths = self._ttbar_stub()
        widths[-6] = {0: 6.0, 1: 6.0}          # t~ hadronic in both groups
        br = stub._resolve_group_rates(gen_jobs, widths)
        # br_1 = (2/9)(6/9), br_2 = (6/9)(6/9)
        self.assertAlmostEqual(br, (2 * 6 + 6 * 6) / 81., places=12)
        self.assertAlmostEqual(stub._decay_groups['prob'][0], 12. / 48.)
        self.assertAlmostEqual(stub._decay_groups['prob'][1], 36. / 48.)

    def test_zero_rate_everywhere_is_an_error_not_a_silent_zero(self):
        stub, gen_jobs, widths = self._ttbar_stub()
        widths[6] = {0: 0.0, 1: 0.0}
        self.assertRaises(Exception, stub._resolve_group_rates,
                          gen_jobs, widths)

    # -------------------------------------------------------------- group draw
    def test_group_is_drawn_with_its_probability(self):
        import random
        stub, gen_jobs, widths = self._ttbar_stub()
        stub._decay_groups['prob'] = [0.25, 0.75]
        random.seed(7)
        drawn = collections.Counter(stub._draw_decay_group()
                                    for _ in range(20000))
        self.assertAlmostEqual(drawn['1'] / 20000., 0.25, places=2)
        self.assertAlmostEqual(drawn['2'] / 20000., 0.75, places=2)

    def test_no_group_declared_draws_none(self):
        stub = self._Stub()
        stub._decay_groups = None
        self.assertIsNone(stub._draw_decay_group())

    # ------------------------------------------------- the draw inside a group
    def test_every_particle_takes_the_drawn_group_channel(self):
        import random
        stub, gen_jobs, widths = self._ttbar_stub()
        stub._resolve_group_rates(gen_jobs, widths)
        production = [self._Part(6), self._Part(-6)]
        random.seed(3)
        seen = collections.Counter()
        for _ in range(400):
            evt_decayfile = {6:  {0: self._Pool('t_lep'), 1: self._Pool('t_had')},
                             -6: {0: self._Pool('tx_had'), 1: self._Pool('tx_lep')}}
            out = stub.get_decay_from_file(production, evt_decayfile, 10)
            seen[(out[6][0].split(':')[0], out[-6][0].split(':')[0])] += 1
        # only the two tagged assignments, never lep+lep or had+had
        self.assertEqual(set(seen), {('t_lep', 'tx_had'), ('t_had', 'tx_lep')})
        for count in seen.values():
            self.assertGreater(count, 150)      # both groups are used

    def test_identical_parents_inside_a_group_are_positional(self):
        """p p > t t t~ t~: the group's two lines for a pdg go to its two
        particles in order."""
        stub = self._Stub()
        stub.model = self._Model()
        stub.list_branches = {'t': ['a @1', 'b @1', 'c @2', 'd @2']}
        stub._decay_groups = {'tags': ['1', '2'],
                              'lines': {6: {'1': [0, 1], '2': [2, 3]}},
                              'prob': [1.0, 0.0]}
        production = [self._Part(6), self._Part(6)]
        evt_decayfile = {6: dict((i, self._Pool('c%d' % i)) for i in range(4))}
        out = stub.get_decay_from_file(production, evt_decayfile, 10)
        self.assertEqual([d.split(':')[0] for d in out[6]], ['c0', 'c1'])

        stub._decay_groups['prob'] = [0.0, 1.0]
        evt_decayfile = {6: dict((i, self._Pool('c%d' % i)) for i in range(4))}
        out = stub.get_decay_from_file(production, evt_decayfile, 10)
        self.assertEqual([d.split(':')[0] for d in out[6]], ['c2', 'c3'])


class TestUnweightingDecisionTable(unittest.TestCase):
    """The upfront / unweighting decision logic, exhaustively.

    `_unweighting_mode` and the predicates built on it (`_sequential_active`,
    `_sequential_upfront`, `_sequential_offshell`, `_sequential_pool_ladder`)
    decide, per run, how the accept/reject is organised. The branching is
    spinmode-specific and layered -- `auto` resolves on the spinmode family and
    the decay multiplicity, then several fallbacks can still send the run back
    to the joint test -- so it is pinned here over *every* reachable
    combination, against the rules restated independently of the implementation
    (`_reference_mode` below, written from the `unweighting` option comment).
    """

    # every declared spinmode, plus 'bridge' and a name no branch knows about
    SPINMODES = ('full', 'madspin', 'none', 'onshell', 'PA', 'madspin_v1',
                 'onshell_v1', 'bridge', 'not_a_spinmode')
    UNWEIGHTING = ('auto', 'joint', 'two_stage', 'sequential',
                   'sequential_global_retry', 'sequential_with_mass')
    NB_DECAYING = (0, 1, 2, 3, 4, 7)
    POLE_APPROXIMATION = ('PA', 'onshell')
    DENSITY_SPINMODES = ('madspin', 'full', 'PA', 'onshell')
    # 'p p > t{+} t~{+}' as _production_polarization reports it
    POLARIZATIONS = (None, {6: ((1,),), -6: ((1,),)})

    class _Part(object):
        def __init__(self, spin):
            self.spin = spin

        def get(self, key):
            assert key == 'spin'
            return self.spin

    class _Model(object):
        SPINS = {6: 2, -6: 2, 24: 3, 23: 3, 25: 1}

        def get_particle(self, pdg):
            if pdg not in self.SPINS:
                raise Exception('unknown particle')
            return TestUnweightingDecisionTable._Part(self.SPINS[pdg])

    class _Stub(object):
        """The real methods, on the smallest object that can carry them."""
        _borrow_decision_helpers(locals())
        for _name in ('_unweighting_mode', '_announce_mode', '_log_once',
                      '_sequential_active', '_sequential_upfront',
                      '_sequential_offshell', '_sequential_pool_ladder',
                      '_sequential_spin_order', '_decay_pool_ladder'):
            # getattr_static keeps the staticmethod wrappers intact
            locals()[_name] = inspect.getattr_static(
                interface_madspin.MadSpinInterface, _name)
        del _name

        # keyword-only: a new knob must not be able to shift the meaning of an
        # existing argument at a call site
        def __init__(self, *, spinmode='madspin', unweighting='auto',
                     nb_decaying=2, fixed_order=False, decay_groups=None,
                     polarization=None):
            self.options = {'spinmode': spinmode, 'unweighting': unweighting,
                            'fixed_order': fixed_order,
                            'sequential_spin_order': '2 3 1'}
            self._nb_decaying = nb_decaying
            self._decay_groups = decay_groups
            # seed _production_polarization's cache rather than a banner: '{}'
            # is what it returns for a process line without braces
            self._production_polarization_cache = polarization or {}
            self.model = TestUnweightingDecisionTable._Model()
            self._logged_once = set()

    # one point of the exhaustive product below. Named, so that adding a
    # dimension cannot silently reinterpret the existing ones -- which is
    # exactly what `polarization` would have done as a seventh tuple slot.
    _Case = collections.namedtuple(
        '_Case', ('spinmode', 'unweighting', 'nb_decaying', 'fixed_order',
                  'decay_groups', 'polarization', 'density_method'))

    @classmethod
    def _reference_mode(cls, case):
        """The rules as documented on the `unweighting` option, restated here
        rather than read off the implementation, so this is a check and not a
        tautology."""
        if not case.density_method:
            return 'joint'                       # only scheme outside density
        mode = case.unweighting
        if mode == 'auto':
            if case.spinmode in cls.POLE_APPROXIMATION:
                mode = 'sequential'              # fastest at every measured n
            elif case.polarization and case.spinmode in cls.DENSITY_SPINMODES:
                mode = 'sequential'              # restricted convolution: the
                                                 # joint weight sits orders of
                                                 # magnitude below its bound
            elif case.nb_decaying <= 2:
                mode = 'joint'                   # offshell, too few decays
            else:
                mode = 'sequential'
        if mode == 'joint':
            return 'joint'
        if case.fixed_order:
            return 'joint'                       # counter-events ride along
        if case.decay_groups:
            return 'joint'                       # '@' groups self-normalise
        if case.spinmode not in ('PA', 'onshell', 'madspin', 'full'):
            return 'joint'                       # no density matrix to stage
        if (mode == 'sequential_with_mass'
                and case.spinmode not in cls.POLE_APPROXIMATION):
            return 'sequential'                  # needs a per-particle mass
        return mode

    def _case_stub(self, case):
        """The stub a case describes. `density_method` stays out of it: it is
        the argument the decision methods take, not stub state."""
        return self._Stub(spinmode=case.spinmode,
                          unweighting=case.unweighting,
                          nb_decaying=case.nb_decaying,
                          fixed_order=case.fixed_order,
                          decay_groups=case.decay_groups,
                          polarization=case.polarization)

    def _cases(self):
        for spinmode in self.SPINMODES:
            for unweighting in self.UNWEIGHTING:
                for nb_decaying in self.NB_DECAYING:
                    for fixed_order in (False, True):
                        for groups in (None, {'tags': ['1', '2']}):
                            for pol in self.POLARIZATIONS:
                                for density_method in (True, False):
                                    yield self._Case(
                                        spinmode=spinmode,
                                        unweighting=unweighting,
                                        nb_decaying=nb_decaying,
                                        fixed_order=fixed_order,
                                        decay_groups=groups,
                                        polarization=pol,
                                        density_method=density_method)

    def test_every_combination_matches_the_documented_rules(self):
        seen = set()
        for case in self._cases():
            stub = self._case_stub(case)
            got = stub._unweighting_mode(case.density_method)
            seen.add(got)
            self.assertEqual(got, self._reference_mode(case), msg=str(case))
        # the table is not degenerate: every scheme is reachable through it
        self.assertEqual(seen, {'joint', 'two_stage', 'sequential',
                                'sequential_global_retry',
                                'sequential_with_mass'})

    def test_auto_resolves_on_the_family_then_the_multiplicity(self):
        """Spelled out, since it is the branch a user never sets by hand:
        sequential everywhere under PA/onshell, joint offshell up to two
        decaying particles and sequential from three -- and sequential at every
        multiplicity once the production carries a polarisation brace."""
        pol = self.POLARIZATIONS[1]
        for spinmode in ('madspin', 'full', 'PA', 'onshell'):
            for nb in self.NB_DECAYING:
                stub = self._Stub(spinmode=spinmode, unweighting='auto',
                                  nb_decaying=nb, polarization=pol)
                self.assertEqual(stub._auto_unweighting_mode(), 'sequential',
                                 ('polarised', spinmode, nb))
        for spinmode in ('PA', 'onshell'):
            for nb in self.NB_DECAYING:
                stub = self._Stub(spinmode=spinmode, unweighting='auto',
                                  nb_decaying=nb)
                self.assertEqual(stub._auto_unweighting_mode(),
                                 'sequential', (spinmode, nb))
        for spinmode in ('madspin', 'full'):
            for nb, expected in ((0, 'joint'), (1, 'joint'), (2, 'joint'),
                                 (3, 'sequential'), (4, 'sequential'),
                                 (7, 'sequential')):
                stub = self._Stub(spinmode=spinmode, unweighting='auto',
                                  nb_decaying=nb)
                self.assertEqual(stub._auto_unweighting_mode(),
                                 expected, (spinmode, nb))

    def test_auto_without_a_measured_multiplicity_assumes_two(self):
        """`_nb_decaying` is set while the decays are prepared; anything asking
        before that must not crash."""
        stub = self._Stub(spinmode='madspin', unweighting='auto')
        del stub._nb_decaying
        self.assertEqual(stub._unweighting_mode(), 'joint')

    def test_sequential_active_is_exactly_not_joint(self):
        for case in self._cases():
            stub = self._case_stub(case)
            self.assertEqual(stub._sequential_active(case.density_method),
                             stub._unweighting_mode(case.density_method)
                                != 'joint',
                             msg=str(case))

    def test_upfront_is_every_scheme_but_joint_and_with_mass(self):
        for mode, expected in (('joint', False), ('two_stage', True),
                               ('sequential', True),
                               ('sequential_global_retry', True),
                               ('sequential_with_mass', False)):
            self.assertEqual(self._Stub()._is_upfront_scheme(mode), expected,
                             mode)
        for case in self._cases():
            stub = self._case_stub(case)
            self.assertEqual(
                stub._sequential_upfront(case.density_method),
                stub._unweighting_mode(case.density_method) not in
                    ('joint', 'sequential_with_mass'),
                msg=str(case))

    def test_with_mass_falls_back_to_sequential_offshell_only(self):
        """It needs a per-particle mass draw; the offshell spinmodes reshuffle
        the whole production onto the mass set at once."""
        for spinmode in ('PA', 'onshell'):
            stub = self._Stub(spinmode=spinmode, nb_decaying=2,
                              unweighting='sequential_with_mass')
            self.assertEqual(stub._unweighting_mode(), 'sequential_with_mass')
            self.assertFalse(stub._sequential_upfront())
        for spinmode in ('madspin', 'full'):
            stub = self._Stub(spinmode=spinmode, nb_decaying=2,
                              unweighting='sequential_with_mass')
            self.assertEqual(stub._unweighting_mode(), 'sequential')
            self.assertTrue(stub._sequential_upfront())

    def test_the_spinmode_family_predicates(self):
        for spinmode in self.SPINMODES:
            stub = self._Stub(spinmode=spinmode)
            self.assertEqual(stub._density_pole_approximation(),
                             spinmode in ('PA', 'onshell'), spinmode)
            self.assertEqual(stub._density_do_reshuffle(), spinmode == 'PA',
                             spinmode)
            self.assertEqual(stub._spinmode_has_density(),
                             spinmode in ('PA', 'onshell', 'madspin', 'full'),
                             spinmode)
            # offshell is exactly the complement of the pole approximation
            self.assertEqual(stub._sequential_offshell(),
                             not stub._density_pole_approximation(), spinmode)

    def test_needs_reshuffle_is_offshell_or_pa_inside_density_mode(self):
        for spinmode in self.SPINMODES:
            stub = self._Stub(spinmode=spinmode)
            self.assertFalse(stub._density_needs_reshuffle(False), spinmode)
            self.assertEqual(bool(stub._density_needs_reshuffle(True)),
                             spinmode != 'onshell', spinmode)

    def test_pool_ladder_is_empty_unless_a_staged_scheme_is_in_use(self):
        to_decay, nb_event = {6: 100, -6: 100}, 100
        for case in self._cases():
            stub = self._case_stub(case)
            ladder = stub._sequential_pool_ladder(dict(to_decay), nb_event,
                                                  case.density_method)
            if stub._unweighting_mode(case.density_method) == 'joint':
                self.assertEqual(ladder, {}, msg=str(case))
            else:
                self.assertEqual(sorted(ladder), [-6, 6], msg=str(case))
                self.assertEqual(sorted(ladder.values()), [1.5, 2.0],
                                 msg=str(case))

    def test_pool_ladder_gives_up_on_a_particle_the_model_does_not_know(self):
        stub = self._Stub(spinmode='PA', unweighting='sequential',
                          nb_decaying=2)
        self.assertEqual(stub._sequential_pool_ladder({6: 100, 999: 100}, 100,
                                                      True), {})


class TestCheckWeightIdentitySlotPairing(unittest.TestCase):
    """_check_weight_identity: the PA branch undoes each accepted decay's boost
    with ``parents[slot]``, and picks the slot with a free-running index over
    the *grouped* walk of the ``decays`` dict.

    Those two orderings coincide by construction, and this pins that down:

      * ``_sequential_slots`` lays the slots out as "for pdg in decays_key, for
        particle in production order", so a pdg owns a *contiguous* block of
        slots and the blocks come in decays_key order;
      * ``sequential_accept_reject`` builds the returned dict by ascending slot
        (``for slot in range(len(order))``), never in accept/reject order --
        ``_decay_slot_order`` decides which slot is *drawn* next and must never
        permute the layout;
      * so the dict's keys are in decays_key order, each pdg's list is that
        pdg's slot block in ascending order, and the flat walk enumerates
        slots 0 .. n-1.

    A mis-pairing here would undo a boost with another particle's momentum and
    corrupt the joint weight the check compares against. It is a *debug-only*
    path (``sequential_debug``), so it could never move a physics result -- but
    a silently wrong cross-check is worse than none.
    """

    MT = 173.0
    MW = 79.8

    # production final states, deliberately interleaved so that slot order and
    # production order genuinely differ
    LAYOUTS = {
        # p p > t t~ t : slots are (t@0, t@2, t~@1) -- slot 1 belongs to the
        # *third* final-state particle, so a free-running index over the
        # production would hand it the anti-top
        'ttxt': [6, -6, 6],
        # three pdgs, two of them owning two slots each
        'ttxwtxw': [6, -6, 24, -6, 24],
        # a single pdg owning every slot
        'tttt': [6, 6, 6, 6],
    }

    @staticmethod
    def _energy(m, px, py, pz):
        return math.sqrt(m * m + px * px + py * py + pz * pz)

    @staticmethod
    def _lhe(parts):
        head = (' %d      1 +1.0000000e+00 1.00000000e+02 7.54677100e-03'
                ' 1.17102600e-01' % len(parts))
        lines = [head]
        for (pid, status, m1, m2, px, py, pz, energy, mass) in parts:
            lines.append(' %5d %2d %4d %4d    0    0 %+.10e %+.10e %+.10e'
                         ' %.10e %.10e 0.0000e+00 9.0000e+00'
                         % (pid, status, m1, m2, px, py, pz, energy, mass))
        return '<event>\n' + '\n'.join(lines) + '\n</event>'

    def _mass(self, pid):
        return self.MW if abs(pid) == 24 else self.MT

    def _production(self, pids):
        """A production event whose final state carries ``pids``, each with its
        own distinctive momentum so a mis-paired boost cannot go unnoticed."""
        momenta = [(60.0 + 17 * i, 30.0 - 23 * i, 120.0 - 47 * i)
                   for i in range(len(pids))]
        tot_pz = sum(p[2] for p in momenta)
        tot_e = sum(self._energy(self._mass(pid), *mom)
                    for pid, mom in zip(pids, momenta))
        parts = [(2, -1, 0, 0, 0.0, 0.0, (tot_e + tot_pz) / 2,
                  (tot_e + tot_pz) / 2, 0.0),
                 (-2, -1, 0, 0, 0.0, 0.0, -(tot_e - tot_pz) / 2,
                  (tot_e - tot_pz) / 2, 0.0)]
        for pid, (px, py, pz) in zip(pids, momenta):
            mass = self._mass(pid)
            parts.append((pid, 1, 1, 2, px, py, pz,
                          self._energy(mass, px, py, pz), mass))
        return lhe_parser.Event(self._lhe(parts))

    def _decay_on(self, parent, channel=0):
        """A two-body decay of ``parent``, built in its rest frame and then
        boosted onto it -- which is the frame PA hands its accepted decays back
        in, and the boost _check_weight_identity has to undo.

        ``channel`` picks a different opening angle, i.e. a different decay
        channel out of the pool: the pairing must not depend on it.
        """
        mass = self._mass(parent.pid)
        sign = 1 if parent.pid > 0 else -1
        half = mass / 2.0
        angle = 0.3 + 0.7 * channel
        parts = [(parent.pid, -1, 0, 0, 0.0, 0.0, 0.0, mass, mass),
                 (sign * 5, 1, 1, 1, half * math.sin(angle), 0.0,
                  half * math.cos(angle), half, 0.0),
                 (sign * -5, 1, 1, 1, -half * math.sin(angle), 0.0,
                  -half * math.cos(angle), half, 0.0)]
        decay = lhe_parser.Event(self._lhe(parts))
        # rest -> lab: Event.boost takes the event *into* the rest frame of the
        # momentum it is given (it flips the spatial part), so the momentum
        # that boosts out of it is the parent with its 3-momentum reversed
        decay.boost(lhe_parser.FourMomentum(parent.E, -parent.px,
                                            -parent.py, -parent.pz))
        return decay

    def _slots(self, production, pools):
        interface = interface_madspin.MadSpinInterface
        decays_key = interface._decaying_pdgs(production, pools)
        particles, slot_to_index = interface._sequential_slots(production,
                                                               decays_key)
        # exactly what _density_basis puts in init_part, i.e. what PA hands
        # _check_weight_identity as ``parents``
        init_part = [part for pdg in decays_key for part in production
                     if part.pid == pdg and part.status == 1]
        return decays_key, particles, slot_to_index, init_part

    @staticmethod
    def _pools(pids):
        # two decay channels per pdg: grouped iteration then has something to
        # group, and the pool multiplicity must not enter the pairing
        return dict((pid, {0: 'f', 1: 'f'}) for pid in set(pids))

    @staticmethod
    def _build_decays(particles, slot_to_index, slot_decays):
        """The dict layout sequential_accept_reject returns -- copied verbatim
        from its tail, so this test tracks it."""
        decays = collections.defaultdict(list)
        for slot in range(len(slot_to_index)):
            decays[particles[slot_to_index[slot]].pid].append(slot_decays[slot])
        return decays

    # ------------------------------------------------------------------
    # the ordering invariant itself
    # ------------------------------------------------------------------

    def test_grouped_walk_enumerates_the_slots_in_order(self):
        """The flat index over decays.items() *is* the slot index, for every
        layout -- including one where slot order and production order differ."""
        for name, pids in self.LAYOUTS.items():
            production = self._production(pids)
            _, particles, slot_to_index, init_part = \
                            self._slots(production, self._pools(pids))
            slot_decays = dict((slot, ('slot', slot))
                               for slot in range(len(slot_to_index)))
            decays = self._build_decays(particles, slot_to_index, slot_decays)

            index = 0
            for pdg, decay_list in decays.items():
                for decay in decay_list:
                    self.assertEqual(decay[1], index,
                                     '%s: grouped index %d is slot %d'
                                     % (name, index, decay[1]))
                    # and therefore parents[index] is that slot's particle
                    self.assertIs(init_part[index],
                                  particles[slot_to_index[decay[1]]])
                    self.assertEqual(init_part[index].pid, pdg)
                    index += 1
            self.assertEqual(index, len(slot_to_index))

    def test_slot_order_really_differs_from_production_order(self):
        """Guard on the guard: if the layouts stopped being interleaved the
        test above would pass vacuously."""
        production = self._production(self.LAYOUTS['ttxt'])
        _, _, slot_to_index, _ = self._slots(production,
                                             self._pools(self.LAYOUTS['ttxt']))
        self.assertEqual(slot_to_index, [0, 2, 1])
        production = self._production(self.LAYOUTS['ttxwtxw'])
        _, _, slot_to_index, _ = self._slots(
                        production, self._pools(self.LAYOUTS['ttxwtxw']))
        # t t~ W+ t~ W+ -> decays_key is (6, -6, 24), so the slots are
        # (t@0), (t~@1, t~@3), (W@2, W@4): grouped by pdg, not production order
        self.assertEqual(slot_to_index, [0, 1, 3, 2, 4])

    def test_every_final_state_layout_pairs_by_slot(self):
        """Swept over every final state of up to four particles drawn from
        three pdgs: the invariant is a property of the construction, not of the
        handful of layouts above."""
        import itertools
        interface = interface_madspin.MadSpinInterface
        for size in range(1, 5):
            for pids in itertools.product((6, -6, 24), repeat=size):
                production = self._production(list(pids))
                _, particles, slot_to_index, init_part = \
                                self._slots(production, self._pools(pids))
                slot_decays = dict((slot, slot)
                                   for slot in range(len(slot_to_index)))
                decays = self._build_decays(particles, slot_to_index,
                                            slot_decays)
                flat = [slot for lst in decays.values() for slot in lst]
                self.assertEqual(flat, list(range(len(slot_to_index))),
                                 'layout %s' % (pids,))
                # the dict keys are decays_key order, i.e. first appearance
                self.assertEqual(list(decays),
                                 list(interface._decaying_pdgs(
                                            production, self._pools(pids))))

    # ------------------------------------------------------------------
    # and what it buys: the real routine, boosting each decay back
    # ------------------------------------------------------------------

    class _Stub(object):
        _check_weight_identity = \
                interface_madspin.MadSpinInterface._check_weight_identity

        def calculate_matrix_element_from_density(self, prod, decays, dd):
            self.seen = decays
            return 1.0, None, 1.0, 1.0, 1.0

    def _run_check(self, production, decays, parents, nslot):
        stub = self._Stub()
        stats = collections.defaultdict(int)
        stub._check_weight_identity(production, decays, {}, 1.0,
                                    [[1, -1]] * nslot, stats,
                                    offshell=False, keep_jac=False,
                                    parents=parents)
        return stub.seen, stats

    def test_each_decay_is_boosted_by_its_own_slot_parent(self):
        """The observable consequence: every decay was handed back boosted onto
        its own parent, so undoing that boost with parents[slot] must leave
        every mother at rest. Pairing slot k with anything else leaves it
        moving."""
        for name, pids in self.LAYOUTS.items():
            production = self._production(pids)
            _, particles, slot_to_index, init_part = \
                            self._slots(production, self._pools(pids))
            nslot = len(slot_to_index)
            slot_decays = dict((slot, self._decay_on(init_part[slot],
                                                     channel=slot % 2))
                               for slot in range(nslot))
            decays = self._build_decays(particles, slot_to_index, slot_decays)

            seen, stats = self._run_check(production, decays, init_part, nslot)
            self.assertEqual(stats['nb_identity_check'], 1)
            nb = 0
            for pdg, decay_list in seen.items():
                for copy_evt in decay_list:
                    mother = copy_evt[0]
                    for comp in (mother.px, mother.py, mother.pz):
                        self.assertAlmostEqual(comp, 0.0, places=5,
                                               msg='%s: mother not at rest'
                                                   % name)
                    self.assertAlmostEqual(mother.E, self._mass(pdg), places=4)
                    nb += 1
            self.assertEqual(nb, nslot)

    def test_swapping_two_same_pdg_decays_is_caught(self):
        """The negative control the test above needs: give slot 0 the decay of
        slot 1 (both tops, so no pdg tells them apart) and the boost no longer
        undoes -- which is exactly the corruption a free-running index that did
        not match the slots would produce."""
        pids = self.LAYOUTS['ttxt']
        production = self._production(pids)
        _, particles, slot_to_index, init_part = \
                        self._slots(production, self._pools(pids))
        slot_decays = dict((slot, self._decay_on(init_part[slot]))
                           for slot in range(len(slot_to_index)))
        slot_decays[0], slot_decays[1] = slot_decays[1], slot_decays[0]
        decays = self._build_decays(particles, slot_to_index, slot_decays)

        seen, _ = self._run_check(production, decays, init_part,
                                  len(slot_to_index))
        moving = [max(abs(evt[0].px), abs(evt[0].py), abs(evt[0].pz))
                  for lst in seen.values() for evt in lst]
        self.assertEqual(sum(1 for m in moving if m > 1e-3), 2)

    def test_production_order_parents_trip_the_assertion(self):
        """And the cross-pdg case the assertion in _check_weight_identity
        covers: selecting the parent by a free-running index over the
        *production* final state instead of by slot hands slot 1 the anti-top.
        That must be loud, not silently wrong."""
        pids = self.LAYOUTS['ttxt']
        production = self._production(pids)
        _, particles, slot_to_index, init_part = \
                        self._slots(production, self._pools(pids))
        self.assertNotEqual([p.pid for p in particles],
                            [p.pid for p in init_part])
        slot_decays = dict((slot, self._decay_on(init_part[slot]))
                           for slot in range(len(slot_to_index)))
        decays = self._build_decays(particles, slot_to_index, slot_decays)
        self.assertRaises(AssertionError, self._run_check, production, decays,
                          particles, len(slot_to_index))


class TestGetPdirUnpackArity(unittest.TestCase):
    """``get_pdir`` grew from returning 2 values to 4 (single f2py library) to 5
    (loop-induced production) without every call site following along, and the
    result is a ``ValueError: too many values to unpack`` that only shows up when
    the call is actually made. These tests turn that into a test failure instead:
    change ``get_pdir``'s return and the arity check below goes red."""

    SOURCE = pjoin(MG5DIR, 'MadSpin', 'interface_madspin.py')

    # methods knowingly left behind an arity bump, by name. Empty: every call
    # site agrees with get_pdir today, and it stays that way.
    KNOWN_PENDING = set()

    def _tree(self):
        import ast
        with open(self.SOURCE) as fsock:
            return ast.parse(fsock.read()), ast

    def _interface_class(self):
        tree, ast = self._tree()
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == 'MadSpinInterface':
                return node, ast
        self.fail('MadSpinInterface not found in %s' % self.SOURCE)

    def _return_arity(self):
        """number of values ``MadSpinInterface.get_pdir`` returns"""
        klass, ast = self._interface_class()
        for meth in klass.body:
            if isinstance(meth, ast.FunctionDef) and meth.name == 'get_pdir':
                arities = [len(n.value.elts) for n in ast.walk(meth)
                           if isinstance(n, ast.Return) and n.value is not None
                           and isinstance(n.value, ast.Tuple)]
                self.assertTrue(arities, 'get_pdir returns no tuple')
                self.assertEqual(len(set(arities)), 1,
                                 'get_pdir returns tuples of differing length: %s'
                                 % arities)
                return arities[0]
        self.fail('MadSpinInterface.get_pdir not found')

    def _call_sites(self):
        """[(method name, line, number of targets)] for every ``... = self.get_pdir(...)``"""
        klass, ast = self._interface_class()
        out = []
        for meth in klass.body:
            if not isinstance(meth, ast.FunctionDef):
                continue
            for node in ast.walk(meth):
                if not isinstance(node, ast.Assign):
                    continue
                call = node.value
                if not isinstance(call, ast.Call):
                    continue
                fct = call.func
                if not (isinstance(fct, ast.Attribute) and fct.attr == 'get_pdir'):
                    continue
                target = node.targets[0]
                nb = len(target.elts) if isinstance(target, ast.Tuple) else 1
                out.append((meth.name, node.lineno, nb))
        return out

    def test_get_pdir_return_arity(self):
        """the arity the call sites are checked against -- a bump here is the
        signal to update them all"""
        self.assertEqual(self._return_arity(), 5)

    def test_every_call_site_matches(self):
        """every ``self.get_pdir(...)`` unpack in MadSpinInterface agrees with
        what get_pdir actually returns"""
        expected = self._return_arity()
        sites = self._call_sites()
        # the sweep is worthless if the AST walk found nothing. Three remain:
        # _frame_boost, get_density and get_iden (get_inter_value/get_nhel were
        # dead code and have been removed).
        self.assertTrue(len(sites) >= 3, 'no get_pdir call site found')
        bad = ['%s (line %s) unpacks %s' % (name, line, nb)
               for name, line, nb in sites
               if nb != expected and name not in self.KNOWN_PENDING]
        self.assertEqual(bad, [],
                         'get_pdir returns %s value(s) but: %s'
                         % (expected, ', '.join(bad)))

    def test_dead_f2py_cluster_stays_removed(self):
        """``get_inter_value``/``get_nhel``/``get_mymod`` were the last users of
        the pre-e16ac171b single-module f2py layout and had no caller left; they
        are gone, and so are the caches only they touched."""
        klass, ast = self._interface_class()
        methods = set(m.name for m in klass.body
                      if isinstance(m, ast.FunctionDef))
        for name in ('get_inter_value', 'get_nhel', 'get_mymod'):
            self.assertNotIn(name, methods)
        for cache in ('all_amp', 'all_jamp', 'all_inter', 'all_matrix',
                      'all_nhel'):
            self.assertFalse(
                hasattr(interface_madspin.MadSpinInterface, cache),
                '%s should not come back as a class attribute' % cache)
        with open(self.SOURCE) as fsock:
            source = fsock.read()
        for cache in ('all_amp', 'all_jamp', 'all_inter', 'all_matrix',
                      'all_nhel'):
            self.assertNotIn('self.%s' % cache, source)


class TestDecayedEventsPath(unittest.TestCase):
    """The legacy (madspin_v1) decay writes its events to an intermediate
    ``decayed_events.lhe`` and the interface then gzips that file into
    ``<events>_decayed.lhe.gz``. Writer and reader used to spell the directory
    out separately -- ``decay_all_events.decaying_events`` used ``path_me``,
    ``MadSpinInterface.do_launch``/``run_from_pickle`` used ``curr_dir`` -- and
    disagreed whenever those two differ.

    They differ exactly when ``ms_dir`` is set *and* ``curr_dir`` is not the
    ms_dir. That is not exotic: ``post_set_ms_dir`` points ``curr_dir`` at the
    ms_dir, so a card that says ``set ms_dir`` and *then* imports the event file
    gets ``curr_dir`` pointed back at the event file's directory by
    ``do_import``. The whole (expensive) decay then completes -- correct
    branching ratio, all events written, correct <init> -- and the run dies on
    the very last step with FileNotFoundError on a file that is sitting in the
    ms_dir. Without ``ms_dir`` the two coincide by construction (``path_me`` is
    *defined* as ``realpath(curr_dir)``), which is why this never showed up in
    the common case.

    ``curr_dir`` is the correct end: it is the run's output directory, whereas
    ``path_me`` means "where the matrix-element directories live" everywhere
    else it is used, and under ``ms_dir`` it is a directory built once and
    reused by later runs.

    These tests pin the two ends *together* rather than each to a literal, so a
    future edit to either side cannot re-open the gap without failing here. No
    MadEvent round trip is needed: the disagreement is entirely about paths.
    """

    NAME = 'decayed_events.lhe'

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp(prefix='ms_decayed_path_')
        self.evt_dir = pjoin(self.tmpdir, 'events')
        self.ms_dir = pjoin(self.tmpdir, 'gridpack')
        os.makedirs(self.evt_dir)
        os.makedirs(self.ms_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    class _Interface(object):
        """Stands in for the live MadSpinInterface: the accessor only needs its
        ``options``."""
        def __init__(self, options):
            self.options = options

    def _options(self, ms_dir=None, curr_dir=None):
        """A real MadSpinOptions, driven the way a card drives it -- so the
        ``set ms_dir`` -> ``import`` interaction that creates the mismatch is
        reproduced by the production code, not imitated here."""
        options = interface_madspin.MadSpinOptions()
        if ms_dir:
            options['ms_dir'] = ms_dir          # post_set_ms_dir moves curr_dir
        if curr_dir:
            options['curr_dir'] = curr_dir      # ...and do_import moves it back
        return options

    def _writer(self, options):
        """A ``decay_all_events`` with only what the accessor touches. Built
        without ``__init__`` on purpose: constructing the real thing needs a
        banner, a model and a compiled matrix element, none of which has any say
        in where the output goes."""
        writer = object.__new__(madspin.decay_all_events)
        writer.options = options
        writer.mscmd = self._Interface(options)
        # path_me exactly as decay_all_events.__init__ computes it
        writer.path_me = os.path.realpath(options['curr_dir'])
        if options['ms_dir']:
            writer.path_me = os.path.realpath(options['ms_dir'])
        return writer

    @staticmethod
    def _reader_path(writer):
        """What the interface gzips. Kept as a named indirection so it is
        obvious that ``test_both_gzip_call_sites_use_the_accessor`` is what
        makes this stand for the real read sites."""
        return writer.decayed_events_path

    # ------------------------------------------------------------------
    # the two ends agree
    # ------------------------------------------------------------------

    def test_writer_and_reader_agree_without_ms_dir(self):
        writer = self._writer(self._options(curr_dir=self.evt_dir))
        self.assertEqual(os.path.realpath(writer.decayed_events_path),
                         os.path.realpath(self._reader_path(writer)))
        self.assertEqual(os.path.realpath(os.path.dirname(
                             writer.decayed_events_path)),
                         os.path.realpath(self.evt_dir))

    def test_writer_and_reader_agree_with_ms_dir(self):
        """The regression: ms_dir set, curr_dir left pointing at the event file
        (a card that imports after ``set ms_dir``)."""
        options = self._options(ms_dir=self.ms_dir, curr_dir=self.evt_dir)
        # the setup really is the mismatching one
        self.assertNotEqual(os.path.realpath(options['curr_dir']),
                            os.path.realpath(options['ms_dir']))
        writer = self._writer(options)
        self.assertEqual(os.path.realpath(writer.decayed_events_path),
                         os.path.realpath(self._reader_path(writer)))

    def test_writer_and_reader_agree_when_ms_dir_is_curr_dir(self):
        """The historically working case -- a card that sets ms_dir and lets
        post_set_ms_dir carry curr_dir with it -- must stay working."""
        options = self._options(ms_dir=self.ms_dir)
        self.assertEqual(os.path.realpath(options['curr_dir']),
                         os.path.realpath(self.ms_dir))
        writer = self._writer(options)
        self.assertEqual(os.path.realpath(writer.decayed_events_path),
                         os.path.realpath(self._reader_path(writer)))
        self.assertEqual(os.path.realpath(os.path.dirname(
                             writer.decayed_events_path)),
                         os.path.realpath(self.ms_dir))

    # ------------------------------------------------------------------
    # ...and agree on the *right* directory
    # ------------------------------------------------------------------

    def test_the_output_goes_to_curr_dir_and_not_into_the_ms_dir(self):
        """Fixing this at the other end -- teaching the reader to look in
        path_me -- would also have silenced the traceback, and would have been
        wrong: it would leave per-run event output inside a gridpack directory
        that later runs reuse and may share."""
        writer = self._writer(self._options(ms_dir=self.ms_dir,
                                            curr_dir=self.evt_dir))
        self.assertEqual(os.path.realpath(os.path.dirname(
                             writer.decayed_events_path)),
                         os.path.realpath(self.evt_dir))
        self.assertNotEqual(os.path.realpath(writer.decayed_events_path),
                            os.path.realpath(pjoin(writer.path_me, self.NAME)))

    def test_without_ms_dir_path_me_and_curr_dir_still_coincide(self):
        """The no-ms_dir case is the one most likely to break if the fix is made
        at the wrong end, so pin that the file lands where it always did."""
        writer = self._writer(self._options(curr_dir=self.evt_dir))
        self.assertEqual(os.path.realpath(writer.decayed_events_path),
                         os.path.realpath(pjoin(writer.path_me, self.NAME)))

    # ------------------------------------------------------------------
    # the accessor reads the live run, not the pickled one
    # ------------------------------------------------------------------

    def test_the_path_follows_the_live_interface_not_the_stored_options(self):
        """Under ms_dir the writer is restored from ``madspin.pkl``, so its own
        ``options`` are those of the run that *built* the gridpack -- including
        that run's curr_dir. ``run_from_pickle`` re-points ``mscmd`` at the live
        interface, so the accessor must read the location from there."""
        stale_dir = pjoin(self.tmpdir, 'the_run_that_built_the_gridpack')
        os.makedirs(stale_dir)
        writer = self._writer(self._options(ms_dir=self.ms_dir,
                                            curr_dir=stale_dir))
        # what run_from_pickle does: hand the restored object the live interface
        live = self._options(ms_dir=self.ms_dir, curr_dir=self.evt_dir)
        writer.mscmd = self._Interface(live)
        self.assertEqual(os.path.realpath(os.path.dirname(
                             writer.decayed_events_path)),
                         os.path.realpath(self.evt_dir))
        self.assertNotIn('the_run_that_built_the_gridpack',
                         writer.decayed_events_path)

    # ------------------------------------------------------------------
    # a real write/read round trip
    # ------------------------------------------------------------------

    def _round_trip(self, options):
        """Write at the writer's path, gzip from the reader's path, exactly as
        decaying_events and do_launch do."""
        writer = self._writer(options)
        with open(writer.decayed_events_path, 'w') as fsock:
            fsock.write('<LesHouchesEvents version="1.0">\n'
                        '</LesHouchesEvents>\n')
        out = pjoin(self.tmpdir, 'events_decayed.lhe')
        misc.gzip(self._reader_path(writer), stdout=out)
        return out + '.gz'

    def test_round_trip_without_ms_dir(self):
        import gzip as gziplib
        out = self._round_trip(self._options(curr_dir=self.evt_dir))
        self.assertTrue(os.path.exists(out))
        self.assertIn('LesHouchesEvents', gziplib.open(out, 'rt').read())

    def test_round_trip_with_ms_dir(self):
        """On the buggy code this raised FileNotFoundError -- after the whole
        decay had already been done."""
        import gzip as gziplib
        out = self._round_trip(self._options(ms_dir=self.ms_dir,
                                             curr_dir=self.evt_dir))
        self.assertTrue(os.path.exists(out))
        self.assertIn('LesHouchesEvents', gziplib.open(out, 'rt').read())

    # ------------------------------------------------------------------
    # nobody rebuilds the path by hand any more
    # ------------------------------------------------------------------

    def test_the_writer_opens_the_accessor(self):
        source = inspect.getsource(madspin.decay_all_events.decaying_events)
        self.assertIn('self.decayed_events_path', source)
        self.assertNotIn(self.NAME, source)

    def test_both_gzip_call_sites_use_the_accessor(self):
        """This is what lets the round-trip tests above stand for the real read
        sites: neither of them may rebuild the path from an option.

        Read from the module file rather than through ``inspect.getsource`` on
        the methods: ``do_launch`` is wrapped by ``misc.mute_logger`` and
        getsource returns the decorator's body."""
        with open(interface_madspin.__file__.replace('.pyc', '.py')) as fsock:
            source = fsock.read()
        self.assertEqual(
            source.count('misc.gzip(generate_all.decayed_events_path'), 2,
            'do_launch and run_from_pickle must both ask the writer where the '
            'decayed events are')
        self.assertNotIn("pjoin(self.options['curr_dir'],'%s')" % self.NAME,
                         source)

    def test_the_accessor_is_not_derived_from_path_me(self):
        """path_me is the matrix-element directory; the guard is here because
        making the traceback go away by pointing the *reader* at path_me is the
        tempting wrong fix."""
        source = inspect.getsource(
            madspin.decay_all_events.decayed_events_path.fget)
        code = source.split('"""')[-1]
        self.assertNotIn('path_me', code)
        self.assertIn("curr_dir", code)


class TestMadSpinCardArchive(unittest.TestCase):
    """MadSpin keeps a copy of the card it ran next to the events it produced
    (``madspin_card_for_<events>.dat``). That copy is the record of what was
    actually run, and its absence is only noticed much later, by whoever tries
    to reproduce the result.

    The source used to be ``pjoin(self.options['curr_dir'], 'Cards',
    'madspin_card.dat')``, and ``curr_dir`` does not mean that. It is where the
    run's *output* goes (that is what #365 pinned it to for the decayed
    events), and ``MadSpinOptions.post_set_ms_dir`` re-points it at the
    gridpack. ``do_import`` points it back at the event file, so the card
    *ordering* decided whether the archiving worked:

      set ms_dir ... ; import events   -> curr_dir is the process dir  -> worked
      import events ; set ms_dir ...   -> curr_dir is the ms_dir       -> silent

    and the second is the ordering MadEvent always produces, because
    ``do_decay_events`` imports the event file in the ``MadSpinInterface``
    constructor and only then runs the card. So *every* MadEvent run whose
    madspin_card.dat says ``set ms_dir`` lost its card, with no warning: the
    guard was ``if os.path.exists(ms_card_path)``, and MadSpin never creates a
    ``Cards`` directory inside an ms_dir, so the whole block was skipped.

    The fix is to stop deriving the card from an output directory at all and
    ask which card was *run* -- ``MadSpinInterface.madspin_card_path``. These
    tests drive the real ``MadSpinOptions`` and the real ``do_import``, so the
    ``post_set_ms_dir`` -> ``do_import`` interaction that creates the mismatch
    is produced by production code rather than imitated here, and they pin the
    two orderings *to each other* so neither end can drift again.
    """

    class _Interface(interface_madspin.MadSpinInterface):
        """A MadSpinInterface carrying only the state the methods under test
        touch. Subclassed, not faked: ``do_import``, ``import_command_file``,
        ``madspin_card_path`` and ``_archive_madspin_card`` are the production
        ones. ``__init__`` is skipped because building the real interface pulls
        in a MasterCmd and a model, neither of which has any say in which card
        gets archived."""

        def __init__(self, options):
            self.options = options
            self.ms_card_path = None
            self.event_base_dir = None
            self.mother = None
            self.child = None
            self.stored_line = None
            self.history = []
            self.inputfile = None
            self.use_rawinput = False

    CARD = '# the card this run actually used\nset spinmode madspin_v1\n'

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp(prefix='ms_card_archive_')
        self.proc = pjoin(self.tmpdir, 'PROC')
        self.run_dir = pjoin(self.proc, 'Events', 'run_01')
        self.ms_dir = pjoin(self.tmpdir, 'gridpack')
        os.makedirs(pjoin(self.proc, 'Cards'))
        os.makedirs(self.run_dir)
        os.makedirs(self.ms_dir)
        self.card = pjoin(self.proc, 'Cards', 'madspin_card.dat')
        with open(self.card, 'w') as fsock:
            fsock.write(self.CARD)
        # the event file need not exist: do_import decides the directories
        # before it looks at the file (see _import_events)
        self.events = pjoin(self.run_dir, 'unweighted_events.lhe')
        self.decayed = pjoin(self.run_dir, 'unweighted_events_decayed.lhe')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # helpers -- everything below goes through production code
    # ------------------------------------------------------------------

    def _interface(self):
        return self._Interface(interface_madspin.MadSpinOptions())

    def _import_events(self, iface):
        """Run the *real* ``do_import`` for its directory bookkeeping.

        It sets ``curr_dir``/``event_base_dir`` in its first few lines and only
        afterwards checks that the file is there, so pointing it at a file that
        does not exist runs exactly the part under test and stops before the
        banner and model reading that would need a whole MadEvent output."""
        self.assertRaises(iface.InvalidCmd, iface.do_import, self.events)

    def _set_ms_dir(self, iface):
        """`set ms_dir` as a card does it -- post_set_ms_dir carries curr_dir
        along, which is the whole reason the orderings differ."""
        iface.options['ms_dir'] = self.ms_dir

    def _ordering_madevent(self):
        """import first, ms_dir second: the constructor imported the events and
        the card then says `set ms_dir`. This is what MadEvent always does."""
        iface = self._interface()
        self._import_events(iface)
        self._set_ms_dir(iface)
        return iface

    def _ordering_card_first(self):
        """`set ms_dir` first, import second -- the ordering that happened to
        work, and which must go on working."""
        iface = self._interface()
        self._set_ms_dir(iface)
        self._import_events(iface)
        return iface

    # ------------------------------------------------------------------
    # the interaction that made this invisible really is there
    # ------------------------------------------------------------------

    def test_the_two_orderings_really_do_disagree_about_curr_dir(self):
        """The premise of everything below, asserted rather than assumed.

        It doubles as the guard against the other tempting wrong fix: making
        ``post_set_ms_dir`` stop moving ``curr_dir``. That would silence this
        bug and break the decayed-events path of #365, which needs curr_dir to
        follow ms_dir when the card sets no event file after it."""
        after = self._ordering_madevent()
        before = self._ordering_card_first()
        self.assertEqual(os.path.realpath(after.options['curr_dir']),
                         os.path.realpath(self.ms_dir))
        self.assertEqual(os.path.realpath(before.options['curr_dir']),
                         os.path.realpath(self.proc))
        # ...and the old expression is exactly what that breaks
        self.assertFalse(os.path.exists(
            pjoin(after.options['curr_dir'], 'Cards', 'madspin_card.dat')))

    # ------------------------------------------------------------------
    # the card is found whatever the ordering
    # ------------------------------------------------------------------

    def test_the_card_is_found_in_the_madevent_ordering(self):
        """The regression: this returned a path inside the gridpack, which does
        not exist, so nothing was archived and nothing was said."""
        iface = self._ordering_madevent()
        self.assertEqual(os.path.realpath(iface.madspin_card_path),
                         os.path.realpath(self.card))

    def test_the_card_is_found_in_the_card_first_ordering(self):
        iface = self._ordering_card_first()
        self.assertEqual(os.path.realpath(iface.madspin_card_path),
                         os.path.realpath(self.card))

    def test_the_ordering_cannot_change_the_answer(self):
        """The pin: the two orderings are tied to each other, not each to a
        literal, so no future edit can re-open the gap on one side only."""
        self.assertEqual(
            os.path.realpath(self._ordering_madevent().madspin_card_path),
            os.path.realpath(self._ordering_card_first().madspin_card_path))

    def test_without_ms_dir_the_card_is_still_found(self):
        """The case that breaks if the fix is made at the wrong end -- by far
        the most common way MadSpin is run."""
        iface = self._interface()
        self._import_events(iface)
        self.assertEqual(iface.options['ms_dir'], '')
        self.assertEqual(os.path.realpath(iface.madspin_card_path),
                         os.path.realpath(self.card))

    # ------------------------------------------------------------------
    # ...and it is the card that was *run*
    # ------------------------------------------------------------------

    def test_import_command_file_records_the_card_it_runs(self):
        """``import_command_file`` is how every driver hands MadSpin a card,
        and the path it is given is the only thing that knows where the card
        lives. Run the real method on an empty card so nothing is executed."""
        elsewhere = pjoin(self.tmpdir, 'my_own_card.dat')
        open(elsewhere, 'w').close()
        iface = self._interface()
        iface.import_command_file(elsewhere)
        self.assertEqual(iface.ms_card_path, os.path.realpath(elsewhere))

    def test_a_card_from_elsewhere_wins_over_the_process_directory(self):
        """MadEvent passes ``<me_dir>/Cards/madspin_card.dat`` so the two agree
        there, but a card handed in from anywhere else is still *the* card that
        was run, and is what must be archived."""
        elsewhere = pjoin(self.tmpdir, 'my_own_card.dat')
        with open(elsewhere, 'w') as fsock:
            fsock.write('# a card that lives somewhere else\n')
        iface = self._ordering_madevent()
        iface.ms_card_path = os.path.realpath(elsewhere)
        self.assertEqual(iface.madspin_card_path, os.path.realpath(elsewhere))

    def test_no_card_means_no_archive_rather_than_a_wrong_one(self):
        """An interactive session types its commands; there is no file to keep.
        Returning None (and archiving nothing) is the honest answer -- better
        than reaching for whatever madspin_card.dat happens to be nearby."""
        iface = self._interface()
        self._import_events(iface)
        os.remove(self.card)
        self.assertIsNone(iface.madspin_card_path)
        self.assertIsNone(iface._archive_madspin_card(self.decayed))
        self.assertEqual([f for f in os.listdir(self.run_dir)
                          if f.startswith('madspin_card_for_')], [])

    # ------------------------------------------------------------------
    # a real archiving round trip
    # ------------------------------------------------------------------

    def test_the_archive_holds_the_card_that_was_run(self):
        """What the user goes looking for months later: the file is there, it
        is named after the events, and its content is the card."""
        iface = self._ordering_madevent()
        written = iface._archive_madspin_card(self.decayed)
        self.assertEqual(
            os.path.realpath(written),
            os.path.realpath(pjoin(
                self.run_dir,
                'madspin_card_for_unweighted_events_decayed.dat')))
        self.assertTrue(os.path.exists(written))
        self.assertEqual(open(written).read(), self.CARD)

    def test_both_orderings_archive_the_same_bytes(self):
        first = self._ordering_madevent()._archive_madspin_card(self.decayed)
        second = self._ordering_card_first()._archive_madspin_card(self.decayed)
        self.assertNotEqual(first, second)   # the counter kept them apart
        self.assertEqual(open(first).read(), open(second).read())

    def test_a_second_run_does_not_overwrite_the_first(self):
        """Rerunning against an existing ms_dir writes a new event file into the
        same directory; its card must not silently replace the earlier one."""
        iface = self._ordering_madevent()
        first = iface._archive_madspin_card(self.decayed)
        with open(self.card, 'w') as fsock:
            fsock.write('# a different card, second run\n')
        second = iface._archive_madspin_card(self.decayed)
        self.assertTrue(second.endswith(
            'madspin_card_for_unweighted_events_decayed_1.dat'))
        self.assertEqual(open(first).read(), self.CARD)
        self.assertEqual(open(second).read(), '# a different card, second run\n')

    def test_the_archive_goes_inside_RunMaterial_when_there_is_one(self):
        """aMC@NLO runs keep their cards inside RunMaterial.tar.gz; the copy
        must end up in the tarball, not loose beside it."""
        os.makedirs(pjoin(self.run_dir, 'RunMaterial'))
        misc.call(['tar', '-czpf', 'RunMaterial.tar.gz', 'RunMaterial'],
                  cwd=self.run_dir)
        shutil.rmtree(pjoin(self.run_dir, 'RunMaterial'))
        iface = self._ordering_madevent()
        iface._archive_madspin_card(self.decayed)
        self.assertFalse(os.path.exists(pjoin(
            self.run_dir, 'madspin_card_for_unweighted_events_decayed.dat')))
        misc.call(['tar', '-xzpf', 'RunMaterial.tar.gz'], cwd=self.run_dir)
        inside = pjoin(self.run_dir, 'RunMaterial',
                       'madspin_card_for_unweighted_events_decayed.dat')
        self.assertTrue(os.path.exists(inside))
        self.assertEqual(open(inside).read(), self.CARD)

    # ------------------------------------------------------------------
    # nobody rebuilds the source by hand any more
    # ------------------------------------------------------------------

    def test_every_call_site_goes_through_the_helper(self):
        """do_launch archived the card, run_from_pickle -- the path every rerun
        against an existing ms_dir takes -- did not archive it at all. Both go
        through one helper now, so they cannot disagree about what to keep or
        about whether to keep it.

        Read the module file rather than using ``inspect.getsource`` on the
        methods: ``do_launch`` is wrapped by ``misc.mute_logger`` and getsource
        returns the decorator's body."""
        with open(interface_madspin.__file__.replace('.pyc', '.py')) as fsock:
            source = fsock.read()
        self.assertEqual(
            source.count('self._archive_madspin_card(decayed_evt_file)'), 2,
            'do_launch and run_from_pickle must both archive the card')
        self.assertNotIn(
            "pjoin(self.options['curr_dir'],'Cards','madspin_card.dat')",
            source)

    def test_the_accessor_does_not_look_at_curr_dir(self):
        """curr_dir is an output directory. Rebuilding the card's location from
        it is the mistake being fixed, and it is the tempting one because it
        works in every setup that has no ms_dir."""
        source = inspect.getsource(
            interface_madspin.MadSpinInterface.madspin_card_path.fget)
        code = source.split('"""')[-1]
        self.assertNotIn('curr_dir', code)
        self.assertIn('ms_card_path', code)
        self.assertIn('event_base_dir', code)

    def test_do_import_keeps_the_directory_under_its_own_name(self):
        """``event_base_dir`` exists so that the answer survives a later
        ``set ms_dir``; if do_import ever stopped setting it the fallback would
        quietly go back to finding nothing."""
        source = inspect.getsource(interface_madspin.MadSpinInterface.do_import)
        self.assertIn('self.event_base_dir', source)


class TestZeroDensityGuard(unittest.TestCase):
    """The guards that turn a MadSpin accept/reject which can never accept into
    an immediate, named failure instead of an unbounded retry loop.

    Context: in the density spin modes every trial weight is built from the
    production spin-density matrix. A rho_prod that is identically zero makes
    every weight zero (offshell) or NaN (PA/onshell, which divide by its trace),
    so nothing is ever accepted, the decay pools are drained and regenerated for
    ever and MadSpin never writes an event. These tests pin both that the guards
    fire on that state and -- the part that matters just as much -- that they
    cannot fire on a run that is merely inefficient.
    """

    class _Trace(object):
        def __init__(self, value):
            self.real = value

    class _Density(object):
        """The bit of MadSpin.decay.DensityMatrix the guard touches."""
        def __init__(self, trace):
            self._trace = trace
        def trace(self):
            return TestZeroDensityGuard._Trace(self._trace)

    class _Event(object):
        def get_tag_and_order(self):
            return ((2, -2), (24, -24)), None

    def _stub(self, me_prod=1.0, nb_sigma=0.):
        interface = interface_madspin.MadSpinInterface
        class Stub(object):
            _check_production_density = interface._check_production_density
            _raise_degenerate_weight = interface._raise_degenerate_weight
            _dead_trial = interface._dead_trial
            _combine_maxwgt = interface._combine_maxwgt
            def calculate_matrix_element(self, event):
                if me_prod is None:
                    raise RuntimeError('no matrix element here')
                return me_prod
        stub = Stub()
        stub.options = {'nb_sigma': nb_sigma}
        return stub

    # ---------------- the production density check ----------------

    def test_a_healthy_density_is_accepted_and_its_trace_returned(self):
        """The common case must be a pure pass-through: no exception, and the
        trace handed back so the caller does not recompute it."""
        stub = self._stub()
        self.assertEqual(stub._check_production_density(self._Event(),
                                                        self._Density(3.5)),
                         3.5)

    def test_a_tiny_but_positive_trace_is_healthy(self):
        """A small weight is a slow run, not a broken one: the guard keys on
        zero, never on smallness."""
        stub = self._stub()
        self.assertEqual(stub._check_production_density(self._Event(),
                                                        self._Density(1e-300)),
                         1e-300)

    def test_zero_trace_raises_and_names_the_cause(self):
        stub = self._stub(me_prod=1.7e-3)
        try:
            stub._check_production_density(self._Event(), self._Density(0.0),
                                           'joint accept/reject')
        except interface_madspin.MadSpinDegenerateWeight as error:
            msg = str(error)
        else:
            self.fail('a zero production density matrix must raise')
        # what happened
        self.assertIn('production spin-density matrix', msg)
        self.assertIn('identically zero', msg)
        self.assertIn('EVERY trial rejected', msg)
        self.assertIn('joint accept/reject', msg)
        # why it is not just an unlucky phase-space point
        self.assertIn('0.0017', msg)
        # the plausible causes
        self.assertIn('polarised', msg)
        self.assertIn('ALLOW_HEL', msg)
        self.assertIn('ms_dir', msg)
        self.assertIn('beampol', msg)

    def test_a_nan_trace_raises_too(self):
        """PA/onshell divide by the trace, so a broken rho shows up as NaN
        rather than 0; both are 'can never accept'."""
        stub = self._stub()
        self.assertRaises(interface_madspin.MadSpinDegenerateWeight,
                          stub._check_production_density,
                          self._Event(), self._Density(float('nan')))

    def test_a_vanishing_full_me_is_reported_as_its_own_case(self):
        """Tr(rho)=0 *and* |M_prod|^2=0 is a different diagnosis -- the event
        carries no matrix element at all -- and must not be blamed on the
        helicity basis."""
        stub = self._stub(me_prod=0.0)
        try:
            stub._check_production_density(self._Event(), self._Density(0.0))
        except interface_madspin.MadSpinDegenerateWeight as error:
            msg = str(error)
        else:
            self.fail('a zero production density matrix must raise')
        self.assertIn('vanishes as well', msg)
        self.assertNotIn('which is NOT zero', ' '.join(msg.split()))

    def test_the_cross_check_may_fail_without_hiding_the_error(self):
        stub = self._stub(me_prod=None)
        self.assertRaises(interface_madspin.MadSpinDegenerateWeight,
                          stub._check_production_density,
                          self._Event(), self._Density(0.0))

    # ---------------- the bounded dead-trial backstop ----------------

    def test_a_positive_weight_resets_the_dead_trial_counter(self):
        stub = self._stub()
        self.assertEqual(stub._dead_trial(17, 1e-12, 'x'), 0)

    def test_dead_trials_accumulate_and_raise_at_the_bound(self):
        stub = self._stub()
        limit = interface_madspin.MS_MAX_DEAD_TRIALS
        self.assertEqual(stub._dead_trial(0, 0.0, 'x'), 1)
        self.assertEqual(stub._dead_trial(limit - 2, 0.0, 'x'), limit - 1)
        try:
            stub._dead_trial(limit - 1, 0.0, 'the joint accept/reject')
        except interface_madspin.MadSpinDegenerateWeight as error:
            msg = str(error)
        else:
            self.fail('an unbounded run of dead trials must raise')
        self.assertIn('consecutive trials', msg)
        self.assertIn('the joint accept/reject', msg)
        self.assertIn('not a low acceptance', msg)

    def test_negative_and_nan_weights_count_as_dead(self):
        stub = self._stub()
        for wgt in (0.0, -1.0, float('nan'), float('inf')):
            self.assertEqual(stub._dead_trial(0, wgt, 'x'), 1)

    def test_an_atrocious_but_correct_efficiency_never_trips_the_bound(self):
        """The separation that makes this guard safe. A 1-in-100000 acceptance
        is a slow *correct* run: it burns a huge number of trials, but each one
        computes a small POSITIVE weight and is merely rejected by the
        `random()*maxwgt < wgt` test -- which this counter never sees. Only a
        weight that is itself zero/NaN is counted. Ten times the bound of such
        trials, with one in ten of them a legitimate zero (an infeasible
        virtuality), and nothing raises."""
        stub = self._stub()
        counter = 0
        for i in range(10 * interface_madspin.MS_MAX_DEAD_TRIALS):
            wgt = 0.0 if i % 10 == 0 else 1e-9
            counter = stub._dead_trial(counter, wgt, 'x')
            self.assertTrue(counter < interface_madspin.MS_MAX_DEAD_TRIALS)

    # ---------------- the max-weight bound ----------------

    def test_a_healthy_probe_still_gives_a_bound(self):
        stub = self._stub()
        self.assertTrue(stub._combine_maxwgt([1.0, 3.0, 2.0, 0.0]) > 0)

    def test_an_all_zero_probe_is_refused(self):
        """A bound of 0 rejects every trial for ever: `random()*0 < 0` is never
        true. Refuse it before the unweighting starts."""
        stub = self._stub()
        self.assertRaises(interface_madspin.MadSpinDegenerateWeight,
                          stub._combine_maxwgt, [0.0, 0.0, 0.0])

    def test_a_nan_in_the_probe_is_refused(self):
        """0/0 weights used to reach _combine_maxwgt and trip a bare
        `assert all_maxwgt[0] >= all_maxwgt[1], "ERROR: "` with an empty
        message."""
        stub = self._stub()
        self.assertRaises(interface_madspin.MadSpinDegenerateWeight,
                          stub._combine_maxwgt,
                          [float('nan'), float('nan'), 1.0])


class TestReusedMsDirBranchingRatio(unittest.TestCase):
    """Reusing an ``ms_dir`` must reproduce the branching ratio of the run that
    built it -- and a branching ratio that cannot be computed must stop the run
    rather than scale every event to zero.

    Context: ``ms_dir`` selects the gridpack decay-generation path, on which the
    partial width of a channel is measured once, while the gridpack is being
    built. A later run finds the ``decay_<pdg>_<i>`` directory already there,
    skips the whole build block and only *runs* the gridpack -- so nothing
    re-measures the width, the accumulator keeps its neutral value (0.0 under
    ``cumul``, the common case) and the branching ratio comes out exactly 0.
    That zero multiplies every event weight and the <init> cross-section, so
    the run completes, writes a well-formed LHE file of +/-0.0 and reports
    success.
    """

    _INIT_ONLY_LHE = (
        '<LesHouchesEvents version="1.0">\n'
        '<header>\n'
        '</header>\n'
        '<init>\n'
        '2212 2212 6.500000e+03 6.500000e+03 0 0 247000 247000 -4 1\n'
        ' 3.000000e+00 1.000000e-02 3.000000e+00 1\n'
        '</init>\n'
        '</LesHouchesEvents>\n'
    )

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp(prefix='ms_dir_reuse_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ---------------- the reuse path of generate_events ----------------

    class _Particle(object):
        def __init__(self, name):
            self._name = name
        def get_name(self):
            return self._name

    class _Model(object):
        def get_particle(self, pdg):
            return TestReusedMsDirBranchingRatio._Particle('t')

    def _prebuilt_decay_dir(self, stored_width=None, gridpack_cross=True):
        """An ``ms_dir`` holding one already-built decay directory, exactly as a
        previous run leaves it: a run.sh, the events its gridpack produces and
        (unless ``stored_width`` is None) the partial-width record."""
        import gzip
        decay_dir = pjoin(self.tmpdir, 'decay_6_0')
        os.makedirs(decay_dir)
        open(pjoin(decay_dir, 'run.sh'), 'w').write('#!/bin/sh\n')
        if gridpack_cross:
            with gzip.open(pjoin(decay_dir, 'events.lhe.gz'), 'wt') as fsock:
                fsock.write(self._INIT_ONLY_LHE)
        else:
            # a file EventFile cannot get a cross-section out of
            with gzip.open(pjoin(decay_dir, 'events.lhe.gz'), 'wt') as fsock:
                fsock.write('<LesHouchesEvents version="1.0">\n'
                            '</LesHouchesEvents>\n')
        if stored_width is not None:
            interface_madspin.MadSpinInterface._store_partial_width(
                decay_dir, stored_width)
        return decay_dir

    def _stub(self):
        interface = interface_madspin.MadSpinInterface
        class Stub(object):
            generate_events = interface.generate_events
            _store_partial_width = interface._store_partial_width
            _load_partial_width = interface._load_partial_width
            PARTIAL_WIDTH_FILE = interface.PARTIAL_WIDTH_FILE
            _DECAY_GROUP_TAG = interface._DECAY_GROUP_TAG
            _split_group_tag = interface._split_group_tag
            def _resolve_nb_core(self):
                return 1
            def _run_gridpack(self, cmd, cwd):
                # the gridpack's events are already on disk in these tests
                return 0, ''
        stub = Stub()
        stub.path_me = self.tmpdir
        stub.options = {'ms_dir': self.tmpdir, 'seed': 7}
        stub.seed = 7
        stub.model = self._Model()
        stub.list_branches = {'t': ['t > w+ b, w+ > all all']}
        return stub

    def test_reuse_recovers_the_partial_width_that_was_stored(self):
        """The regression itself: with the record in place the reuse path
        reports the width the building run measured -- not 0."""
        self._prebuilt_decay_dir(stored_width=1.4594692)
        stub = self._stub()
        out, width, channel_widths = stub.generate_events(
            6, 100, None, cumul=True, output_width=True)
        self.assertEqual(width, 1.4594692)
        self.assertEqual(channel_widths, {0: 1.4594692})
        self.assertEqual(list(out), [0])

    def test_reuse_without_the_record_falls_back_to_the_generated_events(self):
        """An ms_dir built by a version that stored nothing must still work:
        the cross-section of the events the gridpack just produced is the same
        quantity, measured on this run's sample."""
        self._prebuilt_decay_dir(stored_width=None)
        stub = self._stub()
        out, width, channel_widths = stub.generate_events(
            6, 100, None, cumul=True, output_width=True)
        self.assertEqual(width, 3.0)          # the <init> cross of the events
        self.assertEqual(channel_widths, {0: 3.0})

    def test_reuse_with_no_recoverable_width_raises(self):
        """Neither source available: fail, never fall back to a default. Any
        default here is a wrong branching ratio in a well-formed file."""
        self._prebuilt_decay_dir(stored_width=None, gridpack_cross=False)
        stub = self._stub()
        self.assertRaises(interface_madspin.MadSpinUnknownPartialWidth,
                          lambda: stub.generate_events(6, 100, None,
                                                       cumul=True,
                                                       output_width=True))

    def test_the_stored_width_survives_a_round_trip(self):
        decay_dir = self._prebuilt_decay_dir(stored_width=2.5e-3)
        self.assertEqual(
            interface_madspin.MadSpinInterface._load_partial_width(decay_dir),
            2.5e-3)

    def test_a_corrupt_record_falls_back_instead_of_propagating(self):
        """A truncated/garbage record must not become the branching ratio."""
        decay_dir = self._prebuilt_decay_dir(stored_width=1.0)
        open(pjoin(decay_dir,
                   interface_madspin.MadSpinInterface.PARTIAL_WIDTH_FILE),
             'w').write('not a number\n')
        stub = self._stub()
        out, width, channel_widths = stub.generate_events(
            6, 100, None, cumul=True, output_width=True)
        self.assertEqual(width, 3.0)

    def test_a_zero_record_is_not_trusted_either(self):
        """0 is exactly the value the bug produced; reading it back from disk
        must not resurrect it."""
        decay_dir = self._prebuilt_decay_dir(stored_width=0.0)
        stub = self._stub()
        out, width, channel_widths = stub.generate_events(
            6, 100, None, cumul=True, output_width=True)
        self.assertEqual(width, 3.0)

    def test_a_fresh_directory_is_untouched_by_the_reuse_branch(self):
        """The recovery only ever runs for a channel this run did not measure:
        a width already in ``channel_widths`` must never be added twice."""
        self._prebuilt_decay_dir(stored_width=1.5)
        stub = self._stub()
        out, width, channel_widths = stub.generate_events(
            6, 100, None, cumul=False, output_width=True)
        # cumul=False multiplies into a neutral 1.0, so a double fold would
        # square it
        self.assertEqual(width, 1.5)

    # ---------------- the branching-ratio guard ----------------

    def test_a_healthy_branching_ratio_passes_through(self):
        check = interface_madspin.MadSpinInterface._check_branching_ratio
        self.assertEqual(check(0.543), 0.543)

    def test_a_tiny_branching_ratio_is_healthy(self):
        """A rare decay is a small BR, not a broken one: the guard keys on zero
        and on non-finiteness, never on smallness."""
        check = interface_madspin.MadSpinInterface._check_branching_ratio
        self.assertEqual(check(1e-12), 1e-12)

    def test_a_zero_branching_ratio_raises_and_names_the_cause(self):
        check = interface_madspin.MadSpinInterface._check_branching_ratio
        try:
            check(0.0, {6: {'kind': 'simple'}})
        except interface_madspin.MadSpinZeroBranchingRatio as error:
            msg = str(error)
        else:
            self.fail('a zero branching ratio must raise')
        # what would have happened
        self.assertIn('every weight is zero', msg)
        self.assertIn('<init>', msg)
        # the plausible causes, the first of which is this bug
        self.assertIn('ms_dir', msg)
        self.assertIn('use_old_dir', msg)
        self.assertIn('param_card', msg)
        self.assertIn('cross_section', msg)
        # and what the run was actually doing
        self.assertIn('simple', msg)

    def test_a_non_finite_branching_ratio_raises(self):
        check = interface_madspin.MadSpinInterface._check_branching_ratio
        for bad in (float('nan'), float('inf'), -1.0):
            self.assertRaises(interface_madspin.MadSpinZeroBranchingRatio,
                              check, bad)

    def test_run_onshell_checks_before_it_decays(self):
        """The guard has to sit on the branching ratio *before* the events are
        written, not after; pin the call site so it cannot drift below the
        decay loop."""
        source = inspect.getsource(
            interface_madspin.MadSpinInterface.run_onshell)
        self.assertIn('_check_branching_ratio(br', source)
        self.assertLess(source.index('_check_branching_ratio(br'),
                        source.index('self.branching_ratio = br'))



class TestDensityContractionIsReal(unittest.TestCase):
    """The accept/reject weight of the density spin modes is a *real* number,
    and the code now says so.

    Background. Every weight is built from ``rho_dec . rho_prod``, a sum over
    the packed (h1,h2) index set of products of complex64 entries. That sum is
    real by construction -- the index set is closed under (h1,h2) -> (h2,h1),
    both matrices are hermitian, so each term appears together with its own
    complex conjugate -- and what survives it is float32 rounding. Measured on
    `p p > t t~` with both tops decayed (53655 contractions, joint unweighting):
    24% of them had a non-zero imaginary part, and the largest |Im|/|Re| was
    1.5e-7, i.e. float32 epsilon.

    The weight that reached the accept/reject was nevertheless a *complex*
    scalar with an exactly zero imaginary part, because the propagator
    denominator was built as ``complex(0, m*Gamma) * conj(...)`` -- a real
    number written in complex form -- and that type rode through the division.
    ``math.isfinite`` in ``_dead_trial`` then coerced it back and emitted
    "ComplexWarning: Casting complex values to real discards the imaginary
    part" on every run of every density mode using the joint scheme.
    """

    # ---------------- the reality check itself ----------------

    def test_a_negligible_imaginary_part_passes_through_as_the_real_part(self):
        """The physical case: rounding residue of an exact cancellation. The
        real part comes back unchanged and nothing is reported."""
        import numpy as np
        value = np.complex64(complex(43065.3125, -0.00048828125))  # measured
        with _CaptureCritical() as reported:
            got = interface_madspin.ms_density_real(value, 'a contraction')
        self.assertEqual(got, value.real)
        self.assertFalse(hasattr(got, 'imag') and got.imag)
        self.assertEqual(reported.messages, [])

    def test_an_exactly_real_value_passes_through(self):
        with _CaptureCritical() as reported:
            self.assertEqual(
                interface_madspin.ms_density_real(complex(2.5, 0.0), 'x'), 2.5)
        self.assertEqual(reported.messages, [])

    def test_a_plain_float_is_returned_unchanged(self):
        """The helper has to be safe on the paths whose weight is already real
        (the sequential stages take their own ``.real``)."""
        import numpy as np
        with _CaptureCritical() as reported:
            self.assertEqual(interface_madspin.ms_density_real(1.25, 'x'), 1.25)
            self.assertEqual(
                interface_madspin.ms_density_real(np.float32(0.5), 'x'), 0.5)
        self.assertEqual(reported.messages, [])

    def test_a_large_imaginary_part_is_reported_loudly(self):
        """The part that must not be silent. A contraction with a real
        imaginary part means a non-hermitian density matrix or two sides in
        different helicity bases -- a bug, not a tolerance to widen."""
        with _CaptureCritical() as reported:
            got = interface_madspin.ms_density_real(complex(1.0, 0.5),
                                                    'the test contraction')
        self.assertEqual(got, 1.0)
        self.assertEqual(len(reported.messages), 1)
        msg = reported.messages[0]
        self.assertIn('imaginary part', msg)
        self.assertIn('the test contraction', msg)
        self.assertIn('real by construction', msg)

    def test_a_reported_imaginary_part_does_not_abort_the_run(self):
        """Reported, not raised: the run still produces events (they are just
        not to be trusted), which is what the weight-identity and
        ``density_debug`` checks in this file do as well."""
        with _CaptureCritical():
            interface_madspin.ms_density_real(complex(1.0, 1.0), 'no-raise')

    def test_a_vanishing_real_part_with_an_imaginary_one_is_reported(self):
        """No ZeroDivisionError on the way to the message: a zero weight is
        exactly the state the dead-trial guards are watching for, so this path
        has to survive it."""
        with _CaptureCritical() as reported:
            got = interface_madspin.ms_density_real(complex(0.0, 1e-30), 'z')
        self.assertEqual(got, 0.0)
        self.assertEqual(len(reported.messages), 1)

    def test_the_tolerance_leaves_room_over_float32_rounding(self):
        """The measured worst case was 1.5e-7 (float32 epsilon). The bound has
        to sit far above that and far below anything that means something."""
        self.assertGreater(interface_madspin.MS_DENSITY_IMAG_TOL, 1e-5)
        self.assertLess(interface_madspin.MS_DENSITY_IMAG_TOL, 1e-1)
        with _CaptureCritical() as reported:
            for exponent in range(7, 12):
                interface_madspin.ms_density_real(complex(1.0, 10.0 ** -exponent),
                                                  'tol %d' % exponent)
        self.assertEqual(reported.messages, [])

    def test_each_site_is_reported_once(self):
        """A broken basis repeats on every trial of every event; one line, not
        millions."""
        with _CaptureCritical() as reported:
            for _ in range(50):
                interface_madspin.ms_density_real(complex(1.0, 1.0), 'one site')
        self.assertEqual(len(reported.messages), 1)

    # ---------------- the propagator denominator ----------------

    def test_the_real_propagator_denominator_is_bit_identical(self):
        """``prod_denominators`` used to be accumulated as
        ``complex(0, m*Gamma) * conj(complex(0, m*Gamma))``: the value |D|^2 of
        a real number written in complex form. Replacing it by ``mw * mw`` is
        what makes the weight real, and it is only a legitimate replacement if
        it changes no bit of any weight -- which is why the decayed output of a
        real run is byte-identical across the change. Pin that.

        ``mw ** 2`` is NOT the same thing: pow() re-rounds and differs from the
        product in the last bit for roughly one value in 700, which would move
        weights and so move the accepted events."""
        import struct
        bits = lambda x: struct.pack('<d', x)
        differs_under_pow = 0
        values = [173.0 * 1.5, 80.379 * 2.085, 91.1876 * 2.4952,
                  1e-30, 1e12, 4.7, 0.0, 1.0]
        import random as _random
        random_ = _random.Random(20250819)
        values += [random_.uniform(1e-12, 1e6) for _ in range(20000)]
        for mw in values:
            old = (complex(0, mw) * complex(0, mw).conjugate())
            self.assertEqual(old.imag, 0.0)
            self.assertEqual(bits(old.real), bits(mw * mw))
            if bits(old.real) != bits(mw ** 2):
                differs_under_pow += 1
        self.assertGreater(differs_under_pow, 0,
                           'if pow() ever becomes exact here the comment above '
                           'is stale, but mw*mw stays the safe spelling')

    def test_the_denominator_is_no_longer_built_as_a_complex(self):
        """The single ``complex(...)`` construction in the density weight was
        the only reason it was ever complex; keep it gone."""
        source = inspect.getsource(
            interface_madspin.MadSpinInterface.calculate_matrix_element_from_density)
        self.assertNotIn('complex(0,', source.replace(' ', ''))
        self.assertIn('mw * mw', source)
        self.assertIn('ms_density_real(me', source)

    # ---------------- what _dead_trial accepts ----------------

    def _dead_trial_stub(self):
        interface = interface_madspin.MadSpinInterface
        class Stub(object):
            _dead_trial = interface._dead_trial
            _raise_degenerate_weight = interface._raise_degenerate_weight
        return Stub()

    def test_dead_trial_verdicts_are_unchanged_by_the_weight_becoming_real(self):
        """The invariant the fix must not disturb: exactly the weights that
        reset the counter before still reset it, and exactly those that were
        counted as dead still are -- whether they arrive as a python float, as
        the numpy real scalar the density path now produces, or as the numpy
        complex scalar with a negligible imaginary part it used to produce."""
        import numpy as np
        import warnings
        stub = self._dead_trial_stub()
        alive = [1.0, 1e-12, 1e30, 3.5]
        dead = [0.0, -0.0, -1.0, float('nan'), float('inf'), -float('inf')]
        # the complex entries below coerce on purpose; that they do is the
        # subject of the two tests underneath, not noise this one has to print
        cm = warnings.catch_warnings()
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        warnings.simplefilter('ignore')
        for value in alive:
            for wgt in (value, np.float32(value), np.float64(value),
                        np.complex64(complex(value, 0.0)),
                        np.complex64(complex(value, 1e-9 * value))):
                self.assertEqual(stub._dead_trial(17, wgt, 'x'), 0,
                                 'a positive weight must reset: %r' % (wgt,))
        for value in dead:
            for wgt in (value, np.float32(value), np.float64(value),
                        np.complex64(complex(value, 0.0))):
                self.assertEqual(stub._dead_trial(0, wgt, 'x'), 1,
                                 'a dead weight must count: %r' % (wgt,))

    def test_a_python_complex_weight_would_be_counted_as_dead(self):
        """A sharp edge worth knowing about rather than discovering. A *numpy*
        complex scalar defines ``__float__`` and so survives ``math.isfinite``
        (with the ComplexWarning); a plain python ``complex`` does not, the
        ``except TypeError`` branch takes over, and the trial is scored dead --
        which after MS_MAX_DEAD_TRIALS in a row would abort a healthy run.

        It is unreachable today, and the fix makes it more so: the weight is
        built from numpy scalars, so a python complex never wins the mixed
        arithmetic. Pinned so that a future change which does hand one over is
        seen for what it is instead of surfacing as a mystery
        MadSpinDegenerateWeight."""
        stub = self._dead_trial_stub()
        self.assertEqual(stub._dead_trial(17, complex(1.0, 0.0), 'x'), 18)

    def test_dead_trial_does_not_coerce_the_weights_the_code_now_produces(self):
        """The warning the user saw, pinned from the other side: the real
        scalars the density path hands over must reach ``math.isfinite``
        without any complex-to-real cast at all."""
        import warnings
        import numpy as np
        stub = self._dead_trial_stub()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for wgt in (1.0, np.float32(2.5), np.float64(0.0), -1.0):
                stub._dead_trial(0, wgt, 'x')
        self.assertEqual([w for w in caught
                          if 'ComplexWarning' in w.category.__name__], [])

    def test_dead_trial_still_reports_a_complex_weight_rather_than_hiding_it(self):
        """``math.isfinite`` was kept on purpose over ``numpy.isfinite`` or an
        explicit ``.real``: if a complex weight is ever reintroduced upstream,
        this is the line that says so."""
        import warnings
        import numpy as np
        stub = self._dead_trial_stub()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            stub._dead_trial(0, np.complex64(complex(1.0, 0.0)), 'x')
        self.assertTrue([w for w in caught
                         if 'ComplexWarning' in w.category.__name__],
                        'a complex weight must not pass in silence')
        source = inspect.getsource(interface_madspin.MadSpinInterface._dead_trial)
        self.assertIn('math.isfinite(wgt)', source)


class TestFixedOrderGroupDecays(unittest.TestCase):
    """Every member of a ``fixed_order`` event group comes back decayed.

    ``Event.add_decays`` used to copy the *mapping* it is given
    (``dict(pdg_to_decay)``) but share the lists inside it, which the recursion
    then ``pop(0)``s. It looks like a defensive copy and is not one: the first
    production event it is applied to drained the caller's own lists, so every
    later one silently attached nothing and came back with the bare production
    particles.

    MadSpin applies one draw to several events in exactly the place where that
    matters -- ``_unweight_range`` attaches the born event's decays to the born
    event and then to each of its counter-events
    (``[full_evt] + [evt.add_decays(decays) for evt in counterevt]``, the 2017
    design of the option). So no counter-event had ever been decayed, in any
    spinmode; and under the pole approximation, where
    ``get_onshell_evt_and_wgt`` builds the event first and the caller then
    rebuilds it to fold in the reshuffling jacobian, not even the born event
    was.
    """

    # ---------------- fixtures ----------------

    PROD = """<event>
 4      1 +1.0000000e+00 1.00000000e+02 7.54677160e-03 1.02860750e-01
       21 -1    0    0  501  502 +0.00000000000e+00 +0.00000000000e+00 +%(E).11e  %(E).11e  0.00000000000e+00 0. 9.
       21 -1    0    0  503  501 +0.00000000000e+00 +0.00000000000e+00 -%(E).11e  %(E).11e  0.00000000000e+00 0. 9.
        6  1    1    2  503    0 +%(px).11e +0.00000000000e+00 +0.00000000000e+00  %(E).11e  1.73000000000e+02 0. 9.
       -6  1    1    2    0  502 -%(px).11e +0.00000000000e+00 +0.00000000000e+00  %(E).11e  1.73000000000e+02 0. 9.
</event>
"""

    DECAY = """<event>
 3      1 +1.0000000e+00 1.00000000e+02 7.54677160e-03 1.02860750e-01
     %(pdg)4d -1    0    0  %(c1)3d  %(c2)3d +0.00000000000e+00 +0.00000000000e+00 +0.00000000000e+00  1.73000000000e+02  1.73000000000e+02 0. 9.
     %(d1)4d  1    1    1  %(c1)3d    0 +0.00000000000e+00 +0.00000000000e+00 +%(E).11e  %(E).11e  0.00000000000e+00 0. 9.
     %(d2)4d  1    1    1    0  %(c2)3d +0.00000000000e+00 +0.00000000000e+00 -%(E).11e  %(E).11e  0.00000000000e+00 0. 9.
</event>
"""

    # 4 production particles + 2 extra per decayed top
    UNDECAYED = 4
    DECAYED = 8

    @classmethod
    def _production(cls, px):
        """g g > t t~, the tops back to back along x with the given |px|."""
        E = math.sqrt(173.0 ** 2 + px ** 2)
        return lhe_parser.Event(cls.PROD % dict(px=px, E=E))

    @classmethod
    def _decay(cls, pdg, d1, d2, c1, c2):
        """t > d1 d2 in the top rest frame, the two daughters massless."""
        return lhe_parser.Event(cls.DECAY % dict(pdg=pdg, d1=d1, d2=d2,
                                                 c1=c1, c2=c2, E=173.0 / 2))

    @classmethod
    def _decays(cls):
        return {6: [cls._decay(6, 5, 24, 601, 0)],
                -6: [cls._decay(-6, -5, -24, 0, 602)]}

    # ---------------- the contract, at the lhe_parser level ----------------

    def test_the_same_decays_can_be_attached_to_several_events(self):
        """The bug in one line: three production events, one decays dict."""
        decays = self._decays()
        for i in range(3):
            event = self._production(100.0 + i).add_decays(decays)
            self.assertEqual(len(event), self.DECAYED,
                             'event %d came back undecayed -- add_decays '
                             'drained the caller\'s lists' % i)

    def test_add_decays_leaves_the_callers_dict_untouched(self):
        decays = self._decays()
        before = dict((pdg, list(value)) for pdg, value in decays.items())
        self._production(100.0).add_decays(decays)
        self.assertEqual(sorted(decays), sorted(before))
        for pdg in before:
            self.assertEqual(decays[pdg], before[pdg],
                             'the %s list was consumed' % pdg)

    def test_the_decay_events_themselves_are_reusable(self):
        """Not just the lists: the decay *events* are attached by copy, so the
        second production event gets the same rest-frame decay boosted into its
        own frame rather than a mutated one."""
        decays = self._decays()
        text = str(decays[6][0])
        self._production(100.0).add_decays(decays)
        self._production(400.0).add_decays(decays)
        self.assertEqual(str(decays[6][0]), text)

    # ---------------- the same thing through _unweight_range ----------------

    class _Pool(object):
        """An inexhaustible decay pool for one channel."""
        cross = 1.0

        def __init__(self, maker):
            self.maker = maker

        def __next__(self):
            return self.maker()
        next = __next__

    class _Generate(object):
        def __init__(self, mode):
            self.mode = mode
            self.all_me = collections.defaultdict(dict)

    class _Output(object):
        def __init__(self):
            self.written = []

        def write_events(self, event):
            self.written.append(event)

    class _Options(dict):
        """The knobs _unweight_range reads that this stub does not care about
        are all falsy; spelling every one of them out would only hide which
        ones the test actually sets."""
        def __missing__(self, key):
            return False

    def _stub(self, spinmode, density_method):
        stub = interface_madspin.MadSpinInterface.__new__(
            interface_madspin.MadSpinInterface)
        stub.options = self._Options({'fixed_order': True,
                                      'spinmode': spinmode,
                                      'density_tolerance': 1e-2})
        stub.generate_all = self._Generate(
            'density' if density_method else 'onshell')
        stub.efficiency = 1.0
        stub.branching_ratio = 1.0
        stub._shard_tag = None
        stub._decay_groups = None
        # the matrix elements are not what is under test: a flat weight makes
        # the joint accept/reject take the first trial
        stub.calculate_matrix_element = lambda event, *a, **kw: 1.0
        stub.calculate_matrix_element_from_density = \
            lambda production, decays, decay_dict, cached=None: \
            (1.0, {'cached': True}, 1.0, 1.0, 1.0)
        return stub

    def _run_group(self, spinmode, density_method, nb_member=3):
        """Decay one event group of ``nb_member`` members and return the
        particle count of each member as written out."""
        stub = self._stub(spinmode, density_method)
        evt_decayfile = {
            6: {0: self._Pool(lambda: self._decay(6, 5, 24, 601, 0))},
            -6: {0: self._Pool(lambda: self._decay(-6, -5, -24, 0, 602))},
        }
        group = [self._production(100.0 + 0.1 * k) for k in range(nb_member)]
        output = self._Output()
        ctx = {'maxwgt': 1e-9,          # accept the first trial
               'maxwgts': [], 'sequential': False,
               'decay_dict': {6: (173.0, 1.5), -6: (173.0, 1.5)},
               'drop_prob_per_pdg': None, 'mixed_pdgs_set': set(),
               'density_method': density_method,
               'density_pole_approximation': stub._density_pole_approximation(),
               'density_needs_reshuffle':
                   stub._density_needs_reshuffle(density_method),
               'shard_nb_event': 1}
        stub._unweight_range(iter([group]), evt_decayfile, output, ctx)
        self.assertEqual(len(output.written), 1)
        return [len(event) for event in output.written[0]]

    def test_every_group_member_is_decayed_in_onshell_v1(self):
        """spinmode=onshell_v1: no density, no reshuffling."""
        self.assertEqual(self._run_group('onshell_v1', density_method=False),
                         [self.DECAYED] * 3)

    def test_every_group_member_is_decayed_in_onshell(self):
        """spinmode=onshell: density matrices, still no reshuffling -- the
        combination ``fixed_order`` is documented to support."""
        self.assertEqual(self._run_group('onshell', density_method=True),
                         [self.DECAYED] * 3)

    def test_a_lone_born_event_is_still_decayed(self):
        """A group of one is the degenerate case and must not regress."""
        self.assertEqual(
            self._run_group('onshell', density_method=True, nb_member=1),
            [self.DECAYED])


class TestFixedOrderReshufflingSpinmodesRefused(unittest.TestCase):
    """``fixed_order`` is refused in the spinmodes that reshuffle the
    production.

    An event group is decayed once and the born event's decays are attached to
    every member unchanged. PA and madspin/full then reshuffle the production
    onto the sampled virtualities -- but only the born member goes through that
    reshuffling, so its resonance sits at the sampled mass while the
    counter-events subtracting it stay onshell. That is a decayed-looking
    sample whose members disagree, which is worse than the undecayed output it
    replaced, so launch refuses instead.
    """

    def _stub(self, spinmode, fixed_order):
        stub = interface_madspin.MadSpinInterface.__new__(
            interface_madspin.MadSpinInterface)
        stub.options = {'spinmode': spinmode, 'fixed_order': fixed_order}
        return stub

    def _refuses(self, spinmode):
        stub = self._stub(spinmode, fixed_order=True)
        try:
            stub._check_fixed_order_spinmode(spinmode)
        except interface_madspin.MadSpinInterface.InvalidCmd as error:
            return str(error)
        self.fail('fixed_order + spinmode=%s was accepted' % spinmode)

    def test_pa_is_refused(self):
        message = self._refuses('PA')
        self.assertIn('fixed_order', message)
        self.assertIn('spinmode=PA', message)
        self.assertIn('onshell', message)   # names the way out

    def test_madspin_is_refused(self):
        self.assertIn('spinmode=madspin', self._refuses('madspin'))

    def test_the_supported_spinmodes_are_not_refused(self):
        for spinmode in ('onshell', 'onshell_v1', 'none', 'madspin_v1'):
            stub = self._stub(spinmode, fixed_order=True)
            stub._check_fixed_order_spinmode(spinmode)   # must not raise

    def test_nothing_is_refused_when_fixed_order_is_off(self):
        for spinmode in ('PA', 'madspin', 'onshell'):
            stub = self._stub(spinmode, fixed_order=False)
            stub._check_fixed_order_spinmode(spinmode)   # must not raise

    def test_the_refused_set_is_exactly_the_reshuffling_ones(self):
        """The list is hand-kept; tie it to what actually reshuffles so a new
        reshuffling spinmode cannot be added without landing here."""
        for spinmode in ('PA', 'madspin', 'onshell'):
            stub = self._stub(spinmode, fixed_order=True)
            reshuffles = stub._density_needs_reshuffle(True)
            refused = spinmode in stub.FIXED_ORDER_RESHUFFLING_SPINMODES
            self.assertEqual(refused, reshuffles, spinmode)


class _CaptureCritical(object):
    """Collect the CRITICAL records ``ms_density_real`` emits, and reset its
    once-per-site memory so the tests do not shadow one another."""

    def __enter__(self):
        import logging
        self.messages = []
        capture = self

        class _Handler(logging.Handler):
            def emit(self, record):
                capture.messages.append(record.getMessage())

        interface_madspin._MS_IMAG_REPORTED.clear()
        self._handler = _Handler(level=logging.CRITICAL)
        self._logger = interface_madspin.logger
        self._level = self._logger.level
        self._propagate = self._logger.propagate
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.CRITICAL)
        self._logger.propagate = False
        return self

    def __exit__(self, *args):
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._level)
        self._logger.propagate = self._propagate
        interface_madspin._MS_IMAG_REPORTED.clear()
        return False
