"""Default of the real-emission integration channels (real_channels_x0),
chosen at output from the real-emission diagrams."""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import madgraph.interface.master_interface as mgcmd
import madgraph.fks.fks_helas_objects as fks_helas
import madgraph.fks.fks_real_channels_auto as auto
from madgraph.interface.amcatnlo_interface import aMCatNLOInterface
from madgraph.various import banner


class TestRealChannelsDecision(unittest.TestCase):
    """The diagram analysis on generated processes"""

    def decide(self, model, process):
        interface = mgcmd.MasterCmd()
        interface.no_notification()
        interface.exec_cmd('import model %s' % model, printcmd=False)
        interface.exec_cmd('generate %s' % process, printcmd=False)
        mes = fks_helas.FKSHelasMultiProcess(interface._fks_multi_proc)
        directories = [auto.real_structures(me)
                       for me in mes['matrix_elements']]
        return auto.decide(directories, 10)

    def test_photon_initiated_t_channel(self):
        """u a > w+ h d: the photon couples to the t-channel W, which no
        Born diagram (s-channel W only) has"""
        x0, summary = self.decide('loop_qcd_qed_sm', 'p p > w+ h [real=QED]')
        self.assertEqual(x0, auto.AUTO_X0)
        uncovered = dict((tuple(pdgs), diags) for pdgs, diags, n
                         in summary['reals_uncovered'])
        self.assertIn((2, 22, 1, 24, 25), uncovered)
        # q q~ > w+ h a is a pure emission off the Born
        self.assertNotIn((2, -1, 22, 24, 25), uncovered)

    def test_gluon_fusion_in_real_emission(self):
        """u g > w+ b b~ d: g g > b b~ through a t-channel gluon off the
        light-quark line (4FS)"""
        x0, summary = self.decide('loop_sm', 'p p > w+ b b~ [real=QCD]')
        self.assertEqual(x0, auto.AUTO_X0)
        self.assertTrue(all(21 in pdgs[:2]
                            for pdgs, diags, n in summary['reals_uncovered']))

    def test_born_like_real_emission(self):
        """Drell-Yan and t t~ at NLO QCD: all the real-emission diagrams
        are followed by the Born channels"""
        for process in ['p p > e+ e- [real=QCD]', 'p p > t t~ [real=QCD]']:
            x0, summary = self.decide('loop_sm', process)
            self.assertEqual(x0, -1., process)
            self.assertEqual(summary['n_uncovered'], 0, process)

    def test_resonance_or_decay_products(self):
        """The decision does not change when a resonance is replaced by its
        decay products: the W bounds the propagators that contain it, as
        the external W does (and the leptons are not FKS partons)"""
        for model, processes in [
                ('loop_sm', ['p p > w+ j [real=QCD]',
                             'p p > e+ ve j [real=QCD]']),
                ('loop_qcd_qed_sm', ['p p > w+ [real=QED]',
                                     'p p > e+ ve [real=QED]'])]:
            for process in processes:
                x0, summary = self.decide(model, process)
                self.assertEqual(x0, -1., process)
                self.assertEqual(summary['n_uncovered'], 0, process)
        # and both flagged when the real emission opens a new structure
        for process in ['p p > w+ h [real=QED]', 'p p > e+ ve h [real=QED]']:
            x0, summary = self.decide('loop_qcd_qed_sm', process)
            self.assertEqual(x0, auto.AUTO_X0, process)

    def test_too_many_channels(self):
        x0, summary = auto.decide([], 0)
        self.assertEqual(x0, -1.)
        x0, summary = self.decide('loop_qcd_qed_sm', 'p p > w+ h [real=QED]')
        self.assertEqual(x0, auto.AUTO_X0)
        directories = [{'born': {'legs': [], 'props': []}, 'reals': []}]
        self.assertEqual(auto.decide(directories,
                                     auto.MAX_HARD_CHANNELS + 1)[0], -1.)


class TestRealChannelsAnalyserCache(unittest.TestCase):
    """The parameter values are evaluated once per model of an output"""

    def setUp(self):
        auto.clear_analysers()
        patcher = mock.patch.object(aMCatNLOInterface, 'get_parameter_values',
                                    side_effect=lambda model: {'MT': 173.})
        self.evaluate = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(auto.clear_analysers)

    def test_alternating_models_are_evaluated_once(self):
        """The Born and the real amplitudes of a directory carry different
        model objects, used in turn"""
        born, real = object(), object()
        for _ in range(5):
            self.assertIs(auto.get_analyser(born).model, born)
            self.assertIs(auto.get_analyser(real).model, real)
        self.assertEqual(self.evaluate.call_count, 2)

    def test_reused_id_does_not_return_another_model(self):
        old, new = object(), object()
        auto._analysers[id(new)] = auto.RealChannelAnalyser(old, {})
        self.assertIs(auto.get_analyser(new).model, new)
        self.assertEqual(self.evaluate.call_count, 1)

    def test_clear_releases_the_models(self):
        auto.get_analyser(object())
        auto.clear_analysers()
        self.assertEqual(auto._analysers, {})


class TestRealChannelsSectors(unittest.TestCase):
    """The coverage rule on hand-made sectors"""

    def test_identical_permutations(self):
        legs = [(1, 21, False), (2, 21, False), (3, 21, True),
                (4, 21, True), (5, 21, True)]
        sectors = auto.identical_permutations([(5, 4, True, 4)], legs)
        self.assertIn((3, 4, True, 4), sectors)
        self.assertIn((5, 3, True, 4), sectors)
        self.assertEqual(len(sectors), 6)

    def test_follows(self):
        # 1 2 > 3 4 5, i=5 soft, j=4 (Born leg 4), Born 1 2 > 3 4
        mapping = {1: 1, 2: 2, 3: 3, 4: 4}
        born_props = set([(frozenset([2, 4]), 6)])
        sector = auto.Sector(5, 4, True, mapping, born_props, 4)
        # does not separate i from j
        self.assertTrue(sector.follows({4, 5}, 21, 5))
        self.assertTrue(sector.follows({1, 3}, 21, 5))
        # i attached to the external leg 3: soft enhancement only
        self.assertTrue(sector.follows({3, 5}, 6, 5))
        # i emitted off a propagator that the Born has
        self.assertTrue(sector.follows({2, 4}, 6, 5))
        # a propagator that the Born does not have
        self.assertFalse(sector.follows({2, 4}, 24, 5))
        hard = auto.Sector(5, 4, False, mapping, born_props, 4)
        self.assertFalse(hard.follows({3, 5}, 6, 5))


    def test_merged_flavours(self):
        """d g > d g g is a flavour of the u g > u g directory, where it has
        its soft sectors, and the representative of a real process of
        g g > g g, where it only has the hard sector (3, 1)"""
        diagrams = [[[[3, 5], 2, True, 0., 0.]]]
        masses = [None, None, 0., 0., 0.]
        quark_gluon = {
            'born': {'legs': [[1, 2, False], [2, 21, False], [3, 2, True],
                              [4, 21, True]],
                     'flavours': [[2, 21, 2, 21], [1, 21, 1, 21]],
                     'props': [], 'threshold': 0.},
            'reals': [{'legs': [[1, 2, False], [2, 21, False], [3, 2, True],
                                [4, 21, True], [5, 21, True]],
                       'flavours': [[2, 21, 2, 21, 21], [1, 21, 1, 21, 21]],
                       'sectors': [(5, 4, True, 4)], 'masses': masses,
                       'coloured': [1, 2, 3, 4, 5],
                       'diagrams': diagrams}]}
        gluons = {
            'born': {'legs': [[1, 21, False], [2, 21, False], [3, 21, True],
                              [4, 21, True]],
                     'flavours': [[21, 21, 21, 21]],
                     'props': [], 'threshold': 0.},
            'reals': [{'legs': [[1, 1, False], [2, 21, False], [3, 1, True],
                                [4, 21, True], [5, 21, True]],
                       'flavours': [[1, 21, 1, 21, 21]],
                       'sectors': [(3, 1, False, 1)], 'masses': masses,
                       'coloured': [1, 2, 3, 4, 5],
                       'diagrams': diagrams}]}
        key = ((1, False), (21, False), (1, True), (21, True), (21, True))
        reals = auto.collect_reals([gluons, quark_gluon])
        self.assertEqual(sorted((s.i, s.j, s.soft) for s in
                                reals[key]['sectors']),
                         [(3, 1, False), (4, 5, True), (5, 4, True)])
        # the quark propagator is a peak, followed by the soft sectors
        x0, summary = auto.decide([gluons, quark_gluon], 10)
        self.assertEqual(summary['n_uncovered'], 0)
        self.assertEqual(x0, -1.)
        # with the first flavour only, d g > d g g has the hard sector only
        del quark_gluon['reals'][0]['flavours']
        reals = auto.collect_reals([gluons, quark_gluon])
        self.assertEqual([(s.i, s.j) for s in reals[key]['sectors']],
                         [(3, 1)])
        x0, summary = auto.decide([gluons, quark_gluon], 10)
        self.assertEqual(summary['reals_uncovered'],
                         [([1, 21, 1, 21, 21], [1], 1)])

    def test_peaking(self):
        initial = frozenset([1, 2])
        # 1 2 > 3 4 5 6, all massless, 3 and 4 the decay products of a W
        masses = {3: 0., 4: 0., 5: 0., 6: 0.}
        w = [[3, 4], 24, True, 80., 2.]
        resonances = auto.diagram_resonances([w], initial, masses)
        self.assertEqual(resonances, {frozenset([3, 4]): 80.})
        quark = [[3, 4, 5], 1, True, 0., 0.]
        partons = {3, 4, 5, 6}
        # bounded by the W, as when the W is external (1 2 > 3 5 6)
        self.assertFalse(auto.peaking(quark, initial, masses, partons,
                                      resonances))
        self.assertFalse(auto.peaking([[3, 5], 1, True, 0., 0.], initial,
                                      {3: 80., 5: 0., 6: 0.}, partons, {}))
        self.assertTrue(auto.peaking(quark, initial, masses, partons, {}))
        # leptons are not FKS partons: bounded by the cuts of the Born
        self.assertFalse(auto.peaking(quark, initial, masses, {5, 6}, {}))
        # a massless t-channel propagator is singular when the legs of
        # either side become collinear to their beam, however it is written
        masses = {3: 80., 4: 0.}
        for side in ([1, 4], [2, 3]):
            gluon = [side, 21, False, 0., 0.]
            self.assertTrue(auto.peaking(gluon, initial, masses, {4}, {}))
        # a massive t-channel propagator is compared to the hard scale of
        # the Born, which includes its resonances
        masses = {3: 0., 4: 0., 5: 0.}
        w_t = [[1, 5], 24, False, 80., 2.]
        self.assertTrue(auto.peaking(w_t, initial, masses, {5}, {}))
        self.assertFalse(auto.peaking(w_t, initial, masses, {5}, {}, 80.))
        self.assertEqual(auto.born_threshold(
            [[[[3, 4], 24, True, 80., 2.]]], initial, {3: 0., 4: 0.}), 80.)
        self.assertEqual(auto.born_threshold(
            [[[[3, 4], 22, True, 0., 0.]], [[[3, 4], 23, True, 91., 2.]]],
            initial, {3: 0., 4: 0.}), 91.)


class TestRealChannelsDefault(unittest.TestCase):
    """The default of real_channels_x0 written in the run_card of a process"""

    def make_card(self, x0):
        charac = banner.ProcCharacteristic()
        charac['real_channels_x0'] = x0
        charac['ninitial'] = 2
        process = self.process
        card = banner.RunCardNLO()
        card.create_default_for_process(charac, None, [process])
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        path = os.path.join(directory, 'run_card.dat')
        card.write(path)
        return card, open(path).read(), path

    def setUp(self):
        interface = mgcmd.MasterCmd()
        interface.no_notification()
        interface.exec_cmd('import model loop_sm', printcmd=False)
        interface.exec_cmd('generate u u~ > e+ e- [real=QCD]', printcmd=False)
        self.process = \
            interface._fks_multi_proc['born_processes'][0].born_amp['process']

    def test_written_in_run_card(self):
        for x0 in (0.05, -1.0):
            card, text, path = self.make_card(x0)
            self.assertEqual(card['real_channels_x0'], x0)
            line = next(l for l in text.splitlines()
                        if '= real_channels_x0' in l)
            self.assertEqual(float(line.split('=')[0]), x0)
            self.assertEqual(banner.RunCardNLO(path)['real_channels_x0'], x0)
