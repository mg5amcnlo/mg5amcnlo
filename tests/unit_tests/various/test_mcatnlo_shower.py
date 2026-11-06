"""Tests of the MC@NLO matching contract and of the shower configuration."""

import os
import tempfile
import unittest

import madgraph.various.banner as banner_mod
import madgraph.various.mcatnlo_shower as mcatnlo_shower

pjoin = os.path.join


def lhe_event(isorh, particles, attributes=''):
    """particles: list of (pdg, status, mother1, col1, col2)."""
    lines = ['<event%s>' % attributes,
             ' %d 1 +1.0e+00 1.0e+02 7.8e-03 1.1e-01' % len(particles)]
    for pdg, status, mother, col1, col2 in particles:
        lines.append(' %d %d %d %d %d %d 0 0 0 0 0 0 9' %
                     (pdg, status, mother, mother if mother else 0, col1, col2))
    lines.append('#aMCatNLO %d 5 4 4 4 0.1E+03 0.1E+03 9 0 0 1.0 1.0 1.0 1.0 1.0' % isorh)
    lines.append('</event>')
    return '\n'.join(lines) + '\n'


HEADER = ('<LesHouchesEvents version="3.0">\n<header>\n<MGRunCard>\n'
          "  10 = nevents\n  'PYTHIA8' = parton_shower\n</MGRunCard>\n</header>\n"
          '<init>\n 2212 2212 6500 6500 0 0 0 0 -4 1\n 1 0 1 1\n</init>\n')


class TestMatchingContract(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='mg5_contract_')
        self.addCleanup(self.directory.cleanup)
        self.root = self.directory.name

    def write(self, name, events):
        path = pjoin(self.root, name)
        with open(path, 'w') as stream:
            stream.write(HEADER + ''.join(events) + '</LesHouchesEvents>\n')
        return path

    def test_round_trip(self):
        contract = mcatnlo_shower.MatchingContract(
            parton_shower='PYTHIA8', mcatnlo_delta=True, ickkw=3, ptj='10d0',
            qed_matching='.true.', pythia8_mmaxgamma='2.5d0',
            nlo_resonance_decays='23, 6', born_coloured_partons=2)
        reread = mcatnlo_shower.MatchingContract.from_text(contract.to_text())
        self.assertEqual(reread, contract)
        self.assertEqual(reread['nlo_resonance_decays'], [6, 23])
        self.assertAlmostEqual(reread['pythia8_mmaxgamma'], 2.5)
        self.assertTrue(reread['qed_matching'])

    def test_scan_born_multiplicity_and_resonances(self):
        # q q~ > t t~ (S) and q q~ > t t~ g (H); a Z decaying to b b~ in
        # one event (an NLO-corrected decay), and Z > e+ e- in another.
        events = [
            lhe_event(1, [(2, -1, 0, 501, 0), (-2, -1, 0, 0, 502),
                          (6, 1, 1, 501, 0), (-6, 1, 1, 0, 502)]),
            lhe_event(2, [(2, -1, 0, 501, 0), (-2, -1, 0, 0, 503),
                          (6, 1, 1, 501, 0), (-6, 1, 1, 0, 502),
                          (21, 1, 1, 502, 503)]),
            lhe_event(1, [(2, -1, 0, 501, 0), (-2, -1, 0, 0, 501),
                          (23, 2, 1, 0, 0), (5, 1, 3, 502, 0), (-5, 1, 3, 0, 502),
                          (6, 1, 1, 503, 0), (-6, 1, 1, 0, 503)]),
            lhe_event(1, [(2, -1, 0, 501, 0), (-2, -1, 0, 0, 501),
                          (23, 2, 1, 0, 0), (11, 1, 3, 0, 0), (-11, 1, 3, 0, 0),
                          (6, 1, 1, 503, 0), (-6, 1, 1, 0, 503)]),
        ]
        scan = mcatnlo_shower.scan_events(self.write('events.lhe', events))
        self.assertEqual(scan.nevents, 4)
        self.assertEqual(scan.nevents_S, 3)
        # Decay products of the Z are not part of the production system.
        self.assertEqual(scan.born_coloured, {2})
        self.assertEqual(scan.born_multiplicity(), 2)
        self.assertEqual(scan.real_coloured, {3})
        self.assertEqual(scan.coloured_decays, {23})
        self.assertEqual(scan.charged_decays, {23})
        self.assertFalse(scan.has_npnlo)

    def test_scan_mixed_multiplicity_and_limits(self):
        events = [lhe_event(1, [(21, -1, 0, 501, 502), (21, -1, 0, 502, 501),
                                (25, 1, 1, 0, 0)], " npNLO='0'"),
                  lhe_event(1, [(21, -1, 0, 501, 502), (21, -1, 0, 502, 503),
                                (25, 1, 1, 0, 0), (21, 1, 1, 501, 503)], " npNLO='1'")]
        path = self.write('events.lhe', events)
        scan = mcatnlo_shower.scan_events(path)
        self.assertEqual(scan.born_multiplicity(), -1)
        self.assertTrue(scan.has_npnlo)
        self.assertEqual(mcatnlo_shower.scan_events(path, max_per_file=1).nevents, 1)

    def test_scan_reads_every_file(self):
        # Many channel files; the resonance appears only in the last one,
        # after more events than were ever read in total in other files.
        plain = lhe_event(1, [(2, -1, 0, 501, 0), (-2, -1, 0, 0, 501),
                              (11, 1, 1, 0, 0), (-11, 1, 1, 0, 0)])
        resonant = lhe_event(1, [(2, -1, 0, 501, 0), (-2, -1, 0, 0, 501),
                                 (23, 2, 1, 0, 0), (5, 1, 3, 502, 0),
                                 (-5, 1, 3, 0, 502)])
        paths = [self.write('ch%d.lhe' % i, [plain] * 30) for i in range(40)]
        paths.append(self.write('last.lhe', [plain] * 1999 + [resonant]))
        scan = mcatnlo_shower.scan_events(paths)
        self.assertEqual(scan.nevents, 40 * 30 + 2000)
        self.assertEqual(scan.coloured_decays, {23})
        # Only the first 2000 events of a file are read.
        paths.append(self.write('late.lhe', [plain] * 2000 + [resonant]))
        self.assertEqual(mcatnlo_shower.scan_events(paths[-1:]).coloured_decays,
                         set())

    def test_contract_in_banner_survives_banner_rewrite(self):
        path = pjoin(self.root, 'banner.txt')
        with open(path, 'w') as stream:
            stream.write(HEADER.split('<init>')[0] + '</LesHouchesEvents>\n')
        contract = mcatnlo_shower.MatchingContract(parton_shower='PYTHIA8',
                                                   pythia8_mmaxgamma=3.)
        mcatnlo_shower.add_contract_to_banner_file(path, contract)
        # Replacing an existing block does not duplicate it.
        contract['pythia8_mmaxgamma'] = 4.
        mcatnlo_shower.add_contract_to_banner_file(path, contract)
        with open(path) as stream:
            self.assertEqual(stream.read().count('<MCatNLOMatching>'), 1)
        banner = banner_mod.Banner(path)
        self.assertEqual(mcatnlo_shower.MatchingContract.from_banner(banner),
                         contract)
        # A banner written again (e.g. by MadSpin) keeps the block.
        rewritten = pjoin(self.root, 'rewritten.txt')
        banner.write(rewritten)
        self.assertEqual(mcatnlo_shower.MatchingContract.from_banner(
            banner_mod.Banner(rewritten)), contract)
        self.assertIsNone(mcatnlo_shower.MatchingContract.from_banner(
            banner_mod.Banner(self.write('old.lhe', []))))

    def test_contract_from_generation(self):
        cards = pjoin(self.root, 'Cards')
        os.mkdir(cards)
        with open(pjoin(cards, 'FKS_params.dat'), 'w') as stream:
            stream.write('! comment\n#Pythia8MMaxGamma\n2.5d0\n'
                         '#FxFxPythia8FiniteTerms\n.false.\n')
        events = [lhe_event(1, [(2, -1, 0, 501, 0), (-2, -1, 0, 0, 501),
                                (23, 2, 1, 0, 0), (11, 1, 3, 0, 0),
                                (-11, 1, 3, 0, 0)])]
        run_card = {'parton_shower': 'pythia8', 'mcatnlo_delta': False,
                    'ickkw': 3, 'ptj': 20., 'shower_scale_factor': 1.,
                    'lpp1': 1, 'lpp2': 1}
        contract = mcatnlo_shower.contract_from_generation(
            self.root, run_card, {'splitting_types': ['QCD', 'QED']},
            [self.write('events.lhe', events)])
        self.assertEqual(contract['parton_shower'], 'PYTHIA8')
        self.assertTrue(contract['qed_matching'])
        self.assertAlmostEqual(contract['pythia8_mmaxgamma'], 2.5)
        # Forced on for FxFx with Pythia8, as in the Fortran code.
        self.assertTrue(contract['fxfx_pythia8_finite_terms'])
        # With QED matching, Z > e+ e- is an NLO-corrected decay.
        self.assertEqual(contract['nlo_resonance_decays'], [23])
        self.assertEqual(contract['born_coloured_partons'], 0)
        self.assertTrue(contract['fks_params_md5'])


class FakePythia8(object):
    """Stands for a Pythia8 installation knowing the given settings."""
    version = '8.318'
    path = '/fake/pythia8'

    def __init__(self, missing=(), lhapdf6=True):
        self.missing = set(m.lower() for m in missing)
        self.lhapdf6 = lhapdf6

    def has_setting(self, name):
        return name.lower() not in self.missing

    def has_lhapdf6(self):
        return self.lhapdf6


class TestPythia8Settings(unittest.TestCase):

    def setUp(self):
        from madgraph.various.shower_card import ShowerCard
        self.card = ShowerCard()
        self.run = {'nevents': 100, 'nevents_total': 200, 'event_norm': 'average',
                    'mc_masses': {1: 0.33, 2: 0.33, 3: 0.5, 4: 1.5, 5: 4.8,
                                  11: 0.000511, 13: 0.10566, 15: 1.777, 21: 0.},
                    'masses': {6: 173., 23: 91.188}, 'widths': {6: 1.49},
                    'pdf': ('lhapdf6', 'NNPDF31_nlo_as_0118', False), 'seed': 7}

    def settings(self, contract=None, installation=None, **card):
        for key, value in card.items():
            self.card.set_param(key, value)
        contract = contract or mcatnlo_shower.MatchingContract(
            parton_shower='PYTHIA8', born_coloured_partons=2)
        return mcatnlo_shower.pythia8_settings(
            contract, self.card, self.run, installation or FakePythia8())

    def locked(self, settings):
        return dict((s.key, s.value) for s in settings.locked())

    def test_default_tiers(self):
        s = self.settings()
        locked = self.locked(s)
        for key, value in [('TimeShower:pTmaxMatch', '1'),
                           ('SpaceShower:pTmaxMatch', '1'),
                           ('TimeShower:globalRecoil', 'on'),
                           ('TimeShower:globalRecoilMode', '2'),
                           ('TimeShower:nPartonsInBorn', '2'),
                           ('TimeShower:recoilDeadCone', 'on'),
                           ('SpaceShower:dipoleRecoil', 'off'),
                           ('SpaceShower:MEcorrections', 'off'),
                           ('TimeShower:MEextended', 'off'),
                           ('Beams:setProductionScalesFromLHEF', 'off'),
                           ('5:m0', '4.8'), ('6:m0', '173.0')]:
            self.assertEqual(locked[key], value, key)
        # Free defaults: couplings, PDF, decay MECs (arXiv:2308.06389).
        self.assertEqual(s.tier('TimeShower:alphaSvalue'), mcatnlo_shower.FREE)
        self.assertEqual(s.tier('PDF:pSet'), mcatnlo_shower.FREE)
        self.assertEqual(s.get('TimeShower:MEcorrections').value, 'on')
        self.assertEqual(s.tier('TimeShower:MEcorrections'), mcatnlo_shower.FREE)
        self.assertNotIn('21:m0', locked)
        text = s.render()
        # Locked settings come after everything a user may set (tunes).
        self.assertLess(text.index('PDF:pSet'), text.index('TimeShower:pTmaxMatch'))
        self.assertIn('Random:seed = 7', text)
        self.assertIn('Main:spareMode1 = 100', text)
        self.assertIn('TimeShower:globalRecoil = on', s.render_locked())

    def test_mec_off_for_nlo_resonance_decays(self):
        contract = mcatnlo_shower.MatchingContract(parton_shower='PYTHIA8',
                                                   nlo_resonance_decays=[23])
        s = self.settings(contract)
        self.assertEqual(self.locked(s)['TimeShower:MEcorrections'], 'off')
        self.assertTrue(any('23' in note for note in s.notes))

    def test_delta_fxfx_qed_and_beams(self):
        contract = mcatnlo_shower.MatchingContract(
            parton_shower='PYTHIA8', mcatnlo_delta=True, qed_matching=True,
            pythia8_mmaxgamma=2.5, lpp1=0, lpp2=0)
        locked = self.locked(self.settings(contract))
        self.assertEqual(locked['Beams:setDipoleShowerStartingScalesFromLHEF'], 'on')
        self.assertEqual(locked['Beams:setProductionScalesFromLHEF'], 'on')
        self.assertEqual(locked['TimeShower:mMaxGamma'], '2.5')
        self.assertEqual(locked['TimeShower:QEDshowerByQ'], 'on')
        self.assertEqual(locked['PDF:lepton'], 'off')
        # Delta needs a Pythia8 version with per-dipole starting scales.
        old = FakePythia8(missing=['Beams:setDipoleShowerStartingScalesFromLHEF'])
        with self.assertRaisesRegex(mcatnlo_shower.ShowerInterfaceError,
                                    'setDipoleShowerStartingScalesFromLHEF'):
            self.settings(contract, old)
        fxfx = mcatnlo_shower.MatchingContract(parton_shower='PYTHIA8', ickkw=3,
                                               ptj=20., born_coloured_partons=1)
        with self.assertRaisesRegex(mcatnlo_shower.ShowerInterfaceError, 'Qcut'):
            self.settings(fxfx, qcut=15.)
        s = self.settings(fxfx, qcut=30.)
        locked = self.locked(s)
        self.assertEqual(locked['JetMatching:qCutME'], '20.0')
        self.assertEqual(locked['TimeShower:nPartonsInBorn'], '-1')
        self.assertEqual(s.get('JetMatching:qCut').value, '30.0')

    def test_user_options_and_opt_in(self):
        s = self.settings(pythia8_options="{'Tune:pp': 14, 'TimeShower:alphaSvalue': 0.1365}")
        self.assertEqual([(u.key, u.value) for u in s.user],
                         [('Tune:pp', '14'), ('TimeShower:alphaSvalue', '0.1365')])
        # Same value as required: accepted silently.
        self.settings(pythia8_options="{'TimeShower:globalRecoil': 'on'}")
        with self.assertRaisesRegex(mcatnlo_shower.ShowerInterfaceError,
                                    'allow_unsafe_shower_options'):
            self.settings(pythia8_options="{'TimeShower:globalRecoil': 'off'}")
        with self.assertRaisesRegex(mcatnlo_shower.ShowerInterfaceError,
                                    'SpaceShower:MEcorrections'):
            self.settings(space_shower_me_corrections='T',
                          pythia8_options='{}')
        with self.assertRaisesRegex(mcatnlo_shower.ShowerInterfaceError,
                                    'cannot be changed'):
            self.settings(space_shower_me_corrections='F',
                          pythia8_options="{'Beams:LHEF': 'other.lhe'}")
        s = self.settings(pythia8_options="{'TimeShower:globalRecoil': 'off'}",
                          allow_unsafe_shower_options='T')
        self.assertNotIn('TimeShower:globalRecoil', self.locked(s))
        self.assertIn('TimeShower:globalRecoil = off', s.render().split(
            'overridden on request')[1])
        self.assertTrue(any(n.startswith('UNSAFE') for n in s.notes))

    def test_b_mass_and_pdf_checks(self):
        with self.assertRaisesRegex(mcatnlo_shower.ShowerInterfaceError, 'b_mass'):
            self.settings(b_mass=4.75)
        self.settings(b_mass=4.8)
        with self.assertRaisesRegex(mcatnlo_shower.ShowerInterfaceError, 'LHAPDF6'):
            self.settings(installation=FakePythia8(lhapdf6=False), b_mass=-1)
        self.run['pdf'] = ('internal', None, False)
        s = self.settings(installation=FakePythia8(lhapdf6=False))
        self.assertEqual(s.get('PDF:pSet').value, '13')
        # pdfcode = 1 (same PDF as the generation) falls back on the internal
        # PDF when Pythia8 has no LHAPDF6 interface.
        self.run['pdf'] = ('lhapdf6', 'NNPDF31_nlo_as_0118', False)
        self.run['pdf_fallback'] = True
        s = self.settings(installation=FakePythia8(lhapdf6=False))
        self.assertEqual(s.get('PDF:pSet').value, '13')
        self.assertIsNone(s.get('PDF:extrapolate'))
        self.assertTrue(any('internal PDF' in n for n in s.notes))

    def test_qed_shower_off_switches_off_all_photon_radiation(self):
        s = self.settings(qed_shower='F')
        for key in ('TimeShower:QEDshowerByQ', 'TimeShower:QEDshowerByL',
                    'TimeShower:QEDshowerByOther', 'TimeShower:QEDshowerByGamma',
                    'SpaceShower:QEDshowerByQ', 'SpaceShower:QEDshowerByL'):
            self.assertEqual(s.get(key).value, 'off', key)
            self.assertEqual(s.tier(key), mcatnlo_shower.FREE, key)
        s = self.settings(qed_shower='T')
        self.assertEqual(s.get('TimeShower:QEDshowerByOther').value, 'on')

    def test_qed_shower_card_switch_with_qed_matching(self):
        contract = mcatnlo_shower.MatchingContract(parton_shower='PYTHIA8',
                                                   qed_matching=True)
        s = self.settings(contract, qed_shower='F')
        self.assertEqual(self.locked(s)['TimeShower:QEDshowerByL'], 'on')
        self.assertTrue(any('qed_shower = F is ignored' in n for n in s.notes))

    def test_unknown_settings(self):
        with self.assertRaisesRegex(mcatnlo_shower.ShowerInterfaceError,
                                    'not a setting'):
            self.settings(installation=FakePythia8(missing=['Foo:bar']),
                          pythia8_options="{'Foo:bar': 1}")
        # A default unknown to an older version is dropped, with a note.
        s = self.settings(installation=FakePythia8(missing=['Check:epTolErr']),
                          pythia8_options='{}')
        self.assertIsNone(s.get('Check:epTolErr'))
        self.assertTrue(any('Check:epTolErr' in n for n in s.notes))


class TestPythia8Installation(unittest.TestCase):

    def test_layout_version_and_settings(self):
        with tempfile.TemporaryDirectory() as root:
            xmldoc = pjoin(root, 'share', 'Pythia8', 'xmldoc')
            os.makedirs(xmldoc)
            os.makedirs(pjoin(root, 'bin'))
            config = pjoin(root, 'bin', 'pythia8-config')
            with open(config, 'w') as stream:
                stream.write('#!/bin/sh\ncase "$1" in --with-hepmc3) echo true;; '
                             '--hepmc3) echo "-I/x/include -L/x/lib -lHepMC3";; '
                             '*) echo "";; esac\n')
            os.chmod(config, 0o755)
            os.makedirs(pjoin(root, 'include', 'Pythia8Plugins'))
            open(pjoin(root, 'include', 'Pythia8Plugins', 'HepMC3.h'), 'w').close()
            with open(pjoin(xmldoc, 'Version.xml'), 'w') as stream:
                stream.write('<parm name="Pythia:versionNumber" default="8.316">')
            with open(pjoin(xmldoc, 'Timelike.xml'), 'w') as stream:
                stream.write('<flag name="TimeShower:globalRecoil" default="off">\n'
                             '<modepick name="TimeShower:globalRecoilMode" default="0">')
            p = mcatnlo_shower.Pythia8Installation(root)
            self.assertEqual(p.version, '8.316')
            self.assertEqual(p.version_tuple, (8, 316))
            self.assertTrue(p.has_setting('timeshower:GLOBALRECOIL'))
            self.assertTrue(p.has_setting('TimeShower:globalRecoilMode'))
            self.assertFalse(p.has_setting('TimeShower:recoilDeadCone'))
            self.assertTrue(p.has_setting('5:m0'))
            self.assertEqual(p.hepmc()[0], 3)
            # Pythia 8.1 layout is refused.
            os.makedirs(pjoin(root, 'old', 'xmldoc'))
            with self.assertRaisesRegex(mcatnlo_shower.ShowerInterfaceError, '8.1'):
                mcatnlo_shower.Pythia8Installation(pjoin(root, 'old'))

    def test_lhapdf_set_name(self):
        with tempfile.TemporaryDirectory() as root:
            config = pjoin(root, 'lhapdf-config')
            with open(config, 'w') as stream:
                stream.write('#!/bin/sh\necho %s\n' % root)
            os.chmod(config, 0o755)
            with open(pjoin(root, 'pdfsets.index'), 'w') as stream:
                stream.write('244800 NNPDF23_nlo_as_0119_qed 5\n'
                             '303400 NNPDF31_nlo_as_0118 1\n'
                             '303600 NNPDF31_nnlo_as_0118 1\n')
            os.makedirs(pjoin(root, 'NNPDF31_nlo_as_0118'))
            with open(pjoin(root, 'NNPDF31_nlo_as_0118',
                            'NNPDF31_nlo_as_0118.info'), 'w') as stream:
                stream.write('NumMembers: 101\n')
            name = mcatnlo_shower.lhapdf_set_name
            self.assertEqual(name(config, 244800), 'NNPDF23_nlo_as_0119_qed')
            self.assertEqual(name(config, 303400), 'NNPDF31_nlo_as_0118')
            self.assertEqual(name(config, 303401), 'NNPDF31_nlo_as_0118/1')
            self.assertIsNone(name(config, 303501))   # beyond its 101 members
            self.assertIsNone(name(config, 100))

    def test_shower_log_summary(self):
        with tempfile.TemporaryDirectory() as root:
            log = pjoin(root, 'log')
            with open(log, 'w') as stream:
                stream.write('MG5_MCATNLO_CONTRACT OK (31 settings verified)\n'
                             'MG5_MCATNLO_SUMMARY read=102 showered=100 failed=2 sigma=1.5\n')
            info = mcatnlo_shower.check_shower_log(log)
            self.assertTrue(info['contract'])
            self.assertEqual(info['showered'], '100')
            self.assertEqual(info['read'], '102')


if __name__ == '__main__':
    unittest.main()
