"""Numerical export test; enable with MG5_MACNLOPS_EXPORT_TESTS=1.

The fixture builds real matrix elements and the production evaluator. It
needs a Fortran/C++ toolchain and the usual MG5 vendor libraries, so the
ordinary fast unit-test run does not build a process implicitly.
"""

import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(os.environ.get('MG5_MACNLOPS_EXPORT_TESTS') == '1',
                     'set MG5_MACNLOPS_EXPORT_TESTS=1 to build a real process')
class TestMacNLOPSExport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not shutil.which('gfortran') or not shutil.which('g++'):
            raise unittest.SkipTest('the numerical export needs gfortran and g++')
        cls.temporary = tempfile.TemporaryDirectory(prefix='macnlops-evaluator-')
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        cls.process = directory / 'process'
        commands = directory / 'export.cmd'
        commands.write_text('generate u u~ > e+ e- [real=QCD]\n'
                            'output %s -f\nquit\n' % cls.process)
        cls.run_build([sys.executable, str(ROOT / 'bin/mg5_aMC'),
                       str(commands)], ROOT, directory / 'export.log')

        from madgraph.interface.amcatnlo_run_interface import aMCatNLOCmd
        from madgraph.various.banner import RunCardNLO
        card_path = cls.process / 'Cards/run_card.dat'
        card = RunCardNLO(str(card_path))
        card['parton_shower'] = 'PYTHIA8'
        card['mcatnlo_delta'] = False
        # Exercise the evaluator against the ordinary source D. The full
        # pipeline additionally requires a shower-readable LHAPDF set.
        card['macnlops'] = False
        card['ickkw'] = 0
        card['pdlabel'] = 'nn23nlo'
        card['ebeam1'] = card['ebeam2'] = 6500.
        card.write(str(card_path))
        interface = aMCatNLOCmd(str(cls.process))
        interface.do_treatcards('run')
        interface.do_treatcards('param')
        (cls.process / 'SubProcesses/analyse_opts').write_text(
            'FO_ANALYSE=analysis_dummy.o dbook.o '
            'open_output_files_dummy.o HwU_dummy.o\n')
        cls.run_build(['make', '-j2'], cls.process / 'Source',
                      directory / 'source-build.log')
        cls.subprocess = cls.process / 'SubProcesses/P0_uux_emep'
        cls.run_build(['make', '-j2', 'macnlops_evaluator'], cls.subprocess,
                      directory / 'evaluator-build.log')

    @classmethod
    def run_build(cls, command, cwd, log):
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=str(cwd), stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=600)
        if result.returncode:
            raise AssertionError('Command failed: %s\n%s' %
                                 (command, log.read_text()[-12000:]))

    @staticmethod
    def point(request, owner, energy, boost=0.):
        lepton_energy = (1000. - energy) / 2.
        py = math.sqrt(lepton_energy**2 - (energy / 2.)**2)
        rows = [(2, 500., 0., 0., 500.),
                (-2, 500., 0., 0., -500.),
                (11, lepton_energy, -energy / 2., py, 0.),
                (-11, lepton_energy, -energy / 2., -py, 0.),
                (21, energy, energy, 0., 0.)]
        lines = ['MACNLOPS_POINT_V1 %d %d 5' % (request, owner)]
        for identity, e, px, py, pz in rows:
            transformed_e = math.cosh(boost) * e + math.sinh(boost) * pz
            transformed_z = math.sinh(boost) * e + math.cosh(boost) * pz
            lines.append('%d %.17e %.17e %.17e %.17e' %
                         (identity, transformed_e, px, py, transformed_z))
        return '\n'.join(lines + ['END_POINT']) + '\n'

    def evaluate(self, points, config=None):
        configuration = '' if config is None else (
            'MACNLOPS_CONFIG_V1 %s\n' % ' '.join(map(str, config)))
        result = subprocess.run(['./macnlops_evaluator'],
                                cwd=str(self.subprocess),
                                input=configuration + ''.join(points),
                                text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout[-12000:])
        records = []
        for line in result.stdout.splitlines():
            if line.startswith('MACNLOPS_DENSITY_V1 '):
                fields = line.split()
                self.assertEqual(int(fields[2]), 0, result.stdout[-12000:])
                records.append(tuple(float(value) for value in fields[3:]))
        self.assertEqual(len(records), len(points), result.stdout[-12000:])
        return records

    def test_complete_density_is_owner_and_query_order_independent(self):
        records = self.evaluate([self.point(1, 1, 20.),
                                 self.point(2, 2, 20.),
                                 self.point(3, 1, 5., boost=.7),
                                 self.point(4, 1, .2),
                                 self.point(5, 1, 20.)])
        self.assertEqual(records[0], records[1])
        self.assertEqual(records[0], records[4])
        for record in records:
            self.assertTrue(all(math.isfinite(value) for value in record))
            self.assertGreater(record[0], 0.)
            self.assertGreater(record[2], 0.)
        # Source bogus P vanishes below .5 GeV, so its complete H is zero.
        self.assertEqual(records[3][0], records[3][1])

    def test_original_g_replacement_changes_the_acceptance(self):
        real, subtraction, proposal = self.evaluate([self.point(1, 1, 20.)])[0]
        expected = (134012.48964838282, 140894.55029855983,
                    134068.30492936928)
        for actual, reference in zip((real, subtraction, proposal), expected):
            self.assertAlmostEqual(actual / reference, 1., delta=2.e-8)
        acceptance = 1. - max(subtraction - real, 0.) / proposal
        naive = min(1., real / proposal)
        self.assertGreater(naive - acceptance, .04)
        self.assertGreater(acceptance, 0.)
        self.assertLess(acceptance, 1.)

    def test_shower_configuration_changes_only_proposal_support(self):
        point = [self.point(1, 1, 20.)]
        baseline = self.evaluate(point)[0]
        cut = self.evaluate(point, config=(1., 1., 100., 100.))[0]
        outside_beam = self.evaluate(point, config=(.01, .01, 0., 0.))[0]
        self.assertEqual(baseline[:2], cut[:2])
        self.assertEqual(baseline[:2], outside_beam[:2])
        self.assertEqual(cut[2], 0.)
        self.assertEqual(outside_beam[2], 0.)

    def test_complete_h_matches_ordinary_generation_density(self):
        from docs.audits.check_macnlops_source_h import check_source_h
        result = check_source_h(self.subprocess)
        self.assertEqual(result['points'], 100)
        self.assertLess(result['max_error_relative_real_subtraction'], 1.e-10)

    def test_pure_source_pairs_and_negative_h_bound(self):
        from docs.audits.check_macnlops_source_pairs import check_source_pairs
        # A density-only fixture supplies an explicit profile. Production
        # generation measures these values from its actual initialized shower.
        points = [self.point(i + 1, owner, energy)
                  for i, (owner, energy) in enumerate(
                      ((1, .2), (2, .2), (1, 1.), (2, 1.), (1, 20.)))]
        result = check_source_pairs(self.subprocess, points,
            profile=(.5, .5867333369950679, 1.2, 4.,
                     .9998556507692308, .9998556507692308),
            resolved_source=True)
        self.assertEqual(result['points'], 5)
        self.assertEqual(result['transferred_real_points'], 2)
        self.assertEqual(result['resolved_real_points'], 3)
        self.assertLess(result['max_real_pair_error'], 1.e-11)
        self.assertLess(result['max_mc_pair_error'], 1.e-11)
        self.assertLess(result['max_bound_excess'], 1.e-11)

    def test_ordinary_source_keeps_bogus_p_and_undamped_proposal(self):
        from docs.audits.check_macnlops_source_pairs import check_source_pairs
        points = [self.point(i + 1, 1, energy)
                  for i, energy in enumerate((.2, 3., 20.))]
        profile = (.5, .5867333369950679, 1.2, 4.,
                   .9998556507692308, .9998556507692308)
        ordinary = check_source_pairs(self.subprocess, points, profile=profile)
        resolved = check_source_pairs(self.subprocess, points, profile=profile,
                                      resolved_source=True)
        self.assertEqual(ordinary['probne_values'][0], 0.)
        self.assertGreater(ordinary['probne_values'][1], 0.)
        self.assertLess(ordinary['probne_values'][1], 1.)
        self.assertEqual(ordinary['probne_values'][2], 1.)
        for expected, actual in zip(ordinary['probne_values'],
                                    ordinary['hard_real_fractions']):
            self.assertAlmostEqual(actual, expected, delta=1.e-11)
        for old, new in zip(ordinary['densities'], resolved['densities']):
            self.assertAlmostEqual(old[1], new[1], delta=abs(old[1])*1.e-12)
            self.assertAlmostEqual(old[3], new[3], delta=max(abs(old[3]),1.)*1.e-12)
        # Source G/P is independent of the chosen proposal support gate.
        cut = check_source_pairs(self.subprocess, [points[1]],
            profile=(100., 100., *profile[2:]))
        for actual, expected in zip(cut['densities'][0][1:3],
                                    ordinary['densities'][1][1:3]):
            self.assertAlmostEqual(actual, expected, delta=abs(expected)*1.e-12)
        self.assertEqual(cut['densities'][0][3], 0.)

    def test_pure_source_requires_measured_shower_profile(self):
        from docs.audits.check_macnlops_source_pairs import check_source_pairs
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            check_source_pairs(self.subprocess, [self.point(1, 1, 20.)],
                               profile=(-1.,) * 6)
        self.assertIn('MAcNLOPS shower profile is absent or invalid',
                      failure.exception.stdout)


if __name__ == '__main__':
    unittest.main()
