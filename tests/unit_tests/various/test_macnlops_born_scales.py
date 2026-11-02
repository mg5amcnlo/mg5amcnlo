"""Native Born-scale factors must enter the same S/H transfer density.

The provider owns the exact model-dependent coupling and hard-PDF ratios.
These tests compile the C++ consumer and exercise its actual decision paths;
the separate FKS fixture tests how the Fortran provider obtains those ratios.
"""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from madgraph.various import macnlops as mc
from tests.unit_tests.various import test_macnlops_first_emission as first


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc'
COUPLING = 'MACNLOPS_STUB_BORN_COUPLING'
PDF = 'MACNLOPS_STUB_BORN_PDF'
READY = 'MACNLOPS_STUB_BORN_READY'


@unittest.skipUnless(shutil.which('g++'), 'requires g++')
class TestBornScaleConsumerDomain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='macnlops_born_scale_domain_')
        cls.addClassCleanup(cls.tmp.cleanup)
        work = Path(cls.tmp.name)
        source = SOURCE.read_text()
        function = source[source.index('double bornScaleFactor('):
                          source.index('// Signed-residual acceptances')]
        (work / 'factor.cc').write_text(r'''
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
struct Density {
  struct BornScale {int ready;double coupling,pdf;} bornScale;
};
''' + function + r'''
int main() {
  int ready;std::string coupling,pdf;
  while(std::cin>>ready>>coupling>>pdf) {
    Density x{{ready,std::stod(coupling),std::stod(pdf)}};
    try {const double value=bornScaleFactor(x);
      std::cout<<std::setprecision(17)<<value<<"\n";}
    catch(const std::runtime_error&) {std::cout<<"invalid\n";}
  }
}
''')
        cls.exe = work / 'factor'
        subprocess.run(['g++', '-std=c++11', '-O2', str(work / 'factor.cc'),
                        '-o', str(cls.exe)], check=True, capture_output=True, text=True)

    def factors(self, cases):
        result = subprocess.run([str(self.exe)], input=''.join(
            '%s %s %s\n' % case for case in cases), capture_output=True,
            text=True, check=True)
        return [float(value) if value != 'invalid' else value
                for value in result.stdout.splitlines()]

    def test_exact_coupling_ratios_do_not_acquire_an_assumed_qcd_power(self):
        # Ratios of complete provider Born weights, including a mixed-order
        # effective coupling. The consumer must use them without re-exponentiation.
        alpha_born, alpha_real = .17, .23
        born_weights = [1., alpha_born, alpha_born**2,
                        (alpha_born + 3*alpha_born**2)**2]
        real_weights = [1., alpha_real, alpha_real**2,
                        (alpha_real + 3*alpha_real**2)**2]
        ratios = [born/real for born, real in zip(born_weights, real_weights)]
        values = self.factors([(1, ratio, .7) for ratio in ratios])
        for ratio, value in zip(ratios, values):
            self.assertAlmostEqual(value, .7*ratio, delta=2.e-15)

    def test_zero_support_and_small_positive_factors_have_distinct_domains(self):
        values = self.factors([(1, 1., 1.), (1, 0., 2.), (1, 3., 0.),
                               (1, 1.e-300, 2.), (1, 2., 3.)])
        self.assertEqual(values[:3], [1., 0., 0.])
        self.assertEqual(values[3], 2.e-300)
        self.assertEqual(values[4], 6.)

    def test_unready_nonfinite_negative_and_overflowing_factors_fail(self):
        cases = [(0, 1., 1.), (2, 1., 1.), (1, -1., 1.), (1, 1., -1.),
                 (1, 'nan', 1.), (1, 1., 'nan'), (1, 'inf', 1.),
                 (1, 1., '-inf'), (1, 0., 'nan'), (1, 1.e300, 1.e300)]
        self.assertEqual(self.factors(cases), ['invalid']*len(cases))


class TestPairedBornScaleNative(unittest.TestCase):
    """Actual native S/H decisions, retaining every original signed event."""
    REGISTRY = first.TestFirstEmissionNative.REGISTRY
    run_match = first.TestFirstEmissionNative.run_match

    @classmethod
    def setUpClass(cls):
        try:
            first.TestFirstEmissionNative.setUpClass.__func__(cls)
        except subprocess.CalledProcessError as error:
            raise AssertionError((error.stdout or '')+(error.stderr or '')) from error
        cls.original_input = (cls.work / 'events.lhe').read_text()
        cls.base_environment = dict(cls.env)
        for key in (COUPLING, PDF, READY):
            cls.base_environment.pop(key, None)

    def setUp(self):
        self.env = dict(self.base_environment, MACNLOPS_STUB_MODE='4')
        (self.work / 'events.lhe').write_text(
            self.original_input.replace('source="delta"', 'source="delta_native_tc"'))

    def tearDown(self):
        (self.work / 'events.lhe').write_text(self.original_input)

    @staticmethod
    def decision(block):
        return mc.xml_attributes(re.search(
            r'<macnlops_decision\b([^<>]*?)/>', block)[1])

    @staticmethod
    def physical_prefix(block):
        return block.split('<rwgt>', 1)[0]

    def test_unit_reference_factors_preserve_the_fixed_scale_result(self):
        before = self.run_match('unit_default')
        self.env.update({COUPLING: '1', PDF: '1', READY: '1'})
        after = self.run_match('unit_explicit')
        self.assertEqual(before[:2], after[:2])
        evaluated = [self.decision(block) for block in after[1]
                     if 'born_scale_factor=' in block]
        self.assertTrue(evaluated)
        for decision in evaluated:
            self.assertEqual(float(decision['born_scale_factor']), 1.)

    def test_both_factors_rescale_s_and_h_without_changing_source_endpoints(self):
        _, before, _ = self.run_match('factor_before')
        self.env.update({COUPLING: '2', PDF: '1.5'})
        _, after, _ = self.run_match('factor_after')
        self.assertEqual(len(before), len(after))
        checked_s = checked_h = 0
        for old, new in zip(before, after):
            self.assertEqual(self.physical_prefix(old), self.physical_prefix(new))
            a, b = self.decision(old), self.decision(new)
            if 'born_scale_factor' not in a:
                continue
            self.assertEqual(float(b['born_coupling_factor']), 2.)
            self.assertEqual(float(b['born_pdf_factor']), 1.5)
            self.assertEqual(float(b['born_scale_factor']), 3.)
            self.assertAlmostEqual(float(b['kernel_scale'])/float(a['kernel_scale']),
                                   3., delta=2.e-14)
            if mc.Event(old).kind == 'S':
                self.assertAlmostEqual(1.-float(b['keep']),
                                       (1.-float(a['keep']))/3., delta=3.e-15)
                checked_s += 1
            else:
                # This fixture's ordinary H points are source-limited.
                self.assertEqual(float(b['keep']), float(a['keep']))
                checked_h += 1
        self.assertGreater(checked_s, 0)
        self.assertGreater(checked_h, 0)

    def test_active_common_bound_changes_h_removal_with_the_same_factor(self):
        self.env.update({'MACNLOPS_STUB_SOURCE': '1', 'MACNLOPS_STUB_PS': '0.000001'})
        _, before, _ = self.run_match('active_before')
        self.env.update({COUPLING: '2', PDF: '3'})
        _, after, _ = self.run_match('active_after')
        checked = 0
        for old, new in zip(before, after):
            a, b = self.decision(old), self.decision(new)
            if a['rule'] != 'h_removal':
                continue
            u, source = float(b['d'])-float(b['r']), float(b['delta_source_mean'])
            available = (float(b['k'])*float(b['kernel_scale'])*
                         float(b['delta_shower_mean']))
            transferred = u*source*(1.-float(b['keep']))
            self.assertLess(available, min(u, float(b['k']))*source)
            self.assertAlmostEqual(transferred, available, delta=3.e-15)
            self.assertAlmostEqual(1.-float(b['keep']),
                                   6.*(1.-float(a['keep'])), delta=3.e-15)
            self.assertNotEqual(float(a['keep']), float(b['keep']))
            checked += 1
        self.assertGreater(checked, 0)

    def test_zero_reference_support_retains_both_signed_event_classes(self):
        path = self.work / 'events.lhe'
        with mc.LHEStream(path) as stream:
            header, blocks = stream.header, list(stream)
        for i, block in enumerate(blocks):
            if mc.Event(block).kind != 'S':
                continue
            lines = block.splitlines(keepends=True)
            tokens = lines[1].split();tokens[2] = '-' + tokens[2].lstrip('+-')
            lines[1] = ' '.join(tokens)+'\n';blocks[i] = ''.join(lines)
        path.write_text(header+''.join(blocks)+'</LesHouchesEvents>\n')
        for coupling, pdf in [('0', '1'), ('1', '0')]:
            with self.subTest(coupling=coupling, pdf=pdf):
                self.env.update({COUPLING: coupling, PDF: pdf})
                _, output, _ = self.run_match('unsupported_'+coupling+pdf)
                self.assertEqual(len(output), len(blocks))
                kinds = set()
                for block in output:
                    event, decision = mc.Event(block), self.decision(block)
                    if decision['rule'] != 'born_scale_no_transfer':
                        continue
                    kinds.add(event.kind)
                    self.assertLess(event.weight, 0.)
                    self.assertEqual((decision['keep'], decision['accepted']), ('1', '1'))
                    weight = re.search(r"<wgt id='macnlops'> (\S+) </wgt>", block)[1]
                    self.assertEqual(mc.number(weight), event.weight)
                self.assertEqual(kinds, {'S', 'H'})

    def test_legacy_v2_v3_do_not_use_the_v4_reference_factor(self):
        for version in ('v2', 'v3'):
            with self.subTest(version=version):
                self.env.pop(COUPLING, None);self.env.pop(PDF, None);self.env.pop(READY, None)
                before = self.run_match('legacy_before_'+version, '--s-veto', version)
                self.env.update({COUPLING: 'nan', PDF: '0', READY: '0'})
                after = self.run_match('legacy_after_'+version, '--s-veto', version)
                self.assertEqual(before[:2], after[:2])

    def test_invalid_native_reference_is_fatal_instead_of_unit_fallback(self):
        for key, value in [(READY, '0'), (COUPLING, 'nan'), (PDF, '-1')]:
            with self.subTest(key=key, value=value), patch.dict(self.env, {key: value}):
                result = subprocess.run(['./match.exe', '--input', 'events.lhe',
                    '--output', 'invalid.lhe', '--summary', 'invalid.json',
                    '--routes', 'routes.dat', '--settings', str(self.settings),
                    '--pdf', self.pdf, '--seed', '7349'], cwd=self.work,
                    env=self.env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Born-scale factor', result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
