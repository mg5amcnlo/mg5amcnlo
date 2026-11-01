"""Paired transfer protection where the native Born PDF ratio is undefined.

The native integration fixture uses the real matching driver and its existing
stand-in density provider. Only the provider's canonical Born profile is
controlled; Pythia, source events and shower random streams remain ordinary.
"""
import os
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


@unittest.skipUnless(shutil.which('g++'), 'requires g++')
class TestBornPdfSupportDomain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='macnlops_pdf_support_')
        cls.addClassCleanup(cls.tmp.cleanup)
        work = Path(cls.tmp.name)
        source = SOURCE.read_text()
        function = source[source.index('bool hasBornPdfSupport('):
                          source.index('struct GaussRule {')]
        program = r'''
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
struct Hook {
  double values[2]; int calls[2]={0,0};
  double beamXf(int side,int,double,double) {++calls[side];return values[side];}
};
struct Context {Hook* hook;int flavours;};
struct SudakovProfile {int ids[2];double x[2];double pt;};
''' + function + r'''
int main(int argc,char**argv) {
  if(argc!=9)return 2;
  Hook hook;hook.values[0]=std::stod(argv[6]);hook.values[1]=std::stod(argv[7]);
  Context c{&hook,std::stoi(argv[8])};
  SudakovProfile p{{std::stoi(argv[2]),std::stoi(argv[3])},
    {std::stod(argv[4]),std::stod(argv[5])},std::stod(argv[1])};
  try {const bool result=hasBornPdfSupport(c,p);std::cout<<(result?"yes":"no");}
  catch(const std::runtime_error&){std::cout<<"invalid";}
  std::cout<<" "<<hook.calls[0]<<" "<<hook.calls[1]<<"\n";
}
'''
        (work / 'support.cc').write_text(program)
        cls.exe = work / 'support'
        subprocess.run(['g++', '-std=c++11', '-O2', str(work / 'support.cc'),
                        '-o', str(cls.exe)], check=True, capture_output=True, text=True)

    def support(self, values=(1., 1.), ids=(4, -3), x=(.022904539932296634, .01),
                pt=1.7028105905461708, flavours=4):
        command = [str(self.exe), str(pt), *map(str, ids), *map(str, x),
                   *map(str, values), str(flavours)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        state, a, b = result.stdout.split()
        return state, (int(a), int(b))

    def test_both_competing_beams_need_support(self):
        self.assertEqual(self.support(values=(0., 2.)), ('no', (1, 1)))
        self.assertEqual(self.support(values=(2., 0.)), ('no', (1, 1)))
        self.assertEqual(self.support(values=(-.00038605536778315, 2.)), ('no', (1, 1)))
        # A positive gluon PDF is part of the same domain as the quarks.
        self.assertEqual(self.support(values=(2., 0.), ids=(4, 21)), ('no', (1, 1)))

    def test_positive_values_are_not_replaced_by_an_artificial_floor(self):
        for small in (1.e-300, 1.e-12, 2.716191890459e-5, 1.):
            with self.subTest(value=small):
                self.assertEqual(self.support(values=(small, 2.)), ('yes', (1, 1)))

    def test_nonfinite_second_pdf_is_fatal_even_after_a_zero_first_pdf(self):
        for bad in ('nan', 'inf', '-inf'):
            with self.subTest(value=bad):
                self.assertEqual(self.support(values=(0., bad)), ('invalid', (1, 1)))
        for pt in (0., 'nan', 'inf'):
            with self.subTest(pt=pt):
                self.assertEqual(self.support(pt=pt)[0], 'invalid')
        for x in ((0., .01), (1., .01), (.01, 'nan')):
            with self.subTest(x=x):
                self.assertEqual(self.support(x=x)[0], 'invalid')

    def test_colorless_and_inactive_flavours_do_not_query_hadron_pdfs(self):
        self.assertEqual(self.support(ids=(11, -11), x=(1., 1.),
                                      values=('nan', 'nan')), ('yes', (0, 0)))
        self.assertEqual(self.support(ids=(5, 21), values=('nan', 2.), flavours=4),
                         ('yes', (0, 1)))


class TestPairedBornPdfSupportNative(unittest.TestCase):
    """Real S/H decision paths must both retain an unsupported component."""
    REGISTRY = first.TestFirstEmissionNative.REGISTRY
    run_match = first.TestFirstEmissionNative.run_match

    @classmethod
    def setUpClass(cls):
        replacement = r'''
  const char* support=getenv("MACNLOPS_STUB_BORN_PDF_SUPPORT");
  if(support && atoi(support)>=1 && atoi(support)<=2) {
    const int side=atoi(support)-1;
    // Bottom support vanishes below its PDF threshold. These canonical
    // profile labels intentionally differ from the source's representative
    // incoming flavour, ensuring the guard consults the native profile.
    ids[side]=5;x[side]=.022904539932296634;*pt=.6;q[0]=*pt;
  }
'''
        stub = first.STUB.replace('  *mass=13000.*sqrt(x[0]*x[1]);',
                                  replacement + '  *mass=13000.*sqrt(x[0]*x[1]);')
        if stub == first.STUB:
            raise AssertionError('Native density fixture has changed')
        with patch.object(first, 'STUB', stub):
            first.TestFirstEmissionNative.setUpClass.__func__(cls)
        cls.original_input = (cls.work / 'events.lhe').read_text()
        cls.base_environment = dict(cls.env)
        native = r'''
#define main unused_matching_main
#include "MATCH_SOURCE"
#undef main
class PdfHook : public FirstEmission {
public:
  void setBeams(Pythia8::BeamParticle& a,Pythia8::BeamParticle& b) {
    beamAPtr=&a;beamBPtr=&b;
  }
};
int main(int argc,char**argv) {
  Pythia8::Pythia engine;
  if(!engine.readString("PDF:pSet = LHAPDF6:NNPDF40_nlo_as_01180_nf_4/0"))return 77;
  auto pdf=engine.getPDFPtr(2212);
  if(!pdf || !pdf->isSetup())return 77;
  Pythia8::BeamParticle a,b;a.initPDFPtr(pdf,pdf);b.initPDFPtr(pdf,pdf);
  PdfHook hook;hook.setBeams(a,b);
  Context c;c.hook=&hook;c.flavours=4;
  SudakovProfile p;p.ids[0]=4;p.ids[1]=21;
  p.x[0]=.022904539932296634;p.x[1]=.058242641107197977;
  p.pt=std::stod(argv[1]);
  const double charm=hook.beamXf(0,4,p.x[0],p.pt*p.pt);
  const double gluon=hook.beamXf(1,21,p.x[1],p.pt*p.pt);
  const bool nativeSupport=hasBornPdfSupport(c,p);
  // A zero real-mother charm PDF does not make a positive Born-gluon
  // denominator unsupported. Mother/daughter zero cases are different.
  p.ids[0]=21;
  const bool gluonBornSupport=hasBornPdfSupport(c,p);
  std::cout<<std::setprecision(17)<<"SUPPORT "<<nativeSupport<<" "<<charm
           <<" "<<gluon<<" "<<gluonBornSupport<<"\n";
}
'''.replace('MATCH_SOURCE', str(SOURCE))
        (cls.work / 'native_support.cc').write_text(native)
        cls.native_probe = cls.work / 'native_support'
        command = [os.environ.get('CXX', 'g++'), '-std=c++11', '-O1',
                   str(cls.work / 'native_support.cc'), str(cls.work / 'stub.o'),
                   '-I' + str(cls.prefix / 'include'), '-L' + str(cls.prefix / 'lib'),
                   '-Wl,-rpath,' + str(cls.prefix / 'lib'), '-lpythia8', '-ldl',
                   '-o', str(cls.native_probe)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def setUp(self):
        self.env = dict(self.base_environment, MACNLOPS_STUB_MODE='3')
        (self.work / 'events.lhe').write_text(
            self.original_input.replace('source="delta"', 'source="delta_native_ir"'))

    def tearDown(self):
        (self.work / 'events.lhe').write_text(self.original_input)

    def decisions(self, blocks):
        for block in blocks:
            decision = mc.xml_attributes(re.search(
                r'<macnlops_decision\b([^<>]*?)/>', block)[1])
            yield mc.Event(block), decision

    def test_default_v4_retains_both_s_and_h_using_either_profile_beam(self):
        for side in (1, 2):
            with self.subTest(side=side):
                self.env['MACNLOPS_STUB_BORN_PDF_SUPPORT'] = str(side)
                _, blocks, report = self.run_match('unsupported_beam%d' % side)
                self.assertEqual(len(blocks), len(self.source))
                kinds = set()
                for event, decision in self.decisions(blocks):
                    if decision['rule'] != 'pdf_support_no_transfer':
                        continue
                    kinds.add(event.kind)
                    self.assertEqual((decision['keep'], decision['accepted']), ('1', '1'))
                    weight = re.search(r"<wgt id='macnlops'> (\S+) </wgt>", event.block)[1]
                    self.assertEqual(mc.number(weight), event.weight)
                self.assertEqual(kinds, {'S', 'H'})
                self.assertEqual(report['status'], 'complete')

    def test_negative_s_is_preserved_in_the_same_no_transfer_domain(self):
        self.env['MACNLOPS_STUB_BORN_PDF_SUPPORT'] = '1'
        path = self.work / 'events.lhe'
        with mc.LHEStream(path) as stream:
            header, blocks = stream.header, list(stream)
        for i, block in enumerate(blocks):
            if mc.Event(block).kind != 'S':
                continue
            lines = block.splitlines(keepends=True)
            fields = lines[1].split();fields[2] = '-' + fields[2].lstrip('+-')
            lines[1] = ' '.join(fields) + '\n';blocks[i] = ''.join(lines)
        path.write_text(header + ''.join(blocks) + '</LesHouchesEvents>\n')
        _, output, _ = self.run_match('negative_s_support')
        found = 0
        for event, decision in self.decisions(output):
            if event.kind == 'S' and decision['rule'] == 'pdf_support_no_transfer':
                self.assertLess(event.weight, 0.)
                self.assertEqual((decision['keep'], decision['accepted']), ('1', '1'))
                found += 1
        self.assertGreater(found, 0)

    def test_legacy_v2_v3_do_not_acquire_the_new_v4_support_rule(self):
        for version in ('v2', 'v3'):
            with self.subTest(version=version):
                self.env.pop('MACNLOPS_STUB_BORN_PDF_SUPPORT', None)
                before = self.run_match('legacy_' + version, '--s-veto', version)
                self.env['MACNLOPS_STUB_BORN_PDF_SUPPORT'] = '1'
                after = self.run_match('legacy_support_' + version, '--s-veto', version)
                self.assertEqual(before[:2], after[:2])
                self.assertFalse(any(d['rule'] == 'pdf_support_no_transfer'
                                     for _, d in self.decisions(after[1])))

    def native_support(self, scale):
        result = subprocess.run([str(self.native_probe), repr(scale)], cwd=self.work,
                                env=self.env, capture_output=True, text=True)
        if result.returncode == 77:
            self.skipTest('NNPDF40_nlo_as_01180_nf_4 is not installed')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(row for row in result.stdout.splitlines() if row.startswith('SUPPORT '))
        support, charm, gluon, gluon_born = line.split()[1:]
        return bool(int(support)), float(charm), float(gluon), bool(int(gluon_born))

    def test_exact_four_flavour_pdf_failure_and_nearby_positive_support(self):
        unsupported = self.native_support(1.7028105905461708)
        self.assertEqual(unsupported[0], False)
        self.assertEqual(unsupported[1], 0.)
        self.assertAlmostEqual(unsupported[2], 2.005076188159679, delta=2.e-12)
        positive = self.native_support(1.71)
        self.assertTrue(positive[0])
        self.assertGreater(positive[1], 0.)
        self.assertAlmostEqual(positive[1], 2.716191890459e-5, delta=2.e-12)
        # Zero charm as a real mother has a positive Born-gluon denominator;
        # the guard must not confuse that with this failed daughter case.
        self.assertTrue(unsupported[3])
        self.assertTrue(positive[3])


if __name__ == '__main__':
    unittest.main()
