"""Check candidate thinning and signed LHE normalization for AmpliCol."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

from tests.unit_tests.fks.test_momentum_maps import TEMPLATE


DRIVER = """
program check_ampli_lhe
  use rewrite_under_test
  implicit none
  character(512) :: source,destination,arg
  double precision,allocatable :: weights(:)
  integer :: expected,i
  call get_command_argument(1,source)
  call get_command_argument(2,destination)
  call get_command_argument(3,arg)
  read(arg,*)expected
  allocate(weights(command_argument_count()-3))
  do i=1,size(weights)
    call get_command_argument(i+3,arg)
    read(arg,*)weights(i)
  enddo
  call ampli_rewrite_events(trim(source),trim(destination),weights,expected)
end program
"""


def candidate_file(weights):
    events = []
    for index, weight in enumerate(weights):
        events.append('''  <event npNLO="0" npLO="-1">
 2 42 %.16e 91.1876 0.00729735 0.118
 11 -1 0 0 0 0 0 0 45.0 45.0 0.0 0.0 9.0
 -11 1 1 1 0 0 0 0 -45.0 45.0 0.0 0.0 -9.0
 # candidate %d
 <rwgt>
  <wgt id="scale1"> 0.3141592653589793E+02 </wgt>
 </rwgt>
 <mgrwgt>
  17 1.23456789012345D+00
 </mgrwgt>
 </event>
''' % (weight, index))
    return '''<LesHouchesEvents version="3.0">
<header><generator>candidate test</generator></header>
<init>
 11 -11 45.0 45.0 0 0 0 0 -4 1
 20.0 0.1 20.0 42
</init>
''' + ''.join(events) + '</LesHouchesEvents>\n'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestAmpliLHE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_ampli_lhe_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.work = work = Path(cls.tempdir.name)
        source = (TEMPLATE / 'ampli_mint_adapter.f90').read_text()
        start = source.index('  subroutine ampli_rewrite_events(')
        end = source.index('  end subroutine ampli_rewrite_events', start)
        routine = source[start:end] + '  end subroutine ampli_rewrite_events\n'
        (work / 'rewrite.f90').write_text(
            'module rewrite_under_test\nimplicit none\ncontains\n' +
            routine + 'end module rewrite_under_test\n')
        (work / 'driver.f90').write_text(DRIVER)
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-fcheck=all',
            '-ffpe-trap=invalid,zero,overflow', str(work / 'rewrite.f90'),
            str(work / 'driver.f90'), '-o', str(work / 'check')],
            cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def rewrite(self, content, weights, expected):
        source = self.work / 'candidates.lhe'
        destination = self.work / 'events.lhe'
        source.write_text(content)
        result = subprocess.run([
            str(self.work / 'check'), str(source), str(destination),
            str(expected), *map(str, weights)], cwd=self.work,
            capture_output=True, text=True)
        return result, destination.read_text()

    def test_selection_preserves_sign_and_global_normalization(self):
        for normalization in (1.0, 0.2, 20.0):
            with self.subTest(normalization=normalization):
                original = candidate_file([normalization, -normalization, normalization])
                result, rewritten = self.rewrite(original, [0.5, 1.5, 0.0], 2)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                tree = ET.fromstring(rewritten)
                events = tree.findall('event')
                self.assertEqual(len(events), 2)
                for event, expected_weight in zip(events, [0.5*normalization, -1.5*normalization]):
                    header = event.text.splitlines()[1].split()
                    self.assertEqual(header[:2], ['2', '42'])
                    self.assertAlmostEqual(float(header[2]), expected_weight)
                    for actual, expected in zip(header[3:], [91.1876, 0.00729735, 0.118]):
                        self.assertAlmostEqual(float(actual), expected)
                self.assertNotIn('# candidate 2', rewritten)

    def test_retained_event_payloads_and_init_are_preserved(self):
        original = candidate_file([15.0, -25.0, 20.0])
        result, rewritten = self.rewrite(original, [2.0, 0.4, 0.0], 2)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        before, after = ET.fromstring(original), ET.fromstring(rewritten)
        self.assertEqual(ET.tostring(before.find('init')), ET.tostring(after.find('init')))
        self.assertEqual(ET.tostring(before.find('header')), ET.tostring(after.find('header')))
        for before_event, after_event in zip(before.findall('event'), after.findall('event')):
            self.assertEqual(before_event.attrib, after_event.attrib)
            self.assertEqual(before_event.text.splitlines()[2:], after_event.text.splitlines()[2:])
            for before_child, after_child in zip(before_event, after_event):
                self.assertEqual(ET.tostring(before_child), ET.tostring(after_child))
        self.assertAlmostEqual(float(after.findall('event')[0].text.split()[2]), 30.0)
        self.assertAlmostEqual(float(after.findall('event')[1].text.split()[2]), -10.0)

    def test_plain_event_tags_and_empty_final_sample(self):
        original = candidate_file([20.0]).replace(' npNLO="0" npLO="-1"', '')
        result, rewritten = self.rewrite(original, [0.0], 0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(ET.fromstring(rewritten).findall('event'), [])

    def test_event_count_mismatches_fail(self):
        for weights, expected, diagnostic in (
                ([1.0], 1, 'Too many AmpliCol candidate events'),
                ([1.0, 1.0, 0.0], 2, 'Incomplete AmpliCol candidate selection'),
                ([1.0, 0.0], 2, 'Incomplete AmpliCol candidate selection')):
            with self.subTest(weights=weights, expected=expected):
                result, unused = self.rewrite(candidate_file([20.0, -20.0]), weights, expected)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stdout + result.stderr)

    def test_truncated_event_and_invalid_header_fail(self):
        original = candidate_file([20.0])
        for content, diagnostic in (
                (original[:original.index(' </event>')], 'Incomplete AmpliCol candidate selection'),
                (original.replace(' 2 42 ', ' invalid header '), 'Invalid AmpliCol candidate event header')):
            with self.subTest(diagnostic=diagnostic):
                result, unused = self.rewrite(content, [1.0], 1)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stdout + result.stderr)
