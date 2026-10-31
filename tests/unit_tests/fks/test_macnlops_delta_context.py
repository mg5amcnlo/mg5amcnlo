"""Small conditional-Delta fixtures; no exported process or shower runtime.

Only extracted routines and controlled contribution stubs are compiled.
The tests need gfortran, but no PDFs, Sudakov tables or generated process.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMacNLOPSDeltaContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='macnlops-delta-context-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        source = (TEMPLATE / 'handling_lhe_events.f').read_text()
        (cls.work / 'packets.f').write_text(source[source.index(
            '      subroutine clear_macnlops_hard_event()'):])
        cls.executable = cls.work / 'check'
        result = subprocess.run(['gfortran', '-O0', '-fcheck=all',
            '-ffixed-line-length-none', str(TEMPLATE / 'weight_lines.f'),
            str(cls.work / 'packets.f'),
            str(ROOT / 'tests/input_files/check_macnlops_delta_context.f90'),
            '-o', str(cls.executable)], cwd=cls.work, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def invoke(self, mode, data=''):
        return subprocess.run([str(self.executable), str(mode)], cwd=self.work,
            input=data, capture_output=True, text=True, timeout=10)

    def test_context_growth_replay_and_zero_mid_one_transfer(self):
        result = self.invoke(0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS conditional Delta context', result.stdout)

    def test_missing_duplicate_nonfinite_or_repeated_keys_fail(self):
        for mode in range(1, 6):
            with self.subTest(mode=mode):
                self.assertNotEqual(self.invoke(mode).returncode, 0)

    def test_complete_and_empty_packets_preserved_by_generic_reader(self):
        for mode in (7, 8):
            with self.subTest(mode=mode):
                written = self.invoke(mode)
                self.assertEqual(written.returncode, 0, written.stderr)
                self.assertIn('source="delta" partition="global_sum"', written.stdout)
                self.assertEqual(written.stdout.count('<macnlops_delta_context'), 1)
                replayed = self.invoke(6, written.stdout)
                self.assertEqual(replayed.returncode, 0, replayed.stderr)
                self.assertEqual(replayed.stdout, written.stdout)

    def test_winner_source_survives_native_collect_roundtrip(self):
        written = self.invoke(10)
        self.assertEqual(written.returncode, 0, written.stderr)
        self.assertIn('source="delta_winner"', written.stdout)
        replayed = self.invoke(6, written.stdout)
        self.assertEqual(replayed.returncode, 0, replayed.stderr)
        self.assertEqual(replayed.stdout, written.stdout)

    def test_native_ir_source_survives_native_collect_roundtrip(self):
        written = self.invoke(11)
        self.assertEqual(written.returncode, 0, written.stderr)
        self.assertIn('source="delta_native_ir"', written.stdout)
        replayed = self.invoke(6, written.stdout)
        self.assertEqual(replayed.returncode, 0, replayed.stderr)
        self.assertEqual(replayed.stdout, written.stdout)

    def test_malformed_probability_or_registry_fails(self):
        source = self.invoke(7).stdout
        rows = source.splitlines()
        first_probability = next(i for i, line in enumerate(rows)
                                 if '<macnlops_delta_context ' in line) + 1
        fields = rows[first_probability].split()
        fields[-1] = 'NaN'
        rows[first_probability] = ' '.join(fields)
        for invalid in (source.replace('registry="' + 'a' * 64,
                                       'registry="' + 'g' * 64),
                        '\n'.join(rows) + '\n',
                        source.replace('</macnlops_delta_context>', '')):
            with self.subTest(packet=invalid[-160:]):
                self.assertNotEqual(self.invoke(6, invalid).returncode, 0)

    def test_delta_scales_accept_old_inline_and_full_precision_multiline(self):
        value = .31415926535897931
        for data in ("<scales scalup_1_2='3.14159265E-01'>\n</scales>\n",
                     "<scales muf='-1' mur='-1'\n"
                     " scalup_12_1='%.17e'\n>\n</scales>\n" % value):
            result = self.invoke(9, data)
            self.assertEqual(result.returncode, 0, result.stderr)
            i, j, parsed = result.stdout.split()
            self.assertEqual((int(i), int(j)), (1, 2) if '1_2=' in data else (12, 1))
            self.assertEqual(float(parsed), .314159265 if int(i) == 1 else value)
        for invalid in ("<scales scalup_13_1='1'>\n</scales>\n",
                        "<scales scalup_1_2='NaN'>\n</scales>\n",
                        "<scales scalup_1_2='1' scalup_1_2='2'>\n</scales>\n",
                        "<scales scalup_1_2='1'>\n"):
            self.assertNotEqual(self.invoke(9, invalid).returncode, 0)


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMacNLOPSDeltaSource(unittest.TestCase):
    def test_native_wrapper_preserves_delta_compensation_and_ordinary_limit(self):
        from tests.unit_tests.fks.test_momentum_maps import fortran_routine
        with tempfile.TemporaryDirectory(prefix='macnlops-delta-source-') as temporary:
            work = Path(temporary)
            includes = {
                'nexternal.inc': 'integer,parameter :: nexternal=5,nincoming=2',
                'orders.inc': '''integer,parameter :: amp_split_size=1,nsplitorders=1,
     $ qcd_pos=1,cpower_pos=0''',
                'coupl.inc': 'double precision g\nparameter(g=1d0)',
                'timing_variables.inc': '''logical,parameter :: timing_sampled=.false.
integer,parameter :: timing_mc=1''',
                'run.inc': '''logical macnlops,macnlops_resolved_source
common/test_source_mode/macnlops,macnlops_resolved_source
integer ickkw
parameter(ickkw=0)''',
            }
            for name, text in includes.items():
                (work / name).write_text('\n'.join(
                    row if row.startswith('     $') else '      ' + row
                    for row in text.splitlines()) + '\n')
            (work / 'counterterm.f').write_text('\n'.join(fortran_routine(
                TEMPLATE / 'fks_singular.f', name) for name in (
                    'compute_MC_subt_term', 'compute_native_NLOPS_weights')))
            result = subprocess.run(['gfortran', '-O0', '-fcheck=all',
                '-ffixed-line-length-none', '-I.', str(TEMPLATE / 'weight_lines.f'),
                str(ROOT / 'tests/input_files/check_macnlops_delta_source.f90'),
                str(work / 'counterterm.f'), '-o', str(work / 'check')],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS native Delta MC term', result.stdout)
            self.assertIn('PASS native Delta wrapper', result.stdout)
            self.assertIn('PASS native ordinary bogus-P wrapper', result.stdout)


if __name__ == '__main__':
    unittest.main()
