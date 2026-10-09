"""Native MC subtraction cuts must use the underlying Born jet definition."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran') and shutil.which('g++'),
                     'requires gfortran and g++')
class TestMCBornCuts(unittest.TestCase):
    def test_native_born_cuts_and_real_state_restoration(self):
        with tempfile.TemporaryDirectory(prefix='mg5_mc_born_cuts_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=7,nincoming=2)',
                'run.inc': 'integer ickkw\ncommon/test_run/ickkw',
                'cuts.inc': 'integer maxjetflavor\nparameter(maxjetflavor=5)\n'
                            'double precision ptj,ptgmin\nlogical gamma_is_j\n'
                            'common/test_cuts/ptj,ptgmin,gamma_is_j',
            }
            for name, source in includes.items():
                (work / name).write_text(''.join(
                    '      ' + line + '\n' for line in source.splitlines()))
            routines = [fortran_routine(TEMPLATE / 'driver_mintMC.f',
                                        'passcuts_native_born')]
            routines += [fortran_routine(TEMPLATE / 'cuts.f', name)
                         for name in ('identify_QCD_partons', 'passcuts_fxfx')]
            (work / 'cuts.f').write_text('\n'.join(routines))
            executable = work / 'check_mc_born_cuts'
            # Use the real jet finder and cut routines. The fixture supplies
            # native clustering metadata and the particle IDs, as an exported
            # process would, without needing a generated W+2j process in CI.
            commands = [
                ['g++', '-O1', '-std=c++11', '-c',
                 str(TEMPLATE / 'fastjetfortran_madfks_core.cc'),
                 str(TEMPLATE / 'fjcore.cc')],
                ['gfortran', '-O2', '-fcheck=all', '-fbacktrace',
                 '-ffixed-line-length-none', '-I', str(work), '-I', str(TEMPLATE),
                 str(work / 'cuts.f'),
                 str(ROOT / 'tests/input_files/check_mc_born_cuts.f90'),
                 str(TEMPLATE / 'fastjet_wrapper.f'),
                 'fastjetfortran_madfks_core.o', 'fjcore.o', '-lstdc++',
                 '-o', str(executable)],
                [str(executable)],
            ]
            for command in commands:
                result = subprocess.run(command, cwd=work, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS native Born cuts', result.stdout)
