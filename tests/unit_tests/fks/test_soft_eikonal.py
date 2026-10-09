"""Small, resolved native soft products must not be rounded to zero."""

from decimal import Decimal, localcontext
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestSoftEikonal(unittest.TestCase):
    def test_captured_native_soft_point_and_energy_scaling(self):
        with tempfile.TemporaryDirectory(prefix='mg5_soft_eikonal_') as tmp:
            work = Path(tmp)
            (work / 'nexternal.inc').write_text(
                '      integer nexternal,nincoming\n'
                '      parameter(nexternal=7,nincoming=2)\n')
            (work / 'coupl.inc').write_text('')
            (work / 'pmass.inc').write_text('      pmass=0d0\n')
            (work / 'geometry.f').write_text(
                '      module fks_phase_space_helpers\n      contains\n' +
                fortran_routine(TEMPLATE / 'genps_fks_helpers.f', 'dot') +
                '      end module\n')
            (work / 'eikonal.f').write_text(
                fortran_routine(TEMPLATE / 'fks_singular.f', 'eikonal_reduced') +
                fortran_routine(ROOT / 'Template/NLO/Source/kin_functions.f', 'dot'))
            executable = work / 'check'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all',
                '-ffixed-line-length-none', '-I', str(work),
                str(TEMPLATE / 'fks_phase_space_data.f'),
                str(work / 'geometry.f'), str(work / 'eikonal.f'),
                str(ROOT / 'tests/input_files/check_soft_eikonal.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # Independent high-precision oracle from the captured vectors.
            with localcontext() as ctx:
                ctx.prec = 60
                p = list(map(Decimal, ('1.9412798495029867e-4', '4.2281074989622250e-5',
                                      '1.0792948936341049e-4', '-1.5572158027819647e-4')))
                q = list(map(Decimal, ('175.38639907330929', '38.668406976089592',
                                      '98.371914059055683', '-139.95752857989231')))
                denominator = p[0]*q[0] - sum(a*b for a, b in zip(p[1:], q[1:]))
                reference = float((p[0]-p[3]) / (q[0]*denominator))
            self.assertGreater(denominator, 0)
            self.assertLess(denominator, Decimal('1e-6'))
            rows = [line.split() for line in result.stdout.splitlines() if line.startswith('RESULT')]
            self.assertEqual(len(rows), 3)
            for _, scale, forward, reverse, general in rows:
                scale = float(scale)
                for value in (forward, reverse):
                    self.assertAlmostEqual(float(value)*scale**2/reference, 1.0, delta=2e-11)
                # For m=5,n=6,j=1 with leg 6 along -z, no denominator is
                # replaced by the analytic FKS-collinear limit.
                with localcontext() as ctx:
                    ctx.prec = 60
                    general_ref = float((p[0]+p[3])*(1+Decimal('0.79799533669307987')) /
                                        ((q[0]+q[3])*denominator))
                self.assertAlmostEqual(float(general)*scale**2/general_ref, 1.0, delta=2e-11)
