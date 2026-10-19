"""IEEE validation must not add state handling to unrelated module users."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_born_spreading import mint_support, TEMPLATE


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestIEEEScope(unittest.TestCase):
    def test_module_users_do_not_save_floating_point_state(self):
        with tempfile.TemporaryDirectory(prefix='mg5_ieee_scope_') as tmp:
            work = Path(tmp)
            # Compile the production modules: checking only a USE statement
            # misses IEEE imports propagated through their .mod interfaces.
            mint_support(work)
            result = subprocess.run([
                'gfortran', '-O', '-c', 'params.f90',
                str(TEMPLATE / 'mint_module.f90'),
                str(TEMPLATE / 'qed_shower_support.f90'),
                str(TEMPLATE.parent / 'Source/BornSupport/mc_born_types.f90')],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            probes = {
                'mint': 'integer function probe()\n'
                        'use mint_module, only: iconfig\n'
                        'probe=iconfig\nend function\n',
                'qed': 'logical function probe(pdg)\n'
                       'use qed_shower_support, only: pythia8_qed_radiator\n'
                       'integer pdg\nprobe=pythia8_qed_radiator(pdg,1d0)\n'
                       'end function\n',
                'born': 'logical function probe(x)\n'
                        'use mc_born_types, only: born_is_finite\n'
                        'real(8) x\nprobe=born_is_finite(x)\nend function\n',
            }
            for name, source in probes.items():
                with self.subTest(module=name):
                    (work / 'probe.f90').write_text(source)
                    result = subprocess.run([
                        'gfortran', '-O', '-S', 'probe.f90'],
                        cwd=work, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn('ieee_procedure_', (work / 'probe.s').read_text())
