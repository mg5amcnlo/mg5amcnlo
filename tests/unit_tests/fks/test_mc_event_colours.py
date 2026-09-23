"""Keep sampled H colours with the selected outer sector and fold."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMCEventColours(unittest.TestCase):
    def test_sector_fold_and_native_colour_ownership(self):
        with tempfile.TemporaryDirectory(prefix='mg5_mc_colours_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=5,nincoming=2)',
                'genps.inc': 'integer maxproc,maxflow\nparameter(maxproc=1,maxflow=1)',
                'born_nhel.inc': 'integer max_bcol\nparameter(max_bcol=1)',
                'born_leshouche.inc': 'common/test_born_colours/idup,mothup,icolup',
                'orders.inc': 'integer nsplitorders,qcd_pos,qed_pos\n'
                              'parameter(nsplitorders=2,qcd_pos=1,qed_pos=2)',
                'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=2)',
                'fks_info.inc': 'integer pdg_type_d(2,5),fks_i_d(2)\n'
                                'logical need_color_links_d(2),need_charge_links_d(2)',
            }
            for name, text in includes.items():
                (work / name).write_text(''.join('      '+s+'\n' for s in text.splitlines()))
            for name in ('timing_variables.inc', 'fks_powers.inc'):
                shutil.copyfile(TEMPLATE / name, work / name)
            (work / 'contexts.f90').write_text(
                'module mc_native_context\ninteger :: history_flavours(1,1)=1\nend module\n'
                'module mint_module\ninteger :: imode=0\nlogical :: only_virt=.false.\nend module\n')
            routines = [('driver_mintMC.f', 'init_process_module_n1body_wrapper'),
                        ('add_write_info.f', 'fill_icolor_H'),
                        ('add_write_info.f', 'fill_icolor_S'),
                        ('fks_singular.f', 'pick_unweight_contr'),
                        ('fks_singular.f', 'sum_identical_contributions'),
                        ('fks_singular.f', 'pdg_equal'),
                        ('fks_singular.f', 'momenta_equal')]
            (work / 'colours.f').write_text('\n'.join(
                fortran_routine(TEMPLATE / filename, name) for filename, name in routines))
            executable = work / 'check_colours'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
                '-ffixed-line-length-none', '-ffunction-sections', '-fdata-sections',
                '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                '-I', str(work), str(work / 'contexts.f90'),
                str(TEMPLATE / 'process_module.f90'),
                str(TEMPLATE / 'kinematics_module.f90'),
                str(TEMPLATE / 'scale_module.f90'), str(TEMPLATE / 'weight_lines.f'),
                str(work / 'colours.f'),
                str(ROOT / 'tests/input_files/check_mc_event_colours.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS event colours', result.stdout)
