"""Numerical checks of Born-spreading normalization and MINT's restart."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'


def mint_support(work, real_weight_lines=False):
    (work / 'params.f90').write_text(
        'module FKSParams\n'
        'logical :: use_poly_virtual=.false.\n'
        'double precision :: virt_fraction=1d0,min_virt_fraction=0.01d0\n'
        'end module\n'
        'module mc_native_context\n'
        'logical :: native_mapping=.false.\n'
        'end module\n')
    # These deterministic checks need neither plots nor FKS sampling.
    unused = ('accum', 'hwu_accum_iter', 'hwu_add_points', 'initplot',
              'empty_mc_integer', 'regrid_mc_integer', 'reset_mc_grid')
    if not real_weight_lines:
        unused += ('deallocate_born_spread_lines',)
    polyfit = ('add_point_polyfit', 'do_polyfit', 'get_polyfit',
               'init_polyfit', 'restore_polyfit', 'save_polyfit')
    (work / 'stubs.f90').write_text(
        ''.join('subroutine %s\nend subroutine\n' % name for name in unused) +
        ''.join('subroutine %s\nstop 99\nend subroutine\n' % name for name in polyfit))
    return [str(work / 'params.f90'), str(TEMPLATE / 'mint_module.f90'),
            str(work / 'stubs.f90')]


def routine(path, name):
    """Compile the production routines without unrelated matrix elements."""
    pattern = r'^      (?:subroutine|(?:double precision|logical) function) ' + name + r'\b.*?^      end\s*$'
    return re.search(pattern, path.read_text(), re.M | re.S | re.I).group(0) + '\n'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestBornSpreading(unittest.TestCase):
    def test_sector_fit_grouping_and_application(self):
        """Conflicting sector optima stay normalized and ignore residual virtuals."""
        with tempfile.TemporaryDirectory(prefix='mg5_born_sectors_') as tmp:
            work = Path(tmp)
            support = mint_support(work, real_weight_lines=True)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=5,nincoming=2)',
                'genps.inc': 'integer maxproc\nparameter(maxproc=2)',
                'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=3)',
                'fks_info.inc': 'integer pdg_type_d(3,5),fks_i_d(3)\n'
                                'common/test_fks_info/pdg_type_d,fks_i_d',
            }
            for name, source in includes.items():
                (work / name).write_text(''.join('      ' + line + '\n'
                                               for line in source.splitlines()))
            shutil.copyfile(TEMPLATE / 'timing_variables.inc', work / 'timing_variables.inc')
            # Only S grouping is exercised; these supply the unused H metadata.
            (work / 'group_context.f90').write_text(
                'module process_module\ninteger :: ndelH=1\n'
                'integer :: event_colour_H(2,5,3,2)=0\nend module\n'
                'module scale_module\ndouble precision :: emsca_H(3,2,1,1)=1d0\nend module\n')
            (work / 'grouping.f').write_text(''.join(
                routine(TEMPLATE / 'fks_singular.f', name) for name in
                ('sum_identical_contributions', 'apply_born_spread_weight',
                 'pdg_equal', 'momenta_equal')))
            executable = work / 'check_sectors'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all', '-fno-automatic',
                '-ffixed-line-length-132'] + support + [
                str(work / 'group_context.f90'), str(TEMPLATE / 'weight_lines.f'),
                str(work / 'grouping.f'),
                str(ROOT / 'tests/input_files/check_born_spreading_sectors.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], cwd=work, capture_output=True,
                                    text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS sector fits and S grouping', result.stdout)
            table = work / 'born_spreading.dat'
            saved = table.read_text().splitlines()
            # An overall mean of one cannot compensate a biased sector.
            lines = saved[:4]
            for factor in (0.5, 1.5, 1.0):
                lines.extend([' '.join([str(factor)] * 40)] * 40)
            table.write_text('\n'.join(lines) + '\n')
            result = subprocess.run([str(executable), 'load'], cwd=work,
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('sector normalization constraint', result.stdout)
            for version in (1, 2):
                with self.subTest(obsolete_version=version):
                    saved[0] = 'BORN_SPREAD %d' % version
                    table.write_text('\n'.join(saved) + '\n')
                    result = subprocess.run([str(executable), 'load'], cwd=work,
                                            capture_output=True, text=True, timeout=30)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('rerun integration step 0', result.stdout)

    def test_restart_preserves_integral_normalization(self):
        """The first restarted iteration must average its sampled weights."""
        with tempfile.TemporaryDirectory(prefix='mg5_born_spread_') as tmp:
            work = Path(tmp)
            executable = work / 'check_restart'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all', '-fno-automatic'] +
                mint_support(work) + [
                str(ROOT / 'tests/input_files/check_born_spreading_restart.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for channels, enabled in ((1, 0), (1, 1), (2, 1)):
                with self.subTest(channels=channels, enabled=enabled):
                    result = subprocess.run(
                        [str(executable), str(channels), str(enabled)],
                        cwd=work, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0,
                                     result.stdout + result.stderr)
                    self.assertIn('PASS MINT restart normalization', result.stdout)

    def test_born_radiation_measure(self):
        """A fitted factor must preserve ISR and massless/massive FSR Born weights."""
        with tempfile.TemporaryDirectory(prefix='mg5_born_measure_') as tmp:
            work = Path(tmp)
            support = mint_support(work)
            (work / 'nexternal.inc').write_text(
                '      integer nexternal,nincoming\n'
                '      parameter (nexternal=5,nincoming=2)\n')
            maps = ('generate_momenta_initial', 'generate_momenta_massless_final',
                    'generate_momenta_massive_final', 'getangles')
            source = ''.join(routine(TEMPLATE / 'genps_fks.f', name) for name in maps)
            source += routine(TEMPLATE / 'fks_singular.f', 'set_born_spread_point')
            source += routine(TEMPLATE / 'fks_singular.f', 'rotate_invar')
            source += routine(ROOT / 'Template/NLO/Source/kin_functions.f', 'rho')
            source += routine(ROOT / 'Template/NLO/Source/kin_functions.f', 'threedot')
            (work / 'maps.f').write_text(source)
            executable = work / 'check_measure'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all', '-fno-automatic',
                '-ffixed-line-length-132'] + support + [str(work / 'maps.f'),
                str(TEMPLATE / 'boostwdir2.f'),
                str(ROOT / 'tests/input_files/check_born_spreading_measure.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], cwd=work, capture_output=True,
                                    text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS Born radiation normalization', result.stdout)
            # Even a unit table in the obsolete measure must require a refit.
            table = work / 'born_spreading.dat'
            lines = table.read_text().splitlines()
            lines[0] = 'BORN_SPREAD 1'
            lines[4:] = [' '.join(['1.0'] * 40)] * 40
            table.write_text('\n'.join(lines) + '\n')
            result = subprocess.run([str(executable), 'load'], cwd=work,
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('rerun integration step 0', result.stdout)
