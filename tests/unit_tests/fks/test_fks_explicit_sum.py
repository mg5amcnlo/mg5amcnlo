"""Outer FKS summation, sampling normalization and signed unweighting."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_born_spreading import mint_support
from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestFKSExplicitSum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_fks_sum_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.work = work = Path(cls.tempdir.name)
        includes = {
            'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=5,nincoming=2)',
            'nFKSconfigs.inc': 'integer fks_configs,fks_integrated\n'
                               'parameter(fks_configs=7,fks_integrated=6)',
            'genps.inc': 'integer maxproc\nparameter(maxproc=1)',
            'run.inc': 'integer ickkw\ndouble precision xbk(2)\ncommon/test_run/xbk,ickkw',
            'orders.inc': 'integer nsplitorders\nparameter(nsplitorders=1)',
        }
        for name, source in includes.items():
            (work / name).write_text(''.join('      ' + line + '\n'
                                           for line in source.splitlines()))
        (work / 'map.f').write_text(fortran_routine(
            TEMPLATE / 'driver_mintMC.f', 'setup_proc_map'))
        (work / 'explicit.dat').write_text('#FKSExplicitSum\n.true.\n')
        (work / 'sampled.dat').write_text('#FKSExplicitSum\n.false.\n')
        (work / 'old.dat').write_text('#MCExplicitKLSum\n.true.\n')
        (work / 'legacy_matching.dat').write_text(
            '#FKSExplicitSum\n.true.\n#MCExplicitKLSum\n.false.\n')
        # A stale multi-bin grid must never supply a nonunit explicit-sum weight.
        (work / 'grid.MC_integer').write_text('AVE 0 0.25 0.75 1\n' * 50)
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-fcheck=all', '-fno-automatic',
            '-ffixed-line-length-132', '-I', str(work),
            str(TEMPLATE / 'FKSParams.f90'), str(work / 'map.f'),
            str(TEMPLATE / 'MC_integer.f'),
            str(ROOT / 'tests/input_files/check_fks_explicit_sum.f90'),
            '-o', str(work / 'check')], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def run_mode(self, *args):
        return subprocess.run([str(self.work / 'check'), *map(str, args)],
                              cwd=self.work, capture_output=True, text=True, timeout=30)

    def test_card_defaults_and_overrides(self):
        result = self.run_mode('card', TEMPLATE.parent / 'Cards/FKS_params.dat')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS card', result.stdout)
        self.assertIn('FKSExplicitSum', result.stdout)

    def test_sector_coverage_and_unit_sampling_weight(self):
        for channel in (0, 1, 2):
            with self.subTest(channel=channel):
                result = self.run_mode('map', channel)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('PASS map and sampling', result.stdout)

    def test_unlops_rejects_explicit_sum(self):
        result = self.run_mode('unlops')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FKSExplicitSum is not supported for UNLOPS', result.stdout)

    def test_common_born_chart_with_jet_cuts(self):
        with tempfile.TemporaryDirectory(prefix='mg5_fks_sum_chart_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=5,nincoming=2)',
                'genps.inc': 'integer lmaxconfigs,max_branch,maxproc,maxflow\n'
                             'parameter(lmaxconfigs=1,max_branch=2,maxproc=1,maxflow=1)',
                'nFKSconfigs.inc': 'integer fks_configs,fks_integrated\n'
                                   'parameter(fks_configs=3,fks_integrated=2)',
                'run.inc': 'double precision ebeam(2)\nparameter(ebeam=[6500d0,6500d0])',
                'cuts.inc': 'double precision ptj,ptgmin,ptl,mll,mll_sf\n'
                            'parameter(ptj=30d0,ptgmin=0d0,ptl=0d0,mll=0d0,mll_sf=0d0)',
                'orders.inc': 'integer nsplitorders\nparameter(nsplitorders=1)',
                'coupl.inc': '',
                'born_props.inc': 'pmass=0d0\npwidth=0d0',
                'fks_info.inc': 'integer fks_i_d(3),fks_j_d(3)\n'
                                'parameter(fks_i_d=[5,5,5],fks_j_d=[1,3,2])',
            }
            for name, source in includes.items():
                (work / name).write_text(''.join('      ' + line + '\n'
                                               for line in source.splitlines()))
            (work / 'contexts.f90').write_text(
                'module mint_module\nuse FKSParams\n'
                'integer,parameter :: maxchannels=1\ninteger :: iconfig=1,ichan=1\n'
                'logical :: nlo_ps=.true.\nend module\n'
                'module mc_native_context\n'
                'integer :: native_epoch=1,active_context=1,native_context_ids(3)=[1,1,2]\n'
                'logical :: native_mapping=.false.\nend module\n')
            (work / 'chart.f').write_text(fortran_routine(TEMPLATE / 'setcuts.f', 'set_tau_min'))
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all', '-fno-automatic',
                '-ffixed-line-length-132', '-I', str(work), str(TEMPLATE / 'FKSParams.f90'),
                str(TEMPLATE / 'fks_phase_space_data.f'), str(work / 'contexts.f90'),
                str(work / 'chart.f'),
                str(ROOT / 'tests/input_files/check_fks_sum_born_chart.f90'),
                '-o', str(work / 'check')], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS common Born chart and local recoil', result.stdout)

    def test_negative_weight_cancellation_before_unweighting(self):
        with tempfile.TemporaryDirectory(prefix='mg5_fks_sum_weights_') as tmp:
            work = Path(tmp)
            support = mint_support(work, real_weight_lines=True)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=5,nincoming=2)',
                'genps.inc': 'integer maxproc\nparameter(maxproc=1)',
                'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=2)',
                'orders.inc': 'integer amp_split_size\nparameter(amp_split_size=1)',
                'fks_info.inc': 'integer pdg_type_d(2,5),fks_i_d(2)\n'
                                'common/test_fks_info/pdg_type_d,fks_i_d',
            }
            for name, source in includes.items():
                (work / name).write_text(''.join('      ' + line + '\n'
                                               for line in source.splitlines()))
            shutil.copyfile(TEMPLATE / 'timing_variables.inc', work / 'timing_variables.inc')
            (work / 'contexts.f90').write_text(
                'module process_module\ninteger :: ndelH=1\n'
                'integer :: event_colour_H(2,5,2,2)=0\nend module\n'
                'module scale_module\nreal(8) :: emsca_H(2,2,1,1)=1d0\nend module\n')
            (work / 'weights.f').write_text('\n'.join(fortran_routine(
                TEMPLATE / 'fks_singular.f', name) for name in
                ('sum_identical_contributions', 'fill_mint_function_NLOPS',
                 'pdg_equal', 'momenta_equal')))
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all', '-fno-automatic',
                '-ffixed-line-length-132', *support, str(work / 'contexts.f90'),
                str(TEMPLATE / 'weight_lines.f'), str(work / 'weights.f'),
                str(ROOT / 'tests/input_files/check_fks_sum_weights.f90'),
                '-o', str(work / 'check')], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS signed integral and negative fractions', result.stdout)
