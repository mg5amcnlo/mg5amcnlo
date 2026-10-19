"""Compiled checks of clustering reuse, candidate lookup and scale requests."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_fxfx_clustering import CLUSTER_ROUTINES
from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestClusterOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_cluster_optimization_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.work = work = Path(cls.tempdir.name)
        includes = {
            'run.inc': 'integer lpp(2)\ndouble precision bwcutoff\n'
                       'common/test_run/bwcutoff,lpp',
            'cuts.inc': 'integer maxjetflavor\nparameter(maxjetflavor=5)',
            'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=6,nincoming=2)',
            'maxconfigs.inc': 'integer lmaxconfigs\nparameter(lmaxconfigs=70)',
            'maxparticles.inc': 'integer max_branch\nparameter(max_branch=4)',
            'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=2)',
            'real_from_born_configs.inc': 'integer real_from_born_conf(2,2)\n'
                'parameter(real_from_born_conf=reshape([1,2,2,1],[2,2]))',
        }
        for name, source in includes.items():
            (work / name).write_text(''.join('      ' + line + '\n'
                                           for line in source.splitlines()))
        shutil.copyfile(TEMPLATE / 'timing_variables.inc', work / 'timing_variables.inc')
        (work / 'state.f90').write_text(
            'module fks_phase_space_data\n'
            'double precision :: p_born(0:3,5),p_ev(0:3,6)\nend module\n'
            'module mc_native_context\ninteger :: active_context=1,native_epoch=1\n'
            'integer,parameter :: context_providers(2)=[1,2]\nend module\n'
            'module weight_lines\ninteger :: pdg_uborn(6,0:0),pdg(6,0:0)\nend module\n')
        cached = fortran_routine(TEMPLATE / 'cluster.f', 'cluster_and_reweight')
        fresh = cached.replace('subroutine cluster_and_reweight(',
                               'subroutine cluster_and_reweight_fresh(')
        # Force the reference to rebuild all derived tables at every call,
        # independently of the production cache's identity and mass checks.
        fresh = fresh.replace('      if (.not.allocated(table%ipdg)) then',
            '      if (allocated(table%ipdg)) deallocate(table%ipdg,\n'
            '     $ table%cluster_list,table%cluster_pdg,table%cluster_type,\n'
            '     $ table%config_masks,table%qcd_mask,\n'
            '     $ table%type_pdg,table%type_mass)\n'
            '      if (.not.allocated(table%ipdg)) then')
        for name, source, tag in [('cached.f', cached, 1), ('fresh.f', fresh, 2)]:
            source = source.replace('            call iforest_to_list(',
                '            call count_cluster_builds(%d)\n' % tag +
                '            call iforest_to_list(')
            (work / name).write_text(source)
        routines = [fortran_routine(TEMPLATE / 'cluster.f', name)
                    for name in CLUSTER_ROUTINES[1:] + ('QCDsudakov',)]
        routines += [fortran_routine(ROOT / 'Template/NLO/Source/kin_functions.f', name)
                     for name in ('dot', 'SumDot')]
        (work / 'clustering.f').write_text('\n'.join(routines))
        cls.executable = work / 'check_cluster_optimization'
        result = subprocess.run([
            shutil.which('gfortran'), str(TEMPLATE.parent / 'Source/timing.f'), '-O2', '-g', '-std=legacy',
            '-ffixed-line-length-132', '-fcheck=all', '-fbacktrace',
            '-finit-integer=-777', '-finit-real=snan', '-ffpe-trap=invalid,zero,overflow',
            '-I', str(work), str(work / 'state.f90'), str(work / 'cached.f'),
            str(work / 'fresh.f'), str(work / 'clustering.f'), str(ROOT / 'HELAS/boostx.F'),
            str(ROOT / 'tests/input_files/check_cluster_optimization.f90'),
            '-o', str(cls.executable)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def check_clustering(self, mode):
        result = subprocess.run([str(self.executable), mode], cwd=self.work,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS cluster optimization ' + mode, result.stdout)

    def test_tables_survive_context_sector_and_channel_switches(self):
        self.check_clustering('reuse')

    def test_model_masses_refresh_particle_types(self):
        self.check_clustering('mass')

    def test_resonance_inputs_and_beam_measure_remain_live(self):
        self.check_clustering('resonance')

    def test_shower_scale_requests_skip_sudakov_bookkeeping(self):
        self.check_clustering('scales')

    def test_candidate_membership_preserves_order_and_resonances(self):
        self.check_clustering('candidates')

    def test_configuration_masks_cross_machine_word_boundaries(self):
        self.check_clustering('words')

    def test_vertex_masks_distinguish_coloured_mother_and_daughters(self):
        self.check_clustering('vertices')
