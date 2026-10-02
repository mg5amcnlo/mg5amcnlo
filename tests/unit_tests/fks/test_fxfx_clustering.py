"""Compiled regressions for FxFx resonance selection and 2 -> 1 processes."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestFxFxClustering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_fxfx_clustering_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        includes = {
            'run.inc': ('      integer lpp(2)\n'
                        '      double precision bwcutoff\n'
                        '      common/test_run/bwcutoff,lpp\n'),
            'cuts.inc': ('      integer maxjetflavor\n'
                         '      parameter(maxjetflavor=5)\n'),
            'nexternal.inc': ('      integer nexternal,nincoming\n'
                             '      parameter(nexternal=4,nincoming=2)\n'),
            'maxconfigs.inc': ('      integer lmaxconfigs\n'
                              '      parameter(lmaxconfigs=2)\n'),
            'maxparticles.inc': ('      integer max_branch\n'
                                '      parameter(max_branch=2)\n'),
            'nFKSconfigs.inc': ('      integer fks_configs\n'
                               '      parameter(fks_configs=1)\n'),
            'real_from_born_configs.inc': (
                '      integer real_from_born_conf(2,1)\n'
                '      parameter(real_from_born_conf=reshape([1,2],[2,1]))\n'),
        }
        for filename, contents in includes.items():
            (work / filename).write_text(contents)
        # Exported topology and model data are supplied by the fixture. Retain
        # the production wrapper, clustering, reweighting and kinematic code.
        (work / 'state.f90').write_text(
            'module fks_phase_space_data\n'
            'double precision :: p_born(0:3,3),p_ev(0:3,4)\nend module\n'
            'module mc_native_context\n'
            'integer :: native_epoch=0\nend module\n'
            'module weight_lines\n'
            'integer :: pdg_uborn(4,0:0),pdg(4,0:0)\nend module\n')
        names = (
            'cluster_and_reweight', 'set_array_indices', 'iforest_to_list',
            'cluster', 'Reweighting', 'set_particle_type', 'reset_valid_confs',
            'limit_cluster_iconfig', 'IsBreitWigner', 'remove_confs_BW',
            'cluster_one_step', 'update_valid_confs', 'update_momenta',
            'update_imap', 'set_cluster_conf', 'link_clustering_to_iforest',
            'set_cluster_pdg_2_1_process', 'update_cluster_scales', 'in_list',
            'cluster_scale', 'get_clustering_type', 'QCDchangeline',
            'QCDvertex', 'numberQCDcharged', 'startQCDvertex', 'IR_cluster',
            'matching_particles', 'fill_type', 'get_type', 'update_type',
            'djb_clus', 'dj_clus', 'crossp', 'rotate', 'constr')
        routines = [fortran_routine(TEMPLATE / 'cluster.f', name)
                    for name in names]
        routines += [fortran_routine(ROOT / 'Template/NLO/Source/kin_functions.f', name)
                     for name in ('dot', 'SumDot')]
        (work / 'clustering.f').write_text('\n'.join(routines))
        cls.executable = work / 'check_clustering'
        result = subprocess.run([
            shutil.which('gfortran'), '-O0', '-g', '-std=legacy',
            '-ffixed-line-length-none', '-fcheck=all',
            '-finit-integer=-777', '-finit-real=snan',
            '-ffpe-trap=invalid,zero,overflow', '-I', str(work),
            str(work / 'state.f90'), str(work / 'clustering.f'),
            str(ROOT / 'HELAS/boostx.F'),
            str(ROOT / 'tests/input_files/check_fxfx_clustering.f90'),
            '-o', str(cls.executable)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def check_clustering(self, mode):
        result = subprocess.run([str(self.executable), mode],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS ' + mode, result.stdout)

    def test_winning_resonance_keeps_its_invariant_mass(self):
        self.check_clustering('resonance')

    def test_later_resonance_does_not_change_winning_nonresonance(self):
        self.check_clustering('nonresonance')

    def test_initial_state_candidate_clears_resonance_flag(self):
        self.check_clustering('initial_state')

    def test_zero_branching_clustering(self):
        self.check_clustering('zero')

    def test_zero_branching_wrapper_scales_and_weights(self):
        self.check_clustering('wrapper')

    def test_nonzero_branching_qcd_core_scale(self):
        self.check_clustering('qcd_core')
