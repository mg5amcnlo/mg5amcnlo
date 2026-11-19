"""Cache tau bounds and recoil metadata across native chart switches."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestTauCutCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_tau_cut_cache_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.work = work = Path(cls.tempdir.name)
        includes = {
            'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=7,nincoming=2)',
            'genps.inc': 'integer lmaxconfigs,max_branch,maxproc,maxflow\n'
                         'parameter(lmaxconfigs=2,max_branch=4,maxproc=1,maxflow=1)',
            'nFKSconfigs.inc': 'integer fks_configs,fks_integrated\n'
                               'parameter(fks_configs=4,fks_integrated=4)',
            'orders.inc': 'integer nsplitorders\nparameter(nsplitorders=1)',
            'run.inc': 'double precision ebeam(2)\ncommon/test_beams/ebeam',
            'cuts.inc': 'double precision ptj,ptgmin,ptl,mll,mll_sf\n'
                        'common/test_cuts/ptj,ptgmin,ptl,mll,mll_sf',
            'coupl.inc': 'double precision resmass(2),reswidth(2),parentmass(2),parentwidth(2),g\n'
                         'common/test_model/resmass,reswidth,parentmass,parentwidth,g',
            'born_props.inc': 'call fixture_props(pmass,pwidth,pow)',
            'born_conf.inc': 'integer iforest(2,-4:-1,2)\ncommon/test_forest/iforest',
            'fks_info.inc': 'integer fks_i_d(4),fks_j_d(4)\n'
                            'parameter(fks_i_d=[7,7,7,7],fks_j_d=[1,3,2,3])',
        }
        for name, source in includes.items():
            (work / name).write_text(''.join('      ' + line + '\n'
                                           for line in source.splitlines()))
        (work / 'contexts.f90').write_text(
            'module mint_module\nuse FKSParams\n'
            'integer,parameter :: maxchannels=2\ninteger :: iconfig=1,ichan=1\n'
            'logical :: nlo_ps=.true.\nend module\n'
            'module mc_native_context\n'
            'integer :: native_epoch=1,active_context=1,native_context_ids(4)=[1,1,2,2]\n'
            'integer :: active_history=0\n'
            'logical :: native_mapping=.false.\nend module\n')
        cached = fortran_routine(TEMPLATE / 'setcuts.f', 'set_tau_min')
        # The reference takes the production calculation on every call. Only
        # lookup is disabled, independently of the cache's input predicates.
        fresh = cached.replace('subroutine set_tau_min(', 'subroutine set_tau_min_fresh(')
        fresh = fresh.replace('end subroutine set_tau_min', 'end subroutine set_tau_min_fresh')
        fresh = fresh.replace('cache_matches=.true.', 'cache_matches=.false.')
        for name, source, tag in [('cached.f', cached, 1), ('fresh.f', fresh, 2)]:
            source = source.replace('      call set_granny(',
                                    '      call count_tau_builds(%d)\n      call set_granny(' % tag)
            (work / name).write_text(source)
        (work / 'granny.f').write_text(fortran_routine(TEMPLATE / 'fks_singular.f', 'set_granny'))
        cls.executable = work / 'check_tau_cache'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-fcheck=all', '-fbacktrace',
            '-ffixed-line-length-132', '-I', str(work), str(TEMPLATE / 'FKSParams.f90'),
            str(TEMPLATE / 'fks_phase_space_data.f'), str(work / 'contexts.f90'),
            str(work / 'cached.f'), str(work / 'fresh.f'), str(work / 'granny.f'),
            str(ROOT / 'tests/input_files/check_tau_cut_cache.f90'),
            '-o', str(cls.executable)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def check_cache(self, mode):
        result = subprocess.run([str(self.executable), mode], cwd=self.work,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS tau cache ' + mode, result.stdout)

    def test_context_sector_channel_and_topology_reuse(self):
        self.check_cache('reuse')

    def test_mass_width_cut_and_particle_invalidation(self):
        self.check_cache('inputs')

    def test_mapping_modes_and_common_born_chart(self):
        self.check_cache('mapping')

    def test_conflicting_resonances_and_recoil_restoration(self):
        self.check_cache('resonance')

    def test_bounded_cache_eviction(self):
        self.check_cache('eviction')
