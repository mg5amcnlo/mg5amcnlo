"""Numerical regressions for Delta matching without a shower runtime library."""

from pathlib import Path
import math
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import fortran_routine


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMCatNLODeltaScales(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_delta_scales_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        cls.executable = work / 'check_delta_scales'
        # Linking only these two Fortran sources verifies that scale
        # reconstruction needs no PYTHIA library, wrappers or initialization.
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-std=f2003', '-fcheck=all',
            str(TEMPLATE / 'mcatnlo_delta_scales.f90'),
            str(ROOT / 'tests/input_files/check_mcatnlo_delta_scales.f90'),
            '-o', str(cls.executable)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def check_case(self, name):
        result = subprocess.run([str(self.executable), name],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS ' + name, result.stdout)

    def test_mixed_dipoles_and_lorentz_invariance(self):
        self.check_case('mixed')

    def test_final_dipoles_and_emission_remapping(self):
        self.check_case('final')

    def test_initial_double_colour_connection(self):
        self.check_case('double')

    def test_dead_zone_and_small_virtuality_conventions(self):
        self.check_case('sentinels')

    def test_heavy_flavour_thresholds(self):
        self.check_case('thresholds')

    def test_invalid_colours_arguments_and_pair_threshold(self):
        self.check_case('invalid')


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMCatNLODeltaMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_delta_matching_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        includes = {
            'nexternal.inc': 'integer nexternal,nincoming\n'
                             'parameter(nexternal=5,nincoming=2)',
            'born_nhel.inc': 'integer max_bcol\nparameter(max_bcol=1)',
            'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=1)',
            'run.inc': 'integer lpp(2)\ncommon/test_lpp/lpp',
            'orders.inc': '',
            'genps.inc': 'integer maxproc\nparameter(maxproc=1)',
            'born_leshouche.inc': 'integer iproc_born\nparameter(iproc_born=1)\n'
                                  'common/test_born/idup,mothup,icolup',
            'leshouche_decl.inc': 'integer idup_d(1,5,1),mothup_d(1,2,5,1)\n'
                                  'integer icolup_d(1,2,5,1),niprocs_d(1)',
        }
        for name, contents in includes.items():
            (work / name).write_text(''.join('      ' + line + '\n'
                                            for line in contents.splitlines()))
        shutil.copyfile(TEMPLATE / 'MCmasses_PYTHIA8.inc',
                        work / 'MCmasses_PYTHIA8.inc')
        shutil.copyfile(TEMPLATE / 'fks_powers.inc', work / 'fks_powers.inc')
        (work / 'delta_matching.f').write_text('\n'.join(
            fortran_routine(TEMPLATE / 'montecarlocounter.f', name)
            for name in ('compute_delta', 'gl_safe', 'get_parton_id', 'setSudType')))
        cls.executable = work / 'check_delta_matching'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
            '-ffixed-line-length-none', '-ffunction-sections', '-fdata-sections',
            '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
            '-I', str(work), str(TEMPLATE / 'process_module.f90'),
            str(TEMPLATE / 'kinematics_module.f90'),
            str(TEMPLATE / 'scale_module.f90'),
            str(TEMPLATE / 'mcatnlo_delta_scales.f90'),
            str(work / 'delta_matching.f'),
            str(ROOT / 'tests/input_files/check_mcatnlo_delta_matching.f90'),
            '-o', str(cls.executable)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def test_native_scales_feed_sudakov_and_h_event_assignment(self):
        result = subprocess.run([str(self.executable), 'matching'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS native Delta matching and starting-scale veto', result.stdout)

    def test_reconstruction_error_is_fatal(self):
        result = subprocess.run([str(self.executable), 'invalid'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('MC@NLO-Delta scale reconstruction failed', result.stdout)


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMCatNLODeltaModelMasses(unittest.TestCase):
    def check_model_masses(self, model_name, charm_mass, bottom_mass):
        from madgraph.iolibs import export_fks, file_writers
        from models import import_ufo, model_reader

        model = model_reader.ModelReader(import_ufo.import_model(model_name))
        model.set_parameters_and_couplings()
        with tempfile.TemporaryDirectory(prefix='mg5_delta_model_masses_') as tmp:
            work = Path(tmp)
            makeinc = work / 'makeinc.inc'
            makeinc.write_text('MODEL = \n')
            getter = work / 'get_mass_width_fcts.f'
            with file_writers.FortranWriter(str(getter)) as writer:
                export_fks.ProcessExporterFortranFKS.write_get_mass_width_file(
                    None, writer, str(makeinc), model)
            # Supply the model's evaluated mass/width parameters to the actual
            # generated getter, without exporting unrelated matrix elements.
            parameters = {particle[kind] for particle in model['particles']
                          for kind in ('mass', 'width')
                          if particle[kind].lower() != 'zero'}
            declarations = []
            for name in sorted(parameters):
                value = complex(model['parameter_dict'][name])
                self.assertEqual(value.imag, 0.0)
                number = ('%.17e' % value.real).replace('e', 'd')
                declarations.extend(('      double precision ' + name,
                                     '      parameter (%s=%s)' % (name, number)))
            (work / 'coupl.inc').write_text('\n'.join(declarations) + '\n')
            executable = work / 'check_model_masses'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all', '-I', str(work),
                str(TEMPLATE / 'mcatnlo_delta_scales.f90'), str(getter),
                str(ROOT / 'tests/input_files/check_mcatnlo_delta_model_masses.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            values = [float(value) for value in result.stdout.split()]
            self.assertEqual(len(values), 6, result.stdout)
            self.assertEqual(values[:4], [charm_mass, charm_mass, bottom_mass, bottom_mass])
            # The input kinematics give Q2=200 and 1-z=0.002. A nonzero
            # model mass activates the threshold correction for these inputs.
            for actual, mass in zip(values[4:], (charm_mass, bottom_mass)):
                self.assertAlmostEqual(actual, math.sqrt(0.002*(200.0 + mass**2)), places=10)

    def test_five_flavour_model_has_massless_charm_and_bottom(self):
        self.check_model_masses('loop_sm-no_b_mass', 0.0, 0.0)

    def test_default_model_has_massless_charm_and_massive_bottom(self):
        self.check_model_masses('loop_sm', 0.0, 4.7)

    def test_charm_mass_model_uses_both_heavy_flavour_thresholds(self):
        self.check_model_masses('loop_sm-c_mass', 1.55, 4.7)
