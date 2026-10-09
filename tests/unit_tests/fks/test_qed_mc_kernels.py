"""Compiled QED MC kernels: coupling orders, charges and photon multiplicity.

References use the fermion and photon Altarelli-Parisi functions directly.
These tests do not assert full shower matching or an integrated S/H result.
"""

import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import (
    ROOT, TEMPLATE, fortran_routine, mc_counterterm_test_module)


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestQEDMCKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_qed_mc_kernels_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        files = {
            'nexternal.inc': 'integer nexternal,nincoming\n'
                             'parameter(nexternal=5,nincoming=2)',
            'born_nhel.inc': 'integer max_bcol\nparameter(max_bcol=1)',
            'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=1)',
            # Deliberately reverse the positions: kernel index and order index
            # belong to different spaces.
            'orders.inc': 'integer nsplitorders,qcd_pos,qed_pos,amp_split_size\n'
                          'parameter(nsplitorders=2,qcd_pos=2,qed_pos=1,amp_split_size=1)\n'
                          'double complex amp_split_cnt(amp_split_size,2,nsplitorders)\n'
                          'common/test_amp_split/amp_split_cnt',
            'coupl.inc': 'double precision g\ndouble complex gal(2)\n'
                         'common/test_couplings/g,gal',
        }
        for name, source in files.items():
            (work / name).write_text(''.join('      '+line+'\n'
                                           for line in source.splitlines()))
        (work / 'scale.f90').write_text(
            'module scale_module\ninteger :: born_flow_picked\nend module\n')
        names = ('prepare_MCsubtraction_born', 'get_mbar', 'compute_splitting_kernels',
                 'compute_splitting_kernel_icode1', 'compute_splitting_kernel_icode2',
                 'compute_splitting_kernel_icode3', 'compute_splitting_kernel_icode4',
                 'xfact_ileg12', 'xfact_ileg3', 'xfact_ileg4', 'py8_gluon_recoil_weight')
        source = mc_counterterm_test_module(names, include_kinematics=False,
            external_functions=(('double complex', 'mc_born_azimuth_phase'),))
        source += '\n'.join(fortran_routine(TEMPLATE / 'fks_singular.f', name)
                            for name in ('AP_reduced', 'AP_reduced_SUSY',
                                         'AP_reduced_massive', 'Qterms_reduced_timelike',
                                         'Qterms_reduced_spacelike'))
        (work / 'routines.f').write_text(source)
        cls.executable = work / 'check_qed_mc_kernels'
        command = [shutil.which('gfortran'), '-O0', '-std=legacy',
                   '-ffixed-line-length-none', '-fcheck=all', '-finit-real=snan',
                   '-ffpe-trap=invalid,zero,overflow', '-I', str(work),
                   str(TEMPLATE / 'qed_shower_support.f90'),
                str(TEMPLATE / 'process_module.f90'), 'scale.f90', 'routines.f',
                   str(ROOT / 'tests/input_files/check_qed_mc_kernels.f90'),
                   '-o', str(cls.executable)]
        subprocess.run(command, cwd=work, check=True, capture_output=True, text=True)

        # Exercise the actual H-event color insertion, with a controlled Born
        # flow that has either a photon, a quark or an antiquark mother.
        (work / 'colors.f').write_text(fortran_routine(
            TEMPLATE / 'add_write_info.f', 'fill_icolor_H'))
        cls.color_executable = work / 'check_qed_mc_colors'
        subprocess.run([shutil.which('gfortran'), '-O0', '-std=legacy',
                        '-ffixed-line-length-none', '-fcheck=all', '-I', str(work),
                        'colors.f', str(ROOT / 'tests/input_files/check_qed_mc_colors.f90'),
                        '-o', str(cls.color_executable)], cwd=work, check=True,
                       capture_output=True, text=True)

        identity = work / 'identity'
        identity.mkdir()
        for name in ('nexternal.inc', 'orders.inc'):
            shutil.copyfile(work / name, identity / name)
        (identity / 'nFKSconfigs.inc').write_text(
            '      integer fks_configs\n      parameter(fks_configs=2)\n')
        tables = '''integer fks_i_D(2),fks_j_D(2),extra_cnt_D(2)
integer isplitorder_born_D(2),isplitorder_cnt_D(2)
integer fks_j_from_i_D(2,5,0:5),particle_type_D(2,5),pdg_type_D(2,5)
logical need_color_links_D(2),need_charge_links_D(2)
logical particle_tag_D(2,5),split_type_D(2,2)
double precision particle_charge_D(2,5)
common/test_int/ fks_i_D,fks_j_D,extra_cnt_D,isplitorder_born_D,isplitorder_cnt_D,fks_j_from_i_D,particle_type_D,pdg_type_D
common/test_logical/ need_color_links_D,need_charge_links_D,particle_tag_D,split_type_D
common/test_charges/ particle_charge_D'''
        (identity / 'fks_info.inc').write_text(''.join(
            ('     &'+line[1:] if line.startswith('&') else '      '+line)+'\n'
            for line in tables.splitlines()))
        (identity / 'chooser.f').write_text('\n'.join(fortran_routine(
            TEMPLATE / 'chooser_functions.f', name)
            for name in ('fks_inc_chooser', 'get_mother_col_charge')))
        cls.identity_executable = identity / 'check_qed_mc_identity'
        subprocess.run([shutil.which('gfortran'), '-O0', '-std=legacy',
                        '-ffixed-line-length-none', '-ffree-line-length-none',
                        '-fcheck=all', '-I', str(identity),
                        str(ROOT / 'tests/input_files/check_qed_mc_identity.f90'),
                        'chooser.f', '-o', str(cls.identity_executable)],
                       cwd=identity, check=True, capture_output=True, text=True)

    def run_driver(self, mode, data, check=True):
        return subprocess.run([str(self.executable), mode], input=data,
                              capture_output=True, text=True, check=check)

    def kernels(self, leg, collinear, species, g=1., z=.6, shower='PYTHIA8'):
        data = '{} {}\n'.format(shower, z)
        data += '{} {} {} {} {} {} {} {} {} {}\n'.format(
            leg, 'T' if collinear else 'F', *species, g)
        return list(map(float, self.run_driver('radiators', data).stdout.split()))

    def assertRelative(self, actual, reference):
        self.assertAlmostEqual(actual/reference, 1., delta=3e-14)

    def test_correction_order_dispatch(self):
        for flags, expected in [('T F', [2., 1., -1.]),
                                ('F T', [1., 2., -2.])]:
            result = self.run_driver('dispatch', flags+' PYTHIA8 F 0\n')
            self.assertEqual(list(map(float, result.stdout.split())), expected)

    def test_unsupported_qed_configurations_fail_explicitly(self):
        for data, message in [
                ('T T PYTHIA8 F 0', 'exactly one QCD or QED'),
                ('F F PYTHIA8 F 0', 'exactly one QCD or QED'),
                ('T F HERWIG7 F 0', 'requires PYTHIA8'),
                ('T F PYTHIA8 T 0', 'MC@NLO-Delta or FxFx'),
                ('T F PYTHIA8 F 3', 'MC@NLO-Delta or FxFx')]:
            with self.subTest(configuration=data):
                result = self.run_driver('dispatch', data+'\n', check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout)

    def test_born_spin_for_both_fermion_daughter_orderings(self):
        for data in ('1 1 1 0 -1 -1', '1 1 1 -1 0 -1',
                     '1 3 3 0 0.666666666666667 0.666666666666667',
                     '3 1 3 0.666666666666667 0 0.666666666666667'):
            result = self.run_driver('born', data+'\n')
            self.assertEqual(list(map(float, result.stdout.split())), [1., 0.])
        result = self.run_driver('born', '1 1 1 -1 1 0\n')
        self.assertEqual(list(map(float, result.stdout.split())), [1., -1.])

    def test_qed_h_event_color_insertions(self):
        # (j_fks, i_type, j_type, Born color, Born anticolor), then
        # the expected colors (i_color, i_anticolor, j_color, j_anticolor).
        cases = [((3, 1, 1, 0, 0), (0, 0, 0, 0)),
                 ((1, 1, 1, 0, 0), (0, 0, 0, 0)),
                 ((3, -3, 3, 0, 0), (0, 504, 504, 0)),
                 ((3, 3, -3, 0, 0), (504, 0, 0, 504)),
                 ((3, 3, 1, 503, 0), (503, 0, 0, 0)),
                 ((3, -3, 1, 0, 503), (0, 503, 0, 0)),
                 ((1, 3, 3, 0, 0), (504, 0, 504, 0)),
                 ((1, -3, -3, 0, 0), (0, 504, 0, 504)),
                 ((1, 3, 1, 0, 503), (503, 0, 0, 0)),
                 ((1, -3, 1, 503, 0), (0, 503, 0, 0)),
                 ((1, 1, 3, 503, 0), (0, 0, 503, 0)),
                 ((1, 1, -3, 0, 503), (0, 0, 0, 503))]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                result = subprocess.run([str(self.color_executable)],
                    input=' '.join(map(str, inputs))+'\n', capture_output=True,
                    text=True, check=True)
                self.assertEqual(tuple(map(int, result.stdout.split())), expected)

    def test_photon_mother_identity_when_switching_qcd_qed_sectors(self):
        result = subprocess.run([str(self.identity_executable)],
                                capture_output=True, text=True, check=True)
        rows = [list(map(float, line.split())) for line in result.stdout.splitlines()]
        self.assertEqual(rows, [[1., 0.], [8., 0.], [1., 0.]]*2)

    @staticmethod
    def measure(leg):
        x, y, s = .7, .2, 1000.
        if leg <= 2:
            return 4*(1-y)*(1-x)/(s*x)
        if leg == 3:
            return abs((1+x)*100+(1-x)*y*math.sqrt(10000+4))/10000*120*(1-x)*(1-y)*2/s
        return (2-(1-x)*(1-y))/.6*(1-x)*(1-y)*2/s

    def test_photon_conversion_colour_and_single_recoiler(self):
        z, e2, t, jac = .6, .09, 20., 7.
        for colour, charge, pdg in [(1, -1., 11), (3, 2/3., 2), (-3, -1/3., -1)]:
            # FSR photon -> f fbar: mother neutral, both daughters charged.
            species = (colour, -colour if abs(colour) == 3 else 1, 1,
                       charge, -charge, 0., -pdg)
            values = self.kernels(4, False, species)
            norm = self.measure(4)*e2*charge**2*abs(colour)*jac/t
            self.assertRelative(values[1], norm*(z*z+(1-z)**2))
            self.assertRelative(values[3], norm*4*z*(1-z))
            # The analytic collinear branch must also survive alpha_s = 0.
            values = self.kernels(4, True, species, g=0.)
            x, s = .7, 1000.
            norm = 4*e2*charge**2*abs(colour)/s
            self.assertRelative(values[1], norm*(1-x)*(x*x+(1-x)**2)/x)
            self.assertRelative(values[3], norm*4*(1-x)**2)
            # Cross the photon into the real incoming leg: the Born leg is
            # a charged fermion and has no azimuthal correlation.
            species = (colour, 1, -colour if abs(colour) == 3 else 1,
                       charge, 0., -charge, 22)
            values = self.kernels(1, False, species)
            norm = self.measure(1)*e2*charge**2*abs(colour)*jac/t
            self.assertRelative(values[1], norm*(z*z+(1-z)**2))
            self.assertEqual(values[3], 0.)
            values = self.kernels(1, True, species, g=0.)
            self.assertRelative(values[1], 4*e2*charge**2*abs(colour)/s*
                                (1-x)*(x*x+(1-x)**2)/x)

    def test_fermion_radiation_and_crossed_photon_born(self):
        z, e2, t, jac, x, s = .6, .09, 20., 7., .7, 1000.
        radiation = (1, 1, 1, 0., -1., -1., 11)
        for leg in (1, 2, 3, 4):
            values = self.kernels(leg, False, radiation)
            reference = self.measure(leg)*e2*(1+z*z)*jac/(t*(1-z))
            self.assertRelative(values[1], reference)
            if leg != 3:
                values = self.kernels(leg, True, radiation, g=0.)
                self.assertRelative(values[1], 4*e2*(1+x*x)/(s*x))
        # The three charged massive leptons must retain the fermion numerator.
        for pdg in (11, -11, 13, -13, 15, -15):
            values = self.kernels(3, False, (*radiation[:-1], pdg))
            self.assertRelative(values[1], self.measure(3)*e2*(1+z*z)*jac/(t*(1-z)))
        # Backwards evolution from a photon Born leg uses one connection.
        species = (1, 1, 1, -1., -1., 0., 11)
        values = self.kernels(1, False, species)
        norm = self.measure(1)*e2*jac/t
        self.assertRelative(values[1], norm*(1+(1-z)**2)/z)
        self.assertRelative(values[3], -norm*4*(1-z)/z)
        values = self.kernels(1, True, species, g=0.)
        self.assertRelative(values[1], 4*e2*(1-x)*(1+(1-x)**2)/(s*x*x))
        self.assertRelative(values[3], -16*e2*(1-x)**2/(s*x*x))

    def test_crossed_final_state_daughter_charges(self):
        # f -> photon f: the neutral daughter's charge must be passed to AP.
        species = (1, 1, 1, -1., 0., -1., 22)
        z = .6
        for leg in (3, 4):
            values = self.kernels(leg, False, species)
            norm = self.measure(leg)*.09*7/20
            self.assertRelative(values[1], norm*(1+(1-z)**2)/z)

    def test_pythia_massive_w_and_squark_radiation(self):
        # SimpleTimeShower uses the same 1+z^2 numerator for charged
        # resonances and colour-triplet radiators when MECs are off.
        # Test the actual dispatch, including both squark coupling orders.
        species = [(24, 1, 1.), (-24, 1, -1.),
                   (1000002, 3, 2/3.), (-1000002, -3, -2/3.),
                   (2000001, 3, -1/3.), (-2000001, -3, 1/3.)]
        for pdg, colour, charge in species:
            for emitted, index, coupling in ((1, 1, .09*charge**2),
                                             (8, 0, 4/3.)):
                if index == 0 and abs(colour) != 3:
                    continue
                state = (emitted, colour, colour, 0., charge, charge, pdg)
                for z in (.2, .6, .92, .9999):
                    norm = self.measure(3)*coupling*7/(20*(1-z))
                    with self.subTest(pdg=pdg, emitted=emitted, z=z):
                        values = self.kernels(3, False, state, z=z)
                        self.assertRelative(values[index], norm*(1+z*z))
                        self.assertEqual(values[index+2], 0.)
                        # Other shower choices retain their SUSY kernels.
                        for shower in ('HERWIG6', 'HERWIG7', 'PYTHIA6Q'):
                            values = self.kernels(3, False, state, z=z,
                                                  shower=shower)
                            self.assertRelative(values[index], norm*2*z)

    def test_pythia_gluino_colour_connection_normalization(self):
        # In SimpleTimeShower 8.318 with MECs off a non-gluon octet
        # receives two colour ends, each treated as colType=+/-1 (CF).
        # The physical SUSY kernel retains CA/2 per end for other showers.
        species = (8, 8, 8, 0., 0., 0., 1000021)
        for g in (.7, 1.3):
            for z in (.2, .6, .92, .9999):
                norm = self.measure(3)*g*g*(1+z*z)*7/(20*(1-z))
                for shower, per_end in (('PYTHIA8', 4/3.),
                                        ('HERWIG6', 3/2.),
                                        ('HERWIG7', 3/2.),
                                        ('PYTHIA6Q', 3/2.)):
                    with self.subTest(shower=shower, z=z, g=g):
                        values = self.kernels(3, False, species, g=g, z=z,
                                              shower=shower)
                        self.assertRelative(values[0], norm*per_end)
                        self.assertEqual(values[1:], [0., 0., 0.])
