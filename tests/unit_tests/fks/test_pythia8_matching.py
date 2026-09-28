"""Production regressions for the shared massive measure and PYTHIA gluon recoil.

The forward FSR momenta follow SimpleTimeShower::branch. The independent
measure uses phase-space factorization, and endpoint references solve energy
conservation and its implicit derivative with 80-digit Decimal arithmetic.
These are scalar matching tests, not a full shower or S/H expansion test.
"""

from decimal import Decimal, localcontext
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


def dot(a, b):
    return a[0]*b[0] - sum(x*y for x, y in zip(a[1:], b[1:]))


def boost(p, velocity):
    b2 = sum(x*x for x in velocity)
    if b2 == 0:
        return list(p)
    gamma = 1/math.sqrt(1-b2)
    bp = sum(x*y for x, y in zip(velocity, p[1:]))
    return [gamma*(p[0]+bp)] + [
        p[i+1] + ((gamma-1)*bp/b2 + gamma*p[0])*velocity[i]
        for i in range(3)]


def rotate(p):
    theta, phi = .6, .4
    a = math.cos(theta)*p[1] + math.sin(theta)*p[3]
    return [p[0], math.cos(phi)*a-math.sin(phi)*p[2],
            math.sin(phi)*a+math.cos(phi)*p[2],
            -math.sin(theta)*p[1]+math.cos(theta)*p[3]]


def fsr(m, recoil_mass, z, t):
    """A hard system with two individually massless recoil spectators."""
    s, root = 1e6, 1000.
    pair = m*m + t/(z*(1-z))
    energy = (s+pair-recoil_mass**2)/(2*root)
    k = math.sqrt(energy**2-pair)
    kz1 = (energy**2*z-pair/2)/k
    kz2 = (energy**2*(1-z)-pair/2)/k
    kt = math.sqrt(pair*(energy**2*z*(1-z)-pair/4)/k**2)
    fraction = m*m/pair
    kt *= 1-fraction
    kz1 += fraction*kz2
    kz2 *= 1-fraction
    radiator = rotate([math.sqrt(m*m+kt*kt+kz1*kz1), kt, 0, kz1])
    emitted = rotate([math.hypot(kt,kz2), -kt, 0, kz2])
    if recoil_mass:
        velocity = [0, 0, -k/(root-energy)]
        spectators = [rotate(boost([recoil_mass/2, sign*recoil_mass/2, 0, 0],
                                   velocity)) for sign in (1, -1)]
    else:
        spectators = [rotate([(root-energy)/2, 0, 0, -k/2])]*2
    born_energy = (s+m*m-recoil_mass**2)/(2*root)
    born_k = math.sqrt(born_energy**2-m*m)
    measure = ((1-fraction)*(s+pair-recoil_mass**2) /
               (32*math.pi**3*2*root*born_k*z*(1-z)))
    return [[500., 0, 0, 500.], [500., 0, 0, -500.],
            radiator, *spectators, emitted], measure


def event_input(p, m=0., kind=1, shower='PYTHIA8', scales=(1000., 1000.)):
    header = '{} {} {} {} {}\n'.format(m, kind, shower, *scales)
    return header + ''.join(' '.join(format(x, '.17g') for x in q)+'\n' for q in p)


def ordered_recoil(z, s, recoil2, pair2):
    """Literal PYTHIA trial weight, with its rejection of negative weights."""
    r, v = recoil2/s, pair2/s
    x1, x2 = (1-r+v)*z, 1+r-v
    return max(0., 1-r/max(1e-12, x1+x2-1-r) *
               max(1e-12, 1+r-x2)/max(1e-12, 1-r-x1))


def endpoint_reference(m, recoil_mass, rad, delta, branch=1):
    """Solve (2-xi) E + xi*y*k = (S*(1-xi)+m^2-M^2)/sqrt(S)."""
    with localcontext() as context:
        context.prec = 80
        s, root = Decimal(1000000), Decimal(1000)
        # Match the representable FKS coordinates used by the Fortran routines.
        xi = Decimal.from_float(1-float(1-rad))
        y = Decimal.from_float(float(1-delta))
        m2 = Decimal.from_float(float(m*m))
        mr2 = Decimal.from_float(float(recoil_mass*recoil_mass))
        a, b = 2-xi, xi*y
        c = (s*(1-xi)+m2-mr2)/root
        disc = c*c-(a*a-b*b)*m2
        if disc <= 0:
            raise ValueError('Outside the massive FKS support')
        k = (-b*c+branch*a*disc.sqrt())/(a*a-b*b)
        if k <= 0:
            raise ValueError('Nonpositive radiator momentum')
        energy = (k*k+m2).sqrt()
        w = root*xi*(energy-y*k)
        z = 1-s*xi*(m2+w)/(w*(s+m2+w-mr2))
        t = z*(1-z)*w
        dkdy = -xi*k/(a*k/energy+b)
        dwdy = root*xi*((k/energy-y)*dkdy-k)
        jac = abs(s*(m2+w)/(w*(s+m2+w-mr2))*dwdy*z*(1-z))
        return float(k), float(z), float(t), float(jac)


def massive_shower_reference(shower, m, recoil_mass, rad, y, e0sq, branch):
    """Independent finite-difference determinant of the shower-variable map."""
    with localcontext() as context:
        context.prec = 80
        s, root = Decimal(1000000), Decimal(1000)
        m2, mr2 = (Decimal.from_float(float(mass*mass))
                    for mass in (m, recoil_mass))
        rad, y, e0sq = map(Decimal.from_float, (rad, y, e0sq))

        def variables(xi, cosine):
            a, b = 2-xi, xi*cosine
            c = (s*(1-xi)+m2-mr2)/root
            k = (-b*c+branch*a*(c*c-(a*a-b*b)*m2).sqrt())/(a*a-b*b)
            energy = (k*k+m2).sqrt()
            w1 = root*xi*(energy-cosine*k)
            w2 = s*xi-w1
            if shower in ('PYTHIA8', 'PYTHIA6Q'):
                z = 1-s*xi*(m2+w1)/(w1*(s+m2+w1-mr2))
                return z, z*(1-z)*w1 if shower == 'PYTHIA8' else w1
            eps = 1-(m2-mr2)/(s-w1)
            beta = (eps*eps-4*s*mr2/(s-w1)**2).sqrt()
            zeta = ((2*s-(s-w1)*eps)*w2 +
                    (s-w1)*((w1+w2)*beta-eps*w1)) / (
                    (s-w1)*beta*(2*s-(s-w1)*eps+(s-w1)*beta))
            if shower == 'HERWIG7':
                z = 1-zeta
                return z, w1/(z*(1-z))
            tbeta = (1-(w1+m2)/e0sq).sqrt()
            z = 1-tbeta*zeta-w1/(2*(1+tbeta)*e0sq)
            return z, w1/(2*z*(1-z)*e0sq)

        z, t = variables(rad, y)
        h = Decimal('1e-25')
        xp, xm = variables(rad+h, y), variables(rad-h, y)
        yp, ym = variables(rad, y+h), variables(rad, y-h)
        jac = abs(((xp[0]-xm[0])*(yp[1]-ym[1]) -
                   (yp[0]-ym[0])*(xp[1]-xm[1]))/(4*h*h))
        return float(z), float(t), float(jac)


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestPythia8Matching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_pythia8_matching_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        (work / 'nexternal.inc').write_text(
            '      integer nexternal,nincoming\n'
            '      parameter(nexternal=6,nincoming=2)\n')
        (work / 'genps.inc').write_text(
            '      integer max_branch,max_particles\n'
            '      parameter(max_branch=8,max_particles=8)\n')
        (work / 'coupl.inc').write_text(
            '      double precision g\n      double complex gal(2)\n'
            '      common/test_couplings/g,gal\n')
        shutil.copyfile(TEMPLATE / 'fks_powers.inc', work / 'fks_powers.inc')
        (work / 'native.f90').write_text(
            'module mc_native_context\nlogical :: native_mapping=.true.\nend module\n')
        (work / 'scale.f90').write_text(
            'module scale_module\n'
            'double precision :: shower_scale_nbody_min(5,5),'
            'shower_scale_nbody_max(5,5)\nend module\n')
        selections = {
            'genps_fks.f': ('invert_fks_radiation', 'generate_momenta_initial_inverse',
                           'generate_momenta_massive_final_inverse',
                           'generate_momenta_massless_final_inverse',
                           'native_fsr_angle', 'get_recoil', 'getangles'),
            'montecarlocounter.f': (
                'zPY8', 'xiPY8', 'xjacPY8',
                'get_shower_variables', 'get_zeta',
                'zHW6', 'xiHW6', 'xjacHW6', 'zHW7', 'xiHW7', 'xjacHW7',
                'zPY6Q', 'xiPY6Q', 'xjacPY6Q', 'zPY6PT', 'xiPY6PT', 'xjacPY6PT',
                'dinvariants_dFKS', 'xfact_ileg12',
                'xfact_ileg3', 'xfact_ileg4', 'compute_splitting_kernels',
                'compute_splitting_kernel_icode1', 'compute_splitting_kernel_icode2',
                'compute_splitting_kernel_icode3', 'compute_splitting_kernel_icode4',
                'py8_gluon_recoil_weight', 'limits', 'get_dead_zone', 'get_angle',
                'compute_damping_weight', 'emscafun'),
            'fks_singular.f': ('AP_reduced', 'AP_reduced_SUSY', 'AP_reduced_massive',
                              'Qterms_reduced_timelike', 'Qterms_reduced_spacelike'),
            'fks_Sij.f': ('fks_Hij', 'h_damp'),
        }
        routines = [fortran_routine(TEMPLATE / filename, name)
                    for filename, names in selections.items() for name in names]
        routines += [fortran_routine(ROOT / 'Template/NLO/Source/kin_functions.f', name)
                     for name in ('dot', 'rho', 'threedot')]
        (work / 'routines.f').write_text('\n'.join(routines))
        cls.executables = {}
        variants = {
            'optimized': ['-O2'],
            'poisoned': ['-O2', '-finit-real=inf'],
            'checked': ['-O0', '-fcheck=all', '-finit-real=snan',
                        '-ffpe-trap=invalid,zero,overflow'],
        }
        for variant, flags in variants.items():
            executable = work / ('check_' + variant)
            command = [shutil.which('gfortran'), *flags, '-std=legacy',
                       '-ffixed-line-length-none', '-fno-automatic',
                       '-ffunction-sections', '-fdata-sections',
                       '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                       '-I', str(work), str(TEMPLATE / 'process_module.f90'),
                       str(TEMPLATE / 'kinematics_module.f90'), 'native.f90', 'scale.f90',
                       'routines.f', str(TEMPLATE / 'boostwdir2.f'),
                       str(ROOT / 'tests/input_files/check_pythia8_matching.f90'),
                       '-o', str(executable)]
            result = subprocess.run(command, cwd=work, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            cls.executables[variant] = executable

    def run_driver(self, mode, data, variant='optimized'):
        result = subprocess.run([str(self.executables[variant]), mode], input=data,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [list(map(float, line.split())) for line in result.stdout.splitlines()]

    def assertRelative(self, actual, expected, tolerance=1e-9):
        self.assertLessEqual(abs(actual-expected), tolerance*max(abs(expected), 1e-300))

    def test_massive_second_branch_and_shared_kernel_controls(self):
        z, t = .23921928965797884, 41764.76709871376
        p, measure = fsr(173., 173., z, t)
        data = ''.join(event_input(p, 173., kind, shower)
                       for kind in (2, 3, 4) for shower in ('PYTHIA8', 'HERWIG6'))
        rows = self.run_driver('event', data)
        for kind, (pythia, herwig) in zip((2, 3, 4), zip(rows[::2], rows[1::2])):
            np = 2 if kind == 4 else 1
            coefficient = pythia[3]*pythia[2]/(pythia[4]**2*(1-pythia[5]))
            self.assertRelative(coefficient, 1/(16*math.pi**3*np*measure))
            self.assertGreater(pythia[6], 0)
            self.assertEqual(pythia[12], 1)  # Accepted by the local dipole check.
            self.assertRelative(pythia[3], herwig[3])
        self.assertRelative(rows[0][3]*rows[0][2]/(rows[0][4]**2*(1-rows[0][5])),
                            .3139879195061142)

    def test_shared_massive_measure_across_showers(self):
        showers = ('PYTHIA8', 'HERWIG6', 'HERWIG7', 'PYTHIA6Q')
        cases = []
        for z, t in ((.8, 5000.), (.23921928965797884, 41764.76709871376)):
            p, measure = fsr(173., 173., z, t)
            cases.extend((shower, p, measure) for shower in showers)
        data = ''.join(event_input(p, 173., 2, shower) for shower, p, _ in cases)
        reference_rows = self.run_driver('measure', data)
        self.assertEqual(len(reference_rows), 4*len(cases))
        accepted = {shower: 0 for shower in showers}
        accepted_second = dict(accepted)
        for (shower, p, measure), start in zip(cases, range(0, len(reference_rows), 4)):
            for row in reference_rows[start:start+4]:
                with self.subTest(shower=shower, emitted=p[5], e0sq=row[6]):
                    z, t, jac, prefactor, rad, y, e0sq, _, zone = row
                    k = math.sqrt(sum(v*v for v in p[2][1:]))
                    geometry = (2-rad)*k+rad*y*p[2][0]
                    branch = 1 if geometry > 0 else -1
                    self.assertGreater(prefactor, 0)
                    self.assertGreaterEqual(jac, 0)
                    accepted[shower] += int(zone)
                    if branch == -1:
                        accepted_second[shower] += int(zone)
                    if z < 0:
                        self.assertEqual(jac, 0)
                        self.assertEqual(zone, 0)
                        continue
                    expected = massive_shower_reference(
                        shower, 173., 173., rad, y, e0sq, branch)
                    for actual, wanted in zip((z, t, jac), expected):
                        self.assertRelative(actual, wanted)
                    _, _, _, py8jac = endpoint_reference(173., 173., rad, 1-y, branch)
                    actual = prefactor*jac/(rad*rad*(1-y))
                    self.assertRelative(actual, expected[2]/(16*math.pi**3*measure*py8jac))
        self.assertTrue(all(accepted.values()), accepted)
        self.assertGreater(accepted_second['PYTHIA8'], 0)
        # At this fixture the other showers veto the second solution.
        # Correcting the common measure must preserve that support decision.
        for shower in ('HERWIG6', 'HERWIG7', 'PYTHIA6Q'):
            self.assertEqual(accepted_second[shower], 0)
        for variant in ('poisoned', 'checked'):
            rows = self.run_driver('measure', data, variant)
            self.assertEqual(len(rows), len(reference_rows))
            for row, expected in zip(rows, reference_rows):
                for actual, wanted in zip(row, expected):
                    self.assertAlmostEqual(actual, wanted,
                                           delta=1e-12*max(abs(wanted), 1e-20))

    def test_full_measure_on_both_massive_branches(self):
        rng = random.Random(8318)
        cases = []
        for m in (0., 50., 173.):
            for mr in (40., 180.):
                for _ in range(30):
                    z, t = rng.uniform(.6, .88), rng.uniform(500., 3500.)
                    p, measure = fsr(m, mr, z, t)
                    cases.append((p, m, z, t, measure))
        for m in (0., 50., 173.):
            for mr in (40., 173., 180.):
                count = 0
                while count < 180:
                    z = rng.uniform(.015, .985)
                    t = rng.uniform(.001, .98)*((1000-mr)**2-m*m)*z*(1-z)
                    try:
                        p, measure = fsr(m, mr, z, t)
                    except ValueError:
                        continue
                    cases.append((p, m, z, t, measure))
                    count += 1
        self.assertEqual(len(cases), 1800)
        rows = self.run_driver('event', ''.join(event_input(p, m, 2 if m else 1)
                                               for p, m, *_ in cases))
        self.assertEqual(len(rows), len(cases))
        second_branches = 0
        for row, (p, m, z, t, measure) in zip(rows, cases):
            self.assertRelative(row[0], z)
            self.assertRelative(row[1], t)
            coefficient = row[3]*row[2]/(row[4]**2*(1-row[5]))
            np = 1 if m else 2
            self.assertRelative(coefficient, 1/(16*math.pi**3*np*measure))
            if m:
                k = math.sqrt(sum(v*v for v in p[2][1:]))
                if 2-row[4]*(1-p[2][0]/k*row[5]) < 0:
                    second_branches += 1
        self.assertEqual(second_branches, 12)

    def test_guarded_recoil_and_massless_control(self):
        cases = [(z, 1e6, mr**2, pair) for mr in (0., 40., 400.)
                 for z in (.2, .5, .8) for pair in (1e-10, 1000., 93750.)]
        # Both denominator guards, the numerator guard, and rejected trials.
        cases += [(z, 1e6, 160000., pair)
                  for z in (0., 1e-14, 1-1e-14, 1.) for pair in (0., 93750.)]
        data = ''.join('{} {} {} {}\n'.format(*case) for case in cases)
        values = self.run_driver('weight', data)
        for case, (value,) in zip(cases, values):
            self.assertAlmostEqual(value, ordered_recoil(*case), delta=3e-12)
        audit = self.run_driver('weight', '.8 1e6 160000 93750\n')[0][0]
        self.assertAlmostEqual(audit, .7530955643618137, delta=1e-14)

    def test_massive_branch_meeting_and_small_radiator_momentum(self):
        m, mr, y = 173., 173., -.8
        a = 1e6-m*m*(1-y*y)
        b = -2e6+4*m*m
        c = 1e6-4*m*m
        boundary = (-b-math.sqrt(b*b-4*a*c))/(2*a)
        coordinates = [(boundary-eps, 1-y, branch)
                       for eps in (1e-3, 1e-6, 1e-9, 1e-12) for branch in (1, -1)]
        for k in (1., 1e-3, 1e-6):
            energy = math.hypot(m,k)
            xi = (1000-2*energy)/(1000-energy+y*k)
            coordinates.append((xi, 1-y, -1))
        data, references = '', []
        for xi, delta, branch in coordinates:
            k, z, t, _ = endpoint_reference(m,mr,xi,delta,branch)
            data += '3 173 173 {:.17g} {:.17g} {:.17g}\n'.format(xi,delta,k)
            pair = m*m+t/(z*(1-z))
            born_k = math.sqrt((1e6+m*m-mr*mr)**2/4e6-m*m)
            measure = ((1-m*m/pair)*(1e6+pair-mr*mr) /
                       (32*math.pi**3*2000*born_k*z*(1-z)))
            references.append(1/(16*math.pi**3*measure))
        for variant in self.executables:
            rows = self.run_driver('radiation', data, variant)
            for (xi, delta, _), row, reference in zip(coordinates, rows, references):
                effective_xi, effective_delta = 1-float(1-xi), 1-float(1-delta)
                self.assertRelative(row[2]*row[3]/(effective_xi**2*effective_delta),
                                    reference)

    def test_complete_gluon_daughter_and_connection_sum(self):
        for z, t in ((z, t) for z in (.2, .35, .5, .8) for t in (15000., 20000.)):
            p, measure = fsr(0., 400., z, t)
            swapped = list(p)
            swapped[2], swapped[5] = p[5], p[2]
            for scales in ((1000., 1000.), (160., 250.), (100., 250.)):
                rows = self.run_driver('event', event_input(p, scales=scales) +
                                       event_input(swapped, scales=scales))
                self.assertRelative(rows[0][15], 160000.)
                for spectator in p[3:5]:
                    self.assertAlmostEqual(dot(spectator,spectator), 0, delta=1e-9)
                self.assertAlmostEqual(rows[0][10]+rows[1][10], 1, delta=1e-14)
                if scales == (1000., 1000.):
                    self.assertEqual(rows[0][11], float(t == 15000.))
                    self.assertEqual(rows[0][12], 1.)
                for partner in range(2):
                    self.assertEqual(rows[0][11+partner], rows[1][11+partner])
                    self.assertRelative(rows[0][13+partner], rows[1][13+partner])
                    actual = sum(row[6]/(row[4]**2*(1-row[5]))*row[10] *
                                 row[11+partner]*row[13+partner] for row in rows)
                    weighted = sum(1.5*(1+zz**3)/(1-zz) *
                                   ordered_recoil(zz, 1e6, 160000., t/(z*(1-z)))
                                   for zz in (z, 1-z))
                    expected = (weighted/t/(16*math.pi**3*measure) *
                                rows[0][11+partner]*rows[0][13+partner])
                    self.assertAlmostEqual(actual, expected,
                                           delta=1e-9*max(abs(expected), 1e-20))

    def test_recoil_preserves_helicity_other_channels_and_g_limits(self):
        for m, kind in ((0., 1), (0., 2), (0., 5), (0., 6),
                        (173., 2), (173., 3), (173., 4)):
            p, _ = fsr(m, 400., .8, 15000.)
            pythia, control = self.run_driver('event', event_input(p, m, kind) +
                                             event_input(p, m, kind, 'HERWIG6'))
            self.assertEqual(pythia[7:10], control[7:10])
            expected = 1.
            if kind == 1:
                expected = ordered_recoil(.8, 1e6, 160000., 93750.)
            self.assertRelative(pythia[6]/control[6], expected)
        p, _ = fsr(0., 400., .99, 1e-4)
        row = self.run_driver('event', event_input(p))[0]
        self.assertEqual(row[16], 0.)  # Raw soft term off; retain the G replacement.
        self.assertEqual(row[18], 1.)  # Exact helicity prescription unchanged.
        for t in (1., 1e-2, 1e-4, 1e-8):
            p, _ = fsr(0., 400., .8, t)
            row, control = self.run_driver('event', event_input(p) +
                                           event_input(p, shower='HERWIG6'))
            self.assertLess(abs(row[6]/control[6]-1), 10*t/1e6)
            self.assertEqual(row[7], control[7])
            if 1-row[5] < 1e-6:
                self.assertEqual(row[6], control[6])

    def test_endpoint_expansions_against_high_precision(self):
        cases, references = [], []
        switches = (1e-3, 1e-4, 1.001e-5, .999e-5, 1e-6, 1e-8, 1e-10, 1e-12)
        for m, mr in ((0., 0.), (0., 400.), (.1, 400.), (50., 180.),
                      (173., 173.), (173., 820.)):
            for xi in switches:
                for delta in (*switches, .5, 1.7):
                    try:
                        k, z, t, jac = endpoint_reference(m, mr, xi, delta)
                    except ValueError:
                        continue
                    if not 0 < z < 1:
                        continue
                    cases.append((3 if m else 4, m, mr, xi, delta, k))
                    references.append((z, t, jac))
        data = ''.join(' '.join(format(x, '.17g') for x in case)+'\n' for case in cases)
        rows = self.run_driver('radiation', data)
        self.assertEqual(len(rows), len(cases))
        for case, row, reference in zip(cases, rows, references):
            with self.subTest(m=case[1], recoil=case[2], xi=case[3], delta=case[4]):
                xi, delta = case[3:5]
                tolerance = 1e-12
                if not case[1]:
                    # The retained massless soft/collinear expansions have
                    # relative remainders O(xi) / O(1-y), respectively.
                    tolerance = 1e-11
                    if xi < 1e-5:
                        tolerance += 5*xi/(1-case[2]**2/1e6)**2
                    elif delta < 1e-5:
                        tolerance += 5*delta/(1-case[2]**2/1e6)**2
                for actual, expected in zip(row[:3], reference):
                    self.assertTrue(math.isfinite(actual))
                    self.assertRelative(actual, expected, tolerance)
                # The exact pT used by support/damping also stays stable.
                self.assertRelative(row[4], reference[1], 1e-12)
                z, _, _ = reference
                effective_xi, effective_delta = 1-float(1-xi), 1-float(1-delta)
                energy = math.hypot(case[1],case[5])
                gap = case[1]**2/(energy+case[5])+effective_delta*case[5]
                emitted = 500*effective_xi
                w = 1000*effective_xi*gap
                pair = case[1]**2+w
                omz = (case[1]**2/(2*gap)+emitted)/(energy+emitted)
                born_k = math.sqrt((1e6+case[1]**2-case[2]**2)**2/4e6-case[1]**2)
                measure = (w/pair*(1e6+pair-case[2]**2) /
                           (32*math.pi**3*2000*born_k*z*omz))
                coefficient = row[3]*row[2]/(effective_xi**2*effective_delta)
                self.assertRelative(coefficient, 1/(16*math.pi**3*measure),
                                    2*tolerance)

    def test_initialization_independence_and_fpe_checks(self):
        p, _ = fsr(173., 173., .23921928965797884, 41764.76709871376)
        event_data = event_input(p, 173., 2)
        event_data += event_input(fsr(0., 400., .8, 15000.)[0])
        radiation_data = ''
        for m in (0., 50., 173.):
            for xi, delta in ((1e-3, .5), (1.001e-5, .999e-5),
                              (.999e-5, 1.001e-5), (1e-12, 1e-12)):
                k, *_ = endpoint_reference(m, 400., xi, delta)
                radiation_data += '{} {} 400 {} {} {:.17g}\n'.format(
                    3 if m else 4, m, xi, delta, k)
        for mode, data in (('event', event_data), ('radiation', radiation_data)):
            reference = self.run_driver(mode, data)
            for variant in ('poisoned', 'checked'):
                rows = self.run_driver(mode, data, variant)
                self.assertEqual(len(rows), len(reference))
                for row, expected in zip(rows, reference):
                    for a, b in zip(row, expected):
                        self.assertAlmostEqual(a, b, delta=1e-12*max(abs(b), 1e-20))
