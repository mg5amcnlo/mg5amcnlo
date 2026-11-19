"""Regressions for the angular failures seen in the stable-top local map."""
import math
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which("gfortran"), "gfortran is required")
class TestGeometryStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fks_geometry_")
        cls.work = Path(cls.temporary.name)
        helpers = (ROOT / "Template/NLO/SubProcesses/genps_fks_helpers.f").read_text()
        start = helpers.index("      subroutine getangles(")
        stop = helpers.index("      end subroutine getangles", start)
        angles = helpers[start:stop] + "      end subroutine getangles\n"
        sij = (ROOT / "Template/NLO/SubProcesses/fks_Sij.f").read_text()
        start = sij.index("      double precision function get_cms_energy(")
        stop = sij.index("      double precision function costh_fks", start)
        (cls.work / "kernels.f").write_text(angles + "\n" + sij[start:stop])
        (cls.work / "point_data.f90").write_text("""
module fks_phase_space_data
  implicit none
  real(8) :: ybst_til_tocm, shat
end module

real(8) function dot(a,b)
  implicit none
  real(8), intent(in) :: a(0:3), b(0:3)
  dot=a(0)*b(0)-sum(a(1:3)*b(1:3))
end function
""")
        (cls.work / "driver.f90").write_text("""
program geometry
  use fks_phase_space_data
  implicit none
  integer :: mode, count, i
  real(8) :: p1(0:3),p2(0:3),ka(0:3),kb(0:3),q(0:3)
  real(8) :: th,cth,sth,phi,cphi,sphi,m1,m2,e1,e2,b1,b2,ct
  read(*,*) count
  do i=1,count
    read(*,*) mode
    if(mode.eq.1) then
      read(*,*) p1
      call getangles(p1,th,cth,sth,phi,cphi,sphi)
      write(*,'(6ES26.17E3)') th,cth,sth,phi,cphi,sphi
    else
      read(*,*) ybst_til_tocm,m1,m2,e1,e2
      read(*,*) p1
      read(*,*) p2
      read(*,*) ka
      read(*,*) kb
      q=ka+kb
      shat=q(0)**2-sum(q(1:3)**2)
      call get_cms_costh_fks(p1,p2,ka,kb,e1,e2,m1,m2,b1,b2,ct)
      write(*,'(5ES26.17E3)') b1,b2,ct,e1,e2
    endif
  enddo
end program
""")
        cls.executable = cls.work / "geometry"
        subprocess.run(
            ["gfortran", "-O2", "-ffixed-line-length-132",
             "-ffpe-trap=invalid,zero,overflow", "point_data.f90", "kernels.f",
             "driver.f90", "-o", str(cls.executable)],
            cwd=cls.work, check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_cases(self, cases):
        lines = [str(len(cases))]
        for case in cases:
            lines.extend(case)
        result = subprocess.run(
            [str(self.executable)], input="\n".join(lines) + "\n",
            capture_output=True, text=True, check=True)
        rows = [[float(x) for x in line.split()] for line in result.stdout.splitlines()]
        self.assertEqual(len(rows), len(cases))
        self.assertTrue(all(math.isfinite(x) for row in rows for x in row))
        return rows

    @staticmethod
    def line(values):
        return " ".join(format(x, ".17g") for x in values)

    def angle_case(self, vector):
        return ["1", self.line([0.] + list(vector))]

    def cms_case(self, p1, p2, ka, kb, masses, ybst=1., energies=(-1., -1.)):
        return ["2", self.line([ybst, *masses, *energies]), self.line(p1),
                self.line(p2), self.line(ka), self.line(kb)]

    @staticmethod
    def boost(vector, velocity):
        b2 = sum(x*x for x in velocity)
        if b2 == 0.:
            return list(vector)
        gamma = 1. / math.sqrt(1.-b2)
        bp = sum(x*y for x, y in zip(vector[1:], velocity))
        # Positive boost, from the incoming-pair rest frame to its moving frame.
        coefficient = gamma*gamma/(gamma+1.)*bp + gamma*vector[0]
        return [gamma*(vector[0]+bp)] + [
            x + coefficient*b for x, b in zip(vector[1:], velocity)]

    @staticmethod
    def momentum(mass, spatial):
        return [math.hypot(mass, *spatial), *spatial]

    def test_getangles_retains_transverse_direction_at_both_beam_axes(self):
        vectors = []
        for sign in (-1., 1.):
            for epsilon in (1.e-4, 1.e-9, 1.e-14, 1.e-100):
                for scale in (1.e-200, 1.e-10, 1., 1.e200):
                    for phi in (-2.7, -.4, 1.3, 2.9):
                        vectors.append([scale*epsilon*math.cos(phi),
                                        scale*epsilon*math.sin(phi), sign*scale])
        rows = self.run_cases([self.angle_case(v) for v in vectors])
        for vector, (_, cth, sth, phi, cphi, sphi) in zip(vectors, rows):
            scale = max(abs(x) for x in vector)
            normalized = [x/scale for x in vector]
            norm = math.hypot(*normalized)
            transverse = math.hypot(*normalized[:2])
            self.assertGreater(sth, 0.)
            self.assertAlmostEqual(sth/(transverse/norm), 1., delta=5.e-15)
            self.assertAlmostEqual(cphi, normalized[0]/transverse, delta=5.e-15)
            self.assertAlmostEqual(sphi, normalized[1]/transverse, delta=5.e-15)
            self.assertAlmostEqual(cth, normalized[2]/norm, delta=5.e-15)
            self.assertAlmostEqual(phi, math.atan2(vector[1], vector[0]), delta=5.e-15)

    def test_getangles_axis_and_zero_conventions(self):
        rows = self.run_cases([self.angle_case(v) for v in
                              ([0., 0., 0.], [0., 0., 9.], [0., 0., -9.])])
        self.assertEqual(rows[0], [0., 1., 0., 0., 1., 0.])
        self.assertEqual(rows[1], rows[0])
        self.assertEqual(rows[2], [math.pi, -1., 0., 0., 1., 0.])

    def test_getangles_agrees_with_ordinary_spherical_coordinates(self):
        rng = random.Random(93211)
        vectors = [[rng.uniform(-100., 100.) for _ in range(3)] for _ in range(1000)]
        rows = self.run_cases([self.angle_case(v) for v in vectors])
        for v, (th, cth, sth, phi, cphi, sphi) in zip(vectors, rows):
            norm = math.sqrt(sum(x*x for x in v))
            old_cth = v[2]/norm
            old_phi = math.atan2(v[1], v[0])
            self.assertAlmostEqual(cth, old_cth, delta=5.e-15)
            self.assertAlmostEqual(th, math.acos(old_cth), delta=1.e-13)
            self.assertAlmostEqual(sth, math.sqrt(1.-old_cth**2), delta=1.e-13)
            self.assertAlmostEqual(phi, old_phi, delta=5.e-15)
            self.assertAlmostEqual(cphi, math.cos(old_phi), delta=5.e-15)
            self.assertAlmostEqual(sphi, math.sin(old_phi), delta=5.e-15)

    def test_cms_velocities_and_angle_agree_with_well_conditioned_legacy(self):
        rng = random.Random(32912)
        cases, expected = [], []
        for _ in range(1000):
            masses = [rng.choice((0., 32., 173.)) for _ in range(2)]
            spatial = [[rng.uniform(-400., 400.) for _ in range(3)] for _ in range(2)]
            vectors = [self.momentum(m, p) for m, p in zip(masses, spatial)]
            velocity = [rng.uniform(-.35, .35) for _ in range(3)]
            ka = self.boost([500., 0., 0., 500.], velocity)
            kb = self.boost([500., 0., 0., -500.], velocity)
            p1, p2 = [self.boost(p, velocity) for p in vectors]
            energy1, energy2 = [p[0] for p in vectors]
            beta1, beta2 = [math.sqrt(1.-(m/p[0])**2)
                            for m, p in zip(masses, vectors)]
            dot = p1[0]*p2[0]-sum(x*y for x, y in zip(p1[1:], p2[1:]))
            legacy_cos = (1.-dot/(energy1*energy2))/(beta1*beta2)
            cases.append(self.cms_case(p1, p2, ka, kb, masses))
            expected.append([beta1, beta2, legacy_cos, energy1, energy2])
        for row, target in zip(self.run_cases(cases), expected):
            for value, reference in zip(row[:2], target[:2]):
                self.assertAlmostEqual(value, reference, delta=8.e-15)
            self.assertAlmostEqual(row[2], target[2], delta=2.e-13)
            for value, reference in zip(row[3:], target[3:]):
                self.assertAlmostEqual(value/reference, 1., delta=8.e-15)

    def test_near_rest_top_bad_energy_uses_resolved_spatial_velocity(self):
        # The exact soft counterevent of the first C_LOC log failure.
        top = [172.99999999999616, 1.1946661016943396e-8,
               7.806717122411141e-8, 3.6385847579367692e-6]
        light = [3.88953357144652e-6, -2.0791375990462636e-7,
                 -1.3586423077798258e-6, -3.638589606711522e-6]
        ka, kb = [86.5000019447668, 0., 0., 86.5000019447668], [
            86.5000019447668, 0., 0., -86.5000019447668]
        self.assertLess(1.-(173./top[0])**2, 0.)
        rows = self.run_cases([self.cms_case(top, light, ka, kb, (173., 0.), ybst=y)
                               for y in (0., 1.)])
        norm1, norm2 = math.hypot(*top[1:]), math.hypot(*light[1:])
        cosine = sum(x*y for x, y in zip(top[1:], light[1:]))/(norm1*norm2)
        for beta1, beta2, costh, _, _ in rows:
            self.assertAlmostEqual(beta1/(norm1/top[0]), 1., delta=8.e-15)
            self.assertEqual(beta2, 1.)
            self.assertAlmostEqual(costh, cosine, delta=8.e-15)

    def test_top_rest_limit_and_massless_velocity(self):
        cases = []
        for velocity in ([0., 0., 0.], [.13, -.21, .3]):
            ka = self.boost([500., 0., 0., 500.], velocity)
            kb = self.boost([500., 0., 0., -500.], velocity)
            top = self.boost([173., 0., 0., 0.], velocity)
            light = self.boost([5., 3., 4., 0.], velocity)
            cases.append(self.cms_case(top, light, ka, kb, (173., 0.)))
        for beta1, beta2, costh, _, _ in self.run_cases(cases):
            self.assertAlmostEqual(beta1, 0., delta=1.e-15)
            self.assertEqual(beta2, 1.)
            self.assertAlmostEqual(1.-beta1*costh, 1., delta=1.e-15)

    def test_near_rest_velocity_and_partition_angle_survive_a_boost(self):
        top = self.momentum(173., [1.e-7, -2.e-6, 3.e-6])
        light = self.momentum(0., [.4, .7, -.2])
        norm1, norm2 = math.hypot(*top[1:]), math.hypot(*light[1:])
        beta = norm1/top[0]
        cosine = sum(x*y for x, y in zip(top[1:], light[1:]))/(norm1*norm2)
        cases = []
        for velocity in ([0., 0., 1.e-12], [.13, -.21, .3], [0., 0., .95]):
            ka = self.boost([500., 0., 0., 500.], velocity)
            kb = self.boost([500., 0., 0., -500.], velocity)
            cases.append(self.cms_case(self.boost(top, velocity),
                self.boost(light, velocity), ka, kb, (173., 0.)))
        for beta1, beta2, costh, _, _ in self.run_cases(cases):
            self.assertAlmostEqual(beta1, beta, delta=3.e-14)
            self.assertEqual(beta2, 1.)
            self.assertAlmostEqual(1.-beta1*costh, 1.-beta*cosine, delta=3.e-14)

    def test_cms_flag_and_cached_energy_semantics(self):
        p1 = self.momentum(173., [10., 20., 30.])
        p2 = self.momentum(0., [6., 8., 0.])
        # A zero frame-change flag uses the vectors' current frame and does
        # not overwrite the energy cache, irrespective of ka/kb.
        ka, kb = [600., 0., 0., 600.], [400., 0., 0., -400.]
        row = self.run_cases([self.cms_case(p1, p2, ka, kb, (173., 0.),
            ybst=0., energies=(-1., -2.))])[0]
        self.assertEqual(row[3:], [-1., -2.])
        self.assertAlmostEqual(row[0], math.hypot(*p1[1:])/p1[0], delta=5.e-15)
        # A positive precomputed energy is retained in the nontrivial branch.
        row = self.run_cases([self.cms_case(p1, p2,
            [500., 0., 0., 500.], [500., 0., 0., -500.], (173., 0.),
            energies=(p1[0], p2[0]))])[0]
        self.assertEqual(row[3:], [p1[0], p2[0]])

    def test_large_incoming_beam_ratio_keeps_threshold_partition_finite(self):
        top = [172.99999999999616, 1.1946661016943396e-8,
               7.806717122411141e-8, 3.6385847579367692e-6]
        light = self.momentum(0., [.4, .7, -.2])
        norm = math.hypot(*top[1:])
        beta = norm/top[0]
        cosine = sum(x*y for x, y in zip(top[1:], light[1:]))/(
            norm*math.hypot(*light[1:]))
        target = 1.-beta*cosine
        ratios = (1., 10., 100., 1000., 5650., 1.e6)
        cases = []
        for ratio in ratios:
            root = math.sqrt(ratio)

            def rapidity_boost(vector):
                plus = (vector[0]+vector[3])*root
                minus = (vector[0]-vector[3])/root
                return [(plus+minus)/2., vector[1], vector[2], (plus-minus)/2.]

            ka = [86.5*root, 0., 0., 86.5*root]
            kb = [86.5/root, 0., 0., -86.5/root]
            cases.append(self.cms_case(rapidity_boost(top), rapidity_boost(light),
                                       ka, kb, (173., 0.)))
        for ratio, row in zip(ratios, self.run_cases(cases)):
            # 5650 covers the largest beam-fraction ratio at 13 TeV with
            # a 173 GeV top; 1e6 is an additional conditioning stress case.
            tolerance = 5.e-13 if ratio <= 5650. else 5.e-11
            self.assertAlmostEqual(1.-row[0]*row[2], target, delta=tolerance)
            self.assertEqual(row[1], 1.)


if __name__ == "__main__":
    unittest.main()
