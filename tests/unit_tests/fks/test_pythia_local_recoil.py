"""Independent kinematic and measure checks for PYTHIA local IF FKS recoil."""

import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tests.unit_tests.fks.test_momentum_maps import (
    ROOT, TEMPLATE, fks_test_module, fortran_routine,
)


def dot(a, b):
    return a[0]*b[0] - np.dot(a[1:], b[1:])


def boost(p, velocity):
    b2 = np.dot(velocity, velocity)
    gamma = 1/math.sqrt(1-b2)
    bp = np.dot(velocity, p[1:])
    return np.r_[gamma*(p[0]+bp), p[1:] + (
        (gamma-1)*bp/b2 + gamma*p[0])*velocity]


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestPythiaLocalRecoil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix="mg5_pythia_local_")
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        for name in ("fks_powers.inc", "timing_variables.inc"):
            shutil.copyfile(TEMPLATE / name, work / name)
        (work / "nexternal.inc").write_text(
            "      integer nexternal,nincoming\n      parameter(nexternal=5,nincoming=2)\n")
        (work / "orders.inc").write_text(
            "      integer nsplitorders\n      parameter(nsplitorders=1)\n")
        (work / "genps.inc").write_text(
            "      integer max_branch,max_particles\n      parameter(max_branch=10,max_particles=10)\n")
        (work / "run.inc").write_text(
            "      double precision ebeam(2),xbk(2)\n      integer lpp(2)\n"
            "      common/test_run/ebeam,xbk,lpp\n")
        (work / "pmass.inc").write_text("      common/to_mass/pmass\n")
        (work / "coupl.inc").write_text("")
        (work / "native_context.f90").write_text(
            "module mc_native_context\nlogical::native_mapping=.true.\nend module\n")
        (work / "fks_phase_space.f").write_text(fks_test_module())
        routines = [fortran_routine(TEMPLATE / "fks_singular.f", name) for name in (
            "rotate_invar", "trp_rotate_invar", "phspncheck_nocms", "xlen4",
            "xmom_compare", "xmcompare", "xprintout",
            "compute_prefactors_n1body", "set_cms_stuff")]
        routines += [fortran_routine(ROOT / "Template/NLO/Source/kin_functions.f", name)
                     for name in ("dot", "rho", "threedot")]
        (work / "maps.f").write_text("\n".join(routines))
        cls.executable = work / "check_local"
        command = [shutil.which("gfortran"), "-O2", "-std=legacy", "-fcheck=all",
                   "-ffpe-trap=invalid,zero,overflow",
                   "-ffixed-line-length-none", "-ffunction-sections", "-fdata-sections",
                   "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                   "-I", str(work), str(work / "native_context.f90")]
        command += [str(TEMPLATE / name) for name in (
            "fks_phase_space_data.f", "genps_fks_helpers.f", "FKSParams.f90",
            "genps_fks_radiation.f")]
        command += [str(work / "fks_phase_space.f"), str(work / "maps.f")]
        command += [str(TEMPLATE / name) for name in (
            "resonance_recoil.f", "initial_recoil.f", "boostwdir2.f")]
        command += [str(ROOT / name) for name in (
            "Template/NLO/Source/timing.f", "HELAS/boostx.F", "HELAS/rotxxx.F",
            "tests/input_files/check_pythia_local_recoil.f90")]
        command += ["-o", str(cls.executable)]
        result = subprocess.run(command, cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        (work / "selector_orders.inc").write_text(
            "      integer nsplitorders,qed_pos\n      parameter(nsplitorders=2,qed_pos=2)\n")
        selector = fortran_routine(TEMPLATE / "mc_native_runtime.f", "mc_select_pythia_pair")
        (work / "selector.f").write_text(selector.replace("'orders.inc'", "'selector_orders.inc'"))
        cls.selector_executable = work / "check_selector"
        result = subprocess.run([
            shutil.which("gfortran"), "-O2", "-ffixed-line-length-none", "-fcheck=all",
            "-I", str(work), str(ROOT / "tests/input_files/check_pythia_local_selection.f90"),
            str(work / "selector.f"),
            "-o", str(cls.selector_executable)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def run_map(self, beam=1, partner=3, mass=173., rad=(.6, .4, .3),
                mode="map", extra="", emitted=5):
        data = f"{beam} {partner} {mass} " + " ".join(map(str, rad)) + "\n" + extra
        result = subprocess.run([str(self.executable), mode, str(emitted)], input=data,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        if mode != "map":
            return result.stdout
        values = np.fromstring(result.stdout, sep=" ")
        self.assertEqual(len(values), 48, result.stdout)
        return values[:8], values[8:28].reshape(5, 4), values[28:].reshape(5, 4)

    def test_pythia8318_branch_kinematics_and_projection(self):
        # Literal IF branch relations from SimpleSpaceShower::branch,
        # pythia8318. The reference uses the old dipole CM, whereas the
        # production chart uses the full real-process CM.
        for beam in (1, 2):
            for partner in (3, 4):
                for mass in (0., 173.):
                    for rad in ((.2, .1, .8), (.6, .4, .3), (.93, .85, .9)):
                        with self.subTest(beam=beam, partner=partner, mass=mass, rad=rad):
                            h, born, p = self.run_map(beam, partner, mass, rad)
                            a, f, k = p[beam-1], p[partner-1], p[4]
                            ba, bf = born[beam-1], born[partner-1]
                            z = 1-h[2]
                            self.assertAlmostEqual(1-dot(f, k)/dot(a, f+k), z, delta=1e-12)
                            np.testing.assert_allclose(z*a, ba, rtol=1e-12, atol=1e-9)
                            np.testing.assert_allclose(f+k-(1-z)*a, bf, rtol=1e-12, atol=1e-9)
                            for leg in range(4):
                                if leg not in (beam-1, partner-1):
                                    np.testing.assert_array_equal(p[leg], born[leg])
                            np.testing.assert_allclose(p[0]+p[1], p[2]+p[3]+p[4], atol=1e-9)
                            q = ba+bf
                            md2 = dot(q, q)
                            md = math.sqrt(md2)
                            mf2 = max(0., dot(bf, bf))
                            red = md2-mf2
                            baxis = boost(ba, -q[1:]/q[0])[1:]
                            baxis /= np.linalg.norm(baxis)
                            kc = boost(k, -q[1:]/q[0])
                            kz = np.dot(kc[1:], baxis)
                            q2 = 2*dot(a, k)
                            energy = (1-z)*red/(2*z*md) + (z-.5)*q2/md
                            longitudinal = ((1-z)*red/(2*z*md)-q2/(2*md)
                                            -z*q2*mf2/(md*red))
                            pt2 = (1-z)*q2*(1-z*q2/red)-mf2*(z*q2/red)**2
                            self.assertAlmostEqual(kc[0], energy, delta=2e-8)
                            self.assertAlmostEqual(kz, longitudinal, delta=2e-8)
                            self.assertAlmostEqual(np.dot(kc[1:], kc[1:])-kz*kz, pt2, delta=2e-5)

    def test_independent_hadronic_determinant(self):
        for beam in (1, 2):
            for partner in (3, 4):
                h, born, p = self.run_map(beam, partner)
                # Compute d^3k/(2E) directly in the sampled random coordinates.
                derivatives = []
                for axis in range(3):
                    up, down = np.array([.6, .4, .3]), np.array([.6, .4, .3])
                    up[axis] += 1e-5
                    down[axis] -= 1e-5
                    kp = self.run_map(beam, partner, rad=up)[2][4, 1:]
                    km = self.run_map(beam, partner, rad=down)[2][4, 1:]
                    derivatives.append((kp-km)/2e-5)
                measure = abs(np.linalg.det(derivatives))/(2*p[4, 0]*(2*math.pi)**3)
                measure *= dot(born[beam-1], born[partner-1])/dot(born[beam-1], p[partner-1])
                expected = h[4]*h[1]*h[5]*(1-(.12 if beam == 1 else .20))*2*math.pi
                self.assertAlmostEqual(measure/expected, 1., delta=1e-6)

    def test_phase_space_volume(self):
        nodes, weights = np.polynomial.legendre.leggauss(56)
        extra = "56\n" + "".join(f"{(x+1)/2:.17g} {w/2:.17g}\n"
                                    for x, w in zip(nodes, weights))
        for beam in (1, 2):
            for partner in (3, 4):
                out = self.run_map(beam, partner, mode="volume", extra=extra)
                value, reference = np.fromstring(out, sep=" ")
                self.assertAlmostEqual(value/reference, 1., delta=5e-4)

    def test_production_counterevents_native_inverse_and_state(self):
        for beam in (1, 2):
            for partner in (3, 4):
                self.assertIn("PASS production", self.run_map(beam, partner, mode="production"))

    def test_emitted_leg_permutations(self):
        for emitted in (3, 4):
            for beam in (1, 2):
                for partner in (3, 4):
                    self.assertIn("PASS production", self.run_map(
                        beam, partner, mode="production", emitted=emitted))

    def test_dimensional_soft_conversion_and_unchanged_pdf_prefactors(self):
        for beam in (1, 2):
            for partner in (3, 4):
                output = self.run_map(beam, partner, mode="endpoints")
                observed, scale = np.fromstring(output, sep=" ")
                a = 1/scale
                epsilon = 1e-8
                # Finite difference between the two regulated soft integrals
                # after their common -1/(2 epsilon) pole cancels.
                finite = -math.expm1(-2*epsilon*math.log(a))/(2*epsilon)
                self.assertAlmostEqual(observed, finite, delta=2e-7)

    def test_soft_collinear_and_beam_endpoint_corners(self):
        for beam in (1, 2):
            for partner in (3, 4):
                for rad in ((1e-5, .4, .3), (.6, 1e-5, .3), (.999999, .8, .3)):
                    self.run_map(beam, partner, rad=rad)

    def test_turning_point_is_covered_without_a_singular_sample_weight(self):
        # At fixed physical xi the quadratic has two branches. Sampling the
        # incoming rescaling must traverse their merger and cover both sides.
        _, born, _ = self.run_map()
        f = born[2]
        y, phi = .8, .4
        fm, fp = f[0]-f[3], f[0]+f[3]
        v = fm*(1+y)+fp*(1-y)
        b = 2*(f[1]*math.cos(phi)+f[2]*math.sin(phi))*math.sqrt(1-y*y)
        q = (v+math.sqrt(v*v-b*b))/b
        w = 1-1/(q*q)
        self.assertLess(w, .88)
        x2, x3 = math.sqrt((1-y)/2), phi/(2*math.pi)
        centre = self.run_map(rad=(math.sqrt(w/.88), x2, x3))[0]
        for trial in (w-.02, w+.02):
            header = self.run_map(rad=(math.sqrt(trial/.88), x2, x3))[0]
            self.assertLess(header[1], centre[1])
        self.assertTrue(np.isfinite(centre[4]))

    def test_native_colour_pair_selection_and_scope(self):
        for mode in ("quark", "antiquark", "flows", "gluon", "no_partner",
                     "beam", "qed", "top_mass"):
            with self.subTest(mode=mode):
                result = subprocess.run([str(self.selector_executable), mode],
                                        capture_output=True, text=True)
                if mode in ("quark", "antiquark"):
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("PASS selector", result.stdout)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("five-flavour single-top", result.stdout)

    def test_invalid_momenta_and_beam_fraction_are_rejected(self):
        self.assertIn("PASS invalid", self.run_map(mode="invalid"))

    def test_native_soft_jet_with_hard_nearly_collinear_emission(self):
        # Captured from a matched single-top run. A light Born partner's
        # roundoff mass used to turn a 10 keV real jet into a 38 MeV jet.
        p = [[76.8177389880016648, 0., 0., 76.8177389880016648],
             [117.981307376411678, 0., 0., -117.981307376411678],
             [174.220399736325419, -.0253216612150490676, -.0608140089019293881,
              -20.5850271983018018],
             [1.00079085213186498e-5, 6.15576511107626379e-9, 1.47837861887234288e-8,
              -1.00078957087532355e-5],
             [20.5786366180093658, .0253216550591598248, .060813994118144526,
              -20.5785311800426776]]
        extra = "".join(" ".join(format(x, ".17g") for x in q)+"\n" for q in p)
        for beam, partner in ((1, 3), (2, 4)):
            self.assertIn("PASS native point", self.run_map(
                beam, partner, mode="native_point", extra=extra))


if __name__ == "__main__":
    unittest.main()
