"""Stock and patched IF densities, endpoint limits and local G profiles."""

import math
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import numpy as np

from tests.unit_tests.fks.test_momentum_maps import (
    ROOT, TEMPLATE, fortran_routine, mc_counterterm_test_module,
)
from tests.unit_tests.fks import test_pythia_local_recoil as recoil_tests

boost, dot = recoil_tests.boost, recoil_tests.dot


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestPythiaLocalMatching(unittest.TestCase):
    run_map = recoil_tests.TestPythiaLocalRecoil.run_map

    @classmethod
    def setUpClass(cls):
        recoil_tests.TestPythiaLocalRecoil.setUpClass.__func__(cls)
        work = Path(cls.tempdir.name)
        (work / "orders.inc").write_text(
            "      integer nsplitorders,amp_split_size\n"
            "      parameter(nsplitorders=1,amp_split_size=1)\n")
        (work / "born_nhel.inc").write_text(
            "      integer max_bcol\n      parameter(max_bcol=1)\n")
        (work / "coupl.inc").write_text(
            "      double precision g\n      double complex gal(2)\n"
            "      common/test_couplings/g,gal\n")
        (work / "scale.f90").write_text(
            "module scale_module\n"
            "real(8)::shower_scale_nbody_min(4,4),shower_scale_nbody_max(4,4)\n"
            "end module\n")
        kernel = mc_counterterm_test_module((
            "prepare_pythia_local_kinematics", "compute_pythia_local_counterterm", "pythia_local_gprofile",
            "emscafun"), include_kinematics=False)
        kernel += fortran_routine(TEMPLATE / "fks_singular.f", "get_mc_lum")
        kernel += (TEMPLATE / "fks_Sij.f").read_text()
        (work / "local_kernel.f").write_text(kernel)
        cls.kernel_executable = work / "check_matching"
        command = [shutil.which("gfortran"), "-O2", "-std=legacy", "-fcheck=all",
                   "-ffpe-trap=invalid,zero,overflow", "-ffixed-line-length-none",
                   "-ffunction-sections", "-fdata-sections",
                   "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                   "-I", str(work), str(work / "native_context.f90"),
                   str(work / "scale.f90")]
        command += [str(TEMPLATE / name) for name in (
            "fks_phase_space_data.f", "genps_fks_helpers.f", "FKSParams.f90",
            "genps_fks_radiation.f")]
        command += [str(work / name) for name in (
            "fks_phase_space.f", "maps.f", "local_kernel.f")]
        command += [str(TEMPLATE / name) for name in (
            "resonance_recoil.f", "initial_recoil.f", "boostwdir2.f")]
        command += [str(ROOT / name) for name in (
            "Template/NLO/Source/timing.f", "HELAS/boostx.F", "HELAS/rotxxx.F",
            "tests/input_files/check_pythia_local_matching.f90")]
        command += ["-o", str(cls.kernel_executable)]
        result = subprocess.run(command, cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def run_kernel(self, p, beam, partner, end, mass=0., kind=1,
                   matched=False, width=.01, scales=(1e8, 2e8), patched=False,
                   soft_patched=False, common_profile=False, factors=False):
        matched = matched or factors
        header = f"{beam} {partner} {end} {kind} {mass} .12 .20\n"
        if matched:
            header += f"{width} {scales[0]} {scales[1]}\n"
        data = header + "".join(" ".join(format(x, ".17g") for x in q) + "\n" for q in p)
        result = subprocess.run([str(self.kernel_executable),
                                 "matched_factors" if factors else
                                 "matched_density" if matched else "density",
                                 "2patch" if soft_patched else
                                 "patch" if patched else "stock",
                                 "1" if common_profile else "0"],
                                input=data, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        values = np.fromstring(result.stdout, sep=" ")
        self.assertEqual(len(values), 14 if factors else 12, result.stdout)
        return values

    def test_common_w_profile_has_the_same_factors_for_all_three_kernels(self):
        def profile(v):
            u = min(1., max(0., v/.01))
            return u*u/(u*u+(1-u)**2)

        for beam in (1, 2):
            for partner, mass in ((3, 173.), (4, 0.)):
                _, born, p = self.run_map(beam, partner, rad=(1e-3, .4, .3))
                if partner == 4:
                    d = 2*dot(born[beam-1], born[partner-1])
                    p = self.branch(born, beam, partner, .995, .2*d)
                for end in (beam, partner):
                    y = np.dot(p[end-1, 1:], p[4, 1:])/(
                        np.linalg.norm(p[end-1, 1:])*np.linalg.norm(p[4, 1:]))
                    for patched, soft in ((False, False), (True, False), (True, True)):
                        opts = dict(mass=mass, patched=patched, soft_patched=soft)
                        raw = self.run_kernel(p, beam, partner, end, **opts)
                        matched = self.run_kernel(p, beam, partner, end,
                            common_profile=True, factors=True, **opts)
                        self.assertAlmostEqual(matched[0]/raw[0], profile(raw[5]), delta=1e-12)
                        self.assertAlmostEqual(matched[11], profile(raw[5]), delta=1e-12)
                        self.assertAlmostEqual(matched[12], profile(1-y), delta=1e-12)
                        self.assertEqual(matched[13], 0.)

    def test_common_w_profile_retains_damping_and_finite_endpoint_remainders(self):
        _, born, _ = self.run_map(1, 4, mass=0.)
        d = 2*dot(born[0], born[3])
        for patched, soft in ((True, False), (True, True)):
            opts = dict(patched=patched, soft_patched=soft, common_profile=True)
            remainders = []
            for w in (1e-5, 1e-6, 1e-7):
                p = self.branch(born, 1, 4, 1-w, .5*d/(1-w))
                matched = self.run_kernel(p, 1, 4, 4, matched=True, **opts)
                # G(w) removes the raw 1/s final-collinear pole before the
                # native C-SC replacement restores the QCD endpoint.
                remainders.append(matched[0])
            self.assertLess(remainders[-1]/remainders[0], .02)
            p = self.branch(born, 1, 4, .995, .2*d)
            unbounded = self.run_kernel(p, 1, 4, 4, matched=True, **opts)
            scale = math.sqrt(unbounded[5]*unbounded[2])
            damped = self.run_kernel(p, 1, 4, 4, matched=True,
                scales=(scale/2, 3*scale/2), **opts)
            self.assertAlmostEqual(damped[0]/unbounded[0], .5, delta=1e-12)
            self.assertAlmostEqual(1-damped[11], .5*(1-unbounded[11]), delta=1e-14)
            outside = self.run_kernel(p, 1, 4, 4, matched=True, scales=(0., scale/2), **opts)
            self.assertEqual(outside[0], 0.)
            self.assertEqual(outside[11], 1.)

    def test_common_profile_card_default_validation_and_local_scope(self):
        work = Path(self.tempdir.name)
        card = work / 'FKS_params.dat'
        for mode, local, success in ((None, 0, True), (0, 0, True), (1, 1, True),
                                     (1, 0, False), (-1, 1, False), (2, 1, False)):
            text = f'#FKSPythiaLocalRecoil\n{local}\n#FKSExplicitSum\n.true.\n'
            if mode is not None:
                text += f'#PythiaLocalGProfile\n{mode}\n'
            card.write_text(text)
            result = subprocess.run([str(self.kernel_executable), 'parameters'],
                                    cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, success, result.stdout+result.stderr)
            if success:
                self.assertIn(f'PROFILE           {mode or 0}', result.stdout)
            else:
                self.assertIn('PythiaLocalGProfile', result.stdout)

    def test_common_w_soft_completion_cancels_with_native_endpoint_measures(self):
        for kernel in ('stock', 'patch', '2patch'):
            for mass in (0., 173.):
                for angle in ((.3, .7), (.7, .2)):
                    result = subprocess.run(
                        [str(self.kernel_executable), 'common_soft', kernel, '1'],
                        input=f'{mass} 0.0001 {angle[0]} {angle[1]}\n',
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    values = np.fromstring(result.stdout, sep=' ')
                    self.assertEqual(len(values), 4)
                    self.assertLess(abs(values[2]), 2e-6)

    def test_two_patches_reproduce_massive_soft_current_without_G(self):
        for beam in (1, 2):
            _, _, p = self.run_map(beam, 3, rad=(1e-4, .4, .3))
            pa, pf, k = p[beam-1], p[2], p[4]
            expected = 2*(4/3)*(dot(pa, pf)/(dot(pa, k)*dot(pf, k))
                                -173.**2/(2*dot(pf, k)**2))
            values = [self.run_kernel(p, beam, 3, end, mass=173.,
                                      soft_patched=True) for end in (beam, 3)]
            self.assertAlmostEqual(sum(v[0] for v in values)/expected,
                                   1., delta=1e-7)
            for end, raw in zip((beam, 3), values):
                for width in (.005, .01, .5):
                    matched = self.run_kernel(p, beam, 3, end, mass=173.,
                                              soft_patched=True, matched=True,
                                              width=width)
                    np.testing.assert_allclose(raw, matched, rtol=1e-14)
                    self.assertEqual(matched[-1], 1.)

    def test_soft_patch_is_bounded_and_matches_invariant_ratio(self):
        for beam in (1, 2):
            _, born, _ = self.run_map(beam, 3)
            d = 2*dot(born[beam-1], born[2])
            for z in (.4, .75, .99):
                max_delta = (1-z)*d/((1-z)*d+z*173.**2)
                for fraction in (1e-4, .1, .75, .999999):
                    q2 = fraction*max_delta*d/z
                    p = self.branch(born, beam, 3, z, q2, .43)
                    one = sum(self.run_kernel(p, beam, 3, end, mass=173.,
                                             patched=True)[0] for end in (beam, 3))
                    two = sum(self.run_kernel(p, beam, 3, end, mass=173.,
                                             soft_patched=True)[0] for end in (beam, 3))
                    pa, pf, k = p[beam-1], p[2], p[4]
                    ratio = 1-173.**2*dot(pa, k)/(2*dot(pa, pf)*dot(pf, k))
                    self.assertGreaterEqual(two/one, -1e-12)
                    self.assertLessEqual(two/one, 1+1e-12)
                    self.assertAlmostEqual(two/one, ratio, delta=2e-10)

    def test_soft_patch_preserves_massless_line_and_conversion(self):
        _, _, p = self.run_map(1, 4)
        one = self.run_kernel(p, 1, 4, 4, patched=True)
        two = self.run_kernel(p, 1, 4, 4, soft_patched=True)
        np.testing.assert_allclose(one, two, rtol=1e-14)
        for kind in (1, 2):
            _, born, _ = self.run_map(1, 3)
            d = 2*dot(born[0], born[2])
            p = self.branch(born, 1, 3, .7, d*1e-9)
            one = self.run_kernel(p, 1, 3, 1, mass=173., kind=kind, patched=True)
            two = self.run_kernel(p, 1, 3, 1, mass=173., kind=kind, soft_patched=True)
            self.assertAlmostEqual(two[0]/one[0], 1., delta=1e-8)

    def test_two_patch_soft_limit_in_common_S_coordinates(self):
        for mass in (0., 173.):
            for angle in ((.3, .7), (.7, .2)):
                result = subprocess.run([str(self.kernel_executable), 'soft', '2patch'],
                    input=f'{mass} 0.0001 {angle[0]} {angle[1]}\n',
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                values = np.fromstring(result.stdout, sep=' ')
                self.assertEqual(len(values), 4)
                self.assertLess(abs(values[2]), 2e-6)

    def test_soft_ratio_is_Lorentz_invariant(self):
        _, _, p = self.run_map(1, 3, rad=(.2, .4, .3))
        shifted = np.array([boost(q, np.array([.1, .2, -.1])) for q in p])
        a = self.run_kernel(p, 1, 3, 3, mass=173., soft_patched=True)
        b = self.run_kernel(shifted, 1, 3, 3, mass=173., soft_patched=True)
        np.testing.assert_allclose(a[:8], b[:8], rtol=2e-12)

    def test_patched_density_matches_corrected_IF_probability(self):
        for beam in (1, 2):
            for partner in (3, 4):
                _, _, p = self.run_map(beam, partner)
                mass = 173. if partner == 3 else 0.
                a = self.run_kernel(p, beam, partner, beam, mass, patched=True)
                b = self.run_kernel(p, beam, partner, partner, mass, patched=True)
                z, q2, zeta, w = a[1], a[2], a[4], a[5]
                expected = 2*(4/3)*(z*z+zeta*zeta)/(z*q2*w)
                self.assertAlmostEqual((a[0]+b[0])/expected, 1., delta=2e-12)
                stock = sum(self.run_kernel(p, beam, partner, end, mass)[0]
                            for end in (beam, partner))
                self.assertAlmostEqual((a[0]+b[0])/stock,
                                       (z*z+zeta*zeta)/(1+z*z), delta=2e-12)

    def test_patch_repairs_hard_final_collinear_limit_without_G(self):
        for beam in (1, 2):
            _, born, _ = self.run_map(beam, 4, mass=0.)
            d = 2*dot(born[beam-1], born[3])
            for fraction in (.2, .5, .8):
                p = self.branch(born, beam, 4, 1-1e-8,
                                d*(1-fraction)/(1-1e-8))
                raw = self.run_kernel(p, beam, 4, 4, patched=True)
                matched = self.run_kernel(p, beam, 4, 4, patched=True,
                                          matched=True)
                invariant = 2*dot(p[3], p[4])
                ap = (4/3)*(1+fraction*fraction)/(1-fraction)
                self.assertAlmostEqual(raw[0]*invariant/(2*ap), 1., delta=1e-6)
                self.assertEqual(matched[-1], 1.)  # No G completion.
                np.testing.assert_allclose(raw, matched, rtol=1e-14)

    def test_patch_preserves_initial_collinear_and_gluon_conversion(self):
        _, born, _ = self.run_map(1, 4, mass=0.)
        d = 2*dot(born[0], born[3])
        p = self.branch(born, 1, 4, .7, d*1e-9)
        stock = self.run_kernel(p, 1, 4, 1)
        patched = self.run_kernel(p, 1, 4, 1, patched=True)
        self.assertAlmostEqual(patched[0]/stock[0], 1., delta=1e-8)
        for matched in (False, True):
            stock = self.run_kernel(p, 1, 4, 1, kind=2, matched=matched)
            patched = self.run_kernel(p, 1, 4, 1, kind=2, matched=matched,
                                      patched=True)
            np.testing.assert_allclose(stock, patched, rtol=1e-14)

    def test_patch_keeps_only_soft_energy_G_profile(self):
        _, _, p = self.run_map(1, 3, rad=(1e-3, .4, .3))
        raw = self.run_kernel(p, 1, 3, 3, mass=173., patched=True)
        matched = self.run_kernel(p, 1, 3, 3, mass=173., patched=True,
                                  matched=True)
        total = p[0]+p[1]
        soft_energy = 2*dot(total, p[4])/dot(total, total)
        u = soft_energy/.01
        profile = u*u/(u*u+(1-u)**2)
        self.assertLess(soft_energy, .01)
        self.assertAlmostEqual(matched[0]/raw[0], profile, delta=1e-12)
        self.assertAlmostEqual(matched[-1], profile, delta=1e-12)

    def test_G_profile_endpoint_scaling_and_compact_support(self):
        for width in (.005, .01, .02):
            values = []
            for fraction in (0., 1e-5, 1e-4, .25, .5, .75, 1., 2.):
                result = subprocess.run([str(self.kernel_executable), "profile"],
                                        input=f"{width} {width*fraction}\n",
                                        capture_output=True, text=True, check=True)
                values.append(float(result.stdout))
            self.assertEqual(values[0], 0.)
            self.assertEqual(values[-2:], [1., 1.])
            self.assertTrue(all(x <= y for x, y in zip(values, values[1:])))
            self.assertAlmostEqual(values[2]/values[1], 100., delta=.02)

    def test_G_suppresses_stock_final_collinear_remainder(self):
        for beam in (1, 2):
            _, born, _ = self.run_map(beam, 4, mass=0.)
            d = 2*dot(born[beam-1], born[3])
            for fraction in (.2, .5, .8):
                residual = []
                for w in (1e-5, 1e-6, 1e-7):
                    p = self.branch(born, beam, 4, 1-w, d*(1-fraction)/(1-w))
                    raw = self.run_kernel(p, beam, 4, 4)
                    matched = self.run_kernel(p, beam, 4, 4, matched=True)
                    # The raw difference still has a 1/s pole. Its G-weighted
                    # remainder must vanish at least linearly in s ~ w.
                    residual.append(matched[0]*(1-fraction*fraction)/2)
                    self.assertAlmostEqual(matched[0]/raw[0], matched[-1], delta=1e-15)
                self.assertLess(residual[-1]/residual[0], .02)

    def test_G_preserves_hard_stock_density_and_gluon_conversion(self):
        _, _, p = self.run_map(1, 4)
        for kind in (1, 2):
            raw = self.run_kernel(p, 1, 4, 1, kind=kind)
            matched = self.run_kernel(p, 1, 4, 1, kind=kind, matched=True)
            self.assertGreater(raw[5], .01)
            np.testing.assert_allclose(raw, matched, rtol=1e-14)
        # g -> q qbar has no soft or outgoing collinear mismatch, even w -> 0.
        _, born, _ = self.run_map(1, 4)
        d = 2*dot(born[0], born[3])
        p = self.branch(born, 1, 4, 1-1e-6, 1e-6*d)
        raw = self.run_kernel(p, 1, 4, 1, kind=2)
        matched = self.run_kernel(p, 1, 4, 1, kind=2, matched=True)
        np.testing.assert_allclose(raw, matched, rtol=1e-14)

    def test_G_replacement_shares_starting_scale_damping(self):
        _, born, _ = self.run_map(1, 4)
        d = 2*dot(born[0], born[3])
        p = self.branch(born, 1, 4, .995, .2*d)
        raw = self.run_kernel(p, 1, 4, 4)
        profile = self.run_kernel(p, 1, 4, 4, matched=True)
        scale = math.sqrt(raw[5]*raw[2])
        damped = self.run_kernel(p, 1, 4, 4, matched=True, scales=(scale/2, 3*scale/2))
        damping = damped[0]/profile[0]
        self.assertAlmostEqual(damping, .5, delta=1e-12)
        self.assertAlmostEqual(1-damped[-1], (1-profile[-1])*damping, delta=1e-14)
        outside = self.run_kernel(p, 1, 4, 4, matched=True, scales=(0., scale/2))
        self.assertEqual(outside[0], 0.)
        self.assertEqual(outside[-1], 1.)

    @staticmethod
    def branch(born, beam, partner, z, q2, phi=0.):
        """Literal IF branch in the Born dipole rest frame, independent of FKS."""
        ba, bf = born[beam-1], born[partner-1]
        pair = ba+bf
        md = math.sqrt(dot(pair, pair))
        mass2 = max(0., dot(bf, bf))
        d = 2*dot(ba, bf)
        axis = boost(ba, -pair[1:]/pair[0])[1:]
        axis /= np.linalg.norm(axis)
        transverse = np.cross(axis, [1., 0., 0.])
        transverse /= np.linalg.norm(transverse)
        energy = (1-z)*d/(2*z*md)+(z-.5)*q2/md
        kz = (1-z)*d/(2*z*md)-q2/(2*md)-z*q2*mass2/(md*d)
        pt2 = (1-z)*q2*(1-z*q2/d)-mass2*(z*q2/d)**2
        transverse = math.cos(phi)*transverse + math.sin(phi)*np.cross(axis, transverse)
        k = boost(np.r_[energy, kz*axis+math.sqrt(pt2)*transverse], pair[1:]/pair[0])
        p = born.copy()
        p[4] = k
        p[beam-1] = ba/z
        p[partner-1] = bf+(1/z-1)*ba-k
        return p

    def test_density_partition_backward_pdfs_and_shower_scale(self):
        for beam in (1, 2):
            for partner in (3, 4):
                _, born, p = self.run_map(beam, partner)
                mass = 173. if partner == 3 else 0.
                incoming = self.run_kernel(p, beam, partner, beam, mass)
                outgoing = self.run_kernel(p, beam, partner, partner, mass)
                density, z, q2, d, zeta, w, pt2, chi = incoming[:8]
                reference = 2*(4/3)*(1+z*z)/(z*q2*w)
                self.assertAlmostEqual((density+outgoing[0])/reference, 1., delta=2e-12)
                self.assertAlmostEqual(chi+outgoing[7], 1., delta=2e-15)
                self.assertAlmostEqual(q2, 2*dot(p[beam-1], p[4]), delta=1e-8)
                self.assertAlmostEqual(d, 2*dot(born[beam-1], born[partner-1]), delta=1e-8)
                self.assertAlmostEqual(incoming[8+beam-1], (.12, .20)[beam-1]/z)
                np.testing.assert_allclose(incoming[8:10], outgoing[8:10], rtol=1e-13)
                self.assertEqual(incoming[10], 1.)  # No additional PDF Jacobian.
                self.assertEqual(incoming[11], 1.)  # No G replacement.

    def test_stock_final_collinear_coefficient_differs_from_qcd(self):
        for beam in (1, 2):
            _, born, _ = self.run_map(beam, 4, mass=0.)
            d = 2*dot(born[beam-1], born[3])
            for fraction in (.2, .5, .8):
                z = 1/(1+1e-8)
                p = self.branch(born, beam, 4, z, d*(1-fraction)/z)
                values = self.run_kernel(p, beam, 4, 4)
                invariant = 2*dot(p[3], p[4])
                coefficient = values[0]*invariant/2
                ap = (4/3)*(1+fraction*fraction)/(1-fraction)
                self.assertAlmostEqual(coefficient/ap, 2/(1+fraction*fraction), delta=1e-6)
                self.assertAlmostEqual(coefficient-ap, (4/3)*(1+fraction), delta=1e-6)

    def test_shower_distribution_with_independent_cartesian_measure(self):
        for beam in (1, 2):
            for partner in (3, 4):
                _, born, _ = self.run_map(beam, partner)
                mass = 173. if partner == 3 else 0.
                d = 2*dot(born[beam-1], born[partner-1])
                variables = np.array([.72, .18*d, .43])
                p = self.branch(born, beam, partner, *variables)
                derivatives = []
                for axis, step in enumerate((1e-6, .18*d*1e-6, 1e-6)):
                    up, down = variables.copy(), variables.copy()
                    up[axis] += step
                    down[axis] -= step
                    derivatives.append((self.branch(born, beam, partner, *up)[4, 1:]-
                                        self.branch(born, beam, partner, *down)[4, 1:])/(2*step))
                z, q2, _ = variables
                # Emission measure, the IF on-shell constraint and flux ratio.
                measure = abs(np.linalg.det(derivatives))/(2*p[4, 0]*(2*math.pi)**3)
                measure *= d/(d-z*q2)*z
                density = sum(self.run_kernel(p, beam, partner, end, mass)[0]
                              for end in (beam, partner))
                splitting = (4/3)*(1+z*z)/(1-z)
                # alpha_s=1/(4*pi), dQ2/Q2=dt/t at fixed z, flat dipole phi.
                shower = splitting/(16*math.pi**3*z*q2)
                self.assertAlmostEqual(density*measure/shower, 1., delta=2e-8)

    def test_initial_collinear_and_gluon_conversion(self):
        for beam in (1, 2):
            _, born, _ = self.run_map(beam, 4, mass=0.)
            d = 2*dot(born[beam-1], born[3])
            for z in (.3, .7):
                p = self.branch(born, beam, 4, z, d*1e-9)
                qcd = self.run_kernel(p, beam, 4, beam)
                self.assertAlmostEqual(qcd[0]*z*qcd[2]/2/((4/3)*(1+z*z)/(1-z)),
                                       1., delta=1e-7)
                converted = self.run_kernel(p, beam, 4, beam, kind=2)
                ap = .5*(z*z+(1-z)**2)
                self.assertAlmostEqual(converted[0]*z*converted[2]/2/ap, 1., delta=1e-12)

    def test_stock_soft_limit_has_no_top_self_eikonal(self):
        for beam in (1, 2):
            _, _, p = self.run_map(beam, 3, rad=(1e-4, .4, .3))
            pa, pf, k = p[beam-1], p[2], p[4]
            eikonal = 2*(4/3)*(dot(pa, pf)/(dot(pa, k)*dot(pf, k))
                                      -173.**2/(2*dot(pf, k)**2))
            value = sum(self.run_kernel(p, beam, 3, end, mass=173.)[0] for end in (beam, 3))
            pair = 2*(4/3)*dot(pa, pf)/(dot(pa, k)*dot(pf, k))
            self_term = (4/3)*173.**2/dot(pf, k)**2
            self.assertAlmostEqual(value/pair, 1., delta=1e-7)
            self.assertAlmostEqual((value-eikonal)/self_term, 1., delta=1e-6)

    def test_stock_S_soft_difference_survives_common_coordinates(self):
        for angle in ((.3, .7), (.7, .2)):
            r, phi = angle
            st = 2*r*math.sqrt(1-r*r)
            n = np.array([1., st*math.cos(2*math.pi*phi),
                          st*math.sin(2*math.pi*phi), 1-2*r*r])
            for mass in (0., 173.):
                _, born, _ = self.run_map(1, 3, mass=mass)
                self_term = mass**2/(2*dot(born[2], n)**2)
                current = sum(dot(born[a], born[a+2])/
                              (dot(born[a], n)*dot(born[a+2], n)) for a in (0, 1))
                expected = self_term/(current-self_term)
                for energy in (1e-3, 1e-4):
                    data = f"{mass} {energy} {angle[0]} {angle[1]}\n"
                    result = subprocess.run([str(self.kernel_executable), "soft"], input=data,
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    values = np.fromstring(result.stdout, sep=" ")
                    self.assertEqual(len(values), 4, result.stdout)
                    self.assertAlmostEqual(values[2], expected, delta=2e-5)
                    if mass:
                        self.assertGreater(values[2], 1e-4)
                self.assertLess(values[3], 1e-6)  # The same physical soft-energy regulator.


if __name__ == "__main__":
    unittest.main()
