"""FKS kinematic reshuffling of the FxFx EW-Sudakov clustering (fxfx_ewsudakov).

Every fallback method of _fks_fsr_mapping has an engineered configuration
except fks_fsr_degenerate_collinear (reachable only for 0 < R < 1e-10, since
R <= 0 is caught as fks_fsr_unreachable_target) and fks_fsr_superluminal
(boundary-only: every real root of the squared quadratic has |beta| <= 1).
"""
from __future__ import division

import math
import os
import random
import sys
import unittest

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import madgraph.various.fxfx_ewsudakov as fx  # noqa: E402  (path bootstrap above)
import madgraph.various.lhe_parser as lhe_parser  # noqa: E402

FourMomentum = fx.FourMomentum


def _m2(p):
    return p.E**2 - p.px**2 - p.py**2 - p.pz**2


def _mk_particle(event, pid, status, E, px, py, pz, mass=None):
    part = lhe_parser.Particle(event=event)
    part.pid = pid
    part.status = status
    part.E = E
    part.px = px
    part.py = py
    part.pz = pz
    if mass is None:
        mass = math.sqrt(max(0.0, E**2 - px**2 - py**2 - pz**2))
        if mass < 1e-9:
            mass = 0.0
    part.mass = mass
    part.mother1 = 0
    part.mother2 = 0
    part.color1 = 0
    part.color2 = 0
    part.vtim = 0
    part.helicity = 9
    return part


def _mk_event(entries):
    """entries: list of (pid, status, E, px, py, pz)."""
    event = lhe_parser.Event()
    for pid, status, E, px, py, pz in entries:
        event.append(_mk_particle(event, pid, status, E, px, py, pz))
    return event


def _fs_sum(event):
    total = FourMomentum()
    for idx in range(2, len(event)):
        total += FourMomentum(event[idx])
    return total


def _split_massless_z(E, pz):
    """Split (E,0,0,pz) into two massless momenta along +/- z (light-cone)."""
    plus = (E + pz) / 2.0
    minus = (E - pz) / 2.0
    return (plus, 0.0, 0.0, plus), (minus, 0.0, 0.0, -minus)


class TestFSRSuccessPath(unittest.TestCase):
    """The mapping must place the mother exactly on the pole mass."""

    def test_massless_spectator_onshell(self):
        beams = [FourMomentum([250, 0, 0, 250]), FourMomentum([250, 0, 0, -250])]
        spec = FourMomentum([100, 100, 0, 0])
        p_a = FourMomentum([150, 150, 0, 0])
        p_b = FourMomentum([250, -250, 0, 0])
        momenta = beams + [p_a, p_b, spec]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self.assertEqual(res["method"], "fks_fsr_massive_onshell")
        mother = res["mother"]
        self.assertAlmostEqual(math.sqrt(_m2(mother)), fx.MW_POLE, places=6)
        total = FourMomentum(mother)
        for p in res["spectators"].values():
            total += p
        q = beams[0] + beams[1]
        for comp in ("E", "px", "py", "pz"):
            self.assertAlmostEqual(getattr(total, comp), getattr(q, comp), places=6)

    def test_massive_spectator_onshell_random(self):
        random.seed(7)
        n_ok = 0
        for _ in range(200):
            ecm = random.uniform(300.0, 2000.0)
            beams = [
                FourMomentum([ecm / 2, 0, 0, ecm / 2]),
                FourMomentum([ecm / 2, 0, 0, -ecm / 2]),
            ]
            q = beams[0] + beams[1]
            parts = []
            for _n in range(3):
                px, py, pz = (random.uniform(-ecm / 6, ecm / 6) for _c in range(3))
                E = math.sqrt(px**2 + py**2 + pz**2)
                parts.append(FourMomentum([E, px, py, pz]))
            tot = parts[0] + parts[1] + parts[2]
            last = FourMomentum([q.E - tot.E, q.px - tot.px, q.py - tot.py, q.pz - tot.pz])
            if last.E <= 0 or _m2(last) <= 0:
                continue
            momenta = beams + parts + [last]
            res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 23}], momenta)
            if res["method"] != "fks_fsr_massive_onshell":
                continue
            n_ok += 1
            self.assertAlmostEqual(
                math.sqrt(_m2(res["mother"])) / fx.MZ_POLE, 1.0, places=6
            )
            # spectator-system invariant mass is preserved by the boost
            krec = FourMomentum()
            krec_new = FourMomentum()
            for idx, p in res["spectators"].items():
                krec += momenta[idx]
                krec_new += p
            self.assertAlmostEqual(_m2(krec_new) / _m2(krec), 1.0, places=6)
        self.assertGreater(n_ok, 50)

    def test_no_spectators_beam_scaling(self):
        beams = [FourMomentum([50, 0, 0, 50]), FourMomentum([50, 0, 0, -50])]
        p_a = FourMomentum([50, 50, 0, 0])
        p_b = FourMomentum([50, -50, 0, 0])
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 23}], beams + [p_a, p_b])
        self.assertEqual(res["method"], "fks_fsr_no_spectators_beam_scaled")
        self.assertAlmostEqual(math.sqrt(_m2(res["mother"])), fx.MZ_POLE, places=6)
        # beam rapidity (E ratio) is preserved
        self.assertAlmostEqual(res["beams"][0].E / res["beams"][1].E, 1.0, places=9)


class TestFSRFallbackBranches(unittest.TestCase):
    """One engineered kinematic configuration per fallback branch."""

    def _naive(self, momenta, i, j):
        return momenta[i] + momenta[j]

    def _assert_fallback(self, res, momenta, method):
        self.assertEqual(res["method"], method)
        naive = self._naive(momenta, 2, 3)
        for comp in ("E", "px", "py", "pz"):
            self.assertAlmostEqual(
                getattr(res["mother"], comp), getattr(naive, comp), places=9
            )

    def test_zero_krec_momentum(self):
        beams = [FourMomentum([100, 0, 0, 100]), FourMomentum([100, 0, 0, -100])]
        p_a = FourMomentum([50, 50, 0, 0])
        p_b = FourMomentum([50, -50, 0, 0])
        spec1 = FourMomentum([50, 0, 0, 50])
        spec2 = FourMomentum([50, 0, 0, -50])
        momenta = beams + [p_a, p_b, spec1, spec2]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 23}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_zero_krec_momentum")

    def test_negative_energy(self):
        # sqrt(s) = 60 GeV < M_W: no on-shell W + spectator fits in the event
        beams = [FourMomentum([30, 0, 0, 30]), FourMomentum([30, 0, 0, -30])]
        spec = FourMomentum([10, 10, 0, 0])
        p_a = FourMomentum([20, -20, 0, 0])
        p_b = FourMomentum([30, 10, 0, 0])
        pair = p_a + p_b
        self.assertAlmostEqual(pair.E, 50.0)
        momenta = beams + [p_a, p_b, spec]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_negative_energy")

    def test_negative_r(self):
        # sqrt(s) = M_W exactly: target boosted spectator energy is exactly zero
        M = fx.MW_POLE
        beams = [FourMomentum([M / 2, 0, 0, M / 2]), FourMomentum([M / 2, 0, 0, -M / 2])]
        spec = FourMomentum([10, 10, 0, 0])
        p_a = FourMomentum([(M - 20) / 2, (M - 20) / 2, 0, 0])
        p_b = FourMomentum([M / 2, -M / 2, 0, 0])
        momenta = beams + [p_a, p_b, spec]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_negative_r")

    def test_unreachable_target(self):
        # sqrt(s)=60, light massive spectator system (m_rec=5): R < 0, i.e. the
        # event cannot host an on-shell W plus this spectator system. The
        # squared quadratic still has |beta|<1 roots, all on the spurious
        # branch gamma(A+beta B) = -R, with the mother at 28 GeV; the R <= 0
        # guard rejects the configuration first.
        beams = [FourMomentum([30, 0, 0, 30]), FourMomentum([30, 0, 0, -30])]
        E_r, p_r = math.sqrt(34.0), 3.0  # m_rec^2 = 25
        (k1E, _x, _y, k1z), (k2E, _x2, _y2, k2z) = _split_massless_z(E_r, p_r)
        spec1 = FourMomentum([k1E, 0, 0, k1z])
        spec2 = FourMomentum([k2E, 0, 0, k2z])
        pair = FourMomentum([60 - E_r, 0, 0, -p_r])
        (paE, _px, _py, paz), (pbE, _px2, _py2, pbz) = _split_massless_z(pair.E, pair.pz)
        p_a = FourMomentum([paE, 0, 0, paz])
        p_b = FourMomentum([pbE, 0, 0, pbz])
        momenta = beams + [p_a, p_b, spec1, spec2]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_unreachable_target")
        self.assertLess(math.sqrt(max(0.0, _m2(res["mother"]))), fx.MW_POLE - 10.0)

    def test_no_solution(self):
        # sqrt(s)=100, m_rec=30: 0 < R but R^2 < A^2 - B^2 -> negative discriminant
        beams = [FourMomentum([50, 0, 0, 50]), FourMomentum([50, 0, 0, -50])]
        E_r = 30.4
        p_r = math.sqrt(E_r**2 - 900.0)
        (k1E, _x, _y, k1z), (k2E, _x2, _y2, k2z) = _split_massless_z(E_r, p_r)
        spec1 = FourMomentum([k1E, 0, 0, k1z])
        spec2 = FourMomentum([k2E, 0, 0, k2z])
        pair = FourMomentum([100 - E_r, 0, 0, -p_r])
        (paE, _px, _py, paz), (pbE, _px2, _py2, pbz) = _split_massless_z(pair.E, pair.pz)
        p_a = FourMomentum([paE, 0, 0, paz])
        p_b = FourMomentum([pbE, 0, 0, pbz])
        momenta = beams + [p_a, p_b, spec1, spec2]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_no_solution")

    def test_degenerate_lightlike_q(self):
        # q lightlike along the (massless) spectator direction: denominator = 0
        beams = [FourMomentum([50, 0, 0, 50]), FourMomentum([30, 0, 0, 30])]
        spec = FourMomentum([10, 0, 0, 10])
        p_a = FourMomentum([35, 0, 0, 35])
        p_b = FourMomentum([35, 0, 0, 35])
        momenta = beams + [p_a, p_b, spec]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_degenerate")

    def test_collinear_unphysical(self):
        # Massive spectator collinear with q direction (B = 0), gamma^2 < 1
        beams = [FourMomentum([60, 0, 0, 60]), FourMomentum([40, 0, 0, -40])]
        q = beams[0] + beams[1]  # (100, 0, 0, 20)
        k_E = 20.0
        k_z = k_E * q.pz / q.E  # 4.0 -> B = 0 exactly, m_rec^2 = 384
        spec = FourMomentum([k_E, 0, 0, k_z])
        pair = FourMomentum([q.E - k_E, 0, 0, q.pz - k_z])
        (paE, _px, _py, paz), (pbE, _px2, _py2, pbz) = _split_massless_z(pair.E, pair.pz)
        p_a = FourMomentum([paE, 0, 0, paz])
        p_b = FourMomentum([pbE, 0, 0, pbz])
        momenta = beams + [p_a, p_b, spec]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self._assert_fallback(res, momenta, "fks_fsr_collinear_unphysical")

    def test_threshold_collinear_is_unreachable(self):
        # B = 0 and R = 0: s + m_rec^2 = M^2 with a q-collinear massive
        # spectator, exactly at threshold (beta = +-1 would be needed).
        M2 = fx.MW_POLE**2
        beams = [FourMomentum([40, 0, 0, 40]), FourMomentum([24, 0, 0, -24])]
        q = beams[0] + beams[1]  # (64, 0, 0, 16)
        s = _m2(q)
        m_rec2 = M2 - s
        self.assertGreater(m_rec2, 0.0)
        ratio = q.pz / q.E
        k_E = math.sqrt(m_rec2 / (1.0 - ratio**2))
        k_z = k_E * ratio
        spec = FourMomentum([k_E, 0, 0, k_z])
        pair = FourMomentum([q.E - k_E, 0, 0, q.pz - k_z])
        p_a = FourMomentum([pair.E / 2, pair.px / 2, 0, pair.pz / 2])
        p_b = FourMomentum([pair.E / 2, pair.px / 2, 0, pair.pz / 2])
        momenta = beams + [p_a, p_b, spec]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        # R is an exact float 0 here (unreachable_target, while 0 < R < 1e-10
        # would give degenerate_collinear); only the non-success is pinned.
        self.assertNotIn(res["method"], fx.FKS_SUCCESS_METHODS)
        naive = self._naive(momenta, 2, 3)
        for comp in ("E", "px", "py", "pz"):
            self.assertAlmostEqual(
                getattr(res["mother"], comp), getattr(naive, comp), places=9
            )


def _onshell_tolerance(momenta, mother_mass):
    # floating-point accuracy of a mapping solved for the input momenta
    q2 = _m2(momenta[0] + momenta[1])
    return 1e-9 * max(q2, mother_mass**2)


def _within_pole(mother, mother_mass):
    # on-shell criterion of a successful mapping
    m = math.sqrt(max(0.0, _m2(mother)))
    return abs(m - mother_mass) <= 1e-4 * mother_mass


def _massless_momentum(E, theta, phi):
    return FourMomentum(
        [
            E,
            E * math.sin(theta) * math.cos(phi),
            E * math.sin(theta) * math.sin(phi),
            E * math.cos(theta),
        ]
    )


def _lhe_event(rows):
    """rows: (pid, status, mass, E, px, py, pz) with the LHE mass column."""
    event = lhe_parser.Event()
    for pid, status, mass, E, px, py, pz in rows:
        event.append(_mk_particle(event, pid, status, E, px, py, pz, mass=mass))
    return event


class TestFSRNumericalRobustness(unittest.TestCase):
    """Soft and nearly lightlike recoil systems: every success reaches the pole,
    and the spectators, the mother and the transported decay leaves follow one map."""

    def _assert_naive_fallback(self, res, momenta):
        self.assertNotIn(res["method"], fx.FKS_SUCCESS_METHODS)
        self.assertEqual(res["spectator_transform"], [])
        naive = momenta[2] + momenta[3]
        for comp in ("E", "px", "py", "pz"):
            self.assertEqual(getattr(res["mother"], comp), getattr(naive, comp))
        for idx, p in res["spectators"].items():
            for comp in ("E", "px", "py", "pz"):
                self.assertEqual(getattr(p, comp), getattr(momenta[idx], comp))

    @staticmethod
    def _soft_massless_recoil_event(E_rec):
        # sqrt(s) = 1 TeV, one massless spectator of energy E_rec; the pair
        # (2, 3) carries the remaining momentum and becomes a W
        beams = [FourMomentum([500, 0, 0, 500]), FourMomentum([500, 0, 0, -500])]
        spec = _massless_momentum(E_rec, 2.0, 1.3)
        p_a = FourMomentum([300, 300, 0, 0])
        p_b = beams[0] + beams[1] - p_a - spec
        return beams + [p_a, p_b, spec]

    def test_soft_massless_recoil_onshell(self):
        for E_rec in (1e-1, 1e-3, 3e-4, 1e-5, 1e-8):
            momenta = self._soft_massless_recoil_event(E_rec)
            res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
            self.assertEqual(res["method"], "fks_fsr_massive_onshell", E_rec)
            self.assertLessEqual(
                abs(_m2(res["mother"]) - fx.MW_POLE**2),
                _onshell_tolerance(momenta, fx.MW_POLE),
                E_rec,
            )
            total = res["mother"] + res["spectators"][4]
            q = momenta[0] + momenta[1]
            for comp in ("E", "px", "py", "pz"):
                self.assertAlmostEqual(
                    getattr(total, comp), getattr(q, comp), delta=1e-9 * q.E
                )

    def test_soft_massless_recoil_rescales_spectator(self):
        for E_rec in (1e-1, 1e-3, 3e-4, 1e-5, 1e-8):
            momenta = self._soft_massless_recoil_event(E_rec)
            res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
            self.assertEqual(res["spectator_transform"][0][0], "rescale", E_rec)
            spec = res["spectators"][4]
            r = res["spectator_transform"][0][1]
            for comp in ("E", "px", "py", "pz"):
                self.assertAlmostEqual(
                    getattr(spec, comp),
                    r * getattr(momenta[4], comp),
                    delta=1e-12 * spec.E,
                )
            # a decay leaf of the spectator follows the identical map
            leaf = fx._apply_fxfx_frame_transform(
                momenta[4], res["spectator_transform"]
            )
            for comp in ("E", "px", "py", "pz"):
                self.assertEqual(getattr(leaf, comp), getattr(spec, comp))

    def test_boosted_lightlike_spectator(self):
        # q Z -> q with beta_q = -0.995 and a gluon spectator whose LHE-rounded
        # momentum has m^2 = 5.8e-7 GeV^2: the clustering maps it as a lightlike
        # recoil system and the mother reaches the pole.
        event = _lhe_event(
            [
                (21, -1, 0.0, 8.520120244, 0.0, 0.0, 8.520120244),
                (2, -1, 0.0, 3419.6635543, 0.0, 0.0, -3419.6635543),
                (2, 1, 0.0, 392.37478144, 58.688524463, 77.749701885, -380.09026568),
                (
                    23,
                    1,
                    91.1876,
                    2929.8757065,
                    -75.57807841,
                    -76.403583926,
                    -2926.4837132,
                ),
                (21, 1, 0.0, 105.93318663, 16.889553947, -1.3461179586, -104.56945521),
            ]
        )
        q = FourMomentum(event[0]) + FourMomentum(event[1])
        res = fx._fks_fsr_mapping_cm(
            2, 3, [{"number": 3, "id": 2}], [FourMomentum(p) for p in event]
        )
        self.assertEqual(res["spectator_transform"][1][0], "rescale")
        method = fx.fxfx_merge_particles_kinematics(
            event, 2, 3, [{"number": 3, "id": 2}]
        )
        self.assertEqual(method, "fks_fsr_massive_onshell")
        self.assertEqual(event[2].pid, 2)
        self.assertEqual(event[2].mass, 0.0)
        self.assertLessEqual(abs(_m2(FourMomentum(event[2]))), 1e-12 * _m2(q))
        event.check_kinematics_only()

    def test_rounded_soft_photon_is_lightlike(self):
        # A 10 MeV photon whose momentum carries rounding-level mass
        # (m^2 / E^2 = 1e-8), in a frame where q is not at rest.
        E = 0.01
        photon_p = E * (1.0 - 5e-9)
        photon = FourMomentum([E, photon_p * 0.6, photon_p * 0.0, photon_p * 0.8])
        beams = [FourMomentum([800, 0, 0, 800]), FourMomentum([200, 0, 0, -200])]
        p_a = _massless_momentum(250.0, 1.1, 0.4)
        p_b = beams[0] + beams[1] - p_a - photon
        momenta = beams + [p_a, p_b, photon]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self.assertEqual(res["method"], "fks_fsr_massive_onshell")
        self.assertEqual(res["spectator_transform"][0][0], "rescale")
        self.assertLessEqual(
            abs(_m2(res["mother"]) - fx.MW_POLE**2),
            _onshell_tolerance(momenta, fx.MW_POLE),
        )

    def test_massive_spectator_keeps_its_mass(self):
        # A 600 GeV muon: m^2 / E^2 = 3e-8 is small, but m^2 = 0.011 GeV^2 is a
        # physical mass, so the recoil system is boosted, not rescaled.
        m_mu = 0.1056583755
        E = 600.0
        p = math.sqrt(E**2 - m_mu**2)
        muon = FourMomentum(
            [
                E,
                p * math.sin(2.0) * math.cos(1.3),
                p * math.sin(2.0) * math.sin(1.3),
                p * math.cos(2.0),
            ]
        )
        beams = [FourMomentum([1000, 0, 0, 1000]), FourMomentum([1000, 0, 0, -1000])]
        p_a = FourMomentum([300, 300, 0, 0])
        p_b = beams[0] + beams[1] - p_a - muon
        momenta = beams + [p_a, p_b, muon]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self.assertEqual(res["method"], "fks_fsr_massive_onshell")
        self.assertEqual(res["spectator_transform"][0][0], "boost_along_direction")
        m_after = math.sqrt(max(0.0, _m2(res["spectators"][4])))
        self.assertAlmostEqual(m_after / m_mu, 1.0, delta=1e-6)

    def test_wide_angle_soft_pair_is_massive(self):
        # Two 0.7 MeV photons at 90 degrees: m^2_rec = 9.8e-7 GeV^2 is below
        # 1e-6 GeV^2 but comparable to E^2_rec, so the recoil system is massive.
        g1 = FourMomentum([0.7e-3, 0.7e-3, 0, 0])
        g2 = FourMomentum([0.7e-3, 0, 0.7e-3, 0])
        beams = [FourMomentum([500, 0, 0, 500]), FourMomentum([500, 0, 0, -500])]
        p_a = FourMomentum([300, 300, 0, 0])
        p_b = beams[0] + beams[1] - p_a - g1 - g2
        momenta = beams + [p_a, p_b, g1, g2]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self.assertNotIn("rescale", [op for op, _ref in res["spectator_transform"]])
        if res["method"] in fx.FKS_SUCCESS_METHODS:
            self.assertTrue(_within_pole(res["mother"], fx.MW_POLE))
        else:
            self._assert_naive_fallback(res, momenta)
        # the merged event keeps a mass label consistent with its kinematics
        event = _lhe_event(
            [
                (21, -1, 0.0, 500, 0, 0, 500),
                (21, -1, 0.0, 500, 0, 0, -500),
                (13, 1, 0.0, p_a.E, p_a.px, p_a.py, p_a.pz),
                (-14, 1, 0.0, p_b.E, p_b.px, p_b.py, p_b.pz),
                (22, 1, 0.0, g1.E, g1.px, g1.py, g1.pz),
                (22, 1, 0.0, g2.E, g2.px, g2.py, g2.pz),
            ]
        )
        fx.fxfx_merge_particles_kinematics(event, 2, 3, [{"number": 3, "id": -24}])
        m_kin = math.sqrt(max(0.0, _m2(FourMomentum(event[2]))))
        if event[2].mass != fx.MW_POLE:
            self.assertAlmostEqual(event[2].mass, m_kin, delta=1e-9 * max(1.0, m_kin))
        event.check_kinematics_only()

    def test_soft_massive_recoil_fallbacks(self):
        # Wide-angle two-photon recoil systems of 0.35 MeV and 14 MeV: the boost
        # needs |beta| -> 1. The first fails the unsquared root check, the
        # second misses the pole by more than 1e-4 M. Both labels are stable
        # under 1e-12 relative perturbations of the inputs.
        cases = {
            "fks_fsr_spurious_root": [
                [238.4639599737477, 0.0, 0.0, 238.4639599737477],
                [238.4639599737477, 0.0, 0.0, -238.4639599737477],
                [
                    143.07827049475074,
                    117.4467050112949,
                    49.65567044191677,
                    64.89974855669855,
                ],
                [
                    333.8492978210851,
                    -117.44671385487138,
                    -49.65575930015483,
                    -64.89968880261101,
                ],
                [
                    0.00017581582981151224,
                    1.364203365704181e-05,
                    0.00013707204056107185,
                    0.00010925363438547318,
                ],
                [
                    0.00017581582981151224,
                    -4.798457174322873e-06,
                    -4.821380249929515e-05,
                    -0.0001690077219230405,
                ],
            ],
            "fks_fsr_inexact": [
                [1494.1205279391247, 0.0, 0.0, 1494.1205279391247],
                [1494.1205279391247, 0.0, 0.0, -1494.1205279391247],
                [
                    896.4681187528728,
                    735.8715361258292,
                    311.1214953364348,
                    406.63446164798694,
                ],
                [
                    2091.7589437567035,
                    -735.8704682162353,
                    -311.12022499915327,
                    -406.63049952355396,
                ],
                [
                    0.006996684336627818,
                    -0.004486135781554201,
                    -0.005336505604767825,
                    0.0005915111039986608,
                ],
                [
                    0.006996684336627818,
                    0.0034182261876529005,
                    0.00406616832325441,
                    -0.004553635536954992,
                ],
            ],
        }
        for method, rows in cases.items():
            momenta = [FourMomentum(row) for row in rows]
            res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
            self.assertEqual(res["method"], method)
            self._assert_naive_fallback(res, momenta)

    def test_near_pole_massive_mapping_is_kept(self):
        # A soft wide-angle recoil system mapped by the boost to within 1.3e-5 M
        # of the W pole: inside the 1e-4 M criterion, so it remains a success.
        momenta = [
            FourMomentum(row)
            for row in [
                [629.9305504702015, 0.0, 0.0, 629.9305504702015],
                [629.9305504702015, 0.0, 0.0, -629.9305504702015],
                [
                    377.95479907555085,
                    310.24659189082115,
                    131.17015518806517,
                    171.43883093485323,
                ],
                [
                    881.8945311762852,
                    -310.2436303268797,
                    -131.16958340351616,
                    -171.4368399211711,
                ],
                [
                    0.005885344283432136,
                    0.001548872728250166,
                    0.00029903845127215664,
                    -0.005669995291961975,
                ],
                [
                    0.005885344283432136,
                    -0.00451043666968957,
                    -0.0008708230002789891,
                    0.0036789816098252086,
                ],
            ]
        ]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self.assertEqual(res["method"], "fks_fsr_massive_onshell")
        self.assertEqual(res["spectator_transform"][0][0], "boost_along_direction")
        self.assertTrue(_within_pole(res["mother"], fx.MW_POLE))
        m = math.sqrt(max(0.0, _m2(res["mother"])))
        self.assertGreater(abs(m - fx.MW_POLE), 5e-6 * fx.MW_POLE)

    def test_massless_mother_off_pole_is_inexact(self):
        # Two 16 keV photons at 90 degrees at sqrt(s) = 5 TeV with a gluon
        # mother: the boost leaves the mother with m^2 = 0.7 q^2, far beyond the
        # 1e-6 q^2 allowed for a massless mother.
        momenta = [
            FourMomentum(row)
            for row in [
                [2498.6410511744325, 0.0, 0.0, 2498.6410511744325],
                [2498.6410511744325, 0.0, 0.0, -2498.6410511744325],
                [
                    1499.1846208719342,
                    1230.6151962570434,
                    520.29575985361,
                    680.0243293283839,
                ],
                [
                    3498.09744870118,
                    -1230.6152075448172,
                    -520.2957711088405,
                    -680.0243130837324,
                ],
                [
                    1.638787557311009e-05,
                    1.1604288060042795e-05,
                    1.1570832316503377e-05,
                    1.3712739568612155e-07,
                ],
                [
                    1.638787557311009e-05,
                    -3.165143032392387e-07,
                    -3.156017766541575e-07,
                    -1.6381778902614822e-05,
                ],
            ]
        ]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 21}], momenta)
        self.assertEqual(res["method"], "fks_fsr_inexact")
        self._assert_naive_fallback(res, momenta)

    def test_near_massless_mother_is_kept(self):
        # A soft wide-angle recoil system with a gluon mother: the boost leaves
        # m^2 = 6.4e-08 q^2, inside the 1e-6 q^2 allowed for a
        # massless mother, so it remains a success.
        momenta = [
            FourMomentum(row)
            for row in [
                [406.01081044718563, 0.0, 0.0, 406.01081044718563],
                [406.01081044718563, 0.0, 0.0, -406.01081044718563],
                [
                    243.60297044172427,
                    199.96304197983744,
                    84.54301814732963,
                    110.49736256011568,
                ],
                [
                    568.40693103069,
                    -199.97011817858854,
                    -84.54313011806855,
                    -110.50285489100789,
                ],
                [
                    0.005859710978462199,
                    0.0012220761141021812,
                    1.933760912009812e-05,
                    0.005730826186452077,
                ],
                [
                    0.005859710978462199,
                    0.00585412263699704,
                    9.26331297936664e-05,
                    -0.0002384952942384235,
                ],
            ]
        ]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 21}], momenta)
        self.assertEqual(res["method"], "fks_fsr_massive_onshell")
        q2 = _m2(momenta[0] + momenta[1])
        self.assertLessEqual(abs(_m2(res["mother"])), 1e-6 * q2)
        self.assertGreater(abs(_m2(res["mother"])), 1e-8 * q2)

    def test_near_collinear_soft_pair_is_boosted(self):
        # Two 25 MeV photons 0.02 rad apart (m^2_rec = 2.5e-7 GeV^2,
        # m^2_rec / E^2_rec = 1e-4): the recoil system is massive, and the boost
        # keeps its invariant mass.
        momenta = [
            FourMomentum(row)
            for row in [
                [150.0, 0.0, 0.0, 150.0],
                [150.0, 0.0, 0.0, -150.0],
                [89.985, 73.86475747782474, 31.229518565363225, 40.81684698648058],
                [
                    209.96499999999997,
                    -73.90191386291258,
                    -31.249817191050308,
                    -40.84343869290857,
                ],
                [
                    0.025,
                    0.01846150656510322,
                    0.010085567002783374,
                    0.013507557646703494,
                ],
                [0.025, 0.0186948785227415, 0.010213058684300886, 0.013084148781291239],
            ]
        ]
        res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
        self.assertEqual(res["method"], "fks_fsr_massive_onshell")
        self.assertEqual(res["spectator_transform"][0][0], "boost_along_direction")
        k = momenta[4] + momenta[5]
        k_bar = res["spectators"][4] + res["spectators"][5]
        self.assertAlmostEqual(_m2(k_bar) / _m2(k), 1.0, delta=1e-3)

    def test_every_success_reaches_the_pole(self):
        # Soft two-photon recoil systems and ordinary spectators, with unequal
        # beam energies so that q is not at rest.
        random.seed(11)
        n_success = 0
        for n in range(1200):
            e1, e2 = random.uniform(100.0, 1500.0), random.uniform(100.0, 1500.0)
            beams = [FourMomentum([e1, 0, 0, e1]), FourMomentum([e2, 0, 0, -e2])]
            q = beams[0] + beams[1]
            rs = math.sqrt(_m2(q))
            if n % 2:
                E = 10 ** random.uniform(-3.5, 0.0)
                th, ph = random.uniform(0.2, 2.9), random.uniform(0.0, 6.28)
                opening = random.uniform(0.05, 3.0)
                specs = [
                    _massless_momentum(E / 2, th, ph),
                    _massless_momentum(E / 2, th + opening, ph),
                ]
            else:
                specs = [
                    _massless_momentum(
                        random.uniform(5.0, rs / 4),
                        random.uniform(0.1, 3.0),
                        random.uniform(0.0, 6.28),
                    )
                    for _s in range(random.choice([1, 2]))
                ]
            p_a = _massless_momentum(random.uniform(0.1, 0.4) * rs, 1.1, 0.4)
            p_b = q - p_a
            for spec in specs:
                p_b = p_b - spec
            if p_b.E <= 0 or _m2(p_b) <= 0:
                continue
            momenta = beams + [p_a, p_b] + specs
            res = fx._fks_fsr_mapping(2, 3, [{"number": 3, "id": 24}], momenta)
            if res["method"] not in fx.FKS_SUCCESS_METHODS:
                self._assert_naive_fallback(res, momenta)
                continue
            n_success += 1
            self.assertTrue(_within_pole(res["mother"], fx.MW_POLE))
            if res["spectator_transform"][0][0] == "rescale":
                self.assertLessEqual(
                    abs(_m2(res["mother"]) - fx.MW_POLE**2),
                    _onshell_tolerance(momenta, fx.MW_POLE),
                )
            total = FourMomentum(res["mother"])
            for idx, p in res["spectators"].items():
                total += p
                leaf = fx._apply_fxfx_frame_transform(
                    momenta[idx], res["spectator_transform"]
                )
                for comp in ("E", "px", "py", "pz"):
                    self.assertEqual(getattr(leaf, comp), getattr(p, comp))
            for comp in ("E", "px", "py", "pz"):
                self.assertAlmostEqual(
                    getattr(total, comp), getattr(q, comp), delta=1e-9 * q.E
                )
        self.assertGreater(n_success, 300)

    def test_rescale_frame_operation(self):
        p = FourMomentum([5.0, 3.0, 0.0, 4.0])
        out = fx._apply_fxfx_frame_transform(p, [("rescale", 2.5)])
        for comp in ("E", "px", "py", "pz"):
            self.assertEqual(getattr(out, comp), 2.5 * getattr(p, comp))


class TestMergeLabelConsistency(unittest.TestCase):
    """fxfx_merge_particles_kinematics: mass label matches sqrt(p^2), the
    method is returned and counted, check_kinematics_only passes for successes
    and fallbacks."""

    def _merge(self, entries, i, j, mother_pdg):
        event = _mk_event(entries)
        before = dict(fx.FKS_METHOD_COUNTS)
        cons_before = fx.FKS_QUALITY_COUNTS["merge_momentum_conservation_violation"]
        # In production _cluster_fxfx_event opens the counting gate for the
        # first clustering pass of each main-loop event.
        prev_gate = fx._FKS_COUNT_ACTIVE
        fx._FKS_COUNT_ACTIVE = True
        try:
            method = fx.fxfx_merge_particles_kinematics(
                event, i, j, [{"number": i + 1, "id": mother_pdg}]
            )
        finally:
            fx._FKS_COUNT_ACTIVE = prev_gate
        self.assertEqual(
            fx.FKS_METHOD_COUNTS[method], before.get(method, 0) + 1,
            "method counter did not increment",
        )
        self.assertEqual(
            fx.FKS_QUALITY_COUNTS["merge_momentum_conservation_violation"],
            cons_before,
            "momentum conservation violated by merge",
        )
        return event, method

    def test_success_label_is_pole_mass(self):
        entries = [
            (21, -1, 250, 0, 0, 250),
            (21, -1, 250, 0, 0, -250),
            (13, 1, 150, 150, 0, 0),
            (-14, 1, 250, -250, 0, 0),
            (21, 1, 100, 100, 0, 0),
        ]
        event, method = self._merge(entries, 2, 3, -24)
        self.assertEqual(method, "fks_fsr_massive_onshell")
        mother = event[2]
        self.assertAlmostEqual(mother.mass, fx.MW_POLE, places=9)
        self.assertAlmostEqual(
            math.sqrt(max(0.0, _m2(FourMomentum(mother)))), fx.MW_POLE, places=6
        )
        event.check_kinematics_only()

    def test_fallback_label_is_invariant_mass(self):
        # no_solution configuration (see TestFSRFallbackBranches.test_no_solution)
        E_r = 30.4
        p_r = math.sqrt(E_r**2 - 900.0)
        (k1E, _x, _y, k1z), (k2E, _x2, _y2, k2z) = _split_massless_z(E_r, p_r)
        pair = FourMomentum([100 - E_r, 0, 0, -p_r])
        (paE, _px, _py, paz), (pbE, _px2, _py2, pbz) = _split_massless_z(pair.E, pair.pz)
        entries = [
            (21, -1, 50, 0, 0, 50),
            (21, -1, 50, 0, 0, -50),
            (13, 1, paE, 0, 0, paz),
            (-14, 1, pbE, 0, 0, pbz),
            (21, 1, k1E, 0, 0, k1z),
            (21, 1, k2E, 0, 0, k2z),
        ]
        event, method = self._merge(entries, 2, 3, -24)
        self.assertEqual(method, "fks_fsr_no_solution")
        mother = event[2]
        m_kin = math.sqrt(max(0.0, _m2(FourMomentum(mother))))
        self.assertAlmostEqual(mother.mass, m_kin, places=9)
        self.assertNotAlmostEqual(mother.mass, fx.MW_POLE, places=1)
        # passes the 0.1% label/kinematics mass gate
        event.check_kinematics_only()

    def test_zero_krec_label_is_invariant_mass(self):
        entries = [
            (21, -1, 100, 0, 0, 100),
            (21, -1, 100, 0, 0, -100),
            (13, 1, 50, 50, 0, 0),
            (-14, 1, 50, -50, 0, 0),
            (21, 1, 50, 0, 0, 50),
            (21, 1, 50, 0, 0, -50),
        ]
        event, method = self._merge(entries, 2, 3, 23)
        self.assertEqual(method, "fks_fsr_zero_krec_momentum")
        mother = event[2]
        m_kin = math.sqrt(max(0.0, _m2(FourMomentum(mother))))
        self.assertAlmostEqual(mother.mass, m_kin, places=9)
        event.check_kinematics_only()

    def test_isr_returns_method(self):
        entries = [
            (21, -1, 500, 0, 0, 500),
            (1, -1, 500, 0, 0, -500),
            (23, 1, 500, 100, 0, 20),
            (21, 1, 500, -100, 0, -20),
        ]
        event = _mk_event(entries)
        method = fx.fxfx_merge_particles_kinematics(
            event, 0, 3, [{"number": 1, "id": 1}]
        )
        self.assertEqual(method, "fks_isr_cm_boost")
        self.assertEqual(len(event), 3)


class TestFSRFrame(unittest.TestCase):
    """The FSR clustering is defined in the partonic CM frame: clustering a
    z-boosted event and boosting the result back reproduces the CM result."""

    @staticmethod
    def _cm_event(spectators):
        beams = [FourMomentum([400, 0, 0, 400]), FourMomentum([400, 0, 0, -400])]
        p_a = _massless_momentum(220.0, 1.1, 0.4)
        p_b = beams[0] + beams[1] - p_a
        for spec in spectators:
            p_b = p_b - spec
        return beams + [p_a, p_b] + list(spectators)

    def _check_boost_covariance(self, spectators, spectator_pids, masses):
        z_hat = (0, 0, 1)
        cm = self._cm_event(spectators)
        lab = [fx._boost_along_direction(p, -0.6, z_hat) for p in cm]
        pids = [21, 21, 13, -14] + list(spectator_pids)
        labels = [0.0, 0.0, 0.0, 0.0] + list(masses)
        events = []
        for momenta in (cm, lab):
            event = _lhe_event(
                [
                    (pid, -1 if k < 2 else 1, mass, p.E, p.px, p.py, p.pz)
                    for k, (pid, mass, p) in enumerate(zip(pids, labels, momenta))
                ]
            )
            method = fx.fxfx_merge_particles_kinematics(
                event, 2, 3, [{"number": 3, "id": -24}]
            )
            self.assertEqual(method, "fks_fsr_massive_onshell")
            events.append(event)
        for p_cm, p_lab in zip(events[0], events[1]):
            back = fx._boost_along_direction(FourMomentum(p_lab), 0.6, z_hat)
            for comp in ("E", "px", "py", "pz"):
                self.assertAlmostEqual(
                    getattr(back, comp), getattr(p_cm, comp), delta=1e-9 * 800.0
                )

    def test_massive_spectator(self):
        # A single W spectator, as in the q + W -> q' clusterings of WW events.
        p = 60.0
        w = FourMomentum(
            [
                math.sqrt(p**2 + fx.MW_POLE**2),
                p * math.sin(2.0) * math.cos(1.3),
                p * math.sin(2.0) * math.sin(1.3),
                p * math.cos(2.0),
            ]
        )
        self._check_boost_covariance([w], [24], [fx.MW_POLE])

    def test_massive_spectator_system(self):
        self._check_boost_covariance(
            [_massless_momentum(60.0, 2.0, 1.3), _massless_momentum(45.0, 0.7, 2.9)],
            [21, 21],
            [0.0, 0.0],
        )


class TestISRMapping(unittest.TestCase):
    def test_conservation_and_beams(self):
        beams = [FourMomentum([400, 0, 0, 400]), FourMomentum([300, 0, 0, -300])]
        rad = FourMomentum([50, 30, -20, 10])
        q = beams[0] + beams[1]
        rest = FourMomentum([q.E - rad.E, q.px - rad.px, q.py - rad.py, q.pz - rad.pz])
        p_a = FourMomentum([rest.E / 2, rest.px / 2 + 5, rest.py / 2, rest.pz / 2 - 5])
        p_b = FourMomentum(
            [rest.E - p_a.E, rest.px - p_a.px, rest.py - p_a.py, rest.pz - p_a.pz]
        )
        momenta = beams + [p_a, p_b, rad]
        res = fx._fks_isr_mapping(0, 4, [{"number": 1, "id": 1}], momenta)
        b0, b1 = res["beams"]
        self.assertAlmostEqual(_m2(b0), 0.0, places=5)
        self.assertAlmostEqual(_m2(b1), 0.0, places=5)
        pin = b0 + b1
        self.assertAlmostEqual(_m2(pin) / _m2(rest), 1.0, places=9)
        pfs = FourMomentum()
        for p in res["final_state"].values():
            pfs += p
        for comp in ("E", "px", "py", "pz"):
            self.assertAlmostEqual(getattr(pin, comp), getattr(pfs, comp), places=5)

    def test_lightlike_guard(self):
        beams = [FourMomentum([100, 0, 0, 100]), FourMomentum([1, 0, 0, -1])]
        rad = FourMomentum([1, 0, 0, -1])
        p_a = FourMomentum([100, 0, 0, 100])
        momenta = beams + [p_a, rad]
        with self.assertRaises(ValueError):
            fx._fks_isr_mapping(0, 3, [{"number": 1, "id": 1}], momenta)


class TestDecayTransform(unittest.TestCase):
    def test_two_body_conservation_and_masses(self):
        # off-shell W (70 GeV) -> two massless; retarget to on-shell moving W
        minv = 70.0
        pstar = minv / 2
        d1 = FourMomentum([pstar, pstar, 0, 0])
        d2 = FourMomentum([pstar, -pstar, 0, 0])
        boost_dir = (0.6, 0.0, 0.8)
        d1lab = fx._boost_along_direction(d1, -0.7, boost_dir)
        d2lab = fx._boost_along_direction(d2, -0.7, boost_dir)
        orig = d1lab + d2lab
        wrest = FourMomentum([fx.MW_POLE, 0, 0, 0])
        target = fx._boost_along_direction(wrest, -0.5, (0.0, 1.0, 0.0))
        out = fx._transform_decay_products_to_onshell([d1lab, d2lab], orig, target)
        tot = out[0] + out[1]
        for comp in ("E", "px", "py", "pz"):
            self.assertAlmostEqual(getattr(tot, comp), getattr(target, comp), places=6)
        self.assertAlmostEqual(_m2(out[0]), 0.0, places=5)
        self.assertAlmostEqual(_m2(out[1]), 0.0, places=5)

    def test_forbidden_returns_original(self):
        # target mass below the sum of (massive) decay-product masses
        d1 = FourMomentum([60, 0, 0, 10])  # m ~ 59.2
        d2 = FourMomentum([60, 0, 0, -10])
        orig = d1 + d2
        target = FourMomentum([fx.MW_POLE, 0, 0, 0])
        out = fx._transform_decay_products_to_onshell([d1, d2], orig, target)
        self.assertAlmostEqual(out[0].E, d1.E, places=9)
        self.assertAlmostEqual(out[1].pz, d2.pz, places=9)

    def test_decay_follows_prepared_frame_without_wigner_rotation(self):
        # A longitudinal event-frame boost, non-collinear with this W. With zero
        # mass shift the decay transport reduces exactly to that frame boost.
        mass = fx.MW_POLE
        w_lab = FourMomentum(
            [math.sqrt(mass**2 + 60.0**2 + 150.0**2), 60.0, 0.0, 150.0]
        )
        direction = (0.3, 0.8, math.sqrt(1.0 - 0.3**2 - 0.8**2))
        d1_rest = FourMomentum(
            [
                mass / 2.0,
                mass / 2.0 * direction[0],
                mass / 2.0 * direction[1],
                mass / 2.0 * direction[2],
            ]
        )
        d2_rest = FourMomentum(
            [
                mass / 2.0,
                -mass / 2.0 * direction[0],
                -mass / 2.0 * direction[1],
                -mass / 2.0 * direction[2],
            ]
        )
        w_p = math.sqrt(w_lab.px**2 + w_lab.py**2 + w_lab.pz**2)
        w_dir = (w_lab.px / w_p, w_lab.py / w_p, w_lab.pz / w_p)
        daughters_lab = [
            fx._boost_along_direction(d1_rest, -w_p / w_lab.E, w_dir),
            fx._boost_along_direction(d2_rest, -w_p / w_lab.E, w_dir),
        ]

        # beta_z = 0.55, in the Event.boost convention recorded by the
        # preparation routine.
        frame_transform = [
            ("boost_to_cm", FourMomentum([1000.0, 0.0, 0.0, 550.0]))
        ]
        w_prepared = fx._apply_fxfx_frame_transform(w_lab, frame_transform)
        expected = [
            fx._apply_fxfx_frame_transform(p, frame_transform)
            for p in daughters_lab
        ]
        transformed = fx._transform_decay_products_to_onshell(
            daughters_lab,
            w_lab,
            w_prepared,
            frame_transform=frame_transform,
        )

        for got, want in zip(transformed, expected):
            for component in ("E", "px", "py", "pz"):
                self.assertAlmostEqual(
                    getattr(got, component), getattr(want, component), places=10
                )

        # Without frame_transform the lab daughters are mapped directly onto the
        # prepared mother: the total four-momentum is conserved, but the
        # daughters differ from the frame boost by a Wigner rotation.
        mixed_frame = fx._transform_decay_products_to_onshell(
            daughters_lab, w_lab, w_prepared
        )
        self.assertGreater(
            max(
                abs(getattr(mixed_frame[0], component) - getattr(expected[0], component))
                for component in ("E", "px", "py", "pz")
            ),
            1.0e-3,
        )


class TestDensityKernelLimits(unittest.TestCase):
    def test_fixed_buffer_limits(self):
        self.assertTrue(fx.density_dimensions_supported(9, 2))
        self.assertTrue(fx.density_dimensions_supported(36, 4))
        self.assertFalse(fx.density_dimensions_supported(81, 4))
        self.assertFalse(fx.density_dimensions_supported(36, 9))
        self.assertFalse(fx.density_dimensions_supported(1, 11))


class TestSudakovSources(unittest.TestCase):
    def test_numeric_source_ids_remain_distinct(self):
        class Event:
            wgt = 2.0

            def parse_reweight(self):
                return {"1001": 2.0, "2001": 200.0, "2101": 210.0, "2201": 220.0}

        driver = fx.FxFxEWSudakovMixin()
        event = Event()
        factors = [1.1, 0.9, 1.05]
        result = driver._build_sudakov_rwgt_dict(event, factors)
        self.assertEqual(len(result), 13)
        self.assertFalse(set(event.parse_reweight()) & set(result))
        for source, value in event.parse_reweight().items():
            for variant, factor in enumerate(factors):
                key = driver._sudakov_weight_map[0, variant, source]
                self.assertAlmostEqual(result[key], value * factor)


class TestForcedClusteringPlumbing(unittest.TestCase):
    """_apply_forced_clustering: fks_methods threading, conditional mass rule,
    and the skip-path sentinel."""

    def _dummy_self(self):
        return type("DummyMixin", (), {})()

    def _madspin_event(self):
        # LHE 1-based: 1,2 beams; 3: mu+; 4: vm; 5: W+ mother (status 2);
        # 6: recoil parton. FS total = q = (500,0,0,0).
        event = _mk_event(
            [
                (21, -1, 250, 0, 0, 250),
                (21, -1, 250, 0, 0, -250),
                (-13, 1, 150, 150, 0, 0),
                (14, 1, 250, -250, 0, 0),
                (24, 2, 400, -100, 0, 0),
                (21, 1, 100, 100, 0, 0),
            ]
        )
        return event

    def test_success_threading_and_mass(self):
        event = self._madspin_event()
        step = fx.ForcedClusterStep(
            children_lhe_idx=[3, 4], mother_lhe_idx=5, mother_pdg=24,
            scale=100.0, depth=0,
        )
        out, groups = fx.FxFxEWSudakovMixin._apply_forced_clustering(
            self._dummy_self(), event, [step]
        )
        self.assertEqual(out.fks_methods, ["fks_fsr_massive_onshell"])
        mothers = [p for p in out if abs(getattr(p, "pid", 0)) == 24]
        self.assertEqual(len(mothers), 1)
        self.assertEqual(mothers[0].status, 1)
        self.assertAlmostEqual(mothers[0].mass, fx.MW_POLE, places=9)
        m_kin = math.sqrt(max(0.0, _m2(FourMomentum(mothers[0]))))
        self.assertAlmostEqual(m_kin, fx.MW_POLE, places=6)

    def test_skipped_step_records_sentinel(self):
        event = self._madspin_event()
        # 1->3 step: not a MadSpin 1->2 topology, so it is skipped and recorded
        step = fx.ForcedClusterStep(
            children_lhe_idx=[3, 4, 6], mother_lhe_idx=5, mother_pdg=24,
            scale=100.0, depth=0,
        )
        out, groups = fx.FxFxEWSudakovMixin._apply_forced_clustering(
            self._dummy_self(), event, [step]
        )
        self.assertEqual(out.fks_methods, ["forced_step_skipped"])
        self.assertNotIn("forced_step_skipped", fx.FKS_SUCCESS_METHODS)
        # mother left un-restored (still status 2)
        mothers = [p for p in out if abs(getattr(p, "pid", 0)) == 24]
        self.assertEqual(mothers[0].status, 2)

    def test_missing_children_records_sentinel(self):
        event = self._madspin_event()
        step = fx.ForcedClusterStep(
            children_lhe_idx=[3, 99], mother_lhe_idx=5, mother_pdg=24,
            scale=100.0, depth=0,
        )
        out, groups = fx.FxFxEWSudakovMixin._apply_forced_clustering(
            self._dummy_self(), event, [step]
        )
        self.assertEqual(out.fks_methods, ["forced_step_skipped"])


class TestAccountingGateAndReset(unittest.TestCase):
    """Counting gate semantics and per-launch reset."""

    def test_gate_off_means_no_counting(self):
        entries = [
            (21, -1, 250, 0, 0, 250),
            (21, -1, 250, 0, 0, -250),
            (13, 1, 150, 150, 0, 0),
            (-14, 1, 250, -250, 0, 0),
            (21, 1, 100, 100, 0, 0),
        ]
        event = _mk_event(entries)
        prev_gate = fx._FKS_COUNT_ACTIVE
        fx._FKS_COUNT_ACTIVE = False
        before = dict(fx.FKS_METHOD_COUNTS)
        try:
            method = fx.fxfx_merge_particles_kinematics(
                event, 2, 3, [{"number": 3, "id": -24}]
            )
        finally:
            fx._FKS_COUNT_ACTIVE = prev_gate
        self.assertEqual(method, "fks_fsr_massive_onshell")
        self.assertEqual(
            fx.FKS_METHOD_COUNTS.get(method, 0), before.get(method, 0),
            "counter incremented although the gate was off",
        )

    def test_reset_clears_counters_and_gate_state(self):
        fx.FKS_METHOD_COUNTS["fks_fsr_massive_onshell"] += 3
        fx.FKS_QUALITY_COUNTS["events_clustering_failed"] += 1
        fx._FKS_COUNT_ACTIVE = True
        fx._FKS_COUNTED_UPTO = 42
        fx.fks_reset_accounting()
        self.assertEqual(sum(fx.FKS_METHOD_COUNTS.values()), 0)
        self.assertEqual(sum(fx.FKS_QUALITY_COUNTS.values()), 0)
        self.assertFalse(fx._FKS_COUNT_ACTIVE)
        self.assertEqual(fx._FKS_COUNTED_UPTO, 0)

    def test_summary_reports_fallbacks(self):
        fx.fks_reset_accounting()
        fx.FKS_METHOD_COUNTS["fks_fsr_massive_onshell"] += 5
        fx.FKS_METHOD_COUNTS["fks_fsr_unreachable_target"] += 2
        text = fx.fks_accounting_summary()
        self.assertIn("7 clusterings", text)
        self.assertIn("2 fallback(s)", text)
        self.assertIn("fks_fsr_unreachable_target", text)
        self.assertIn("<-- fallback", text)
        fx.fks_reset_accounting()


class TestSpuriousRootSafety(unittest.TestCase):
    def test_A_geq_absB(self):
        """A >= |B|: A±B = (q_E±q_n)(E_rec∓p_n) >= 0 for timelike q, physical k_rec.

        Hence for R > 0 every |beta| < 1 root of the squared quadratic solves
        the unsquared constraint gamma(A + beta B) = R."""
        random.seed(11)
        for _ in range(2000):
            qE = random.uniform(50, 3000)
            qvec = [random.uniform(-0.9, 0.9) * qE / math.sqrt(3.0) for _c in range(3)]
            if qE**2 - sum(x * x for x in qvec) <= 0:
                continue
            kE = random.uniform(1, 500)
            p_n = random.uniform(0, 1) * kE
            n = [random.gauss(0, 1) for _c in range(3)]
            norm = math.sqrt(sum(x * x for x in n))
            q_n = sum(a * b / norm for a, b in zip(qvec, n))
            A = qE * kE - q_n * p_n
            B = q_n * kE - qE * p_n
            self.assertGreaterEqual(A, abs(B) - 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
