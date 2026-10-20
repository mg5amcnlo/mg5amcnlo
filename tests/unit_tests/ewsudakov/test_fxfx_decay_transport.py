"""Decay transport through recoil maps and resonance mass projections."""

import copy
import math
import unittest
from types import SimpleNamespace

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from madgraph.various import fxfx_ewsudakov as fx, lhe_parser


def vector(p):
    return np.array([p.E, p.px, p.py, p.pz])


def mass2(p):
    return p[0]**2 - p[1:] @ p[1:]


def boost(beta):
    """Construct a passive Lorentz matrix with metric (+---)."""
    beta = np.asarray(beta, dtype=float)
    square = beta @ beta
    matrix = np.eye(4)
    if square:
        gamma = 1 / math.sqrt(1 - square)
        matrix[0, 0] = gamma
        matrix[0, 1:] = matrix[1:, 0] = -gamma * beta
        matrix[1:, 1:] += (gamma - 1) * np.outer(beta, beta) / square
    return matrix


def two_body(parent, masses, direction):
    """Construct two daughters from invariant masses and a rest-frame axis."""
    square = mass2(parent)
    mass = math.sqrt(square)
    m1, m2 = masses
    magnitude = math.sqrt(max(0, (square - (m1 + m2)**2) *
                              (square - (m1 - m2)**2))) / (2 * mass)
    direction = np.asarray(direction) / np.linalg.norm(direction)
    to_lab = boost(-parent[1:] / parent[0])
    return [to_lab @ np.r_[math.sqrt(m**2 + magnitude**2), sign * magnitude * direction]
            for m, sign in zip(masses, (1, -1))]


def make_event(rows, mothers=(), tag=""):
    event = lhe_parser.Event()
    event.aqcd = 0.118
    for pid, status, momentum in rows:
        particle = lhe_parser.Particle(event=event)
        particle.pid, particle.status = pid, status
        particle.E, particle.px, particle.py, particle.pz = map(float, momentum)
        particle.mass = math.sqrt(max(0, mass2(momentum))) if abs(pid) in (6, 23, 24) else 0.0
        particle.mother1 = particle.mother2 = 0
        particle.color1 = particle.color2 = 0
        particle.vtim, particle.helicity = 0, 9
        event.append(particle)
    for child, parent in mothers:
        event[child - 1].mother1 = event[child - 1].mother2 = event[parent - 1]
    event.tag = tag
    return event


def ww_event(jets=(), masses=None, beta_z=0.0):
    masses = masses or (fx.MW_POLE, fx.MW_POLE)
    beams = [np.array([500., 0, 0, 500.]), np.array([500., 0, 0, -500.])]
    recoil = sum(beams) - sum(jets, np.zeros(4))
    parents = two_body(recoil, masses, [0.6, 0.5, math.sqrt(0.39)])
    daughters = [p for parent, axis in zip(parents, ([0.3, 0.8, math.sqrt(0.27)],
                                                    [-0.4, 0.2, math.sqrt(0.8)]))
                 for p in two_body(parent, (0, 0), axis)]
    lab = boost([0, 0, -beta_z])
    pids = [2, -2, 24, -24] + [21] * len(jets) + [-13, 14, 11, -12]
    statuses = [-1, -1, 2, 2] + [1] * (len(jets) + 4)
    rows = [(pid, status, lab @ p) for pid, status, p in
            zip(pids, statuses, beams + parents + list(jets) + daughters)]
    first = 5 + len(jets)
    mothers = [(first, 3), (first + 1, 3), (first + 2, 4), (first + 3, 4)]
    steps = ['step_%d="(%d,%d);mom=%d"' % (i + 1, i % 2 + 1, i + 5, 2 * (-1)**i)
             for i in range(len(jets))]
    return make_event(rows, mothers, '<clustering type="test" %s></clustering>' % " ".join(steps))


class ReferenceTransport:
    """Recoil calculation using Lorentz matrices and a rapidity solution."""

    def __init__(self, event):
        self.beams = [vector(p) for p in event[:2]]
        self.particles = {i: vector(p) for i, p in enumerate(event, 1) if p.status == 1}
        self.leaves = {i: {i: p.copy()} for i, p in self.particles.items()}

    def transform(self, matrix):
        self.beams = [matrix @ p for p in self.beams]
        self.particles = {i: matrix @ p for i, p in self.particles.items()}
        self.leaves = {i: {j: matrix @ p for j, p in leaves.items()}
                       for i, leaves in self.leaves.items()}

    def fsr(self, pair, parent, mass):
        total = sum(self.beams)
        velocity = np.array([0., 0., total[3] / total[0]])
        self.transform(boost(velocity))
        total = sum(self.beams)
        recoil = sum((p for i, p in self.particles.items() if i not in pair), np.zeros(4))
        n = recoil[1:] / np.linalg.norm(recoil[1:])
        a = total[0] * recoil[0] - (total[1:] @ n) * np.linalg.norm(recoil[1:])
        b = (total[1:] @ n) * recoil[0] - total[0] * np.linalg.norm(recoil[1:])
        rhs = (mass2(total) + mass2(recoil) - mass**2) / 2
        offset = math.atanh(b / a)
        distance = math.acosh(max(1, rhs / math.sqrt(a * a - b * b)))
        rapidity = min((distance - offset, -distance - offset), key=abs)
        matrix = boost(math.tanh(rapidity) * n)
        mother = total - matrix @ recoil
        source = [self.particles[i] for i in pair]
        source_parent = sum(source)
        axis = (boost(source_parent[1:] / source_parent[0]) @ source[0])[1:]
        targets = two_body(mother, [math.sqrt(max(0, mass2(p))) for p in source], axis)
        products = {}
        for i, p, target in zip(pair, source, targets):
            leaves = self.leaves.pop(i)
            if len(leaves) == 1:
                products[next(iter(leaves))] = target
            else:
                transport = boost(-target[1:] / target[0]) @ boost(p[1:] / p[0])
                products.update({j: transport @ leaf for j, leaf in leaves.items()})
            del self.particles[i]
        self.particles = {i: matrix @ p for i, p in self.particles.items()}
        self.leaves = {i: {j: matrix @ p for j, p in leaves.items()}
                       for i, leaves in self.leaves.items()}
        self.particles[parent], self.leaves[parent] = mother, products
        self.transform(boost(-velocity))

    def isr(self, radiation):
        recoil = sum(self.beams) - self.particles.pop(radiation)
        del self.leaves[radiation]
        matrix = boost([recoil[1] / recoil[0], recoil[2] / recoil[0], 0])
        self.transform(matrix)
        total = matrix @ recoil
        self.beams = [np.array([(total[0] + sign * total[3]) / 2, 0, 0,
                               sign * (total[0] + sign * total[3]) / 2]) for sign in (1, -1)]


@unittest.skipIf(np is None, "NumPy is required for Lorentz matrix checks")
class TestDecayTransport(unittest.TestCase):
    def setUp(self):
        self.driver = fx.FxFxEWSudakovMixin()
        self.driver.model = None
        self.driver.fxfx_clustered_final_states = set()

    def assert_momenta(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-8)

    def check_ww(self, jets, masses=None, beta_z=0.0):
        event = ww_event(jets, masses, beta_z)
        source = str(event)
        expected = ReferenceTransport(event)
        first = 5 + len(jets)
        expected.fsr((first, first + 1), 3, fx.MW_POLE)
        expected.fsr((first + 2, first + 3), 4, fx.MW_POLE)
        for i in range(len(jets)):
            expected.isr(5 + i)
        clustered, tag, groups = self.driver._cluster_fxfx_event(event, return_groups=True)
        scalar, scalar_tag = self.driver._cluster_fxfx_event(event)
        self.assertEqual(tag, scalar_tag)
        self.assertEqual(clustered.fks_methods, scalar.fks_methods)
        np.testing.assert_array_equal([vector(p) for p in clustered], [vector(p) for p in scalar])
        self.assertEqual(len(clustered.fks_methods), 2 + len(jets))
        for index, label in ((2, 3), (3, 4)):
            self.assert_momenta(vector(clustered[index]), expected.particles[label])
        total = sum(expected.beams)
        expected.transform(boost(total[1:] / total[0]))
        catalogue = SimpleNamespace(original_pdg_list_dict={tag: clustered.get_tag_and_order()[1]})
        prepared = self.driver._prepare_fxfx_sudakov_inputs(event, catalogue, clustered)
        for index, label in ((2, 3), (3, 4)):
            labels = sorted(expected.leaves[label])
            self.assertTrue(set(labels).issubset(groups[index]))
            decay = self.driver._build_decay_event_from_cluster(
                event, clustered, index, event[label - 1].pid, labels,
                frame_transform=prepared["frame_transform"],
            )
            actual = np.array([vector(p) for p in decay[1:]])
            self.assert_momenta(actual, [expected.leaves[label][i] for i in labels])
            self.assert_momenta(actual.sum(axis=0), vector(clustered[index]))
            self.assert_momenta(vector(clustered[index]), expected.particles[label])
            for p in actual:
                self.assertAlmostEqual(mass2(p), 0, delta=1e-7)
        self.assertEqual(str(event), source)
        self.assertFalse(hasattr(event, "_fxfx_decay_products"))

    def test_transverse_recoil_in_longitudinal_frames(self):
        for px, py, beta_z in ((20, 10, 0), (20, 10, .55), (-10, 20, -.35)):
            with self.subTest(px=px, py=py, beta_z=beta_z):
                self.check_ww([np.array([50., px, py, math.sqrt(2500 - px**2 - py**2)])], beta_z=beta_z)

    def test_collinear_recoil(self):
        self.check_ww([np.array([50., 0, 0, 50.])])

    def test_no_radiation(self):
        self.check_ww([])

    def test_two_transverse_recoils(self):
        self.check_ww([np.array([50., 20, 10, math.sqrt(2000)]),
                       np.array([35., -12, 15, -math.sqrt(856)])])

    def test_mass_projection_then_transverse_recoil(self):
        self.check_ww([np.array([50., 20, 10, math.sqrt(2000)])], masses=(89., 84.), beta_z=.3)

    def test_transport_is_copied_with_event(self):
        event = ww_event([np.array([50., 20, 10, math.sqrt(2000)])])
        clustered, _, _ = self.driver._cluster_fxfx_event(event, return_groups=True)
        clone = copy.deepcopy(clustered)
        label = next(iter(clone._fxfx_decay_products[2]))
        clone._fxfx_decay_products[2][label].px += 1
        self.assertNotEqual(clone._fxfx_decay_products[2][label].px,
                            clustered._fxfx_decay_products[2][label].px)

    def check_sequence(self, event, expected, identities, methods):
        source = str(event)
        clustered, tag, groups = self.driver._cluster_fxfx_event(event, return_groups=True)
        self.assertEqual(clustered.fks_methods, methods)
        scalar, _ = self.driver._cluster_fxfx_event(event)
        np.testing.assert_array_equal([vector(p) for p in clustered], [vector(p) for p in scalar])
        total = sum(expected.beams)
        expected.transform(boost(total[1:] / total[0]))
        catalogue = SimpleNamespace(original_pdg_list_dict={tag: clustered.get_tag_and_order()[1]})
        prep = self.driver._prepare_fxfx_sudakov_inputs(event, catalogue, clustered)
        decays = []
        for index, label in identities:
            self.assert_momenta(vector(clustered[index]), expected.particles[label])
            labels = sorted(expected.leaves[label])
            self.assertTrue(set(labels).issubset(groups[index]))
            decay = self.driver._build_decay_event_from_cluster(
                event, clustered, index, clustered[index].pid, labels,
                frame_transform=prep["frame_transform"],
            )
            self.assert_momenta([vector(p) for p in decay[1:]],
                                [expected.leaves[label][i] for i in labels])
            self.assert_momenta(sum(vector(p) for p in decay[1:]), vector(clustered[index]))
            decays.append(decay)
        self.assertEqual(str(event), source)
        return decays

    def test_isr_fsr_isr_sequence(self):
        jets = [np.array([50., 20, 10, math.sqrt(2000)]),
                np.array([35., -12, 15, -math.sqrt(856)])]
        source = ww_event(jets, masses=(89., 84.))
        rows = [(p.pid, p.status, vector(p)) for i, p in enumerate(source) if i != 3]
        event = make_event(rows, [(6, 3), (7, 3)],
                           '<clustering step_1="(1,4);mom=2" step_2="(6,7);mom=-24" '
                           'step_3="(2,5);mom=-2"></clustering>')
        expected = ReferenceTransport(event)
        expected.fsr((6, 7), 3, fx.MW_POLE)
        expected.isr(4)
        expected.fsr((8, 9), 8, fx.MW_POLE)
        expected.isr(5)
        self.check_sequence(event, expected, [(2, 3), (3, 8)],
                            ["fks_fsr_massive_onshell", "fks_isr_cm_boost",
                             "fks_fsr_massive_onshell", "fks_isr_cm_boost"])

    def test_nested_decay_mass_projections(self):
        beams = [np.array([500., 0, 0, 500.]), np.array([500., 0, 0, -500.])]
        jet = np.array([50., 20, 10, math.sqrt(2000)])
        top, antitop = two_body(sum(beams) - jet, (190., fx.MT_POLE), [.6, .5, .4])
        w, bottom = two_body(top, (89., 0), [.2, .3, .8])
        leptons = two_body(w, (0, 0), [.4, -.3, .7])
        lab = boost([0, 0, -.3])
        momenta = beams + [top, antitop, jet, w, bottom] + leptons
        rows = [(pid, status, lab @ p) for pid, status, p in
                zip([2, -2, 6, -6, 21, 24, 5, -13, 14],
                    [-1, -1, 2, 1, 1, 2, 1, 1, 1], momenta)]
        event = make_event(rows, [(6, 3), (7, 3), (8, 6), (9, 6)],
                           '<clustering step_1="(1,5);mom=2"></clustering>')
        expected = ReferenceTransport(event)
        expected.fsr((8, 9), 6, fx.MW_POLE)
        expected.fsr((6, 7), 3, fx.MT_POLE)
        expected.isr(5)
        decays = self.check_sequence(event, expected, [(2, 3)],
                                     ["fks_fsr_massive_onshell", "fks_fsr_massive_onshell",
                                      "fks_isr_cm_boost"])
        self.assertAlmostEqual(math.sqrt(mass2(vector(decays[0][2]) + vector(decays[0][3]))),
                               fx.MW_POLE, places=7)

    def test_resonances_from_external_fermions(self):
        source = ww_event([np.array([50., 20, 10, math.sqrt(2000)])], masses=(89., 84.))
        rows = [(p.pid, p.status, vector(p)) for i, p in enumerate(source) if i not in (2, 3)]
        event = make_event(rows, tag='<clustering step_1="(1,3);mom=2" '
                           'step_2="(4,5);mom=24" step_3="(6,7);mom=-24"></clustering>')
        expected = ReferenceTransport(event)
        expected.isr(3)
        expected.fsr((4, 5), 4, fx.MW_POLE)
        expected.fsr((6, 7), 6, fx.MW_POLE)
        self.check_sequence(event, expected, [(2, 4), (3, 6)],
                            ["fks_isr_cm_boost", "fks_fsr_massive_onshell", "fks_fsr_massive_onshell"])

    def test_massless_jet_constituents(self):
        event = ww_event([np.array([50., 20, 10, math.sqrt(2000)]),
                          np.array([35., -12, 15, -math.sqrt(856)])])
        event.tag = '<clustering step_1="(5,6);mom=21"></clustering>'
        clustered, _, groups = self.driver._cluster_fxfx_event(event, return_groups=True)
        self.assertEqual(len(clustered), 5)
        for index in range(2, len(clustered)):
            leaves = clustered._fxfx_decay_products[index]
            self.assertTrue(set(leaves).issubset(groups[index]))
            self.assert_momenta(sum(vector(p) for p in leaves.values()), vector(clustered[index]))
        for p in clustered._fxfx_decay_products[4].values():
            self.assertAlmostEqual(mass2(vector(p)), 0, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
