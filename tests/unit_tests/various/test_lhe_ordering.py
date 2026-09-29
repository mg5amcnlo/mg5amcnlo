################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""The order an LHE writes its particles in must not reach the physics.

Everything that evaluates a matrix element on an event -- MadSpin, the
reweighting -- has to put the event's particles on the legs of a matrix element
generated with its *own* leg order, and the two orders need not agree: aMC@NLO
writes p p > t t~ t t~ as ``t t t~ t~`` while the matrix element MadSpin
generates for it has ``t t~ t t~``. Every accessor that answers "which slot does
this particle go to" must therefore give the same, flavour-correct answer, and
that answer must not depend on how the event happened to order its lines.

The bugs this file exists for were all silent or near-silent:

* MadSpin's density modes read their leg positions off the LHE record while the
  momenta went through ``get_momenta``: every top's density block contracted
  with an anti-top's decay, the spin correlations gone and nothing else moved;
* ``get_momenta``, ``get_helicity`` and ``get_all_momenta`` each carried their
  own copy of the slot assignment, and the copies had drifted: on a
  charge-reversed order ``get_momenta`` returned a Fortran-formatted *string*
  (its retry went through ``get_momenta_str``) and ``get_all_momenta`` could
  not map it at all. All three now go through ``get_mapping``.

The tests below are therefore phrased as invariants over *permutations of the
event's lines*, not as fixed expected lists: a new accessor, or a change in the
tie-breaking, that breaks the agreement fails here whichever order it breaks.
"""

from __future__ import absolute_import
import itertools
import unittest

import madgraph.various.lhe_parser as lhe_parser


def _event(initial, final, resonance_after=None, mothers=None):
    """An lhe_parser.Event with ``initial`` then ``final`` pdgs, in that order.

    Every particle carries a momentum and a helicity that identify it uniquely,
    so where it lands can be read back. ``resonance_after`` inserts a status-2
    line after that many final-state particles (the s-channel resonance
    MadEvent writes for ``p p > z z j j``). ``mothers[k]`` overrides the
    (mother1, mother2) of final-state particle k.
    """
    lines = []
    nb = len(initial) + len(final) + (resonance_after is not None)
    lines.append(' %d 1 1.0 100.0 0.0078 0.118' % nb)
    for k, pid in enumerate(initial):
        pz = 500. + k if k == 0 else -500. - k
        lines.append(' %d -1 0 0 0 0 0.0 0.0 %r %r 0.0 0. %d'
                     % (pid, pz, abs(pz), -1 if k else 1))
    default_mothers = (1, 2) if len(initial) == 2 else (1, 1)
    for k, pid in enumerate(final):
        if resonance_after == k:
            lines.append(' 25 2 1 2 0 0 0.0 0.0 0.0 600.0 125.0 0. 9')
        m1, m2 = mothers[k] if mothers else default_mothers
        # helicity alternates, so identical particles are told apart by momentum
        lines.append(' %d 1 %d %d 0 0 %r %r %r %r 1.0 0. %d'
                     % (pid, m1, m2, 11. * (k + 1), 22. * (k + 1),
                        33. * (k + 1), 250. + k, 1 if k % 2 == 0 else -1))
    if resonance_after is not None and resonance_after >= len(final):
        lines.append(' 25 2 1 2 0 0 0.0 0.0 0.0 600.0 125.0 0. 9')
    return lhe_parser.Event('<event>\n' + '\n'.join(lines) + '\n</event>')


def _mom(part):
    return (part.E, part.px, part.py, part.pz)


def _external(event):
    return [p for p in event if abs(p.status) == 1]


class TestOneSlotAssignment(unittest.TestCase):
    """get_mapping, get_momenta, get_helicity and get_all_momenta describe ONE
    assignment of the event's particles to the matrix element's
    legs, and it is flavour-correct."""

    # the density matrix element MadSpin generates for p p > t t~ t t~
    TTTT = [(21, 21), (6, -6, 6, -6)]

    def assertOneAssignment(self, event, order):
        """The invariant, checked accessor by accessor against get_mapping."""
        mapping, inverse = event.get_mapping(order)
        ext = _external(event)
        flat = list(order[0]) + list(order[1])
        self.assertEqual(sorted(mapping), list(range(len(ext))))
        self.assertEqual(sorted(mapping.values()), list(range(len(flat))))
        for i, slot in mapping.items():
            self.assertEqual(inverse[slot], i)

        # flavour: every particle sits on a leg of its own (or, for a
        # charge-reversed order, the conjugate) flavour
        signs = set()
        for i, part in enumerate(ext):
            leg = flat[mapping[i]]
            pid = part.pid
            self.assertIn(leg, (pid, -pid),
                          'particle %d (pid %d) put on a leg %d'
                          % (i, part.pid, leg))
            signs.add(leg == pid)
        # a charge-reversed order reverses every leg, never only some
        self.assertEqual(len(signs), 1)

        momenta = event.get_momenta(order)
        helicities = event.get_helicity(order)
        all_momenta = event.get_all_momenta(order)
        self.assertIsInstance(momenta, list)
        for i, part in enumerate(ext):
            slot = mapping[i]
            self.assertEqual(momenta[slot], _mom(part),
                             'get_momenta: slot %d does not hold particle %d'
                             % (slot, i))
            self.assertEqual(helicities[slot], int(part.helicity),
                             'get_helicity: slot %d does not hold particle %d'
                             % (slot, i))
        self.assertIn(momenta, [list(p) for p in all_momenta],
                      'get_all_momenta does not contain get_momenta\'s ordering')

    def test_every_ordering_of_four_tops(self):
        """All 24 orderings of the four top lines, identical ones included."""
        for final in itertools.permutations([6, 6, -6, -6]):
            with self.subTest(final=final):
                self.assertOneAssignment(_event([21, 21], final), self.TTTT)

    def test_a_mixed_final_state(self):
        """Distinct and identical flavours together, every ordering."""
        order = [(21, 21), (6, 23, -6, 23, 25)]
        for final in set(itertools.permutations([6, -6, 23, 23, 25])):
            with self.subTest(final=final):
                self.assertOneAssignment(_event([21, 21], final), order)

    def test_the_initial_state_in_either_order(self):
        order = [(2, -2), (23, 23)]
        for initial in ([2, -2], [-2, 2]):
            with self.subTest(initial=initial):
                self.assertOneAssignment(_event(initial, [23, 23]), order)

    def test_a_status_two_line_is_not_a_leg(self):
        """The record index counts it, the matrix element does not."""
        for where in range(5):
            with self.subTest(resonance_after=where):
                event = _event([21, 21], [6, 6, -6, -6], resonance_after=where)
                self.assertOneAssignment(event, self.TTTT)

    def test_a_charge_reversed_order(self):
        """MadSpin evaluates a t~ decay with the t matrix element when only
        that one exists (get_pdir's anti-particle fallback). get_momenta used
        to return a Fortran-formatted string here and get_all_momenta to
        raise, and calculate_matrix_element goes through the latter."""
        self.assertOneAssignment(_event([-6], [-24, -5]), [(6,), (24, 5)])
        self.assertOneAssignment(_event([-24], [11, -12]), [(24,), (-11, 12)])
        self.assertOneAssignment(_event([2, -2], [-6, -6, 6, 6]),
                                 [(-2, 2), (6, -6, 6, -6)])


class TestIdenticalParticleTieBreak(unittest.TestCase):
    """Among identical particles the k-th in event order goes to the k-th slot
    of that flavour. MadSpin's decay bookkeeping (_density_basis' init_part,
    _sequential_slots, add_decays) and the polarisation braces all count
    identical particles in event order, and rely on the matrix element seeing
    them in that same order."""

    def test_kth_particle_of_a_flavour_takes_the_kth_slot(self):
        order = [(21, 21), (6, -6, 6, -6)]
        flat = list(order[0]) + list(order[1])
        for final in set(itertools.permutations([6, 6, -6, -6])):
            event = _event([21, 21], final)
            mapping, _ = event.get_mapping(order)
            ext = _external(event)
            for pdg in (6, -6):
                in_event = [mapping[i] for i, p in enumerate(ext)
                            if p.pid == pdg]
                in_order = [s for s, leg in enumerate(flat) if leg == pdg]
                self.assertEqual(in_event, in_order, final)


class TestGetAllMomenta(unittest.TestCase):
    """get_all_momenta permutes identical particles with different mothers,
    starting from get_momenta's own assignment."""

    def test_two_decays_to_identical_products(self):
        """t t~ with both W's decaying to e: the two electrons have different
        mothers, so both assignments are candidates -- and the unpermuted one
        is exactly get_momenta's."""
        # 1 2 incoming; 3 = W+ (4-body toy: W+ > e+ ve, W- > e- ve~ written
        # as two e+ e- pairs from two mothers)
        lines = ['<event>', ' 8 1 1.0 100.0 0.0078 0.118',
                 ' 2 -1 0 0 0 0 0.0 0.0 500.0 500.0 0.0 0. 1',
                 ' -2 -1 0 0 0 0 0.0 0.0 -500.0 500.0 0.0 0. -1',
                 ' 23 2 1 2 0 0 1.0 2.0 3.0 300.0 91.0 0. 9',
                 ' 23 2 1 2 0 0 -1.0 -2.0 -3.0 300.0 91.0 0. 9',
                 ' 11 1 3 3 0 0 11.0 1.0 1.0 150.0 0.0 0. 1',
                 ' -11 1 3 3 0 0 12.0 2.0 2.0 150.0 0.0 0. -1',
                 ' 11 1 4 4 0 0 13.0 3.0 3.0 150.0 0.0 0. 1',
                 ' -11 1 4 4 0 0 14.0 4.0 4.0 150.0 0.0 0. -1',
                 '</event>']
        event = lhe_parser.Event('\n'.join(lines))
        order = [(2, -2), (11, -11, 11, -11)]
        base = event.get_momenta(order)
        with_swaps = event.get_all_momenta(order, permutate_two_decay=True)
        self.assertIn(base, [list(p) for p in with_swaps])
        for perm in with_swaps:
            # same particles, only moved among the slots of their own flavour
            flat = list(order[0]) + list(order[1])
            for slot, mom in enumerate(perm):
                self.assertEqual(flat[base.index(mom)], flat[slot])


def _chain_event(layout):
    """u d~ > w+ z, w+ > e+ ve, z > e+ e-, written with its lines in
    ``layout`` (names below). Each particle's momentum identifies it, and the
    mothers follow the lines wherever they are written."""
    parts = {'u': (2, -1, None, (0., 0., 223.), 0.),
             'dx': (-1, -1, None, (0., 0., -133.), 0.),
             'W': (24, 2, 'ini', (25., 37., 200.), 80.4),
             'Z': (23, 2, 'ini', (-25., -37., -110.), 91.19),
             'e+W': (-11, 1, 'W', (35., 12., 140.), 0.),
             've': (12, 1, 'W', (-10., 25., 60.), 0.),
             'e+Z': (-11, 1, 'Z', (-40., -30., -20.), 0.),
             'e-': (11, 1, 'Z', (15., -7., -90.), 0.)}
    idx = dict((name, k + 1) for k, name in enumerate(layout))
    lines = [' %d 1 1.0 100.0 0.0078 0.118' % len(layout)]
    for name in layout:
        pid, status, mother, (px, py, pz), mass = parts[name]
        if mother is None:
            m1 = m2 = 0
        elif mother == 'ini':
            m1, m2 = idx['u'], idx['dx']
        else:
            m1 = m2 = idx[mother]
        e = (mass ** 2 + px ** 2 + py ** 2 + pz ** 2) ** 0.5
        hel = -1 if name == 'e+Z' else 1
        lines.append(' %d %d %d %d 0 0 %r %r %r %r %r 0. %d'
                     % (pid, status, m1, m2, px, py, pz, e, mass, hel))
    return lhe_parser.Event('<event>\n' + '\n'.join(lines) + '\n</event>')


class TestDecayChainLayout(unittest.TestCase):
    """decay_chain=True: identical particles from different resonances are put
    where a decay-chain matrix element expects them -- the products of each
    resonance on consecutive legs -- whatever order the event writes them in.

    u d~ > w+ z, w+ > e+ ve, z > e+ e- has legs u d~ e+ ve e+ e-: leg 3 is
    the W's e+. Dealt in line order, an event writing the Z's e+ first put it
    on leg 3, so the matrix element (and me_frame = [3, 4], "the W rest frame")
    got an e+ ve pair that is not a W."""

    ORDER = [(2, -1), (-11, 12, -11, 11)]
    E_W = (35., 12., 140.)
    LAYOUTS = [['u', 'dx', 'W', 'Z', 'e+W', 've', 'e+Z', 'e-'],
               ['dx', 'u', 'Z', 'e-', 'e+Z', 'W', 've', 'e+W'],
               ['u', 'dx', 'e+Z', 'Z', 'e-', 've', 'W', 'e+W'],
               ['u', 'Z', 'dx', 'W', 'e+Z', 'e+W', 'e-', 've']]

    def test_the_w_products_take_legs_3_and_4(self):
        for layout in self.LAYOUTS:
            event = _chain_event(layout)
            p = event.get_momenta(self.ORDER, decay_chain=True)
            self.assertEqual(p[2][1:], self.E_W, layout)
            all_p = event.get_all_momenta(self.ORDER, decay_chain=True)
            self.assertEqual(len(all_p), 1)
            self.assertEqual(all_p[0], p)
            # legs 3+4 rebuild the W line
            w = lhe_parser.FourMomentum(p[2]) + lhe_parser.FourMomentum(p[3])
            self.assertEqual((w.px, w.py, w.pz), (25., 37., 200.))

    def test_the_helicity_follows_the_momentum(self):
        """get_helicity is dealt with the same mapping (the Z's e+ carries
        helicity -1, every other particle +1)"""
        for layout in self.LAYOUTS:
            event = _chain_event(layout)
            hel = event.get_helicity(self.ORDER, decay_chain=True)
            self.assertEqual(hel[2], 1, layout)
            self.assertEqual(hel[4], -1, layout)

    def test_the_default_keeps_the_line_order(self):
        """MadSpin relies on the k-th e+ taking the k-th e+ slot; without the
        option nothing changes"""
        event = _chain_event(self.LAYOUTS[1])
        p = event.get_momenta(self.ORDER)
        self.assertEqual(p[2][1:], (-40., -30., -20.))

    def test_a_process_not_written_as_a_chain_keeps_the_line_order(self):
        """u d~ > e+ e+ ve e-: no assignment puts both resonances on
        consecutive legs, and the two e+ are exchangeable in that matrix
        element, so the line order stays"""
        order = [(2, -1), (-11, -11, 12, 11)]
        for layout in self.LAYOUTS:
            event = _chain_event(layout)
            self.assertEqual(event.get_momenta(order, decay_chain=True),
                             event.get_momenta(order), layout)


if __name__ == '__main__':
    unittest.main()
