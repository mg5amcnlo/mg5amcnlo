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
"""The reweighting's choice of legs must not depend on the LHE's line order.

Two places picked "leg n" by walking the event's own lines while every other
quantity they were combined with -- the momenta handed to the matrix element,
pdg, POS of GET_DENSITY -- is laid out in the matrix element's leg order:

* ReweightInterface.method_boost_event's ``frame_id`` branch (a non-default
  ``me_frame``). It never ran: it died on an undefined ``FourMomenta`` and on
  ``str.reverse``. Fixed by name only, it would have boosted to the rest frame
  of whatever particle the LHE wrote at that place.
* DensityInterface: its boost counted every LHE line, status-2 ones included,
  against matrix-element leg numbers, and ranking by an observable paired
  ``pdg[i]`` (a leg) with ``event[i]`` (a line) -- so the boost frame, the
  helicity axis and the POS given to GET_DENSITY could all belong to another
  particle.

The tests feed the same physical event written in several line orders and
require identical answers.
"""

from __future__ import absolute_import
import math
import unittest

import madgraph
import madgraph.various.lhe_parser as lhe_parser
import madgraph.interface.reweight_interface as reweight_interface


def _event(final, resonance=False):
    """g g > final, each final-state particle with its own momentum (fixed by
    its identity, not its place), so that permuting ``final`` permutes lines
    of one and the same physical event."""
    momenta = {('t', 0): (40., 10., 30.), ('t', 1): (-25., 60., -80.),
               ('tx', 0): (-35., -50., 20.), ('tx', 1): (20., -20., 130.)}
    lines = [' %d 1 1.0 100.0 0.0078 0.118' % (2 + len(final) + resonance),
             ' 21 -1 0 0 501 502 0.0 0.0 600.0 600.0 0.0 0. 9',
             ' 21 -1 0 0 503 501 0.0 0.0 -600.0 600.0 0.0 0. 9']
    if resonance:
        lines.append(' 25 2 1 2 0 0 0.0 0.0 0.0 700.0 125.0 0. 9')
    for name in final:
        pid = 6 if name[0] == 't' else -6
        px, py, pz = momenta[name]
        e = math.sqrt(173. ** 2 + px ** 2 + py ** 2 + pz ** 2)
        lines.append(' %d 1 1 2 0 0 %r %r %r %r 173.0 0. 9'
                     % (pid, px, py, pz, e))
    return lhe_parser.Event('<event>\n' + '\n'.join(lines) + '\n</event>')


ORDER = [(21, 21), (6, -6, 6, -6)]
# the four tops, identified by name; the event writes them in these orders
LAYOUTS = [[('t', 0), ('tx', 0), ('t', 1), ('tx', 1)],   # madevent
           [('t', 0), ('t', 1), ('tx', 0), ('tx', 1)],   # aMC@NLO
           [('tx', 0), ('t', 0), ('tx', 1), ('t', 1)]]


def _rest(momenta, legs):
    """the summed three-momentum of ``legs`` (0-based)"""
    return [sum(momenta[leg][k] for leg in legs) for k in (1, 2, 3)]


class _BaseStub(object):
    class _Banner(object):
        def __init__(self, frame_id):
            self.run_card = {'frame_id': frame_id}

    def __init__(self, frame_id):
        self.banner = self._Banner(frame_id)
        self.keep_ordering = False
        self.boost_event = False

    method_boost_event = reweight_interface.ReweightInterface.method_boost_event
    boost_momenta_to_rest_frame = staticmethod(
        reweight_interface.ReweightInterface.boost_momenta_to_rest_frame)
    boost_momenta_to_me_frame = classmethod(
        reweight_interface.ReweightInterface.boost_momenta_to_me_frame.__func__)


class TestFrameIdBoost(unittest.TestCase):
    """me_frame / frame_id in the reweighting: bit n is leg n of the matrix
    element, counted from 1."""

    def _boosted(self, frame_id, layout, resonance=False):
        event = _event(layout, resonance)
        all_p = [event.get_momenta(ORDER)]
        return _BaseStub(frame_id).method_boost_event(event, all_p, ORDER, 0)

    def test_the_chosen_legs_end_up_at_rest(self):
        """me_frame = [3, 4]: the first top and first anti-top of the matrix
        element, whichever LHE line they sit on."""
        frame_id = 2 ** 3 + 2 ** 4
        for layout in LAYOUTS:
            out = self._boosted(frame_id, layout)
            self.assertEqual(len(out), 1)
            for comp in _rest(out[0], [2, 3]):
                self.assertAlmostEqual(comp, 0., places=8)

    def test_the_line_order_does_not_matter(self):
        frame_id = 2 ** 3 + 2 ** 4
        ref = self._boosted(frame_id, LAYOUTS[0])
        for layout in LAYOUTS[1:]:
            for resonance in (False, True):
                out = self._boosted(frame_id, layout, resonance)
                for a, b in zip(out[0], ref[0]):
                    for x, y in zip(a, b):
                        self.assertAlmostEqual(x, y, places=8)

    def test_a_single_leg_is_exactly_at_rest(self):
        """HELAS takes the frame's z axis only for a momentum exactly at rest;
        the boost arithmetic alone leaves ~1e-14."""
        out = self._boosted(2 ** 5, LAYOUTS[1])
        self.assertEqual(out[0][4][1:], (0., 0., 0.))
        self.assertAlmostEqual(out[0][4][0], 173., places=8)

    def test_no_selected_leg_changes_nothing(self):
        event = _event(LAYOUTS[0])
        all_p = [event.get_momenta(ORDER)]
        # bit 0 is not a leg
        self.assertIs(_BaseStub(1).method_boost_event(event, all_p, ORDER, 0),
                      all_p)


class _DensityStub(object):
    """the DensityInterface pieces that choose, rank and boost"""

    def __init__(self, momenta_boost):
        self.momenta_boost = momenta_boost
        self.keep_ordering = False

    method_boost_event = reweight_interface.DensityInterface.method_boost_event
    chose_particle_user_input = \
        reweight_interface.DensityInterface.chose_particle_user_input
    find_position_particles_default_order = \
        reweight_interface.DensityInterface.find_position_particles_default_order
    find_position_particles_new_order = \
        reweight_interface.DensityInterface.find_position_particles_new_order
    find_position_particles_with_observable = \
        reweight_interface.DensityInterface.find_position_particles_with_observable
    _boost_to_chosen_legs = reweight_interface.DensityInterface._boost_to_chosen_legs
    boost_momenta_to_rest_frame = staticmethod(
        reweight_interface.ReweightInterface.boost_momenta_to_rest_frame)


class TestDensityInterfaceLegs(unittest.TestCase):

    PROPERTIES = [p for p in dir(lhe_parser.FourMomentum)
                  if isinstance(getattr(lhe_parser.FourMomentum, p), property)]
    PDG = list(ORDER[0]) + list(ORDER[1])

    def _choose(self, layout, user_input, resonance=False, fortran=False):
        event = _event(layout, resonance)
        lab_p = event.get_momenta(ORDER)
        stub = _DensityStub(user_input)
        return stub.chose_particle_user_input(lab_p, self.PDG, self.PROPERTIES,
                                              ORDER, user_input, 'test',
                                              fortran), lab_p, event

    def test_ranked_choice_is_the_same_particle_in_every_layout(self):
        """'the hardest top' names one particle, however the LHE lists them."""
        chosen = set()
        for layout in LAYOUTS:
            for resonance in (False, True):
                positions, lab_p, _ = self._choose(layout, [[6], 'pt', []],
                                                   resonance)
                self.assertEqual(len(positions), 1)
                self.assertEqual(self.PDG[positions[0]], 6)
                pts = [math.hypot(p[1], p[2]) for p, pid in
                       zip(lab_p, self.PDG) if pid == 6]
                chosen_p = lab_p[positions[0]]
                self.assertEqual(math.hypot(chosen_p[1], chosen_p[2]), max(pts))
                chosen.add(tuple(round(x, 9) for x in chosen_p))
        self.assertEqual(len(chosen), 1)

    def test_default_choice_is_a_leg_of_the_right_flavour(self):
        for layout in LAYOUTS:
            positions, _, _ = self._choose(layout, [[6, -6], '', []])
            self.assertEqual([self.PDG[p] for p in positions], [6, -6])

    def test_boost_puts_the_chosen_legs_at_rest(self):
        """the boost is built from the chosen legs of all_p, in the matrix
        element's order, not from LHE lines -- a status-2 line included"""
        for layout in LAYOUTS:
            for resonance in (False, True):
                user_input = [[6, -6], '', []]
                positions, lab_p, event = self._choose(layout, user_input,
                                                       resonance)
                out = _DensityStub(user_input).method_boost_event(
                                    event, [list(lab_p)], ORDER, 0, positions)
                for comp in _rest(out[0], positions):
                    self.assertAlmostEqual(comp, 0., places=8)

    def test_boost_keeps_the_momenta_it_was_given(self):
        """the momenta boosted are the ones given, not the event's own read
        back from a boosted copy of it, as this did"""
        user_input = [[6], '', []]
        positions, lab_p, event = self._choose(LAYOUTS[1], user_input)
        shifted = [tuple(x * 1.01 for x in p) for p in lab_p]
        out = _DensityStub(user_input).method_boost_event(
                                event, [shifted], ORDER, 0, positions)
        # the chosen leg of the *given* momenta is at rest, which it would not
        # be if the event's own momenta had been read back
        self.assertEqual(len(out), 1)
        for comp in _rest(out[0], positions):
            self.assertAlmostEqual(comp, 0., places=8)


# u d~ > w+ z with unequal parton energies: the partonic CM moves along z
# (rapidity 0.26), so the lab and the partonic CM are different frames
_W = (35. - 10., 12. + 25., 140. + 60.)
_Z = (-40. + 15. - 25., -30. - 7. - 37., -20. - 90.)


def _wz_event(layout):
    """``layout``: the order of 'u', 'dx', 'W', 'Z' in the event"""
    mom = {'W': (24, 1, _W, 80.4), 'Z': (23, 1, _Z, 91.19)}
    e_w = math.sqrt(80.4 ** 2 + sum(x * x for x in _W))
    e_z = math.sqrt(91.19 ** 2 + sum(x * x for x in _Z))
    etot, pztot = e_w + e_z, _W[2] + _Z[2]
    mom['u'] = (2, -1, (0., 0., (etot + pztot) / 2), 0.)
    mom['dx'] = (-1, -1, (0., 0., -(etot - pztot) / 2), 0.)
    idx = dict((name, k + 1) for k, name in enumerate(layout))
    lines = [' 4 1 1.0 100.0 0.0078 0.118']
    for name in layout:
        pid, status, (px, py, pz), mass = mom[name]
        m1, m2 = (idx['u'], idx['dx']) if status == 1 else (0, 0)
        e = math.sqrt(mass ** 2 + px ** 2 + py ** 2 + pz ** 2)
        lines.append(' %d %d %d %d 0 0 %r %r %r %r %r 0. 9'
                     % (pid, status, m1, m2, px, py, pz, e, mass))
    return lhe_parser.Event('<event>\n' + '\n'.join(lines) + '\n</event>')


WZ_ORDER = [(2, -1), (24, 23)]
WZ_LAYOUTS = [['u', 'dx', 'W', 'Z'], ['dx', 'u', 'Z', 'W'],
              ['u', 'Z', 'dx', 'W']]


def _madevent_frame(lab, legs):
    """What madevent hands the matrix element: the momenta in the partonic
    CM (genps.f), then boost_to_frame on the legs selected, a single one set
    exactly at rest -- written out independently of the code under test."""
    pcm = lhe_parser.FourMomentum(lab[0]) + lhe_parser.FourMomentum(lab[1])
    cm = [lhe_parser.FourMomentum(p).zboost(pcm) for p in lab]
    pboost = lhe_parser.FourMomentum()
    for n in legs:
        pboost += cm[n - 1]
    neg = lhe_parser.FourMomentum(pboost.E, -pboost.px, -pboost.py, -pboost.pz)
    out = [p.boost(neg) for p in cm]
    out = [(p.E, p.px, p.py, p.pz) for p in out]
    if len(legs) == 1:
        out[legs[0] - 1] = (out[legs[0] - 1][0], 0., 0., 0.)
    return out


class TestMeFrameConvention(unittest.TestCase):
    """me_frame has to be madevent's frame, for both matrix elements of the
    weight.

    The four-top events above have back-to-back beams of equal energy, so their
    lab *is* the partonic CM and none of this can show there."""

    def _boosted(self, me_frame, layout=WZ_LAYOUTS[0], hypp_id=0, stub=None):
        event = _wz_event(layout)
        all_p = [event.get_momenta(WZ_ORDER)]
        stub = stub or _BaseStub(sum(2 ** n for n in me_frame))
        return stub.method_boost_event(event, all_p, WZ_ORDER, hypp_id)[0]

    def assertMomentaEqual(self, first, second, places=8):
        for a, b in zip(first, second):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=places)

    def test_the_frame_goes_through_the_partonic_cm(self):
        """madevent boosts to the frame from the partonic CM. Straight from
        the lab is another frame -- a Wigner rotation, 2.8 degrees here -- and
        turns the quantisation axis of a leg at rest with it."""
        for me_frame in ([3], [4], [3, 4]):
            lab = _wz_event(WZ_LAYOUTS[0]).get_momenta(WZ_ORDER)
            ref = _madevent_frame(lab, me_frame)
            for layout in WZ_LAYOUTS:
                self.assertMomentaEqual(self._boosted(me_frame, layout), ref)
        # the test can tell: from the lab, the beam is not where madevent has it
        lab = _wz_event(WZ_LAYOUTS[0]).get_momenta(WZ_ORDER)
        direct = _BaseStub(8).boost_momenta_to_rest_frame(
                                lab, lhe_parser.FourMomentum(lab[2]), 2)
        ref = _madevent_frame(lab, [3])
        self.assertGreater(max(abs(x - y) for a, b in zip(direct, ref)
                               for x, y in zip(a, b)), 10.)

    def test_a_single_leg_is_exactly_at_rest(self):
        for leg in (3, 4):
            out = self._boosted([leg])
            self.assertEqual(out[leg - 1][1:], (0., 0., 0.))

    def test_both_matrix_elements_are_in_the_frame(self):
        """the weight is w_new / w_orig: the new matrix element (hypp_id 1)
        was left in the partonic CM while the original one was in me_frame"""
        for me_frame in ([3], [3, 4]):
            self.assertMomentaEqual(self._boosted(me_frame, hypp_id=1),
                                    self._boosted(me_frame, hypp_id=0))

    def test_a_boost_set_for_the_new_matrix_element_wins(self):
        """'change boost' (boost_event) is the user's explicit choice for the
        new matrix element; True means: leave the momenta as they are"""
        stub = _BaseStub(8)
        stub.boost_event = True
        event = _wz_event(WZ_LAYOUTS[0])
        all_p = [event.get_momenta(WZ_ORDER)]
        self.assertIs(stub.method_boost_event(event, all_p, WZ_ORDER, 1), all_p)

    def test_a_massless_frame_is_refused(self):
        """a single massless leg has no rest frame: this used to be a
        ZeroDivisionError deep inside FourMomentum.boost"""
        event = _event(LAYOUTS[0])
        all_p = [event.get_momenta(ORDER)]
        # leg 1, a gluon
        self.assertRaises(madgraph.InvalidCmd, _BaseStub(2).method_boost_event,
                          event, all_p, ORDER, 0)
        self.assertRaises(madgraph.InvalidCmd,
                          _BaseStub(2).boost_momenta_to_rest_frame,
                          all_p[0], lhe_parser.FourMomentum(all_p[0][0]))

    def test_each_assignment_gets_its_own_frame(self):
        """with identical particles all_p holds one assignment per guess of
        which particle is which leg; each is boosted to the rest frame of *its*
        leg -- one boost from all_p[0] left the other guesses' leg moving"""
        event = _event(LAYOUTS[0])
        first = event.get_momenta(ORDER)
        swapped = list(first)
        swapped[2], swapped[4] = swapped[4], swapped[2]
        for stub, hypp_id in ((_BaseStub(2 ** 3), 0), (_BaseStub(2 ** 3), 1)):
            out = stub.method_boost_event(event, [first, swapped], ORDER,
                                          hypp_id)
            for p in out:
                self.assertEqual(p[2][1:], (0., 0., 0.))
        # and the same in DensityInterface's boost, legs given 0-based
        density = _DensityStub([[6], '', []])
        out = density.method_boost_event(event, [first, swapped], ORDER, 0, [2])
        for p in out:
            for comp in _rest(p, [2]):
                self.assertAlmostEqual(comp, 0., places=8)


if __name__ == '__main__':
    unittest.main()
