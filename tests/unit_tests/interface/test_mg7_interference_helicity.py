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
"""[generation] interference_helicity: the helicity of an interference.

A helicity can contribute negatively to an interference (squared split orders
dropping a component). "exact", the default, draws the helicity of each event
as madevent does and the weight carries its sign, so the helicity is exact;
"summed" keeps the helicity sum as the weight (lower variance), and its
helicities, which then mean nothing, are written as 9 in the LHE. The matrix
element library is told through the umami parameter
interference_helicity_summed, sent only when asked for: a library made before
it existed refuses an unknown parameter.
"""

from __future__ import absolute_import

import io
import unittest

from madgraph.various.banner import RunCardMG7

# helicity rows of a 2 -> 2 subprocess
HELICITIES = [[-1, 1, -1, 1], [1, -1, 1, -1]]


def card(value=None):
    rc = RunCardMG7()
    if value is not None:
        rc.set('generation.interference_helicity', value, user=True)
    return rc


class TestInterferenceHelicity(unittest.TestCase):

    def setUp(self):
        try:
            import madgraph.mg7.launch as launch
        except ImportError as error:  # the mg7 runtime (madspace) is not there
            self.skipTest('mg7 launch not importable: %s' % error)
        self.launch = launch

    def test_default_is_exact(self):
        rc = card()
        self.assertEqual(rc['generation']['interference_helicity'], 'exact')
        self.assertFalse(self.launch.interference_helicities_summed(rc))
        # nothing new sent to the libraries, so that older ones keep working
        self.assertEqual(set(self.launch.me_parameters(rc)), {'bwcutoff'})
        meta = {'helicities': HELICITIES, 'interference': True}
        self.assertEqual(self.launch.lhe_helicities(meta, rc), HELICITIES)

    def test_summed(self):
        rc = card('summed')
        self.assertTrue(self.launch.interference_helicities_summed(rc))
        self.assertEqual(
            self.launch.me_parameters(rc)['interference_helicity_summed'], 1.)
        meta = {'helicities': HELICITIES, 'interference': True}
        self.assertEqual(self.launch.lhe_helicities(meta, rc),
                         [[9, 9, 9, 9], [9, 9, 9, 9]])

    def test_summed_leaves_the_other_subprocesses_alone(self):
        """Only an interference subprocess has helicities that can be
        negative; the others keep theirs (as an output made before the
        "interference" key, which has none)."""
        rc = card('summed')
        for meta in ({'helicities': HELICITIES, 'interference': False},
                     {'helicities': HELICITIES}):
            self.assertEqual(self.launch.lhe_helicities(meta, rc), HELICITIES)

    def test_card_round_trip(self):
        rc = card('summed')
        out = io.StringIO()
        rc.write(out)
        self.assertIn('interference_helicity = "summed"', out.getvalue())
        back = RunCardMG7(out.getvalue())
        self.assertEqual(back['generation']['interference_helicity'], 'summed')


if __name__ == '__main__':
    unittest.main()
