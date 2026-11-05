################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""MG7 (madspace) event generation for an interference: a signed integrand.

``p p > u u~ QCD^2==2`` keeps the QCD x EW cross term alone, whose total is
negative (85% of the events carry a negative weight). madspace integrates the
signed weights and unweights on |w|, keeping the sign, so the sample must:

  * reproduce the reference cross section, negative;
  * declare its negative weights in the LHE <init> block: IDWTUP = -4, as
    madevent writes, for which a shower takes the cross section from the signed
    event weights (Pythia8 reads -3 as |XSECUP| times the mean sign, and +3 is
    for positive weights only), with XMAXUP the unit weight sigma_abs and
    XSECUP the signed cross section;
  * carry events of both signs whose mean weight is the cross section;
  * with [generation] interference_helicity = "summed" (the helicity sum as the
    weight), keep the cross section and write the helicities as 9.

Configuration of test_check_xsec_processes_mg7.py (fixed scale mu = 91.188
GeV, NNPDF23_lo_as_0130_qed, its helpers are reused), the default mg7 cuts.
Reference: three mg7 runs of 200k events (seeds 31, 32, 33: -12248(17),
-12238(16), -12274(16) pb); madevent with the same settings gives
-12340(42) pb, 0.7% away.

Run locally with::

    ./tests/test_manager.py test_.*interference.*_mg7 -pA -t0

``MG7_INTERF_EVENTS`` sets the events per run (default 5000).
"""

from __future__ import absolute_import
from __future__ import division

import glob
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import madgraph.interface.master_interface as MGCmd
from tests.acceptance_tests.test_check_xsec_processes_mg7 import (
    _edit_run_card, _require_mg7_runtime, _tail)

pjoin = os.path.join

_PROCESS = 'p p > u u~ QCD^2==2'
_REFERENCE_CROSS = -12253.
_REFERENCE_ERROR = 10.
_TOLERANCE = 0.01
_NSIGMA = 3
_EVENTS = int(os.environ.get('MG7_INTERF_EVENTS', 5000))


def _read_lhe(path):
    """(the <init> lines, the event weights, the set of the helicities of
    every particle line) of an LHE file."""
    init, weights, helicities = [], [], set()
    with gzip.open(path, 'rt') as f:
        in_init = False
        particles = -1  # particle lines left in the current event, -1 outside
        for line in f:
            if particles == 0:
                # first line of the event block: NUP IDPRUP XWGTUP ...
                fields = line.split()
                weights.append(float(fields[2]))
                particles = int(fields[0])
            elif particles > 0:
                # IDUP ISTUP MOTHUP(2) ICOLUP(2) PUP(5) VTIMUP SPINUP
                helicities.add(float(line.split()[12]))
                particles -= 1
                if particles == 0:
                    particles = -1
            elif line.startswith('<event'):
                particles = 0
            elif line.startswith('<init>'):
                in_init = True
            elif line.startswith('</init>'):
                in_init = False
            elif in_init:
                init.append(line.split())
    return init, weights, helicities


class MG7InterferenceTest(unittest.TestCase):

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='mg7_interference_')

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def generate_sample(self, interference_helicity=None):
        """Generate the sample, optionally with [generation]
        interference_helicity set: (info.json "process", the LHE <init>
        lines, the event weights, the set of the LHE helicities), after
        checking the cross section against the reference."""
        _require_mg7_runtime(self)

        run_dir = pjoin(self.path, 'PROC')
        mg = MGCmd.MasterCmd()
        mg.no_notification()
        mg.exec_cmd('set automatic_html_opening False --no_save')
        mg.exec_cmd('generate %s' % _PROCESS)
        mg.exec_cmd('output mg7 %s' % run_dir)

        toml = pjoin(run_dir, 'Cards', 'run_card.toml')
        _edit_run_card(toml, _EVENTS, False)
        card = open(toml).read()
        edits = [(r'(?m)^output_format = \S+', 'output_format = "lhe"')]
        if interference_helicity:
            edits.append((r'(?m)^interference_helicity = \S+',
                          'interference_helicity = "%s"' % interference_helicity))
        for pattern, value in edits:
            card, count = re.subn(pattern, value, card)
            self.assertEqual(count, 1, pattern)
        open(toml, 'w').write(card)

        log = pjoin(run_dir, 'mg7_gen.log')
        with open(log, 'w') as logfh:
            ret = subprocess.call(
                [sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
                cwd=run_dir, stdout=logfh, stderr=subprocess.STDOUT)
        self.assertEqual(ret, 0, 'mg7 generate_events failed:\n%s' % _tail(log))

        infos = sorted(glob.glob(pjoin(run_dir, 'Events', '*', 'info.json')))
        self.assertTrue(infos, 'no info.json:\n%s' % _tail(log))
        with open(infos[-1]) as f:
            status = json.load(f)['process']
        cross, error = status['mean'], status['error']

        # the signed cross section
        self.assertLess(cross, 0)
        self.assertGreater(status['mean_abs'], abs(cross))
        sigma = math.hypot(error, _REFERENCE_ERROR)
        self.assertLessEqual(
            abs(cross - _REFERENCE_CROSS),
            _TOLERANCE * abs(_REFERENCE_CROSS) + _NSIGMA * sigma,
            'mg7 %.6g +- %.3g pb, reference %.6g pb' % (cross, error,
                                                       _REFERENCE_CROSS))
        return (status,) + _read_lhe(pjoin(os.path.dirname(infos[-1]),
                                           'events.lhe.gz'))

    def test_interference_signed_sample_mg7(self):
        status, init, weights, helicities = self.generate_sample()
        cross, error = status['mean'], status['error']
        abs_cross = status['mean_abs']
        # exact helicities (the default): real ones, not the 9 of "summed"
        self.assertTrue(helicities & {-1., 1.}, helicities)

        # the <init> block declares the negative weights
        self.assertEqual(int(init[0][8]), -4, 'IDWTUP: %s' % init[0])
        xsecup, _xerrup, xmaxup = (float(x) for x in init[1][:3])
        self.assertAlmostEqual(xsecup, cross, delta=1e-8 * abs(cross))
        self.assertAlmostEqual(xmaxup, abs_cross, delta=1e-8 * abs_cross)

        # events of both signs, at +-sigma_abs, averaging to the cross section
        self.assertEqual(len(weights), _EVENTS)
        negative = sum(1 for w in weights if w < 0)
        self.assertTrue(0 < negative < len(weights))
        unit = sorted(abs(w) for w in weights)[len(weights) // 2]
        self.assertAlmostEqual(unit, abs_cross, delta=0.05 * abs_cross)
        mean = sum(weights) / len(weights)
        # the sign is a binomial draw with p(-) = (1 - cross/abs_cross) / 2
        ratio = cross / abs_cross
        spread = 2 * abs_cross * math.sqrt((1 - ratio ** 2) / 4 / len(weights))
        self.assertLessEqual(abs(mean - cross), 5 * spread + 3 * error)

    def test_interference_summed_helicity_mg7(self):
        """[generation] interference_helicity = "summed": the weights are the
        helicity sum (same cross section), and the helicities, which then mean
        nothing, are 9 on every particle line."""
        status, init, weights, helicities = self.generate_sample('summed')
        self.assertEqual(helicities, {9.})
        self.assertEqual(int(init[0][8]), -4, 'IDWTUP: %s' % init[0])
        self.assertEqual(len(weights), _EVENTS)
        self.assertTrue(0 < sum(1 for w in weights if w < 0) < len(weights))


if __name__ == '__main__':
    unittest.main()
