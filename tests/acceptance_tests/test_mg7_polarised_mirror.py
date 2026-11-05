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
################################################################################
"""The initial-state mirror against a matrix element that is not Lorentz
invariant (mg7 / madspace).

madspace covers the beam-swapped copy of an asymmetric initial state by drawing
a per-event "mirror" choice and, when it comes up, writing the event with
``py, pz -> -py, -pz`` -- the rotation by pi about x that moves each leg onto
the other beam. The weight it pairs with that event is the matrix element, and
the two only match if |M|^2 is invariant under that rotation.

It is, for every Lorentz invariant matrix element, and for a longitudinally
polarised one. It is **not** for a transversely polarised particle that
``me_frame`` holds at rest: HELAS then quantises the spin along the *frame* z
axis (the ``pp == 0`` branch of ``vxxxxx``) rather than along the particle's own
momentum, the mirror flips that axis, and the + and - states swap.

The observable consequence is a symmetry that must hold and did not: on a
symmetric proton-proton collider, with the quantisation axis tied to the beam
axis, sigma(z{+}) and sigma(z{-}) are equal -- there is no way to tell the two
beams apart, and the mirror is precisely what averages over them. Evaluating the
matrix element on the unmirrored momenta while writing the mirrored ones broke
that: measured 4485 +- 39 pb against 6060 +- 54 pb, 23 sigma apart. Note that
the *sum* over the three polarisations stayed right, which is why no unpolarised
test ever saw this.

Run locally with::

    ./tests/test_manager.py test_polarised_mirror_symmetry_mg7 -pA -t0 -l INFO
"""

from __future__ import absolute_import
from __future__ import division

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc

pjoin = os.path.join

# Small enough to stay affordable in CI, large enough that the 23 sigma split
# this guards against cannot hide in the statistics.
_EVENTS = int(os.environ.get('MG7_MIRROR_EVENTS', 4000))
_SEED = 7


def _require_mg7_runtime(test):
    """``skipTest`` unless madspace and an LHAPDF grid for the run_card's
    default PDF are available, the same gate the other mg7 acceptance tests
    use."""
    try:
        import madspace
        has_mg7 = hasattr(madspace, 'ChannelEventGenerator')
    except ImportError:
        has_mg7 = False
    if not has_mg7:
        test.skipTest('mg7 runtime stack (madspace) unavailable')

    # The run itself locates LHAPDF from the MadGraph configuration, the way a
    # user's bin/generate_events does, so the gate has to look there too and
    # not only at $LHAPDF_DATA_PATH.
    from madgraph.various.banner import RunCardMG7
    pdf = RunCardMG7()['beam']['pdf']
    if not misc.resolve_lhapdf().find_set(pdf):
        test.skipTest('%s LHAPDF data not found (set the lhapdf option or '
                      '$LHAPDF_DATA_PATH)' % pdf)


class MG7PolarisedMirrorTest(unittest.TestCase):
    """sigma(z{+}) == sigma(z{-}) for p p > z j evaluated in the Z rest frame."""

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='mg7_pol_mirror_')

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def _cross_section(self, process, tag, me_frame):
        """Generate *process* as an mg7 output, run it with the given me_frame
        and return (cross section, error) in pb."""
        run_dir = pjoin(self.path, tag)
        mg = MGCmd.MasterCmd()
        mg.no_notification()
        mg.exec_cmd('set automatic_html_opening False --no_save')
        mg.exec_cmd('generate %s' % process)
        mg.exec_cmd('output mg7 %s' % run_dir)

        # set through the card's own sections rather than by editing its text:
        # several sections have an 'enable' key, and which one comes first is
        # the template's business
        from madgraph.various.banner import RunCardMG7
        toml = pjoin(run_dir, 'Cards', 'run_card.toml')
        card = RunCardMG7(toml)
        card.set('generation.events', _EVENTS, user=True)
        card.set('run.seed', _SEED, user=True)
        card.set('run.me_frame', me_frame, user=True)
        # the scale/PDF variation weights are irrelevant here and cost time
        card.set('systematics.enable', False, user=True)
        card.write(toml)

        log = pjoin(run_dir, 'mg7_gen.log')
        with open(log, 'w') as logfh:
            ret = subprocess.call(
                [sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
                cwd=run_dir, stdout=logfh, stderr=subprocess.STDOUT)
        self.assertEqual(ret, 0, 'mg7 generate_events failed for %s (see %s)'
                         % (process, log))
        infos = sorted(glob.glob(pjoin(run_dir, 'Events', '*', 'info.json')))
        self.assertTrue(infos, 'no info.json produced for %s (see %s)'
                        % (process, log))
        with open(infos[-1]) as infofh:
            info = json.load(infofh)['process']
        return float(info['mean']), float(info.get('error') or 0.0)

    def test_polarised_mirror_symmetry_mg7(self):
        """The two transverse polarisations must come out equal.

        Both are computed in the Z rest frame (me_frame = [3]), which is the
        standard way to ask for Z polarisation and the one case where the spin
        axis is the frame's rather than the particle's -- exactly where the
        mirror used to swap + and - without the weight following.
        """
        _require_mg7_runtime(self)

        plus, plus_err = self._cross_section('p p > z{+} j', 'zplus', [3])
        minus, minus_err = self._cross_section('p p > z{-} j', 'zminus', [3])

        spread = (plus_err ** 2 + minus_err ** 2) ** 0.5
        self.assertAlmostEqual(
            plus, minus, delta=max(5 * spread, 0.01 * plus),
            msg=('p p > z{+} j and p p > z{-} j must have the same cross '
                 'section on a symmetric pp collider (the beams cannot be told '
                 'apart, and the initial-state mirror is what averages over '
                 'them), got %.6g +- %.3g pb and %.6g +- %.3g pb. A split here '
                 'means the matrix element is being evaluated on momenta the '
                 'event is not written with -- see Integrand::build_common_part '
                 'in madspace/src/phasespace/integrand.cpp.'
                 % (plus, plus_err, minus, minus_err)))
