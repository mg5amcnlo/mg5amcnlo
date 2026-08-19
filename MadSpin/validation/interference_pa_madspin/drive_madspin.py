#!/usr/bin/env python3
"""Run a MadSpin card, with the reshuffling jacobian instrumented.

    drive_madspin.py <card.dat>

Identical to ``MadSpin/madspin <card.dat>`` except that, before the card is
read, ``lhe_parser.Event.reshuffle_production`` is wrapped so that every call
is counted and every return value that is *not* a finite positive number is
recorded.

Why: under ``spinmode = PA`` (and, through
``calculate_matrix_element_from_density``, under ``madspin``) the
pure-interference convolution is ``W = wgt * jac`` with ``jac`` the return of
``reshuffle_production``.  A ``jac <= 0`` is the one route by which ``W <= 0``
can mean "this trial is invalid" rather than "this trial has a sign", and the
two output paths of PR #363 treat a sign as physics.  So the rate has to be
*measured*, not argued about.

``reshuffle_production(_allow_retry=True)`` -- the default, and what both
production paths use -- resamples the masses and recurses when RAMBO reports
``jac in (0, -1)``, so those internal failures do not reach the caller.  They
are counted separately through ``Event.nb_reshuffle_issue``, the class counter
the retry already increments.  What *can* reach the caller is a zero from the
decay-level ``reshuffle_decay``, which is multiplied in after the retry check.

Counters are written to ``$MS_PA_JACLOG.<pid>.json`` at the end of every
``_unweight_range`` call, i.e. once per forked shard as well as in the parent
(multiprocessing's fork children do not run ``atexit`` handlers, so the dump is
hung off the function that does the work instead).
"""

import json
import math
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))
sys.path.insert(0, root)

import madgraph.various.lhe_parser as lhe_parser        # noqa: E402
import MadSpin.interface_madspin as interface           # noqa: E402

JACLOG = os.environ.get('MS_PA_JACLOG')

STATS = dict(n_calls=0, n_bad=0, bad_values=[], n_zero=0, n_negative=0,
             n_nonfinite=0)


def _install():
    orig = lhe_parser.Event.reshuffle_production

    def wrapper(self, _allow_retry=True):
        STATS['n_calls'] += 1
        jac = orig(self, _allow_retry=_allow_retry)
        try:
            ok = math.isfinite(jac) and jac > 0
        except TypeError:
            ok = False
        if not ok:
            STATS['n_bad'] += 1
            if len(STATS['bad_values']) < 200:
                STATS['bad_values'].append(repr(jac))
            try:
                if not math.isfinite(jac):
                    STATS['n_nonfinite'] += 1
                elif jac == 0:
                    STATS['n_zero'] += 1
                else:
                    STATS['n_negative'] += 1
            except TypeError:
                STATS['n_nonfinite'] += 1
        return jac

    lhe_parser.Event.reshuffle_production = wrapper

    orig_range = interface.MadSpinInterface._unweight_range

    def range_wrapper(self, *args, **kw):
        try:
            return orig_range(self, *args, **kw)
        finally:
            _dump()

    interface.MadSpinInterface._unweight_range = range_wrapper


def _dump():
    if not JACLOG:
        return
    out = dict(STATS)
    out['pid'] = os.getpid()
    # the retry counter the reshuffle itself keeps: internal RAMBO failures
    # that were resolved by resampling the resonance masses
    out['nb_reshuffle_issue'] = getattr(lhe_parser.Event,
                                        'nb_reshuffle_issue', None)
    with open('%s.%d.json' % (JACLOG, os.getpid()), 'w') as f:
        json.dump(out, f, indent=1)


def main():
    _install()
    card = os.path.realpath(sys.argv[1])
    cmd_line = interface.MadSpinInterface()
    cmd_line.use_rawinput = False
    cmd_line.import_command_file(card)
    _dump()


if __name__ == '__main__':
    main()
