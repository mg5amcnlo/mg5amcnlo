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

Environment
-----------
``MS_PA_JACLOG=<prefix>``
    Write ``<prefix>.<stage>.<pid>.json`` counters.  Counters are reset when
    each instrumented stage starts and dumped when it ends, so summing every
    file gives the true total with no double counting across forked shards
    (multiprocessing's fork children inherit the parent's counters, and do not
    run ``atexit`` handlers).  Two stages are instrumented: ``probe``
    (``_joint_maxwgt_range``, where ``c``, ``<|W|>`` and ``max|W|`` are
    measured) and ``unweight`` (``_unweight_range``, the loop that writes the
    events).

``MS_PA_FORCE_BADJAC=<p>[:<value>]``
    Fault injection.  With probability ``p``, ``reshuffle_production`` returns
    ``value`` (default ``0.0``) instead of the jacobian it computed.  That is
    the *only* way to exercise the failed-reshuffle handling on a process where
    it does not fire on its own, and it is what makes the difference between
    "invalid trial" and "negative interference weight" testable: inject
    ``0.0`` and the trial's ``W`` is zero, inject ``-1.0`` and its ``W`` has
    the sign and the full magnitude of a legitimate interference weight.
"""

import json
import math
import os
import random
import sys

root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))
sys.path.insert(0, root)

import madgraph.various.lhe_parser as lhe_parser        # noqa: E402
import MadSpin.interface_madspin as interface           # noqa: E402

JACLOG = os.environ.get('MS_PA_JACLOG')

_force = os.environ.get('MS_PA_FORCE_BADJAC')
FORCE_P, FORCE_VALUE = 0.0, 0.0
if _force:
    parts = _force.split(':')
    FORCE_P = float(parts[0])
    if len(parts) > 1:
        FORCE_VALUE = float(parts[1])

STATS = {}
# Fault injection is confined to the event loop.  Injecting it into the
# maximum-weight probe as well would corrupt c, <|W|> and max|W|, and the
# question under test is what the WRITING loop does with a failed reshuffle,
# not what a mis-measured constant does.
STAGE = [None]


def _reset():
    STATS.clear()
    STATS.update(n_calls=0, n_bad=0, bad_values=[], n_zero=0, n_negative=0,
                 n_nonfinite=0, n_forced=0,
                 reshuffle_issue_at_start=getattr(
                     lhe_parser.Event, 'nb_reshuffle_issue', 0))


_reset()


def _install():
    orig = lhe_parser.Event.reshuffle_production
    rng = random.Random()

    def wrapper(self, _allow_retry=True):
        STATS['n_calls'] += 1
        jac = orig(self, _allow_retry=_allow_retry)
        if FORCE_P and STAGE[0] == 'unweight' and rng.random() < FORCE_P:
            jac = FORCE_VALUE
            STATS['n_forced'] += 1
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

    for stage, name in (('probe', '_joint_maxwgt_range'),
                        ('unweight', '_unweight_range')):
        original = getattr(interface.MadSpinInterface, name)

        def make(original, stage):
            def stage_wrapper(self, *args, **kw):
                _reset()
                STAGE[0] = stage
                try:
                    return original(self, *args, **kw)
                finally:
                    _dump(stage)
                    STAGE[0] = None
            return stage_wrapper

        setattr(interface.MadSpinInterface, name, make(original, stage))


def _dump(stage):
    if not JACLOG:
        return
    out = dict(STATS)
    out.pop('bad_values', None)
    out['bad_values'] = STATS['bad_values']
    out['pid'] = os.getpid()
    out['stage'] = stage
    # internal RAMBO failures the reshuffle resolved by resampling the masses
    out['nb_reshuffle_issue'] = (
        getattr(lhe_parser.Event, 'nb_reshuffle_issue', 0)
        - out.pop('reshuffle_issue_at_start', 0))
    with open('%s.%s.%d.json' % (JACLOG, stage, os.getpid()), 'w') as f:
        json.dump(out, f, indent=1)


def main():
    _install()
    card = os.path.realpath(sys.argv[1])
    cmd_line = interface.MadSpinInterface()
    cmd_line.use_rawinput = False
    cmd_line.import_command_file(card)


if __name__ == '__main__':
    main()
