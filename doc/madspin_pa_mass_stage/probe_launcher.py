#!/usr/bin/env python3
"""An instrumented stand-in for the ``MadSpin/madspin`` executable.

It runs *exactly* the shipped MadSpin -- no line of ``interface_madspin.py`` is
touched -- but before handing over it wraps four methods so that every quantity
the mass/virtuality stage of the sequential accept/reject actually uses is
written to a stream file:

  ``U``  one call to ``_upfront_production``: the production reshuffling
         jacobian ``jac_prod`` it settled for that mass set, the per-slot
         Breit-Wigner sampling jacobians ``jac_bw``, and the drawn virtualities.
         ``U -`` is a mass set the production could not be reshuffled onto
         (``_upfront_production`` returned None and the chain restarts).
  ``Z``  one call to ``_zhat``: the tabulated conditional normalisation the
         mass-set weight multiplies in, per slot.
  ``W``  the mass-set weight ``w_mass`` at the instant it is tested against
         ``maxwgts[0]``.  Read off ``_dead_trial``, which the accept/reject
         calls with that exact value on the line above ``random.random() *
         maxwgts[0] >= w_mass`` -- so this is the tested quantity itself, not a
         reconstruction of it.
  ``M``  the bound vector returned by ``get_sequential_maxwgt``; ``M[0]`` is the
         mass stage's bound.

A ``P`` line brackets the maximum-weight scan, so probe-phase draws (which make
no accept/reject) can be told apart from production ones.

Within one chain attempt the calls come in the order ``U (Z Z ...) W``, which is
what ``analyse.py`` pairs on: the ``Z`` records strictly between a ``U`` and the
next ``W`` are that mass set's, and any later ``_zhat`` call belongs to the
angle stage.

Usage (identical to ``MadSpin/madspin``)::

    MS_PROBE_OUT=/path/to/stream.txt python3 -O probe_launcher.py card.dat
"""

from __future__ import absolute_import

import os
import sys

_here = os.path.dirname(os.path.realpath(__file__))
_root = os.path.split(os.path.split(_here)[0])[0]
if _root not in sys.path:
    sys.path.insert(0, _root)

import logging
import logging.config

import madgraph.interface.coloring_logging  # noqa: F401
import MadSpin.interface_madspin as interface

try:
    logging.config.fileConfig(os.path.join(_root, 'madgraph', 'interface',
                                           '.mg5_logging.conf'))
    logging.root.setLevel(logging.INFO)
    logging.getLogger('madgraph').setLevel(logging.INFO)
    logging.getLogger('madevent').setLevel(logging.INFO)
except Exception:
    pass


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------
_OUT = open(os.environ['MS_PROBE_OUT'], 'w', buffering=1 << 20)
_STATE = {'probe': 0}

MS = interface.MadSpinInterface


def _fmt(value):
    return '%.10g' % value


_orig_upfront = MS._upfront_production


def _upfront_production(self, *args, **kwargs):
    result = _orig_upfront(self, *args, **kwargs)
    if result is None:
        _OUT.write('U -\n')
        return result
    _, jac_prod, slot_mass, _, _ = result
    slots = sorted(slot_mass)
    _OUT.write('U %s %s %s\n' % (
        _fmt(jac_prod),
        ','.join(_fmt(slot_mass[s][2]) for s in slots) or '-',
        ','.join(_fmt(slot_mass[s][0]) for s in slots) or '-'))
    return result


MS._upfront_production = _upfront_production


_orig_zhat = MS._zhat


def _zhat(self, key, mass):
    value = _orig_zhat(self, key, mass)
    _OUT.write('Z %s %s %s\n' % (key, _fmt(mass), _fmt(value)))
    return value


MS._zhat = _zhat


_orig_dead_trial = MS._dead_trial


def _dead_trial(self, counter, wgt, stage):
    if 'mass-set stage' in stage:
        try:
            _OUT.write('W %s\n' % _fmt(wgt))
        except (TypeError, ValueError):
            _OUT.write('W nan\n')
    return _orig_dead_trial(self, counter, wgt, stage)


MS._dead_trial = _dead_trial


_orig_maxwgt = MS.get_sequential_maxwgt


def get_sequential_maxwgt(self, orig_lhe, evt_decayfile):
    _STATE['probe'] += 1
    _OUT.write('P start\n')
    try:
        bounds = _orig_maxwgt(self, orig_lhe, evt_decayfile)
    finally:
        _OUT.write('P stop\n')
    _OUT.write('M %s\n' % ','.join(_fmt(b) for b in bounds))
    return bounds


MS.get_sequential_maxwgt = get_sequential_maxwgt


# --------------------------------------------------------------------------
def main():
    card = os.path.realpath(sys.argv[1])
    cmd_line = interface.MadSpinInterface()
    cmd_line.use_rawinput = False
    cmd_line.haspiping = False
    cmd_line.import_command_file(card)
    cmd_line.run_cmd('quit')


if __name__ == '__main__':
    try:
        main()
    finally:
        _OUT.flush()
        _OUT.close()
