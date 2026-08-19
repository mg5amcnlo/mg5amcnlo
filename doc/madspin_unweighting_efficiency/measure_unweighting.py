#!/usr/bin/env python3
"""Measure the per-stage MadSpin unweighting cost of every (spinmode,
unweighting) combination on one production sample.

Process: ``p p > t t~``, both tops decayed (fully leptonic chain), ONE madevent
production sample and ONE MadSpin seed shared by every configuration, so the
columns differ only by the accept/reject scheme.

For each configuration we report, straight off the counters MadSpin itself
prints at the end of a run:

  eps_m     "MadSpin sequential mass stage: X mass sets per accepted event"
            -- virtuality sets drawn per written event.
  eps_t     "MadSpin sequential slot P: X decay events per accepted one"
  eps_tbar  ... same line, the other position. Which position is which particle
            is resolved from the slot layout, not assumed (see slot_map()).
  N_dec     total decay events drawn over the whole run
            = sum over positions of the "(T drawn)" counts for the staged
              schemes; = (number of decaying particles) x trials for ``joint``,
              since one joint trial draws one decay for every decaying particle
              (interface_madspin.get_decay_from_file / _draw_all_decays).

  joint has no stages: it makes one accept/reject over the virtualities and
  both decays at once, so it yields the single number of
  "MadSpin unweight efficiency: ... (W written / T trials, R trials/event)".

Every eps here is "generated points per accepted point": LOWER IS BETTER, the
floor is 1.

Usage:
    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 doc/madspin_unweighting_efficiency/measure_unweighting.py \
        --nevents 10000 --outdir /some/scratch/dir

Writes results.json (raw counters), table.tex (the LaTeX table) and copies the
per-run madspin.log files into <outdir>/logs/.
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import collections
import json
import math
import os
import re
import shutil
import sys
import time

pjoin = os.path.join

_here = os.path.dirname(os.path.realpath(__file__))
_root = os.path.split(os.path.split(_here)[0])[0]
if _root not in sys.path:
    sys.path.insert(0, _root)

from tests.parallel_tests.madspin_comparator import (  # noqa: E402
    MadSpinFactory, SpinModeConfig, _parse_unweighting)
import madgraph.various.lhe_parser as lhe_parser  # noqa: E402


# --------------------------------------------------------------------------
# The grid.  sequential_global_retry is deliberately absent for ``onshell``;
# see the module docstring of the report and the check_onshell_retry() note
# below -- onshell draws no virtuality, so the Z_hat factor the scheme exists
# to cancel is identically 1 there and the scheme degenerates into ``sequential``
# plus wasted work.
# --------------------------------------------------------------------------
GRID = [
    ('PA', 'joint'),
    ('PA', 'sequential'),
    ('PA', 'sequential_global_retry'),
    ('madspin', 'joint'),
    ('madspin', 'sequential'),
    ('madspin', 'sequential_global_retry'),
    ('onshell', 'joint'),
    ('onshell', 'sequential'),
]

# Same physics setup as tests/parallel_tests/test_madspin_factory.TTBAR_LEPTONIC,
# so these numbers are directly comparable with the campaign the ``unweighting``
# card comment quotes.
TTBAR_LEPTONIC = dict(
    production_process='p p > t t~',
    decays=['t > b w+, w+ > l+ vl',
            't~ > b~ w-, w- > l- vl~'],
    multiparticles={'p': 'g u d s c u~ d~ s~ c~',
                    'l+': 'e+ mu+', 'vl': 've vm',
                    'l-': 'e- mu-', 'vl~': 've~ vm~'},
    extra_run_card={'ebeam1': 6500, 'ebeam2': 6500},
)


# --------------------------------------------------------------------------
# Log parsing.  Every number below is one MadSpin already computes and prints;
# nothing here re-derives an efficiency from the event file.
# --------------------------------------------------------------------------
_RE_EFF = re.compile(
    r'MadSpin unweight efficiency:\s*([0-9.eE+\-]+)\s*'
    r'\((\d+)\s*written\s*/\s*(\d+)\s*trials,\s*([0-9.eE+\-]+)\s*trials/event\)')
_RE_MASS = re.compile(
    r'MadSpin sequential mass stage:\s*([0-9.eE+\-]+)\s*mass sets per accepted '
    r'event\s*\((\d+)\s*drawn,\s*(\d+)\s*rejected(?:,\s*(\d+)\s*dropped by a '
    r'rejected decay)?\)')
_RE_SLOT = re.compile(
    r'MadSpin sequential slot (\d+):\s*([0-9.eE+\-]+)\s*decay events per '
    r'accepted one\s*\((\d+)\s*drawn\)')
_RE_ANGLE = re.compile(
    r'MadSpin sequential angle stage:\s*([0-9.eE+\-]+)\s*angle sets per '
    r'accepted event\s*\((\d+)\s*drawn,\s*(\d+)\s*rejected\)')
_RE_RESTART = re.compile(
    r'MadSpin sequential:\s*(\d+)\s*chains restarted on a mass set')
_RE_OVERFLOW = re.compile(
    r'MadSpin sequential:\s*(\d+)\s*weights exceeded their per-particle maximum')


def _flat(text):
    return re.sub(r'[ \t]+', ' ', text)


def parse_run(log_text):
    """Every counter this measurement needs, out of one MadSpin log."""
    flat = _flat(log_text)
    out = {}

    mode, why = _parse_unweighting(log_text)
    out['reported_mode'] = mode
    out['reported_why'] = why

    match = None
    for match in _RE_EFF.finditer(flat):
        pass
    if match:
        out['eff'] = float(match.group(1))
        out['n_written'] = int(match.group(2))
        out['n_trials'] = int(match.group(3))
        out['trials_per_event'] = float(match.group(4))

    match = None
    for match in _RE_MASS.finditer(flat):
        pass
    if match:
        out['mass_stage'] = {
            'per_accepted': float(match.group(1)),
            'drawn': int(match.group(2)),
            'rejected': int(match.group(3)),
            'dropped_by_rejected_decay': int(match.group(4) or 0),
        }

    slots = {}
    for match in _RE_SLOT.finditer(flat):
        slots[int(match.group(1))] = {
            'per_accepted': float(match.group(2)),
            'drawn': int(match.group(3)),
        }
    out['slots'] = slots

    match = None
    for match in _RE_ANGLE.finditer(flat):
        pass
    if match:
        out['angle_stage'] = {'per_accepted': float(match.group(1)),
                              'drawn': int(match.group(2)),
                              'rejected': int(match.group(3))}

    found = _RE_RESTART.search(flat)
    out['chain_restarts'] = int(found.group(1)) if found else 0
    found = _RE_OVERFLOW.search(flat)
    out['overflows'] = int(found.group(1)) if found else 0
    return out


# --------------------------------------------------------------------------
# Slot -> particle mapping.  Never assume the card order: MadSpin lays the
# density-matrix slots out by _decaying_pdgs (pdgs in order of first appearance
# among the production final state) and then orders the accept/reject by
# _decay_slot_order (sequential_spin_order, ties by slot index).  Both tops are
# spin-1/2 here, so the spin order cannot separate them and the answer is
# entirely in the production event layout -- which we read off the LHE.
# --------------------------------------------------------------------------
def slot_map(events_file, spin_order='2 3 1', nmax=2000):
    """Return (position -> pdg, diagnostics) for the production sample."""
    layouts = collections.Counter()
    lhe = lhe_parser.EventFile(events_file)
    for i, event in enumerate(lhe):
        if i >= nmax:
            break
        finals = [p.pdg for p in event if int(p.status) == 1]
        layouts[tuple(p for p in finals if abs(p) == 6)] += 1
    try:
        lhe.close()
    except Exception:
        pass

    # _decaying_pdgs: pdgs in order of first appearance among the final state
    # -> slot k.  _decay_slot_order: both spin 2 (2S+1) so the spin ranks tie
    # and the tie-break is the slot index, i.e. position k == slot k.
    layout = layouts.most_common(1)[0][0] if layouts else ()
    position_to_pdg = {k: pdg for k, pdg in enumerate(layout)}
    return position_to_pdg, {
        'layouts': {'/'.join(str(x) for x in k): v
                    for k, v in layouts.items()},
        'sequential_spin_order': spin_order,
        'note': ('both decaying particles are spin-1/2, so sequential_spin_order '
                 'cannot reorder them: position k == slot k == the k-th distinct '
                 'decaying pdg in production-event order'),
    }


# --------------------------------------------------------------------------
def eps_error(eps, n_written):
    """Rough 1-sigma on a trials-per-acceptance ratio.

    The per-event trial count of a redraw-until-accepted loop is geometric with
    mean eps, hence variance eps*(eps-1); the mean over n_written independent
    events then has sd sqrt(eps*(eps-1)/n_written).  For a stage that is not a
    plain geometric (a restart scheme couples the slots) this is indicative
    only, which is why it is reported separately from the table.
    """
    if not n_written or eps is None or eps <= 1:
        return 0.0
    return math.sqrt(eps * (eps - 1.0) / n_written)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nevents', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--basedir', default=None,
                        help='factory working tree (kept, so it can be reused)')
    parser.add_argument('--nb-core', type=int, default=1)
    parser.add_argument('--only', default=None,
                        help='comma separated spinmode:unweighting to restrict to')
    parser.add_argument('--extra', default=None,
                        help='extra spinmode:unweighting runs appended to the grid')
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(pjoin(outdir, 'logs'), exist_ok=True)

    grid = list(GRID)
    if args.extra:
        for item in args.extra.split(','):
            spinmode, unw = item.split(':')
            grid.append((spinmode, unw))
    if args.only:
        keep = set(args.only.split(','))
        grid = [g for g in grid if '%s:%s' % g in keep]

    factory = MadSpinFactory(
        name='unweighting_grid',
        nevents=args.nevents,
        seed=args.seed,
        base_dir=args.basedir,
        extra_madspin_settings={'nb_core': args.nb_core},
        **TTBAR_LEPTONIC)

    t0 = time.time()
    events_file = factory.produce_events()
    production_wall = time.time() - t0

    position_to_pdg, slot_diag = slot_map(events_file)

    results = collections.OrderedDict()
    for spinmode, unw in grid:
        label = '%s_%s' % (spinmode, unw)
        config = SpinModeConfig(label, spinmode)
        print('=== running %s' % label, flush=True)
        start = time.time()
        res = factory.run_mode(config, extra_settings={'unweighting': unw})
        wall = time.time() - start
        with open(res.log_path) as fp:
            log_text = fp.read()
        shutil.copy(res.log_path, pjoin(outdir, 'logs', '%s.log' % label))
        parsed = parse_run(log_text)
        parsed['spinmode'] = spinmode
        parsed['unweighting_asked'] = unw
        parsed['wall_seconds'] = wall
        parsed['BR'] = res.BR
        parsed['cross_out'] = res.cross_out
        results[label] = parsed
        print('    -> mode=%s (%s) written=%s slots=%s [%.0f s]'
              % (parsed.get('reported_mode'), parsed.get('reported_why'),
                 parsed.get('n_written'), parsed.get('slots'), wall), flush=True)

    payload = {
        'process': 'p p > t t~',
        'decays': TTBAR_LEPTONIC['decays'],
        'nevents_requested': args.nevents,
        'seed': args.seed,
        'nb_core': args.nb_core,
        'events_file': events_file,
        'cross_in': getattr(factory, 'cross_in', None),
        'production_wall_seconds': production_wall,
        'position_to_pdg': {str(k): v for k, v in position_to_pdg.items()},
        'slot_diagnostics': slot_diag,
        'runs': results,
    }
    with open(pjoin(outdir, 'results.json'), 'w') as fp:
        json.dump(payload, fp, indent=2, sort_keys=False)
    print('wrote %s' % pjoin(outdir, 'results.json'))
    print('base_dir = %s' % factory.base_dir)


if __name__ == '__main__':
    main()
