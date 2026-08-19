#!/usr/bin/env python3
"""What the two pure-interference output paths do with a failed reshuffle.

    analyse_badjac.py <workdir> <out_dir>

Reads the fault-injection runs of ``run_badjac.sh`` and answers, per case:

* how many trials had their reshuffling jacobian forced to a non-positive
  value (from the instrumentation, which counts the injections it made);
* how many events the file holds, and how many carry ``w == 0`` exactly;
* what the banner's dead-trial counter says (``nb_pi_dead`` -- the counter that
  is supposed to mean "this trial is invalid");
* what the zero-cross-section check ``z`` says.

The two injections answer two different questions.  ``jac = 0`` gives ``W = 0``:
an invalid trial that happens to be indistinguishable from "contributes
nothing", and both output paths should absorb it harmlessly.  ``jac = -1`` --
RAMBO's own sentinel for an impossible mass set -- gives ``W = -|wgt|``, a
full-magnitude NEGATIVE weight.  If the code conflates that with a legitimate
interference sign, the injected trials enter the sample carrying a sign and
``z`` blows up.  The ``decay_output = weighted`` cases are the contrast: that
mode DOES treat ``signed <= 0`` as dead, so its dead counter must move.
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from analyse_pa_madspin import (read_lhe, read_pi_block, jac_counters,   # noqa: E402
                                observables, moment)

CASES = [
    ('ctrl_w',    'pure_interference', 'weighted',   0.0),
    ('ctrl_u',    'pure_interference', 'unweighted', 0.0),
    ('bad0_w',    'pure_interference', 'weighted',   0.0),
    ('bad0_u',    'pure_interference', 'unweighted', 0.0),
    ('badm1_w',   'pure_interference', 'weighted',  -1.0),
    ('badm1_u',   'pure_interference', 'unweighted', -1.0),
    ('bad0_dw',   'decay_output',      'weighted',   0.0),
    ('badm1_dw',  'decay_output',      'weighted',  -1.0),
]

# what each case is measured against
CONTROL = {'bad0_w': 'ctrl_w', 'badm1_w': 'ctrl_w',
           'bad0_u': 'ctrl_u', 'badm1_u': 'ctrl_u'}


def read_dead(header):
    """nb_pi_dead, from whichever banner block the mode wrote."""
    b = read_pi_block(header)
    if b.get('dead') is not None:
        return b['dead'], b
    import re
    m = re.search(r'Trials with a dead weight\s*:\s*(\d+)', header)
    return (int(m.group(1)) if m else None), b


def main():
    work = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for tag, mode, out, jacval in CASES:
        d = os.path.join(work, 'badjac_' + tag)
        path = os.path.join(d, 'events_decayed.lhe.gz')
        r = dict(tag=tag, mode=mode, output=out, forced_jac=jacval,
                 exists=os.path.exists(path))
        p = os.path.join(d, 'exit_code')
        r['exit_code'] = int(open(p).read().strip()) if os.path.exists(p) else None
        r['jac'] = jac_counters(d)
        if r['exists']:
            ev = read_lhe(path)
            w = ev['w']
            dead, banner = read_dead(ev['header'])
            n = len(w)
            sumw2 = float((w * w).sum())
            r.update(n_file=n, n_zero_weight=int((w == 0).sum()),
                     n_pos=int((w > 0).sum()), n_neg=int((w < 0).sum()),
                     dead=dead,
                     n_read=banner.get('n_read'),
                     S=float(w.sum()),
                     sqrt_sum_w2=math.sqrt(sumw2),
                     z=float(w.sum()) / math.sqrt(sumw2) if sumw2 else 0.0,
                     n_magnitudes=int(len(np.unique(np.round(
                         np.abs(w) / (np.abs(w).max() or 1.0), 9)))))
            if mode == 'pure_interference':
                # The z check cannot see a sign flip: W is already signed with
                # mean zero, so flipping a random subset of signs leaves S at
                # zero.  What a sign flip DOES move is any observable whose
                # sign correlates with the kinematics, i.e. the physics.  So
                # measure that instead.
                o = observables(ev)
                val, err = moment(w, o['cnn'], n, banner['ref'])
                r['cnn'], r['cnn_err'] = val, err
        res[tag] = r

    for tag, ctrl in CONTROL.items():
        if 'cnn' in res.get(tag, {}) and 'cnn' in res.get(ctrl, {}):
            a, b = res[tag], res[ctrl]
            res[tag]['cnn_ratio'] = a['cnn'] / b['cnn'] if b['cnn'] else None
            res[tag]['cnn_shift_sigma'] = (
                (a['cnn'] - b['cnn'])
                / math.sqrt(a['cnn_err'] ** 2 + b['cnn_err'] ** 2))

    with open(os.path.join(out_dir, 'badjac.json'), 'w') as f:
        json.dump(res, f, indent=1)

    print('%-9s %-17s %-11s %6s %8s %8s %8s %8s %8s %9s'
          % ('tag', 'mode', 'output', 'jac', 'forced', 'N_read', 'N_file',
             'w==0', 'dead', 'z'))
    for tag, mode, out, jacval in CASES:
        r = res[tag]
        if not r['exists']:
            print('%-9s %-17s %-11s %6.1f  MISSING (rc=%s)'
                  % (tag, mode, out, jacval, r['exit_code']))
            continue
        print('%-9s %-17s %-11s %6.1f %8d %8s %8d %8d %8s %+9.2f'
              % (tag, mode, out, jacval, r['jac']['n_forced'], r['n_read'],
                 r['n_file'], r['n_zero_weight'], r['dead'], r['z']))

    print('\n== the physics, which is where a conflated sign actually shows ==')
    print('%-9s %22s %9s %9s' % ('tag', '<C_nn> interference', 'vs ctrl',
                                 'sigma'))
    for tag, mode, out, jacval in CASES:
        r = res[tag]
        if 'cnn' not in r:
            continue
        print('%-9s %+12.6f +- %.6f %9s %9s'
              % (tag, r['cnn'], r['cnn_err'],
                 ('%.3f' % r['cnn_ratio']) if r.get('cnn_ratio') else '-',
                 ('%+.2f' % r['cnn_shift_sigma'])
                 if r.get('cnn_shift_sigma') is not None else '-'))
    print('\nwrote', os.path.join(out_dir, 'badjac.json'))


if __name__ == '__main__':
    main()
