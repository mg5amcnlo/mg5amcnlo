#!/usr/bin/env python3
"""Turn the results.json of measure_unweighting.py into the LaTeX table.

    python3 doc/madspin_unweighting_efficiency/make_table.py results.json > table.tex

Column definitions -- all four come straight from counters MadSpin prints at
the end of the run, none is re-derived here:

  eps_m     "MadSpin sequential mass stage: X mass sets per accepted event"
  eps_t     "MadSpin sequential slot P: X decay events per accepted one", for
  eps_tbar  the position whose slot holds t / t~ (resolved from the LHE layout)
  N_dec     staged schemes: sum over positions of the "(N drawn)" counts on
            those same slot lines.  joint: (number of decaying particles) x the
            trial count of "MadSpin unweight efficiency: ... (W written /
            T trials, R trials/event)", because one joint trial draws exactly
            one decay per decaying particle.

eps is always "generated points per accepted point": lower is better, floor 1.
"""

from __future__ import division

import argparse
import collections
import json
import math
import sys


ROW_ORDER = [
    ('PA', 'joint'),
    ('PA', 'sequential'),
    ('PA', 'sequential_global_retry'),
    ('madspin', 'joint'),
    ('madspin', 'sequential'),
    ('madspin', 'sequential_global_retry'),
    ('onshell', 'joint'),
    ('onshell', 'sequential'),
]

SCHEME_TEX = {
    'joint': r'\texttt{joint}',
    'sequential': r'\texttt{sequential}',
    'sequential\_global\_retry': r'\texttt{sequential\_global\_retry}',
}


def eps_error(eps, n_written):
    """Indicative 1-sigma on a trials-per-acceptance ratio: the per-event trial
    count of a redraw-until-accepted loop is geometric with mean eps, hence
    variance eps(eps-1), so the mean over n_written events has sd
    sqrt(eps(eps-1)/n_written)."""
    if not n_written or eps is None or eps <= 1.0:
        return 0.0
    return math.sqrt(eps * (eps - 1.0) / n_written)


def row_numbers(run, position_to_pdg):
    """(eps_m, eps_t, eps_tbar, N_dec, joint_eps) for one run.

    ``joint_eps`` is not None only for the joint scheme, where there are no
    stages and the single accept/reject spans the three eps columns.
    """
    mode = run.get('reported_mode')
    n_written = run.get('n_written')
    if mode == 'joint':
        n_dec = run['n_trials'] * len(position_to_pdg)
        return None, None, None, n_dec, run['trials_per_event']

    slots = {int(k): v for k, v in run['slots'].items()}
    pdg_to_eps = {}
    n_dec = 0
    for position, info in slots.items():
        pdg = position_to_pdg[str(position)]
        pdg_to_eps[pdg] = info['per_accepted']
        n_dec += info['drawn']

    mass = run.get('mass_stage')
    eps_m = mass['per_accepted'] if mass else None
    return eps_m, pdg_to_eps.get(6), pdg_to_eps.get(-6), n_dec, None


def fmt(value, digits=3):
    if value is None:
        return r'--'
    return ('%.' + str(digits) + 'f') % value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('--label', default='tab:madspin-unweighting-eps')
    args = parser.parse_args()

    with open(args.results) as fp:
        data = json.load(fp)
    position_to_pdg = data['position_to_pdg']
    runs = data['runs']

    nev = None
    lines = []
    errors = []
    for spinmode, unw in ROW_ORDER:
        key = '%s_%s' % (spinmode, unw)
        run = runs.get(key)
        if run is None:
            continue
        assert run['reported_mode'] == unw, (
            '%s: asked for %s, ran %s' % (key, unw, run['reported_mode']))
        assert run['reported_why'] == 'set explicitly', (
            '%s: scheme was not set explicitly (%s)'
            % (key, run['reported_why']))
        nev = run['n_written'] if nev is None else nev
        eps_m, eps_t, eps_tb, n_dec, joint = row_numbers(run, position_to_pdg)
        spin_tex = r'\texttt{%s}' % spinmode
        unw_tex = r'\texttt{%s}' % unw.replace('_', r'\_')
        if joint is not None:
            cells = r'\multicolumn{3}{c}{%s}' % fmt(joint)
            errors.append((key, 'joint', joint, eps_error(joint, run['n_written'])))
        else:
            cells = ' & '.join([fmt(eps_m), fmt(eps_t), fmt(eps_tb)])
            for name, value in (('eps_m', eps_m), ('eps_t', eps_t),
                                ('eps_tbar', eps_tb)):
                if value is not None:
                    errors.append((key, name, value,
                                   eps_error(value, run['n_written'])))
        lines.append('%s & %s & %s & %s \\\\' % (spin_tex, unw_tex, cells,
                                                 '{:,}'.format(n_dec)
                                                 .replace(',', r'\,')))

    out = []
    out.append(r'\begin{table}[htbp]')
    out.append(r'  \centering')
    out.append(r'  \begin{tabular}{llrrrr}')
    out.append(r'    \toprule')
    out.append(r'    spinmode & unweighting & $\epsilon_m$ & $\epsilon_t$ '
               r'& $\epsilon_{\bar t}$ & $N_{\mathrm{dec}}$ \\')
    out.append(r'    \midrule')
    previous = None
    for (spinmode, unw), line in zip(
            [g for g in ROW_ORDER if '%s_%s' % g in runs], lines):
        if previous is not None and spinmode != previous:
            out.append(r'    \midrule')
        previous = spinmode
        out.append('    ' + line)
    out.append(r'    \bottomrule')
    out.append(r'  \end{tabular}')
    out.append(r'  \caption{MadSpin unweighting cost per accept/reject stage '
               r'for $p\,p \to t\bar t$ with both tops decayed '
               r'($t \to b\,W^+$, $W^+ \to \ell^+\nu$ and charge conjugate), '
               r'one production sample of %s events and one MadSpin seed '
               r'shared by every row. Each $\epsilon$ is the average number of '
               r'generated points needed to accept one, so \emph{lower is '
               r'better and the floor is~1}: $\epsilon_m$ counts virtuality '
               r'(mass) sets drawn per written event, $\epsilon_t$ and '
               r'$\epsilon_{\bar t}$ count decay events drawn for the $t$ and '
               r'$\bar t$ slot per written event, and $N_{\mathrm{dec}}$ is the '
               r'total number of decay events consumed over the whole run. '
               r'\texttt{joint} has no separate phases -- it makes a single '
               r'accept/reject over the virtualities and both decays at once -- '
               r'so it yields one efficiency, spanning the three $\epsilon$ '
               r'columns. \texttt{onshell} keeps the production kinematics and '
               r'never samples a virtuality, so it has no mass stage at all and '
               r'$\epsilon_m$ is not defined there (dash, not~1).}'
               % '{:,}'.format(nev or 0).replace(',', r'\,'))
    out.append(r'  \label{%s}' % args.label)
    out.append(r'\end{table}')
    print('\n'.join(out))

    sys.stderr.write('\n%-42s %-9s %8s %8s\n'
                     % ('run', 'column', 'value', '1sigma'))
    for key, name, value, err in errors:
        sys.stderr.write('%-42s %-9s %8.4f %8.4f\n' % (key, name, value, err))


if __name__ == '__main__':
    main()
