#!/usr/bin/env python3
"""Compare the source Delta of H events with the first-step shower's Sudakov.

MC@NLO-Delta multiplies each H event by Delta_src, a product over Born legs
from the S-event starting scales down to stopping scales reconstructed from
the H kinematics (compute_delta in montecarlocounter.f). The S events of the
same sample measure the actual first-step shower: with starting scale Q
(SCALUP) and first-emission scale t_S, the empirical no-emission probability
inside the shower range is

    Delta_PS(t) = P(t_S < t | Q > t),

and P(t_S < t) over all S events also counts those starting below t. Each H
event is paired with Delta_PS at its own scale. H events are generated
proportional to Delta_src |H_0|, so the comparison over the undamped H
measure weights them by |w|/Delta_src; it is heavy-tailed, and its largest
single-event share is reported. The as-generated |w| average is robust but
favours large Delta_src, so it overstates Delta_src relative to Delta_PS.
H events with Delta_src < 1 are compared with the conditional Delta_PS, all
H events with the unconditional one.

Scales are compared in two variables: Pythia's evolution pT (S: the first
emission; H: the smallest reconstructed dipole stopping scale) and the
hard-system pT binning of diagnose_macnlops_transfer.py. The H events' Delta
starting scales follow the same D(mu) law as the S events' SCALUP but are
not recorded, so this compares averages over that law. K, beta and the veto
do not enter. Nothing is generated, showered or rewritten.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'macnlops_transfer_diagnostic', ROOT / 'docs/audits/diagnose_macnlops_transfer.py')
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)
mc = transfer.mc

VARIABLES = (('evolution', 'Pythia evolution pT [GeV]'), ('system', 'hard-system pT [GeV]'))
# S columns: |w|, Q, t_evolution, t_system; H columns: |w|, Delta, t_evolution, t_system.
S_COLUMNS = dict(weight=0, start=1, evolution=2, system=3)
H_COLUMNS = dict(weight=0, delta=1, evolution=2, system=3)
GRID = np.geomspace(1e-2, 1e4, 1201)


def read(paths, process, window=None):
    """Per-event scales of all S and H proposals, optionally in a hard-system mass window."""
    s_rows, h_rows = [], []
    for job in transfer.jobs(paths):
        for _, event, record in transfer.job_records(job, 'shower.jsonl'):
            system = transfer.hard_system(event, process)
            if system is None:
                continue
            if window is not None:
                mass2 = system[3]**2 - system[0]**2 - system[1]**2 - system[2]**2
                if not window[0] <= math.sqrt(max(mass2, 0.)) < window[1]:
                    continue
            pt = transfer.prior.pt(system)
            if event.kind == 'S':
                s_rows.append((abs(event.weight), record['shower_scalup'],
                               record['pt_evol'] if record['emitted'] else 0., pt))
                continue
            if event.component is None:
                raise transfer.DiagnosticError('H events need sampled-component packets '
                                               'carrying their source Delta')
            stops = [scale for _, _, scale in record['source_dipole_scales'] if scale > 0]
            h_rows.append((abs(event.weight), event.component.source_factor,
                           min(stops) if stops else math.nan, pt))
    if not s_rows or not h_rows:
        raise transfer.DiagnosticError('Need both S and H proposals')
    return np.array(s_rows), np.array(h_rows)


def weight_below(values, weight, scales, side='left'):
    """Per scale, the summed weight of the values below it ('right': or equal to it)."""
    order = np.argsort(values)
    cumulative = np.concatenate(([0.], np.cumsum(weight[order])))
    return cumulative[np.searchsorted(values[order], scales, side=side)]


def shower_no_emission(s, variable):
    """Delta_PS on GRID: over all S events, and conditional on Q above the scale."""
    weight, start, scale = (s[:, S_COLUMNS[key]] for key in ('weight', 'start', variable))
    every = weight_below(scale, weight, GRID) / weight.sum()
    above = weight_below(-start, weight, -GRID)
    # scale < t < Q needs scale < Q, and then it is scale < t but not Q <= t.
    ordered = scale < start
    emitted = (weight_below(scale[ordered], weight[ordered], GRID) -
               weight_below(start[ordered], weight[ordered], GRID, 'right'))
    inside = np.full(len(GRID), np.nan)
    np.divide(emitted, above, out=inside, where=above > 0)
    return every, inside


def at(values, scales):
    """Interpolate a GRID function in log scale; NaN outside its defined range."""
    defined = np.isfinite(values)
    return np.interp(np.log(scales), np.log(GRID[defined]), values[defined],
                     left=np.nan, right=np.nan)


def compare(s, h, variable, edges):
    every, inside = shower_no_emission(s, variable)
    weight, delta, scale = (h[:, H_COLUMNS[key]] for key in ('weight', 'delta', variable))
    paired = dict(all=at(every, scale), inside=at(inside, scale))
    upper = list(edges[1:]) + [math.inf]
    rows = []
    for lo, hi in zip(edges, upper):
        row = dict(lo=lo, hi=None if math.isinf(hi) else hi)
        in_bin = (scale >= lo) & (scale < hi)
        row['H_events'] = int(in_bin.sum())
        row['delta_one_fraction'] = (float((delta[in_bin] == 1.).mean())
                                     if in_bin.any() else None)
        for name, select in (('all', in_bin), ('inside', in_bin & (delta < 1.))):
            select = select & np.isfinite(paired[name])
            if not select.any():
                row[name] = None
                continue
            w, d, p = weight[select], delta[select], paired[name][select]
            undamped = w / d
            row[name] = dict(
                events=int(select.sum()),
                undamped_source=float(w.sum() / undamped.sum()),
                undamped_shower=float((undamped * p).sum() / undamped.sum()),
                undamped_largest_share=float(undamped.max() / undamped.sum()),
                generated_source=float((w * d).sum() / w.sum()),
                generated_shower=float((w * p).sum() / w.sum()))
        rows.append(row)
    return rows


def table(result):
    lines = ['MC@NLO-Delta Sudakov check: %s, %d S and %d H proposals%s' % (
        result['process'], result['S_events'], result['H_events'],
        '' if result['mass_window'] is None else
        ', hard-system mass in [%g, %g) GeV' % tuple(result['mass_window']))]
    for variable, label in VARIABLES:
        lines += ['', label + ': Delta_src (H) versus Delta_PS from the S first emissions',
                  '%-10s %6s %6s | %-27s %6s | %-27s' % (
                      'scale', 'N_H', 'D=1', 'inside: undamped (gen.)', 'max', 'all H: undamped (gen.)'),
                  '%-10s %6s %6s | %13s %13s %6s | %13s %13s' % (
                      '', '', '', 'src', 'shower', 'share', 'src', 'shower')]
        for row in result[variable]:
            cells = []
            for name in ('inside', 'all'):
                values = row[name]
                if values is None:
                    cells.append('%13s %13s' % ('n/a', 'n/a') + ('%7s' % 'n/a' if name == 'inside' else ''))
                    continue
                cells.append('%6.3f (%5.3f) %6.3f (%5.3f)' % (
                    values['undamped_source'], values['generated_source'],
                    values['undamped_shower'], values['generated_shower']) +
                    ('%7.2f' % values['undamped_largest_share'] if name == 'inside' else ''))
            fraction = row['delta_one_fraction']
            lines.append('%-10s %6d %6s | %s | %s' % (
                transfer.bin_label(row), row['H_events'], 'n/a' if fraction is None else '%.2f' % fraction,
                cells[0], cells[1]))
    return '\n'.join(lines)


def check(paths, process, edges, window=None):
    transfer.check_edges(edges)
    s, h = read(paths, process, window)
    result = dict(process=process, mass_window=window, S_events=len(s), H_events=len(h),
                  edges=list(edges))
    for variable, _ in VARIABLES:
        result[variable] = compare(s, h, variable, edges)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', type=Path,
                        help='collected macnlops output directories or single matching jobs')
    parser.add_argument('--process', required=True, choices=transfer.PROCESSES)
    parser.add_argument('--edges', type=transfer.float_list,
                        help='comma-separated scale edges in GeV; an overflow bin is added')
    parser.add_argument('--mass-window', type=transfer.float_list,
                        help='LO,HI hard-system mass window in GeV for both S and H')
    parser.add_argument('--json', type=Path, help='write all averages')
    args = parser.parse_args(argv)
    result = check(args.paths, args.process, args.edges or transfer.default_edges(args.process),
                   args.mass_window)
    print(table(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (transfer.DiagnosticError, mc.MatchingError, transfer.prior.ValidationError,
            OSError) as error:
        print('MC@NLO-Delta Sudakov check: %s' % error, file=sys.stderr)
        sys.exit(1)
