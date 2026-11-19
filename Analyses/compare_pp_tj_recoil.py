#!/usr/bin/env python3
"""Overlay stable-top recoil analyses, retaining signed-weight errors."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from madgraph.various.histograms import HwUList
from Analyses.matched_run_inputs import CENTRAL_RUNS, find_input

CENTRAL = ROOT / 'central_runs_fixed'


def read_histograms(path):
    data = {}
    for hist in HwUList(str(path)):
        if hist.type == 'AUX':
            continue
        bounds = np.array([b.boundaries for b in hist.bins])
        edges = np.r_[bounds[:, 0], bounds[-1, 1]]
        values = np.array([b.wgts['central'] for b in hist.bins])
        errors = np.array([b.wgts['stat_error'] for b in hist.bins])
        data[hist.type, hist.title] = (edges, values, errors)
    return data


def subset_fraction(numerator, denominator, err_num, err_den):
    """N/D with Cov(N,D)=Var(N), including signed event weights."""
    fraction = np.full_like(numerator, np.nan, dtype=float)
    error = fraction.copy()
    good = denominator > 0
    fraction[good] = numerator[good] / denominator[good]
    variance = ((1 - fraction[good])**2 * err_num[good]**2
                + fraction[good]**2
                * np.maximum(err_den[good]**2 - err_num[good]**2, 0))
    error[good] = np.sqrt(variance) / denominator[good]
    return fraction, error


def independent_ratio(local, default, err_local, err_default):
    ratio = np.full_like(local, np.nan, dtype=float)
    error = ratio.copy()
    good = default > 0
    ratio[good] = local[good] / default[good]
    error[good] = np.hypot(err_local[good] / default[good],
                          local[good] * err_default[good] / default[good]**2)
    return ratio, error


def plot_pair(pyplot, pdf, key, curves, styles, shape=False, efficiency=False):
    selection, title = key
    fig, (main, ratios) = pyplot.subplots(
        2, 1, sharex=True, figsize=(7.4, 6),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})
    displayed = []
    for (_, label, colour), (edges, values, errors) in zip(styles, curves):
        values, errors = values.copy(), errors.copy()
        rate = len(values) == 1
        if shape and not rate and not efficiency:
            total = values.sum()
            if total <= 0:
                raise ValueError('Nonpositive shape normalization: %s %s' % key)
            values, errors = subset_fraction(
                values, np.full_like(values, total), errors,
                np.full_like(errors, np.linalg.norm(errors)))
        if not rate and not efficiency:
            values /= np.diff(edges)
            errors /= np.diff(edges)
        mid = (edges[1:] + edges[:-1]) / 2
        main.stairs(values, edges, label=label, color=colour, linewidth=1.4)
        main.errorbar(mid, values, yerr=errors, fmt='none', color=colour,
                      alpha=0.5, linewidth=0.7)
        displayed.append((values, errors))
    default, err_default = displayed[0]
    ratios.axhline(1, color='0.5', linewidth=0.8)
    for index, (values, errors) in enumerate(displayed[1:], start=1):
        ratio, error = independent_ratio(values, default, errors, err_default)
        ratios.errorbar(mid, ratio, yerr=error, fmt='.', color=styles[index][2],
                        markersize=3, linewidth=0.7)
    ratios.set_ylabel('ratio to default')
    ratios.set_xlabel('veto threshold Q0 [GeV]' if efficiency else title)
    main.set_title(selection + ': ' + title)
    main.set_ylabel('gap-veto efficiency' if efficiency else
                    'cross section [pb]' if rate else
                    'normalized density' if shape else
                    'cross section / bin width [pb / x unit]')
    main.legend(frameon=False, fontsize=9)
    main.axhline(0, color='0.7', linewidth=0.5)
    for axis in (main, ratios):
        axis.grid(alpha=0.18)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.12, top=0.93)
    pdf.savefig(fig)
    pyplot.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--central-dir', type=Path, default=CENTRAL,
                        help='corrected C_GLB/C_LOC/C_Q/C_QT exports')
    parser.add_argument('default', nargs='?', type=Path,
                        help='global-recoil export/HwU (default: C_GLB under --central-dir)')
    parser.add_argument('local', nargs='?', type=Path,
                        help='local-recoil export/HwU (default: C_LOC under --central-dir)')
    parser.add_argument('--patched', nargs='?', type=Path,
                        const=True,
                        help='also overlay C_Q, or an explicit export/HwU file')
    parser.add_argument('--two-patches', nargs='?', type=Path,
                        const=True,
                        help='also overlay both patched predictions, or an explicit export/HwU file')
    parser.add_argument('--powheg', nargs='?', type=Path,
                        const=ROOT / 'pp_tj_powheg',
                        help='also overlay local-recoil POWHEG, or an explicit HwU file')
    parser.add_argument('--run', help='event run name (default: infer C_/G_/P_/S_ name, otherwise run_01)')
    parser.add_argument('--out', type=Path, default=ROOT / 'Analyses/pp_tj_recoil')
    parser.add_argument('--shape', action='store_true',
                        help='normalize each distribution over its plotted range')
    args = parser.parse_args()
    args.default = args.default or args.central_dir / CENTRAL_RUNS['default']
    args.local = args.local or args.central_dir / CENTRAL_RUNS['local']
    if args.patched is True:
        args.patched = args.central_dir / CENTRAL_RUNS['local_patch']
    if args.two_patches is True:
        args.two_patches = args.central_dir / CENTRAL_RUNS['local_2patch']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as pyplot
    from matplotlib.backends.backend_pdf import PdfPages

    try:
        paths = [args.default, args.local]
        styles = [('default', 'default recoil', '#2463a4'),
                  ('local', 'local recoil', '#d35400')]
        if args.patched is not None or args.two_patches is not None:
            paths.append(args.patched if args.patched is not None
                         else args.central_dir / CENTRAL_RUNS['local_patch'])
            styles.append(('local_patch', 'local + collinear patch', '#228545'))
        if args.two_patches is not None:
            paths.append(args.two_patches)
            styles.append(('local_2patch', 'local + both patches', '#9553a4'))
        if args.powheg is not None:
            paths.append(args.powheg)
            styles.append(('powheg', 'POWHEG local recoil', '#9b7010'))
        inputs = [find_input(p, args.run) for p in paths]
        data = [read_histograms(p) for p in inputs]
        if any(len(d) != 75 for d in data):
            raise ValueError('Each shower analysis must contain 75 physical histograms')
        if any(data[0].keys() != d.keys() for d in data[1:]):
            raise ValueError('The analyses contain different histograms')
        for key in data[0]:
            if any(not np.array_equal(data[0][key][0], d[key][0]) for d in data[1:]):
                raise ValueError('Incompatible bins: %s %s' % key)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        output = str(args.out) + ('_shapes' if args.shape else '')
        with PdfPages(output + '.pdf') as pdf:
            for key in data[0]:
                if key[1].startswith('gap veto '):
                    continue  # Cumulative scans are turned into efficiencies below.
                plot_pair(pyplot, pdf, key, [d[key] for d in data], styles, args.shape)
            for selection in ('TAG', 'GAP'):
                curves = []
                for sample in data:
                    edges, num, err_num = sample[
                        selection, 'gap veto survived Q0 [GeV]']
                    _, den, err_den = sample[
                        selection, 'gap veto reference Q0 [GeV]']
                    values, errors = subset_fraction(num, den, err_num, err_den)
                    curves.append((edges, values, errors))
                plot_pair(pyplot, pdf, (selection, 'gap veto efficiency'), curves, styles,
                          efficiency=True)
        rates = {'inputs': [str(p.resolve()) for p in inputs], 'rates_pb': {}}
        for key, (_, value, _) in data[0].items():
            if len(value) == 1:
                rates['rates_pb'][' / '.join(key)] = {
                    name: {'value': float(d[key][1][0]),
                           'stat_error': float(d[key][2][0])}
                    for (name, _, _), d in zip(styles, data)}
        Path(output + '.json').write_text(json.dumps(rates, indent=2) + '\n')
        print('Wrote %s.pdf and %s.json' % (output, output))
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
