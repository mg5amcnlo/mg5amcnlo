#!/usr/bin/env python3
"""Plot the retained WZ LHE source shower scales for every saved variant."""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'docs/audits/fxfx_delta_wz_backward_isr'
ORDER = ('historical_2', 'historical_1', 'reference', 'new')
STYLE = {
    'historical_2': ('Historical hard scalar', '#777777', '--'),
    'historical_1': ('Historical geometric mean', '#d89000', ':'),
    'reference': ('Fresh geometric mean, ISR off', '#202124', '-'),
    'new': ('Fresh geometric mean, ISR on', '#6f42a1', '-'),
}
CATEGORIES = ('SUM', '0', '1')


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def source_records(path):
    """Read event headers and their explicit S/H label, without particle copies."""
    scales, samples, kinds, weights = [], [], [], []
    in_event = need_header = False
    kind = None
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            text = line.strip()
            if text == '<event>' or text.startswith('<event '):
                if in_event:
                    raise ValueError(f'Nested event in {path}')
                in_event, need_header, kind = True, True, None
            elif in_event and text:
                if need_header:
                    fields = text.split()
                    sample = int(fields[1])
                    weight, scale = float(fields[2]), float(fields[3])
                    if sample not in (0, 1) or not np.isfinite((weight, scale)).all() or scale <= 0:
                        raise ValueError(f'Invalid source event header: {text}')
                    need_header = False
                elif text.startswith('#aMCatNLO'):
                    kind = int(text.split()[1])
                elif text == '</event>':
                    if kind not in (1, 2):
                        raise ValueError(f'Missing S/H label in {path}')
                    scales.append(scale)
                    samples.append(sample)
                    kinds.append(kind)
                    weights.append(weight)
                    in_event = False
    if in_event:
        raise ValueError(f'Truncated source event in {path}')
    return (np.asarray(scales), np.asarray(samples, dtype=np.int8),
            np.asarray(kinds, dtype=np.int8), np.asarray(weights))


def common_edges(comparison):
    result = {}
    for kind in ('S', 'H'):
        values = [summary for key in ORDER
                  for name, summary in comparison['runs'][key]['lhe']['scalup_GeV'].items()
                  if name.endswith(':' + kind)]
        low = int(np.floor(np.log10(min(v['min'] for v in values))))
        high = int(np.ceil(np.log10(max(v['max'] for v in values))))
        result[kind] = np.logspace(low, high, 10 * (high - low) + 1)
    return result


def bin_records(comparison, edges):
    results, provenance = {}, {}
    for key in ORDER:
        run = comparison['runs'][key]
        path = Path(run['lhe']['file'])
        print(f'Reading {STYLE[key][0]}: {path}', flush=True)
        scales, samples, kinds, weights = source_records(path)
        count = len(scales)
        if count != run['lhe']['events']:
            raise ValueError(f'{key}: event count differs from saved audit')
        if count < 2:
            raise ValueError(f'{key}: at least two events are needed for a sampling error')
        if np.count_nonzero(weights < 0) != round(count * run['lhe']['negative_fraction']):
            raise ValueError(f'{key}: event signs differ from saved audit')
        digest = sha256(path)
        if digest != run['lhe']['sha256']:
            raise ValueError(f'{key}: LHE hash differs from saved audit')
        results[key] = {}
        for code, kind in ((1, 'S'), (2, 'H')):
            results[key][kind] = {}
            for category in CATEGORIES:
                mask = kinds == code
                if category != 'SUM':
                    mask &= samples == int(category)
                current = scales[mask]
                w = weights[mask]
                if np.any(current < edges[kind][0]) or np.any(current > edges[kind][-1]):
                    raise ValueError(f'{key}: source scale outside the plotted support')
                if category != 'SUM':
                    expected = run['lhe']['sample_type_signs'][category + ':' + kind]
                    if len(w) != expected['events']:
                        raise ValueError(f'{key}: source category count differs')
                    if not np.isclose(w.sum(), expected['signed_weight_sum'], rtol=1.e-12, atol=1.e-7):
                        raise ValueError(f'{key}: source category weight differs')
                nbins = len(edges[kind]) - 1
                indices = np.minimum(np.searchsorted(edges[kind], current, side='right') - 1,
                                     nbins - 1)
                if len(w) and np.all(np.abs(w) == abs(w[0])):
                    # Integer signed counts avoid spurious residuals in
                    # cancelled bins of these uniform-magnitude samples.
                    positive = np.bincount(indices[w > 0], minlength=nbins)
                    negative = np.bincount(indices[w < 0], minlength=nbins)
                    signed_sum = (positive - negative) * abs(w[0])
                    square_sum = (positive + negative) * abs(w[0])**2
                else:
                    signed_sum = np.bincount(indices, weights=w, minlength=nbins)
                    square_sum = np.bincount(indices, weights=w*w, minlength=nbins)
                central = signed_sum / count
                error = np.sqrt(np.maximum(square_sum - signed_sum**2/count, 0) /
                                (count * (count-1)))
                if not np.isclose(central.sum(), w.sum()/count, rtol=1.e-11, atol=1.e-10):
                    raise ValueError(f'{key}: source bin integral failed')
                results[key][kind][category] = {
                    'central': central, 'error': error, 'events': len(w),
                    'integral_pb': float(w.sum() / count),
                }
            values = results[key][kind]
            if not np.allclose(values['SUM']['central'],
                               values['0']['central'] + values['1']['central'],
                               rtol=1.e-10, atol=1.e-10):
                raise ValueError(f'{key}: combined source bins do not equal @0 + @1')
        signed_mean = sum(results[key][kind]['SUM']['integral_pb'] for kind in ('S', 'H'))
        if not np.isclose(signed_mean, run['lhe']['signed_weight_mean_pb'], rtol=1.e-12):
            raise ValueError(f'{key}: source integral differs from the saved mean LHE weight')
        provenance[key] = {'label': STYLE[key][0], 'file': str(path), 'sha256': digest,
                           'events': count, 'source_label': run['label']}
        del scales, samples, kinds, weights
    return results, provenance


def error_bars(axis, centers, values, errors, color, mask=None):
    if mask is None:
        mask = np.ones(len(values), dtype=bool)
    axis.errorbar(centers[mask], values[mask], yerr=errors[mask],
                  fmt='none', color=color, alpha=.42, linewidth=.65)


def ratio(axis, centers, numerator, denominator, color, linestyle, label):
    nv, ne = numerator['central'], numerator['error']
    dv, de = denominator['central'], denominator['error']
    valid = (dv > 0) & (dv > 3 * de)
    values = np.divide(nv, dv, out=np.full_like(nv, np.nan), where=valid)
    axis.plot(centers, values, color=color, linestyle=linestyle,
              drawstyle='steps-mid', label=label)
    error_bars(axis, centers, values, np.divide(ne, dv, out=np.zeros_like(ne), where=valid),
               color, valid)
    return valid


def plot(results, edges, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    def signed_power(value, position):
        if value == 0:
            return '0'
        exponent = int(round(np.log10(abs(value))))
        sign = '-' if value < 0 else ''
        return f'${sign}10^{{{exponent}}}$'

    with PdfPages(output / 'source_scales.pdf') as pdf:
        for kind in ('S', 'H'):
            bins = edges[kind]
            centers = np.sqrt(bins[:-1] * bins[1:])
            fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex='col',
                                     gridspec_kw={'height_ratios': [3, 1.15, 1.15]})
            for column, category in enumerate(CATEGORIES):
                top, common, paired = axes[:, column]
                all_values = []
                for key in ORDER:
                    item = results[key][kind][category]
                    label, color, linestyle = STYLE[key]
                    top.stairs(item['central'], bins, color=color, linestyle=linestyle,
                               linewidth=1.15, label=label)
                    nonzero = (item['central'] != 0) | (item['error'] != 0)
                    error_bars(top, centers, item['central'], item['error'], color, nonzero)
                    all_values.extend(item['central'])
                absolute = np.abs(np.asarray(all_values))
                positive = absolute[absolute > 0]
                if np.any(np.asarray(all_values) < 0):
                    threshold = max(float(np.quantile(positive, .1)), 1.e-6)
                    top.set_yscale('symlog', linthresh=threshold)
                    # Suppress crowded logarithmic ticks at the linear join.
                    low, high = top.get_ylim()
                    minimum_power = int(np.ceil(np.log10(10 * threshold)))
                    maximum_power = int(np.ceil(np.log10(max(abs(low), abs(high)))))
                    ticks = [0.] + [sign * 10.**power
                                    for power in range(minimum_power, maximum_power + 1)
                                    for sign in (-1, 1)
                                    if low <= sign * 10.**power <= high]
                    top.yaxis.set_major_locator(FixedLocator(sorted(ticks)))
                    top.yaxis.set_major_formatter(FuncFormatter(signed_power))
                    top.yaxis.set_minor_formatter(NullFormatter())
                else:
                    top.set_yscale('log')
                top.set_title('Combined source' if category == 'SUM' else f'Source @{category}')
                top.set_ylabel('Signed cross section per bin [pb]')
                top.legend(fontsize=7, loc='best')
                reference = results['reference'][kind][category]
                dv, de = reference['central'], reference['error']
                valid = (dv > 0) & (dv > 3 * de)
                relative_error = np.divide(de, dv, out=np.full_like(de, np.nan), where=valid)
                common.fill_between(centers, 1-relative_error, 1+relative_error,
                                    step='mid', color='#202124', alpha=.14)
                for key in ORDER:
                    if key != 'reference':
                        label, color, linestyle = STYLE[key]
                        ratio(common, centers, results[key][kind][category], reference,
                              color, linestyle, label)
                common.axhline(1, color='#202124', linewidth=.7)
                common.set_ylabel('Variant / fresh OFF')
                for a, b, color, linestyle, label in (
                        ('historical_1', 'historical_2', '#d89000', ':', 'Historical geom / hard'),
                        ('new', 'reference', '#6f42a1', '-', 'Fresh ON / OFF')):
                    ratio(paired, centers, results[a][kind][category], results[b][kind][category],
                          color, linestyle, label)
                paired.axhline(1, color='#202124', linewidth=.7)
                paired.set_ylabel('Paired ratios')
                paired.legend(fontsize=7, loc='best')
                paired.set_xlabel(r'LHE SCALUP $Q$ [GeV]')
                for axis in (top, common, paired):
                    axis.set_xscale('log')
                    axis.set_xlim(bins[0], bins[-1])
                    axis.grid(alpha=.18)
            fig.suptitle(f'WZ+0,1j source shower scales: {kind} events before showering', fontsize=14)
            fig.text(.5, .047,
                     'All bins retained; signed LHE weights / complete source event count; no area normalization. '
                     'Ratio points require a positive denominator above 3 sampling errors.',
                     ha='center', fontsize=8)
            fig.text(.5, .029,
                     'Bars: numerator fixed-N sampling error / denominator; common-ratio band: fresh OFF. '
                     'Source-normalization uncertainty and cross-run covariance are not included.',
                     ha='center', fontsize=8)
            fig.text(.5, .011,
                     'Spectrum axes use symlog where signed bins require it. '
                     'Historical and fresh source epochs differ; these scales are measured before showering.',
                     ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .07, 1, .96))
            fig.savefig(output / ('scalup_' + kind.lower() + '.png'), dpi=160)
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=AUDIT / 'comparison.json')
    parser.add_argument('--out', type=Path, default=AUDIT / 'plots/source_scales')
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    edges = common_edges(comparison)
    results, provenance = bin_records(comparison, edges)
    with (args.out / 'source_scales.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['variant', 'event_type', 'sample', 'q_min_GeV', 'q_max_GeV',
                         'signed_pb', 'conditional_error_pb'])
        for key in ORDER:
            for kind in ('S', 'H'):
                for category in CATEGORIES:
                    values = results[key][kind][category]
                    for low, high, central, error in zip(edges[kind][:-1], edges[kind][1:],
                                                         values['central'], values['error']):
                        writer.writerow([key, kind, category, low, high, central, error])
    plot(results, edges, args.out)
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'comparison_file': str(args.comparison.resolve()),
               'comparison_sha256': sha256(args.comparison), 'inputs': provenance,
               'plot_script_sha256': sha256(Path(__file__)),
               'normalization': 'Signed source LHE weights / complete source event count; pb per bin. No unit-area scaling.',
               'uncertainty': 'sqrt(max(sum(w**2) - sum(w)**2/N, 0)/(N*(N-1))) for fixed-N conditional source sampling; N is the complete source count. No source integration uncertainty or cross-run covariance.',
               'ratio_policy': 'Positive denominator above three conditional sampling errors; other ratios omitted.',
               'source_epoch_note': 'Historical source epochs also differ from fresh integration and numerical fixes. Historical shower steering omitted the corrected charged-resonance QED-off setting; QED does not enter the pre-shower SCALUP histograms.',
               'logarithmic_edges_GeV': {kind: values.tolist() for kind, values in edges.items()},
               'integrals_pb': {key: {kind: {category: values['integral_pb']
                                          for category, values in categories.items()}
                                     for kind, categories in items.items()}
                                for key, items in results.items()}}
    summary['outputs'] = {name: sha256(args.out/name) for name in
                          ('source_scales.pdf', 'scalup_s.png', 'scalup_h.png', 'source_scales.csv')}
    (args.out / 'source_scales.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.out / 'README.md').write_text('''# Source shower-scale comparisons

[Two-page PDF](source_scales.pdf): S- and H-event LHE SCALUP, before showering,
with combined, @0 and @1 sources. Every retained event enters the plotted
logarithmic support. The four source epochs are overlaid, with a common fresh
backward-ISR-off denominator and the historical geometric-mean/hard-scale and
fresh backward-ISR-on/off paired ratios.

These are absolute signed source cross sections per logarithmic bin, obtained
from the original LHE weight divided by the **entire** source event count.
There is no normalization to S/H counts, multiplicity counts, survivors or
unit area. Fixed-source-count sampling errors are
`sqrt(max(sum(w²) - sum(w)²/N, 0)/(N*(N-1)))`, with the **complete** source
count `N`. Source integration-normalization errors are reported separately
in comparison.json and are not propagated here. Cross-run and bin
covariances are unavailable. Ratio points require a positive
reference bin above three conditional sampling errors. Signed bins remain
visible with a symmetric-logarithmic axis when needed.

Historical comparisons include different source epochs; the old shower QED
setting does not enter these pre-shower scale distributions. The source-scale
prescription is unchanged by the backward-ISR switch, but changed weights and
event populations can change its sampled distribution.

![S-event scales](scalup_s.png)

![H-event scales](scalup_h.png)

[Bin data](source_scales.csv), [hashes and integral checks](source_scales.json).

```sh
python3 fxfx_diagnostics/plot_wz_source_scales.py
```
''')
    print('Wrote', args.out / 'source_scales.pdf', flush=True)


if __name__ == '__main__':
    main()
