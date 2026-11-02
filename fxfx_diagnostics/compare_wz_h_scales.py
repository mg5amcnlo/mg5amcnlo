#!/usr/bin/env python3
"""Audit the WZ FxFx-Delta H-scale rerun and compare its HwU distributions."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from madgraph.various.histograms import HwUList


def audit_lhe(path):
    """Inspect event records only; the banner also has scalesfunctionalform."""
    counts = defaultdict(Counter)
    groups = defaultdict(list)
    scales, samples, kinds, fingerprints, weights = [], [], [], [], []
    physical = hashlib.sha256()
    headers = hashlib.sha256()
    in_event = False
    need_header = False
    rows = 0
    tags = 0
    attributes = re.compile(r'pt_clust_(\d+)=[\'"]([^\'"]+)[\'"]')
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            text = line.strip()
            if text == '<event>' or text.startswith('<event '):
                if in_event:
                    raise ValueError('Nested event')
                in_event, need_header = True, True
                kind = None
                event_tags = 0
                continue
            if not in_event or not text:
                continue
            if need_header:
                fields = text.split()
                rows, sample = int(fields[0]), int(fields[1])
                weight, scale = float(fields[2]), float(fields[3])
                if sample not in (0, 1) or not np.isfinite([weight, scale]).all() or scale <= 0:
                    raise ValueError('Invalid event header: ' + text)
                headers.update((' '.join(fields[:3] + fields[4:]) + '\n').encode())
                physical.update((str(rows) + ':' + str(sample) + '\n').encode())
                fingerprint = hashlib.blake2b(digest_size=16)
                fingerprint.update((str(rows) + ':' + str(sample) + '\n').encode())
                need_header = False
            elif rows:
                record = (' '.join(text.split()) + '\n').encode()
                physical.update(record)
                fingerprint.update(record)
                rows -= 1
            elif text.startswith('#aMCatNLO'):
                kind = int(text.split()[1])
                if kind not in (1, 2):
                    raise ValueError('Invalid S/H type')
            elif text.startswith('<scales '):
                if any(name in text for name in ('scalup_', 'fxfxdelta', 'fxfxmatch')):
                    raise ValueError('Unexpected directed FxFx metadata')
                markers = [float(value) for _, value in attributes.findall(text)]
                if not markers or abs(max(markers) - scale) > 1.e-5 + 1.e-6 * scale:
                    raise ValueError('FxFx marker / SCALUP mismatch: ' + text)
                tags += 1
                event_tags += 1
            elif text == '</event>':
                if kind is None or event_tags != 1:
                    raise ValueError('Missing S/H type or FxFx scale tags')
                name = 'S' if kind == 1 else 'H'
                counts[str(sample)]['events'] += 1
                counts[str(sample)][name] += 1
                counts[str(sample)]['negative' if weight < 0 else 'positive'] += 1
                groups[f'{sample}:{name}'].append(scale)
                scales.append(scale)
                samples.append(sample)
                kinds.append(kind)
                weights.append(weight)
                fingerprint.update(str(kind).encode())
                fingerprints.append(fingerprint.digest())
                in_event = False
    if in_event:
        raise ValueError('Truncated event file')
    weights = np.asarray(weights)
    absolute_weights = np.abs(weights)
    summary = {
        'file': str(path), 'events': len(scales), 'scales_tags': tags,
        'samples': dict(counts),
        'negative_fraction': sum(g['negative'] for g in counts.values()) / len(scales),
        'absolute_weight_min': float(absolute_weights.min()),
        'absolute_weight_max': float(absolute_weights.max()),
        'physical_event_order_sha256': physical.hexdigest(),
        'headers_without_scalup_sha256': headers.hexdigest(),
        'scalup_GeV': {},
    }
    for name, values in sorted(groups.items()):
        quantiles = np.quantile(values, [0, .01, .1, .5, .9, .99, 1])
        summary['scalup_GeV'][name] = dict(zip(
            ('min', 'p01', 'p10', 'median', 'p90', 'p99', 'max'), map(float, quantiles)))
    return (summary, np.asarray(scales), np.asarray(samples), np.asarray(kinds),
            np.frombuffer(b''.join(fingerprints), dtype='V16'), weights)


def read_histograms(path):
    histograms = HwUList(str(path))
    lookup = {(h.title, h.type): h for h in histograms}
    residual = 0.
    for title, category in lookup:
        if category != 'SUM':
            continue
        zero, one, total = [lookup[title, kind] for kind in ('0', '1', 'SUM')]
        if len(zero.bins) != len(one.bins) or len(zero.bins) != len(total.bins):
            raise ValueError('Inconsistent sample bin counts')
        for a, b, result in zip(zero.bins, one.bins, total.bins):
            if a.boundaries != b.boundaries or a.boundaries != result.boundaries:
                raise ValueError('Inconsistent histogram bins')
            x, y, z = [item.get_weight('central') for item in (a, b, result)]
            errors = [item.get_weight('stat_error') for item in (a, b, result)]
            if not np.isfinite([x, y, z, *errors]).all():
                raise ValueError('Nonfinite histogram weight or error')
            relative = abs(z - x - y) / max(1.e-300, abs(x) + abs(y), abs(z))
            residual = max(residual, relative)
            if relative > 5.e-6:
                raise ValueError('Histogram sample sum failed')
    rates = {kind: lookup['total rate', kind].bins[0].wgts for kind in ('0', '1', 'SUM')}
    return lookup, {'file': str(path), 'rates_pb': rates,
                    'max_histogram_sum_relative_residual': residual}


def read_shower(process):
    logs = sorted((process / 'MCatNLO/RUN_PYTHIA8_1').glob('mcatnlo_run_[0-9]*.log'))
    counts = defaultdict(Counter)
    messages = Counter()
    for path in logs:
        text = path.read_text()
        total = re.search(r'Les Houches User Process\(es\)\s+9999\s*\|\s*(\d+)\s+(\d+)\s+(\d+)', text)
        if total is None:
            raise ValueError('Missing shower statistics: ' + str(path))
        for name, value in zip(('tried', 'selected', 'accepted'), total.groups()):
            counts['SUM'][name] += int(value)
        for category, tried, selected, accepted in re.findall(
                r'whereof user classification code\s+([01])\s*\|\s*(\d+)\s+(\d+)\s+(\d+)', text):
            for name, value in zip(('tried', 'selected', 'accepted'), (tried, selected, accepted)):
                counts[category][name] += int(value)
        for number, message in re.findall(r'^\s*\|\s*(\d+)\s+((?:Abort|Error|Warning|Info) [^|]+)\|', text, re.M):
            messages[message.strip()] += int(number)
    if not logs:
        raise ValueError('No split shower logs: ' + str(process))
    for card in (process / 'MCatNLO/RUN_PYTHIA8_1').glob('run_*/Pythia8.cmd'):
        text = card.read_text()
        if not re.search(r'JetMatching:doFxFx\s*=\s*on', text):
            raise ValueError('FxFx is disabled: ' + str(card))
        if re.search(r'Beams:setDipoleShowerStartingScalesFromLHEF\s*=\s*on', text):
            raise ValueError('Directed scales enabled for scalar comparison: ' + str(card))
    unexpected = sum(number for message, number in messages.items()
                     if message.startswith('Error ') or
                     (message.startswith('Abort ') and 'reached end of Les Houches Events File' not in message))
    return {'jobs': len(logs), 'counts': dict(counts), 'messages': dict(messages),
            'errors_or_unexpected_aborts': unexpected}


def infer_accepted_signs(run):
    """Recover signed counts from the uniform event weights and normalized rate."""
    lhe = run['lhe']
    if lhe['absolute_weight_min'] != lhe['absolute_weight_max'] or lhe['absolute_weight_min'] <= 0.:
        return None
    weight = lhe['absolute_weight_min']
    result = {}
    for category in ('0', '1', 'SUM'):
        accepted = run['shower']['counts'][category]['accepted']
        signed = run['hwu']['rates_pb'][category]['central'] * lhe['events'] / weight
        negative_float = (accepted - signed) / 2.
        negative = round(negative_float)
        # HwU stores seven decimals in scientific notation; allow its rounding.
        if abs(negative_float - negative) > .05 or negative < 0 or negative > accepted:
            raise ValueError('Uniform-weight signed-count reconstruction failed')
        result[category] = {'positive': accepted-negative, 'negative': negative,
                            'negative_fraction': negative/accepted}
    return result


def plot_comparison(old, new, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    observables = [('pT W+Z [GeV]', r'$p_T(W^+Z)$ [GeV]', 'pt_wz'),
                   ('pT j1 [GeV]', r'$p_T(j_1)$ [GeV]', 'pt_j1'),
                   ('sqrt d01 [GeV]', r'$\sqrt{d_{01}}$ [GeV]', 'djr01')]
    with PdfPages(output / 'distributions.pdf') as pdf:
        for title, xlabel, filename in observables:
            fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex='col',
                                     gridspec_kw={'height_ratios': [3, 1]})
            for column, category in enumerate(('SUM', '0', '1')):
                a, b = old[title, category], new[title, category]
                nshow = sum(x.boundaries[1] <= 400. for x in a.bins)
                abins, bbins = a.bins[:nshow], b.bins[:nshow]
                edges = np.array([abins[0].boundaries[0]] + [x.boundaries[1] for x in abins])
                centers = .5 * (edges[:-1] + edges[1:])
                av, bv = [np.array([x.get_weight('central') for x in bins]) for bins in (abins, bbins)]
                ae, be = [np.array([x.get_weight('stat_error') for x in bins]) for bins in (abins, bbins)]
                top, bottom = axes[:, column]
                for values, errors, color, label in (
                        (av, ae, '#555555', 'Old hard scalar'),
                        (bv, be, '#6f42a1', 'Geometric mean')):
                    top.stairs(np.where(values > 0., values, np.nan), edges, color=color, label=label)
                    top.errorbar(centers, np.where(values > 0., values, np.nan),
                                 yerr=errors, fmt='none', color=color, alpha=.4, linewidth=.7)
                top.set_yscale('log')
                top.set_title('Merged' if category == 'SUM' else f'Sample @{category}')
                top.set_ylabel('Cross section per bin [pb]')
                # Show ratios where the old denominator is statistically resolved.
                valid = (av > 3. * ae) & (av > 0.)
                ratio = np.divide(bv, av, out=np.full_like(av, np.nan), where=valid)
                reference_error = np.divide(ae, av, out=np.full_like(av, np.nan), where=valid)
                bottom.fill_between(centers, 1.-reference_error, 1.+reference_error,
                                    step='mid', color='#777777', alpha=.2)
                bottom.plot(centers, ratio, color='#6f42a1', drawstyle='steps-mid')
                bottom.errorbar(centers[valid], ratio[valid], yerr=be[valid]/av[valid],
                                fmt='none', color='#6f42a1', linewidth=.7, alpha=.6)
                bottom.axhline(1., color='#777777', linewidth=.8)
                bottom.set_ylabel('New / old')
                bottom.set_xlabel(xlabel)
                for ax in (top, bottom):
                    ax.set_xlim(0., 400.)
                    ax.grid(alpha=.2)
                top.legend(fontsize=8)
            fig.suptitle(r'$pp\to W^+Z+0,1j$, FxFx–Delta, $q_{cut}=70$ GeV, Pythia 8.317')
            fig.text(.5, .01, 'Ratio band: old error; bars: new error / old central. No cross-run covariance. Old denominator > 3 statistical errors.',
                     ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .035, 1, .95))
            fig.savefig(output / (filename + '.png'), dpi=160)
            pdf.savefig(fig)
            plt.close(fig)


def write_report(result, output):
    old, new = result['old'], result['new']
    lines = [
        '# WZ FxFx Delta comparison of H-event shower scales', '',
        'The old hard scalar and the geometric mean were compared with one million '
        'events each for `p p > w+ z [QCD] @0` and `p p > w+ z j [QCD] @1`.', '',
        'Both runs use 13 TeV beams, the same model and physics cards, `ptj=10` GeV, '
        '`Qcut=70` GeV, maximum merged multiplicity one, stable W/Z bosons, and '
        'Pythia 8.317 with MPI, hadronization and primordial transverse momentum disabled and `qed_shower=F`. '
        'The archived steering did not explicitly disable `TimeShower:QEDshowerByOther`; '
        'this comparison predates that switch fix and does not establish the absence '
        'of charged-resonance QED radiation. '
        'The original integration grids and Fortran seed 33 were reused. The integration '
        'normalization is therefore shared; this is a test of event generation and showering, '
        'rather than an independent redetermination of the NLO integral.', '',
        'The baseline is commit `a4d369829a647ace3d8b87e1466593062c197f2f`. '
        '[The source patch](source.patch), [setup and hashes](setup.json), and '
        '[effective cards](cards/) record the comparison inputs.', '',
        '## Merged rates', '',
        '| Sample | Old scalar (pb) | Geometric mean (pb) | Change |',
        '| --- | ---: | ---: | ---: |',
    ]
    for category in ('0', '1', 'SUM'):
        a, b = [run['hwu']['rates_pb'][category] for run in (old, new)]
        change = result['rate_changes'][category]['relative_change'] * 100.
        label = 'Merged' if category == 'SUM' else '@' + category
        lines.append(f'| {label} | {a["central"]:.6f} ± {a["stat_error"]:.6f} | '
                     f'{b["central"]:.6f} ± {b["stat_error"]:.6f} | {change:+.3f}% |')
    lines += ['', 'Errors are the statistical errors reported by each HwU run. '
              'The samples share integration information and many hard events; '
              'no cross-run covariance or uncertainty on their difference is estimated.', '',
              '## Event and scale audit', '',
              f'The negative-weight fractions before showering are '
              f'{old["lhe"]["negative_fraction"] * 100:.4f}% (old) and '
              f'{new["lhe"]["negative_fraction"] * 100:.4f}% (new). '
              f'The comparison matched {result["common_unique_event_records"]:,} unique '
              f'physical records by particle data and S/H type, with '
              f'{result["common_records_with_changed_weight"]:,} changed weights among them. '
              'The Python allocation of event counts to integration jobs can differ despite '
              'the same Fortran seed, so identical event order is not assumed.', '',
              '| Sample | Median old H scale (GeV) | Median new H scale (GeV) | '
              'Common S records with changed scale |', '| --- | ---: | ---: | ---: |']
    for sample in (0, 1):
        a, b = [run['lhe']['scalup_GeV'][f'{sample}:H']['median'] for run in (old, new)]
        s = result['paired_scalup_ratios'][f'{sample}:S']
        lines.append(f'| @{sample} | {a:.6f} | {b:.6f} | {s["changed"]:,} / {s["count"]:,} |')
    lines += ['', 'The H scale is the unweighted geometric mean of positive directed '
              'H-event starts mapped from the stopping scales. Diagonal and nonpositive '
              'entries are excluded. The selected outer flow and fold own the scalar, '
              'with no Born hard-scale floor. The ordinary FxFx fallback is retained '
              'where no Delta stopping scales were reconstructed. This scalar approximation '
              'does not reproduce competition with separate directed limits.', '',
              '## Shower and histogram checks', '',
              '| Prescription | Selected LHE records | Accepted after shower and merging | '
              'Pythia errors or unexpected aborts |', '| --- | ---: | ---: | ---: |']
    for label, run in (('Old scalar', old), ('Geometric mean', new)):
        shower = run['shower']
        counts = shower['counts']['SUM']
        lines.append(f'| {label} | {counts["selected"]:,} | {counts["accepted"]:,} | '
                     f'{shower["errors_or_unexpected_aborts"]:,} |')
    if old['accepted_signs_from_uniform_weights'] and new['accepted_signs_from_uniform_weights']:
        a, b = [run['accepted_signs_from_uniform_weights']['SUM']['negative_fraction'] for run in (old, new)]
        lines += ['', f'The negative-weight fractions among accepted merged events are '
                  f'{a*100:.4f}% (old) and {b*100:.4f}% (new). These counts are inferred '
                  'from the accepted totals and signed rates after checking that the '
                  'LHE absolute weights are uniform; the HwU rounding is consistent '
                  'with integer positive and negative counts.']
    lines += ['', 'Each histogram was checked bin by bin for `SUM = @0 + @1`. '
              'The maximum relative residuals are '
              f'{old["hwu"]["max_histogram_sum_relative_residual"]:.3g} (old) and '
              f'{new["hwu"]["max_histogram_sum_relative_residual"]:.3g} (new). '
              'Expected end-of-file messages are excluded from the abort count; '
              'all warning counts are retained in the JSON audits.', '',
              '## Distributions', '',
              '[Three-page comparison PDF](distributions.pdf): merged, @0 and @1 '
              'spectra with per-run statistical errors and new/old ratios. '
              'The ratio band shows the old relative statistical error, and ratio bars '
              'show the new error divided by the old central value; they do not include '
              'a cross-run covariance. '
              'Plots show 0–400 GeV; the [CSV](histogram_comparison.csv) contains every bin. '
              'Ratios omit bins whose old denominator is below three statistical errors.', '',
              '![WZ transverse momentum](pt_wz.png)', '',
              '![Leading jet transverse momentum](pt_j1.png)', '',
              '![Differential jet rate](djr01.png)', '',
              '## Reproduction', '',
              'The generation and shower commands are recorded in `setup.json`. '
              'The original workflow audit stopped on the banner tag '
              '`scalesfunctionalform`; this audit checks scale tags only inside '
              'event records. '
              'Once the old and new run outputs are available, run from the checkout root:', '',
              '```sh', 'python3 fxfx_diagnostics/compare_wz_h_scales.py', '```', '',
              '[Full numerical comparison](comparison.json), '
              '[old audit](old_audit.json), [new audit](new_audit.json).', '']
    (output / 'comparison.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, default=ROOT / 'PROC_pp_wz01j_EWjets_Delta')
    parser.add_argument('--new', type=Path, default=ROOT / 'PROC_pp_wz01j_EWjets_Delta_Hgeom')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/audits/fxfx_delta_wz_h_scales')
    parser.add_argument('--old-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    old_run = args.old / 'Events/run_01'
    old_audit, old_scales, old_samples, old_kinds, old_ids, old_weights = audit_lhe(old_run / 'events.lhe.gz')
    old_hist, old_hwu = read_histograms(old_run / 'plot_PYTHIA8_1_0.HwU')
    old_shower = read_shower(args.old)
    if old_shower['counts']['SUM']['selected'] != old_audit['events']:
        raise ValueError('Old shower did not select every LHE record')
    old_result = {'lhe': old_audit, 'hwu': old_hwu, 'shower': old_shower}
    old_result['accepted_signs_from_uniform_weights'] = infer_accepted_signs(old_result)
    (args.out / 'old_audit.json').write_text(json.dumps(old_result, indent=2) + '\n')
    print('Old audit:', old_audit['events'], 'events;', old_hwu['rates_pb'], flush=True)
    if args.old_only:
        return
    new_run = args.new / 'Events/run_hgeom'
    new_audit, new_scales, new_samples, new_kinds, new_ids, new_weights = audit_lhe(new_run / 'events.lhe.gz')
    expected_events = json.loads((args.out / 'setup.json').read_text())['nevents']
    if old_audit['events'] != expected_events or new_audit['events'] != expected_events:
        raise ValueError('Generated event count differs from comparison setup')
    new_hist, new_hwu = read_histograms(new_run / 'plot_PYTHIA8_1_0.HwU')
    new_shower = read_shower(args.new)
    if new_shower['counts']['SUM']['selected'] != new_audit['events']:
        raise ValueError('New shower did not select every LHE record')
    new_result = {'lhe': new_audit, 'hwu': new_hwu, 'shower': new_shower}
    new_result['accepted_signs_from_uniform_weights'] = infer_accepted_signs(new_result)
    (args.out / 'new_audit.json').write_text(json.dumps(new_result, indent=2) + '\n')
    result = {'old': old_result, 'new': new_result, 'rate_changes': {}, 'binned_integrals': []}
    aligned = (old_audit['physical_event_order_sha256'] == new_audit['physical_event_order_sha256']
               and np.array_equal(old_samples, new_samples) and np.array_equal(old_kinds, new_kinds))
    result['identical_ordered_event_kinematics_and_types'] = aligned
    result['identical_headers_except_scalup'] = old_audit['headers_without_scalup_sha256'] == new_audit['headers_without_scalup_sha256']
    common, old_index, new_index = np.intersect1d(old_ids, new_ids, return_indices=True)
    result['common_unique_event_records'] = int(len(common))
    result['common_records_with_changed_weight'] = int(np.count_nonzero(old_weights[old_index] != new_weights[new_index]))
    result['old_duplicate_event_records'] = int(len(old_ids) - len(np.unique(old_ids)))
    result['new_duplicate_event_records'] = int(len(new_ids) - len(np.unique(new_ids)))
    result['paired_scalup_ratios'] = {}
    for sample in (0, 1):
        for kind in (1, 2):
            mask = (old_samples[old_index] == sample) & (old_kinds[old_index] == kind)
            a, b = old_scales[old_index[mask]], new_scales[new_index[mask]]
            ratios = b / a
            result['paired_scalup_ratios'][f'{sample}:{"S" if kind == 1 else "H"}'] = {
                'changed': int(np.count_nonzero(b != a)), 'count': int(mask.sum()),
                'quantiles': dict(zip(('min', 'p10', 'median', 'p90', 'max'),
                                     map(float, np.quantile(ratios, [0, .1, .5, .9, 1])))) if len(ratios) else {}}
    for category in ('0', '1', 'SUM'):
        a, b = [h['rates_pb'][category]['central'] for h in (old_hwu, new_hwu)]
        result['rate_changes'][category] = {'old_pb': a, 'new_pb': b, 'difference_pb': b-a, 'relative_change': b/a-1.}
    with (args.out / 'histogram_comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['observable', 'sample', 'xmin', 'xmax', 'old_pb', 'old_error_pb', 'new_pb', 'new_error_pb', 'new_over_old'])
        for title, category in old_hist:
            a, b = old_hist[title, category], new_hist[title, category]
            if len(a.bins) != len(b.bins):
                raise ValueError('Old/new bin counts differ')
            for x, y in zip(a.bins, b.bins):
                if x.boundaries != y.boundaries:
                    raise ValueError('Old/new bins differ')
                av, bv = x.get_weight('central'), y.get_weight('central')
                writer.writerow([title, category, *x.boundaries, av, x.get_weight('stat_error'),
                                 bv, y.get_weight('stat_error'), bv/av if av else ''])
            if title not in ('pT W+Z [GeV]', 'pT j1 [GeV]', 'sqrt d01 [GeV]'):
                continue
            for low, high in ((0, 20), (20, 70), (70, 150), (150, 300), (300, 1000)):
                values = []
                for hist in (a, b):
                    bins = [x for x in hist.bins if x.boundaries[0] >= low and x.boundaries[1] <= high]
                    values.append({'central_pb': sum(x.get_weight('central') for x in bins),
                                   'error_pb': float(np.sqrt(sum(x.get_weight('stat_error')**2 for x in bins)))})
                av, bv = [x['central_pb'] for x in values]
                result['binned_integrals'].append({'observable': title, 'sample': category,
                    'range_GeV': [low, high], 'old': values[0], 'new': values[1],
                    'relative_change': bv/av-1. if av else None})
    (args.out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    plot_comparison(old_hist, new_hist, args.out)
    write_report(result, args.out)
    print(json.dumps({'rates': result['rate_changes'], 'aligned_events': aligned}, indent=2))


if __name__ == '__main__':
    main()
