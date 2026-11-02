#!/usr/bin/env python3
"""Compare WZ FxFx-Delta backward-ISR samples and optional historical runs."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fxfx_diagnostics.compare_wz_h_scales import (
    audit_lhe, infer_accepted_signs, read_histograms, read_shower,
)

CATEGORIES = ('0', '1', 'SUM')
OBSERVABLES = (
    ('pT W+Z [GeV]', r'$p_T(W^+Z)$ [GeV]', 'pt_wz'),
    ('pT j1 [GeV]', r'$p_T(j_1)$ [GeV]', 'pt_j1'),
    ('sqrt d01 [GeV]', r'$\sqrt{d_{01}}$ [GeV]', 'djr01'),
)
BANDS = ((0, 10), (10, 20), (0, 20), (20, 70), (70, 150),
         (150, 300), (300, 1000))
UNCERTAINTY_NOTE = (
    'HwU rate/bin errors and plot bands are conditional event/shower sampling '
    'errors at the estimated source normalization. Separately reported source '
    'integration-normalization errors are not propagated into them. Cross-run '
    'and bin covariance are not estimated; no significance is assigned to '
    'differences or ratios.'
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def numeric(value):
    return float(value.replace('D', 'e').replace('d', 'e'))


def integration_summary(run_dir):
    """Read the last saved signed/absolute integration totals, not LHE means."""
    candidates = sorted(run_dir.glob('res_*.txt'),
                        key=lambda path: int(re.search(r'res_(\d+)', path.name)[1]))
    pattern = re.compile(r'Total ABS and\s+Total:\s*'
                         r'([\d.eEdD+\-]+)\s*\+-\s*([\d.eEdD+\-]+)[^\n]*\n\s*'
                         r'([\d.eEdD+\-]+)\s*\+-\s*([\d.eEdD+\-]+)')
    for path in reversed(candidates):
        match = pattern.search(path.read_text())
        if match:
            a, ae, s, se = map(numeric, match.groups())
            return {'file': str(path), 'absolute_pb': a, 'absolute_error_pb': ae,
                    'signed_pb': s, 'signed_error_pb': se}
    return None


def read_settings(path):
    settings = {}
    for line in path.read_text().splitlines():
        text = line.split('!', 1)[0].strip()
        if '=' in text:
            key, value = map(str.strip, text.split('=', 1))
            settings[key.lower()] = value
    return settings


def input_provenance(process, run_dir, saved_steering=None):
    files = {}
    for base, names in ((process / 'Cards', ('run_card.dat', 'shower_card.dat',
                                            'FKS_params.dat', 'param_card.dat',
                                            'MadLoopParams.dat', 'proc_card_mg5.dat')),
                        (process / 'SubProcesses', ('montecarlocounter.f', 'FKSParams.f90',
                                                   'sudakov.f', 'scale_module.f90',
                                                   'driver_mintMC.f'))):
        for name in names:
            path = base / name
            if path.exists():
                files[str(path.relative_to(process))] = sha256(path)
    for path in sorted(run_dir.glob('*_banner.txt')):
        files[str(path.relative_to(process))] = sha256(path)
    steering_files = sorted((process / 'MCatNLO/RUN_PYTHIA8_1').glob('run_*/Pythia8.cmd'))
    variants = defaultdict(list)
    settings = {}
    for path in steering_files:
        digest = sha256(path)
        variants[digest].append(str(path.relative_to(process)))
        settings[digest] = read_settings(path)
    if not steering_files and saved_steering:
        digest = sha256(saved_steering)
        variants[digest].append(str(saved_steering.resolve()))
        settings[digest] = read_settings(saved_steering)
    return {'file_sha256': files,
            'steering_variants': dict(variants),
            'steering_settings': settings,
            'run_summary': (run_dir / 'summary.txt').read_text()
                           if (run_dir / 'summary.txt').exists() else None}


def audit_run(process, run_name, label, expected_events,
              saved_audit=None, saved_steering=None):
    process = process.resolve()
    run_dir = process / 'Events' / run_name
    lhe_path = run_dir / 'events.lhe.gz'
    summary, scales, samples, kinds, _, weights = audit_lhe(lhe_path)
    if expected_events is not None and summary['events'] != expected_events:
        raise ValueError(f'{label}: expected {expected_events} events, found {summary["events"]}')
    summary['sample_type_signs'] = {}
    for sample in (0, 1):
        for kind, name in ((1, 'S'), (2, 'H')):
            mask = (samples == sample) & (kinds == kind)
            selected = weights[mask]
            count = int(mask.sum())
            negative = int(np.count_nonzero(selected < 0.))
            summary['sample_type_signs'][f'{sample}:{name}'] = {
                'events': count, 'negative': negative, 'positive': count - negative,
                'negative_fraction': negative / count if count else None,
                'signed_weight_sum': float(selected.sum()),
                'absolute_weight_sum': float(np.abs(selected).sum()),
            }
    summary['signed_weight_mean_pb'] = float(weights.mean())
    summary['sha256'] = sha256(lhe_path)
    del scales, samples, kinds, weights
    histogram_path = run_dir / 'plot_PYTHIA8_1_0.HwU'
    histograms, hwu = read_histograms(histogram_path)
    hwu['sha256'] = sha256(histogram_path)
    if list((process / 'MCatNLO/RUN_PYTHIA8_1').glob('mcatnlo_run_[0-9]*.log')):
        shower = read_shower(process)
        shower_provenance = {'mode': 'live split logs'}
    elif saved_audit:
        saved = json.loads(saved_audit.read_text())
        # The archived logs must belong to the retained event cohort, not
        # merely to another run with the same number of events.
        checks = ('events', 'physical_event_order_sha256',
                  'headers_without_scalup_sha256', 'scalup_GeV')
        candidates = [('root', saved)] if 'lhe' in saved and 'shower' in saved else [
            (name, value) for name, value in saved.items()
            if isinstance(value, dict) and 'lhe' in value and 'shower' in value]
        matches = [(name, value) for name, value in candidates
                   if all(value['lhe'].get(check) == summary[check] for check in checks)]
        if len(matches) != 1:
            raise ValueError(f'{label}: saved audit has no unique matching event cohort; '
                             'supply an audit containing both LHE and shower statistics')
        selection, cached = matches[0]
        shower = cached['shower']
        shower_provenance = {'mode': 'saved audit; split logs no longer retained',
                             'file': str(saved_audit.resolve()),
                             'selection': selection,
                             'sha256': sha256(saved_audit),
                             'event_cohort_checks': list(checks)}
    else:
        raise ValueError(f'{label}: no live shower logs; supply the saved audit explicitly')
    if shower['counts']['SUM']['selected'] != summary['events']:
        raise ValueError(f'{label}: shower did not select every LHE record')
    if shower['counts']['SUM']['tried'] != summary['events']:
        raise ValueError(f'{label}: shower tried count differs from LHE count')
    result = {'label': label, 'process': str(process), 'run': run_name,
              'lhe': summary, 'hwu': hwu, 'shower': shower,
              'shower_provenance': shower_provenance,
              'integration': integration_summary(run_dir),
              'provenance': input_provenance(process, run_dir, saved_steering)}
    result['accepted_signs_from_uniform_weights'] = infer_accepted_signs(result)
    return result, histograms


def changes(reference, new):
    result = {}
    for category in CATEGORIES:
        a, b = [run['hwu']['rates_pb'][category] for run in (reference, new)]
        av, bv = a['central'], b['central']
        result[category] = {
            'reference_pb': av, 'reference_error_pb': a['stat_error'],
            'new_pb': bv, 'new_error_pb': b['stat_error'],
            'difference_pb': bv - av,
            'relative_change': bv / av - 1. if av else None,
        }
    return result


def steering_differences(reference, new):
    a = list(reference['provenance']['steering_settings'].values())
    b = list(new['provenance']['steering_settings'].values())
    # Event counts and file names vary by shard; record the physics-setting
    # values observed over all shards, including absent entries.
    keys = set().union(*(set(s) for s in a + b)) if a + b else set()
    result = {}
    for key in sorted(keys):
        if key.startswith('main:') or key == 'beams:lhef':
            continue
        av = sorted({s.get(key, '<absent>') for s in a})
        bv = sorted({s.get(key, '<absent>') for s in b})
        if av != bv:
            result[key] = {'reference': av, 'new': bv}
    return result


def compare_bins(histograms, output):
    keys = list(histograms)
    reference = histograms[keys[0]]
    integrals = []
    with (output / 'histogram_comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['run', 'observable', 'sample', 'xmin', 'xmax', 'central_pb', 'error_pb'])
        for title, category in reference:
            for key, lookup in histograms.items():
                current = lookup[title, category]
                if [b.boundaries for b in current.bins] != [b.boundaries for b in reference[title, category].bins]:
                    raise ValueError(f'{key}: histogram binning differs for {title}, {category}')
                for item in current.bins:
                    writer.writerow([key, title, category, *item.boundaries,
                                     item.get_weight('central'), item.get_weight('stat_error')])
            if title not in {item[0] for item in OBSERVABLES}:
                continue
            for low, high in BANDS:
                values = {}
                for key, lookup in histograms.items():
                    bins = [b for b in lookup[title, category].bins
                            if b.boundaries[0] >= low and b.boundaries[1] <= high]
                    if not bins or bins[0].boundaries[0] != low or bins[-1].boundaries[1] != high:
                        raise ValueError(f'{key}: requested integral [{low}, {high}] is not a complete bin interval')
                    values[key] = {
                        'central_pb': sum(b.get_weight('central') for b in bins),
                        'error_pb': float(np.sqrt(sum(b.get_weight('stat_error')**2 for b in bins))),
                    }
                new = values['new']['central_pb']
                relative = {key: new / value['central_pb'] - 1. if value['central_pb'] else None
                            for key, value in values.items() if key != 'new'}
                integrals.append({'observable': title, 'sample': category,
                                  'range_GeV': [low, high], 'runs': values,
                                  'new_relative_changes': relative})
    return integrals


def plot_comparison(runs, histograms, references, output, prefix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = ['#555555', '#c77c00', '#1f77b4', '#2ca02c']
    with PdfPages(output / (prefix + 'distributions.pdf')) as pdf:
        for title, xlabel, filename in OBSERVABLES:
            fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex='col',
                                     gridspec_kw={'height_ratios': [3, 1]})
            for column, category in enumerate(('SUM', '0', '1')):
                new = histograms['new'][title, category]
                nshow = sum(b.boundaries[1] <= 400. for b in new.bins)
                new_bins = new.bins[:nshow]
                edges = np.array([new_bins[0].boundaries[0]] + [b.boundaries[1] for b in new_bins])
                centers = .5 * (edges[:-1] + edges[1:])
                nv = np.array([b.get_weight('central') for b in new_bins])
                ne = np.array([b.get_weight('stat_error') for b in new_bins])
                top, bottom = axes[:, column]
                for index, key in enumerate(references):
                    bins = histograms[key][title, category].bins[:nshow]
                    rv = np.array([b.get_weight('central') for b in bins])
                    re = np.array([b.get_weight('stat_error') for b in bins])
                    color = colors[index % len(colors)]
                    top.stairs(np.where(rv > 0., rv, np.nan), edges, color=color, label=runs[key]['label'])
                    top.errorbar(centers, np.where(rv > 0., rv, np.nan), yerr=re,
                                 fmt='none', color=color, alpha=.35, linewidth=.6)
                    valid = (rv > 3. * re) & (rv > 0.)
                    ratio = np.divide(nv, rv, out=np.full_like(nv, np.nan), where=valid)
                    bottom.plot(centers, ratio, color=color, drawstyle='steps-mid',
                                label='ON / ' + runs[key]['label'])
                    if index == 0:
                        err = np.divide(re, rv, out=np.full_like(rv, np.nan), where=valid)
                        bottom.fill_between(centers, 1.-err, 1.+err, step='mid', color=color, alpha=.15)
                        bottom.errorbar(centers[valid], ratio[valid], yerr=ne[valid] / rv[valid],
                                        fmt='none', color=color, linewidth=.6, alpha=.5)
                top.stairs(np.where(nv > 0., nv, np.nan), edges, color='#6f42a1', label=runs['new']['label'])
                top.errorbar(centers, np.where(nv > 0., nv, np.nan), yerr=ne,
                             fmt='none', color='#6f42a1', alpha=.4, linewidth=.7)
                top.set_yscale('log')
                top.set_title('Merged' if category == 'SUM' else f'Sample @{category}')
                top.set_ylabel('Cross section per bin [pb]')
                bottom.axhline(1., color='#777777', linewidth=.8)
                bottom.set_ylabel('ON / reference')
                bottom.set_xlabel(xlabel)
                top.legend(fontsize=7)
                if len(references) > 1:
                    bottom.legend(fontsize=6)
                for ax in (top, bottom):
                    ax.set_xlim(0., 400.)
                    ax.grid(alpha=.2)
            fig.suptitle(r'$pp\to W^+Z+0,1j$, FxFx–Delta, backward-ISR source correction')
            fig.text(.5, .01,
                     'Ratio band: first reference error; bars: ON error / reference. No cross-run covariance. Reference > 3 statistical errors.',
                     ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .035, 1, .95))
            fig.savefig(output / (prefix + filename + '.png'), dpi=160)
            pdf.savefig(fig)
            plt.close(fig)


def percentage(value):
    return 'undefined' if value is None else f'{100.*value:+.3f}%'


def assess_comparison(result):
    """Retain the fixed-setup interpretation separately from sampled errors."""
    reference, new = [result['runs'][key] for key in ('reference', 'new')]
    before = 100.*(new['lhe']['negative_fraction'] - reference['lhe']['negative_fraction'])
    accepted = [run['accepted_signs_from_uniform_weights'] for run in (reference, new)]
    after = (100.*(accepted[1]['SUM']['negative_fraction'] -
                   accepted[0]['SUM']['negative_fraction'])
             if all(accepted) else None)
    bands = {
        '{}-{}'.format(*item['range_GeV']): item['new_relative_changes']['reference']
        for item in result['binned_integrals']
        if item['observable'] == 'pT W+Z [GeV]' and item['sample'] == 'SUM'
    }
    return {
        'scope': 'One fixed WZ+0,1-jet FxFx-Delta setup and merging scale.',
        'interpretation': (
            'The observed total-rate and broad merged-spectrum changes are '
            'modest, giving a reasonable implementation check for this fixed '
            'setup. The comparison does not demonstrate a reduction of '
            'negative-weight fractions.'
        ),
        'uncertainty_note': UNCERTAINTY_NOTE,
        'source_normalizations': {key: result['runs'][key]['integration']
                                  for key in ('reference', 'new')},
        'merged_rate_relative_change': result['rate_changes']['reference']['SUM']['relative_change'],
        'merged_pt_wz_relative_changes': bands,
        'negative_fraction_changes_percentage_points': {'before_shower': before,
                                                       'after_merging': after},
        'demonstrated_negative_weight_reduction_claim': False,
        'formal_accuracy_limit': (
            'This numerical check establishes neither an exact or universal '
            'NLO-matching theorem nor logarithmic accuracy or exact '
            'competing-shower correspondence.'
        ),
    }


def write_report(result, output):
    runs = result['runs']
    primary = ('reference', 'new')
    assessment = result['assessment'] if 'assessment' in result else assess_comparison(result)
    lines = ['# WZ FxFx-Delta backward-ISR comparison', '',
             'The primary comparison uses `p p > w+ z [QCD] @0` and '
             '`p p > w+ z j [QCD] @1`, with geometric-mean H-event scales '
             'in both samples. The backward-ISR source switch is off in the '
             'reference and on in the new sample. Source changes require fresh '
             'integration and generation; absolute rates retain each run’s '
             'own normalization.', '',
             'The correction uses the active Delta table’s coupling and running: '
             '`alpha_s(MZ)=0.1365`, one-loop for the shipped provider. '
             'This is local to the no-emission probability; the hard matrix '
             'elements, hard PDFs and analytic FxFx factors keep their existing '
             'coupling settings. '
             'The shower’s retained `alpha_s(MZ)=0.118` is a separate setting '
             'and does not override the source correction. The table profile '
             'is an audited archive identification; its original producing log '
             'is unavailable.', '',
             '## Absolute merged rates', '',
             f'| Sample | {runs["reference"]["label"]} (pb) | {runs["new"]["label"]} (pb) | ON/OFF change |',
             '| --- | ---: | ---: | ---: |']
    for category in CATEGORIES:
        a, b = [runs[key]['hwu']['rates_pb'][category] for key in primary]
        change = result['rate_changes']['reference'][category]['relative_change']
        sample = 'Merged' if category == 'SUM' else '@' + category
        lines.append(f'| {sample} | {a["central"]:.6f} ± {a["stat_error"]:.6f} | '
                     f'{b["central"]:.6f} ± {b["stat_error"]:.6f} | {percentage(change)} |')
    lines += ['', 'Errors are each HwU run’s reported conditional event/shower '
              'sampling errors at the estimated source normalization. The '
              'separate source integration-normalization errors shown below '
              'are not propagated into these rate errors or the plot bands. '
              'No cross-run covariance or error on a difference/ratio is estimated. '
              'The binned-integral errors below combine the reported bin errors '
              'in quadrature; bin covariance is unavailable.', '',
              '## Source normalization and event signs', '',
              '| Sample | Events | Integrated signed rate (pb) | Negative before shower | Negative after merging |',
              '| --- | ---: | ---: | ---: | ---: |']
    for key in primary:
        run = runs[key]
        source = run['integration']
        value = (f'{source["signed_pb"]:.6f} ± {source["signed_error_pb"]:.6f}'
                 if source else 'not available')
        signs = run['accepted_signs_from_uniform_weights']
        after = f'{100.*signs["SUM"]["negative_fraction"]:.4f}%' if signs else 'not inferred'
        lines.append(f'| {run["label"]} | {run["lhe"]["events"]:,} | {value} | '
                     f'{100.*run["lhe"]["negative_fraction"]:.4f}% | {after} |')
    lines += ['', 'The integration errors in the signed-rate column are a '
              'separate source-normalization uncertainty. They are not part '
              'of the conditional HwU/bin errors or ratio bands. No combined '
              'uncertainty or significance for the ON/OFF difference is assigned.', '',
              'Accepted positive/negative counts are inferred only when all LHE '
              'absolute weights are equal and the signed HwU rate reproduces '
              'integer counts within its rounding precision. The JSON retains '
              'separate pre-shower signs for every sample and S/H event type.', '',
              '| Sample/type | OFF events | ON events | OFF median SCALUP (GeV) | ON median SCALUP (GeV) |',
              '| --- | ---: | ---: | ---: | ---: |']
    for sample in (0, 1):
        for kind in ('S', 'H'):
            key = f'{sample}:{kind}'
            a, b = [runs[name]['lhe'] for name in primary]
            av = a['scalup_GeV'].get(key, {}).get('median')
            bv = b['scalup_GeV'].get(key, {}).get('median')
            medians = [f'{value:.6f}' if value is not None else 'absent' for value in (av, bv)]
            lines.append(f'| @{sample} {kind} | {a["sample_type_signs"][key]["events"]:,} | '
                         f'{b["sample_type_signs"][key]["events"]:,} | {medians[0]} | {medians[1]} |')
    lines += ['', 'Event populations and weights can change under the source correction. '
              'No common-event or unchanged-weight assumption is made.', '',
              '## WZ transverse-momentum integrals', '',
              '| Sample | Range (GeV) | OFF (pb) | ON (pb) | Change |',
              '| --- | ---: | ---: | ---: | ---: |']
    for item in result['binned_integrals']:
        if item['observable'] != 'pT W+Z [GeV]' or item['range_GeV'] not in ([0, 10], [10, 20], [20, 70], [70, 150]):
            continue
        a, b = [item['runs'][key] for key in primary]
        sample = 'Merged' if item['sample'] == 'SUM' else '@' + item['sample']
        low, high = item['range_GeV']
        lines.append(f'| {sample} | {low}–{high} | {a["central_pb"]:.6f} ± {a["error_pb"]:.6f} | '
                     f'{b["central_pb"]:.6f} ± {b["error_pb"]:.6f} | '
                     f'{percentage(item["new_relative_changes"]["reference"])} |')
    lines += ['', '## Shower checks and distributions', '',
              '| Sample | Tried / selected | Accepted | Errors or unexpected aborts |',
              '| --- | ---: | ---: | ---: |']
    for key in primary:
        run = runs[key]
        count = run['shower']['counts']['SUM']
        lines.append(f'| {run["label"]} | {count["tried"]:,} / {count["selected"]:,} | '
                     f'{count["accepted"]:,} | {run["shower"]["errors_or_unexpected_aborts"]:,} |')
    lines += ['', 'All histograms satisfy the checked `SUM = @0 + @1` tolerance. '
              'Expected end-of-file messages are excluded from unexpected aborts; '
              'the JSON retains warnings and sample-specific shower counts.', '',
              '[Primary three-page PDF](distributions.pdf) and [all histogram bins](histogram_comparison.csv). '
              'Plots retain absolute cross sections. Ratio bands show the reference '
              'relative error and bars show the ON error divided by the reference '
              'central value. Ratios omit bins whose reference is below three '
              'statistical errors. Negative/nonpositive bins remain in the CSV.', '',
              '![WZ transverse momentum](pt_wz.png)', '',
              '![Leading jet transverse momentum](pt_j1.png)', '',
              '![Differential jet rate](djr01.png)', '',
              '## Actual shower steering', '']
    primary_diff = result['steering_differences']['reference']
    if primary_diff:
        lines += ['The primary samples differ in these non-event-count steering entries:', '',
                  '| Setting | Reference | ON |', '| --- | --- | --- |']
        for key, value in primary_diff.items():
            lines.append(f'| `{key}` | `{", ".join(value["reference"])}` | `{", ".join(value["new"])}` |')
    else:
        lines.append('The primary samples have identical observed physics-setting values across all shower shards.')
    historical = [key for key in runs if key not in primary]
    if historical:
        lines += ['', '## Historical comparisons', '',
                  'These compare complete run epochs. In particular, historical '
                  'steering omitted an explicit `TimeShower:QEDshowerByOther` setting, '
                  'while the fresh samples use the corrected QED-off switch. '
                  'Historical differences cannot be attributed exclusively to '
                  'the backward-ISR source correction. Their original source '
                  'normalizations and steering differences are retained in JSON.', '',
                  '| Historical reference | @0 rate (pb) | @1 rate (pb) | Merged rate (pb) | ON/merged change |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for key in historical:
            run = runs[key]
            rates = [run['hwu']['rates_pb'][cat] for cat in CATEGORIES]
            entries = [f'{value["central"]:.6f} ± {value["stat_error"]:.6f}' for value in rates]
            change = result['rate_changes'][key]['SUM']['relative_change']
            lines.append('| ' + run['label'] + ' | ' + ' | '.join(entries) + ' | ' + percentage(change) + ' |')
        lines += ['', '| Historical reference | Negative before shower | Negative after merging | Median H SCALUP @0 / @1 (GeV) |',
                  '| --- | ---: | ---: | ---: |']
        for key in historical:
            run = runs[key]
            before = 100.*run['lhe']['negative_fraction']
            accepted = run['accepted_signs_from_uniform_weights']
            after = f'{100.*accepted["SUM"]["negative_fraction"]:.4f}%' if accepted else 'not inferred'
            medians = ' / '.join(f'{run["lhe"]["scalup_GeV"][str(sample)+":H"]["median"]:.6f}'
                                 for sample in (0, 1))
            lines.append(f'| {run["label"]} | {before:.4f}% | {after} | {medians} |')
        lines += ['', '[Historical spectrum comparisons](historical_distributions.pdf).']
    changes_pp = assessment['negative_fraction_changes_percentage_points']
    after_pp = changes_pp['after_merging']
    negative_change = (f'The central negative fractions change by '
                       f'{changes_pp["before_shower"]:+.4f} percentage points before showering')
    if after_pp is not None:
        negative_change += f' and {after_pp:+.4f} points after merging'
    lines += ['', '## Assessment', '', assessment['interpretation'], '',
              negative_change + '. These shifts do not support a negative-weight '
              'reduction claim. Source integration uncertainty and the unavailable '
              'cross-run covariance limit interpretation of the small rate changes.', '',
              assessment['formal_accuracy_limit'], '',
              '## Inputs and reproduction', '',
              'Each run audit records the LHE/HwU hashes, source and card hashes, '
              'banner hashes, saved integration totals and actual shower steering. '
              '[Setup](setup.json), when supplied, records the generation commands '
              'and source snapshot. [Full comparison](comparison.json) retains '
              'all scale quantiles, signs, rates and spectrum integrals.', '',
              '```sh', result['reproduction_command'], '```', '',
              'This fixed setup tests the implemented source prescription. It '
              'does not establish exact competing-shower agreement or '
              'logarithmic accuracy of the combined FxFx–Delta scheme.', '']
    (output / 'comparison.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-process', type=Path, required=True)
    parser.add_argument('--reference-run', required=True)
    parser.add_argument('--new-process', type=Path, required=True)
    parser.add_argument('--new-run', required=True)
    parser.add_argument('--reference-label', default='Fresh backward ISR OFF')
    parser.add_argument('--new-label', default='Fresh backward ISR ON')
    parser.add_argument('--historical', nargs=3, action='append', default=[],
                        metavar=('PROCESS', 'RUN', 'LABEL'), help='Repeat for saved historical references')
    parser.add_argument('--historical-audit', type=Path, action='append', default=[],
                        help='Saved audit if historical split logs were removed; same order as --historical')
    parser.add_argument('--historical-steering', type=Path, action='append', default=[],
                        help='Archived actual steering if historical shower directory was removed; same order as --historical')
    parser.add_argument('--expected-events', type=int, default=1000000)
    parser.add_argument('--setup', type=Path)
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'docs/audits/fxfx_delta_wz_backward_isr')
    args = parser.parse_args(argv)
    for name in ('historical_audit', 'historical_steering'):
        if len(getattr(args, name)) not in (0, len(args.historical)):
            parser.error(f'--{name.replace("_", "-")} must be supplied once per --historical')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.setup:
        setup = json.loads(args.setup.read_text())
        if args.setup.resolve() != (args.out / 'setup.json').resolve():
            (args.out / 'setup.json').write_text(json.dumps(setup, indent=2) + '\n')
    runs, histograms = {}, {}
    inputs = [('reference', args.reference_process, args.reference_run, args.reference_label, None, None),
              ('new', args.new_process, args.new_run, args.new_label, None, None)]
    for i, (process, run, label) in enumerate(args.historical):
        saved_audit = args.historical_audit[i] if args.historical_audit else None
        saved_steering = args.historical_steering[i] if args.historical_steering else None
        inputs.append((f'historical_{i+1}', Path(process), run, label, saved_audit, saved_steering))
    for key, process, run, label, saved_audit, saved_steering in inputs:
        print(f'Auditing {label}: {process}/Events/{run}', flush=True)
        runs[key], histograms[key] = audit_run(process, run, label, args.expected_events,
                                             saved_audit, saved_steering)
    import shlex
    command = shlex.join([sys.executable, 'fxfx_diagnostics/compare_wz_backward_isr.py',
                          *(sys.argv[1:] if argv is None else argv)])
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'runs': runs, 'rate_changes': {}, 'steering_differences': {},
              'uncertainty_note': UNCERTAINTY_NOTE,
              'reproduction_command': command}
    for key in runs:
        if key == 'new':
            continue
        result['rate_changes'][key] = changes(runs[key], runs['new'])
        result['steering_differences'][key] = steering_differences(runs[key], runs['new'])
    result['binned_integrals'] = compare_bins(histograms, args.out)
    result['assessment'] = assess_comparison(result)
    (args.out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    plot_comparison(runs, histograms, ['reference'], args.out)
    historical = [key for key in runs if key not in ('reference', 'new')]
    if historical:
        plot_comparison(runs, histograms, historical, args.out, prefix='historical_')
    write_report(result, args.out)
    print(json.dumps({'rates': result['rate_changes'],
                      'primary_steering_differences': result['steering_differences']['reference']}, indent=2))


if __name__ == '__main__':
    main()
