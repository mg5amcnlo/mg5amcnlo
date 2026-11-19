#!/usr/bin/env python3
"""Analyse completed G/P/S/POWHEG samples and make the manuscript figures.

Run from any directory after run_all.py. Corrected central/G/P/S exports are
required by default; --allow-partial reports only completed corrected samples.
Production cards, events and running jobs are read only.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from madgraph.various.histograms import HwUList
from Analyses.matched_run_inputs import CENTRAL_RUNS, check_run, find_input, source_name

RATE = {'ALL': 'total rate [pb]', 'TAG': 'fiducial rate [pb]',
        'GAP': 'fiducial rate [pb]'}
SEVEN = [(1., 1.), (.5, .5), (.5, 1.), (1., .5),
         (1., 2.), (2., 1.), (2., 2.)]
VARIATION_RUNS = {
    'G': ['G_LOC_H005', 'G_LOC_H020', 'G_Q_H005', 'G_Q_H020'],
    'P': ['P_Q_COMMON_W', 'P_QT_COMMON_W'],
    'S': [f'S_{variant}_Q{q}' for variant in ['GLB', 'LOC', 'Q', 'QT']
          for q in ['0P5', '1', '2']],
}


def fraction(n, en, d, ed):
    """Subset ratio, including Cov(n,d)=Var(n), for signed weights."""
    n, en, d, ed = np.broadcast_arrays(n, en, d, ed)
    if np.any(d <= 0):
        raise ValueError('Nonpositive denominator')
    if np.any(en**2 > ed**2 * (1 + 2e-5)):
        raise ValueError('Numerator is not a variance subset of denominator')
    f = n / d
    v = ((1 - f)**2 * en**2 + f**2 * np.maximum(ed**2 - en**2, 0)) / d**2
    return f, np.sqrt(v)


def independent(a, ea, b, eb):
    # A relative change is not useful for a zero/negative or statistically
    # unresolved reference bin. Its signed difference remains meaningful.
    valid = b > 3 * eb
    ratio = float(a / b) if valid else None
    error = float(np.hypot(ea / b, a * eb / b**2)) if valid else None
    difference_error = float(np.hypot(ea, eb))
    return {'ratio': ratio, 'ratio_error': error,
            'difference': float(a - b), 'difference_error': difference_error,
            'pull': float((a - b) / difference_error) if difference_error else None}


class Sample:
    def __init__(self, name, directory, run):
        self.name, self.directory, self.run = name, directory, run
        check_run(directory, run)
        self.events = directory / 'Events' / run
        self.path = find_input(directory, run)
        self.hists = {}
        for hist in HwUList(str(self.path)):
            if hist.type == 'AUX':
                continue
            self.hists[hist.type, hist.title] = (
                np.array([b.boundaries[0] for b in hist.bins] + [hist.bins[-1].boundaries[1]]),
                {k: np.array([b.wgts[k] for b in hist.bins]) for k in hist.bins[0].wgts})
        if len(self.hists) != 75:
            raise ValueError(f'{name}: expected 75 physical histograms')
        for edges, weights in self.hists.values():
            if not np.all(np.diff(edges) > 0) or any(not np.all(np.isfinite(v)) for v in weights.values()):
                raise ValueError(f'{name}: invalid bins or weights')

    def hist(self, selection, title, weight='central'):
        edges, values = self.hists[selection, title]
        return edges, values[weight], values['stat_error']

    def rate(self, selection, weight='central'):
        _, values, errors = self.hist(selection, RATE[selection], weight)
        return values[0], errors[0]

    def interval(self, selection, title, lo, hi):
        edges, values, errors = self.hist(selection, title)
        if not any(np.isclose(edges, lo)) or not any(np.isclose(edges, hi)):
            raise ValueError(f'{title}: interval must end at stored bin edges')
        mask = (edges[:-1] >= lo - 1e-8) & (edges[1:] <= hi + 1e-8)
        return values[mask].sum(), np.linalg.norm(errors[mask])

    def curve(self, selection, title, normalized=False, veto=False, weight='central'):
        if veto:
            edges, n, en = self.hist(selection, 'gap veto survived Q0 [GeV]', weight)
            _, d, ed = self.hist(selection, 'gap veto reference Q0 [GeV]', weight)
            # Only central weights have per-bin statistical errors in the HwU.
            values, errors = fraction(n, en, d, ed) if weight == 'central' else (n / d, np.zeros_like(n))
        else:
            edges, values, errors = self.hist(selection, title, weight)
            if normalized:
                d, ed = self.rate(selection, weight)
                values, errors = fraction(values, errors, d, ed) if weight == 'central' else (values / d, np.zeros_like(values))
            values, errors = values / np.diff(edges), errors / np.diff(edges)
        return edges, values, errors

    def metrics(self):
        metrics = {s: self.rate(s) for s in RATE}
        for selection in ['TAG', 'GAP']:
            metrics['A_' + selection] = fraction(*self.rate(selection), *self.rate('ALL'))
            for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (5, 10), (20, 50)]:
                metrics[f'{selection}_recoil_{lo}_{hi}'] = self.interval(selection, 'pT(t+j1) low [GeV]', lo, hi)
            for lo, hi in [(-.5, .5), (-1., -.5), (.5, 1.)]:
                metrics[f'{selection}_zstar_{lo}_{hi}'] = self.interval(selection, 'j2 zstar (dy>1)', lo, hi)
            edges, values, errors = self.curve(selection, '', veto=True)
            for q0 in [21, 23, 25, 27, 29, 41, 61]:
                i = np.flatnonzero(np.isclose((edges[1:] + edges[:-1]) / 2, q0))[0]
                metrics[f'{selection}_veto_{q0}'] = values[i], errors[i]
        metrics['top_pT_200_400'] = self.interval('ALL', 'top pT [GeV]', 200, 400)
        metrics['GAP_beyond_j1_0_0.96'] = self.interval('GAP', 'j2 dy from j1 (oriented)', 0, .96)
        metrics['GAP_beyond_top_-1.92_0'] = self.interval('GAP', 'j2 dy from top (oriented)', -1.92, 0)
        for key, (value, error) in list(metrics.items()):
            selection = key.split('_')[0]
            if selection in ['TAG', 'GAP'] and '_' in key and '_veto_' not in key:
                metrics['fraction_' + key] = fraction(value, error, *self.rate(selection))
        metrics['fraction_top_pT_200_400'] = fraction(*metrics['top_pT_200_400'], *self.rate('ALL'))
        return {k: {'value': float(v), 'error': float(e)} for k, (v, e) in metrics.items()}


def discover(central_dir, g_dir, scale_dir, powheg_dir, require_complete=True):
    definitions = [(name, central_dir / run, run, 'C')
                   for name, run in CENTRAL_RUNS.items()]
    definitions += [(name, (scale_dir if group == 'S' else g_dir) / name, name, group)
                    for group, names in VARIATION_RUNS.items() for name in names]
    missing = [str(directory / 'Events' / run / 'campaign_done.json')
               for _, directory, run, _ in definitions
               if not (directory / 'Events' / run / 'campaign_done.json').is_file()]
    if require_complete and missing:
        raise ValueError('Corrected campaign is incomplete. Missing completion markers:\n  '
                         + '\n  '.join(missing)
                         + '\nUse --allow-partial only to inspect finished subsets.')
    samples, complete = {}, {'C': [], 'G': [], 'P': [], 'S': []}
    for name, directory, run, group in definitions:
        if not (directory / 'Events' / run / 'campaign_done.json').is_file():
            continue
        samples[name] = Sample(name, directory, run)
        complete[group].append(run)
    if require_complete or (powheg_dir / 'Events/run_01/plot_PYTHIA8_1_0.HwU').is_file():
        samples['powheg'] = Sample('powheg', powheg_dir, 'run_01')
    if not samples:
        raise ValueError('No completed corrected samples found')
    return samples, complete, missing


def hard_envelope(sample, selection, title, normalized=False, veto=False):
    _, weights = sample.hists[selection, 'gap veto survived Q0 [GeV]' if veto else title]
    keys = {tuple(k[-2:]): k for k in weights if isinstance(k, tuple) and k[0] in ('scale', 'scale_adv')}
    if not set(SEVEN) <= keys.keys():
        raise ValueError('Missing seven-point weights')
    for edges, values in sample.hists.values():
        # LHE nominal and reweight entries have different printed precision;
        # signed cancellations can amplify their relative difference near zero.
        tolerance = 2e-4 * np.maximum(abs(values['central']), values['stat_error'])
        if np.any(abs(values[keys[1., 1.]] - values['central']) > tolerance + 1e-12):
            raise ValueError('Central hard-scale weight is inconsistent')
    curves = [sample.curve(selection, title, normalized, veto,
                           weight='central' if pair == (1., 1.) else keys[pair])[1] for pair in SEVEN]
    return np.min(curves, axis=0), np.max(curves, axis=0)


def matched_analysis_count(directory, generated, run_spec):
    """Account for the current MG5 driver's final unanalysed event per job."""
    events = int(run_spec.get('nevents', generated))
    if events != generated:
        raise ValueError(f'{directory}: generated count disagrees with run_spec.json')
    card = (directory / 'Cards/shower_card.dat').read_text()
    settings = re.findall(r'^\s*nsplit_jobs\s*=\s*([+-]?\d+)\b', card, re.M | re.I)
    jobs = int(settings[-1]) if settings else 1
    if jobs < 1 or events % jobs or events <= jobs:
        raise ValueError(f'{directory}: invalid shower split count for {events} events')
    return events, jobs, events - jobs


def report(samples, complete):
    result = {'completed': complete, 'samples': {}, 'comparisons': {}, 'scale_envelopes': {},
              'error_convention': 'Signed squared-weight statistical errors. Subset ratios include Cov(N,D)=Var(N). Different samples are independent; integration-normalisation errors are quoted separately.',
              'normalization': 'Differential comparisons use the ALL/TAG/GAP cross section of each sample, including covariance with the selected bin. No rescaling is applied to absolute rates.',
              'rapidity_intervals': 'Use exact stored edges: 0 < s*(y_j2-y_j1) < 0.96 and -1.92 < s*(y_j2-y_t) < 0. Never split a bin.'}
    for name, sample in samples.items():
        metrics = sample.metrics()
        record = {'input': source_name(sample.path),
                  'sha256': hashlib.sha256(sample.path.read_bytes()).hexdigest(), 'metrics': metrics}
        completed = sample.events / 'campaign_done.json'
        if completed.is_file():
            metadata = json.loads(completed.read_text())
            generated_inputs = metadata.get('generation_inputs', metadata['inputs'])
            record['normalization_fix'] = generated_inputs['normalization_fix']
            record['numerical_fix'] = generated_inputs['numerical_fix']
            record['matching_source_sha256'] = generated_inputs['source_sha256']
            if 'source_migration_sha256' in metadata:
                record['source_migration_sha256'] = metadata['source_migration_sha256']
                record['compatible_current_source_sha256'] = metadata['inputs']['source_sha256']
            record['completion_record_sha256'] = hashlib.sha256(completed.read_bytes()).hexdigest()
        spec_path = sample.directory / 'run_spec.json'
        if spec_path.exists():
            record['run_spec'] = json.loads(spec_path.read_text())
            fks_path = sample.directory / 'Cards/FKS_params.dat'
            fks = fks_path.read_text()
            record['fks_settings'] = {}
            for key in ['FKSPythiaLocalRecoil', 'PythiaLocalGWidth', 'PythiaLocalGProfile',
                        'PythiaLocalCollinearPatch', 'PythiaLocalSoftPatch']:
                actual = re.search(r'^#' + key + r'\s*\n\s*(\S+)', fks, re.M)
                if actual:
                    record['fks_settings'][key] = actual[1]
            expected_profile = '1' if record['run_spec'].get('profile') == 'w_forced' else '0'
            if record['fks_settings'].get('PythiaLocalGProfile', '0') != expected_profile:
                raise ValueError(f'{sample.directory}: G-profile card disagrees with run_spec.json')
            record['fks_card_sha256'] = hashlib.sha256(fks_path.read_bytes()).hexdigest()
        summary = sample.events / 'summary.txt'
        if summary.exists():
            text = summary.read_text()
            match = re.search(r'Total cross section:\s*([\deE+.-]+)\s*\+-\s*([\deE+.-]+)', text)
            record['integration_sigma_pb'], record['integration_error_pb'] = map(float, match.groups())
            generated = int(re.search(r'Number of events generated:\s*(\d+)', text)[1])
            events, jobs, analysed = matched_analysis_count(
                sample.directory, generated, record.get('run_spec', {}))
            record.update(generated=events, shower_jobs=jobs, analysed=analysed)
            # Every nominal LHE event has the same absolute weight. This is the
            # same inference as eq. (neff), using the actual analysis-call count.
            total = metrics['ALL']
            record['negative_fraction_inferred'] = .5 * (1 - total['value'] / (total['error'] * np.sqrt(analysed)))
        else:
            info = json.loads((sample.events / 'summary.json').read_text())
            record.update(generated=info['events'], analysed=info['analysed'],
                          negative_fraction=info['negative'] / info['analysed'])
        result['samples'][name] = record
    pairs = [(n, ref) for n, ref in [
        ('G_LOC_H005', 'local'), ('G_LOC_H020', 'local'),
        ('G_Q_H005', 'local_patch'), ('G_Q_H020', 'local_patch'),
        ('P_Q_COMMON_W', 'local'), ('P_QT_COMMON_W', 'P_Q_COMMON_W'),
        ('P_Q_COMMON_W', 'local_patch'), ('P_QT_COMMON_W', 'local_2patch'),
        ('local_patch', 'local'), ('local_2patch', 'local_patch'),
        ('powheg', 'local')]]
    for a, b in pairs:
        if a not in samples or b not in samples:
            continue
        ma, mb = result['samples'][a]['metrics'], result['samples'][b]['metrics']
        result['comparisons'][a + '/' + b] = {
            k: independent(v['value'], v['error'], mb[k]['value'], mb[k]['error']) for k, v in ma.items()}
    for variant in ['GLB', 'LOC', 'Q', 'QT']:
        names = [f'S_{variant}_Q{x}' for x in ['0P5', '1', '2']]
        if not all(n in samples for n in names):
            continue
        central = samples[names[1]]
        entry = {}
        for selection in ['ALL', 'TAG', 'GAP']:
            edges, weights = central.hists[selection, RATE[selection]]
            lookup = {tuple(k[-2:]): k for k in weights if isinstance(k, tuple) and k[0] in ('scale', 'scale_adv')}
            hard = [central.rate(selection, 'central' if pair == (1., 1.) else lookup[pair])[0] for pair in SEVEN]
            shower = [samples[n].rate(selection)[0] for n in names]
            grid = [samples[n].rate(selection, 'central' if pair == (1., 1.) else lookup[pair])[0]
                    for n in names for pair in SEVEN]
            entry[selection] = {'central': central.rate(selection)[0],
                                'hard_seven_point': list(map(float, hard)),
                                'shower_scales_0p5_1_2': list(map(float, shower)),
                                'hard_envelope': [float(min(hard)), float(max(hard))],
                                'shower_envelope': [float(min(shower)), float(max(shower))],
                                'combined_3_by_7': [float(min(grid)), float(max(grid))]}
        result['scale_envelopes'][variant] = entry
    return result


def write_tables(result):
    output = ROOT / 'paper/tables'
    output.mkdir(exist_ok=True)
    def value(v, e, factor=1., places=2):
        if v is None or e is None:
            return r'---'
        return f'${v * factor:.{places}f}\\pm{e * factor:.{places}f}$'
    def table(filename, columns, header, rows, caption, label):
        content = [r'\begin{table}[t]', r'  \centering',
                   r'  \begin{tabular}{@{}' + columns + r'@{}}',
                   r'    \toprule', '    ' + header + r'\\', r'    \midrule']
        content += ['    ' + ' & '.join(row) + r'\\' if row else r'    \midrule' for row in rows]
        content += [r'    \bottomrule', r'  \end{tabular}',
                    r'  \caption{' + caption + '}', r'  \label{' + label + '}', r'\end{table}']
        (output / filename).write_text('% Generated by analyse_matched_runs.py\n' + '\n'.join(content) + '\n')
    if {'local', 'local_patch', 'G_LOC_H005', 'G_LOC_H020', 'G_Q_H005', 'G_Q_H020'} <= result['samples'].keys():
        rows = []
        for prediction, names in [(r'\loc', ['G_LOC_H005', 'local', 'G_LOC_H020']),
                                  (r'\locq', ['G_Q_H005', 'local_patch', 'G_Q_H020'])]:
            if rows:
                rows.append(None)
            for i, (h, name) in enumerate(zip(['0.005', '0.010', '0.020'], names)):
                m = result['samples'][name]['metrics']
                rows.append([prediction if i == 0 else '', '$' + h + '$'] + [
                    value(m[k]['value'], m[k]['error'], factor) for k, factor in [
                        ('ALL', 1), ('A_TAG', 100), ('A_GAP', 100), ('fraction_GAP_recoil_0_5', 100)]])
        table('g_width.tex', 'lccccc',
              r'Prediction & $h$ & $\sigma_{\rm ALL}$ [pb] & $A_{\rm TAG}$ [\%] & $A_{\rm GAP}$ [\%] & $F_{<5}$ [\%]', rows,
              r'G-function width dependence with the production setup. Each row uses $10^6$ independent LHE events. '
              r'$A_{\rm TAG,GAP}=\sigma_{\rm TAG,GAP}/\sigma_{\rm ALL}$ and $F_{<5}$ is defined in eq.~\eqref{eq:lowrecoilfrac}. '
              r'The uncertainties are statistical; subset fractions include their numerator--denominator covariance. '
              r'The integration uncertainty of the absolute normalisation is additional.', 'tab:gwidth')
    if {'local', 'local_patch', 'local_2patch', 'P_Q_COMMON_W', 'P_QT_COMMON_W'} <= result['samples'].keys():
        rows = []
        for correction, quantity, metric, native, common in [
            (r'$r_q$', r'beyond $j_1$', 'fraction_GAP_beyond_j1_0_0.96', 'local_patch/local', 'P_Q_COMMON_W/local'),
            (r'$r_t$', r'beyond $t$', 'fraction_GAP_beyond_top_-1.92_0', 'local_2patch/local_patch', 'P_QT_COMMON_W/P_Q_COMMON_W'),
            (r'$r_q$', r'$F_{<5}$', 'fraction_GAP_recoil_0_5', 'local_patch/local', 'P_Q_COMMON_W/local'),
            (r'$r_t$', r'$F_{<5}$', 'fraction_GAP_recoil_0_5', 'local_2patch/local_patch', 'P_QT_COMMON_W/P_Q_COMMON_W')]:
            rows.append([correction, quantity] + [
                value(result['comparisons'][key][metric]['ratio'],
                      result['comparisons'][key][metric]['ratio_error'], places=3) for key in [native, common]])
        table('common_g.tex', 'llcc', r'Correction & Quantity & Production profiles & Common $g_h(w)$', rows,
              r'Ratios after/before the indicated kernel correction, with each rate normalised to $\sigma_{\rm GAP}$. '
              r'The production profiles are $g_h(w)$, $g_h(\xi)$ and no G functions for \loc, \locq and \locqt, respectively. '
              r'The common-profile comparison uses $g_h(w)$ and the same angular replacement in all three samples, with $h=0.010$. '
              r'The rapidity windows are $0<s[y(j_2)-y(j_1)]<0.96$ and $-1.92<s[y(j_2)-y(t)]<0$, using exact histogram edges. '
              r'Errors are statistical and samples are independent.', 'tab:commong')
    if {'local', 'powheg'} <= result['samples'].keys():
        rows = []
        for label, key, factor, places in [
            ('ALL [pb]', 'ALL', 1, 2), ('TAG [pb]', 'TAG', 1, 2), ('GAP [pb]', 'GAP', 1, 2),
            (r'$A_{\rm TAG}$ [\%]', 'A_TAG', 100, 2), (r'$A_{\rm GAP}$ [\%]', 'A_GAP', 100, 2),
            (r'$F_{<5}$ [\%]', 'fraction_GAP_recoil_0_5', 100, 2),
            (r'$\sigma_{\rm GAP}(|z^*_{j_2}|<1/2)/\sigma_{\rm GAP}$ [\%]', 'fraction_GAP_zstar_-0.5_0.5', 100, 2),
            (r'$f_{\rm gap}(21\GeV)$', 'GAP_veto_21', 1, 4)]:
            rows.append([label] + [value(result['samples'][name]['metrics'][key]['value'],
                                        result['samples'][name]['metrics'][key]['error'], factor, places)
                                   for name in ['local', 'powheg']])
        table('powheg.tex', 'lcc', r'Quantity & MC@NLO, \loc & POWHEG, local recoil', rows,
              r'Comparison with POWHEG showered with unmodified \pythia and local recoil. '
              r'Both samples contain $10^6$ events and use the same hard-process parameters and analysis. '
              r'Errors are signed-weight statistical errors; integration-normalisation uncertainties are additional.', 'tab:powheg')
    rows = []
    for variant, prediction in [('GLB', r'\glob'), ('LOC', r'\loc')]:
        if variant not in result['scale_envelopes']:
            continue
        for selection in ['ALL', 'TAG', 'GAP']:
            entry = result['scale_envelopes'][variant][selection]
            rate = result['samples'][f'S_{variant}_Q1']['metrics'][selection]
            envelopes = []
            for kind in ['hard_envelope', 'shower_envelope']:
                low, high = entry[kind]
                envelopes.append(f'$_{{-{100 * (1 - low / rate["value"]):.2f}}}^{{+{100 * (high / rate["value"] - 1):.2f}}}$')
            rows.append([prediction if selection == 'ALL' else '', selection,
                         value(rate['value'], rate['error'])] + envelopes)
        if variant == 'GLB':
            rows.append(None)
    if rows:
        table('scale_rates.tex', 'llccc',
              r'Prediction & Selection & $\sigma$ [pb] & Hard scales [\%] & Starting scale [\%]', rows,
              r'Completed scale variations for \glob and \loc. Central results are from the new, independent scale-weighted samples. '
              r'Hard-scale bands are seven-point envelopes at central starting scale; starting-scale bands use factors $1/2$, $1$, $2$ at central hard scales. '
              r'The quoted errors on central rates are statistical. The envelope widths are not statistical errors, and independently integrated starting-scale samples also carry integration noise.', 'tab:scales')


def plots(samples, output):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    mpl.rcParams.update({'font.family': 'serif', 'font.size': 11, 'mathtext.fontset': 'cm'})
    def panel(axes, names, labels, colors, selection, title, xlabel, xlim,
              normalized=True, veto=False, ylim=None, difference=False, envelope=None):
        top, bottom = axes
        reference = samples[names[0]].curve(selection, title, normalized, veto)
        edges, ref, eref = reference
        bottom.axhline(0 if difference else 1, color='0.4', lw=.7)
        def band(ax, lo, hi, color, alpha=.17):
            ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]], step='post', color=color, alpha=alpha, lw=0)
        valid = ref > 0
        if difference:
            band(bottom, -eref, eref, 'gray', .3)
        else:
            relerr = np.divide(eref, ref, out=np.full_like(ref, np.nan), where=valid)
            band(bottom, 1 - relerr, 1 + relerr, 'gray', .3)
        minimum = 0.
        for name, label, color in zip(names, labels, colors):
            e, y, dy = samples[name].curve(selection, title, normalized, veto)
            assert np.array_equal(e, edges)
            shown = (edges[:-1] >= xlim[0]) & (edges[1:] <= xlim[1])
            minimum = min(minimum, np.min((y - dy)[shown]))
            top.stairs(y, edges, color=color, label=label, lw=1.2)
            if envelope:
                if envelope == 'hard':
                    low, high = hard_envelope(samples[name], selection, title, normalized, veto)
                else:
                    variant = name.split('_')[1]
                    variations = [samples[f'S_{variant}_Q{q}'].curve(selection, title, normalized, veto)[1]
                                  for q in ['0P5', '1', '2']]
                    low, high = np.min(variations, axis=0), np.max(variations, axis=0)
                minimum = min(minimum, np.min(low[shown]))
                band(top, low, high, color)
                low_ratio = np.divide(low, ref, out=np.full_like(low, np.nan), where=valid)
                high_ratio = np.divide(high, ref, out=np.full_like(high, np.nan), where=valid)
                band(bottom, low_ratio, high_ratio, color)
                top.errorbar((edges[1:] + edges[:-1]) / 2, y, yerr=dy, fmt='none',
                             color=color, lw=.5, alpha=.6)
            else:
                band(top, y - dy, y + dy, color)
            if name != names[0]:
                ratio = y - ref if difference else np.divide(y, ref, out=np.full_like(y, np.nan), where=valid)
                error = dy if difference else np.divide(dy, ref, out=np.full_like(y, np.nan), where=valid)
                bottom.stairs(ratio, edges, color=color, lw=1.2)
                if not envelope:
                    band(bottom, ratio - error, ratio + error, color)
        top.set_ylabel('gap fraction' if veto else r'$\sigma_{\rm sel}^{-1}\,d\sigma/dx$' if normalized else r'$d\sigma/dx$ [pb/unit]')
        top.set_title(selection)
        top.legend(frameon=False, fontsize=9)
        top.set_ylim(bottom=1.2 * minimum if minimum < 0 else 0 if not veto else .75)
        top.axhline(0, color='0.6', lw=.6)
        bottom.set_ylabel(r'$\Delta(\sigma_{\rm sel}^{-1}d\sigma/dx)$' if difference else 'ratio')
        bottom.set_xlabel(xlabel)
        if ylim:
            bottom.set_ylim(*ylim)
        elif not difference:
            bottom.set_ylim(.75, 1.25)
        bottom.set_xlim(*xlim)
        for ax in axes:
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.yaxis.set_minor_locator(AutoMinorLocator())
    def pair(out, panels):
        fig, ax = plt.subplots(2, 2, figsize=(10.4, 5), sharex='col',
                               gridspec_kw={'height_ratios': [2.5, 1], 'hspace': .06, 'wspace': .28})
        for i, cfg in enumerate(panels): panel(ax[:, i], **cfg)
        fig.savefig(output / out, bbox_inches='tight')
        plt.close(fig)
    colors = ['black', '#2a5db0', '#c40524']
    width = [('local', 'G_LOC_H005', 'G_LOC_H020'), ('local_patch', 'G_Q_H005', 'G_Q_H020')]
    if all(name in samples for names in width for name in names):
        for suffix, title, xlabel, xlim, veto, ylim in [
            ('recoil', 'pT(t+j1) low [GeV]', r'$p_T(t+j_1)$ [GeV]', (0, 10), False, None),
            ('zstar', 'j2 zstar (dy>1)', r'$z^*_{j_2}$', (-1.5, 1.2), False, (.8, 1.2)),
            ('veto', '', r'$Q_0$ [GeV]', (20, 50), True, (.985, 1.015))]:
            cfgs = [dict(names=list(n), labels=[r'$h=0.010$', r'$h=0.005$', r'$h=0.020$'], colors=colors,
                         selection='GAP', title=title, xlabel=xlabel, xlim=xlim, veto=veto, ylim=ylim,
                         difference=suffix == 'recoil') for n in width]
            pair('g_width_' + suffix + '.pdf', cfgs)
    if {'local', 'local_patch', 'local_2patch', 'P_Q_COMMON_W', 'P_QT_COMMON_W'} <= samples.keys():
        common = ['local', 'P_Q_COMMON_W', 'P_QT_COMMON_W']
        labels = [r'local, $g_h(w)$', r'local $\times r_q$, $g_h(w)$', r'local $\times r_qr_t$, $g_h(w)$']
        pair('common_g_rapidity.pdf', [dict(names=common, labels=labels, colors=colors, selection='GAP',
            title=title, xlabel=xlabel, xlim=xlim, ylim=(.7, 1.2)) for title, xlabel, xlim in [
            ('j2 dy from j1 (oriented)', r'$s[y(j_2)-y(j_1)]$', (-3, 2)),
            ('j2 dy from top (oriented)', r'$s[y(j_2)-y(t)]$', (-3, 4))]])
        pair('common_g_recoil.pdf', [dict(names=[native, forced], labels=labels,
            colors=['black', '#c40524'], selection='GAP', title='pT(t+j1) low [GeV]',
            xlabel=r'$p_T(t+j_1)$ [GeV]', xlim=(0, 10), difference=True) for native, forced, labels in [
            ('local_patch', 'P_Q_COMMON_W', [r'$r_q$, $g_h(\xi)$', r'$r_q$, $g_h(w)$']),
            ('local_2patch', 'P_QT_COMMON_W', [r'$r_qr_t$, no G', r'$r_qr_t$, $g_h(w)$'])]])
    if {'local', 'powheg'} <= samples.keys():
        for out, configs in [
            ('powheg_standard.pdf', [('ALL', 'top pT [GeV]', r'$p_T(t)$ [GeV]', (0, 400)),
                                     ('TAG', 'j1 y', r'$y(j_1)$', (-4.5, 4.5))]),
            ('powheg_radiation.pdf', [('GAP', 'j2 zstar (dy>1)', r'$z^*_{j_2}$', (-1.5, 1.2)),
                                      ('GAP', '', r'$Q_0$ [GeV]', (20, 100))])]:
            pair(out, [dict(names=['local', 'powheg'], labels=['MC@NLO, local', 'POWHEG, local'],
                           colors=['black', '#a06010'], selection=sel, title=title, xlabel=xlabel,
                           xlim=xlim, veto=not title, ylim=(.94, 1.04) if not title else (.7, 1.4))
                       for sel, title, xlabel, xlim in configs])
    if all(f'S_{v}_Q{q}' in samples for v in ['GLB', 'LOC'] for q in ['0P5', '1', '2']):
        for kind, cfgs in [('hard', [
            ('ALL', 'top pT [GeV]', r'$p_T(t)$ [GeV]', (0, 400), (.8, 1.2)),
            ('GAP', '', r'$Q_0$ [GeV]', (20, 100), (.98, 1.15))]), ('shower', [
            ('GAP', 'pT(t+j1) low [GeV]', r'$p_T(t+j_1)$ [GeV]', (0, 10), (-.5, 2.6)),
            ('GAP', '', r'$Q_0$ [GeV]', (20, 100), (.85, 1.25))])]:
            pair('scale_' + kind + '.pdf', [dict(
                names=['S_GLB_Q1', 'S_LOC_Q1'], labels=['global', 'local'], colors=['black', '#2a5db0'],
                selection=sel, title=title, xlabel=xlabel, xlim=xlim, veto=not title, ylim=ylim,
                envelope=kind) for sel, title, xlabel, xlim, ylim in cfgs])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--central-dir', type=Path, default=ROOT / 'central_runs_fixed')
    parser.add_argument('--g-dir', type=Path, default=ROOT / 'g_runs_fixed')
    parser.add_argument('--scale-dir', type=Path, default=ROOT / 'scale_runs_fixed')
    parser.add_argument('--powheg-dir', type=Path, default=ROOT / 'pp_tj_powheg')
    completion = parser.add_mutually_exclusive_group()
    completion.add_argument('--require-complete', action='store_true', dest='require_complete',
                            help='require all 22 corrected matched samples (default)')
    completion.add_argument('--allow-partial', action='store_false', dest='require_complete',
                            help='inspect finished samples; omit unavailable comparisons and envelopes')
    parser.set_defaults(require_complete=True)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    try:
        samples, complete, missing = discover(args.central_dir.resolve(), args.g_dir.resolve(),
                                              args.scale_dir.resolve(), args.powheg_dir.resolve(),
                                              args.require_complete)
        result = report(samples, complete)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + '\n')
    result['incomplete_samples'] = missing
    result['input_directories'] = {key: source_name(path) for key, path in [
        ('central', args.central_dir), ('g', args.g_dir),
        ('scales', args.scale_dir), ('powheg', args.powheg_dir)]}
    (ROOT / 'paper/matched_results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    write_tables(result)
    if not args.no_plots:
        plots(samples, ROOT / 'paper/figures')
    print('Completed samples:', complete)
    if missing:
        print('Partial campaign: unavailable comparisons, tables and figures were omitted; '
              'previous files for those outputs have not been refreshed.')
    for key, values in result['comparisons'].items():
        print(key, {k: {n: round(v[n], 5) if v[n] is not None else None
                       for n in ['ratio', 'ratio_error', 'pull']}
                    for k, v in values.items() if k in ['ALL', 'A_TAG', 'A_GAP',
                    'fraction_GAP_recoil_0_5', 'fraction_GAP_zstar_-0.5_0.5',
                    'GAP_beyond_j1_0_0.96', 'GAP_beyond_top_-1.92_0', 'GAP_veto_21']})


if __name__ == '__main__':
    main()
