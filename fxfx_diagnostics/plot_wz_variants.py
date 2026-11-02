#!/usr/bin/env python3
"""Plot every saved WZ shower spectrum for the four matching prescriptions."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fxfx_diagnostics.compare_wz_h_scales import read_histograms


VARIANTS = (
    ('historical_2', 'Original hard scale (historical)', '#777777', '--'),
    ('historical_1', 'Geometric mean (historical)', '#d89000', ':'),
    ('reference', 'Geometric mean / ISR off (fresh)', '#202124', '-'),
    ('new', 'Geometric mean / ISR on (fresh)', '#6f42a1', '-'),
)
OBSERVABLES = (
    ('pT W+Z [GeV]', r'$p_T(W^+Z)$ [GeV]', 'pt_wz', False),
    ('log10 pT W+Z/GeV', r'$\log_{10}(p_T(W^+Z)/\mathrm{GeV})$', 'log_pt_wz', True),
    ('pT j1 [GeV]', r'$p_T(j_1)$ [GeV]', 'pt_j1', False),
    ('log10 pT j1/GeV', r'$\log_{10}(p_T(j_1)/\mathrm{GeV})$', 'log_pt_j1', True),
    ('sqrt d01 [GeV]', r'$\sqrt{d_{01}}$ [GeV]', 'djr01', False),
    ('log10 sqrt d01/GeV', r'$\log_{10}(\sqrt{d_{01}}/\mathrm{GeV})$', 'log_djr01', True),
)
CATEGORIES = (('SUM', 'Merged'), ('0', 'Sample @0'), ('1', 'Sample @1'))
PAIRS = (
    ('historical_1', 'historical_2', 'Historical geometric / hard', '#d89000', '--'),
    ('new', 'reference', 'Fresh ISR on / off', '#6f42a1', '-'),
)
FOOTER = (
    '13 TeV; Qcut=70 GeV; stable W/Z; Pythia 8.317; one million events per prescription.\n'
    'Bands: denominator error; bars: numerator error / denominator. '
    'Ratios require denominator > 3 statistical errors. No cross-run/bin covariance.\n'
    'Errors condition on source normalization; integration-normalization errors are separate. '
    'Historical and fresh shower steering differ (charged-resonance QED).'
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binned(histogram, factor=1):
    edges = np.array([histogram.bins[0].boundaries[0]] +
                     [item.boundaries[1] for item in histogram.bins])
    values = np.array([item.get_weight('central') for item in histogram.bins])
    errors = np.array([item.get_weight('stat_error') for item in histogram.bins])
    if factor > 1:
        if len(values) % factor:
            raise ValueError('Rebinning must preserve every original bin')
        values = values.reshape(-1, factor).sum(axis=1)
        errors = np.sqrt((errors.reshape(-1, factor)**2).sum(axis=1))
        edges = edges[::factor]
    return edges, values, errors


def ratio(numerator, denominator):
    _, a, ae = numerator
    _, b, be = denominator
    valid = (b > 0.) & (b > 3.*be)
    values = np.divide(a, b, out=np.full_like(a, np.nan), where=valid)
    numerator_error = np.divide(ae, b, out=np.full_like(ae, np.nan), where=valid)
    denominator_error = np.divide(be, b, out=np.full_like(be, np.nan), where=valid)
    return values, numerator_error, denominator_error, valid


def draw_ratio(ax, numerator, denominator, color, style='-', label=None, band=True):
    edges = numerator[0]
    centers = .5*(edges[:-1] + edges[1:])
    values, error, reference_error, valid = ratio(numerator, denominator)
    ax.stairs(values, edges, color=color, linestyle=style, linewidth=1.15, label=label)
    ax.errorbar(centers[valid], values[valid], yerr=error[valid],
                fmt='none', color=color, alpha=.45, linewidth=.55)
    if band:
        lower = np.where(valid, 1.-reference_error, np.nan)
        upper = np.where(valid, 1.+reference_error, np.nan)
        ax.fill_between(centers, lower, upper, step='mid', color=color, alpha=.09)
    return values, error, valid


def limits(ax, series):
    """Keep every defined ratio visible, including sparse component tails."""
    arrays = [item[np.isfinite(item)] for item in series]
    values = np.concatenate(arrays) if arrays else np.array([])
    if values.size:
        low, high = min(.9, float(values.min())), max(1.1, float(values.max()))
        padding = max(.025, .08*(high-low))
        ax.set_ylim(low-padding, high+padding)
    else:
        ax.set_ylim(.8, 1.2)
        ax.text(.5, .5, 'No resolved denominator bins', transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='#666666')


def signed_ticks(ax, low, high, threshold):
    first = int(np.ceil(np.log10(threshold)))
    last = int(np.ceil(np.log10(max(abs(low), abs(high), threshold))))
    positive = [10.**power for power in range(first, last+1) if 10.**power <= high]
    negative = [-10.**power for power in range(first, last+1) if -10.**power >= low]
    ax.set_yticks(sorted(negative)+[0.]+positive)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: '0' if value == 0. else
        ('$' + ('-' if value < 0. else '') +
         r'10^{' + str(int(round(np.log10(abs(value))))) + '}$')))


def distribution_page(histograms, title, xlabel, slug, logarithmic, view,
                      xlim, factor, output, pdf, page, ratio_writer):
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex='col',
                             gridspec_kw={'height_ratios': [3.2, 1.2, 1.2]})
    handles = []
    negative_bins = {}
    for column, (category, category_label) in enumerate(CATEGORIES):
        top, common, paired = axes[:, column]
        data = {key: binned(histograms[key][title, category], factor)
                for key, _, _, _ in VARIANTS}
        edges = data['reference'][0]
        for entry in data.values():
            if not np.array_equal(entry[0], edges):
                raise ValueError('Input binning differs')
        visible = (edges[:-1] >= xlim[0]-1.e-10) & (edges[1:] <= xlim[1]+1.e-10)
        centers = .5*(edges[:-1]+edges[1:])
        content = np.concatenate([item[1][visible] for item in data.values()])
        shown_errors = np.concatenate([item[2][visible] for item in data.values()])
        has_content = np.any(content != 0.) or np.any(shown_errors > 0.)
        has_negative = np.any(content < 0.) or np.any((content == 0.) & (shown_errors > 0.))
        negative_bins[category] = {key: int(np.count_nonzero(item[1][visible] < 0.))
                                   for key, item in data.items()}
        for key, label, color, style in VARIANTS:
            _, values, errors = data[key]
            plotted = np.where(visible & (((values != 0.) | (errors > 0.)) if has_negative else (values > 0.)),
                               values, np.nan)
            line = top.stairs(plotted, edges, color=color, linestyle=style,
                              linewidth=1.35, label=label)
            if column == 0:
                handles.append(line)
            selected = visible & ((values != 0.) | (errors != 0.))
            top.errorbar(centers[selected], plotted[selected], yerr=errors[selected],
                         fmt='none', color=color, alpha=.4, linewidth=.55)
        if has_negative:
            positive = content[content > 0.]
            threshold = max(float(np.max(np.abs(content)))*1.e-6,
                            float(np.min(positive)) if positive.size else 1.e-9)
            top.set_yscale('symlog', linthresh=threshold)
            low = min(0., float(np.min(content-shown_errors)))*1.5
            high = max(threshold, float(np.max(content+shown_errors)))*1.5
            top.set_ylim(low, high)
            signed_ticks(top, low, high, threshold)
            top.axhline(0., color='#999999', linewidth=.5)
            top.text(.02, .02, 'Signed/cancelled bins: symlog scale', transform=top.transAxes,
                     fontsize=7, color='#555555')
        elif has_content:
            top.set_yscale('log')
        else:
            top.set_ylim(0., 1.)
            top.text(.5, .5, 'No entries in this range', transform=top.transAxes,
                     ha='center', va='center', fontsize=9, color='#666666')
        top.set_title(category_label, fontsize=12)
        top.set_ylabel('Cross section per bin [pb]')
        common_series, paired_series = [], []
        reference = data['reference']
        for key, _, color, style in VARIANTS:
            if key == 'reference':
                continue
            values, error, valid = draw_ratio(common, data[key], reference, color,
                                              style, band=(key == 'new'))
            common_series.extend(((values-error)[visible], (values+error)[visible]))
            if not logarithmic or view == 'full':
                for i in np.flatnonzero(visible):
                    ratio_writer.writerow([title, view, category, key, 'reference',
                                           edges[i], edges[i+1], bool(valid[i]),
                                           values[i] if valid[i] else '',
                                           error[i] if valid[i] else '',
                                           reference[2][i]/reference[1][i] if valid[i] else ''])
        for numerator, denominator, label, color, style in PAIRS:
            values, error, valid = draw_ratio(paired, data[numerator], data[denominator],
                                              color, style, label)
            paired_series.extend(((values-error)[visible], (values+error)[visible]))
            for i in np.flatnonzero(visible):
                ratio_writer.writerow([title, view, category, numerator, denominator,
                                       edges[i], edges[i+1], bool(valid[i]),
                                       values[i] if valid[i] else '',
                                       error[i] if valid[i] else '',
                                       data[denominator][2][i]/data[denominator][1][i] if valid[i] else ''])
        limits(common, common_series)
        limits(paired, paired_series)
        common.set_ylabel('Variant / fresh OFF')
        paired.set_ylabel('Controlled ratios')
        if column == 0:
            paired.legend(fontsize=7, loc='best')
        for ax in axes[:, column]:
            ax.axhline(1., color='#888888', linewidth=.6) if ax is not top else None
            ax.set_xlim(*xlim)
            ax.grid(alpha=.2)
        paired.set_xlabel(xlabel)
    suffix = '0.1-decade bins' if logarithmic else f'{10*factor:g} GeV bins'
    title_view = {'full': 'full saved range', 'low': 'low-scale zoom',
                  'tail': 'high-scale tail; rebinned'}[view]
    fig.suptitle(f'WZ + 0,1 jets: {title} — {title_view}, {suffix}', fontsize=14, y=.985)
    fig.legend(handles, [item[1] for item in VARIANTS], loc='upper center',
               bbox_to_anchor=(.5, .955), ncol=2, fontsize=9)
    note = (' Log histograms exclude exact zeros and underflow; they are not normalized to the total rate.'
            if logarithmic else '')
    if 'j1' in slug:
        note += ' Leading jets require anti-kt R=0.4, pT>20 GeV and |eta|<4.5.'
    if factor > 1:
        note += ' Rebinning sums conditional bin variances; bin covariance is unavailable.'
    fig.text(.5, .018, FOOTER+note, ha='center', va='bottom', fontsize=7.4)
    fig.tight_layout(rect=(0, .085, 1, .91), h_pad=.7)
    filename = f'{slug}_{view}.png'
    fig.savefig(output/filename, dpi=155)
    pdf.savefig(fig)
    plt.close(fig)
    return {'page': page, 'observable': title, 'view': view, 'png': filename,
            'range': list(xlim), 'rebin_factor': factor, 'negative_bins': negative_bins}


def rate_page(result, output, pdf):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    for ax, (category, title) in zip(axes, CATEGORIES):
        for index, (key, label, color, _) in enumerate(VARIANTS):
            value = result['runs'][key]['hwu']['rates_pb'][category]
            ax.errorbar(index, value['central'], yerr=value['stat_error'],
                        fmt='o', color=color, capsize=4)
            ax.annotate(f'{value["central"]:.5f}', (index, value['central']),
                        xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
        ax.set_xticks(range(4), ['Old hard', 'Old geometric', 'Fresh OFF', 'Fresh ON'], rotation=20)
        ax.set_title(title)
        ax.set_ylabel('Merged cross section [pb]')
        ax.grid(alpha=.2)
        ax.margins(x=.2, y=.25)
    fig.suptitle('WZ + 0,1 jets: absolute rates for all four prescriptions', fontsize=14)
    fig.text(.5, .025,
             '13 TeV; Qcut=70 GeV; stable W/Z; Pythia 8.317; one million events per prescription.\n'
             'Point errors: conditional event/shower sampling; source integration uncertainty is separate. '
             'No cross-run covariance.\nHistorical and fresh shower steering differ '
             '(charged-resonance QED).', ha='center', va='bottom', fontsize=7.5)
    fig.tight_layout(rect=(0, .13, 1, .92))
    fig.savefig(output/'rates.png', dpi=155)
    pdf.savefig(fig)
    plt.close(fig)


def integral(lookup, title, category, lower, upper):
    bins = [item for item in lookup[title, category].bins
            if item.boundaries[0] >= lower-1.e-10 and item.boundaries[1] <= upper+1.e-10]
    return {'pb': sum(item.get_weight('central') for item in bins),
            'conditional_error_pb': float(np.sqrt(sum(item.get_weight('stat_error')**2 for item in bins)))}


def write_index(output, manifest, combined):
    book = 'full_comparison.pdf' if combined else 'all_variants.pdf'
    rows = ['# Complete WZ prescription comparison plots', '',
            f'[Download the {15 if combined else 13}-page plot book]({book}) · '
            '[Shower spectra only](all_variants.pdf) · '
            '[Source scales](source_scales/source_scales.pdf)', '',
            'All four available one-million-event samples are overlaid: original '
            'hard H scale, previous geometric mean, fresh geometric mean with '
            'backward ISR off, and fresh geometric mean with backward ISR on. '
            'Columns show the merged result, @0 and @1 separately.', '',
            'The middle row uses fresh OFF as a common denominator. The bottom '
            'row separates the two controlled comparisons: historical '
            'geometric/hard scales, and fresh backward ISR on/off. Historical '
            'and fresh epochs have different charged-resonance QED steering; '
            'cross-epoch ratios therefore do not isolate one source change.', '',
            '## Complete coverage', '',
            '| PDF page | Observable | View | Image |', '| ---: | --- | --- | --- |']
    for page in manifest['pages']:
        rows.append(f'| {page["page"]} | {page["observable"]} | {page["view"]} | '
                    f'[PNG]({page["png"]}) |')
    if combined:
        rows += ['| 14 | S-event source SCALUP | Full source support | [PNG](source_scales/scalup_s.png) |',
                 '| 15 | H-event source SCALUP | Full source support | [PNG](source_scales/scalup_h.png) |']
    rows += ['', 'Linear spectra retain their full 0–1000 GeV range. The '
             'low-scale views use native 10 GeV bins over 0–150 GeV. Tail '
             'views cover 400–1000 GeV with 50 GeV bins; all original tail '
             'bins remain visible in the full-range pages. Logarithmic spectra '
             'retain log10(x/GeV) from −3 to 3. Signed or cancelled bins use '
             'symlog axes, and empty component ranges are marked explicitly.', '',
             '## Findings from the expanded range', '',
             '| Merged tail, 400–1000 GeV | Fresh OFF (pb) | Fresh ON (pb) | ON/OFF change |',
             '| --- | ---: | ---: | ---: |']
    for slug, label in (('pt_wz', 'pT(WZ)'), ('pt_j1', 'Leading-jet pT'), ('djr01', 'sqrt(d01)')):
        a, b = [manifest['diagnostic_integrals'][slug][key] for key in ('reference', 'new')]
        rows.append(f'| {label} | {a["pb"]:.6f} ± {a["conditional_error_pb"]:.6f} | '
                    f'{b["pb"]:.6f} ± {b["conditional_error_pb"]:.6f} | '
                    f'{100.*(b["pb"]/a["pb"]-1.):+.2f}% |')
    rows += ['', 'These central tail changes were not shown by the earlier '
             '0–400 GeV figures. The three observables are correlated; their '
             'tail shifts are not independent evidence. Conditional sampling '
             'errors, separate source integration uncertainty and unavailable '
             'cross-run covariance preclude a significance claim.', '',
             '| Very soft pT(WZ), 0.001–0.1 GeV | Rate (pb) |', '| --- | ---: |']
    for key, label, _, _ in VARIANTS:
        value = manifest['diagnostic_integrals']['log_pt_wz'][key]
        rows.append(f'| {label} | {value["pb"]:.7f} ± {value["conditional_error_pb"]:.7f} |')
    rows += ['', 'This tiny absolute-rate region shows a relative historical/fresh '
             'difference. Fresh ON/OFF are close here. Different historical '
             'W-boson QED steering prevents assigning the epoch shift to the '
             'H scale or backward-ISR prescription alone.', '',
             '## Normalization and uncertainty', '',
             'All curves retain absolute cross sections per bin; no unit-area '
             'rescaling is applied. Log spectra exclude exactly zero and '
             'underflow values below 0.001 GeV. Leading-jet spectra omit events '
             'without an anti-kt R=0.4 jet with pT>20 GeV and |eta|<4.5; their '
             'integrals therefore do not equal the total cross section.', '',
             'The saved HwU errors are conditional event/shower errors. '
             'Source integration-normalization errors are separate. Rebinning '
             'combines bin errors in quadrature without bin covariance. Ratio '
             'bands show denominator errors and bars numerator error divided '
             'by the denominator; these are not combined ratio confidence '
             'intervals. Ratios require a positive denominator above three '
             'sampling errors. The source-scale companion documents its '
             'fixed-N sampling-error estimator separately.', '',
             '[Original numerical comparison](../comparison.md) · '
             '[All native histogram bins](../histogram_comparison.csv) · '
             '[Plot ratios](ratios.csv) · [Plot manifest](manifest.json) · '
             '[Source-scale bin table](source_scales/source_scales.csv)', '',
             '## Reproduction', '', '```sh',
             'python3 fxfx_diagnostics/plot_wz_source_scales.py',
             'python3 fxfx_diagnostics/plot_wz_variants.py', '```', '',
             'The source-scale script reads the retained compressed LHE files. '
             'The shower script checks and reads the four retained HwU files. '
             'If `pdfunite` is available, their two books are combined into '
             '`full_comparison.pdf`.', '']
    (output/'README.md').write_text('\n'.join(rows))
    figures = [(page['observable']+' — '+page['view'], page['png']) for page in manifest['pages']]
    if combined:
        figures += [('S-event source shower scales', 'source_scales/scalup_s.png'),
                    ('H-event source shower scales', 'source_scales/scalup_h.png')]
    cards = ''.join(f'<details><summary>{html.escape(label)}</summary>'
                    f'<a href="{filename}"><img src="{filename}" loading="lazy" '
                    f'alt="{html.escape(label)}"></a></details>' for label, filename in figures)
    (output/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>WZ prescription comparisons</title>'
        '<style>body{font:16px system-ui;max-width:1500px;margin:2em auto;padding:0 1em;'
        'color:#202124}a{color:#5840a0}details{margin:1em 0;border:1px solid #ddd;'
        'border-radius:6px;padding:.8em}summary{cursor:pointer;font-weight:600}'
        'img{width:100%;height:auto;margin-top:1em}</style>'
        '<h1>Complete WZ prescription comparisons</h1>'
        f'<p><a href="{book}">Download complete PDF</a> · '
        '<a href="README.md">Coverage, findings and uncertainty notes</a> · '
        '<a href="ratios.csv">Ratio data</a></p>'
        '<p>Four curves, merged / @0 / @1 columns. Middle row: common fresh-OFF '
        'reference. Bottom row: historical geometric/hard and fresh ISR on/off. '
        'Historical and fresh shower steering differ. Expand a plot to inspect it.</p>'+cards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path,
                        default=ROOT/'docs/audits/fxfx_delta_wz_backward_isr/comparison.json')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT/'docs/audits/fxfx_delta_wz_backward_isr/plots')
    args = parser.parse_args()
    result = json.loads(args.comparison.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    histograms, sources = {}, {}
    for key, _, _, _ in VARIANTS:
        path = Path(result['runs'][key]['hwu']['file'])
        if sha256(path) != result['runs'][key]['hwu']['sha256']:
            raise ValueError(f'Saved histogram changed: {path}')
        histograms[key], _ = read_histograms(path)
        sources[key] = {'file': str(path), 'sha256': sha256(path)}
    actual = {title for title, category in histograms['reference'] if title != 'total rate'}
    if actual != {item[0] for item in OBSERVABLES}:
        raise ValueError(f'Uncovered or missing observables: {actual}')
    pages = [{'page': 1, 'observable': 'total rate', 'view': 'overview', 'png': 'rates.png'}]
    with PdfPages(args.output_dir/'all_variants.pdf') as pdf, \
            (args.output_dir/'ratios.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['observable', 'view', 'sample', 'numerator', 'denominator',
                         'xmin', 'xmax', 'resolved_denominator', 'ratio',
                         'numerator_error_over_denominator', 'denominator_relative_error'])
        rate_page(result, args.output_dir, pdf)
        for title, xlabel, slug, logarithmic in OBSERVABLES:
            views = [('full', (-3., 3.) if logarithmic else (0., 1000.), 1)]
            if not logarithmic:
                views += [('low', (0., 150.), 1), ('tail', (400., 1000.), 5)]
            for view, xlim, factor in views:
                pages.append(distribution_page(histograms, title, xlabel, slug, logarithmic,
                                               view, xlim, factor, args.output_dir, pdf,
                                               len(pages)+1, writer))
    bands = {}
    for title, _, slug, logarithmic in OBSERVABLES:
        lower, upper = (-3., -1.) if logarithmic else (400., 1000.)
        bands[slug] = {key: integral(lookup, title, 'SUM', lower, upper)
                       for key, lookup in histograms.items()}
    manifest = {'comparison_file': str(args.comparison.resolve()),
                'comparison_sha256': sha256(args.comparison), 'histogram_sources': sources,
                'plot_script_sha256': sha256(Path(__file__)),
                'all_saved_observables_covered': True, 'categories': [x[0] for x in CATEGORIES],
                'pages': pages, 'diagnostic_integrals': bands,
                'uncertainty_note': FOOTER,
                'source_scale_book': 'source_scales/source_scales.pdf'}
    source_pdf = args.output_dir/'source_scales/source_scales.pdf'
    source_metadata = args.output_dir/'source_scales/source_scales.json'
    combined = False
    if source_pdf.exists() and source_metadata.exists() and shutil.which('pdfunite'):
        source_info = json.loads(source_metadata.read_text())
        if sha256(source_pdf) != source_info['outputs']['source_scales.pdf']:
            raise ValueError('Source-scale PDF does not match its completed audit')
        subprocess.run([shutil.which('pdfunite'), str(args.output_dir/'all_variants.pdf'),
                        str(source_pdf), str(args.output_dir/'full_comparison.pdf')], check=True)
        combined = True
    manifest['combined_book'] = 'full_comparison.pdf' if combined else None
    write_index(args.output_dir, manifest, combined)
    manifest['outputs'] = {path.name: sha256(path) for path in args.output_dir.iterdir()
                           if path.is_file() and path.name != 'manifest.json'}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Wrote {len(pages)} pages covering all six saved spectra to {args.output_dir}')


if __name__ == '__main__':
    main()
