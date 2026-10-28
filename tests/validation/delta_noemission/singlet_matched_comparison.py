#!/usr/bin/env python3
"""Compare original and historical-settings-matched DY/ggH shower references.

Usage: singlet_matched_comparison.py ORIGINAL_STUDY MATCHED_STUDY OUTPUT
Run singlet_matched_analysis.py separately for all four samples in MATCHED_STUDY/pdf13
and MATCHED_STUDY/pdf15 first. The Delta values are recalculated there; this
driver checks they agree bitwise with the earlier calculation where available.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from singlet_analysis import ABSOLUTE, RELATIVE, NAMES as OLD_NAMES, SAMPLES, ROOT, load
from singlet_matched_analysis import NAMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(numerator, denominator):
    return float(numerator/denominator) if denominator > 0 else None


def bootstrap_ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)


def standard_error(samples):
    return float(samples.std(ddof=1)) if np.isfinite(samples).all() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('original', type=Path)
    parser.add_argument('matched', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    original, matched, out = [p.resolve() for p in
                              [args.original, args.matched, args.output]]
    out.mkdir(parents=True, exist_ok=True)
    rows, checks, inputs = [], [], {}
    for pdf in ['13', '15']:
        folder = 'analysis_pdf13' if pdf == '13' else 'analysis'
        old_provenance_path = ROOT/'docs/audits/delta_noemission/singlet_validation'
        if pdf == '13': old_provenance_path /= 'pdf13_control'
        old_provenance_path /= 'analysis_provenance.json'
        new_provenance_path = out/('pdf'+pdf)/'analysis_provenance.json'
        old_provenance = json.loads(old_provenance_path.read_text())
        new_provenance = json.loads(new_provenance_path.read_text())
        for p in [old_provenance_path, new_provenance_path]: inputs[str(p)] = sha(p)
        for key in ['pdf_grid_sha256', 'pdf_source_sha256']:
            assert old_provenance[key] == new_provenance[key], (pdf, key)
        assert old_provenance['script_sha256'] == new_provenance['original_driver_sha256']
        assert new_provenance['table_order'] == NAMES
        for table in NAMES:
            b = new_provenance['tables'][table]
            if table in OLD_NAMES:
                a = old_provenance['tables'][table]
                for key in ['table_sha256', 'source_sha256', 'extracted_routines']:
                    assert a[key] == b[key], (pdf, table, key)
            table_path = Path(b['table'])
            table_bytes = table_path.read_bytes()
            if table_path.suffix == '.gz': table_bytes = gzip.decompress(table_bytes)
            assert hashlib.sha256(table_bytes).hexdigest() == b['table_sha256']
            for filename, expected in b['source_sha256'].items():
                assert sha(ROOT/filename) == expected
        for sample in SAMPLES:
            path = matched/('pdf'+pdf)/folder/(sample+'.npz')
            new = load(path)
            inputs[str(path)] = sha(path)
            born_new = matched/('pdf'+pdf)/(sample+'.born.jsonl')
            born_old = original/(sample+'.born.jsonl')
            assert born_new.read_bytes() == born_old.read_bytes()
            inputs[str(born_new)] = sha(born_new)
            inputs[str(born_old)] = sha(born_old)
            # Independently recompute the empirical CDF from all retained trials.
            recomputed = np.array([np.searchsorted(np.sort(p), q, side='right')/len(p)
                                   for p, q in zip(new['trials'], new['q'])])
            assert np.array_equal(recomputed, new['reference'])
            assert np.all(new['trials'] >= 0)
            assert np.all(new['trials'] <= new['scalup'][:, None]*(1+1.e-10))
            old_sample = sample + ('_pdf13' if pdf == '13' else '')
            old_path = original/folder/(old_sample+'.npz')
            old = load(old_path) if old_path.exists() else None
            if old is not None:
                inputs[str(old_path)] = sha(old_path)
                for key in ['q', 'ids', 'rapidity', 'scalup']:
                    assert np.array_equal(new[key], old[key]), (pdf, sample, key)
                assert np.array_equal(new['values'][:, :, [NAMES.index(t) for t in OLD_NAMES]],
                                      old['values']), (pdf, sample, 'values')
            checks.append(dict(pdf=pdf, sample=sample, born=len(new['reference']),
                trials=int(new['trials'].size), born_records_byte_identical=True,
                cdf_recomputed_bitwise_identical=True,
                table_pdf_and_native_source_hashes_match_old=True,
                old_delta_available=old is not None,
                overlapping_native_delta_values_bitwise_identical=(True if old is not None else None),
                overlapping_tables=OLD_NAMES if old is not None else []))
            n = len(new['reference'])
            boots = np.random.default_rng(102828).integers(0, n, size=(500, n))
            for j in range(new['q'].shape[1]):
                threshold_qualified = (int(sum(new['q'][:, j] <= np.sqrt(2.)*4.8))
                    if sample in new_provenance.get('threshold_error_samples', []) else 0)
                after_p = new['reference'][:, j]
                after_bp = after_p[boots].mean(axis=1)
                before_p = old['reference'][:, j] if old is not None else None
                before_bp = before_p[boots].mean(axis=1) if old is not None else None
                for k, table in enumerate(NAMES):
                    had_old_delta = old is not None and table in OLD_NAMES
                    d = new['values'][:, j, k, 0]
                    bd = d[boots].mean(axis=1)
                    after_br = bootstrap_ratio(bd, after_bp)
                    before_br = bootstrap_ratio(bd, before_bp) if had_old_delta else None
                    after_ratio = ratio(d.mean(), after_p.mean())
                    before_ratio = ratio(d.mean(), before_p.mean()) if had_old_delta else None
                    row = dict(pdf=pdf, sample=sample, table=table,
                        cut_kind='GeV' if j<len(ABSOLUTE) else 'fraction_of_start',
                        cut_value=float(ABSOLUTE[j] if j<len(ABSOLUTE) else RELATIVE[j-len(ABSOLUTE)]),
                        born=n, delta=float(d.mean()),
                        shower_before=float(before_p.mean()) if old is not None else None,
                        shower_after=float(after_p.mean()),
                        ratio_before=before_ratio, ratio_after=after_ratio,
                        ratio_before_se=standard_error(before_br) if had_old_delta else None,
                        ratio_after_se=standard_error(after_br),
                        ratio_change=(after_ratio-before_ratio
                                      if after_ratio is not None and before_ratio is not None else None),
                        ratio_change_se=(standard_error(after_br-before_br)
                                         if had_old_delta else None),
                        shower_change=float((after_p-before_p).mean()) if old is not None else None,
                        shower_change_se=float((after_bp-before_bp).std(ddof=1)) if old is not None else None,
                        threshold_qualified_born=threshold_qualified,
                        undefined_born=int((new['values'][:, j, k, 4]>0).sum()))
                    rows.append(row)
    with (out/'comparison_before_after.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    selected = [r for r in rows if r['cut_kind']=='GeV'
                and r['cut_value'] in [5, 10, 20, 30, 50, 75, 100]]
    (out/'comparison_before_after.json').write_text(json.dumps(selected, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for i, pdf in enumerate(['13', '15']):
        for col, process in enumerate(['dy', 'higgs']):
            ax = axes[i, col]
            table = 'PDFmode1' if process == 'dy' else 'PDFmode1_root'
            for start, color in [('fixed', '#0072B2'), ('damped', '#D55E00')]:
                rs = [r for r in selected if r['pdf']==pdf and r['sample']==process+'_'+start
                      and r['table']==table]
                x = np.array([r['cut_value'] for r in rs])
                y = np.array([r['ratio_after'] if r['undefined_born']==0 and r['threshold_qualified_born']==0 and r['ratio_after'] is not None
                              else np.nan for r in rs])
                e = np.array([r['ratio_after_se'] if r['ratio_after_se'] is not None else np.nan for r in rs])
                ax.plot(x, y, color=color, label=start+', matched settings')
                ax.fill_between(x, y-e, y+e, color=color, alpha=.15)
                if all(r['ratio_before'] is not None for r in rs):
                    ax.plot(x, [r['ratio_before'] if r['undefined_born']==0 and r['threshold_qualified_born']==0 else np.nan for r in rs], '--', color=color,
                            label=start+', original settings')
            ax.axhline(1, color='black', linewidth=.7)
            ax.set_xscale('log'); ax.set_xlim(5, 100); ax.set_ylim(0, 1.6)
            ax.grid(alpha=.25)
            ax.set_title(('Drell–Yan' if process=='dy' else 'gg → H')+f', PDF{pdf}')
            ax.set_xlabel('Common cutoff q [GeV]')
            ax.set_ylabel('Mean Delta / mean shower CDF')
            ax.legend(fontsize=8)
    fig.suptitle('Historical shower settings: PDFmode 1, incoming-gluon square root\n'
                 'Simple Pythia 8.313; PDF15 DY cuts ≤6.79 GeV omitted because of threshold errors')
    fig.tight_layout()
    for suffix in ['pdf', 'png']:
        fig.savefig(out/('comparison_before_after.'+suffix), dpi=170)
    plt.close(fig)
    provenance = dict(checks=checks, input_sha256=inputs, script_sha256=sha(__file__),
        definitions=dict(ratio='Mean Delta divided by mean first-QCD-emission CDF',
            before='Original singlet reference settings, including ISR pT0Ref=2GeV',
            after='Historical-settings-matched reference; see shower_provenance.json',
            bootstrap='500 paired Born-cluster resamples, seed102828; identical resamples for before/after',
            zero_probability='Ratios or bootstrap errors with zero denominators are null; no such ratio is interpreted as a probability',
            threshold_qualification='For samples whose full showers report xPDF=0 in pT2nearThreshold, '
                'cuts at/below sqrt(2)*4.8GeV are counted and masked in plots. Their retained numerical '
                'ratios are qualified diagnostics, not unqualified physics validation.',
            unavailable='No old PDF13 damped reference exists; before values are null for those rows. '
                'PDFmode0 is newly evaluated, so its old Delta ratios are null.'),
        note='Every matched native Delta value was recalculated. All six overlapping old/new '
             'sample/PDF combinations agree bitwise for all five previously tested table variants, '
             'including clipping and undefined flags. '
             'Only the empirical shower reference changes for those comparisons.')
    (out/'comparison_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    for r in selected:
        if r['table'] == ('PDFmode1' if r['sample'].startswith('dy') else 'PDFmode1_root'):
            print(f'PDF{r["pdf"]} {r["sample"]:13} q={r["cut_value"]:3.0f} '
                  f'old={r["ratio_before"]} new={r["ratio_after"]} +/-{r["ratio_after_se"]}')


if __name__ == '__main__':
    main()
