#!/usr/bin/env python3
"""Swap only archived tables in the existing paired Simple-shower comparison.

Usage: archive_real_validation.py STUDY_DIRECTORY OUTPUT_DIRECTORY
The prior branch_exact analysis supplies independent reference CDFs and the
default-table baseline. The same actual real-emission configurations are
reevaluated with each archive; no new showers or normalization changes.
"""
import argparse
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import multiprocessing
import time

import numpy as np

from analyse import EDGES, grouped
from delta_evaluator import DeltaEvaluator, ROOT
from delta_native_pdf import NativePDF


# Forked workers inherit separate Fortran state and PDF caches. Each chunk
# contains independent Born probes; no mutable state is shared between workers.
WORKER_BORN = None
WORKER_PDF = None
WORKER_EVALUATORS = None


def evaluate_chunk(chunk):
    values = []
    for index, event, real in chunk:
        b = WORKER_BORN[event]
        result = {}
        for label, evaluator in WORKER_EVALUATORS.items():
            r = evaluator.evaluate_real(b, real, WORKER_PDF, mc=0., mb=0.)
            result[label] = [r['delta'],r['isr'],r['fsr'],
                             int(any(r['clipped'])),int(any(r['nonfinite_raw']))]
        values.append((index, result))
    return values


def writecsv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize(arrays, out):
    n = len(arrays['default']['common'])
    boots = np.random.default_rng(102727).integers(0, n, size=(500, n))
    common, real = [], []
    for label, z in arrays.items():
        for channel, mask in [('all', np.ones(n, bool)), ('gg', z['channels']),
                              ('qqbar', ~z['channels'])]:
            m = sum(mask)
            for j, q in enumerate(z['q']):
                d, p = z['common'][mask, j, 0], z['refcdf'][mask, j]
                common.append(dict(table=label, channel=channel, pt_GeV=q,
                    delta=d.mean(), pythia=p.mean(), delta_over_pythia=d.mean()/p.mean(),
                    difference_se=(d-p).std(ddof=1)/np.sqrt(m), born=m))
            r = z['probes']; event = r[:, 0].astype(int)
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                selected = (r[:, 2]>=lo)&(r[:, 2]<hi)&mask[event]
                if sum(selected)<20:
                    continue
                s, ids = r[selected], event[selected]
                counts = np.bincount(ids, minlength=n)
                ds = np.bincount(ids, weights=s[:, 5], minlength=n)
                ps = np.bincount(ids, weights=s[:, 6], minlength=n)
                bn = counts[boots].sum(axis=1); good = bn>0
                bd = ds[boots[good]].sum(axis=1)/bn[good]
                bp = ps[boots[good]].sum(axis=1)/bn[good]
                finite = s[:, 10]==0
                real.append(dict(table=label, channel=channel, pt_low=lo, pt_high=hi,
                    delta=s[:, 5].mean(), pythia=s[:, 6].mean(),
                    delta_over_pythia=s[:, 5].mean()/s[:, 6].mean(),
                    ratio_se=(bd/bp).std(ddof=1),
                    difference_se=(bd-bp).std(ddof=1),
                    ratio_excluding_undefined=s[finite,5].mean()/s[finite,6].mean(),
                    undefined_raw_count=int(sum(~finite)), finite_clipped_count=int(sum(s[:,9]>0)),
                    fsr_delta=s[:,8].mean(), probes=len(s), born=int(sum(counts>0))))
    writecsv(out/'common_cut.csv', common)
    writecsv(out/'real_emission.csv', real)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label in arrays:
        c = [r for r in common if r['table']==label and r['channel']=='all']
        axes[0].plot([r['pt_GeV'] for r in c], [r['delta_over_pythia'] for r in c], label=label)
        r = [r for r in real if r['table']==label and r['channel']=='all']
        axes[1].errorbar([np.sqrt(v['pt_low']*v['pt_high']) for v in r],
                        [v['delta_over_pythia'] for v in r],
                        yerr=[v['ratio_se'] for v in r], label=label, marker='.', capsize=2)
    for ax, title in zip(axes, ['Common cutoff on Born states', 'Actual first-emission configurations']):
        ax.set_title(title); ax.set_xscale('log'); ax.axhline(1, color='gray', linestyle='--')
        ax.set_xlabel('Shower evolution scale [GeV]'); ax.set_ylabel('Mean Delta / mean Simple-shower CDF')
        ax.set_ylim(0, 1.5); ax.set_xlim(5, 200); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('Untouched archive tables, unchanged production caller and shower sample')
    fig.tight_layout(); fig.savefig(out/'comparison.pdf'); fig.savefig(out/'comparison.png', dpi=150)
    return real


def main():
    global WORKER_BORN, WORKER_PDF, WORKER_EVALUATORS
    parser = argparse.ArgumentParser()
    parser.add_argument('study', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(); study, out = args.study.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = study/'archive_validation'; work.mkdir(exist_ok=True)
    base = study/'branch_exact'
    prior = np.load(str(base)+'.analysis.npz')
    arrays = {'default': {k: prior[k] for k in prior.files}}
    baseline_probes = arrays['default']['probes']
    if args.reuse:
        for mode in range(3):
            archive = np.load(work/f'PDFmode{mode}.npz')
            arrays[f'PDFmode{mode}'] = {k: archive[k] for k in archive.files}
    else:
        born = [grouped(json.loads(l)) for l in Path(str(base)+'.born.jsonl').open()]
        n = len(born); assert n==len(prior['common'])
        started = time.monotonic()
        with ExitStack() as stack:
            pdf = stack.enter_context(NativePDF(study/'TT', work/'native_pdf'))
            evaluators = {}
            for mode in range(3):
                label = f'PDFmode{mode}'
                table = ROOT/f'vendor/SudGen/sudakov_r515_{label}.f.gz'
                evaluators[label] = stack.enter_context(DeltaEvaluator(work=work/label, table=table))
                arrays[label] = {k: prior[k].copy() for k in prior.files}
                # Prior aggregate flags belong to the default table; the new
                # probe-level flags are recorded in columns9/10 below.
                arrays[label].pop('flags', None)
                arrays[label]['common'][:] = np.nan
                arrays[label]['probes'][:, [5, 7, 8, 9, 10]] = np.nan
            for i, b in enumerate(born):
                for label, evaluator in evaluators.items():
                    for j, q in enumerate(prior['q']):
                        r = evaluator.evaluate(b, q, pdf)
                        arrays[label]['common'][i,j] = [r['delta'],r['isr'],r['fsr']]
                if (i+1)%100==0:
                    print(f'Common cutoff: {i+1}/{n} Born states, {time.monotonic()-started:.1f}s', flush=True)
            k = 0; inputs = []
            with Path(str(base)+'.trials.jsonl').open() as stream:
                for line in stream:
                    trial = json.loads(line)
                    assert trial['ok']
                    if trial['trial']>=60 or trial['pt']==0:
                        continue
                    baseline = baseline_probes[k]
                    assert (trial['event'],trial['trial'],trial['pt'])==tuple(baseline[:3])
                    inputs.append((k, trial['event'], trial['real']))
                    k += 1
            assert k==len(baseline_probes)
            WORKER_BORN, WORKER_PDF, WORKER_EVALUATORS = born, pdf, evaluators
            serial_check = evaluate_chunk(inputs[:8])
            completed = 0; next_report = 5000; seen = np.zeros(k, dtype=bool)
            chunks = [inputs[i:i+240] for i in range(0, k, 240)]
            with ProcessPoolExecutor(max_workers=args.workers,
                    mp_context=multiprocessing.get_context('fork')) as pool:
                for values in pool.map(evaluate_chunk, chunks):
                    for index, result in values:
                        assert not seen[index]; seen[index] = True
                        for label, row in result.items():
                            arrays[label]['probes'][index,[5,7,8,9,10]] = row
                    completed += len(values)
                    if completed>=next_report:
                        print(f'Real probes: {completed}/{k}, {time.monotonic()-started:.1f}s', flush=True)
                        next_report += 5000
            assert seen.all()
            for index, result in serial_check:
                for label, row in result.items():
                    assert np.array_equal(arrays[label]['probes'][index,[5,7,8,9,10]], row)
            for label in evaluators:
                assert np.isfinite(arrays[label]['probes']).all()
                assert np.isfinite(arrays[label]['common']).all()
                np.savez_compressed(work/f'{label}.npz', **arrays[label])
            provenance = dict(table_provenance={label:e.provenance for label,e in evaluators.items()},
                native_pdf=pdf.provenance, input_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(str(base)+'.born.jsonl'),Path(str(base)+'.trials.jsonl'),Path(str(base)+'.analysis.npz')]},
                definition='Production caller unchanged; archived tables unchanged; no roots applied; same Simple-shower probes/CDFs',
                probes=k, born=n, probes_per_born=60, references_per_born=240,
                uncertainty='500 bootstrap resamples of Born clusters, same resamples for every table',
                parallel_validation='First8 probes for all3 archives agree bitwise between serial and forked workers',
                caveats=['Common cutoff is auxiliary; real comparison uses native real-dependent stopping scales.',
                         'Existing shower is Pythia8.313 Simple, alphaS=.118, PDF13; archives generation settings are not matched.',
                         'Native Delta PDF is nn23nlo, with production grouped flavour averaging.',
                         'This tests the archived tables as replacements, not normalized/regenerated variants.'])
            (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    rows = summarize(arrays, out)
    selected = [r for r in rows if r['channel']=='all' and r['pt_low']>=10 and r['pt_high']<=150]
    (out/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')
    report = ['Archived-table substitution on the existing Simple-shower sample',
              '', 'Same 871 Born states, 52,185 real probes, and independent 240 reference showers/Born.',
              'Only the table changes; the production caller and PDF ratios are unchanged.',
              'No normalization corrections or new shower runs.',
              'The original Pythia8.313 Simple-shower settings are not matched to the archive-generation settings.',
              '', 'Ratios of mean Delta to mean conditional Simple-shower CDF:',
              'Table        evolution-pT bin [GeV]       ratio          Born-cluster SE']
    for row in selected:
        report.append(f'{row["table"]:12} {row["pt_low"]:6.0f}-{row["pt_high"]:<6.0f}'
                      f'                 {row["delta_over_pythia"]:.6f}        {row["ratio_se"]:.6f}')
    report += ['', 'The common-cutoff comparison is auxiliary; real_emission.csv uses native',
               'real-dependent stopping scales. Full channel splits, nonfinite counts, and',
               'clipping diagnostics are in the CSV files. Errors use 500 paired Born-cluster',
               'bootstrap resamples; different scale bins remain correlated.',
               '', 'Reproduce:',
               'python tests/validation/delta_noemission/archive_real_validation.py STUDY_DIR OUTPUT_DIR --workers 8']
    (out/'report.txt').write_text('\n'.join(report)+'\n')
    print(json.dumps(selected,indent=2))


if __name__=='__main__':
    main()
