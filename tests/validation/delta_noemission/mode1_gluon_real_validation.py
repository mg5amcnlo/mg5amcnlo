#!/usr/bin/env python3
"""Test incoming-gluon normalization on stored actual Simple-shower ttbar probes.

Usage: mode1_gluon_real_validation.py STUDY_DIRECTORY OUTPUT_DIRECTORY [--reuse]
Only an isolated copy of the PDFmode1 table is changed. Every native caller,
PDF ratio, starting/stopping scale and independent shower reference is retained.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from analyse import grouped
from delta_evaluator import DeltaEvaluator, ROOT
from delta_native_pdf import NativePDF
from mode1_analysis import bin_probes, writecsv


BORN = PDF = EVALUATOR = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def evaluate_chunk(chunk):
    result = []
    for index, event, real in chunk:
        r = EVALUATOR.evaluate_real(BORN[event], real, PDF, mc=0., mb=0.)
        result.append((index, [r['delta'], r['isr'], r['fsr'],
                       int(any(r['clipped'])), int(any(r['nonfinite_raw']))],
                       r['legs'], [len(c) for c in r['connections']],
                       [len(c) for c in r['connections'][:2]
                        if any(x['type'] != 1 for x in c)],
                       [x['type'] for c in r['connections'][2:] for x in c]))
    return result


def calculate(study, work, table, samples, workers):
    global BORN, PDF, EVALUATOR
    born_path = study/'branch_exact.born.jsonl'
    raw_born = [json.loads(l) for l in born_path.open()]
    BORN = [grouped(b) for b in raw_born]
    result, provenance = {}, {}
    started = time.monotonic()
    with NativePDF(study/'TT', work/'native_pdf') as pdf, \
            DeltaEvaluator(work=work/'delta', table=table) as evaluator:
        PDF, EVALUATOR = pdf, evaluator
        common = np.array([[[r['delta'], r['isr'], r['fsr']]
                           for q in samples['baseline']['q']
                           for r in [evaluator.evaluate(b, q, pdf)]] for b in BORN])
        old = load(study/'mode1_study/gluon_normalization/common_cut.npz')
        indices = [int(np.flatnonzero(samples['baseline']['q'] == q)[0]) for q in old['q']]
        assert np.array_equal(common[:, indices], old['diagnostic'])
        for name, raw in samples.items():
            prefix = study/('branch_exact' if name=='baseline' else 'mode1_study/'+name)
            born_file = Path(str(prefix)+'.born.jsonl')
            assert [json.loads(l) for l in born_file.open()] == raw_born
            trial_file = Path(str(prefix)+'.trials.jsonl')
            inputs, references = [], [[] for _ in BORN]
            for line in trial_file.open():
                t = json.loads(line)
                assert t['ok']
                if t['trial'] >= 60:
                    references[t['event']].append(t['pt'])
                elif t['pt'] > 0:
                    index = len(inputs)
                    assert tuple(raw['probes'][index, :3]) == (t['event'], t['trial'], t['pt'])
                    inputs.append((index, t['event'], t['real']))
            assert len(inputs) == len(raw['probes'])
            assert all(len(r)==240 for r in references)
            refcdf = np.array([np.searchsorted(np.sort(r), raw['q'], side='right')/240
                               for r in references])
            assert np.array_equal(refcdf, raw['refcdf'])
            for event, r in enumerate(references):
                selected = raw['probes'][:, 0] == event
                actual = np.searchsorted(np.sort(r), raw['probes'][selected, 2], side='right')/240
                assert np.array_equal(actual, raw['probes'][selected, 6])
            z = {k: v.copy() for k, v in raw.items() if k not in ('flags', 'fixed_probes')}
            z['common'] = common.copy()
            z['legs'] = np.empty((len(inputs), 4))
            z['connections'] = np.empty((len(inputs), 4), dtype=int)
            serial = evaluate_chunk(inputs[:8])
            seen = np.zeros(len(inputs), bool)
            final_types = set()
            chunks = [inputs[i:i+240] for i in range(0, len(inputs), 240)]
            with ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('fork')) as pool:
                for values in pool.map(evaluate_chunk, chunks):
                    for index, row, legs, connections, nonII, types in values:
                        assert not seen[index] and not nonII
                        seen[index] = True
                        z['probes'][index, [5, 7, 8, 9, 10]] = row
                        z['legs'][index] = legs
                        z['connections'][index] = connections
                        final_types.update(types)
            assert seen.all() and final_types == {4}, final_types
            for index, row, legs, connections, _, _ in serial:
                assert np.array_equal(z['probes'][index, [5, 7, 8, 9, 10]], row)
                assert np.array_equal(z['legs'][index], legs)
                assert np.array_equal(z['connections'][index], connections)
            event = z['probes'][:, 0].astype(int)
            quarks = ~z['channels'][event]
            assert np.array_equal(z['probes'][quarks], raw['probes'][quarks])
            assert np.array_equal(z['common'][~z['channels']], raw['common'][~z['channels']])
            assert np.array_equal(z['common'][:, :, 2], raw['common'][:, :, 2])
            assert np.array_equal(z['probes'][:, 8], raw['probes'][:, 8])
            if 'legs' in raw:
                assert np.array_equal(z['legs'][:, 2:], raw['legs'][:, 2:])
            assert np.isfinite(z['probes']).all() and np.isfinite(z['common']).all()
            assert ((z['probes'][:, 5:9] >= 0) & (z['probes'][:, 5:9] <= 1)).all()
            np.savez_compressed(work/(name+'.npz'), **z)
            result[name] = z
            provenance[name] = dict(born=len(BORN), emitting_probes=len(inputs),
                trials=len(BORN)*300, independent_references_per_born=240,
                undefined_raw=int(sum(z['probes'][:, 10]>0)),
                clipped=int(sum(z['probes'][:, 9]>0)),
                incoming_sudakov_types=[1], final_sudakov_types=sorted(final_types),
                input_sha256={str(p): sha(p) for p in [born_file, trial_file, Path(str(prefix)+'.settings.cmnd')]})
            print(name, len(inputs), 'probes done;', round(time.monotonic()-started, 1), 's', flush=True)
        provenance['evaluator'] = evaluator.provenance
        provenance['native_pdf'] = pdf.provenance
    provenance['checks'] = [
        'All Born records identical including starts, colours and order',
        'Every independent reference CDF reconstructed bitwise from raw successful trials',
        'Prior three-cut square-root diagnostic reproduced bitwise',
        'First eight probes agree bitwise in serial and parallel evaluation for each sample',
        'All common and actual-real FSR factors unchanged bitwise',
        'All common and actual-real qqbar probabilities unchanged bitwise',
        'All actual-real incoming table calls are II; final calls are FI, with no FF calls',
        'All final probabilities finite and bounded']
    return result, provenance


def summarize(samples, corrected, out):
    n = len(samples['baseline']['channels'])
    boots = np.random.default_rng(102727).integers(0, n, size=(500, n))
    real, common = [], []
    for name, raw in samples.items():
        for variant, z in [('raw', raw), ('gluon_root', corrected[name])]:
            rows = bin_probes(name, z, boots)
            for row in rows:
                row['variant'] = variant
            real += rows
            for channel, mask in [('all', np.ones(n, bool)), ('gg', z['channels']), ('qqbar', ~z['channels'])]:
                ids = np.flatnonzero(mask)
                cb = np.random.default_rng(12732).integers(0, len(ids), size=(500, len(ids)))
                for j, q in enumerate(z['q']):
                    d, p = z['common'][mask, j, 0], z['refcdf'][mask, j]
                    bd, bp = d[cb].mean(axis=1), p[cb].mean(axis=1)
                    common.append(dict(sample=name, variant=variant, channel=channel,
                        pt_GeV=q, delta=d.mean(), pythia=p.mean(), delta_over_pythia=d.mean()/p.mean(),
                        ratio_se=(bd/bp).std(ddof=1), difference=(d-p).mean(),
                        difference_se=(bd-bp).std(ddof=1), born=len(ids)))
    writecsv(out/'real_emission.csv', real)
    writecsv(out/'common_cut.csv', common)
    selected = [r for r in real if r['channel']=='all' and r['first_type']=='all'
                and r['variant']=='gluon_root' and r['pt_low']>=10 and r['pt_high']<=150]
    (out/'selected.json').write_text(json.dumps(selected, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(baseline='#0072B2', matched_pdf13='#D55E00', matched_pdf15='#009E73')
    labels = dict(baseline='ISR/FSR alphaS=.118, PDF13',
                  matched_pdf13='ISR .130 / FSR .1365, PDF13',
                  matched_pdf15='ISR .130 / FSR .1365, PDF15')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey='row')
    for row, (source, title) in enumerate([(common, 'Common cutoff'), (real, 'Actual first-emission configurations')]):
        for ax, channel in zip(axes[row], ['all', 'gg', 'qqbar']):
            for name in samples:
                for variant, style in [('raw', '--'), ('gluon_root', '-')]:
                    rs = [r for r in source if r['sample']==name and r['channel']==channel
                          and r['variant']==variant and r.get('first_type', 'all')=='all']
                    x = [r.get('pt_GeV', r.get('pt_mean')) for r in rs]
                    y = np.array([r['delta_over_pythia'] for r in rs])
                    ax.plot(x, y, color=colors[name], linestyle=style,
                            label=labels[name] if variant=='gluon_root' else None)
                    if variant=='gluon_root':
                        e = np.array([r['ratio_se'] for r in rs])
                        ax.fill_between(x, y-e, y+e, color=colors[name], alpha=.15)
            ax.set_xscale('log'); ax.set_xlim(5, 200)
            ax.set_ylim(0, 1.4 if row==0 else 5)
            ax.axhline(1, color='gray', linewidth=.8); ax.grid(alpha=.25)
            ax.set_title(title+'; '+channel); ax.set_xlabel('Shower evolution scale [GeV]')
    axes[0, 0].set_ylabel('Mean Delta / mean shower probability')
    axes[1, 0].set_ylabel('Mean Delta / mean shower probability')
    axes[1, 1].legend(fontsize=7, loc='lower right')
    fig.suptitle('PDFmode1 in stable ttbar, Pythia8.313 Simple\n'
                 'Dashed: original table; solid: square root of incoming-gluon table values only')
    fig.tight_layout(); fig.savefig(out/'comparison.pdf'); fig.savefig(out/'comparison.png', dpi=170)
    plt.close(fig)
    for r in selected:
        print(f'{r["sample"]:16} {r["pt_low"]:3.0f}-{r["pt_high"]:<3.0f} '
              f'{r["delta_over_pythia"]:.6f} +/- {r["ratio_se"]:.6f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('study', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    study, out = args.study.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = study/'mode1_gluon_real_validation'; work.mkdir(exist_ok=True)
    source = ROOT/'vendor/SudGen/sudakov_r515_PDFmode1.f.gz'
    original = gzip.decompress(source.read_bytes()).decode()
    target = '      pysudakov=tmp\n'
    assert original.count(target)==1
    replacement = target + '''      if(abs(id).eq.21.and.(itype.eq.1.or.itype.eq.3))then
        pysudakov=sqrt(max(tmp,0d0))
      endif
'''
    table = work/'mode1_incoming_gluon_root.f'
    table.write_text(original.replace(target, replacement))
    paths = dict(baseline=study/'archive_validation/PDFmode1.npz',
        **{name: study/'mode1_study'/(name+'.mode1.npz') for name in ['matched_pdf13', 'matched_pdf15']})
    samples = {name: load(path) for name, path in paths.items()}
    if args.reuse:
        corrected = {name: load(work/(name+'.npz')) for name in samples}
        provenance = json.loads((out/'provenance.json').read_text())
    else:
        corrected, provenance = calculate(study, work, table, samples, args.workers)
    summarize(samples, corrected, out)
    provenance.update(script_sha256=sha(__file__), archive_sha256=sha(source),
        corrected_table_sha256=sha(table), input_npz_sha256={str(p): sha(p) for p in paths.values()},
        output_npz_sha256={str(work/(name+'.npz')): sha(work/(name+'.npz')) for name in samples},
        modification=replacement, no_production_changes=True,
        bootstrap_seed=102727, bootstrap_replicas=500,
        definition='Square root only of incoming-gluon table return; caller PDF ratios untouched. '
                   'Ratios of means, independent 240-shower CDF per Born, actual native real stopping scales.')
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__=='__main__':
    main()
