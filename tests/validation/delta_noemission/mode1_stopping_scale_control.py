#!/usr/bin/env python3
"""Isolate real-dependent stopping scales on identical corrected-mode1 probes.

Usage: mode1_stopping_scale_control.py STUDY_DIRECTORY OUTPUT_DIRECTORY [--reuse]
No production files are changed. All variants share probe kinematics, native
PDF ratios, starts, and independent reference CDFs. This is a diagnostic.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from analyse import grouped
from delta_evaluator import DeltaEvaluator
from delta_native_pdf import NativePDF
from mode1_analysis import bin_probes, writecsv


BORN = PDF = EVALUATOR = None
VARIANTS = {
    'native': 'Native real stopping scales, real masses, native kinematic dead zones and start veto.',
    'uniform_real_keep_dead': 'Every stopping scale is actual shower pT; real masses and native kinematic dead zones retained; start veto recomputed.',
    'uniform_real': 'Every stopping scale is actual shower pT; real masses; only q>=start veto.',
    'uniform_born': 'Every stopping scale is actual shower pT; Born soft-limit masses; only q>=start veto (evaluator.evaluate).',
    'isr_uniform': 'Incoming stopping scales are actual shower pT with only q>=start veto; final stopping scales/dead zones remain native; real masses.',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def pack(r):
    return [r['delta'], r['isr'], r['fsr'], int(any(r['clipped'])), int(any(r['nonfinite_raw']))]


def evaluate_chunk(chunk):
    result = []
    for index, event, q, real in chunk:
        b = BORN[event]
        start = EVALUATOR.starting_scales(b)
        stop, mass, dead = EVALUATOR.real_scales(b, real, mc=0., mb=0.)
        uniform = np.full_like(stop, q)
        isr_stop, isr_dead = stop.copy(), (dead | (stop > start)).copy()
        isr_stop[:2] = q
        isr_dead[:2] = uniform[:2] >= start[:2]
        rs = dict(
            native=EVALUATOR.evaluate_matrices(b, start, stop, mass, dead | (stop > start), PDF),
            uniform_real_keep_dead=EVALUATOR.evaluate_matrices(b, start, uniform, mass, dead | (uniform >= start), PDF),
            uniform_real=EVALUATOR.evaluate_matrices(b, start, uniform, mass, uniform >= start, PDF),
            uniform_born=EVALUATOR.evaluate(b, q, PDF),
            isr_uniform=EVALUATOR.evaluate_matrices(b, start, isr_stop, mass, isr_dead, PDF))
        assert rs['isr_uniform']['fsr'] == rs['native']['fsr']
        assert rs['isr_uniform']['isr'] == rs['uniform_real']['isr']
        for r in rs.values():
            assert np.isfinite(pack(r)).all()
        result.append((index, {name: (pack(r), r['legs']) for name, r in rs.items()}))
    return result


def calculate(study, work, out, samples, workers):
    global BORN, PDF, EVALUATOR
    table = study/'mode1_gluon_real_validation/mode1_incoming_gluon_root.f'
    born_file = study/'branch_exact.born.jsonl'
    raw_born = [json.loads(l) for l in born_file.open()]
    BORN = [grouped(b) for b in raw_born]
    arrays, provenance = {}, {}
    started = time.monotonic()
    with NativePDF(study/'TT', work/'native_pdf') as pdf, \
            DeltaEvaluator(work=work/'delta', table=table) as evaluator:
        PDF, EVALUATOR = pdf, evaluator
        for name in samples:
            source = study/'mode1_gluon_real_validation'/(name+'.npz')
            raw = load(source)
            prefix = study/'mode1_study'/name
            bf, tf = Path(str(prefix)+'.born.jsonl'), Path(str(prefix)+'.trials.jsonl')
            assert [json.loads(l) for l in bf.open()] == raw_born
            inputs, refs = [], [[] for _ in BORN]
            for line in tf.open():
                t = json.loads(line)
                assert t['ok']
                if t['trial'] >= 60:
                    refs[t['event']].append(t['pt'])
                elif t['pt'] > 0:
                    i = len(inputs)
                    assert (t['event'], t['trial'], t['pt']) == tuple(raw['probes'][i, :3])
                    inputs.append((i, t['event'], t['pt'], t['real']))
            assert len(inputs) == len(raw['probes'])
            for event, ref in enumerate(refs):
                assert len(ref)==240
                mask = raw['probes'][:, 0] == event
                actual = np.searchsorted(np.sort(ref), raw['probes'][mask, 2], side='right')/240
                assert np.array_equal(actual, raw['probes'][mask, 6])
            z = {v: dict(probes=raw['probes'].copy(), legs=np.empty((len(inputs), 4)),
                         channels=raw['channels']) for v in VARIANTS}
            serial = evaluate_chunk(inputs[:8])
            seen = np.zeros(len(inputs), bool)
            chunks = [inputs[i:i+160] for i in range(0, len(inputs), 160)]
            with ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('fork')) as pool:
                for values in pool.map(evaluate_chunk, chunks):
                    for index, results in values:
                        assert not seen[index]
                        seen[index] = True
                        for variant, (row, legs) in results.items():
                            z[variant]['probes'][index, [5, 7, 8, 9, 10]] = row
                            z[variant]['legs'][index] = legs
            assert seen.all()
            for index, results in serial:
                for variant, (row, legs) in results.items():
                    assert np.array_equal(z[variant]['probes'][index, [5, 7, 8, 9, 10]], row)
                    assert np.array_equal(z[variant]['legs'][index], legs)
            assert np.array_equal(z['native']['probes'], raw['probes'])
            assert np.array_equal(z['native']['legs'], raw['legs'])
            assert np.array_equal(z['isr_uniform']['probes'][:, 8], raw['probes'][:, 8])
            assert np.array_equal(z['isr_uniform']['probes'][:, 7], z['uniform_real']['probes'][:, 7])
            for variant, values in z.items():
                assert np.isfinite(values['probes']).all()
                assert ((values['probes'][:, 5:9]>=0) & (values['probes'][:, 5:9]<=1)).all()
            data = dict(channels=raw['channels'])
            data.update({variant+'_'+key: val for variant, a in z.items()
                         for key, val in a.items() if key!='channels'})
            np.savez_compressed(work/(name+'.npz'), **data)
            arrays[name] = z
            provenance[name] = dict(probes=len(inputs), born=len(BORN),
                input_sha256={str(p): sha(p) for p in [source, bf, tf, Path(str(prefix)+'.settings.cmnd')]},
                diagnostics={variant: dict(undefined_raw=int(sum(a['probes'][:, 10]>0)),
                    clipped=int(sum(a['probes'][:, 9]>0))) for variant, a in z.items()})
            print(name, len(inputs), 'probes / five variants done;', round(time.monotonic()-started, 1), 's', flush=True)
        provenance['evaluator'] = evaluator.provenance
        provenance['native_pdf'] = pdf.provenance
    provenance['checks'] = ['All Born input records identical',
        'Independent reference CDFs reconstructed bitwise from 240 raw showers per Born',
        'First eight probes for every variant agree bitwise in serial and parallel',
        'Native probabilities and individual leg factors reproduce saved corrected results bitwise',
        'Incoming-only replacement retains native FSR bitwise and matches uniform-real ISR bitwise',
        'All returned probabilities finite and bounded']
    return arrays, provenance


def summarize(arrays, out):
    n = len(next(iter(arrays.values()))['native']['channels'])
    boots = np.random.default_rng(102727).integers(0, n, size=(500, n))
    rows = []
    for name, variants in arrays.items():
        for variant, z in variants.items():
            rs = bin_probes(name, z, boots)
            for r in rs: r['variant'] = variant
            rows += rs
    writecsv(out/'stop_control.csv', rows)
    selected = [r for r in rows if r['channel']=='all' and r['first_type']=='all'
                and r['pt_low']>=10 and r['pt_high']<=150]
    (out/'stop_control.selected.json').write_text(json.dumps(selected, indent=2)+'\n')
    report = ['Stopping-scale controls on identical actual first-emission probes', '',
        'Corrected PDFmode1: incoming-gluon table values replaced by sqrt(T), other table entries unchanged.',
        'Same actual probe configurations and reference CDF for every variant within a sample.',
        'These are probe-conditioned averages; they differ from common-cut averages over all Born states.',
        'Same native caller, starting scales, grouped native nn23nlo PDF prescription and Simple-shower references.',
        'matched_pdf13/PDF15 use ISR alphaS=.130, FSR=.1365, one loop, CMWoff.', '', 'Variants:']
    report += [key+': '+value for key, value in VARIANTS.items()]
    report += ['', 'Mean Delta / mean Simple-shower CDF, all channels:',
               'sample          variant                   pTbin     ratio       SE']
    for r in selected:
        report.append(f'{r["sample"]:15} {r["variant"]:26} {r["pt_low"]:3.0f}-{r["pt_high"]:<3.0f} '
                      f'{r["delta_over_pythia"]:.6f} {r["ratio_se"]:.6f}')
    report += ['', 'Uncertainties:500 paired Born-cluster bootstrap replicas; bins and variants correlated.',
               'CSV includes gg/qqbar and first-emission ISR/FSR channel splits, individual legs and clipping counts.',
               'This diagnostic does not prescribe replacing the production stopping-scale definition.',
               'Reproduce: python tests/validation/delta_noemission/mode1_stopping_scale_control.py STUDY_DIR OUTPUT_DIR']
    (out/'stop_control.txt').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('study', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    study, out = args.study.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = study/'mode1_gluon_real_validation/stop_control'; work.mkdir(exist_ok=True)
    samples = ['matched_pdf13', 'matched_pdf15']
    if args.reuse:
        provenance = json.loads((out/'stop_control.provenance.json').read_text())
        arrays = {}
        for name in samples:
            z = load(work/(name+'.npz'))
            arrays[name] = {variant: dict(probes=z[variant+'_probes'], legs=z[variant+'_legs'],
                            channels=z['channels']) for variant in VARIANTS}
    else:
        arrays, provenance = calculate(study, work, out, samples, args.workers)
    summarize(arrays, out)
    provenance.update(script_sha256=sha(__file__), variants=VARIANTS, no_production_changes=True,
        output_npz_sha256={str(work/(name+'.npz')): sha(work/(name+'.npz')) for name in samples},
        bootstrap_seed=102727, bootstrap_replicas=500)
    (out/'stop_control.provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__=='__main__':
    main()
