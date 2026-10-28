#!/usr/bin/env python3
"""Compare the untouched PDFmode1 table with repeated Simple ttbar showers.

Usage: mode1_analysis.py STUDY_DIRECTORY OUTPUT_DIRECTORY
       [--prefix matched_pdf13 --prefix matched_pdf15] [--reuse]

Each new prefix lives in STUDY_DIRECTORY/mode1_study and contains the same
871 Born states, 60 probe showers and 240 independent reference showers/Born.
The baseline mode1 calculation is reused from archive_validation/PDFmode1.npz.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import time

import numpy as np

from analyse import EDGES, Q, grouped
from delta_evaluator import DeltaEvaluator, ROOT
from delta_native_pdf import NativePDF


BORN = TRIALS = EVALUATOR = PDF = None
NPROBE = 60
LABELS = {
    'baseline': r'$\alpha_s^{\rm ISR,FSR}=0.118$, PDF13',
    'matched_pdf13': r'$\alpha_s^{\rm ISR}=0.130$, $\alpha_s^{\rm FSR}=0.1365$, PDF13',
    'matched_pdf15': r'$\alpha_s^{\rm ISR}=0.130$, $\alpha_s^{\rm FSR}=0.1365$, PDF15',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def writecsv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def evaluate_born(index):
    b, ts = BORN[index], TRIALS[index]
    reference = np.sort([t['pt'] for t in ts[NPROBE:]])
    assert len(reference) == 240
    refcdf = np.searchsorted(reference, Q, side='right')/len(reference)
    common = []
    flags = np.zeros(4, dtype=int)
    for q in Q:
        r = EVALUATOR.evaluate(b, q, PDF)
        common.append([r['delta'], r['isr'], r['fsr']])
        flags[0] += any(r['clipped']); flags[1] += any(r['nonfinite_raw'])
    probes, legs = [], []
    for t in ts[:NPROBE]:
        if t['pt'] == 0:
            continue
        r = EVALUATOR.evaluate_real(b, t['real'], PDF, mc=0., mb=0.)
        f = np.searchsorted(reference, t['pt'], side='right')/len(reference)
        p = t['real']['p'][-1]
        probes.append([index, t['trial'], t['pt'], np.hypot(p[1], p[2]), t['type'],
                       r['delta'], f, r['isr'], r['fsr'],
                       int(any(r['clipped'])), int(any(r['nonfinite_raw']))])
        legs.append(r['legs'])
        flags[2] += any(r['clipped']); flags[3] += any(r['nonfinite_raw'])
    return index, np.array(common), refcdf, np.array(probes), np.array(legs), flags


def calculate(study, name, baseline, out, workers):
    global BORN, TRIALS, EVALUATOR, PDF
    prefix = study/'mode1_study'/name
    raw_born = [json.loads(line) for line in Path(str(prefix)+'.born.jsonl').open()]
    original = [json.loads(line) for line in (study/'branch_exact.born.jsonl').open()]
    assert raw_born == original, 'Changed Born states, weights, order or starting scales'
    BORN = [grouped(b) for b in raw_born]
    n = len(BORN); TRIALS = [[] for _ in BORN]
    for line in Path(str(prefix)+'.trials.jsonl').open():
        t = json.loads(line)
        assert t['ok'], ('Failed shower', t)
        if t['trial'] >= NPROBE:
            t.pop('real', None)
        TRIALS[t['event']].append(t)
    assert all([t['trial'] for t in ts] == list(range(300)) for ts in TRIALS)
    settings_path = Path(str(prefix)+'.settings.cmnd')
    settings = settings_path.read_text()
    selected_settings = {}
    for key in ['PartonShowers:model', 'PDF:pSet', 'SpaceShower:pdfMode',
                'TimeShower:pdfMode', 'SpaceShower:alphaSvalue', 'TimeShower:alphaSvalue',
                'SpaceShower:alphaSorder', 'TimeShower:alphaSorder',
                'SpaceShower:alphaSuseCMW', 'TimeShower:alphaSuseCMW']:
        m = re.search(r'^'+re.escape(key)+r' = (.+)$', settings, re.M)
        selected_settings[key] = m.group(1) if m else None
    assert selected_settings['PartonShowers:model'] == '1'
    assert selected_settings['SpaceShower:pdfMode'] == '0'
    assert selected_settings['TimeShower:pdfMode'] == '0'
    work = prefix.parent/(name+'_analysis_build')
    started = time.monotonic()
    table = ROOT/'vendor/SudGen/sudakov_r515_PDFmode1.f.gz'
    with NativePDF(study/'TT', work/'native_pdf') as pdf, DeltaEvaluator(work=work/'delta', table=table) as evaluator:
        PDF, EVALUATOR = pdf, evaluator
        serial = evaluate_born(0)
        with ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('fork')) as pool:
            values = list(pool.map(evaluate_born, range(n), chunksize=8))
        for expected, actual in zip(serial[1:], values[0][1:]):
            assert np.array_equal(expected, actual), 'Serial/parallel discrepancy'
        assert [v[0] for v in values] == list(range(n))
        z = dict(q=Q, common=np.array([v[1] for v in values]),
                 refcdf=np.array([v[2] for v in values]),
                 probes=np.concatenate([v[3] for v in values]),
                 legs=np.concatenate([v[4] for v in values]),
                 flags=np.array([v[5] for v in values]),
                 channels=baseline['channels'].copy(), weights=baseline['weights'].copy(),
                 scalup=baseline['scalup'].copy())
        assert np.array_equal(z['common'], baseline['common']), 'Common Delta changed despite identical Born state'
        assert np.isfinite(z['probes']).all()
        fixed = baseline['probes'].copy()
        for i, ts in enumerate(TRIALS):
            reference = np.sort([t['pt'] for t in ts[NPROBE:]])
            selected = fixed[:, 0] == i
            fixed[selected, 6] = np.searchsorted(reference, fixed[selected, 2], side='right')/len(reference)
        z['fixed_probes'] = fixed
        np.savez_compressed(str(prefix)+'.mode1.npz', **z)
        provenance = dict(table=evaluator.provenance, native_pdf=pdf.provenance,
            settings=selected_settings, input_files={str(p):sha(p) for p in
                [Path(str(prefix)+'.born.jsonl'), Path(str(prefix)+'.trials.jsonl'), settings_path]},
            born=n, trials=n*300, probes=len(z['probes']), reference_per_born=240,
            checks=['Born JSON content exactly equals baseline including scales, weights and order',
                    'Every Born has successful trials0-299',
                    'First Born serial and parallel results agree bitwise',
                    'Every common-cut Delta agrees bitwise with baseline',
                    'All evaluated probabilities are finite'],
            elapsed_seconds=time.monotonic()-started)
        (out/(name+'.analysis_provenance.json')).write_text(json.dumps(provenance,indent=2)+'\n')
    print(name, 'evaluated', len(z['probes']), 'probes in', time.monotonic()-started, 's', flush=True)
    return z


def bin_probes(label, z, boots, fixed=False):
    r = z.get('fixed_probes', z['probes']) if fixed else z['probes']
    n = len(z['channels']); event = r[:, 0].astype(int)
    result = []
    for channel, mask in [('all', np.ones(n, bool)), ('gg', z['channels']), ('qqbar', ~z['channels'])]:
        for first_type in ['all', 'isr', 'fsr']:
            component = np.ones(len(r), bool) if first_type == 'all' else r[:, 4] == (2 if first_type=='isr' else 3)
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                selected = (r[:, 2]>=lo)&(r[:, 2]<hi)&mask[event]&component
                if sum(selected)<20:
                    continue
                s, ids = r[selected], event[selected]
                counts = np.bincount(ids, minlength=n)
                ds = np.bincount(ids, weights=s[:,5], minlength=n)
                ps = np.bincount(ids, weights=s[:,6], minlength=n)
                bn = counts[boots].sum(axis=1); good = bn>0
                bd = ds[boots[good]].sum(axis=1)/bn[good]
                bp = ps[boots[good]].sum(axis=1)/bn[good]
                ratio = s[:,5].mean()/s[:,6].mean() if s[:,6].mean()>0 else float('nan')
                finite = s[:,10] == 0
                row = dict(sample=label, channel=channel, first_type=first_type,
                    pt_low=lo, pt_high=hi, pt_mean=s[:,2].mean(),
                    delta=s[:,5].mean(), pythia=s[:,6].mean(), delta_over_pythia=ratio,
                    delta_se=bd.std(ddof=1), pythia_se=bp.std(ddof=1),
                    ratio_se=(bd/bp).std(ddof=1) if np.all(bp>0) else float('nan'),
                    difference=(s[:,5]-s[:,6]).mean(), difference_se=(bd-bp).std(ddof=1),
                    isr_delta=s[:,7].mean(), fsr_delta=s[:,8].mean(),
                    actual_fsr_fraction=np.mean(s[:,4]==3),
                    undefined_raw_count=int(sum(~finite)), clipped_count=int(sum(s[:,9]>0)),
                    probes=len(s), born=int(sum(counts>0)))
                if not fixed and 'legs' in z:
                    for i, value in enumerate(z['legs'][selected].mean(axis=0)):
                        row['leg'+str(i+1)] = value
                else:
                    for i in range(4): row['leg'+str(i+1)] = float('nan')
                result.append(row)
    return result


def summarize(arrays, out):
    n = len(arrays['baseline']['channels'])
    boots = np.random.default_rng(102727).integers(0,n,size=(500,n))
    common, real, fixed = [], [], []
    for label,z in arrays.items():
        for channel,mask in [('all',np.ones(n,bool)),('gg',z['channels']),('qqbar',~z['channels'])]:
            m = sum(mask)
            for j,q in enumerate(z['q']):
                d,p = z['common'][mask,j,0], z['refcdf'][mask,j]
                common.append(dict(sample=label,channel=channel,pt_GeV=q,
                    delta=d.mean(),pythia=p.mean(),delta_over_pythia=d.mean()/p.mean(),
                    difference=(d-p).mean(),difference_se=(d-p).std(ddof=1)/np.sqrt(m),
                    pythia_se=p.std(ddof=1)/np.sqrt(m),delta_se=d.std(ddof=1)/np.sqrt(m),
                    isr_delta=z['common'][mask,j,1].mean(),fsr_delta=z['common'][mask,j,2].mean(),born=int(m)))
        real += bin_probes(label,z,boots)
        fixed += bin_probes(label,z,boots,fixed=True)
    writecsv(out/'common_cut.csv',common)
    writecsv(out/'real_emission.csv',real)
    writecsv(out/'fixed_probe_reference_change.csv',fixed)
    make_plots(arrays, common, real, fixed, out)
    selected = [r for r in real if r['channel']=='all' and r['first_type']=='all'
                and r['pt_low']>=10 and r['pt_high']<=150]
    (out/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')
    report = ['Archived PDFmode1 versus actual Simple-shower ttbar no-emission probabilities',
        '', 'Same871 S-event Born configurations and colour flows, pp13TeV, stable top mass173GeV.',
        'Every sample uses60 probe showers and240 independent reference showers per Born.',
        'Only vendor/SudGen/sudakov_r515_PDFmode1.f.gz is used, without normalization changes.',
        'Production Delta caller, native real-dependent scales, and native nn23nlo beam-PDF ratios are unchanged.',
        'The sampled conditional no-emission probability is F_B(q)=P(first accepted shower pT<=q | Born B).',
        'Each Delta is evaluated on an actual first-emission configuration and its shower Born preimage.',
        'Reported ratios are ratios of means in shower-evolution-pT bins, not means of eventwise ratios.',
        'This is an absolute-S-weight conditional study, not a signed NLO cross section or the full FKS integrand.',
        '', 'Samples:',
        'baseline: Simple8.313, ISR/FSR alphaS(mZ)=.118, one loop, CMWoff, PDF13.',
        'matched_pdf13: ISRalphaS=.130, FSRalphaS=.1365, one loop, CMWoff, PDF13.',
        'matched_pdf15: same increased couplings, PDF15 (NNPDF2.3NLO .119).',
        'Actual shower PDFmode remains0 (ordinary backwards evolution); mode1 labels the lookup table.',
        'The increased couplings match the recorded historical wrapper/defaults, not a proven archived-generation log.',
        'The old archive likely used patched8.225; the present shower uses8.313. ID6 table mass is zero.',
        'Therefore the coupling controls do not constitute an exact recreation of the historical generation setup.',
        '', 'Actual-real comparison, all channels:',
        'sample             pTbin[GeV]       Delta          Pythia         Delta/Pythia   ratioSE']
    for r in selected:
        report.append(f'{r["sample"]:18} {r["pt_low"]:5.0f}-{r["pt_high"]:<5.0f} '
                      f'{r["delta"]:13.6f} {r["pythia"]:13.6f} {r["delta_over_pythia"]:13.6f} {r["ratio_se"]:10.6f}')
    diagnostics = {}
    for label,z in arrays.items():
        probes = z['probes']
        diagnostics[label] = dict(born=n, probes=len(probes),
            undefined_raw_count=int(sum(probes[:,10]>0)),
            clipped_count=int(sum(probes[:,9]>0)),
            first_fsr_fraction=float(np.mean(probes[:,4]==3)))
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    report += ['', 'Diagnostics:']
    for label,d in diagnostics.items():
        report.append(f'{label}: {d["probes"]} probes, {d["undefined_raw_count"]} undefined raw Delta values, '
                      f'{d["clipped_count"]} probes with finite clipping, first-FSR fraction={d["first_fsr_fraction"]:.6f}.')
    report += ['', 'Uncertainties:500 Born-cluster bootstrap resamples with common resamples across samples.',
        'Bins and sample differences are correlated. The uncertainty is statistical, not a modelling error.',
        'common_cut.csv is the auxiliary common-cutoff comparison on Born states.',
        'fixed_probe_reference_change.csv keeps the original52185 probe configurations and Delta values fixed,',
        'changing only the independent reference-shower CDF. It isolates reference-setting effects.',
        'real_emission.csv also contains gg/qqbar and first-emission ISR/FSR splits plus Delta components.',
        '', 'Checks: new Born records and common-cut Delta values agree exactly with the baseline;',
        'first Born calculations agree bitwise between serial and parallel evaluation; all trials succeeded.',
        '', 'Reproduce:',
        'python tests/validation/delta_noemission/mode1_analysis.py STUDY_DIR OUTPUT_DIR --prefix matched_pdf13 --prefix matched_pdf15']
    (out/'report.txt').write_text('\n'.join(report)+'\n')
    return selected


def make_plots(arrays, common, real, fixed, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(baseline='#0072B2',matched_pdf13='#D55E00',matched_pdf15='#009E73')
    fig,axes = plt.subplots(2,2,figsize=(11,7),sharex='col',gridspec_kw={'height_ratios':[2,1]})
    for col,(source,title) in enumerate([(common,'Common cutoff on Born states'),
                                       (real,'Delta on actual first-emission configurations')]):
        for name in arrays:
            rows = [r for r in source if r['sample']==name and r['channel']=='all'
                    and r.get('first_type','all')=='all']
            x = [r.get('pt_GeV',r.get('pt_mean')) for r in rows]
            axes[0,col].plot(x,[r['delta'] for r in rows],color=colors[name],linestyle='--')
            axes[0,col].errorbar(x,[r['pythia'] for r in rows],yerr=[r['pythia_se'] for r in rows],
                               color=colors[name],marker='.',linestyle='-',label=LABELS[name])
            axes[1,col].plot(x,[r['delta_over_pythia'] for r in rows],color=colors[name])
            if col==1:
                y=np.array([r['delta_over_pythia'] for r in rows]); e=np.array([r['ratio_se'] for r in rows])
                axes[1,col].fill_between(x,y-e,y+e,color=colors[name],alpha=.18)
        axes[0,col].set_title(title);axes[0,col].set_ylim(0,1.05)
        axes[1,col].axhline(1,color='gray',linewidth=.8);axes[1,col].set_ylim(0,1.5)
        axes[1,col].set_xlabel('Shower evolution scale [GeV]')
        for ax in axes[:,col]:ax.set_xscale('log');ax.set_xlim(5,200);ax.grid(alpha=.25)
    axes[0,0].set_ylabel('No-emission probability');axes[1,0].set_ylabel('Mean Delta / mean Pythia')
    axes[0,1].legend(fontsize=7,loc='lower right')
    fig.suptitle('PDFmode1 table, unchanged caller; Simple Pythia8.313, stable ttbar at13TeV\n'
                 'Dashed: Delta; solid/points: independent shower reference')
    fig.tight_layout();fig.savefig(out/'comparison.pdf');fig.savefig(out/'comparison.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,channel in zip(axes,['all','gg','qqbar']):
        for name in arrays:
            rows=[r for r in real if r['sample']==name and r['channel']==channel and r['first_type']=='all']
            ax.errorbar([r['pt_mean'] for r in rows],[r['delta_over_pythia'] for r in rows],
                        yerr=[r['ratio_se'] for r in rows],color=colors[name],marker='.',label=LABELS[name])
        ax.set_title(channel);ax.axhline(1,color='gray',linewidth=.8);ax.set_xscale('log')
        ax.set_xlim(5,200);ax.set_ylim(0,1.5);ax.grid(alpha=.25);ax.set_xlabel('First-emission shower pT [GeV]')
    axes[0].set_ylabel('Mean Delta / mean conditional Pythia CDF')
    axes[1].legend(fontsize=6,loc='lower right')
    fig.suptitle('Untouched PDFmode1: actual-real comparison by Born channel')
    fig.tight_layout();fig.savefig(out/'channels.pdf');fig.savefig(out/'channels.png',dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for name in arrays:
        rows=[r for r in fixed if r['sample']==name and r['channel']=='all' and r['first_type']=='all']
        ax.errorbar([r['pt_mean'] for r in rows],[r['delta_over_pythia'] for r in rows],
                    yerr=[r['ratio_se'] for r in rows],color=colors[name],marker='.',label=LABELS[name])
    ax.set_xscale('log');ax.set_xlim(5,200);ax.set_ylim(0,1.5);ax.grid(alpha=.25)
    ax.axhline(1,color='gray',linewidth=.8);ax.legend(fontsize=7)
    ax.set_xlabel('Original probe first-emission shower pT [GeV]')
    ax.set_ylabel('Fixed mean Delta / changed reference CDF')
    ax.set_title('Same52185 probes: only reference-shower settings change')
    fig.tight_layout();fig.savefig(out/'fixed_probe_reference.pdf');fig.savefig(out/'fixed_probe_reference.png',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('study',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--prefix',action='append',default=[])
    parser.add_argument('--reuse',action='store_true');parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();study=args.study.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    path=study/'archive_validation/PDFmode1.npz'
    with np.load(path) as z:baseline={k:z[k] for k in z.files}
    arrays={'baseline':baseline}
    for name in args.prefix:
        if args.reuse:
            with np.load(study/'mode1_study'/(name+'.mode1.npz')) as z:
                arrays[name]={k:z[k] for k in z.files}
        else:
            arrays[name]=calculate(study,name,baseline,out,args.workers)
    selected=summarize(arrays,out)
    provenance=dict(baseline_npz=str(path),baseline_sha256=sha(path),
        script_sha256=sha(__file__),table_sha256=sha(ROOT/'vendor/SudGen/sudakov_r515_PDFmode1.f.gz'),
        samples=list(arrays),bootstrap_seed=102727,bootstrap_replicas=500,
        quantity='Ratio of mean production Delta to mean conditional Simple-shower no-emission probability',
        no_production_changes=True)
    (out/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(selected,indent=2))


if __name__=='__main__':main()
