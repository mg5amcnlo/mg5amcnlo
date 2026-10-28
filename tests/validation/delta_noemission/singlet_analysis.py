#!/usr/bin/env python3
"""Common-cutoff table/shower comparison for DY and gg->H Born ensembles.

Usage: singlet_analysis.py STUDY_DIRECTORY PYTHIA_INSTALL OUTPUT_DIRECTORY
       [--reuse] [--workers 8] [--pdf 13|15] [--sample NAME ...]
Run singlet_run_showers.py first (singlet_pdf13_control.py for PDF13).
The exact shower PDF and definite incoming
flavours are used in the unchanged native Delta caller. No actual-real Delta
is compared with a scalar shower CDF.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import ctypes as C
from functools import lru_cache
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np

from delta_evaluator import DeltaEvaluator, ROOT
from mode1_analysis import writecsv

ABSOLUTE = np.array([2., 3., 5., 7., 10., 15., 20., 30., 40., 50., 75., 100., 125., 150.])
# Keep every relative cut above the native sub-1.01 GeV continuation region.
RELATIVE = np.array([.2, .3, .5, .75, .9, 1.])
NAMES = ['default', 'PDFmode2', 'PDFmode2_root', 'PDFmode1', 'PDFmode1_root']
SAMPLES = ['dy_fixed', 'dy_damped', 'higgs_fixed', 'higgs_damped']
BORN = EVALUATORS = PDF = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def table_value(evaluator, q, mass, pdg):
    # Public table function, called independently of delta_leg_probability.
    f = evaluator.lib.pysudakov_
    f.argtypes = [C.POINTER(C.c_double), C.POINTER(C.c_double),
                  C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_double)]
    f.restype = C.c_double
    mq = (C.c_double*21)(.33, .33, .5, 1.5, 4.8, *([0.]*16))
    a, b, i, t = C.c_double(q), C.c_double(mass), C.c_int(pdg), C.c_int(1)
    return f(C.byref(a), C.byref(b), C.byref(i), C.byref(t), mq)


def evaluate_born(index):
    b = BORN[index]
    qs = np.r_[ABSOLUTE, RELATIVE*b['scalup']]
    assert min(qs)>=1.01
    higgs = b['ids'][0] == 21
    expected_counts = [2, 2, 0] if higgs else [1, 1, 0, 0]
    values = np.empty((len(qs), len(NAMES), 5))
    formula_error = 0.
    for j, q in enumerate(qs):
        results = {}
        for k, name in enumerate(NAMES):
            r = EVALUATORS[name].evaluate(b, q, PDF)
            assert r['fsr'] == 1 and all(v == 1 for v in r['legs'][2:])
            counts = [len(c) for c in r['connections']]
            assert counts == (expected_counts if q < b['scalup'] else [0]*len(b['ids']))
            for c in r['connections'][:2]:
                for line in c:
                    assert line['type']==1
                    assert line['start']==b['scalup'] and line['stop']==q
            values[j, k] = [r['delta'], r['isr'], r['fsr'],
                             sum(r['clipped']), sum(r['nonfinite_raw'])]
            results[name] = r
        if not higgs:
            for mode in [1, 2]:
                assert results[f'PDFmode{mode}'] == results[f'PDFmode{mode}_root']
        elif q < b['scalup']:
            mass = results['PDFmode1']['connections'][0][0]['mass']
            for mode in [1, 2]:
                e = EVALUATORS[f'PDFmode{mode}']
                ratio = table_value(e, q, mass, 21)/table_value(e, b['scalup'], mass, 21)
                for root, power in [('', 2), ('_root', 1)]:
                    r = results[f'PDFmode{mode}'+root]
                    if any(r['nonfinite_raw']):
                        # Both native logarithmic weights vanish when R=1.
                        # Retain and flag the caller output; it is not a valid
                        # probability or an algebra/normalization comparison.
                        assert ratio >= 1.
                        continue
                    legs = [np.clip(PDF(21, x, q)/PDF(21, x, b['scalup'])*ratio**power, 0, 1)
                            for x in b['x']]
                    actual = r['legs'][:2]
                    error = max(abs(np.array(actual)-legs))
                    assert error < 2.e-12, (index, q, mode, root, actual, legs)
                    formula_error = max(formula_error, error)
    assert np.isfinite(values).all()
    return values, formula_error


def calculate(study, install, out, workers, pdfset, samples):
    global BORN, EVALUATORS, PDF
    work = study/('analysis' if pdfset=='15' else 'analysis_pdf13'); work.mkdir(exist_ok=True)
    source = ROOT/'tests/validation/delta_noemission/pythia_pdf.cc'
    bridge = work/'pythia_pdf.so'
    build = ['c++', '-O2', '-std=c++17', '-shared', '-fPIC', str(source),
        '-I'+str(install/'include'), '-L'+str(install/'lib'),
        '-Wl,-rpath,'+str(install/'lib'), '-lpythia8', '-o', str(bridge)]
    subprocess.run(build, check=True)
    lib = C.CDLL(str(bridge))
    lib.pythia_pdf_init.argtypes = [C.c_char_p, C.c_char_p]
    lib.pythia_pdf_xf.argtypes = [C.c_int, C.c_double, C.c_double]
    lib.pythia_pdf_xf.restype = C.c_double
    assert lib.pythia_pdf_init(str(install/'share/Pythia8/xmldoc').encode(), pdfset.encode())==0
    @lru_cache(maxsize=100000)
    def pdf(pdg, x, q):
        return lib.pythia_pdf_xf(pdg, x, q)
    PDF = pdf
    paths = {'default': ROOT/'vendor/SudGen/sudakov_template.f'}
    for mode in [1, 2]:
        path = ROOT/f'vendor/SudGen/sudakov_r515_PDFmode{mode}.f.gz'
        paths[f'PDFmode{mode}'] = path
        original = gzip.decompress(path.read_bytes()).decode()
        target = '      pysudakov=tmp\n'
        assert original.count(target)==1
        table = work/f'PDFmode{mode}_incoming_gluon_root.f'
        table.write_text(original.replace(target, target+'''      if(abs(id).eq.21.and.(itype.eq.1.or.itype.eq.3))then
        pysudakov=sqrt(max(tmp,0d0))
      endif
'''))
        paths[f'PDFmode{mode}_root'] = table
    arrays, provenance = {}, {}
    started = time.monotonic()
    with ExitStack() as stack:
        EVALUATORS = {name: stack.enter_context(DeltaEvaluator(work=work/name, table=paths[name]))
                      for name in NAMES}
        previous = {}
        for name in samples:
            born_path, trials_path = study/(name+'.born.jsonl'), study/(name+'.trials.jsonl')
            BORN = [json.loads(l) for l in born_path.open()]
            base_name = name.removesuffix('_pdf13')
            if name != base_name:
                assert BORN == [json.loads(l) for l in (study/(base_name+'.born.jsonl')).open()]
            n = len(BORN)
            assert [b['event'] for b in BORN] == list(range(n))
            assert np.array_equal([b['weight'] for b in BORN], np.ones(n))
            original_key = name.split('_')[0]
            if base_name.endswith('fixed'):
                previous[original_key] = BORN
            else:
                for fixed, damped in zip(previous[original_key], BORN):
                    for key in ['ids', 'p', 'masses', 'colors', 'x', 'weight']:
                        assert fixed[key] == damped[key], (name, key)
                    assert .1*fixed['scalup'] <= damped['scalup'] <= fixed['scalup']
            pts = [[] for _ in BORN]
            for l in trials_path.open():
                t = json.loads(l)
                assert t['ok']
                i = t['event']
                assert t['trial']==len(pts[i])
                assert 0<=t['pt']<=BORN[i]['scalup']*(1+1.e-9)
                assert t['pt']==0 or t['type']==2, 'Non-ISR first QCD emission'
                pts[i].append(t['pt'])
            assert len(set(map(len, pts)))==1 and len(pts[0])>=300
            qs = np.array([np.r_[ABSOLUTE, RELATIVE*b['scalup']] for b in BORN])
            reference = np.array([np.searchsorted(np.sort(p), q, side='right')/len(p)
                                  for p, q in zip(pts, qs)])
            serial = evaluate_born(0)
            with ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('fork')) as pool:
                results = list(pool.map(evaluate_born, range(n), chunksize=8))
            assert np.array_equal(serial[0], results[0][0]) and serial[1]==results[0][1]
            values = np.array([r[0] for r in results])
            end = qs>=np.array([b['scalup'] for b in BORN])[:, None]
            assert np.array_equal(reference[end], np.ones(end.sum()))
            assert (values[:, :, :, 0][end] == 1).all()
            ids = np.array([b['ids'][0] for b in BORN])
            rapidity = .5*np.log(np.array([b['x'][0]/b['x'][1] for b in BORN]))
            z = dict(values=values, reference=reference, q=qs, ids=ids, rapidity=rapidity,
                     scalup=np.array([b['scalup'] for b in BORN]), trials=np.array(pts))
            arrays[name] = z
            np.savez_compressed(work/(name+'.npz'), **z)
            provenance[name] = dict(born=n, trials=sum(map(len, pts)),
                repeats=len(pts[0]), no_emission=int(sum(np.array(pts).ravel()==0)),
                max_direct_gluon_leg_formula_error=max(r[1] for r in results),
                undefined_raw_by_table=dict(zip(NAMES, values[:, :, :, 4].sum(axis=(0, 1)).astype(int).tolist())),
                clipped_legs_by_table=dict(zip(NAMES, values[:, :, :, 3].sum(axis=(0, 1)).astype(int).tolist())),
                input_sha256={str(p): sha(p) for p in [born_path, trials_path, study/(name+'.settings.cmnd')]})
            print(name, 'done;', round(time.monotonic()-started, 1), 's', flush=True)
        provenance['tables'] = {name:e.provenance for name,e in EVALUATORS.items()}
    grid = 'NNPDF23_nlo_as_0119_qed_0000.dat' if pdfset=='15' else 'NNPDF23_lo_as_0130_qed_0000.dat'
    provenance.update(pdf=f'Pythia PDF{pdfset}, same provider and definite flavours as shower',
        pdf_build=build, pdf_bridge_sha256=sha(bridge), pdf_source_sha256=sha(source),
        pdf_grid_sha256=sha(install/'share/Pythia8/pdfdata'/grid),
        checks=['Every trial successful, first emission ISR and below its start',
                'DY raw and rooted archive calculations identical for every Born/cut',
                'Colourless leg factors and total FSR exactly one',
                'Native selector yields two lines per incoming Higgs gluon and one per DY quark',
                'Every finite native gluon factor agrees with independent F*R and F*R^2 formulas; undefined zero-weight cases separately flagged',
                'Fixed/damped samples share exact Born kinematics, flavours and colours',
                'First Born serial/forked results bitwise identical',
                'Delta and shower CDF exactly one for q>=Q'])
    return arrays, provenance


def summarize(arrays, out, pdfset):
    rows = []
    for name, z in arrays.items():
        n = len(z['reference'])
        masks = [('all', np.ones(n, bool))]
        if name.startswith('dy'):
            masks += [('flavour_'+str(i), abs(z['ids'])==i) for i in range(1, 6)]
        masks += [('abs_y_0_1', abs(z['rapidity'])<1),
                  ('abs_y_1_2', (abs(z['rapidity'])>=1)&(abs(z['rapidity'])<2)),
                  ('abs_y_2_inf', abs(z['rapidity'])>=2)]
        for group, mask in masks:
            ids = np.flatnonzero(mask)
            if len(ids)<10: continue
            boots = np.random.default_rng(102828).integers(0, len(ids), size=(500, len(ids)))
            for j in range(z['q'].shape[1]):
                p = z['reference'][mask, j]
                bp = p[boots].mean(axis=1)
                for k, table in enumerate(NAMES):
                    values = z['values'][mask, j, k]
                    d = values[:, 0]; bd = d[boots].mean(axis=1)
                    finite = values[:, 4]==0
                    ratio_se = (bd/bp).std(ddof=1) if np.all(bp>0) else float('nan')
                    rows.append(dict(sample=name, group=group, table=table,
                        cut_kind='GeV' if j<len(ABSOLUTE) else 'fraction_of_start',
                        cut_value=ABSOLUTE[j] if j<len(ABSOLUTE) else RELATIVE[j-len(ABSOLUTE)],
                        q_mean=z['q'][mask, j].mean(), delta=d.mean(), shower=p.mean(),
                        ratio=d.mean()/p.mean() if p.mean()>0 else float('nan'), ratio_se=ratio_se,
                        difference=(d-p).mean(), difference_se=(bd-bp).std(ddof=1),
                        delta_se=bd.std(ddof=1), shower_se=bp.std(ddof=1),
                        undefined_born=int(sum(~finite)), clipped_legs=int(values[:, 3].sum()),
                        ratio_excluding_undefined=d[finite].mean()/p[finite].mean() if finite.any() and p[finite].mean()>0 else float('nan'),
                        born=len(ids)))
    writecsv(out/'common_cut.csv', rows)
    selected = [r for r in rows if r['group']=='all' and r['cut_kind']=='GeV'
                and r['cut_value'] in [5, 10, 20, 30, 50, 75, 100]]
    (out/'selected.json').write_text(json.dumps(selected, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(default='#777777', PDFmode2='#0072B2', PDFmode2_root='#0072B2',
                  PDFmode1='#D55E00', PDFmode1_root='#D55E00')
    for kind, filename in [('GeV', 'comparison'), ('fraction_of_start', 'relative_cut')]:
        order = [name for start in ['fixed', 'damped'] for process in ['dy', 'higgs']
                 for name in arrays if name.removesuffix('_pdf13')==process+'_'+start]
        nrows = (len(order)+1)//2
        fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.5*nrows), sharey=True, squeeze=False)
        for ax, name in zip(axes.flat, order):
            for table in NAMES:
                if name.startswith('dy') and table.endswith('_root'): continue
                rs = [r for r in rows if r['group']=='all' and r['sample']==name and r['table']==table and r['cut_kind']==kind]
                x = [r['cut_value'] for r in rs]
                y = np.array([r['ratio'] if r['undefined_born']==0 else np.nan for r in rs])
                e = np.array([r['ratio_se'] for r in rs])
                style = '--' if table in ['PDFmode1', 'PDFmode2'] and name.startswith('higgs') else '-'
                ax.plot(x, y, linestyle=style, color=colors[table], label=table)
                ax.fill_between(x, y-e, y+e, color=colors[table], alpha=.12)
            ax.axhline(1, color='black', linewidth=.7); ax.grid(alpha=.25)
            ax.set_xscale('log'); ax.set_xlim((2, 150) if kind=='GeV' else (.2, 1))
            ax.set_ylim(0, 1.6); ax.set_title(name.replace('_', ' '))
            ax.set_xlabel('Common cutoff q [GeV]' if kind=='GeV' else 'Common cutoff q / starting scale Q')
            ax.set_ylabel('Mean Delta / mean shower CDF'); ax.legend(fontsize=8)
        fig.suptitle(f'DY and gg → H: Simple Pythia 8.313, PDF{pdfset}, ISR alphaS=.130\n'
                     'Common cutoff in both calculations; shaded bands: Born-bootstrap statistical errors')
        fig.tight_layout(); fig.savefig(out/(filename+'.pdf')); fig.savefig(out/(filename+'.png'), dpi=170)
        plt.close(fig)
    for r in selected:
        if r['table'] in ['default', 'PDFmode1', 'PDFmode1_root']:
            print(f'{r["sample"]:13} {r["table"]:14} q={r["cut_value"]:5.0f} '
                  f'ratio={r["ratio"]:.6f} +/- {r["ratio_se"]:.6f} '
                  f'undefined={r["undefined_born"]}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('study', type=Path); p.add_argument('pythia', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--workers', type=int, default=8); p.add_argument('--reuse', action='store_true')
    p.add_argument('--pdf', choices=['13', '15'], default='15')
    p.add_argument('--sample', action='append', default=[])
    args = p.parse_args()
    study, install, out = args.study.resolve(), args.pythia.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    samples = args.sample or (SAMPLES if args.pdf=='15' else ['dy_fixed_pdf13', 'higgs_fixed_pdf13'])
    work = study/('analysis' if args.pdf=='15' else 'analysis_pdf13')
    if args.reuse:
        arrays = {name:load(work/(name+'.npz')) for name in samples}
        provenance = json.loads((out/'analysis_provenance.json').read_text())
    else:
        arrays, provenance = calculate(study, install, out, args.workers, args.pdf, samples)
    summarize(arrays, out, args.pdf)
    provenance.update(script_sha256=sha(__file__), no_production_changes=True,
        bootstrap_replicas=500, bootstrap_seed=102828,
        definition='Same common cutoff on every live line and in direct first-emission CDF; '
                   'all repeats used, no emission-conditioned probes, equal Born weights.',
        output_sha256={str(work/(name+'.npz')): sha(work/(name+'.npz')) for name in samples})
    (out/'analysis_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__=='__main__':
    main()
