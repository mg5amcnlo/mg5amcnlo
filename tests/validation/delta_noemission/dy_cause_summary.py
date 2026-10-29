#!/usr/bin/env python3
"""Compare the central DY shower, kernel integrals and native table caller."""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np

from delta_evaluator import DeltaEvaluator, ROOT
from singlet_run_showers import sha


def main():
    p = argparse.ArgumentParser()
    for key in ['matched', 'output', 'work']:
        p.add_argument(key, type=Path)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    central_path = a.output/'central_probe.json'
    quadrature_path = a.output/'phase_space_quadrature.json'
    central = json.loads(central_path.read_text())
    quadrature = json.loads(quadrature_path.read_text())
    bridge = a.matched/'pdf13/analysis_pdf13/pythia_pdf.so'
    lib = C.CDLL(str(bridge))
    lib.pythia_pdf_init.argtypes = [C.c_char_p, C.c_char_p]
    lib.pythia_pdf_xf.argtypes = [C.c_int, C.c_double, C.c_double]
    lib.pythia_pdf_xf.restype = C.c_double
    assert lib.pythia_pdf_init(central['jobs'][0]['command'][1].encode(), b'13')==0
    def pdf(pdg, x, q):
        return lib.pythia_pdf_xf(pdg, x, q)
    born_path = Path(central['jobs'][0]['prefix']+'.born.jsonl')
    born = json.loads(born_path.read_text())
    rows = []
    table = ROOT/'vendor/SudGen/sudakov_r515_PDFmode1.f.gz'
    with DeltaEvaluator(work=a.work/'native', table=table) as evaluator:
        for measured in central['rows']:
            cut = measured['q']
            q = next(r for r in quadrature['rows'] if r['cut']==cut)
            event = q['events']['2']
            native = evaluator.evaluate(born, cut, pdf)
            assert native['fsr']==1 and not any(native['nonfinite_raw'])
            assert abs(native['delta']-event['archive_times_pdf_probability'])<1.e-12
            prob = measured['no_emission_probability']
            error = measured['standard_error']
            rows.append(dict(q=cut, shower=prob, shower_se=error,
                backward_quadrature=event['backward_kernel_probability'],
                fresh_forward_times_pdf=event['forward_times_pdf_probability'],
                native_delta=native['delta'],
                backward_quadrature_pull=(event['backward_kernel_probability']-prob)/error,
                native_over_shower=native['delta']/prob,
                fresh_forward_over_shower=event['forward_times_pdf_probability']/prob))
        provenance = evaluator.provenance
    result = dict(rows=rows, born=central['born'], native_provenance=provenance,
        source_sha256={str(f):sha(f) for f in
            [Path(__file__),central_path,quadrature_path,bridge,born_path,table]},
        definition='Same central u-ubar Born point, Q=M=91.1876GeV. Direct empirical '
            'first-emission CDF from100000fullSimple showers versus continuous backward '
            'kernel quadrature and unmodified native forward-table/PDF product.',
        checks=['Native Delta agrees with independent archived-table/PDF product at all6cuts',
                'Quadrature and full-shower probabilities compared with binomial sampling errors'],
        limitations=['One central light-quark Born point, not an attribution for every Born flavour',
                     'Quadrature omits explicit valence/sea bookkeeping and finite sister masses',
                     'Quoted shower sampling errors exclude modeling and archived-grid uncertainty'])
    (a.output/'central_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5,4.5))
    x = [r['q'] for r in rows]
    ax.errorbar(x,[r['shower'] for r in rows],yerr=[r['shower_se'] for r in rows],
                fmt='ko', label='Actual Simple shower (100,000 trials)', capsize=3)
    for key, label, style in [
        ('backward_quadrature','Backward kernel with shower phase space','-'),
        ('fresh_forward_times_pdf','Fresh forward Sudakov × PDF ratios','--'),
        ('native_delta','Archived mode 1 × PDF ratios (native caller)',':')]:
        ax.plot(x,[r[key] for r in rows],style,label=label)
    ax.set_xscale('log');ax.set_ylim(0,1.03);ax.grid(alpha=.25)
    ax.set_xlabel('Common cutoff q [GeV]');ax.set_ylabel('No-emission probability')
    ax.set_title('Central u ubar → e− e+: Q = M = 91.1876 GeV, PDF13')
    ax.legend(fontsize=8);fig.tight_layout()
    for suffix in ['pdf','png']:
        fig.savefig(a.output/('central_comparison.'+suffix),dpi=170)
    plt.close(fig)
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    main()
