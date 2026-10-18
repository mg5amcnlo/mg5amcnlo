#!/usr/bin/env python3
"""Aggregate and plot the controlled W+1j finite-term comparison."""

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from madgraph.various.histograms import HwUList


def aggregate(paths, target):
    result = None
    for path in paths:
        histos = HwUList(str(path))
        if result is None:
            result = histos
        else:
            assert [(h.title, h.type) for h in result] == [(h.title, h.type) for h in histos]
            result = HwUList([a+b for a, b in zip(result, histos)])
    result.output(str(target), format='HwU', auto_open=False)
    return result


def array(histogram):
    return np.array([(*b.boundaries, b.wgts['central'], b.wgts['stat_error'])
                     for b in histogram.bins])


def main():
    records = json.loads((HERE/'w1j_finite_fo_runs.json').read_text())
    data, rates, selected, output = {}, {}, {}, []
    for variant in ('finite_legacy', 'finite_pythia'):
        paths = [Path(r['directory'])/'MADatNLO.HwU' for r in records if r['variant']==variant]
        histos = aggregate(paths, HERE/('w1j_'+variant))
        for kind in ('NLO', 'LO'):
            rate = next(h for h in histos if h.title=='total rate' and h.type==kind)
            rates[variant+'_'+kind] = array(rate)[0,2:].tolist()
            djr = next(h for h in histos if h.title=='log10 DJR 0-1/GeV' and h.type==kind)
            data[variant+'_'+kind] = a = array(djr)
            selected[variant+'_'+kind] = a[np.isclose(a[:,0],1.3)][0,2:].tolist()
            for lo, hi, value, error in a:
                output.append([variant, kind, lo, hi, value, error])
    np.savez(HERE/'finite_fo_djr.npz', **data)
    with (HERE/'finite_fo_djr.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['scheme','order','log10_djr_low','log10_djr_high','bin_pb','stat_pb'])
        writer.writerows(output)
    summary = dict(rates_pb=rates, djr_19p95_25p12_pb=selected,
        note='Six subprocesses, 2 x 65536 points each, corrected Born-scale restoration. '
             'Schemes share initial grids and random seeds; their integration errors are correlated. '
             'These are fixed-order FxFx-weighted results before showering. '
             'LO curves have no NLO compensation; NLO curves include it.')
    (HERE/'finite_fo_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(8.2, 7.8), sharex=True,
                                  gridspec_kw={'height_ratios':[3,1]})
    for kind, style in [('NLO','-'), ('LO','--')]:
        ref = data['finite_legacy_'+kind]
        for variant, label, colour in [('finite_legacy','Standard NLL','#0072b2'),
                                      ('finite_pythia','Pythia8-inspired finite terms','#d55e00')]:
            a = data[variant+'_'+kind]
            x = 10**((a[:,0]+a[:,1])/2)
            keep = (x>=8)&(x<=160)&(ref[:,2]>0)&(a[:,2]>0)
            width = a[:,1]-a[:,0]
            y,e = a[:,2]/width,a[:,3]/width
            ax.plot(x[keep],y[keep],style,color=colour,label=label+' ('+kind+')')
            ax.fill_between(x[keep],(y-e)[keep],(y+e)[keep],color=colour,alpha=.10)
            if variant == 'finite_pythia':
                ratio.plot(x[keep],a[keep,2]/ref[keep,2],style,color=colour,label=kind)
    ax.set_yscale('log')
    ratio.set_xscale('log')
    ratio.set_xlim(8,160)
    ratio.set_ylim(.5,1.15)
    ratio.axhline(1,color='0.4',lw=.8)
    for a in (ax,ratio):
        a.axvspan(20,23,color='0.5',alpha=.12)
        a.grid(alpha=.2)
    ratio.set_xticks([10,20,30,40,60,100,160],labels=['10','20','30','40','60','100','160'])
    ratio.set_xlabel(r'$d_{1\to0}$ [GeV], $k_T$, $R=1$')
    ratio.set_ylabel('Finite / standard')
    ax.set_ylabel(r'$d\sigma/d\log_{10}(d_{1\to0}/{\rm GeV})$ [pb]')
    ax.legend(fontsize=9)
    ratio.legend(fontsize=8)
    fig.suptitle('W+1j: FxFx finite terms, before showering')
    fig.text(.13,.015,'All six subprocesses, 2 x 65536 integration points each; identical starting grids and seeds.\n'
             'NLO includes the finite-term compensation. Bands show individual integration errors.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.95))
    fig.savefig(HERE/'finite_fo_comparison.pdf')
    fig.savefig(HERE/'finite_fo_comparison.png',dpi=160)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
