#!/usr/bin/env python3
"""Compare the production FxFx factors with the actual MC@NLO-Delta tables.

The comparison excludes the separate beam PDF factors. All quantities here
are raw Sudakovs, without either matching prescription's compensation terms.
"""

import csv
import ctypes
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import sudakov_audit as audit


HERE, ROOT = Path(__file__).resolve().parent, audit.ROOT
TYPES = {1: 'II', 2: 'FF', 3: 'IF', 4: 'FI'}
MASS = [0.33, 0.33, 0.5, 1.5, 4.8, 0, 0, 0, 0, 0,
        .00051099893, 0, .10565837, 0, 1.77682, 0, 0, 0, 0, 0, 0]


class DeltaTable:
    def __init__(self, work, source, label):
        work.mkdir()
        data = gzip.decompress(source.read_bytes()) if source.suffix == '.gz' else source.read_bytes()
        local = work/'sudakov.f'
        local.write_bytes(data)
        libpath = work/'delta.so'
        cern = ROOT/'Template/NLO/Source/CERNLIB'
        result = subprocess.run(['gfortran', '-O2', '-std=legacy', '-shared', '-fPIC',
            '-ffixed-line-length-none', str(local),
            *(str(cern/f) for f in ('dfint.f','kerset.f','abend.f')), '-o',str(libpath)],
            capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        self.lib = ctypes.CDLL(str(libpath))
        self.lib.pysudakov_.restype = ctypes.c_double
        self.masses = (ctypes.c_double*21)(*MASS)
        self.label = label
        self.provenance = dict(source=str(source.relative_to(ROOT)),
                               sha256_uncompressed=hashlib.sha256(data).hexdigest())

    def value(self, k, mass, pdg, dipole):
        value = self.lib.pysudakov_(*[ctypes.byref(v) for v in (
            ctypes.c_double(k),ctypes.c_double(mass),ctypes.c_int(pdg),ctypes.c_int(dipole))], self.masses)
        assert math.isfinite(value) and 0 <= value <= 1+1e-12, value
        return value

    def interval(self, stop, start, mass, pdg, dipole):
        numerator = self.value(stop,mass,pdg,dipole)
        denominator = self.value(start,mass,pdg,dipole)
        assert denominator > 0, (self.label,stop,start,mass,pdg,dipole)
        value = numerator/denominator
        return value


def fxfx_library(work, asmz=.119, nloop=2):
    work.mkdir()
    lib = audit.build_library(work)
    coupling = audit.Coupling.in_dll(lib,'a_block_')
    coupling.asmz,coupling.nloop = asmz,nloop
    for name in ('sudakov_exp_', 'expanded_sudakov_exp_', 'alphas_'):
        getattr(lib,name).restype = ctypes.c_double
    return lib


def fxfx(lib, stop, start, pdg, finite):
    ctypes.c_int.in_dll(lib,'__fksparams_MOD_fxfxpythia8finiteterms').value = int(finite)
    e = lib.sudakov_exp_(*[ctypes.byref(v) for v in (
        ctypes.c_double(stop),ctypes.c_double(start),ctypes.c_int(3 if pdg==21 else 1),ctypes.c_double(0))])
    return min(math.exp(e),1)


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        out=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        out.writeheader()
        out.writerows(rows)


def main():
    tablesources = [('shipped_Delta',ROOT/'vendor/SudGen/sudakov.f')]
    tablesources += [('archive_pdfmode'+str(i),ROOT/f'vendor/SudGen/sudakov_r515_PDFmode{i}.f.gz') for i in (0,1,2)]
    rows=[]
    with tempfile.TemporaryDirectory(prefix='fxfx_delta_compare_') as tmp:
        work=Path(tmp)
        tables=[DeltaTable(work/label,source,label) for label,source in tablesources]
        lib=fxfx_library(work/'fxfx')
        coupling_control=fxfx_library(work/'fxfx_0130_1loop',.130,1)
        masses=[audit.QHARD,200.,500.]
        for hard in masses:
            lows=np.unique(np.r_[np.geomspace(max(2,hard*.02),hard,90),
                                 [q for q in (8.,10.,15.,20.,21.5,23.,30.,40.) if q<hard]])
            for pdg in (1,2,21):
                # Delta tables describe individual colour lines. Compare a
                # complete gluon with two identical live colour connections.
                lines=2 if pdg==21 else 1
                for q in lows:
                    old=fxfx(lib,q,hard,pdg,False)
                    new=fxfx(lib,q,hard,pdg,True)
                    control=fxfx(coupling_control,q,hard,pdg,True)
                    for dipole,name in TYPES.items():
                        for table in tables:
                            line=table.interval(q,hard,hard,pdg,dipole)
                            delta=min(line**lines,1.)
                            rows.append(dict(table=table.label,pdg=pdg,dipole=name,
                                stop_GeV=q,start_GeV=hard,dipole_mass_GeV=hard,
                                colour_lines=lines,delta_table_per_line=line,
                                delta_table_per_leg=delta,fxfx_legacy=old,fxfx_finite=new,
                                finite_over_delta=new/delta if delta>0 else None,
                                legacy_over_delta=old/delta if delta>0 else None,
                                fxfx_finite_asmz0130_1loop=control))
        # Separate dipole-mass dependence from evolution interval: the FxFx
        # analytic kernel has no independent mass/geometry argument.
        geometry=[]
        hard=audit.QHARD
        for pdg in (2,21):
            lines=2 if pdg==21 else 1
            for q in (20.,23.):
                for m in (hard*.5,hard,hard*2,hard*4):
                    for dipole,name in TYPES.items():
                        table=tables[0]
                        delta=min(table.interval(q,hard,m,pdg,dipole)**lines,1.)
                        new=fxfx(lib,q,hard,pdg,True)
                        geometry.append(dict(pdg=pdg,dipole=name,stop_GeV=q,
                            start_GeV=hard,dipole_mass_GeV=m,delta_table_per_leg=delta,
                            fxfx_finite=new,finite_over_delta=new/delta if delta else None))
        provenance={t.label:t.provenance for t in tables}
    write_csv(HERE/'fxfx_vs_delta_tables.csv',rows)
    write_csv(HERE/'fxfx_vs_delta_geometry.csv',geometry)
    selected=[r for r in rows if r['start_GeV']==audit.QHARD and
              r['stop_GeV'] in (10,15,20,21.5,23,30) and r['table']=='shipped_Delta']
    (HERE/'fxfx_vs_delta_selected.json').write_text(json.dumps(selected,indent=2)+'\n')
    summary=dict(provenance=provenance,fxfx_coupling=dict(asmz=.119,nloop=2),
        alpha_control=dict(asmz=.130,nloop=1),
        delta_normalization='Table(stop,mass)/table(start,mass), one factor per live colour line.',
        notes=['PDF ratios and matching compensations are excluded.',
               'The shipped table generation settings are not encoded in the table source.',
               'Alpha control changes only the analytic factor; it does not identify the table coupling.',
               'Two identical colour connections are assumed for each gluon leg.',
               'Gluons use the symmetric-connection limit (product of the two line ratios), not a complete event Delta evaluation.',
               'Raw table ratios are retained; per-leg probabilities are bounded at one.',
               'The active Delta caller uses type II for all incoming emitters, including IF.'])
    summary['generated_process_tables']={}
    for process in ('w0j','w1j','w0j_new','w1j_new'):
        path=ROOT/process/'SubProcesses/sudakov.f'
        if path.exists():
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==provenance['shipped_Delta']['sha256_uncompressed']
            summary['generated_process_tables'][process]=digest
    overshoots=[r for r in rows if r['delta_table_per_line']>1+1e-12]
    summary['raw_interpolation_overshoots']=dict(count=len(overshoots),
        max_ratio=max((r['delta_table_per_line'] for r in overshoots),default=1))
    (HERE/'fxfx_vs_delta_provenance.json').write_text(json.dumps(summary,indent=2)+'\n')
    with PdfPages(HERE/'fxfx_vs_delta_tables.pdf') as pdf:
        for hard in masses:
            fig,axes=plt.subplots(2,2,figsize=(10,8),sharex='col')
            for col,pdg in enumerate((2,21)):
                ax,ratio=axes[:,col]
                baseline=[r for r in rows if r['start_GeV']==hard and r['pdg']==pdg
                          and r['dipole']=='II' and r['table']=='shipped_Delta']
                x=np.array([r['stop_GeV'] for r in baseline])
                ax.plot(x,[r['fxfx_finite'] for r in baseline],color='black',label='FxFx finite')
                ax.plot(x,[r['fxfx_legacy'] for r in baseline],color='0.5',ls='--',label='FxFx legacy')
                for dipole,colour in [('II','#0072b2'),('FF','#d55e00'),('IF','#009e73'),('FI','#cc79a7')]:
                    a=[r for r in rows if r['start_GeV']==hard and r['pdg']==pdg
                       and r['dipole']==dipole and r['table']=='shipped_Delta']
                    ax.plot(x,[r['delta_table_per_leg'] for r in a],label='Delta '+dipole,color=colour)
                    ratio.plot(x,[r['finite_over_delta'] for r in a],color=colour)
                ax.set_title('Quark leg' if pdg==2 else 'Gluon leg (two identical colour lines)')
                ax.set_ylabel('Raw Sudakov')
                ratio.set_ylabel('FxFx finite / Delta')
                if pdg==21:
                    ratio.set_yscale('log')
                ratio.set_xlabel('Stopping scale [GeV]')
                ratio.axhline(1,color='0.4',lw=.8)
                ax.legend(fontsize=8)
                for a in (ax,ratio):
                    a.set_xscale('log')
                    a.grid(alpha=.2)
                    a.set_xlim(max(2,hard*.02),hard)
                if hard==audit.QHARD:
                    for a in (ax,ratio): a.axvspan(20,23,color='0.5',alpha=.12)
            fig.suptitle(f'Production FxFx versus shipped Delta tables: start = dipole mass = {hard:.3f} GeV')
            fig.text(.1,.015,'FxFx: alpha_s(mZ)=0.119, two loops, A2 included. Delta: shipped tables with their original settings.\n'
                     'PDF factors and matching compensations excluded; this is not a complete event Delta factor.',fontsize=8)
            fig.tight_layout(rect=(0,.07,1,.95))
            pdf.savefig(fig)
            if hard==audit.QHARD: fig.savefig(HERE/'fxfx_vs_delta_tables.png',dpi=160)
            plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True)
    hard=audit.QHARD
    baseline=[r for r in rows if r['start_GeV']==hard and r['pdg']==2
              and r['dipole']=='II' and r['table']=='shipped_Delta']
    x=[r['stop_GeV'] for r in baseline]
    axes[0].plot(x,[r['fxfx_finite']**2 for r in baseline],color='black',label='FxFx finite')
    axes[0].plot(x,[r['fxfx_legacy']**2 for r in baseline],color='0.5',ls='--',label='FxFx legacy')
    for table,label,colour in [('shipped_Delta','Delta shipped (active)','#0072b2'),
        ('archive_pdfmode1','Delta archive, momentum tagging','#009e73'),
        ('archive_pdfmode2','Delta archive, z > 1/2','#d55e00')]:
        selected=[r for r in rows if r['start_GeV']==hard and r['pdg']==2
                  and r['dipole']=='II' and r['table']==table]
        down={r['stop_GeV']:r for r in rows if r['start_GeV']==hard and r['pdg']==1
              and r['dipole']=='II' and r['table']==table}
        pair=[r['delta_table_per_leg']*down[r['stop_GeV']]['delta_table_per_leg'] for r in selected]
        axes[0].plot(x,pair,color=colour,label=label)
        axes[1].plot(x,[r['fxfx_finite']**2/d for r,d in zip(selected,pair)],color=colour)
    for a in axes:
        a.axvspan(20,23,color='0.5',alpha=.12)
        a.set_xlim(8,40)
        a.grid(alpha=.2)
    axes[0].set_ylabel(r'Two-quark raw Sudakov $\Delta_q^2$')
    axes[0].legend(fontsize=9)
    axes[1].axhline(1,color='0.5',lw=.8)
    axes[1].set_ylabel('FxFx finite / Delta (pair)')
    axes[1].set_xlabel('Stopping scale [GeV]')
    fig.suptitle('Incoming u dbar pair: production FxFx versus stored Delta tables')
    fig.text(.12,.015,r'$Q=m_{dip}=m_W$; FxFx uses $\alpha_s(m_Z)=0.119$, two loops and A2.'+'\n'
             'The stored tables retain their own generation settings. PDF ratios and matching compensations excluded.',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.95))
    fig.savefig(HERE/'fxfx_vs_delta_quark_pair.pdf')
    fig.savefig(HERE/'fxfx_vs_delta_quark_pair.png',dpi=160)
    plt.close(fig)
    for r in selected:
        if r['stop_GeV'] in (20.,23.):
            print(f"pdg={r['pdg']:2} {r['dipole']} q={r['stop_GeV']:4.1f} "
                  f"Delta={r['delta_table_per_leg']:.6f} FxFx={r['fxfx_finite']:.6f} "
                  f"ratio={r['finite_over_delta']:.4f}")


if __name__=='__main__':
    main()
