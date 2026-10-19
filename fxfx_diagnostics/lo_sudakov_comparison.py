#!/usr/bin/env python3
"""Compare the actual LO-template analytic kernels with FxFx at common alpha_s.

The LO analytic function is used for FSR in its ickkw=2 path. Its mode 2 is
also evaluated to document the second parameterization in sudakov.inc.
These kernel comparisons are not complete hadronic LO merging predictions:
the LO ISR path calls a separate x- and PDF-dependent grid.
"""

from pathlib import Path
import csv
import ctypes
import json
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sudakov_audit as nlo
from tests.unit_tests.fks.test_momentum_maps import fortran_routine


HERE = Path(__file__).resolve().parent
LO = HERE.parent / 'Template/LO'


class Settings(ctypes.Structure):
    _fields_ = [('color', ctypes.c_double * 28),
                ('double_log', ctypes.c_double * 28),
                ('single_log', ctypes.c_double * 28),
                ('powers', ctypes.c_double * (28*3*2)),
                ('masses', ctypes.c_double * 21),
                ('as_factor', ctypes.c_double), ('K', ctypes.c_double),
                ('last_as', ctypes.c_double), ('pca', ctypes.c_int * (28*2))]


def build_lo(work):
    includes = {
        'nexternal.inc': '', 'maxamps.inc': '', 'maxparticles.inc': '',
        'message.inc': '      integer mlevel\n      common/test_messages/mlevel\n',
        'run.inc': '      double precision alpsfact\n      common/test_run/alpsfact\n',
    }
    source = (LO / 'SubProcesses/cluster.inc').read_text()
    includes['cluster.inc'] = source[source.index('      double precision m_colfac'):source.index('      integer vec_igraph')]
    for name, text in includes.items():
        (work / name).write_text(text)
    shutil.copyfile(LO / 'SubProcesses/sudakov.inc', work / 'sudakov.inc')
    shutil.copyfile(nlo.TEMPLATE / 'Source/alfas.inc', work / 'alfas.inc')
    shutil.copyfile(nlo.TEMPLATE / 'SubProcesses/timing_variables.inc', work / 'timing_variables.inc')
    routines = [fortran_routine(LO / 'SubProcesses/reweight.f', name)
                for name in ('gamma', 'sud', 'sudwgt')]
    # Use exactly the same running coupling as the FxFx audit, to isolate the
    # kernel choices from the PDF set, loop order, and alpsfact setting.
    routines.append((nlo.TEMPLATE / 'Source/alfas_functions.f').read_text())
    (work / 'lo_kernels.f').write_text('\n'.join(routines))
    target = work / 'lo_sudakov.so'
    subprocess.run(['gfortran', '-O2', '-std=legacy', '-shared', '-fPIC',
                    '-ffixed-line-length-none', '-I', str(work),
                    str(work / 'lo_kernels.f'), str(LO / 'Source/dgauss.f'),
                    '-o', str(target)], check=True, capture_output=True, text=True)
    lib = ctypes.CDLL(str(target))
    coupling = nlo.Coupling.in_dll(lib, 'a_block_')
    coupling.asmz, coupling.nloop = 0.119, 2
    ctypes.c_double.in_dll(lib, 'test_run_').value = 1
    ctypes.c_int.in_dll(lib, 'test_messages_').value = 0
    lib.sud_.restype = ctypes.c_double
    return lib


def main():
    with tempfile.TemporaryDirectory(prefix='lo_fxfx_sudakov_') as tmp:
        work = Path(tmp)
        lp, npth = work / 'lo', work / 'nlo'
        lp.mkdir(); npth.mkdir()
        lo = build_lo(lp)
        fxfx = nlo.build_library(npth)
        coupling = nlo.Coupling.in_dll(fxfx, 'a_block_')
        coupling.asmz, coupling.nloop = 0.119, 2
        fxfx.sudakov_exp_.restype = ctypes.c_double
        settings = Settings.in_dll(lo, 'cl_sud_')
        assert settings.K == 0
        q = np.unique(np.r_[np.linspace(5,45,161),20,21.5,23])
        rows = []
        for low in q:
            exponent = 2 * fxfx.sudakov_exp_(ctypes.byref(ctypes.c_double(low)),
                         ctypes.byref(ctypes.c_double(nlo.QHARD)),
                         ctypes.byref(ctypes.c_int(1)), ctypes.byref(ctypes.c_double(0)))
            row = dict(q_GeV=low, Q_GeV=nlo.QHARD, fxfx_NLL=min(float(np.exp(exponent)),1.0))
            for mode in [1,2]:
                settings.K = 0
                delta = lo.sud_(ctypes.byref(ctypes.c_double(low)),
                               ctypes.byref(ctypes.c_double(nlo.QHARD)),
                               ctypes.byref(ctypes.c_int(2)), ctypes.byref(ctypes.c_int(mode)))
                row[f'LO_mode{mode}_default_K0'] = delta**2
                settings.K = 3*(67/18-np.pi**2/6)-5*5/9
                delta = lo.sud_(ctypes.byref(ctypes.c_double(low)),
                               ctypes.byref(ctypes.c_double(nlo.QHARD)),
                               ctypes.byref(ctypes.c_int(2)), ctypes.byref(ctypes.c_int(mode)))
                row[f'LO_mode{mode}_with_K'] = delta**2
            rows.append(row)
    with (HERE / 'lo_vs_fxfx_sudakov.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    selected = [r for r in rows if r['q_GeV'] in [20,21.5,23]]
    (HERE / 'lo_vs_fxfx_sudakov_selected.json').write_text(json.dumps(selected, indent=2)+'\n')
    print(json.dumps(selected, indent=2))
    fig, ax = plt.subplots(figsize=(8,5.5))
    for key, label in [('fxfx_NLL','FxFx NLL, final cap'),
                       ('LO_mode1_default_K0','LO mode 1 (FSR), default K=0'),
                       ('LO_mode1_with_K','LO mode 1, with NLL K coefficient'),
                       ('LO_mode2_default_K0','LO mode 2, default K=0')]:
        ax.plot(q,[r[key] for r in rows],label=label)
    ax.axvspan(20,23,color='0.5',alpha=0.12)
    ax.set(xlabel='Lower scale q [GeV]',ylabel=r'$\Delta_q(q,Q)^2$',xlim=(5,45),ylim=(0.4,1.02),
           title=r'Actual template kernels at $Q=m_W$, common $\alpha_s(m_Z)=0.119$')
    ax.grid(alpha=0.2); ax.legend()
    fig.text(0.12,0.02,'Common two-loop running and alpsfact=1; two massless quark lines.\n'
                        'The analytic LO FSR kernel is not the separate LO ISR grid.',fontsize=8)
    fig.tight_layout(rect=(0,0.09,1,1))
    fig.savefig(HERE/'lo_vs_fxfx_sudakov.pdf')
    fig.savefig(HERE/'lo_vs_fxfx_sudakov.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
