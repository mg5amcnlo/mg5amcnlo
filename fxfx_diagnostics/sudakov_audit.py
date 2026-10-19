#!/usr/bin/env python3
"""Evaluate production FxFx Sudakovs and controlled finite-scale alternatives.

Alternatives are diagnostics. This script does not change a generation driver.
The positive-rate prescription clips the branching integrand; the finite-z
illustration retains the power terms in the integral of Pqq over 0<z<1-q/Q.
Neither illustration is a prediction of a Pythia no-emission probability.
"""

from pathlib import Path
import csv
import ctypes
import json
import math
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from tests.unit_tests.fks.test_momentum_maps import fortran_routine

TEMPLATE = ROOT / 'Template/NLO'
CF = 4.0 / 3.0
QHARD = 80.419002445756163


class Coupling(ctypes.Structure):
    _fields_ = [('asmz', ctypes.c_double), ('nloop', ctypes.c_int)]


class GammaState(ctypes.Structure):
    _fields_ = [('Q', ctypes.c_double), ('mass', ctypes.c_double),
                ('type', ctypes.c_int), ('mode', ctypes.c_int)]


def build_library(work):
    (work / 'cuts.inc').write_text('      integer maxjetflavor\n'
                                  '      parameter(maxjetflavor=5)\n')
    (work / 'orders.inc').write_text('      integer nsplitorders\n'
                                   '      parameter(nsplitorders=1)\n')
    shutil.copyfile(TEMPLATE / 'Source/alfas.inc', work / 'alfas.inc')
    shutil.copyfile(TEMPLATE / 'SubProcesses/timing_variables.inc', work / 'timing_variables.inc')
    routines = [fortran_routine(TEMPLATE / 'SubProcesses/cluster.f', name)
                for name in ('sudakov_exp', 'expanded_sudakov_exp', 'Gamma')]
    routines.append((TEMPLATE / 'Source/alfas_functions.f').read_text())
    # Only the massless-quark kernel is evaluated. Gamma initializes this list
    # for its unused gluon-to-heavy-quark branch as well.
    routines.append('      double precision function get_mass_from_id(i)\n'
                    '      integer i\n      get_mass_from_id=0d0\n'
                    '      if (i.eq.6) get_mass_from_id=173d0\n      end\n')
    (work / 'kernels.f').write_text('\n'.join(routines))
    library = work / 'sudakov.so'
    subprocess.run(['gfortran', '-O2', '-std=legacy', '-shared', '-fPIC',
                    '-ffixed-line-length-none', '-I', str(work),
                    '-J', str(work), str(TEMPLATE / 'SubProcesses/FKSParams.f90'),
                    str(work / 'kernels.f'), str(TEMPLATE / 'Source/dgauss.f'),
                    '-o', str(library)], check=True, capture_output=True, text=True)
    return ctypes.CDLL(str(library))


def evaluate(library):
    coupling = Coupling.in_dll(library, 'a_block_')
    coupling.asmz, coupling.nloop = 0.119, 2
    state = GammaState.in_dll(library, 'to_sud_exp_')
    for name in ('alphas_', 'gamma_', 'sudakov_exp_', 'expanded_sudakov_exp_'):
        getattr(library, name).restype = ctypes.c_double

    def alphas(k):
        return library.alphas_(ctypes.byref(ctypes.c_double(k)))

    assert abs(alphas(20) - 0.1550028093532384) < 1e-9, 'Coupling differs from the generated W study'

    def gamma(k, mode):
        state.Q, state.mass, state.type, state.mode = QHARD, 0.0, 1, mode
        return library.gamma_(ctypes.byref(ctypes.c_double(k)))

    def exponent(q, function):
        # Alpha_s uses a piecewise-linear grid in log(q). Resolve its knots
        # explicitly in the independent integrator.
        knots = np.exp(np.linspace(math.log(4.7), math.log(100000), 256))
        points = [k for k in knots if q < k < QHARD]
        return 2.0 * quad(function, q, QHARD, points=points,
                          epsabs=1e-9, epsrel=1e-9, limit=200)[0]

    def production_exponent(q, name):
        return 2.0 * getattr(library, name)(
            ctypes.byref(ctypes.c_double(q)), ctypes.byref(ctypes.c_double(QHARD)),
            ctypes.byref(ctypes.c_int(1)), ctypes.byref(ctypes.c_double(0)))

    def kernel(k, variant, mode):
        a = alphas(k) if mode == 2 else 1.0
        if variant == 'LL':
            return -2 * CF * a / (math.pi * k) * math.log(QHARD / k)
        value = gamma(k, mode)
        if variant == 'positive_NLL_rate':
            return min(value, 0.0)
        if variant == 'finite_z_illustration':
            r = k / QHARD
            value -= 2 * CF * a / (math.pi * k) * (r - r*r/4)
        return value

    rows = []
    qvalues = np.unique(np.r_[np.linspace(5, 45, 161), 8, 10, 15, 18, 20, 21.5, 23, 30])
    for q in qvalues:
        L = math.log(QHARD / q)
        closed_form = -4 * CF / math.pi * (L*L/2 - 3*L/4)
        for variant in ['production_NLL', 'LL', 'positive_NLL_rate', 'finite_z_illustration']:
            if variant == 'production_NLL':
                E = production_exponent(q, 'sudakov_exp_')
                E1 = production_exponent(q, 'expanded_sudakov_exp_')
                assert abs(E1 - closed_form) < 1e-10, 'Wrong first-order normalization'
                independent = exponent(q, lambda k: gamma(k, 2))
                assert abs(E - independent) < 1e-5, ('Quadratures disagree', q, E, independent)
            else:
                E = exponent(q, lambda k: kernel(k, variant, 2))
                E1 = exponent(q, lambda k: kernel(k, variant, 1))
            raw = math.exp(E)
            delta = min(raw, 1.0)
            first = alphas(q) * min(E1, 0.0)
            rows.append(dict(variant=variant, q_GeV=q, Q_GeV=QHARD, alpha_s_q=alphas(q),
                             log_delta=E, delta_raw=raw, delta_applied=delta,
                             expanded_coefficient_raw=E1, expanded_applied=first,
                             born_plus_compensation=delta*(1-first)))
    return rows


def main():
    with tempfile.TemporaryDirectory(prefix='fxfx_sudakov_audit_') as tmp:
        rows = evaluate(build_library(Path(tmp)))
    with (HERE / 'sudakov_factors.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8), sharex=True)
    colors = ['#0072b2', '0.4', '#d55e00', '#009e73']
    labels = ['Production NLL, capped at 1', 'LL only', 'NLL with positive branching rate',
              'Finite-z illustration']
    for name, label, color in zip(['production_NLL', 'LL', 'positive_NLL_rate', 'finite_z_illustration'], labels, colors):
        selected = [r for r in rows if r['variant'] == name]
        x = [r['q_GeV'] for r in selected]
        axes[0].plot(x, [r['delta_applied'] for r in selected], color=color, label=label)
        axes[1].plot(x, [r['born_plus_compensation'] for r in selected], color=color)
    for ax in axes:
        ax.axvspan(20, 23, color='0.5', alpha=0.12)
        ax.grid(alpha=0.2)
        ax.set_xlim(5, 45)
    axes[0].set_ylabel(r'$\Delta_{q\bar q}(q,Q)$')
    axes[1].set_ylabel(r'$\Delta\,[1-\Delta^{(1)}]$ (Born + compensation)')
    axes[1].set_xlabel('Lower scale q [GeV]')
    axes[0].legend(fontsize=9)
    fig.suptitle(r'Two incoming quarks, $Q=m_W=80.419$ GeV, $\alpha_s(m_Z)=0.119$')
    fig.text(0.12, 0.015, 'Lower panel isolates the Born term and its compensator; it is not a complete NLO cross section.\n'
                         'Alternative kernels illustrate finite-scale sensitivity and are not Pythia predictions.', fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(HERE / 'sudakov_factors.pdf')
    fig.savefig(HERE / 'sudakov_factors.png', dpi=160)
    plt.close(fig)
    chosen = [r for r in rows if r['q_GeV'] in (10, 15, 20, 21.5, 23)]
    (HERE / 'sudakov_factors_selected.json').write_text(json.dumps(chosen, indent=2)+'\n')
    for r in chosen:
        print(f"{r['variant']:24} q={r['q_GeV']:4.1f} raw={r['delta_raw']:.6f} "
              f"applied={r['delta_applied']:.6f} Born+comp={r['born_plus_compensation']:.6f}")
    print('Production first-order coefficients and independent quadratures agree.')


if __name__ == '__main__':
    main()
