#!/usr/bin/env python3
"""Evaluate both production FxFx schemes at the W-study coupling."""

import csv
import ctypes
import json
import math
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sudakov_audit as audit


HERE = Path(__file__).resolve().parent


def evaluate():
    rows = []
    with tempfile.TemporaryDirectory(prefix='fxfx_pythia_finite_') as tmp:
        lib = audit.build_library(Path(tmp))
        coupling = audit.Coupling.in_dll(lib, 'a_block_')
        coupling.asmz, coupling.nloop = .119, 2
        setting = ctypes.c_int.in_dll(lib, '__fksparams_MOD_fxfxpythia8finiteterms')
        for name in ('alphas_', 'sudakov_exp_', 'expanded_sudakov_exp_'):
            getattr(lib, name).restype = ctypes.c_double
        lows = np.unique(np.r_[np.linspace(5, 79, 149), 8, 10, 15, 20, 21.5, 23, 30])
        for q in lows:
            args = [ctypes.byref(x) for x in (ctypes.c_double(q),
                ctypes.c_double(audit.QHARD), ctypes.c_int(1), ctypes.c_double(0))]
            alpha = lib.alphas_(ctypes.byref(ctypes.c_double(q)))
            for enabled in (False, True):
                setting.value = int(enabled)
                e = 2*lib.sudakov_exp_(*args)
                e1 = 2*lib.expanded_sudakov_exp_(*args)
                delta = min(math.exp(e), 1)
                first = alpha*min(e1, 0)
                rows.append(dict(scheme='Pythia8_finite' if enabled else 'legacy_NLL',
                    q_GeV=q, Q_GeV=audit.QHARD, delta_raw=math.exp(e),
                    delta_applied=delta, expanded_coefficient=e1,
                    expanded_applied=first, born_plus_compensation=delta*(1-first)))
    return rows


def main():
    rows = evaluate()
    with (HERE / 'pythia_finite_terms.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    selected = [r for r in rows if r['q_GeV'] in (8, 10, 15, 20, 21.5, 23, 30)]
    (HERE / 'pythia_finite_terms_selected.json').write_text(json.dumps(selected, indent=2)+'\n')
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for scheme, label, colour in [('legacy_NLL', 'Standard NLL', '#0072b2'),
        ('Pythia8_finite', 'Pythia8-inspired finite terms', '#d55e00')]:
        data = [r for r in rows if r['scheme'] == scheme]
        axes[0].plot([r['q_GeV'] for r in data], [r['delta_applied'] for r in data],
                     label=label, color=colour)
        axes[1].plot([r['q_GeV'] for r in data],
                     [r['born_plus_compensation'] for r in data], color=colour)
    for ax in axes:
        ax.axvspan(20, 23, color='0.5', alpha=.12)
        ax.grid(alpha=.2)
        ax.set_xlim(5, 45)
    axes[0].set_ylabel(r'$\Delta_{q\bar q}(q,Q)$')
    axes[1].set_ylabel(r'$\Delta\,[1-\Delta^{(1)}]$')
    axes[1].set_xlabel('Lower scale q [GeV]')
    axes[0].legend()
    fig.suptitle(r'Two massless quarks, $Q=m_W$, $\alpha_s(m_Z)=0.119$, two-loop running')
    fig.text(.13, .02, 'Lower panel: Born term and its NLO compensator only.\n'
             'The finite-term prescription is not an event-dependent Pythia no-emission probability.', fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(HERE / 'pythia_finite_terms.pdf')
    fig.savefig(HERE / 'pythia_finite_terms.png', dpi=160)
    for r in selected:
        print(f"{r['scheme']:16} q={r['q_GeV']:4.1f} delta={r['delta_applied']:.8f} "
              f"Born+comp={r['born_plus_compensation']:.8f}")


if __name__ == '__main__':
    main()
