#!/usr/bin/env python3
"""Plots and tables for ``per_event_weight.py``: is ``<w | production event>``
a constant, and what does a per-event bound buy?

Produces

``plots/per_event_mean_weight.png``
    the decisive picture.  ``A_e = <w | e>`` against ``sqrt(shat)``, its
    decomposition into ``<J>`` and ``<jac_BW . Zhat>``, the distribution of
    ``A_e/<A>`` against the pure-Monte-Carlo spread, and the resulting
    distortion of the ``sqrt(shat)`` spectrum.

``plots/per_event_bound.png``
    the per-event maximum ``max_e w`` against ``sqrt(shat)`` with the global
    bound and the *analytic* 2 -> 2 bound
    ``1.1 . <jac_BW.Zhat> . |p'|(m_lo,m_lo)/|p|(m_pole,m_pole)`` drawn on, and
    the per-event acceptance cost it would give.

Usage::

    export PATH="$HOME/.pyenv/versions/mg-3.12/bin:$PATH"
    python3 doc/madspin_pa_mass_stage/analyse_per_event.py \
        --data <dir with per_event.npz> --plots doc/madspin_pa_mass_stage/plots \
        --out doc/madspin_pa_mass_stage/data
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt        # noqa: E402

pjoin = os.path.join

_here = os.path.dirname(os.path.realpath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from jacobian_analytic import analytic_bound, analytic_A, MT, WT   # noqa: E402

BLUE, RED, GREY, GREEN = '#1f77b4', '#d62728', '#555555', '#2ca02c'


def profile(x, y, edges, weights=None):
    """Mean of ``y`` in bins of ``x`` (optionally a ratio of sums)."""
    idx = np.digitize(x, edges) - 1
    mids, vals, errs, counts = [], [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        n = int(sel.sum())
        if n < 3:
            continue
        mids.append(0.5 * (edges[b] + edges[b + 1]))
        if weights is None:
            vals.append(y[sel].mean())
            errs.append(y[sel].std() / math.sqrt(n))
        else:
            vals.append(y[sel].sum() / weights[sel].sum())
            errs.append(0.0)
        counts.append(n)
    return np.array(mids), np.array(vals), np.array(errs), np.array(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--plots', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--ztables', default=None,
                        help='the same cache per_event_weight.py used; enables '
                             'the noise-free analytic A_e curve')
    args = parser.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    d = np.load(pjoin(args.data, 'per_event.npz'))
    with open(pjoin(args.data, 'per_event.json')) as fp:
        summary = json.load(fp)

    sqrts = d['sqrts']
    A = d['A']                 # <w|e>, infeasible counted as 0
    Aerr = d['A_err']
    Af = d['Af']               # <w|e, feasible>
    Anz = d['A_nozhat']
    n = d['n']
    nfeas = d['nfeas']
    jbar = d['sum_jac'] / np.maximum(nfeas, 1)
    maxw = d['max_w']
    Abar = float(A.mean())
    bound = summary.get('overweight', {}).get('bound')

    tabs = None
    if args.ztables:
        with open(args.ztables) as fp:
            zt = json.load(fp)['z_tables']
        keys = sorted(zt, key=lambda k: -int(k.split('_')[0]))
        tabs = (zt[keys[0]], zt[keys[1]])

    # distance above the t t~ threshold: the natural variable, since
    # J = |p'|/|p| and |p| -> 0 there.  Log scale, so the last GeV -- where
    # the whole effect lives -- is not squeezed into one pixel.
    above = sqrts - 2 * MT

    # ---------------------------------------------------------------- fig 1
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0][0]
    a.plot(above, A / Abar, '.', ms=2.5, color=BLUE, alpha=.4, rasterized=True,
           label=r'$A_e$, %g free mass sets per event' % summary['draws_base'])
    if tabs:
        grid = np.exp(np.linspace(math.log(0.05), math.log(3600), 300))
        a.plot(grid, [analytic_A(2 * MT + g, tabs) / Abar for g in grid], '-',
               lw=1.8, color=RED,
               label=r'exact quadrature, $A_e=\frac{1}{\pi^2}\int\!dR_1dR_2\,'
                     r"J\hat Z\hat Z$")
    a.axhline(1.0, color=GREY, lw=1.2, ls='--',
              label=r'the single global $\langle w\rangle$ MadSpin uses')
    a.set_xscale('log')
    a.set_xlim(0.05, 3600)
    a.set_yscale('log')
    a.set_xlabel(r'$\sqrt{\hat s} - 2m_t$  [GeV]')
    a.set_ylabel(r'$A_e/\langle A\rangle$,  $A_e=\langle w\,|\,e\rangle$')
    a.set_title(r'$\langle w\,|\,e\rangle$ is not a constant: it diverges like '
                r'$1/\beta_t$ at threshold')
    a.legend(loc='upper right', fontsize=8)
    a.grid(alpha=.25, which='both')

    edges = np.concatenate([np.arange(346, 400, 2.), np.array(
        [400, 410, 425, 450, 475, 500, 550, 600, 700, 850, 1100, 1600, 4000.])])

    # the two competing pieces of A_e: the reshuffling jacobian (with the
    # feasible fraction folded in, since an infeasible set contributes 0) and
    # everything else -- the Breit-Wigner sampling jacobian and Zhat, whose
    # window the budget truncates from above exactly where <J> blows up.
    a = ax[0][1]
    Jincl = d['sum_jac'] / n
    rest = A / np.maximum(Jincl, 1e-300)
    e2 = np.arange(346, 421, 1.5)
    sel = sqrts < 420
    a.plot(sqrts[sel], (A / Abar)[sel], '.', ms=3, color=BLUE, alpha=.4)
    for arr, col, lab in (
            (A / Abar, RED, r'$A_e/\langle A\rangle$  (the product)'),
            (Jincl / Jincl.mean(), GREEN,
             r'$\langle J\rangle$ (infeasible sets counted as 0)'),
            (rest / rest.mean(), '#9467bd',
             r'$\langle {\rm jac}_{BW}\hat Z\rangle$ (truncated window)')):
        mx, mv, me, _ = profile(sqrts, arr, e2)
        a.plot(mx, mv, '-', lw=1.8, color=col, label=lab)
    a.axhline(1.0, color=GREY, lw=1, ls='--')
    a.set_xlim(345, 420)
    a.set_xlabel(r'$\sqrt{\hat s}$  [GeV]')
    a.set_ylabel('relative to its own global mean')
    a.set_title(r'Threshold region: $\langle J\rangle$ up, the truncated '
                r'Breit-Wigner window down')
    a.legend(loc='best', fontsize=8)
    a.grid(alpha=.25)

    a = ax[1][0]
    r = A / Abar
    bins = np.linspace(0.90, 1.20, 150)
    a.hist(np.clip(r, bins[0], bins[-1]), bins=bins, color=BLUE, alpha=.8,
           label=r'measured $A_e/\langle A\rangle$')
    # the same histogram if A_e really were one constant: only MC noise
    rng = np.random.default_rng(7)
    fake = 1.0 + rng.normal(size=len(A)) * Aerr / Abar
    a.hist(np.clip(fake, bins[0], bins[-1]), bins=bins, histtype='step', lw=1.6,
           color=RED, label='same statistics if $A_e$ were constant')
    a.set_yscale('log')
    a.set_xlabel(r'$A_e/\langle A\rangle$')
    a.set_ylabel('production events')
    a.set_title('Spread of the per-event mean, against its own MC error')
    a.legend(fontsize=9)
    a.grid(alpha=.25)

    a = ax[1][1]
    e3 = np.concatenate([np.array([346., 346.5, 347, 348, 349, 350, 352, 355]),
                         np.arange(358, 400, 4.),
                         np.array([400, 420, 450, 500, 600, 800, 1200, 4000.])])
    mx, mv, me, cnt = profile(sqrts, A, e3)
    frac = cnt / cnt.sum()
    a.errorbar(mx - 2 * MT, 100 * (mv / Abar - 1), yerr=100 * me / Abar,
               fmt='o-', ms=4, lw=1.3, color=RED,
               label=r'$\langle A_e\rangle_{\rm bin}/\langle A\rangle - 1$')
    a.axhline(0.0, color=GREY, lw=1, ls='--')
    a.set_xscale('log')
    a.set_xlim(0.05, 3600)
    a.set_xlabel(r'$\sqrt{\hat s} - 2m_t$  [GeV]')
    a.set_ylabel(r'missing reweighting of the bin  [%]')
    a.set_title('What redraw-until-accept normalises away, bin by bin')
    a.legend(fontsize=9)
    a.grid(alpha=.25, which='both')
    a2 = a.twinx()
    a2.step(mx - 2 * MT, 100 * frac, where='mid', color=GREY, lw=1, alpha=.6)
    a2.set_ylabel('% of the production sample in the bin', color=GREY,
                  fontsize=9)
    a2.tick_params(axis='y', labelcolor=GREY, labelsize=8)

    fig.tight_layout()
    out1 = pjoin(args.plots, 'per_event_mean_weight.png')
    fig.savefig(out1, dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- fig 2
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

    a = ax[0]
    a.plot(sqrts, maxw, '.', ms=2, color=BLUE, alpha=.4, rasterized=True,
           label=r'$\max_e w$ measured (%d free draws / event)'
                 % summary['draws_base'])
    grid = np.exp(np.linspace(math.log(346.1), math.log(4000), 400))
    jbw = float(np.median(Af / np.maximum(jbar, 1e-12)))
    a.plot(grid, [jbw * analytic_bound(s) for s in grid], '-', lw=1.8,
           color=GREEN, label=r'analytic $2\to2$: '
                              r"$\langle j\rangle\,|p'|_{m_{lo}}/|p|_{m_{pole}}$")
    if bound:
        a.axhline(bound, color=RED, lw=1.5,
                  label='global bound $C$ = %.3f' % bound)
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlim(345, 4000)
    a.set_xlabel(r'$\sqrt{\hat s}$  [GeV]')
    a.set_ylabel(r'$\max w$ for that production event')
    a.set_title('One global bound for a per-event quantity')
    a.legend(fontsize=8, loc='upper right')
    a.grid(alpha=.25)

    a = ax[1]
    eps_pe = 1.1 * maxw / Af
    a.plot(sqrts, eps_pe, '.', ms=2, color=GREEN, alpha=.4, rasterized=True,
           label=r'per-event bound $C_e = 1.1\max_e w$')
    if bound:
        a.plot(sqrts, bound / Af, '.', ms=2, color=RED, alpha=.4,
               rasterized=True, label='global bound $C$')
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlim(345, 4000)
    a.set_xlabel(r'$\sqrt{\hat s}$  [GeV]')
    a.set_ylabel(r'$\varepsilon_m$ = mass sets per accepted event')
    a.set_title('Cost of the mass stage, per production event')
    a.legend(fontsize=8)
    a.grid(alpha=.25)

    fig.tight_layout()
    out2 = pjoin(args.plots, 'per_event_bound.png')
    fig.savefig(out2, dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- tables
    report = {'global_mean_A': Abar, 'plots': [out1, out2]}
    rows = []
    for lo, hi in zip([346, 350, 355, 360, 370, 380, 400, 450, 500, 600, 800,
                       1200], [350, 355, 360, 370, 380, 400, 450, 500, 600,
                               800, 1200, 1e9]):
        sel = (sqrts >= lo) & (sqrts < hi)
        if sel.sum() < 3:
            continue
        rows.append({
            'lo': lo, 'hi': None if hi > 1e8 else hi,
            'events': int(sel.sum()),
            'fraction': float(sel.mean()),
            'A_over_Abar': float(d['sum_w'][sel].sum() / n[sel].sum() / Abar),
            'mean_J': float(jbar[sel].mean()),
            'feas': float(nfeas[sel].sum() / n[sel].sum()),
            'median_eps_per_event': float(np.median(eps_pe[sel])),
            'median_eps_global': (float(np.median(bound / Af[sel]))
                                  if bound else None),
        })
    report['by_sqrts'] = rows
    if bound:
        report['eps_global_overall'] = float(bound / Abar)
    report['eps_per_event_overall'] = float(np.mean(eps_pe))
    # the same per-event bound built from the ANALYTIC 2 -> 2 maximum instead
    # of a probe: exact, free, and necessarily more conservative (it sits at
    # the corner of the window, which the Breit-Wigner almost never reaches)
    jbw = float(np.median(Af / np.maximum(jbar, 1e-12)))
    eps_ana = np.array([1.1 * jbw * analytic_bound(s) for s in sqrts]) / Af
    report['eps_analytic_bound_overall'] = float(eps_ana.mean())
    report['eps_analytic_bound_median'] = float(np.median(eps_ana))
    for r, (lo, hi) in zip(rows, [(x['lo'], x['hi'] or 1e9) for x in rows]):
        sel = (sqrts >= lo) & (sqrts < hi)
        r['median_eps_analytic'] = float(np.median(eps_ana[sel]))
    # how far A_e strays, and how many events it moves
    rel = A / Abar
    report['A_rel_sd'] = float(rel.std())
    report['A_rel_mad'] = float(np.abs(rel - 1).mean())
    report['frac_beyond_1pc'] = float((np.abs(rel - 1) > 0.01).mean())
    report['frac_beyond_5pc'] = float((np.abs(rel - 1) > 0.05).mean())
    report['A_rel_min'] = float(rel.min())
    report['A_rel_max'] = float(rel.max())
    report['mean_stat_error_rel'] = float((Aerr / Abar).mean())
    if bound:
        report['overweight'] = summary['overweight']
    with open(pjoin(args.out, 'per_event_report.json'), 'w') as fp:
        json.dump(report, fp, indent=2, sort_keys=True)

    print('%-14s %7s %8s %8s %9s %9s %9s' % (
        'sqrt(shat)', 'frac', 'A/<A>', '<J>',
        'eps(C)', 'eps(C_e)', 'eps(ana)'))
    for r in rows:
        print('%6.0f-%-7s %6.2f%% %8.4f %8.4f %9.2f %9.2f %9.2f' % (
            r['lo'], r['hi'] if r['hi'] else 'inf', 100 * r['fraction'],
            r['A_over_Abar'], r['mean_J'],
            r['median_eps_global'] or float('nan'),
            r['median_eps_per_event'], r['median_eps_analytic']))
    print('\nA_e/<A>: sd %.4f, mean |dev| %.4f, range [%.4f, %.4f]; '
          'MC error %.5f' % (report['A_rel_sd'], report['A_rel_mad'],
                             report['A_rel_min'], report['A_rel_max'],
                             report['mean_stat_error_rel']))
    print('%.2f%% of events beyond 1%%, %.2f%% beyond 5%%'
          % (100 * report['frac_beyond_1pc'], 100 * report['frac_beyond_5pc']))
    if bound:
        ow = summary['overweight']
        print('overweight against C = %.4f: %.1f +- %.1f per 100k written '
              'events, %.2f%% of production events can produce one'
              % (ow['bound'], ow['per_100k_written'],
                 ow['per_100k_written_sd'],
                 100 * ow['fraction_of_events_that_can_overflow']))
    print('eps_m: global %.2f  per-event probe %.2f  per-event analytic %.2f'
          % (report.get('eps_global_overall', float('nan')),
             report['eps_per_event_overall'],
             report['eps_analytic_bound_overall']))
    print('\nwrote %s\n      %s' % (out1, out2))


if __name__ == '__main__':
    main()
