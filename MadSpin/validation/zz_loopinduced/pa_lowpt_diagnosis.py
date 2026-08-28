#!/usr/bin/env python3
"""Diagnosis of the ``spinmode = PA`` low-``pt`` deficit of ``pt(e+ e-)``.

Everything ``PA_LOWPT_DIAGNOSIS.md`` claims is computed here, from the
per-event unbinned columns the 200 000-event run wrote (paths are read out of
``data/meta.json``, key ``runs/<mode>/event_columns``).  Nothing here needs
MG5, MadSpin or an LHE file; only numpy, and matplotlib for the figure.

    python3 pa_lowpt_diagnosis.py                # measure + write data/ and plots/
    python3 pa_lowpt_diagnosis.py --from-cache   # redraw from data/pa_lowpt_diagnosis.npz
    python3 pa_lowpt_diagnosis.py --selftest     # assert the three load-bearing claims

The columns are ~110 MB and live outside the repository; the derived
histograms this writes into ``data/pa_lowpt_diagnosis.npz`` are 30 kB and are
enough to redraw the figure and re-derive every table.

The one piece of physics this file contains, and the reason it can answer the
"is it the reshuffling?" question without regenerating anything:

``Event.reshuffle_production`` (RAMBO, ``lhe_parser.mass_shuffle``) scales all
final-state three-momenta in the partonic CM by one common ``chi`` and leaves
their directions alone.  For a ``2 -> 2`` production that makes

    pt(after) / pt(before)  =  chi  =  lambda^(1/2)(shat, m1^2, m2^2)
                                      / lambda^(1/2)(shat, mZ^2, mZ^2)

exactly, with ``shat = m_4l`` untouched.  ``spinmode = onshell`` draws no
virtuality and never reshuffles, so its ``pt(e+ e-)`` *is* the before-reshuffle
value of the very same production event -- the samples are paired, event by
event, which ``check_pairing`` verifies rather than assumes.  ``chi`` is
therefore recoverable from ``(m_4l, m_ee, m_mumu)`` alone, and the accepted
sample of a run with ``density_keep_jacobian = False`` is the shipped ``PA``
sample reweighted by ``1/chi``.
"""

import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'data', 'pa_lowpt_diagnosis.npz')

MODES = ('PA', 'madspin', 'onshell', 'none')

# The ``pt(e+ e-)`` binning of this file.  Coarser than the study's 2 GeV grid
# on purpose: the claim is about a coherent trend over a decade of pt, and 5 GeV
# bins below 60 GeV make the per-bin errors small enough to read the trend off
# the table instead of off a fit.
PT_EDGES = np.array([0., 5., 10., 15., 20., 25., 30., 35., 40., 50., 60.,
                     80., 100., 130., 170., 220., 300., 400., 600.])


# --------------------------------------------------------------------------
def kstar(s, m1, m2):
    """``|p*|``, the two-body momentum of masses ``m1, m2`` at invariant mass ``s``."""
    return np.sqrt(np.maximum(0., (s ** 2 - (m1 + m2) ** 2)
                              * (s ** 2 - (m1 - m2) ** 2))) / (2 * s)


def load_columns(meta):
    """The five per-event column files, with ``chi`` added to the two off-shell modes."""
    mz = meta['m_Z']
    cols = {}
    for key in ('truth',) + MODES:
        path = meta['runs'][key]['event_columns']
        if not os.path.exists(path):
            raise SystemExit(
                'missing event columns for %r at %s\n'
                'They are the durable output of run_zz_loopinduced.py --stage harvest;\n'
                'use --from-cache to work from data/pa_lowpt_diagnosis.npz instead.'
                % (key, path))
        cols[key] = dict(np.load(path))
    for key in ('PA', 'madspin'):
        d = cols[key]
        d['chi'] = (kstar(d['m_4l'], d['m_ee'], d['m_mumu'])
                    / kstar(d['m_4l'], mz, mz))
    return cols


def check_pairing(cols, mz):
    """The two facts the ``chi`` reconstruction rests on, measured not assumed.

    Returns ``(max |d m_4l|, {mode: max |chi - pt_after/pt_before|})``.
    """
    dm4l = max(float(np.abs(cols[k]['m_4l'] - cols['none']['m_4l']).max())
               for k in ('PA', 'madspin', 'onshell'))
    dchi = {}
    for k in ('PA', 'madspin'):
        r = cols[k]['pt_ee'] / cols['none']['pt_ee']
        dchi[k] = float(np.abs(cols[k]['chi'] - r).max())
    return dm4l, dchi


# --------------------------------------------------------------------------
def hist(x, w, edges):
    y = np.histogram(x, bins=edges, weights=w)[0]
    e = np.sqrt(np.histogram(x, bins=edges, weights=w ** 2)[0])
    return y, e


def shape_test(y, e, yt, et):
    """``(rate, chi2, ndf, n_dropped)`` -- the study's own shape metric.

    ``rate`` is the ratio of integrals over the bins where the truth has
    support; ``chi2`` is the per-bin ratios against that flat line.
    ``n_dropped`` is the number of bins where the truth has support and the
    mode has *none* -- the bins ``plot_zz_loopinduced.py`` silently drops,
    counted here so that they can never be silent again.
    """
    ok = yt > 0
    live = ok & (y > 0)
    dropped = int(ok.sum() - live.sum())
    rate = float(y[ok].sum() / yt[ok].sum())
    r = y[live] / yt[live]
    er = r * np.sqrt((e[live] / y[live]) ** 2 + (et[live] / yt[live]) ** 2)
    ndf = int(live.sum()) - 1
    chi2 = float((((r - rate) / er) ** 2).sum())
    return rate, chi2, ndf, dropped


def measure(cols, meta):
    """Every histogram the report and the figure need, as a flat dict for npz."""
    mz = meta['m_Z']
    t = cols['truth']
    sub = t['m_4l'] < 2 * mz            # the truth region MadSpin cannot reach
    out = {'pt_edges': PT_EDGES, 'm_Z': np.array(mz)}

    y, e = hist(t['pt_ee'], t['w'], PT_EDGES)
    out['truth/all/y'], out['truth/all/e'] = y, e
    y, e = hist(t['pt_ee'][~sub], t['w'][~sub], PT_EDGES)
    out['truth/hi/y'], out['truth/hi/e'] = y, e
    y, e = hist(t['pt_ee'][sub], t['w'][sub], PT_EDGES)
    out['truth/sub/y'], out['truth/sub/e'] = y, e

    for k in MODES:
        y, e = hist(cols[k]['pt_ee'], cols[k]['w'], PT_EDGES)
        out['%s/y' % k], out['%s/e' % k] = y, e
    d = cols['PA']
    y, e = hist(d['pt_ee'], d['w'] / d['chi'], PT_EDGES)
    out['PA_nojac/y'], out['PA_nojac/e'] = y, e

    out['sigma'] = np.array([t['w'].sum()] + [cols[k]['w'].sum() for k in MODES])
    out['sigma_labels'] = np.array(['truth'] + list(MODES))
    out['sigma_truth_hi'] = np.array(t['w'][~sub].sum())
    out['sigma_truth_sub'] = np.array(t['w'][sub].sum())

    # m_4l of the truth, finely binned across the 2 m_Z edge: the shape of the
    # region that is missing, and the reason the study's 10 GeV grid hid it.
    m4l_edges = np.concatenate([np.arange(110., 180., 5.),
                                [180., 181., 182., 2 * mz, 183., 185.],
                                np.arange(190., 320., 10.)])
    out['m4l_edges'] = m4l_edges
    y, e = hist(t['m_4l'], t['w'], m4l_edges)
    out['truth/m4l/y'], out['truth/m4l/e'] = y, e
    y, e = hist(cols['PA']['m_4l'], cols['PA']['w'], m4l_edges)
    out['PA/m4l/y'], out['PA/m4l/e'] = y, e

    # chi, the per-event reshuffle ratio
    for k in ('PA', 'madspin'):
        c = cols[k]['chi']
        out['%s/chi_stats' % k] = np.array(
            [c.min(), np.percentile(c, 5), np.median(c), c.mean(),
             np.percentile(c, 95), c.max(), (c > 1).mean()])

    # leakage below the truth's own pt > 1 GeV cut
    out['pt_below_cut'] = np.array(
        [float(cols[k]['w'][cols[k]['pt_ee'] < 1.0].sum() / cols[k]['w'].sum())
         for k in MODES])

    # per-observable shape table, against both supports
    rows = []
    for obs in meta['observables']:
        edges = np.array(meta['bins'][obs])
        for label, x, w in ([(k, cols[k][obs], cols[k]['w']) for k in MODES]
                            + [('PA_nojac', cols['PA'][obs],
                                cols['PA']['w'] / cols['PA']['chi'])]):
            y, e = hist(x, w, edges)
            for tag, m in (('all', slice(None)), ('hi', ~sub)):
                yt, et = hist(t[obs][m], t['w'][m], edges)
                rate, chi2, ndf, drop = shape_test(y, e, yt, et)
                rows.append((obs, label, tag, rate, chi2, ndf, drop))
    out['shape_rows'] = np.array(
        [(o, l, tg, '%.6f' % r, '%.4f' % c, str(n), str(dp))
         for o, l, tg, r, c, n, dp in rows], dtype=object)

    # the low-mass tail of m(e+ e-), the claim RESULTS.md section 5 makes
    out['frac_mee_below80'] = np.array([
        float(t['w'][t['m_ee'] < 80].sum() / t['w'].sum()),
        float(t['w'][(~sub) & (t['m_ee'] < 80)].sum() / t['w'][~sub].sum()),
        float(cols['madspin']['w'][cols['madspin']['m_ee'] < 80].sum()
              / cols['madspin']['w'].sum()),
        float(cols['PA']['w'][cols['PA']['m_ee'] < 80].sum()
              / cols['PA']['w'].sum())])

    dm4l, dchi = check_pairing(cols, mz)
    out['pairing'] = np.array([dm4l, dchi['PA'], dchi['madspin']])
    return out


# --------------------------------------------------------------------------
def report(D):
    A, out = (lambda s='': out.append(s)), []
    mz = float(D['m_Z'])
    edges = D['pt_edges']

    A('PA low-pt deficit: the measurement')
    A('=' * 72)
    A('')
    A('--- the samples are paired, event by event ---')
    p = D['pairing']
    A('  max |m_4l(mode) - m_4l(none)|          %.2e GeV   (LHE write precision)'
      % p[0])
    A('  max |chi - pt_after/pt_before|  PA     %.2e' % p[1])
    A('                                  madspin %.2e' % p[2])
    A('  so pt(onshell) is the BEFORE-reshuffle pt of the same production event,')
    A('  and chi = lambda^(1/2)(m_4l, m1, m2) / lambda^(1/2)(m_4l, mZ, mZ) is the')
    A('  whole of what the reshuffle does to pt.')
    A('')

    A('--- cross sections [fb of the 200 000-event files] ---')
    st = float(D['sigma'][0])
    for lab, s in zip(D['sigma_labels'], D['sigma']):
        A('  %-9s %10.4f   ratio to truth %.5f' % (lab, s, s / st))
    hi, sb = float(D['sigma_truth_hi']), float(D['sigma_truth_sub'])
    A('  truth m_4l >= 2 mZ  %10.4f   (%.4f %% of the truth is BELOW 2 mZ = %.3f GeV)'
      % (hi, 100 * sb / st, 2 * mz))
    A('  every MadSpin mode has EXACTLY zero support there: the production sample')
    A('  is g g > z z with both z on shell, so m_4l = sqrt(shat) >= 2 mZ, and the')
    A('  RAMBO reshuffle holds sqrt(shat) fixed.')
    A('')

    A('--- pt(e+ e-): the deficit, and what is left of it on the reachable support ---')
    A('  %-13s %9s %8s | %-24s | %-24s'
      % ('pt [GeV]', 'truth', 'f_sub', 'PA / truth (all)', 'PA / truth (m_4l >= 2 mZ)'))
    yt, et = D['truth/all/y'], D['truth/all/e']
    yh, eh = D['truth/hi/y'], D['truth/hi/e']
    ys = D['truth/sub/y']
    y, e = D['PA/y'], D['PA/e']
    ra = float(y.sum() / yt.sum())
    rh = float(y.sum() / yh.sum())
    for i in range(len(edges) - 1):
        r1 = y[i] / yt[i]
        s1 = r1 * math.sqrt((e[i] / y[i]) ** 2 + (et[i] / yt[i]) ** 2)
        r2 = y[i] / yh[i]
        s2 = r2 * math.sqrt((e[i] / y[i]) ** 2 + (eh[i] / yh[i]) ** 2)
        A('  %5.0f - %5.0f %9.3f %7.2f%% | %7.4f +- %6.4f  %+6.2f | %7.4f +- %6.4f  %+6.2f'
          % (edges[i], edges[i + 1], yt[i], 100 * ys[i] / yt[i],
             r1, s1, (r1 / ra - 1) / (s1 / ra),
             r2, s2, (r2 / rh - 1) / (s2 / rh)))
    A('  ("f_sub" is the share of that truth bin that lies below 2 mZ; the two')
    A('   pull columns are against each comparison\'s own flat line, i.e. they')
    A('   are shape pulls with the normalisation divided out.)')
    A('')

    A('--- chi, the reshuffle ratio pt_after/pt_before ---')
    A('  %-9s %8s %8s %8s %8s %8s %8s %8s'
      % ('mode', 'min', 'p5', 'median', 'mean', 'p95', 'max', 'frac>1'))
    for k in ('PA', 'madspin'):
        s = D['%s/chi_stats' % k]
        A('  %-9s %8.4f %8.4f %8.4f %8.4f %8.4f %8.2f %8.4f' % ((k,) + tuple(s)))
    A('')

    A('--- pt below the truth\'s own pt(ll) > 1 GeV cut (an acceptance mismatch')
    A('    in the OTHER direction: sample A applies it to the on-shell z, before')
    A('    the reshuffle, the truth to the reconstructed pair) ---')
    for k, f in zip(MODES, D['pt_below_cut']):
        A('  %-9s %.5f %% of sigma' % (k, 100 * f))
    A('')

    A('--- per-observable shape test, against both supports ---')
    A('  %-12s %-9s | %-26s | %-26s'
      % ('observable', 'mode', 'full truth support', 'truth m_4l >= 2 mZ'))
    A('  %-12s %-9s | %8s %9s %5s %5s | %8s %9s %5s %5s'
      % ('', '', 'rate', 'chi2/ndf', 'ndf', 'drop', 'rate', 'chi2/ndf', 'ndf', 'drop'))
    rows = {}
    for o, l, tg, r, c, n, dp in D['shape_rows']:
        rows.setdefault((o, l), {})[tg] = (float(r), float(c), int(n), int(dp))
    seen = []
    for o, l in rows:
        if (o, l) in seen:
            continue
        seen.append((o, l))
    last = None
    for o, l in seen:
        a, h = rows[(o, l)]['all'], rows[(o, l)]['hi']
        def fmt(v):
            return ('%8.4f %9s %5d %5d'
                    % (v[0], '--' if v[2] <= 0 else '%.2f' % (v[1] / v[2]),
                       v[2], v[3]))
        A('  %-12s %-9s | %s | %s'
          % (o if o != last else '', l, fmt(a), fmt(h)))
        last = o
    A('')
    A('  "drop" is the number of bins where the truth has support and the mode')
    A('  has none.  plot_zz_loopinduced.py drops those bins from its chi2 without')
    A('  saying so; the m_4l row is where that mattered.')
    A('')

    A('--- fraction of sigma with m(e+ e-) < 80 GeV ---')
    f = D['frac_mee_below80']
    for lab, v in zip(['truth (all)', 'truth (m_4l >= 2 mZ)', 'madspin', 'PA'], f):
        A('  %-22s %.4f %%' % (lab, 100 * v))
    A('  madspin / truth(all) = %.4f   but madspin / truth(reachable) = %.4f'
      % (f[2] / f[0], f[2] / f[1]))
    A('  PA      / truth(all) = %.4f              / truth(reachable) = %.4f'
      % (f[3] / f[0], f[3] / f[1]))
    return '\n'.join(out)


# --------------------------------------------------------------------------
def figure(D, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    edges = D['pt_edges']
    ctr = 0.5 * (edges[1:] + edges[:-1])
    wid = np.diff(edges)
    yt, et = D['truth/all/y'], D['truth/all/e']
    yh, eh = D['truth/hi/y'], D['truth/hi/e']
    ys = D['truth/sub/y']

    fig, ax = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1.15],
                                        'hspace': 0.07})
    styles = {'PA': ('C3', 'o'), 'madspin': ('C0', 's'), 'onshell': ('C2', '^')}
    for k, (c, mk) in styles.items():
        y, e = D['%s/y' % k], D['%s/e' % k]
        r = y / yt
        er = r * np.sqrt((e / y) ** 2 + (et / yt) ** 2)
        ax[0].errorbar(ctr, r, yerr=er, xerr=wid / 2, fmt=mk, ms=3.5, lw=1.1,
                       color=c, label=k)
        r2 = y / yh
        er2 = r2 * np.sqrt((e / y) ** 2 + (eh / yh) ** 2)
        ax[1].errorbar(ctr, r2, yerr=er2, xerr=wid / 2, fmt=mk, ms=3.5, lw=1.1,
                       color=c, label=k)
    for a in ax:
        a.axhline(1.0, color='k', lw=0.8, ls='-')
        a.set_xscale('log')
        a.set_xlim(1.3, 620)
        a.grid(alpha=0.25, lw=0.5)
    twin = ax[0].twinx()
    twin.step(np.concatenate([[edges[0]], edges[1:]]),
              np.concatenate([[100 * ys[0] / yt[0]], 100 * ys / yt]),
              where='pre', color='0.6', lw=1.2, ls='--', zorder=0)
    twin.set_ylabel('truth below $2m_Z$  [% of the bin]', color='0.4')
    twin.tick_params(axis='y', colors='0.4')
    twin.set_ylim(0, 30)
    ax[0].set_ylabel('mode / truth, full truth support')
    ax[1].set_ylabel(r'mode / truth, $m_{4\ell} \geq 2m_Z$')
    ax[1].set_xlabel(r'$p_T(e^+e^-)$  [GeV]')
    ax[0].set_ylim(0.85, 1.13)
    ax[1].set_ylim(0.93, 1.21)
    ax[0].legend(loc='lower right', ncol=3, fontsize=9, frameon=True,
                 framealpha=0.95, edgecolor='0.8')
    twin.plot([], [], color='0.6', lw=1.2, ls='--',
              label='truth below $2m_Z$')
    twin.legend(loc='center left', fontsize=8.5, frameon=False,
                labelcolor='0.4')
    ax[0].set_title('$gg\\to ZZ$ (loop induced) + MadSpin vs the off-shell truth\n'
                    'top: as published.  bottom: truth cut to the support MadSpin '
                    'can reach', fontsize=10)
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------
def selftest(D):
    """The three claims PA_LOWPT_DIAGNOSIS.md rests on, as assertions."""
    edges, yt, yh = D['pt_edges'], D['truth/all/y'], D['truth/hi/y']
    y = D['PA/y']
    low = edges[1:] <= 20.0

    # 1. against the full truth, PA is coherently low below 20 GeV
    ra = y.sum() / yt.sum()
    shape_all = (y / yt) / ra
    assert shape_all[low].max() < 0.98, shape_all[low]

    # 2. against the reachable truth it is not
    rh = y.sum() / yh.sum()
    shape_hi = (y / yh) / rh
    assert shape_hi[low].min() > 0.96 and shape_hi[low].max() < 1.02, shape_hi[low]

    # 3. removing the reshuffling jacobian does NOT remove the deficit
    yn = D['PA_nojac/y']
    sn = (yn / yt) / (yn.sum() / yt.sum())
    assert sn[low].max() < 0.99, sn[low]

    # 4. the pairing that makes claim 3 meaningful
    assert D['pairing'].max() < 1e-4, D['pairing']
    print('selftest: 4/4 claims hold')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--from-cache', action='store_true',
                    help='read data/pa_lowpt_diagnosis.npz instead of the columns')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--no-figure', action='store_true')
    args = ap.parse_args()

    if args.from_cache:
        D = dict(np.load(CACHE, allow_pickle=True))
    else:
        meta = json.load(open(os.path.join(HERE, 'data', 'meta.json')))
        D = measure(load_columns(meta), meta)
        np.savez_compressed(CACHE, **D)

    txt = report(D)
    print(txt)
    open(os.path.join(HERE, 'plots', 'pa_lowpt_diagnosis.txt'), 'w').write(txt + '\n')
    if not args.no_figure:
        figure(D, os.path.join(HERE, 'plots', 'pa_lowpt_diagnosis.pdf'))
    if args.selftest:
        selftest(D)


if __name__ == '__main__':
    main()
