#!/usr/bin/env python3
"""``C_kk`` and ``f_0`` as functions of the four-lepton mass, for the ``g g``
quark box and the ``q q~`` continuum, in the MG7 paper's plotting style.

    plot_ckk_mass.py [--data DIR] [--out DIR] [--nbins N] [--check-minus]

with ``<data>`` holding ``events.npz`` as written by ``run_ckk_mass.py``.
Nothing here needs MadGraph or MadSpin.

What is plotted
---------------
Upper pane
    ``C_kk(m_4l) = (4/eta_l^2) <cos th1 cos th2>`` in bins of the four-lepton
    invariant mass, for the two production mechanisms.  ``C_kk`` is the
    helicity-sign correlation ``P(++) + P(--) - P(+-) - P(-+)`` between the two
    ``Z`` on their own helicity axes; ``SPIN_COEFFICIENTS.md`` sections 1 and 4
    of ``../zz_loopinduced`` derive it and fix the calibration.  A zero line is
    drawn because the two mechanisms sit on opposite sides of it, and whether
    that survives differentially is the question the figure exists to answer.

Lower pane
    ``f_0(m_4l) = 2 - 5 <cos^2 theta>``, the longitudinal fraction of one ``Z``,
    averaged over the two.  Free: it is the same per-event columns, and it is
    the rank-2 companion that carries no ``eta_l`` dilution at all.

Neither pane carries a MadSpin spinmode.  Both curves are the fully off-shell
four-lepton matrix element -- ``g g > e+ e- mu+ mu- / a [noborn=QCD]`` and
``p p > e+ e- mu+ mu- / a [QCD]``.  This is a physics figure; showing that
``spinmode = madspin`` reproduces the differential coefficient would be a
separate figure and is not attempted here.

Why the calibration makes this hard, and what the bin edges are for
------------------------------------------------------------------
``<cos th1 cos th2> = (eta_l^2/4) C_kk`` with ``eta_l = 0.2193``, so the
measured moment is ``C_kk/83.15``.  That factor multiplies the ERROR too: the
per-event spread of ``cos th1 cos th2`` is about ``0.38`` whatever the
correlation is, so

    sigma(C_kk) = 83.15 * 0.38 / sqrt(N) ~ 31.6 / sqrt(N)

and a ``+-0.1`` bin costs 100 000 events.  ``--stats`` prints this from the
sample's own spread rather than from the estimate.

Bin edges are therefore chosen for **equal statistical power, not equal width**:
they are the quantiles of the ``gg`` sample's own ``m_4l`` distribution, so every
``gg`` bin holds the same number of events and every ``gg`` error bar is the same
size.  The ``gg`` side sets them because it is the one that is expensive to
generate; the ``qq`` spectrum is harder, so its bins are not exactly equal-N and
its bars vary a little.  The top bin is open-ended and wide -- the spectrum falls
steeply -- which is stated on the axis rather than hidden by re-binning.
"""

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator

_HERE = os.path.dirname(os.path.abspath(__file__))
_VAL = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_VAL, 'zz_loopinduced'))
import observables as OBS                                        # noqa: E402
from plot_zz_loopinduced import (_fix_type1_subset_minus, check_minus,  # noqa
                                 _have_latex)

MINUS_FIX = _fix_type1_subset_minus()
USETEX = _have_latex()
mpl.rcParams.update({"text.usetex": USETEX, "font.family": "serif",
                     "font.size": 14, 'lines.markersize': 8})
if not USETEX:
    mpl.rcParams.update({'mathtext.fontset': 'cm'})

LW = 1.2
allcolors = list(mcolors.TABLEAU_COLORS.values())
allcolors[1] = 'black'
allcolors[0] = 'blue'
allcolors[3] = 'red'

CAL = 4.0 / OBS.ETA_L ** 2                      # 83.154

TAGS = ['gg', 'qq']
COLOR = {'gg': 'blue', 'qq': 'red'}
MARK = {'gg': 'o', 'qq': 's'}
LABEL_TEX = {
    'gg': r'$gg \to e^{+}e^{-}\mu^{+}\mu^{-}$ (quark box, loop induced)',
    'qq': r'$pp \to e^{+}e^{-}\mu^{+}\mu^{-}$ (continuum, NLO QCD)',
}
LABEL_PLAIN = {
    'gg': 'gg -> e+ e- mu+ mu- (quark box, loop induced)',
    'qq': 'pp -> e+ e- mu+ mu- (continuum, NLO QCD)',
}


# --------------------------------------------------------------------------
# weighted moments
# --------------------------------------------------------------------------
def wmean(x, w):
    """Weighted mean and its error.

    ``sum w^2 (x - <x>)^2 / (sum w)^2`` rather than ``var/N``: the ``qq`` sample
    is MC@NLO and carries negative weights, and an unweighted error on it would
    be wrong in the direction that flatters the figure.  On a unit-weight sample
    this reduces to the ordinary error of the mean exactly.
    """
    sw = w.sum()
    m = (w * x).sum() / sw
    var = (w ** 2 * (x - m) ** 2).sum() / sw ** 2
    return m, np.sqrt(var)


def neff(w):
    return w.sum() ** 2 / (w ** 2).sum()


def coefficients(x, w):
    """``(C_kk, err)`` from the per-event product ``cos th1 cos th2``."""
    m, e = wmean(x, w)
    return CAL * m, CAL * e


# --------------------------------------------------------------------------
def load(path):
    d = np.load(path)
    out = {}
    for tag in TAGS:
        if '%s/w' % tag not in d:
            continue
        out[tag] = {k.split('/', 1)[1]: d[k] for k in d.files
                    if k.startswith(tag + '/')}
    return out


def equal_stat_edges(m, w, nbins):
    """Quantile edges of ``m``, so that every bin carries the same weight.

    Weighted quantiles, because the sample that sets the edges may not be
    unit-weight.  The outer edges are the sample's own extremes, and the top bin
    is therefore open-ended in practice.
    """
    o = np.argsort(m)
    ms, ws = m[o], w[o]
    c = np.cumsum(ws) / ws.sum()
    q = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    inner = np.interp(q, c, ms)
    return np.concatenate([[ms[0]], inner, [ms[-1]]])


def bin_median(data, edges):
    """The weighted median of ``m_4l`` inside each bin.

    The marker is drawn there and not at the bin centre.  The top bin is open
    ended and runs to several TeV while holding a quarter of the rate within a
    few tens of GeV of its lower edge, so a centre-drawn marker would sit where
    no event is.  The x error bar still spans the whole bin, so nothing about
    the binning is hidden by this.
    """
    m, w = data['m_4l'], data['w']
    idx = np.clip(np.digitize(m, edges) - 1, 0, len(edges) - 2)
    out = []
    for i in range(len(edges) - 1):
        s = idx == i
        if s.sum() == 0:
            out.append(0.5 * (edges[i] + edges[i + 1]))
            continue
        o = np.argsort(m[s])
        c = np.cumsum(w[s][o])
        out.append(float(m[s][o][np.searchsorted(c, 0.5 * c[-1])]))
    return np.array(out)


def binned(data, edges, key):
    """``(value, error, N, N_eff)`` per bin for a per-event estimator."""
    m, w, x = data['m_4l'], data['w'], data[key]
    idx = np.digitize(m, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    val, err, n, ne = [], [], [], []
    for i in range(len(edges) - 1):
        s = idx == i
        if s.sum() < 2:
            val.append(np.nan); err.append(np.nan)
            n.append(int(s.sum())); ne.append(0.0)
            continue
        a, b = wmean(x[s], w[s])
        val.append(a); err.append(b)
        n.append(int(s.sum())); ne.append(neff(w[s]))
    return (np.array(val), np.array(err), np.array(n), np.array(ne))


# --------------------------------------------------------------------------
def stats_report(data, nbins, targets=(0.30, 0.20, 0.15, 0.10)):
    """The feasibility arithmetic, from the sample's own per-event spread."""
    lines = []
    lines.append('Per-event spread and what it costs')
    lines.append('=' * 74)
    lines.append('C_kk = %.3f * <cos th1 cos th2>   (eta_l = %.5f, on shell)'
                 % (CAL, OBS.ETA_L))
    lines.append('')
    lines.append('%-6s %9s %9s %11s %11s %10s'
                 % ('sample', 'N', 'N_eff', 'sd(c1c2)', 'k = cal*sd', 'C_kk'))
    ks = {}
    for tag, d in data.items():
        w, x = d['w'], d['cos1cos2']
        mu, err = wmean(x, w)
        sd = np.sqrt((w * (x - mu) ** 2).sum() / w.sum())
        k = CAL * sd
        ks[tag] = k
        lines.append('%-6s %9d %9.0f %11.4f %11.2f %10s'
                     % (tag, len(w), neff(w), sd, k,
                        '%+.3f +- %.3f' % (CAL * mu, CAL * err)))
    lines.append('')
    lines.append('sigma(C_kk) = k / sqrt(N_eff).  Events needed PER BIN:')
    lines.append('%-10s %s' % ('target', '  '.join('%8s' % t for t in TAGS)))
    for t in targets:
        lines.append('%-10s %s'
                     % ('+-%.2f' % t,
                        '  '.join('%8.0f' % (ks[tag] / t) ** 2 for tag in TAGS
                                  if tag in ks)))
    lines.append('')
    for tag in TAGS:
        if tag not in ks:
            continue
        ne = neff(data[tag]['w'])
        lines.append('%-6s at %d bins: N_eff/bin = %.0f -> sigma(C_kk) = %.3f'
                     % (tag, nbins, ne / nbins, ks[tag] / np.sqrt(ne / nbins)))
    return '\n'.join(lines)


def numbers(data, edges, nbins):
    lines = []
    lo, hi = edges[:-1], edges[1:]
    for tag in TAGS:
        if tag not in data:
            continue
        ck, cke, n, ne = binned(data[tag], edges, 'cos1cos2')
        ck, cke = CAL * ck, CAL * cke
        f0a, f0ae, _, _ = binned(data[tag], edges, 'pol0_1')
        f0b, f0be, _, _ = binned(data[tag], edges, 'pol0_2')
        lines.append('')
        lines.append('%s  --  %s' % (tag, LABEL_PLAIN[tag]))
        lines.append('%-18s %9s %9s %20s %20s'
                     % ('m_4l bin [GeV]', 'N', 'N_eff', 'C_kk', 'f_0 (mean of 2)'))
        for i in range(nbins):
            f0 = 0.5 * (f0a[i] + f0b[i])
            f0e = 0.5 * np.hypot(f0ae[i], f0be[i])
            lines.append('%7.1f - %7.1f %9d %9.0f  %+9.3f +- %-7.3f %+9.4f +- %-7.4f'
                         % (lo[i], hi[i], n[i], ne[i], ck[i], cke[i], f0, f0e))
        a, b = wmean(data[tag]['cos1cos2'], data[tag]['w'])
        lines.append('%-18s %9d %9.0f  %+9.3f +- %-7.3f'
                     % ('inclusive', len(data[tag]['w']),
                        neff(data[tag]['w']), CAL * a, CAL * b))
    return '\n'.join(lines)


# --------------------------------------------------------------------------
def figure(data, edges, nbins, out, meta):
    lo, hi = edges[:-1], edges[1:]

    fig, ax = plt.subplots(2, 1, figsize=(7.2, 7.8), sharex=True,
                           gridspec_kw={'height_ratios': [1.45, 1.0],
                                        'hspace': 0.07})
    a0, a1 = ax

    # |C_kk| <= 1 is a bound on a spin-1 correlation, not a fit range.  Drawn so
    # that a bin whose central value runs past it is visibly a fluctuation and
    # not a measurement of something impossible.
    for y in (-1.0, 1.0):
        a0.axhline(y, color='0.75', lw=0.9, zorder=1)
    a0.axhline(0.0, color='0.35', lw=1.0, ls=(0, (4, 3)), zorder=2)
    # 1/3 is the isotropic longitudinal fraction.
    a1.axhline(1.0 / 3.0, color='0.35', lw=1.0, ls=(0, (4, 3)), zorder=2)

    for tag in TAGS:
        if tag not in data:
            continue
        ck, cke, n, ne = binned(data[tag], edges, 'cos1cos2')
        ck, cke = CAL * ck, CAL * cke
        f0a, f0ae, _, _ = binned(data[tag], edges, 'pol0_1')
        f0b, f0be, _, _ = binned(data[tag], edges, 'pol0_2')
        f0 = 0.5 * (f0a + f0b)
        f0e = 0.5 * np.hypot(f0ae, f0be)
        lab = LABEL_TEX[tag] if USETEX else LABEL_PLAIN[tag]
        x = bin_median(data[tag], edges)
        xerr = np.vstack([x - lo, hi - x])
        a0.errorbar(x, ck, yerr=cke, xerr=xerr, fmt=MARK[tag],
                    color=COLOR[tag], lw=LW, capsize=0, label=lab, zorder=4)
        a1.errorbar(x, f0, yerr=f0e, xerr=xerr, fmt=MARK[tag],
                    color=COLOR[tag], lw=LW, capsize=0, zorder=4)

    a0.set_ylabel(r'$C_{kk}$' if USETEX else 'C_kk')
    a1.set_ylabel(r'$f_{0}$' if USETEX else 'f_0')
    xl = (r'$m_{4\ell}\ $ [GeV]' if USETEX else 'm_4l [GeV]')
    a1.set_xlabel(xl)
    a0.legend(frameon=False, fontsize=10.5, loc='upper center',
              handletextpad=0.5, borderaxespad=0.9)
    # LOG x, and stated.  The bins hold equal numbers of events and the spectrum
    # falls steeply, so the top bin is several times wider than the whole rest of
    # the axis; on a linear scale it would squash the four bins into the first
    # tenth of the pane.  Nothing is re-binned to make the picture nicer -- the
    # scale is changed instead, and the marker sits at the bin's own median.
    for a in ax:
        a.set_xscale('log')
        a.yaxis.set_minor_locator(AutoMinorLocator())
        a.tick_params(which='both', direction='in', top=True, right=True)
    a1.set_xticks([200, 300, 500, 1000, 2000, 3000])
    a1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # matplotlib labels log MINOR ticks too when a decade is not fully spanned,
    # and those labels land on top of the ones just set.
    a1.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    a0.set_xlim(edges[0] * 0.97, edges[-1] * 1.03)

    note = (r'%d equal-statistics bins, from the quantiles of the $%s$ spectrum;'
            '\n' r'marker at the bin median, bar spans the bin'
            % (nbins, meta['edge_sample'])) if USETEX else \
           ('%d equal-statistics bins; marker at bin median' % nbins)
    a1.text(0.5, -0.34, note, transform=a1.transAxes, ha='center',
            fontsize=9)

    ttl = (r'$ZZ$ spin correlation and longitudinal fraction vs.\ '
           r'four-lepton mass') if USETEX else \
          'ZZ spin correlation and longitudinal fraction vs four-lepton mass'
    a0.set_title(ttl, fontsize=12)
    a0.set_ylim(-1.35, 1.35)

    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--nbins', type=int, default=4)
    ap.add_argument('--edge-sample', default='gg')
    ap.add_argument('--check-minus', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = load(os.path.join(args.data, 'events.npz'))
    if not data:
        raise SystemExit('no samples in %s/events.npz' % args.data)

    src = args.edge_sample if args.edge_sample in data else sorted(data)[0]
    edges = equal_stat_edges(data[src]['m_4l'], data[src]['w'], args.nbins)

    txt = stats_report(data, args.nbins) + '\n\n' \
        + 'Bin edges: quantiles of the %s sample (equal statistical power)\n' % src \
        + '  ' + '  '.join('%.1f' % e for e in edges) + '\n' \
        + numbers(data, edges, args.nbins) + '\n'
    print(txt)
    open(os.path.join(args.data, 'numbers.txt'), 'w').write(txt)

    np.savez_compressed(
        os.path.join(args.data, 'ckk_mass.npz'),
        edges=edges,
        **{'%s/%s' % (t, k): v
           for t in data if t in TAGS
           for k, v in zip(('C_kk', 'C_kk_err', 'N', 'N_eff'),
                           binned(data[t], edges, 'cos1cos2'))},
        **{'%s/f0_%d' % (t, j): binned(data[t], edges, 'pol0_%d' % j)[0]
           for t in data if t in TAGS for j in (1, 2)},
        **{'%s/f0_%d_err' % (t, j): binned(data[t], edges, 'pol0_%d' % j)[1]
           for t in data if t in TAGS for j in (1, 2)})

    meta = {'calibration': CAL, 'eta_l': OBS.ETA_L, 'nbins': args.nbins,
            'edge_sample': src, 'edges': list(map(float, edges)),
            'samples': {t: {'N': int(len(data[t]['w'])),
                            'N_eff': float(neff(data[t]['w']))}
                        for t in data}}
    json.dump(meta, open(os.path.join(args.data, 'plot_meta.json'), 'w'),
              indent=1)

    base = os.path.join(args.out, 'ckk_mass')
    figure(data, edges, args.nbins, base, meta)
    print('wrote %s.pdf / .png  (usetex=%s, minus fix applied=%s)'
          % (base, USETEX, MINUS_FIX))
    if args.check_minus:
        ok, detail = check_minus(base + '.pdf')
        print('check-minus: %s' % detail)
        if not ok:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
