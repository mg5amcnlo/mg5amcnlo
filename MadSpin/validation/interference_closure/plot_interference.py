#!/usr/bin/env python3
"""Plots and numbers of the MadSpin interference-closure test.

Runs entirely off the committed raw histograms, so nothing here needs MadSpin:

    plot_interference.py <data_dir> <out_dir>

with ``<data_dir>`` holding ``histograms.npz`` and ``meta.json`` as written by
``analyse_interference.py``.

Style follows the MG7 paper's ``plotexample/dummyplot.py``: LaTeX text, serif,
base font size 14, step histograms of line width 1.2, the paper's fixed figure
width (7*0.75 in -- "do not change the horizontal size"), tableau colours with
black/blue/red promoted, frameless legends, minor tick locators.
"""

import json
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator

# --- MG7 paper style ------------------------------------------------------
def _have_latex():
    import shutil
    if os.environ.get('NO_USETEX'):
        return False
    for extra in ('/Library/TeX/texbin', '/usr/local/texlive/bin',
                  '/usr/bin', '/usr/local/bin'):
        if os.path.isdir(extra) and extra not in os.environ.get('PATH', ''):
            os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + extra
    return bool(shutil.which('latex')) and bool(shutil.which('dvipng'))


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

C_UNPOL = 'black'
C_SUM4 = allcolors[0]           # blue   -- the previous (diagonal-only) result
C_SUM9 = allcolors[3]           # red    -- diagonal + interference
C_DIAG = allcolors[7]           # grey
C_INT = allcolors[4]            # purple


# --------------------------------------------------------------------------
OBS = [
    ('cos_k_p',  r'$\cos\theta^{k}_{\ell^+}$', 'diagonal'),
    ('cos_k_m',  r'$\cos\theta^{k}_{\ell^-}$', 'diagonal'),
    ('ckk',      r'$\cos\theta^{k}_{\ell^+}\cos\theta^{k}_{\ell^-}$   ($C_{kk}$)',
     'diagonal'),
    ('cnn',      r'$\cos\theta^{n}_{\ell^+}\cos\theta^{n}_{\ell^-}$   ($C_{nn}$)',
     'off-diagonal'),
    ('crr',      r'$\cos\theta^{r}_{\ell^+}\cos\theta^{r}_{\ell^-}$   ($C_{rr}$)',
     'off-diagonal'),
    ('cos_phi',  r'$\cos\varphi_{\ell\ell}$', 'off-diagonal'),
    ('cos_n_p',  r'$\cos\theta^{n}_{\ell^+}$', 'off-diagonal'),
    ('dphi_lab', r'$\Delta\phi(\ell^+,\ell^-)$ (lab)', 'off-diagonal'),
    ('pt_t',     r'$p_T(t)$ [GeV]', 'control'),
    ('m_tt',     r'$m(t\bar t)$ [GeV]', 'control'),
]

DIAG = ['pp', 'pm', 'mp', 'mm']
# the nine-term total, route 1:  the four diagonal blocks, the two blocks with
# an interference on the antitop side, and x_t = (I_t x 1_tbar) which is
# (I,D+) + (I,D-) + (I,I) in one sample.
ROUTE1 = DIAG + ['x_t', 'i_tp', 'i_tm']
# route 2 swaps the roles of the two legs -- an independent determination
ROUTE2 = DIAG + ['x_tb', 'i_tbp', 'i_tbm']


class Data(object):
    def __init__(self, ddir):
        self.z = np.load(os.path.join(ddir, 'histograms.npz'))
        self.meta = json.load(open(os.path.join(ddir, 'meta.json')))

    def bins(self, key):
        return self.z['bins__%s' % key]

    def h(self, tags, key):
        """(sum of weights, error) of the sum of the samples in ``tags``."""
        s = sum(self.z['sumw__%s__%s' % (t, key)] for t in tags)
        e2 = sum(self.z['sumw2__%s__%s' % (t, key)] for t in tags)
        return s, np.sqrt(e2)

    def hdiff(self, plus, minus, key):
        s = (sum(self.z['sumw__%s__%s' % (t, key)] for t in plus)
             - sum(self.z['sumw__%s__%s' % (t, key)] for t in minus))
        e2 = sum(self.z['sumw2__%s__%s' % (t, key)] for t in list(plus) + list(minus))
        return s, np.sqrt(e2)

    def mom(self, tags, key):
        return sum(self.z['mom__%s__%s' % (t, key)] for t in tags)

    def mean(self, tags, key):
        """<O> and its MC error for the (signed) combination of ``tags``."""
        sw, swo, sw2, sw2o, sw2o2, _n = self.mom(tags, key)
        m = swo / sw
        var = (sw2o2 - 2 * m * sw2o + m * m * sw2) / (sw * sw)
        return m, math.sqrt(max(var, 0.0))


def ratio(tot, tote, ref, refe):
    r = tot / np.where(ref == 0, np.nan, ref)
    re = np.abs(r) * np.sqrt((tote / np.where(tot == 0, np.nan, tot)) ** 2
                             + (refe / np.where(ref == 0, np.nan, ref)) ** 2)
    # a bin whose content is compatible with zero must not get a zero error
    re = np.where(np.isfinite(re), re,
                  tote / np.where(ref == 0, np.nan, ref))
    return r, re


def implied_scale(s4, s4e, inter, intere, u, ue):
    """Best-fit k in  (4 diagonal blocks) + k * (interference) = unpolarised.

    The predicted value is k = 1: the interference samples are normalised by
    ``max_weight_i / c``, a number taken from the run, never fitted.  This is
    what says how well that prediction holds, and is quoted alongside the
    chi2 rather than used anywhere.
    """
    d = u - s4
    s2 = ue ** 2 + s4e ** 2 + intere ** 2
    good = (s2 > 0) & np.isfinite(d) & (np.abs(inter) > 0)
    num = (inter[good] * d[good] / s2[good]).sum()
    den = (inter[good] ** 2 / s2[good]).sum()
    if den <= 0:
        return float('nan'), float('nan')
    return num / den, 1.0 / math.sqrt(den)


def chi2(tot, tote, ref, refe):
    d = tot - ref
    s2 = tote ** 2 + refe ** 2
    good = (s2 > 0) & np.isfinite(d)
    return float((d[good] ** 2 / s2[good]).sum()), int(good.sum())


def update_legend(ax, ncol=1, loc='best', size=9):
    handles, labels = ax.get_legend_handles_labels()
    new = []
    for h in handles:
        if 'fill' in dir(h):
            new.append(mlines.Line2D([], [], c=h.get_edgecolor(),
                                     linestyle=h.get_linestyle()))
        else:
            new.append(h)
    ax.legend(handles=new, labels=labels, prop={'size': size}, frameon=False,
              ncol=ncol, loc=loc, handlelength=2.0, columnspacing=1.4)


def one_figure(d, key, label, kind, out):
    bins = d.bins(key)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    wid = bins[1] - bins[0]

    u, ue = d.h(['unpol'], key)
    s4, s4e = d.h(DIAG, key)
    s9, s9e = d.h(ROUTE1, key)
    s9b, s9be = d.h(ROUTE2, key)
    it, ite = d.h(['x_t', 'i_tp', 'i_tm'], key)      # the whole interference

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                             gridspec_kw={'height_ratios': [2.5, 1.4]})
    fig.set_size_inches(7 * 0.75, 6.0)
    fig.subplots_adjust(hspace=0.06)
    for ax in axes:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax = axes[0]
    ax.errorbar(ctr, u / wid, yerr=ue / wid, fmt='o', ms=4, color=C_UNPOL,
                lw=1.0, capsize=2, label=r'unpolarised')
    ax.hist(x=ctr, weights=s4 / wid, histtype='step', bins=len(ctr),
            range=(bins[0], bins[-1]), linewidth=LW, color=C_SUM4,
            linestyle='dashed', label=r'4 diagonal blocks')
    ax.hist(x=ctr, weights=s9 / wid, histtype='step', bins=len(ctr),
            range=(bins[0], bins[-1]), linewidth=LW, color=C_SUM9,
            label=r'all 9 blocks')
    ax.hist(x=ctr, weights=it / wid, histtype='step', bins=len(ctr),
            range=(bins[0], bins[-1]), linewidth=LW, color=C_INT,
            linestyle='dotted', label=r'interference only')
    ax.axhline(0.0, color='gray', lw=0.6, linestyle='dashed')
    ax.set_ylabel(r'$d\sigma/dX$ [pb]')
    if kind == 'control':
        ax.set_yscale('log')
        ax.set_ylim(bottom=max(1e-4, np.nanmin(u[u > 0] / wid) * 0.3),
                    top=np.nanmax(u / wid) * 12.0)
    else:
        lo0 = min(0.0, np.nanmin(it / wid) * 1.3)
        ax.set_ylim(lo0, np.nanmax(u / wid) * 1.42)
    update_legend(ax, ncol=2, loc='upper right')
    plt.setp(ax.get_xticklabels(), visible=False)

    ax = axes[1]
    r4, r4e = ratio(s4, s4e, u, ue)
    r9, r9e = ratio(s9, s9e, u, ue)
    r9b, r9be = ratio(s9b, s9be, u, ue)
    ax.axhline(1.0, color='gray', lw=0.8, linestyle='dashed')
    ax.errorbar(ctr - wid / 6, r4, yerr=r4e, fmt='v', ms=4, color=C_SUM4,
                lw=1.0, capsize=2, label=r'4 diagonal')
    ax.errorbar(ctr + wid / 6, r9, yerr=r9e, fmt='o', ms=4, color=C_SUM9,
                lw=1.0, capsize=2, label=r'9 blocks')
    ax.set_ylabel(r'sum / unpol.')
    ax.set_xlabel(label)
    # scale on the bins that are actually measured: a sparse edge bin with a
    # 100% error otherwise sets the range and hides everything else
    ok = np.isfinite(r9e) & (r9e < 0.35) & np.isfinite(r4e) & (r4e < 0.35)
    if ok.sum() < 4:
        ok = np.isfinite(r9e) & np.isfinite(r4e)
    lo = np.nanmin(np.concatenate([(r4 - r4e)[ok], (r9 - r9e)[ok]]))
    hi = np.nanmax(np.concatenate([(r4 + r4e)[ok], (r9 + r9e)[ok]]))
    pad = 0.12 * max(hi - lo, 0.05)
    ax.set_ylim(min(lo - pad, 0.96), max(hi + pad, 1.04))
    update_legend(ax, ncol=2, loc='best')
    fig.savefig(os.path.join(out, 'closure_%s.pdf' % key), bbox_inches='tight')
    fig.savefig(os.path.join(out, 'closure_%s.png' % key), dpi=160,
                bbox_inches='tight')
    plt.close(fig)

    c4 = chi2(s4, s4e, u, ue)
    c9 = chi2(s9, s9e, u, ue)
    c9b = chi2(s9b, s9be, u, ue)
    k = implied_scale(s4, s4e, it, ite, u, ue)
    # the interference integrates to zero over the DECAY phase space at every
    # production point, so its contribution to a purely production-level
    # observable must vanish bin by bin -- a null test with 20 bins
    czero = chi2(it, ite, np.zeros_like(it), np.zeros_like(ite))
    return c4, c9, c9b, k, czero


def summary_figure(d, keys, out):
    """One figure, the three observables the diagonal-only sum failed on."""
    fig = plt.figure()
    fig.set_size_inches(7 * 0.75 * len(keys), 5.6)
    gs = fig.add_gridspec(2, len(keys), height_ratios=[2.5, 1.4],
                          hspace=0.06, wspace=0.30)
    for col, key in enumerate(keys):
        label = dict((k, l) for k, l, _ in OBS)[key]
        bins = d.bins(key)
        ctr = 0.5 * (bins[1:] + bins[:-1])
        wid = bins[1] - bins[0]
        u, ue = d.h(['unpol'], key)
        s4, s4e = d.h(DIAG, key)
        s9, s9e = d.h(ROUTE1, key)
        it, ite = d.h(['x_t', 'i_tp', 'i_tm'], key)

        ax = fig.add_subplot(gs[0, col])
        rax = fig.add_subplot(gs[1, col], sharex=ax)
        for a in (ax, rax):
            a.yaxis.set_minor_locator(AutoMinorLocator())
            a.xaxis.set_minor_locator(AutoMinorLocator())
        ax.errorbar(ctr, u / wid, yerr=ue / wid, fmt='o', ms=4, color=C_UNPOL,
                    lw=1.0, capsize=2, label='unpolarised')
        ax.hist(x=ctr, weights=s4 / wid, histtype='step', bins=len(ctr),
                range=(bins[0], bins[-1]), linewidth=LW, color=C_SUM4,
                linestyle='dashed', label='4 diagonal blocks')
        ax.hist(x=ctr, weights=s9 / wid, histtype='step', bins=len(ctr),
                range=(bins[0], bins[-1]), linewidth=LW, color=C_SUM9,
                label='all 9 blocks')
        ax.hist(x=ctr, weights=it / wid, histtype='step', bins=len(ctr),
                range=(bins[0], bins[-1]), linewidth=LW, color=C_INT,
                linestyle='dotted', label='interference only')
        ax.axhline(0.0, color='gray', lw=0.6, linestyle='dashed')
        ax.set_ylim(min(0.0, np.nanmin(it / wid) * 1.3),
                    np.nanmax(u / wid) * 1.62)
        if col == 0:
            ax.set_ylabel(r'$d\sigma/dX$ [pb]')
            rax.set_ylabel(r'sum / unpolarised')
        if col == len(keys) - 1:
            update_legend(ax, ncol=1, loc='upper right', size=8)
        plt.setp(ax.get_xticklabels(), visible=False)

        r4, r4e = ratio(s4, s4e, u, ue)
        r9, r9e = ratio(s9, s9e, u, ue)
        rax.axhline(1.0, color='gray', lw=0.8, linestyle='dashed')
        rax.errorbar(ctr - wid / 6, r4, yerr=r4e, fmt='v', ms=4, color=C_SUM4,
                     lw=1.0, capsize=2, label='4 diagonal')
        rax.errorbar(ctr + wid / 6, r9, yerr=r9e, fmt='o', ms=4, color=C_SUM9,
                     lw=1.0, capsize=2, label='9 blocks')
        ok = np.isfinite(r9e) & (r9e < 0.35) & np.isfinite(r4e) & (r4e < 0.35)
        if ok.sum() < 4:
            ok = np.isfinite(r9e) & np.isfinite(r4e)
        lo = np.nanmin(np.concatenate([(r4 - r4e)[ok], (r9 - r9e)[ok]]))
        hi = np.nanmax(np.concatenate([(r4 + r4e)[ok], (r9 + r9e)[ok]]))
        pad = 0.12 * max(hi - lo, 0.05)
        rax.set_ylim(min(lo - pad, 0.96), max(hi + pad, 1.04))
        rax.set_xlabel(label)
        if col == 0:
            update_legend(rax, ncol=2, loc='best', size=8)
    fig.suptitle(r'MadSpin interference closure: '
                 r'$pp\to t\bar t$, 13 TeV, LO, dileptonic', y=0.96)
    fig.savefig(os.path.join(out, 'closure_summary.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(out, 'closure_summary.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def blocks_figure(d, key, label, out):
    """The nine blocks separately, on one observable."""
    bins = d.bins(key)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    wid = bins[1] - bins[0]
    u, ue = d.h(['unpol'], key)

    named = [('pp', r'$(D^+\!,D^+)$'), ('pm', r'$(D^+\!,D^-)$'),
             ('mp', r'$(D^-\!,D^+)$'), ('mm', r'$(D^-\!,D^-)$'),
             ('i_tbp', r'$(I,D^+)$'), ('i_tbm', r'$(I,D^-)$'),
             ('i_tp', r'$(D^+\!,I)$'), ('i_tm', r'$(D^-\!,I)$')]

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                             gridspec_kw={'height_ratios': [2.5, 1.4]})
    fig.set_size_inches(7 * 0.75, 6.0)
    fig.subplots_adjust(hspace=0.06)
    for ax in axes:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax = axes[0]
    for i, (tag, lab) in enumerate(named):
        s, _e = d.h([tag], key)
        ax.hist(x=ctr, weights=s / wid, histtype='step', bins=len(ctr),
                range=(bins[0], bins[-1]), linewidth=1.0,
                color=allcolors[(i + 4) % len(allcolors)],
                linestyle='solid' if i < 4 else 'dashed', label=lab)
    s, e = d.hdiff(['x_t'], ['i_tbp', 'i_tbm'], key)
    ax.hist(x=ctr, weights=s / wid, histtype='step', bins=len(ctr),
            range=(bins[0], bins[-1]), linewidth=1.4, color=C_INT,
            linestyle='dotted', label=r'$(I,I)$')
    ax.axhline(0.0, color='gray', lw=0.6, linestyle='dashed')
    ax.set_ylabel(r'$d\sigma/dX$ [pb]')
    top = max(np.nanmax(d.h([t], key)[0] / wid) for t, _ in named)
    bot = min(0.0, np.nanmin(s / wid) * 1.25)
    ax.set_ylim(bot, top * 1.55)
    update_legend(ax, ncol=3, loc='upper right', size=7)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax = axes[1]
    s2, e2 = d.hdiff(['x_tb'], ['i_tp', 'i_tm'], key)
    ax.axhline(0.0, color='gray', lw=0.8, linestyle='dashed')
    ax.errorbar(ctr - wid / 6, s / wid, yerr=e / wid, fmt='o', ms=4,
                color=C_INT, lw=1.0, capsize=2,
                label=r'$x_t-(I,D^+)-(I,D^-)$')
    ax.errorbar(ctr + wid / 6, s2 / wid, yerr=e2 / wid, fmt='s', ms=4,
                mfc='none', color=allcolors[2], lw=1.0, capsize=2,
                label=r'$x_{\bar t}-(D^+\!,I)-(D^-\!,I)$')
    ax.set_ylabel(r'$(I,I)$ [pb]')
    ax.set_xlabel(label)
    update_legend(ax, ncol=1, loc='best', size=8)
    fig.savefig(os.path.join(out, 'blocks_%s.pdf' % key), bbox_inches='tight')
    fig.savefig(os.path.join(out, 'blocks_%s.png' % key), dpi=160,
                bbox_inches='tight')
    plt.close(fig)
    return chi2(s, e, s2, e2)


def main():
    ddir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    d = Data(ddir)

    L = []
    m = d.meta
    L.append('MadSpin interference closure -- p p > t t~, 13 TeV, LO, '
             'dileptonic')
    L.append('')
    L.append('normalisation constant of the pure-interference samples')
    L.append('  c = <W> over the decay phase space, measured on the '
             'unpolarised sample')
    L.append('  run in the ordinary joint scheme:  c = eff * max_weight')
    L.append('    eff        = %.6f' % m['_c']['eff'])
    L.append('    max_weight = %.6e' % m['_c']['maxwgt'])
    L.append('    c          = %.6e' % m['_c']['measured'])
    L.append('  every interference sample is multiplied by max_weight_i / c '
             '(one number,')
    L.append('  predicted, not fitted -- see analyse_interference.py)')
    L.append('')
    L.append('samples')
    L.append('  %-6s %-6s %9s %14s %12s %10s %8s'
             % ('tag', 'kind', 'written', 'unit [pb]', 'max_weight', 'keep',
                'z'))
    for tag in ['unpol'] + DIAG + ['i_tbp', 'i_tbm', 'i_tp', 'i_tm',
                                   'x_t', 'x_tb']:
        s = m[tag]
        keep = ('%.4f' % (s['n_written'] / float(s['n_read']))
                if s['kind'] == 'inter' else '1.0')
        L.append('  %-6s %-6s %9d %14.6g %12.5g %10s %8s'
                 % (tag, s['kind'], s['n_written'], s['unit'],
                    s['maxwgt'] if s['maxwgt'] else float('nan'), keep,
                    ('%+.2f' % s['z']) if s['kind'] == 'inter' else '-'))
    L.append('')

    # -------- rates --------------------------------------------------------
    L.append('total rate')
    s4 = sum(m[t]['xsec'] for t in DIAG)
    e4 = math.sqrt(sum(m[t]['xerr'] ** 2 for t in DIAG))
    su, eu = m['unpol']['xsec'], m['unpol']['xerr']
    L.append('  sum of the 4 diagonal samples : %12.5f +- %.5f pb' % (s4, e4))
    L.append('  unpolarised                   : %12.5f +- %.5f pb' % (su, eu))
    L.append('  ratio                         : %.6f  (%.2f sigma)'
             % (s4 / su, (s4 - su) / math.sqrt(e4 ** 2 + eu ** 2)))
    tot_int = 0.0
    tot_int_e = 0.0
    for t in ['x_t', 'i_tp', 'i_tm']:
        sw, _swo, sw2 = d.mom([t], 'cos_k_p')[:3]
        tot_int += sw
        tot_int_e += sw2
    L.append('  the 5 interference blocks     : %+12.5f +- %.5f pb  '
             '(zero by construction)' % (tot_int, math.sqrt(tot_int_e)))
    L.append('  -> the 9-block total is the 4-block one, unchanged: the '
             'interference')
    L.append('     terms carry no cross-section.')
    L.append('')

    # -------- means --------------------------------------------------------
    L.append('means.  The interference column is the exact shift it produces,')
    L.append('  <O>_9 - <O>_4 = (sum_int w O) / (sum_4 w), since sum_int w = 0.')
    L.append('  %-10s %20s %20s %20s %20s %8s %8s'
             % ('observable', '4 diagonal', 'interference', '9 blocks',
                'unpolarised', 'pull(4)', 'pull(9)'))
    for key, label, kind in OBS:
        m4, s4_ = d.mean(DIAG, key)
        m9, s9_ = d.mean(ROUTE1, key)
        mu, su_ = d.mean(['unpol'], key)
        norm = d.mom(DIAG, key)[0]
        swo = d.mom(['x_t', 'i_tp', 'i_tm'], key)[1]
        sw2o2 = d.mom(['x_t', 'i_tp', 'i_tm'], key)[4]
        p4 = (m4 - mu) / math.sqrt(s4_ ** 2 + su_ ** 2)
        p9 = (m9 - mu) / math.sqrt(s9_ ** 2 + su_ ** 2)
        L.append('  %-10s %12.5f+-%-7.5f %12.5f+-%-7.5f %12.5f+-%-7.5f '
                 '%12.5f+-%-7.5f %8.2f %8.2f'
                 % (key, m4, s4_, swo / norm, math.sqrt(sw2o2) / norm,
                    m9, s9_, mu, su_, p4, p9))
    L.append('')

    # -------- per-bin chi2 -------------------------------------------------
    L.append('per-bin chi2 against the unpolarised sample (20 bins), and the')
    L.append('best-fit scale k of the interference contribution '
             '(predicted k = 1, never fitted)')
    L.append('  %-10s %14s %14s %14s %16s %14s  %s'
             % ('observable', 'chi2 (4 diag)', 'chi2 (9, r1)',
                'chi2 (9, r2)', 'k', 'chi2 int vs 0', 'kind'))
    for key, label, kind in OBS:
        c4, c9, c9b, k, cz = one_figure(d, key, label, kind, out)
        L.append('  %-10s %9.1f /%-3d %9.1f /%-3d %9.1f /%-3d %8.3f+-%-6.3f '
                 '%9.1f /%-3d  %s'
                 % (key, c4[0], c4[1], c9[0], c9[1], c9b[0], c9b[1],
                    k[0], k[1], cz[0], cz[1], kind))
    L.append('  (chi2 int vs 0: the interference integrates to zero over the '
             'decay phase')
    L.append('   space at every production point, so for a production-level '
             'observable')
    L.append('   -- pt_t, m_tt -- it must vanish bin by bin; elsewhere a large '
             'value IS')
    L.append('   the signal.)')
    L.append('')

    # the identity cos phi_ll = C_kk + C_rr + C_nn, term by term
    L.append('identity  <cos phi_ll> = <C_kk> + <C_rr> + <C_nn>')
    for name, tags in (('4 diagonal', DIAG), ('9 blocks (r1)', ROUTE1),
                       ('9 blocks (r2)', ROUTE2), ('unpolarised', ['unpol'])):
        kk = d.mean(tags, 'ckk')[0]
        rr = d.mean(tags, 'crr')[0]
        nn = d.mean(tags, 'cnn')[0]
        ph = d.mean(tags, 'cos_phi')[0]
        L.append('  %-14s %+.5f %+.5f %+.5f = %+.5f   (measured %+.5f, '
                 'diff %+.1e)' % (name, kk, rr, nn, kk + rr + nn, ph,
                                  kk + rr + nn - ph))
    L.append('')

    summary_figure(d, ['cnn', 'cos_phi', 'dphi_lab'], out)

    L.append('the (I,I) block from the two routes (must agree bin by bin)')
    L.append('  %-10s %14s' % ('observable', 'chi2 / nbins'))
    for key, label, kind in OBS:
        if key not in ('cnn', 'cos_phi', 'dphi_lab', 'ckk'):
            continue
        cc = blocks_figure(d, key, label, out)
        L.append('  %-10s %9.1f /%-3d' % (key, cc[0], cc[1]))

    open(os.path.join(out, 'closure_numbers.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwrote plots + closure_numbers.txt to', out)


if __name__ == '__main__':
    main()
