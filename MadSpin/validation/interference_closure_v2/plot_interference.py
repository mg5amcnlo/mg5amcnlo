#!/usr/bin/env python3
"""Plots and numbers of the MadSpin interference-closure test, second run.

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
C_SUM4 = allcolors[0]           # blue   -- the diagonal-only result
C_SUM9 = allcolors[3]           # red    -- diagonal + interference
C_INT = allcolors[4]            # purple -- the interference sum

# The four diagonal blocks, when they are shown one by one on top of a closure
# figure.  Line style encodes the polarisation of the TOP (the first index):
# solid for the two blocks with the top in D+, dash-dot for the two with the
# top in D-, so the grouping on the page is the grouping in the physics.
DIAG_COLOR = {'pp': allcolors[2],       # tab:green
              'pm': 'tab:orange',
              'mp': 'tab:brown',
              'mm': 'tab:cyan'}
DIAG_LS = {'pp': 'solid', 'pm': 'solid', 'mp': 'dashdot', 'mm': 'dashdot'}

# Observables whose closure figure shows the four diagonal blocks as well.
# For cos(theta^k) of the l+ the polarisation of the top -- the FIRST index of
# the block -- fixes the slope, so (D+,D+) and (D+,D-) rise while (D-,D+) and
# (D-,D-) fall, and the unscaled sum of the four is the flat unpolarised curve.
AUGMENT_WITH_DIAG_BLOCKS = ('cos_k_p',)

# What the four individual block curves actually are.  Stated on the figure
# itself, because a rescaled curve next to an unscaled reference is exactly the
# kind of thing a reader mis-reads.
BLOCK_NOTE = ('individual blocks scaled by '
              r'$\sigma_{\mathrm{unpol}}/\sigma_{\mathrm{block}}$'
              ' -- shape only, they do not add up;' '\n'
              'the reference and the 4- and 9-block sums are unscaled')


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

DIAG = ['pp', 'pm', 'mp', 'mm']                        # 4 diagonal-diagonal
INTER = ['i_dp', 'i_dm', 'dp_i', 'dm_i', 'ii']         # the 5 with an I index
ALL9 = DIAG + INTER
BLOCK_LABEL = {'pp': r'$(D^+\!,D^+)$', 'pm': r'$(D^+\!,D^-)$',
               'mp': r'$(D^-\!,D^+)$', 'mm': r'$(D^-\!,D^-)$',
               'i_dp': r'$(I,D^+)$', 'i_dm': r'$(I,D^-)$',
               'dp_i': r'$(D^+\!,I)$', 'dm_i': r'$(D^-\!,I)$',
               'ii': r'$(I,I)$'}


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
    """Best-fit k in  (4 diagonal blocks) + k * (5 interference) = unpolarised.

    k = 1 is a prediction with nothing fitted anywhere: the fully weighted mode
    normalises itself (w = sigma_ref*BR*W/c, bin content = sum w / N_file), so
    unlike the first run of this test there is no measured constant standing
    between the samples.  Quoted, never used.
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


def one_figure(d, key, label, kind, out, show_diag_blocks=False, name=None):
    """Closure figure: unpolarised reference vs the 4- and 9-block sums.

    With ``show_diag_blocks`` the upper panel additionally carries the four
    diagonal blocks one by one.  Their cross sections differ by a factor two
    (7.93 pb for (D+,D+) and (D-,D-), 3.94 pb for the mixed pair), so drawn raw
    they would be neither comparable with each other nor with the reference;
    each is therefore multiplied by sigma_unpol / sigma_block, i.e. rescaled to
    a common normalisation, and only its SHAPE is on the page.  The factor is
    written into the legend entry and the panel carries a note, because the
    rescaled curves deliberately do NOT add up to the '4 diagonal blocks' curve
    next to them -- that one is the unscaled sum, and it is the sum the ratio
    panel below tests.
    """
    bins = d.bins(key)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    wid = bins[1] - bins[0]

    u, ue = d.h(['unpol'], key)
    s4, s4e = d.h(DIAG, key)
    s9, s9e = d.h(ALL9, key)
    it, ite = d.h(INTER, key)

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
            linestyle='dotted', label=r'5 interference blocks')

    top_extra = 0.0
    if show_diag_blocks:
        su = d.meta['unpol']['xsec']
        for tag in DIAG:
            s, _e = d.h([tag], key)
            f = su / d.meta[tag]['xsec']
            ax.hist(x=ctr, weights=f * s / wid, histtype='step', bins=len(ctr),
                    range=(bins[0], bins[-1]), linewidth=1.0,
                    color=DIAG_COLOR[tag], linestyle=DIAG_LS[tag], zorder=1,
                    label=r'%s $\times\,%.1f$' % (BLOCK_LABEL[tag], f))
            top_extra = max(top_extra, float(np.nanmax(f * s / wid)))

    ax.axhline(0.0, color='gray', lw=0.6, linestyle='dashed')
    ax.set_ylabel(r'$d\sigma/dX$ [pb]')
    if kind == 'control':
        ax.set_yscale('log')
        ax.set_ylim(bottom=max(1e-4, np.nanmin(u[u > 0] / wid) * 0.3),
                    top=np.nanmax(u / wid) * 12.0)
    else:
        lo0 = min(0.0, np.nanmin(it / wid) * 1.3)
        ax.set_ylim(lo0, max(np.nanmax(u / wid), top_extra)
                    * (1.55 if show_diag_blocks else 1.42))
    if show_diag_blocks:
        # above the frame: the legend already fills the top of the panel, and
        # this caption must not be something the eye can skip
        ax.text(0.0, 1.015, BLOCK_NOTE, transform=ax.transAxes, va='bottom',
                ha='left', fontsize=8, color='dimgray', linespacing=1.4)
    update_legend(ax, ncol=2, loc='upper right',
                  size=8 if show_diag_blocks else 9)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax = axes[1]
    r4, r4e = ratio(s4, s4e, u, ue)
    r9, r9e = ratio(s9, s9e, u, ue)
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
    name = name or ('closure_%s' % key)
    fig.savefig(os.path.join(out, '%s.pdf' % name), bbox_inches='tight')
    fig.savefig(os.path.join(out, '%s.png' % name), dpi=160,
                bbox_inches='tight')
    plt.close(fig)

    c4 = chi2(s4, s4e, u, ue)
    c9 = chi2(s9, s9e, u, ue)
    k = implied_scale(s4, s4e, it, ite, u, ue)
    # the interference integrates to zero over the DECAY phase space at every
    # production point, so its contribution to a purely production-level
    # observable must vanish bin by bin -- a null test with 20 bins
    czero = chi2(it, ite, np.zeros_like(it), np.zeros_like(ite))
    return c4, c9, k, czero


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
        s9, s9e = d.h(ALL9, key)
        it, ite = d.h(INTER, key)

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
                linestyle='dotted', label='5 interference blocks')
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
    """The nine blocks separately, every one of them a sample of its own.

    Upper panel: the four diagonal-diagonal blocks and the five interference
    ones.  Lower panel: the five interference blocks alone, on their own scale
    -- which is where the (I,I) block shows that it carries the whole effect.
    """
    bins = d.bins(key)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    wid = bins[1] - bins[0]

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1.7]})
    fig.set_size_inches(7 * 0.75, 6.4)
    fig.subplots_adjust(hspace=0.06)
    for ax in axes:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax = axes[0]
    for i, tag in enumerate(ALL9):
        s, _e = d.h([tag], key)
        ax.hist(x=ctr, weights=s / wid, histtype='step', bins=len(ctr),
                range=(bins[0], bins[-1]), linewidth=1.0,
                color=allcolors[(i + 4) % len(allcolors)],
                linestyle='solid' if i < 4 else 'dashed',
                label=BLOCK_LABEL[tag])
    ax.axhline(0.0, color='gray', lw=0.6, linestyle='dashed')
    ax.set_ylabel(r'$d\sigma/dX$ [pb]')
    top = max(np.nanmax(d.h([t], key)[0] / wid) for t in DIAG)
    bot = min(0.0, min(np.nanmin(d.h([t], key)[0] / wid) for t in INTER) * 1.25)
    ax.set_ylim(bot, top * 1.60)
    update_legend(ax, ncol=3, loc='upper right', size=7)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax = axes[1]
    ax.axhline(0.0, color='gray', lw=0.8, linestyle='dashed')
    for i, tag in enumerate(INTER):
        s, e = d.h([tag], key)
        off = (i - 2) * wid / 8.0
        ax.errorbar(ctr + off, s / wid, yerr=e / wid,
                    fmt='os^vD'[i], ms=3.5, mfc='none' if i < 4 else None,
                    color=allcolors[(i + 8) % len(allcolors)] if i < 4 else C_INT,
                    lw=1.0, capsize=1.5, label=BLOCK_LABEL[tag])
    ax.set_ylabel(r'interference [pb]')
    ax.set_xlabel(label)
    update_legend(ax, ncol=3, loc='best', size=7)
    fig.savefig(os.path.join(out, 'blocks_%s.pdf' % key), bbox_inches='tight')
    fig.savefig(os.path.join(out, 'blocks_%s.png' % key), dpi=160,
                bbox_inches='tight')
    plt.close(fig)
    # how much of the interference the (I,I) block alone carries, and the
    # null test on the four single-I blocks
    ii, iie = d.h(['ii'], key)
    single, singlee = d.h(['i_dp', 'i_dm', 'dp_i', 'dm_i'], key)
    zeros = np.zeros_like(ii)
    return (chi2(ii, iie, zeros, zeros),
            chi2(single, singlee, zeros, zeros))


def main():
    ddir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    d = Data(ddir)

    L = []
    m = d.meta
    L.append('MadSpin interference closure (v2, reworked interface)')
    L.append('  p p > t t~, 13 TeV, LO, dileptonic')
    L.append('')
    L.append('The fully weighted pure-interference mode normalises itself:')
    L.append('  w = sigma_ref * BR * W / c   and   bin [pb] = sum_bin(w) / N_file')
    L.append('for every sample in this test, diagonal and interference alike.')
    L.append('Nothing below is fitted and nothing is read out of a log.')
    L.append('')
    L.append('samples')
    L.append('  %-6s %-6s %9s %14s %14s %14s %8s %6s'
             % ('tag', 'kind', 'N_file', 'sigma [pb]', 'mean(w) [pb]',
                'c (banner)', 'z', 'dead'))
    for tag in ['unpol'] + DIAG + INTER:
        s = m[tag]
        inter = s['kind'] == 'inter'
        L.append('  %-6s %-6s %9d %14.6g %+14.6e %14s %8s %6s'
                 % (tag, s['kind'], s['n_file'], s['xsec'], s['mean_w'],
                    ('%.6e' % s['c']) if inter else '-',
                    ('%+.2f' % s['z']) if inter else '-',
                    ('%d' % s['dead']) if inter else '-'))
    cs = [m[t]['c'] for t in INTER]
    L.append('  c is measured independently in each interference run (the')
    L.append('  maximum-weight scan): %.6e .. %.6e, spread %.2f%% -- it is a'
             % (min(cs), max(cs), 100 * (max(cs) - min(cs)) / np.mean(cs)))
    L.append('  decay-side constant, so this is a free consistency check.')
    L.append('')

    # -------- rates --------------------------------------------------------
    L.append('total rate')
    s4 = sum(m[t]['xsec'] for t in DIAG)
    e4 = math.sqrt(sum(m[t]['xerr'] ** 2 for t in DIAG))
    su, eu = m['unpol']['xsec'], m['unpol']['xerr']
    L.append('  sum of the 4 diagonal samples : %12.5f +- %.5f pb' % (s4, e4))
    L.append('  unpolarised                   : %12.5f +- %.5f pb' % (su, eu))
    L.append('  ratio                         : %.6f +- %.6f (%.2f sigma)'
             % (s4 / su, (s4 / su) * math.sqrt((e4 / s4) ** 2 + (eu / su) ** 2),
                (s4 - su) / math.sqrt(e4 ** 2 + eu ** 2)))
    L.append('  production, sum of the 4      : %12.5f +- %.5f pb'
             % (sum(m[t]['prod_xsec'] for t in DIAG),
                math.sqrt(sum(m[t]['prod_err'] ** 2 for t in DIAG))))
    L.append('  production, unpolarised       : %12.5f +- %.5f pb'
             % (m['unpol']['prod_xsec'], m['unpol']['prod_err']))
    L.append('')
    L.append('  each interference block, integrated (its own XSECUP is 0):')
    tot_int = 0.0
    tot_int_e2 = 0.0
    for t in INTER:
        sw, _swo, sw2 = d.mom([t], 'cos_k_p')[:3]
        tot_int += sw
        tot_int_e2 += sw2
        L.append('    %-6s %+12.6f +- %.6f pb   (%+.2f sigma)'
                 % (t, sw, math.sqrt(sw2), sw / math.sqrt(sw2)))
    L.append('    %-6s %+12.6f +- %.6f pb   (%+.2f sigma)'
             % ('all 5', tot_int, math.sqrt(tot_int_e2),
                tot_int / math.sqrt(tot_int_e2)))
    L.append('  -> the 9-block total is the 4-block one, unchanged: the '
             'interference')
    L.append('     terms carry no cross-section.  %.2g of the %.2f pb total.'
             % (abs(tot_int) / su, su))
    L.append('')

    # -------- means --------------------------------------------------------
    L.append('means.  The interference column is the exact shift it produces,')
    L.append('  <O>_9 - <O>_4 = (sum_int w O) / (sum_4 w), since sum_int w = 0.')
    L.append('  %-10s %20s %20s %20s %20s %8s %8s'
             % ('observable', '4 diagonal', 'interference', '9 blocks',
                'unpolarised', 'pull(4)', 'pull(9)'))
    for key, label, kind in OBS:
        m4, s4_ = d.mean(DIAG, key)
        m9, s9_ = d.mean(ALL9, key)
        mu, su_ = d.mean(['unpol'], key)
        norm = d.mom(DIAG, key)[0]
        swo = d.mom(INTER, key)[1]
        sw2o2 = d.mom(INTER, key)[4]
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
    L.append('  %-10s %14s %14s %16s %14s  %s'
             % ('observable', 'chi2 (4 diag)', 'chi2 (9 blocks)',
                'k', 'chi2 int vs 0', 'kind'))
    for key, label, kind in OBS:
        if key in AUGMENT_WITH_DIAG_BLOCKS:
            # the figure the write-up shows carries the four diagonal blocks;
            # the earlier, plain rendering is kept next to it under *_plain
            one_figure(d, key, label, kind, out, name='closure_%s_plain' % key)
            c4, c9, k, cz = one_figure(d, key, label, kind, out,
                                       show_diag_blocks=True)
        else:
            c4, c9, k, cz = one_figure(d, key, label, kind, out)
        L.append('  %-10s %9.1f /%-3d %9.1f /%-3d %8.3f+-%-6.3f '
                 '%9.1f /%-3d  %s'
                 % (key, c4[0], c4[1], c9[0], c9[1],
                    k[0], k[1], cz[0], cz[1], kind))
    L.append('  (chi2 int vs 0: the interference integrates to zero over the '
             'decay phase')
    L.append('   space at every production point, so for a production-level '
             'observable')
    L.append('   -- pt_t, m_tt -- it must vanish bin by bin; elsewhere a large '
             'value IS')
    L.append('   the signal.)')
    L.append('')

    # -------- the null test, bin by bin -----------------------------------
    # A production-level observable must get NO contribution from the
    # interference in ANY bin, so the per-bin pulls are worth seeing and not
    # just their chi2: a chi2 that sits a little high because of one bin is a
    # fluctuation, one spread over many bins would be a bias.
    L.append('null test, bin by bin: pull of the 5 interference blocks against')
    L.append('  zero, for the two production-level observables')
    for key in ('pt_t', 'm_tt'):
        s, e = d.h(INTER, key)
        pull = np.where(e > 0, s / np.where(e > 0, e, 1), 0.0)
        L.append('  %-6s %s' % (key, ' '.join('%+.1f' % p for p in pull)))
        L.append('  %-6s chi2 = %.1f / %d, largest |pull| = %.2f'
                 % ('', float((pull ** 2).sum()), len(pull),
                    float(np.abs(pull).max())))
    L.append('')

    # the identity cos phi_ll = C_kk + C_rr + C_nn, term by term
    L.append('identity  <cos phi_ll> = <C_kk> + <C_rr> + <C_nn>')
    for name, tags in (('4 diagonal', DIAG), ('9 blocks', ALL9),
                       ('unpolarised', ['unpol'])):
        kk = d.mean(tags, 'ckk')[0]
        rr = d.mean(tags, 'crr')[0]
        nn = d.mean(tags, 'cnn')[0]
        ph = d.mean(tags, 'cos_phi')[0]
        L.append('  %-14s %+.5f %+.5f %+.5f = %+.5f   (measured %+.5f, '
                 'diff %+.1e)' % (name, kk, rr, nn, kk + rr + nn, ph,
                                  kk + rr + nn - ph))
    L.append('')

    summary_figure(d, ['cnn', 'cos_phi', 'dphi_lab'], out)

    L.append('where the interference lives: the (I,I) block against the four')
    L.append('single-I blocks, each measured directly (no subtraction)')
    L.append('  %-10s %16s %22s' % ('observable', 'chi2(I,I vs 0)',
                                    'chi2(4 single-I vs 0)'))
    for key, label, kind in OBS:
        if key not in ('cnn', 'cos_phi', 'dphi_lab', 'ckk', 'pt_t'):
            continue
        cii, csingle = blocks_figure(d, key, label, out)
        L.append('  %-10s %11.1f /%-3d %17.1f /%-3d'
                 % (key, cii[0], cii[1], csingle[0], csingle[1]))

    open(os.path.join(out, 'closure_numbers.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwrote plots + closure_numbers.txt to', out)


if __name__ == '__main__':
    main()
