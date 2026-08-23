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
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator


# --- work around a matplotlib bug that eats every minus sign in the PDFs ---
def _fix_type1_subset_minus():
    """Stop matplotlib's usetex PDF path from dropping every minus sign.

    ``_type1font.Type1Font.subset`` in matplotlib 3.11.0 ends its encoding
    filter with an unconditional

        encoding[0] = '.notdef'

    For a text font slot 0 really is ``.notdef``, but TeX's CMSY10 -- the font
    every math minus comes from -- carries ``minus`` there.  Subsetting a
    figure's fonts therefore throws the minus away: the glyph walk follows
    ``.notdef`` instead, so the outline never reaches the embedded font, and
    the PDF font dictionary comes out as

        /BaseFont /XXXXXX+CMSY10 /Encoding << /Differences [ ] >>

    with no /BaseEncoding.  Viewers fall back to StandardEncoding, which has
    nothing at code 0, and the sign silently disappears -- no warning, no
    error, a wrong figure.  Only PDF is affected: the PNGs go through dvipng,
    which rasterises before any of this happens.

    In these figures that removed every ``D^-`` from the block labels *and*
    the minus from every negative axis tick label.  Slot 0 becomes a default
    here rather than an override, which is what upstream should do.

    Guarded on the exact upstream line, so a fixed or restructured matplotlib
    is left alone.  Returns True if the patch was applied.
    """
    import inspect
    import textwrap
    from matplotlib import _type1font

    bad = "encoding[0] = '.notdef'"
    good = "encoding.setdefault(0, '.notdef')"
    try:
        src = textwrap.dedent(inspect.getsource(_type1font.Type1Font.subset))
    except (OSError, TypeError):            # no source available
        return False
    if bad not in src:                      # already fixed upstream
        return False
    ns = vars(_type1font).copy()
    exec(compile(src.replace(bad, good), '<type1font-minus-fix>', 'exec'), ns)
    _type1font.Type1Font.subset = ns['subset']
    return True


MINUS_FIX = _fix_type1_subset_minus()


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

# Observables that also get a stacked rendering (see ``signed_stack``).
# closure: the contributions of 'all 9 blocks' laid on top of one another.
# blocks:  all nine one by one, plus the five interference ones on their own
#          scale in the lower panel.
STACKED_CLOSURE_FOR = ('cos_k_p', 'cnn', 'dphi_lab')
STACKED_BLOCKS_FOR = ('cnn', 'dphi_lab')

# The sign convention, written onto every stacked figure and repeated in
# RESULTS.md.  Kept short enough to sit above the frame without wrapping.
STACK_NOTE = ('stacked: positive contributions up from 0, negative down; '
              'net = line')


# --------------------------------------------------------------------------
# Axis labels.  Every figure in this directory histograms a PER-EVENT quantity;
# a spin-correlation coefficient is the MEAN of one of them, never the
# histogram.  Where the two could be confused the label therefore names the
# per-event product and points at the coefficient with an arrow that reads
# "whose mean is", '(mean -> C_ij)' -- and never puts '(C_ij)' next to a curve.
# The three C_ij panes are the ones that need it; cos_k_p, cos_k_m, cos_n_p,
# cos_phi, dphi_lab, pt_t and m_tt already name exactly what is drawn.
OBS = [
    ('cos_k_p',  r'$\cos\theta^{k}_{\ell^+}$', 'diagonal'),
    ('cos_k_m',  r'$\cos\theta^{k}_{\ell^-}$', 'diagonal'),
    ('ckk',      r'$\cos\theta^{k}_{\ell^+}\cos\theta^{k}_{\ell^-}$'
                 r'   (mean $\to C_{kk}$)',
     'diagonal'),
    ('cnn',      r'$\cos\theta^{n}_{\ell^+}\cos\theta^{n}_{\ell^-}$'
                 r'   (mean $\to C_{nn}$)',
     'off-diagonal'),
    ('crr',      r'$\cos\theta^{r}_{\ell^+}\cos\theta^{r}_{\ell^-}$'
                 r'   (mean $\to C_{rr}$)',
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

# One colour per block, used in BOTH panels of the stacked block figure.  The
# unstacked ``blocks_figure`` gives (I,I) green in its upper panel and purple
# in its lower one; that is harmless when the mark shape changes too, but not
# when the only thing identifying a band is its fill.
BLOCK_COLOR = dict((t, allcolors[(i + 4) % len(allcolors)])
                   for i, t in enumerate(ALL9))


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
    diagonal blocks one by one, each at its own cross section -- nothing is
    rescaled anywhere on this figure, so the four block curves add up, bin by
    bin, to the '4 diagonal blocks' curve drawn next to them, which is the sum
    the ratio panel below tests.  Line style encodes the polarisation of the
    top, the first block index: solid for D+, dash-dot for D-.
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

    if show_diag_blocks:
        for tag in DIAG:
            s, _e = d.h([tag], key)
            ax.hist(x=ctr, weights=s / wid, histtype='step', bins=len(ctr),
                    range=(bins[0], bins[-1]), linewidth=1.0,
                    color=DIAG_COLOR[tag], linestyle=DIAG_LS[tag], zorder=1,
                    label=BLOCK_LABEL[tag])

    ax.axhline(0.0, color='gray', lw=0.6, linestyle='dashed')
    ax.set_ylabel(r'$d\sigma/dX$ [pb]')
    if kind == 'control':
        ax.set_yscale('log')
        ax.set_ylim(bottom=max(1e-4, np.nanmin(u[u > 0] / wid) * 0.3),
                    top=np.nanmax(u / wid) * 12.0)
    else:
        lo0 = min(0.0, np.nanmin(it / wid) * 1.3)
        # eight legend entries instead of four when the blocks are shown
        ax.set_ylim(lo0, np.nanmax(u / wid)
                    * (1.62 if show_diag_blocks else 1.42))
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


# --------------------------------------------------------------------------
# Stacked rendering.
#
# Four of the nine blocks are strictly positive in every bin of every
# observable here; the five interference blocks are not -- (I,I) in particular
# is negative over half the range of C_nn and of Delta phi, and its whole point
# is that it changes sign.  A stack that just piles |contribution| on top of
# the previous one would draw those bins the wrong way up and silently turn a
# subtraction into an addition, so the sign has to be carried explicitly.
#
# Convention used here, and stated on every stacked figure:
#
#   in each bin, contributions with a positive content are stacked upward
#   from zero and contributions with a negative content are stacked downward
#   from zero.  No band ever crosses zero and no band is drawn on the wrong
#   side of it, so a colour below the axis is a subtraction and can be read as
#   one.  The consequence is that the *envelope* is NOT the total: the top of
#   the stack is the sum of the positive contributions alone.  The total is
#   the algebraic sum of the two piles and is drawn explicitly as a line.
#
# The alternative -- laying each band from the running cumulative sum, so that
# the last band's top is the total -- keeps the envelope meaningful but makes
# negative bands overlap the ones under them, which with nine components is
# unreadable.  The sign-split convention plus an explicit net line is the one
# used here, and ``stack_closure_check`` verifies bin by bin that the net of
# the stack really is the total it claims to be.
def signed_stack(comps):
    """Sign-split cumulative bands for a list of per-bin contributions.

    Returns ``(bands, pos_top, neg_bot, net)`` where ``bands[i]`` is the
    ``(lo, hi)`` pair to fill for component ``i``.  ``net`` is
    ``pos_top + neg_bot``, i.e. the plain sum of the components, which is what
    the figure's total curve must equal bin by bin.
    """
    up = np.zeros_like(comps[0], dtype=float)
    dn = np.zeros_like(comps[0], dtype=float)
    bands = []
    for c in comps:
        c = np.asarray(c, dtype=float)
        lo = np.where(c >= 0, up, dn)
        hi = lo + c
        bands.append((lo, hi))
        up = np.where(c >= 0, hi, up)
        dn = np.where(c >= 0, dn, hi)
    return bands, up, dn, up + dn


def stack_closure_check(comps, total):
    """max |net of the stack - total| and the same relative to the peak."""
    _b, _u, _d, net = signed_stack(comps)
    dev = float(np.nanmax(np.abs(net - total)))
    scale = float(np.nanmax(np.abs(total))) or 1.0
    return dev, dev / scale


def _stepfill(ax, edges, lo, hi, color, label=None, alpha=1.0, hatch=None):
    """One stacked band, drawn as a post-step filled region."""
    return ax.fill_between(edges, np.append(lo, lo[-1]), np.append(hi, hi[-1]),
                           step='post', facecolor=color, edgecolor='white',
                           linewidth=0.35, alpha=alpha, hatch=hatch,
                           label=label, zorder=1)


def stack_legend(ax, handles, labels, ncol=1, loc='upper right', size=8):
    """Frameless legend for a stack: patches keep their own face colour."""
    ax.legend(handles=handles, labels=labels, prop={'size': size},
              frameon=False, ncol=ncol, loc=loc, handlelength=1.5,
              columnspacing=1.1, handletextpad=0.5)


def _stack_note(ax):
    ax.set_title(STACK_NOTE, fontsize=7, color='0.35', loc='left', pad=4)


def stacked_closure_figure(d, key, label, kind, out):
    """The 'all 9 blocks' curve, drawn as the stack of what makes it up.

    For the observables in ``AUGMENT_WITH_DIAG_BLOCKS`` the four diagonal
    blocks are stacked one by one -- they are positive everywhere, so that
    part of the stack is an ordinary one and its top is exactly the '4
    diagonal blocks' curve of the unstacked figure.  Everywhere else the four
    are stacked as a single band, because the unstacked figure does not draw
    them individually either.  The five interference blocks are then added as
    one signed band, which is the piece that can go either side of zero.
    """
    bins = d.bins(key)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    wid = bins[1] - bins[0]

    u, ue = d.h(['unpol'], key)
    s4, s4e = d.h(DIAG, key)
    s9, s9e = d.h(ALL9, key)
    it, _ite = d.h(INTER, key)

    if key in AUGMENT_WITH_DIAG_BLOCKS:
        comps = [d.h([t], key)[0] for t in DIAG] + [it]
        colors = [DIAG_COLOR[t] for t in DIAG] + [C_INT]
        labels = [BLOCK_LABEL[t] for t in DIAG] + ['5 interference blocks']
    else:
        comps = [s4, it]
        colors = [C_SUM4, C_INT]
        labels = ['4 diagonal blocks', '5 interference blocks']

    bands, pos_top, neg_bot, net = signed_stack(comps)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                             gridspec_kw={'height_ratios': [2.5, 1.4]})
    fig.set_size_inches(7 * 0.75, 6.0)
    fig.subplots_adjust(hspace=0.06)
    for ax in axes:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax = axes[0]
    handles = []
    for (lo, hi), color, lab in zip(bands, colors, labels):
        _stepfill(ax, bins, lo / wid, hi / wid, color)
        handles.append(mpatches.Patch(facecolor=color, edgecolor='white',
                                      linewidth=0.35))
    ax.axhline(0.0, color='gray', lw=0.6, linestyle='dashed', zorder=2)
    ax.hist(x=ctr, weights=net / wid, histtype='step', bins=len(ctr),
            range=(bins[0], bins[-1]), linewidth=LW, color=C_SUM9, zorder=3)
    handles.append(mlines.Line2D([], [], color=C_SUM9, lw=LW))
    labels = labels + ['net: all 9 blocks']
    ax.errorbar(ctr, u / wid, yerr=ue / wid, fmt='o', ms=4, color=C_UNPOL,
                lw=1.0, capsize=2, zorder=4)
    handles.append(mlines.Line2D([], [], color=C_UNPOL, lw=0, marker='o',
                                 ms=4))
    labels = labels + ['unpolarised']

    ax.set_ylabel(r'$d\sigma/dX$ [pb]')
    top = float(np.nanmax(pos_top / wid))
    bot = float(np.nanmin(neg_bot / wid))
    ax.set_ylim(min(bot * 1.35, -0.02 * top),
                top * (1.46 if len(comps) > 2 else 1.34))
    stack_legend(ax, handles, labels, ncol=2, loc='upper right',
                 size=8 if len(comps) > 2 else 9)
    _stack_note(ax)
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
    ok = np.isfinite(r9e) & (r9e < 0.35) & np.isfinite(r4e) & (r4e < 0.35)
    if ok.sum() < 4:
        ok = np.isfinite(r9e) & np.isfinite(r4e)
    lo = np.nanmin(np.concatenate([(r4 - r4e)[ok], (r9 - r9e)[ok]]))
    hi = np.nanmax(np.concatenate([(r4 + r4e)[ok], (r9 + r9e)[ok]]))
    pad = 0.12 * max(hi - lo, 0.05)
    ax.set_ylim(min(lo - pad, 0.96), max(hi + pad, 1.04))
    update_legend(ax, ncol=2, loc='best')

    name = 'closure_%s_stacked' % key
    fig.savefig(os.path.join(out, '%s.pdf' % name), bbox_inches='tight')
    fig.savefig(os.path.join(out, '%s.png' % name), dpi=160,
                bbox_inches='tight')
    plt.close(fig)
    return stack_closure_check(comps, s9)


def stacked_blocks_figure(d, key, label, out):
    """The nine blocks stacked, and the five interference ones stacked alone.

    Upper panel: all nine, in the order (4 diagonal, then the 5 with an I
    index), so the positive pile is built bottom-up out of the four that carry
    the cross section and the interference lands last, on whichever side of
    zero its sign puts it.  Lower panel: the five interference blocks only, on
    their own scale, which is the panel the convention actually matters on --
    (I,I) is negative over half the range and the four single-I blocks are
    consistent with zero, so the stack there is (I,I) and four hairlines.
    """
    bins = d.bins(key)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    wid = bins[1] - bins[0]

    u, ue = d.h(['unpol'], key)
    s9, _s9e = d.h(ALL9, key)
    it, _ite = d.h(INTER, key)

    comps9 = [d.h([t], key)[0] for t in ALL9]
    cols9 = [BLOCK_COLOR[t] for t in ALL9]
    bands9, pos9, neg9, net9 = signed_stack(comps9)

    compsI = [d.h([t], key)[0] for t in INTER]
    colsI = [BLOCK_COLOR[t] for t in INTER]
    bandsI, posI, negI, netI = signed_stack(compsI)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1.7]})
    fig.set_size_inches(7 * 0.75, 6.4)
    fig.subplots_adjust(hspace=0.06)
    for ax in axes:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax = axes[0]
    handles, labels = [], []
    for tag, (lo, hi), color in zip(ALL9, bands9, cols9):
        _stepfill(ax, bins, lo / wid, hi / wid, color)
        handles.append(mpatches.Patch(facecolor=color, edgecolor='white',
                                      linewidth=0.35))
        labels.append(BLOCK_LABEL[tag])
    ax.axhline(0.0, color='gray', lw=0.6, linestyle='dashed', zorder=2)
    ax.hist(x=ctr, weights=net9 / wid, histtype='step', bins=len(ctr),
            range=(bins[0], bins[-1]), linewidth=LW, color=C_SUM9, zorder=3)
    handles.append(mlines.Line2D([], [], color=C_SUM9, lw=LW))
    labels.append('net: all 9')
    ax.errorbar(ctr, u / wid, yerr=ue / wid, fmt='o', ms=3.5, color=C_UNPOL,
                lw=1.0, capsize=2, zorder=4)
    handles.append(mlines.Line2D([], [], color=C_UNPOL, lw=0, marker='o',
                                 ms=3.5))
    labels.append('unpolarised')
    ax.set_ylabel(r'$d\sigma/dX$ [pb]')
    top = float(np.nanmax(pos9 / wid))
    bot = float(np.nanmin(neg9 / wid))
    ax.set_ylim(min(bot * 1.35, -0.03 * top), top * 1.46)
    stack_legend(ax, handles, labels, ncol=3, loc='upper right', size=7)
    _stack_note(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax = axes[1]
    handles, labels = [], []
    for tag, (lo, hi), color in zip(INTER, bandsI, colsI):
        _stepfill(ax, bins, lo / wid, hi / wid, color)
        handles.append(mpatches.Patch(facecolor=color, edgecolor='white',
                                      linewidth=0.35))
        labels.append(BLOCK_LABEL[tag])
    ax.axhline(0.0, color='gray', lw=0.8, linestyle='dashed', zorder=2)
    ax.hist(x=ctr, weights=netI / wid, histtype='step', bins=len(ctr),
            range=(bins[0], bins[-1]), linewidth=LW, color=C_UNPOL, zorder=3)
    handles.append(mlines.Line2D([], [], color=C_UNPOL, lw=LW))
    labels.append('net: all 5')
    ax.set_ylabel(r'interference [pb]')
    ax.set_xlabel(label)
    topI = float(np.nanmax(posI / wid))
    botI = float(np.nanmin(negI / wid))
    span = max(topI - botI, 1e-9)
    ax.set_ylim(botI - 0.10 * span, topI + 0.42 * span)
    stack_legend(ax, handles, labels, ncol=3, loc='upper right', size=7)

    name = 'blocks_%s_stacked' % key
    fig.savefig(os.path.join(out, '%s.pdf' % name), bbox_inches='tight')
    fig.savefig(os.path.join(out, '%s.png' % name), dpi=160,
                bbox_inches='tight')
    plt.close(fig)
    return (stack_closure_check(comps9, s9),
            stack_closure_check(compsI, it))


def stacked_main(d, out):
    """The stacked variants, written next to the unstacked ones."""
    lab = dict((k, l) for k, l, _ in OBS)
    kind = dict((k, kk) for k, _l, kk in OBS)
    L = []
    L.append('Stacked renderings of the interference-closure figures.')
    L.append('')
    L.append('Sign convention (also written on every stacked figure):')
    L.append('  in each bin, contributions with POSITIVE content are stacked')
    L.append('  upward from zero and contributions with NEGATIVE content are')
    L.append('  stacked downward from zero.  A band below the axis is a')
    L.append('  subtraction.  The top of the stack is therefore the sum')
    L.append('  of the POSITIVE contributions only and is NOT the total;')
    L.append('  the total is the algebraic sum of the two piles, drawn as')
    L.append('  an explicit line ("net: ...").  Read the line, not the')
    L.append('  envelope.')
    L.append('')
    L.append('This matters because four of the nine blocks are positive in')
    L.append('every bin while the five interference ones are not: (I,I) is')
    L.append('negative over roughly half the range of C_nn and of dphi_lab,')
    L.append('and its sign change is the physics the figure exists to show.')
    L.append('')
    L.append('Closure of the stack: the net of the stacked bands must equal,')
    L.append('bin by bin, the total curve drawn on the same panel.  It is the')
    L.append('same addition regrouped: the residual is floating point only.')
    L.append('')
    L.append('  %-32s %14s %14s' % ('figure / panel', 'max|net-tot|',
                                    'relative'))
    for key in STACKED_CLOSURE_FOR:
        dev, rel = stacked_closure_figure(d, key, lab[key], kind[key], out)
        L.append('  %-32s %14.3e %14.3e'
                 % ('closure_%s_stacked' % key, dev, rel))
    for key in STACKED_BLOCKS_FOR:
        c9, ci = stacked_blocks_figure(d, key, lab[key], out)
        L.append('  %-32s %14.3e %14.3e'
                 % ('blocks_%s_stacked (9)' % key, c9[0], c9[1]))
        L.append('  %-32s %14.3e %14.3e'
                 % ('blocks_%s_stacked (5 I)' % key, ci[0], ci[1]))
    L.append('  (units are pb; "relative" is against the peak bin of the same')
    L.append('   total.  Machine epsilon on a double is 2.2e-16.)')
    L.append('')
    L.append('How much of the stack sits below zero, per observable: the')
    L.append('ratio of the summed negative pile to the summed positive pile')
    L.append('over all bins, for the 9-block stack and for the 5')
    L.append('interference blocks alone.  The second column is why the')
    L.append('convention is needed.')
    L.append('  %-10s %18s %18s' % ('observable', 'neg/pos (9 blocks)',
                                    'neg/pos (5 inter.)'))
    for key, _l, _k in OBS:
        c9 = [d.h([t], key)[0] for t in ALL9]
        ci = [d.h([t], key)[0] for t in INTER]
        f9 = (sum(np.clip(c, None, 0).sum() for c in c9)
              / sum(np.clip(c, 0, None).sum() for c in c9))
        fi = (sum(np.clip(c, None, 0).sum() for c in ci)
              / sum(np.clip(c, 0, None).sum() for c in ci))
        L.append('  %-10s %18.4f %18.4f' % (key, abs(f9), abs(fi)))
    L.append('')
    path = os.path.join(out, 'stacked_numbers.txt')
    open(path, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('wrote the stacked figures + stacked_numbers.txt to', out)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = set(a for a in sys.argv[1:] if a.startswith('--'))
    ddir, out = argv[0], argv[1]
    os.makedirs(out, exist_ok=True)
    d = Data(ddir)
    if '--stacked-only' in flags:
        # regenerate the stacked variants without touching the existing
        # figures (a PDF carries a timestamp, so rewriting one is a diff)
        stacked_main(d, out)
        return

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
    L.append('Naming: ckk, cnn, crr are the PER-EVENT products')
    L.append('  c_ij = cos(theta^i_l+) cos(theta^j_l-),  one number per event.')
    L.append('  Every histogram and every chi2 below, and every figure, is of')
    L.append('  that product.  The spin-correlation coefficient C_ij is the')
    L.append('  MEAN of it (up to the standard 1/9 of the two decay analysing')
    L.append('  powers), so it is a single number per sample and appears only')
    L.append('  in the "means" table.  A block whose C_ij vanishes still has a')
    L.append('  cross section and a perfectly non-zero histogram: what vanishes')
    L.append('  is the first moment.  The axis labels say "mean -> C_ij" for')
    L.append('  this reason.')
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

    # the identity cos phi_ll = c_kk + c_rr + c_nn, term by term.  It is an
    # identity between the per-event quantities themselves, so it survives the
    # averaging; the means are what is printed.
    L.append('identity  <cos phi_ll> = <ckk> + <crr> + <cnn>   (per-event')
    L.append('  products, whose means give the coefficients C_ij)')
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

    if '--no-stacked' not in flags:
        stacked_main(d, out)


if __name__ == '__main__':
    main()
