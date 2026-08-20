#!/usr/bin/env python3
"""The m(t) / m(tbar) resonance lineshape of every MadSpin spinmode x
unweighting cell, in the MG7 paper's plotting style.

Runs entirely off the committed raw histograms, so nothing here needs MadSpin:

    plot_lineshape.py [--data DIR] [--out DIR]

with ``<data>`` holding ``histograms.npz`` and ``meta.json`` as written by
``run_lineshape.py``.

What is drawn is a PER-EVENT quantity and the axes say so: the invariant mass
sqrt(E^2 - |p|^2) of the status-2 resonance in the decayed LHE record, i.e. the
virtuality MadSpin assigned to that top in that event.  It is not a fitted
width, not a pole parameter and not a mean -- those appear only in RESULTS.md.

Style follows the MG7 paper's ``plotexample/dummyplot.py``: LaTeX text, serif,
base font size 14, step histograms of line width 1.2, the paper's fixed figure
width (7*0.75 in -- "do not change the horizontal size"), tableau colours with
black/blue/red promoted, frameless legends, minor tick locators.
"""

import argparse
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

_HERE = os.path.dirname(os.path.abspath(__file__))


# --- work around a matplotlib bug that eats every minus sign in the PDFs ---
def _fix_type1_subset_minus():
    """Stop matplotlib's usetex PDF path from dropping every minus sign.

    ``_type1font.Type1Font.subset`` in matplotlib 3.11.0 ends its encoding
    filter with an unconditional

        encoding[0] = '.notdef'

    For a text font slot 0 really is ``.notdef``, but TeX's CMSY10 -- the font
    every math minus comes from -- carries ``minus`` there.  Subsetting a
    figure's fonts therefore throws the minus away and the PDF comes out with
    an empty /Differences and no /BaseEncoding, so viewers silently drop the
    sign: no warning, no error, a wrong figure.  Only PDF is affected (the PNGs
    go through dvipng, which rasterises first).

    These figures carry minus signs in every ratio tick label and in the
    ``m - m_t`` secondary axis, so the bug would be visible.  Guarded on the
    exact upstream line, so a fixed matplotlib is left alone.
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


# --------------------------------------------------------------------------
# Which cells exist, and what each one is.
#
# ``kind``:
#   'shape' -- the cell samples a virtuality, so it has a lineshape to draw.
#   'delta' -- the cell assigns the pole mass exactly, in every event.  There is
#              no lineshape; drawing a histogram of it would be a spike one bin
#              wide and would misrepresent it, so these are marked on the figure
#              with a vertical line at the pole instead.
#   'alias' -- the run is bit-identical to another cell (proved in
#              RESULTS.md from the raw histograms, not asserted from the source).
# --------------------------------------------------------------------------
CELLS = [
    # key,                  label,                                family,  kind
    ('madspin_joint',       'madspin / joint',                    'madspin', 'shape'),
    ('madspin_sequential',  'madspin / sequential',               'madspin', 'shape'),
    ('madspin_seqglobal',   'madspin / sequential global retry',  'madspin', 'shape'),
    ('PA_joint',            'PA / joint',                         'PA', 'shape'),
    ('PA_sequential',       'PA / sequential',                    'PA', 'shape'),
    ('PA_seqglobal',        'PA / sequential global retry',       'PA', 'shape'),
    ('PA_seqwithmass',      'PA / sequential with mass',          'PA', 'shape'),
    ('madspin_v1_joint',    'madspin v1 (unweighting inert)',     'v1', 'shape'),
    ('onshell_joint',       'onshell / joint',                    'onshell', 'delta'),
    ('onshell_sequential',  'onshell / sequential',               'onshell', 'delta'),
    ('onshell_seqglobal',   'onshell / sequential global retry',  'onshell', 'delta'),
    ('none_joint',          'none (unweighting inert)',           'none', 'delta'),
    ('PAnojac_joint',       'PA / joint, jacobian off',           'PAnojac', 'shape'),
    ('PAnojac_sequential',  'PA / sequential, jacobian off',      'PAnojac', 'shape'),
    ('PAnojac_seqglobal',   'PA / seq. global retry, jac. off',   'PAnojac', 'shape'),
    ('PAnojac_seqwithmass', 'PA / seq. with mass, jac. off',      'PAnojac', 'shape'),
    ('madspin_joint_rep',   'madspin / joint, replica seed',      'replica', 'shape'),
    ('PA_joint_rep',        'PA / joint, replica seed',           'replica', 'shape'),
]

LABEL = {k: l for k, l, _f, _kd in CELLS}
KIND = {k: kd for k, _l, _f, kd in CELLS}

# The reference every ratio pane divides by.
#
#   * ``madspin`` (== ``full``; run_madspin rewrites the name) is the shipped
#     default spinmode and the only one that takes the resonance shape from the
#     full off-shell matrix element rather than from a fixed-width Breit-Wigner
#     multiplying an on-shell density.  It is therefore the most complete
#     lineshape the code can produce, and the natural absolute standard.
#   * ``joint`` is the historical, unsplit accept/reject: ONE test over the
#     virtualities and every decay at once.  It has no staged decomposition, so
#     no stage has a conditional normalisation to divide out and no tabulated
#     Z-hat factor is needed anywhere.  Every staged scheme is a
#     re-decomposition of exactly this test and is supposed to reproduce it.
#
# So madspin/joint is the cell with the fewest moving parts.  A staged scheme
# that departs from it is a bug in the decomposition; PA and onshell departing
# from it is the size of their physics approximation, which is the other thing
# this figure is for.
REF = 'madspin_joint'

COLOR = {
    'madspin_joint':      'blue',
    'madspin_sequential': allcolors[2],      # tab:green
    'madspin_seqglobal':  allcolors[4],      # purple
    'PA_joint':           'red',
    'PA_sequential':      'tab:orange',
    'PA_seqglobal':       'tab:brown',
    'PA_seqwithmass':     'tab:pink',
    'madspin_v1_joint':   'tab:cyan',
    'PAnojac_joint':      'red',
    'PAnojac_sequential': 'tab:orange',
    'PAnojac_seqglobal':  'tab:brown',
    'PAnojac_seqwithmass': 'tab:pink',
    'madspin_joint_rep':  'gray',
    'PA_joint_rep':       'gray',
}
LS = {
    'madspin_joint': 'solid', 'madspin_sequential': 'dashed',
    'madspin_seqglobal': 'dashdot',
    'PA_joint': 'solid', 'PA_sequential': 'dashed',
    'PA_seqglobal': 'dashdot', 'PA_seqwithmass': 'dotted',
    'madspin_v1_joint': 'solid',
    'PAnojac_joint': 'solid', 'PAnojac_sequential': 'dashed',
    'PAnojac_seqglobal': 'dashdot', 'PAnojac_seqwithmass': 'dotted',
    'madspin_joint_rep': 'dotted', 'PA_joint_rep': 'dotted',
}

XLABEL = {
    't': r'$m(t)$ [GeV] \ \ (per-event $\sqrt{E^2-|\vec{p}|^2}$'
         r' of the intermediate $t$)',
    'tbar': r'$m(\bar t)$ [GeV] \ \ (per-event $\sqrt{E^2-|\vec{p}|^2}$'
            r' of the intermediate $\bar t$)',
}
XLABEL_PLAIN = {
    't': r'$m(t)$ [GeV]  (per-event $\sqrt{E^2-|p|^2}$ of the intermediate $t$)',
    'tbar': r'$m(\bar t)$ [GeV]  (per-event $\sqrt{E^2-|p|^2}$ of the '
            r'intermediate $\bar t$)',
}
YLABEL = r'$\frac{1}{N}\,\mathrm{d}N/\mathrm{d}m$ [GeV$^{-1}$]'
YLABEL_PLAIN = r'$(1/N)\ \mathrm{d}N/\mathrm{d}m$ [GeV$^{-1}$]'


# --------------------------------------------------------------------------
# Rebinning.  ``run_lineshape.py`` writes a fine UNIFORM grid (Gamma/12, 360
# bins over the full +-15 Gamma Breit-Wigner support) so the committed .npz is
# the raw measurement.  Plotting wants something else: a Breit-Wigner falls by
# three decades between the pole and the truncation, so a binning that resolves
# the peak leaves single-digit counts in the tail -- and the tail is exactly
# where a bound problem would show.  Whole numbers of fine bins are therefore
# grouped into three zones, so no bin edge moves and nothing is interpolated:
#
#   |m - M| < 3.5 Gamma      Gamma/6   (2 fine bins)  -- resolves the peak
#   3.5 .. 9 Gamma           Gamma/2   (6 fine bins)  -- the shoulders
#   9 .. 15 Gamma            2 Gamma  (24 fine bins)  -- the far tails
#
# 70 bins in total.  At 2e5 events that keeps every bin above ~300 entries, so
# every per-bin statistical error is below ~6%; the exact per-bin errors are
# drawn on the figures and tabulated in RESULTS.md.
ZONES = [(15.0, 9.0, 24), (9.0, 3.5, 6), (3.5, -3.5, 2),
         (-3.5, -9.0, 6), (-9.0, -15.0, 24)]


def rebin_map(nfine, per_width=12):
    """Group sizes over the fine grid, as a list of (start, stop) index pairs."""
    groups = []
    i = 0
    for lo_n, hi_n, g in ZONES:
        span = int(round((lo_n - hi_n) * per_width))
        for _ in range(span // g):
            groups.append((i, i + g))
            i += g
    assert i == nfine, (i, nfine)
    return groups


def _apply(vec, groups):
    return np.array([vec[a:b].sum() for a, b in groups])


class Data(object):
    """The raw fine-grid histograms, and the rebinned normalised densities."""

    def __init__(self, ddir):
        self.z = np.load(os.path.join(ddir, 'histograms.npz'))
        self.meta = json.load(open(os.path.join(ddir, 'meta.json')))
        fine = self.z['bins']
        self.fine = fine
        self.groups = rebin_map(len(fine) - 1)
        self.edges = np.array([fine[a] for a, _b in self.groups]
                              + [fine[self.groups[-1][1]]])
        self.width = np.diff(self.edges)
        self.centre = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.pole = self.meta['pole']
        self.gamma = self.meta['width']

    def has(self, key, tag='t'):
        return ('sumw__%s__%s' % (key, tag)) in self.z

    def raw(self, key, tag='t'):
        """(sum of weights, sum of squared weights, entries) on the PLOT bins."""
        g = self.groups
        return (_apply(self.z['sumw__%s__%s' % (key, tag)], g),
                _apply(self.z['sumw2__%s__%s' % (key, tag)], g),
                _apply(self.z['n__%s__%s' % (key, tag)], g))

    def density(self, key, tag='t'):
        """(p, dp) -- the unit-area density (1/N) dN/dm and its MC error.

        The normalisation is the sum over the plotted bins; the runs put nothing
        outside them (meta.json records the out-of-range counts, all zero), so
        that is the whole sample.  The error is sqrt(sum w^2)/(W * bin width):
        the normalisation's own error is correlated with the bin's, is a
        1/sqrt(N) effect on an O(1) quantity, and cancels to that order in the
        ratio of two densities -- so it is left out, and the replica rows are
        there to show the resulting floor empirically.
        """
        w, w2, _n = self.raw(key, tag)
        norm = w.sum()
        return w / (norm * self.width), np.sqrt(w2) / (norm * self.width)

    def bw(self):
        """Per-bin Breit-Wigner density, integrated EXACTLY over each bin.

        MadSpin draws the virtuality from a fixed-width relativistic
        Breit-Wigner in m^2 (``lhe_parser.Event.generate_random_mass``):

            dN/dm^2  ~  1 / ((m^2 - M^2)^2 + M^2 Gamma^2)

        truncated to |m - M| < BW_cut * Gamma with BW_cut = 15, the value the
        ``BW_cut = -1`` card default resolves to in ``_draw_mass_value``.  The
        primitive is atan((m^2 - M^2)/(M Gamma))/(M Gamma), so a bin's
        probability is an exact difference of arctangents -- no midpoint
        approximation, which matters because the core bins are narrower than
        the peak is sharp.

        M and Gamma are the pole and width the run actually used, read out of
        the production banner's param_card; nothing is assumed or fitted.

        This is an absolute standard to read the figure against, NOT a
        prediction.  What is accepted is the Breit-Wigner times the decay
        matrix element and the phase space, so the measured curve is expected
        to lie near this line and not on it.
        """
        M, G = self.pole, self.gamma
        def F(m):
            return np.arctan((m ** 2 - M ** 2) / (M * G))
        lo, hi = self.meta['bw_range']
        gap = F(hi) - F(lo)
        frac = (F(self.edges[1:]) - F(self.edges[:-1])) / gap
        return frac / self.width


def ratio(num, nume, den, dene):
    r = num / np.where(den == 0, np.nan, den)
    re = np.abs(r) * np.sqrt(
        (nume / np.where(num == 0, np.nan, num)) ** 2
        + (dene / np.where(den == 0, np.nan, den)) ** 2)
    re = np.where(np.isfinite(re), re, nume / np.where(den == 0, np.nan, den))
    return r, re


def chi2(num, nume, den, dene, ndof_penalty=1):
    """chi2 and dof of two normalised densities.

    Both are unit-area, so one linear constraint is shared between them and the
    bins are (weakly) correlated through it: dof = nbins - 1, and the off
    diagonal terms of the covariance are dropped.  This makes the chi2 slightly
    conservative; the replica rows in RESULTS.md are the empirical noise floor
    that it should be judged against rather than against the nominal 1.0.
    """
    d = num - den
    s2 = nume ** 2 + dene ** 2
    good = (s2 > 0) & np.isfinite(d)
    return float((d[good] ** 2 / s2[good]).sum()), int(good.sum()) - ndof_penalty


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
              ncol=ncol, loc=loc, handlelength=2.2, columnspacing=1.2)


def lineshape_figure(d, keys, tag, out, name, title=None, ref=REF,
                     show_delta=False, ncol=2):
    """Main pane: the normalised lineshapes plus the Breit-Wigner standard.
    Ratio pane: every cell divided by ``ref``.
    """
    edges, ctr = d.edges, d.centre
    lab = XLABEL[tag] if USETEX else XLABEL_PLAIN[tag]
    ylab = YLABEL if USETEX else YLABEL_PLAIN

    rp, rpe = d.density(ref, tag)
    bw = d.bw()

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                             gridspec_kw={'height_ratios': [2.3, 1.5]})
    fig.set_size_inches(7 * 0.75, 6.6)
    fig.subplots_adjust(hspace=0.06)
    for ax in axes:
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    # ---- main pane -------------------------------------------------------
    ax = axes[0]
    ax.set_yscale('log')
    ax.plot(ctr, bw, color='black', lw=1.4, linestyle=(0, (6, 3)), zorder=1,
            label=(r'Breit-Wigner, $M=%.2f$, $\Gamma=%.4f$ GeV'
                   % (d.pole, d.gamma)))
    for key in keys:
        p, pe = d.density(key, tag)
        ax.errorbar(ctr, p, yerr=pe, fmt='none', ecolor=COLOR[key],
                    elinewidth=0.8, capsize=1.2, zorder=3)
        ax.step(edges, np.append(p, p[-1]), where='post', color=COLOR[key],
                lw=LW, linestyle=LS[key], label=LABEL[key], zorder=3)
    if show_delta:
        # onshell and none assign the pole mass in every event.  Drawing a
        # histogram of a delta would be a one-bin spike whose height depends
        # only on the binning, which would misrepresent it; the vertical line
        # says what they actually are.
        # ymax keeps the line clear of the (frameless) legend, which it would
        # otherwise be drawn straight through
        ax.axvline(d.pole, color='dimgray', lw=1.0, linestyle=(0, (1, 2)),
                   zorder=0, ymax=0.68, label=DELTA_NOTE)
    ax.set_ylabel(ylab)
    ax.set_ylim(np.nanmin(bw[bw > 0]) * 0.30,
                np.nanmax(bw) * 10 ** (2.3 if ncol > 1 else 1.7))
    update_legend(ax, ncol=ncol, loc='upper left', size=7.5)
    plt.setp(ax.get_xticklabels(), visible=False)
    if title:
        ax.set_title(title, fontsize=10.5)

    # ---- ratio pane ------------------------------------------------------
    ax = axes[1]
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.axhline(1.0, color='gray', lw=0.8, linestyle='dashed')
    rb, _ = ratio(bw, np.zeros_like(bw), rp, rpe)
    ax.plot(ctr, rb, color='black', lw=1.2, linestyle=(0, (6, 3)), zorder=1)
    span = []
    others = [k for k in keys if k != ref]
    for i, key in enumerate(others):
        p, pe = d.density(key, tag)
        r, re = ratio(p, pe, rp, rpe)
        # a small horizontal offset per series, so overlapping error bars stay
        # readable; it is cosmetic and never more than a fifth of a bin
        off = (i - (len(others) - 1) / 2.0) * (0.16 / max(len(others), 1))
        ax.errorbar(ctr + off * d.width, r, yerr=re, fmt='o', ms=2.6,
                    color=COLOR[key], lw=0.9, capsize=1.2, zorder=3)
        ok = np.isfinite(r) & np.isfinite(re)
        if ok.any():
            span += [float((r - re)[ok].min()), float((r + re)[ok].max())]
    # the Breit-Wigner is a standard, not a measurement: it is allowed to set
    # the range only where it stays within a factor of a few, so a single
    # far-tail excursion cannot compress everything else into a line
    keep = np.isfinite(rb) & (rb > 0.2) & (rb < 4.0)
    if keep.any():
        span += [float(rb[keep].min()), float(rb[keep].max())]
    lo, hi = min(span), max(span)
    pad = 0.08 * max(hi - lo, 0.04)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel('ratio to %s' % LABEL[ref], fontsize=9.5)
    ax.set_xlabel(lab, fontsize=10.5)

    paths = []
    for ext, kw in (('pdf', {}), ('png', {'dpi': 160})):
        pth = os.path.join(out, '%s.%s' % (name, ext))
        fig.savefig(pth, bbox_inches='tight', **kw)
        paths.append(pth)
    plt.close(fig)
    return paths


FAMILIES = [
    ('all', 't', 'lineshape_mt_all',
     ['madspin_joint', 'madspin_sequential', 'madspin_seqglobal',
      'PA_joint', 'PA_sequential', 'PA_seqglobal', 'PA_seqwithmass',
      'madspin_v1_joint'],
     r'MadSpin $m(t)$ lineshape: every distinct spinmode / unweighting cell'),
    ('all', 'tbar', 'lineshape_mtbar_all',
     ['madspin_joint', 'madspin_sequential', 'madspin_seqglobal',
      'PA_joint', 'PA_sequential', 'PA_seqglobal', 'PA_seqwithmass',
      'madspin_v1_joint'],
     r'MadSpin $m(\bar t)$ lineshape: every distinct spinmode / unweighting cell'),
    ('madspin', 't', 'lineshape_mt_madspin',
     ['madspin_joint', 'madspin_sequential', 'madspin_seqglobal',
      'madspin_joint_rep'],
     'spinmode = madspin: the three unweighting schemes, plus a replica seed'),
    ('PA', 't', 'lineshape_mt_PA',
     ['PA_joint', 'PA_sequential', 'PA_seqglobal', 'PA_seqwithmass',
      'PA_joint_rep'],
     'spinmode = PA: the four unweighting schemes, plus a replica seed'),
    ('nojac', 't', 'lineshape_mt_nojac',
     ['PA_joint', 'PAnojac_joint', 'PAnojac_sequential', 'PAnojac_seqglobal',
      'PAnojac_seqwithmass'],
     'spinmode = PA: density keep jacobian on (reference) vs off'),
]

# onshell (all three schemes) and none put every event at the pole mass, to
# machine precision -- verified run by run in RESULTS.md, not assumed.
DELTA_NOTE = (r'onshell ($\times3$), none: $\delta(m-M)$, no lineshape'
              if USETEX else
              r'onshell ($\times$3), none: $\delta(m-M)$, no lineshape')


# --------------------------------------------------------------------------
# The numbers.  Everything below is computed from the same raw histograms the
# figures are drawn from, plus the unbinned moments run_lineshape.py stored, so
# nothing is read back off a plot.
# --------------------------------------------------------------------------
SHAPE_CELLS = ['madspin_joint', 'madspin_sequential', 'madspin_seqglobal',
               'PA_joint', 'PA_sequential', 'PA_seqglobal', 'PA_seqwithmass',
               'PAnojac_joint', 'PAnojac_sequential', 'PAnojac_seqglobal',
               'PAnojac_seqwithmass', 'madspin_v1_joint',
               'madspin_joint_rep', 'PA_joint_rep']
DELTA_CELLS = ['onshell_joint', 'onshell_sequential', 'onshell_seqglobal',
               'none_joint']

# Every cell also gets compared to its OWN family's joint scheme, not only to
# the global reference.  The two questions are different and must not be
# conflated:
#   vs the global reference (madspin/joint) -- how far this cell's lineshape is
#     from the most complete one the code can produce.  For a PA cell most of
#     that distance is the pole approximation itself, which is physics, not a
#     defect.
#   vs the family reference -- whether the accept/reject scheme changed the
#     lineshape *within* one spinmode.  This is the closure test: these cells
#     are supposed to sample the same distribution, so any deviation here is a
#     bug in the decomposition and nothing else.
# The replica rows carry the family reference of the cell they replicate, so
# each family's noise floor is read off the same column as its closure test.
FAMILY_REF = {
    'madspin_joint': 'madspin_joint',
    'madspin_sequential': 'madspin_joint',
    'madspin_seqglobal': 'madspin_joint',
    'madspin_joint_rep': 'madspin_joint',
    'madspin_v1_joint': 'madspin_joint',      # no family peer; v1 has one cell
    'PA_joint': 'PA_joint',
    'PA_sequential': 'PA_joint',
    'PA_seqglobal': 'PA_joint',
    'PA_seqwithmass': 'PA_joint',
    'PA_joint_rep': 'PA_joint',
    'PAnojac_joint': 'PAnojac_joint',
    'PAnojac_sequential': 'PAnojac_joint',
    'PAnojac_seqglobal': 'PAnojac_joint',
    'PAnojac_seqwithmass': 'PAnojac_joint',
}
# (run, the run it should be bit-identical to, why)
DEGENERATE = [
    ('full_joint', 'madspin_joint',
     "run_madspin rewrites spinmode 'full' to 'madspin' before anything "
     "else reads it"),
    ('madspin_seqwithmass', 'madspin_sequential',
     'sequential_with_mass needs a per-particle mass draw (PA); the offshell '
     'spinmodes fall back to sequential'),
    ('madspin_v1_sequential', 'madspin_v1_joint',
     'madspin_v1 has no density matrix, so the unweighting card entry is '
     'inert'),
    ('none_sequential', 'none_joint',
     'spinmode=none has no density matrix, so the unweighting card entry is '
     'inert'),
]


def moments(meta, key, tag):
    """(mean, err, rms, err) of m, unbinned, from the stored raw moments."""
    mm = meta['runs'][key]['%s_moments' % tag]
    s0, s1, s2, sq0 = mm['s0'], mm['s1'], mm['s2'], mm['sq0']
    mean = s1 / s0
    var = max(s2 / s0 - mean * mean, 0.0)
    neff = s0 * s0 / sq0 if sq0 > 0 else 0.0
    dmean = math.sqrt(var / neff) if neff > 0 else float('nan')
    rms = math.sqrt(var)
    # The error on the rms is NOT rms/sqrt(2n): that is the gaussian formula,
    # and this distribution is a truncated Breit-Wigner whose fourth central
    # moment is completely tail-dominated -- mu4/mu2^2 is of order 10 here, not
    # 3.  The general result se(s)^2 = (mu4 - mu2^2)/(4 mu2 n) is used instead,
    # with mu4 taken off the fine grid (Gamma/12 bins, far finer than the
    # structure being integrated).  Using the gaussian formula would have made
    # ordinary width fluctuations look like 5-sigma effects.
    return mean, dmean, rms, None, neff


def rms_with_error(d, key, tag):
    """(rms, its error) of m, using the true fourth central moment."""
    w = d.z['sumw__%s__%s' % (key, tag)]
    c = 0.5 * (d.fine[1:] + d.fine[:-1])
    n = float(d.z['n__%s__%s' % (key, tag)].sum())
    tot = w.sum()
    mean = float((w * c).sum() / tot)
    dev = c - mean
    mu2 = float((w * dev ** 2).sum() / tot)
    mu4 = float((w * dev ** 4).sum() / tot)
    rms = math.sqrt(mu2)
    var_s2 = (mu4 - mu2 ** 2) / n
    drms = math.sqrt(max(var_s2, 0.0)) / (2 * rms) if rms > 0 else float('nan')
    return rms, drms, mu4 / (mu2 ** 2) if mu2 > 0 else float('nan')


def core_fraction(d, key, tag, nwidth):
    """Weight fraction within +- nwidth*Gamma of the pole, off the fine grid,
    with its MC error.  A binning-free shape number to sit next to the rms,
    which a truncated Breit-Wigner's tails dominate."""
    w = d.z['sumw__%s__%s' % (key, tag)]
    w2 = d.z['sumw2__%s__%s' % (key, tag)]
    c = 0.5 * (d.fine[1:] + d.fine[:-1])
    sel = np.abs(c - d.pole) < nwidth * d.gamma
    tot = w.sum()
    f = w[sel].sum() / tot
    df = math.sqrt(w2[sel].sum()) / tot
    return f, df


def bw_mean(d):
    """Mean and rms of m under the exact truncated Breit-Wigner, by numerical
    quadrature on the fine grid.  Quoted so the measured means have an absolute
    number to sit next to and not only each other."""
    M, G = d.pole, d.gamma
    c = 0.5 * (d.fine[1:] + d.fine[:-1])
    def F(m):
        return np.arctan((m ** 2 - M ** 2) / (M * G))
    lo, hi = d.meta['bw_range']
    w = (F(d.fine[1:]) - F(d.fine[:-1])) / (F(hi) - F(lo))
    mean = float((w * c).sum())
    rms = math.sqrt(float((w * c * c).sum()) - mean * mean)
    return mean, rms


def write_numbers(d, out):
    L = []
    m = d.meta
    L.append('MadSpin m(t) / m(tbar) lineshape across spinmode x unweighting')
    L.append('  %s, both tops decayed (fully leptonic), 13 TeV LO'
             % m['process'])
    L.append('  %s' % '; '.join(m['decays']))
    L.append('  %d production events, madevent+MadSpin seed %d, code %s'
             % (m['nevents_requested'], m['seed'], m['code_sha'][:12]))
    L.append('  pole M = %.6f GeV, width Gamma = %.6f GeV (production banner '
             'param_card)' % (d.pole, d.gamma))
    L.append('  Breit-Wigner support |m-M| < 15 Gamma = [%.4f, %.4f] GeV'
             % tuple(m['bw_range']))
    L.append('')
    L.append('OBSERVABLE: the per-event invariant mass sqrt(E^2-|p|^2) of the')
    L.append('  status-2 resonance in the decayed LHE.  Not a fitted width,')
    L.append('  not a pole parameter, not a mean -- the means below are means')
    L.append('  OF this quantity and are single numbers, the histograms are of')
    L.append('  the quantity itself.')
    L.append('')

    L.append('cells that collapse onto another cell (checked bin by bin on the')
    L.append('  raw fine grid, not asserted from the source)')
    for a, b, why in DEGENERATE:
        if not (d.has(a) and d.has(b)):
            continue
        same = all(np.array_equal(d.z['sumw__%s__%s' % (a, t)],
                                  d.z['sumw__%s__%s' % (b, t)])
                   for t in ('t', 'tbar'))
        L.append('  %-22s == %-20s : %s' % (a, b, 'IDENTICAL' if same
                                            else '*** DIFFERENT ***'))
        L.append('      %s' % why)
        L.append('      reported unweighting: %s -> %s'
                 % (m['runs'][a]['unweighting_asked'],
                    m['runs'][a]['reported_mode']))
    L.append('')

    L.append('cells with no lineshape at all: every event at the pole mass')
    for k in DELTA_CELLS:
        if not d.has(k):
            continue
        r = m['runs'][k]
        L.append('  %-22s m(t) in [%.10f, %.10f]  m(tbar) in [%.10f, %.10f]'
                 % (k, r['t_moments']['mmin'], r['t_moments']['mmax'],
                    r['tbar_moments']['mmin'], r['tbar_moments']['mmax']))
    L.append('  -> onshell samples no virtuality (_density_do_reshuffle is')
    L.append('     False, slot_mass comes back empty) and spinmode=none does')
    L.append('     not smear the resonance either.  There is nothing to put on')
    L.append('     a lineshape plot for these four.')
    L.append('')

    for tag in ('t', 'tbar'):
        L.append('=== m(%s) ===' % ('t' if tag == 't' else 'tbar'))
        L.append('  %-22s %9s %14s %10s %10s %9s %9s'
                 % ('cell', 'N', 'mean [GeV]', 'rms [GeV]',
                    'f(|dm|<G/2)', 'f(<3G)', 'sigma [pb]'))
        for k in SHAPE_CELLS:
            if not d.has(k, tag):
                continue
            mean, dmean, _r, _dr, _neff = moments(m, k, tag)
            rms, drms, _kurt = rms_with_error(d, k, tag)
            f1, df1 = core_fraction(d, k, tag, 0.5)
            f3, df3 = core_fraction(d, k, tag, 3.0)
            L.append('  %-22s %9d %8.4f+-%-.4f %6.4f+-%-.4f %6.4f+-%-.4f '
                     '%6.4f+-%-.4f %9.5f'
                     % (k, m['runs'][k]['nevents'], mean, dmean, rms, drms,
                        f1, df1, f3, df3, m['runs'][k]['cross_out'] or 0.0))
        L.append('')
        L.append('  per-bin chi2 on the %d plot bins.  Both entries of every'
                 % len(d.centre))
        L.append('  pair are unit-area densities, so one linear constraint is')
        L.append('  shared: dof = nbins - 1.  "vs BW" is against the exact')
        L.append('  truncated Breit-Wigner at the run\'s own M and Gamma,')
        L.append('  which is a standard and NOT a prediction: the accepted')
        L.append('  lineshape is that Breit-Wigner times the decay matrix')
        L.append('  element and phase space.')
        L.append('  The two "replica seed" rows are the SAME cell as the row')
        L.append('  above them, run off the SAME production events with a')
        L.append('  different MadSpin seed.  They are the noise floor: no')
        L.append('  chi2/dof at or below theirs means anything.')
        rp, rpe = d.density(REF, tag)
        bw = d.bw()
        zeros = np.zeros_like(bw)
        L.append('  %-22s %13s %13s %13s %12s %10s'
                 % ('cell', 'vs madspin/', 'vs its own', 'vs Breit-', 'family',
                    'largest'))
        L.append('  %-22s %13s %13s %13s %12s %10s'
                 % ('', 'joint', 'family joint', 'Wigner', 'reference',
                    'pull (fam)'))
        for k in SHAPE_CELLS:
            if not d.has(k, tag):
                continue
            p, pe = d.density(k, tag)
            c, n = chi2(p, pe, rp, rpe)
            cb, nb = chi2(p, pe, bw, zeros)
            fam = FAMILY_REF.get(k, REF)
            fp, fpe = d.density(fam, tag)
            cf, nf = chi2(p, pe, fp, fpe)
            if k == fam:
                pull_s, at = 0.0, float('nan')
            else:
                sig = np.sqrt(pe ** 2 + fpe ** 2)
                pull = np.where(sig > 0, (p - fp) / np.where(sig > 0, sig, 1),
                                0.0)
                j = int(np.argmax(np.abs(pull)))
                pull_s, at = float(pull[j]), float(d.centre[j])
            L.append('  %-22s %8.1f /%-3d %8.1f /%-3d %8.1f /%-3d %12s '
                     '%+6.2f@%.0f'
                     % (k, c, n, cf, nf, cb, nb, fam.replace('_joint', ''),
                        pull_s, at))
        L.append('  sqrt(2*dof) = %.1f, so a chi2 that many above %d is 1 sigma.'
                 % (math.sqrt(2 * (len(d.centre) - 1)), len(d.centre) - 1))
        L.append('')

        # -- the means, differenced against the family reference ------------
        bwm, bwr = bw_mean(d)
        L.append('  mean m shifted against the family reference, in MeV.  The')
        L.append('  quoted error treats the two samples as independent, which')
        L.append('  they are not (they share the production events), so read')
        L.append('  it against the replica row rather than against 1 sigma:')
        L.append('  the replica is the same scheme with a different MadSpin')
        L.append('  seed, so whatever shift it shows is what a shift costs.')
        L.append('  For scale, the exact truncated Breit-Wigner has')
        L.append('  mean = %.4f GeV and rms = %.4f GeV.' % (bwm, bwr))
        L.append('  %-22s %14s %12s %10s'
                 % ('cell', 'mean-ref [MeV]', 'family ref', 'n sigma'))
        for k in SHAPE_CELLS:
            if not d.has(k, tag):
                continue
            fam = FAMILY_REF.get(k, REF)
            if k == fam:
                continue
            mk, dk, _r, _dr, _n = moments(m, k, tag)
            mf, df, _r, _dr, _n = moments(m, fam, tag)
            diff = (mk - mf) * 1000.0
            err = math.sqrt(dk ** 2 + df ** 2) * 1000.0
            L.append('  %-22s %8.1f +- %-4.1f %12s %9.1f'
                     % (k, diff, err, fam.replace('_joint', ''),
                        diff / err if err else float('nan')))
        L.append('')

    # ---- the decisive test on the means -------------------------------
    # t and tbar are two independent virtuality draws in the same event, so a
    # scheme that really biased the lineshape would move BOTH the same way.  A
    # shift that appears in one and not the other -- or with the opposite sign
    # -- is a fluctuation, and with 11 comparisons per resonance some of them
    # will look like 3 sigma on one resonance alone.  Averaging the two halves
    # the error and is the number to read.
    L.append('mean shift against the family reference, m(t) and m(tbar)')
    L.append('  separately and combined [MeV].  A scheme that biased the')
    L.append('  lineshape moves both resonances the same way; a shift in one')
    L.append('  only is a fluctuation, and with 11 comparisons per resonance')
    L.append('  there will be some.  The combined column is the one to read.')
    bwm, _bwr = bw_mean(d)
    L.append('  (exact truncated Breit-Wigner mean = %.4f GeV)' % bwm)
    L.append('  %-22s %14s %14s %16s %8s'
             % ('cell', 'm(t)', 'm(tbar)', 'combined', 'n sigma'))
    for k in SHAPE_CELLS:
        fam = FAMILY_REF.get(k, REF)
        if k == fam or not d.has(k):
            continue
        row = []
        for tag in ('t', 'tbar'):
            mk, dk, _r, _dr, _n = moments(m, k, tag)
            mf, df, _r, _dr, _n = moments(m, fam, tag)
            row.append(((mk - mf) * 1000.0,
                        math.sqrt(dk ** 2 + df ** 2) * 1000.0))
        comb = 0.5 * (row[0][0] + row[1][0])
        combe = 0.5 * math.sqrt(row[0][1] ** 2 + row[1][1] ** 2)
        L.append('  %-22s %7.1f+-%-5.1f %7.1f+-%-5.1f %8.1f+-%-6.1f %8.1f'
                 % (k, row[0][0], row[0][1], row[1][0], row[1][1],
                    comb, combe, comb / combe if combe else float('nan')))
    L.append('')

    # ---- and the same for the shape test ------------------------------
    L.append('closure chi2 against the family reference, m(t) and m(tbar) and')
    L.append('  the two summed (140 bins, 138 dof).  Read every row against')
    L.append('  the replica row of its own family at the bottom: that is the')
    L.append('  same scheme with a different MadSpin seed, so it is what')
    L.append('  agreement costs, and 1 sigma on a chi2 of 138 dof is 16.6.')
    L.append('  %-22s %12s %12s %16s %9s'
             % ('cell', 'm(t)', 'm(tbar)', 'combined', 'n sigma'))
    for k in SHAPE_CELLS:
        fam = FAMILY_REF.get(k, REF)
        if k == fam or not d.has(k):
            continue
        tot, dof = 0.0, 0
        each = []
        for tag in ('t', 'tbar'):
            p_, pe_ = d.density(k, tag)
            f_, fe_ = d.density(fam, tag)
            c, n = chi2(p_, pe_, f_, fe_)
            each.append((c, n))
            tot += c
            dof += n
        L.append('  %-22s %7.1f /%-3d %7.1f /%-3d %9.1f /%-4d %9.1f'
                 % (k, each[0][0], each[0][1], each[1][0], each[1][1],
                    tot, dof, (tot - dof) / math.sqrt(2.0 * dof)))
    L.append('')

    # ---- the differences that are physics, not closure -----------------
    # These three pairs differ by something the code does on purpose, so a
    # non-zero answer here is the size of that choice and not a defect.  They
    # are separated from the closure table above so the two are never read as
    # the same kind of number.
    PHYSICS_PAIRS = [
        ('PA_joint', 'madspin_joint',
         'the pole approximation: PA evaluates the production density at '
         'onshell momenta, madspin at the reshuffled offshell ones'),
        ('PAnojac_joint', 'PA_joint',
         'density_keep_jacobian = False: the reshuffling jacobian is left out '
         'of the accept/reject weight and applied afterwards as a dressing'),
        ('madspin_v1_joint', 'madspin_joint',
         'the pre-density implementation: different mass smearing and a '
         'factorised on-shell branching ratio'),
    ]
    L.append('differences that are a deliberate physics choice, not a closure')
    L.append('  failure.  Same statistics, same reference machinery as above.')
    L.append('  %-22s %-18s %14s %14s'
             % ('cell', 'against', 'combined chi2', 'mean shift'))
    for k, base, why in PHYSICS_PAIRS:
        if not (d.has(k) and d.has(base)):
            continue
        tot, dof, shift, serr = 0.0, 0, 0.0, 0.0
        for tag in ('t', 'tbar'):
            p_, pe_ = d.density(k, tag)
            f_, fe_ = d.density(base, tag)
            c, n = chi2(p_, pe_, f_, fe_)
            tot += c
            dof += n
            mk, dk, _r, _dr, _n = moments(m, k, tag)
            mf, df, _r, _dr, _n = moments(m, base, tag)
            shift += 0.5 * (mk - mf) * 1000.0
            serr += 0.25 * (dk ** 2 + df ** 2) * 1e6
        L.append('  %-22s %-18s %8.1f /%-4d %7.1f+-%-5.1f MeV'
                 % (k, base, tot, dof, shift, math.sqrt(serr)))
        L.append('      %s' % why)
    L.append('')

    L.append('rates (decayed-LHE banner cross section, and the branching ratio')
    L.append('  MadSpin reports).  These are NOT normalised away on the')
    L.append('  figures -- the figures show unit-area densities, so a rate')
    L.append('  difference would not appear there and is quoted here instead.')
    L.append('  %-22s %12s %12s %10s %9s'
             % ('cell', 'sigma [pb]', 'BR', 'unw. eff.', 'overflows'))
    for k in SHAPE_CELLS + DELTA_CELLS:
        if k not in m['runs']:
            continue
        r = m['runs'][k]
        L.append('  %-22s %12.6f %12.8f %10.4f %9d'
                 % (k, r['cross_out'] or 0.0, r['BR'] or 0.0,
                    r['efficiency'] or 0.0, r['overflows']))
    L.append('  sigma(production, banner) = %s pb' % m['cross_in'])
    L.append('')
    L.append('out-of-range: events whose virtuality fell outside the +-15 Gamma')
    L.append('  grid (there should be none: that is the range MadSpin samples)')
    for k in SHAPE_CELLS:
        if k not in m['runs']:
            continue
        r = m['runs'][k]
        tot = sum(r['%s_moments' % t][s] for t in ('t', 'tbar')
                  for s in ('under_n', 'over_n'))
        if tot:
            L.append('  %-22s %d' % (k, tot))
    L.append('  (nothing listed = none, in any cell)')

    txt = '\n'.join(L) + '\n'
    open(os.path.join(out, 'lineshape_numbers.txt'), 'w').write(txt)
    return txt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = Data(args.data)
    written = []
    for fam, tag, name, keys, title in FAMILIES:
        keys = [k for k in keys if d.has(k, tag)]
        ref = REF if REF in keys else keys[0]
        written += lineshape_figure(
            d, keys, tag, args.out, name, title=title, ref=ref,
            show_delta=(fam == 'all'),
            ncol=2 if len(keys) > 4 else 1)
    print(write_numbers(d, args.out))
    written.append(os.path.join(args.out, 'lineshape_numbers.txt'))
    for p in written:
        print(p)
    print('\n%d files in %s' % (len(written), args.out))


if __name__ == '__main__':
    main()
