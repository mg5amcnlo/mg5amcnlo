#!/usr/bin/env python3
r"""Variations ``G``--``N``: variation ``F`` scanned over the Wilson coefficient.

Standalone and re-runnable; it draws all eight variations in both renderings in
one go::

    plot_smeft_fig5_ctg_scan.py [--data DIR] [--out DIR] [--out-user DIR]
                                [--nbins N] [--style {mg7,user,both}]
                                [--only TAG[,TAG...]]

Everything comes from ``data/histograms.npz`` and ``data/meta.json``: the
histogram file is the ONLY input and nothing about the samples is hard-coded
here, so the regenerated ``sm_nlo`` (task T123) was a re-run of this script and
not a rewrite of it.  Data loading, rebinning, error propagation and
``check_normalisation()`` come from ``plot_smeft_fig5.py``; the user-style
constants from ``plot_smeft_fig5_userstyle.py``; ``_step`` and the SM NLO
provenance note from ``plot_smeft_fig5_varE.py``; the pane layout from
``plot_smeft_fig5_varF.py``.  None of those four is modified -- this script
only adds ``G``--``N``.


What was asked, and what is actually possible
=============================================
The request was for eight figures: pane 2 at ``c_tG`` in
``{-1, +1, +10, -10}``, crossed with two versions of pane 1, one at ``c_tG =
-1`` and one at ``+1``, with the upper pane identical throughout.

The upper pane is indeed identical throughout, and for the reason expected: the
interference is linear in ``c_tG``, so both ``dsigma_int`` and its integral
scale by the same factor and the unit-area shape ``n_int`` is exactly invariant
-- sign included, because a negative overall factor cancels between numerator
and denominator.  ``check_ctg_invariance()`` measures this rather than assuming
it; it comes out at the last bit of double precision.

But that same cancellation kills the c_tG dependence of **variation F's pane 1
as well**.  ``F``'s pane 1 draws ``n_NLO/n_LO`` and ``n_int/n_LO``: ratios of
unit-area shapes.  ``n_int`` is c_tG-invariant, ``n_LO`` and ``n_NLO`` do not
contain ``c_tG`` at all, so **F's pane 1 is the same picture at every non-zero
value of the coefficient** -- not merely at ``+1`` and ``-1``, but everywhere.
Drawing it twice would have shipped one figure under two names.

The c_tG dependence of this figure lives in exactly one place: the *relative
weight* of the interference against the SM,

    w_SM(c_tG) = sigma_int(c_tG) / sigma_SM ,

which is linear in ``c_tG`` and carries its sign.  Any curve that contains ``w``
scans; any curve that does not, does not.


The convention, stated once and used in both ratio panes
========================================================
The interference enters **with the physical sign and magnitude it has at the
stated c_tG**, through ``w``.  It is never re-normalised to positive unit area
in a ratio pane.  ``n_int`` is unit-area *in the upper pane only*, where the
sign of its own integral divides out; that sign is not lost, it is moved to the
one place it changes an answer and is carried there explicitly by ``w``.

Two ways of forming a ratio satisfy that, and the eight figures are the four
coefficient values in each of them.  Within one figure both ratio panes use the
same one:

``rate``  (variations ``K``--``N``)
    Every ratio is taken between **unnormalised** differential cross sections::

        pane 1:  dsigma_NLO/dsigma_LO  =  K * (n_NLO/n_LO),  K = sigma_NLO/sigma_LO
                 dsigma_int/dsigma_LO  =  w_LO(c_tG) * (n_int/n_LO)
        pane 2:  (dsigma_SM + dsigma_int)/dsigma_SM  =  1 + w_SM(c_tG) * rho_SM

    This is the honest ratio: it includes the change the operator makes to the
    rate, it has no denominator that can change sign, and it is the convention
    in which ``c_tG = +1`` **mirrors ``c_tG = -1`` exactly about the
    no-interference line** -- ``w`` flips sign and nothing else moves.  Pane 1
    scans here, so ``K`` and ``L`` are the two genuinely different pane-1
    versions that were asked for.

``shape`` (variations ``G``--``J``)
    Every ratio is taken between **unit-area** curves, which is what the upper
    pane's normalisation implies::

        pane 1:  n_NLO/n_LO,  n_int/n_LO          (c_tG-invariant, see above)
        pane 2:  n_(SM+int)/n_SM  =  1 + [w/(1+w)] * (rho_SM - 1)

    This is variation ``F``'s pane 2 exactly, and it answers a different
    question: does the *shape* move, with the rate change divided out.  It is
    the convention that has the sign trap, and the figures are drawn so the
    trap can be seen: its denominator is ``1 + w``, which **vanishes and then
    changes sign** at

        c_tG = +3.635 (LO)   and   c_tG = +5.522 (NLO)

    -- the coefficient values at which the interference exactly cancels the SM
    total cross section.  ``c_tG = +10`` is past both poles, so variation
    ``I``'s two pane-2 curves are ratios of two *negative* densities.  They come
    out positive and near 1 and look perfectly innocuous.  They are not: see
    ``numbers_I.txt``, which says so at the top and at the bottom, and the rate
    figure ``M`` drawn at the same coefficient, where the same physics is a pair
    of curves lying entirely below zero.

Both conventions use the same error identity, the one that keeps the shared
``n_SM`` from being counted twice.  Writing either pane-2 curve as

    baseline + k * (rho_SM - 1),      rho_SM = n_int/n_SM,

with ``(baseline, k) = (1, w/(1+w))`` for ``shape`` and ``(1 + w, w)`` for
``rate``, the SM measurement appears exactly once, inside ``rho_SM``, and the
error is ``|k| * sigma(rho_SM)``.  ``rho_SM``'s own error adds ``n_int``'s and
``n_SM``'s in quadrature; the two samples are independent.


Does the sum stay physically sensible?
======================================
No, not at ``|c_tG| = 10``, and the figures are drawn so that this is visible
rather than hidden.  With ``w_LO(c_tG) = -0.275092 * c_tG`` and
``w_NLO(c_tG) = -0.181081 * c_tG``:

* ``c_tG = -1``  : ``w = +0.2751 / +0.1811``.  A 28 % / 18 % correction on the
  rate.  Sound.
* ``c_tG = +1``  : ``w = -0.2751 / -0.1811``.  Same size, other sign.  Sound.
* ``c_tG = -10`` : ``w = +2.7512 / +1.8108``.  The dimension-six interference
  is now 2.8 (LO) and 1.8 (NLO) times the whole SM cross section.  Everything
  stays positive, so nothing looks broken, but a "correction" three times the
  size of what it corrects is not a correction; the dimension-six *squared*
  term that was dropped is of the same order or larger, and the truncation is
  meaningless here.
* ``c_tG = +10`` : ``w = -2.7512 / -1.8108``.  ``SM + interference`` is
  **negative in every one of the 20 bins**, at both LO and NLO -- a negative
  differential cross section.  It is drawn as it comes out, with no clipping
  and no floor.

So ``G``, ``H``, ``K`` and ``L`` are physics; ``I``, ``J``, ``M`` and ``N`` are
a demonstration that the linear EFT expansion has failed, and must be presented
as that and never as a small correction.


No text on the figures
======================
Axis labels and legends only, as everywhere else in this series.  **The
coefficient a figure is drawn at is not written on it.**  It is in the file
name, in ``README.md`` and in that figure's ``numbers_*.txt``, which opens and
closes with both the c_tG value and the SM NLO provenance note.  The file names
are

    smeft_fig5_G_ctg_m1_shape     smeft_fig5_K_ctg_m1_rate
    smeft_fig5_H_ctg_p1_shape     smeft_fig5_L_ctg_p1_rate
    smeft_fig5_I_ctg_p10_shape    smeft_fig5_M_ctg_p10_rate
    smeft_fig5_J_ctg_m10_shape    smeft_fig5_N_ctg_m10_rate

-- letter for continuity with ``A``--``F``, then the coefficient (``m`` minus,
``p`` plus) and the convention, so no figure can be mistaken for another.


Provenance of the SM NLO sample
===============================
``sm_nlo``'s MadSpin density matrices used to be evaluated at model defaults
rather than the run's card (``173 / 1.4915 / 2.4414 / 2.0476`` against the
run's ``172.76 / 1.33 / 2.4952 / 2.085``).  That is fixed, and the
sample drawn here was regenerated with the fix (task T123).  The measured
effect on this observable is ``-0.0001 %`` on the cross section and, on the
shape, smaller than reseeding the MadSpin run.  The defect was in the
**density** path, which ``spinmode = none`` does not use at all, so only the
**solid** NLO curves were ever touched.  See ``README.md``.
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import plot_smeft_fig5 as P                                    # noqa: E402
_MG7_RC = {k: v for k, v in mpl.rcParams.items()
           if k in ('text.usetex', 'font.family', 'font.size',
                    'lines.markersize', 'mathtext.fontset')}
import plot_smeft_fig5_userstyle as U                          # noqa: E402
import plot_smeft_fig5_varE as E                               # noqa: E402
import plot_smeft_fig5_varF as F                               # noqa: E402
matplotlib.use('Agg')
_USER_RC = {k: mpl.rcParamsDefault[k] for k in _MG7_RC}

MODES = F.MODES                     # ('onshell', 'none')
SAMPLES = F.SAMPLES                 # ('eft_int', 'sm_lo', 'sm_nlo')
CURVES = F.CURVES                   # upper pane, unchanged from F
PANE1 = F.PANE1                     # [('sm_nlo','sm_lo'), ('eft_int','sm_lo')]
PANE2 = F.PANE2                     # ['sm_lo', 'sm_nlo']

NLO_PROVENANCE_SHORT = E.NLO_PROVENANCE_SHORT
NLO_PROVENANCE_LONG = E.NLO_PROVENANCE_LONG

NLO_REGEN_NOTE = """\
    The regeneration (task T123, branch claude/ms-smeft-fig5-nlo-redo) needed
    no change to this script: nothing about the samples is hard-coded here,
    every cross section, weight and curve being read out of the histogram file
    and its meta.json.  Dropping a new data/histograms.npz in place and
    re-running is all a further regeneration would take."""


def _ctg_tag(c):
    """``-1 -> 'm1'``, ``+10 -> 'p10'``: file-name-safe and unambiguous."""
    s = ('%g' % abs(c)).replace('.', 'p')
    return ('m' if c < 0 else 'p') + s


# The eight variations.  Letter for continuity with A--F; the coefficient and
# the convention are both in the name so no two figures can be confused.
# Order follows the request: -1 (the point F was drawn at), +1, +10, -10.
POINTS = [
    ('G', -1.0, 'shape'),
    ('H', +1.0, 'shape'),
    ('I', +10.0, 'shape'),
    ('J', -10.0, 'shape'),
    ('K', -1.0, 'rate'),
    ('L', +1.0, 'rate'),
    ('M', +10.0, 'rate'),
    ('N', -10.0, 'rate'),
]

CONVENTION_BLURB = {
    'rate': 'ratios of unnormalised d sigma/d phi; the interference keeps its '
            'physical sign and size',
    'shape': 'ratios of unit-area curves; the rate change is divided out '
             '(F\'s convention)',
}


def stem(tag, c, conv):
    return 'smeft_fig5_%s_ctg_%s_%s' % (tag, _ctg_tag(c), conv)


# --------------------------------------------------------------------------
class Scan(object):
    """Every curve of the eight figures, at one ``c_tG`` in one convention.

    The generated sample sits at ``c_ref`` (``ctGRe`` from ``meta.json``, which
    is ``-1``).  The interference is linear in the coefficient, so every
    interference cross section is the measured one times ``c_tG / c_ref`` and
    every interference *shape* is the measured one unchanged.
    """

    def __init__(self, d, ctg, convention):
        if convention not in ('rate', 'shape'):
            raise ValueError('convention must be rate or shape')
        self.d = d
        self.ctg = float(ctg)
        self.convention = convention
        self.c_ref = float(d.meta['samples']['eft_int']
                           ['wilson_coefficients']['ctGRe'])
        self.Lambda = float(d.meta['samples']['eft_int']
                            ['wilson_coefficients']['LambdaSMEFT_GeV'])
        self.scale = self.ctg / self.c_ref

    # -- the pieces -------------------------------------------------------
    def sigma_int(self, mode):
        """``sigma_int`` at this ``c_tG``, linear in the coefficient."""
        return self.scale * self.d.sigma('eft_int', mode)

    def weight(self, sm, mode):
        """``w = sigma_int(c_tG) / sigma_SM``, signed.

        The only place ``c_tG`` enters the figure.  ``none`` curves use the
        ``none`` samples' own cross sections, as in ``F`` -- measured to agree
        with the ``onshell`` ones to 6e-8 (LO) and 6e-5 (NLO), not assumed.
        """
        return self.sigma_int(mode) / self.d.sigma(sm, mode)

    def shape_ratio(self, num, den, mode):
        """``n_num/n_den`` of the unit-area shapes; errors in quadrature."""
        return F.VarF(self.d).shape_ratio(num, den, mode)

    def kfactor(self, num, den, mode):
        """``sigma_num/sigma_den``; for ``eft_int`` it carries ``c_tG``."""
        top = (self.sigma_int(mode) if num == 'eft_int'
               else self.d.sigma(num, mode))
        bot = (self.sigma_int(mode) if den == 'eft_int'
               else self.d.sigma(den, mode))
        return top / bot

    # -- pane 1 -----------------------------------------------------------
    def pane1(self, num, den, mode):
        """One pane-1 curve in this figure's convention.

        ``shape``: ``n_num/n_den``, which does not contain ``c_tG``.
        ``rate`` : ``dsigma_num/dsigma_den = (sigma_num/sigma_den) *
        (n_num/n_den)``, which does when the numerator is the interference.
        """
        r, e = self.shape_ratio(num, den, mode)
        if self.convention == 'shape':
            return r, e
        k = self.kfactor(num, den, mode)
        return k * r, abs(k) * e

    def pane1_reference(self):
        """The horizontal lines pane 1 is read against."""
        if self.convention == 'shape':
            return [1.0]
        return [0.0, 1.0]

    # -- pane 2 -----------------------------------------------------------
    def pane2_terms(self, sm, mode):
        """``(baseline, k, w)`` of ``baseline + k*(rho-1)`` for this pane.

        One identity, two conventions.  Writing the pane-2 curve this way is
        what keeps the shared ``n_SM`` from being counted as two independent
        measurements in a numerator and a denominator:

            shape : n_(SM+int)/n_SM = (1 + w*rho)/(1 + w)
                                    = 1 + [w/(1+w)] * (rho - 1)
            rate  : (dsigma_SM + dsigma_int)/dsigma_SM
                                    = 1 + w*rho
                                    = (1 + w) + w * (rho - 1)

        so ``(baseline, k)`` is ``(1, w/(1+w))`` or ``(1 + w, w)`` and the error
        is ``|k| * sigma(rho)`` in both cases.
        """
        w = self.weight(sm, mode)
        if self.convention == 'shape':
            return 1.0, w / (1.0 + w), w
        return 1.0 + w, w, w

    def pane2(self, sm, mode):
        rho, rho_e = self.shape_ratio('eft_int', sm, mode)
        base, k, _ = self.pane2_terms(sm, mode)
        return base + k * (rho - 1.0), abs(k) * rho_e

    def pane2_reference(self):
        return [1.0]

    # -- the health of the sum -------------------------------------------
    def pole(self, sm, mode):
        """The ``c_tG`` at which ``1 + w = 0``: sigma(SM+int) passes zero.

        Past it the ``shape`` convention divides by a negative total and its
        curves are ratios of two negative densities -- positive, near 1, and
        meaningless.
        """
        w_ref = self.d.sigma('eft_int', mode) / self.d.sigma(sm, mode)
        return self.c_ref / w_ref * -1.0 if w_ref else float('inf')

    def diagnose(self, sm, mode):
        """``(w, 1+w, n_bins_negative, past_pole)`` for the sum at this point."""
        w = self.weight(sm, mode)
        rho, _ = self.shape_ratio('eft_int', sm, mode)
        dens = 1.0 + w * rho                      # (SM+int)/SM, rate level
        return (w, 1.0 + w, int(np.sum(dens < 0)), (1.0 + w) < 0)


# --------------------------------------------------------------------------
def check_ctg_invariance(d, fh=sys.stdout):
    """Measure -- do not assume -- that the upper pane does not move.

    The claim is that ``n_int`` is exactly invariant under ``c_tG``, so the
    upper pane is identical in all eight figures.  Rescaling ``sumw`` and
    ``sumw2`` by ``c_tG/c_ref`` and renormalising is the same computation the
    figure does, so if the claim holds the two agree to the last bit.
    """
    p = lambda *a: print(*a, file=fh)
    p('-' * 78)
    p('the upper pane is identical in all eight figures -- measured, not assumed')
    p('-' * 78)
    p('  n_int = (dsigma_int/dphi) / sigma_int.  Both scale by c_tG/c_ref, so')
    p('  the factor cancels -- INCLUDING its sign, which is why c_tG = +1 and')
    p('  c_tG = -1 give the same upper pane and not mirrored ones.')
    p('')
    p('%-10s %-8s %14s %14s' % ('mode', 'c_tG', 'max |dn/n|', 'max |de/e|'))
    worst = 0.0
    for mode in MODES:
        y0, e0 = d.shape('eft_int', mode)
        n = d.nevents('eft_int', mode)
        sumw = d._rebin('eft_int_%s_sumw' % mode)
        sumw2 = d._rebin('eft_int_%s_sumw2' % mode)
        for _, ctg, conv in POINTS:
            if conv != 'shape':
                continue
            s = ctg / -1.0
            dens = s * sumw / n / d.width
            err = np.sqrt(s * s * sumw2) / n / d.width
            total = float(s * sumw.sum() / n)
            y, e = dens / total, err / abs(total)
            dy = float(np.max(np.abs(y / y0 - 1)))
            de = float(np.max(np.abs(e / e0 - 1)))
            worst = max(worst, dy, de)
            p('%-10s %-8g %14.3e %14.3e' % (mode, ctg, dy, de))
    p('')
    p('  Same statement for pane 1 in the SHAPE convention: it is built only')
    p('  from unit-area shapes, so it too is the same picture at every c_tG.')
    p('  That is why G, H, I and J share a pane 1, and why the two pane-1')
    p('  versions that were asked for are the RATE ones, K and L.')
    return [('the upper pane is c_tG-invariant', worst < 1e-12,
             'worst relative move %.2e over c_tG = -1, +1, +10, -10' % worst)]


def check_sum_health(d, fh=sys.stdout):
    """Where the linear EFT truncation stops meaning anything."""
    p = lambda *a: print(*a, file=fh)
    p('-' * 78)
    p('is SM + interference still a sensible object at each c_tG?')
    p('-' * 78)
    p('%-6s %-8s %-8s %10s %10s %8s  %s'
      % ('c_tG', 'SM', 'mode', 'w', '1 + w', 'bins<0', 'verdict'))
    bad = []
    for _, ctg, conv in POINTS:
        if conv != 'rate':
            continue
        s = Scan(d, ctg, 'rate')
        for sm in PANE2:
            for mode in MODES:
                w, opw, nneg, past = s.diagnose(sm, mode)
                if nneg:
                    v = 'NEGATIVE d sigma in %d/%d bins' % (nneg, d.nbins)
                elif abs(w) > 1.0:
                    v = 'positive, but |w| > 1: the correction exceeds the SM'
                else:
                    v = 'sound'
                if nneg or abs(w) > 1.0:
                    bad.append((ctg, sm, mode))
                p('%-6g %-8s %-8s %10.4f %10.4f %8d  %s'
                  % (ctg, sm, mode, w, opw, nneg, v))
    p('')
    p('  The shape convention divides by (1 + w).  It vanishes at')
    for sm in PANE2:
        s = Scan(d, -1.0, 'shape')
        p('      %-8s c_tG = %+.4f (onshell)  %+.4f (none)'
          % (sm, s.pole(sm, 'onshell'), s.pole(sm, 'none')))
    p('  and c_tG = +10 is past both poles, so variations I\'s pane-2 curves')
    p('  are ratios of two NEGATIVE densities.  They come out positive and')
    p('  near 1.  That is the arbitrary-sign normalisation this series has')
    p('  been avoiding since it began; read M (the rate figure at the same')
    p('  coefficient) instead, where the same physics lies below zero.')
    return [('the sum is honest wherever it is drawn', True,
             '%d of the scanned (c_tG, SM, mode) points are outside EFT '
             'validity and are flagged, not clipped' % len(bad))]


# --------------------------------------------------------------------------
def _room(ax, series, anchors, bottom_extra=0.0, pad_frac=0.10):
    """Fit every point plus its error and every reference line, then make room.

    ``F._room`` anchors on 1.0, which is right for a ratio that stays near 1.
    The ``rate`` panes do not: at ``c_tG = +10`` pane 2 lies entirely below
    zero, and the zero line has to be on the pane for that to be readable.
    """
    lo = min(anchors)
    hi = max(anchors)
    for r, e in series:
        lo = min(lo, float((r - e).min()))
        hi = max(hi, float((r + e).max()))
    span = hi - lo
    ax.set_ylim(lo - pad_frac * span - bottom_extra * span,
                hi + pad_frac * span)


def _refline(ax, y):
    ax.axhline(y, color='black', lw=0.9 if y == 1.0 else 0.7,
               ls='solid' if y == 1.0 else (0, (1.5, 1.5)), zorder=2)


def _label_p1(s, num, den, mode, plain=False):
    """Pane 1's legend entry.  Names the quantity, never the coefficient."""
    tex = not plain and P.USETEX
    name = P.SAMPLE_TEX if tex else P.SAMPLE_PLAIN
    mtag = P.MODE_TEX if tex else P.MODE_PLAIN
    sep = r'\,/\,' if tex else ' / '
    return '%s%s%s, %s' % (name[num], sep, name[den], mtag[mode])


def _label_p2(sm, mode, plain=False):
    return F._label_p2(sm, mode, plain=plain)


def _ylabel_p1(s, tx=None):
    if s.convention == 'shape':
        return (tx(r'shape ratio', 'shape ratio') if tx else 'Shape ratio')
    if tx:
        return tx(r'$\mathrm{d}\sigma/\mathrm{d}\sigma_{\mathrm{LO}}$',
                  'd sigma / d sigma(LO)')
    return r'$d\sigma\,/\,d\sigma_{\rm LO}$'


def _ylabel_p2(s, tx=None):
    if tx:
        return tx(r'$(\mathrm{SM}+\mathcal{O}_{tG})/\mathrm{SM}$',
                  '(SM + O_tG) / SM')
    return '(SM + $\\mathcal{O}_{tG}$) / SM'


# --------------------------------------------------------------------------
def make_figure_mg7(d, s, tag, out):
    """The MG7 paper style, three panes, four curves in each ratio pane."""
    mpl.rcParams.update(_MG7_RC)
    tx = P._tx

    fig = plt.figure(figsize=(7 * 0.75 * 1.35, 7 * 0.75 * 1.5 * 1.40))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.2], hspace=0.07)
    ax = fig.add_subplot(gs[0])
    r1 = fig.add_subplot(gs[1], sharex=ax)
    r2 = fig.add_subplot(gs[2], sharex=ax)
    lo, hi = 0.0, float(np.pi)

    # --- upper pane: all six curves, identical in all eight figures ------
    for sample, mode in CURVES:
        y, ye = d.shape(sample, mode)
        E._step(ax, d, y, color=P.COLOR[sample], ls=P.LS[mode], lw=P.LW,
                label=F._label_top(sample, mode), zorder=4)
        ax.errorbar(d.centres, y, yerr=ye, fmt='none', ecolor=P.COLOR[sample],
                    elinewidth=0.9, capsize=0, zorder=4)

    ax.set_ylabel(tx(
        r'$\frac{1}{\sigma}\,\mathrm{d}\sigma/\mathrm{d}\Delta\phi(e^-e^+)$'
        r'\ \ [rad$^{-1}$]',
        r'$(1/\sigma)\,d\sigma/d\Delta\phi(e^-e^+)$  [1/rad]'))
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(MultipleLocator(np.pi / 4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelbottom=False)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(ymin, 0.15), ymax + 0.30 * (ymax - ymin))
    ax.legend(frameon=False, loc='upper left', fontsize=9.5,
              handlelength=2.8, borderaxespad=1.0, labelspacing=0.4, ncol=2,
              columnspacing=1.4)

    # --- pane 1 ----------------------------------------------------------
    for y in s.pane1_reference():
        _refline(r1, y)
    series = []
    for mode in MODES:
        for num, den in PANE1:
            r, e = s.pane1(num, den, mode)
            series.append((r, e))
            E._step(r1, d, r, color=P.COLOR[num], ls=P.LS[mode], lw=P.LW,
                    zorder=4, label=_label_p1(s, num, den, mode))
            r1.errorbar(d.centres, r, yerr=e, fmt='none', ecolor=P.COLOR[num],
                        elinewidth=0.9, capsize=0, zorder=4)
    _room(r1, series, s.pane1_reference(), bottom_extra=0.46)
    r1.set_ylabel(_ylabel_p1(s, tx), fontsize=11.5)
    r1.legend(frameon=False, loc='lower left', fontsize=7.5,
              handlelength=2.4, borderaxespad=0.5, ncol=2, columnspacing=1.4)

    # --- pane 2 ----------------------------------------------------------
    for y in s.pane2_reference():
        _refline(r2, y)
    series, handles = [], []
    for mode in MODES:
        for sm in PANE2:
            r, e = s.pane2(sm, mode)
            series.append((r, e))
            E._step(r2, d, r, color=P.COLOR[sm], ls=P.LS[mode], lw=P.LW,
                    zorder=4)
            r2.errorbar(d.centres, r, yerr=e, fmt='none', ecolor=P.COLOR[sm],
                        elinewidth=0.9, capsize=0, zorder=4)
            handles.append(Line2D([], [], color=P.COLOR[sm], lw=P.LW,
                                  ls=P.LS[mode], label=_label_p2(sm, mode)))
    anchors = list(s.pane2_reference())
    if min(min((r - e).min() for r, e in series), 1.0) < 0:
        anchors.append(0.0)
        _refline(r2, 0.0)
    _room(r2, series, anchors, bottom_extra=0.46)
    r2.set_ylabel(_ylabel_p2(s, tx), fontsize=11.5)
    r2.legend(handles=handles, frameon=False, loc='lower left', fontsize=7.5,
              handlelength=2.4, borderaxespad=0.5, ncol=2, columnspacing=1.4)

    r2.set_xlabel(tx(r'$\Delta\phi(e^-e^+)$ [rad]',
                     r'$\Delta\phi(e^-e^+)$ [rad]'))
    for rx in (r1, r2):
        rx.xaxis.set_major_locator(MultipleLocator(np.pi / 4))
        rx.xaxis.set_minor_locator(AutoMinorLocator(2))
        rx.yaxis.set_minor_locator(AutoMinorLocator())
        rx.set_xlim(lo, hi)
    r1.tick_params(labelbottom=False)
    r2.xaxis.set_major_formatter(P._pi_formatter())

    fig.subplots_adjust(left=0.155, right=0.975, top=0.988, bottom=0.075)
    base = os.path.join(out, stem(tag, s.ctg, s.convention))
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=300)
    plt.close(fig)
    return base


# --------------------------------------------------------------------------
def make_figure_user(d, s, tag, out):
    """The user's own style: marker fill carries the spinmode."""
    mpl.rcParams.update(_USER_RC)

    fig = plt.figure(figsize=(6, 8.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.2], hspace=U.HSPACE)
    ax = fig.add_subplot(gs[0])
    r1 = fig.add_subplot(gs[1], sharex=ax)
    r2 = fig.add_subplot(gs[2], sharex=ax)

    def draw(axis, y, ye, colour, mode, label):
        axis.step(d.edges, np.concatenate([y[:1], y]), where='pre',
                  color=colour, lw=1.0, alpha=U.STEP_ALPHA, zorder=3)
        axis.errorbar(d.centres, y, yerr=ye, fmt='o', ms=U.MS, color=colour,
                      mfc=(colour if U.FILLED[mode] else 'white'), mew=1.1,
                      label=label, zorder=4)

    for sample, mode in CURVES:
        y, ye = d.shape(sample, mode)
        draw(ax, y, ye, U.COLOR[sample], mode,
             F._label_top(sample, mode, plain=True))

    ax.set_ylabel(r'$(1/\sigma)\,d\sigma/d\Delta\phi(e^-e^+)$  [1/rad]')
    ax.set_xlim(0.0, np.pi)
    ax.tick_params(labelbottom=False)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.32 * (ymax - ymin))
    ax.legend(loc='upper left', fontsize=8.0, ncol=2)

    for y in s.pane1_reference():
        r1.axhline(y, color='black', ls='--', lw=0.9, zorder=2)
    series = []
    for mode in MODES:
        for num, den in PANE1:
            r, e = s.pane1(num, den, mode)
            series.append((r, e))
            draw(r1, r, e, U.COLOR[num], mode,
                 _label_p1(s, num, den, mode, plain=True))
    _room(r1, series, s.pane1_reference(), bottom_extra=0.40)
    r1.set_ylabel(_ylabel_p1(s))
    r1.legend(loc='lower left', fontsize=6.5, ncol=2)
    r1.tick_params(labelbottom=False)

    for y in s.pane2_reference():
        r2.axhline(y, color='black', ls='--', lw=0.9, zorder=2)
    series = []
    for mode in MODES:
        for sm in PANE2:
            r, e = s.pane2(sm, mode)
            series.append((r, e))
            draw(r2, r, e, U.COLOR[sm], mode, _label_p2(sm, mode, plain=True))
    anchors = list(s.pane2_reference())
    if min(min((r - e).min() for r, e in series), 1.0) < 0:
        anchors.append(0.0)
        r2.axhline(0.0, color='black', ls=':', lw=0.9, zorder=2)
    _room(r2, series, anchors, bottom_extra=0.40)
    r2.set_ylabel(_ylabel_p2(s))
    r2.legend(loc='lower left', fontsize=6.5, ncol=2)
    r2.set_xlabel(r'$\Delta\phi(e^-e^+)$ [rad]')
    r2.set_xlim(0.0, np.pi)
    U._pi_ticks(r2)

    fig.subplots_adjust(hspace=0.1, left=0.155, right=0.97,
                        bottom=0.07, top=0.985)
    base = os.path.join(out, stem(tag, s.ctg, s.convention))
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=U.DPI)
    plt.close(fig)
    return base


# --------------------------------------------------------------------------
def write_curves(d, s, path):
    """Every number this variation draws, beside the figure."""
    out = {'edges': d.edges, 'centres': d.centres,
           'bin_width_rad': np.array(d.width),
           'ctGRe': np.array(s.ctg),
           'ctGRe_generated': np.array(s.c_ref),
           'Lambda_GeV': np.array(s.Lambda),
           'convention': np.array(s.convention)}
    for sample, m in CURVES:
        y, ye = d.shape(sample, m)
        out['%s_%s_shape' % (sample, m)] = y
        out['%s_%s_shape_err' % (sample, m)] = ye
        out['%s_%s_sigma_pb' % (sample, m)] = np.array(d.sigma(sample, m))
    for m in MODES:
        out['eft_int_%s_sigma_pb_at_ctg' % m] = np.array(s.sigma_int(m))
        for num, den in PANE1:
            r, e = s.pane1(num, den, m)
            out['pane1_%s_over_%s_%s' % (num, den, m)] = r
            out['pane1_%s_over_%s_%s_err' % (num, den, m)] = e
        for sm in PANE2:
            r, e = s.pane2(sm, m)
            base, k, w = s.pane2_terms(sm, m)
            out['pane2_%s_plus_int_over_%s_%s' % (sm, sm, m)] = r
            out['pane2_%s_plus_int_over_%s_%s_err' % (sm, sm, m)] = e
            out['pane2_w_%s_%s' % (sm, m)] = np.array(w)
            out['pane2_baseline_%s_%s' % (sm, m)] = np.array(base)
            out['pane2_k_%s_%s' % (sm, m)] = np.array(k)
    np.savez(path, **out)
    return path


def _header(s, tag, p):
    p('=' * 78)
    p('Fig. 5 variation %s: %s, at c_tG = %+g, %s convention'
      % (tag, 'F scanned over the Wilson coefficient', s.ctg, s.convention))
    p('=' * 78)
    p('')
    p('  THE COEFFICIENT IS NOT WRITTEN ON THE FIGURE.  This figure is drawn')
    p('  at')
    p('')
    p('        ctGRe = %+g          Lambda = %g GeV' % (s.ctg, s.Lambda))
    p('        convention: %s' % s.convention)
    p('        file stem : %s' % stem(tag, s.ctg, s.convention))
    p('')
    p('  and a caption must say so.  The samples were generated at ctGRe =')
    p('  %+g; the interference is linear in the coefficient, so everything' % s.c_ref)
    p('  here is the measured interference times %+g.' % s.scale)
    p('')


def write_numbers(d, s, tag, fh=sys.stdout):
    p = lambda *a: print(*a, file=fh)
    setup = d.meta['setup']
    _header(s, tag, p)
    p(NLO_PROVENANCE_LONG)
    p('')
    p(NLO_REGEN_NOTE)
    p('')
    p('  On this figure the defect was in the DENSITY path, which spinmode =')
    p('  none does not use.  So the DASHED NLO curves never moved at all; the')
    p('  SOLID ones moved by less than reseeding the run.')
    p('')

    # -- what the eight figures are, so any one of them locates the others
    p('=' * 78)
    p('the eight variations, and why they are eight')
    p('=' * 78)
    p('  The upper pane is IDENTICAL in all eight.  So is pane 1 in the four')
    p('  `shape\' figures.  Both facts are consequences of the interference')
    p('  being linear in c_tG: n_int = (dsigma_int)/(sigma_int) has the factor')
    p('  in numerator and denominator and it cancels, sign and all.  The')
    p('  coefficient enters the figure in exactly one place, the signed weight')
    p('  w = sigma_int(c_tG)/sigma_SM.')
    p('')
    p('  Two versions of pane 1 were asked for, at c_tG = -1 and +1.  In the')
    p('  shape convention they would have been the SAME PICTURE, so the two')
    p('  versions are the RATE ones, K and L, where pane 1 does contain w and')
    p('  c_tG = +1 mirrors c_tG = -1 exactly about the no-interference line.')
    p('')
    p('%-4s %-8s %-8s  %s' % ('tag', 'c_tG', 'conv.', 'file stem'))
    for t, c, conv in POINTS:
        mark = '  <== this file' if t == tag else ''
        p('%-4s %-8g %-8s  %s%s' % (t, c, conv, stem(t, c, conv), mark))
    p('')
    p('  rate  : %s' % CONVENTION_BLURB['rate'])
    p('  shape : %s' % CONVENTION_BLURB['shape'])
    p('')

    p('observable   : %s' % setup['observable'])
    p('sqrt(s)      : %g GeV      scale: %s' % (setup['sqrt_s_GeV'],
                                                setup['scale']))
    p('PDF          : %s' % setup['pdf'])
    p('cuts         : %s' % setup['cuts'])
    p('sign         : %s' % setup['sign_convention'])
    p('parameter pt : ctGRe = %+g (generated at %+g), Lambda = %g GeV'
      % (s.ctg, s.c_ref, s.Lambda))
    p('plot binning : %d uniform bins over [0, pi], width %.6f rad '
      '(%d stored bins each)' % (d.nbins, d.width, d.group))
    p('encoding     : line style = spinmode (solid onshell, dashed none), '
      'colour = quantity')
    p('')
    P.check_normalisation(d, fh)
    p('')
    check_ctg_invariance(d, fh)
    p('')
    check_sum_health(d, fh)

    # -- the weights, which are the whole c_tG dependence -----------------
    p('')
    p('=' * 78)
    p('the weights: the only place c_tG enters')
    p('=' * 78)
    p('  w(c_tG) = sigma_int(c_tG)/sigma_SM is linear in c_tG and signed.')
    p('  At the generated point ctGRe = %+g it is measured to be' % s.c_ref)
    for mode in MODES:
        p('      %-8s  w(LO) = %+.6f   w(NLO) = %+.6f'
          % (mode, d.sigma('eft_int', mode) / d.sigma('sm_lo', mode),
             d.sigma('eft_int', mode) / d.sigma('sm_nlo', mode)))
    p('  so w(LO) = %+.6f * c_tG and w(NLO) = %+.6f * c_tG.'
      % (d.sigma('eft_int', 'onshell') / d.sigma('sm_lo', 'onshell') / s.c_ref,
         d.sigma('eft_int', 'onshell') / d.sigma('sm_nlo', 'onshell')
         / s.c_ref))
    p('')
    p('%-8s %-8s %-8s %11s %11s %11s'
      % ('c_tG', 'SM', 'mode', 'w', '1 + w', 'k'))
    for _, ctg, conv in POINTS:
        if conv != s.convention:
            continue
        sc = Scan(d, ctg, s.convention)
        for sm in PANE2:
            for mode in MODES:
                base, k, w = sc.pane2_terms(sm, mode)
                p('%-8g %-8s %-8s %11.6f %11.6f %11.6f'
                  % (ctg, sm, mode, w, 1.0 + w, k))
    p('')
    p('  k is the coefficient of (rho - 1) in this convention:')
    p('      shape : baseline 1,     k = w/(1+w)')
    p('      rate  : baseline 1 + w, k = w')
    p('  and the error on the pane-2 curve is |k| * sigma(rho) in both, which')
    p('  is the identity that keeps the shared n_SM from being counted twice.')

    # -- the curves --------------------------------------------------------
    p('')
    p('=' * 78)
    p('what this figure draws')
    p('=' * 78)
    rows = []
    for mode in MODES:
        for num, den in PANE1:
            rows.append(('p1 %s/%s, %s' % (num, den, mode),
                         s.pane1(num, den, mode)))
    for mode in MODES:
        for sm in PANE2:
            rows.append(('p2 (%s+int)/%s, %s' % (sm, sm, mode),
                         s.pane2(sm, mode)))
    p('%-32s %11s %11s %11s %11s'
      % ('curve', 'first bin', 'last bin', 'min', 'max'))
    for name, (r, e) in rows:
        p('%-32s %11.4f %11.4f %11.4f %11.4f'
          % (name, r[0], r[-1], r.min(), r.max()))
    p('')
    p('  and with their errors:')
    for name, (r, e) in rows:
        p('    %-32s first %+9.4f +- %.4f   last %+9.4f +- %.4f'
          % (name, r[0], e[0], r[-1], e[-1]))

    p('')
    p('=' * 78)
    p('bin by bin')
    p('=' * 78)
    p('%9s' % 'phi/pi' + ''.join(' %21s' % n for n, _ in rows))
    for i in range(d.nbins):
        line = '%9.3f' % (d.centres[i] / np.pi)
        for _, (r, e) in rows:
            line += '   %8.4f +- %.4f' % (r[i], e[i])
        p(line)

    p('')
    p('=' * 78)
    p('what a caption must carry, since the figure carries no text')
    p('=' * 78)
    p('    * ctGRe = %+g, Lambda = %g GeV.  Without it the ratio panes have'
      % (s.ctg, s.Lambda))
    p('      no scale at all.')
    p('    * the convention: %s -- %s'
      % (s.convention, CONVENTION_BLURB[s.convention]))
    p('    * the SM NLO provenance note: the defect touched the SOLID red')
    p('      curves only, spinmode = none using no density matrices, and its')
    p('      measured effect on this observable is below the seed noise;')
    p('    * that pane 1 is %s;'
      % ('a SHAPE ratio and not a K-factor (the K-factor is %.3f)'
         % (d.sigma('sm_nlo', 'onshell') / d.sigma('sm_lo', 'onshell'))
         if s.convention == 'shape' else
         'a ratio of unnormalised d sigma, so the SM NLO / SM LO curve IS '
         'the differential K-factor (its integral is %.3f)'
         % (d.sigma('sm_nlo', 'onshell') / d.sigma('sm_lo', 'onshell'))))
    p('    * that the dashed curves are the non-spin part of each ratio and')
    p('      the solid-to-dashed gap is the spin-correlation effect;')
    worst = [s.diagnose(sm, m) for sm in PANE2 for m in MODES]
    if any(n for _, _, n, _ in worst):
        p('    * !! THIS FIGURE IS OUTSIDE EFT VALIDITY.  SM + interference is')
        p('      NEGATIVE in up to %d of the %d bins -- a negative'
          % (max(n for _, _, n, _ in worst), d.nbins))
        p('      differential cross section.  It must NOT be presented as a')
        p('      small correction: at |c_tG| = 10 the dropped dimension-six')
        p('      SQUARED term is of the same order or larger than what is')
        p('      kept.')
        if s.convention == 'rate':
            p('      Pane 2 draws that as it comes out, unclipped and below')
            p('      zero.')
        else:
            p('      PANE 2 DOES NOT SHOW IT: see the next point.')
    elif any(abs(w) > 1 for w, _, _, _ in worst):
        p('    * !! THIS FIGURE IS OUTSIDE EFT VALIDITY.  |w| > 1: the')
        p('      interference is larger than the whole SM cross section it')
        p('      corrects.  Everything stays positive so nothing looks broken,')
        p('      but the dropped dimension-six SQUARED term is of the same')
        p('      order or larger than what is kept.')
    if any(past for _, _, _, past in worst):
        p('    * !! and this is the SHAPE convention past its (1 + w) pole, so')
        p('      its pane-2 curves are ratios of two NEGATIVE densities.  They')
        p('      look innocuous and are not.  Read the rate figure at the same')
        p('      coefficient instead.')
    p('')
    p(NLO_PROVENANCE_LONG)
    p('')
    p(NLO_REGEN_NOTE)
    p('')
    _header(s, tag, p)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--out-user', dest='out_user',
                    default=os.path.join(_HERE, 'plots_userstyle'))
    ap.add_argument('--nbins', type=int, default=20,
                    help='plot bins over [0, pi]; must divide the stored 720')
    ap.add_argument('--style', choices=('mg7', 'user', 'both'), default='both')
    ap.add_argument('--only', default=None,
                    help='comma-separated subset of G,H,I,J,K,L,M,N')
    ap.add_argument('--check-minus', dest='check_minus', action='store_true',
                    default=True)
    ap.add_argument('--no-check-minus', dest='check_minus',
                    action='store_false')
    args = ap.parse_args()

    d = P.Data(args.data, nbins=args.nbins)
    devnull = open(os.devnull, 'w')
    rows = (P.check_normalisation(d, fh=devnull)
            + F.check_weights(d, fh=devnull)
            + check_ctg_invariance(d, fh=devnull)
            + check_sum_health(d, fh=devnull))
    bad = [name for name, ok, _ in rows if not ok]
    if bad:
        raise SystemExit('checks failed: %s' % '; '.join(bad))
    for name, ok, detail in rows:
        print('  check: %-52s %s -- %s'
              % (name, 'ok' if ok else 'FAILED', detail))
    print('')

    wanted = None
    if args.only:
        wanted = {t.strip().upper() for t in args.only.split(',')}

    failures = []
    for tag, ctg, conv in POINTS:
        if wanted and tag not in wanted:
            continue
        s = Scan(d, ctg, conv)
        name = stem(tag, ctg, conv)
        if args.style in ('mg7', 'both'):
            os.makedirs(args.out, exist_ok=True)
            base = make_figure_mg7(d, s, tag, args.out)
            print('wrote %s.pdf / .png   (MG7 paper style)' % base)
            if args.check_minus:
                ok, detail = P.check_minus(base + '.pdf')
                print('   minus check: %s -- %s'
                      % ({True: 'ok', False: 'FAILED', None: 'n/a'}[ok],
                         detail))
                if ok is False:
                    failures.append(base)
            write_curves(d, s, os.path.join(args.out, name + '_curves.npz'))
            with open(os.path.join(args.out, 'numbers_%s.txt' % tag),
                      'w') as fh:
                write_numbers(d, s, tag, fh)
            print('   %s_curves.npz and numbers_%s.txt' % (name, tag))

        if args.style in ('user', 'both'):
            os.makedirs(args.out_user, exist_ok=True)
            base = make_figure_user(d, s, tag, args.out_user)
            print('wrote %s.pdf / .png   (user style)' % base)
            write_curves(d, s,
                         os.path.join(args.out_user, name + '_curves.npz'))
            with open(os.path.join(args.out_user, 'numbers_%s.txt' % tag),
                      'w') as fh:
                write_numbers(d, s, tag, fh)
            print('   %s_curves.npz and numbers_%s.txt' % (name, tag))

    print('')
    check_ctg_invariance(d)
    print('')
    check_sum_health(d)
    print('')
    print('*** %s ***' % NLO_PROVENANCE_SHORT)
    print('*** the defect hits the SOLID red curves only: '
          'spinmode = none uses no density matrices ***')
    print('*** I, J, M, N are drawn at |c_tG| = 10, outside EFT validity: '
          'see their numbers_*.txt ***')

    if failures:
        raise SystemExit('the usetex minus sign was lost in: %s'
                         % ', '.join(failures))


if __name__ == '__main__':
    main()
