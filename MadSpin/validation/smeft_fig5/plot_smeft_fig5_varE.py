#!/usr/bin/env python3
r"""Variation ``E`` of the remade Fig. 5: three ``onshell`` curves and a single
no-spin-correlation curve, over **two** ratio panes, with no text on the plot.

Standalone and re-runnable; it draws both renderings in one go::

    plot_smeft_fig5_varE.py [--data DIR] [--out DIR] [--out-user DIR]
                            [--nbins N] [--style {mg7,user,both}]

Everything comes from ``data/histograms.npz`` and ``data/meta.json``.  The data
loading, rebinning, error propagation and ``check_normalisation()`` are imported
from ``plot_smeft_fig5.py``; the user-style constants from
``plot_smeft_fig5_userstyle.py``.  Neither module is modified, and variations
``A``--``D`` are left exactly as they are -- this script only adds ``E``.

What ``E`` changes, and why
---------------------------

**No text on the plot.**  Axis labels and legends, and nothing else: no header
block, no parameter point written on the figure, no notes on the ratio panes, no
title, no footnote, and legend entries that identify rather than explain.
Everything that would otherwise be written there lives in ``README.md`` and in
``numbers_E.txt``, which ends with an explicit list of what a caption must carry
-- the parameter point above all, since pane 2's magnitude is proportional to
``c_tG/Lambda^2`` and means nothing without it.

**Upper pane.**  Three ``onshell`` curves -- the SMEFT interference, the SM at
LO and the SM at NLO -- and exactly one ``spinmode = none`` curve.

That one ``none`` curve covers the **two LO samples only**, and its legend entry
names them: "SMEFT O_tG interference & SM, LO, none".  Measured on these
histograms at the plotted binning::

    pair of `none' shapes      max |r-1|    chi2/ndf
    SMEFT  / SM LO                 2.1 %         1.9
    SMEFT  / SM NLO                8.0 %        10.5
    SM LO  / SM NLO                8.7 %        13.8

The two LO shapes are one curve; the NLO shape is four times further away, and
the gap does not close at finer binning (8.1 % at 36 bins, 11.1 % at 72).  This
confirms T110's 8 %.  The cause is radiation and not spin -- the extra emission
changes the ``t t~`` boost -- so a ``none`` curve claiming to cover NLO as well
would be false.  The claim is narrowed rather than a curve added: SM NLO is on
the pane with its ``onshell`` curve alone, exactly as SM LO was in variation
``A``, and nothing in the legend suggests it has a dashed partner hidden under
the blue one.

The SMEFT ``none`` is the one kept because ``E``'s LO content is then exactly
the published figure's curve set; because it keeps the blue sample complete,
solid *and* dashed, which is right for a figure whose subject is the operator;
and because both LO ``none`` samples have 1 M events and the same 0.48 % median
per-bin error, so statistics does not decide it.

**Two ratio panes**, replacing ``B``'s single ``onshell/none`` pane.  Both use
the ``onshell`` curves throughout, since that is the physical prediction.

* Pane 1: ``NLO/LO`` and ``SMEFT/LO``.  This is **not** a K-factor.  Every curve
  in the upper pane is normalised to unit area, so the rates have been divided
  out before the ratio is taken; the pane shows only how the *shape* moves.  The
  NLO/LO K-factor of this sample is 10.900/7.175 = 1.52 and appears nowhere on
  the figure.
* Pane 2: ``(LO + SMEFT)/LO`` and ``(NLO + SMEFT)/NLO``.

The relative weight in pane 2
-----------------------------
Adding two unit-area shapes needs a relative weight, and that weight sets the
whole size of pane 2.  The weight used here is the **ratio of the decayed cross
sections**, which the manifest carries for every sample::

    n_sum(x) = [sigma_SM * n_SM(x) + sigma_int * n_int(x)] / (sigma_SM + sigma_int)

with ``sigma_int = 1.9739 pb``, ``sigma_SM(LO) = 7.1753 pb`` and
``sigma_SM(NLO) = 10.9004 pb``, all ``onshell``.  That is the real relative rate
of the two contributions, so pane 2 is the physical statement "what the operator
does to this distribution" and not an assertion about how big the operator is.
The alternative -- averaging the two unit-area shapes with equal weight -- is
well defined but corresponds to no parameter point: it would claim the
dimension-six interference is as large as the SM, which at ``c_tG = -1``,
``Lambda = 1 TeV`` it is not (it is 27.5 % of it at LO).

Three consequences, all stated on the figure:

1. **The pane scales with** ``c_tG/Lambda^2``.  The samples are at
   ``ctGRe = -1``, ``Lambda = 1 TeV``; at another parameter point pane 2 scales
   linearly and, for ``c_tG > 0``, *mirrors about 1*.  An unlabelled arbitrary
   normalisation would make the pane meaningless, so the parameter point is
   written on the figure.

2. **Pane 2 carries no information beyond pane 1 and the two cross sections.**
   The shared ``n_SM`` cancels algebraically::

       (SM + int)/SM = 1 + [w/(1+w)] * (n_int/n_SM - 1),   w = sigma_int/sigma_SM

   so pane 2's LO curve is pane 1's blue curve shrunk towards 1 by
   ``w/(1+w) = 0.2157``, and its NLO curve is ``SMEFT/NLO`` shrunk by 0.1533.
   This is worth knowing rather than hiding: it is why the errors in pane 2 are
   computed from that identity (the ``n_SM`` in numerator and denominator is one
   and the same measurement and must not be counted as two independent ones).

3. The NLO curve in pane 2 is *smaller* than the LO one, and the reason is
   arithmetic rather than physics: ``sigma_SM`` grows by the 1.52 K-factor while
   the interference is only available at LO, so ``w`` falls from 0.275 to 0.181.
   Read pane 2's NLO curve as "an LO interference against an NLO SM", which is
   what it is, and not as "the operator matters less at NLO".

Provenance of the SM NLO sample
-------------------------------
Both new panes use the SM NLO sample, which once had a real defect: its MadSpin
density matrices were evaluated at the *model default* parameters
(``173 / 1.4915 / 2.4414 / 2.0476``) rather than the run's own
(``172.76 / 1.33 / 2.4952 / 2.085``).  That is fixed, and the sample drawn here
was regenerated with the fix (task T123).  The measured effect on this
observable is ``-0.0001 %`` on the cross section and, on the shape, smaller than
reseeding the MadSpin run -- see ``NLO_PROVENANCE_LONG`` below for the control
numbers.

The figure says nothing on its face -- no text is allowed on it -- so the note
survives **only** in ``README.md`` and in ``numbers_E.txt``, which opens and
closes with it.  A caption written from this figure must carry it by hand.
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

# plot_smeft_fig5 installs the usetex minus-sign fix and the MG7 rcParams at
# import time; plot_smeft_fig5_userstyle resets rcParams to the stock defaults
# at import time.  Both styles are wanted here, so each set is captured and
# re-applied explicitly around the drawing, and neither module is modified.
import plot_smeft_fig5 as P                                    # noqa: E402
_MG7_RC = {k: v for k, v in mpl.rcParams.items()
           if k in ('text.usetex', 'font.family', 'font.size',
                    'lines.markersize', 'mathtext.fontset')}
import plot_smeft_fig5_userstyle as U                          # noqa: E402
matplotlib.use('Agg')
_USER_RC = {k: mpl.rcParamsDefault[k] for k in _MG7_RC}

TAG = 'E'

# The upper pane.  Colour names the sample and the dash names the spinmode, as
# in A--D.  Three `onshell' curves and exactly one `none'; the solid ones are
# listed first so the legend groups them.
CURVES = [('eft_int', 'onshell'), ('sm_lo', 'onshell'), ('sm_nlo', 'onshell'),
          ('eft_int', 'none')]

# The one no-spin-correlation curve that is drawn, and the samples it covers.
#
# It covers the two LO samples ONLY.  Their `none' shapes agree to 2.1 %
# (chi2/ndf 1.9), so one curve honestly stands for both.  The SM NLO `none'
# shape does NOT join them: it differs from the SMEFT one by 8.0 %
# (chi2/ndf 10.5) and from the SM LO one by 8.7 % (chi2/ndf 13.8), measured on
# these histograms and confirming T110's 8 %.  That difference is a
# radiation effect -- the extra emission changes the t t~ boost -- and has
# nothing to do with spin, so folding NLO into the drawn curve would be simply
# false.  The legend entry therefore NAMES the two samples the curve covers and
# says nothing at all about NLO; SM NLO appears on the pane with its `onshell'
# curve only, exactly as SM LO did in variation A.
NONE_KEPT = 'eft_int'
NONE_COVERS = ('eft_int', 'sm_lo')
NONE_EXCLUDES = 'sm_nlo'
NONE_STANDS_FOR = 'sm_lo'          # the partner of NONE_KEPT, for the numbers

# Ratio panes.  ``(numerator, denominator)`` in pane 1; ``(sm, interference)``
# in pane 2.  Everything ``onshell``.
PANE1 = [('sm_nlo', 'sm_lo'), ('eft_int', 'sm_lo')]
PANE2 = ['sm_lo', 'sm_nlo']

# The user style's ratio-limit ladder (imported ``U.RATIO_LADDER``) jumps
# straight from +-1 % to +-15 %.  Variation B's onshell/none ratio spans +-30 %
# so that gap never showed; pane 2 here spans +-2.5 %, and the +-15 % rung would
# draw it as a flat line.  The ladder is therefore extended with intermediate
# rungs *locally*, in this script only -- the user's own module is unchanged and
# the rungs it does have are kept exactly where they are.
LADDER_E = [(0.99, 1.01), (0.98, 1.02), (0.95, 1.05),
            (0.92, 1.08), (0.85, 1.15), (0.8, 1.2)] + list(U.RATIO_LADDER[2:])


def choose_ylim_E(series):
    lo = hi = 1.0
    for r, e in series:
        lo = min(lo, float((r - e).min()))
        hi = max(hi, float((r + e).max()))
    for cand in LADDER_E:
        if lo >= cand[0] and hi <= cand[1]:
            return cand
    half = max(1.0 - lo, hi - 1.0)
    return (1.0 - 1.1 * half, 1.0 + 1.1 * half)


NLO_PROVENANCE_SHORT = ('SM NLO regenerated with the param_card fix; '
                        'measured effect -0.0001 % on sigma, below seed '
                        'noise on the shape -- see README')
NLO_PROVENANCE_LONG = """\
SM NLO provenance: the param_card defect, measured and fixed

    MadSpin's density path used to initialise its standalone matrix-element
    library from <madspin_me>/Cards/param_card.dat -- the card `output
    standalone' wrote from the MODEL DEFAULTS -- and never from the run's own
    card.  For PROC_sm_nlo that meant

        run :  MT 172.76   WT 1.33     WZ 2.4952   WW 2.085
        ME  :  MT 173.0    WT 1.4915   WZ 2.4414   WW 2.0476

    a top mass 0.24 GeV too high and a top width 12 % too large in the density
    matrix.  The defect was real.  It is fixed (0a1007bc2), and the sm_nlo
    sample plotted here was regenerated with the fix (T123), reusing the same
    500k production events at the same MadSpin seeds -- so the comparison
    below is exact and not Monte-Carlo noise in the production.

    The measured effect on this observable is nil:

      rate   10.9004361 -> 10.9004235 pb, i.e. -0.0001 %, against a 0.020 pb
             integration error.  The <init> XSECUP is 10.899373 +- 0.020274 pb
             before and after, to every digit the banner carries.

      shape  below the reseeding noise floor.  Buggy against fixed moves the
             Delta phi forward-backward asymmetry by +0.0076; changing only
             the MadSpin seed, 42 -> 99, with the fixed code moves it by
             -0.0050; buggy against that seed-99 run, by +0.0025.  The 720-bin
             chi2/ndf is 1.002, 0.975 and 0.997 for the three pairings: all
             equally indistinguishable.

    So `immaterial' here is a measurement, not a hope.  spinmode = none was
    never affected -- it builds no density matrix, has no madspin_me directory,
    and its histograms reproduced bit for bit.  The two LO samples differ from
    their ME card only in alpha_s(M_Z) (0.1190025 against 0.1179), which
    cancels exactly in MadSpin's Tr(rho_prod)-normalised accept/reject
    weight."""


# --------------------------------------------------------------------------
class VarE(object):
    """The four ratio curves of the two new panes, from a ``P.Data``."""

    def __init__(self, d):
        self.d = d

    def sigma(self, sample):
        return self.d.sigma(sample, 'onshell')

    def shape_ratio(self, num, den):
        """``n_num / n_den`` of the unit-area ``onshell`` shapes.

        A ratio of *shapes*: both curves were normalised to unit area before
        the division, so all rate information is already gone.  This is not a
        K-factor and must not be read as one.  The two samples are statistically
        independent, so the errors add in quadrature.
        """
        a, ae = self.d.shape(num, 'onshell')
        b, be = self.d.shape(den, 'onshell')
        r = a / b
        e = np.abs(r) * np.sqrt((ae / a) ** 2 + (be / b) ** 2)
        return r, e

    def weight(self, sm):
        """``w = sigma_int / sigma_sm``, both decayed and ``onshell``."""
        return self.sigma('eft_int') / self.sigma(sm)

    def sum_ratio(self, sm):
        """``(SM + interference) / SM``, cross-section weighted.

        ``[sigma_sm n_sm + sigma_int n_int] / [(sigma_sm + sigma_int) n_sm]``,
        i.e. the unit-area sum of the two contributions divided by the SM shape.
        The ``n_sm`` cancels analytically, leaving

            1 + [w/(1+w)] * (n_int/n_sm - 1)

        which is used as written: it makes the cancellation exact rather than
        approximate, and it keeps the shared SM measurement from being counted
        as two independent ones in the error.
        """
        rho, rho_e = self.shape_ratio('eft_int', sm)
        w = self.weight(sm)
        f = w / (1.0 + w)
        return 1.0 + f * (rho - 1.0), f * rho_e

    def none_agreement(self, other=None):
        """``(max |ratio-1|, chi2/ndf, median per-bin relative error)``.

        ``NONE_KEPT``'s ``none`` shape against ``other``'s (default the sample
        the drawn curve also covers).  Pass ``NONE_EXCLUDES`` to measure the
        NLO sample, which is exactly the comparison that decides the drawn
        curve may not claim to cover it.
        """
        a, ae = self.d.shape(NONE_KEPT, 'none')
        b, be = self.d.shape(other or NONE_STANDS_FOR, 'none')
        r = a / b
        e = np.abs(r) * np.sqrt((ae / a) ** 2 + (be / b) ** 2)
        chi2 = float(np.sum(((r - 1) / e) ** 2)) / self.d.nbins
        # the error quoted is the one the chi2 is measured against, i.e. the
        # error on the RATIO (the two samples' 0.48 % errors in quadrature),
        # not the error on either curve alone
        rel = float(np.median(np.abs(e / r)))
        return float(np.max(np.abs(r - 1))), chi2, rel


# --------------------------------------------------------------------------
def _label_E(sample, mode, plain=False):
    """The upper pane's legend entry: identifying, never explaining.

    The one ``none`` curve names the two samples it covers and stops there.  It
    does not name SM NLO, whose ``none`` shape is 8 % away and is not on the
    pane; nothing in the legend invites the reader to assume otherwise.
    """
    tex = not plain and P.USETEX
    name = P.SAMPLE_TEX if tex else P.SAMPLE_PLAIN
    mtag = P.MODE_TEX if tex else P.MODE_PLAIN
    if mode == 'none':
        covers = ' \\& ' if tex else ' & '
        return (covers.join(name[c] for c in NONE_COVERS)
                + ', %s' % mtag[mode])
    return '%s, %s' % (name[sample], mtag[mode])


def _step(ax, d, y, **kw):
    ax.step(d.edges, np.concatenate([y[:1], y]), where='pre', **kw)


def _autoscale(ax, series, pad_frac=0.12, top_extra=0.0):
    """Fit every point plus its error, then leave room above for the note."""
    lo = hi = 1.0
    for r, e in series:
        lo = min(lo, float((r - e).min()))
        hi = max(hi, float((r + e).max()))
    pad = pad_frac * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad + top_extra * (hi - lo))


# --------------------------------------------------------------------------
def make_figure_mg7(d, out):
    """The MG7 paper style, three panes."""
    mpl.rcParams.update(_MG7_RC)
    v = VarE(d)
    tx = P._tx

    # Same width as A--D; the height grows with the extra pane.
    fig = plt.figure(figsize=(7 * 0.75 * 1.35, 7 * 0.75 * 1.5 * 1.34))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.15, 1.15], hspace=0.07)
    ax = fig.add_subplot(gs[0])
    r1 = fig.add_subplot(gs[1], sharex=ax)
    r2 = fig.add_subplot(gs[2], sharex=ax)
    lo, hi = 0.0, float(np.pi)

    # --- upper pane ------------------------------------------------------
    # No free-floating text anywhere on this figure: axis labels and legends
    # only.  The parameter point (ctGRe = -1, Lambda = 1 TeV), the size of the
    # two LO `none' curves' agreement and the SM NLO provenance note all live in
    # README.md and numbers_E.txt instead.
    for sample, mode in CURVES:
        y, ye = d.shape(sample, mode)
        _step(ax, d, y, color=P.COLOR[sample], ls=P.LS[mode], lw=P.LW,
              label=_label_E(sample, mode), zorder=4)
        ax.errorbar(d.centres, y, yerr=ye, fmt='none',
                    ecolor=P.COLOR[sample], elinewidth=0.9, capsize=0,
                    zorder=4)

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
    ax.set_ylim(min(ymin, 0.15), ymax + 0.34 * (ymax - ymin))
    ax.legend(frameon=False, loc='upper left', fontsize=10.0,
              handlelength=2.8, borderaxespad=1.0, labelspacing=0.5)

    # --- pane 1: shape ratios, NOT a K-factor ----------------------------
    r1.axhline(1.0, color='black', lw=0.9, zorder=2)
    series = []
    for num, den in PANE1:
        r, e = v.shape_ratio(num, den)
        series.append((r, e))
        _step(r1, d, r, color=P.COLOR[num], ls='solid', lw=P.LW, zorder=4,
              label=tx(r'%s\,/\,%s' % (P.SAMPLE_TEX[num], P.SAMPLE_TEX[den]),
                       '%s / %s' % (P.SAMPLE_PLAIN[num],
                                    P.SAMPLE_PLAIN[den])))
        r1.errorbar(d.centres, r, yerr=e, fmt='none', ecolor=P.COLOR[num],
                    elinewidth=0.9, capsize=0, zorder=4)
    _autoscale(r1, series, top_extra=0.55)
    r1.set_ylabel(tx(r'shape ratio', 'shape ratio'), fontsize=11.5)
    r1.legend(frameon=False, loc='lower left', fontsize=8.5,
              handlelength=2.4, borderaxespad=0.5, ncol=2)
    # --- pane 2: the operator's effect on the SM prediction --------------
    r2.axhline(1.0, color='black', lw=0.9, zorder=2)
    series = []
    handles = []
    for sm in PANE2:
        r, e = v.sum_ratio(sm)
        series.append((r, e))
        _step(r2, d, r, color=P.COLOR[sm], ls='solid', lw=P.LW, zorder=4)
        r2.errorbar(d.centres, r, yerr=e, fmt='none', ecolor=P.COLOR[sm],
                    elinewidth=0.9, capsize=0, zorder=4)
        handles.append(Line2D(
            [], [], color=P.COLOR[sm], lw=P.LW,
            label=tx(r'$(\,$%s$\,+\,$int.$\,)\,/\,$%s'
                     % (P.SAMPLE_TEX[sm], P.SAMPLE_TEX[sm]),
                     '(%s + int.) / %s'
                     % (P.SAMPLE_PLAIN[sm], P.SAMPLE_PLAIN[sm]))))
    _autoscale(r2, series, pad_frac=0.20, top_extra=0.70)
    r2.set_ylabel(tx(r'$(\mathrm{SM}+\mathcal{O}_{tG})/\mathrm{SM}$',
                     '(SM + O_tG) / SM'), fontsize=11.5)
    r2.legend(handles=handles, frameon=False, loc='lower left', fontsize=8.5,
              handlelength=2.4, borderaxespad=0.5, ncol=2)
    r2.set_xlabel(tx(r'$\Delta\phi(e^-e^+)$ [rad]',
                     r'$\Delta\phi(e^-e^+)$ [rad]'))
    for rx in (r1, r2):
        rx.xaxis.set_major_locator(MultipleLocator(np.pi / 4))
        rx.xaxis.set_minor_locator(AutoMinorLocator(2))
        rx.yaxis.set_minor_locator(AutoMinorLocator())
        rx.set_xlim(lo, hi)
    r1.tick_params(labelbottom=False)
    r2.xaxis.set_major_formatter(P._pi_formatter())

    fig.subplots_adjust(left=0.145, right=0.975, top=0.988, bottom=0.078)
    base = os.path.join(out, 'smeft_fig5_%s' % TAG)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=300)
    plt.close(fig)
    return base


# --------------------------------------------------------------------------
def make_figure_user(d, out):
    """The user's own style, three panes."""
    mpl.rcParams.update(_USER_RC)
    v = VarE(d)

    fig = plt.figure(figsize=(6, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.15, 1.15],
                          hspace=U.HSPACE)
    ax = fig.add_subplot(gs[0])
    r1 = fig.add_subplot(gs[1], sharex=ax)
    r2 = fig.add_subplot(gs[2], sharex=ax)

    # As in the MG7 rendering: axis labels and legends, and nothing else.
    for sample, mode in CURVES:
        y, ye = d.shape(sample, mode)
        c = U.COLOR[sample]
        ax.step(d.edges, np.concatenate([y[:1], y]), where='pre', color=c,
                lw=1.0, alpha=U.STEP_ALPHA, zorder=3)
        ax.errorbar(d.centres, y, yerr=ye, fmt='o', ms=U.MS, color=c,
                    mfc=(c if U.FILLED[mode] else 'white'), mew=1.1,
                    label=_label_E(sample, mode, plain=True), zorder=4)

    ax.set_ylabel(r'$(1/\sigma)\,d\sigma/d\Delta\phi(e^-e^+)$  [1/rad]')
    ax.set_xlim(0.0, np.pi)
    ax.tick_params(labelbottom=False)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.34 * (ymax - ymin))
    ax.legend(loc='upper left', fontsize=8.5, ncol=1)

    r1.axhline(1.0, color='black', ls='--', lw=0.9, zorder=2)
    series = []
    for num, den in PANE1:
        r, e = v.shape_ratio(num, den)
        series.append((r, e))
        c = U.COLOR[num]
        r1.step(d.edges, np.concatenate([r[:1], r]), where='pre', color=c,
                lw=1.0, alpha=U.STEP_ALPHA, zorder=3)
        r1.errorbar(d.centres, r, yerr=e, fmt='o', ms=U.MS, color=c, zorder=4,
                    label='%s / %s' % (P.SAMPLE_PLAIN[num],
                                       P.SAMPLE_PLAIN[den]))
    r1.set_ylim(*choose_ylim_E(series))
    r1.set_ylabel('Shape ratio')
    r1.legend(loc='lower left', fontsize=7, ncol=2)
    r1.tick_params(labelbottom=False)

    r2.axhline(1.0, color='black', ls='--', lw=0.9, zorder=2)
    series = []
    for sm in PANE2:
        r, e = v.sum_ratio(sm)
        series.append((r, e))
        c = U.COLOR[sm]
        r2.step(d.edges, np.concatenate([r[:1], r]), where='pre', color=c,
                lw=1.0, alpha=U.STEP_ALPHA, zorder=3)
        r2.errorbar(d.centres, r, yerr=e, fmt='o', ms=U.MS, color=c, zorder=4,
                    label='(%s + int.) / %s'
                          % (P.SAMPLE_PLAIN[sm], P.SAMPLE_PLAIN[sm]))
    r2.set_ylim(*choose_ylim_E(series))
    r2.set_ylabel('(SM + $\\mathcal{O}_{tG}$) / SM')
    r2.legend(loc='lower left', fontsize=7, ncol=2)
    r2.set_xlabel(r'$\Delta\phi(e^-e^+)$ [rad]')
    r2.set_xlim(0.0, np.pi)
    U._pi_ticks(r2)

    fig.subplots_adjust(hspace=0.1, left=0.145, right=0.97,
                        bottom=0.075, top=0.985)
    base = os.path.join(out, 'smeft_fig5_%s' % TAG)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=U.DPI)
    plt.close(fig)
    return base


# --------------------------------------------------------------------------
def write_curves_E(d, path):
    """The numbers this variation actually draws, beside the figure."""
    v = VarE(d)
    out = {'edges': d.edges, 'centres': d.centres,
           'bin_width_rad': np.array(d.width),
           'ctGRe': np.array(-1.0), 'Lambda_GeV': np.array(1000.0)}
    for s, m in CURVES:
        y, ye = d.shape(s, m)
        out['%s_%s_shape' % (s, m)] = y
        out['%s_%s_shape_err' % (s, m)] = ye
    for s in ('eft_int', 'sm_lo', 'sm_nlo'):
        out['%s_sigma_onshell_pb' % s] = np.array(v.sigma(s))
    # The `none' shapes that are not drawn.  SM LO's is what the single drawn
    # curve also stands for; SM NLO's is the one it explicitly does NOT, and it
    # belongs here so that claim can be rechecked without regenerating anything.
    for s_ in (NONE_STANDS_FOR, NONE_EXCLUDES):
        y, ye = d.shape(s_, 'none')
        out['%s_none_shape_not_drawn' % s_] = y
        out['%s_none_shape_not_drawn_err' % s_] = ye
    for num, den in PANE1:
        r, e = v.shape_ratio(num, den)
        out['pane1_%s_over_%s' % (num, den)] = r
        out['pane1_%s_over_%s_err' % (num, den)] = e
    for sm in PANE2:
        r, e = v.sum_ratio(sm)
        out['pane2_%s_plus_int_over_%s' % (sm, sm)] = r
        out['pane2_%s_plus_int_over_%s_err' % (sm, sm)] = e
        out['pane2_w_%s' % sm] = np.array(v.weight(sm))
    np.savez(path, **out)
    return path


def write_numbers_E(d, fh=sys.stdout):
    p = lambda *a: print(*a, file=fh)
    v = VarE(d)
    setup = d.meta['setup']
    p('=' * 78)
    p('Fig. 5 variation E: one no-spin-correlation curve, two ratio panes')
    p('=' * 78)
    p(NLO_PROVENANCE_LONG)
    p('')
    p('observable   : %s' % setup['observable'])
    p('sqrt(s)      : %g GeV      scale: %s' % (setup['sqrt_s_GeV'],
                                                setup['scale']))
    p('PDF          : %s' % setup['pdf'])
    p('cuts         : %s' % setup['cuts'])
    p('sign         : %s' % setup['sign_convention'])
    p('parameter pt : ctGRe = %g, Lambda = %g GeV'
      % (d.meta['samples']['eft_int']['wilson_coefficients']['ctGRe'],
         d.meta['samples']['eft_int']['wilson_coefficients']
         ['LambdaSMEFT_GeV']))
    p('plot binning : %d uniform bins over [0, pi], width %.6f rad '
      '(%d stored bins each)' % (d.nbins, d.width, d.group))
    p('ratio panes  : spinmode = onshell throughout')
    p('')
    P.check_normalisation(d, fh)

    p('')
    p('=' * 78)
    p('upper pane: which spinmode = none curve is drawn')
    p('=' * 78)
    maxdev, chi2, rel = v.none_agreement()
    ndev, nchi2, nrel = v.none_agreement(NONE_EXCLUDES)
    p('  drawn        : %s, spinmode = none' % NONE_KEPT)
    p('  covers       : %s' % ', '.join(NONE_COVERS))
    p('  does NOT     : %s -- see the second row of the table below'
      % NONE_EXCLUDES)
    p('')
    p('  %-26s %12s %10s %12s' % ('pair of spinmode=none shapes',
                                  'max |r-1|', 'chi2/ndf', 'per-bin err'))
    for other, verdict in ((NONE_STANDS_FOR, 'covered by the drawn curve'),
                           (NONE_EXCLUDES, 'NOT covered -- 4x further away')):
        md, c2, rl = v.none_agreement(other)
        p('  %-26s %11.2f%% %10.2f %11.2f%%   %s'
          % ('%s / %s' % (NONE_KEPT, other), 100 * md, c2, 100 * rl, verdict))
    p('')
    p('  The SM NLO `none\' shape is %.1f %% away from the drawn one'
      % (100 * ndev))
    p('  (chi2/ndf %.1f), which confirms T110\'s 8 %%.  The cause is radiation,'
      % nchi2)
    p('  not spin: the extra emission changes the t t~ boost.  A single `none\'')
    p('  curve claiming to cover NLO as well would therefore be false, so the')
    p('  legend entry names only %s and says nothing'
      % ' and '.join(NONE_COVERS))
    p('  about SM NLO, which appears with its `onshell\' curve alone.')
    a, ae = d.shape(NONE_KEPT, 'none')
    b, be = d.shape(NONE_STANDS_FOR, 'none')
    c, ce = d.shape(NONE_EXCLUDES, 'none')
    p('')
    p('  n_events     : %s %d, %s %d -- same statistics, same %.2f %% error'
      % (NONE_KEPT, d.nevents(NONE_KEPT, 'none'), NONE_STANDS_FOR,
         d.nevents(NONE_STANDS_FOR, 'none'),
         100 * float(np.median(np.abs(be / b)))))
    p('')
    p('  %9s %12s %12s %12s %9s %9s'
      % ('phi/pi', 'EFT none', 'SM LO none', 'SM NLO none',
         'LO/EFT', 'NLO/EFT'))
    for i in range(d.nbins):
        p('  %9.3f %12.5f %12.5f %12.5f %8.4f %9.4f'
          % (d.centres[i] / np.pi, a[i], b[i], c[i],
             b[i] / a[i], c[i] / a[i]))

    p('')
    p('=' * 78)
    p('cross sections used as the pane-2 weights (decayed, onshell, pb)')
    p('=' * 78)
    for s in ('eft_int', 'sm_lo', 'sm_nlo'):
        p('  %-8s %12.6f' % (s, v.sigma(s)))
    for sm in PANE2:
        w = v.weight(sm)
        p('  w(%-6s) = sigma_int/sigma_SM = %.4f   ->  shrink factor '
          'w/(1+w) = %.4f' % (sm, w, w / (1 + w)))
    p('  LO -> NLO K-factor of these samples = %.4f  (NOT what pane 1 shows)'
      % (v.sigma('sm_nlo') / v.sigma('sm_lo')))

    p('')
    p('=' * 78)
    p('pane 1 (shape ratios of unit-area onshell curves -- not K-factors)')
    p('and pane 2 ((SM + interference)/SM, cross-section weighted)')
    p('=' * 78)
    p('%9s %19s %19s %19s %19s'
      % ('phi/pi', 'NLO/LO', 'SMEFT/LO', '(LO+int)/LO', '(NLO+int)/NLO'))
    p1 = [v.shape_ratio(n, dd) for n, dd in PANE1]
    p2 = [v.sum_ratio(sm) for sm in PANE2]
    for i in range(d.nbins):
        row = '%9.3f' % (d.centres[i] / np.pi)
        for r, e in p1 + p2:
            row += '   %7.4f +- %.4f' % (r[i], e[i])
        p(row)

    p('')
    p('the ends of every ratio curve, on the plotted binning:')
    names = ['NLO / LO           (pane 1)', 'SMEFT / LO         (pane 1)',
             '(LO + int) / LO    (pane 2)', '(NLO + int) / NLO  (pane 2)']
    for name, (r, e) in zip(names, p1 + p2):
        p('  %-28s first bin %+7.2f%% +- %.2f%%   last bin %+7.2f%% +- %.2f%%'
          '   max |dev| %5.2f%%'
          % (name, 100 * (r[0] - 1), 100 * e[0], 100 * (r[-1] - 1),
             100 * e[-1], 100 * np.max(np.abs(r - 1))))
    p('')
    p('  Pane 2 is pane 1 rescaled, exactly:')
    p('    (SM + int)/SM = 1 + [w/(1+w)] * (n_int/n_SM - 1)')
    p('  so it carries no information beyond pane 1 and the two cross')
    p('  sections -- but it is the pane with a physical magnitude, and that')
    p('  magnitude is proportional to c_tG/Lambda^2.  At c_tG = +1 the two')
    p('  pane-2 curves mirror about 1.')
    p('')
    p('  The NLO curve of pane 2 is smaller than the LO one because sigma_SM')
    p('  grows by the %.2f K-factor while the interference is only available'
      % (v.sigma('sm_nlo') / v.sigma('sm_lo')))
    p('  at LO, so w falls from %.3f to %.3f.  Read it as "an LO interference'
      % (v.weight('sm_lo'), v.weight('sm_nlo')))
    p('  against an NLO SM", not as "the operator matters less at NLO".')
    p('')
    p('=' * 78)
    p('where the provenance note and the parameter point live')
    p('=' * 78)
    p('  The figure carries NO text beyond its axis labels and legends, by')
    p('  request.  Everything below therefore exists ONLY here and in')
    p('  README.md, and whoever writes the caption must carry it across:')
    p('')
    p('    * the SM NLO provenance note, repeated below;')
    p('    * the parameter point ctGRe = %g, Lambda = %g GeV, without which'
      % (d.meta['samples']['eft_int']['wilson_coefficients']['ctGRe'],
         d.meta['samples']['eft_int']['wilson_coefficients']
         ['LambdaSMEFT_GeV']))
    p('      pane 2 has no scale at all -- its magnitude is proportional to')
    p('      c_tG/Lambda^2 and its two curves mirror about 1 for c_tG > 0;')
    p('    * the pane-2 weights w = %.4f (LO) and %.4f (NLO);'
      % (v.weight('sm_lo'), v.weight('sm_nlo')))
    p('    * that pane 1 is a SHAPE ratio and not a K-factor (the K-factor is')
    p('      %.2f);' % (v.sigma('sm_nlo') / v.sigma('sm_lo')))
    p('    * that the drawn `none\' curve covers the two LO samples only.')
    p('')
    p(NLO_PROVENANCE_LONG)


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
    ap.add_argument('--check-minus', dest='check_minus', action='store_true',
                    default=True)
    ap.add_argument('--no-check-minus', dest='check_minus',
                    action='store_false')
    args = ap.parse_args()

    d = P.Data(args.data, nbins=args.nbins)
    rows = P.check_normalisation(d, fh=open(os.devnull, 'w'))
    bad = [name for name, ok, _ in rows if not ok]
    if bad:
        raise SystemExit('normalisation check failed: %s' % '; '.join(bad))

    failures = []
    if args.style in ('mg7', 'both'):
        os.makedirs(args.out, exist_ok=True)
        base = make_figure_mg7(d, args.out)
        print('wrote %s.pdf / .png   (MG7 paper style)' % base)
        if args.check_minus:
            ok, detail = P.check_minus(base + '.pdf')
            print('   minus check: %s -- %s'
                  % ({True: 'ok', False: 'FAILED', None: 'n/a'}[ok], detail))
            if ok is False:
                failures.append(base)
        write_curves_E(d, os.path.join(args.out,
                                       'smeft_fig5_E_curves.npz'))
        with open(os.path.join(args.out, 'numbers_E.txt'), 'w') as fh:
            write_numbers_E(d, fh)
        print('wrote %s/smeft_fig5_E_curves.npz and numbers_E.txt' % args.out)

    if args.style in ('user', 'both'):
        os.makedirs(args.out_user, exist_ok=True)
        base = make_figure_user(d, args.out_user)
        print('wrote %s.pdf / .png   (user style)' % base)
        write_curves_E(d, os.path.join(args.out_user,
                                       'smeft_fig5_E_curves.npz'))
        with open(os.path.join(args.out_user, 'numbers_E.txt'), 'w') as fh:
            write_numbers_E(d, fh)
        print('wrote %s/smeft_fig5_E_curves.npz and numbers_E.txt'
              % args.out_user)

    write_numbers_E(d)
    print('')
    print('*** %s ***' % NLO_PROVENANCE_SHORT)

    if failures:
        raise SystemExit('the usetex minus sign was lost in: %s'
                         % ', '.join(failures))


if __name__ == '__main__':
    main()
