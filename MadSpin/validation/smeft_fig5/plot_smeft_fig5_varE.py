#!/usr/bin/env python3
r"""Variation ``E`` of the remade Fig. 5: variation ``B``'s upper pane with a
single no-spin-correlation curve, over **two** ratio panes.

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

**Upper pane.**  Variation ``B`` draws two ``spinmode = none`` curves, one for
the SMEFT interference and one for the SM at LO, and they are indistinguishable:
they agree to 2.1 % with a chi2/ndf of 1.9 against a 0.7 % per-bin error.  Only
**one** is drawn here -- the SMEFT one -- and it is labelled as standing for
both.

The SMEFT ``none`` curve is the one kept, for three reasons.  It is the curve
the published figure already had (variation ``A``'s curve set is exactly ``E``'s
upper pane), so ``E`` is a minimal edit of the published figure rather than a
new one.  It keeps the blue sample complete, solid *and* dashed, which is right
for a figure whose subject is the operator; dropping the SMEFT dash instead
would leave the operator with no "without spin correlations" reference at all.
And the two ``none`` samples have the same 1 M events and the same 0.48 % median
per-bin error, so nothing is lost by statistics either way.

The coincidence of the two ``none`` curves *is* this figure's main result -- it
is what says that the whole separation between the two solid curves is a
spin-correlation effect and not a shape difference the operator would produce
anyway.  Drawing one curve must not lose it, so the agreement is stated twice:
in the legend entry itself and in a note under the header block.

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

Health warning on the SM NLO sample
-----------------------------------
Both new panes use the SM NLO sample, and that sample has a known defect: its
MadSpin density matrices were evaluated at the *model default* parameters rather
than the run's, so the events were made at ``MT = 172.76``, ``WT = 1.33``,
``WZ = 2.4952``, ``WW = 2.085`` while the matrix-element directories held
``173 / 1.4915 / 2.4414 / 2.0476``.  See ``README.md``; the figure says so on
its face, and every NLO number in ``numbers_E.txt`` is flagged.
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

# The three curves of the upper pane: variation A's set, relabelled.  Colour
# names the sample and the dash names the spinmode, as in A--D.
CURVES = [('eft_int', 'onshell'), ('eft_int', 'none'), ('sm_lo', 'onshell')]

# The one no-spin-correlation curve that is drawn, and the one it stands for.
NONE_KEPT = 'eft_int'
NONE_STANDS_FOR = 'sm_lo'

# Ratio panes.  ``(numerator, denominator)`` in pane 1; ``(sm, interference)``
# in pane 2.  Everything ``onshell``.
PANE1 = [('sm_nlo', 'sm_lo'), ('eft_int', 'sm_lo')]
PANE2 = ['sm_lo', 'sm_nlo']

NLO_WARNING_SHORT = ('SM NLO density matrices ran at model defaults '
                     '(MT 173, WT 1.4915) -- see README')
NLO_WARNING_LONG = """\
!! HEALTH WARNING on every number and curve involving the SM NLO sample !!

    MadSpin's density path initialises its standalone matrix-element library
    from <madspin_me>/Cards/param_card.dat -- the card `output standalone'
    wrote from the MODEL DEFAULTS -- and never from the run's own card.  For
    PROC_sm_nlo the run used

        MT 172.76   WT 1.33   WZ 2.4952   WW 2.085

    while its ME directories held

        MT 173.0    WT 1.4915 WZ 2.4414   WW 2.0476

    so the spin-density matrices of the SM NLO sample were evaluated at a top
    mass 0.24 GeV too high and a top width 12 % too large.  The `onshell'
    curve of that sample, and therefore both ratio panes' NLO entries, are
    affected at an unquantified level.  The figure is drawn anyway because it
    is wanted, but the NLO curves must not be quoted as a result until the
    sample is regenerated.

    The two LO samples were audited for this report and are clear: the only
    parameter that differs between their event card and their ME card is
    alpha_s(M_Z) (0.1190025 against 0.1179).  At LO both the QCD production
    density matrix and the O_tG interference are proportional to g_s^4, one
    overall factor, and MadSpin normalises its accept/reject weight on
    Tr(rho_prod), so that factor cancels exactly.  The upper pane's two LO
    curves and pane 1's denominator are therefore sound.  (The `none' runs
    build no density matrices at all -- they have no madspin_me directory --
    so they cannot be affected.)"""


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

    def none_agreement(self):
        """``(max |ratio-1|, chi2/ndf, median per-bin relative error)``."""
        a, ae = self.d.shape(NONE_KEPT, 'none')
        b, be = self.d.shape(NONE_STANDS_FOR, 'none')
        r = a / b
        e = np.abs(r) * np.sqrt((ae / a) ** 2 + (be / b) ** 2)
        chi2 = float(np.sum(((r - 1) / e) ** 2)) / self.d.nbins
        rel = float(np.median(np.abs(ae / a)))
        return float(np.max(np.abs(r - 1))), chi2, rel


# --------------------------------------------------------------------------
def _step(ax, d, y, **kw):
    ax.step(d.edges, np.concatenate([y[:1], y]), where='pre', **kw)


def _autoscale(ax, series, pad_frac=0.12, floor=None):
    lo = hi = 1.0
    for r, e in series:
        lo = min(lo, float((r - e).min()))
        hi = max(hi, float((r + e).max()))
    pad = pad_frac * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    if floor is not None:
        ax.set_ylim(min(ax.get_ylim()[0], floor[0]),
                    max(ax.get_ylim()[1], floor[1]))


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

    maxdev, chi2, rel = v.none_agreement()

    # --- upper pane ------------------------------------------------------
    for sample, mode in CURVES:
        y, ye = d.shape(sample, mode)
        if mode == 'none':
            # The single no-spin-correlation curve.  The label says what it
            # stands for: the other sample's `none' shape lies on top of it.
            label = tx(r'\texttt{none} (both): %s' % P.SAMPLE_TEX[sample]
                       + '\n' + r'\quad and %s, agreeing to %.0f\%%'
                       % (P.SAMPLE_TEX[NONE_STANDS_FOR], 100 * maxdev),
                       'none (both): %s and %s, to %.0f%%'
                       % (P.SAMPLE_PLAIN[sample],
                          P.SAMPLE_PLAIN[NONE_STANDS_FOR], 100 * maxdev))
        else:
            label = P._label(sample, mode)
        _step(ax, d, y, color=P.COLOR[sample], ls=P.LS[mode], lw=P.LW,
              label=label, zorder=4)
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
    ax.set_ylim(min(ymin, 0.15), ymax + 0.72 * (ymax - ymin))

    ax.text(0.028, 0.974,
            tx(r'$pp \to t\bar t$ at $\sqrt{s} = 13$~TeV, '
               r'$\mu_R = \mu_F = 173$~GeV, no cuts',
               r'$pp \to t\bar t$ at $\sqrt{s}=13$ TeV, '
               r'$\mu_R=\mu_F=173$ GeV, no cuts'),
            transform=ax.transAxes, ha='left', va='top', fontsize=10.5)
    ax.text(0.028, 0.926,
            tx(r'$t \to W^+b\,(W^+\!\to e^+\nu_e)$, '
               r'$\bar t \to W^-\bar b\,(W^-\!\to e^-\bar\nu_e)$; '
               r'every curve normalised to unit area',
               r'$t\to W^+b(\to e^+\nu)$, $\bar t\to W^-\bar b(\to e^-\bar'
               r'\nu)$; every curve normalised to unit area'),
            transform=ax.transAxes, ha='left', va='top', fontsize=10.5)
    ax.text(0.028, 0.878,
            tx(r'interference at $c_{tG} = -1$, $\Lambda = 1$~TeV, i.e.\ '
               r'$-2\,\mathrm{Re}(\mathcal{M}^{*}_{\mathrm{SM}}'
               r'\mathcal{M}_{tG})$',
               r'interference at $c_{tG}=-1$, $\Lambda=1$ TeV, i.e. '
               r'$-2Re(M^*_{SM}M_{tG})$'),
            transform=ax.transAxes, ha='left', va='top', fontsize=10.5,
            color='blue')
    ax.text(0.028, 0.830,
            tx(r'one \texttt{none} curve is drawn for both samples: without '
               r'spin correlations they agree to'
               '\n'
               r'%.1f\%% ($\chi^2/\mathrm{ndf} = %.1f$ against a %.1f\%% '
               r'per-bin error) -- \emph{that} is the result'
               % (100 * maxdev, chi2, 100 * rel),
               'one none curve is drawn for both samples: without spin '
               'correlations they\nagree to %.1f%% (chi2/ndf = %.1f against '
               'a %.1f%% per-bin error) -- that is the result'
               % (100 * maxdev, chi2, 100 * rel)),
            transform=ax.transAxes, ha='left', va='top', fontsize=9.5,
            color='0.25')

    ax.legend(frameon=False, loc='upper left', fontsize=10.0,
              handlelength=2.8, borderaxespad=0.9, labelspacing=0.65,
              bbox_to_anchor=(0.015, 0.735))

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
    _autoscale(r1, series)
    r1.set_ylabel(tx(r'shape ratio', 'shape ratio'), fontsize=11.5)
    r1.legend(frameon=False, loc='lower left', fontsize=8.5,
              handlelength=2.4, borderaxespad=0.5, ncol=2)
    r1.text(0.982, 0.93,
            tx(r'\textbf{not a $K$-factor}: both curves are unit-area shapes, '
               r'so the rates are already divided out'
               '\n'
               r'(the LO$\,\to\,$NLO $K$-factor of these samples is '
               r'$%.2f$ and appears nowhere on this figure)'
               % (v.sigma('sm_nlo') / v.sigma('sm_lo')),
               'NOT a K-factor: both curves are unit-area shapes, so the '
               'rates are already divided out\n(the LO->NLO K-factor of '
               'these samples is %.2f and appears nowhere on this figure)'
               % (v.sigma('sm_nlo') / v.sigma('sm_lo'))),
            transform=r1.transAxes, ha='right', va='top', fontsize=8.0,
            color='0.3')

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
            label=tx(r'$(\,$%s$\,+\,$int.$\,)\,/\,$%s, $w = %.3f$'
                     % (P.SAMPLE_TEX[sm], P.SAMPLE_TEX[sm], v.weight(sm)),
                     '(%s + int.) / %s, w = %.3f'
                     % (P.SAMPLE_PLAIN[sm], P.SAMPLE_PLAIN[sm],
                        v.weight(sm)))))
    _autoscale(r2, series, pad_frac=0.35)
    r2.set_ylabel(tx(r'$(\mathrm{SM}+\mathcal{O}_{tG})/\mathrm{SM}$',
                     '(SM + O_tG) / SM'), fontsize=11.5)
    r2.legend(handles=handles, frameon=False, loc='lower left', fontsize=8.5,
              handlelength=2.4, borderaxespad=0.5, ncol=2)
    r2.text(0.982, 0.93,
            tx(r'cross-section weighted: '
               r'$(\sigma_{\rm SM}n_{\rm SM}+\sigma_{\rm int}n_{\rm int})'
               r'/(\sigma_{\rm SM}+\sigma_{\rm int})$, '
               r'$w=\sigma_{\rm int}/\sigma_{\rm SM}$;'
               '\n'
               r'this pane scales with $c_{tG}/\Lambda^2$ and mirrors about 1 '
               r'for $c_{tG}>0$. Interference is LO in both curves.',
               'cross-section weighted: (s_SM n_SM + s_int n_int)/'
               '(s_SM + s_int), w = s_int/s_SM;\nthis pane scales with '
               'c_tG/Lambda^2 and mirrors about 1 for c_tG > 0. '
               'Interference is LO in both curves.'),
            transform=r2.transAxes, ha='right', va='top', fontsize=8.0,
            color='0.3')

    r2.set_xlabel(tx(r'$\Delta\phi(e^-e^+)$ [rad]',
                     r'$\Delta\phi(e^-e^+)$ [rad]'))
    for rx in (r1, r2):
        rx.xaxis.set_major_locator(MultipleLocator(np.pi / 4))
        rx.xaxis.set_minor_locator(AutoMinorLocator(2))
        rx.yaxis.set_minor_locator(AutoMinorLocator())
        rx.set_xlim(lo, hi)
    r1.tick_params(labelbottom=False)
    r2.xaxis.set_major_formatter(P._pi_formatter())

    # The NLO caveat, on the face of the figure.
    fig.text(0.5, 0.006,
             tx(r'\textbf{Caveat:} %s' % NLO_WARNING_SHORT.replace('%', r'\%'),
                'Caveat: ' + NLO_WARNING_SHORT),
             ha='center', va='bottom', fontsize=8.0, color='red')

    fig.subplots_adjust(left=0.145, right=0.975, top=0.988, bottom=0.093)
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
    maxdev, chi2, rel = v.none_agreement()

    fig = plt.figure(figsize=(6, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.15, 1.15],
                          hspace=U.HSPACE)
    ax = fig.add_subplot(gs[0])
    r1 = fig.add_subplot(gs[1], sharex=ax)
    r2 = fig.add_subplot(gs[2], sharex=ax)

    for sample, mode in CURVES:
        y, ye = d.shape(sample, mode)
        c = U.COLOR[sample]
        if mode == 'none':
            label = ('none (both): %s\nand %s, agreeing to %.0f%%'
                     % (P.SAMPLE_PLAIN[sample],
                        P.SAMPLE_PLAIN[NONE_STANDS_FOR], 100 * maxdev))
        else:
            label = '%s, %s' % (P.SAMPLE_PLAIN[sample], P.MODE_PLAIN[mode])
        ax.step(d.edges, np.concatenate([y[:1], y]), where='pre', color=c,
                lw=1.0, alpha=U.STEP_ALPHA, zorder=3)
        ax.errorbar(d.centres, y, yerr=ye, fmt='o', ms=U.MS, color=c,
                    mfc=(c if U.FILLED[mode] else 'white'), mew=1.1,
                    label=label, zorder=4)

    ax.set_ylabel(r'$(1/\sigma)\,d\sigma/d\Delta\phi(e^-e^+)$  [1/rad]')
    ax.set_xlim(0.0, np.pi)
    ax.tick_params(labelbottom=False)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.75 * (ymax - ymin))
    ax.legend(loc='upper left', fontsize=8.0, ncol=1)
    ax.set_title(r'$pp\to t\bar{t}$, 13 TeV, $\mu_R=\mu_F=173$ GeV, no cuts; '
                 'all curves normalised to unit area', fontsize=9)
    ax.text(0.985, 0.965,
            'interference at $c_{tG}=-1$, $\\Lambda=1$ TeV,\n'
            'i.e. $-2\\,\\mathrm{Re}(\\mathcal{M}^*_{SM}\\mathcal{M}_{tG})$',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color=U.COLOR['eft_int'])
    ax.text(0.985, 0.855,
            'one "none" curve for both samples:\n'
            'they agree to %.1f%% ($\\chi^2$/ndf = %.1f,\n'
            'per-bin error %.1f%%)' % (100 * maxdev, chi2, 100 * rel),
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color='0.25')

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
    r1.set_ylim(*U.choose_ratio_ylim(series))
    r1.set_ylabel('Shape ratio')
    r1.legend(loc='lower left', fontsize=7, ncol=2)
    r1.text(0.985, 0.94,
            'shapes only (unit area) -- not a K-factor; the LO$\\to$NLO\n'
            'K-factor is %.2f and is not shown'
            % (v.sigma('sm_nlo') / v.sigma('sm_lo')),
            transform=r1.transAxes, ha='right', va='top', fontsize=7,
            color='0.3')
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
                    label='(%s + int.) / %s, w = %.3f'
                          % (P.SAMPLE_PLAIN[sm], P.SAMPLE_PLAIN[sm],
                             v.weight(sm)))
    r2.set_ylim(*U.choose_ratio_ylim(series))
    r2.set_ylabel('(SM + $\\mathcal{O}_{tG}$) / SM')
    r2.legend(loc='lower left', fontsize=7, ncol=2)
    r2.text(0.985, 0.94,
            'cross-section weighted, $w=\\sigma_{int}/\\sigma_{SM}$;\n'
            'scales with $c_{tG}/\\Lambda^2$, mirrors about 1 for $c_{tG}>0$',
            transform=r2.transAxes, ha='right', va='top', fontsize=7,
            color='0.3')
    r2.set_xlabel(r'$\Delta\phi(e^-e^+)$ [rad]')
    r2.set_xlim(0.0, np.pi)
    U._pi_ticks(r2)

    fig.text(0.5, 0.004, 'Caveat: ' + NLO_WARNING_SHORT, ha='center',
             va='bottom', fontsize=7, color='red')
    fig.subplots_adjust(hspace=0.1, left=0.145, right=0.97,
                        bottom=0.085, top=0.945)
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
    # the SM LO `none' shape is not drawn, but it is what the single drawn
    # `none' curve stands for, so it belongs in the file
    y, ye = d.shape(NONE_STANDS_FOR, 'none')
    out['%s_none_shape_not_drawn' % NONE_STANDS_FOR] = y
    out['%s_none_shape_not_drawn_err' % NONE_STANDS_FOR] = ye
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
    p(NLO_WARNING_LONG)
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
    p('  drawn        : %s, spinmode = none' % NONE_KEPT)
    p('  stands for   : %s, spinmode = none (not drawn)' % NONE_STANDS_FOR)
    p('  agreement    : max |ratio - 1| = %.2f %%, chi2/ndf = %.2f, against a'
      % (100 * maxdev, chi2))
    p('                 median per-bin relative error of %.2f %%'
      % (100 * rel))
    a, ae = d.shape(NONE_KEPT, 'none')
    b, be = d.shape(NONE_STANDS_FOR, 'none')
    p('  n_events     : %s %d, %s %d -- same statistics, same %.2f %% error'
      % (NONE_KEPT, d.nevents(NONE_KEPT, 'none'), NONE_STANDS_FOR,
         d.nevents(NONE_STANDS_FOR, 'none'),
         100 * float(np.median(np.abs(be / b)))))
    p('')
    p('  %9s %13s %13s %11s' % ('phi/pi', 'EFT none', 'SM LO none', 'ratio'))
    for i in range(d.nbins):
        p('  %9.3f %13.5f %13.5f %11.5f'
          % (d.centres[i] / np.pi, a[i], b[i], a[i] / b[i]))

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
    p(NLO_WARNING_LONG)


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
    print('*** %s ***' % NLO_WARNING_SHORT)

    if failures:
        raise SystemExit('the usetex minus sign was lost in: %s'
                         % ', '.join(failures))


if __name__ == '__main__':
    main()
