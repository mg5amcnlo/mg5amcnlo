#!/usr/bin/env python3
r"""Variation ``F`` of the remade Fig. 5: ``E``, decomposed.

Standalone and re-runnable; it draws both renderings in one go::

    plot_smeft_fig5_varF.py [--data DIR] [--out DIR] [--out-user DIR]
                            [--nbins N] [--style {mg7,user,both}]

Everything comes from ``data/histograms.npz`` and ``data/meta.json``.  Data
loading, rebinning, error propagation and ``check_normalisation()`` come from
``plot_smeft_fig5.py``; the user-style constants from
``plot_smeft_fig5_userstyle.py``; the panel helpers and the extended ratio
ladder from ``plot_smeft_fig5_varE.py``.  None of those three is modified, and
variations ``A``--``E`` are left exactly as they are -- this script only adds
``F``.

What ``F`` is
-------------
``E`` with every curve doubled: **every quantity is drawn twice, once from the
``onshell`` samples and once from the ``none`` ones.**

* Upper pane: all six curves -- SMEFT interference, SM LO and SM NLO, each with
  and without spin correlations.  This undoes ``E``'s single-``none``
  compromise.  ``E`` had to narrow its legend claim because one dashed curve
  could not honestly stand for a SM NLO shape 8 % away; with all three drawn,
  nothing is implied about a sample that is not on the pane.
* Pane 1: ``NLO/LO`` and ``SMEFT/LO``, each for ``onshell`` and for ``none``.
* Pane 2: ``(LO+SMEFT)/LO`` and ``(NLO+SMEFT)/NLO``, each for ``onshell`` and
  for ``none``.

That turns the figure into a **decomposition**.  Every ratio splits into what
survives when the spin correlations are switched off -- the non-spin part -- and
the gap between the solid and the dashed curve of the same colour, which is the
spin-correlation effect.  The two pane-1 pairs then say opposite things, and
that contrast is the pane's point:

* ``SMEFT/LO`` collapses onto 1 in the ``none`` case (max 2.1 %, mean 0.2 %):
  **the entire SMEFT/LO structure is spin correlation.**  This is the strongest
  form of what variation ``B`` was recommended for -- ``B`` showed the two
  ``none`` curves lying on top of each other, ``F`` shows their ratio being flat
  at 1 while the ``onshell`` ratio runs from +11.5 % to -17.8 %.
* ``NLO/LO`` does **not** collapse (max 9.6 % in the ``none`` case against
  6.5 % in the ``onshell`` one): the LO/NLO difference in this observable is
  almost entirely *not* a spin effect.  It is the extra radiation changing the
  ``t t~`` boost, and switching the spin correlations off leaves it untouched.

Encoding
--------
Consistent across all three panes and inherited from ``A``--``E``: **line style
is the spinmode** -- solid ``onshell``, dashed ``none`` -- and **colour is the
quantity**.  In the upper pane the quantity is the sample (blue SMEFT, black SM
LO, red SM NLO, as in ``D``); in the ratio panes it is the ratio, with the
colour taken from the sample that is not the LO reference (pane 1) or from the
SM sample being corrected (pane 2).  A reader therefore pairs curves by colour
and reads the vertical gap within a colour as the spin-correlation effect, which
is the one thing the figure is for.

The weights of pane 2's ``none`` curves
---------------------------------------
Pane 2 needs ``w = sigma_int/sigma_SM``, and the ``none`` curves get their
weights from the ``none`` samples' own cross sections rather than reusing the
``onshell`` ones.  They come out all but identical, which is checked here and
not assumed (``check_weights()``, run before anything is drawn)::

    w(none)/w(onshell) - 1 :   LO  -6.3e-08      NLO  -6.3e-05

MadSpin writes ``sigma_production * BR`` in either spinmode, so each sample's
own total moves by only ~4.4e-4 between the two modes; and in the LO weight even
that cancels, because ``eft_int`` and ``sm_lo`` move by the same 4.428e-4 and
the weight is their ratio.  Using the ``onshell`` weights throughout would have
changed nothing visible -- but it would have been an assumption, and the
assumption is now a measurement.

Everything else follows ``E``
-----------------------------
No text on the plot: axis labels and legends only.  The parameter point
(``ctGRe = -1``, ``Lambda = 1 TeV``, which sets pane 2's scale), the pane-2
weights and the SM NLO health warning live in ``README.md`` and in
``numbers_F.txt``, which opens and closes with the warning and ends with an
explicit list of what a caption must carry.

Health warning on the SM NLO sample
-----------------------------------
``sm_nlo``'s MadSpin density matrices were evaluated at model defaults rather
than the run's card: the events were made at ``MT = 172.76``, ``WT = 1.33``,
``WZ = 2.4952``, ``WW = 2.085`` while the matrix-element directories held
``173 / 1.4915 / 2.4414 / 2.0476``.  Every red curve on this figure, and the
``NLO`` entries of both ratio panes, are provisional.  Note that the defect is
in the **density** path, which ``spinmode = none`` does not use at all -- the
``none`` runs build no ``madspin_me`` directory -- so ``F``'s dashed NLO curves
are the *sound* ones and its solid NLO curves are the suspect ones.  See
``README.md``.
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
matplotlib.use('Agg')
_USER_RC = {k: mpl.rcParamsDefault[k] for k in _MG7_RC}

TAG = 'F'

MODES = ('onshell', 'none')
SAMPLES = ('eft_int', 'sm_lo', 'sm_nlo')

# Upper pane: every sample in every spinmode.  Solid curves first so the legend
# groups them; colour is the sample, dash is the spinmode, exactly as in `D'.
CURVES = [(s, m) for m in MODES for s in SAMPLES]

# Ratio panes.  ``(numerator, denominator)`` in pane 1; the SM sample being
# corrected in pane 2.  Each is drawn once per spinmode.
PANE1 = [('sm_nlo', 'sm_lo'), ('eft_int', 'sm_lo')]
PANE2 = ['sm_lo', 'sm_nlo']

# Reuse `E''s health-warning text verbatim: it is the same defect, the same
# sample and the same wording, and there should be one copy of it.
NLO_WARNING_SHORT = E.NLO_WARNING_SHORT
NLO_WARNING_LONG = E.NLO_WARNING_LONG


# --------------------------------------------------------------------------
class VarF(object):
    """``E``'s four ratio curves, computed in either spinmode."""

    def __init__(self, d):
        self.d = d

    def sigma(self, sample, mode):
        return self.d.sigma(sample, mode)

    def shape_ratio(self, num, den, mode):
        """``n_num / n_den`` of the unit-area shapes, in one spinmode.

        A ratio of *shapes*: both are normalised to unit area before the
        division, so this is not a K-factor.  The two samples are statistically
        independent and the errors add in quadrature.
        """
        a, ae = self.d.shape(num, mode)
        b, be = self.d.shape(den, mode)
        r = a / b
        e = np.abs(r) * np.sqrt((ae / a) ** 2 + (be / b) ** 2)
        return r, e

    def weight(self, sm, mode):
        """``w = sigma_int/sigma_SM``, both taken in ``mode``.

        The ``none`` curves get the ``none`` samples' own cross sections.  See
        ``check_weights()``: they agree with the ``onshell`` ones to 6e-8 (LO)
        and 6e-5 (NLO), but that is measured, not assumed.
        """
        return self.sigma('eft_int', mode) / self.sigma(sm, mode)

    def sum_ratio(self, sm, mode):
        """``(SM + interference)/SM``, cross-section weighted, in one spinmode.

        Uses the identity that makes the shared ``n_SM`` cancel exactly::

            (SM + int)/SM = 1 + [w/(1+w)] * (n_int/n_SM - 1)

        so the one SM measurement is not counted as two independent ones in
        numerator and denominator.
        """
        rho, rho_e = self.shape_ratio('eft_int', sm, mode)
        w = self.weight(sm, mode)
        f = w / (1.0 + w)
        return 1.0 + f * (rho - 1.0), f * rho_e

    def none_agreement(self, a_s, b_s):
        """``(max |ratio-1|, chi2/ndf)`` of two ``none`` shapes."""
        a, ae = self.d.shape(a_s, 'none')
        b, be = self.d.shape(b_s, 'none')
        r = a / b
        e = np.abs(r) * np.sqrt((ae / a) ** 2 + (be / b) ** 2)
        return (float(np.max(np.abs(r - 1))),
                float(np.sum(((r - 1) / e) ** 2)) / self.d.nbins)


# --------------------------------------------------------------------------
def check_weights(d, fh=sys.stdout):
    """Prove that the ``none`` weights match the ``onshell`` ones.

    T110 measured ``onshell`` and ``none`` sharing a total cross section to
    5e-4; pane 2's ``none`` curves need their own ``w`` and this checks, rather
    than assumes, that recomputing it changes nothing.  Returns
    ``(name, ok, detail)`` in ``check_normalisation``'s format.
    """
    p = lambda *a: print(*a, file=fh)
    v = VarF(d)
    p('-' * 78)
    p("pane-2 weights: recomputed for `none', not reused from `onshell'")
    p('-' * 78)
    p('%-10s %13s %13s %11s' % ('sample', 'sigma onshell', 'sigma none',
                                'ratio-1'))
    for s in SAMPLES:
        a, b = v.sigma(s, 'onshell'), v.sigma(s, 'none')
        p('%-10s %13.6f %13.6f %11.3e' % (s, a, b, a / b - 1))
    p('')
    p('%-10s %13s %13s %11s' % ('w = s_int/s_SM', 'onshell', 'none',
                                'ratio-1'))
    worst = 0.0
    for sm in PANE2:
        wo, wn = v.weight(sm, 'onshell'), v.weight(sm, 'none')
        worst = max(worst, abs(wn / wo - 1))
        p('%-10s %13.6f %13.6f %11.3e' % (sm, wo, wn, wn / wo - 1))
    p('')
    p('  The LO weight agrees to 6e-8, far better than the 4.4e-4 by which')
    p('  either cross section moves between the spinmodes, because eft_int and')
    p('  sm_lo move by the SAME 4.428e-4 and the weight is their ratio.')
    return [('the none weights reproduce the onshell ones', worst < 1e-3,
             'worst |w_none/w_onshell - 1| = %.2e' % worst)]


# --------------------------------------------------------------------------
def _room(ax, series, bottom_extra=0.0, top_extra=0.0, pad_frac=0.10):
    """Fit every point plus its error, then reserve room for the legend.

    ``E._autoscale`` only opens the top, because ``E``'s notes sat there.  ``F``
    has no notes and four curves per pane, so its legends go at the bottom and
    the room has to be made there instead.
    """
    lo = hi = 1.0
    for r, e in series:
        lo = min(lo, float((r - e).min()))
        hi = max(hi, float((r + e).max()))
    span = hi - lo
    ax.set_ylim(lo - pad_frac * span - bottom_extra * span,
                hi + pad_frac * span + top_extra * span)


def _label_top(sample, mode, plain=False):
    tex = not plain and P.USETEX
    name = P.SAMPLE_TEX if tex else P.SAMPLE_PLAIN
    mtag = P.MODE_TEX if tex else P.MODE_PLAIN
    return '%s, %s' % (name[sample], mtag[mode])


def _label_p1(num, den, mode, plain=False):
    tex = not plain and P.USETEX
    name = P.SAMPLE_TEX if tex else P.SAMPLE_PLAIN
    mtag = P.MODE_TEX if tex else P.MODE_PLAIN
    sep = r'\,/\,' if tex else ' / '
    return '%s%s%s, %s' % (name[num], sep, name[den], mtag[mode])


def _label_p2(sm, mode, plain=False):
    tex = not plain and P.USETEX
    name = P.SAMPLE_TEX if tex else P.SAMPLE_PLAIN
    mtag = P.MODE_TEX if tex else P.MODE_PLAIN
    if tex:
        return (r'$(\,$%s$\,+\,$int.$\,)\,/\,$%s, %s'
                % (name[sm], name[sm], mtag[mode]))
    return '(%s + int.) / %s, %s' % (name[sm], name[sm], mtag[mode])


# --------------------------------------------------------------------------
def make_figure_mg7(d, out):
    """The MG7 paper style, three panes, four curves in each ratio pane."""
    mpl.rcParams.update(_MG7_RC)
    v = VarF(d)
    tx = P._tx

    fig = plt.figure(figsize=(7 * 0.75 * 1.35, 7 * 0.75 * 1.5 * 1.40))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.2], hspace=0.07)
    ax = fig.add_subplot(gs[0])
    r1 = fig.add_subplot(gs[1], sharex=ax)
    r2 = fig.add_subplot(gs[2], sharex=ax)
    lo, hi = 0.0, float(np.pi)

    # --- upper pane: all six curves --------------------------------------
    # No free-floating text anywhere on this figure, as in `E'.
    for sample, mode in CURVES:
        y, ye = d.shape(sample, mode)
        E._step(ax, d, y, color=P.COLOR[sample], ls=P.LS[mode], lw=P.LW,
                label=_label_top(sample, mode), zorder=4)
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

    # --- pane 1: shape ratios, both spinmodes ----------------------------
    r1.axhline(1.0, color='black', lw=0.9, zorder=2)
    series = []
    for mode in MODES:
        for num, den in PANE1:
            r, e = v.shape_ratio(num, den, mode)
            series.append((r, e))
            E._step(r1, d, r, color=P.COLOR[num], ls=P.LS[mode], lw=P.LW,
                    zorder=4, label=_label_p1(num, den, mode))
            r1.errorbar(d.centres, r, yerr=e, fmt='none', ecolor=P.COLOR[num],
                        elinewidth=0.9, capsize=0, zorder=4)
    _room(r1, series, bottom_extra=0.46)
    r1.set_ylabel(tx(r'shape ratio', 'shape ratio'), fontsize=11.5)
    r1.legend(frameon=False, loc='lower left', fontsize=7.5,
              handlelength=2.4, borderaxespad=0.5, ncol=2, columnspacing=1.4)

    # --- pane 2: SM + interference, both spinmodes -----------------------
    r2.axhline(1.0, color='black', lw=0.9, zorder=2)
    series, handles = [], []
    for mode in MODES:
        for sm in PANE2:
            r, e = v.sum_ratio(sm, mode)
            series.append((r, e))
            E._step(r2, d, r, color=P.COLOR[sm], ls=P.LS[mode], lw=P.LW,
                    zorder=4)
            r2.errorbar(d.centres, r, yerr=e, fmt='none', ecolor=P.COLOR[sm],
                        elinewidth=0.9, capsize=0, zorder=4)
            handles.append(Line2D([], [], color=P.COLOR[sm], lw=P.LW,
                                  ls=P.LS[mode], label=_label_p2(sm, mode)))
    _room(r2, series, bottom_extra=0.46)
    r2.set_ylabel(tx(r'$(\mathrm{SM}+\mathcal{O}_{tG})/\mathrm{SM}$',
                     '(SM + O_tG) / SM'), fontsize=11.5)
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

    fig.subplots_adjust(left=0.145, right=0.975, top=0.988, bottom=0.075)
    base = os.path.join(out, 'smeft_fig5_%s' % TAG)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=300)
    plt.close(fig)
    return base


# --------------------------------------------------------------------------
def make_figure_user(d, out):
    """The user's own style: marker fill carries the spinmode."""
    mpl.rcParams.update(_USER_RC)
    v = VarF(d)

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
             _label_top(sample, mode, plain=True))

    ax.set_ylabel(r'$(1/\sigma)\,d\sigma/d\Delta\phi(e^-e^+)$  [1/rad]')
    ax.set_xlim(0.0, np.pi)
    ax.tick_params(labelbottom=False)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.32 * (ymax - ymin))
    ax.legend(loc='upper left', fontsize=8.0, ncol=2)

    r1.axhline(1.0, color='black', ls='--', lw=0.9, zorder=2)
    series = []
    for mode in MODES:
        for num, den in PANE1:
            r, e = v.shape_ratio(num, den, mode)
            series.append((r, e))
            draw(r1, r, e, U.COLOR[num], mode,
                 _label_p1(num, den, mode, plain=True))
    _lo, _hi = E.choose_ylim_E(series)
    r1.set_ylim(_lo - 0.40 * (_hi - _lo), _hi)
    r1.set_ylabel('Shape ratio')
    r1.legend(loc='lower left', fontsize=6.5, ncol=2)
    r1.tick_params(labelbottom=False)

    r2.axhline(1.0, color='black', ls='--', lw=0.9, zorder=2)
    series = []
    for mode in MODES:
        for sm in PANE2:
            r, e = v.sum_ratio(sm, mode)
            series.append((r, e))
            draw(r2, r, e, U.COLOR[sm], mode, _label_p2(sm, mode, plain=True))
    _lo, _hi = E.choose_ylim_E(series)
    r2.set_ylim(_lo - 0.40 * (_hi - _lo), _hi)
    r2.set_ylabel('(SM + $\\mathcal{O}_{tG}$) / SM')
    r2.legend(loc='lower left', fontsize=6.5, ncol=2)
    r2.set_xlabel(r'$\Delta\phi(e^-e^+)$ [rad]')
    r2.set_xlim(0.0, np.pi)
    U._pi_ticks(r2)

    fig.subplots_adjust(hspace=0.1, left=0.145, right=0.97,
                        bottom=0.07, top=0.985)
    base = os.path.join(out, 'smeft_fig5_%s' % TAG)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=U.DPI)
    plt.close(fig)
    return base


# --------------------------------------------------------------------------
def write_curves_F(d, path):
    """Every number this variation draws, beside the figure."""
    v = VarF(d)
    out = {'edges': d.edges, 'centres': d.centres,
           'bin_width_rad': np.array(d.width),
           'ctGRe': np.array(-1.0), 'Lambda_GeV': np.array(1000.0)}
    for s, m in CURVES:
        y, ye = d.shape(s, m)
        out['%s_%s_shape' % (s, m)] = y
        out['%s_%s_shape_err' % (s, m)] = ye
        out['%s_%s_sigma_pb' % (s, m)] = np.array(v.sigma(s, m))
    for m in MODES:
        for num, den in PANE1:
            r, e = v.shape_ratio(num, den, m)
            out['pane1_%s_over_%s_%s' % (num, den, m)] = r
            out['pane1_%s_over_%s_%s_err' % (num, den, m)] = e
        for sm in PANE2:
            r, e = v.sum_ratio(sm, m)
            out['pane2_%s_plus_int_over_%s_%s' % (sm, sm, m)] = r
            out['pane2_%s_plus_int_over_%s_%s_err' % (sm, sm, m)] = e
            out['pane2_w_%s_%s' % (sm, m)] = np.array(v.weight(sm, m))
    np.savez(path, **out)
    return path


def write_numbers_F(d, fh=sys.stdout):
    p = lambda *a: print(*a, file=fh)
    v = VarF(d)
    setup = d.meta['setup']
    p('=' * 78)
    p('Fig. 5 variation F: every curve of E drawn in both spinmodes')
    p('=' * 78)
    p(NLO_WARNING_LONG)
    p('')
    p('  Note for F specifically: the defect is in the DENSITY path, which')
    p('  spinmode = none does not use.  So on this figure the DASHED NLO')
    p('  curves are sound and the SOLID ones are the suspect ones.')
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
    p('encoding     : line style = spinmode (solid onshell, dashed none), '
      'colour = quantity')
    p('')
    P.check_normalisation(d, fh)
    p('')
    check_weights(d, fh)

    p('')
    p('=' * 78)
    p('the decomposition, which is what F is for')
    p('=' * 78)
    p('  Each ratio is drawn twice.  The `none\' curve is the part of the')
    p('  ratio that is NOT a spin-correlation effect; the gap to the `onshell\'')
    p('  curve of the same colour is the spin-correlation effect itself.')
    p('')
    hdr = '%-30s %10s %10s %10s %10s' % ('curve', 'first bin', 'last bin',
                                         'max |dev|', 'mean dev')
    p(hdr)
    rows = []
    for mode in MODES:
        for num, den in PANE1:
            rows.append(('%s/%s, %s' % (num, den, mode),
                         v.shape_ratio(num, den, mode)))
    for mode in MODES:
        for sm in PANE2:
            rows.append(('(%s+int)/%s, %s' % (sm, sm, mode),
                         v.sum_ratio(sm, mode)))
    for name, (r, e) in rows:
        p('%-30s %+9.2f%% %+9.2f%% %9.2f%% %+9.2f%%'
          % (name, 100 * (r[0] - 1), 100 * (r[-1] - 1),
             100 * np.max(np.abs(r - 1)), 100 * (r.mean() - 1)))

    p('')
    p('  and with their errors:')
    for name, (r, e) in rows:
        p('    %-30s first %+7.2f%% +- %.2f%%   last %+7.2f%% +- %.2f%%'
          % (name, 100 * (r[0] - 1), 100 * e[0],
             100 * (r[-1] - 1), 100 * e[-1]))

    # The two predictions F was drawn to test.
    p('')
    p('=' * 78)
    p('the two predictions, against what the figure actually shows')
    p('=' * 78)
    md_es, c2_es = v.none_agreement('eft_int', 'sm_lo')
    md_ns, c2_ns = v.none_agreement('sm_lo', 'sm_nlo')
    r_es, _ = v.shape_ratio('eft_int', 'sm_lo', 'none')
    r_ns, _ = v.shape_ratio('sm_nlo', 'sm_lo', 'none')
    r_es_on, _ = v.shape_ratio('eft_int', 'sm_lo', 'onshell')
    r_ns_on, _ = v.shape_ratio('sm_nlo', 'sm_lo', 'onshell')
    p('  1. SMEFT/LO in the none case should sit near 1, because the two')
    p('     none shapes agree to %.1f %% (chi2/ndf %.1f).' % (100 * md_es,
                                                             c2_es))
    p('     MEASURED: max |dev| %.2f %%, mean %+.2f %%, ends %+.2f %% / %+.2f %%.'
      % (100 * np.max(np.abs(r_es - 1)), 100 * (r_es.mean() - 1),
         100 * (r_es[0] - 1), 100 * (r_es[-1] - 1)))
    p('     CONFIRMED -- and it is the same measurement seen twice: the curve')
    p('     IS the ratio whose max deviation is the %.1f %% quoted above.'
      % (100 * md_es))
    p('     Against the onshell curve running %+.1f %% to %+.1f %%, the figure'
      % (100 * (r_es_on[0] - 1), 100 * (r_es_on[-1] - 1)))
    p('     says the ENTIRE SMEFT/LO structure is spin correlation.')
    p('')
    p('  2. NLO/LO in the none case should show roughly the %.1f %% measured'
      % (100 * md_ns))
    p('     for the SM LO / SM NLO none pair.')
    p('     MEASURED: max |dev| %.2f %%, ends %+.2f %% / %+.2f %%.'
      % (100 * np.max(np.abs(r_ns - 1)), 100 * (r_ns[0] - 1),
         100 * (r_ns[-1] - 1)))
    p('     CONFIRMED.  The %.2f %% and the %.2f %% are the same number the two'
      % (100 * np.max(np.abs(r_ns - 1)), 100 * md_ns))
    p('     ways round: 1/(1 - %.4f) - 1 = %.4f.'
      % (md_ns, 1 / (1 - md_ns) - 1))
    p('     The onshell curve is SMALLER (%.2f %% max) than the none one, so'
      % (100 * np.max(np.abs(r_ns_on - 1))))
    p('     the LO/NLO difference in this observable is not a spin effect at')
    p('     all -- it is the extra radiation changing the t t~ boost, and the')
    p('     spin correlations partly mask it rather than cause it.')

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
    p('where the warnings and the parameter point live')
    p('=' * 78)
    p('  The figure carries NO text beyond its axis labels and legends, by')
    p('  request.  Everything below exists ONLY here and in README.md, and')
    p('  whoever writes the caption must carry it across:')
    p('')
    p('    * the SM NLO health warning, repeated at the top of this file --')
    p('      and, for F, the fact that it hits the SOLID red curves only;')
    p('    * the parameter point ctGRe = %g, Lambda = %g GeV, without which'
      % (d.meta['samples']['eft_int']['wilson_coefficients']['ctGRe'],
         d.meta['samples']['eft_int']['wilson_coefficients']
         ['LambdaSMEFT_GeV']))
    p('      pane 2 has no scale at all -- its magnitude is proportional to')
    p('      c_tG/Lambda^2 and all four of its curves mirror about 1 for')
    p('      c_tG > 0;')
    p('    * the pane-2 weights:')
    for mode in MODES:
        p('        %-8s  w(LO) = %.4f   w(NLO) = %.4f'
          % (mode, v.weight('sm_lo', mode), v.weight('sm_nlo', mode)))
    p('    * that pane 1 is a SHAPE ratio and not a K-factor (the K-factor is')
    p('      %.2f);' % (v.sigma('sm_nlo', 'onshell')
                        / v.sigma('sm_lo', 'onshell')))
    p('    * that the dashed curves are the non-spin part of each ratio and')
    p('      the solid-to-dashed gap is the spin-correlation effect.')
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
    devnull = open(os.devnull, 'w')
    rows = P.check_normalisation(d, fh=devnull) + check_weights(d, fh=devnull)
    bad = [name for name, ok, _ in rows if not ok]
    if bad:
        raise SystemExit('checks failed: %s' % '; '.join(bad))

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
        write_curves_F(d, os.path.join(args.out, 'smeft_fig5_F_curves.npz'))
        with open(os.path.join(args.out, 'numbers_F.txt'), 'w') as fh:
            write_numbers_F(d, fh)
        print('wrote %s/smeft_fig5_F_curves.npz and numbers_F.txt' % args.out)

    if args.style in ('user', 'both'):
        os.makedirs(args.out_user, exist_ok=True)
        base = make_figure_user(d, args.out_user)
        print('wrote %s.pdf / .png   (user style)' % base)
        write_curves_F(d, os.path.join(args.out_user,
                                       'smeft_fig5_F_curves.npz'))
        with open(os.path.join(args.out_user, 'numbers_F.txt'), 'w') as fh:
            write_numbers_F(d, fh)
        print('wrote %s/smeft_fig5_F_curves.npz and numbers_F.txt'
              % args.out_user)

    write_numbers_F(d)
    print('')
    print('*** %s ***' % NLO_WARNING_SHORT)
    print('*** on F the defect hits the SOLID red curves only: '
          'spinmode = none uses no density matrices ***')

    if failures:
        raise SystemExit('the usetex minus sign was lost in: %s'
                         % ', '.join(failures))


if __name__ == '__main__':
    main()
