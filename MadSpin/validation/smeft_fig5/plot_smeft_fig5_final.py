#!/usr/bin/env python3
r"""Variation ``O``: the FINAL Fig. 5 -- ``H`` with pane 2 rebuilt.

Standalone and re-runnable; it needs nothing but ``data/histograms.npz`` and
``data/meta.json``::

    plot_smeft_fig5_final.py [--data DIR] [--out DIR] [--out-user DIR]
                             [--nbins N] [--style {mg7,user,both}]
                             [--only TAG[,TAG...]]

Data loading, rebinning, error propagation and ``check_normalisation()`` come
from ``plot_smeft_fig5.py``; the user-style constants from
``plot_smeft_fig5_userstyle.py``; ``_step`` and the SM NLO provenance note from
``plot_smeft_fig5_varE.py``; the pane layout and ``check_weights()`` from
``plot_smeft_fig5_varF.py``; ``Scan``, the conventions and the ``c_tG``
machinery from ``plot_smeft_fig5_ctg_scan.py``.  **None of those five is
modified**, so variations ``A``--``N`` stay byte-identical; this script only
adds ``O`` and ``P``.


What ``O`` is
=============
``smeft_fig5_H_ctg_p1_shape`` -- the ``shape`` convention at ``c_tG = +1`` --
with three changes, all of them in **pane 2** or on the axes.  The upper pane
and pane 1 are ``H``'s, curve for curve.

1. **Pane 2 drops its two ``spinmode = none`` curves.**  It goes from four
   lines to two, both ``onshell``: ``(LO + SMEFT)/LO`` and
   ``(NLO + SMEFT)/NLO``.
2. **``onshell`` is written once, in pane 2's top-left corner**, and removed
   from the two legend entries there.  This is the one deliberate exception to
   the "axis labels and legends only" rule of this series: with every curve in
   the pane in one spinmode, repeating it twice in the legend is noise.  The
   upper pane and pane 1 still carry both spinmodes and still name them per
   entry.
3. **The axes carry no units and decimal ticks.**  ``[rad]`` is off the
   ``Delta phi`` axis and ``[rad^-1]`` off the upper pane's vertical axis, and
   the x ticks are ``0, 0.5, ... , 3.0`` on an axis running to ``pi``, instead
   of the ``pi/4`` ladder ``A``--``N`` use.  This matches the paper's other
   figures.  It is a consistency choice and not an improvement: the ``pi``
   ticks are the nicer ones.

Nothing else moved.  There is still no other text on the figure: no
coefficient, no convention, no header, no note, no title.


The third curve that was asked for, and why it is not drawn
===========================================================
``O`` was asked for with a **third** pane-2 curve, ``(NLO + K*int)/NLO`` with
``K`` the bin-by-bin ``NLO/LO`` K-factor -- an estimate of the missing NLO
interference, the ``O_tG`` interference being available at LO only.  It is not
drawn, because it is the curve already on the pane.

Write the ansatz out.  With ``K(x) = dsigma_NLO(x)/dsigma_LO(x)``,

    dsigma_NLO(x) + K(x) dsigma_int(x)
        = dsigma_NLO(x) * [ 1 + dsigma_int(x)/dsigma_LO(x) ]

so at the level of **unnormalised** cross sections -- the ``rate`` convention
of ``K``--``N`` -- the ratio to ``dsigma_NLO`` is

    1 + dsigma_int(x)/dsigma_LO(x)

with ``dsigma_NLO`` cancelling identically.  That is ``(LO + int)/LO``, bit for
bit: ``check_k_degeneracy()`` measures the difference and gets ``0.0``, not a
small number.

``O`` is in the ``shape`` convention, where both the numerator and ``n_NLO``
are renormalised to unit area, so the cancellation is not quite complete.  What
survives is a **single multiplicative constant**:

    curve3(x) = [1 + w_LO rho_LO(x)] / Z ,      Z = 1 + w_LO <rho_LO>_NLO
    curveLO(x) = [1 + w_LO rho_LO(x)] / (1 + w_LO)

so ``curve3 / curveLO = (1 + w_LO)/Z`` with **no x in it**.  Measured, at
``c_tG = +1``: the ratio is ``1.0014992469`` in every one of the 20 bins, the
spread across bins being ``7e-16`` -- the last bit of double precision.  The
two curves are separated by at most ``0.0016``, which is ``0.91`` of the
plotted one-sigma error bar: they are inside each other's errors everywhere.

So the third curve is the LO curve, rigidly rescaled by 0.15 % by nothing but
the renormalisation of the K-weighted sum.  It carries **no** information about
the shape.  Drawing it would put two lines on top of each other and invite a
reader to measure a vertical gap that is a normalisation artefact.

This is not a defect of the calculation, it is what the ansatz says.  "The
interference receives the same bin-by-bin multiplicative NLO correction as the
SM" is exactly "the NLO correction does not change the operator's relative
effect", and the pane's LO curve is already that statement.  The useful
consequence is a **re-reading of the curve that is there**: ``(LO + SMEFT)/LO``
may be quoted as the NLO estimate under a bin-by-bin K-factor ansatz.  That
belongs in the caption, not in a fourth colour.

What ``K`` would carry information: variation ``P``
---------------------------------------------------
Only the **x-dependence** of ``K`` can.  Split it,

    K(x) = <K> * k(x) ,    <K> = sigma_NLO/sigma_LO = 1.519 ,  <k> = 1 ,

and the two factors do different things.  ``<K>`` multiplies the interference's
rate, which is algebraically the same knob as ``c_tG`` -- the figure already
scans that, in ``G``--``N``, and in the ``shape`` convention it divides straight
back out.  Only ``k(x) = n_NLO(x)/n_LO(x)``, the K-factor with its inclusive
size divided out, moves the curve relative to the two that are drawn.

``P`` (``smeft_fig5_P_ctg_p1_shape_kshape``) is ``O`` with that third curve
added:

    (NLO + k*int)/NLO ,   k(x) = n_NLO(x)/n_LO(x)   (unit-area K-factor)
        = [1 + w_NLO rho_LO(x)] / Z' ,   Z' = 1 + w_NLO <rho_LO>_NLO

i.e. the interference keeps its LO **rate** but is given the NLO **shape**
distortion.  It is a genuinely different curve: ``6.3 sigma`` from the NLO
curve at its furthest bin and ``15.5 sigma`` from the LO one, lying between
them in 17 of 20 bins.  ``P`` is offered as the proposal; ``O`` is the figure.

Both statements are measured by ``check_k_degeneracy()``, which runs before
anything is drawn and writes its table into every ``numbers_*.txt``.


Provenance of the SM NLO sample
===============================
``sm_nlo``'s MadSpin density matrices used to be evaluated at model defaults
rather than the run's card (``173 / 1.4915 / 2.4414 / 2.0476`` against the
run's ``172.76 / 1.33 / 2.4952 / 2.085``).  That is fixed, and the sample drawn
here was regenerated with the fix (task T123).  The measured effect on this
observable is ``-0.0001 %`` on the cross section and, on the shape, smaller
than reseeding the MadSpin run.  The defect was in the **density** path, which
``spinmode = none`` does not use at all, so only the **solid** NLO curves were
ever touched.  It is not on the figure; see ``README.md`` and the head and foot
of ``numbers_O.txt``.
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
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter

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
import plot_smeft_fig5_ctg_scan as G                           # noqa: E402
matplotlib.use('Agg')
_USER_RC = {k: mpl.rcParamsDefault[k] for k in _MG7_RC}

MODES = F.MODES                     # ('onshell', 'none')
CURVES = F.CURVES                   # upper pane, unchanged from H
PANE1 = F.PANE1                     # [('sm_nlo','sm_lo'), ('eft_int','sm_lo')]
PANE2 = F.PANE2                     # ['sm_lo', 'sm_nlo']

# Pane 2 is one spinmode now, and it is named once in the corner instead of
# twice in the legend.
PANE2_MODE = 'onshell'

NLO_PROVENANCE_SHORT = E.NLO_PROVENANCE_SHORT
NLO_PROVENANCE_LONG = E.NLO_PROVENANCE_LONG
NLO_REGEN_NOTE = G.NLO_REGEN_NOTE

# The two variations this script adds.  (tag, c_tG, convention, k_curve)
POINTS = [
    ('O', +1.0, 'shape', None),
    ('P', +1.0, 'shape', 'kshape'),
]

STEMS = {
    'O': 'smeft_fig5_O_ctg_p1_shape_final',
    'P': 'smeft_fig5_P_ctg_p1_shape_kshape',
}

# The K-factor curve, in the two definitions the degeneracy analysis separates.
K_DEFS = {
    # K(x) = dsigma_NLO/dsigma_LO -- the rate K-factor, as asked for.
    # DEGENERATE: the resulting curve is (LO+int)/LO times a constant.
    'krate': dict(
        weight='sm_lo',
        label='rate K-factor  K = dsigma_NLO/dsigma_LO  (<K> = sigma_NLO/sigma_LO)',
        short='K_rate'),
    # k(x) = n_NLO/n_LO -- the same K with its inclusive size divided out.
    # The only part of K that is not already on the pane.
    'kshape': dict(
        weight='sm_nlo',
        label='shape K-factor  k = n_NLO/n_LO  (<k> = 1)',
        short='k_shape'),
}

# Pane 2 of `P` has one spinmode, so line style is free there and is used for
# the K ansatz instead.  Stated in numbers_P.txt; `O` does not use it.
K_LS = (0, (4.5, 1.6, 1.0, 1.6))


def stem(tag):
    return STEMS[tag]


# --------------------------------------------------------------------------
class Final(G.Scan):
    """``G.Scan`` plus the K-factor curve, and pane 2 restricted to one mode.

    Everything about ``c_tG``, the two conventions, the weights and the pane-2
    error identity is inherited unchanged from ``plot_smeft_fig5_ctg_scan.py``.
    """

    def __init__(self, d, ctg, convention, k_curve=None):
        G.Scan.__init__(self, d, ctg, convention)
        if k_curve is not None and k_curve not in K_DEFS:
            raise ValueError('unknown K definition %r' % (k_curve,))
        self.k_curve = k_curve

    # -- the K-factor curve ----------------------------------------------
    def kterms(self, which, mode=PANE2_MODE):
        r"""``(baseline, k, Z)`` of ``(NLO + K*int)/NLO`` in this convention.

        Both definitions of ``K`` are proportional to ``n_NLO/n_LO``, so both
        give a numerator ``n_NLO(x) * [1 + W rho_LO(x)]`` with

            W = w_LO   for the rate K-factor   (K = dsigma_NLO/dsigma_LO)
            W = w_NLO  for the shape K-factor  (k = n_NLO/n_LO)

        and the ``shape`` convention then divides by ``Z = 1 + W <rho_LO>_NLO``
        rather than by ``1 + W``: the K-weighted sum is not normalised by the
        SM's own weights.  Written as ``baseline + k*(rho_LO - 1)`` the shared
        SM measurement appears once, exactly as in ``Scan.pane2_terms``, and
        the error is ``|k| sigma(rho_LO)``.  ``Z`` is treated as an exact
        constant for the same reason ``Data.shape`` treats ``sigma`` as one --
        it is a global scale common to every bin.
        """
        W = self.weight(K_DEFS[which]['weight'], mode)
        rho, _ = self.shape_ratio('eft_int', 'sm_lo', mode)
        n_nlo, _ = self.d.shape('sm_nlo', mode)
        if self.convention == 'shape':
            Z = float(((n_nlo * (1.0 + W * rho)) * self.d.width).sum())
        else:
            Z = 1.0
        return (1.0 + W) / Z, W / Z, Z

    def kcurve(self, which, mode=PANE2_MODE):
        rho, rho_e = self.shape_ratio('eft_int', 'sm_lo', mode)
        base, k, _ = self.kterms(which, mode)
        return base + k * (rho - 1.0), abs(k) * rho_e

    # -- pane 2, one spinmode --------------------------------------------
    def pane2_series(self):
        """``[(name, colour_key, is_k, (r, e))]`` in drawing order."""
        out = []
        for sm in PANE2:
            out.append((sm, sm, False, self.pane2(sm, PANE2_MODE)))
        if self.k_curve:
            out.append((self.k_curve, 'sm_nlo', True,
                        self.kcurve(self.k_curve)))
        return out


# --------------------------------------------------------------------------
def check_k_degeneracy(d, ctg=+1.0, convention='shape', fh=sys.stdout):
    r"""Measure -- do not assume -- that the rate-K curve is already drawn.

    Two claims, both measured here:

    * in the ``rate`` convention ``(NLO + K*int)/NLO`` with
      ``K = dsigma_NLO/dsigma_LO`` is ``(LO + int)/LO`` **exactly**;
    * in the ``shape`` convention it is that curve times a constant, the
      bin-to-bin spread of the ratio being at the last bit of double
      precision, and the two differ by less than the plotted error bar.
    """
    p = lambda *a: print(*a, file=fh)
    s = Final(d, ctg, convention)
    mode = PANE2_MODE
    p('-' * 78)
    p('the third curve that was asked for: (NLO + K*int)/NLO, K = NLO/LO')
    p('-' * 78)
    p('  K(x) = dsigma_NLO(x)/dsigma_LO(x) gives')
    p('')
    p('      dsigma_NLO + K dsigma_int = dsigma_NLO * [1 + dsigma_int/dsigma_LO]')
    p('')
    p('  so dsigma_NLO cancels and the ratio to dsigma_NLO is')
    p('  1 + dsigma_int/dsigma_LO -- which IS (LO + int)/LO.')
    p('')

    # -- rate convention: exact identity ---------------------------------
    sr = Final(d, ctg, 'rate')
    lo_rate, _ = sr.pane2('sm_lo', mode)
    k_rate, _ = sr.kcurve('krate', mode)
    exact = float(np.max(np.abs(k_rate - lo_rate)))
    p('  rate convention  : max |(NLO + K int)/NLO  -  (LO + int)/LO| = %.1e'
      % exact)
    p('                     (an exact algebraic identity, not a small number)')

    # -- shape convention: constant rescaling ----------------------------
    ss = Final(d, ctg, 'shape')
    lo_sh, lo_sh_e = ss.pane2('sm_lo', mode)
    k_sh, _ = ss.kcurve('krate', mode)
    ratio = k_sh / lo_sh
    spread = float(ratio.max() / ratio.min() - 1.0)
    sep = float(np.max(np.abs(k_sh - lo_sh)))
    sig = float(np.max(np.abs(k_sh - lo_sh) / lo_sh_e))
    _, _, Z = ss.kterms('krate', mode)
    w_lo = ss.weight('sm_lo', mode)
    p('  shape convention : curve3/curveLO = (1 + w_LO)/Z, with no x in it')
    p('                     w_LO = %+.8f   Z = %.8f' % (w_lo, Z))
    p('                     ratio = %.10f in every bin, spread %.1e'
      % (float(ratio.mean()), spread))
    p('                     max separation %.6f = %.2f x the plotted error'
      % (sep, sig))
    p('')
    p('  So the rate-K curve is the LO curve rescaled by a CONSTANT %.4f %%,'
      % (100.0 * (float(ratio.mean()) - 1.0)))
    p('  from nothing but renormalising the K-weighted sum to unit area.  It')
    p('  carries no shape information and is NOT drawn on O.  Read the LO')
    p('  curve as the NLO estimate under a bin-by-bin K-factor ansatz instead;')
    p('  that belongs in the caption.')
    p('')

    # -- what does carry information -------------------------------------
    p('  Only the x-dependence of K can move the curve.  Splitting')
    p('  K(x) = <K> k(x) with <K> = sigma_NLO/sigma_LO = %.4f and <k> = 1:'
      % (d.sigma('sm_nlo', mode) / d.sigma('sm_lo', mode)))
    p('    <K> rescales the interference rate, which is the same knob as c_tG')
    p('        (scanned in G-N) and divides back out in the shape convention;')
    p('    k(x) = n_NLO/n_LO is the only part that is not already on the pane.')
    p('')
    ks, ks_e = ss.kcurve('kshape', mode)
    nlo_sh, nlo_sh_e = ss.pane2('sm_nlo', mode)
    d_nlo = float(np.max(np.abs(ks - nlo_sh) / nlo_sh_e))
    d_lo = float(np.max(np.abs(ks - lo_sh) / lo_sh_e))
    between = int(np.sum((ks - lo_sh) * (ks - nlo_sh) < 0))
    p('  (NLO + k int)/NLO with k = n_NLO/n_LO -- variation P -- is distinct:')
    p('      %.2f sigma from the NLO curve at its furthest bin' % d_nlo)
    p('      %.2f sigma from the LO curve' % d_lo)
    p('      between the two in %d of the %d bins' % (between, d.nbins))
    p('      ends %+.4f +- %.4f  ...  %+.4f +- %.4f'
      % (ks[0], ks_e[0], ks[-1], ks_e[-1]))
    p('')
    ok = (exact == 0.0) and (spread < 1e-12) and (sig < 1.0)
    return [('the rate-K curve is (LO+int)/LO, exactly or up to a constant',
             ok,
             'rate: identical to %.1e; shape: constant ratio %.10f, spread '
             '%.1e, separation %.2f sigma'
             % (exact, float(ratio.mean()), spread, sig))]


# --------------------------------------------------------------------------
def _decimal_ticks(ax):
    """0, 0.5, ... 3.0 on an axis running to pi, no unit anywhere.

    ``A``--``N`` use the ``pi/4`` ladder, which is the nicer one.  The paper's
    other figures use decimal radians and consistency wins.
    """
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: '%g' % x))


def _room(ax, series, anchors, bottom_extra=0.0, top_extra=0.0,
          pad_frac=0.10):
    """``G._room`` with headroom for pane 2's corner label."""
    lo, hi = min(anchors), max(anchors)
    for r, e in series:
        lo = min(lo, float((r - e).min()))
        hi = max(hi, float((r + e).max()))
    span = hi - lo
    ax.set_ylim(lo - pad_frac * span - bottom_extra * span,
                hi + pad_frac * span + top_extra * span)


def _label_p2(sm, plain=False):
    """Pane 2's legend entry, with the spinmode taken OUT.

    It is in the corner of the pane instead -- the one piece of text this
    series puts on a figure, and it is there because every curve in the pane
    shares it.
    """
    tex = not plain and P.USETEX
    name = P.SAMPLE_TEX if tex else P.SAMPLE_PLAIN
    if tex:
        return (r'$(\,$%s$\,+\,$int.$\,)\,/\,$%s'
                % (name[sm], name[sm]))
    return '(%s + int.) / %s' % (name[sm], name[sm])


def _label_k(which, plain=False):
    """The K-factor curve's legend entry (variation ``P`` only)."""
    tex = not plain and P.USETEX
    name = P.SAMPLE_TEX if tex else P.SAMPLE_PLAIN
    if tex:
        return (r'$(\,$%s$\,+\,K\,$int.$\,)\,/\,$%s'
                % (name['sm_nlo'], name['sm_nlo']))
    return '(%s + K int.) / %s' % (name['sm_nlo'], name['sm_nlo'])


def _corner_mode(ax, plain=False):
    """``onshell``, once, in pane 2's top-left corner."""
    txt = (P.MODE_TEX if (not plain and P.USETEX)
           else P.MODE_PLAIN)[PANE2_MODE]
    ax.text(0.018, 0.94, txt, transform=ax.transAxes, ha='left', va='top',
            fontsize=8.5 if not plain else 8.0, zorder=6)


# --------------------------------------------------------------------------
def make_figure_mg7(d, s, tag, out):
    """The MG7 paper style: upper pane and pane 1 as ``H``, pane 2 rebuilt."""
    mpl.rcParams.update(_MG7_RC)
    tx = P._tx

    fig = plt.figure(figsize=(7 * 0.75 * 1.35, 7 * 0.75 * 1.5 * 1.40))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.2], hspace=0.07)
    ax = fig.add_subplot(gs[0])
    r1 = fig.add_subplot(gs[1], sharex=ax)
    r2 = fig.add_subplot(gs[2], sharex=ax)
    lo, hi = 0.0, float(np.pi)

    # --- upper pane: H's six curves, unchanged but for the unit ----------
    for sample, mode in CURVES:
        y, ye = d.shape(sample, mode)
        E._step(ax, d, y, color=P.COLOR[sample], ls=P.LS[mode], lw=P.LW,
                label=F._label_top(sample, mode), zorder=4)
        ax.errorbar(d.centres, y, yerr=ye, fmt='none', ecolor=P.COLOR[sample],
                    elinewidth=0.9, capsize=0, zorder=4)

    ax.set_ylabel(tx(
        r'$\frac{1}{\sigma}\,\mathrm{d}\sigma/\mathrm{d}\Delta\phi(e^-e^+)$',
        r'$(1/\sigma)\,d\sigma/d\Delta\phi(e^-e^+)$'))
    ax.set_xlim(lo, hi)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelbottom=False)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(ymin, 0.15), ymax + 0.30 * (ymax - ymin))
    ax.legend(frameon=False, loc='upper left', fontsize=9.5,
              handlelength=2.8, borderaxespad=1.0, labelspacing=0.4, ncol=2,
              columnspacing=1.4)

    # --- pane 1: H's four curves, unchanged ------------------------------
    for y in s.pane1_reference():
        G._refline(r1, y)
    series = []
    for mode in MODES:
        for num, den in PANE1:
            r, e = s.pane1(num, den, mode)
            series.append((r, e))
            E._step(r1, d, r, color=P.COLOR[num], ls=P.LS[mode], lw=P.LW,
                    zorder=4, label=G._label_p1(s, num, den, mode))
            r1.errorbar(d.centres, r, yerr=e, fmt='none', ecolor=P.COLOR[num],
                        elinewidth=0.9, capsize=0, zorder=4)
    _room(r1, series, s.pane1_reference(), bottom_extra=0.46)
    r1.set_ylabel(G._ylabel_p1(s, tx), fontsize=11.5)
    r1.legend(frameon=False, loc='lower left', fontsize=7.5,
              handlelength=2.4, borderaxespad=0.5, ncol=2, columnspacing=1.4)

    # --- pane 2: onshell only, named once in the corner -------------------
    for y in s.pane2_reference():
        G._refline(r2, y)
    series, handles = [], []
    for name, colour, is_k, (r, e) in s.pane2_series():
        ls = K_LS if is_k else P.LS[PANE2_MODE]
        lab = _label_k(name) if is_k else _label_p2(name)
        series.append((r, e))
        E._step(r2, d, r, color=P.COLOR[colour], ls=ls, lw=P.LW, zorder=4)
        r2.errorbar(d.centres, r, yerr=e, fmt='none', ecolor=P.COLOR[colour],
                    elinewidth=0.9, capsize=0, zorder=4)
        handles.append(Line2D([], [], color=P.COLOR[colour], lw=P.LW, ls=ls,
                              label=lab))
    _room(r2, series, s.pane2_reference(), bottom_extra=0.46, top_extra=0.30)
    r2.set_ylabel(G._ylabel_p2(s, tx), fontsize=11.5)
    r2.legend(handles=handles, frameon=False, loc='lower left', fontsize=7.5,
              handlelength=2.4, borderaxespad=0.5, ncol=1, columnspacing=1.4)
    _corner_mode(r2)

    r2.set_xlabel(tx(r'$\Delta\phi(e^-e^+)$', r'$\Delta\phi(e^-e^+)$'))
    for rx in (r1, r2):
        rx.yaxis.set_minor_locator(AutoMinorLocator())
        rx.set_xlim(lo, hi)
    r1.tick_params(labelbottom=False)
    for axis in (ax, r1, r2):
        _decimal_ticks(axis)

    fig.subplots_adjust(left=0.155, right=0.975, top=0.988, bottom=0.075)
    base = os.path.join(out, stem(tag))
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

    def draw(axis, y, ye, colour, filled, label, ls=None):
        axis.step(d.edges, np.concatenate([y[:1], y]), where='pre',
                  color=colour, lw=1.0, alpha=U.STEP_ALPHA, zorder=3,
                  ls=ls or 'solid')
        axis.errorbar(d.centres, y, yerr=ye, fmt='o', ms=U.MS, color=colour,
                      mfc=(colour if filled else 'white'), mew=1.1,
                      label=label, zorder=4)

    for sample, mode in CURVES:
        y, ye = d.shape(sample, mode)
        draw(ax, y, ye, U.COLOR[sample], U.FILLED[mode],
             F._label_top(sample, mode, plain=True))

    ax.set_ylabel(r'$(1/\sigma)\,d\sigma/d\Delta\phi(e^-e^+)$')
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
            draw(r1, r, e, U.COLOR[num], U.FILLED[mode],
                 G._label_p1(s, num, den, mode, plain=True))
    _room(r1, series, s.pane1_reference(), bottom_extra=0.40)
    r1.set_ylabel(G._ylabel_p1(s))
    r1.legend(loc='lower left', fontsize=6.5, ncol=2)
    r1.tick_params(labelbottom=False)

    for y in s.pane2_reference():
        r2.axhline(y, color='black', ls='--', lw=0.9, zorder=2)
    series = []
    for name, colour, is_k, (r, e) in s.pane2_series():
        series.append((r, e))
        draw(r2, r, e, U.COLOR[colour], not is_k,
             _label_k(name, plain=True) if is_k
             else _label_p2(name, plain=True),
             ls='-.' if is_k else None)
    _room(r2, series, s.pane2_reference(), bottom_extra=0.40, top_extra=0.30)
    r2.set_ylabel(G._ylabel_p2(s))
    r2.legend(loc='lower left', fontsize=6.5, ncol=1)
    _corner_mode(r2, plain=True)
    r2.set_xlabel(r'$\Delta\phi(e^-e^+)$')
    r2.set_xlim(0.0, np.pi)
    for axis in (ax, r1, r2):
        _decimal_ticks(axis)

    fig.subplots_adjust(hspace=0.1, left=0.155, right=0.97,
                        bottom=0.07, top=0.985)
    base = os.path.join(out, stem(tag))
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=U.DPI)
    plt.close(fig)
    return base


# --------------------------------------------------------------------------
def write_curves(d, s, tag, path):
    """Every number this variation draws, beside the figure."""
    out = {'edges': d.edges, 'centres': d.centres,
           'bin_width_rad': np.array(d.width),
           'ctGRe': np.array(s.ctg),
           'ctGRe_generated': np.array(s.c_ref),
           'Lambda_GeV': np.array(s.Lambda),
           'convention': np.array(s.convention),
           'pane2_spinmode': np.array(PANE2_MODE),
           'variation': np.array(tag),
           'file_stem': np.array(stem(tag))}
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
        r, e = s.pane2(sm, PANE2_MODE)
        base, k, w = s.pane2_terms(sm, PANE2_MODE)
        out['pane2_%s_plus_int_over_%s' % (sm, sm)] = r
        out['pane2_%s_plus_int_over_%s_err' % (sm, sm)] = e
        out['pane2_w_%s' % sm] = np.array(w)
        out['pane2_baseline_%s' % sm] = np.array(base)
        out['pane2_k_%s' % sm] = np.array(k)
    # Both K definitions are stored on BOTH variations, drawn or not: the
    # degeneracy is a result and belongs in the data file.
    for which in K_DEFS:
        r, e = s.kcurve(which)
        base, k, Z = s.kterms(which)
        out['pane2_nlo_plus_%s_int_over_nlo' % which] = r
        out['pane2_nlo_plus_%s_int_over_nlo_err' % which] = e
        out['pane2_%s_baseline' % which] = np.array(base)
        out['pane2_%s_k' % which] = np.array(k)
        out['pane2_%s_Z' % which] = np.array(Z)
        out['pane2_%s_drawn' % which] = np.array(which == (s.k_curve or ''))
    np.savez(path, **out)
    return path


def _header(s, tag, p):
    p('=' * 78)
    p('Fig. 5 variation %s: the FINAL figure -- H with pane 2 rebuilt'
      % tag)
    p('=' * 78)
    p('')
    p('  NOTHING ON THIS FIGURE SAYS SO.  It is drawn at')
    p('')
    p('        ctGRe = %+g          Lambda = %g GeV' % (s.ctg, s.Lambda))
    p('        convention : %s' % s.convention)
    p('        pane 2     : spinmode = %s only, named once in the corner'
      % PANE2_MODE)
    p('        file stem  : %s' % stem(tag))
    p('')
    p('  and a caption must say so.  The samples were generated at ctGRe =')
    p('  %+g; the interference is linear in the coefficient, so everything' % s.c_ref)
    p('  here is the measured interference times %+g.' % s.scale)
    p('')
    p('  The ONE piece of text on the figure is the word `%s\' in pane 2\'s'
      % PANE2_MODE)
    p('  top-left corner.  It is there because every curve in that pane is in')
    p('  that spinmode, so naming it per legend entry was pure repetition.')
    p('  The upper pane and pane 1 still carry both spinmodes and still name')
    p('  them per entry.  Everything else on the figure is an axis label or a')
    p('  legend entry, as in A-N.')
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
    p('  none does not use.  Pane 2 is now onshell only, so it is entirely')
    p('  built from curves the defect could in principle have touched; the')
    p('  measured move is 0.83 % at most, 1.8 sigma, below the seed noise.')
    p('')

    # -- what changed from H ---------------------------------------------
    p('=' * 78)
    p('what O changes, against H (smeft_fig5_H_ctg_p1_shape)')
    p('=' * 78)
    p('  upper pane : unchanged, curve for curve.  Six curves, three samples')
    p('               x two spinmodes, all unit-area.')
    p('  pane 1     : unchanged, curve for curve.  NLO/LO and SMEFT/LO in')
    p('               both spinmodes, shape ratios.')
    p('  pane 2     : the two spinmode = none curves are DROPPED.  Two curves')
    p('               remain, both onshell:')
    p('                   (LO  + SMEFT)/LO')
    p('                   (NLO + SMEFT)/NLO')
    p('               `onshell\' is written once in the pane\'s top-left')
    p('               corner and is out of both legend entries.')
    p('  axes       : the units are gone -- no [rad] on the Delta phi axis,')
    p('               no [rad^-1] on the upper pane\'s vertical axis -- and')
    p('               the x ticks are DECIMAL (0, 0.5, ... 3.0 on an axis')
    p('               running to pi = 3.14159) instead of the pi/4 ladder of')
    p('               A-N.  Consistency with the paper\'s other figures; the')
    p('               pi ticks were the nicer ones and were given up for it.')
    p('')
    p('  A-N are untouched and byte-identical: this script imports the five')
    p('  that draw them and modifies none of them.')
    p('')

    P.check_normalisation(d, fh)
    p('')
    check_k_degeneracy(d, s.ctg, s.convention, fh)
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
    p('encoding     : upper pane and pane 1 -- line style = spinmode (solid')
    p('               onshell, dashed none), colour = quantity.  Pane 2 is')
    p('               one spinmode, so every line there is solid%s.'
      % (' except the K-factor curve, which is dash-dot' if s.k_curve else ''))
    p('')

    # -- the weights -------------------------------------------------------
    p('=' * 78)
    p('the weights: the only place c_tG enters')
    p('=' * 78)
    p('  w(c_tG) = sigma_int(c_tG)/sigma_SM is linear in c_tG and signed.')
    p('  At the generated point ctGRe = %+g it is measured to be' % s.c_ref)
    for mode in MODES:
        p('      %-8s  w(LO) = %+.6f   w(NLO) = %+.6f'
          % (mode, d.sigma('eft_int', mode) / d.sigma('sm_lo', mode),
             d.sigma('eft_int', mode) / d.sigma('sm_nlo', mode)))
    p('  so w(LO) = %+.6f * c_tG and w(NLO) = %+.6f * c_tG,'
      % (d.sigma('eft_int', 'onshell') / d.sigma('sm_lo', 'onshell') / s.c_ref,
         d.sigma('eft_int', 'onshell') / d.sigma('sm_nlo', 'onshell')
         / s.c_ref))
    p('  and this figure is at c_tG = %+g.' % s.ctg)
    p('')
    p('%-10s %-8s %11s %11s %11s' % ('curve', 'SM', 'w', '1 + w', 'k'))
    for sm in PANE2:
        base, k, w = s.pane2_terms(sm, PANE2_MODE)
        p('%-10s %-8s %11.6f %11.6f %11.6f' % ('pane 2', sm, w, 1.0 + w, k))
    for which in K_DEFS:
        base, k, Z = s.kterms(which)
        p('%-10s %-8s %11s %11s %11.6f  (Z = %.6f, %s)'
          % ('K curve', which, '', '', k, Z,
             'DRAWN' if which == s.k_curve else 'not drawn'))
    p('')
    p('  k is the coefficient of (rho - 1), rho = n_int/n_SM for the two SM')
    p('  curves and rho = n_int/n_LO for the K ones; the error is')
    p('  |k| * sigma(rho) throughout, which is the identity that keeps the')
    p('  shared SM measurement from being counted twice.')
    p('')

    # -- the curves --------------------------------------------------------
    p('=' * 78)
    p('what this figure draws')
    p('=' * 78)
    rows = []
    for mode in MODES:
        for num, den in PANE1:
            rows.append(('p1 %s/%s, %s' % (num, den, mode),
                         s.pane1(num, den, mode)))
    for name, _c, is_k, re_ in s.pane2_series():
        rows.append(('p2 %s' % ('(nlo + %s int)/nlo' % K_DEFS[name]['short']
                                if is_k else '(%s+int)/%s' % (name, name)),
                     re_))
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
    p('  pane 2 as percentages, which is how it reads:')
    for name, _c, is_k, (r, e) in s.pane2_series():
        p('    %-32s first %+7.2f %% +- %.2f   last %+7.2f %% +- %.2f'
          % (name, 100 * (r[0] - 1), 100 * e[0],
             100 * (r[-1] - 1), 100 * e[-1]))
    p('')
    p('  and the curve that was asked for and is NOT drawn, for the record:')
    r, e = s.kcurve('krate')
    p('    %-32s first %+7.2f %% +- %.2f   last %+7.2f %% +- %.2f'
      % ('krate  (== the LO curve x const)', 100 * (r[0] - 1), 100 * e[0],
         100 * (r[-1] - 1), 100 * e[-1]))

    p('')
    p('=' * 78)
    p('bin by bin')
    p('=' * 78)
    p('%9s' % 'phi' + ''.join(' %21s' % n for n, _ in rows))
    for i in range(d.nbins):
        line = '%9.4f' % d.centres[i]
        for _, (r, e) in rows:
            line += '   %8.4f +- %.4f' % (r[i], e[i])
        p(line)

    p('')
    p('=' * 78)
    p('what a caption must carry, since the figure carries no text')
    p('=' * 78)
    p('    * ctGRe = %+g, Lambda = %g GeV.  Without it the ratio panes have'
      % (s.ctg, s.Lambda))
    p('      no scale at all.  The samples were generated at ctGRe = %+g, so'
      % s.c_ref)
    p('      the plotted interference is -2 Re(M*_SM M_tG) times %+g.'
      % s.scale)
    p('    * the convention: shape -- %s' % G.CONVENTION_BLURB['shape'])
    p('    * the x axis is Delta phi in RADIANS; the unit is not on the axis.')
    p('      So is the upper pane\'s vertical axis, in rad^-1.')
    p('    * pane 1 is a SHAPE ratio and not a K-factor.  The LO->NLO')
    p('      K-factor of these samples is %.3f and is not shown.'
      % (d.sigma('sm_nlo', 'onshell') / d.sigma('sm_lo', 'onshell')))
    p('    * in the upper pane and pane 1 the dashed curve is the part of the')
    p('      ratio that is NOT a spin-correlation effect and the gap up to')
    p('      the solid curve of the same colour is the spin-correlation')
    p('      effect itself.  PANE 2 IS ONSHELL ONLY -- the word is in its')
    p('      corner -- so no such reading applies there.')
    p('    * the two pane-2 curves may ALSO be read as the estimate of the')
    p('      missing NLO interference: with the interference scaled by the')
    p('      bin-by-bin K-factor K = dsigma_NLO/dsigma_LO, (NLO + K int)/NLO')
    p('      IS the (LO + int)/LO curve -- identically in the rate')
    _const = float((s.kcurve('krate')[0]
                    / s.pane2('sm_lo', PANE2_MODE)[0]).mean())
    p('      convention, and here up to a constant %+.4f %%.  See the'
      % (100.0 * (_const - 1.0)))
    p('      degeneracy table above.')
    if s.k_curve:
        p('    * THIS VARIATION (P) adds a third pane-2 curve, the same')
        p('      ansatz with the K-factor\'s inclusive size divided out,')
        p('      k = n_NLO/n_LO: the interference keeps its LO rate and is')
        p('      given the NLO shape distortion.  That is the only part of')
        p('      the K-factor that is not already on the pane.')
    p('    * the SM NLO provenance note: the defect touched the SOLID red')
    p('      curves only, spinmode = none using no density matrices, and its')
    p('      measured effect on this observable is below the seed noise.')
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
    ap.add_argument('--only', default=None, help='subset of O,P')
    ap.add_argument('--check-minus', dest='check_minus', action='store_true',
                    default=True)
    ap.add_argument('--no-check-minus', dest='check_minus',
                    action='store_false')
    args = ap.parse_args()

    d = P.Data(args.data, nbins=args.nbins)
    devnull = open(os.devnull, 'w')
    rows = (P.check_normalisation(d, fh=devnull)
            + F.check_weights(d, fh=devnull)
            + G.check_ctg_invariance(d, fh=devnull)
            + check_k_degeneracy(d, fh=devnull))
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
    for tag, ctg, conv, kc in POINTS:
        if wanted and tag not in wanted:
            continue
        s = Final(d, ctg, conv, kc)
        name = stem(tag)
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
            write_curves(d, s, tag,
                         os.path.join(args.out, name + '_curves.npz'))
            with open(os.path.join(args.out, 'numbers_%s.txt' % tag),
                      'w') as fh:
                write_numbers(d, s, tag, fh)
            print('   %s_curves.npz and numbers_%s.txt' % (name, tag))

        if args.style in ('user', 'both'):
            os.makedirs(args.out_user, exist_ok=True)
            base = make_figure_user(d, s, tag, args.out_user)
            print('wrote %s.pdf / .png   (user style)' % base)
            write_curves(d, s, tag,
                         os.path.join(args.out_user, name + '_curves.npz'))
            with open(os.path.join(args.out_user, 'numbers_%s.txt' % tag),
                      'w') as fh:
                write_numbers(d, s, tag, fh)
            print('   %s_curves.npz and numbers_%s.txt' % (name, tag))

    print('')
    check_k_degeneracy(d)
    print('')
    print('*** O is the final figure: %s ***' % stem('O'))
    print('*** P is the proposal, the only K definition that is not already '
          'on the pane ***')
    print('*** %s ***' % NLO_PROVENANCE_SHORT)

    if failures:
        raise SystemExit('the usetex minus sign was lost in: %s'
                         % ', '.join(failures))


if __name__ == '__main__':
    main()
