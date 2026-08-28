#!/usr/bin/env python3
"""``dsigma/dm_tt`` near the ``2 m_t`` threshold -- MadSpin's spinmodes against
the doubly-resonant off-shell MadGraph truth, in the MG7 paper's plotting style.

Runs entirely off the committed raw histograms, so nothing here needs MadSpin:

    plot_mtt_threshold.py [--data DIR] [--out DIR]

with ``<data>`` holding ``histograms.npz`` and ``meta.json`` as written by
``run_mtt_threshold.py``.

What is drawn is a PER-EVENT quantity: the invariant mass of ``(W+ b) + (W- b~)``,
built from the four status-1 particles with ``|pid|`` in ``{24, 5}`` of each
event, identically on both sides.

The vertical scale is ``(1/sigma) dsigma/dm_tt`` in 1/GeV: every curve is
divided by its OWN total cross section, so the figure is a SHAPE comparison.
``sigma`` is the sample total over the full ``m_tt`` range (``sum(w)/N`` of the
whole file, ``meta.json`` ``runs[key].sumw``), NOT the integral of the plotted
316-420 GeV window -- that window holds only a few percent of the rate and
normalising to it would define away part of the difference under study.

Why shape and not absolute.  The truth and the MadSpin samples do not share a
total cross section: ``truth/PA = truth/onshell = truth/madspin_v1 = 0.96614``,
so an absolute ratio pane plateaus at 1.035 and a +-5 % band around it is mostly
a statement about the sample size.  That 3.4 % is understood -- MG5's
decay-chain truth truncates each top Breit-Wigner at ``bwcutoff`` widths and
MadSpin normalises to ``sigma_prod * BR`` with no such loss; see
``write_numbers`` and RESULTS.md section 1a -- and it is flat in ``m_tt``, which
is what makes dividing it out legitimate.  It is NOT hidden by the choice: every
absolute number stays in ``numbers.txt`` and in RESULTS.md.

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
    figure's fonts therefore throws the minus away and the PDF comes out with an
    empty /Differences and no /BaseEncoding, so viewers silently drop the sign:
    no warning, no error, a wrong figure.  Only PDF is affected (the PNGs go
    through dvipng, which rasterises first).

    This figure carries a minus sign in every main-pane tick label: the
    vertical scale is ``(1/sigma) dsigma/dm_tt``, which runs 1e-7 to 1e-2, so
    the axis is a column of ``10^{-n}`` and the bug would eat all of it.  (It
    used to be carried by the ratio ticks and the annotation text as well;
    stripping the annotations and centring the ratio pane on 1 removed those,
    which is why the premise of the check is restated here rather than assumed.)
    Guarded on the exact upstream line, so a fixed matplotlib is left alone.
    ``--check-minus`` re-opens the written PDF and asserts the sign survived;
    that check has had to be made twice in this project already, and it is
    verified to still discriminate: ``NO_MINUS_FIX=1`` makes it report False.
    """
    import inspect
    import textwrap
    from matplotlib import _type1font

    # NO_MINUS_FIX=1 disables the workaround.  Its only purpose is to prove that
    # ``check_minus`` is discriminating: with the fix off, the check must fail.
    # If it passed either way it would be worthless as a guard.
    if os.environ.get('NO_MINUS_FIX'):
        return False

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
# The curves.
#
#   truth       p p > t t~ j, t > w+ b, t~ > w- b~ generated by MG5.  Full
#               doubly-resonant off-shell matrix element, both tops carrying
#               their Breit-Wigner propagator out to ``bwcutoff`` widths.  This
#               is the reference every ratio divides by.
#   madspin     MadSpin's default off-shell spinmode: the virtuality comes from
#               the off-shell density matrix, i.e. from the matrix element.
#   PA          pole approximation: a per-particle Breit-Wigner draw multiplying
#               an on-shell density, then the production reshuffle.
#   onshell     no virtuality is drawn at all and ``_density_do_reshuffle`` is
#               False, so the production momenta are untouched.  Its m_tt is the
#               production sample's m_tt bit for bit -- verified in RESULTS.md
#               from the event-by-event pairing, max |Delta m_tt| = 0 exactly.
#   madspin_v1  the legacy Fortran path.  It does not go through run_onshell at
#               all: it draws the virtualities inside its own Fortran driver and
#               then REGENERATES the whole phase-space point from the
#               decay-chain topology, holding sqrt(shat) and the production
#               tree's invariants at the values it extracted from the production
#               event (``generate_momenta_conf`` / ``keep_inv`` in
#               MadSpin/src/driver.f).  That is a different reshuffle from the
#               RAMBO mass_shuffle the density modes use, so where it lands is
#               a measurement, not a prediction.
#   production  the undecayed p p > t t~ j sample.  Harvested and tabulated but
#               deliberately NOT drawn: ``onshell`` reproduces it event by event
#               (max |Delta m_tt| = 0 exactly), so a second curve on top of it
#               would only add ink.  RESULTS.md carries the comparison.
# --------------------------------------------------------------------------
CURVES = [
    ('truth',      r'$pp \to t\bar t j$, $t \to W^+ b$ (off shell)'),
    ('madspin',    r'MadSpin, \texttt{spinmode = madspin}'),
    ('PA',         r'MadSpin, \texttt{spinmode = PA}'),
    ('onshell',    r'MadSpin, \texttt{spinmode = onshell}'),
    ('madspin_v1', r'MadSpin, \texttt{spinmode = madspin\_v1} (legacy)'),
]
CURVES_PLAIN = {
    'truth':      r'$pp \to t\bar t j$, $t \to W^+ b$ (off shell)',
    'madspin':    'MadSpin, spinmode = madspin',
    'PA':         'MadSpin, spinmode = PA',
    'onshell':    'MadSpin, spinmode = onshell',
    'madspin_v1': 'MadSpin, spinmode = madspin_v1 (legacy)',
}
REF = 'truth'
MODES = ['madspin', 'PA', 'onshell', 'madspin_v1']

# The production process, as it appears in the setup line above the curves and
# in the truth curve's legend entry.  A module global so the sibling ``2 -> 2``
# study can re-point this module at ``p p > t t~`` without a second copy of the
# figure code -- the style is then shared by construction rather than by
# somebody remembering to copy a change across.
PROC_TEX = r'pp \to t\bar t j'
PROC_PLAIN = r'pp \to t\bar t j'

COLOR = {'truth': 'black', 'madspin': 'blue', 'PA': 'red',
         'onshell': allcolors[2], 'madspin_v1': allcolors[4],
         'production': 'gray'}
LS = {'truth': 'solid', 'madspin': 'solid', 'PA': 'dashed',
      'onshell': 'dashdot', 'madspin_v1': (0, (1, 1.4)),
      'production': 'dotted'}


# --------------------------------------------------------------------------
# Binning.
#
# ``run_mtt_threshold.py`` writes a fine UNIFORM 0.25 GeV grid so the committed
# .npz stays the raw measurement.  Plotting wants something else: the truth
# spectrum falls by three decades from the peak into the sub-threshold tail, so
# a binning that resolves the threshold turn-on leaves single-digit counts
# 20 GeV below it -- and that tail is exactly where the schemes differ.  Whole
# numbers of fine bins are grouped into zones, so no bin edge moves and nothing
# is interpolated:
#
#   316 -- 326  10   GeV   the deep sub-threshold tail, ~0.001% of sigma
#   326 -- 336   5   GeV
#   336 -- 344   2   GeV   approaching the threshold
#   344 -- 356   1   GeV   the turn-on itself, on both sides of 2 m_t
#   356 -- 380   2   GeV   where the schemes come back together
#   380 -- 420   5   GeV   the region that must be flat at unity
#
# Every edge is a multiple of 0.25 GeV, and 346.0 = 2 m_t is an edge, so no bin
# straddles the threshold: a bin either is entirely below it (where onshell has
# exactly zero support) or entirely above it.  That is not cosmetic -- a
# straddling bin would smear "structurally empty" into "small but non-zero".
# --------------------------------------------------------------------------
ZONES = [(316.0, 326.0, 10.0),
         (326.0, 336.0, 5.0),
         (336.0, 344.0, 2.0),
         (344.0, 356.0, 1.0),
         (356.0, 380.0, 2.0),
         (380.0, 420.0, 5.0)]

# The window the "does it agree" question is asked in.
AGREE_HI = 420.0


def zone_edges():
    out = [ZONES[0][0]]
    for lo, hi, w in ZONES:
        n = int(round((hi - lo) / w))
        for k in range(1, n + 1):
            out.append(lo + k * w)
    return np.array(out)


class Data(object):
    """The raw fine grid, and the rebinned absolute differential cross section.

    ``density`` is ``dsigma/dm_tt`` in pb/GeV.  The LHE files are written with
    ``event_norm = average``, i.e. every event carries the full cross section as
    its weight and the *mean* weight is sigma; the density of a bin is therefore
    ``sum(w)_bin / N_total / width``, and integrating it over all bins returns
    sigma.  ``check_norm`` records the agreement with the banner's own cross
    section, which is an independent number.
    """

    def __init__(self, ddir):
        self.z = np.load(os.path.join(ddir, 'histograms.npz'))
        self.meta = json.load(open(os.path.join(ddir, 'meta.json')))
        self.fine = self.z['bins']
        self.edges = zone_edges()
        self.centres = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.widths = np.diff(self.edges)
        self.two_mt = float(self.meta.get('two_mt', 346.0))
        self._groups = self._group_map()

    def _group_map(self):
        """(start, stop) index pairs into the fine grid, one per plot bin."""
        fine = self.fine
        groups = []
        for lo, hi in zip(self.edges[:-1], self.edges[1:]):
            a = int(np.argmin(np.abs(fine - lo)))
            b = int(np.argmin(np.abs(fine - hi)))
            assert abs(fine[a] - lo) < 1e-9 and abs(fine[b] - hi) < 1e-9, \
                'plot bin edge %s/%s is not on the fine grid' % (lo, hi)
            groups.append((a, b))
        return groups

    def _rebin(self, vec):
        return np.array([vec[a:b].sum() for a, b in self._groups])

    def nevents(self, key):
        return float(self.meta['runs'][key]['nevents'])

    def sigma(self, key):
        """Total cross section of the sample, from its own event weights."""
        return self.meta['runs'][key]['sumw'] / self.nevents(key)

    def banner_sigma(self, key):
        return self.meta['runs'][key].get('banner_cross_pb')

    def density(self, key):
        """(dsigma/dm [pb/GeV], its statistical error, raw event count)."""
        n = self.nevents(key)
        sumw = self._rebin(self.z['%s_sumw' % key])
        sumw2 = self._rebin(self.z['%s_sumw2' % key])
        cnt = self._rebin(self.z['%s_cnt' % key])
        return sumw / n / self.widths, np.sqrt(sumw2) / n / self.widths, cnt

    def shape(self, key):
        """((1/sigma) dsigma/dm [1/GeV], its error, raw event count).

        This is what the figure draws.  ``sigma`` is :meth:`sigma`, the sample's
        TOTAL cross section over the full ``m_tt`` range -- ``sum(w)/N`` of the
        whole file -- and deliberately not the integral of the plotted window:
        316-420 GeV holds a few percent of the rate, so normalising to it would
        divide out part of the very region under study and would also make the
        curves depend on the plot limits.

        The error carries the per-bin statistics only.  The relative error on
        the total is 0.03 % or below on every sample here (five truth runs
        agreeing to 0.011 %; MadEvent's own quoted integration errors are
        651.8 +- 0.22 pb and 674.4 +- 0.21 pb), i.e. two orders of magnitude
        below the per-bin errors near threshold, so it is not propagated.
        """
        y, ye, cnt = self.density(key)
        s = self.sigma(key)
        return y / s, ye / s, cnt

    def fine_density(self, key):
        n = self.nevents(key)
        w = np.diff(self.fine)
        return self.z['%s_sumw' % key] / n / w

    def integral(self, key, lo, hi):
        """(sigma, error, count) over ``lo <= m_tt < hi`` on the FINE grid.

        Done on the fine grid rather than the plot bins so the answer does not
        depend on the plotting choices above.
        """
        fine = self.fine
        c = 0.5 * (fine[:-1] + fine[1:])
        mask = (c >= lo) & (c < hi)
        n = self.nevents(key)
        s = self.z['%s_sumw' % key][mask].sum() / n
        e = math.sqrt(self.z['%s_sumw2' % key][mask].sum()) / n
        k = int(self.z['%s_cnt' % key][mask].sum())
        return s, e, k


# How close ``m_tt`` has to come back for a mode to count as not having moved
# it.  Two scales bracket the choice and they are four orders of magnitude
# apart, so nothing hinges on where in between it sits: the LHE writes momenta
# as decimal text, which puts a floor of ~1e-5 GeV under any mode that provably
# does not touch them (``onshell`` measures 2.6e-5 GeV in the ``t t~ j`` study),
# while a mode that does move ``m_tt`` moves it by tens of GeV (38 to 141 GeV
# there).  1e-3 GeV is above the text-precision floor and 1/1000 of the
# narrowest plot bin.
STRUCTURAL_TOL = 1e-3


def preserves_mtt(d, key):
    """Did ``key`` return every event's ``m_tt`` unchanged?

    Answered from the event-by-event pairing ``run_mtt_threshold.pair_delta``
    measured, not from the mode's name: ``max |Delta m_tt|`` over the whole
    sample, together with the ``max |Delta sqrt(shat)|`` that proves the two
    event streams were actually paired.

    Which modes this is true of is a property of the PROCESS, which is why it
    is measured.  For ``p p > t t~ j`` only ``onshell`` qualifies -- it draws no
    virtuality and never reshuffles -- while the density modes' RAMBO reshuffle
    rescales the recoil jet and moves ``m_tt``, and ``madspin_v1`` regenerates
    the phase-space point outright.  For a ``2 -> 2`` production there is no
    recoil to rescale, ``m_tt = sqrt(shat)`` is what the reshuffle holds fixed,
    and every mode qualifies.

    Older data files that predate the ``delta_mtt`` measurement fall back to
    ``onshell`` alone, which is the one case the code path guarantees without a
    measurement (``_density_do_reshuffle`` is False).
    """
    m = d.meta.get('delta_mtt', {}).get(key)
    if m is None:
        return key == 'onshell'
    return (m['max_abs'] < STRUCTURAL_TOL
            and m['max_dshat'] < STRUCTURAL_TOL)


def structurally_empty(d, key):
    """Bins where ``key`` has no support *by construction*, not by bad luck.

    The production sample is on-shell ``t t~``, so it has no support at all
    below ``2 m_t``.  A mode that provably returns each event's ``m_tt``
    unchanged (:func:`preserves_mtt`) therefore has none either -- for any
    sample size.  A mode that does move ``m_tt`` *can* land below threshold, so
    an empty sub-threshold bin of its is a statement about ``N`` and is drawn as
    a gap rather than as a zero.

    The claim is checked against the data rather than asserted: if a bin marked
    structurally empty turned out to hold events, this raises.
    """
    mask = d.centres < d.two_mt
    if not preserves_mtt(d, key):
        return np.zeros_like(mask)
    _, _, cnt = d.density(key)
    bad = mask & (cnt > 0)
    if bad.any():
        raise AssertionError(
            '%s has %d events below 2 m_t -- the "structurally empty" '
            'claim is false and the figure must not be drawn'
            % (key, int(cnt[bad].sum())))
    return mask


def ratio(num, nume, den, dene):
    """``num/den`` with the two errors combined in quadrature.

    Where the denominator is zero the ratio is NaN (nothing to divide by).
    Where the *numerator* is zero but the denominator is not, the ratio is a
    true 0.0, not a NaN: that is the whole point of the sub-threshold region for
    ``onshell`` and it must be drawn, not dropped.
    """
    r = np.full_like(num, np.nan, dtype=float)
    re = np.full_like(num, np.nan, dtype=float)
    good = den > 0
    r[good] = num[good] / den[good]
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_n = np.where(num[good] > 0, nume[good] / num[good], 0.0)
        rel_d = dene[good] / den[good]
    re[good] = np.abs(r[good]) * np.sqrt(rel_n ** 2 + rel_d ** 2)
    # A bin with zero numerator still has an error bar: it is the reference's.
    zero = good & (num == 0)
    re[zero] = nume[zero] / den[zero] if nume[zero].size else 0.0
    return r, re


# --------------------------------------------------------------------------
# The ratio pane is CLIPPED to +-20 %.
#
# A measured ratio whose central value lies outside that window still gets an
# arrow at the boundary it left through, so it does not quietly disappear: a
# clipped point drawn as an ordinary marker sitting at 1.2 would be worse than
# not clipping at all.  The axis label says the pane is clipped.
#
# A structurally empty bin (see :func:`structurally_empty`) carries no arrow and
# no marker of its own: it is an exact zero, it drops off the bottom of the pane
# with its step, and its content is reported in numbers.txt rather than drawn.
# --------------------------------------------------------------------------
RATIO_CLIP = (0.8, 1.2)


def offscale_arrows(ax, x, r, colour, clip=RATIO_CLIP, dx=None, slot=0,
                    nslot=1, lw=1.0, scale=9):
    """Draw one arrow per point of ``r`` that lies outside ``clip``.

    ``dx``/``slot``/``nslot`` spread the arrows of several curves horizontally
    inside their shared bin, so four modes leaving the same bin give four
    visible arrows rather than one drawn four times.

    Returns ``(n_below, n_above)``.
    """
    lo, hi = clip
    span = hi - lo
    x = np.asarray(x, dtype=float)
    if dx is None:
        dx = np.zeros_like(x)
    shift = ((slot - 0.5 * (nslot - 1)) / max(nslot, 1)) * 0.55 * np.asarray(dx)
    n_lo = n_hi = 0
    for xi, sh, ri in zip(x, shift, r):
        if not np.isfinite(ri):
            continue
        if ri < lo:
            tail, head = lo + 0.13 * span, lo + 0.015 * span
            n_lo += 1
        elif ri > hi:
            tail, head = hi - 0.13 * span, hi - 0.015 * span
            n_hi += 1
        else:
            continue
        ax.annotate('', xy=(xi + sh, head), xytext=(xi + sh, tail),
                    arrowprops=dict(arrowstyle='-|>', color=colour, lw=lw,
                                    shrinkA=0, shrinkB=0,
                                    mutation_scale=scale),
                    annotation_clip=False, zorder=7)
    return n_lo, n_hi


# The window used to divide out the overall normalisation for the SHAPE-only
# variant of the agreement question.  High enough that every scheme is well
# inside its asymptotic regime and the statistics are good; the figure itself is
# always in absolute normalisation and is never rescaled.
ANCHOR = (380.0, 420.0)


def shape_scale(d, key):
    """``sigma_truth / sigma_key`` on the TOTAL cross sections.

    Multiplying a mode's absolute density by this is identical to dividing both
    sides by their own total, i.e. it is exactly the ratio the figure's lower
    pane shows.  It is the figure's normalisation, so the agreement thresholds
    quoted *for the figure* are the ones computed with it.

    The two sides do NOT share a total cross section and the difference is a
    real, understood effect rather than a bug: MG5's decay-chain truth truncates
    each top's Breit-Wigner at ``bwcutoff`` widths in the MASS (``myamp.f``:
    ``abs(xmass - prmass) < bwcutoff * prwidth`` under ``gForceBW = 1``, the same
    convention MadSpin's ``BW_cut`` uses), removing
    ``1 - 2*arctan(2*bwcutoff)/pi`` = 2.12 % per resonance and 4.20 % for the
    pair, while MadSpin's output cross section is ``sigma_production * BR`` and
    carries no such loss.  ``write_numbers`` reproduces that estimate and says
    what it does and does not settle.

    Reporting both answers separates the two questions: "is the rate right"
    (it is not, by a known and quantified amount) and "is the SHAPE near
    threshold right" (which is what the reshuffle controls).
    """
    st, sk = d.sigma(REF), d.sigma(key)
    if sk <= 0:
        return float('nan')
    return st / sk


def anchor_scale(d, key):
    """``sigma_truth / sigma_key`` over :data:`ANCHOR`, and its relative error.

    A robustness check on :func:`shape_scale`, not the figure's normalisation:
    if the offset really is a flat rate effect then dividing it out locally, on
    a window well above the turn-on, must give the same agreement thresholds as
    dividing out the global total.  It does -- see ``numbers.txt`` -- and that
    agreement is the evidence that the 3.4 % is a normalisation and not a shape.
    """
    st, ste, _ = d.integral(REF, *ANCHOR)
    sk, ske, _ = d.integral(key, *ANCHOR)
    if sk <= 0:
        return float('nan'), float('nan')
    scale = st / sk
    rel = math.sqrt((ste / st) ** 2 + (ske / sk) ** 2)
    return scale, rel


def agreement_threshold(d, key, tol, scale=1.0):
    """Lowest plot-bin edge above which every bin agrees with truth to ``tol``.

    ``scale`` multiplies the mode's density before the comparison; 1.0 is the
    absolute question, :func:`anchor_scale` gives the shape-only one.

    Two answers, because they say different things:

      * ``strict``   -- the lowest edge X such that every bin centred above X
                        has ``|ratio - 1| <= tol`` on its CENTRAL value.
      * ``compat``   -- the same, but a bin counts as agreeing when it is
                        within ``tol`` once its own 1 sigma error is allowed,
                        i.e. ``|ratio - 1| - err <= tol``.  This is the honest
                        statement when the per-bin errors are a few percent:
                        the strict answer is then partly a statement about the
                        sample size, not about the physics.

    Scanned downwards from ``AGREE_HI`` so the answer is "agreement holds from
    here up", not "there exists one bin that agrees".
    """
    num, nume, _ = d.density(key)
    den, dene, _ = d.density(REF)
    r, re = ratio(num * scale, nume * scale, den, dene)
    out = {}
    for name, ok in (('strict', lambda i: abs(r[i] - 1) <= tol),
                     ('compat', lambda i: abs(r[i] - 1) - re[i] <= tol)):
        edge = None
        for i in range(len(r) - 1, -1, -1):
            if d.centres[i] > AGREE_HI:
                continue
            if not np.isfinite(r[i]):
                break
            if ok(i):
                edge = d.edges[i]
            else:
                break
        out[name] = edge
        if edge is not None:
            j = int(np.argmin(np.abs(d.edges - edge)))
            out[name + '_ratio'] = float(r[j])
            out[name + '_err'] = float(re[j])
    return out


# --------------------------------------------------------------------------
def _panels():
    # The MG7 paper fixes the horizontal size; only the height is ours.
    fig = plt.figure(figsize=(7 * 0.75 * 1.35, 7 * 0.75 * 1.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    rx = fig.add_subplot(gs[1], sharex=ax)
    return fig, ax, rx


def _tx(s_tex, s_plain):
    return s_tex if USETEX else s_plain


def make_figure(d, out, style_tag=''):
    fig, ax, rx = _panels()
    two_mt = d.two_mt
    lo, hi = d.edges[0], d.edges[-1]

    # --- the structurally empty region -----------------------------------
    # Shaded on BOTH panes, and labelled for what it is: below 2 m_t the
    # production sample has no support at all (on-shell tops), and any mode
    # that does not move m_tt inherits that exactly.  Whether the other modes
    # reach it is a property of the process -- for ``p p > t t~ j`` the
    # production reshuffle rescales the recoil jet and so moves m_tt, for a
    # ``2 -> 2`` production there is no recoil to rescale -- so the shading
    # marks the *on-shell kinematic boundary*, not an empty region of the
    # figure.
    for a in (ax, rx):
        a.axvspan(lo, two_mt, facecolor='0.90', edgecolor='none', zorder=0)
        a.axvline(two_mt, color='0.35', lw=1.0, ls=(0, (6, 3)), zorder=1)

    # Every curve divided by its OWN total cross section: the figure is a shape
    # comparison, so the 3.4 % normalisation difference between the truth and
    # MadSpin cancels and what is left is the turn-on.  The absolute rates are
    # not lost -- they are in numbers.txt and in RESULTS.md sections 1a and 2.
    den, dene, dcnt = d.shape(REF)

    for key, _lab in CURVES:
        y, ye, cnt = d.shape(key)
        lab = _lab if USETEX else CURVES_PLAIN[key]
        draw = np.where(cnt > 0, y, np.nan)
        ax.step(d.edges, np.concatenate([draw[:1], draw]), where='pre',
                color=COLOR[key], ls=LS[key], lw=LW, label=lab,
                zorder=5 if key == REF else 4)
        ax.errorbar(d.centres, draw, yerr=np.where(cnt > 0, ye, np.nan),
                    fmt='none', ecolor=COLOR[key], elinewidth=0.9,
                    capsize=0, zorder=4)

    ax.set_yscale('log')
    ax.set_ylabel(_tx(r'$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}m_{t\bar t}$'
                      r' [1/GeV]',
                      r'$(1/\sigma)\,d\sigma/dm_{t\bar t}$ [1/GeV]'))
    ax.set_xlim(lo, hi)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelbottom=False)
    ax.legend(frameon=False, loc='lower right', fontsize=10.5,
              handlelength=2.6, borderaxespad=0.8)

    # One line of setup above the curves, and nothing else in the pane.  The
    # prose that used to sit here -- what the shaded region means, how each mode
    # gets into it, the sub-threshold event counts -- is not on the figure: it
    # is in numbers.txt and in RESULTS.md, where it can carry its errors.  What
    # stays is the setup line, which cannot be read off the curves, and the
    # ``2 m_t`` tag, which names the drawn threshold line rather than commenting
    # on the physics.  ``m_tt`` is defined in RESULTS.md and in this module's
    # docstring; the axis names the variable and no more.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 3.2)
    ax.text(0.028, 0.965,
            _tx(r'$%s$ at $\sqrt{s} = 13$~TeV, LO, '
                r'$\mu_R = \mu_F = m_t$, BW cut $=%g\,\Gamma_t$ on both sides'
                % (PROC_TEX, d.meta.get('bwcutoff', 15.0)),
                r'$%s$ at $\sqrt{s}=13$ TeV, LO, '
                r'$\mu_R=\mu_F=m_t$, BW cut = %g $\Gamma_t$ on both sides'
                % (PROC_PLAIN, d.meta.get('bwcutoff', 15.0))),
            transform=ax.transAxes, ha='left', va='top', fontsize=11)
    ax.annotate(_tx(r'$2m_t$', r'$2m_t$'),
                xy=(two_mt, 0.03), xycoords=('data', 'axes fraction'),
                xytext=(3, 0), textcoords='offset points',
                ha='left', va='bottom', fontsize=11, color='0.35')

    # --- ratio pane -------------------------------------------------------
    rx.axhspan(0.9, 1.1, facecolor=allcolors[0], alpha=0.10, zorder=0)
    rx.axhspan(0.95, 1.05, facecolor=allcolors[0], alpha=0.16, zorder=0)
    rx.axhline(1.0, color='black', lw=0.9, zorder=2)
    rx.text(0.993, 0.93, _tx(r'bands: $\pm5\%$, $\pm10\%$',
                             r'bands: $\pm5\%$, $\pm10\%$'),
            transform=rx.transAxes, ha='right', va='top',
            fontsize=8.5, color=allcolors[0])

    # The pane is clipped to RATIO_CLIP, so it has to be set BEFORE anything is
    # drawn into it: the step lines below run far outside the window and would
    # otherwise drag the autoscale out with them.
    rx.set_ylim(*RATIO_CLIP)

    # Which modes are structurally zero below threshold is a property of the
    # PROCESS -- for ``p p > t t~ j`` only ``onshell``, for a ``2 -> 2``
    # production all four -- and the pane no longer marks them: a structural
    # zero is an exact 0, its step drops off the bottom of the clipped window,
    # and the bins are listed in numbers.txt.  The set is still computed,
    # because it decides which empty bins get an arrow (none of these do).
    struct_of = {key: structurally_empty(d, key) & (dcnt > 0) for key in MODES}

    n_out = 0
    for slot, key in enumerate(MODES):
        # Both sides already divided by their own total sigma, so this pane is
        # a ratio of SHAPES and sits on 1 rather than on the 1.035 the absolute
        # normalisation gave.  Consequence to keep in mind when reading it: an
        # absolute statement -- "onshell misses 16.2 % of the cross section
        # below 2 m_t + 5 GeV" -- can no longer be read off this pane.  Those
        # numbers are in numbers.txt and in RESULTS.md section 2.
        y, ye, cnt = d.shape(key)
        r, re = ratio(y, ye, den, dene)
        # Two kinds of empty bin, handled differently on purpose.
        #   structural -- a mode that provably does not move m_tt, below 2 m_t.
        #                 A real, exact zero: it is 0 and not "somewhere below
        #                 the pane", so it gets NO arrow.  Its step runs to 0
        #                 and leaves the clipped window; the bins and their
        #                 counts are in numbers.txt.
        #   statistical -- any other bin with no entries.  Drawn as a gap: the
        #                 sample simply did not reach there, which is a
        #                 statement about N and not about the scheme.
        struct = struct_of[key]
        stat = (cnt == 0) & (dcnt > 0) & ~struct
        rr = np.where(struct, 0.0, np.where(stat, np.nan, r))
        rx.step(d.edges, np.concatenate([rr[:1], rr]), where='pre',
                color=COLOR[key], ls=LS[key], lw=LW, zorder=4)
        rx.errorbar(d.centres, np.where(struct | stat, np.nan, r),
                    yerr=np.where(struct | stat, np.nan, re), fmt='none',
                    ecolor=COLOR[key], elinewidth=0.9, capsize=0, zorder=4)
        live = np.where(struct | stat, np.nan, r)
        nb, na = offscale_arrows(rx, d.centres, live, COLOR[key],
                                 dx=d.widths, slot=slot, nslot=len(MODES))
        n_out += nb + na

    # A count, printed rather than silent: it is the check that the clipping
    # did not swallow anything.  Every point it counts carries an arrow.
    print('ratio pane clipped to %s: %d point(s) outside it, each drawn as an '
          'arrow at the boundary it left through' % (RATIO_CLIP, n_out))

    # Say it on the axis, not only in the caption: a reader who crops the
    # figure out of the document must still be told the pane is clipped, and
    # that what it compares is shapes.
    rx.set_ylabel(_tx(r'shape ratio' '\n' r'(clipped to $\pm20\%$)',
                      'shape ratio\n(clipped to $\\pm20\\%$)'),
                  fontsize=10.5)
    # The variable and its unit, and nothing else.  What m_tt is built from is
    # in RESULTS.md and in meta.json['observable'].
    rx.set_xlabel(_tx(r'$m_{t\bar t}$ [GeV]', r'$m_{t\bar t}$ [GeV]'))
    rx.xaxis.set_minor_locator(AutoMinorLocator())
    rx.yaxis.set_minor_locator(AutoMinorLocator())
    rx.set_xlim(lo, hi)

    fig.subplots_adjust(left=0.135, right=0.975, top=0.985, bottom=0.105)
    base = os.path.join(out, 'mtt_threshold%s' % style_tag)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=300)
    plt.close(fig)
    return base


def check_minus(pdf_path):
    """Re-open the PDF and confirm a math minus actually made it in.

    matplotlib's usetex PDF subsetting has silently eaten every minus sign in
    this project twice.  The figure carries them in the ratio ticks, so a PDF
    without one is wrong.  Returns (found, detail).
    """
    if not USETEX:
        return None, 'usetex is off, the Type1 subsetting path is not used'
    # Read the file, not a renderer.  The failure mode is precise: the Type1
    # subset of CMSY10 comes out with an /Encoding whose /Differences no longer
    # names ``/minus``, so the glyph is unreachable and viewers drop the sign.
    # A correct PDF for this figure -- which has a minus in every negative ratio
    # tick label and in the annotation -- must therefore carry ``/minus``.
    try:
        with open(pdf_path, 'rb') as fp:
            raw = fp.read()
    except OSError as exc:
        return None, 'cannot read the PDF (%s)' % exc
    if b'/minus' in raw:
        return True, '/minus is in the PDF font encoding'
    # pdftotext, when present, is a second opinion rather than the primary one.
    try:
        import subprocess
        txt = subprocess.run(['pdftotext', pdf_path, '-'],
                             capture_output=True).stdout.decode('utf8', 'replace')
        for ch in ('−', '-'):
            if ch in txt:
                return True, 'pdftotext extracted %r' % ch
    except Exception:
        pass
    return False, ('/minus absent from the PDF font encoding -- the usetex '
                   'Type1 subsetting bug has eaten the sign')


# --------------------------------------------------------------------------
# The normalisation difference, verified rather than asserted.
#
# truth/PA = truth/onshell = truth/madspin_v1 = 0.9661, and the figure divides
# it out, so it has to be understood BEFORE it is divided out.  Everything below
# is computed here, from the model parameters in meta.json, so the estimate can
# be re-run and disagreed with.
#
# MG5 electroweak inputs.  The ``sm`` model takes (Gf, MZ, aEWM1) and DERIVES
# MW; none of the three is in meta.json, so they are the shipped defaults here.
# That is not taken on trust: the derived MW feeds the LO t -> W b width below,
# and that width has to come out equal to the param card's WT, which meta.json
# does record.  It does, to five digits -- see ``normalisation_report``.
# --------------------------------------------------------------------------
SM_GF = 1.166390e-5
SM_MZ = 91.1876
SM_AEWM1 = 132.507


def _sm_mw():
    aew = 1.0 / SM_AEWM1
    return math.sqrt(SM_MZ ** 2 / 2.0
                     + math.sqrt(SM_MZ ** 4 / 4.0
                                 - (aew * math.pi * SM_MZ ** 2)
                                 / (SM_GF * math.sqrt(2))))


def top_width_lo(m, mb, mw):
    """LO ``Gamma(t -> W b)`` at top mass ``m``, keeping ``m_b``.

    Needed at ``m != m_t``: it is the numerator the truth's Breit-Wigner
    carries, and it is emphatically not constant across a +-15 Gamma window.
    """
    xw, xb = (mw / m) ** 2, (mb / m) ** 2
    lam = 1 + xw * xw + xb * xb - 2 * xw - 2 * xb - 2 * xw * xb
    if lam <= 0:
        return 0.0
    return (SM_GF * m ** 3 / (8 * math.pi * math.sqrt(2)) * math.sqrt(lam)
            * ((1 - xb) ** 2 + xw * (1 + xb) - 2 * xw * xw))


def bw_kept_fraction(mt, gt, mb, mw, bwcut, dlnsig_dm=0.0, decay_numerator=True,
                     relativistic=True, npt=200001):
    """Fraction of the NWA rate that survives ``|m - m_t| < bwcut * Gamma_t``.

    The reference is the narrow-width limit, which is what MadSpin's
    ``sigma_production * BR`` is, so a return of 1 would mean "the truncated
    off-shell calculation gives the same rate as the NWA".

    ``relativistic=False`` reproduces the non-relativistic (Cauchy) shortcut,
    ``1 - 2*arctan(2*bwcut)/pi`` removed.  Otherwise the fixed-width propagator
    MG5 actually generates is used -- ``p^2 - M(M - i*Gamma)``, verified in
    ``aloha_writers.py`` -- through the substitution ``s = M^2 + M*Gamma*tan(t)``
    that makes the propagator weight flat in ``t`` and the quadrature exact.

    Two numerator effects, both off by default in the naive estimate and both
    the size of the residual it leaves:

    * ``decay_numerator`` puts back ``m*Gamma(m)/(M*Gamma(M))``, the decay side
      of the resonance.  Over a +-15 Gamma_t window that runs 0.52 to 1.71.
    * ``dlnsig_dm`` is ``d ln sigma_production / d m_t`` PER TOP, which the
      production side supplies and which pulls the other way.  It is an input,
      not a measurement: this study never varied m_t, so it cannot be got from
      the data here.
    """
    th_hi = math.atan(((mt + bwcut * gt) ** 2 - mt ** 2) / (mt * gt))
    th_lo = -math.atan((mt ** 2 - (mt - bwcut * gt) ** 2) / (mt * gt))
    if not relativistic:
        th_hi, th_lo = math.atan(2 * bwcut), -math.atan(2 * bwcut)
        th = np.linspace(th_lo, th_hi, npt)
        m = mt + 0.5 * gt * np.tan(th)
    else:
        th = np.linspace(th_lo, th_hi, npt)
        m = np.sqrt(np.clip(mt ** 2 + mt * gt * np.tan(th), 1e-9, None))
    f = np.exp(dlnsig_dm * (m - mt))
    if decay_numerator:
        g0 = mt * top_width_lo(mt, mb, mw)
        f = f * np.array([mm * top_width_lo(mm, mb, mw) for mm in m]) / g0
    return float(np.trapezoid(f, th) / math.pi)


def normalisation_report(d, p):
    """Why truth/MadSpin is 0.966, how much of it is understood, and how much
    is not.  Printed into ``numbers.txt``; RESULTS.md section 1a quotes it."""
    bw = float(d.meta['bwcutoff'])
    masses = d.meta.get('param_card_masses', {})
    mt = float(masses.get('MT', 173.0))
    gt = float(masses.get('WT', 1.4915))
    mb = float(masses.get('MB', 4.7))
    mw = _sm_mw()

    p('')
    p('-- the normalisation difference, and what accounts for it ------------')
    p('measured, from each sample\'s own sum(w)/N:')
    for key in MODES:
        p('   truth / %-11s = %.5f' % (key, d.sigma(REF) / d.sigma(key)))
    p('   (madspin is the odd one out because of its `joint` overweights;')
    p('    forcing sequential puts it at %.5f, on top of the other three --'
      % (d.sigma(REF) / d.sigma('madspin_seq')
         if 'madspin_seq' in d.meta['runs'] else float('nan')))
    p('    see the control section below.  So there is ONE number to explain.)')
    p('')
    p('   it is a RATE effect, not a shape one: the truth/mode ratio measured')
    p('   in 20 GeV slices from 380 GeV up is flat at 0.966, and dividing out')
    p('   the global total gives the same agreement thresholds as dividing out')
    p('   a local 380-420 GeV anchor (both are listed below).')
    p('')
    ig = d.meta.get('mg5_integration_pb', {})
    if ig:
        p('   not statistics: MadEvent quotes %.1f +- %.4f pb for the truth and'
          % (ig['truth']['cross_pb'], ig['truth']['error_pb']))
        p('   %.1f +- %.4f pb for the production, i.e. 0.03%%, and the five'
          % (ig['production']['cross_pb'], ig['production']['error_pb']))
        p('   truth runs agree to 0.011%.  3.4% is ~100 sigma of that.')
    p('')
    p('   not a branching-ratio mismatch either.  MadSpin normalises to')
    p('   sigma_prod * BR and the truth\'s decay-chain propagator is normalised')
    p('   by the param card\'s WT, so a WT that did not equal the model\'s own')
    p('   LO Gamma(t -> W b) would show up here as a pure rate offset:')
    p('      LO Gamma(t -> W b) at m_t = %g, m_b = %g, MW = %.5f : %.5f GeV'
      % (mt, mb, mw, top_width_lo(mt, mb, mw)))
    p('      param card WT                                       : %.5f GeV'
      % gt)
    p('      implied BR                                          : %.5f'
      % (top_width_lo(mt, mb, mw) / gt))
    p('   0.002%.  (This also validates the derived MW the lines below use.)')
    p('')
    p('what IS in it: MG5\'s decay-chain truth cuts every phase-space point')
    p('with |m - m_t| >= %g Gamma_t (myamp.f, the gForceBW=1 branch: not' % bw)
    p('"onshell" bookkeeping but `cut_bw = .true.`, so the point is rejected')
    p('and the truncation is in the integrated cross section).  MadSpin takes')
    p('no such loss.  Four evaluations of the same integral, NWA-normalised:')
    p('')
    p('   %-58s %8s %8s' % ('per-resonance kept fraction', 'per res', 'pair'))
    rows = [
        ('non-relativistic BW, flat numerator  [1-2*atan(2*%g)/pi]' % bw,
         bw_kept_fraction(mt, gt, mb, mw, bw, decay_numerator=False,
                          relativistic=False)),
        ('fixed-width relativistic BW, flat numerator',
         bw_kept_fraction(mt, gt, mb, mw, bw, decay_numerator=False)),
        ('  + the decay numerator m*Gamma(m)/(m_t*Gamma_t)',
         bw_kept_fraction(mt, gt, mb, mw, bw)),
        ('  + d ln sigma_prod/dm_t = -1.5%/GeV per top (an INPUT)',
         bw_kept_fraction(mt, gt, mb, mw, bw, dlnsig_dm=-0.015)),
    ]
    for lab, k in rows:
        p('   %-58s %8.5f %8.5f' % (lab, k, k * k))
    p('')
    p('   %-58s %8s %8.5f'
      % ('MEASURED truth/mode (PA, onshell, madspin_v1)', '',
         d.sigma(REF) / d.sigma('PA')))
    p('')
    p('VERDICT.  The truncation is confirmed and it is the dominant term: it')
    p('removes about 4% of the rate where the whole difference is 3.4%, so')
    p('nothing else in this comparison is allowed to be large.  The')
    p('non-relativistic shortcut is a good approximation to the PROPAGATOR')
    p('part (%.5f against %.5f for the pair, 0.02%%) -- the relativistic'
      % (rows[0][1] ** 2, rows[1][1] ** 2))
    p('corrections to the two tails very nearly cancel in arctan.')
    p('')
    p('   What is NOT established is the size of the residual.  The estimate')
    p('   that leaves "+0.8% of genuine off-shell rate" holds the numerator')
    p('   flat across a +-%.1f GeV window, and it is not flat: the decay side'
      % (bw * gt))
    p('   alone moves the pair prediction from %.4f to %.4f, and a production'
      % (rows[1][1] ** 2, rows[2][1] ** 2))
    p('   slope of the size m_t variations usually show (-1.5%/GeV per top)')
    p('   moves it back to %.4f.  Against a measurement of %.4f the residual'
      % (rows[3][1] ** 2, d.sigma(REF) / d.sigma('PA')))
    p('   is therefore anywhere from +%.1f%% to +%.1f%% depending on an input'
      % (100 * ((d.sigma(REF) / d.sigma('PA')) / rows[2][1] ** 2 - 1),
         100 * ((d.sigma(REF) / d.sigma('PA')) / rows[3][1] ** 2 - 1)))
    p('   this study does not have.  It is the right SIZE for a finite-width')
    p('   correction -- Gamma_t/m_t = %.3f%% -- and its sign is positive in'
      % (100 * gt / mt))
    p('   every variant tried, but "0.8%" is a leftover, not a measurement,')
    p('   and it should be quoted as a range.  Pinning it down needs a truth')
    p('   run at a second bwcutoff, which was not done: only 15 was run.')
    p('')


# --------------------------------------------------------------------------
def write_numbers(d, out, fh=sys.stdout):
    p = lambda *a: print(*a, file=fh)
    two_mt = d.two_mt
    p('=' * 78)
    p('m_tt near threshold: MadSpin spinmodes vs the doubly-resonant truth')
    p('=' * 78)
    p('code            : %s (%s)' % (d.meta['code_sha'][:12],
                                     d.meta.get('code_branch')))
    p('production      : %s' % d.meta['production_process'])
    p('truth           : %s' % d.meta['truth_process'])
    p('MadSpin decays  : %s' % ', '.join(d.meta['madspin_decays']))
    p('bwcutoff/BW_cut : %s / %s' % (d.meta['bwcutoff'],
                                     d.meta['madspin_BW_cut']))
    p('2 m_t           : %.4f GeV  (param card MT = %s)'
      % (two_mt, d.meta.get('param_card_masses', {}).get('MT')))
    p('')
    p('%-11s %12s %14s %14s' % ('sample', 'events', 'sigma [pb]',
                                'banner [pb]'))
    for key in ['truth'] + MODES + ['production']:
        if key not in d.meta['runs']:
            continue
        p('%-11s %12s %14.4f %14s'
          % (key, '%d' % d.meta['runs'][key]['nevents'], d.sigma(key),
             '%.4f' % d.banner_sigma(key) if d.banner_sigma(key) else '-'))
    p('')
    p('accept/reject scheme each run ACTUALLY used, from its own log:')
    for key in MODES + list(d.meta.get('controls', [])):
        if key not in d.meta['runs']:
            continue
        r = d.meta['runs'][key]
        p('   %-11s %-10s  (%s)' % (key, r.get('unweighting'),
                                    r.get('unweighting_why')))
    p('   NB: "legacy" is not one of the four `set unweighting` schemes.')
    p('   spinmode=madspin_v1 never reaches _unweighting_mode -- that')
    p('   dispatcher lives in run_onshell, which the legacy path does not')
    p('   call -- so the card\'s `unweighting` value is inert for it and the')
    p('   scheme it ran is the legacy one: one max_weight per decay channel,')
    p('   probed before the event loop, then one test of the whole decay')
    p('   chain\'s weight against it per trial, inside the Fortran driver.')

    normalisation_report(d, p)

    # --- the sub-threshold region ----------------------------------------
    st, ste, stk = d.integral(REF, d.fine[0], two_mt)
    # The FULL cross section of the sample, overflow included -- the histogram
    # range stops at 520 GeV and roughly half the events are above it, so
    # integrating the in-range bins would understate every denominator by 2x.
    tot = d.sigma(REF)
    p('-- below 2 m_t ------------------------------------------------------')
    p('truth       sigma(m_tt < 2m_t) = %.4f +- %.4f pb   (%d events, '
      '%.3f%% +- %.3f%% of the truth total %.3f pb)'
      % (st, ste, stk, 100 * st / tot, 100 * ste / tot, tot))
    for key in MODES:
        s, e, k = d.integral(key, d.fine[0], two_mt)
        stot = d.sigma(key)
        rel = (s / st) if st else float('nan')
        rele = rel * math.sqrt((e / s) ** 2 + (ste / st) ** 2) if s > 0 else 0.0
        # A zero here means two different things depending on the mode, and
        # the line must not paper over the difference: ``onshell`` cannot reach
        # this region at any sample size, anything else simply did not.
        if s > 0:
            verdict = '%.3f +- %.3f' % (rel, rele)
        elif key == 'onshell':
            verdict = '0 exactly (structurally empty)'
        else:
            verdict = '0 in this sample (NOT structural -- %s can reach ' \
                      'here; this is a statement about N)' % key
        p('%-11s sigma(m_tt < 2m_t) = %.4f +- %.4f pb   (%d events, '
          '%.3f%% of its own total)   ratio to truth = %s'
          % (key, s, e, k, 100 * s / stot, verdict))
    p('')

    # --- the brief's window ----------------------------------------------
    for width in (5.0, 10.0):
        lo_w, hi_w = d.fine[0], two_mt + width
        s0, e0, k0 = d.integral(REF, lo_w, hi_w)
        p('-- m_tt < 2 m_t + %g GeV  (= %.1f GeV) -------------------------'
          % (width, hi_w))
        p('truth       %.4f +- %.4f pb  (%d events, %.3f%% of the truth total)'
          % (s0, e0, k0, 100 * s0 / tot))
        for key in MODES:
            s, e, k = d.integral(key, lo_w, hi_w)
            diff = s - s0
            diffe = math.sqrt(e ** 2 + e0 ** 2)
            p('%-11s %.4f +- %.4f pb  (%d events)   %+.4f +- %.4f pb '
              'vs truth = %+.1f%% +- %.1f%%'
              % (key, s, e, k, diff, diffe, 100 * diff / s0,
                 100 * diffe / s0))
        p('')

    # --- where agreement returns -----------------------------------------
    p('-- where each spinmode enters agreement with the truth --------------')
    p('   scanned downwards from %g GeV over the plot binning; "strict" uses'
      % AGREE_HI)
    p('   the central ratio, "within errors" allows each bin its own 1 sigma.')
    p('')
    p('   Three normalisations.  The FIRST is the figure\'s.')
    p('     SHAPE, total sigma  -- each side divided by its own total cross')
    p('                            section over the full m_tt range.  This is')
    p('                            what the figure draws, so these are the')
    p('                            thresholds to quote for it.')
    p('     SHAPE, %g-%g GeV  -- the same idea with a local anchor instead.'
      % ANCHOR)
    p('                            A cross-check: it agrees edge for edge with')
    p('                            the total-sigma version, which is the')
    p('                            evidence that the 3.4% is a flat rate')
    p('                            offset and not a shape.')
    p('     ABSOLUTE            -- no rescaling.  Kept because the rate')
    p('                            difference is a real result, but a 5% band')
    p('                            around a plateau at 1.035 is only 1.5%')
    p('                            wide, so that row is half a statement about')
    p('                            the sample size.')
    p('')
    shape_sc, anchor_sc = {}, {}
    for key in MODES:
        shape_sc[key] = shape_scale(d, key)
        sc, rel = anchor_scale(d, key)
        anchor_sc[key] = sc
        p('   %-11s truth/mode: total %.4f   %g-%g GeV %.4f +- %.4f'
          % (key, shape_sc[key], ANCHOR[0], ANCHOR[1], sc, sc * rel))
    p('')
    for what, scales in (
            ('SHAPE, each side over its own TOTAL sigma  (THE FIGURE)',
             shape_sc),
            ('SHAPE, each mode rescaled by its %g-%g GeV anchor (cross-check)'
             % ANCHOR, anchor_sc),
            ('ABSOLUTE normalisation', None)):
        p('   -- %s --' % what)
        for tol in (0.05, 0.10):
            p('   tolerance %d%%:' % int(100 * tol))
            for key in MODES:
                a = agreement_threshold(d, key, tol,
                                        1.0 if scales is None else scales[key])
                def fmt(name):
                    if a.get(name) is None:
                        return 'never, up to %g GeV' % AGREE_HI
                    return ('m_tt >= %.0f GeV  (= 2 m_t %+.0f GeV; first bin '
                            'ratio %.3f +- %.3f)'
                            % (a[name], a[name] - two_mt,
                               a[name + '_ratio'], a[name + '_err']))
                p('      %-11s strict       : %s' % (key, fmt('strict')))
                p('      %-11s within errors: %s' % ('', fmt('compat')))
        p('')

    # --- the mechanism ----------------------------------------------------
    if d.meta.get('delta_mtt'):
        p('-- the mechanism: m_tt(decayed) - m_tt(production), event by event --')
        p('   MadSpin writes one decayed event per production event, in order.')
        p('   sqrt(shat) is RAMBO-invariant, so max |Delta sqrt(shat)| = 0 is')
        p('   what proves the two streams are actually paired.')
        p('   %-11s %9s %8s %9s %13s %9s %7s %10s'
          % ('mode', 'mean', 'rms', 'max|d|', 'max|dsqrt(s)|',
             'down-xing', 'up-xing', 'unchanged'))
        for key in MODES:
            m = d.meta['delta_mtt'].get(key)
            if not m:
                continue
            unch = ('%.2f%%' % (100.0 * m['n_tiny'] / m['n'])
                    if 'n_tiny' in m else '-')
            p('   %-11s %+9.4f %8.4f %9.3f %13.3g %9d %7d %10s'
              % (key, m['mean'], m['rms'], m['max_abs'], m['max_dshat'],
                 m['crossed_down'], m['crossed_up'], unch))
        p('   "unchanged" = |Delta m_tt| / m_tt < 1e-6, i.e. m_tt came back the')
        p('   same to the precision the LHE text carries.  Read that column and')
        p('   not a bit-exact one: onshell provably never touches the momenta')
        p('   and is still only %.2f%% bit-exact, because the LHE round-trips'
          % (100.0 * d.meta['delta_mtt']['onshell']['n_exact']
             / d.meta['delta_mtt']['onshell']['n']))
        p('   the momenta as decimal text.  That is the precision floor; the')
        p('   1e-6 column sits above it.')
        p('   This is the column that explains madspin_v1: its rms is the')
        p('   LARGEST of the four, yet it pushes the FEWEST events across the')
        p('   threshold, because it holds m_tt exactly fixed in half its')
        p('   events and moves the rest further than anything else does.')
        p('')

    # --- controls ---------------------------------------------------------
    for key in d.meta.get('controls', []):
        if key not in d.meta['runs']:
            continue
        base = key.split('_')[0]
        p('-- control: %s ------------------------------------------------'
          % key)
        p('   %s repeated with the accept/reject scheme forced, so the'
          % base)
        p('   %s-vs-PA difference can be attributed to the spinmode and not' % base)
        p('   to "auto" resolving to a different scheme for each of them.')
        p('   resolved unweighting: %-10s vs %-10s for %s'
          % (d.meta['runs'][key].get('unweighting'),
             d.meta['runs'][base].get('unweighting'), base))
        s0, e0, k0 = d.integral(base, d.fine[0], two_mt)
        s1, e1, k1 = d.integral(key, d.fine[0], two_mt)
        p('   sigma(m_tt < 2m_t): %-10s %.4f +- %.4f pb (%d events)'
          % (base, s0, e0, k0))
        p('                       %-10s %.4f +- %.4f pb (%d events)'
          % (key, s1, e1, k1))
        if s0 > 0:
            rr = s1 / s0
            rre = rr * math.sqrt((e1 / s1) ** 2 + (e0 / s0) ** 2)
            p('   control / default = %.4f +- %.4f  (%.1f sigma from 1)'
              % (rr, rre, abs(rr - 1) / rre if rre else float('nan')))
        p('   total sigma: %-10s %.4f pb   %-10s %.4f pb'
          % (base, d.sigma(base), key, d.sigma(key)))
        p('')

    # --- onshell IS the production sample --------------------------------
    if 'production' in d.meta['runs']:
        a = d.z['onshell_cnt']
        b = d.z['production_cnt']
        same = bool(np.array_equal(a, b))
        p('-- onshell vs the undecayed production sample -----------------------')
        p('   Same fine-grid histogram, bin for bin: %s '
          '(%d vs %d entries in range, max |difference| = %d)'
          % (same, int(a.sum()), int(b.sum()), int(np.abs(a - b).max())))
        p('   That is the claim "onshell has no lineshape and no reshuffle"')
        p('   measured rather than asserted, and it is what makes its zero')
        p('   below 2 m_t structural.')
        p('')

    # --- what the clipped ratio pane cannot show --------------------------
    # The figure clips its ratio pane to RATIO_CLIP and marks every excursion
    # with an arrow, but an arrow has no value on it.  The values live here, so
    # nothing that leaves the pane is lost.
    # In the FIGURE's normalisation -- shape ratios, both sides over their own
    # total sigma -- because this list has to be the list of arrows actually
    # drawn.  The absolute ratios are in the first per-bin table below.
    den_c, dene_c, dcnt_c = d.shape(REF)
    p('-- SHAPE ratio points outside the figure\'s clipped pane %s -------'
      % (RATIO_CLIP,))
    any_out = False
    for key in MODES:
        y, ye, cnt = d.shape(key)
        r, re = ratio(y, ye, den_c, dene_c)
        struct = structurally_empty(d, key) & (dcnt_c > 0)
        stat = (cnt == 0) & (dcnt_c > 0) & ~struct
        for i in range(len(r)):
            if struct[i] or stat[i] or not np.isfinite(r[i]):
                continue
            if RATIO_CLIP[0] <= r[i] <= RATIO_CLIP[1]:
                continue
            any_out = True
            p('   %-11s %4.0f-%-4.0f  ratio = %.3f +- %.3f   (drawn as an '
              'arrow at %.1f)'
              % (key, d.edges[i], d.edges[i + 1], r[i], re[i],
                 RATIO_CLIP[1] if r[i] > RATIO_CLIP[1] else RATIO_CLIP[0]))
    if not any_out:
        p('   none.')
    p('   onshell below 2 m_t is NOT in this list: it is an exact structural')
    p('   zero, drawn with its open marker on the lower boundary and no arrow.')
    p('')

    # --- per-bin table ----------------------------------------------------
    # ABSOLUTE, deliberately: the figure is now a shape comparison, so this is
    # the only place the rates themselves survive.  The shape ratios the figure
    # actually draws follow in the second table.
    p('-- per-bin table (absolute, pb/GeV; ratios are ABSOLUTE) -------------')
    den, dene, dcnt = d.density(REF)
    head = '%9s %12s %9s' % ('bin [GeV]', 'truth', '+-')
    for key in MODES:
        head += ' %12s %8s %8s' % (key, '+-', 'ratio')
    p(head)
    for i in range(len(d.centres)):
        row = '%4.0f-%4.0f %12.5g %9.2g' % (d.edges[i], d.edges[i + 1],
                                            den[i], dene[i])
        for key in MODES:
            y, ye, cnt = d.density(key)
            r, re = ratio(y, ye, den, dene)
            if cnt[i] == 0:
                row += ' %12.5g %8s %8s' % (0.0, '-', '0 exact')
            else:
                row += ' %12.5g %8.2g %8s' % (
                    y[i], ye[i],
                    '%.3f' % r[i] if np.isfinite(r[i]) else '-')
        p(row)
    p('')

    # --- the same table in the figure's own normalisation ------------------
    # So a point can be checked against the figure without re-deriving the
    # scaling by hand.  Same bins, same errors; every column is
    # (1/sigma) dsigma/dm in 1/GeV, sigma being the sample's own TOTAL.
    p('-- per-bin table in the FIGURE\'s normalisation ((1/sigma) dsigma/dm,')
    p('   1/GeV; ratios are SHAPE ratios and are what the lower pane draws) --')
    sden, sdene, _ = d.shape(REF)
    head = '%9s %12s %9s' % ('bin [GeV]', 'truth', '+-')
    for key in MODES:
        head += ' %12s %8s %8s' % (key, '+-', 'ratio')
    p(head)
    for i in range(len(d.centres)):
        row = '%4.0f-%4.0f %12.5g %9.2g' % (d.edges[i], d.edges[i + 1],
                                            sden[i], sdene[i])
        for key in MODES:
            y, ye, cnt = d.shape(key)
            r, re = ratio(y, ye, sden, sdene)
            if cnt[i] == 0:
                row += ' %12.5g %8s %8s' % (0.0, '-', '0 exact')
            else:
                row += ' %12.5g %8.2g %8s' % (
                    y[i], ye[i],
                    '%.3f' % r[i] if np.isfinite(r[i]) else '-')
        p(row)
    p('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--check-minus', action='store_true', default=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = Data(args.data)
    base = make_figure(d, args.out)
    print('wrote %s.pdf / .png   (usetex=%s, minus fix applied=%s)'
          % (base, USETEX, MINUS_FIX))
    if args.check_minus:
        ok, detail = check_minus(base + '.pdf')
        print('minus-sign check: %s -- %s' % (ok, detail))
    with open(os.path.join(args.out, 'numbers.txt'), 'w') as fh:
        write_numbers(d, args.out, fh)
    write_numbers(d, args.out)


if __name__ == '__main__':
    main()
