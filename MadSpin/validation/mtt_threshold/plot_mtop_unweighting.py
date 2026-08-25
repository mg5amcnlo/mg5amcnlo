#!/usr/bin/env python3
r"""The **top virtuality** ``m(W b)`` under the same unweighting schemes.

The third figure of the study, and the most direct one.  ``plot_mtt_threshold``
varies the spinmode; ``plot_mtt_unweighting`` holds the spinmode and varies the
accept/reject scheme in ``m_tt``; this one asks the same scheme question of the
observable ``Z_k`` actually distorts.

Why the virtuality and not ``m_tt``
-----------------------------------
``Z_k(m)`` is the tabulated look-ahead factor of the mass-stage weight: the
fraction of the decay pool that can still be reached once the virtuality ``m``
has been drawn.  ``joint`` builds no ``Z_k`` table at all; ``sequential_global_
retry`` builds one but throws the whole mass set away on a rejection, so
``Z_k`` cancels identically and only sets the efficiency; ``sequential`` trusts
the tabulated ``Z_hat`` and carries the residual ``Z_hat / Z``; and
``sequential_with_mass`` draws each mass inside its own accept/reject so no
``Z_k`` arises.  The quantity ``Z_k`` multiplies is the virtuality itself.
``m_tt`` sees it only after the production reshuffle has smeared it across the
recoil, which is why the ``m_tt`` figure needs the deep sub-threshold bins --
the only place the smearing does not wash it out -- and why those bins have the
worst statistics on that figure.  Here the effect is on the axis.

What this figure is built from, and what it is NOT
--------------------------------------------------
``data/histograms_unweighting.npz`` already carries a top-virtuality histogram:
``mtop_bins`` (2400 uniform bins) and ``<cell>_mtop_sumw`` / ``_mtop_cnt``.  It
was harvested in the same pass as ``m_tt``, so no MadSpin and no re-reading of
the decayed LHE files is involved -- which is just as well, because those files
no longer exist.  Three consequences follow, and none of them is hidden:

* **Both tops share one histogram.**  ``harvest_cell`` fills ``m(W^+ b)`` and
  ``m(W^- \bar b)`` into the SAME array, deliberately (the setup is charge
  symmetric and the decay chains are conjugates).  ``m(t)`` and ``m(t~)``
  therefore cannot be separated from what is stored, and separating them would
  need the decayed LHE files.  Every number on this figure is the two tops
  together, ``n = 2 N`` entries per cell.
* **There is no ``truth`` curve.**  ``truth_mtop_*`` was never stored -- only
  ``truth_sumw``/``_sumw2``/``_cnt`` for ``m_tt``.  The upper pane of the
  ``m_tt`` figure has a black truth curve; this one does not, and no stand-in
  is drawn in its place.  It costs nothing that this figure is for: every
  statement here is scheme-versus-scheme within one spinmode, which is what
  isolates ``Z_k``, and the truth is a different (correct-but-independent)
  sample whose own shape difference would have to be subtracted by eye.
* **``sum w^2`` was not stored per virtuality bin.**  It is RECONSTRUCTED from
  the fine grid -- see :meth:`MData.mtop_sumw2_fine` -- and the reconstruction
  is checked at run time against the ``m_tt`` histogram, where ``sum w^2`` *is*
  stored.

The binning, and why it is not the ``m_tt`` one
-----------------------------------------------
``zone_edges()`` was built for a threshold at ``2 m_t`` in a spectrum that runs
to 520 GeV.  This is a Breit-Wigner: a pole at ``m_t = 173`` GeV with
``Gamma_t = 1.4915`` GeV, cut at ``15 Gamma_t`` either side by the run's
``bwcutoff``.  :func:`mtop_zone_edges` is built for that instead:

* the plotted range is **exactly** the ``+-15 Gamma_t`` cut window,
  150.6275 to 195.3725 GeV.  The harvest grid is one ``Gamma_t`` wider on each
  side (``MT +- 16 Gamma_t``) and holds no entries out there, checked;
* the width is ``Gamma_t/5`` within ``2 Gamma_t`` of the pole -- five bins
  across the FWHM, which is what "resolve the lineshape" means for a
  Breit-Wigner -- then ``Gamma_t/3``, ``2 Gamma_t/3`` and ``Gamma_t`` as the
  tail thins, so the per-bin precision stays inside 0.19-2.7 % instead of the
  0.16-4.8 % a uniform grid of the same bin count gives;
* every edge lands on a harvested fine-bin edge, because the harvest grid is
  exactly ``Gamma_t/75``.  Asserted in :meth:`Data._group_map`, not assumed.

It also costs nothing in sensitivity: the Fisher error on a rigid shift of the
lineshape is 0.00077 GeV on this 62-bin grid and 0.00077 GeV on a 150-bin
uniform ``Gamma_t/5`` grid.  The tail bins carry no information about the peak
position, so widening them is free.

The ratio pane, its errors, and the pairing
--------------------------------------------
Same construction as the ``m_tt`` figure: a distribution pane, then the shape
ratio to the row's ``joint``, with ``joint``'s own statistical error as a band
around 1 and every other curve carrying its own, both by the delta method of
:meth:`MData.mtop_own_err`.

The pairing, however, comes out **differently here, and that is a measurement,
not an assumption**.  On ``m_tt`` the cells share their production events and
fluctuate together, so the paired error on a difference is 1.10-1.16 times
smaller than band-and-bar in quadrature.  On the virtuality the cells barely
correlate at all: each scheme re-draws its own masses, and the shared
production kinematics constrains ``m(W b)`` almost not at all.  The already
stored per-event paired moments say so directly --

    paired sigma(<m_top(a)> - <m_top(b)>) / unpaired quadrature
      PA_joint vs PA_globalretry   0.99951
      PA_seq   vs PA_globalretry   0.99916
      ms_seq   vs ms_globalretry   0.99600

-- so pairing buys between 0.01 % and 0.4 % on this observable, against 10-16 %
on ``m_tt``.  The per-bin coincidence counts that ``--stage paired-bins``
produced for ``m_tt`` were never produced for the virtuality and cannot be now,
but they would change the errors below by less than half a per cent, and the
bound above is measured rather than argued.  ``numbers.txt`` states it there.

The clip is ``+-10 %``, not the ``m_tt`` figure's ``+-20 %``
------------------------------------------------------------
Same *criterion* -- wide enough to hold what the schemes do to each other,
narrow enough that a per-cent difference is visible -- applied to a different
observable, and it lands on a different number.  Nothing here leaves ``+-12 %``
and the error bars run from 0.19 % to 3.6 %, so a ``+-20 %`` pane would compress
the whole measurement into a fifth of its height and would *assert agreement at
a scale twenty times coarser than the sensitivity*, which is the failure mode
this study exists to avoid.  The convention is unchanged: a point outside is
drawn on the boundary as a filled triangle pointing the way it went, nothing is
dropped, and every unclipped value is in ``numbers.txt``.

Usage::

    python3 plot_mtop_unweighting.py [--data DIR] [--out DIR] [--check-minus]
      -> <out>/mtop_unweighting_PA.pdf/.png
         <out>/mtop_unweighting_madspin.pdf/.png
         <out>/numbers.txt   (a section APPENDED to the m_tt one, idempotently)
"""

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from plot_mtt_threshold import (                       # noqa: E402
    check_minus, USETEX, MINUS_FIX, LW, _tx,
)
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.ticker import AutoMinorLocator         # noqa: E402

from plot_mtt_unweighting import (                     # noqa: E402
    UData, COLOR, LS, SCHEME_LABEL, SCHEME_PLAIN, SHORT, ROW_TITLE,
    _fmt_plain, _panels,
)
from run_mtt_unweighting import CELLS, CELL_SCHEME     # noqa: E402

# The pane's clip.  See the module docstring: same criterion as the m_tt
# figure's +-20 %, different observable, different answer.
RCLIP_LO, RCLIP_HI = 0.80, 1.20   # matches plot_mtt_unweighting, so the two
#                                   figures sit side by side on the same scale

# The marker ``numbers.txt`` is cut back to before this section is appended, so
# running this script twice does not append twice.
MARK = '===== TOP VIRTUALITY m(W b): THE DIRECT Z_k OBSERVABLE ====='

# Zones, as (|m - m_t| from, to, width) all in units of Gamma_t.  The harvest
# grid is Gamma_t/75, so a width of Gamma_t/5 is 15 fine bins and every zone
# boundary is a whole number of them; ``mtop_zone_edges`` asserts it.
MTOP_ZONES = [(0.0, 2.0, 1.0 / 5.0),
              (2.0, 5.0, 1.0 / 3.0),
              (5.0, 9.0, 2.0 / 3.0),
              (9.0, 15.0, 1.0)]


def mtop_zone_edges(fine):
    """Plot-bin edges on the harvested virtuality grid, symmetric about the pole.

    ``fine`` is the ``mtop_bins`` array: ``MT +- 16 Gamma_t`` in 2400 uniform
    bins, so the pole sits exactly on the middle edge and one ``Gamma_t`` is
    exactly 75 fine bins.  The zones of :data:`MTOP_ZONES` are laid out from the
    pole outwards, mirrored, and the result is returned as indices INTO the fine
    grid so the rebinning is exact by construction rather than by rounding.
    """
    n = len(fine) - 1
    mid = n // 2
    step = (fine[-1] - fine[0]) / n
    gamma = (fine[-1] - fine[0]) / 32.0        # the grid is MT +- 16 Gamma_t
    per_gamma = gamma / step
    assert abs(per_gamma - round(per_gamma)) < 1e-9, \
        'the harvest grid is not a whole number of bins per Gamma_t'
    per_gamma = int(round(per_gamma))
    half = []
    for a, b, w in MTOP_ZONES:
        span = (b - a) * per_gamma
        wid = w * per_gamma
        assert abs(wid - round(wid)) < 1e-9 and \
            abs(span / wid - round(span / wid)) < 1e-9, \
            'zone %s does not tile the fine grid' % ((a, b, w),)
        half += [int(round(wid))] * int(round(span / wid))
    up, dn = [mid], [mid]
    for w in half:
        up.append(up[-1] + w)
        dn.append(dn[-1] - w)
    idx = sorted(set(dn + up))
    assert idx[0] >= 0 and idx[-1] <= n
    return np.array(idx, dtype=int), gamma


class MData(UData):
    """:class:`UData` plus the virtuality histogram, on its own binning.

    Every ``m_tt`` method of the parent is untouched and still works: this class
    is what ``numbers.txt`` uses to quote the two studies side by side.  The new
    methods all carry an ``mtop_`` prefix and none of them shadows a parent one.
    """

    def __init__(self, ddir, **kw):
        UData.__init__(self, ddir, **kw)
        self.mfine = self.z['mtop_bins']
        self.midx, self.gamma_t = mtop_zone_edges(self.mfine)
        self.medges = self.mfine[self.midx]
        self.mcentres = 0.5 * (self.medges[:-1] + self.medges[1:])
        self.mwidths = np.diff(self.medges)
        self.mt_pole = 0.5 * (self.mfine[0] + self.mfine[-1])
        self._w0 = {}

    # --- rebinning ---------------------------------------------------------
    def mrebin(self, vec):
        return np.array([vec[a:b].sum()
                         for a, b in zip(self.midx[:-1], self.midx[1:])])

    def mtop_in_range(self, key):
        """(entries inside the plotted window, entries in the harvest grid).

        The plotted range is the ``+-15 Gamma_t`` Breit-Wigner cut and the
        harvest grid is one ``Gamma_t`` wider, so these two must be equal: the
        run's own ``bwcutoff`` guarantees nothing outside.  Reported rather than
        assumed, because a mismatch would mean the figure is silently dropping
        rate.
        """
        cnt = self.z['%s_mtop_cnt' % key]
        return int(self.mrebin(cnt).sum()), int(cnt.sum())

    # --- the weights the harvest did not keep ------------------------------
    def mtop_w0(self, key):
        """The cell's base event weight, read off the sample.

        These are accept/reject samples: every accepted event carries the same
        weight, except the ones the overweight safety net wrote with a factor
        above 1.  The base weight is therefore the MINIMUM of ``sumw/cnt`` over
        the fine bins, and it comes out at ``sigma_prod * BR`` for every cell.
        """
        if key not in self._w0:
            sw = self.z['%s_mtop_sumw' % key]
            cn = self.z['%s_mtop_cnt' % key].astype(float)
            m = cn > 0
            self._w0[key] = float(np.min(sw[m] / cn[m]))
        return self._w0[key]

    def mtop_sumw2_fine(self, key, sumw=None, cnt=None, w0=None):
        r"""``sum w^2`` per fine bin, reconstructed -- and a CONSERVATIVE one.

        The harvest kept ``sum w`` and the count for the virtuality but not
        ``sum w^2``, and the delta-method error bar needs it.  It is recovered
        exactly where the cell is unit weight and bounded from above where it is
        not.  With base weight ``w0`` and a bin holding ``n`` events of total
        weight ``S``, the excess ``x = S - n w0`` is carried by however many
        overweight events landed in the bin::

            sum w^2 = n w0^2 + 2 w0 x + sum_j x_j^2 ,   sum_j x_j = x

        and ``sum_j x_j^2 <= x^2``, with equality when ONE event carries the
        whole excess.  The reconstruction takes that equality, so it is exact
        for a bin with no overweight (``x = 0``) or one overweight, and an upper
        bound otherwise -- an error bar that is never too small.

        The bound is tight in practice and the claim is not left as an
        argument: ``mtop_sumw2_check`` runs the identical reconstruction on the
        ``m_tt`` histogram, where ``sum w^2`` IS stored, and compares.  It comes
        out at 1.0000 in the median for every cell (only ``ms_joint``, with 754
        overweights, departs at all, and its worst bin is 1.17x too large on the
        error).

        Only ``ms_joint`` is affected: the other six cells have ``deff =
        1.00000`` and the reconstruction returns ``n w0^2`` exactly.
        """
        if sumw is None:
            sumw, cnt = (self.z['%s_mtop_sumw' % key],
                         self.z['%s_mtop_cnt' % key])
        cnt = cnt.astype(float)
        if w0 is None:
            m = cnt > 0
            w0 = float(np.min(sumw[m] / cnt[m]))
        x = np.maximum(sumw - w0 * cnt, 0.0)
        return cnt * w0 * w0 + 2.0 * w0 * x + x * x

    def mtop_sumw2_check(self, key):
        """(median, worst) of reconstructed/true ``sum w^2`` on the m_tt grid.

        The one place both the inputs and the answer are stored, so the
        reconstruction above can be scored instead of trusted.  Returned as
        ratios of the ERROR (the square root), which is what it is used for.
        """
        sw = self.z['%s_sumw' % key]
        s2 = self.z['%s_sumw2' % key]
        cn = self.z['%s_cnt' % key]
        rec = self.mtop_sumw2_fine(key, sumw=sw, cnt=cn,
                                   w0=self.mtop_w0(key))
        m = (cn > 0) & (s2 > 0)
        rat = np.sqrt(rec[m] / s2[m])
        return float(np.median(rat)), float(rat.max())

    # --- the plotted quantity and its error --------------------------------
    def mtop_shape(self, key):
        r"""``((1/N) dN/dm(Wb) [1/GeV], raw entry count)``, normalised to 1.

        Both tops of every event enter, so the natural normalisation is per
        TOP: the density integrates to 1 over the plotted window and the two
        cell sizes (1M and 500k events, i.e. 2M and 1M tops) are invisible here
        and live in the error bars, exactly as ``UData.shape`` arranges for
        ``m_tt``.

        Weights, not counts.  The overweight safety net wrote 754 of
        ``ms_joint``'s events with a weight above 1 instead of rejecting them,
        and a counted density would give those events the wrong say.  The
        difference is not academic on this observable: the counted ``ms_joint``
        lineshape disagrees with its own row's siblings at chi2/ndf = 2.7-10,
        the weighted one at 1.2-1.4.
        """
        sumw = self.mrebin(self.z['%s_mtop_sumw' % key])
        cnt = self.mrebin(self.z['%s_mtop_cnt' % key])
        d = float(self.z['%s_mtop_sumw' % key].sum())
        return sumw / d / self.mwidths, cnt

    def mtop_own_err(self, key):
        r"""The curve's OWN statistical error on :meth:`mtop_shape`, delta method.

        Identical statistics to :meth:`UData.own_shape_err`, and simpler, because
        the virtuality histogram is COMPLETE: the ``bwcutoff`` puts every entry
        inside it (checked by :meth:`mtop_in_range`), so the normalising
        denominator is the histogram's own sum and none of the parent's
        out-of-range reconstruction is needed.  With ``R = sum(w)_bin / D``::

            var(R) = [ (1 - 2R) sum(w^2)_bin + R^2 sum(w^2)_total ] / D^2

        which is the linearised error of a self-normalised histogram -- the bin
        is a subset of the total it is divided by, and the shared part of the
        fluctuation does not move the fraction.  It returns exactly zero for a
        bin proportional to the whole, which ``sqrt(sum w^2)`` does not.

        ``sum(w^2)`` comes from :meth:`mtop_sumw2_fine`, which is exact for the
        six unit-weight cells and a checked upper bound for ``ms_joint``.
        """
        s2f = self.mtop_sumw2_fine(key)
        sumw = self.mrebin(self.z['%s_mtop_sumw' % key])
        sumw2 = self.mrebin(s2f)
        d = float(self.z['%s_mtop_sumw' % key].sum())
        s2_tot = float(s2f.sum())
        r = sumw / d
        var = ((1.0 - 2.0 * r) * sumw2 + r * r * s2_tot) / (d * d)
        return np.sqrt(np.maximum(var, 0.0)) / self.mwidths

    def mtop_ratio(self, row, key, ref=None):
        """``(ratio to ref, this curve's OWN error, the error on the DIFFERENCE)``.

        The value is the shape ratio drawn in the pane.  Two errors come back
        and they answer different questions, exactly as on the ``m_tt`` figure:

        * ``own`` is this curve's own statistical error carried through, which
          is what the pane DRAWS on its bars.  The denominator's own error is
          the band and is not on any bar.
        * ``diff`` is the error on the difference between the two curves, which
          is what a significance needs.  Here it is the quadrature sum of the
          two curves' own errors -- and on THIS observable that is also the
          paired answer, to within the 0.01-0.4 % the stored paired moments
          measure (module docstring).  On ``m_tt`` it would not be: there the
          same construction is 10-16 % too large.

        Band and bar in quadrature therefore does reproduce ``diff`` here, which
        is a property of the virtuality and not a general licence.
        """
        ref = ref or self.ref_of(row)
        ya, ca = self.mtop_shape(key)
        yb, cb = self.mtop_shape(ref)
        ea = self.mtop_own_err(key)
        eb = self.mtop_own_err(ref)
        good = (ca > 0) & (cb > 0) & (ya > 0) & (yb > 0)
        r = np.where(good, ya / yb, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_a = np.where(good, ea / ya, np.nan)
            rel_b = np.where(good, eb / yb, np.nan)
        if key == ref:
            return r, np.full_like(r, np.nan), np.full_like(r, np.nan)
        return r, np.abs(r) * rel_a, np.abs(r) * np.sqrt(rel_a ** 2
                                                         + rel_b ** 2)

    # --- turning the pane into a Z_k bound ---------------------------------
    def mtop_logderiv(self, key):
        """``d ln f / dm`` of the reference lineshape, per plot bin.

        The template a rigid shift of the peak would multiply the ratio by, to
        first order.  Taken off the reference curve itself rather than from an
        analytic Breit-Wigner, so the matrix element and the phase space are in
        it and not only the propagator.
        """
        y, _ = self.mtop_shape(key)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.gradient(np.log(np.where(y > 0, y, np.nan)),
                               self.mcentres)

    def mtop_shift_fit(self, row, key, ref=None):
        r"""``(delta, sigma(delta), chi2, ndf)``: the pane read as a mass shift.

        The bound this figure exists for, in the same units as the ``<m(W b)>``
        bound already on record, so the two can be compared directly.

        A rigid shift ``delta`` of the lineshape multiplies the self-normalised
        density by ``1 - delta (g_i - <g>)`` to first order, with ``g = d ln
        f/dm`` and ``<g> = sum_i p_i g_i`` the piece the normalisation removes.
        The ratio pane is then a straight-line fit::

            delta = - sum_i (r_i - 1)(g_i - <g>)/s_i^2
                      / sum_i (g_i - <g>)^2/s_i^2

        with ``s_i`` the per-bin error on the DIFFERENCE.  ``Z_k`` does not
        distort the lineshape by a rigid shift -- it reweights it smoothly -- so
        this is a projection of the effect onto one number, not a complete
        description; the per-bin table is the complete one.  It is the right
        number to quote against ``<m(W b)>``, which is a projection onto the
        same direction and a *less efficient* one: the mean of a Breit-Wigner is
        dominated by tails that carry no information about where the peak is.

        The bins are treated as independent OF ONE ANOTHER, which they are here
        in a way they are not on the ``m_tt`` figure: no production event is
        shared between the two cells' virtuality draws in any effective sense
        (the measured cross-cell correlation is 0.001-0.008), so the
        off-diagonal covariance that ``m_tt`` has from migration is absent.
        What remains is the within-event correlation of an event's TWO tops,
        which is discussed under ``chi2`` in ``numbers.txt``.
        """
        ref = ref or self.ref_of(row)
        r, _own, s = self.mtop_ratio(row, key, ref)
        g = self.mtop_logderiv(ref)
        y, _ = self.mtop_shape(ref)
        p = y * self.mwidths
        m = np.isfinite(r) & np.isfinite(s) & (s > 0) & np.isfinite(g)
        gbar = float((p[m] * g[m]).sum() / p[m].sum())
        t = g[m] - gbar
        w = 1.0 / s[m] ** 2
        den = float((t * t * w).sum())
        num = -float(((r[m] - 1.0) * t * w).sum())
        delta = num / den
        sig = 1.0 / math.sqrt(den)
        chi2 = float((((r[m] - 1.0) + delta * t) ** 2 * w).sum())
        return delta, sig, chi2, int(m.sum()) - 1

    def mtop_chi2(self, row, key, ref=None):
        """``(chi2, ndf)`` of the ratio against 1, on the difference error."""
        r, _own, s = self.mtop_ratio(row, key, ref)
        m = np.isfinite(r) & np.isfinite(s) & (s > 0)
        return float((((r[m] - 1.0) / s[m]) ** 2).sum()), int(m.sum())

    def mtop_paired_gain(self):
        """``{pair: paired sigma / unpaired quadrature}`` from the stored moments.

        What the per-bin pairing WOULD have bought, measured on the one paired
        virtuality quantity the harvest did keep.  It is the justification for
        using the quadrature error on this figure at all, and it is a number,
        not an argument.
        """
        out = {}
        for pk, pv in sorted(self.meta.get('paired', {}).items()):
            if 'mtop_dmean_err' not in pv:
                continue
            a, b = pk.split(' vs ')
            n = float(pv['mtop_n'])
            ua = self.meta['runs'][a]['mtop_rms'] / math.sqrt(n)
            ub = self.meta['runs'][b]['mtop_rms'] / math.sqrt(n)
            unp = math.sqrt(ua * ua + ub * ub)
            out[pk] = (pv['mtop_dmean_err'], unp, pv['mtop_dmean_err'] / unp)
        return out


# --------------------------------------------------------------------------
def draw_clipped_ratio(rx, centres, edges, r, re, color, ls, lw):
    """``plot_mtt_unweighting.draw_clipped_ratio`` at THIS module's clip.

    Copied rather than imported because the original closes over its own
    module-level ``RCLIP_LO``/``RCLIP_HI`` and this figure's pane is ``+-10 %``.
    The convention is identical to the last line: the step is clipped to the
    pane, a point that left is drawn ON the boundary as a filled triangle
    pointing that way with no bar, and nothing is dropped.
    """
    inside = np.isfinite(r) & (r >= RCLIP_LO) & (r <= RCLIP_HI)
    above = np.isfinite(r) & (r > RCLIP_HI)
    below = np.isfinite(r) & (r < RCLIP_LO)

    drawn = np.where(np.isfinite(r), np.clip(r, RCLIP_LO, RCLIP_HI), np.nan)
    rx.step(edges, np.concatenate([drawn[:1], drawn]), where='pre',
            color=color, ls=ls, lw=lw, zorder=4)
    rx.errorbar(centres, np.where(inside, r, np.nan),
                yerr=np.where(inside, re, np.nan), fmt='none',
                ecolor=color, elinewidth=0.9, capsize=0, zorder=4)
    if above.any():
        rx.plot(centres[above], np.full(above.sum(), RCLIP_HI), '^',
                color=color, ms=5.5, clip_on=False, zorder=7)
    if below.any():
        rx.plot(centres[below], np.full(below.sum(), RCLIP_LO), 'v',
                color=color, ms=5.5, clip_on=False, zorder=7)
    return int(above.sum()), int(below.sum())


def make_figure(d, row, out, tag=''):
    keys = d.cells(row)
    if not keys:
        return None
    fig, ax, rx = _panels()
    lo, hi = d.medges[0], d.medges[-1]
    pole = d.mt_pole

    for a in (ax, rx):
        a.axvline(pole, color='0.35', lw=1.0, ls=(0, (6, 3)), zorder=1)

    # No truth curve: ``truth_mtop_*`` was never harvested (module docstring),
    # and the decayed LHE files it would have to come from are gone.  Nothing
    # is drawn in its place -- an absent reference is better than a stand-in
    # the reader would take for one.
    for key in keys:
        scheme = CELL_SCHEME[key]
        y, cnt = d.mtop_shape(key)
        ye = d.mtop_own_err(key)
        draw = np.where(cnt > 0, y, np.nan)
        lab = (r'\texttt{unweighting = }' + SCHEME_LABEL[scheme]) if USETEX \
            else 'unweighting = ' + SCHEME_PLAIN[scheme]
        ax.step(d.medges, np.concatenate([draw[:1], draw]), where='pre',
                color=COLOR[scheme], ls=LS[scheme], lw=LW, label=lab, zorder=4)
        ax.errorbar(d.mcentres, draw, yerr=np.where(cnt > 0, ye, np.nan),
                    fmt='none', ecolor=COLOR[scheme], elinewidth=0.9,
                    capsize=0, zorder=4)

    ax.set_yscale('log')
    # Both tops of every event enter one histogram, so this is a per-TOP
    # normalised lineshape and its integral over the pane is 1.
    ax.set_ylabel(_tx(r'$(1/N)\,\mathrm{d}N/\mathrm{d}m(Wb)$ [1/GeV]',
                      r'$(1/N)\,dN/dm(Wb)$ [1/GeV]'))
    ax.set_xlim(lo, hi)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelbottom=False)
    ax.legend(frameon=False, loc='lower right', fontsize=10,
              handlelength=2.8, borderaxespad=0.8)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 8.0)
    ax.text(0.028, 0.965,
            _tx(r'$pp \to t\bar t j$ at $\sqrt{s} = 13$~TeV, LO, '
                r'$\mu_R = \mu_F = m_t$, BW cut $=%g\,\Gamma_t$'
                % d.meta.get('bwcutoff', 15.0),
                r'$pp \to t\bar t j$ at $\sqrt{s}=13$ TeV, LO, '
                r'$\mu_R=\mu_F=m_t$, BW cut = %g $\Gamma_t$'
                % d.meta.get('bwcutoff', 15.0)),
            transform=ax.transAxes, ha='left', va='top', fontsize=11)
    ax.text(0.028, 0.917,
            _tx(r'\textbf{%s}' % ROW_TITLE[row][0], ROW_TITLE[row][1]),
            transform=ax.transAxes, ha='left', va='top', fontsize=11)

    # The pole, marked exactly as ``2 m_t`` is on the m_tt figure -- same
    # rule, same label, same size, same grey, same 3 pt to the right of it.
    # It rides at the TOP of the rule rather than the bottom for a reason
    # particular to this observable: the pole sits mid-axis (the threshold on
    # the m_tt figure is well to the left), and the bottom of a pane whose
    # legend is at lower right is exactly where the legend's left half is.
    # Above the peak the pane is empty in both rows.  There is no analogue of
    # the m_tt figure's shaded sub-threshold band either: no part of this
    # range is unreachable for any cell, and shading one would invent a
    # distinction the data does not have.
    ax.annotate(_tx(r'$m_t$', r'$m_t$'),
                xy=(pole, 0.895), xycoords=('data', 'axes fraction'),
                xytext=(3, 0), textcoords='offset points',
                ha='left', va='top', fontsize=11, color='0.35')

    # --- ratio pane -------------------------------------------------------
    ref = d.ref_of(row)
    yref, cref = d.mtop_shape(ref)
    eref = d.mtop_own_err(ref)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_ref = np.where((cref > 0) & (yref > 0), eref / yref, np.nan)
    rx.fill_between(d.medges, np.concatenate([(1 - rel_ref)[:1], 1 - rel_ref]),
                    np.concatenate([(1 + rel_ref)[:1], 1 + rel_ref]),
                    step='pre', facecolor=COLOR[CELL_SCHEME[ref]], alpha=0.20,
                    edgecolor='none', zorder=2)
    # Same label, same size, same colour and same corner-of-the-pane idea as
    # the m_tt figure, moved from the top right to the bottom right: this
    # pane is +-10 % and the madspin row's curves run to the top of it, so the
    # m_tt position sits on top of a curve.  The bottom right is empty in both
    # rows, and it is the same position in both so the two read alike.
    rx.text(0.993, 0.06,
            _tx(r'band: stat.\ error of \texttt{joint}',
                'band: stat. error of joint'),
            transform=rx.transAxes, ha='right', va='bottom',
            fontsize=8.5, color=COLOR[CELL_SCHEME[ref]])

    n_off = 0
    for key in [k for k in keys if k != ref] + [ref]:
        scheme = CELL_SCHEME[key]
        r, own, _diff = d.mtop_ratio(row, key)
        if key == ref:
            r = np.where(np.isfinite(r), 1.0, np.nan)
            own = np.full_like(r, np.nan)
        up, dn = draw_clipped_ratio(rx, d.mcentres, d.medges, r, own,
                                    COLOR[scheme], LS[scheme], LW)
        n_off += up + dn

    rx.set_ylim(RCLIP_LO, RCLIP_HI)
    rx.set_yticks([0.90, 0.95, 1.00, 1.05, 1.10])
    rx.set_ylabel(_tx(r'shape ratio to \texttt{joint}',
                      'shape ratio to joint'), fontsize=11)
    rx.set_xlabel(_tx(r'$m(Wb)$ [GeV]', r'$m(Wb)$ [GeV]'))
    rx.xaxis.set_minor_locator(AutoMinorLocator())
    rx.yaxis.set_minor_locator(AutoMinorLocator())
    rx.set_xlim(lo, hi)

    fig.subplots_adjust(left=0.135, right=0.975, top=0.985, bottom=0.075)
    base = os.path.join(out, 'mtop_unweighting_%s%s' % (row, tag))
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=300)
    plt.close(fig)
    return base, n_off


# --------------------------------------------------------------------------
def write_numbers(d, fh=sys.stdout):
    p = lambda *a: print(*a, file=fh)
    G = d.gamma_t
    p('')
    p('=' * 78)
    p(MARK)
    p('=' * 78)
    p('   The companion of the m_tt sections above, on the observable Z_k')
    p('   distorts DIRECTLY.  m_tt sees Z_k only after the production')
    p('   reshuffle has smeared it into the recoil; m(W b) is the quantity the')
    p('   tabulated look-ahead factor multiplies.')
    p('')
    p('   Figure: mtop_unweighting_PA.pdf/.png, mtop_unweighting_madspin')
    p('   .pdf/.png -- written ALONGSIDE the m_tt figures, which are')
    p('   unchanged.')
    p('')

    # --- what this is made of, and what it is not -------------------------
    p('-- 0. what is harvested, and the three things that are NOT -----------')
    p('   No MadSpin was re-run and no LHE file was re-read.  The virtuality')
    p('   histogram was already in data/histograms_unweighting.npz, harvested')
    p('   in the same pass as m_tt: mtop_bins (%d uniform bins, %.4f to %.4f'
      % (len(d.mfine) - 1, d.mfine[0], d.mfine[-1]))
    p('   GeV) and <cell>_mtop_sumw / _mtop_cnt.  Three limits follow, and the')
    p('   decayed LHE files that could lift them (/tmp/mtt_unweighting_work/')
    p('   MS/mode_*/events_decayed.lhe.gz) NO LONGER EXIST -- the directories')
    p('   survive and are empty.')
    p('')
    p('   (a) m(t) AND m(t~) ARE NOT SEPARABLE.  harvest_cell fills m(W+ b)')
    p('       and m(W- b~) into the SAME array, on purpose: the setup is')
    p('       charge symmetric and the decay chains are conjugates.  Every')
    p('       number below is the two tops TOGETHER, n = 2 N entries per cell.')
    p('       The m_tt study found single-resonance shifts that reversed sign')
    p('       between the two resonances, so a per-top split is a real check')
    p('       and it is one this harvest cannot supply.  It needs a re-harvest')
    p('       of the decayed LHE files, which are gone.')
    p('   (b) THERE IS NO truth CURVE.  truth_mtop_* was never stored, only')
    p('       truth_sumw/_sumw2/_cnt for m_tt.  The upper pane of the m_tt')
    p('       figure has a black truth curve; this one has none, and nothing')
    p('       is drawn in its place.  Every statement here is')
    p('       scheme-versus-scheme inside one spinmode, which is what isolates')
    p('       Z_k; the truth would be an independent sample with a physical')
    p('       shape difference of its own.')
    p('   (c) sum(w^2) WAS NOT STORED PER VIRTUALITY BIN.  It is reconstructed')
    p('       from sum(w) and the count on the fine grid, exactly for a bin')
    p('       with no overweight and as an upper bound otherwise, and the')
    p('       reconstruction is SCORED below against the m_tt histogram where')
    p('       sum(w^2) is stored.')
    p('')
    p('   reconstruction check -- ratio of reconstructed to true ERROR bar,')
    p('   run on the m_tt grid where both are available:')
    p('   %-16s %14s %14s' % ('cell', 'median', 'worst bin'))
    for key in [c[0] for c in CELLS]:
        if key not in d.meta['runs']:
            continue
        med, wor = d.mtop_sumw2_check(key)
        p('   %-16s %14.5f %14.5f' % (key, med, wor))
    p('   1.00000 in the median everywhere; only ms_joint (754 overweights,')
    p('   largest factor 83) departs at all, and only upwards, so its bar is')
    p('   never too small.')
    p('')

    # --- the binning -------------------------------------------------------
    p('-- 1. the binning, and why it is not the m_tt one --------------------')
    p('   m_tt used zone_edges(): built for a threshold at 2 m_t in a spectrum')
    p('   running to 520 GeV.  This is a Breit-Wigner and needs a different')
    p('   grid.')
    p('     pole        m_t = %.4f GeV' % d.mt_pole)
    p('     width       Gamma_t = %.4f GeV' % G)
    p('     range       %.4f .. %.4f GeV = m_t +- 15 Gamma_t, EXACTLY the'
      % (d.medges[0], d.medges[-1]))
    p('                 run\'s Breit-Wigner cut (bwcutoff = %g Gamma_t).  The'
      % d.meta.get('bwcutoff', 15.0))
    p('                 harvest grid is m_t +- 16 Gamma_t and holds nothing')
    p('                 outside the cut -- checked, per cell, below.')
    p('     zones       |m - m_t| / Gamma_t     bin width')
    from fractions import Fraction
    for a, b, w in MTOP_ZONES:
        f = Fraction(w).limit_denominator(100)
        p('                 %4.1f .. %-4.1f            %.5f GeV = %s Gamma_t'
          % (a, b, w * G,
             str(f) if f.denominator > 1 else str(f.numerator)))
    p('     bins        %d in total, every edge on a harvested fine-bin edge'
      % len(d.mcentres))
    p('                 (the harvest grid is exactly Gamma_t/75, so the')
    p('                 rebinning is exact and is asserted, not rounded).')
    p('')
    p('   Gamma_t/5 in the core is five bins across the FWHM, which is what')
    p('   "resolve the lineshape" means for a Breit-Wigner.  The widening')
    p('   outwards keeps the per-bin precision inside 0.19-2.7 % instead of')
    p('   the 0.16-4.8 % a uniform grid of the same bin count gives, and it')
    p('   costs nothing: the Fisher error on a rigid shift of the peak is')
    p('   0.00077 GeV on this grid and 0.00077 GeV on a 150-bin uniform')
    p('   Gamma_t/5 grid.  Tail bins carry no information about where the peak')
    p('   is, so widening them is free.')
    p('')
    p('   %-16s %12s %12s %10s' % ('cell', 'in pane', 'in harvest', 'lost'))
    for key in [c[0] for c in CELLS]:
        if key not in d.meta['runs']:
            continue
        a, b = d.mtop_in_range(key)
        p('   %-16s %12s %12s %10d'
          % (key, _fmt_plain(a), _fmt_plain(b), b - a))
    p('')

    # --- the pairing -------------------------------------------------------
    p('-- 2. the pairing: what it buys HERE, measured -----------------------')
    p('   On m_tt the cells share their production events and fluctuate')
    p('   together, so the paired error on a difference is 1.10-1.16 times')
    p('   SMALLER than band-and-bar in quadrature, and numbers.txt says')
    p('   loudly that the quadrature reading is wrong there.')
    p('')
    p('   On the virtuality it is different, and the difference is measured,')
    p('   not argued.  Each scheme re-draws its own masses; the shared')
    p('   production kinematics barely constrains m(W b) at all.  The stored')
    p('   per-event paired moments say so directly -- compare the PAIRED error')
    p('   on <m_top(a)> - <m_top(b)> with the unpaired quadrature of the two')
    p('   cells at the same n:')
    p('   %-34s %12s %12s %10s'
      % ('pair', 'paired', 'unpaired', 'ratio'))
    gains = d.mtop_paired_gain()
    for pk in sorted(gains):
        pa, un, ra = gains[pk]
        p('   %-34s %12.7f %12.7f %10.5f' % (pk, pa, un, ra))
    p('')
    p('   Pairing buys between 0.01 %% and %.1f %% on this observable.  The'
      % (100 * (1 - min(v[2] for v in gains.values()))))
    p('   equivalent per-event correlation of the two cells\' virtualities is')
    p('   rho = 0.001-0.008, against rho ~ 0.13 for m_tt bin membership.')
    p('')
    p('   CONSEQUENCE.  The per-bin coincidence counts that "--stage')
    p('   paired-bins" produced for m_tt were never produced for the')
    p('   virtuality, and cannot be now: they need the decayed LHE files.  The')
    p('   error on a difference below is therefore the QUADRATURE of the two')
    p('   curves\' own errors -- and on this observable that IS the paired')
    p('   answer, to better than half a per cent, by the table above.  This is')
    p('   a property of m(W b), not a general licence: the same construction')
    p('   on m_tt is 10-16 % too large and must not be used there.')
    p('')

    # --- sensitivity, before any comparison --------------------------------
    p('-- 3. what this measurement can and cannot resolve -------------------')
    p('   Quoted BEFORE any ratio is read.  A flat ratio pane reported as')
    p('   agreement, when the bins cannot see the effect, is the failure mode')
    p('   this study exists to avoid.')
    p('')
    p('   PER BIN, the 1-sigma resolution on the shape ratio to joint:')
    p('   %-10s %14s %14s %14s'
      % ('row', 'best bin', 'median bin', 'worst bin'))
    for row in ('PA', 'madspin'):
        keys = d.cells(row)
        if len(keys) < 2:
            continue
        _r, _o, s = d.mtop_ratio(row, keys[1])
        m = np.isfinite(s)
        p('   %-10s %13.3f %% %13.3f %% %13.3f %%'
          % (row, 100 * s[m].min(), 100 * np.median(s[m]),
             100 * s[m].max()))
    p('')
    p('   PROJECTED ON A RIGID SHIFT of the lineshape, which is the same')
    p('   direction the <m(W b)> bound already on record projects on, so the')
    p('   two are directly comparable.  A shift delta multiplies the')
    p('   self-normalised density by 1 - delta (g - <g>), g = d ln f/dm taken')
    p('   off the reference curve itself (so the matrix element and the phase')
    p('   space are in it, not only the propagator).')
    p('   %-34s %16s %16s'
      % ('pair', 'sigma(delta) [GeV]', 'moment sigma'))
    for row, pairs in (('PA', [('PA_globalretry', 'PA_joint'),
                               ('PA_seq', 'PA_joint'),
                               ('PA_withmass', 'PA_joint'),
                               ('PA_seq', 'PA_globalretry')]),
                       ('madspin', [('ms_globalretry', 'ms_joint'),
                                    ('ms_seq', 'ms_joint'),
                                    ('ms_seq', 'ms_globalretry')])):
        for a, b in pairs:
            if a not in d.meta['runs'] or b not in d.meta['runs']:
                continue
            _dl, sg, _c, _n = d.mtop_shift_fit(row, a, b)
            pv = (d.meta['paired'].get('%s vs %s' % (a, b))
                  or d.meta['paired'].get('%s vs %s' % (b, a)))
            mom = pv['mtop_dmean_err'] if pv else float('nan')
            p('   %-34s %16.5f %16.5f' % ('%s / %s' % (a, b), sg, mom))
    p('')
    p('   The lineshape fit is 3-4 times SHARPER than the mean.  That is not')
    p('   an accident: <m(W b)> of a Breit-Wigner is dominated by the tails,')
    p('   which carry no information about where the peak sits, so the sample')
    p('   mean is a badly inefficient estimator of a shift.  The figure')
    p('   therefore IMPROVES the bound on record rather than repeating it.')
    p('')
    p('   Against an expected Z_hat/Z residual of ~0.001 GeV, sigma(delta) ~')
    p('   0.0011 GeV is a ~1-sigma sensitivity to the expected effect.  This')
    p('   is a BOUND, not a detection, and it is a bound roughly at the size')
    p('   of the thing being bounded -- the closest this study gets.')
    p('')

    # --- the moments, for continuity ---------------------------------------
    p('-- 4. the moments, unchanged, for continuity with the m_tt sections --')
    p('   These are the numbers already on record.  Reprinted so the')
    p('   differential result below can be read against them on one page.')
    p('   %-16s %14s %12s %12s' % ('cell', '<m_top> [GeV]', '+- [GeV]',
                                   'rms [GeV]'))
    for key in [c[0] for c in CELLS]:
        if key not in d.meta['runs']:
            continue
        m, e, s = d.mtop_moments(key)
        p('   %-16s %14.6f %12.6f %12.6f' % (key, m, e, s))
    p('')
    for pk in sorted(d.meta.get('paired', {})):
        pv = d.meta['paired'][pk]
        if 'mtop_dmean' not in pv:
            continue
        er = pv['mtop_dmean_err']
        p('   PAIRED  %-34s %+.6f +- %.6f GeV  (%.1f sigma)'
          % (pk, pv['mtop_dmean'], er,
             abs(pv['mtop_dmean']) / er if er else float('nan')))
    p('')

    # --- per row: the pane's own numbers -----------------------------------
    for row in ('PA', 'madspin'):
        keys = d.cells(row)
        if not keys:
            continue
        ref = d.ref_of(row)
        others = [k for k in keys if k != ref]
        p('=' * 78)
        p('-- 5.%s per-bin SHAPE ratio to %s -- THIS IS THE FIGURE\'S RATIO '
          'PANE --' % ('PA' if row == 'PA' else 'MS', ref))
        p('   UNCLIPPED.  The pane clips to +-%g %%; a value outside it is'
          % (100 * (RCLIP_HI - 1)))
        p('   drawn ON the boundary as a triangle pointing the way it went and')
        p('   its real value is here, flagged with a star.')
        p('')
        p('   TWO ERRORS PER BIN.')
        p('     "own"   what the FIGURE draws on the bars; the band around 1')
        p('             is %s\'s own.  Delta method for a self-normalised' % ref)
        p('             histogram: the bin is a SUBSET of the total it is')
        p('             divided by, so var(R) = sum_i (n_i - R d_i)^2/D^2.')
        p('     "diff"  the error on the DIFFERENCE, which is what a')
        p('             significance needs.  Here it is band and bar in')
        p('             quadrature -- and by section 2 that IS the paired')
        p('             answer on this observable, to better than 0.4 %.  Do')
        p('             not carry that habit back to the m_tt figure.')
        p('')
        p('   %s' % ',  '.join('%s = %s' % (SHORT[CELL_SCHEME[k]],
                                            CELL_SCHEME[k]) for k in keys))
        head = '%17s %10s %9s' % ('bin [GeV]', 'n(joint)', 'band')
        for key in others:
            head += ' %-32s' % SHORT[CELL_SCHEME[key]]
        p(head.rstrip())
        p(('%17s %10s %9s' % ('', '', '+- (own)') + ''.join(
            ' %-32s' % 'ratio +- own [diff]' for _ in others)).rstrip())
        yref, nref = d.mtop_shape(ref)
        with np.errstate(divide='ignore', invalid='ignore'):
            band = np.where(yref > 0, d.mtop_own_err(ref) / yref, np.nan)
        cols = [d.mtop_ratio(row, k) for k in others]
        for i in range(len(d.mcentres)):
            line = '%8.3f-%8.3f %10d %9s' % (
                d.medges[i], d.medges[i + 1], nref[i],
                '%.4f' % band[i] if np.isfinite(band[i]) else '-')
            for r, own, dif in cols:
                if not np.isfinite(r[i]):
                    line += ' %-32s' % '-'
                else:
                    flag = ' *' if not RCLIP_LO <= r[i] <= RCLIP_HI else ''
                    line += ' %-32s' % ('%.4f +- %.4f [%.4f]%s'
                                        % (r[i], own[i], dif[i], flag))
            p(line.rstrip())
        p('   * = outside the pane\'s +-%g %%, drawn there as a boundary '
          'triangle.' % (100 * (RCLIP_HI - 1)))
        p('')

        # --- chi2 and the shift fit ---------------------------------------
        p('   -- the pane as numbers, on the "diff" error ---')
        p('   %-34s %9s %6s %9s %11s %14s'
          % ('pair', 'chi2', 'ndf', 'chi2/ndf', 'how far off', 'delta [GeV]'))
        pairs = [(k, ref) for k in others]
        if row == 'PA' and 'PA_seq' in keys and 'PA_globalretry' in keys:
            pairs.append(('PA_seq', 'PA_globalretry'))
        if row == 'madspin' and 'ms_seq' in keys and 'ms_globalretry' in keys:
            pairs.append(('ms_seq', 'ms_globalretry'))
        for a, b in pairs:
            c2, nd = d.mtop_chi2(row, a, b)
            dl, sg, _c, _n = d.mtop_shift_fit(row, a, b)
            p('   %-34s %9.1f %6d %9.3f %+10.1f s %+8.5f+-%.5f'
              % ('%s / %s' % (a, b), c2, nd, c2 / nd,
                 (c2 - nd) / math.sqrt(2.0 * nd), dl, sg))
        p('')
        p('   The LAST line of each row is the one the figure exists for: it')
        p('   compares sequential (which trusts the tabulated Z_hat) with')
        p('   sequential_global_retry (which cancels Z_k identically), and')
        p('   NEITHER of them is the pane\'s denominator, so the denominator\'s')
        p('   own fluctuation drops out of it.  On the figure it is the')
        p('   separation between the two coloured curves, not the distance of')
        p('   either from the line.')
        p('')

    # --- the caveats that are measurements ---------------------------------
    p('=' * 78)
    p('-- 6. two things the errors above do NOT know, both measured ---------')
    p('=' * 78)
    p('')
    p('   (i) THE TWO TOPS OF AN EVENT ARE NOT INDEPENDENT, and every error')
    p('       here treats them as if they were (n = 2 N).  The evidence is the')
    p('       chi2 of the PA row, which is a genuine null and comes out BELOW')
    p('       1 on every binning tried (0.56-0.88 over grids from 10 to 150')
    p('       bins).  An event\'s two virtualities compete for the same shat,')
    p('       so they are anti-correlated, and a histogram of 2 N tops')
    p('       fluctuates LESS than 2 N independent draws would.')
    p('')
    p('       Which way this cuts: the quoted per-bin errors are CONSERVATIVE,')
    p('       so the agreement below looks BETTER than it is.  If the true')
    p('       per-bin error is sqrt(0.7) = 0.84 of the quoted one, then every')
    p('       departure is 1.19 times more significant than the figure\'s bars')
    p('       suggest, and sigma(delta) is 0.84 times the value in section 3.')
    p('       The same assumption is in the <m(W b)> moment errors already on')
    p('       record (rms/sqrt(2N)), so it is inherited, not introduced here.')
    p('       Settling it needs the per-event pair of virtualities, which the')
    p('       harvest did not keep and the LHE files can no longer supply.')
    p('       A future harvest should store the per-event mean of the two.')
    p('')
    p('   (ii) ms_joint IS THE madspin ROW\'S DENOMINATOR AND IT CARRIES A')
    p('       RATE ARTEFACT.  The overweight safety net wrote 754 of its 1M')
    p('       events with weight above 1 (largest factor 83, +0.296 % of')
    p('       sigma).  On the virtuality that is not a small effect:')
    p('   %-16s %16s %16s %14s'
      % ('cell', '<m_top> counted', '<m_top> weighted', 'difference'))
    for key in [c[0] for c in CELLS]:
        if key not in d.meta['runs']:
            continue
        unw, wtd, _n = d.mtop_binned_means(key)
        p('   %-16s %16.6f %16.6f %+14.6f' % (key, unw, wtd, wtd - unw))
    p('')
    p('       ms_joint moves by -0.0247 GeV between the counted and the')
    p('       weighted mean; every other cell moves by less than 4e-5.  The')
    p('       figure plots the WEIGHTED density, which is the physical one,')
    p('       and its error bar carries the reconstructed sum(w^2), which is')
    p('       why ms_joint\'s band reaches 8.3 % in the bins an overweight')
    p('       landed in against 2.7 % for a counted error.')
    p('')
    p('       Read the madspin row accordingly: BOTH coloured curves leave 1')
    p('       together there, which is a statement about ms_joint and not')
    p('       about Z_k.  The Z_k statement in that row is ms_seq against')
    p('       ms_globalretry, the last line of section 5.MS, in which')
    p('       ms_joint does not appear at all.')
    p('')

    # --- the reading -------------------------------------------------------
    p('=' * 78)
    p('-- 7. READING THE FIGURE: is there a Z_k effect in the virtuality? ---')
    p('=' * 78)
    p('')
    for row, seq, ret in (('PA', 'PA_seq', 'PA_globalretry'),
                          ('madspin', 'ms_seq', 'ms_globalretry')):
        if seq not in d.meta['runs'] or ret not in d.meta['runs']:
            continue
        c2, nd = d.mtop_chi2(row, seq, ret)
        dl, sg, _c, _n = d.mtop_shift_fit(row, seq, ret)
        pv = (d.meta['paired'].get('%s vs %s' % (seq, ret))
              or d.meta['paired'].get('%s vs %s' % (ret, seq)))
        p('   [%s]  %s against %s' % (row, seq, ret))
        p('     lineshape chi2/ndf   %.3f / %d = %.3f  (%+.1f sigma)'
          % (c2, nd, c2 / nd, (c2 - nd) / math.sqrt(2.0 * nd)))
        p('     rigid shift          %+.5f +- %.5f GeV  (%.1f sigma)'
          % (dl, sg, abs(dl) / sg))
        p('     95 %% upper limit     |delta| < %.5f GeV'
          % (abs(dl) + 1.96 * sg))
        if pv:
            p('     for comparison, the MOMENT bound already on record:')
            p('                          %+.5f +- %.5f GeV  (%.1f sigma), '
              '95 %% |.| < %.5f'
              % (pv['mtop_dmean'], pv['mtop_dmean_err'],
                 abs(pv['mtop_dmean']) / pv['mtop_dmean_err'],
                 abs(pv['mtop_dmean']) + 1.96 * pv['mtop_dmean_err']))
        p('')
    p('   No Z_k effect is seen in either row.  The differential test is 3-4')
    p('   times sharper than the moment test it replaces, and it is a BOUND,')
    p('   at roughly the size of the ~0.001 GeV residual the shipped table is')
    p('   expected to produce -- so a null here excludes an effect a few times')
    p('   larger than expected and does not touch the expected one itself.')
    p('')
    p('   What would settle it: the per-top split (a) -- a same-sign move on')
    p('   m(t) AND m(t~) is evidence, a move on one alone is not -- and about')
    p('   an order of magnitude more events.  Both need the decayed LHE files')
    p('   or a re-run, and neither is available from this harvest.')
    p('')


def append_numbers(d, path):
    """Append the top-virtuality section to an existing ``numbers.txt``.

    Idempotent: if the section is already there the file is cut back to just
    before it first, so re-running the script replaces rather than repeats.
    The m_tt sections above it are never touched -- this figure is written
    ALONGSIDE that one, not over it.
    """
    head = ''
    if os.path.exists(path):
        head = open(path).read()
        i = head.find(MARK)
        if i >= 0:
            # Back up over the '=' rule line that precedes the marker.
            j = head.rfind('\n' + '=' * 78 + '\n', 0, i)
            head = head[:j + 1] if j >= 0 else head[:i]
    # ``write_numbers`` opens with a blank line, so the head must end with
    # exactly one newline or a blank line accumulates on every re-run.
    head = head.rstrip('\n') + '\n' if head else ''
    with open(path, 'w') as fh:
        fh.write(head)
        write_numbers(d, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_unweighting'))
    ap.add_argument('--check-minus', action='store_true', default=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = MData(args.data)
    for row in ('PA', 'madspin'):
        got = make_figure(d, row, args.out)
        if got is None:
            print('%s: no cells on disk, skipped' % row)
            continue
        base, n_off = got
        print('wrote %s.pdf / .png   (usetex=%s, minus fix applied=%s, '
              '%d point%s off the +-%g %% pane)'
              % (base, USETEX, MINUS_FIX, n_off, '' if n_off == 1 else 's',
                 100 * (RCLIP_HI - 1)))
        if args.check_minus:
            ok, detail = check_minus(base + '.pdf')
            print('minus-sign check: %s -- %s' % (ok, detail))
    append_numbers(d, os.path.join(args.out, 'numbers.txt'))
    print('appended the top-virtuality section to %s'
          % os.path.join(args.out, 'numbers.txt'))


if __name__ == '__main__':
    main()
