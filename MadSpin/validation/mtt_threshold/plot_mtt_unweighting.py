#!/usr/bin/env python3
"""``dsigma/dm_tt`` near the ``2 m_t`` threshold, one **unweighting scheme**
against another within a fixed off-shell spinmode, in the MG7 paper's style.

The companion of ``plot_mtt_threshold.py``.  That figure varies the spinmode;
this one holds the spinmode fixed and varies the accept/reject scheme, so that
what is left on the figure is the ``Z_k`` question:

  * ``joint`` has no mass stage and never reads the tabulated ``Z_k``;
  * ``sequential_global_retry`` has a mass stage but a rejected decay throws the
    mass set away, so ``Z_k`` cancels identically and only sets the efficiency;
  * ``sequential`` trusts the tabulated ``Z_hat``, and its residual bias is
    exactly ``Z_hat / Z``;
  * ``sequential_with_mass`` (``PA`` only) draws each mass inside its own
    accept/reject, so no ``Z_k`` arises at all.

The first two must agree within statistics.  Whether this measurement can *see*
a sub-per-cent departure of the third is a question about the sample size, and
``numbers_unweighting.txt`` answers it explicitly rather than reporting an
inconclusive ratio as agreement.

The ratio pane divides by ``joint``, not by the truth
-----------------------------------------------------
``joint`` builds no ``Z_k`` table and ``sequential_global_retry`` cancels
``Z_k`` identically, so *those two agreeing is the null hypothesis* -- and a
pane that divides both by the truth makes the reader subtract two large,
common, physical shape differences by eye to see it.  Dividing by ``joint``
puts the null on the line: ``sequential_global_retry`` flat at 1 is the null
holding, and ``sequential`` -- the only scheme that trusts the tabulated
``Z_hat`` -- departing from it is the residual ``Z_hat/Z``.  The truth is not
drawn in the pane at all; it is still the black curve of the upper pane, where
the absolute lineshape is the thing being shown.

What the pane's error bars are, and what they are NOT
-----------------------------------------------------
The pane draws **each curve's own statistical error**.  The band around 1 is
``joint``'s own error, bin by bin -- ``joint`` is the denominator, so its
statistics are not a property of any other curve and cannot be carried on their
bars; and every other curve carries its own.  Both come from
:meth:`UData.own_shape_err`, the **delta-method** error for a self-normalised
histogram: the plotted quantity is ``(1/sigma) dsigma/dm``, the bin content is a
*subset* of the ``sigma`` it is divided by, and

    R = N/D,  N = sum_i n_i, D = sum_i d_i,  var(R) = sum_i (n_i - R d_i)^2/D^2

is the linearised (jackknife) error that keeps that within-sample correlation.
It returns exactly zero when ``n_i ∝ d_i``, which a plain ``sqrt(sum w^2)``
does not; ``sqrt(sum w^2)`` is wrong here and, on these bins, 0.2-0.6 % too
large.  This is the same estimator ``zz_pol_weights/pol_analysis.ratio`` uses
for exactly this case.

**These bars must not be added in quadrature to read a difference off the
pane.**  The cells are *not* independent of each other: every one of them
decays the SAME production events in the same order (``max |Delta sqrt(shat)|``
is 0 across a row, checked), so the production-level fluctuation is common to a
curve and to ``joint`` and largely cancels in their difference.  Combining the
band and a bar in quadrature discards that cancellation and **overestimates**
the error on the difference -- by a median factor 1.16 per bin in the ``PA``
row and 1.10 in ``madspin``, measured, not assumed.

The paired error is still computed, still in ``numbers.txt`` and is still the
number to quote for any significance.  ``run_mtt_unweighting --stage
paired-bins`` re-reads the decayed LHE files and counts, per plot bin, how many
production events land in *that* bin under a cell and under its row's
``joint``; :meth:`UData.paired_ratio` turns those into the covariance a
scheme-versus-``joint`` error needs.  Per-window coincidences cannot do this: an
event can be in one window under both schemes and in a different BIN under each.
The window table at the end of ``numbers.txt`` is the exact statement, and
``sequential_global_retry``/``joint`` = 0.99996 +- 0.00109 against
``sequential``/``joint`` = 0.99800 +- 0.00108 is the physics this figure exists
for.  Read that from the table, never by eye off the bars.

Two figures, one per spinmode:

    plot_mtt_unweighting.py [--data DIR] [--out DIR]
      -> <out>/mtt_unweighting_PA.pdf/.png
         <out>/mtt_unweighting_madspin.pdf/.png
         <out>/numbers.txt

The ratio pane is **clipped to +-20 %**.  Points outside are drawn *on* the
boundary as filled triangles pointing the way they went, never silently cut off.
The unclipped value of every such point is in the per-bin table of
``numbers.txt``, which is also where the pairing is spelled out curve by curve.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Everything shared with the first figure -- the minus-sign workaround, the
# rcParams, the zone binning, the ratio error propagation, the integrals -- is
# imported.  ``plot_mtt_threshold`` is NOT modified by this file.
from plot_mtt_threshold import (                       # noqa: E402
    Data, zone_edges, check_minus, USETEX, MINUS_FIX, LW, allcolors,
    ANCHOR, AGREE_HI, _tx,
)
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.ticker import AutoMinorLocator         # noqa: E402

from run_mtt_unweighting import (                      # noqa: E402
    CELLS, CELL_SPINMODE, CELL_SCHEME, ROWS, NULL_PAIRS, WINDOWS,
)

REF = 'truth'


def _fmt_plain(n):
    """Thousands-separated integer for the TEXT report.

    ``plot_mtt_threshold._fmt_int`` groups with a LaTeX thin space (``\,``),
    which is right on the figure and wrong in ``numbers.txt`` -- there it just
    prints the backslashes.  Same grouping, plain space.
    """
    return '{:,}'.format(int(n)).replace(',', ' ')


def _naive_over_delta(d, key):
    """Median ``sqrt(sum w^2)`` bar divided by the delta-method one.

    The size of the mistake the pane would make if it took its own statistical
    error the easy way.  Reported rather than asserted; see
    :meth:`UData.own_shape_err`.
    """
    _, naive, cnt = d.shape(key)
    delta = d.own_shape_err(key)
    m = (cnt > 0) & (delta > 0)
    return float(np.median(naive[m] / delta[m])) if m.any() else float('nan')


def _quad_over_paired(d, row):
    """Median of (band and bar in quadrature) / (paired error), over the row.

    The price of reading a difference off the drawn bars.  It is above 1 by
    construction -- the quadrature sum is the paired error with the covariance
    thrown away -- and how far above is a measured property of these samples,
    not a rule of thumb.
    """
    ref = d.ref_of(row)
    yref, _, cref = d.shape(ref)
    with np.errstate(divide='ignore', invalid='ignore'):
        band = np.where(cref > 0, d.own_shape_err(ref) / yref, np.nan)
    out = []
    for key in d.cells(row):
        if key == ref:
            continue
        r, re = d.paired_ratio(row, key)
        _, ro = d.own_ratio_err(row, key)
        quad = np.sqrt(ro ** 2 + (np.abs(r) * band) ** 2)
        m = np.isfinite(quad) & np.isfinite(re) & (re > 0)
        out.append(quad[m] / re[m])
    return float(np.median(np.concatenate(out))) if out else float('nan')

# The ratio pane's clip.  +-20 % is wide enough to hold everything the schemes
# do to each other above threshold and narrow enough that a per-cent-level
# difference is visible at all; the deep sub-threshold bins go outside it, and
# that is what the arrows are for.
RCLIP_LO, RCLIP_HI = 0.80, 1.20

SCHEME_LABEL = {
    'joint': r'\texttt{joint}',
    'sequential': r'\texttt{sequential}',
    'sequential_global_retry': r'\texttt{sequential\_global\_retry}',
    'sequential_with_mass': r'\texttt{sequential\_with\_mass}',
}
SCHEME_PLAIN = {k: v.replace(r'\texttt{', '').replace('}', '')
                   .replace('\\_', '_') for k, v in SCHEME_LABEL.items()}
# Column headings for the per-bin table, where the full names do not fit.
SHORT = {'joint': 'joint', 'sequential': 'seq',
         'sequential_global_retry': 'seq_glob_retry',
         'sequential_with_mass': 'seq_with_mass'}

# joint and global_retry are the null-hypothesis pair, so they get the two
# strong colours; sequential -- the one that reads the table -- gets the third.
COLOR = {'joint': 'blue', 'sequential': 'red',
         'sequential_global_retry': allcolors[2],
         'sequential_with_mass': allcolors[4]}
LS = {'joint': 'solid', 'sequential': 'dashed',
      'sequential_global_retry': 'dashdot',
      'sequential_with_mass': (0, (1, 1.4))}

ROW_TITLE = {
    'PA': (r'\texttt{spinmode = PA}', 'spinmode = PA'),
    'madspin': (r'\texttt{spinmode = madspin}', 'spinmode = madspin'),
}


class UData(Data):
    """:class:`plot_mtt_threshold.Data` over this study's ``.npz``/``.json``.

    The parent hard-codes ``histograms.npz`` and ``meta.json`` in its
    constructor.  Rather than edit that file -- it is being changed in parallel
    for the spinmode figure -- the seven lines of setup are repeated here with
    the two names swapped.  Every *method* (``density``, ``integral``,
    ``sigma``, ``_rebin``, ``_group_map``) is the parent's, unmodified, so the
    two studies rebin and normalise identically by construction.
    """

    def __init__(self, ddir, npz='histograms_unweighting.npz',
                 meta='meta_unweighting.json',
                 pbins='paired_bins_unweighting.json'):
        self.z = np.load(os.path.join(ddir, npz))
        self.meta = json.load(open(os.path.join(ddir, meta)))
        p = os.path.join(ddir, pbins)
        if not os.path.exists(p):
            raise SystemExit(
                '%s is missing.  The ratio pane divides one cell by another '
                'cell of the same production sample; the error on that '
                'DIFFERENCE is paired and needs the per-bin coincidence '
                'counts, and numbers.txt quotes it for every significance '
                'even though the pane itself draws per-curve statistics.  '
                'Produce them with\n'
                '    python3 run_mtt_unweighting.py --stage paired-bins\n'
                'which re-reads the decayed LHE files and runs no MadSpin.'
                % p)
        self.pb = json.load(open(p))
        self.fine = self.z['bins']
        self.edges = zone_edges()
        # The coincidences are counted on a grid, and the pane is drawn on a
        # grid.  If the two ever drift apart the ratios would keep their
        # errors from the wrong bins and nothing would look wrong, so the
        # agreement is asserted rather than assumed.
        if not np.allclose(self.pb['edges'], self.edges):
            raise SystemExit(
                '%s was counted on a different binning from the one this '
                'figure draws.  Re-run "run_mtt_unweighting.py --stage '
                'paired-bins".' % p)
        self.centres = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.widths = np.diff(self.edges)
        self.two_mt = float(self.meta.get('two_mt', 346.0))
        self._groups = self._group_map()

    def shape(self, key):
        """((1/sigma) dsigma/dm [1/GeV], its error, raw event count).

        The companion figure moved to a self-normalised spectrum and this one
        follows it.  ``sigma`` is the parent's :meth:`sigma`, the sample's TOTAL
        cross section over the full ``m_tt`` range (``sum(w)/N`` of the whole
        file), not the integral of the plotted window -- normalising to the
        window would divide out part of the region under study and would make
        the curves depend on the plot limits.

        It is added HERE rather than in ``plot_mtt_threshold`` for the reason
        the class docstring gives: that file is being changed in parallel on
        another branch, which is where its own ``shape`` lives.  The definition
        is copied from it verbatim so the two studies normalise identically.

        The per-cell ``N`` comes from ``meta['runs'][key]['nevents']`` via the
        parent's :meth:`nevents`, which matters more here than it did there:
        the cells are NOT all the same size (five hold 1M, ``ms_seq`` and
        ``ms_globalretry`` hold 500k), so every density on this figure is a
        per-event quantity by construction and the 2:1 split cannot leak into
        the normalisation.  It survives only in the error bars, which is where
        it belongs.
        """
        y, ye, cnt = self.density(key)
        s = self.sigma(key)
        return y / s, ye / s, cnt

    def _sumw2_total(self, key):
        """``sum_i w_i^2`` over the WHOLE file, which the harvest half-holds.

        The normalisation of :meth:`shape` is the sample's total ``sigma`` over
        the full ``m_tt`` range, so the delta-method sum below runs over every
        event of the file -- but ``histograms_unweighting.npz`` only carries
        ``sum w^2`` inside the histogram's 290-520 GeV grid, which holds 55.7 %
        of the rate.  The rest is reconstructed from the design effect measured
        IN range, ``deff = n sum(w^2)/sum(w)^2``:

            sum(w^2)_out ~ deff_in * sum(w)_out^2 / n_out

        with ``sum(w)_out`` and ``n_out`` both exact (``meta`` has the file's
        total ``sumw`` and ``nevents``, the ``.npz`` has the in-range ones).
        Assuming the same weight dispersion outside the window as inside is the
        only approximation on this figure's error bars, and it is a tiny one:
        the term it feeds enters ``var(R)`` multiplied by ``R^2``, so a 3 %
        error on it -- the whole spread of ``deff`` across these cells -- moves
        a bar by 4e-5 of itself.  Six of the seven cells are unit weight to the
        last digit (``deff = 1.00000``) and the assumption is exact for them.
        """
        s2_in = float(self.z['%s_sumw2' % key].sum())
        s_in = float(self.z['%s_sumw' % key].sum())
        n_in = float(self.z['%s_cnt' % key].sum())
        n = self.nevents(key)
        d = float(self.meta['runs'][key]['sumw'])
        n_out, s_out = n - n_in, d - s_in
        if n_out <= 0 or s_out <= 0:
            return s2_in
        deff = n_in * s2_in / (s_in * s_in) if s_in > 0 else 1.0
        return s2_in + deff * s_out * s_out / n_out

    def own_shape_err(self, key):
        """The curve's OWN statistical error on :meth:`shape`, by the delta method.

        Not ``sqrt(sum w^2)/sigma``.  The plotted quantity is self-normalised:
        the bin content is a *subset* of the ``sigma`` it is divided by, so the
        two are correlated and the part of a bin's fluctuation that is shared
        with the normalisation does not move the normalised fraction at all.
        The linearised (delta-method / jackknife) error for a ratio of two sums
        over a common sample keeps that::

            R = N/D,  N = sum_i n_i,  D = sum_i d_i
            dR = (dN - R dD)/D
            var(R) = sum_i (n_i - R d_i)^2 / D^2

        Here ``d_i = w_i`` over every event of the file and ``n_i = w_i`` if
        the event is in the bin and 0 otherwise, so the sum collapses onto
        quantities the harvest already holds::

            var(R) = [ (1 - 2R) sum(w^2)_bin + R^2 sum(w^2)_total ] / D^2

        with no per-event loop.  For a unit-weight sample this is exactly the
        binomial ``R^2 (1 - n/N)/n`` that :meth:`paired_ratio` uses term by
        term, which is the consistency check that the two error treatments on
        this figure are the same statistics seen from two sides; the weighted
        form above is kept because ``ms_joint`` is NOT unit weight (the
        overweight safety net gives it ``deff = 1.028``) and its bar has to
        know that.

        It returns exactly zero when ``n_i ∝ d_i``, which is the property that
        makes it right and ``sqrt(sum w^2)`` wrong.  On these bins the naive
        form is 0.2-0.6 % too large; small, but it is the difference between an
        error bar that is defined and one that merely looks about right.

        This is a WITHIN-sample correlation.  It is unrelated to, and no
        substitute for, the BETWEEN-sample pairing of :meth:`paired_ratio`.
        """
        sumw = self._rebin(self.z['%s_sumw' % key])
        sumw2 = self._rebin(self.z['%s_sumw2' % key])
        d = float(self.meta['runs'][key]['sumw'])
        s2_tot = self._sumw2_total(key)
        r = sumw / d
        var = ((1.0 - 2.0 * r) * sumw2 + r * r * s2_tot) / (d * d)
        # ``shape`` is a density: the same 1/width the value carries.
        return np.sqrt(np.maximum(var, 0.0)) / self.widths

    def own_ratio_err(self, row, key):
        """``(shape ratio to joint, this curve's OWN error on it)``.

        The value is :meth:`paired_ratio`'s -- one definition of the ratio on
        the figure -- and the error is this curve's own, relative error carried
        through.  ``joint`` divided by itself is ``(1, 0)`` as a *ratio*; its
        own statistics are the band, drawn from :meth:`own_shape_err` directly,
        because they belong to the denominator and not to any curve's bar.
        """
        r, _ = self.paired_ratio(row, key)
        y, _, _ = self.shape(key)
        e = self.own_shape_err(key)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(y > 0, e / y, np.nan)
        return r, np.abs(r) * rel

    # --- the top-virtuality histogram, which the parent has no notion of ---
    def cells(self, row):
        """The cells of ``row`` that are actually on disk, in scheme order."""
        return [c for c in ROWS[row] if c in self.meta['runs']]

    def mtop_moments(self, key):
        r = self.meta['runs'][key]
        return r['mtop_mean'], r['mtop_mean_err'], r['mtop_rms']

    def mtop_binned_means(self, key):
        """``(unweighted mean, weight-weighted mean, n)`` off the m_top histogram.

        The harvest's ``mtop_mean`` counts every written event once, ignoring
        its weight.  That is the right estimator for a sample where every
        weight is 1, and every cell here is *meant* to be such a sample -- but
        the overweight safety net writes a handful of events with a weight
        above 1 rather than rejecting them, and in ``ms_joint`` those carry
        +0.3 % of the total cross section.  For that cell the unweighted mean
        and the physical (weight-weighted) mean are not the same number, and a
        difference between them is a RATE artefact of the accept/reject
        machinery, not a ``Z_k`` effect.

        Both are computed here on the SAME binned support -- ``mtop_sumw``
        against ``mtop_cnt``, bin centres, identical truncation at the
        histogram edges -- so the difference between them isolates the weights
        and nothing else.  With 2400 bins over ~48 GeV the binning error on
        either mean is ~1e-5 GeV, an order below the statistical error.
        """
        e = self.z['mtop_bins']
        c = 0.5 * (e[:-1] + e[1:])
        cnt = self.z['%s_mtop_cnt' % key].astype(float)
        sw = self.z['%s_mtop_sumw' % key]
        n = cnt.sum()
        unw = float((cnt * c).sum() / n) if n else float('nan')
        w = float((sw * c).sum() / sw.sum()) if sw.sum() else float('nan')
        return unw, w, int(n)

    def mtop_deff(self, key):
        """Kish's design effect, ``N * sum(w^2) / sum(w)^2``, from the sample.

        A sample whose weights are dispersed is worth fewer events than it
        holds, and this measures by how much.  It is computed off the ``m_tt``
        histogram, which is the only place the harvest kept ``sum(w^2)``; the
        event set is the same one the top virtualities come from (both tops of
        an event share its weight), and restricting to the histogram's range
        divides out of the ratio because numerator and denominator are taken
        over the same events.

        Exactly 1 for a unit-weight sample, so it doubles as an independent
        readout of which cells the overweight safety net touched -- one that
        does not go through the log line at all.
        """
        sw = self.z['%s_sumw' % key].sum()
        sw2 = self.z['%s_sumw2' % key].sum()
        nin = float(self.z['%s_cnt' % key].sum())
        if sw <= 0 or nin <= 0:
            return float('nan')
        return float(nin * sw2 / (sw * sw))

    def mtop_weighted_err(self, key):
        """Error on the WEIGHTED <m_top>: the unweighted one times sqrt(deff)."""
        return self.mtop_moments(key)[1] * math.sqrt(self.mtop_deff(key))

    # --- the ratio pane's denominator, and its pairing ---------------------
    def ref_of(self, row):
        """The cell the ratio pane divides by: the row's ``joint``."""
        return self.pb['rows'][row]['ref']

    def paired_ratio(self, row, key):
        """Per-bin ``shape(key) / shape(joint)``, with the PAIRED error.

        The value is the shape ratio of the two curves in the pane above, so
        the pane is exactly ``coloured curve / blue curve`` and nothing is
        renormalised behind the reader's back.

        The error is where the sibling denominator earns its keep.  Write the
        plotted quantity as a fraction of each sample, ``q = n / N`` (the
        self-normalisation divides by ``sum(w)/N`` over the whole file, so the
        per-event rate is what the pane compares, and ``N`` cancels out of the
        density).  Then, with ``R = q_a / q_b``,

            var(R)/R^2 = var(n_a)/n_a^2 + var(n_b)/n_b^2
                         - 2 cov(n_a, n_b)/(n_a n_b)

        and every term is measured:

          ``var(n) = n (1 - n/N)``      binomial, not Poisson: the bin is a
                                        fraction of a fixed-size sample, and
                                        the ``(1-f)`` is what makes a curve
                                        divided by itself come out with zero
                                        error rather than a spurious one.
          ``cov    = n_both - n_a^pre n_b^pre / N_pair``
                                        the coincidences counted on THIS grid
                                        by ``--stage paired-bins``, over the
                                        events the two cells actually share.

        The third term is the whole point.  Where the two cells put the same
        production event in the same bin, the fluctuation is common and drops
        out; the identity limit ``a == b`` gives exactly 0, and the rare-bin
        limit reduces to McNemar's ``sqrt(n_a + n_b - 2 n_both) / n``, which is
        the estimator ``run_mtt_unweighting`` already quotes per window.

        The ``madspin`` row is only *partly* paired and the formula handles it
        without a special case.  ``ms_seq`` and ``ms_globalretry`` hold 500k
        events, a front truncation of the 1M ``ms_joint`` decays, so the
        coincidences exist over that prefix only while ``n_b`` is the full 1M
        count -- the denominator keeps all its statistics and only the shared
        half of it is credited as correlated.  In the perfectly-concordant
        limit the expression then collapses to the variance of ``ms_joint``'s
        unshared second half, which is the right answer.

        The reference divided by itself is returned as exactly ``(1, 0)``: it
        is the definition of the pane, and its own statistical error is already
        inside every other curve's PAIRED error through ``n_b``.  It is not
        inside the bars the figure draws -- those are each curve's own -- which
        is why the reference's statistics are drawn separately, as the band,
        from :meth:`own_shape_err`.

        This method is no longer what the pane's error bars are made of.  It
        stays because it is the CORRECT error on a scheme-versus-``joint``
        difference and is what ``numbers.txt`` quotes for every significance;
        see :meth:`own_ratio_err` for what is drawn, and the module docstring
        for why the two must not be confused.

        Counts, not weights, throughout.  These are accept/reject samples and
        are meant to be unit weight; six of the seven cells are, to the last
        digit (``deff = 1.00000`` in ``numbers.txt``), and for those the
        count-based relative error is the weighted one.  The exception is
        ``ms_joint``, whose overweight safety net gives it ``deff = 1.02814``,
        so its count-based relative error is 1.4 % low -- on an error, not on a
        value.  Counts are nonetheless what is used, because they are the only
        form in which the pairing is available at all: the coincidences are
        counted events, not summed weights.  ``own_shape_err`` does carry the
        weights, which is why the ``own`` and ``paired`` columns of
        ``numbers.txt`` differ by that much for ``ms_joint`` and by nothing
        anywhere else.
        """
        r = self.pb['rows'][row]
        ref = r['ref']
        ya, _, na = self.shape(key)
        yb, _, nb = self.shape(ref)
        out = np.full(len(na), np.nan)
        err = np.full(len(na), np.nan)
        good = (na > 0) & (nb > 0)
        out[good] = ya[good] / yb[good]
        if key == ref:
            return np.where(good, 1.0, np.nan), np.where(good, 0.0, np.nan)
        npair = float(r['n_pairs'])
        apre = np.array(r['cells'][key]['pre'], dtype=float)
        bpre = np.array(r['cells'][ref]['pre'], dtype=float)
        both = np.array(r['cells'][key]['both'], dtype=float)
        na = na.astype(float)
        nb = nb.astype(float)
        cov = both - apre * bpre / npair
        with np.errstate(divide='ignore', invalid='ignore'):
            var = ((1.0 - na / self.nevents(key)) / na
                   + (1.0 - nb / self.nevents(ref)) / nb
                   - 2.0 * cov / (na * nb))
        err[good] = out[good] * np.sqrt(np.maximum(var[good], 0.0))
        return out, err

    def unpaired_ratio_err(self, row, key):
        """The same error with the third term dropped: what pairing bought.

        Reported in ``numbers.txt`` beside the paired one so the gain is a
        measured number on the page rather than a claim in a docstring.
        """
        ref = self.ref_of(row)
        r, _ = self.paired_ratio(row, key)
        _, _, na = self.shape(key)
        _, _, nb = self.shape(ref)
        na = na.astype(float)
        nb = nb.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            var = ((1.0 - na / self.nevents(key)) / na
                   + (1.0 - nb / self.nevents(ref)) / nb)
            return np.where((na > 0) & (nb > 0), r * np.sqrt(var), np.nan)


# --------------------------------------------------------------------------
def _panels():
    fig = plt.figure(figsize=(7 * 0.75 * 1.35, 7 * 0.75 * 1.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    rx = fig.add_subplot(gs[1], sharex=ax)
    return fig, ax, rx


def draw_clipped_ratio(rx, centres, edges, r, re, color, ls, lw):
    """Draw a ratio series into a pane clipped to ``[RCLIP_LO, RCLIP_HI]``.

    Nothing is dropped.  A point that lands outside the pane is drawn *on* the
    boundary it left through, as a filled triangle pointing that way, and its
    error bar is not drawn (it would be meaningless once the centre has moved).
    A point inside is drawn normally.  The step curve is clipped to the pane so
    it cannot leave and re-enter through the frame without a marker.

    Returns the number of points that went off each way, so the caption can say
    how many there were.
    """
    inside = np.isfinite(r) & (r >= RCLIP_LO) & (r <= RCLIP_HI)
    above = np.isfinite(r) & (r > RCLIP_HI)
    below = np.isfinite(r) & (r < RCLIP_LO)

    drawn = np.clip(r, RCLIP_LO, RCLIP_HI)
    drawn = np.where(np.isfinite(r), drawn, np.nan)
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
    two_mt = d.two_mt
    lo, hi = d.edges[0], d.edges[-1]

    for a in (ax, rx):
        a.axvspan(lo, two_mt, facecolor='0.90', edgecolor='none', zorder=0)
        a.axvline(two_mt, color='0.35', lw=1.0, ls=(0, (6, 3)), zorder=1)

    # Shape comparison: every curve divided by its OWN total cross section, so
    # a rate difference between the truth and MadSpin cancels and the pane
    # below is a pure shape ratio.  It also makes the 1M/500k split invisible
    # here -- ``shape`` is per-event throughout -- leaving it in the error bars
    # only, which is the honest place for it.
    den, dene, _dcnt = d.shape(REF)
    ax.step(d.edges, np.concatenate([den[:1], den]), where='pre',
            color='black', lw=LW, zorder=5,
            label=_tx(r'truth: $pp \to t\bar t j$, $t \to W^+ b$ (off shell)',
                      'truth: pp -> tt~j, t -> W+ b (off shell)'))
    ax.errorbar(d.centres, den, yerr=dene, fmt='none', ecolor='black',
                elinewidth=0.9, capsize=0, zorder=5)

    for key in keys:
        scheme = CELL_SCHEME[key]
        y, ye, cnt = d.shape(key)
        draw = np.where(cnt > 0, y, np.nan)
        lab = (r'\texttt{unweighting = }' + SCHEME_LABEL[scheme]) if USETEX \
            else 'unweighting = ' + SCHEME_PLAIN[scheme]
        ax.step(d.edges, np.concatenate([draw[:1], draw]), where='pre',
                color=COLOR[scheme], ls=LS[scheme], lw=LW, label=lab, zorder=4)
        ax.errorbar(d.centres, draw, yerr=np.where(cnt > 0, ye, np.nan),
                    fmt='none', ecolor=COLOR[scheme], elinewidth=0.9,
                    capsize=0, zorder=4)

    ax.set_yscale('log')
    ax.set_ylabel(_tx(r'$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}m_{t\bar t}$'
                      r' [1/GeV]',
                      r'$(1/\sigma)\,d\sigma/dm_{t\bar t}$ [1/GeV]'))
    ax.set_xlim(lo, hi)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelbottom=False)
    ax.legend(frameon=False, loc='lower right', fontsize=10,
              handlelength=2.8, borderaxespad=0.8)

    # Setup above the curves, and nothing else in the pane.  The prose that used
    # to sit here -- that the scheme is the only thing changing between the
    # coloured curves, and which schemes do and do not read the tabulated Z_k --
    # is not on the figure any more: it is in numbers.txt and RESULTS.md, where
    # it carries its errors and its sensitivity.  ``m_tt`` is defined in
    # RESULTS.md and in this module's docstring, so neither the header nor the
    # axis caption repeats it.  What stays is what cannot be read off the
    # curves: the process line, which row this is, and the sample sizes.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 8.0)
    # Two short lines rather than one long one: at this figure width a single
    # line carrying the process, the scales, the BW cut AND the row title runs
    # off the right edge of the pane.
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

    # The sample size is NOT on the figure.  It used to be, as a third header
    # line: the cells are not all the same size (five hold 1M, ``ms_seq`` and
    # ``ms_globalretry`` hold 500k) and a figure that let the reader assume
    # otherwise would be misleading about the error bars.  It is off now
    # because nothing on the figure is read off it -- every curve is a
    # per-event density and every error bar already carries its own cell's
    # size -- and ``numbers.txt`` states the count of every cell, and the
    # windowed sensitivity each count buys, in a table that a header line
    # could never fit.

    ax.annotate(_tx(r'$2m_t$', r'$2m_t$'),
                xy=(two_mt, 0.03), xycoords=('data', 'axes fraction'),
                xytext=(3, 0), textcoords='offset points',
                ha='left', va='bottom', fontsize=11, color='0.35')

    # --- ratio pane: each scheme over the row's ``joint`` ----------------
    #
    # The denominator is a SIBLING, not the truth.  What that buys, and what it
    # costs, is in :meth:`UData.paired_ratio`; what it *shows* is the null
    # hypothesis directly.  The truth is deliberately absent here: it is an
    # independent sample with a physical shape difference of its own, and
    # carrying it through this pane would make the reader subtract that by eye
    # from the sub-per-cent effect the figure is about.
    ref = d.ref_of(row)

    # The band is no longer a pair of fixed reference rules.  It is ``joint``'s
    # OWN statistical error, bin by bin, from :meth:`UData.own_shape_err` --
    # the delta-method one, because both sides of this pane are self-normalised
    # and the bin is a subset of the sigma it is divided by.  ``joint`` is the
    # denominator, so its statistics are a property of the LINE and not of any
    # coloured curve, and this is the only place they can honestly be drawn.
    #
    # It is NOT an agreement band to add in quadrature with a bar: the cells
    # share their production events, so a curve and ``joint`` fluctuate
    # together and their difference is smaller than the two errors combined.
    # The paired error in numbers.txt is that difference's error.
    yref, _, cref = d.shape(ref)
    eref = d.own_shape_err(ref)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_ref = np.where((cref > 0) & (yref > 0), eref / yref, np.nan)
    rx.fill_between(d.edges, np.concatenate([(1 - rel_ref)[:1], 1 - rel_ref]),
                    np.concatenate([(1 + rel_ref)[:1], 1 + rel_ref]),
                    step='pre', facecolor=COLOR[CELL_SCHEME[ref]], alpha=0.20,
                    edgecolor='none', zorder=2)
    rx.text(0.993, 0.93,
            _tx(r'band: stat.\ error of \texttt{joint}',
                'band: stat. error of joint'),
            transform=rx.transAxes, ha='right', va='top',
            fontsize=8.5, color=COLOR[CELL_SCHEME[ref]])

    n_off = 0
    # The reference LAST, so its flat line sits on top of the others rather
    # than under them, and in its own colour and dash from the pane above --
    # the unity line here IS ``joint``, and drawing it in plain black would
    # invite it to be read as the truth, which is the one curve not in this
    # pane.
    for key in [k for k in keys if k != ref] + [ref]:
        scheme = CELL_SCHEME[key]
        # Both sides self-normalised, so this is a SHAPE ratio.  Any statement
        # about the rate -- including the overweight excess that moves
        # ms_joint's sigma by +0.3 % -- is divided out here on purpose and
        # lives in numbers.txt instead.
        #
        # The bar is this curve's OWN statistical error, not the paired
        # scheme-versus-joint one: the band already carries the denominator's
        # statistics, and putting them on the bars as well would draw them
        # twice.  What the pane no longer shows directly is the error on the
        # DIFFERENCE, which is smaller than band and bar in quadrature and is
        # tabulated per bin and per window in numbers.txt.
        r, re = d.own_ratio_err(row, key)
        # ``joint``'s own error is the BAND; drawing it again as bars on the
        # flat line would show the same statistics twice.
        if key == ref:
            re = np.full_like(r, np.nan)
        # An empty bin here is a statement about the sample size, never a
        # structural zero: every cell on this figure draws a virtuality and
        # reshuffles, so all of them can reach below 2 m_t.  Drawn as a gap,
        # which is why no bin on this figure ever gets the open-circle marker
        # that the companion uses for ``onshell``'s unreachable region.
        up, dn = draw_clipped_ratio(rx, d.centres, d.edges, r, re,
                                    COLOR[scheme], LS[scheme], LW)
        n_off += up + dn

    rx.set_ylim(RCLIP_LO, RCLIP_HI)
    rx.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    rx.set_ylabel(_tx(r'shape ratio to \texttt{joint}',
                      'shape ratio to joint'), fontsize=11)
    # The variable and its unit, nothing else: the definition of ``m_tt`` moved
    # to RESULTS.md with the rest of the prose.
    rx.set_xlabel(_tx(r'$m_{t\bar t}$ [GeV]', r'$m_{t\bar t}$ [GeV]'))
    rx.xaxis.set_minor_locator(AutoMinorLocator())
    rx.yaxis.set_minor_locator(AutoMinorLocator())
    rx.set_xlim(lo, hi)

    # No footnote.  The clipping used to be stated under the figure; it is a
    # property of the pane, the arrows say where every off-scale point went,
    # and ``numbers.txt`` carries each one's unclipped value with its error.
    # The bottom margin is the x-label's now, not a caption's.
    fig.subplots_adjust(left=0.135, right=0.975, top=0.985, bottom=0.075)
    base = os.path.join(out, 'mtt_unweighting_%s%s' % (row, tag))
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=300)
    plt.close(fig)
    return base, n_off


# --------------------------------------------------------------------------
# The numbers.
# --------------------------------------------------------------------------
def _window_integral(d, key, name):
    for n, lo, hi in WINDOWS:
        if n == name:
            return d.integral(key, lo, hi)
    raise KeyError(name)


def write_numbers(d, out, fh=sys.stdout):
    p = lambda *a: print(*a, file=fh)
    two_mt = d.two_mt
    p('=' * 78)
    p('m_tt near threshold: the UNWEIGHTING SCHEMES within each off-shell')
    p('spinmode.  Companion of numbers.txt, which varies the spinmode.')
    p('=' * 78)
    p('code            : %s (%s)' % (d.meta['code_sha'][:12],
                                     d.meta.get('code_branch')))
    p('production      : %s' % d.meta['production_process'])
    p('truth           : %s' % d.meta['truth_process'])
    # NOT one number.  Two cells were run at half size because they were the
    # two slowest, and quoting a single "events per cell" would make every
    # error bar below look better than it is.
    p('events per cell : NOT uniform -- see the table below.  All cells decay '
      'the SAME')
    p('                  production events in the same order; the 500k cells '
      'decay the')
    p('                  first 500k of them (a front truncation of one file), '
      'which is')
    p('                  why they still pair event by event with the 1M cells.')
    for key in [c[0] for c in CELLS]:
        if key in d.meta['runs']:
            p('                  %-16s %9s events'
              % (key, _fmt_plain(int(d.meta['runs'][key]['nevents']))))
    p('2 m_t           : %.4f GeV  (banner MT = %s)'
      % (two_mt, d.meta.get('param_card_masses', {}).get('MT')))
    p('onshell         : not run -- %s' % d.meta['skipped']['onshell'])
    p('')

    # --- what the scheme resolved to, and the overweights ------------------
    p('-- the cells, the scheme each one actually ran, and its overweights --')
    p('   "asked" is the card; "ran" is parsed back out of the log, so a')
    p('   silent fallback cannot masquerade as a measurement.')
    p('%-16s %-9s %-24s %-24s %s'
      % ('cell', 'spinmode', 'asked', 'ran', 'as asked'))
    for key in [c[0] for c in CELLS]:
        if key not in d.meta['runs']:
            continue
        r = d.meta['runs'][key]
        p('%-16s %-9s %-24s %-24s %s'
          % (key, CELL_SPINMODE[key], CELL_SCHEME[key],
             r.get('unweighting'), r.get('scheme_as_asked')))
    p('')
    p('   overweight safety net, per cell.  A joint-vs-global_retry difference')
    p('   caused by these is a RATE effect of the accept/reject machinery and')
    p('   must not be read as a Z_k effect.')
    p('%-16s %10s %10s %14s %14s %10s'
      % ('cell', 'carrying', 'of', 'largest factor', 'excess sum(w)',
         'shift'))
    for key in [c[0] for c in CELLS]:
        if key not in d.meta['runs']:
            continue
        ow = d.meta['runs'][key].get('overweights', {})
        if not ow.get('found'):
            p('%-16s %10s' % (key, 'log line not found'))
            continue
        p('%-16s %10d %10d %14.4f %14.6g %9.4f%%'
          % (key, ow['n'], ow['n_written'], ow['largest'],
             ow.get('excess_w', float('nan')),
             ow.get('percent', float('nan'))))
    p('')
    p('   sigma of each cell from its own event weights, and from its banner.')
    p('   MadSpin normalises to sigma_production * BR, so the banner value is')
    p('   identical across schemes by construction; sum(w)/N is not, and the')
    p('   difference is exactly the overweight carry above.')
    p('%-16s %16s %16s %12s' % ('cell', 'sum(w)/N [pb]', 'banner [pb]',
                                'ratio'))
    for key in [c[0] for c in CELLS]:
        if key not in d.meta['runs']:
            continue
        s = d.sigma(key)
        b = d.banner_sigma(key)
        p('%-16s %16.5f %16.5f %12.6f' % (key, s, b, s / b if b else
                                          float('nan')))
    p('')

    # --- the table itself --------------------------------------------------
    p('-- the tabulated Z_k, as each run logged it --------------------------')
    p('   Z_k(m) is the fraction of the decay pool that can reach virtuality m,')
    p('   fitted during the max-weight scan.  A cell with NO table did not')
    p('   build one, which is the check that it cannot be biased by one.  The')
    p('   span of Z says how much of the lineshape the table is responsible')
    p('   for; the deviation is the code\'s own bin-to-fit residual, a floor on')
    p('   Z_hat/Z rather than the whole of it.')
    p('%-16s %-8s %22s %10s %9s %12s'
      % ('cell', 'slot', 'Z(lo) .. Z(pole) .. Z(hi)', 'span', 'samples',
         'bin/fit dev'))
    for key in [c[0] for c in CELLS]:
        if key not in d.meta['runs']:
            continue
        zt = d.meta['runs'][key].get('z_tables') or []
        if not zt:
            p('%-16s %-8s %22s' % (key, '-', 'no table built'))
            continue
        for t in zt:
            p('%-16s %-8s %22s %10.3f %9d %11.1f%%'
              % (key, t['slot'],
                 '%.3f .. 1 .. %.3f' % (t['Z_lo'], t['Z_hi']),
                 t['Z_hi'] / t['Z_lo'] if t['Z_lo'] else float('nan'),
                 t['samples'], t['bin_fit_deviation_percent']))
    p('')

    # --- the sensitivity, up front -----------------------------------------
    p('-- what this measurement can and cannot resolve ----------------------')
    p('   Quoted BEFORE any comparison is read, because an underpowered null')
    p('   reported as agreement is the failure mode of this whole study.  The')
    p('   entry is the relative difference detectable at 1 sigma.')
    p('')
    p('   The two columns answer different questions and the SIZES DIFFER:')
    p('     unpaired  two independent samples of the sizes this pair actually')
    p('               holds, sqrt((1-f)/(f N_a) + (1-f)/(f N_b)).  A 1M cell')
    p('               against a 500k one is worth barely more than 500k v 500k,')
    p('               so this is NOT the same for every pair.')
    p('     paired    what the shared production sample actually bought,')
    p('               sqrt(discordant)/n_a, MEASURED rather than assumed.')
    p('   The paired column is the WORST (largest) over the measured pairs, so')
    p('   it is a bound on the sensitivity and not the luckiest one.')
    p('%-16s %11s %11s %14s %14s'
      % ('window', 'N_a', 'N_b', '1 sig unpaired', '1 sig paired'))
    # Report against the pair the null hypothesis is really about, per row, so
    # the numbers quoted are the ones the claim rests on.
    for pk in ('PA_joint vs PA_globalretry', 'ms_joint vs ms_globalretry'):
        pv = d.meta.get('paired', {}).get(pk)
        if not pv:
            continue
        p('   [%s]' % pk)
        for name, _lo, _hi in WINDOWS:
            w = pv['windows'].get(name)
            if not w or not w['n_a']:
                continue
            unp = w.get('sigma_rel_unpaired', float('nan'))
            pair = w.get('sigma_rel_paired', float('nan'))
            p('   %-16s %11s %11s %13.3f %% %13.3f %%'
              % (name, _fmt_plain(pv['n_a_total']),
                 _fmt_plain(pv['n_b_total']),
                 100 * unp, 100 * pair))
    p('')
    p('   The sub-threshold window holds ~0.165 % of sigma.  Even at 1M per')
    p('   cell that is a ~3.5 % resolution unpaired, while the Z_k residual')
    p('   expected from the shipped table is sub-per-cent -- so the')
    p('   sub-threshold ratio CANNOT support a claim of agreement, in either')
    p('   direction.  Any agreement read off it alone is a statement about the')
    p('   sample size.  The wider windows and, above all, the top virtuality')
    p('   below are where the test has teeth.')
    p('')

    # --- the direct Z_k observable -----------------------------------------
    p('-- the top virtuality, m(W b), which is what Z_k distorts directly ---')
    p('   m_tt only sees Z_k after the production reshuffle has smeared it.')
    p('   Both tops of every event enter, so n = 2 N.  The error on the mean')
    p('   is rms/sqrt(n) and is four orders of magnitude below the')
    p('   sub-threshold rate error, which is why this is the sensitive test.')
    p('%-16s %14s %12s %12s %14s'
      % ('cell', '<m_top> [GeV]', '+- [GeV]', 'rms [GeV]', 'vs row joint'))
    for row in ('PA', 'madspin'):
        keys = d.cells(row)
        if not keys:
            continue
        base = keys[0]
        m0, e0, _ = d.mtop_moments(base)
        for key in keys:
            m, e, s = d.mtop_moments(key)
            dd = m - m0
            de = math.sqrt(e ** 2 + e0 ** 2)
            p('%-16s %14.6f %12.6f %12.6f %14s'
              % (key, m, e, s,
                 '-' if key == base else '%+.6f +- %.6f (%.1f s)'
                 % (dd, de, abs(dd) / de if de else float('nan'))))
    p('')
    p('   CAVEAT on the "vs row joint" column: it is UNPAIRED, and in the')
    p('   madspin row it compares a 1M cell against 500k cells, so its error')
    p('   is the quadrature sum of two unequal errors.  The paired numbers in')
    p('   the next section are the ones to read -- they compare the same')
    p('   production events and are strictly better.')
    p('')

    # --- is the joint-vs-retry difference a Z_k effect, or the overweights? -
    p('-- unweighted vs weight-weighted <m_top>: the overweight cross-check --')
    p('   The mean above counts every written event ONCE, whatever weight it')
    p('   carries.  That is correct for a unit-weight sample, and the')
    p('   accept/reject schemes are meant to produce one -- but the overweight')
    p('   safety net writes some events with weight > 1 instead of rejecting')
    p('   them.  Where it does, the unweighted mean is not the physical mean,')
    p('   and a joint-vs-global_retry difference driven by THAT is a rate')
    p('   artefact of the accept/reject machinery, not a Z_k effect.')
    p('')
    p('   Both columns are computed on the same binned support, so their')
    p('   difference isolates the weights and nothing else.  A cell with no')
    p('   overweights must show 0 to the last digit; a non-zero entry is the')
    p('   size of the contamination in this observable.')
    p('%-16s %16s %16s %14s %12s'
      % ('cell', '<m_top> unwtd', '<m_top> wtd', 'wtd - unwtd', 'sigma(mean)'))
    for key in [c[0] for c in CELLS]:
        if key not in d.meta['runs']:
            continue
        unw, wtd, _n = d.mtop_binned_means(key)
        _m, e, _s = d.mtop_moments(key)
        p('%-16s %16.6f %16.6f %+14.6f %12.6f'
          % (key, unw, wtd, wtd - unw, e))
    p('')
    p('   The weighted mean also costs precision, because a sample with')
    p('   dispersed weights is worth fewer events than it holds.  Kish\'s')
    p('   design effect deff = N_inrange * sum(w^2) / sum(w)^2 measures that')
    p('   from the sample itself; the error on the weighted mean is the')
    p('   unweighted error times sqrt(deff).  deff = 1 exactly for a')
    p('   unit-weight sample, so this column is a second, independent readout')
    p('   of which cells the safety net actually touched.')
    p('%-16s %12s %12s %14s %14s'
      % ('cell', 'deff', 'sqrt(deff)', 'err unwtd', 'err wtd'))
    for key in [c[0] for c in CELLS]:
        if key not in d.meta['runs']:
            continue
        p('%-16s %12.5f %12.5f %14.6f %14.6f'
          % (key, d.mtop_deff(key), math.sqrt(d.mtop_deff(key)),
             d.mtop_moments(key)[1], d.mtop_weighted_err(key)))
    p('')
    p('   And the comparison REDONE on the weighted means, which are the')
    p('   physical ones.  This is the number to read for any cell the safety')
    p('   net touched; for the others it is identical to the unweighted one.')
    p('   NOTE it is UNPAIRED -- the stored per-event pairing carries no')
    p('   weights, so a paired weighted difference is not available from this')
    p('   harvest.  It therefore has a larger error than the paired numbers')
    p('   below, and where the two disagree it is this one that is unbiased.')
    for row in ('PA', 'madspin'):
        keys = d.cells(row)
        for a, b, what in NULL_PAIRS:
            if a not in keys or b not in keys:
                continue
            da = d.mtop_binned_means(a)[1] - d.mtop_binned_means(b)[1]
            er = math.sqrt(d.mtop_weighted_err(a) ** 2
                           + d.mtop_weighted_err(b) ** 2)
            p('   %-16s - %-16s = %+.6f +- %.6f GeV  (%.2f sigma)'
              % (a, b, da, er, abs(da) / er if er else float('nan')))
    p('')

    # --- the null hypothesis, window by window -----------------------------
    p('-- the pairs, window by window ---------------------------------------')
    p('   n_a, n_b are raw event counts; "both" is the number of production')
    p('   events that landed in the window under BOTH schemes.  Every cell')
    p('   decays the same production events in the same order, so the error on')
    p('   the difference is set by the DISCORDANT pairs (McNemar),')
    p('   sqrt(n_a + n_b - 2 both), not by sqrt(n_a + n_b).')
    for pk, pv in sorted(d.meta.get('paired', {}).items()):
        p('')
        p('   %s' % pk)
        p('     %s' % pv['what'])
        p('     pairing check: %d pairs, max |Delta sqrt(shat)| = %.3g GeV'
          % (pv['n_pairs'], pv['max_dshat']))
        if 'mtop_dmean' in pv:
            err = pv['mtop_dmean_err']
            p('     PAIRED top virtuality: <m_top(a) - m_top(b)> = '
              '%+.6f +- %.6f GeV  (%.1f sigma, %d paired tops, '
              'per-pair rms %.4f GeV)'
              % (pv['mtop_dmean'], err,
                 abs(pv['mtop_dmean']) / err if err else float('nan'),
                 pv['mtop_n'], pv['mtop_drms']))
        p('     %-14s %9s %9s %9s %12s %10s %10s'
          % ('window', 'n_a', 'n_b', 'both', 'ratio a/b', 'sigma', 'pull'))
        for name, _lo, _hi in WINDOWS:
            w = pv['windows'].get(name)
            if not w or not w['n_b']:
                continue
            na, nb, nab = w['n_a'], w['n_b'], w['n_both']
            disc = na + nb - 2 * nab
            r = na / nb
            # sigma on the ratio from the discordant pairs.
            sr = math.sqrt(disc) / nb if nb else float('nan')
            p('     %-14s %9d %9d %9d %12.5f %10.5f %10.2f'
              % (name, na, nb, nab, r, sr,
                 abs(r - 1) / sr if sr else float('nan')))
    p('')

    # --- the ratio pane's own numbers, per bin, unclipped ------------------
    for row in ('PA', 'madspin'):
        keys = d.cells(row)
        if not keys:
            continue
        ref = d.ref_of(row)
        pbr = d.pb['rows'][row]
        p('=' * 78)
        p('-- per-bin SHAPE ratio to %s, %s -- THIS IS THE FIGURE\'S RATIO '
          'PANE --' % (ref, row))
        p('   UNCLIPPED.  The pane clips to +-20 %; every value outside that')
        p('   is drawn on the boundary as a triangle and its real value is')
        p('   here.')
        p('')
        p('   Denominator: %s, the row\'s joint cell -- a SIBLING of every'
          % ref)
        p('   other column, not the truth.  Both sides are divided by their')
        p('   own total sigma first, so this is a shape ratio and the')
        p('   overweight rate carry is divided out of it.')
        p('')
        p('   THREE ERRORS PER BIN, AND THEY ANSWER DIFFERENT QUESTIONS.')
        p('')
        p('   "own"      -- what the FIGURE now draws.  Each curve\'s own')
        p('                 statistical error, and the band around 1 is')
        p('                 %s\'s own (the "band" column).  Delta method:' % ref)
        p('                     var(R) = sum_i (n_i - R d_i)^2 / D^2,')
        p('                 the linearised error of a self-normalised')
        p('                 histogram, in which the bin is a SUBSET of the')
        p('                 sigma it is divided by.  It returns zero when the')
        p('                 bin is proportional to the normalisation, which a')
        p('                 plain sqrt(sum w^2) does not; the naive form is')
        p('                 %.2f %% too large on these bins.'
          % (100.0 * _naive_over_delta(d, ref) - 100.0))
        p('                 DO NOT add the band and a bar in quadrature.  That')
        p('                 discards the correlation below and OVERESTIMATES')
        p('                 the error on the difference.')
        p('')
        p('   "paired"   -- the CORRECT error on the difference between a')
        p('                 scheme and %s, and the one to quote for any' % ref)
        p('                 significance.  The cells decay the SAME production')
        p('                 events in the same order (max |Delta sqrt(shat)| =')
        p('                 %.3g GeV over %s pairs), so the production'
          % (pbr['max_dshat'], _fmt_plain(pbr['n_pairs'])))
        p('                 fluctuation is common to numerator and denominator')
        p('                 and cancels.  The covariance is MEASURED on these')
        p('                 very bins by "run_mtt_unweighting.py --stage')
        p('                 paired-bins", which counts how many production')
        p('                 events land in the SAME bin under both cells; a')
        p('                 per-window coincidence count cannot answer this,')
        p('                 because an event can be in one window under both')
        p('                 schemes and in a different bin under each.')
        p('')
        p('   "unpaired" -- the paired error with the covariance term dropped:')
        p('                 what an independent-samples bar would have been.')
        p('                 The ratio of it to "paired" is what the shared')
        p('                 production sample bought, per bin.  It is also')
        p('                 what band-and-bar-in-quadrature amounts to, which')
        p('                 is why that reading is wrong.')
        p('')
        pairing = ('%s: every cell holds %s events and pairs with %s over all '
                   'of them.'
                   % (row, _fmt_plain(pbr['n_pairs']), ref))
        if any(int(d.meta['runs'][k]['nevents']) != pbr['n_pairs']
               for k in keys):
            pairing = (
                '%s: PARTLY paired.  %s is decayed from %s production events; '
                '\n   the other cells hold %s, a FRONT TRUNCATION of the same '
                'file, so the\n   coincidences exist over that prefix only.  '
                'The denominator keeps all\n   %s of its events -- only its '
                'shared half is credited as correlated,\n   and its unshared '
                'half enters as an independent error, which is why the\n   '
                'pairing gain is smaller in this row than in PA.'
                % (row, ref, _fmt_plain(int(d.meta['runs'][ref]['nevents'])),
                   _fmt_plain(pbr['n_pairs']),
                   _fmt_plain(int(d.meta['runs'][ref]['nevents']))))
        p('   %s' % pairing)
        p('')
        p('   %s' % ',  '.join('%s = %s' % (SHORT[CELL_SCHEME[k]],
                                            CELL_SCHEME[k]) for k in keys))
        others = [k for k in keys if k != ref]
        head = '%9s %9s %9s' % ('bin [GeV]', 'n(joint)', 'band')
        for key in others:
            head += ' %-34s' % SHORT[CELL_SCHEME[key]]
        p(head.rstrip())
        p(('%9s %9s %9s' % ('', '', '+- (own)') + ''.join(
            ' %-34s' % 'ratio +- own [paired] (unpaired)'
            for _ in others)).rstrip())
        den_of_ref, _, nref = d.shape(ref)
        # The band the figure draws around 1: the reference's OWN error.
        band = d.own_shape_err(ref) / np.where(den_of_ref > 0, den_of_ref,
                                               np.nan)
        cols = []
        for key in others:
            r, re = d.paired_ratio(row, key)
            _, ro = d.own_ratio_err(row, key)
            cols.append((r, ro, re, d.unpaired_ratio_err(row, key)))
        for i in range(len(d.centres)):
            line = '%4.0f-%4.0f %9d %9s' % (
                d.edges[i], d.edges[i + 1], nref[i],
                '%.4f' % band[i] if np.isfinite(band[i]) else '-')
            for r, ro, re, ru in cols:
                if not np.isfinite(r[i]):
                    line += ' %-34s' % '-'
                else:
                    flag = ' *' if not RCLIP_LO <= r[i] <= RCLIP_HI else ''
                    line += ' %-34s' % ('%.4f +- %.4f [%.4f] (%.4f)%s'
                                        % (r[i], ro[i], re[i], ru[i], flag))
            p(line.rstrip())
        p('   * = outside the pane\'s +-20 %, drawn there as a boundary '
          'triangle.')
        p('   "band" and "+- own" are what the figure DRAWS; "[paired]" is')
        p('   what to quote.  Band and bar in quadrature reproduces the')
        p('   "(unpaired)" column, which is the wrong, too-large answer.')
        p('')
        # The one-line summary of what the pane is FOR.
        p('   what the pane says, summed over the plotted range %g-%g GeV.'
          % (d.edges[0], d.edges[-1]))
        _f = _quad_over_paired(d, row)
        p('   Built on the PAIRED error, not on the bars the figure draws:')
        p('   this is a statement about a difference, so it needs the error on')
        p('   the difference.  Combining the band and a bar in quadrature')
        p('   instead inflates it by a median factor %.2f, which would shrink'
          % _f)
        p('   every chi2 below by %.0f %% and could hide a real departure.'
          % (100.0 * (1.0 - 1.0 / (_f * _f))))
        p('   %-16s %10s %8s %10s %14s'
          % ('cell', 'chi2', 'ndf', 'chi2/ndf', 'how far off'))
        for key in others:
            r, re = d.paired_ratio(row, key)
            m = np.isfinite(r) & np.isfinite(re) & (re > 0)
            chi2 = float((((r[m] - 1.0) / re[m]) ** 2).sum())
            ndf = int(m.sum())
            # chi2 has mean ndf and variance 2 ndf, so this is how many of its
            # own sigmas the total sits from the null.  Stated that way rather
            # than as a p-value, because the off-diagonal covariance below
            # makes a p-value more precise than the input deserves.
            p('   %-16s %10.1f %8d %10.3f %+13.1f s'
              % (key, chi2, ndf, chi2 / ndf if ndf else float('nan'),
                 (chi2 - ndf) / math.sqrt(2.0 * ndf) if ndf
                 else float('nan')))
        p('   The chi2 treats the bins as independent of ONE ANOTHER.  They')
        p('   are not: a production event that lands in different bins under')
        p('   the two schemes correlates those two bins, and that off-diagonal')
        p('   covariance was not harvested.  The EXPECTATION is still ndf --')
        p('   correlation between bins does not move it, only the diagonal')
        p('   variance could, and that one is measured -- but the SPREAD is')
        p('   not sqrt(2 ndf), so "how far off" is an order of magnitude, not')
        p('   a p-value.  The window table above is the exact statement.')
        p('   The columns are also correlated WITH EACH OTHER: they share the')
        p('   denominator, so a fluctuation of %s moves all of them the same'
          % ref)
        p('   way.  A deviation common to every column is a statement about')
        p('   %s; a deviation in one column only is a statement about that'
          % ref)
        p('   scheme.  The comparison the figure exists for is the latter.')
        p('')

        # --- where the pane's zero point actually sits --------------------
        # The plotted ratio is built from WEIGHTS; the errors from COUNTS.
        # For a unit-weight cell the two are the same number, and where they
        # are not, the difference is the overweight safety net and it belongs
        # on the page rather than in a footnote.
        nref_c = nref
        for key in others:
            ya, _, na_c = d.shape(key)
            rw = ya / den_of_ref
            rc = ((na_c / d.nevents(key))
                  / (nref_c / d.nevents(ref)))
            g = (na_c > 0) & (nref_c > 0)
            off = rw[g] / rc[g] - 1.0
            # The bin that departs furthest from the common offset: that is
            # where an overweight event actually landed.
            worst = int(np.flatnonzero(g)[
                int(np.argmax(np.abs(off - np.median(off))))])
            p('   %-16s weighted/counted - 1: median %+.5f, range %+.5f .. '
              '%+.5f (furthest bin %.0f-%.0f GeV)'
              % (key, float(np.median(off)), float(off.min()),
                 float(off.max()), d.edges[worst], d.edges[worst + 1]))
        p('   A cell with no overweights gives 0 in every bin, and the PA')
        p('   cells do -- they are unit weight to the last digit.')
        p('   The MEDIAN is a normalisation offset of the whole pane: the')
        p('   denominator\'s own sigma was moved by weights the safety net')
        p('   carried, so every curve sits that far off 1 TOGETHER and the')
        p('   difference between two curves -- which is what this figure is')
        p('   read for -- is untouched by it.  ms_joint carries +0.296 % of')
        p('   its sigma in 754 overweight events (largest factor 83), and')
        p('   +0.00295 is exactly that.')
        p('   The RANGE is not: where an overweight event lands inside the')
        p('   plotted window it moves that bin\'s weighted density and not its')
        p('   count, so those bins carry an extra shift of up to 0.3 % on top')
        p('   of the offset.  That is a RATE artefact of the accept/reject')
        p('   machinery, not a Z_k effect, and it is well inside the paired')
        p('   error of the bins it touches -- but it is why the count-based')
        p('   error bar and the weight-based central value are not quite the')
        p('   same measurement in this row, and why section 6\'s caveat about')
        p('   ms_joint applies to the pane as well.')
        p('')

    # --- the reading the pane exists for ----------------------------------
    p('=' * 78)
    p('-- READING THE PANE: sequential against sequential_global_retry ------')
    p('=' * 78)
    p('   The pane divides by joint, and joint is the scheme that builds NO')
    p('   Z_k table at all.  sequential_global_retry builds one but cancels it')
    p('   identically.  So the two of them landing on 1 together is the null')
    p('   hypothesis, and sequential -- the only scheme that trusts the')
    p('   tabulated Z_hat -- leaving them is the residual Z_hat/Z.  Reading')
    p('   that off the pane is a comparison of two CURVES, not of a curve with')
    p('   the line, and that comparison is where the shared denominator\'s own')
    p('   fluctuation drops out.')
    p('')
    p('   THE FIGURE\'S BARS CANNOT GIVE THIS NUMBER.  What the pane draws is')
    p('   each curve\'s OWN statistical error, and the band is joint\'s own.')
    p('   Those are per-curve statistics; the significance of a DIFFERENCE')
    p('   needs the paired error, because the cells share their production')
    p('   events and fluctuate together.  The table below is that error, and')
    p('   it is what to quote.  Anything read by eye off the bars -- band and')
    p('   bar overlapping or not -- is the too-large, uncorrelated answer.')
    p('')
    p('   Integrated, from the paired window counts (exact; the per-bin chi2')
    p('   above cannot be turned into a significance because its bins are')
    p('   correlated).  Every entry is a RATIO and its PAIRED error.')
    p('')
    p('   NOTE for the madspin row: these window counts are FULLY paired, so')
    p('   ms_joint enters them at its shared 500k prefix, while the figure\'s')
    p('   pane uses all 1M of it in the denominator.  The window numbers are')
    p('   therefore the stricter comparison and the pane the more precise')
    p('   one; they are the same measurement to within the error on the half')
    p('   of ms_joint that the two treat differently.')
    p('%-34s %-14s %18s %8s' % ('window', 'pair', 'ratio', 'sigma'))
    for row, seq, ret, jnt in (('PA', 'PA_seq', 'PA_globalretry', 'PA_joint'),
                               ('madspin', 'ms_seq', 'ms_globalretry',
                                'ms_joint')):
        if seq not in d.meta['runs']:
            continue
        p('   [%s]' % row)
        for name, _lo, _hi in WINDOWS:
            for a, b, tag in ((ret, jnt, 'retry/joint'),
                              (seq, jnt, 'seq/joint'),
                              (seq, ret, 'seq/retry')):
                pv = (d.meta['paired'].get('%s vs %s' % (a, b))
                      or d.meta['paired'].get('%s vs %s' % (b, a)))
                if not pv:
                    continue
                w = pv['windows'].get(name)
                if not w or not w['n_a'] or not w['n_b']:
                    continue
                # ``rel_diff`` is stored as (n_a - n_b)/n_a for the pair in
                # the order it was measured; recompute from the raw counts so
                # the orientation printed here is the one the label says.
                flip = d.meta['paired'].get('%s vs %s' % (a, b)) is None
                na, nb = ((w['n_b'], w['n_a']) if flip
                          else (w['n_a'], w['n_b']))
                disc = w['n_a'] + w['n_b'] - 2 * w['n_both']
                r = na / nb
                sr = math.sqrt(disc) / nb
                p('   %-31s %-14s %10.5f +- %.5f %7.2f'
                  % (name, tag, r, sr, abs(r - 1) / sr if sr else
                     float('nan')))
        p('')
    p('   Read the three lines of a window together.  retry/joint consistent')
    p('   with 1 IS the null holding; seq/retry is then the cleanest form of')
    p('   the Z_hat/Z question, because neither of its two samples is the one')
    p('   every other ratio on the figure divides by.')
    p('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_unweighting'))
    ap.add_argument('--check-minus', action='store_true', default=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = UData(args.data)
    for row in ('PA', 'madspin'):
        got = make_figure(d, row, args.out)
        if got is None:
            print('%s: no cells on disk, skipped' % row)
            continue
        base, n_off = got
        # The off-scale count used to be a footnote ON the figure.  It is a
        # property of the drawing, not of the measurement, so it is reported
        # to whoever ran the script instead; the values themselves are in
        # numbers.txt, flagged with a star.
        print('wrote %s.pdf / .png   (usetex=%s, minus fix applied=%s, '
              '%d point%s off the +-20 %% pane)'
              % (base, USETEX, MINUS_FIX, n_off, '' if n_off == 1 else 's'))
        if args.check_minus:
            ok, detail = check_minus(base + '.pdf')
            print('minus-sign check: %s -- %s' % (ok, detail))
    with open(os.path.join(args.out, 'numbers.txt'), 'w') as fh:
        write_numbers(d, args.out, fh)
    write_numbers(d, args.out)


if __name__ == '__main__':
    main()
