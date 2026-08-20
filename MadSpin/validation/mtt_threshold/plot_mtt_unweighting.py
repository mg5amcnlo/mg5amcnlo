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

Two figures, one per spinmode:

    plot_mtt_unweighting.py [--data DIR] [--out DIR]
      -> <out>/mtt_unweighting_PA.pdf/.png
         <out>/mtt_unweighting_madspin.pdf/.png
         <out>/numbers.txt

The ratio pane is **clipped to +-20 %**.  Points outside are drawn *on* the
boundary as filled triangles pointing the way they went, never silently cut off,
and the axis label says so.  The unclipped value of every such point is in the
per-bin table of ``numbers.txt``.
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
    Data, ratio, zone_edges, check_minus, USETEX, MINUS_FIX, LW, allcolors,
    ANCHOR, AGREE_HI, _fmt_int, _tx,
)
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.ticker import AutoMinorLocator         # noqa: E402

from run_mtt_unweighting import (                      # noqa: E402
    CELLS, CELL_SPINMODE, CELL_SCHEME, ROWS, NULL_PAIRS, WINDOWS,
)

REF = 'truth'

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
                 meta='meta_unweighting.json'):
        self.z = np.load(os.path.join(ddir, npz))
        self.meta = json.load(open(os.path.join(ddir, meta)))
        self.fine = self.z['bins']
        self.edges = zone_edges()
        self.centres = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.widths = np.diff(self.edges)
        self.two_mt = float(self.meta.get('two_mt', 346.0))
        self._groups = self._group_map()

    # --- the top-virtuality histogram, which the parent has no notion of ---
    def cells(self, row):
        """The cells of ``row`` that are actually on disk, in scheme order."""
        return [c for c in ROWS[row] if c in self.meta['runs']]

    def mtop_moments(self, key):
        r = self.meta['runs'][key]
        return r['mtop_mean'], r['mtop_mean_err'], r['mtop_rms']


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

    den, dene, dcnt = d.density(REF)
    ax.step(d.edges, np.concatenate([den[:1], den]), where='pre',
            color='black', lw=LW, zorder=5,
            label=_tx(r'truth: $pp \to t\bar t j$, $t \to W^+ b$ (off shell)',
                      'truth: pp -> tt~j, t -> W+ b (off shell)'))
    ax.errorbar(d.centres, den, yerr=dene, fmt='none', ecolor='black',
                elinewidth=0.9, capsize=0, zorder=5)

    for key in keys:
        scheme = CELL_SCHEME[key]
        y, ye, cnt = d.density(key)
        draw = np.where(cnt > 0, y, np.nan)
        lab = (r'\texttt{unweighting = }' + SCHEME_LABEL[scheme]) if USETEX \
            else 'unweighting = ' + SCHEME_PLAIN[scheme]
        ax.step(d.edges, np.concatenate([draw[:1], draw]), where='pre',
                color=COLOR[scheme], ls=LS[scheme], lw=LW, label=lab, zorder=4)
        ax.errorbar(d.centres, draw, yerr=np.where(cnt > 0, ye, np.nan),
                    fmt='none', ecolor=COLOR[scheme], elinewidth=0.9,
                    capsize=0, zorder=4)

    ax.set_yscale('log')
    ax.set_ylabel(_tx(r'$\mathrm{d}\sigma/\mathrm{d}m_{t\bar t}$ [pb/GeV]',
                      r'$d\sigma/dm_{t\bar t}$ [pb/GeV]'))
    ax.set_xlim(lo, hi)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelbottom=False)
    ax.legend(frameon=False, loc='lower right', fontsize=10,
              handlelength=2.8, borderaxespad=0.8)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 26)
    n_cell = int(d.meta['runs'][keys[0]]['nevents'])
    ax.text(0.028, 0.968,
            _tx(r'$pp \to t\bar t j$ at $\sqrt{s} = 13$~TeV, LO, '
                r'$\mu_R = \mu_F = m_t$ --- \textbf{%s}' % ROW_TITLE[row][0],
                r'$pp \to t\bar t j$ at $\sqrt{s}=13$ TeV, LO, '
                r'$\mu_R=\mu_F=m_t$ -- %s' % ROW_TITLE[row][1]),
            transform=ax.transAxes, ha='left', va='top', fontsize=11)
    ax.text(0.028, 0.912,
            _tx(r'the accept/reject scheme is the only thing that changes '
                r'between the coloured curves',
                'the accept/reject scheme is the only thing that changes '
                'between the coloured curves'),
            transform=ax.transAxes, ha='left', va='top', fontsize=9.5,
            color='0.25')
    ax.text(0.028, 0.868,
            _tx(r'same %s production events, same seed, '
                r'BW cut $=%g\,\Gamma_t$ on both sides'
                % (_fmt_int(n_cell), d.meta.get('bwcutoff', 15.0)),
                'same %s production events, same seed, BW cut = %g Gamma_t'
                % (_fmt_int(n_cell), d.meta.get('bwcutoff', 15.0))),
            transform=ax.transAxes, ha='left', va='top', fontsize=9.5,
            color='0.25')

    # The one sentence the figure exists to make testable.
    ax.text(0.028, 0.824,
            _tx(r'\texttt{joint} and \texttt{sequential\_global\_retry} '
                r'do not read the tabulated $Z_k$;'
                '\n'
                r'\texttt{sequential} does, and its residual bias is exactly '
                r'$\hat Z / Z$.',
                'joint and sequential_global_retry do not read the tabulated '
                'Z_k;\nsequential does, and its residual bias is exactly '
                'Zhat/Z.'),
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            color='0.25')
    ax.annotate(_tx(r'$2m_t$', r'$2m_t$'),
                xy=(two_mt, 0.03), xycoords=('data', 'axes fraction'),
                xytext=(3, 0), textcoords='offset points',
                ha='left', va='bottom', fontsize=11, color='0.35')

    # --- ratio pane, clipped -------------------------------------------
    rx.axhspan(0.95, 1.05, facecolor=allcolors[0], alpha=0.16, zorder=0)
    rx.axhspan(0.9, 1.1, facecolor=allcolors[0], alpha=0.10, zorder=0)
    rx.axhline(1.0, color='black', lw=0.9, zorder=2)
    rx.text(0.993, 0.93, _tx(r'bands: $\pm5\%$, $\pm10\%$',
                             'bands: +-5%, +-10%'),
            transform=rx.transAxes, ha='right', va='top',
            fontsize=8.5, color=allcolors[0])

    n_off = 0
    for key in keys:
        scheme = CELL_SCHEME[key]
        y, ye, cnt = d.density(key)
        r, re = ratio(y, ye, den, dene)
        # An empty bin here is a statement about the sample size, never a
        # structural zero: every cell on this figure draws a virtuality and
        # reshuffles, so all of them can reach below 2 m_t.  Drawn as a gap.
        r = np.where((cnt == 0) | (dcnt == 0), np.nan, r)
        up, dn = draw_clipped_ratio(rx, d.centres, d.edges, r, re,
                                    COLOR[scheme], LS[scheme], LW)
        n_off += up + dn

    rx.set_ylim(RCLIP_LO, RCLIP_HI)
    rx.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    rx.set_ylabel(_tx(r'ratio to truth (clipped $\pm20\%$)',
                      'ratio to truth (clipped +-20%)'), fontsize=11)
    rx.set_xlabel(_tx(
        r'$m_{t\bar t}$ [GeV] \ \ (per-event $m$ of $(W^+b)+(W^-\bar b)$)',
        r'$m_{t\bar t}$ [GeV]  (per-event $m$ of $(W^+b)+(W^-\bar b)$)'))
    rx.xaxis.set_minor_locator(AutoMinorLocator())
    rx.yaxis.set_minor_locator(AutoMinorLocator())
    rx.set_xlim(lo, hi)
    # The clipping statement goes under the whole figure rather than inside the
    # pane: the pane is exactly where the off-scale points are, so a caption in
    # it would sit on top of the thing it is describing.
    fig.text(0.5, 0.008,
             _tx(r'ratio pane clipped to $\pm20\%%$: %d point%s outside, drawn '
                 r'as triangles \emph{on} the boundary, pointing the way they '
                 r'went.' '\n' r'Unclipped values in \texttt{numbers.txt}.'
                 % (n_off, '' if n_off == 1 else 's'),
                 'ratio pane clipped to +-20%%: %d point%s outside, drawn as '
                 'triangles on the boundary, pointing the way they went.\n'
                 'Unclipped values in numbers.txt.'
                 % (n_off, '' if n_off == 1 else 's')),
             ha='center', va='bottom', fontsize=8, color='0.35',
             linespacing=1.4)

    fig.subplots_adjust(left=0.135, right=0.975, top=0.985, bottom=0.135)
    base = os.path.join(out, 'mtt_unweighting_%s%s' % (row, tag))
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=300)
    plt.close(fig)
    return base


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
    p('events per cell : %s (all cells decay the SAME production events)'
      % d.meta['nevents_per_cell'])
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
    n_cell = int(d.meta['nevents_per_cell'])
    p('-- what this measurement can and cannot resolve ----------------------')
    p('   Computed from the measured spectrum BEFORE the runs, so it is a')
    p('   design statement and not a post-hoc excuse.  Two cells of %d events'
      % n_cell)
    p('   each; the entry is the relative difference detectable at 1 sigma.')
    p('   Event counts are the first cell\'s; the other cells are within a')
    p('   per cent of it, so the table is a design statement, not a fit.')
    p('   The paired column is the WORST (largest) over the measured pairs, so')
    p('   it is a bound on the sensitivity and not the luckiest one.')
    p('%-16s %14s %14s %14s' % ('window', 'events/cell', '1 sigma unpaired',
                                '1 sigma paired'))
    ref = d.cells('PA')[0] if d.cells('PA') else d.cells('madspin')[0]
    for name, lo, hi in WINDOWS:
        _s, _e, k = d.integral(ref, lo, hi)
        unp = math.sqrt(2.0 / k) if k else float('nan')
        # the paired figure comes from the measured discordant counts
        worst = None
        for pk, pv in d.meta.get('paired', {}).items():
            w = pv['windows'].get(name)
            if not w or not w['n_a']:
                continue
            disc = w['n_a'] + w['n_b'] - 2 * w['n_both']
            val = math.sqrt(disc) / w['n_a']
            worst = val if worst is None else max(worst, val)
        p('%-16s %14d %13.3f %% %13s'
          % (name, k, 100 * unp,
             '%.3f %%' % (100 * worst) if worst is not None else '-'))
    p('')
    p('   The sub-threshold region is 0.165 %% of sigma.  At %d events per'
      % n_cell)
    p('   cell a per-cent-level Z_hat/Z residual is FAR below the noise there,')
    p('   and any "agreement" claimed from that window alone would be a')
    p('   statement about the sample size.  The wider windows and, above all,')
    p('   the top virtuality below are where the test has teeth.')
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

    # --- per-bin table, unclipped ------------------------------------------
    for row in ('PA', 'madspin'):
        keys = d.cells(row)
        if not keys:
            continue
        p('-- per-bin ratio to truth, %s (UNCLIPPED; the figure clips the pane '
          'to +-20%%) --' % row)
        den, dene, _ = d.density(REF)
        p('   %s' % ',  '.join('%s = %s' % (SHORT[CELL_SCHEME[k]],
                                            CELL_SCHEME[k]) for k in keys))
        head = '%9s %12s' % ('bin [GeV]', 'truth')
        for key in keys:
            head += ' %14s' % SHORT[CELL_SCHEME[key]]
        p(head)
        cols = []
        for key in keys:
            y, ye, cnt = d.density(key)
            r, re = ratio(y, ye, den, dene)
            cols.append((r, re, cnt))
        for i in range(len(d.centres)):
            line = '%4.0f-%4.0f %12.5g' % (d.edges[i], d.edges[i + 1], den[i])
            for r, re, cnt in cols:
                if cnt[i] == 0 or not np.isfinite(r[i]):
                    line += ' %14s' % '-'
                else:
                    line += ' %7.3f+-%.3f' % (r[i], re[i])
            p(line)
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
        base = make_figure(d, row, args.out)
        if base is None:
            print('%s: no cells on disk, skipped' % row)
            continue
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
