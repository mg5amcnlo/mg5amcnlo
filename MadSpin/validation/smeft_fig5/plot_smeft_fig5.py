#!/usr/bin/env python3
r"""Fig. 5 of the MadSpin2 paper, remade: ``Delta phi(e-, e+)`` with and without
spin correlations, for the SMEFT ``O_tG`` amplitude interference and for the SM.

Runs entirely off the committed raw histograms -- no MadSpin, no LHE, no event
generation::

    plot_smeft_fig5.py [--data DIR] [--out DIR] [--nbins N] [--only A,B]

with ``<data>`` holding ``histograms.npz`` and ``meta.json`` as written by
``run_smeft_fig5.py``.

Four variations are drawn, all in one style, so the choice of which goes in the
paper can be made by looking:

  A   EFT onshell + EFT none + SM LO onshell
  B   A, plus SM LO none
  C   like B, with the SM at NLO instead of LO
  D   LO and NLO SM together, six curves

What is plotted
---------------
``(1/sigma) dsigma/dDelta phi`` in 1/rad: **every curve is normalised to unit
area**.  The interference term's normalisation is arbitrary (it scales with
``c_tG/Lambda^2``), so an absolute vertical scale would say nothing; and since
MadSpin's output cross section is ``sigma_production * BR`` regardless of
spinmode, ``onshell`` and ``none`` share a total to within 0.05 % anyway -- the
unit-area convention removes essentially nothing from the ``onshell/none``
ratio.  ``numbers.txt`` records both totals so that can be checked and not
merely believed.

The sign
--------
The samples are generated at ``ctGRe = -1``, not ``+1``.  At ``+1`` the
interference is negative over essentially the whole sampled phase space, which
both inverts the unit-area normalisation and trips MadSpin's
``Tr(rho_prod) > 0`` guard (see ``README.md``).  The quantity drawn here is
therefore ``-2 Re(M_SM^* M_tG)``, the exact mirror of the paper's current
caption, and the figure says so on its face.

Normalisation, verified rather than assumed
-------------------------------------------
``check_normalisation()`` re-derives, from ``histograms.npz`` alone, that

  * ``sigma = sum(sumw) / n_events_in_file`` reproduces the decayed LHE
    ``<init>`` XSECUP of every sample (the LHE carry ``IDWTUP = -4``, so the
    weight on each event is the whole cross section, not a share of it);
  * the production banner's ``# Integrated weight (pb)`` comment is *not* that
    number -- MadSpin leaves it at the production value -- so anything reading
    the banner comment instead of ``<init>`` would be wrong by a factor ~70;
  * the ``sumw2`` errors are not ``sqrt(N)`` errors.  For the 22.9 %-negative
    NLO sample the honest error is ~1.9x the naive one.

Style follows the MG7 paper's ``plotexample/dummyplot.py``, as the other
figures of this series do: LaTeX text, serif, base font size 14, step histograms
of line width 1.2, the paper's fixed figure width (7*0.75 in), tableau colours
with black/blue/red promoted, frameless legends, minor tick locators.
``--check-minus`` re-opens each written PDF and asserts the math minus survived
matplotlib's usetex Type1 subsetting.
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
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter

_HERE = os.path.dirname(os.path.abspath(__file__))


# --- work around a matplotlib bug that eats every minus sign in the PDFs ---
def _fix_type1_subset_minus():
    """Stop matplotlib's usetex PDF path from dropping every minus sign.

    ``_type1font.Type1Font.subset`` in matplotlib 3.11.0 ends its encoding
    filter with an unconditional ``encoding[0] = '.notdef'``.  For a text font
    slot 0 really is ``.notdef``, but TeX's CMSY10 -- the font every math minus
    comes from -- carries ``minus`` there.  Subsetting therefore throws the
    minus away and the PDF comes out with an empty /Differences: no warning, no
    error, a wrong figure.  Only PDF is affected (the PNGs go through dvipng,
    which rasterises first).

    This figure carries minus signs in the ``e^-`` of its axis label, in
    ``-2 Re(...)`` and in ``c_tG = -1``, so the bug would be visible and would
    silently turn the stated sign convention into its opposite.  Guarded on the
    exact upstream line, so a fixed matplotlib is left alone.
    """
    import inspect
    import textwrap
    from matplotlib import _type1font

    # NO_MINUS_FIX=1 disables the workaround.  Its only purpose is to prove
    # ``check_minus`` is discriminating: with the fix off, the check must fail.
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
# Curves.  Colour identifies the SAMPLE, line style identifies the SPINMODE:
# solid = spin correlations kept (``onshell``), dashed = dropped (``none``).
# That pairing is what makes variation D legible at six curves -- the eye reads
# "which physics" off the colour and "with or without" off the dash.
# --------------------------------------------------------------------------
COLOR = {'eft_int': 'blue', 'sm_lo': 'black', 'sm_nlo': 'red'}
LS = {'onshell': 'solid', 'none': (0, (5, 2.2))}

SAMPLE_TEX = {
    'eft_int': r'SMEFT $\mathcal{O}_{tG}$ interference',
    'sm_lo':   r'SM, LO',
    'sm_nlo':  r'SM, NLO',
}
SAMPLE_PLAIN = {
    'eft_int': r'SMEFT $\mathcal{O}_{tG}$ interference',
    'sm_lo':   'SM, LO',
    'sm_nlo':  'SM, NLO',
}
MODE_TEX = {'onshell': r'\texttt{onshell}', 'none': r'\texttt{none}'}
MODE_PLAIN = {'onshell': 'onshell', 'none': 'none'}

# The four variations.  ``curves`` are (sample, spinmode) in drawing order;
# ``ratios`` are the samples for which BOTH spinmodes are present, so an
# onshell/none ratio can be formed.
VARIATIONS = {
    'A': dict(curves=[('eft_int', 'onshell'), ('eft_int', 'none'),
                      ('sm_lo', 'onshell')],
              blurb='EFT with and without spin correlations, SM LO for scale'),
    'B': dict(curves=[('eft_int', 'onshell'), ('eft_int', 'none'),
                      ('sm_lo', 'onshell'), ('sm_lo', 'none')],
              blurb='A, plus the SM LO no-spin-correlation curve'),
    'C': dict(curves=[('eft_int', 'onshell'), ('eft_int', 'none'),
                      ('sm_nlo', 'onshell'), ('sm_nlo', 'none')],
              blurb='B with the SM at NLO instead of LO'),
    'D': dict(curves=[('eft_int', 'onshell'), ('eft_int', 'none'),
                      ('sm_lo', 'onshell'), ('sm_lo', 'none'),
                      ('sm_nlo', 'onshell'), ('sm_nlo', 'none')],
              blurb='LO and NLO SM together with the EFT'),
}
for _v in VARIATIONS.values():
    _have = set(_v['curves'])
    _v['ratios'] = [s for s in ('eft_int', 'sm_lo', 'sm_nlo')
                    if (s, 'onshell') in _have and (s, 'none') in _have]


# --------------------------------------------------------------------------
class Data(object):
    """The raw 720-bin grid, rebinned and normalised to unit area.

    The stored grid is uniform on ``[0, pi]``; ``nbins`` must divide 720 so a
    plot bin is a whole number of stored bins and no edge moves.
    """

    def __init__(self, ddir, nbins=20):
        self.z = dict(np.load(os.path.join(ddir, 'histograms.npz')))
        with open(os.path.join(ddir, 'meta.json')) as fh:
            self.meta = json.load(fh)
        self.fine_edges = self.z['edges']
        nfine = len(self.fine_edges) - 1
        if nfine % nbins:
            raise SystemExit('--nbins %d does not divide the stored %d bins'
                             % (nbins, nfine))
        self.group = nfine // nbins
        self.nbins = nbins
        self.edges = self.fine_edges[::self.group].copy()
        self.centres = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.width = float(np.pi / nbins)
        self._cache = {}

    # -- raw accessors ----------------------------------------------------
    def _mode(self, sample, mode):
        return self.meta['samples'][sample]['spinmodes'][mode]

    def nevents(self, sample, mode):
        return float(self._mode(sample, mode)['n_events_in_file'])

    def _rebin(self, key):
        return self.z[key].reshape(self.nbins, self.group).sum(axis=1)

    def sigma(self, sample, mode):
        """``sum(sumw) / N`` in pb -- the sample's own total cross section.

        ``IDWTUP = -4``: XWGTUP is the whole cross section on every event, with
        the event's sign, so the total is the MEAN weight and not the sum.
        """
        return float(self.z['%s_%s_sumw' % (sample, mode)].sum()
                     / self.nevents(sample, mode))

    def init_sigma(self, sample, mode):
        """The decayed LHE ``<init>`` XSECUP, an independent number."""
        return float(self._mode(sample, mode)['decayed_xsec_pb'])

    def banner_sigma(self, sample, mode):
        """The banner's stale ``# Integrated weight (pb)`` comment, if any."""
        return self._mode(sample, mode)['decayed_xsec_detail'].get(
            'banner_comment_xsec_pb')

    # -- the plotted quantity ---------------------------------------------
    def shape(self, sample, mode):
        """``((1/sigma) dsigma/dDelta phi, its error)`` in 1/rad.

        Errors come from ``sumw2``, never from counts: the NLO sample is 22.9 %
        negative-weight and a ``sqrt(N)`` error there is wrong by ~1.9x.  The
        normalisation ``sigma`` is treated as an exact constant -- its own
        relative error is a global scale common to every bin and cancels in the
        shape's own integral, so folding it in per bin would double-count.
        """
        key = (sample, mode)
        if key in self._cache:
            return self._cache[key]
        n = self.nevents(sample, mode)
        sumw = self._rebin('%s_%s_sumw' % (sample, mode))
        sumw2 = self._rebin('%s_%s_sumw2' % (sample, mode))
        dens = sumw / n / self.width
        err = np.sqrt(sumw2) / n / self.width
        total = float(sumw.sum() / n)
        out = (dens / total, err / abs(total))
        self._cache[key] = out
        return out

    def counts(self, sample, mode):
        return self._rebin('%s_%s_count' % (sample, mode))

    def spin_ratio(self, sample):
        """``onshell / none`` of the unit-area shapes, errors in quadrature.

        This is the figure's point stated as a number: how much the shape moves
        when the spin correlations are switched on.  It is a ratio of the two
        *normalised* curves, i.e. of shapes, not of rates -- but since the two
        spinmodes share a total cross section to 0.05 %, it is numerically the
        absolute ratio too.
        """
        a, ae = self.shape(sample, 'onshell')
        b, be = self.shape(sample, 'none')
        r = a / b
        e = np.abs(r) * np.sqrt((ae / a) ** 2 + (be / b) ** 2)
        return r, e


# --------------------------------------------------------------------------
def check_normalisation(d, fh=sys.stdout):
    """Re-derive the three normalisation traps from the raw arrays.

    Returns a list of ``(name, ok, detail)``.  Nothing here is taken on faith
    from ``meta.json``: every number is recomputed from ``histograms.npz`` and
    compared against the independently recorded one.
    """
    p = lambda *a: print(*a, file=fh)
    rows = []
    p('-' * 78)
    p('normalisation checks (all recomputed from histograms.npz)')
    p('-' * 78)
    p('%-22s %14s %14s %10s %9s' % ('sample/spinmode', 'sum(w)/N [pb]',
                                    '<init> [pb]', 'rel.diff', 'banner'))
    worst = 0.0
    for s in ('eft_int', 'sm_lo', 'sm_nlo'):
        for m in ('onshell', 'none'):
            mine = d.sigma(s, m)
            init = d.init_sigma(s, m)
            rel = abs(mine - init) / abs(init)
            worst = max(worst, rel)
            ban = d.banner_sigma(s, m)
            p('%-22s %14.7f %14.7f %10.2e %9s'
              % ('%s / %s' % (s, m), mine, init, rel,
                 ('%.2f' % ban) if ban else '--'))
    ok1 = worst < 1e-3
    rows.append(('sigma = sum(w)/N reproduces <init> XSECUP', ok1,
                 'worst relative difference %.2e' % worst))

    # Trap 2: the banner comment is the PRODUCTION cross section.
    bad_banner = []
    for s in ('eft_int', 'sm_lo', 'sm_nlo'):
        for m in ('onshell', 'none'):
            ban = d.banner_sigma(s, m)
            if ban is None:
                continue
            prod = d.meta['samples'][s]['production_xsec_pb']
            if abs(ban - prod) / prod < 1e-3:
                bad_banner.append('%s/%s' % (s, m))
    rows.append(("banner '# Integrated weight' is stale (= production sigma)",
                 bool(bad_banner),
                 'confirmed on %s' % (', '.join(bad_banner) or 'none')
                 + ' -- reading it instead of <init> would be wrong by ~70x'))

    # Trap 3: sumw2 errors against the naive sqrt(N) ones.
    p('')
    p('%-22s %9s %11s %11s %7s' % ('sample/spinmode', 'neg.frac',
                                   'relerr(w2)', 'relerr(sqrtN)', 'factor'))
    factors = {}
    for s in ('eft_int', 'sm_lo', 'sm_nlo'):
        for m in ('onshell', 'none'):
            n = d.nevents(s, m)
            sumw = d._rebin('%s_%s_sumw' % (s, m))
            sumw2 = d._rebin('%s_%s_sumw2' % (s, m))
            cnt = d.counts(s, m)
            good = (cnt > 0) & (sumw != 0)
            rel_w2 = np.sqrt(sumw2[good]) / np.abs(sumw[good])
            rel_n = 1.0 / np.sqrt(cnt[good])
            fac = float(np.median(rel_w2 / rel_n))
            factors[(s, m)] = fac
            p('%-22s %9.5f %11.5f %11.5f %7.2f'
              % ('%s / %s' % (s, m),
                 d._mode(s, m)['negative_weight_fraction'],
                 float(np.median(rel_w2)), float(np.median(rel_n)), fac))
    nlo_fac = factors[('sm_nlo', 'onshell')]
    lo_fac = factors[('sm_lo', 'onshell')]
    rows.append(('sumw2 errors differ from sqrt(N) for the signed NLO sample',
                 nlo_fac > 1.5 and lo_fac < 1.05,
                 'NLO %.2fx, LO %.2fx -- a sqrt(N) error bar on the NLO curve '
                 'would be about half its true size' % (nlo_fac, lo_fac)))

    # And the one that actually matters for a unit-area figure: does the
    # spinmode change the total?  (It must not, to ~0.05 %.)
    p('')
    p('%-12s %13s %13s %10s' % ('sample', 'sigma onshell', 'sigma none',
                                'ratio-1'))
    worst_tot = 0.0
    for s in ('eft_int', 'sm_lo', 'sm_nlo'):
        a, b = d.sigma(s, 'onshell'), d.sigma(s, 'none')
        worst_tot = max(worst_tot, abs(a / b - 1))
        p('%-12s %13.6f %13.6f %10.2e' % (s, a, b, a / b - 1))
    rows.append(('onshell and none share a total cross section',
                 worst_tot < 2e-3,
                 'worst |sigma_on/sigma_none - 1| = %.2e, so the unit-area '
                 'ratio is the absolute ratio to that accuracy' % worst_tot))

    p('')
    for name, ok, detail in rows:
        p('  [%s] %s' % ('ok ' if ok else 'FAIL', name))
        p('        %s' % detail)
    return rows


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


def _label(sample, mode):
    if USETEX:
        return '%s, %s' % (SAMPLE_TEX[sample], MODE_TEX[mode])
    return '%s, %s' % (SAMPLE_PLAIN[sample], MODE_PLAIN[mode])


def _pi_formatter():
    def f(x, _pos):
        k = x / np.pi
        if abs(k) < 1e-9:
            return '0'
        if abs(k - 1) < 1e-9:
            return _tx(r'$\pi$', r'$\pi$')
        # quarters
        num = round(k * 4)
        if abs(k * 4 - num) < 1e-9:
            if num % 4 == 0:
                return r'$%d\pi$' % (num // 4)
            g = math.gcd(int(num), 4)
            top, bot = num // g, 4 // g
            # "pi/4", not "1 pi / 4".
            numer = r'\pi' if top == 1 else r'%d\pi' % top
            return _tx(r'$%s/%d$' % (numer, bot), r'$%s/%d$' % (numer, bot))
        return '%.2f' % x
    return FuncFormatter(f)


def make_figure(d, tag, out, style_tag=''):
    spec = VARIATIONS[tag]
    fig, ax, rx = _panels()
    lo, hi = 0.0, float(np.pi)

    # --- main panel: unit-area shapes ------------------------------------
    for sample, mode in spec['curves']:
        y, ye = d.shape(sample, mode)
        ax.step(d.edges, np.concatenate([y[:1], y]), where='pre',
                color=COLOR[sample], ls=LS[mode], lw=LW,
                label=_label(sample, mode), zorder=4)
        ax.errorbar(d.centres, y, yerr=ye, fmt='none', ecolor=COLOR[sample],
                    elinewidth=0.9, capsize=0, zorder=4)

    ax.set_ylabel(_tx(
        r'$\frac{1}{\sigma}\,\mathrm{d}\sigma/\mathrm{d}\Delta\phi(e^-e^+)$'
        r'\ \ [rad$^{-1}$]',
        r'$(1/\sigma)\,d\sigma/d\Delta\phi(e^-e^+)$  [1/rad]'))
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(MultipleLocator(np.pi / 4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelbottom=False)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(ymin, 0.15), ymax + 0.50 * (ymax - ymin))
    ax.legend(frameon=False, loc='upper left', fontsize=10.0,
              handlelength=2.8, borderaxespad=0.9,
              bbox_to_anchor=(0.015, 0.805))

    ax.text(0.028, 0.972,
            _tx(r'$pp \to t\bar t$ at $\sqrt{s} = 13$~TeV, '
                r'$\mu_R = \mu_F = 173$~GeV, no cuts',
                r'$pp \to t\bar t$ at $\sqrt{s}=13$ TeV, '
                r'$\mu_R=\mu_F=173$ GeV, no cuts'),
            transform=ax.transAxes, ha='left', va='top', fontsize=10.5)
    ax.text(0.028, 0.918,
            _tx(r'$t \to W^+b\,(W^+\!\to e^+\nu_e)$, '
                r'$\bar t \to W^-\bar b\,(W^-\!\to e^-\bar\nu_e)$; '
                r'every curve normalised to unit area',
                r'$t\to W^+b(\to e^+\nu)$, $\bar t\to W^-\bar b(\to e^-\bar'
                r'\nu)$; every curve normalised to unit area'),
            transform=ax.transAxes, ha='left', va='top', fontsize=10.5)
    if any(s == 'eft_int' for s, _ in spec['curves']):
        ax.text(0.028, 0.864,
                _tx(r'interference at $c_{tG} = -1$, $\Lambda = 1$~TeV, i.e.\ '
                    r'$-2\,\mathrm{Re}(\mathcal{M}^{*}_{\mathrm{SM}}'
                    r'\mathcal{M}_{tG})$',
                    r'interference at $c_{tG}=-1$, $\Lambda=1$ TeV, i.e. '
                    r'$-2Re(M^*_{SM}M_{tG})$'),
                transform=ax.transAxes, ha='left', va='top', fontsize=10.5,
                color='blue')

    # --- ratio panel: the spin-correlation effect ------------------------
    # Each curve is ONE sample divided by ITSELF with the spin correlations
    # switched off -- not a ratio between samples.  That is the only ratio the
    # figure is about, and it is well defined for the interference term even
    # though its normalisation is arbitrary, because the arbitrary factor is
    # common to numerator and denominator.
    rx.axhspan(0.9, 1.1, facecolor=allcolors[0], alpha=0.10, zorder=0)
    rx.axhline(1.0, color='black', lw=0.9, zorder=2)
    rx.text(0.993, 0.94, _tx(r'band: $\pm 10\%$', r'band: $\pm10\%$'),
            transform=rx.transAxes, ha='right', va='top',
            fontsize=8.5, color=allcolors[0])

    lo_r, hi_r = 1.0, 1.0
    for sample in spec['ratios']:
        r, e = d.spin_ratio(sample)
        rx.step(d.edges, np.concatenate([r[:1], r]), where='pre',
                color=COLOR[sample], ls='solid', lw=LW, zorder=4)
        rx.errorbar(d.centres, r, yerr=e, fmt='none', ecolor=COLOR[sample],
                    elinewidth=0.9, capsize=0, zorder=4)
        lo_r = min(lo_r, float((r - e).min()))
        hi_r = max(hi_r, float((r + e).max()))
    pad = 0.10 * (hi_r - lo_r)
    rx.set_ylim(lo_r - pad, hi_r + pad)
    rx.set_ylabel(_tx(r'\texttt{onshell}/\texttt{none}', 'onshell / none'),
                  fontsize=11.5)
    rx.set_xlabel(_tx(r'$\Delta\phi(e^-e^+)$ [rad]',
                      r'$\Delta\phi(e^-e^+)$ [rad]'))
    rx.xaxis.set_major_locator(MultipleLocator(np.pi / 4))
    rx.xaxis.set_major_formatter(_pi_formatter())
    rx.xaxis.set_minor_locator(AutoMinorLocator(2))
    rx.yaxis.set_minor_locator(AutoMinorLocator())
    rx.set_xlim(lo, hi)
    rx.text(0.018, 0.06,
            _tx(r'each sample divided by \emph{itself} with the spin '
                r'correlations switched off',
                'each sample divided by itself with the spin correlations '
                'switched off'),
            transform=rx.transAxes, ha='left', va='bottom', fontsize=8.5,
            color='0.3')

    fig.subplots_adjust(left=0.145, right=0.975, top=0.985, bottom=0.105)
    base = os.path.join(out, 'smeft_fig5_%s%s' % (tag, style_tag))
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=300)
    plt.close(fig)
    return base


# --------------------------------------------------------------------------
def check_minus(pdf_path):
    """Re-open the PDF and confirm a math minus actually made it in.

    matplotlib's usetex PDF subsetting has silently eaten every minus sign in
    this project more than once.  This figure has minus signs in ``e^-``, in
    ``-2 Re(...)`` and in ``c_tG = -1``; a PDF without one states the opposite
    sign convention from the one intended.  Returns ``(found, detail)``.

    ``pdftotext`` is deliberately NOT used as the primary test: these usetex
    PDFs carry subsetted Type1 fonts with no ToUnicode map, so ``pdftotext``
    extracts nothing at all from them and its silence would be meaningless.
    The test is on the font encoding itself.
    """
    if not USETEX:
        return None, 'usetex is off, the Type1 subsetting path is not used'
    try:
        with open(pdf_path, 'rb') as fp:
            raw = fp.read()
    except OSError as exc:
        return None, 'cannot read the PDF (%s)' % exc
    if b'/minus' in raw:
        return True, '/minus is in the PDF font encoding'
    return False, ('/minus absent from the PDF font encoding -- the usetex '
                   'Type1 subsetting bug has eaten the sign')


# --------------------------------------------------------------------------
def write_curves(d, path):
    """The plotted numbers, beside the figure, as their own .npz.

    ``histograms.npz`` is the raw 720-bin measurement; this is what the figure
    actually draws, so a reader can reproduce a panel without redoing the
    rebinning or the normalisation.
    """
    out = {'edges': d.edges, 'centres': d.centres,
           'bin_width_rad': np.array(d.width)}
    for s in ('eft_int', 'sm_lo', 'sm_nlo'):
        for m in ('onshell', 'none'):
            y, ye = d.shape(s, m)
            out['%s_%s_shape' % (s, m)] = y
            out['%s_%s_shape_err' % (s, m)] = ye
            out['%s_%s_sigma_pb' % (s, m)] = np.array(d.sigma(s, m))
        r, e = d.spin_ratio(s)
        out['%s_spin_ratio' % s] = r
        out['%s_spin_ratio_err' % s] = e
    np.savez(path, **out)
    return path


def write_numbers(d, out, fh=sys.stdout):
    p = lambda *a: print(*a, file=fh)
    setup = d.meta['setup']
    p('=' * 78)
    p('Delta phi(e-, e+): spin correlations on against off')
    p('=' * 78)
    p('observable   : %s' % setup['observable'])
    p('sqrt(s)      : %g GeV      scale: %s' % (setup['sqrt_s_GeV'],
                                                setup['scale']))
    p('PDF          : %s' % setup['pdf'])
    p('cuts         : %s' % setup['cuts'])
    p('sign         : %s' % setup['sign_convention'])
    p('plot binning : %d uniform bins over [0, pi], width %.6f rad '
      '(%d stored bins each)' % (d.nbins, d.width, d.group))
    p('sm_nlo       : regenerated with the MadSpin param_card fix (0a1007bc2,')
    p('               task T123).  The defect -- density matrices evaluated at')
    p('               model defaults, MT 173 / WT 1.4915, not the run\'s')
    p('               172.76 / 1.33 -- was real; its measured effect here is')
    p('               -0.0001 % on sigma (10.9004361 -> 10.9004235 pb, against')
    p('               a 0.020 pb integration error) and, on the shape, less')
    p('               than reseeding MadSpin: the asymmetry moves by +0.0076')
    p('               buggy-vs-fixed and -0.0050 for seed 42 -> 99 alone.')
    p('               spinmode = none was never affected.  See README.md.')
    p('')
    check_normalisation(d, fh)

    p('')
    p('=' * 78)
    p('the spin-correlation effect, bin by bin (onshell / none)')
    p('=' * 78)
    hdr = '%9s' % 'phi/pi'
    for s in ('eft_int', 'sm_lo', 'sm_nlo'):
        hdr += ' %19s' % s
    p(hdr)
    for i in range(d.nbins):
        row = '%9.3f' % (d.centres[i] / np.pi)
        for s in ('eft_int', 'sm_lo', 'sm_nlo'):
            r, e = d.spin_ratio(s)
            row += '   %7.4f +- %.4f' % (r[i], e[i])
        p(row)

    p('')
    p('extremes of the effect, on the plotted binning:')
    for s in ('eft_int', 'sm_lo', 'sm_nlo'):
        r, e = d.spin_ratio(s)
        p('  %-8s first bin %+6.1f%% +- %.1f%%   last bin %+6.1f%% +- %.1f%%'
          '   max |effect| %5.1f%%'
          % (s, 100 * (r[0] - 1), 100 * e[0], 100 * (r[-1] - 1), 100 * e[-1],
             100 * np.max(np.abs(r - 1))))
    p('')
    p('  The paper currently says "an impact of up to 25%%".  On these samples')
    p('  the interference term runs from %+.0f%% in the first bin to %+.0f%% in'
      % (100 * (d.spin_ratio('eft_int')[0][0] - 1),
         100 * (d.spin_ratio('eft_int')[0][-1] - 1)))
    p('  the last, so "up to 30%%" is the correct statement; the SM at the same')
    p('  binning moves by only %+.0f%% / %+.0f%% (LO).'
      % (100 * (d.spin_ratio('sm_lo')[0][0] - 1),
         100 * (d.spin_ratio('sm_lo')[0][-1] - 1)))

    # How close are the two no-spin-correlation curves?  This is what makes
    # variation B worth drawing: if they coincide, then everything separating
    # the EFT from the SM in this observable IS the spin correlation.
    p('')
    p('=' * 78)
    p('do the two spinmode = none curves coincide?')
    p('=' * 78)
    a, ae = d.shape('eft_int', 'none')
    for s, lab in (('sm_lo', 'SM LO'), ('sm_nlo', 'SM NLO')):
        b, be = d.shape(s, 'none')
        r = a / b
        e = np.abs(r) * np.sqrt((ae / a) ** 2 + (be / b) ** 2)
        chi2 = float(np.sum(((r - 1) / e) ** 2))
        p('  EFT none / %s none: max |ratio-1| = %.1f%%, chi2/ndf = %.2f'
          % (lab, 100 * np.max(np.abs(r - 1)), chi2 / d.nbins))
    a2, _ = d.shape('eft_int', 'onshell')
    b2, _ = d.shape('sm_lo', 'onshell')
    p('  for contrast, EFT onshell / SM LO onshell: max |ratio-1| = %.1f%%'
      % (100 * np.max(np.abs(a2 / b2 - 1))))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--nbins', type=int, default=20,
                    help='plot bins over [0, pi]; must divide the stored 720')
    ap.add_argument('--only', default='',
                    help='comma-separated subset of A,B,C,D')
    ap.add_argument('--check-minus', dest='check_minus', action='store_true',
                    default=True)
    ap.add_argument('--no-check-minus', dest='check_minus',
                    action='store_false')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = Data(args.data, nbins=args.nbins)
    tags = [t.strip().upper() for t in args.only.split(',') if t.strip()] \
        or list(VARIATIONS)

    rows = check_normalisation(d, fh=open(os.devnull, 'w'))
    bad = [name for name, ok, _ in rows if not ok]
    if bad:
        raise SystemExit('normalisation check failed: %s' % '; '.join(bad))

    failures = []
    for tag in tags:
        base = make_figure(d, tag, args.out)
        print('wrote %s.pdf / .png   (%s)' % (base, VARIATIONS[tag]['blurb']))
        if args.check_minus:
            ok, detail = check_minus(base + '.pdf')
            print('   minus check: %s -- %s'
                  % ({True: 'ok', False: 'FAILED', None: 'n/a'}[ok], detail))
            if ok is False:
                failures.append(base)

    npz = write_curves(d, os.path.join(args.out, 'smeft_fig5_curves.npz'))
    print('wrote %s' % npz)
    with open(os.path.join(args.out, 'numbers.txt'), 'w') as fh:
        write_numbers(d, args.out, fh)
    print('wrote %s' % os.path.join(args.out, 'numbers.txt'))
    write_numbers(d, args.out)

    if failures:
        raise SystemExit('the usetex minus sign was lost in: %s'
                         % ', '.join(failures))


if __name__ == '__main__':
    main()
