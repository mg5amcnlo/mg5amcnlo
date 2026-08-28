#!/usr/bin/env python3
"""MadSpin's spinmodes against the fully off-shell four-lepton calculation for
the loop-induced ``g g > z z``, in the MG7 paper's plotting style.

Runs entirely off the committed raw histograms, so nothing here needs MadSpin
or MadGraph::

    plot_zz_loopinduced.py [--data DIR] [--out DIR] [--check-minus]

with ``<data>`` holding ``histograms.npz`` and ``meta.json`` as written by
``run_zz_loopinduced.py``.

What is compared
----------------
``truth``
    ``g g > e+ e- mu+ mu- / a [noborn=QCD]``: the four leptons come out of the
    loop-induced matrix element directly, with both ``Z`` propagators fully
    off shell and the two decays correlated by construction.  A
    ``|m_ll - m_Z| < 15 Gamma_Z`` window on *both* reconstructed pairs makes it
    comparable to a MadSpin sample drawn at ``BW_cut = 15``.  Every ratio pane
    divides by this.

the four MadSpin modes
    ``g g > z z [noborn=QCD]`` decayed by ``decay z > e+ e-`` plus
    ``decay z > mu+ mu-``.  Two ``z`` and two ``decay`` lines for ``z`` is the
    *positional* rule, so the first ``z`` takes the first line and the second
    the second: exactly one ``e+e-`` and one ``mu+mu-`` pair per event, never
    ``4e`` or ``4mu``.  ``madspin_v1`` and ``onshell_v1`` are absent because
    MadSpin refuses them for a loop-induced process.

The vertical scale is the ABSOLUTE ``dsigma/dx``, not a shape.  On this branch
MadSpin's reported cross section already carries the Breit-Wigner truncation
(PR #379, merged), and sample B's mass window removes the same rate for real,
so the two normalisations are meant to agree and the absolute comparison is the
interesting one rather than something to divide out.  ``numbers.txt`` carries
the totals and their ratios.

Style follows the MG7 paper's ``plotexample/dummyplot.py``, as the sibling
studies under ``MadSpin/validation/`` do: LaTeX text, serif, base font size 14,
step histograms of line width 1.2, the paper's fixed figure width, tableau
colours with black/blue/red promoted, frameless legends, minor tick locators.
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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import observables as OBS                                        # noqa: E402


# --- work around a matplotlib bug that eats every minus sign in the PDFs ---
def _fix_type1_subset_minus():
    """Stop matplotlib's usetex PDF path from dropping every minus sign.

    ``_type1font.Type1Font.subset`` ends its encoding filter with an
    unconditional ``encoding[0] = '.notdef'``.  For a text font slot 0 really is
    ``.notdef``, but TeX's CMSY10 -- the font every math minus comes from --
    carries ``minus`` there.  Subsetting therefore throws the minus away and the
    PDF comes out with an empty /Differences and no /BaseEncoding, so viewers
    silently drop the sign: no warning, no error, a wrong figure.  Only PDF is
    affected; the PNGs go through dvipng, which rasterises first.

    These figures are full of minus signs that matter.  Every mass spectrum is
    log-y, so its axis is a column of ``10^{-n}``; the ``Phi`` figure runs over
    ``[-pi, pi]``, so its x axis is half negative; and ``cos theta_1`` runs over
    ``[-1, 1]``.  A dropped minus on the ``Phi`` axis would silently mirror the
    reader's idea of the figure.

    Guarded on the exact upstream line, so a fixed matplotlib is left alone.
    ``--check-minus`` re-opens the written PDF and asserts the sign survived;
    ``NO_MINUS_FIX=1`` turns the workaround off, which is how that check is
    itself verified to discriminate.
    """
    import inspect
    import textwrap
    from matplotlib import _type1font

    if os.environ.get('NO_MINUS_FIX'):
        return False

    bad = "encoding[0] = '.notdef'"
    good = "encoding.setdefault(0, '.notdef')"
    try:
        src = textwrap.dedent(inspect.getsource(_type1font.Type1Font.subset))
    except (OSError, TypeError):
        return False
    if bad not in src:                      # already fixed upstream
        return False
    ns = vars(_type1font).copy()
    exec(compile(src.replace(bad, good), '<type1font-minus-fix>', 'exec'), ns)
    _type1font.Type1Font.subset = ns['subset']
    return True


MINUS_FIX = _fix_type1_subset_minus()


def check_minus(pdf):
    """Did a minus sign survive into ``pdf``?

    Discriminating only on a usetex-rendered PDF: a non-usetex PDF carries
    ``/minus`` in its font encoding whether or not the workaround ran, so
    running this on the user-style figures would report True either way.  That
    is why ``plot_zz_loopinduced_userstyle.py`` does not call it.
    """
    try:
        blob = open(pdf, 'rb').read()
    except OSError as exc:
        return False, str(exc)
    ok = b'/minus' in blob
    return ok, ('/minus present' if ok else '/minus ABSENT from ' + pdf)


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

REF = 'truth'
MODES = ['madspin', 'PA', 'onshell', 'none']

CURVES_TEX = {
    'truth':   r'$gg \to e^{+}e^{-}\mu^{+}\mu^{-}$ (off shell, loop induced)',
    'madspin': r'MadSpin, \texttt{spinmode = madspin}',
    'PA':      r'MadSpin, \texttt{spinmode = PA}',
    'onshell': r'MadSpin, \texttt{spinmode = onshell}',
    'none':    r'MadSpin, \texttt{spinmode = none}',
}
CURVES_PLAIN = {
    'truth':   'gg -> e+ e- mu+ mu- (off shell, loop induced)',
    'madspin': 'MadSpin, spinmode = madspin',
    'PA':      'MadSpin, spinmode = PA',
    'onshell': 'MadSpin, spinmode = onshell',
    'none':    'MadSpin, spinmode = none',
}

COLOR = {'truth': 'black', 'madspin': 'blue', 'PA': 'red',
         'onshell': allcolors[2], 'none': allcolors[4]}
LS = {'truth': 'solid', 'madspin': 'solid', 'PA': 'dashed',
      'onshell': 'dashdot', 'none': (0, (1, 1.4))}

# Which observables get a log vertical scale.  The mass spectra fall by two to
# four decades across their range; the angles are flat to within a factor of a
# few and a log axis on them would only hide the differences under study.
LOGY = {'m_epmum', 'm_epmup', 'm_4l', 'pt_ee', 'm_ee', 'm_mumu'}

# The ratio pane window.  Wide enough that the modes that genuinely disagree
# leave it (and get arrows), tight enough that the ones that agree are resolved.
RATIO_CLIP = (0.5, 1.5)

PROC_TEX = r'gg \to ZZ \to e^{+}e^{-}\mu^{+}\mu^{-}'


class Data(object):
    """The committed histograms, plus the totals the ratio panes need."""

    def __init__(self, ddir):
        self.z = np.load(os.path.join(ddir, 'histograms.npz'))
        self.meta = json.load(open(os.path.join(ddir, 'meta.json')))
        self.bins = {k: np.array(v) for k, v in self.meta['bins'].items()}

    def has(self, key, obs='m_ee'):
        return '%s/%s/y' % (key, obs) in self.z

    def edges(self, obs):
        return self.bins[obs]

    def centres(self, obs):
        e = self.edges(obs)
        return 0.5 * (e[:-1] + e[1:])

    def density(self, key, obs):
        """``(dsigma/dx, error)`` in pb per unit of ``x``."""
        return (self.z['%s/%s/y' % (key, obs)],
                self.z['%s/%s/e' % (key, obs)])

    def sigma(self, key):
        return self.meta['runs'][key]['sigma_from_events']

    def sigma_err(self, key):
        """The INTEGRATION error on the total, not the spread of the events.

        These files are unweighted: every event carries the same weight and
        ``sqrt(sum w^2)/N`` collapses to ``sigma/sqrt(N)``, i.e. 0.45 % at
        50 000 events.  That number is not an error on sigma at all -- sigma is
        fixed by the integration that produced the sample, not by counting the
        events it wrote out -- and quoting it would inflate every uncertainty
        here by a factor of three and hide the 4.6 sigma normalisation result.

        So the integrator's own number is used: MadEvent's quoted error for the
        truth sample, and ``cmd.error`` for the MadSpin ones (which is the
        production sample's integration error carried through the branching
        ratio).  ``sigma_mc_error`` stays in meta.json, but as what it is -- a
        per-bin statistical scale -- and the per-bin histogram errors, where
        ``sqrt(sum w^2)`` IS the right thing, are unaffected.
        """
        rep = self.meta.get('reported', {}).get(key)
        if rep and rep.get('error'):
            return rep['error']
        ban = self.meta.get('integration_error', {}).get(key)
        if ban:
            return ban
        return self.meta['runs'][key]['sigma_mc_error']

    def nevents(self, key):
        return self.meta['runs'][key]['nevents']


def ratio(num, nerr, den, derr):
    """``num/den`` with both errors, and NaN where the reference is empty."""
    ok = den > 0
    r = np.full_like(num, np.nan, dtype=float)
    e = np.full_like(num, np.nan, dtype=float)
    r[ok] = num[ok] / den[ok]
    rel = np.zeros_like(num)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel[ok] = np.sqrt((nerr[ok] / np.where(num[ok] > 0, num[ok], np.inf)) ** 2
                          + (derr[ok] / den[ok]) ** 2)
    e[ok] = np.abs(r[ok]) * rel[ok]
    return r, e


# Modes that draw NO virtuality for the decaying particle.  ``onshell`` and
# ``none`` both leave the z exactly at its pole mass, so their reconstructed
# pair mass is a delta function at m_Z: every other bin of the 15-width window
# is empty *by construction*, not by chance.
NO_VIRTUALITY = ('onshell', 'none')
PAIR_MASS_OBS = ('m_ee', 'm_mumu')


def structurally_empty(y, key, obs):
    """Bins that are empty BY CONSTRUCTION rather than by sample size.

    The distinction matters and cannot be read off the histogram: an empty bin
    in the ``m(e+ mu-)`` tail of a 50 000-event sample is a statement about the
    sample, while an empty bin in the ``m(e+e-)`` spectrum of ``spinmode =
    onshell`` is a statement about the mode -- it draws no virtuality, so the
    pair mass is exactly ``m_Z`` and no amount of statistics will ever fill the
    window around it.  Only the second kind gets a marker; the first is left as
    a gap, because drawing "measured to be zero" where the truth is "not
    measured" is the same error the ratio pane exists to avoid.

    Testing ``y == 0`` alone gets this wrong in both directions, which is why
    the mode and the observable are arguments and not decoration.
    """
    if key in NO_VIRTUALITY and obs in PAIR_MASS_OBS:
        return y == 0.0
    return np.zeros_like(y, dtype=bool)


def offscale_arrows(ax, x, y, lo, hi, color):
    """Draw an arrow at the pane edge for each point outside ``[lo, hi]``."""
    for xi, yi in zip(x, y):
        if not np.isfinite(yi):
            continue
        if yi > hi:
            ax.annotate('', xy=(xi, hi), xytext=(xi, hi - 0.12 * (hi - lo)),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.0))
        elif yi < lo:
            ax.annotate('', xy=(xi, lo), xytext=(xi, lo + 0.12 * (hi - lo)),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.0))


def draw(data, obs, outdir, modes=MODES):
    xlab, ylab = OBS.LABELS[obs] if USETEX else (OBS.LABELS_TXT[obs], '')
    if not USETEX:
        ylab = 'dsigma/d(%s) [pb per unit]' % OBS.LABELS_TXT[obs].split(' [')[0]
    edges = data.edges(obs)
    x = data.centres(obs)

    fig = plt.figure(figsize=(7 * 0.75, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    rx = fig.add_subplot(gs[1], sharex=ax)

    yref, eref = data.density(REF, obs)

    order = [REF] + [m for m in modes if data.has(m, obs)]
    for key in order:
        y, e = data.density(key, obs)
        lab = CURVES_TEX[key] if USETEX else CURVES_PLAIN[key]
        ax.stairs(y, edges, color=COLOR[key], ls=LS[key], lw=LW, label=lab)
        if key == REF:
            continue
        r, re_ = ratio(y, e, yref, eref)
        rx.errorbar(x, np.clip(r, *RATIO_CLIP), yerr=re_, fmt='none',
                    ecolor=COLOR[key], elinewidth=0.8, alpha=0.55)
        rx.stairs(np.clip(r, *RATIO_CLIP), edges, color=COLOR[key],
                  ls=LS[key], lw=LW)
        offscale_arrows(rx, x, r, RATIO_CLIP[0], RATIO_CLIP[1], COLOR[key])
        # exact structural zeros get an open marker at the pane floor
        empt = structurally_empty(y, key, obs)
        if empt.any():
            rx.plot(x[empt], np.full(empt.sum(), RATIO_CLIP[0]), 'o',
                    mfc='none', mec=COLOR[key], ms=4, lw=0)

    if obs in LOGY:
        ax.set_yscale('log')
    ax.set_ylabel(ylab)
    # Headroom before the legend is placed, so a five-entry frameless legend
    # cannot land on top of the Breit-Wigner peak of the mass spectra.
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * (12.0 if obs in LOGY else 1.45))
    # 'best' only after the headroom is in place: matplotlib minimises overlap
    # against the data as drawn, and the angular figures are U-shaped (high at
    # both edges) while the mass ones peak in the middle, so no single fixed
    # corner works for all of them.
    ax.legend(frameon=False, fontsize=9,
              loc='upper left' if obs in LOGY else 'best')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator()) if obs not in LOGY else None
    plt.setp(ax.get_xticklabels(), visible=False)

    rx.axhline(1.0, color='black', lw=0.8, ls=':')
    rx.set_ylim(*RATIO_CLIP)
    rx.set_xlim(edges[0], edges[-1])
    rx.set_xlabel(xlab)
    rx.set_ylabel(r'ratio' if USETEX else 'ratio', fontsize=11)
    rx.xaxis.set_minor_locator(AutoMinorLocator())
    rx.yaxis.set_minor_locator(AutoMinorLocator())

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, obs)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return base


# --------------------------------------------------------------------------
# the numeric report
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# The named coefficients behind the two angular figures
# --------------------------------------------------------------------------
def _binned_power(data, key, obs, p):
    """``(<x^p>, error)`` of a committed 1-D histogram, without binning bias.

    Using the bin CENTRE is not good enough here.  Under a locally flat density
    the exact bin average of ``x^2`` is ``c^2 + h^2/12``, and on the smooth
    ``cos(theta)`` histograms that already closes to 1e-4 against the unbinned
    moment.  On ``cos1cos2`` it does not: that density has a logarithmic cusp at
    zero, the flat-in-bin assumption is wrong right where the bins are, and the
    residual bias on ``<(c1 c2)^2>`` is +4.4e-4 -- which ``f_00`` multiplies by
    25, i.e. +0.011, the size of the whole statistical error.

    So the density inside each bin is reconstructed as the quadratic through the
    three neighbouring bin averages -- ``f(c+u) = A + B u + C u^2`` with
    ``B, C`` the central first and second differences and ``A`` fixed so that the
    quadratic integrates to the bin's own mass -- and ``x^p`` is integrated
    against it on a sub-grid.  Measured against the unbinned truth of an
    independent 250 000-event sample that leaves +3.3e-5 on ``<(c1 c2)^2>`` and
    -1.6e-5 on ``<cos^2 theta>``, both an order of magnitude under the bars.

    The error is the multinomial one, ``(<x^2p> - <x^p>^2) / N_eff`` with
    ``N_eff = (sum y)^2 / sum e^2``: summing the per-bin errors in quadrature
    instead would ignore that the total is fixed and inflate it by ~60 %.
    """
    e = data.edges(obs)
    y, err = data.density(key, obs)
    tot = y.sum()
    m = y / tot                       # bin masses, summing to 1
    h = np.diff(e)
    dens = m / h
    dm1 = np.concatenate([dens[:1], dens[:-1]])
    dp1 = np.concatenate([dens[1:], dens[-1:]])
    B = (dp1 - dm1) / (2.0 * h)
    C = (dp1 - 2.0 * dens + dm1) / (2.0 * h ** 2)
    A = dens - C * h ** 2 / 12.0      # so that int_bin (A + B u + C u^2) du = m
    c = 0.5 * (e[:-1] + e[1:])
    # Simpson on 21 points per bin; the weights are the same for every bin
    n = 21
    u = np.linspace(-0.5, 0.5, n)
    sw = np.ones(n)
    sw[1:-1:2] = 4.0
    sw[2:-1:2] = 2.0
    sw *= 1.0 / (3.0 * (n - 1))       # int over u in [-1/2, 1/2] of weight = 1
    x = c[:, None] + h[:, None] * u[None, :]
    f = (A[:, None] + B[:, None] * (h[:, None] * u[None, :])
         + C[:, None] * (h[:, None] * u[None, :]) ** 2) * h[:, None]

    def mom(q):
        return float(((f * x ** q) * sw[None, :]).sum())

    m1, m2 = mom(p), mom(2 * p)
    neff = tot ** 2 / float((err ** 2).sum())
    return m1, math.sqrt(max(m2 - m1 ** 2, 0.0) / neff), neff


def _polarisation(data, key):
    """``{name: (value, error, exact?)}`` for one sample.

    Prefers the unbinned moments the harvester stores; falls back to the
    committed histograms for a meta.json written before they existed, which is
    why every row is flagged.  The fallback for ``f_00`` treats the two z as
    independent when it propagates the error -- exact for the value, an
    approximation for the bar, and the bar is far too large to matter anyway.
    """
    mom = data.meta['runs'][key].get('moments', {})
    out = {}
    if 'pol0_1' in mom:
        for src_name, dst in (('pol0_1', 'f_0 (e+ e-)'), ('pol0_2', 'f_0 (mu+ mu-)'),
                              ('pol0_avg', 'f_0 (both)'),
                              ('pol00', 'f_00'), ('polTT', 'f_TT')):
            if src_name in mom:
                out[dst] = (mom[src_name][0], mom[src_name][1], True)
    else:
        c1, e1, neff = _binned_power(data, key, 'cos_theta1', 2)
        c2, e2, _ = _binned_power(data, key, 'cos_theta2', 2)
        q1, _, _ = _binned_power(data, key, 'cos_theta1', 4)
        q2, _, _ = _binned_power(data, key, 'cos_theta2', 4)
        pp, _, _ = _binned_power(data, key, 'cos1cos2', 2)
        # f_0 of one z needs only <cos^2 theta>, and the harvester has always
        # stored that unbinned for the (e+ e-) side, so prefer it.
        ex1 = 'cos2_theta1' in mom
        ex2 = 'cos2_theta2' in mom
        if ex1:
            c1, e1 = mom['cos2_theta1']
        if ex2:
            c2, e2 = mom['cos2_theta2']
        f01, f02 = 2 - 5 * c1, 2 - 5 * c2
        # f_00 = <(2 - 5 c1^2)(2 - 5 c2^2)> = 25 <(c1 c2)^2> - 10 <c1^2> - 10 <c2^2> + 4
        f00 = 25 * pp - 10 * (c1 + c2) + 4
        x2 = [4 - 20 * c + 25 * q for c, q in ((c1, q1), (c2, q2))]
        e00 = math.sqrt(max(x2[0] * x2[1] - (f01 * f02) ** 2, 0.0) / neff)
        out['f_0 (e+ e-)'] = (f01, 5 * e1, ex1)
        out['f_0 (mu+ mu-)'] = (f02, 5 * e2, ex2)
        out['f_00'] = (f00, e00, False)
        out['f_TT'] = (1 - f01 - f02 + f00, e00, False)
        var = [v - f ** 2 for v, f in zip(x2, (f01, f02))]
        out['f_00 - f_0 f_0'] = (f00 - f01 * f02,
                                 math.sqrt(var[0] * var[1] / neff), False)
    if 'pol0_1' in mom:
        n = float(data.nevents(key))
        v1 = mom['pol0_1'][1] ** 2 * n
        v2 = mom['pol0_2'][1] ** 2 * n
        out['f_00 - f_0 f_0'] = (mom['pol00'][0] - mom['pol0_1'][0] * mom['pol0_2'][0],
                                 math.sqrt(v1 * v2 / n), True)
    if 'cos1cos2' in mom:
        out['C_kk'] = OBS.c_kk_from_moment(*mom['cos1cos2']) + (True,)
    return out


def write_polarisation(data, A, modes):
    """The block of numbers.txt that names what the two angular figures measure."""
    keys = [REF] + [k for k in modes if k in data.meta['runs']
                    and data.meta['runs'][k].get('moments')]
    rows = {k: _polarisation(data, k) for k in keys}
    names = ['f_0 (e+ e-)', 'f_0 (mu+ mu-)', 'f_0 (both)', 'f_00',
             'f_00 - f_0 f_0', 'f_TT', 'C_kk']
    names = [n for n in names if any(n in r for r in rows.values())]
    exact = all(v[2] for r in rows.values() for v in r.values())
    A('--- the named coefficients behind the two angular figures ---')
    A('  A Z is a WEAK spin analyser.  Its decay to l+ l- carries the analysing')
    A('  power eta_l = 2 g_V g_A / (g_V^2 + g_A^2) = %.5f for this run'
      % OBS.ETA_L)
    A('  (MW = %.4f from (aEWM1, Gf, MZ), sw2 = 1 - MW^2/MZ^2 = %.6f), and it'
      % (OBS.M_W, OBS.SW2))
    A('  multiplies ONLY the parity-violating cos(theta) term.  So:')
    A('')
    A('    f_0   = 2 - 5 <cos^2 theta>                      UNDILUTED')
    A('            the longitudinal fraction of one z; 1/3 is isotropic.')
    A('    f_00  = <(2 - 5 cos^2 th1)(2 - 5 cos^2 th2)>     UNDILUTED')
    A('            the joint (both longitudinal) fraction; f_0^2 if the two')
    A('            z were independent, so f_00 - f_0^2 is the rank-2')
    A('            correlation and the analogue of t t~\'s C_nn that survives.')
    A('    C_kk  = 4 <cos th1 cos th2> / eta_l^2            DILUTED by eta_l^2')
    A('            = <S_k(1) S_k(2)>, each S on its own helicity axis.  The')
    A('            calibration is 4 / eta_l^2 = %.1f, not the 9 of t t~: 9 is'
      % (4.0 / OBS.ETA_L ** 2))
    A('            the spin-1/2 algebra with kappa = 1, 4 is the spin-1 one.')
    A('')
    if not exact:
        A('  [f_0 / f_00 / f_TT below come from the committed histograms, not')
        A('   from unbinned moments: this meta.json predates the pol0_* moments.')
        A('   The binning bias is removed exactly (see _binned_power); the f_00')
        A('   bar is the independent-z approximation.  Re-harvesting replaces')
        A('   the whole block with the unbinned numbers.]')
        A('')
    A('%-12s %s' % ('sample', ' '.join('%-22s' % n for n in names)))
    for k in keys:
        cells = []
        for n in names:
            v = rows[k].get(n)
            cells.append('%-22s' % ('%+.4f +- %.4f' % (v[0], v[1]) if v else '--'))
        A('%-12s %s' % (k, ' '.join(cells)))
    A('')
    ref = rows[REF]
    A('  Against truth, in sigma (truth and the modes are independent samples):')
    for k in keys[1:]:
        bits = []
        for n in names:
            a, b = rows[k].get(n), ref.get(n)
            if not a or not b:
                continue
            s = math.sqrt(a[1] ** 2 + b[1] ** 2)
            bits.append('%s %+.1f' % (n.split(' ')[0] + n[n.find('('):] if '(' in n
                                      else n, (a[0] - b[0]) / s if s else float('nan')))
        A('    %-10s %s' % (k, '   '.join(bits)))
    A('')
    A('  f_0 = 1/3 exactly is what an ISOTROPIC decay gives, and it is what')
    A('  spinmode = none has to give: decaying each z on its own with no')
    A('  production density leaves it unpolarised, and an unpolarised z decays')
    A('  flat in cos(theta).  That is a prediction, and the "none" row above')
    A('  is the measurement of it.')
    A('')


def write_numbers(data, path, modes=MODES):
    """Everything the figures show, plus what they cannot: the totals."""
    out = []
    A = out.append
    m = data.meta
    A('g g > z z  (loop induced) + MadSpin   against   g g > e+ e- mu+ mu- / a'
      '  (loop induced)')
    A('code %s' % m.get('code_sha', '?'))
    A('m_Z = %.6g GeV   Gamma_Z = %.6g GeV   BW_cut = %s'
      % (m['m_Z'], m['width_Z'], m['BW_cut']))
    A('mass window on both reconstructed pairs of the truth sample: '
      '%.5f .. %.5f GeV' % tuple(m['mass_window']))
    A('')
    A('--- total cross sections, from the event weights (IDWTUP = -4, '
      'sigma = mean(w)) ---')
    A('%-12s %14s %12s %12s' % ('sample', 'sigma [pb]', 'MC error', 'N'))
    for key in [REF] + [k for k in modes if data.has(k)]:
        A('%-12s %14.7g %12.3g %12d'
          % (key, data.sigma(key), data.sigma_err(key), data.nevents(key)))
    A('')
    sref, eref = data.sigma(REF), data.sigma_err(REF)
    A('--- ratio to truth ---')
    for key in [k for k in modes if data.has(k)]:
        s, e = data.sigma(key), data.sigma_err(key)
        r = s / sref
        re_ = r * math.sqrt((e / s) ** 2 + (eref / sref) ** 2)
        A('%-12s %.5f +- %.5f   (%+.2f %%, %.1f sigma)'
          % (key, r, re_, 100 * (r - 1), abs(r - 1) / re_ if re_ else float('nan')))
    A('')
    if 'reported' in m:
        A('--- what MadSpin itself reported (cmd.cross, cmd.branching_ratio) ---')
        for key in modes:
            rep = m['reported'].get(key)
            if not rep or 'cross' not in rep:
                continue
            A('%-12s cross = %.10g   BR = %.10g   efficiency = %s'
              % (key, rep['cross'], rep['br'], rep['efficiency']))
        A('')

    bw = m.get('bw_retained_fraction')
    if bw:
        A('--- the Breit-Wigner truncation ---')
        A('%s' % bw['source'])
        A('  f       = %.12f' % bw['f'])
        A('  f^2     = %.12f      (two resonances)' % bw['f_squared'])
        A('  1/f^2   = %.12f' % (1.0 / bw['f_squared']))
        A('This branch carries PR #379, so MadSpin APPLIES this factor to the')
        A('modes that draw a virtuality and not to the ones that do not.  The')
        A('reported branching ratios are the direct check:')
        rep = m.get('reported', {})
        trunc = [k for k in ('madspin', 'PA') if rep.get(k, {}).get('br')]
        plain = [k for k in ('onshell', 'none') if rep.get(k, {}).get('br')]
        for a in trunc:
            for b in plain:
                A('  BR(%s)/BR(%s) = %.12f   vs f^2 = %.12f   (difference %.2e)'
                  % (a, b, rep[a]['br'] / rep[b]['br'], bw['f_squared'],
                     rep[a]['br'] / rep[b]['br'] - bw['f_squared']))
        A('')

    ctl = m.get('controls') or {}
    if ctl:
        A('--- do sample B\'s cuts actually fire? ---')
        if 'A_no_ptcut' in ctl:
            a = ctl['A_no_ptcut']
            s0, e0 = a['sigma_pb'], a['error_pb']
            s1 = m.get('integration_sigma', {}).get('production')
            e1 = m.get('integration_error', {}).get('production')
            if s1:
                r = s1 / s0
                re_ = r * math.sqrt((e1 / s1) ** 2 + (e0 / s0) ** 2)
                A('  sample A, ptheavy = 1 : %.6g +- %.4g pb' % (s1, e1))
                A('  sample A, ptheavy = 0 : %.6g +- %.4g pb' % (s0, e0))
                A('  the pt(Z) > 1 GeV cut removes %.3f +- %.3f %% (%.1f sigma '
                  'from doing nothing)'
                  % (100 * (1 - r), 100 * re_, abs(1 - r) / re_))
        if 'B_no_masswindow' in ctl:
            b = ctl['B_no_masswindow']
            if 'retained_fraction_measured' in b:
                f = b['retained_fraction_measured']
                fe = b['retained_fraction_error']
                A('  sample B run again with the SAME pt cut and NO mass window:')
                A('    %.6g +- %.4g pb over %d events'
                  % (b.get('sigma_pb', float('nan')),
                     b.get('error_pb', float('nan')), b['nevents']))
                A('    m(e+e-) then spans %.3f .. %.3f GeV -- outside the window,'
                  % tuple(b['m_ee_range']))
                A('    which is by itself proof that the window in the real run '
                  'was doing something')
                A('    fraction of ITS weight inside the 15-width window on both '
                  'pairs:')
                A('      measured  %.5f +- %.5f' % (f, fe))
                if bw:
                    A('      f^2       %.5f        (propagator only, what '
                      'MadSpin applies)' % bw['f_squared'])
                    A('      residual  %+.3f %%          (the part of the '
                      'truncation the propagator factor does not carry: the '
                      'matrix element and phase space also vary across the '
                      'window)' % (100 * (f / bw['f_squared'] - 1)))
            if 'sigma_pb' in b and m.get('integration_sigma', {}).get('truth'):
                st = m['integration_sigma']['truth']
                et = m['integration_error']['truth']
                r = st / b['sigma_pb']
                re_ = r * math.sqrt((et / st) ** 2
                                    + (b['error_pb'] / b['sigma_pb']) ** 2)
                A('    the same ratio from the two integrations: %.5f +- %.5f'
                  % (r, re_))
        A('')

    mom = {k: v.get('moments') for k, v in m['runs'].items() if v.get('moments')}
    if mom:
        A('--- angular moments (weighted mean +- error of the mean) ---')
        names = ['cos_theta1', 'cos2_theta1', 'cos1cos2', 'cos_phi', 'cos_2phi',
                 'm_epmum']
        A('%-12s %s' % ('sample', ' '.join('%-22s' % n for n in names)))
        for key in [REF] + [k for k in modes if k in mom]:
            if key not in mom:
                continue
            A('%-12s %s' % (key, ' '.join(
                '%-22s' % ('%+.5f +- %.5f' % tuple(mom[key][n]))
                for n in names)))
        A('')
        A('  <cos theta1 . cos theta2> is the strict inter-decay correlation: it')
        A('  is exactly zero for any scheme that decays the two z independently,')
        A('  whatever it does to each z on its own.')
        A('')
    write_polarisation(data, A, modes)
    A('--- per-observable agreement over the bins where truth has support ---')
    A('  "rate" is the ratio of INTEGRALS over those bins, sum(mode)/sum(truth).')
    A('  It is deliberately not an inverse-variance weighted mean of the')
    A('  per-bin ratios: the per-bin error is built from the numerator, so a bin')
    A('  that fluctuates low gets a smaller error and more weight, and the mean')
    A('  comes out biased low -- about 5 % on a 2000-event pilot here, which is')
    A('  the size of the effects under study.  The integral ratio has no such')
    A('  bias.  "chi2/ndf" is the SHAPE test: the per-bin ratios against the')
    A('  best-fit flat line (i.e. against "rate"), so a pure normalisation')
    A('  offset does not enter it and only a genuine shape difference does.')
    for obs in m['observables']:
        yref, eref_ = data.density(REF, obs)
        A('  %s' % OBS.LABELS_TXT[obs])
        for key in [k for k in modes if data.has(k, obs)]:
            y, e = data.density(key, obs)
            ok = yref > 0
            if not ok.any():
                A('    %-10s (no overlap with the truth support)' % key)
                continue
            num, den = float(y[ok].sum()), float(yref[ok].sum())
            nume = float(np.sqrt((e[ok] ** 2).sum()))
            dene = float(np.sqrt((eref_[ok] ** 2).sum()))
            rate = num / den
            rate_e = rate * math.sqrt((nume / num) ** 2 + (dene / den) ** 2) \
                if num > 0 else float('nan')
            r, re_ = ratio(y, e, yref, eref_)
            good = ok & np.isfinite(r) & np.isfinite(re_) & (re_ > 0)
            ndf = int(good.sum()) - 1
            if ndf <= 0:
                # A mode with no virtuality draw puts the whole pair-mass
                # spectrum in one bin, so there is no shape left to test.  The
                # chi2 of a delta function against a Breit-Wigner is a large
                # meaningless number; saying so is more use than printing it.
                A('    %-10s rate = %.4f +- %.4f (%+.2f %%)   shape: delta '
                  'function, %d bin with support -- no shape test'
                  % (key, rate, rate_e, 100 * (rate - 1), int(good.sum())))
                continue
            chi2 = float(np.sum(((r[good] - rate) / re_[good]) ** 2))
            # Bins where the truth has support and the mode has NONE carry a
            # zero numerator, hence a zero error, hence fall out of ``good``
            # and out of the chi2 -- silently, until 2026-08-28.  They are the
            # most significant disagreement a mode can have, so they are named
            # here and their share of the truth is quoted.  For this study they
            # are the m_4l < 2 m_Z region, which no spinmode can reach and which
            # PA_LOWPT_DIAGNOSIS.md is about.
            nzero = int(ok.sum() - good.sum())
            if nzero:
                miss = float(yref[ok & ~good].sum() / yref[ok].sum())
                A('    %-10s rate = %.4f +- %.4f (%+.2f %%)   shape chi2/ndf = '
                  '%.1f/%d = %.2f   [%d bin(s) with truth support and NO support '
                  'in this mode, %.2f %% of the truth over this observable, NOT '
                  'in the chi2]'
                  % (key, rate, rate_e, 100 * (rate - 1), chi2, ndf, chi2 / ndf,
                     nzero, 100 * miss))
                continue
            A('    %-10s rate = %.4f +- %.4f (%+.2f %%)   shape chi2/ndf = '
              '%.1f/%d = %.2f'
              % (key, rate, rate_e, 100 * (rate - 1), chi2, ndf, chi2 / ndf))
        A('')
    open(path, 'w').write('\n'.join(out) + '\n')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--check-minus', action='store_true')
    ap.add_argument('--only-numbers', action='store_true',
                    help='rewrite data/numbers.txt and leave the figures alone')
    args = ap.parse_args()

    data = Data(args.data)
    if args.only_numbers:
        print(write_numbers(data, os.path.join(args.data, 'numbers.txt')))
        return
    made = []
    for obs in data.meta['observables']:
        made.append(draw(data, obs, args.out))
        print('wrote %s.pdf / .png   (usetex=%s, minus fix applied=%s)'
              % (made[-1], USETEX, MINUS_FIX))
    txt = write_numbers(data, os.path.join(args.data, 'numbers.txt'))
    print(txt)

    if args.check_minus:
        bad = []
        for base in made:
            ok, detail = check_minus(base + '.pdf')
            print('minus check %-50s %s' % (os.path.basename(base), detail))
            if not ok:
                bad.append(base)
        if bad:
            raise SystemExit('minus sign missing from: %s' % bad)


if __name__ == '__main__':
    main()
