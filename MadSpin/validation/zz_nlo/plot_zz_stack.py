#!/usr/bin/env python3
"""The physics figure: what ``g g \\to Z Z`` adds on top of NLO ``p p \\to Z Z``.

Three samples, one set of cuts, absolute normalisation in pb::

    NLO   p p > z z [QCD]          q q~ Born, plus q g and g q~ real emission
    LI    g g > z z [noborn=QCD]   the loop-induced quark box, O(alpha_s^2)
                                   relative to the Born -- formally beyond NLO,
                                   numerically not negligible
    LO    p p > z z                the curve both ratio panes divide by

The main pane stacks LI on top of NLO and draws LO as a line; the two ratio
panes are ``NLO/LO`` and ``(NLO+LI)/LO``.

**They do not double count.**  The NLO calculation has no ``g g`` initial state
at all: its Born is ``q q~ -> Z Z`` and its real emission is
``q q~ -> Z Z g`` / ``q g -> Z Z q`` / ``g q~ -> Z Z q~``.  ``g g -> Z Z``
proceeds through a closed quark loop with no Born to attach to, which puts it at
``O(alpha_s^2)`` relative to the LO cross section, i.e. it first appears at NNLO.
That is the argument; ``numbers.txt`` carries the *measurement* -- the initial
state of every written event of both samples, summed by weight -- because "they
should not overlap" is exactly the kind of statement this figure would hide if
it were wrong.

Runs entirely off the committed raw histograms::

    plot_zz_stack.py [--data DIR] [--out DIR] [--check-minus]

Style follows the MG7 paper's ``plotexample/dummyplot.py``, as the sibling
studies under ``MadSpin/validation/`` do.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import observables_zz as OZ                                      # noqa: E402
# The usetex Type1 minus-subsetting workaround and its check live in the
# loop-induced study's MG7-style plotter; imported rather than copied, so a
# single fix covers both studies.
_LI = os.path.abspath(os.path.join(_HERE, '..', 'zz_loopinduced'))
if _LI not in sys.path:
    sys.path.insert(0, _LI)
from plot_zz_loopinduced import (                                # noqa: E402
    USETEX, check_minus, MINUS_FIX, LW, ratio,
)


C_NLO = 'blue'
C_LI = 'red'
C_LO = 'black'
C_TOT = 'darkgreen'

LOGY = {'m_zz', 'pt_z_lead', 'pt_z_sublead', 'pt_zz'}

# Which observables the stacked figure is drawn for.  ``pt_zz`` is harvested but
# excluded: it is exactly zero for every LO and every loop-induced event (both
# are 2 -> 2 with no initial-state transverse momentum), so a stack of it would
# be one distribution against two delta functions.
STACK_OBS = ('m_zz', 'pt_z_lead', 'abs_y_zz', 'abs_cos_theta_star',
             'pt_z_sublead')

# The ratio-pane window, per observable.  Fixed in the code rather than
# autoscaled -- the house style of the sibling studies -- but NOT the same
# window for every observable, because these ratios are not all near one: the
# K factor runs from 1.2 to 2.0 across m(ZZ) and to 5.3 across pt(Z), where the
# real emission that LO cannot produce takes over.  One window wide enough for
# pt(Z) would flatten every other figure into a line.
RATIO_WINDOW = {
    'm_zz': (1.0, 2.4),
    'pt_z_lead': (0.8, 6.0),
    'pt_z_sublead': (1.0, 2.1),
    'abs_y_zz': (0.9, 2.4),
    'abs_cos_theta_star': (1.2, 1.8),
}
RATIO_CLIP = (0.8, 2.4)          # the fallback, for an observable not listed


def window(obs):
    return RATIO_WINDOW.get(obs, RATIO_CLIP)


def _drawn_tick_texts(axis, lo, hi):
    """The tick label strings this axis actually PUTS ON THE PAGE.

    ``axis.get_ticklabels()`` is not that, in two ways that both matter here.
    It returns a label for every tick the locator produced, including the ones
    whose location falls outside the current view -- a log axis limited to
    ``(0.12, 370)`` still owns a ``10^{-1}`` tick at 0.1 and a ``10^{3}`` tick
    at 1000, and neither is drawn.  And it ignores ``set_visible(False)``,
    which is how a shared-x upper pane hides its own x labels.  Asking for the
    drawn text needs both filters.
    """
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    out = []
    for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
        loc = tick.get_loc()
        if loc is None or not (a <= loc <= b):
            continue
        for lab in (tick.label1, tick.label2):
            if lab is not None and lab.get_visible() and lab.get_text():
                out.append(lab.get_text())
    return out


def wants_minus(fig):
    """Does this figure actually contain a minus sign to render?

    ``check_minus`` greps a PDF for ``/minus``, which answers "did the Type1
    subsetting bug eat the sign".  On a figure whose every axis is positive --
    ``|y(ZZ)|`` from 0 to 4, ``|cos theta*|`` from 0 to 1, a ratio pane clipped
    above 1 -- there is no minus to eat, and reporting that one as a FAILURE
    would be as wrong as reporting a genuinely eaten sign as a pass.  So the
    figure is asked, after it is drawn, whether any of its text carries one.

    "Its text" has to mean the text that reaches the page.  An earlier version
    of this asked ``get_xticklabels()`` / ``get_yticklabels()`` instead, and
    those hand back labels for ticks the axis does not draw -- which turned a
    log pane whose limits happen to start just above ``0.1`` into a figure
    claiming a ``10^{-1}`` it never renders, and then reported the resulting
    signless PDF as a FAILURE.  That is precisely the false alarm this function
    exists to prevent, so it now filters on the view interval and on
    ``set_visible``; see :func:`_drawn_tick_texts`.
    """
    fig.canvas.draw()
    for ax in fig.axes:
        texts = _drawn_tick_texts(ax.xaxis, *ax.get_xlim())
        texts += _drawn_tick_texts(ax.yaxis, *ax.get_ylim())
        texts = [t for t in texts]
        for t in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if t is not None and t.get_visible():
                texts.append(t.get_text())
        leg = ax.get_legend()
        if leg is not None:
            texts += [t.get_text() for t in leg.get_texts()]
        for s in texts:
            if '-' in s or '\u2212' in s:
                return True
    return False


class Stack(object):
    """The committed production-level histograms and the totals."""

    def __init__(self, ddir):
        self.z = np.load(os.path.join(ddir, 'histograms.npz'))
        self.meta = json.load(open(os.path.join(ddir, 'meta.json')))
        self.bins = {k: np.array(v) for k, v in self.meta['bins_zz'].items()}

    def edges(self, obs):
        return self.bins[obs]

    def centres(self, obs):
        e = self.edges(obs)
        return 0.5 * (e[:-1] + e[1:])

    def density(self, key, obs):
        return (self.z['%s/%s/y' % (key, obs)],
                self.z['%s/%s/e' % (key, obs)])

    def sigma(self, key):
        return self.meta['production'][key]['sigma_from_events']

    def sigma_err(self, key):
        """The INTEGRATION error, not the spread of the written events.

        For the LO and loop-induced samples every event carries the same weight,
        so ``sqrt(sum w^2)/N`` collapses to ``sigma/sqrt(N)`` -- a number that is
        not an uncertainty on sigma at all.  The MC@NLO sample is different: its
        weights carry both signs and several magnitudes, so its event-level
        error IS meaningful, but it is still not the integrator's, which is what
        the ratios here should use.
        """
        p = self.meta['production'][key]
        return p.get('integration_error_pb', p['sigma_mc_error'])

    def nevents(self, key):
        return self.meta['production'][key]['nevents']


def draw(d, obs, outdir):
    xlab, ylab = (OZ.LABELS_ZZ[obs] if USETEX
                  else (OZ.LABELS_ZZ_TXT[obs],
                        'dsigma/d(%s) [pb per unit]'
                        % OZ.LABELS_ZZ_TXT[obs].split(' [')[0]))
    edges = d.edges(obs)
    x = d.centres(obs)

    ynlo, enlo = d.density('nlo', obs)
    yli, eli = d.density('li', obs)
    ylo, elo = d.density('lo', obs)
    ytot = ynlo + yli
    etot = np.sqrt(enlo ** 2 + eli ** 2)

    fig = plt.figure(figsize=(7 * 0.75, 7.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    r1 = fig.add_subplot(gs[1], sharex=ax)
    r2 = fig.add_subplot(gs[2], sharex=ax)

    # --- the stack.  ONLY the band between NLO and NLO+LI is filled.
    #
    # Filling from zero as well would be the textbook stacked histogram, and on
    # a four-decade log axis it is unreadable: the whole plot becomes one solid
    # colour and the 13 % gg band on top of it is a hairline that also hides the
    # LO curve underneath.  Filling the gg band alone puts the ink exactly on
    # the quantity the figure is about -- the height of the shading IS the gg
    # contribution -- and leaves the three curves visible.
    ax.stairs(ytot, edges, baseline=ynlo, fill=True, color=C_LI, alpha=0.30,
              lw=0)
    ax.stairs(ynlo, edges, color=C_NLO, lw=LW,
              label=(r'NLO: $pp \to ZZ$ [QCD] ($q\bar q$, $qg$, $g\bar q$)'
                     if USETEX else 'NLO: p p > z z [QCD]'))
    ax.stairs(ytot, edges, color=C_TOT, lw=LW,
              label=(r'NLO $+$ LI $gg$' if USETEX else 'NLO + LI gg'))
    ax.stairs(yli, edges, color=C_LI, lw=LW, ls='dashed',
              label=(r'loop induced: $gg \to ZZ$ ($\mathcal{O}(\alpha_s^2)$ '
                     r'rel.\ LO)' if USETEX else 'loop induced: g g > z z'))
    ax.stairs(ylo, edges, color=C_LO, lw=LW, ls='dashdot',
              label=(r'LO: $pp \to ZZ$' if USETEX else 'LO: p p > z z'))

    if obs in LOGY:
        ax.set_yscale('log')
    ax.set_ylabel(ylab)
    ylo_, yhi_ = ax.get_ylim()
    ax.set_ylim(ylo_, yhi_ * (25.0 if obs in LOGY else 1.55))
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if obs not in LOGY:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.setp(ax.get_xticklabels(), visible=False)

    clip = window(obs)
    for pane, (ynum, enum, col, lab) in zip(
            (r1, r2),
            ((ynlo, enlo, C_NLO, r'NLO / LO'),
             (ytot, etot, C_TOT, r'(NLO$+$LI) / LO' if USETEX
              else '(NLO+LI) / LO'))):
        r, re_ = ratio(ynum, enum, ylo, elo)
        pane.errorbar(x, np.clip(r, *clip), yerr=re_, fmt='none',
                      ecolor=col, elinewidth=0.8, alpha=0.55)
        pane.stairs(np.clip(r, *clip), edges, color=col, lw=LW)
        pane.axhline(1.0, color='black', lw=0.8, ls=':')
        pane.set_ylim(*clip)
        pane.set_ylabel(lab, fontsize=10)
        pane.xaxis.set_minor_locator(AutoMinorLocator())
        # A window of (1.0, 2.4) gets exactly two automatic ticks, "1" and "2",
        # which is not enough to read a K factor off.  Five majors, and minors
        # between them.
        pane.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        pane.yaxis.set_minor_locator(AutoMinorLocator())
    plt.setp(r1.get_xticklabels(), visible=False)
    r2.set_xlim(edges[0], edges[-1])
    r2.set_xlabel(xlab)

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, 'stack_' + obs)
    want = wants_minus(fig)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return base, want


# --------------------------------------------------------------------------
def write_numbers(d, path):
    out = []
    A = out.append
    m = d.meta
    A('p p > z z at NLO, with the loop-induced g g > z z stacked on top,')
    A('against p p > z z at LO.  Same cuts, same scale, same PDF, same N.')
    A('code %s' % m.get('code_sha', '?'))
    A('scale: %s' % m.get('scale', '?'))
    A('cut:   pt(Z) > %g GeV on BOTH z, via pt_min_pdg = %s'
      % (m['pt_z_min'], m['pt_min_pdg']))
    A('')
    A('--- total cross sections ---')
    A('%-6s %14s %13s %10s %10s' % ('sample', 'sigma [pb]', 'int. error', 'N',
                                    'N(w<0)'))
    for k in ('lo', 'nlo', 'li'):
        p = m['production'][k]
        A('%-6s %14.7g %13.4g %10d %10d'
          % (k, d.sigma(k), d.sigma_err(k), d.nevents(k),
             p['n_negative_weight']))
    A('The quoted sigma is the SUM OF THE EVENT WEIGHTS, because that is what')
    A('normalises the histograms.  The quoted error is the INTEGRATION error,')
    A('because that is the uncertainty on the prediction.  For an unweighted')
    A('sample the two sigmas are identical; for the MC@NLO one the unweighting')
    A('makes them differ by a fraction of the integration error:')
    for k in ('lo', 'nlo', 'li'):
        p = m['production'][k]
        i = p.get('integration_sigma_pb')
        if i is None:
            continue
        A('   %-6s sum of weights %12.6g   integrated %12.6g   '
          'difference %+.3f %% (%+.2f sigma)'
          % (k, d.sigma(k), i, 100 * (d.sigma(k) / i - 1),
             (d.sigma(k) - i) / d.sigma_err(k)))
    A('')
    slo, elo = d.sigma('lo'), d.sigma_err('lo')
    snlo, enlo = d.sigma('nlo'), d.sigma_err('nlo')
    sli, eli = d.sigma('li'), d.sigma_err('li')
    A('')

    def rat(a, ea, b, eb):
        r = a / b
        return r, r * math.sqrt((ea / a) ** 2 + (eb / b) ** 2)

    r, e = rat(snlo, enlo, slo, elo)
    A('K = NLO / LO                 = %.5f +- %.5f' % (r, e))
    r, e = rat(snlo + sli, math.hypot(enlo, eli), slo, elo)
    A('(NLO + LI) / LO              = %.5f +- %.5f' % (r, e))
    r, e = rat(sli, eli, snlo, enlo)
    A('LI / NLO                     = %.5f +- %.5f  (%+.2f %%)'
      % (r, e, 100 * r))
    r, e = rat(sli, eli, slo, elo)
    A('LI / LO                      = %.5f +- %.5f  (%+.2f %%)'
      % (r, e, 100 * r))
    A('')
    if 'fixed_order_nlo' in m:
        f = m['fixed_order_nlo']
        A('--- cross-check: the same process at FIXED ORDER NLO ---')
        A('%s' % f['note'])
        A('sigma = %.6g +- %.4g pb   (MC@NLO event sample: %.6g +- %.4g pb)'
          % (f['sigma_pb'], f['error_pb'], snlo, enlo))
        rr, ee = rat(snlo, enlo, f['sigma_pb'], f['error_pb'])
        A('ratio MC@NLO / fixed order = %.5f +- %.5f' % (rr, ee))
        A('')

    A('--- the double-counting check, measured on the written events ---')
    A('Initial-state parton pairs, with the weight each carries, read off every')
    A('event of each sample.  21 is the gluon.  The NLO sample must carry no')
    A('(21, 21) entry: g g > Z Z has no Born to attach to and is O(alpha_s^2)')
    A('relative to LO, i.e. an NNLO contribution, not an NLO one.')
    for k in ('lo', 'nlo', 'li'):
        p = m['production'][k]
        tot = sum(abs(v) for v in p['initial_states'].values()) or 1.0
        A('')
        A('%s:' % k)
        for st, w in sorted(p['initial_states'].items(),
                            key=lambda kv: -abs(kv[1])):
            A('   %-10s  %8.4f %% of the sample weight' % (st, 100 * w / tot))
        A('   gluon-gluon initial state present: %s'
          % ('YES' if '21 21' in p['initial_states'] else 'no'))
    A('')
    A('--- where the gg contribution sits, differentially ---')
    for obs in STACK_OBS:
        ynlo, enlo_ = d.density('nlo', obs)
        yli, eli_ = d.density('li', obs)
        ylo_, elo_ = d.density('lo', obs)
        edges = d.edges(obs)
        w = np.diff(edges)
        A('')
        A('%s' % OZ.LABELS_ZZ_TXT[obs])
        A('   %12s %12s %12s %10s %10s' % ('bin lo', 'bin hi', 'frac of NLO+LI',
                                           'LI/NLO', 'NLO/LO'))
        for i in range(len(w)):
            tot = (ynlo[i] + yli[i]) * w[i]
            if tot <= 0:
                continue
            frac = tot / ((ynlo + yli) * w).sum()
            if frac < 2e-3:
                continue
            A('   %12.4g %12.4g %12.4f %10.4f %10.4f'
              % (edges[i], edges[i + 1], frac,
                 yli[i] / ynlo[i] if ynlo[i] > 0 else float('nan'),
                 ynlo[i] / ylo_[i] if ylo_[i] > 0 else float('nan')))
    open(path, 'w').write('\n'.join(out) + '\n')
    print('wrote %s' % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--check-minus', action='store_true')
    args = ap.parse_args()
    d = Stack(args.data)
    bases = [draw(d, obs, args.out) for obs in STACK_OBS]
    write_numbers(d, os.path.join(args.data, 'numbers_stack.txt'))
    if args.check_minus:
        print('usetex = %s   minus workaround active = %s' % (USETEX, MINUS_FIX))
        bad = n = 0
        for b, want in bases:
            if not want:
                print('%s: no minus sign in this figure (every axis positive), '
                      'check not applicable' % os.path.basename(b))
                continue
            n += 1
            ok, msg = check_minus(b + '.pdf')
            if not ok:
                bad += 1
                print(msg)
        print('%d/%d applicable PDFs carry /minus' % (n - bad, n))
        if bad:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
