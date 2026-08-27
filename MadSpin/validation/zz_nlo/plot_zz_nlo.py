#!/usr/bin/env python3
"""MadSpin's spinmodes on an **NLO** ``p p > z z`` sample, in the MG7 paper style.

The loop-induced companion of this study
(``MadSpin/validation/zz_loopinduced``) compared four spinmodes against a
directly generated, fully off-shell four-lepton sample.  Two things are
different here and both change what the figures can claim:

1. **Six modes, not four.**  ``p p > z z [QCD]`` is not loop induced, so
   ``madspin_v1`` and ``onshell_v1`` -- which MadSpin refuses outright for a
   loop-induced process -- run as well.  ``fixed_order`` is off throughout and
   is not needed: MC@NLO events are individual events carrying (possibly
   negative) weights, not the counter-event groups a fixed-order LHE carries,
   and ``fixed_order`` is the option for the latter.  It is *refused* in
   ``PA`` and ``madspin``/``full``, so had these events needed it only
   ``onshell``/``onshell_v1`` would have been usable.  They do not.

2. **The reference.**  Whether an off-shell truth exists is read out of the
   data rather than assumed: if ``data/`` carries a ``truth`` sample the ratio
   panes divide by it, and if it does not they divide by ``spinmode = madspin``
   and every axis label says so.  A ratio against another approximation is a
   comparison between modes, not a validation, and the figures must not be
   readable as the second thing.  RESULTS.md states which of the two this
   particular set of figures is.

Everything else -- the observables, the binning, the labels, the boost, the
Type1 minus-sign workaround, the ratio statistics -- is imported from the
loop-induced study rather than reimplemented::

    plot_zz_nlo.py [--data DIR] [--out DIR] [--check-minus]
"""

import argparse
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import observables_zz as OZ                                      # noqa: E402

_LI = os.path.abspath(os.path.join(_HERE, '..', 'zz_loopinduced'))
if _LI not in sys.path:
    sys.path.insert(0, _LI)
from plot_zz_loopinduced import (                                # noqa: E402
    USETEX, check_minus, MINUS_FIX, LW, allcolors, ratio,
    structurally_empty, offscale_arrows, LOGY, RATIO_CLIP,
)
from plot_zz_stack import wants_minus                            # noqa: E402

OBS = OZ.leptonic()

# Every mode this sample allows, in the order the figures draw them.
MODES = ['madspin', 'PA', 'onshell', 'none', 'madspin_v1', 'onshell_v1']

CURVES_TEX = {
    'truth':      r'$pp \to e^{+}e^{-}\mu^{+}\mu^{-}$ [QCD] (off shell)',
    'madspin':    r'MadSpin, \texttt{spinmode = madspin}',
    'PA':         r'MadSpin, \texttt{spinmode = PA}',
    'onshell':    r'MadSpin, \texttt{spinmode = onshell}',
    'none':       r'MadSpin, \texttt{spinmode = none}',
    'madspin_v1': r'MadSpin, \texttt{spinmode = madspin\_v1}',
    'onshell_v1': r'MadSpin, \texttt{spinmode = onshell\_v1}',
}
CURVES_PLAIN = {k: (v.replace(r'\texttt{', '').replace('}', '')
                    .replace('\\_', '_'))
                for k, v in CURVES_TEX.items()}
CURVES_PLAIN['truth'] = 'p p > e+ e- mu+ mu- [QCD] (off shell)'

COLOR = {'truth': 'black', 'madspin': 'blue', 'PA': 'red',
         'onshell': allcolors[2], 'none': allcolors[4],
         'madspin_v1': allcolors[5], 'onshell_v1': allcolors[6]}
LS = {'truth': 'solid', 'madspin': 'solid', 'PA': 'dashed',
      'onshell': 'dashdot', 'none': (0, (1, 1.4)),
      'madspin_v1': (0, (5, 1.5, 1, 1.5)), 'onshell_v1': (0, (3, 1, 1, 1, 1, 1))}


# Modes that draw NO virtuality: their pair mass is a delta function at m_Z, and
# their four-lepton mass therefore cannot fall below 2 m_Z whatever the recoil.
NO_VIRTUALITY = ('onshell', 'none', 'onshell_v1')


def structural(data, y, key, obs):
    """Bins that are empty BY CONSTRUCTION rather than by sample size.

    The loop-induced study's version covers the pair masses; at NLO there is a
    second case, and it is new rather than inherited.  ``m_4l`` below ``2 m_Z``
    is reachable only by a mode that draws a virtuality AND only for an event
    that has recoil to absorb it -- the reshuffle conserves ``sqrt(shat)``, so a
    2 -> 2 event's ``m_4l`` is its production ``m(ZZ)`` and cannot move.  For
    ``onshell`` / ``none`` / ``onshell_v1`` those bins are exactly zero for a
    reason, and marking them as a measured zero would be the error the ratio
    pane exists to avoid.  It also matters numerically: leaving them in turns a
    shape chi2 of 1.6 into one of 30 or more, all of it from three bins carrying
    0.04 % of the rate.
    """
    base = structurally_empty(y, key, obs)
    # The imported helper predates the two ``_v1`` modes -- the loop-induced
    # study could not run them -- and its no-virtuality list is
    # ('onshell', 'none').  ``onshell_v1`` belongs there too: it reports the
    # same branching ratio as ``onshell`` to every digit, i.e. it draws no
    # virtuality either, so its pair mass is the same delta function at m_Z.
    if obs in ('m_ee', 'm_mumu') and key == 'onshell_v1':
        base = base | (y == 0.0)
    if obs == 'm_4l' and key in NO_VIRTUALITY:
        edges = data.edges(obs)
        base = base | ((edges[1:] <= 2 * data.meta['m_Z']) & (y == 0.0))
    return base


def reference(data):
    """Which curve the ratio panes divide by, and whether it is a truth.

    Returns ``(key, is_truth)``.  Read off the data, never assumed: an NLO
    four-lepton reference may or may not have been affordable, and the figures
    have to say which case they are in.
    """
    if 'truth/m_ee/y' in data.z:
        return 'truth', True
    return 'madspin', False


class Data(object):
    """The committed four-lepton histograms, plus the totals."""

    def __init__(self, ddir):
        import json
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
        return (self.z['%s/%s/y' % (key, obs)],
                self.z['%s/%s/e' % (key, obs)])

    def sigma(self, key):
        return self.meta['runs'][key]['sigma_from_events']

    def sigma_err(self, key):
        """The INTEGRATION error where one exists, not the event spread.

        For the loop-induced study every decayed file was unweighted and
        ``sqrt(sum w^2)/N`` collapsed to a meaningless ``sigma/sqrt(N)``.  These
        files are NOT unweighted: they inherit the MC@NLO sample's signed
        weights, so ``sigma_mc_error`` is a real statistical error on the mean.
        MadSpin's own ``cmd.error`` -- the production integration error carried
        through the branching ratio -- is still the better number and is used
        when present; ``numbers.txt`` prints both.
        """
        rep = self.meta.get('reported', {}).get(key)
        if rep and rep.get('error'):
            return rep['error']
        run = self.meta['runs'][key]
        return run.get('integration_error_pb', run['sigma_mc_error'])

    def nevents(self, key):
        return self.meta['runs'][key]['nevents']


def draw(data, obs, outdir, modes=MODES):
    ref, is_truth = reference(data)
    xlab, ylab = OBS.LABELS[obs] if USETEX else (OBS.LABELS_TXT[obs], '')
    if not USETEX:
        ylab = ('dsigma/d(%s) [pb per unit]'
                % OBS.LABELS_TXT[obs].split(' [')[0])
    edges = data.edges(obs)
    x = data.centres(obs)

    fig = plt.figure(figsize=(7 * 0.75, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    rx = fig.add_subplot(gs[1], sharex=ax)

    yref, eref = data.density(ref, obs)
    order = ([ref] if is_truth else []) + [m for m in modes
                                           if data.has(m, obs)]
    for key in order:
        y, e = data.density(key, obs)
        lab = CURVES_TEX[key] if USETEX else CURVES_PLAIN[key]
        ax.stairs(y, edges, color=COLOR[key], ls=LS[key], lw=LW, label=lab)
        if key == ref:
            continue
        r, re_ = ratio(y, e, yref, eref)
        rx.errorbar(x, np.clip(r, *RATIO_CLIP), yerr=re_, fmt='none',
                    ecolor=COLOR[key], elinewidth=0.8, alpha=0.55)
        rx.stairs(np.clip(r, *RATIO_CLIP), edges, color=COLOR[key],
                  ls=LS[key], lw=LW)
        offscale_arrows(rx, x, r, RATIO_CLIP[0], RATIO_CLIP[1], COLOR[key])
        empt = structural(data, y, key, obs)
        if empt.any():
            rx.plot(x[empt], np.full(empt.sum(), RATIO_CLIP[0]), 'o',
                    mfc='none', mec=COLOR[key], ms=4, lw=0)

    if obs in LOGY:
        ax.set_yscale('log')
    ax.set_ylabel(ylab)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * (18.0 if obs in LOGY else 1.55))
    ax.legend(frameon=False, fontsize=8,
              loc='upper left' if obs in LOGY else 'best')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if obs not in LOGY:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.setp(ax.get_xticklabels(), visible=False)

    rx.axhline(1.0, color='black', lw=0.8, ls=':')
    rx.set_ylim(*RATIO_CLIP)
    rx.set_xlim(edges[0], edges[-1])
    rx.set_xlabel(xlab)
    rx.set_ylabel('ratio' if is_truth else 'mode / madspin', fontsize=11)
    rx.xaxis.set_minor_locator(AutoMinorLocator())
    rx.yaxis.set_minor_locator(AutoMinorLocator())

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, obs)
    want = wants_minus(fig)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return base, want


# --------------------------------------------------------------------------
def chi2_flat(r, e):
    """``chi2/ndf`` of a per-bin ratio against its own best-fit flat line.

    A pure normalisation offset does not enter, so only a genuine SHAPE
    difference is measured.  ``None`` if fewer than two bins are usable.
    """
    ok = np.isfinite(r) & np.isfinite(e) & (e > 0)
    if ok.sum() < 2:
        return None
    w = 1.0 / e[ok] ** 2
    mu = float(np.sum(w * r[ok]) / np.sum(w))
    return float(np.sum(w * (r[ok] - mu) ** 2) / (ok.sum() - 1))


def write_numbers(data, path, modes=MODES):
    ref, is_truth = reference(data)
    out = []
    A = out.append
    m = data.meta
    A('p p > z z [QCD]  (MC@NLO events)  +  MadSpin, all six spinmodes')
    A('code %s' % m.get('code_sha', '?'))
    A('scale: %s' % m.get('scale', '?'))
    A('m_Z = %.6g GeV   Gamma_Z = %.6g GeV   BW_cut = %s'
      % (m['m_Z'], m['width_Z'], m['BW_cut']))
    A('')
    if is_truth:
        A('reference: the directly generated off-shell four-lepton sample.')
    else:
        A('reference: spinmode = madspin.  There is NO off-shell truth in this')
        A('study -- see RESULTS.md for the measured cost that decided that --')
        A('so every ratio below is one approximation against another and NOT a')
        A('validation against the exact answer.')
    A('')
    A('--- total cross sections ---')
    A('%-12s %14s %12s %12s %10s %10s'
      % ('sample', 'sigma [pb]', 'MadSpin err', 'event err', 'N', 'N(w<0)'))
    for key in ([ref] if is_truth else []) + [k for k in modes if data.has(k)]:
        r = m['runs'][key]
        A('%-12s %14.7g %12.3g %12.3g %12d %10d'
          % (key, data.sigma(key), data.sigma_err(key), r['sigma_mc_error'],
             data.nevents(key), r['n_negative_weight']))
    A('')
    sref, eref = data.sigma(ref), data.sigma_err(ref)
    A('--- ratio to %s ---' % ref)
    for key in [k for k in modes if data.has(k) and k != ref]:
        s, e = data.sigma(key), data.sigma_err(key)
        rr = s / sref
        re_ = rr * math.sqrt((e / s) ** 2 + (eref / sref) ** 2)
        A('%-12s %.5f +- %.5f   (%+.2f %%, %.1f sigma)'
          % (key, rr, re_, 100 * (rr - 1),
             abs(rr - 1) / re_ if re_ else float('nan')))
    A('')
    if 'production' in m and 'nlo' in m['production']:
        p = m['production']['nlo']
        A('--- the production sample MadSpin was given ---')
        A('p p > z z [QCD], %d events, sigma = %.6g +- %.4g pb'
          % (p['nevents'], p['sigma_from_events'],
             p.get('integration_error_pb', p['sigma_mc_error'])))
        A('negative weights: %d of %d (%.3f %%); sum|w| / sum w = %.5f'
          % (p['n_negative_weight'], p['nevents'],
             100 * p['negative_weight_fraction'],
             p['sum_abs_w'] / (p['sigma_from_events'] * p['nevents'])))
        A('events with an extra final-state parton (H events): %d of %d'
          % (p['n_events_with_extra_parton'], p['nevents']))
        A('')
        A('--- did MadSpin preserve the negative weights? ---')
        A('Every decayed sample should carry the SAME number of negative-weight')
        A('events as the production sample: MadSpin multiplies each event weight')
        A('by an unsigned branching-ratio factor, so it can neither create nor')
        A('destroy a sign.')
        for key in [k for k in modes if data.has(k)]:
            r = m['runs'][key]
            A('   %-12s N(w<0) = %6d   %s'
              % (key, r['n_negative_weight'],
                 'MATCHES production'
                 if r['n_negative_weight'] == p['n_negative_weight']
                 else '*** DIFFERS from production (%d) ***'
                      % p['n_negative_weight']))
    A('')
    if 'reported' in m:
        A('--- what MadSpin itself reported (cmd.cross, cmd.branching_ratio) ---')
        for key in modes:
            rep = m['reported'].get(key)
            if not rep or 'cross' not in rep:
                continue
            A('%-12s cross = %.9g   BR = %.10g   efficiency = %s'
              % (key, rep['cross'], rep['br'], rep['efficiency']))
    A('')
    A('--- angular moments (weighted mean +- error of the mean) ---')
    names = ['cos_theta1', 'cos2_theta1', 'cos1cos2', 'cos_2phi', 'm_epmum']
    A('%-12s %s' % ('sample', ' '.join('%20s' % n for n in names)))
    for key in ([ref] if is_truth else []) + [k for k in modes if data.has(k)]:
        mom = m['runs'][key]['moments']
        A('%-12s %s' % (key, ' '.join('%10.5f+-%8.5f'
                                      % (mom[n][0], mom[n][1]) for n in names)))
    A('')
    A('--- the four-lepton mass below the on-shell ZZ threshold (2 m_Z = '
      '%.4f GeV) ---' % (2 * m['m_Z']))
    A('Reachable only by a mode that draws a virtuality, AND only for a')
    A('production event that has recoil to absorb it: the reshuffle conserves')
    A('sqrt(shat), so a 2 -> 2 event\'s m_4l is its production m(ZZ) and cannot')
    A('move.  onshell / none / onshell_v1 are therefore EXACTLY zero here, which')
    A('is a structural zero and not a measurement.')
    A('%-12s %10s %18s %14s' % ('sample', 'N', 'weight fraction',
                                'frac / truth'))
    tref = m['runs'][ref].get('weight_fraction_below_2mZ')
    for key in ([ref] if is_truth else []) + [k for k in modes if data.has(k)]:
        r = m['runs'][key]
        f = r.get('weight_fraction_below_2mZ')
        if f is None:
            continue
        A('%-12s %10d %18.3e %14s'
          % (key, r['n_below_2mZ'], f,
             '--' if key == ref or not tref else '%.4f' % (f / tref)))
    A('')
    A('--- binned shape test: chi2/ndf of (mode / %s) against a flat line ---'
      % ref)
    A('(a pure normalisation offset does not enter; only a shape difference does)')
    obslist = m['observables']
    A('%-14s %s' % ('observable', ' '.join('%12s' % k for k in modes
                                           if data.has(k) and k != ref)))
    for obs in obslist:
        yref, e0 = data.density(ref, obs)
        row = []
        for key in [k for k in modes if data.has(k) and k != ref]:
            y, e = data.density(key, obs)
            struct = structural(data, y, key, obs)
            r, re_ = ratio(y, e, yref, e0)
            r = np.where(struct, np.nan, r)
            re_ = np.where(struct, np.nan, re_)
            c = chi2_flat(r, re_)
            row.append('%12s' % ('delta fn' if c is None else '%.2f' % c))
        A('%-14s %s' % (obs, ' '.join(row)))
    # m_4l once more, restricted to the bins that lie entirely above 2 m_Z.
    # The unrestricted number above is dominated by the three sub-threshold
    # bins, which carry a few times 1e-4 of the rate and are a statement about
    # the threshold region alone; quoting only that would read as a shape
    # disagreement across the whole spectrum, which is not what it is.
    edges = data.edges('m_4l')
    keep = edges[:-1] >= 2 * m['m_Z']
    yref, e0 = data.density(ref, 'm_4l')
    row = []
    for key in [k for k in modes if data.has(k) and k != ref]:
        y, e = data.density(key, 'm_4l')
        r, re_ = ratio(y, e, yref, e0)
        c = chi2_flat(np.where(keep, r, np.nan), np.where(keep, re_, np.nan))
        row.append('%12s' % ('n/a' if c is None else '%.2f' % c))
    A('%-14s %s' % ('m_4l >= 2mZ', ' '.join(row)))
    open(path, 'w').write('\n'.join(out) + '\n')
    print('wrote %s' % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--check-minus', action='store_true')
    args = ap.parse_args()
    d = Data(args.data)
    bases = [draw(d, obs, args.out) for obs in d.meta['observables']]
    write_numbers(d, os.path.join(args.data, 'numbers.txt'))
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
