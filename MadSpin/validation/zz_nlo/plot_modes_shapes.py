#!/usr/bin/env python3
"""The six spinmodes on the NEW observables -- MG7 paper style.

The parent study compares MadSpin's six spinmodes against the directly
generated off-shell four-lepton truth on eleven observables.  This module does
the same comparison on the observables the shape study added, so that a
variable introduced on the production side is also asked whether MadSpin
reproduces it.

**How the ``Z``-level observables are reconstructed, and how exact that is.**
Every new production observable is a function of the two ``Z`` four-momenta and
of nothing else.  On the decayed samples those are the reconstructed pairs
``(e+ e-)`` and ``(mu+ mu-)``, and the reconstruction is *exact*: the two ``Z``
are flavour tagged, one decaying to electrons and one to muons, so there is no
combinatorial ambiguity, and four-momentum conservation makes each pair's
four-momentum its parent's.  ``|Delta y|``, ``max |y|``, ``|cos theta*_CS|``
and ``pt/m`` therefore carry over without approximation.

The one thing that does **not** carry over is the mass.  A reconstructed pair
is off shell where a produced ``Z`` was not, so ``m_4l`` is not the production
``m(ZZ)`` event by event.  That is a real physical difference -- it is exactly
what ``spinmode`` controls -- and not an approximation in the reconstruction.
It is also why the mass-sliced angular observables are cut on ``m_4l`` here and
on ``m(ZZ)`` there, and why they are named differently.

Everything about the drawing, the colours, the reference logic and the ratio
statistics is imported from ``plot_zz_nlo.py`` rather than copied::

    plot_modes_shapes.py [--data DIR] [--out DIR] [--check-minus]
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import observables_shapes as OS                                  # noqa: E402
_LI = os.path.abspath(os.path.join(_HERE, '..', 'zz_loopinduced'))
if _LI not in sys.path:
    sys.path.insert(0, _LI)
from plot_zz_loopinduced import (                                # noqa: E402
    USETEX, check_minus, MINUS_FIX, ratio,
)
from plot_zz_nlo import (                                        # noqa: E402
    Data, MODES, NO_VIRTUALITY, draw as draw_modes, reference, chi2_flat,
)

NPZ = 'histograms_shapes.npz'
BINS_KEY = 'shapes_bins_4l'
PROBE = 'm_4l_fine'


def load(ddir):
    return Data(ddir, npz=NPZ, bins_key=BINS_KEY)


def structural_shapes(data, y, key, obs):
    """Bins of the NEW observables that are empty by construction.

    Two cases, both confined to the modes that draw no virtuality
    (``onshell`` / ``none`` / ``onshell_v1``):

    ``min_m_ll``
        both pair masses are exactly ``m_Z``, so their minimum is exactly
        ``m_Z`` and every other bin of the window is a structural zero -- the
        same statement the parent study makes about ``m_ee`` and ``m_mumu``,
        and it has to be made again because this is a different observable
        name and the imported helper keys off the name.

    ``m_4l_fine``
        below ``2 m_Z`` for the same reason the parent study gives for
        ``m_4l``: reaching there needs a drawn virtuality *and* recoil to
        absorb it, and a mode with no virtuality cannot get there at any
        statistics.  Marking those as measured zeros would put the whole
        sub-threshold region into a shape chi2 that is meant to be about the
        spectrum.
    """
    base = np.zeros_like(y, dtype=bool)
    if key not in NO_VIRTUALITY:
        return base
    edges = data.edges(obs)
    if obs in OS.PAIR_MASS_OBS_4L:
        return y == 0.0
    if obs == 'm_4l_fine':
        return (edges[1:] <= 2 * data.meta['m_Z']) & (y == 0.0)
    return base


def draw(d, obs, outdir, modes=MODES):
    return draw_modes(d, obs, outdir, modes=modes,
                      labels=OS.LABELS_SHAPES_4L,
                      labels_txt=OS.LABELS_SHAPES_4L_TXT,
                      logy=OS.LOGY_SHAPES_4L,
                      structural_fn=structural_shapes)


# --------------------------------------------------------------------------
def sub_threshold(d, key):
    """Fraction of ``key``'s cross section that the fine ``m_4l`` histogram
    puts below ``2 m_Z``.

    Recomputed off the histogram rather than read out of ``meta.json``: the
    parent study's number is a per-event count, this one is the integral of the
    curve the figure actually draws, and the two agreeing is the check that the
    binning -- in particular its lower edge -- carries the whole effect.  They
    would not agree on the parent's own m_4l grid, which starts at 150 GeV and
    cuts a third of the truth's sub-threshold rate off the left of the figure.
    """
    edges = d.edges('m_4l_fine')
    y, _ = d.density(key, 'm_4l_fine')
    w = np.diff(edges)
    below = edges[1:] <= 2 * d.meta['m_Z']
    return float((y * w)[below].sum() / d.sigma(key))


def write_numbers(d, path, modes=MODES):
    ref, is_truth = reference(d, PROBE)
    out = []
    A = out.append
    m = d.meta
    A('MadSpin spinmodes on the NEW (shape-study) four-lepton observables.')
    A('p p > z z [QCD] (MC@NLO) + MadSpin, against the off-shell NLO truth.')
    A('code %s   (samples unchanged; nothing was regenerated for this)'
      % m.get('code_sha', '?'))
    A('')
    if is_truth:
        A('reference: the directly generated off-shell four-lepton sample.')
    else:
        A('reference: spinmode = madspin -- NOT a validation.  See RESULTS.md.')
    A('')
    A('Every Z-level observable is reconstructed from the two lepton pairs and')
    A('is EXACT: the two Z are flavour tagged (one e+e-, one mu+mu-), so there')
    A('is no combinatorial ambiguity, and each observable is a function of the')
    A('pair four-momenta only.  The exception is the MASS: a reconstructed pair')
    A('is off shell where a produced z was not, so m_4l is not the production')
    A('m(ZZ) event by event -- which is the thing spinmode controls, not a')
    A('defect of the reconstruction.')
    A('')
    A('--- binned shape test: chi2/ndf of (mode / %s) against a flat line ---'
      % ref)
    A('(a pure normalisation offset does not enter; only a shape difference')
    A('does.  Structural zeros are excluded -- see plot_modes_shapes.py.)')
    A('')
    cols = [k for k in modes if d.has(k, PROBE) and k != ref]
    A('%-24s %s' % ('observable', ' '.join('%12s' % k for k in cols)))
    ranked = []
    for obs in OS.SHAPE_OBS_4L:
        yref, e0 = d.density(ref, obs)
        row = []
        worst = 0.0
        for key in cols:
            y, e = d.density(key, obs)
            struct = structural_shapes(d, y, key, obs)
            r, re_ = ratio(y, e, yref, e0)
            r = np.where(struct, np.nan, r)
            re_ = np.where(struct, np.nan, re_)
            # A mode with no virtuality has NO lineshape: its pair masses are
            # a delta function at m_Z, so "the chi2 of its shape" is undefined
            # rather than large, and saying so is the parent study's convention
            # for m_ee / m_mumu.  Reporting a number here would be reporting
            # how the delta happened to land in the binning.
            if obs in OS.PAIR_MASS_OBS_4L and key in NO_VIRTUALITY:
                row.append('%12s' % 'delta fn')
                continue
            c = chi2_flat(r, re_)
            row.append('%12s' % ('n/a' if c is None else '%.2f' % c))
            if c is not None:
                worst = max(worst, c)
        A('%-24s %s' % (obs, ' '.join(row)))
        ranked.append((worst, obs))
    A('')
    A('worst-mode chi2/ndf, ranked -- how strongly each new observable')
    A('separates the modes at all:')
    for w, obs in sorted(ranked, reverse=True):
        A('   %-24s %8.2f' % (obs, w))
    A('')
    A('--- m_4l below the on-shell threshold 2 m_Z = %.4f GeV ---'
      % (2 * m['m_Z']))
    A('The parent study\'s discriminating variable, re-measured on the 4 GeV')
    A('grid.  Integrals of the drawn curve, so they can be compared with the')
    A('per-event counts in numbers.txt directly.')
    A('')
    A('%-12s %18s %14s %20s' % ('sample', 'weight fraction', 'frac / truth',
                                 'per-event (meta)'))
    tref = sub_threshold(d, ref)
    for key in ([ref] if is_truth else []) + [k for k in modes
                                              if d.has(k, PROBE)]:
        f = sub_threshold(d, key)
        A('%-12s %18.4e %14s %20.4e'
          % (key, f, '--' if key == ref or not tref else '%.4f' % (f / tref),
             m['runs'][key]['weight_fraction_below_2mZ']))
    A('')
    A('The last two columns are the same quantity measured two ways -- the')
    A('integral of the drawn curve, and the per-event count in meta.json -- and')
    A('agreeing is what says the histogram\'s lower edge is low enough.')
    A('')
    A('--- min(m(e+e-), m(mu+mu-)): the virtuality handle ---')
    A('Exactly m_Z for every mode that draws no virtuality (onshell, none,')
    A('onshell_v1), so its whole window is a structural zero for those three --')
    A('the same statement the parent study makes about m_ee and m_mumu, made')
    A('again because the minimum of the two is a different observable and picks')
    A('the more off-shell leg of each event.')
    A('')
    A('%-12s %16s %16s' % ('sample', 'mean [GeV]', 'below m_Z frac'))
    for key in ([ref] if is_truth else []) + [k for k in modes
                                              if d.has(k, PROBE)]:
        y, _ = d.density(key, 'min_m_ll')
        e = d.edges('min_m_ll')
        w = np.diff(e)
        c = 0.5 * (e[:-1] + e[1:])
        tot = (y * w).sum()
        if tot <= 0:
            continue
        A('%-12s %16.4f %16.4f'
          % (key, float((y * w * c).sum() / tot),
             float((y * w)[c < m['m_Z']].sum() / tot)))
    open(path, 'w').write('\n'.join(out) + '\n')
    print('wrote %s' % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--check-minus', action='store_true')
    args = ap.parse_args()
    d = load(args.data)
    bases = [draw(d, obs, args.out) for obs in OS.SHAPE_OBS_4L]
    write_numbers(d, os.path.join(args.data, 'numbers_modes_shapes.txt'))
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
