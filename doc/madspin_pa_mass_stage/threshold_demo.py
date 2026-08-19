#!/usr/bin/env python3
"""Why the PA reshuffling jacobian has a tail: a deterministic scan, no MC.

Take one real ``p p > t t~`` production event, put it back-to-back at a chosen
sqrt(shat), give both tops the *same* fixed off-shell mass, and ask the shipped
``Event.reshuffle_production`` for the jacobian.  Nothing is sampled, so what
comes out is the jacobian as a function of two numbers only: how far the
production is above the tt~ threshold, and how far the drawn mass set is below
the pole.

The RAMBO map (``Event.mass_shuffle``) rescales all spatial momenta by a common
chi fixed by ``sum_i sqrt(m_i^2 + chi^2 |p_i|^2) = sqrt(shat)`` and returns

    J = chi^(3n-3) * prod_i (E_i/E'_i) * (sum_i |p_i|^2/E_i) / (sum_i |p'_i|^2/E'_i)

At threshold the on-shell |p_i| goes to zero while the off-shell configuration
still needs a finite |p'_i|, so chi ~ |p'|/|p| diverges and with it J.  That is
the whole story of the tail: it is not a numerical accident, it is the ratio of
an open off-shell phase space to a closed on-shell one.

Usage::

    export PATH="$HOME/.pyenv/versions/mg-3.12/bin:$PATH"
    python3 doc/madspin_pa_mass_stage/threshold_demo.py \
        --events <production .lhe.gz> --out <dir>
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import json
import math
import os
import sys

import numpy as np

pjoin = os.path.join

_here = os.path.dirname(os.path.realpath(__file__))
_root = os.path.split(os.path.split(_here)[0])[0]
if _root not in sys.path:
    sys.path.insert(0, _root)

import madgraph.various.lhe_parser as lhe_parser  # noqa: E402
from MadSpin.interface_madspin import MadSpinInterface  # noqa: E402


def first_event(path):
    lhe = lhe_parser.EventFile(path)
    banner = lhe.get_banner()
    event = next(iter(lhe))
    try:
        lhe.close()
    except Exception:
        pass
    return event, banner


def shape(event, sqrts, mt, costheta=0.0):
    """Rewrite the event's momenta as a back-to-back ``g g > t t~`` at
    ``sqrts``, tops on shell at ``mt``, scattering angle ``costheta``."""
    probe = lhe_parser.Event(str(event))
    initial = [p for p in probe if int(p.status) == -1]
    finals = [p for p in probe if int(p.status) == 1]
    assert len(initial) == 2 and len(finals) == 2
    half = sqrts / 2.0
    for particle, sign in zip(initial, (1.0, -1.0)):
        particle.E, particle.px, particle.py, particle.pz = half, 0., 0., sign * half
        particle.mass = 0.
    pmag = math.sqrt(max(0.0, sqrts ** 2 / 4.0 - mt ** 2))
    sintheta = math.sqrt(max(0.0, 1.0 - costheta ** 2))
    for particle, sign in zip(finals, (1.0, -1.0)):
        particle.E = math.sqrt(pmag ** 2 + mt ** 2)
        particle.px = sign * pmag * sintheta
        particle.py = 0.
        particle.pz = sign * pmag * costheta
        particle.mass = mt
    return probe, pmag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    event, banner = first_event(args.events)
    mt = float(banner.get('param', 'mass', 6).value)
    width = float(banner.get('param', 'decay', 6).value)
    bw_cut = float(banner.get_detail('run_card', 'bwcutoff'))
    print('m_t = %s GeV, Gamma_t = %s GeV, BW_cut = %s (so the drawn mass runs '
          'over %.1f .. %.1f GeV)'
          % (mt, width, bw_cut, mt - bw_cut * width, mt + bw_cut * width))

    # the mass sets to scan: both tops at the same virtuality, from the bottom
    # of the Breit-Wigner window to the top of it
    offsets = [-bw_cut * width, -10 * width, -5 * width, 0.0, +5 * width,
               +bw_cut * width]
    sqrts_grid = np.concatenate([
        np.linspace(2 * mt + 0.05, 2 * mt + 5, 60),
        np.linspace(2 * mt + 5, 2 * mt + 60, 60),
        np.geomspace(2 * mt + 60, 3000, 80)])

    curves = {}
    for offset in offsets:
        mass = mt + offset
        xs, js, betas = [], [], []
        for sqrts in sqrts_grid:
            if 2 * mass >= sqrts:
                continue        # the mass set does not fit: J = -1 by design
            probe, pmag = shape(event, float(sqrts), mt)
            jac = MadSpinInterface._production_jacobian_for(
                probe, [0, 1], {0: (mass, None), 1: (mass, None)})
            if jac in (0, -1):
                continue
            xs.append(float(sqrts))
            js.append(float(jac))
            betas.append(pmag / (sqrts / 2.0))
        curves['%+.1f' % offset] = {'sqrts': xs, 'jac': js, 'beta': betas,
                                    'mass': mass}
        if xs:
            print('  both tops at %7.2f GeV : J = %8.3f at sqrt(s) = %.1f, '
                  '%7.4f at 1 TeV'
                  % (mass, js[0], xs[0],
                     js[min(range(len(xs)), key=lambda i: abs(xs[i] - 1000))]))

    with open(pjoin(args.out, 'threshold_demo.json'), 'w') as fp:
        json.dump({'mt': mt, 'width': width, 'bw_cut': bw_cut,
                   'curves': curves}, fp, indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap('coolwarm')
    keys = sorted(curves, key=lambda k: float(k))
    for i, key in enumerate(keys):
        curve = curves[key]
        if not curve['sqrts']:
            continue
        color = cmap(i / max(1, len(keys) - 1))
        axes[0].plot(curve['sqrts'], curve['jac'], color=color, lw=1.8,
                     label=r'$m_t^{\rm off} = %.1f$ GeV' % curve['mass'])
        axes[1].plot(curve['beta'], curve['jac'], color=color, lw=1.8)
    axes[0].axhline(1.0, color='k', lw=.8, ls=':')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'production $\sqrt{\hat s}$  [GeV]')
    axes[0].set_ylabel(r'reshuffling jacobian  $J$')
    axes[0].set_title(r'$J$ against $\sqrt{\hat s}$ '
                      '(threshold $2m_t = %.0f$ GeV)' % (2 * mt))
    axes[0].legend(fontsize=8)

    beta = np.geomspace(1.2e-2, 1, 50)
    axes[1].plot(beta, 0.5 / beta, 'k--', lw=1,
                 label=r'$0.5/\beta_t$')
    axes[1].axhline(1.0, color='k', lw=.8, ls=':')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel(r'on-shell top velocity  $\beta_t$ in the partonic CM')
    axes[1].set_title(r'the same against $\beta_t$: $J \sim 1/\beta_t$ at '
                      'threshold')
    axes[1].legend(fontsize=8)
    fig.suptitle('The PA production reshuffling jacobian is a phase-space '
                 'volume ratio, and it diverges at the $t\\bar t$ threshold')
    fig.tight_layout()
    fig.savefig(pjoin(args.out, 'threshold_demo.png'), dpi=140)
    print('wrote %s' % pjoin(args.out, 'threshold_demo.png'))


if __name__ == '__main__':
    main()
