#!/usr/bin/env python3
"""Production-level ``Z Z`` observables, for the stacked NLO + loop-induced figure.

The four-lepton observables of this study are **not** redefined here: they are
``MadSpin/validation/zz_loopinduced/observables.py``, imported unchanged by
:func:`leptonic`.  That module owns the four-vector helpers -- in particular
:func:`boost_to_rest`, whose spurious factor of gamma cost the previous study a
full set of wrong angular figures and which now self-tests on import -- and this
one borrows them rather than writing a second boost.

What is new here is the *production*-level side.  The physics figure stacks

  * ``p p > z z [QCD]``          the NLO contribution (``q q~``, ``q g``, ``g q~``)
  * ``g g > z z [noborn=QCD]``   the loop-induced ``g g`` contribution on top
  * ``p p > z z``                the LO curve the two ratio panes divide by

and all three have two ``z`` in the final state, so the observables are read off
the ``z`` directly.  No decay, no MadSpin: this is the production comparison,
and mixing it with the decayed samples would confuse the ``2 -> 2`` LO/LI
kinematics with the ``2 -> 3`` real emission of the NLO sample.

Two of the obvious observables are **degenerate by construction** on two of the
three samples and are recorded but not plotted as a stack:

``pt_zz``
    the transverse momentum of the ``Z Z`` system.  Exactly zero for every LO
    and every loop-induced event (both are ``2 -> 2`` with no initial-state
    transverse momentum); only the NLO sample has recoil.  A stack of a
    distribution against two delta functions is not a figure.

``dphi_zz``
    exactly ``pi`` for the same reason.

The ones that do separate are ``m_zz``, ``pt`` of the harder ``Z``, the ``Z Z``
rapidity and the production angle -- see RESULTS.md.
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LI = os.path.abspath(os.path.join(_HERE, '..', 'zz_loopinduced'))
if _LI not in sys.path:
    sys.path.insert(0, _LI)

# The loop-induced study's module, imported whole: the four-lepton observables,
# the binning, the labels, the LHE reader and the self-testing boost.
import observables as LEP                                        # noqa: E402

mass = LEP.mass
pt = LEP.pt
boost_to_rest = LEP.boost_to_rest
histogram = LEP.histogram

M_Z = LEP.M_Z
W_Z = LEP.W_Z


# --------------------------------------------------------------------------
# reading the two z of a production event
# --------------------------------------------------------------------------
def read_lhe_zz(path):
    """``(weights, z1, z2, extra)`` for a sample whose final state carries two ``z``.

    ``z1`` and ``z2`` are ``(N, 4)`` arrays in the LHE ordering ``(E, px, py,
    pz)``, in the order the ``z`` appear in the event record -- they are NOT
    pt-ordered here, because which one is harder is an observable and is decided
    downstream.  ``extra`` counts the additional final-state partons, which is 0
    for every LO and loop-induced event and 0 or 1 for the NLO ones.

    Events are required to carry exactly two status-1 ``z``.  Anything else --
    a decayed sample fed in by mistake, a ``z`` written as a resonance rather
    than a final state -- is a hard error, not something to skip: silently
    dropping events would renormalise the sample and the whole point of this
    figure is the absolute normalisation.
    """
    weights = []
    z1 = []
    z2 = []
    extra = []
    with LEP._open(path) as fh:
        inev = False
        wgt = None
        zs = []
        nx = 0
        for line in fh:
            s = line.strip()
            if s.startswith('<event'):
                inev, wgt, zs, nx = True, None, [], 0
                continue
            if not inev:
                continue
            if s.startswith('</event>'):
                inev = False
                if len(zs) != 2:
                    raise ValueError('%d status-1 z in one event of %s (expected 2)'
                                     % (len(zs), path))
                weights.append(wgt)
                z1.append(zs[0])
                z2.append(zs[1])
                extra.append(nx)
                continue
            if not s or s.startswith('<') or s.startswith('#'):
                continue
            f = s.split()
            if wgt is None:
                if len(f) >= 6:
                    try:
                        wgt = float(f[2])
                    except ValueError:
                        wgt = None
                continue
            if len(f) < 13:
                continue
            try:
                pdg, status = int(f[0]), int(f[1])
            except ValueError:
                continue
            if status != 1:
                continue
            if pdg == 23:
                zs.append((float(f[9]), float(f[6]), float(f[7]), float(f[8])))
            else:
                nx += 1
    return (np.array(weights, dtype=float), np.array(z1, dtype=float),
            np.array(z2, dtype=float), np.array(extra, dtype=int))


def read_lhe_initial_states(path):
    """``{(pdg1, pdg2): sum of weights}`` over the initial state of every event.

    This is the double-counting check of the stacked figure, done on the events
    that were actually written rather than on the process definition: the NLO
    sample must carry no ``g g`` initial state at all (``g g -> Z Z`` is a
    separate loop-induced contribution at ``O(alpha_s^2)`` relative to the Born,
    formally beyond NLO), and the loop-induced sample must carry nothing else.
    """
    out = {}
    with LEP._open(path) as fh:
        inev = False
        wgt = None
        init = []
        for line in fh:
            s = line.strip()
            if s.startswith('<event'):
                inev, wgt, init = True, None, []
                continue
            if not inev:
                continue
            if s.startswith('</event>'):
                inev = False
                if len(init) == 2:
                    key = tuple(sorted(init))
                    out[key] = out.get(key, 0.0) + (wgt or 0.0)
                continue
            if not s or s.startswith('<') or s.startswith('#'):
                continue
            f = s.split()
            if wgt is None:
                if len(f) >= 6:
                    try:
                        wgt = float(f[2])
                    except ValueError:
                        wgt = None
                continue
            if len(f) < 13:
                continue
            try:
                pdg, status = int(f[0]), int(f[1])
            except ValueError:
                continue
            if status == -1:
                init.append(pdg)
    return out


# --------------------------------------------------------------------------
# the observables
# --------------------------------------------------------------------------
def _rapidity(v):
    num = v[:, 0] + v[:, 3]
    den = v[:, 0] - v[:, 3]
    small = 1e-12
    return 0.5 * np.log(np.clip(num, small, None) / np.clip(den, small, None))


def compute_zz(z1, z2):
    """Dict of production-level arrays, from the two ``z`` four-vectors."""
    zz = z1 + z2
    p1, p2 = pt(z1), pt(z2)
    lead = np.where(p1 >= p2, p1, p2)
    sub = np.where(p1 >= p2, p2, p1)
    # the harder z as a four-vector, for the production angle
    hard = np.where((p1 >= p2)[:, None], z1, z2)

    # cos theta* : the harder z in the ZZ rest frame, against the ZZ direction
    # of flight in the lab.  For a 2 -> 2 sample that is the production angle in
    # the partonic centre of mass up to the (zero) transverse boost; for the NLO
    # sample with recoil it is the natural generalisation, and it is the same
    # definition on all three samples, which is what the stack needs.
    hard_in_zz = boost_to_rest(hard, zz)
    n_h = hard_in_zz[:, 1:4]
    n_zz = zz[:, 1:4].copy()
    # a ZZ system exactly at rest has no axis; use the beam axis there, which is
    # what the 2 -> 2 limit gives anyway.
    flat = np.linalg.norm(n_zz, axis=1) < 1e-9
    n_zz[flat] = np.array([0.0, 0.0, 1.0])
    cs = (np.sum(n_h * n_zz, axis=1)
          / np.clip(np.linalg.norm(n_h, axis=1), 1e-12, None)
          / np.clip(np.linalg.norm(n_zz, axis=1), 1e-12, None))

    dphi = np.arctan2(z1[:, 2], z1[:, 1]) - np.arctan2(z2[:, 2], z2[:, 1])
    return {
        'm_zz': mass(zz),
        'pt_z_lead': lead,
        'pt_z_sublead': sub,
        'pt_zz': pt(zz),
        'y_zz': _rapidity(zz),
        'abs_y_zz': np.abs(_rapidity(zz)),
        'abs_cos_theta_star': np.abs(np.clip(cs, -1.0, 1.0)),
        'dphi_zz': np.abs((dphi + math.pi) % (2 * math.pi) - math.pi),
    }


# --------------------------------------------------------------------------
# binning.  Fixed here so the harvester and the plotters cannot drift apart.
# --------------------------------------------------------------------------
# Widening with the observable, so that the far tail of a 50 000-event sample
# still carries enough weight for its ratio to mean something: at 10 GeV
# uniform binning the m(ZZ) ratio above 400 GeV is a 40 % scatter that says
# nothing about the K factor.  The first bin of m(ZZ) starts at the on-shell
# threshold 2 m_Z = 182.376 GeV, which every one of these three samples respects
# exactly -- they all have two on-shell z in the final state.
_PT_EDGES = np.concatenate([np.arange(0.0, 100.0, 5.0),
                            np.arange(100.0, 200.0, 10.0),
                            np.array([200.0, 225.0, 250.0, 300.0, 400.0,
                                      600.0])])
BINS_ZZ = {
    'm_zz': np.concatenate([np.arange(180.0, 400.0, 10.0),
                            np.arange(400.0, 600.0, 25.0),
                            np.array([600.0, 650.0, 700.0, 800.0, 900.0,
                                      1100.0, 1400.0])]),
    'pt_z_lead': _PT_EDGES,
    'pt_z_sublead': _PT_EDGES,
    'abs_y_zz': np.linspace(0.0, 4.0, 25),
    'abs_cos_theta_star': np.linspace(0.0, 1.0, 21),
    'pt_zz': np.concatenate([np.linspace(0.0, 100.0, 26),
                             np.array([120.0, 150.0, 200.0, 300.0, 500.0])]),
}

# The two observables that are a delta function on the LO and loop-induced
# samples: harvested, reported in numbers.txt, and deliberately kept out of the
# stacked figures.  See the module docstring.
DEGENERATE_ON_2TO2 = ('pt_zz',)

LABELS_ZZ = {
    'm_zz': (r'$m(ZZ)$ [GeV]',
             r'$\mathrm{d}\sigma/\mathrm{d}m(ZZ)$ [pb/GeV]'),
    'pt_z_lead': (r'$p_{T}$ of the harder $Z$ [GeV]',
                  r'$\mathrm{d}\sigma/\mathrm{d}p_{T}(Z_{\mathrm{lead}})$ [pb/GeV]'),
    'pt_z_sublead': (r'$p_{T}$ of the softer $Z$ [GeV]',
                     r'$\mathrm{d}\sigma/\mathrm{d}p_{T}(Z_{\mathrm{sublead}})$ [pb/GeV]'),
    'abs_y_zz': (r'$|y(ZZ)|$ of the $Z Z$ system',
                 r'$\mathrm{d}\sigma/\mathrm{d}|y(ZZ)|$ [pb]'),
    'abs_cos_theta_star': (r'$|\cos\theta^{*}|$ of the harder $Z$ in the $ZZ$ rest frame',
                           r'$\mathrm{d}\sigma/\mathrm{d}|\cos\theta^{*}|$ [pb]'),
    'pt_zz': (r'$p_{T}(ZZ)$ [GeV]',
              r'$\mathrm{d}\sigma/\mathrm{d}p_{T}(ZZ)$ [pb/GeV]'),
}

LABELS_ZZ_TXT = {
    'm_zz': 'm(ZZ) [GeV]',
    'pt_z_lead': 'pt of the harder Z [GeV]',
    'pt_z_sublead': 'pt of the softer Z [GeV]',
    'abs_y_zz': '|y(ZZ)| of the ZZ system',
    'abs_cos_theta_star': '|cos(theta*)| of the harder Z in the ZZ rest frame',
    'pt_zz': 'pt(ZZ) [GeV]',
}


def leptonic():
    """The four-lepton module of the loop-induced study, imported unchanged."""
    return LEP
