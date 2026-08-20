#!/usr/bin/env python3
"""Measure the ``m(t)`` / ``m(tbar)`` lineshape of every MadSpin spinmode x
unweighting cell on one shared production sample.

Process: ``p p > t t~``, both tops decayed (fully leptonic), ONE madevent
production sample and ONE MadSpin seed for every cell, so two cells differ only
by the spinmode / accept-reject scheme under test.

What is histogrammed is a PER-EVENT quantity: the invariant mass of the
intermediate resonance as it stands in the decayed LHE record, i.e.
``sqrt(E^2 - p^2)`` of the status-2 top (and separately of the status-2
anti-top).  That is the virtuality MadSpin actually assigned to the resonance,
and by momentum conservation it is the invariant mass of its decay products.
Nothing here is a fitted or derived parameter.

Usage:
    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 MadSpin/validation/mt_lineshape/run_lineshape.py \
        --nevents 200000 --outdir <data dir> --basedir <scratch tree>

Writes ``histograms.npz`` + ``meta.json`` into ``--outdir`` and copies each
run's ``madspin.log`` into ``<outdir>/logs/``.
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import collections
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

pjoin = os.path.join

_here = os.path.dirname(os.path.realpath(__file__))
_root = os.path.split(os.path.split(os.path.split(_here)[0])[0])[0]
if _root not in sys.path:
    sys.path.insert(0, _root)

from tests.parallel_tests.madspin_comparator import (  # noqa: E402
    MadSpinFactory, SpinModeConfig, _parse_unweighting)
import madgraph.various.lhe_parser as lhe_parser  # noqa: E402


# --------------------------------------------------------------------------
# The grid, and why each cell is or is not its own code path.
#
# ``spinmode`` families (interface_madspin._spinmode_has_density):
#   madspin / full  offshell density mode.  ``full`` is not a separate mode at
#                   all: run_madspin rewrites ``spinmode = full`` to
#                   ``'madspin'`` before anything else looks at it
#                   (interface_madspin.py, "if spinmode == 'full': spinmode =
#                   'madspin'"), so a ``full`` cell is the *same* run.  One
#                   ``full`` cell is kept purely to demonstrate that.
#   PA              pole approximation: per-particle Breit-Wigner draw plus a
#                   production reshuffle.
#   onshell         keeps the production kinematics and samples no virtuality:
#                   ``_density_do_reshuffle`` is False, ``slot_mass`` comes back
#                   empty.  m(t) is therefore the pole mass to machine precision
#                   in *every* onshell cell -- there is no lineshape to compare.
#                   Run anyway, and shown as the delta it is.
#   madspin_v1 /    no density matrix at all.  These never even reach
#   none            ``_unweighting_mode`` -- their logs carry no
#                   "MadSpin: unweighting = ..." line -- so the ``unweighting``
#                   card entry is inert and each has ONE real cell, not three.
#                   ``none`` moreover does not smear the resonance: m(t) comes
#                   out at the pole mass exactly, like onshell.
#                   ``madspin_v1`` does smear, so it has a real lineshape.
#
# ``unweighting``:
#   joint / sequential / sequential_global_retry are three real schemes in the
#   three density spinmodes.
#   sequential_with_mass needs a per-particle mass draw, i.e.
#   ``_density_pole_approximation()`` i.e. spinmode = PA; anywhere else
#   ``_unweighting_mode`` announces 'sequential' instead.  So it is a real cell
#   under PA only, and under madspin/onshell it is literally the sequential
#   cell.  One madspin cell is kept to demonstrate the fallback.
#
# Every claim above is checked at run time against the mode MadSpin itself
# announces in its log ("MadSpin: unweighting = <mode> (<why>)"), and the check
# is recorded in meta.json -- nothing here is asserted from reading the source
# alone.
#
# Entries are (key, spinmode, unweighting, extra ``set`` lines, role, seed).
#   role 'grid'      -- a real, distinct cell that goes on the plot
#   role 'degenerate'-- run to *prove* it collapses onto another cell
#   role 'extra'     -- a second axis (density_keep_jacobian), own figure
#   role 'replica'   -- the SAME cell as a grid entry, off the SAME production
#                       events, with a different MadSpin seed.  This is the
#                       noise floor: whatever chi2/dof two replicas of one
#                       scheme give each other is what "agreement" costs here,
#                       and no smaller difference between two schemes means
#                       anything.  ``seed`` is None -> the factory's seed.
# --------------------------------------------------------------------------
GRID = [
    ('madspin_joint',        'madspin', 'joint', {}, 'grid'),
    ('madspin_sequential',   'madspin', 'sequential', {}, 'grid'),
    ('madspin_seqglobal',    'madspin', 'sequential_global_retry', {}, 'grid'),
    ('madspin_seqwithmass',  'madspin', 'sequential_with_mass', {}, 'degenerate'),
    ('full_joint',           'full',    'joint', {}, 'degenerate'),

    ('PA_joint',             'PA', 'joint', {}, 'grid'),
    ('PA_sequential',        'PA', 'sequential', {}, 'grid'),
    ('PA_seqglobal',         'PA', 'sequential_global_retry', {}, 'grid'),
    ('PA_seqwithmass',       'PA', 'sequential_with_mass', {}, 'grid'),

    ('onshell_joint',        'onshell', 'joint', {}, 'grid'),
    ('onshell_sequential',   'onshell', 'sequential', {}, 'grid'),
    ('onshell_seqglobal',    'onshell', 'sequential_global_retry', {}, 'grid'),

    ('madspin_v1_joint',     'madspin_v1', 'joint', {}, 'grid'),
    ('madspin_v1_sequential', 'madspin_v1', 'sequential', {}, 'degenerate'),
    ('none_joint',           'none', 'joint', {}, 'grid'),
    ('none_sequential',      'none', 'sequential', {}, 'degenerate'),

    ('PAnojac_joint',        'PA', 'joint',
     {'density_keep_jacobian': 'False'}, 'extra'),
    ('PAnojac_sequential',   'PA', 'sequential',
     {'density_keep_jacobian': 'False'}, 'extra'),
    ('PAnojac_seqglobal',    'PA', 'sequential_global_retry',
     {'density_keep_jacobian': 'False'}, 'extra'),
    ('PAnojac_seqwithmass',  'PA', 'sequential_with_mass',
     {'density_keep_jacobian': 'False'}, 'extra'),

    ('madspin_joint_rep',    'madspin', 'joint', {}, 'replica'),
    ('PA_joint_rep',         'PA', 'joint', {}, 'replica'),

    # ``decay_output = weighted`` makes NO accept/reject at all: one decay
    # configuration is drawn per production event and kept with w propto W
    # (_validate_decay_output warns about exactly that).  Histogrammed with the
    # event weights -- which ``harvest`` already does -- it is therefore the
    # TARGET density itself, measured without any bound in the way.  That makes
    # it the reference an accept/reject scheme must reproduce, and the one
    # reference whose value cannot move when a bound changes.  It needs a
    # density spinmode; ``onshell`` has one but no virtuality to sample, so
    # only the two offshell/PA cells are worth running.
    ('madspin_weighted',     'madspin', 'joint',
     {'decay_output': 'weighted'}, 'weighted'),
    ('PA_weighted',          'PA', 'joint',
     {'decay_output': 'weighted'}, 'weighted'),
]

# MadSpin seed for the ``replica`` rows (the factory seed is used everywhere
# else).  Same production events, same scheme, different decay-side random
# stream.
REPLICA_SEED = 20260820

# Same physics setup as tests/parallel_tests/test_madspin_factory.TTBAR_LEPTONIC
# and as doc/madspin_unweighting_efficiency/measure_unweighting.py, so these
# runs are directly comparable with that campaign.
TTBAR_LEPTONIC = dict(
    production_process='p p > t t~',
    decays=['t > b w+, w+ > l+ vl',
            't~ > b~ w-, w- > l- vl~'],
    multiparticles={'p': 'g u d s c u~ d~ s~ c~',
                    'l+': 'e+ mu+', 'vl': 've vm',
                    'l-': 'e- mu-', 'vl~': 've~ vm~'},
    extra_run_card={'ebeam1': 6500, 'ebeam2': 6500},
)


# Threshold slicing of the m histogram (see ``harvest``).  beta is the top's
# velocity in the t tbar rest frame.  0.50 and 0.75 were chosen on a 2 000-event
# pilot to split this sample 20 / 44 / 36 per cent by weight, so the
# near-threshold slice is genuinely near threshold (beta < 0.5 is
# sqrt(s_hat) < 400 GeV, within 54 GeV of the 2M threshold) and still carries a
# fifth of the statistics.  Fixed here rather than per run so slices from
# different runs are comparable.
BETA_EDGES = np.array([0.0, 0.50, 0.75, 1.0])


def make_bins(pole, width, bw_cut, per_width=12):
    """A fine, UNIFORM grid over the whole Breit-Wigner support.

    ``bw_cut * width`` on each side of the pole is exactly what
    ``_draw_mass_value`` allows (``BW_cut = -1`` -> 15), so nothing a run can
    produce falls outside; the harvest counts under/overflow anyway and
    meta.json records it.

    The grid is deliberately finer than any binning worth plotting
    (Gamma/12 = 0.124 GeV, 360 bins).  Rebinning is left to the plotting
    script, which groups whole numbers of these into a coarse-in-the-tails
    scheme -- so the committed .npz stays the raw measurement and the choice of
    binning can be revisited without re-running MadSpin.
    """
    n = int(round(2 * bw_cut * per_width))
    return np.linspace(pole - bw_cut * width, pole + bw_cut * width, n + 1)


def harvest(lhe_path, edges):
    """Histogram m(t) and m(tbar) out of one decayed LHE.

    The LHE is read as text rather than through ``lhe_parser``: at 2e5 events
    x 20 runs the object-building parser dominates the wall time, and the only
    thing needed here is four numbers off the particle lines.  The two readers
    are checked against each other by ``--cross-check`` (and were, on the
    pilot: identical histograms).

    Particle line layout is the LHE standard:
        IDUP ISTUP MOTH1 MOTH2 ICOL1 ICOL2 PX PY PZ E M VTIM SPIN
    and the event header right after ``<event>`` is
        NUP IDPRUP XWGTUP SCALUP AQEDUP AQCDUP.

    Returns per resonance the weighted histogram, its squared weights, the
    out-of-range counts, and enough raw moments to quote mean/rms without
    re-reading the file.

    THRESHOLD SLICING.  The same ``m`` histogram is filled a second time, split
    by how close the event is to the ``t tbar`` production threshold.  The
    slicing variable is the top's velocity in the ``t tbar`` rest frame,

        beta = |p*| / E*   with  s_hat = (p_t + p_tbar)^2,

    computed from the two status-2 four-vectors that are already being read --
    no extra pass, no assumption about the production process beyond there
    being exactly two resonances.  It is worth having because the mass stage's
    per-event bound is placed at the low corner of the Breit-Wigner windows and
    the RAMBO reshuffling jacobian it bounds is a ratio of ``lambda^(1/2)``
    factors, which is where the ``1/beta`` behaviour lives: the low-beta slice
    is where the bound and the weight are furthest apart, and therefore where a
    bound that failed to dominate would show first.  ``BETA_EDGES`` are fixed
    once here so slices from different runs are directly comparable.
    """
    import gzip
    nb = len(edges) - 1
    lo, hi = float(edges[0]), float(edges[-1])
    inv = nb / (hi - lo)                       # the grid is uniform
    nbeta = len(BETA_EDGES) - 1
    acc = {}
    for tag in ('t', 'tbar'):
        acc[tag] = dict(sumw=np.zeros(nb), sumw2=np.zeros(nb),
                        n=np.zeros(nb, dtype=np.int64),
                        sumw_beta=np.zeros((nbeta, nb)),
                        sumw2_beta=np.zeros((nbeta, nb)),
                        n_beta=np.zeros((nbeta, nb), dtype=np.int64),
                        under_w=0.0, over_w=0.0, under_n=0, over_n=0,
                        s0=0.0, s1=0.0, s2=0.0, sq0=0.0,
                        mmin=float('inf'), mmax=float('-inf'),
                        max_dev_record=0.0)
    beta_hist = np.zeros(nbeta)
    nev = 0
    sumw_tot = 0.0
    no_beta = 0                    # events without exactly two status-2 tops

    def flush(pending, wgt):
        """Fill both histograms for one event, once both tops are known."""
        beta = None
        if len(pending) == 2:
            e = px = py = pz = 0.0
            for _tag, _m, p4 in pending:
                px += p4[0]
                py += p4[1]
                pz += p4[2]
                e += p4[3]
            shat = e * e - px * px - py * py - pz * pz
            if shat > 0.0:
                # |p*|/E* of either top in the pair rest frame: the invariant
                # form needs no boost.  m1 = m2 is not assumed.
                m1, m2 = pending[0][1], pending[1][1]
                lam = ((shat - m1 * m1 - m2 * m2) ** 2
                       - 4.0 * m1 * m1 * m2 * m2)
                if lam > 0.0:
                    # E* of top 1 is (shat + m1^2 - m2^2)/(2 sqrt(shat))
                    estar = (shat + m1 * m1 - m2 * m2) / (2.0 * math.sqrt(shat))
                    pstar = math.sqrt(lam) / (2.0 * math.sqrt(shat))
                    if estar > 0.0:
                        beta = pstar / estar
        kb = -1
        if beta is not None:
            kb = int(np.searchsorted(BETA_EDGES, beta, side='right')) - 1
            kb = min(max(kb, 0), nbeta - 1)
            beta_hist[kb] += wgt
        for tag, m, _p4 in pending:
            a = acc[tag]
            a['s0'] += wgt
            a['s1'] += wgt * m
            a['s2'] += wgt * m * m
            a['sq0'] += wgt * wgt
            if m < a['mmin']:
                a['mmin'] = m
            if m > a['mmax']:
                a['mmax'] = m
            if m < lo:
                a['under_w'] += wgt
                a['under_n'] += 1
            elif m >= hi:
                a['over_w'] += wgt
                a['over_n'] += 1
            else:
                k = int((m - lo) * inv)
                if k >= nb:
                    k = nb - 1
                a['sumw'][k] += wgt
                a['sumw2'][k] += wgt * wgt
                a['n'][k] += 1
                if kb >= 0:
                    a['sumw_beta'][kb, k] += wgt
                    a['sumw2_beta'][kb, k] += wgt * wgt
                    a['n_beta'][kb, k] += 1
        return 0 if beta is not None else 1

    opener = gzip.open if lhe_path.endswith('.gz') else open
    with opener(lhe_path, 'rt') as fh:
        in_event = False
        need_header = False
        wgt = 0.0
        pending = []
        for line in fh:
            if line.startswith('<event'):
                in_event = True
                need_header = True
                pending = []
                continue
            if not in_event:
                continue
            if line.startswith('</event'):
                in_event = False
                if pending:
                    no_beta += flush(pending, wgt)
                    pending = []
                continue
            stripped = line.strip()
            if not stripped or stripped[0] in '<#':
                continue
            fields = stripped.split()
            if need_header:
                # NUP IDPRUP XWGTUP ...
                try:
                    wgt = float(fields[2])
                except (IndexError, ValueError):
                    continue
                need_header = False
                nev += 1
                sumw_tot += wgt
                continue
            if len(fields) < 11:
                continue
            if fields[1] != '2':                       # ISTUP, intermediate
                continue
            pdg = fields[0]
            if pdg == '6':
                tag = 't'
            elif pdg == '-6':
                tag = 'tbar'
            else:
                continue
            px, py, pz, en = (float(fields[6]), float(fields[7]),
                              float(fields[8]), float(fields[9]))
            m2 = en * en - px * px - py * py - pz * pz
            m = math.sqrt(m2) if m2 > 0.0 else 0.0
            # how far the record's own mass column sits from the 4-vector mass:
            # a bookkeeping cross-check, never the observable
            a = acc[tag]
            a['max_dev_record'] = max(a['max_dev_record'],
                                      abs(m - float(fields[10])))
            pending.append((tag, m, (px, py, pz, en)))
        if pending:                      # a file whose last </event> is absent
            no_beta += flush(pending, wgt)
    for a in acc.values():
        if a['mmin'] == float('inf'):
            a['mmin'] = a['mmax'] = float('nan')
    return acc, nev, sumw_tot, beta_hist, no_beta


def harvest_reference(lhe_path, edges):
    """The same histogram through ``lhe_parser``.  Slow; used only to check
    that the fast text reader above agrees with the repository's own parser."""
    nb = len(edges) - 1
    out = {t: np.zeros(nb) for t in ('t', 'tbar')}
    nev = 0
    lhe = lhe_parser.EventFile(lhe_path)
    for event in lhe:
        nev += 1
        wgt = float(event.wgt)
        for particle in event:
            if int(particle.status) != 2 or abs(particle.pdg) != 6:
                continue
            tag = 't' if particle.pdg == 6 else 'tbar'
            m2 = (particle.E ** 2 - particle.px ** 2
                  - particle.py ** 2 - particle.pz ** 2)
            m = math.sqrt(max(m2, 0.0))
            if edges[0] <= m < edges[-1]:
                out[tag][int(np.searchsorted(edges, m, side='right') - 1)] += wgt
    try:
        lhe.close()
    except Exception:
        pass
    return out, nev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nevents', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--basedir', default=None)
    parser.add_argument('--nb-core', type=int, default=8)
    parser.add_argument('--only', default=None)
    parser.add_argument('--roles', default='grid,degenerate,extra')
    parser.add_argument('--cross-check', action='store_true',
                        help='also histogram the first run through lhe_parser '
                             'and assert the fast text reader agrees')
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(pjoin(outdir, 'logs'), exist_ok=True)
    if args.basedir:
        os.makedirs(args.basedir, exist_ok=True)

    roles = set(args.roles.split(','))
    grid = [g for g in GRID if g[4] in roles]
    if args.only:
        keep = set(args.only.split(','))
        grid = [g for g in grid if g[0] in keep]

    factory = MadSpinFactory(
        name='mt_lineshape',
        nevents=args.nevents,
        seed=args.seed,
        base_dir=args.basedir,
        extra_madspin_settings={'nb_core': args.nb_core},
        **TTBAR_LEPTONIC)

    t0 = time.time()
    events_file = factory.produce_events()
    production_wall = time.time() - t0

    banner = lhe_parser.EventFile(events_file).get_banner()
    pole = float(banner.get('param', 'mass', 6).value)
    width = float(banner.get('param', 'decay', 6).value)
    bw_cut = 15.0            # MadSpin's BW_cut = -1 default, see _draw_mass_value
    edges = make_bins(pole, width, bw_cut)
    print('pole = %.6f  width = %.6f  bins = %d  range = [%.3f, %.3f]'
          % (pole, width, len(edges) - 1, edges[0], edges[-1]), flush=True)

    payload = {}
    runs = collections.OrderedDict()
    for label, spinmode, unw, extra, role in grid:
        config = SpinModeConfig(label, spinmode)
        settings = {'unweighting': unw}
        settings.update(extra)
        seed = REPLICA_SEED if role == 'replica' else None
        print('=== %s (%s / %s %s seed=%s)'
              % (label, spinmode, unw, extra or '', seed or args.seed),
              flush=True)
        start = time.time()
        res = factory.run_mode(config, extra_settings=settings, seed=seed)
        wall = time.time() - start
        with open(res.log_path) as fp:
            log_text = fp.read()
        shutil.copy(res.log_path, pjoin(outdir, 'logs', '%s.log' % label))
        acc, nev, sumw_tot, beta_hist, no_beta = harvest(res.lhe_path, edges)
        if args.cross_check:
            ref, refn = harvest_reference(res.lhe_path, edges)
            assert refn == nev, (refn, nev)
            for tag in ('t', 'tbar'):
                assert np.allclose(ref[tag], acc[tag]['sumw'], rtol=1e-12), tag
            print('    cross-check vs lhe_parser: identical', flush=True)

        mode, why = _parse_unweighting(log_text)
        entry = dict(
            spinmode=spinmode, unweighting_asked=unw, extra_settings=dict(extra),
            role=role, reported_mode=mode, reported_why=why,
            nevents=nev, sumw_total=sumw_tot, wall_seconds=wall,
            madspin_seed=res.seed,
            BR=res.BR, cross_out=res.cross_out, cross_in=res.cross_in,
            efficiency=res.efficiency, overflows=res.overflows,
            lhe_path=res.lhe_path,
            spinmode_note=[l.strip() for l in log_text.splitlines()
                           if 'keeps the joint accept/reject' in l
                           or 'using sequential instead' in l
                           or 'Running MadSpin in spinmode' in l],
        )
        entry['beta_slices'] = dict(
            edges=[float(x) for x in BETA_EDGES],
            sumw=[float(x) for x in beta_hist],
            events_without_beta=no_beta)
        for tag, a in acc.items():
            payload['sumw__%s__%s' % (label, tag)] = a['sumw']
            payload['sumw2__%s__%s' % (label, tag)] = a['sumw2']
            payload['n__%s__%s' % (label, tag)] = a['n']
            payload['sumwb__%s__%s' % (label, tag)] = a['sumw_beta']
            payload['sumw2b__%s__%s' % (label, tag)] = a['sumw2_beta']
            payload['nb__%s__%s' % (label, tag)] = a['n_beta']
            entry['%s_moments' % tag] = dict(
                s0=a['s0'], s1=a['s1'], s2=a['s2'], sq0=a['sq0'],
                under_w=a['under_w'], over_w=a['over_w'],
                under_n=a['under_n'], over_n=a['over_n'],
                mmin=a['mmin'], mmax=a['mmax'],
                max_dev_record=a['max_dev_record'])
        runs[label] = entry
        print('    -> mode=%s (%s) nev=%d  m(t) in [%.4f, %.4f]  '
              'out-of-range %d/%d  [%.0f s]'
              % (mode, why, nev, acc['t']['mmin'], acc['t']['mmax'],
                 acc['t']['under_n'], acc['t']['over_n'], wall), flush=True)

    payload['bins'] = edges
    payload['beta_edges'] = BETA_EDGES
    np.savez_compressed(pjoin(outdir, 'histograms.npz'), **payload)

    sha = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=_root).decode().strip()
    meta = dict(
        process='p p > t t~', decays=TTBAR_LEPTONIC['decays'],
        multiparticles=TTBAR_LEPTONIC['multiparticles'],
        run_card_overrides=TTBAR_LEPTONIC['extra_run_card'],
        nevents_requested=args.nevents, seed=args.seed, nb_core=args.nb_core,
        code_sha=sha, base_dir=factory.base_dir, events_file=events_file,
        cross_in=getattr(factory, 'cross_in', None),
        production_wall_seconds=production_wall,
        observable=('invariant mass sqrt(E^2-p^2) of the status-2 resonance '
                    'in the decayed LHE record (per event)'),
        pole=pole, width=width, bw_cut=bw_cut,
        bw_range=[pole - bw_cut * width, pole + bw_cut * width],
        bins=[float(x) for x in edges],
        beta_edges=[float(x) for x in BETA_EDGES],
        runs=runs)
    with open(pjoin(outdir, 'meta.json'), 'w') as fp:
        json.dump(meta, fp, indent=2, sort_keys=False)
    print('wrote %s' % pjoin(outdir, 'histograms.npz'))
    print('base_dir = %s' % factory.base_dir)


if __name__ == '__main__':
    main()
