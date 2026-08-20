#!/usr/bin/env python3
"""Measure ``m_tt`` near the ``2 m_t`` threshold: MadSpin's spinmodes against a
doubly-resonant off-shell MadGraph truth, on ``p p > t t~ j``.

What this measures, and why the process is what it is
-----------------------------------------------------
MadSpin decays a production event whose kinematics are already fixed.  In the
off-shell spinmodes it draws a virtuality for each top and then calls
``Event.reshuffle_production``, which RAMBO-shuffles **every top-level
final-state momentum** -- for ``p p > t t~ j`` that is ``t``, ``t~`` *and the
recoiling jet* -- at fixed ``sqrt(shat)``.  Because the jet's momentum is
rescaled by the same RAMBO ``chi``, the ``t t~`` four-momentum is **not**
preserved and the reconstructed ``m_tt`` of the decayed event is **not** the
``m_tt`` of the production event it came from.  In the partonic CM, for a
massless recoil jet,

    m_tt^2(after)  = shat - 2 sqrt(shat) * chi * E_j
    m_tt^2(before) = shat - 2 sqrt(shat) *       E_j

so a mass set lighter than the pole gives ``chi > 1`` and pushes ``m_tt``
*down*, below ``2 m_t``.  That is a ``n >= 3`` effect only: for a ``2 -> 2``
production the two tops are the whole final state, ``m_tt = sqrt(shat)`` is
RAMBO-invariant and the sub-threshold region really is structurally empty.

``onshell`` samples no virtuality and never reshuffles
(``_density_do_reshuffle`` is False), so it keeps the production ``m_tt``
exactly and *is* structurally empty below threshold whatever the multiplicity.

The truth sample is ``p p > t t~ j, t > w+ b, t~ > w- b~`` generated directly by
MG5: the decay-chain matrix element carries the tops' Breit-Wigner propagators,
so the tops go off shell up to ``bwcutoff`` widths each and the sample populates
``m_tt < 2 m_t``, while keeping **only the doubly-resonant diagrams** -- exactly
the diagram set MadSpin also has.  The comparison therefore isolates MadSpin's
approximation of the *kinematics*, and does not conflate it with single- and
non-resonant contributions the framework never tried to reproduce.
``bwcutoff`` is matched between the truth run card and MadSpin's ``BW_cut``: it
sets how far below ``2 m_t`` the truth extends, so it is a parameter of the
figure, not an incidental setting.

The observable is reconstructed identically on both sides: the invariant mass of
the sum of the four status-1 particles with ``|pid|`` in ``{24, 5}``, i.e.
``(W+ b) + (W- b~)``.  The light jet is never a ``b`` (``j`` is defined without
one) and the initial state carries no ``b``, so that sum is unambiguous.

Usage
-----
    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 MadSpin/validation/mtt_threshold/run_mtt_threshold.py \
        --nevents-prod 1000000 --nevents-truth 3000000 \
        --outdir MadSpin/validation/mtt_threshold/data \
        --basedir /tmp/mtt_threshold_work

Stages can be run one at a time with ``--stage prod|truth|madspin|harvest``;
``--stage all`` (the default) does the lot.  Writes ``histograms.npz`` and
``meta.json`` into ``--outdir`` and copies every MadSpin log into
``<outdir>/logs/``.
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import gzip
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


# --------------------------------------------------------------------------
# The physics setup.  One dict, used verbatim by both the production sample and
# the truth sample, so "same model, PDF, scales and cuts" is a property of the
# code and not of a promise in a README.
#
# Fixed renormalisation and factorisation scales are deliberate.  The default
# dynamical scale is CKKW back-clustering *of the generated final state*, which
# is a different final state on the two sides (3 particles vs 5); a fixed scale
# removes that as a source of difference and leaves only the kinematics under
# study.  ``mu_R = mu_F = m_t`` is the conventional choice for this process.
#
# No cut is applied to the top decay products (``cut_decays`` is False by
# default, and ptb/etab are left at their inclusive defaults), so the truth
# sample's acceptance is that of the production sample: one light jet with
# pt > 20 GeV, |eta| < 5.
# --------------------------------------------------------------------------
MODEL = 'sm'
MULTIPARTICLES = {
    'p': 'g u c d s u~ c~ d~ s~',
    'j': 'g u c d s u~ c~ d~ s~',
}
PRODUCTION_PROCESS = 'p p > t t~ j'
DECAYS = ['t > w+ b', 't~ > w- b~']
TRUTH_PROCESS = 'p p > t t~ j, t > w+ b, t~ > w- b~'

BWCUTOFF = 15.0            # matched: run_card bwcutoff == MadSpin BW_cut

RUN_CARD = {
    'ebeam1': 6500, 'ebeam2': 6500,
    'ptj': 20, 'etaj': 5,
    'bwcutoff': BWCUTOFF,
    'fixed_ren_scale': 'True', 'fixed_fac_scale': 'True',
    'scale': 173.0, 'dsqrt_q2fact1': 173.0, 'dsqrt_q2fact2': 173.0,
    'use_syst': 'False',
}

SEED_PROD = 42
SEED_TRUTH = 4242
SEED_MADSPIN = 42

# The three spinmodes the brief asks for, and what each one does to the
# virtuality.  ``full`` is not a fourth mode: run_madspin rewrites it to
# ``madspin`` before anything looks at it.
MODES = [
    ('madspin', 'madspin'),   # off-shell density; virtuality from the ME
    ('PA', 'PA'),             # pole approximation; per-particle BW draw
    ('onshell', 'onshell'),   # no virtuality at all, no reshuffle
]

# A control, not a curve on the figure.
#
# ``unweighting = auto`` does NOT resolve the same way for the three modes:
# ``_auto_unweighting_mode`` sends PA/onshell to ``sequential`` at every
# multiplicity but madspin/full to ``joint`` for up to two decaying particles.
# With two decaying tops that is exactly this setup, so the shipped defaults
# compare madspin-under-joint against PA-under-sequential -- and the overweight
# safety net fires in the joint run and not in the other two.
#
# Every scheme is supposed to sample the same distribution, so this should not
# matter; "supposed to" is not a measurement.  This run repeats madspin with
# ``unweighting`` forced to ``sequential`` so the madspin-vs-PA difference can
# be attributed to the spinmode rather than to the accept/reject scheme.
CONTROLS = [
    ('madspin_seq', 'madspin', {'unweighting': 'sequential'}),
]

# Fine, uniform histogram grid.  0.25 GeV bins.  The truth's Breit-Wigner
# support reaches down to 2 * (m_t - 15 * Gamma_t) ~ 301 GeV, so 290 is below
# anything either side can produce; 520 is well above where the two agree.
# Rebinning is left to the plotting script so the committed .npz stays raw.
MTT_LO, MTT_HI, MTT_NBINS = 290.0, 520.0, 920

# The pole mass and width of the shipped ``sm`` param card.  Both are read back
# out of the production LHE banner at harvest time and the check is recorded in
# meta.json, so nothing downstream depends on these literals being right.
MT_POLE = 173.0
WT_POLE = 1.491500
TWO_MT = 2 * MT_POLE

# Grid for the event-by-event ``m_tt(decayed) - m_tt(production)`` measurement.
DELTA_LO, DELTA_HI, DELTA_NBINS = -40.0, 40.0, 800


def edges():
    return np.linspace(MTT_LO, MTT_HI, MTT_NBINS + 1)


def delta_edges():
    return np.linspace(DELTA_LO, DELTA_HI, DELTA_NBINS + 1)


# --------------------------------------------------------------------------
# Harvest
# --------------------------------------------------------------------------
def _open(path):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'rt')


def harvest(path, bins):
    """Histogram ``m_tt`` out of one LHE file.

    The LHE is read as text rather than through ``lhe_parser``: at millions of
    events the object-building parser dominates the wall time, and the only
    thing needed is four numbers off four particle lines.  ``--cross-check``
    validates this reader against ``lhe_parser`` on a slice of each file.

    Particle line layout is the LHE standard
        IDUP ISTUP MOTH1 MOTH2 ICOL1 ICOL2 PX PY PZ E M VTIM SPIN
    and the event header right after ``<event>`` is
        NUP IDPRUP XWGTUP SCALUP AQEDUP AQCDUP,
    so ``XWGTUP`` is field 2 of the header.  Weights are summed, not counted:
    the samples are unweighted but the sum of weights *is* the cross section
    (``event_norm = average`` gives every event ``sigma/N``), which is what the
    absolute normalisation of the figure needs.
    """
    nb = len(bins) - 1
    lo, hi = float(bins[0]), float(bins[-1])
    inv = nb / (hi - lo)                      # the grid is uniform
    sumw = np.zeros(nb)
    sumw2 = np.zeros(nb)
    cnt = np.zeros(nb, dtype=np.int64)
    out = dict(under_w=0.0, over_w=0.0, under_n=0, over_n=0,
               tot_w=0.0, tot_n=0, nfound_bad=0)

    inev = head = False
    w = 0.0
    px = py = pz = ee = 0.0
    nfound = 0
    with _open(path) as fp:
        for line in fp:
            s = line.strip()
            if not inev:
                if s.startswith('<event'):
                    inev, head = True, True
                continue
            if s.startswith('</event'):
                inev = False
                if nfound != 4:
                    out['nfound_bad'] += 1
                m2 = ee * ee - px * px - py * py - pz * pz
                m = math.sqrt(m2) if m2 > 0 else 0.0
                out['tot_w'] += w
                out['tot_n'] += 1
                if m < lo:
                    out['under_w'] += w
                    out['under_n'] += 1
                elif m >= hi:
                    out['over_w'] += w
                    out['over_n'] += 1
                else:
                    i = int((m - lo) * inv)
                    sumw[i] += w
                    sumw2[i] += w * w
                    cnt[i] += 1
                px = py = pz = ee = 0.0
                nfound = 0
                continue
            if head:
                parts = s.split()
                if len(parts) >= 6:
                    w = float(parts[2])
                    head = False
                continue
            if s.startswith('<'):
                continue
            parts = s.split()
            if len(parts) < 13:
                continue
            try:
                pid = int(parts[0])
                status = int(parts[1])
            except ValueError:
                continue
            if status != 1 or abs(pid) not in (24, 5):
                continue
            px += float(parts[6])
            py += float(parts[7])
            pz += float(parts[8])
            ee += float(parts[9])
            nfound += 1

    out['sumw'] = sumw
    out['sumw2'] = sumw2
    out['cnt'] = cnt
    return out


def _stream_mtt_shat(path, sel):
    """Yield ``(m_tt, sqrt(shat))`` per event, ``m_tt`` built from ``sel`` pids."""
    inev = head = False
    acc = tot = None
    with _open(path) as fp:
        for line in fp:
            s = line.strip()
            if not inev:
                if s.startswith('<event'):
                    inev, head = True, True
                    acc = [0.0] * 4
                    tot = [0.0] * 4
                continue
            if s.startswith('</event'):
                inev = False
                m2 = acc[3] ** 2 - acc[0] ** 2 - acc[1] ** 2 - acc[2] ** 2
                s2 = tot[3] ** 2 - tot[0] ** 2 - tot[1] ** 2 - tot[2] ** 2
                yield math.sqrt(max(0.0, m2)), math.sqrt(max(0.0, s2))
                continue
            if head:
                head = False
                continue
            if s.startswith('<'):
                continue
            p = s.split()
            if len(p) < 13:
                continue
            try:
                pid, status = int(p[0]), int(p[1])
            except ValueError:
                continue
            v = (float(p[6]), float(p[7]), float(p[8]), float(p[9]))
            if status == -1:
                for i in range(4):
                    tot[i] += v[i]
            elif status == 1 and abs(pid) in sel:
                for i in range(4):
                    acc[i] += v[i]


def pair_delta(prod_path, dec_path, dbins):
    """Event-by-event ``m_tt(decayed) - m_tt(production)`` for one mode.

    MadSpin writes exactly one decayed event per production event, in order, so
    the two files pair positionally.  The pairing is *verified* here rather than
    assumed: ``sqrt(shat)`` is RAMBO-invariant, so if the two streams are
    aligned the per-event difference in ``sqrt(shat)`` must be zero to rounding.
    The returned ``max_dshat`` is that check, and it is what makes the rest of
    the numbers mean anything.

    This is the mechanism the whole figure is about: for a ``2 -> 3``
    production the reshuffle rescales the recoil jet too, so ``m_tt`` moves.
    """
    hist = np.zeros(len(dbins) - 1, dtype=np.int64)
    lo, hi = float(dbins[0]), float(dbins[-1])
    inv = (len(dbins) - 1) / (hi - lo)
    n = 0
    s1 = s2 = 0.0
    max_abs = 0.0
    max_dshat = 0.0
    up_cross = down_cross = 0          # events that cross 2 m_t either way
    n_out = 0
    for (mp, sp), (md, sd) in zip(_stream_mtt_shat(prod_path, {6}),
                                  _stream_mtt_shat(dec_path, {24, 5})):
        d = md - mp
        n += 1
        s1 += d
        s2 += d * d
        if abs(d) > max_abs:
            max_abs = abs(d)
        if abs(sd - sp) > max_dshat:
            max_dshat = abs(sd - sp)
        if lo <= d < hi:
            hist[int((d - lo) * inv)] += 1
        else:
            n_out += 1
        if mp >= TWO_MT > md:
            down_cross += 1
        elif md >= TWO_MT > mp:
            up_cross += 1
    mean = s1 / n if n else 0.0
    rms = math.sqrt(max(0.0, s2 / n - mean ** 2)) if n else 0.0
    return dict(hist=hist, n=n, mean=mean, rms=rms, max_abs=max_abs,
                max_dshat=max_dshat, n_out=n_out,
                crossed_down=down_cross, crossed_up=up_cross)


def harvest_many(paths, bins):
    """Sum the histograms of several statistically independent LHE runs.

    MG5 refuses to make more than 1M events in one ``generate_events``
    (``madevent_interface.check_nb_events`` rewrites the run card and says so in
    the log), so more truth statistics can only come from more runs.  They are
    independent samples of the same process with the same cards and consecutive
    random seeds, so summing the weight histograms and dividing by the summed
    event count is the pooled estimator -- and, all runs having the same N, it is
    the plain average of their cross sections.
    """
    total = None
    per_run = []
    for path in paths:
        h = harvest(path, bins)
        per_run.append(dict(path=path, nevents=h['tot_n'],
                            sigma=h['tot_w'] / h['tot_n'] if h['tot_n'] else 0.0,
                            under_n=h['under_n'], over_n=h['over_n'],
                            malformed=h['nfound_bad']))
        if total is None:
            total = h
        else:
            for k in ('sumw', 'sumw2', 'cnt'):
                total[k] = total[k] + h[k]
            for k in ('under_w', 'over_w', 'under_n', 'over_n',
                      'tot_w', 'tot_n', 'nfound_bad'):
                total[k] = total[k] + h[k]
    return total, per_run


def cross_check(path, bins, nmax=2000):
    """Re-read the first ``nmax`` events with ``lhe_parser`` and compare.

    Returns ``(nchecked, max_abs_diff_in_GeV)``.  Anything above ~1e-9 means the
    text reader is picking up the wrong particles.
    """
    import madgraph.various.lhe_parser as lhe_parser
    from madgraph.various.lhe_parser import FourMomentum

    fast = []
    inev = head = False
    px = py = pz = ee = 0.0
    with _open(path) as fp:
        for line in fp:
            s = line.strip()
            if not inev:
                if s.startswith('<event'):
                    inev, head = True, True
                continue
            if s.startswith('</event'):
                inev = False
                m2 = ee * ee - px * px - py * py - pz * pz
                fast.append(math.sqrt(m2) if m2 > 0 else 0.0)
                px = py = pz = ee = 0.0
                if len(fast) >= nmax:
                    break
                continue
            if head:
                head = False
                continue
            if s.startswith('<'):
                continue
            parts = s.split()
            if len(parts) < 13:
                continue
            try:
                pid, status = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if status != 1 or abs(pid) not in (24, 5):
                continue
            px += float(parts[6]); py += float(parts[7])
            pz += float(parts[8]); ee += float(parts[9])

    slow = []
    lhe = lhe_parser.EventFile(path)
    for i, event in enumerate(lhe):
        if i >= len(fast):
            break
        tot = FourMomentum()
        for part in event:
            if part.status == 1 and abs(part.pid) in (24, 5):
                tot = tot + FourMomentum(part)
        slow.append(math.sqrt(max(0.0, tot.mass_sqr)))
    try:
        lhe.close()
    except Exception:
        pass
    n = min(len(fast), len(slow))
    diff = max(abs(a - b) for a, b in zip(fast[:n], slow[:n])) if n else 0.0
    return n, diff


# --------------------------------------------------------------------------
# Truth generation
# --------------------------------------------------------------------------
def write_truth_script(path, procdir, nevents, nb_core):
    lines = ['set automatic_html_opening False --no_save',
             'set nb_core %d' % nb_core,
             'import model %s' % MODEL]
    for name, definition in MULTIPARTICLES.items():
        lines.append('define %s = %s' % (name, definition))
    lines += ['generate %s' % TRUTH_PROCESS,
              'output %s' % procdir,
              'launch %s' % procdir,
              'madspin=OFF', 'shower=OFF', 'detector=OFF', 'analysis=OFF',
              'done',
              'set nevents %d' % nevents,
              'set iseed %d' % SEED_TRUTH]
    for key, val in RUN_CARD.items():
        lines.append('set %s %s' % (key, val))
    lines.append('done')
    with open(path, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')


def run_truth(basedir, nevents, nb_core):
    procdir = pjoin(basedir, 'TRUTH')
    lhe = pjoin(procdir, 'Events', 'run_01', 'unweighted_events.lhe.gz')
    if os.path.exists(lhe):
        print('truth: reusing %s' % lhe)
        return lhe
    script = pjoin(basedir, 'truth.mg5')
    write_truth_script(script, procdir, nevents, nb_core)
    log = pjoin(basedir, 'truth.log')
    print('truth: generating %d events (log %s)' % (nevents, log))
    sys.stdout.flush()
    t0 = time.time()
    with open(log, 'w') as fp:
        ret = subprocess.call([pjoin(_root, 'bin', 'mg5_aMC'), '-f', script],
                              stdout=fp, stderr=subprocess.STDOUT)
    print('truth: %.0f s, rc=%d' % (time.time() - t0, ret))
    if ret != 0 or not os.path.exists(lhe):
        raise RuntimeError('truth generation failed, see %s' % log)
    return lhe


def _banner_masses(path):
    """``MT``/``WT``/``MW``/``MB`` as they actually stand in the run's banner.

    ``MT_POLE`` and ``TWO_MT`` are literals in this file; this reads the values
    the runs were made with, so meta.json records the check rather than the
    assumption.
    """
    out = {}
    # The banner's param card writes the block entries with LOWER-CASE comment
    # names ('# mt', not '# MT'), and 'mt' sits right next to 'ymt', so the
    # match has to be exact and case-folded rather than a substring test.
    want = ('mt', 'wt', 'mw', 'mb', 'ww')
    with _open(path) as fp:
        for line in fp:
            if '<event' in line:
                break
            if '#' not in line:
                continue
            head, _, comment = line.partition('#')
            words = comment.split()
            if not words:
                continue
            name = words[0].strip().lower()
            if name not in want or name.upper() in out:
                continue
            fields = head.split()
            if len(fields) >= 2:
                try:
                    out[name.upper()] = float(fields[-1])
                except ValueError:
                    pass
    return out


def lhe_cross(path):
    import madgraph.various.lhe_parser as lhe_parser
    lhe = lhe_parser.EventFile(path)
    cross = lhe.get_banner().get_cross()
    try:
        lhe.close()
    except Exception:
        pass
    return cross


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nevents-prod', type=int, default=1000000)
    ap.add_argument('--nevents-truth', type=int, default=3000000)
    ap.add_argument('--outdir', default=pjoin(_here, 'data'))
    ap.add_argument('--basedir', default='/tmp/mtt_threshold_work')
    ap.add_argument('--nb-core', type=int, default=8)
    ap.add_argument('--stage', default='all',
                    choices=['all', 'prod', 'truth', 'madspin', 'harvest'])
    ap.add_argument('--cross-check', action='store_true')
    ap.add_argument('--modes', default=None,
                    help='comma-separated subset of the spinmodes to RUN in '
                         'this invocation.  The three modes are independent '
                         'runs off one shared production sample, each in its '
                         'own directory, so they can be launched as three '
                         'concurrent processes; the harvest stage picks up '
                         'whatever is on disk.')
    ap.add_argument('--truth-lhe', default=None,
                    help='comma-separated extra truth LHE files to add to the '
                         'truth histogram.  MG5 caps one generate_events at 1M '
                         'events (madevent_interface.check_nb_events), so more '
                         'truth statistics means more independent runs, and '
                         'they are summed here.')
    args = ap.parse_args()

    os.makedirs(args.basedir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    # MadSpinFactory only creates its base_dir when it makes up a temporary one.
    os.makedirs(pjoin(args.basedir, 'MS'), exist_ok=True)

    run_card = dict(RUN_CARD)
    factory = MadSpinFactory(
        'mtt_threshold',
        production_process=PRODUCTION_PROCESS,
        decays=DECAYS,
        model=MODEL,
        multiparticles=MULTIPARTICLES,
        nevents=args.nevents_prod,
        seed=SEED_PROD,
        extra_run_card=run_card,
        extra_madspin_settings={'nb_core': args.nb_core,
                                'BW_cut': int(BWCUTOFF)},
        base_dir=pjoin(args.basedir, 'MS'),
    )
    # ``set nb_core`` is an mg5 configuration command, not a run_card entry;
    # the factory only writes run_card ``set`` lines, so inject it by hand.
    _orig_write = factory._write_mg5_script

    def _write(script_path):
        _orig_write(script_path)
        with open(script_path) as fp:
            body = fp.read().splitlines()
        body.insert(1, 'set nb_core %d' % args.nb_core)
        with open(script_path, 'w') as fp:
            fp.write('\n'.join(body) + '\n')
    factory._write_mg5_script = _write

    # ``MadSpinFactory.produce_events`` only short-circuits on its own in-process
    # cache, so a later ``--stage madspin`` in a fresh interpreter would happily
    # regenerate a sample that is already on disk.  Adopt it instead.
    _existing = pjoin(factory.proc_dir, 'Events', 'run_01',
                      'unweighted_events.lhe.gz')
    if os.path.exists(_existing):
        factory.events_file = _existing
        factory.cross_in = lhe_cross(_existing)
        print('production: reusing %s (cross %s pb)'
              % (_existing, factory.cross_in))

    if args.stage in ('all', 'prod'):
        t0 = time.time()
        factory.produce_events()
        print('production: %.0f s -> %s (cross %s pb)'
              % (time.time() - t0, factory.events_file, factory.cross_in))
        sys.stdout.flush()

    truth_lhe = pjoin(args.basedir, 'TRUTH', 'Events', 'run_01',
                      'unweighted_events.lhe.gz')
    if args.stage in ('all', 'truth'):
        truth_lhe = run_truth(args.basedir, args.nevents_truth, args.nb_core)

    results = {}
    if args.stage in ('all', 'madspin'):
        wanted = ([m.strip() for m in args.modes.split(',')]
                  if args.modes else [m for m, _ in MODES])
        factory.produce_events()
        for label, spinmode, extra in CONTROLS:
            if label not in wanted:
                continue
            t0 = time.time()
            res = factory.run_mode(SpinModeConfig(label, spinmode),
                                   extra_settings=extra)
            results[label] = res
            print('%s: %.0f s  BR=%s eff=%s cross_out=%s'
                  % (label, time.time() - t0, res.BR, res.efficiency,
                     res.cross_out))
            sys.stdout.flush()
        for label, spinmode in MODES:
            if label not in wanted:
                continue
            t0 = time.time()
            res = factory.run_mode(SpinModeConfig(label, spinmode))
            results[label] = res
            print('%s: %.0f s  BR=%s eff=%s cross_out=%s'
                  % (label, time.time() - t0, res.BR, res.efficiency,
                     res.cross_out))
            sys.stdout.flush()

    if args.stage not in ('all', 'harvest'):
        return

    # ----------------------------------------------------------------- harvest
    factory.produce_events()
    bins = edges()
    logdir = pjoin(args.outdir, 'logs')
    os.makedirs(logdir, exist_ok=True)

    truth_files = [truth_lhe]
    if args.truth_lhe:
        truth_files += [t.strip() for t in args.truth_lhe.split(',') if t.strip()]
    files = [('truth', truth_files)]
    present_controls = []
    for label, _spin, _extra in CONTROLS:
        cand = pjoin(factory.base_dir, 'mode_%s' % label, 'events_decayed.lhe.gz')
        if not os.path.exists(cand):
            cand = cand[:-3]
        if os.path.exists(cand):
            present_controls.append(label)
    meta_controls = list(present_controls)
    for label, _ in MODES:
        run_dir = pjoin(factory.base_dir, 'mode_%s' % label)
        cand = pjoin(run_dir, 'events_decayed.lhe.gz')
        if not os.path.exists(cand):
            cand = cand[:-3]
        files.append((label, cand))
    for label in present_controls:
        cand = pjoin(factory.base_dir, 'mode_%s' % label, 'events_decayed.lhe.gz')
        if not os.path.exists(cand):
            cand = cand[:-3]
        files.append((label, cand))
    files.append(('production', factory.events_file))

    store = {'bins': bins}
    meta = {
        'code_sha': subprocess.check_output(
            ['git', '-C', _root, 'rev-parse', 'HEAD']).decode().strip(),
        'code_branch': subprocess.check_output(
            ['git', '-C', _root, 'rev-parse', '--abbrev-ref', 'HEAD']
        ).decode().strip(),
        'model': MODEL,
        'multiparticles': MULTIPARTICLES,
        'production_process': PRODUCTION_PROCESS,
        'truth_process': TRUTH_PROCESS,
        'madspin_decays': DECAYS,
        'run_card': run_card,
        'bwcutoff': BWCUTOFF,
        'madspin_BW_cut': int(BWCUTOFF),
        'seed_production': SEED_PROD,
        'seed_truth': SEED_TRUTH,
        'seed_madspin': SEED_MADSPIN,
        'nevents_prod_requested': args.nevents_prod,
        'nevents_truth_requested': args.nevents_truth,
        'nb_core': args.nb_core,
        'mtt_bins': {'lo': MTT_LO, 'hi': MTT_HI, 'n': MTT_NBINS},
        'observable': ('invariant mass of the sum of the four status-1 '
                       'particles with |pid| in {24, 5}, i.e. (W+ b) + (W- b~)'),
        'runs': {},
        'cross_check': {},
        'controls': meta_controls,
        'mt_pole_assumed': MT_POLE,
        'wt_pole_assumed': WT_POLE,
        'two_mt': TWO_MT,
    }
    meta['param_card_masses'] = _banner_masses(factory.events_file)

    for label, path in files:
        if label == 'truth':
            h, per_run = harvest_many(path, bins)
            meta['truth_runs'] = per_run
            path = path[0] if len(path) == 1 else path
        elif label == 'production':
            # The production sample has no W/b at all -- its ``m_tt`` is the
            # mass of the two status-1 tops.  Harvested with a separate pid
            # filter, purely so the figure can show that ``onshell`` reproduces
            # it bin for bin.
            h = _harvest_tops(path, bins)
        else:
            h = harvest(path, bins)
        store['%s_sumw' % label] = h['sumw']
        store['%s_sumw2' % label] = h['sumw2']
        store['%s_cnt' % label] = h['cnt']
        entry = dict(path=path, nevents=h['tot_n'], sumw=h['tot_w'],
                     under_n=h['under_n'], over_n=h['over_n'],
                     under_w=h['under_w'], over_w=h['over_w'],
                     malformed=h.get('nfound_bad', 0))
        try:
            entry['banner_cross_pb'] = lhe_cross(
                path[0] if isinstance(path, list) else path)
        except Exception as exc:
            entry['banner_cross_pb'] = None
            entry['banner_cross_error'] = str(exc)
        meta['runs'][label] = entry
        print('%-11s n=%d sum(w)=%.6g pb  banner=%s  under=%d over=%d'
              % (label, h['tot_n'], h['tot_w'], entry['banner_cross_pb'],
                 h['under_n'], h['over_n']))
        sys.stdout.flush()

        if args.cross_check and label != 'production':
            n, diff = cross_check(path[0] if isinstance(path, list) else path,
                                  bins)
            meta['cross_check'][label] = {'nevents': n, 'max_abs_diff_GeV': diff}
            print('  cross-check vs lhe_parser: %d events, max |diff| = %.3g GeV'
                  % (n, diff))

    # --- the mechanism, event by event -------------------------------------
    dbins = delta_edges()
    store['delta_bins'] = dbins
    meta['delta_mtt'] = {}
    for label in [m for m, _ in MODES] + present_controls:
        run_dir = pjoin(factory.base_dir, 'mode_%s' % label)
        dec = pjoin(run_dir, 'events_decayed.lhe.gz')
        if not os.path.exists(dec):
            dec = dec[:-3]
        if not os.path.exists(dec):
            continue
        t0 = time.time()
        d = pair_delta(factory.events_file, dec, dbins)
        store['delta_%s' % label] = d.pop('hist')
        meta['delta_mtt'][label] = d
        print('delta %-8s n=%d mean=%+.4f rms=%.4f max|d|=%.3f '
              'max|dsqrt(shat)|=%.3g crossed down=%d up=%d  (%.0f s)'
              % (label, d['n'], d['mean'], d['rms'], d['max_abs'],
                 d['max_dshat'], d['crossed_down'], d['crossed_up'],
                 time.time() - t0))
        sys.stdout.flush()

    for label in [m for m, _ in MODES] + present_controls:
        run_dir = pjoin(factory.base_dir, 'mode_%s' % label)
        log = pjoin(run_dir, 'madspin.log')
        if os.path.exists(log):
            # ``.log.txt``, not ``.log``: the repository's .gitignore has a
            # blanket ``*.log`` rule, so a copy named ``.log`` would silently
            # never be committed and the provenance would be missing from the
            # very place that promises it.
            shutil.copy(log, pjoin(logdir, 'madspin_%s.log.txt' % label))
            with open(log) as fp:
                text = fp.read()
            mode, why = _parse_unweighting(text)
            meta['runs'][label]['unweighting'] = mode
            meta['runs'][label]['unweighting_why'] = why
        card = pjoin(run_dir, 'madspin_card.dat')
        if os.path.exists(card):
            shutil.copy(card, pjoin(logdir, 'madspin_card_%s.dat' % label))
            meta['runs'][label]['madspin_card'] = open(card).read()

    for name, src in (('mg5_production.log.txt',
                       pjoin(factory.base_dir, 'mg5.log')),
                      ('mg5_truth.log.txt', pjoin(args.basedir, 'truth.log')),
                      ('mg5_production_script.dat',
                       pjoin(factory.base_dir, 'mg5_script.dat')),
                      ('mg5_truth_script.dat', pjoin(args.basedir, 'truth.mg5'))):
        if os.path.exists(src):
            shutil.copy(src, pjoin(logdir, name))

    np.savez_compressed(pjoin(args.outdir, 'histograms.npz'), **store)
    with open(pjoin(args.outdir, 'meta.json'), 'w') as fp:
        json.dump(meta, fp, indent=2, sort_keys=True)
    print('wrote %s and meta.json' % pjoin(args.outdir, 'histograms.npz'))


def _harvest_tops(path, bins):
    """``m_tt`` from the two status-1 tops of the *production* sample."""
    nb = len(bins) - 1
    lo, hi = float(bins[0]), float(bins[-1])
    inv = nb / (hi - lo)
    sumw = np.zeros(nb); sumw2 = np.zeros(nb)
    cnt = np.zeros(nb, dtype=np.int64)
    out = dict(under_w=0.0, over_w=0.0, under_n=0, over_n=0,
               tot_w=0.0, tot_n=0, nfound_bad=0)
    inev = head = False
    w = 0.0
    px = py = pz = ee = 0.0
    nfound = 0
    with _open(path) as fp:
        for line in fp:
            s = line.strip()
            if not inev:
                if s.startswith('<event'):
                    inev, head = True, True
                continue
            if s.startswith('</event'):
                inev = False
                if nfound != 2:
                    out['nfound_bad'] += 1
                m2 = ee * ee - px * px - py * py - pz * pz
                m = math.sqrt(m2) if m2 > 0 else 0.0
                out['tot_w'] += w; out['tot_n'] += 1
                if m < lo:
                    out['under_w'] += w; out['under_n'] += 1
                elif m >= hi:
                    out['over_w'] += w; out['over_n'] += 1
                else:
                    i = int((m - lo) * inv)
                    sumw[i] += w; sumw2[i] += w * w; cnt[i] += 1
                px = py = pz = ee = 0.0
                nfound = 0
                continue
            if head:
                parts = s.split()
                if len(parts) >= 6:
                    w = float(parts[2]); head = False
                continue
            if s.startswith('<'):
                continue
            parts = s.split()
            if len(parts) < 13:
                continue
            try:
                pid, status = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if status != 1 or abs(pid) != 6:
                continue
            px += float(parts[6]); py += float(parts[7])
            pz += float(parts[8]); ee += float(parts[9])
            nfound += 1
    out['sumw'] = sumw; out['sumw2'] = sumw2; out['cnt'] = cnt
    return out


if __name__ == '__main__':
    main()
