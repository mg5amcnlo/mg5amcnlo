#!/usr/bin/env python3
"""``m_tt`` near the ``2 m_t`` threshold, one unweighting scheme against another
*within* a fixed off-shell spinmode.

This is the companion measurement to ``run_mtt_threshold.py``.  That one varied
the **spinmode** and let ``unweighting = auto`` pick a scheme for each; this one
holds the spinmode fixed and varies the **accept/reject scheme**, on the same
production sample and against the same truth.

Why the schemes should be identical, and where the one exception lives
---------------------------------------------------------------------
``Z_k(m)`` is the tabulated look-ahead factor that the up-front-mass schemes
multiply into the mass-set weight -- the fraction of the decay pool that can
reach virtuality ``m``, measured during the max-weight scan.  The three schemes
run here stand in three different relations to it:

``joint``
    one accept/reject over the virtualities and both decays at once.  There is
    no separate mass stage, so ``Z_k`` never enters.

``sequential_global_retry``
    a rejected decay throws away the mass set and restarts the chain, so the
    per-angle stage never renormalises and ``Z_k`` **cancels identically**
    (``interface_madspin.py``: "sequential_global_retry ... stops the per-angle
    stage normalising at all, and then Z_hat cancels identically and only sets
    the efficiency").  It is a pure efficiency preconditioner there.

``sequential``
    trusts the tabulated ``Z_k``.  The per-angle stage divides out the *true*
    ``Z_k`` whatever weight it was given, so the residual bias of this scheme is
    exactly ``Z_hat / Z`` -- quoted elsewhere as a sub-per-cent effect.

So the null hypothesis is sharp: **``joint`` and ``sequential_global_retry``
must agree to within statistics**, because neither of them reads the table.  If
``sequential`` departs from both, the departure is the ``Z_hat/Z`` residual.

``sequential_with_mass`` is a fourth point, and not a variant of the other
three: it draws each slot's virtuality *inside* that slot's own accept/reject,
so no stage freezes a mass and ``Z_k`` does not arise at all.  It needs a
per-particle mass draw, i.e. the ``PA`` spinmode; under ``madspin`` it silently
falls back to ``sequential`` (``_unweighting_mode``), so it is only a cell of
the ``PA`` row.  The resolved scheme is parsed back out of every log and
recorded, so a silent fallback cannot masquerade as a measurement.

``onshell`` is not run.  It draws no virtuality, ``_spinmode_has_density`` is
False for it and ``_unweighting_mode`` therefore forces ``joint`` whatever the
card says: there is no scheme axis to scan.

What this measures, on top of ``m_tt``
--------------------------------------
``Z_k`` is a distortion of the **virtuality** distribution, and ``m_tt`` sees it
only after the production reshuffle has smeared it.  The reconstructed top
virtuality ``m(W^+ b)`` is the direct observable, so it is harvested in the same
pass: same events, no extra cost, and its first two moments carry a statistical
error four orders of magnitude below the sub-threshold rate's.  It is the test
that can actually resolve a sub-per-cent effect; the ``m_tt`` figure is the test
the brief asks for.  Both are reported.

Statistics are compared **paired**, not as two independent samples.  Every cell
decays the *same* production events in the same order, so the production-level
fluctuation is common to all of them and cancels in a difference: the error on
``n_A - n_B`` is set by the discordant pairs (McNemar), not by
``sqrt(n_A + n_B)``.  How much that actually buys is measured, not assumed.

Stages
------
This script never generates the production sample or the truth sample: both are
taken from ``run_mtt_threshold.py``'s work directory (``--prod-lhe``,
``--truth-lhe``), read-only.

    python3 run_mtt_unweighting.py --stage madspin --cells PA_joint,...
    python3 run_mtt_unweighting.py --stage harvest

Writes ``data/histograms_unweighting.npz``, ``data/meta_unweighting.json`` and
``data/logs_unweighting/``, all beside -- and never on top of -- the files of
the first study.
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import json
import math
import os
import re
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
if _here not in sys.path:
    sys.path.insert(0, _here)

from tests.parallel_tests.madspin_comparator import (  # noqa: E402
    MadSpinFactory, SpinModeConfig, _parse_unweighting)

# The physics setup, the binning, the readers and the pairing check all come
# from the first study verbatim.  Importing them rather than copying them is
# what makes "same production sample, same truth, same observable, same grid" a
# property of the code instead of a promise in a README.
from run_mtt_threshold import (                       # noqa: E402
    MODEL, MULTIPARTICLES, PRODUCTION_PROCESS, DECAYS, TRUTH_PROCESS,
    BWCUTOFF, RUN_CARD, SEED_PROD, SEED_MADSPIN, SEED_TRUTH,
    MT_POLE, WT_POLE, TWO_MT, MTT_LO, MTT_HI, MTT_NBINS,
    DELTA_LO, DELTA_HI, DELTA_NBINS,
    edges, delta_edges, harvest, harvest_many, pair_delta, cross_check,
    lhe_cross, _banner_masses, _open, _truncate_lhe,
)


# --------------------------------------------------------------------------
# The cells.  Two rows (the off-shell spinmodes) x the schemes each one can
# actually run.  Every one of them sets ``unweighting`` explicitly: this study
# is about the scheme, so nothing here may be left to ``auto``.
# --------------------------------------------------------------------------
CELLS = [
    # label,             spinmode,  unweighting
    ('PA_joint',         'PA',      'joint'),
    ('PA_seq',           'PA',      'sequential'),
    ('PA_globalretry',   'PA',      'sequential_global_retry'),
    ('PA_withmass',      'PA',      'sequential_with_mass'),
    ('ms_joint',         'madspin', 'joint'),
    ('ms_seq',           'madspin', 'sequential'),
    ('ms_globalretry',   'madspin', 'sequential_global_retry'),
]
CELL_SPINMODE = {c[0]: c[1] for c in CELLS}
CELL_SCHEME = {c[0]: c[2] for c in CELLS}
ROWS = {'PA': ['PA_joint', 'PA_seq', 'PA_globalretry', 'PA_withmass'],
        'madspin': ['ms_joint', 'ms_seq', 'ms_globalretry']}

# The pairs the null hypothesis is about, per row:  (A, B, what it tests).
NULL_PAIRS = [
    ('PA_joint', 'PA_globalretry',
     'neither reads the Z_k table: must agree within statistics'),
    ('ms_joint', 'ms_globalretry',
     'neither reads the Z_k table: must agree within statistics'),
    ('PA_seq', 'PA_globalretry',
     'tabulated Z_hat against exact cancellation: the residual Z_hat/Z'),
    ('ms_seq', 'ms_globalretry',
     'tabulated Z_hat against exact cancellation: the residual Z_hat/Z'),
    ('PA_seq', 'PA_joint',
     'tabulated Z_hat against a scheme with no mass stage'),
    ('ms_seq', 'ms_joint',
     'tabulated Z_hat against a scheme with no mass stage'),
    ('PA_withmass', 'PA_joint',
     'mass drawn inside each slot: no Z_k arises at all'),
]

# The reconstructed top virtuality.  15 Gamma_t either side of the pole is the
# whole of what BW_cut = 15 allows, and 0.02 GeV bins resolve a width of 1.49.
MTOP_LO = MT_POLE - 16.0 * WT_POLE
MTOP_HI = MT_POLE + 16.0 * WT_POLE
MTOP_NBINS = 2400

# Windows the scheme comparison is integrated over.  The sub-threshold one is
# what the figure is about; the others exist because it holds 0.165 % of the
# cross section and a sub-per-cent effect cannot be seen in it -- see
# ``sensitivity`` below, which quantifies that rather than leaving it as a
# remark.
WINDOWS = [
    ('below 2mt',      MTT_LO,  TWO_MT),
    ('2mt+5',          MTT_LO,  TWO_MT + 5.0),
    ('2mt+10',         MTT_LO,  TWO_MT + 10.0),
    ('336-356',        336.0,   356.0),
    ('330-370',        330.0,   370.0),
    ('below 380',      MTT_LO,  380.0),
    ('full 290-520',   MTT_LO,  MTT_HI),
]


def mtop_edges():
    return np.linspace(MTOP_LO, MTOP_HI, MTOP_NBINS + 1)


# --------------------------------------------------------------------------
# Harvest: m_tt and the two top virtualities, one pass.
# --------------------------------------------------------------------------
def harvest_cell(path, bins, mbins):
    """``m_tt`` and the reconstructed top virtualities out of one decayed LHE.

    ``m_tt`` is built exactly as ``run_mtt_threshold.harvest`` builds it -- the
    sum of the four status-1 particles with ``|pid|`` in ``{24, 5}`` -- and
    ``--cross-check`` asserts the two agree bin for bin rather than trusting
    that this loop is a faithful copy.

    The extra output is the per-top virtuality: ``m(W^+ b)`` and ``m(W^- \\bar
    b)`` separately, from the sign of the pid.  That is the quantity ``Z_k``
    distorts *directly*; ``m_tt`` only sees it through the reshuffle.  Both tops
    go into one histogram (the setup is charge-symmetric and the decay chains
    are conjugates), and the running sums give the moments, which is where the
    sensitivity is.
    """
    nb = len(bins) - 1
    lo, hi = float(bins[0]), float(bins[-1])
    inv = nb / (hi - lo)
    nm = len(mbins) - 1
    mlo, mhi = float(mbins[0]), float(mbins[-1])
    minv = nm / (mhi - mlo)

    sumw = np.zeros(nb); sumw2 = np.zeros(nb); cnt = np.zeros(nb, dtype=np.int64)
    msumw = np.zeros(nm); mcnt = np.zeros(nm, dtype=np.int64)
    out = dict(under_w=0.0, over_w=0.0, under_n=0, over_n=0,
               tot_w=0.0, tot_n=0, nfound_bad=0,
               mtop_n=0, mtop_s1=0.0, mtop_s2=0.0, mtop_out=0)

    inev = head = False
    w = 0.0
    tot = [0.0] * 4
    plus = [0.0] * 4
    minus = [0.0] * 4
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
                m2 = tot[3] ** 2 - tot[0] ** 2 - tot[1] ** 2 - tot[2] ** 2
                m = math.sqrt(m2) if m2 > 0 else 0.0
                out['tot_w'] += w
                out['tot_n'] += 1
                if m < lo:
                    out['under_w'] += w; out['under_n'] += 1
                elif m >= hi:
                    out['over_w'] += w; out['over_n'] += 1
                else:
                    i = int((m - lo) * inv)
                    sumw[i] += w; sumw2[i] += w * w; cnt[i] += 1
                if nfound == 4:
                    for vec in (plus, minus):
                        t2 = (vec[3] ** 2 - vec[0] ** 2 - vec[1] ** 2
                              - vec[2] ** 2)
                        mt = math.sqrt(t2) if t2 > 0 else 0.0
                        out['mtop_n'] += 1
                        out['mtop_s1'] += mt
                        out['mtop_s2'] += mt * mt
                        if mlo <= mt < mhi:
                            j = int((mt - mlo) * minv)
                            msumw[j] += w; mcnt[j] += 1
                        else:
                            out['mtop_out'] += 1
                tot = [0.0] * 4; plus = [0.0] * 4; minus = [0.0] * 4
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
                pid = int(parts[0]); status = int(parts[1])
            except ValueError:
                continue
            if status != 1 or abs(pid) not in (24, 5):
                continue
            v = (float(parts[6]), float(parts[7]),
                 float(parts[8]), float(parts[9]))
            dest = plus if pid > 0 else minus
            for i in range(4):
                tot[i] += v[i]
                dest[i] += v[i]
            nfound += 1

    out['sumw'] = sumw; out['sumw2'] = sumw2; out['cnt'] = cnt
    out['mtop_sumw'] = msumw; out['mtop_cnt'] = mcnt
    return out


def _stream_event(path):
    """Yield ``(m_tt, sqrt(shat), m(W+ b), m(W- b~))`` per decayed event.

    ``run_mtt_threshold._stream_mtt_shat`` yields the first two; the two top
    virtualities are the quantity ``Z_k`` distorts directly, and they have to
    come out of the *same* pass if they are to be compared pair by pair.
    """
    inev = head = False
    tot = init = plus = minus = None
    with _open(path) as fp:
        for line in fp:
            s = line.strip()
            if not inev:
                if s.startswith('<event'):
                    inev, head = True, True
                    tot = [0.0] * 4; init = [0.0] * 4
                    plus = [0.0] * 4; minus = [0.0] * 4
                continue
            if s.startswith('</event'):
                inev = False

                def _m(v):
                    q = v[3] ** 2 - v[0] ** 2 - v[1] ** 2 - v[2] ** 2
                    return math.sqrt(q) if q > 0 else 0.0
                yield _m(tot), _m(init), _m(plus), _m(minus)
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
                    init[i] += v[i]
            elif status == 1 and abs(pid) in (24, 5):
                dest = plus if pid > 0 else minus
                for i in range(4):
                    tot[i] += v[i]
                    dest[i] += v[i]


def paired_windows(path_a, path_b, windows):
    """Per-production-event coincidence counts between two decayed files.

    Every cell decays the same production events in the same order, so the two
    decayed files pair positionally and ``sqrt(shat)`` -- RAMBO-invariant, and
    identical to the last bit across schemes -- is the check that they really
    do.  For each window this returns ``(n_a, n_b, n_both, n_pairs)``, from
    which the McNemar error on ``n_a - n_b`` is ``sqrt(n_a + n_b - 2 n_both)``:
    the *discordant* pairs, which is the only thing that can move the
    difference.  Comparing that with ``sqrt(n_a + n_b)`` says how much the
    shared production sample is worth, which is a measurement and not a guess.
    """
    counts = [[0, 0, 0] for _ in windows]
    n = 0
    max_dshat = 0.0
    # The same pairing, applied to the top virtuality.  Both tops of every event
    # contribute one paired difference, so the mean of ``d`` is the shift in
    # <m_top> between the two schemes and its error is rms/sqrt(2N) -- the
    # PAIRED error, which is what the shared production sample buys.
    nt = 0
    ds1 = ds2 = 0.0
    for (ma, sa, pa, na), (mb, sb, pb, nb) in zip(_stream_event(path_a),
                                                  _stream_event(path_b)):
        n += 1
        if abs(sa - sb) > max_dshat:
            max_dshat = abs(sa - sb)
        for d in (pa - pb, na - nb):
            nt += 1
            ds1 += d
            ds2 += d * d
        for k, (_name, lo, hi) in enumerate(windows):
            ina = lo <= ma < hi
            inb = lo <= mb < hi
            if ina:
                counts[k][0] += 1
            if inb:
                counts[k][1] += 1
            if ina and inb:
                counts[k][2] += 1
    mean = ds1 / nt if nt else float('nan')
    var = max(0.0, ds2 / nt - mean ** 2) if nt else float('nan')
    return {'n_pairs': n, 'max_dshat': max_dshat,
            'mtop_n': nt, 'mtop_dmean': mean,
            'mtop_drms': math.sqrt(var) if nt else float('nan'),
            'mtop_dmean_err': (math.sqrt(var / nt) if nt else float('nan')),
            'windows': {w[0]: {'n_a': c[0], 'n_b': c[1], 'n_both': c[2]}
                        for w, c in zip(windows, counts)}}


# --------------------------------------------------------------------------
# The overweight safety net's own end-of-run line.
# --------------------------------------------------------------------------
_RE_OW = re.compile(
    r'overweight safety net:\s*(\d+)/(\d+)\s*written events.*?'
    r'largest factor\s*([0-9.eE+-]+)', re.S)
_RE_OW_ZERO = re.compile(r'overweight safety net:\s*0/(\d+)\s*written events '
                         r'carried a non-unit weight -- no accept/reject')
_RE_OW_ADD = re.compile(
    r'Carrying it added\s*([+-][0-9.eE+-]+)\s*to the summed event weight, '
    r'i\.e\.\s*([+-][0-9.eE+-]+)%')
_RE_OW_JOINT = re.compile(r'(\d+)\s*of them from the joint accept/reject')


def parse_overweights(text):
    """``(n_carrying, n_written, largest_factor, excess_weight, percent)``.

    The safety net prints one line at the end of every run, either the zero
    form or the full one.  Reading it back is the only way to know whether a
    cell's normalisation was moved by a weight that exceeded its bound -- which
    is a *rate* effect of the accept/reject machinery and emphatically not a
    ``Z_k`` effect, so a scheme difference caused by it must not be read as one.
    """
    flat = ' '.join(text.split())
    z = _RE_OW_ZERO.search(flat)
    if z:
        return {'n': 0, 'n_written': int(z.group(1)), 'largest': 1.0,
                'excess_w': 0.0, 'percent': 0.0, 'n_joint': 0, 'found': True}
    m = _RE_OW.search(flat)
    if not m:
        return {'found': False}
    out = {'n': int(m.group(1)), 'n_written': int(m.group(2)),
           'largest': float(m.group(3)), 'found': True}
    a = _RE_OW_ADD.search(flat)
    out['excess_w'] = float(a.group(1)) if a else float('nan')
    out['percent'] = float(a.group(2)) if a else float('nan')
    j = _RE_OW_JOINT.search(flat)
    out['n_joint'] = int(j.group(1)) if j else 0
    return out


_RE_Z = re.compile(
    r'MadSpin sequential: slot (\S+) rate factor '
    r'Z\(([0-9.eE+-]+)\)=([0-9.]+)\s+Z\(([0-9.eE+-]+)\)=1\s+'
    r'Z\(([0-9.eE+-]+)\)=([0-9.]+)\s+'
    r'\((\d+) samples, (\d+) bins, bin/fit deviation up to ([0-9.]+)%\)')


def parse_z_tables(text):
    """The tabulated ``Z_k`` itself, as the run logged it.

    ``_z_tables`` prints one line per slot with the fitted factor at the two
    ends of the mass window, the number of max-weight-scan samples behind it and
    the largest bin-to-fit deviation.  Two things make it worth reading back:
    the span of ``Z`` says how much of the lineshape the table is responsible
    for (if ``Z`` were flat there would be nothing to get wrong), and the
    deviation is the shipped estimate of how well the fit represents the
    measurement.  It is a floor on ``Z_hat/Z``, not the whole of it -- the
    binned values carry their own statistical error too -- but it is the number
    the code itself quotes.

    Only the schemes with a mass stage emit this at all; ``joint`` never does,
    which is itself the check that it does not use the table.
    """
    out = []
    for m in _RE_Z.finditer(' '.join(text.split())):
        out.append({'slot': m.group(1),
                    'lo': float(m.group(2)), 'Z_lo': float(m.group(3)),
                    'pole': float(m.group(4)),
                    'hi': float(m.group(5)), 'Z_hi': float(m.group(6)),
                    'samples': int(m.group(7)), 'bins': int(m.group(8)),
                    'bin_fit_deviation_percent': float(m.group(9))})
    return out


# --------------------------------------------------------------------------
# Sensitivity, computed before the runs rather than after them.
# --------------------------------------------------------------------------
def sensitivity(n_events, ref_fracs):
    """The 1 sigma difference this study could detect, per window.

    ``ref_fracs`` maps a window name to the fraction of events the first study
    measured in it.  Two cells of ``n_events`` each, treated as independent,
    resolve a relative difference of ``sqrt(2/(f N))`` at 1 sigma.  Pairing on
    the shared production sample can only improve on that, so this is the
    conservative statement -- and it is the one to quote *before* the runs, when
    deciding whether the measurement is worth making at all.
    """
    out = {}
    for name, f in ref_fracs.items():
        n = f * n_events
        out[name] = {'expected_events': n,
                     'one_sigma_rel_diff': math.sqrt(2.0 / n) if n > 0 else
                     float('inf')}
    return out


# --------------------------------------------------------------------------
def build_factory(args):
    factory = MadSpinFactory(
        'mtt_unweighting',
        production_process=PRODUCTION_PROCESS,
        decays=DECAYS,
        model=MODEL,
        multiparticles=MULTIPARTICLES,
        nevents=args.nevents,
        seed=SEED_PROD,
        extra_run_card=dict(RUN_CARD),
        extra_madspin_settings={'nb_core': args.nb_core,
                                'BW_cut': int(BWCUTOFF)},
        base_dir=pjoin(args.basedir, 'MS'),
    )
    os.makedirs(factory.base_dir, exist_ok=True)
    # Adopt the first study's production sample instead of making a second one.
    # ``run_mode`` copies the LHE into its own run directory before touching it,
    # so the shared file is only ever read.
    src = args.prod_lhe
    if not os.path.exists(src):
        raise SystemExit('production sample not found: %s' % src)
    if args.nevents and args.nevents < args.prod_nevents:
        src = _truncate_lhe(src, pjoin(args.basedir,
                                       'prod_%d.lhe' % args.nevents),
                            args.nevents)
    factory.events_file = src
    factory.cross_in = lhe_cross(src)
    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basedir', default='/tmp/mtt_unweighting_work')
    ap.add_argument('--outdir', default=pjoin(_here, 'data'))
    ap.add_argument('--prod-lhe',
                    default='/tmp/mtt_threshold_work/MS/PROC/Events/run_01/'
                            'unweighted_events.lhe.gz',
                    help='the FIRST study\'s production sample.  Reused, never '
                         'regenerated: the whole point is that every cell here '
                         'and every curve there decays the same events.')
    ap.add_argument('--prod-nevents', type=int, default=1000000,
                    help='how many events --prod-lhe holds')
    ap.add_argument('--nevents', type=int, default=1000000,
                    help='events per cell (<= --prod-nevents; the sample is '
                         'truncated from the front if smaller)')
    ap.add_argument('--truth-lhe',
                    default=','.join(
                        '/tmp/mtt_threshold_work/TRUTH/Events/run_0%d/'
                        'unweighted_events.lhe.gz' % k for k in range(1, 6)),
                    help='the FIRST study\'s truth sample, comma separated')
    ap.add_argument('--nb-core', type=int, default=4)
    ap.add_argument('--stage', default='all',
                    choices=['all', 'madspin', 'harvest', 'sensitivity'])
    ap.add_argument('--cells', default=None,
                    help='comma-separated subset of the cells to RUN in this '
                         'invocation; they are independent, so they can be '
                         'launched as concurrent processes and the harvest '
                         'stage picks up whatever is on disk.')
    ap.add_argument('--cross-check', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.basedir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    if args.stage == 'sensitivity':
        # The first study's measured fractions, so the estimate is grounded in
        # a real spectrum rather than in a guess about one.
        fracs = {'below 2mt': 0.00165, '2mt+5': 0.0072, '2mt+10': 0.0170,
                 '330-370': 0.0565, 'full 290-520': 0.556}
        s = sensitivity(args.nevents, fracs)
        print('sensitivity at %d events/cell, two cells, unpaired:'
              % args.nevents)
        for name in ['below 2mt', '2mt+5', '2mt+10', '330-370',
                     'full 290-520']:
            print('  %-14s ~%9.0f events   1 sigma on the ratio = %6.3f %%'
                  % (name, s[name]['expected_events'],
                     100 * s[name]['one_sigma_rel_diff']))
        return

    factory = build_factory(args)
    print('production: reusing %s (cross %s pb, %d events)'
          % (factory.events_file, factory.cross_in, args.nevents))
    sys.stdout.flush()

    if args.stage in ('all', 'madspin'):
        wanted = ([c.strip() for c in args.cells.split(',')]
                  if args.cells else [c[0] for c in CELLS])
        for label, spinmode, scheme in CELLS:
            if label not in wanted:
                continue
            t0 = time.time()
            res = factory.run_mode(SpinModeConfig(label, spinmode),
                                   extra_settings={'unweighting': scheme})
            print('%-16s %6.0f s  eff=%s  cross_out=%s  resolved=%s'
                  % (label, time.time() - t0, res.efficiency, res.cross_out,
                     res.unweighting_mode))
            sys.stdout.flush()

    if args.stage not in ('all', 'harvest'):
        return

    # ----------------------------------------------------------------- harvest
    bins = edges()
    mbins = mtop_edges()
    logdir = pjoin(args.outdir, 'logs_unweighting')
    os.makedirs(logdir, exist_ok=True)

    store = {'bins': bins, 'mtop_bins': mbins}
    meta = {
        'code_sha': subprocess.check_output(
            ['git', '-C', _root, 'rev-parse', 'HEAD']).decode().strip(),
        'code_branch': subprocess.check_output(
            ['git', '-C', _root, 'rev-parse', '--abbrev-ref', 'HEAD']
        ).decode().strip(),
        'study': 'unweighting schemes within a fixed off-shell spinmode',
        'companion': 'run_mtt_threshold.py / histograms.npz (spinmode axis)',
        'model': MODEL,
        'multiparticles': MULTIPARTICLES,
        'production_process': PRODUCTION_PROCESS,
        'truth_process': TRUTH_PROCESS,
        'madspin_decays': DECAYS,
        'run_card': dict(RUN_CARD),
        'bwcutoff': BWCUTOFF,
        'madspin_BW_cut': int(BWCUTOFF),
        'seed_production': SEED_PROD,
        'seed_truth': SEED_TRUTH,
        'seed_madspin': SEED_MADSPIN,
        'nevents_per_cell': args.nevents,
        'nb_core': args.nb_core,
        'mtt_bins': {'lo': MTT_LO, 'hi': MTT_HI, 'n': MTT_NBINS},
        'mtop_bins': {'lo': MTOP_LO, 'hi': MTOP_HI, 'n': MTOP_NBINS},
        'observable': ('m_tt: invariant mass of the sum of the four status-1 '
                       'particles with |pid| in {24, 5}.  m_top: the same four '
                       'split by pid sign, i.e. m(W+ b) and m(W- b~), both '
                       'entered.'),
        'cells': {},
        'runs': {},
        'cross_check': {},
        'mt_pole_assumed': MT_POLE,
        'wt_pole_assumed': WT_POLE,
        'two_mt': TWO_MT,
        'windows': [{'name': n, 'lo': lo, 'hi': hi} for n, lo, hi in WINDOWS],
        'null_pairs': [{'a': a, 'b': b, 'what': w} for a, b, w in NULL_PAIRS],
        'skipped': {
            'onshell': ('no virtuality is drawn, _spinmode_has_density is '
                        'False and _unweighting_mode forces joint whatever the '
                        'card says -- there is no scheme axis to scan'),
        },
    }
    meta['param_card_masses'] = _banner_masses(factory.events_file)
    meta['sensitivity_apriori'] = sensitivity(
        args.nevents, {'below 2mt': 0.00165, '2mt+5': 0.0072,
                       '2mt+10': 0.0170, '330-370': 0.0565,
                       'full 290-520': 0.556})

    present = []
    for label, spinmode, scheme in CELLS:
        cand = pjoin(factory.base_dir, 'mode_%s' % label,
                     'events_decayed.lhe.gz')
        if not os.path.exists(cand):
            cand = cand[:-3]
        if os.path.exists(cand):
            present.append((label, cand))
        meta['cells'][label] = {'spinmode': spinmode, 'unweighting_asked':
                                scheme, 'present': os.path.exists(cand)}
    if not present:
        raise SystemExit('no decayed sample found under %s' % factory.base_dir)

    # --- truth, from the first study's files --------------------------------
    truth_files = [t.strip() for t in args.truth_lhe.split(',') if t.strip()]
    h, per_run = harvest_many(truth_files, bins)
    store['truth_sumw'] = h['sumw']; store['truth_sumw2'] = h['sumw2']
    store['truth_cnt'] = h['cnt']
    meta['truth_runs'] = per_run
    meta['runs']['truth'] = dict(path=truth_files, nevents=h['tot_n'],
                                 sumw=h['tot_w'], under_n=h['under_n'],
                                 over_n=h['over_n'], under_w=h['under_w'],
                                 over_w=h['over_w'],
                                 malformed=h.get('nfound_bad', 0),
                                 banner_cross_pb=lhe_cross(truth_files[0]))
    print('truth       n=%d sum(w)=%.6g pb' % (h['tot_n'], h['tot_w']))
    sys.stdout.flush()

    # --- the cells ----------------------------------------------------------
    for label, path in present:
        t0 = time.time()
        hh = harvest_cell(path, bins, mbins)
        store['%s_sumw' % label] = hh['sumw']
        store['%s_sumw2' % label] = hh['sumw2']
        store['%s_cnt' % label] = hh['cnt']
        store['%s_mtop_sumw' % label] = hh['mtop_sumw']
        store['%s_mtop_cnt' % label] = hh['mtop_cnt']
        n = hh['mtop_n']
        mean = hh['mtop_s1'] / n if n else 0.0
        var = max(0.0, hh['mtop_s2'] / n - mean ** 2) if n else 0.0
        entry = dict(path=path, nevents=hh['tot_n'], sumw=hh['tot_w'],
                     under_n=hh['under_n'], over_n=hh['over_n'],
                     under_w=hh['under_w'], over_w=hh['over_w'],
                     malformed=hh['nfound_bad'],
                     mtop_n=n, mtop_mean=mean, mtop_rms=math.sqrt(var),
                     mtop_mean_err=(math.sqrt(var / n) if n else float('nan')),
                     mtop_out_of_range=hh['mtop_out'],
                     banner_cross_pb=lhe_cross(path))
        meta['runs'][label] = entry
        print('%-16s n=%d sum(w)=%.6g pb  banner=%.4f  <m_top>=%.5f +- %.5f  '
              'rms=%.5f  (%.0f s)'
              % (label, hh['tot_n'], hh['tot_w'], entry['banner_cross_pb'],
                 mean, entry['mtop_mean_err'], entry['mtop_rms'],
                 time.time() - t0))
        sys.stdout.flush()

        if args.cross_check:
            # 1. the text reader against lhe_parser, as the first study does.
            nn, diff = cross_check(path, bins)
            # 2. this loop's m_tt against the first study's harvest(), bin for
            #    bin.  They are two different loops over the same file and they
            #    must produce the same histogram; if they do not, the extra
            #    per-top bookkeeping has corrupted the shared part.
            ref = harvest(path, bins)
            same = bool(np.array_equal(ref['cnt'], hh['cnt']))
            meta['cross_check'][label] = {
                'nevents': nn, 'max_abs_diff_GeV': diff,
                'matches_run_mtt_threshold_harvest': same,
                'max_sumw_diff': float(np.abs(ref['sumw']
                                              - hh['sumw']).max())}
            print('  cross-check: lhe_parser %d events max|diff|=%.3g GeV;  '
                  'harvest() identical: %s' % (nn, diff, same))
            if not same:
                raise AssertionError('harvest_cell disagrees with harvest() '
                                     'on %s' % path)
            sys.stdout.flush()

    # --- the mechanism, and the pairing check -------------------------------
    dbins = delta_edges()
    store['delta_bins'] = dbins
    meta['delta_mtt'] = {}
    for label, path in present:
        t0 = time.time()
        d = pair_delta(factory.events_file, path, dbins)
        store['delta_%s' % label] = d.pop('hist')
        meta['delta_mtt'][label] = d
        print('delta %-16s n=%d mean=%+.4f rms=%.4f max|dsqrt(shat)|=%.3g '
              'crossed down=%d (%.0f s)'
              % (label, d['n'], d['mean'], d['rms'], d['max_dshat'],
                 d['crossed_down'], time.time() - t0))
        sys.stdout.flush()

    # --- paired, per-window scheme differences ------------------------------
    have = {lab: p for lab, p in present}
    meta['paired'] = {}
    for a, b, what in NULL_PAIRS:
        if a not in have or b not in have:
            continue
        t0 = time.time()
        pr = paired_windows(have[a], have[b], WINDOWS)
        pr['what'] = what
        # ``_stream_event`` and ``harvest_cell`` are two different loops over
        # the same file; the in-range count of the widest window must be the
        # histogram's own entry count, or one of them is wrong.
        for lab, side in ((a, 'n_a'), (b, 'n_b')):
            got = pr['windows']['full 290-520'][side]
            want = int(store['%s_cnt' % lab].sum())
            if got != want:
                raise AssertionError(
                    '%s: paired stream counts %d events in 290-520 GeV but the '
                    'histogram holds %d' % (lab, got, want))
        meta['paired']['%s vs %s' % (a, b)] = pr
        w = pr['windows']['below 2mt']
        disc = w['n_a'] + w['n_b'] - 2 * w['n_both']
        print('paired %-16s vs %-16s  below 2mt: n_a=%d n_b=%d both=%d  '
              'discordant=%d  sigma(diff)=%.1f (unpaired %.1f)   '
              'Delta<m_top>=%+.6f +- %.6f GeV  (%.0f s)'
              % (a, b, w['n_a'], w['n_b'], w['n_both'], disc,
                 math.sqrt(disc), math.sqrt(w['n_a'] + w['n_b']),
                 pr['mtop_dmean'], pr['mtop_dmean_err'], time.time() - t0))
        sys.stdout.flush()

    # --- logs, cards, and the overweight counters ---------------------------
    for label, _path in present:
        run_dir = pjoin(factory.base_dir, 'mode_%s' % label)
        log = pjoin(run_dir, 'madspin.log')
        if os.path.exists(log):
            # ``.log.txt``: the repository's .gitignore has a blanket ``*.log``
            # rule, so a copy named ``.log`` would silently never be committed.
            shutil.copy(log, pjoin(logdir, 'madspin_%s.log.txt' % label))
            text = open(log).read()
            mode, why = _parse_unweighting(text)
            meta['runs'][label]['unweighting'] = mode
            meta['runs'][label]['unweighting_why'] = why
            meta['runs'][label]['overweights'] = parse_overweights(text)
            meta['runs'][label]['z_tables'] = parse_z_tables(text)
            asked = CELL_SCHEME[label]
            meta['runs'][label]['scheme_as_asked'] = (mode == asked)
            if mode != asked:
                print('NOTE %-16s asked for %s, ran %s (%s)'
                      % (label, asked, mode, why))
        card = pjoin(run_dir, 'madspin_card.dat')
        if os.path.exists(card):
            shutil.copy(card, pjoin(logdir, 'madspin_card_%s.dat' % label))
            meta['runs'][label]['madspin_card'] = open(card).read()

    np.savez_compressed(pjoin(args.outdir, 'histograms_unweighting.npz'),
                        **store)
    with open(pjoin(args.outdir, 'meta_unweighting.json'), 'w') as fp:
        json.dump(meta, fp, indent=2, sort_keys=True)
    print('wrote %s and meta_unweighting.json'
          % pjoin(args.outdir, 'histograms_unweighting.npz'))


if __name__ == '__main__':
    main()
