#!/usr/bin/env python3
"""Generate the statistics needed for a **mass-differential** ``C_kk(m_4l)``.

The inclusive result of ``../zz_loopinduced`` and ``../zz_nlo`` is that the
``g g`` quark box and the ``q q~`` continuum have **opposite-sign** helicity-sign
correlations between the two ``Z``:

    g g -> ZZ   C_kk = +0.570 +- 0.141        (50 000 events)
    q q~ -> ZZ  C_kk = -0.675 +- 0.131        (50 000 events)

This script asks whether that sign flip can be shown *differentially in the
four-lepton invariant mass*, which needs per-event ``(m_4l, cos th1, cos th2)``
and therefore new events: both earlier studies committed 1-D histograms only,
and their LHE files are gone.

Two samples, the same two "truth" processes the earlier studies used, the same
cuts, the same scale, the same PDF:

    gg   g g > e+ e- mu+ mu- / a [noborn=QCD]     (../zz_loopinduced sample B)
    qq   p p > e+ e- mu+ mu- / a [QCD]            (../zz_nlo truth)

Nothing is decayed by MadSpin here.  This is a physics figure, not a MadSpin
validation, so both sides are the full off-shell four-lepton matrix element with
the two decays correlated by construction.

The run cards are not re-typed: they are imported from the two study drivers, so
the samples cannot drift away from the published inclusive numbers by an edit
made in one place and not the other.  Only ``nevents`` and ``iseed`` change.

Statistics are accumulated in **seeded batches** rather than one long run, for
two reasons: a partially finished set is still usable (the batches are
independent and simply concatenate), and the ``gg`` side costs about one event
per CPU-second, so a single 200 000-event run would be a four-hour all-or-nothing
job on 18 cores.

    python3 run_ckk_mass.py --stage all --basedir /tmp/ckk_work \\
            --nb-core 16 --gg-batches 4 --qq-batches 4

Stages: ``prod`` generates, ``harvest`` reads the LHEs into ``data/events.npz``
as per-event columns.  ``all`` does both.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_VAL = os.path.abspath(os.path.join(_HERE, '..'))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
pjoin = os.path.join

sys.path.insert(0, pjoin(_VAL, 'zz_loopinduced'))
import observables as OBS                                        # noqa: E402

# The two study drivers, imported for their run-card dictionaries and card
# writers.  They live in sibling directories and both define a module called
# ``observables``/``observables_zz``, so they are loaded by path rather than by
# putting both directories on sys.path.
import importlib.util                                            # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LI = _load('_run_zz_loopinduced',
           pjoin(_VAL, 'zz_loopinduced', 'run_zz_loopinduced.py'))
NLO = _load('_run_zz_nlo', pjoin(_VAL, 'zz_nlo', 'run_zz_nlo.py'))

PROC = {'gg': LI.PROC_B, 'qq': NLO.PROC['truth']}
OUTDIR = {'gg': 'gg4l_li', 'qq': 'pp4l_nlo'}

# Batch size.  50 000 is the earlier studies' sample size, so one batch
# reproduces their inclusive number and the batch-to-batch spread is a free
# check on the error bars.
BATCH = 50000
# Seeds.  4321 is the earlier studies' seed, kept first so that batch 1 of the
# gg side is bit-for-bit the sample ../zz_loopinduced published.
SEEDS = [4321, 4322, 4323, 4324, 4325, 4326, 4327, 4328,
         4329, 4330, 4331, 4332, 4333, 4334, 4335, 4336]

TIMINGS = []


def sh(cmd, log, cwd=None, tag=None):
    t0 = time.time()
    with open(log, 'w') as fp:
        rc = subprocess.call(cmd, stdout=fp, stderr=subprocess.STDOUT, cwd=cwd)
    dt = time.time() - t0
    TIMINGS.append({'tag': tag or ' '.join(map(str, cmd)), 'seconds': dt,
                    'rc': rc})
    return rc, dt


# --------------------------------------------------------------------------
# stage: prod
# --------------------------------------------------------------------------
def generate_output(basedir, tag, logdir):
    """``output`` one process.  Serial by construction: MG5 builds CutTools and
    IREGI in the *source* tree the first time, and two concurrent outputs race
    there."""
    outdir = pjoin(basedir, OUTDIR[tag])
    if os.path.exists(pjoin(outdir, 'SubProcesses')):
        return outdir
    script = pjoin(basedir, 'gen_%s.mg5' % tag)
    with open(script, 'w') as fp:
        fp.write('set auto_convert_model T\n')
        # The model has to be imported EXPLICITLY.  ``auto_convert_model`` turns
        # ``sm`` into ``loop_sm`` when a loop process asks for it, but only once
        # a model is loaded; with ``set`` as the first line of a command file and
        # no ``import model`` after it, ``g g > e+ e- mu+ mu- / a [noborn=QCD]``
        # dies inside ``loop_interface.validate_model`` with
        # ``AttributeError: 'NoneType' object has no attribute 'get'`` -- and
        # mg5_aMC still exits 0, so the failure is silent unless the output
        # directory is checked afterwards, which is why it is checked below.
        fp.write('import model sm\n')
        fp.write('generate %s\n' % PROC[tag])
        fp.write('output %s\n' % outdir)
    log = pjoin(basedir, 'gen_%s.log' % tag)
    rc, dt = sh([pjoin(_ROOT, 'bin', 'mg5_aMC'), script], log,
                tag='output_%s' % tag)
    LI.copy_log(log, pjoin(logdir, 'gen_%s.log.txt' % tag))
    print('%s: output in %.0f s (rc=%d)' % (tag, dt, rc), flush=True)
    if rc != 0 or not os.path.exists(pjoin(outdir, 'SubProcesses')):
        raise RuntimeError('output of %s failed, see %s' % (tag, log))
    return outdir


def prepare_card(basedir, tag, nevents, seed):
    """Write this sample's run card, from the sibling study's own dictionary."""
    card = pjoin(basedir, OUTDIR[tag], 'Cards', 'run_card.dat')
    if tag == 'gg':
        # ptheavy is hidden from a run card whose process has no heavy final
        # state and has to be written in before it can be set.  Same insertion
        # ../zz_loopinduced makes.
        txt = open(card).read()
        if 'ptheavy' not in txt:
            anchor = (' 0.0  = xptl ! minimum pt for at least one charged '
                      'lepton \n')
            if anchor not in txt:
                raise RuntimeError('cannot find the ptheavy anchor in ' + card)
            txt = txt.replace(anchor, anchor + ' 1.0  = ptheavy   ! minimum pt '
                              'for at least one heavy final state (inert here; '
                              'read by the custom dummy_cuts)\n')
            open(card, 'w').write(txt)
        kv = dict(LI.RUN_CARD_COMMON)
        kv.update(LI.RUN_CARD_B_ONLY)
        kv['nevents'] = nevents
        kv['iseed'] = seed
        LI.set_run_card(card, **kv)
        txt = open(card).read()
        old = '  = custom_fcts ! List of files containing user hook function'
        if old in txt:
            txt = txt.replace(old, ' %s = custom_fcts ! List of files '
                              'containing user hook function' % LI.CUTS_FILE)
            open(card, 'w').write(txt)
        if LI.CUTS_FILE not in open(card).read():
            raise RuntimeError('the custom cut did not make it into ' + card)
    else:
        kv = dict(NLO.RUN_CARD_NLO)
        kv['nevents'] = nevents
        kv['iseed'] = seed
        NLO.set_run_card(card, **kv)
        NLO.set_pt_min_pdg(card, NLO.PT_MIN_PDG)
        import re
        txt = open(card).read()
        m = re.search(r'^[^\n!]*=\s*custom_fcts\b', txt, re.M)
        if not m:
            raise RuntimeError('no custom_fcts entry in ' + card)
        txt = txt[:m.start()] + ' %s = custom_fcts' % NLO.CUTS_FILE_NLO \
            + txt[m.end():]
        open(card, 'w').write(txt)
        if NLO.CUTS_FILE_NLO not in open(card).read():
            raise RuntimeError('the custom cut did not make it into ' + card)
    return card


def batch_lhe(basedir, tag, run_name):
    ev = pjoin(basedir, OUTDIR[tag], 'Events', run_name)
    for name in ('unweighted_events.lhe.gz', 'events.lhe.gz',
                 'unweighted_events.lhe', 'events.lhe'):
        p = pjoin(ev, name)
        if os.path.exists(p):
            return p
    return None


def run_batch(basedir, tag, run_name, nb_core, logdir):
    procdir = pjoin(basedir, OUTDIR[tag])
    lhe = batch_lhe(basedir, tag, run_name)
    if lhe:
        print('%s/%s: reusing %s' % (tag, run_name, lhe), flush=True)
        return lhe
    cmd = pjoin(basedir, 'run_%s_%s.cmd' % (tag, run_name))
    with open(cmd, 'w') as fp:
        fp.write('set run_mode 2\nset nb_core %d\n' % nb_core)
        if tag == 'qq':
            # aMC@NLO stopped at the parton level: MC@NLO events, no shower.
            fp.write('launch aMC@NLO -p -f -n %s\n' % run_name)
            exe = pjoin(procdir, 'bin', 'aMCatNLO')
        else:
            fp.write('generate_events %s -f\n' % run_name)
            exe = pjoin(procdir, 'bin', 'madevent')
    log = pjoin(basedir, 'run_%s_%s.log' % (tag, run_name))
    rc, dt = sh([sys.executable, exe, cmd], log, cwd=procdir,
                tag='sample_%s_%s' % (tag, run_name))
    LI.copy_log(log, pjoin(logdir, 'run_%s_%s.log.txt' % (tag, run_name)))
    lhe = batch_lhe(basedir, tag, run_name)
    print('%s/%s: %.0f s (rc=%d) -> %s' % (tag, run_name, dt, rc, lhe),
          flush=True)
    if rc != 0 or not lhe:
        raise RuntimeError('batch %s/%s failed, see %s' % (tag, run_name, log))
    return lhe


def stage_prod(basedir, tags, batches, nb_core, logdir):
    os.makedirs(basedir, exist_ok=True)
    out = {}
    # Every ``output`` first, and serially: MG5 builds CutTools/IREGI in the
    # source tree the first time and two concurrent outputs race there.  Doing
    # them all up front also means the cheap sample is not queued behind the
    # expensive one's four-hour event generation.
    for tag in tags:
        generate_output(basedir, tag, logdir)
    for tag in tags:
        out[tag] = []
        for i in range(batches[tag]):
            run_name = 'b%02d' % (i + 1)
            prepare_card(basedir, tag, BATCH, SEEDS[i])
            out[tag].append(run_batch(basedir, tag, run_name, nb_core, logdir))
    return out


# --------------------------------------------------------------------------
# stage: harvest -- per-event columns, which is the whole point
# --------------------------------------------------------------------------
KEEP = ('m_4l', 'cos_theta1', 'cos_theta2', 'cos1cos2', 'pol0_1', 'pol0_2')


def harvest(files):
    """Read a list of LHEs into per-event columns.

    ``observables.compute`` is the loop-induced study's own, so ``theta_i`` is
    exactly what ``SPIN_COEFFICIENTS.md`` section 1 documents: the polar angle
    of the ``l+`` in its parent pair's rest frame, against that pair's direction
    in the four-lepton rest frame.

    Per-event columns, not histograms.  That is the entire reason this study
    exists: a 1-D histogram of ``cos th1 cos th2`` -- which is all the two
    earlier studies committed -- cannot be re-binned in ``m_4l`` afterwards.
    """
    cols = {k: [] for k in KEEP}
    cols['w'] = []
    cols['batch'] = []
    for i, path in enumerate(files):
        w, p = OBS.read_lhe(path)
        obs = OBS.compute(p)
        cols['w'].append(w)
        cols['batch'].append(np.full(len(w), i, dtype=np.int16))
        for k in KEEP:
            cols[k].append(obs[k])
        print('  harvested %6d events from %s' % (len(w), path), flush=True)
    return {k: np.concatenate(v) for k, v in cols.items()}


def stage_harvest(files, datadir):
    os.makedirs(datadir, exist_ok=True)
    out = {}
    for tag, paths in files.items():
        if not paths:
            continue
        print('%s: %d file(s)' % (tag, len(paths)), flush=True)
        d = harvest(paths)
        for k, v in d.items():
            out['%s/%s' % (tag, k)] = v
        w = d['w']
        # An MC@NLO sample carries negative weights, so the number that sets the
        # error is the effective count, not the row count.  Recorded here so the
        # figure's bars can never be quoted off the wrong N.
        neff = w.sum() ** 2 / (w ** 2).sum()
        print('%s: N = %d, N_eff = %.0f, sum(w) = %.6g'
              % (tag, len(w), neff, w.sum()), flush=True)
    np.savez_compressed(pjoin(datadir, 'events.npz'), **out)
    return pjoin(datadir, 'events.npz')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['all', 'prod', 'harvest'])
    ap.add_argument('--basedir', default='/tmp/ckk_work')
    ap.add_argument('--nb-core', type=int, default=16)
    ap.add_argument('--gg-batches', type=int, default=4)
    ap.add_argument('--qq-batches', type=int, default=4)
    ap.add_argument('--tags', default='gg,qq')
    ap.add_argument('--logs', default=pjoin(_HERE, 'logs'))
    ap.add_argument('--data', default=pjoin(_HERE, 'data'))
    args = ap.parse_args()

    os.makedirs(args.logs, exist_ok=True)
    tags = [t for t in args.tags.split(',') if t]
    batches = {'gg': args.gg_batches, 'qq': args.qq_batches}

    manifest = pjoin(args.basedir, 'files.json')
    if args.stage in ('all', 'prod'):
        files = stage_prod(args.basedir, tags, batches, args.nb_core, args.logs)
        with open(manifest, 'w') as fp:
            json.dump({'files': files, 'timings': TIMINGS}, fp, indent=1)
        print(json.dumps(TIMINGS, indent=1))
    elif os.path.exists(manifest):
        files = json.load(open(manifest))['files']
    else:
        # No manifest: a ``prod`` stage that was interrupted still left finished
        # batches on disk, and they are complete independent samples.  Harvest
        # is therefore driven by what is actually there, not by a manifest.
        files = {}

    if args.stage in ('all', 'harvest'):
        # Harvest whatever finished.  The batches are independent samples, so a
        # short set is a smaller sample and not a broken one.
        found = {t: [p for p in (batch_lhe(args.basedir, t, 'b%02d' % (i + 1))
                                 for i in range(batches[t])) if p]
                 for t in tags}
        print(stage_harvest(found, args.data))


if __name__ == '__main__':
    main()
