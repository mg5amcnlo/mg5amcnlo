#!/usr/bin/env python3
"""Drive the ``g g > z z`` loop-induced MadSpin-against-truth comparison.

Three stages, selectable with ``--stage``:

``prod``
    generate the two production samples.

    * **A** -- ``g g > z z [noborn=QCD]``, the sample MadSpin decays.
    * **B** -- ``g g > e+ e- mu+ mu- / a [noborn=QCD]``, the fully off-shell
      four-lepton calculation, which is the reference every ratio divides by.

    The two run cards are made identical except where the final states force
    them apart, and the difference is confined to the cuts: sample A carries
    ``ptheavy = 1`` acting on its two ``z`` directly, sample B carries the same
    ``ptheavy`` value plus a ``custom_fcts`` file that applies it -- and the
    ``bwcutoff`` mass window -- to the *reconstructed* ``(e+ e-)`` and
    ``(mu+ mu-)`` systems, because sample B has no ``z`` to cut on.  That file
    is ``zz_equivalent_cuts.f``, next to this script, and it reads every number
    it uses out of the run's own cards, so the two sides cannot drift apart by
    someone editing one card and not the other.

``madspin``
    run MadSpin over sample A in each of ``none``, ``madspin``, ``onshell``
    and ``PA``.  The two ``_v1`` modes are refused by MadSpin itself for a
    loop-induced process (``interface_madspin.py``: "not compatible with
    loop-induced processes"), which is why this study has four modes and not
    six.

    Every mode gets its OWN ``ms_dir``.  Reuse across modes was measured to
    save about 2.5 minutes of MadLoop compilation against a per-mode decay cost
    of tens of minutes, i.e. a few percent, and ``run_from_pickle`` restores the
    *pickled* option object -- so a reused directory carries the first run's
    ``spinmode`` and ``BW_cut`` into every option lookup that goes through
    ``decay_all_events.options``.  The kinematics were verified to follow the
    new spinmode anyway, but a few percent of wall time is not worth reasoning
    about which of the two option objects each normalisation happens to read.
    Reuse *within* a mode (a rerun) is what ``--reuse`` leaves alone.

``harvest``
    read every LHE and write ``data/histograms.npz``, ``data/meta.json`` and
    ``data/numbers.txt``.

Usage::

    python3 run_zz_loopinduced.py --stage all --basedir /tmp/zz_work --nb-core 6

The plotting scripts run off ``data/`` alone and need neither MadGraph nor this
script.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))

pjoin = os.path.join


# --------------------------------------------------------------------------
# Configuration.  Everything that ends up in meta.json as "how this was run".
# --------------------------------------------------------------------------
PROC_A = 'g g > z z [noborn=QCD]'
PROC_B = 'g g > e+ e- mu+ mu- / a [noborn=QCD]'

SEED_PROD = 4321
SEED_MADSPIN = 7777
NEVENTS = 50000
BW_CUT = 15

# Run-card settings applied to BOTH samples.  Anything not listed keeps the
# MG5 default for that process.
RUN_CARD_COMMON = {
    'nevents': NEVENTS,
    'iseed': SEED_PROD,
    # fixed renormalisation and factorisation scale at m_Z, on both sides
    'fixed_ren_scale': 'True',
    'fixed_fac_scale': 'True',
    'scale': '91.1880',
    'dsqrt_q2fact1': '91.1880',
    'dsqrt_q2fact2': '91.1880',
    # the pt cut.  In A this acts on the two z; in B it is inert natively
    # (setcuts.f calls a particle heavy only above 10 GeV) and is read by the
    # custom cut and applied to the reconstructed pairs instead.
    'ptheavy': '1.0',
    # the mass window, likewise: inert in A, read by the custom cut in B, and
    # matched to MadSpin's BW_cut on the A side.
    'bwcutoff': '15.0',
    'use_syst': 'False',
}

# Sample B only: the standard MadEvent lepton cuts have to come OFF.  A default
# run card carries ptl = 10, etal = 2.5, drll = 0.4, and sample A has no lepton
# to apply them to, so leaving them on would compare a cut four-lepton sample
# against an uncut one.  This is the single most important line in this file.
RUN_CARD_B_ONLY = {
    'ptl': '0.0',
    'etal': '-1.0',
    'drll': '0.0',
    'mmll': '0.0',
}

MODES = ['none', 'madspin', 'onshell', 'PA']

CUTS_FILE = pjoin(_HERE, 'zz_equivalent_cuts.f')


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
_TIMINGS = None


def sh(cmd, log, cwd=None, env=None, tag=None):
    t0 = time.time()
    with open(log, 'w') as fp:
        rc = subprocess.call(cmd, stdout=fp, stderr=subprocess.STDOUT,
                             cwd=cwd, env=env)
    dt = time.time() - t0
    if tag and _TIMINGS:
        record_timing(tag, dt, rc)
    return rc, dt


def record_timing(tag, seconds, rc=0):
    """Append one wall time to ``<basedir>/timings.json``.

    Kept on disk rather than in memory because the stages are routinely run as
    separate invocations (and, for the MadSpin modes, in parallel), so the
    harvest that writes meta.json is usually a different process from the one
    that did the work.
    """
    if not _TIMINGS:
        return
    try:
        cur = json.load(open(_TIMINGS)) if os.path.exists(_TIMINGS) else {}
    except ValueError:
        cur = {}
    cur[tag] = {'wall_seconds': round(seconds, 1), 'returncode': rc}
    tmp = _TIMINGS + '.tmp.%d' % os.getpid()
    with open(tmp, 'w') as fp:
        json.dump(cur, fp, indent=1, sort_keys=True)
    os.replace(tmp, _TIMINGS)


def code_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=_ROOT).decode().strip()
    except Exception:
        return 'unknown'


def set_run_card(path, **kv):
    """Rewrite ``key = value`` entries of a run card in place.

    Raises if a key is not present: a silently-ignored run-card setting is the
    exact failure mode this study has to rule out, so it must not be possible
    to ask for one and not get it.
    """
    import re
    lines = open(path).read().split('\n')
    seen = set()
    for i, l in enumerate(lines):
        m = re.match(r'^(\s*)(\S.*?)(\s*=\s*)(\w+)(\s*(?:!.*)?)$', l)
        if not m:
            continue
        key = m.group(4)
        if key in kv:
            val = str(kv[key])
            lines[i] = ' %s %s= %s%s' % (val, ' ' * max(0, 20 - len(val)),
                                         key, m.group(5))
            seen.add(key)
    missing = set(kv) - seen
    if missing:
        raise RuntimeError('run-card entries not found in %s: %s'
                           % (path, sorted(missing)))
    open(path, 'w').write('\n'.join(lines))


def copy_log(src, dst):
    """Copy a log next to the deliverables as ``.log.txt``.

    The repository ``.gitignore`` carries a blanket ``*.log``, so a log copied
    under its own name would be silently untracked.
    """
    if os.path.exists(src):
        shutil.copy(src, dst)


# --------------------------------------------------------------------------
# stage: prod
# --------------------------------------------------------------------------
def _generate(basedir, tag, proc, nb_core, logdir):
    outdir = pjoin(basedir, tag)
    if os.path.exists(pjoin(outdir, 'SubProcesses')):
        print('%s: reusing %s' % (tag, outdir))
        return outdir
    script = pjoin(basedir, 'gen_%s.mg5' % tag)
    with open(script, 'w') as fp:
        fp.write('set auto_convert_model T\n')
        fp.write('generate %s\n' % proc)
        fp.write('output %s\n' % outdir)
    rc, dt = sh([pjoin(_ROOT, 'bin', 'mg5_aMC'), script],
                pjoin(basedir, 'gen_%s.log' % tag), tag='output_%s' % tag)
    copy_log(pjoin(basedir, 'gen_%s.log' % tag),
             pjoin(logdir, 'gen_%s.log.txt' % tag))
    print('%s: output in %.0f s (rc=%d)' % (tag, dt, rc))
    if rc != 0:
        raise RuntimeError('generation of %s failed' % tag)
    # NOTE: MG5 compiles CutTools/IREGI inside the *source* tree the first time
    # a loop-induced output is made.  Two outputs started at once race in that
    # shared directory and one of them dies with a missing include; the two
    # generations are therefore serial here on purpose.
    return outdir


def stage_prod(basedir, nb_core, logdir):
    os.makedirs(basedir, exist_ok=True)
    info = {}

    dir_a = _generate(basedir, 'ggzz', PROC_A, nb_core, logdir)
    set_run_card(pjoin(dir_a, 'Cards', 'run_card.dat'), **RUN_CARD_COMMON)

    dir_b = _generate(basedir, 'gg4l', PROC_B, nb_core, logdir)
    card_b = pjoin(dir_b, 'Cards', 'run_card.dat')
    # ptheavy is hidden from a run card whose process has no heavy final state,
    # so it has to be written in explicitly before it can be set.
    txt = open(card_b).read()
    if 'ptheavy' not in txt:
        anchor = ' 0.0  = xptl ! minimum pt for at least one charged lepton \n'
        if anchor not in txt:
            raise RuntimeError('cannot find the anchor to insert ptheavy into '
                               + card_b)
        txt = txt.replace(anchor, anchor + ' 1.0  = ptheavy   ! minimum pt for '
                          'at least one heavy final state (inert here: no final '
                          'state above 10 GeV; read by the custom dummy_cuts '
                          'and applied to the reconstructed l+l- pairs)\n')
        open(card_b, 'w').write(txt)
    kv = dict(RUN_CARD_COMMON)
    kv.update(RUN_CARD_B_ONLY)
    set_run_card(card_b, **kv)
    txt = open(card_b).read()
    old = '  = custom_fcts ! List of files containing user hook function'
    if old in txt:
        txt = txt.replace(old, ' %s = custom_fcts ! List of files containing '
                          'user hook function' % CUTS_FILE)
        open(card_b, 'w').write(txt)

    for tag, procdir in (('A', dir_a), ('B', dir_b)):
        lhe = pjoin(procdir, 'Events', 'prod', 'unweighted_events.lhe.gz')
        if os.path.exists(lhe):
            print('sample %s: reusing %s' % (tag, lhe))
            info[tag] = lhe
            continue
        cmd = pjoin(basedir, 'run_%s.cmd' % tag)
        with open(cmd, 'w') as fp:
            fp.write('set run_mode 2\nset nb_core %d\ngenerate_events prod -f\n'
                     % nb_core)
        rc, dt = sh([sys.executable, pjoin(procdir, 'bin', 'madevent'), cmd],
                    pjoin(basedir, 'run_%s.log' % tag), cwd=procdir,
                    tag='sample_%s' % tag)
        copy_log(pjoin(basedir, 'run_%s.log' % tag),
                 pjoin(logdir, 'run_%s.log.txt' % tag))
        print('sample %s: %d events in %.0f s (rc=%d)' % (tag, NEVENTS, dt, rc))
        if rc != 0 or not os.path.exists(lhe):
            raise RuntimeError('sample %s failed' % tag)
        info[tag] = lhe
    return info


# --------------------------------------------------------------------------
# stage: madspin
# --------------------------------------------------------------------------
MS_DRIVER = '''\
import sys
sys.path.insert(0, %(root)r)
import MadSpin.interface_madspin as ims
cmd = ims.MadSpinInterface(%(lhe)r)
cmd.import_command_file(%(card)r)
print('MSRESULT', repr({'efficiency': cmd.efficiency, 'cross': cmd.cross,
                        'error': cmd.error, 'br': cmd.branching_ratio,
                        'br_err': cmd.err_branching_ratio}))
'''


def stage_madspin(basedir, lhe_a, modes, nb_core, logdir, reuse=True):
    out = {}
    for mode in modes:
        work = pjoin(basedir, 'ms_%s' % mode)
        os.makedirs(work, exist_ok=True)
        src = pjoin(work, 'events.lhe')
        decayed = pjoin(work, 'events_decayed.lhe.gz')
        if reuse and os.path.exists(decayed):
            print('%s: reusing %s' % (mode, decayed))
            out[mode] = decayed
            continue
        if not os.path.exists(src) and not os.path.exists(src + '.gz'):
            shutil.copy(lhe_a, src + '.gz')
            subprocess.check_call(['gunzip', '-f', src + '.gz'])
        card = pjoin(work, 'madspin_card.dat')
        with open(card, 'w') as fp:
            fp.write('set seed %d\n' % SEED_MADSPIN)
            fp.write('set BW_cut %d\n' % BW_CUT)
            fp.write('set spinmode %s\n' % mode)
            fp.write('set nb_core %d\n' % nb_core)
            fp.write('set ms_dir %s\n' % pjoin(work, 'msdir'))
            # The positional multi-channel rule: two z in the event and two
            # decay lines for z, so the first z takes the first line and the
            # second the second.  One (e+ e-) and one (mu+ mu-) per event --
            # which is what makes this comparable to sample B -- and NOT a
            # random draw that would give 4e and 4mu events.
            fp.write('decay z > e+ e-\n')
            fp.write('decay z > mu+ mu-\n')
            fp.write('launch\n')
        drv = pjoin(work, 'drive.py')
        open(drv, 'w').write(MS_DRIVER % {'root': _ROOT, 'lhe': src,
                                          'card': card})
        log = pjoin(work, 'madspin.log')
        rc, dt = sh([sys.executable, drv], log, cwd=work,
                    tag='madspin_%s' % mode)
        copy_log(log, pjoin(logdir, 'madspin_%s.log.txt' % mode))
        shutil.copy(card, pjoin(logdir, 'madspin_card_%s.dat' % mode))
        print('%s: %.0f s (rc=%d)' % (mode, dt, rc))
        if rc != 0 or not os.path.exists(decayed):
            raise RuntimeError('MadSpin mode %s failed, see %s' % (mode, log))
        out[mode] = decayed
    return out


def banner_cross(lhe):
    """MadEvent's own integrated cross section for ``lhe``, out of its banner.

    An independent number from the ``mean(w)`` the harvester computes off the
    events -- useful precisely because the two can disagree.  The banner does
    not carry the integration *error*, so that is read separately from the run's
    log (:func:`log_cross`).
    """
    import gzip
    import re
    op = gzip.open if lhe.endswith('.gz') else open
    out = {}
    with op(lhe, 'rt', errors='replace') as fh:
        for line in fh:
            if '<init>' in line:
                break
            m = re.search(r'Integrated weight \(pb\)\s*:\s*([-\d.eE+]+)', line)
            if m:
                out['sigma_pb'] = float(m.group(1))
            m = re.search(r'Number of Events\s*:\s*(\d+)', line)
            if m:
                out['nevents'] = int(m.group(1))
    return out


def banner_seed(lhe):
    """The seed a sample was ACTUALLY generated with, out of its own banner.

    Not the same thing as what was asked for.  MadEvent rewrites ``iseed`` to 0
    at the end of a run so that the next run of the same directory gets a fresh
    one, so the value sitting in ``Cards/run_card.dat`` afterwards is 0 and a
    second run started without re-setting ``iseed`` is auto-seeded.  That
    happened here: sample B's pilot consumed the requested 4321 and the 50 000
    event run was auto-seeded.  The banner is the only place the truth is kept.
    """
    import gzip
    import re
    op = gzip.open if lhe.endswith('.gz') else open
    with op(lhe, 'rt', errors='replace') as fh:
        for line in fh:
            if '<init>' in line:
                break
            m = re.search(r'([-\d]+)\s*=\s*iseed', line)
            if m:
                return int(m.group(1))
    return None


def log_cross(logpath):
    """``(sigma, error)`` from a MadEvent run log's "Results Summary" block.

    Returns ``None`` unless the log actually contains that block, i.e. unless
    the run reached the end.  A run that was cancelled mid-refine leaves a log
    full of plausible-looking intermediate numbers, and quoting one of those as
    a cross section would be worse than having no control at all -- it would
    look like a measurement.  The marker is checked first and the number is only
    read from after it.
    """
    import re
    if not os.path.exists(logpath):
        return None
    text = open(logpath, errors='replace').read()
    i = text.rfind('Results Summary')
    if i < 0:
        return None
    got = None
    for line in text[i:].split('\n'):
        m = re.search(r'Cross-section\s*:\s*([-\d.eE+]+)\s*\+-\s*([-\d.eE+]+)',
                      line)
        if m:
            got = (float(m.group(1)), float(m.group(2)))
    return got


def controls(basedir):
    """The evidence that sample B's cuts actually fire.

    A cut that is silently ignored -- a misspelt ``custom_fcts`` path, a
    function name the loader rejects, a run card entry that this process hides
    -- looks exactly like a cut that fires and removes nothing, and it would
    show up in this study as *better* agreement rather than worse.  So both cuts
    are measured against a control that differs from the real run in that cut
    and in nothing else:

    ``ctrl_ggzz_noptcut``
        sample A with ``ptheavy = 0``.  ``sigma(ptheavy=1)/sigma(ptheavy=0)`` is
        what the pt cut removes, on the side where it acts natively.

    ``ctrl_gg4l_nowindow``  -- NOT RUN TO COMPLETION, see RESULTS.md section 2d
        sample B with ``custom_fcts`` pointed at ``zz_ptonly_cuts.f``, i.e. the
        same pt cut and NO mass window.  Two numbers come out of it: the ratio
        of the two integrated cross sections, and -- from its own events, so
        without a second integration's error -- the fraction of its weight that
        falls inside the 15-width window.  The second is the retained fraction
        of the FULL rate (matrix element times Breit-Wigner times phase space),
        which is the quantity MadSpin's ``bw_retained_fraction`` approximates by
        the propagator part alone.  Their difference is the documented residual
        of that approximation, measured here on a process it has not been
        measured on.
    """
    import numpy as np
    sys.path.insert(0, _HERE)
    import observables as O

    out = {}
    got = log_cross(pjoin(basedir, 'ctrlA.log'))
    if got:
        out['A_no_ptcut'] = {'sigma_pb': got[0], 'error_pb': got[1]}
    got = log_cross(pjoin(basedir, 'ctrlB.log'))
    if got:
        out['B_no_masswindow'] = {'sigma_pb': got[0], 'error_pb': got[1]}
    # Only a COMPLETE control may contribute.  ``log_cross`` already refuses a
    # log without a "Results Summary" block, and the event file below only
    # exists once MadEvent has finished writing it, so a cancelled run
    # contributes nothing rather than contributing something half-formed.
    lhe = pjoin(basedir, 'ctrl_gg4l_nowindow', 'Events', 'prod',
                'unweighted_events.lhe.gz')
    if os.path.exists(lhe) and 'B_no_masswindow' in out:
        w, p = O.read_lhe(lhe)
        obs = O.compute(p)
        inside = ((np.abs(obs['m_ee'] - O.M_Z) < O.BW_CUT * O.W_Z)
                  & (np.abs(obs['m_mumu'] - O.M_Z) < O.BW_CUT * O.W_Z))
        # binomial on a ratio of subsets of the SAME sample: much tighter than
        # comparing two independent integrations, and free of their relative
        # normalisation
        frac = float(w[inside].sum() / w.sum())
        n = len(w)
        out['B_no_masswindow'] = dict(out.get('B_no_masswindow', {}), **{
            'nevents': int(n),
            'retained_fraction_measured': frac,
            'retained_fraction_error': float(np.sqrt(frac * (1 - frac) / n)),
            'm_ee_range': [float(obs['m_ee'].min()), float(obs['m_ee'].max())],
            'pt_ee_min': float(obs['pt_ee'].min()),
        })
    return out


def read_ms_result(logpath):
    for line in open(logpath, errors='replace'):
        if line.startswith('MSRESULT '):
            return eval(line[len('MSRESULT '):])
    return {}


# --------------------------------------------------------------------------
# stage: harvest
# --------------------------------------------------------------------------
def stage_harvest(files, datadir, extra_meta):
    sys.path.insert(0, _HERE)
    import numpy as np
    import observables as O

    hist = {}
    meta = {'runs': {}, 'bins': {k: v.tolist() for k, v in O.BINS.items()},
            'observables': list(O.BINS), 'code_sha': code_sha(),
            'm_Z': O.M_Z, 'width_Z': O.W_Z, 'BW_cut': O.BW_CUT,
            'mass_window': [O.M_LO, O.M_HI]}
    meta.update(extra_meta)

    for label, path in files.items():
        w, p = O.read_lhe(path)
        obs = O.compute(p)
        meta['runs'].setdefault(label, {})
        meta['runs'][label].update({
            'lhe': path, 'nevents': int(len(w)),
            'sumw': float(w.sum()), 'sigma_from_events': float(w.mean()),
            'sumw2': float((w ** 2).sum()),
            'sigma_mc_error': float(np.sqrt((w ** 2).sum()) / len(w)),
        })
        for name, edges in O.BINS.items():
            y, e = O.histogram(obs[name], w, edges)
            hist['%s/%s/y' % (label, name)] = y
            hist['%s/%s/e' % (label, name)] = e
        # a couple of scalar diagnostics that the plots do not show
        meta['runs'][label]['m_ee_range'] = [float(obs['m_ee'].min()),
                                             float(obs['m_ee'].max())]
        meta['runs'][label]['m_mumu_range'] = [float(obs['m_mumu'].min()),
                                               float(obs['m_mumu'].max())]
        meta['runs'][label]['pt_ee_min'] = float(obs['pt_ee'].min())
        # Angular moments, with the error of a weighted mean.  These compress
        # each angular figure to one number with an uncertainty, which is what
        # a "these curves lie on top of each other" claim needs in order to say
        # how tightly.
        def moment(v):
            mu = float(np.average(v, weights=w))
            var = float(np.average((v - mu) ** 2, weights=w))
            return [mu, float(np.sqrt(var / len(w)))]

        meta['runs'][label]['moments'] = {
            'cos_theta1': moment(obs['cos_theta1']),
            'cos2_theta1': moment(obs['cos_theta1'] ** 2),
            'cos1cos2': moment(obs['cos1cos2']),
            'cos2_theta2': moment(obs['cos_theta2'] ** 2),
            # the polarisation projectors of observables.py: <pol0_i> is the
            # longitudinal fraction of one z, <pol00> the joint one.  Kept as
            # moments rather than derived in the plotter because the error has
            # to come off the same events -- f_00 built from three separately
            # averaged moments would carry their covariance as an inflation.
            'pol0_1': moment(obs['pol0_1']),
            'pol0_2': moment(obs['pol0_2']),
            'pol00': moment(obs['pol00']),
            'polTT': moment(obs['polTT']),
            'cos_phi': moment(np.cos(obs['phi_planes'])),
            'cos_2phi': moment(np.cos(2 * obs['phi_planes'])),
            'm_epmum': moment(obs['m_epmum']),
        }

    os.makedirs(datadir, exist_ok=True)
    np.savez_compressed(pjoin(datadir, 'histograms.npz'), **hist)
    with open(pjoin(datadir, 'meta.json'), 'w') as fp:
        json.dump(meta, fp, indent=1, sort_keys=True)
    print('wrote %s' % pjoin(datadir, 'histograms.npz'))
    return meta


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['all', 'prod', 'madspin', 'harvest'])
    ap.add_argument('--basedir', default='/tmp/zz_loopinduced_work')
    ap.add_argument('--nb-core', type=int, default=6)
    ap.add_argument('--modes', default=','.join(MODES))
    ap.add_argument('--data', default=pjoin(_HERE, 'data'))
    ap.add_argument('--logs', default=pjoin(_HERE, 'logs'))
    args = ap.parse_args()

    global _TIMINGS
    os.makedirs(args.basedir, exist_ok=True)
    _TIMINGS = pjoin(args.basedir, 'timings.json')
    os.makedirs(args.logs, exist_ok=True)
    modes = args.modes.split(',')
    basedir = args.basedir

    lhe_a = pjoin(basedir, 'ggzz', 'Events', 'prod', 'unweighted_events.lhe.gz')
    lhe_b = pjoin(basedir, 'gg4l', 'Events', 'prod', 'unweighted_events.lhe.gz')

    if args.stage in ('all', 'prod'):
        got = stage_prod(basedir, args.nb_core, args.logs)
        lhe_a, lhe_b = got['A'], got['B']

    if args.stage in ('all', 'madspin'):
        stage_madspin(basedir, lhe_a, modes, args.nb_core, args.logs)

    if args.stage in ('all', 'harvest'):
        # The undecayed production sample is deliberately NOT harvested here.
        # Every observable of this study is built from four charged leptons and
        # sample A has none before MadSpin runs, so there is nothing to
        # histogram; its total cross section is recorded below instead, which is
        # the only number from it that the comparison uses.
        files = {'truth': lhe_b}
        for mode in modes:
            p = pjoin(basedir, 'ms_%s' % mode, 'events_decayed.lhe.gz')
            if os.path.exists(p):
                files[mode] = p
        # The seed asked for, and the seed each sample was actually generated
        # with -- see banner_seed(): they are not always the same, and only the
        # second one reproduces a sample.
        seeds = {'requested_production': SEED_PROD, 'madspin_all_modes':
                 SEED_MADSPIN}
        for key, path in (('sample_A_actual', lhe_a), ('sample_B_actual', lhe_b)):
            if os.path.exists(path):
                seeds[key] = banner_seed(path)
        ctrl_a = pjoin(basedir, 'ctrl_ggzz_noptcut', 'Events', 'prod',
                       'unweighted_events.lhe.gz')
        ctrl_b = pjoin(basedir, 'ctrl_gg4l_nowindow', 'Events', 'prod',
                       'unweighted_events.lhe.gz')
        for key, path in (('control_A_actual', ctrl_a),
                          ('control_B_actual', ctrl_b)):
            if os.path.exists(path):
                seeds[key] = banner_seed(path)
        extra = {'seeds': seeds,
                 'processes': {'A': PROC_A, 'B': PROC_B},
                 'run_card_common': RUN_CARD_COMMON,
                 'run_card_B_only': RUN_CARD_B_ONLY,
                 'madspin_modes': modes,
                 'cuts_file': CUTS_FILE,
                 'reported': {}}
        for mode in modes:
            log = pjoin(basedir, 'ms_%s' % mode, 'madspin.log')
            if os.path.exists(log):
                extra['reported'][mode] = read_ms_result(log)
        extra['controls'] = controls(basedir)
        if os.path.exists(_TIMINGS):
            extra['wall_times'] = json.load(open(_TIMINGS))
        try:
            sys.path.insert(0, _ROOT)
            from MadSpin.decay import bw_retained_fraction
            extra['bw_retained_fraction'] = {
                'f': bw_retained_fraction(91.1880, 2.441404, BW_CUT),
                'source': 'MadSpin.decay.bw_retained_fraction(m_Z, Gamma_Z, %d)'
                          % BW_CUT}
            extra['bw_retained_fraction']['f_squared'] = \
                extra['bw_retained_fraction']['f'] ** 2
        except ImportError:
            pass
        extra['production'] = banner_cross(lhe_a)
        extra['banner_cross'] = {'truth': banner_cross(lhe_b)}
        extra['integration_error'] = {}
        for tag, key in (('A', 'production'), ('B', 'truth')):
            got = log_cross(pjoin(basedir, 'run_%s.log' % tag))
            if got:
                extra['integration_error'][key] = got[1]
                extra.setdefault('integration_sigma', {})[key] = got[0]
        stage_harvest(files, args.data, extra)


if __name__ == '__main__':
    main()
