#!/usr/bin/env python3
"""Systematic sweep of MadSpin's ``ms_dir`` against every spinmode.

For each (spinmode, card ordering) it runs MadSpin twice on the *same*
``ms_dir``: once on a fresh (non-existent) directory, once reusing what the
first run built.  Each run gets its own working directory holding its own copy
of the production events, so the two runs differ only in the state of the
``ms_dir``.

Results are appended to results.jsonl as soon as each run finishes, so an
interrupted sweep costs one row rather than everything.
"""
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
MG5DIR = os.path.dirname(HERE)
WORK = os.path.join(HERE, 'work')
EVENTS = os.path.join(WORK, 'PROC_tt', 'Events', 'run_01', 'unweighted_events.lhe.gz')
RESULTS = os.path.join(WORK, 'results.jsonl')

# cross-section of the undecayed sample, read once from its banner
PROD_XSEC = None

DECAY_LINES = [
    'decay t > w+ b, w+ > e+ ve',
    'decay t~ > w- b~, w- > e- ve~',
]


def read_init(path):
    """(xsec, nb_events, [weights]) of an LHE file (gzipped or not)."""
    opener = gzip.open if path.endswith('.gz') else open
    xsec = None
    nev = 0
    wgts = []
    in_init = False
    in_event = False
    init_lines = 0
    with opener(path, 'rt', errors='replace') as fsock:
        for line in fsock:
            s = line.strip()
            if s.startswith('<init'):
                in_init = True
                init_lines = 0
                continue
            if s.startswith('</init>'):
                in_init = False
                continue
            if in_init:
                init_lines += 1
                if init_lines == 2:
                    try:
                        xsec = float(s.split()[0])
                    except (IndexError, ValueError):
                        pass
                continue
            if s.startswith('<event'):
                in_event = True
                nev += 1
                first = True
                continue
            if s.startswith('</event>'):
                in_event = False
                continue
            if in_event and first:
                first = False
                try:
                    wgts.append(float(s.split()[2]))
                except (IndexError, ValueError):
                    pass
    return xsec, nev, wgts


def summarise(wgts):
    if not wgts:
        return dict(n_wgt=0, wgt_mean=None, wgt_abs_mean=None, wgt_sum=None,
                    n_zero=0)
    n_zero = sum(1 for w in wgts if w == 0.0)
    return dict(n_wgt=len(wgts),
                wgt_mean=sum(wgts) / len(wgts),
                wgt_abs_mean=sum(abs(w) for w in wgts) / len(wgts),
                wgt_sum=sum(wgts),
                n_zero=n_zero)


def make_card(path, evtfile, spinmode, order, ms_dir, use_old_dir, seed=33):
    lines = []
    setms = []
    if ms_dir:
        setms.append('set ms_dir %s' % ms_dir)
    if use_old_dir:
        setms.append('set use_old_dir True')
    if order == 'import_first':
        lines.append('import %s' % evtfile)
        lines.extend(setms)
    else:                       # set ms_dir before import
        lines.extend(setms)
        lines.append('import %s' % evtfile)
    lines.append('set spinmode %s' % spinmode)
    lines.append('set seed %d' % seed)
    lines.append('set max_weight_ps_point 100')
    lines.extend(DECAY_LINES)
    lines.append('launch')
    with open(path, 'w') as fsock:
        fsock.write('\n'.join(lines) + '\n')
    return lines


def run_one(tag, spinmode, order, ms_dir, phase, use_old_dir=False,
            rundir=None, timeout=5400):
    """One MadSpin run.  Returns the record dict (already written to disk)."""
    if rundir is None:
        rundir = os.path.join(WORK, 'runs', tag)
        if os.path.exists(rundir):
            shutil.rmtree(rundir)
        os.makedirs(rundir)
    evt = os.path.join(rundir, 'events.lhe.gz')
    # a fresh copy every run: MadSpin gunzips/gzips its input in place
    for stale in glob.glob(os.path.join(rundir, 'events*.lhe*')):
        os.remove(stale)
    shutil.copy(EVENTS, evt)
    card = os.path.join(rundir, 'madspin_card.dat')
    cardlines = make_card(card, evt, spinmode, order, ms_dir, use_old_dir)
    log = os.path.join(rundir, 'madspin.log')
    env = dict(os.environ)
    env['PATH'] = os.path.expanduser('~/.pyenv/versions/mg-3.14/bin') + \
        os.pathsep + env.get('PATH', '')
    start = time.time()
    timedout = False
    with open(log, 'w') as fsock:
        try:
            rc = subprocess.call([sys.executable, os.path.join(MG5DIR, 'MadSpin', 'madspin'),
                                  '-f', card],
                                 cwd=rundir, stdout=fsock, stderr=subprocess.STDOUT,
                                 env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = -99
            timedout = True
    elapsed = time.time() - start

    out = os.path.join(rundir, 'events_decayed.lhe.gz')
    rec = dict(tag=tag, spinmode=spinmode, order=order, phase=phase,
               use_old_dir=use_old_dir, ms_dir=ms_dir, rundir=rundir,
               exit_code=rc, timeout=timedout, seconds=round(elapsed, 1),
               output=out, output_exists=os.path.exists(out))
    if rec['output_exists']:
        try:
            xsec, nev, wgts = read_init(out)
        except Exception as error:               # noqa: BLE001
            rec['read_error'] = repr(error)
        else:
            rec['init_xsec'] = xsec
            rec['nb_events'] = nev
            rec.update(summarise(wgts))
            if PROD_XSEC and xsec is not None:
                rec['implied_br'] = xsec / PROD_XSEC
    # what else landed in the run dir / ms_dir
    rec['rundir_files'] = sorted(os.path.basename(f) for f in
                                 glob.glob(os.path.join(rundir, '*')))
    if ms_dir and os.path.exists(ms_dir):
        rec['ms_dir_entries'] = sorted(os.listdir(ms_dir))[:25]
        rec['ms_dir_has_decayed_events'] = os.path.exists(
            os.path.join(ms_dir, 'decayed_events.lhe'))
    # tail of the log for triage
    with open(log, errors='replace') as fsock:
        text = fsock.read()
    rec['log_tail'] = text[-2500:]
    for key in ('branching_ratio', 'Branching ratio'):
        pass
    with open(RESULTS, 'a') as fsock:
        fsock.write(json.dumps(rec) + '\n')
    print('[%s] rc=%s xsec=%s nev=%s wgt_mean=%s (%.0fs)' % (
        tag, rc, rec.get('init_xsec'), rec.get('nb_events'),
        rec.get('wgt_mean'), elapsed), flush=True)
    return rec


def main(argv):
    global PROD_XSEC
    PROD_XSEC = read_init(EVENTS)[0]
    print('production <init> xsec = %s' % PROD_XSEC, flush=True)

    combos = argv[1:]
    if not combos:
        combos = ['all']

    spinmodes = ['madspin', 'PA', 'onshell', 'madspin_v1', 'onshell_v1',
                 'none', 'full']
    if combos != ['all']:
        spinmodes = combos

    os.makedirs(os.path.join(WORK, 'runs'), exist_ok=True)
    msroot = os.path.join(WORK, 'msdirs')
    os.makedirs(msroot, exist_ok=True)

    for mode in spinmodes:
        # (0) reference: no ms_dir at all
        run_one('%s__noms' % mode, mode, 'import_first', '', 'baseline')
        for order in ('import_first', 'ms_dir_first'):
            ms_dir = os.path.join(msroot, '%s__%s' % (mode, order))
            if os.path.exists(ms_dir):
                shutil.rmtree(ms_dir)
            run_one('%s__%s__fresh' % (mode, order), mode, order, ms_dir, 'fresh')
            run_one('%s__%s__reuse' % (mode, order), mode, order, ms_dir, 'reuse')


if __name__ == '__main__':
    main(sys.argv)
