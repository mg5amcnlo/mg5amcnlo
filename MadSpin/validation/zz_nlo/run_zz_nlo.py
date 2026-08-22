#!/usr/bin/env python3
"""Drive the ``p p > z z`` NLO + MadSpin study, and the ``gg`` stack on top of it.

This is the NLO continuation of ``MadSpin/validation/zz_loopinduced``: the same
process, the same cuts, the same scale, the same observables, one order higher.
It produces four samples and one set of MadSpin runs.

Stages, selectable with ``--stage``:

``prod``
    the four samples, all with the SAME cuts, scale, PDF and statistics --
    because two ratio panes divide three of them by each other:

    * **NLO** -- ``p p > z z [QCD]``, MC@NLO events (``launch aMC@NLO -p``).
      The Born is ``q q~ -> Z Z``; the real emission adds ``q g`` and ``g q~``.
      There is no ``g g`` initial state anywhere in it, which is the
      double-counting check and is verified on the written events, not asserted.
    * **LO** -- ``p p > z z``, the curve both ratio panes divide by.
    * **LI** -- ``g g > z z [noborn=QCD]``, the loop-induced ``g g`` box, the
      contribution stacked on top of NLO.
    * **truth** -- ``p p > e+ e- mu+ mu- / a [QCD]``, the fully off-shell
      four-lepton reference the MadSpin comparison divides by.  Measured cost
      before committing to it: 88 s for 50 000 events, i.e. affordable.

``controls``
    each cut measured against a run that differs from the real one in that cut
    and in nothing else, plus the fixed-order NLO cross-check on the total.

``madspin``
    MadSpin over the NLO sample in every mode the sample allows.  Unlike the
    loop-induced study, this one is NOT loop induced, so ``madspin_v1`` and
    ``onshell_v1`` are available too and all six modes run.  ``fixed_order`` is
    off and is not needed: these are MC@NLO events, individual events carrying
    (possibly negative) weights, not the counter-event groups a fixed-order LHE
    carries.

``harvest``
    read every LHE and write ``data/histograms.npz`` -- which holds both halves,
    the four-lepton observables of the MadSpin comparison and the
    production-level ``Z Z`` observables of the physics figure -- plus
    ``data/meta.json``.  The two ``numbers*.txt`` reports are written by the
    plotting scripts, so that what they print and what they draw cannot drift
    apart.

Usage::

    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"     # f2py is required
    python3 run_zz_nlo.py --stage all --basedir /tmp/zz_nlo_work --nb-core 6

The plotting scripts run off ``data/`` alone and need neither MadGraph nor this
script.

Two run-card facts that cost time and are worth having written down
------------------------------------------------------------------
* **The NLO sample uses the NLO run card**, and the LO and loop-induced ones use
  the LO card.  They are different classes with different parameters:
  ``drll_sf`` and ``mll_sf`` exist only on ``RunCardNLO``; ``ptheavy``,
  ``use_syst`` and ``sde_strategy`` only on ``RunCardLO``.  The one cut
  parameter that exists on **both** is ``pt_min_pdg``, which is why the
  ``pt(Z) > 1 GeV`` of this study is applied through it on all four samples
  rather than through the loop-induced study's ``ptheavy``.
* ``pt_min_pdg`` is an AND over the particles carrying that PDG on both cards
  (``setcuts.f`` writes it into the per-particle ``etmin`` array).  At ``2 -> 2``
  the two ``Z`` have identical ``pt``, so on the LO and loop-induced samples it
  is the same cut as ``ptheavy = 1`` -- and the loop-induced cross section
  reproduces the value the previous study measured with ``ptheavy``, which is
  the check that the two really are the same cut.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))

pjoin = os.path.join


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
PROC = {
    'nlo': 'p p > z z [QCD]',
    'lo':  'p p > z z',
    'li':  'g g > z z [noborn=QCD]',
    # the off-shell reference: the four leptons come out of the NLO matrix
    # element directly, with both Z propagators off shell and the two decays
    # correlated by construction.  The photon is excluded, as in the
    # loop-induced study, so this tests Z spin correlations and not Z/gamma*
    # interference.
    'truth': 'p p > e+ e- mu+ mu- / a [QCD]',
}
OUTDIR = {'nlo': 'ppzz_nlo', 'lo': 'ppzz_lo', 'li': 'ggzz_li',
          'truth': 'pp4l_nlo'}

# The truth sample has no z to cut on, so ``pt_min_pdg`` is natively inert there
# and is read by this file instead and applied to the reconstructed (l+ l-)
# systems, together with the Breit-Wigner mass window.  Same idea as the
# loop-induced study's ``zz_equivalent_cuts.f``, different dummy_cuts signature.
CUTS_FILE_NLO = pjoin(_HERE, 'zz_equivalent_cuts_nlo.f')

SEED_PROD = 4321
SEED_MADSPIN = 7777
NEVENTS = 50000
BW_CUT = 15
PT_Z_MIN = 1.0
MU = '91.1880'                  # m_Z, as the fixed renormalisation and fact. scale

# Every mode MadSpin allows for this (non loop-induced) sample.  The
# loop-induced study could only run four; here the two ``_v1`` legacy modes are
# available as well, and running them is half the point of doing this at NLO.
MODES = ['none', 'madspin', 'onshell', 'PA', 'madspin_v1', 'onshell_v1']

# --- the LO run card, used by the LO and the loop-induced samples ----------
RUN_CARD_LO = {
    'nevents': NEVENTS,
    'iseed': SEED_PROD,
    'fixed_ren_scale': 'True',
    'fixed_fac_scale': 'True',
    'scale': MU,
    'dsqrt_q2fact1': MU,
    'dsqrt_q2fact2': MU,
    'bwcutoff': '15.0',
    'use_syst': 'False',
    'pdlabel': 'nn23lo1',
    'lhaid': '230000',
}

# --- the NLO run card -----------------------------------------------------
# Same physics content, different parameter names.  ``muR_ref_fixed`` /
# ``muF_ref_fixed`` are the NLO card's ``scale`` / ``dsqrt_q2fact*``; the
# reweighting switches replace ``use_syst``; and the per-lepton cuts have to be
# named here even though this process has no lepton, because the audit in
# RESULTS.md is read off the card that was actually used.
RUN_CARD_NLO = {
    'nevents': NEVENTS,
    'iseed': SEED_PROD,
    'fixed_ren_scale': 'True',
    'fixed_fac_scale': 'True',
    'muR_ref_fixed': MU,
    'muF_ref_fixed': MU,
    'bwcutoff': '15.0',
    # -1 = auto, i.e. the accuracy the requested number of events implies.  Set
    # explicitly because a pilot run in the same directory can leave a looser
    # value behind, and a silently looser integration is exactly the kind of
    # difference between two samples this study must not have.
    'req_acc': '-1.0',
    'reweight_scale': 'False',
    'reweight_PDF': 'False',
    'store_rwgt_info': 'False',
    'parton_shower': 'PYTHIA8',
    # the same PDF as the LO and loop-induced samples.  An LO PDF at NLO is not
    # the conventional choice; using the same one on all three is, because the
    # figure's two ratio panes are meant to show a change of ORDER and not a
    # change of PDF set.  See RESULTS.md.
    'pdlabel': 'nn23lo1',
    'lhaid': '230000',
    # per-lepton cuts off, exactly as in the loop-induced study.  This process
    # has no lepton, so they are inert -- but mll_sf defaults to 30 GeV on the
    # NLO card and would NOT be inert on a four-lepton NLO sample, which is the
    # trap this study inherits.
    'ptl': '0.0',
    'etal': '-1.0',
    'drll': '0.0',
    'drll_sf': '0.0',
    'mll': '0.0',
    'mll_sf': '0.0',
}

# The pt(Z) cut, set through the dict-valued run-card entry that exists on BOTH
# card classes.  Written separately from the scalar entries above because
# ``set_run_card`` matches ``value = key`` lines and this one's value is a dict.
PT_MIN_PDG = "{23: %s}" % PT_Z_MIN


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
    if tag:
        record_timing(tag, dt, rc)
    return rc, dt


def record_timing(tag, seconds, rc=0, note=None):
    """Append one wall time to ``<basedir>/timings.json``.

    On disk rather than in memory: the stages are routinely run as separate
    invocations (and the MadSpin modes in parallel), so the harvest that writes
    meta.json is usually a different process from the one that did the work.
    """
    if not _TIMINGS:
        return
    try:
        cur = json.load(open(_TIMINGS)) if os.path.exists(_TIMINGS) else {}
    except ValueError:
        cur = {}
    cur[tag] = {'wall_seconds': round(seconds, 1), 'returncode': rc}
    if note:
        cur[tag]['note'] = note
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
    """Rewrite scalar ``value = key`` entries of a run card in place.

    Raises if a key is not present.  A silently-ignored run-card setting is the
    exact failure mode this study has to rule out -- the two ratio panes divide
    three samples by each other and a cut that landed on only two of them would
    look like physics -- so it must not be possible to ask for one and not get
    it.
    """
    lines = open(path).read().split('\n')
    # Case-INSENSITIVE on the key, and the file's own spelling is kept.  MG5
    # writes the template's mixed case (``muR_ref_fixed``, ``reweight_PDF``) into
    # a fresh card and rewrites the card in lower case once a run has used it, so
    # a case-sensitive match works on a fresh directory and fails on a reused
    # one -- and fails by raising "entry not found" on a parameter that is
    # plainly there.
    want = {k.lower(): k for k in kv}
    seen = set()
    for i, l in enumerate(lines):
        m = re.match(r'^(\s*)(\S.*?)(\s*=\s*)([A-Za-z_]\w*)(\s*(?:!.*)?)$', l)
        if not m:
            continue
        key = m.group(4)
        if key.lower() in want:
            val = str(kv[want[key.lower()]])
            lines[i] = ' %s %s= %s%s' % (val, ' ' * max(0, 20 - len(val)),
                                         key, m.group(5))
            seen.add(want[key.lower()])
    missing = set(kv) - seen
    if missing:
        raise RuntimeError('run-card entries not found in %s: %s'
                           % (path, sorted(missing)))
    open(path, 'w').write('\n'.join(lines))


def set_pt_min_pdg(path, value):
    """Set the dict-valued ``pt_min_pdg`` entry, on either card class."""
    lines = open(path).read().split('\n')
    done = False
    for i, l in enumerate(lines):
        if re.match(r'^\s*\{.*\}\s*=\s*pt_min_pdg\b', l):
            comment = l.split('!', 1)
            tail = ' !' + comment[1] if len(comment) > 1 else ''
            lines[i] = ' %s = pt_min_pdg%s' % (value, tail)
            done = True
    if not done:
        raise RuntimeError('pt_min_pdg not found in %s' % path)
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
def generate_outputs(basedir, tags, logdir):
    """``output`` the requested processes, SERIALLY.

    Serial on purpose: MG5 builds CutTools and IREGI inside the *source* tree
    the first time a loop-induced or NLO output is made, and two outputs started
    at once race in that shared directory -- one dies with
    ``cp: includects/avh_olo.f90: No such file or directory``.
    """
    todo = [t for t in tags
            if not os.path.exists(pjoin(basedir, OUTDIR[t], 'SubProcesses'))]
    if not todo:
        return
    script = pjoin(basedir, 'gen.mg5')
    with open(script, 'w') as fp:
        fp.write('set auto_convert_model T\n')
        for t in todo:
            fp.write('generate %s\n' % PROC[t])
            fp.write('output %s\n' % pjoin(basedir, OUTDIR[t]))
    log = pjoin(basedir, 'gen.log')
    rc, dt = sh([pjoin(_ROOT, 'bin', 'mg5_aMC'), script], log,
                tag='output_%s' % '_'.join(todo))
    copy_log(log, pjoin(logdir, 'gen_%s.log.txt' % '_'.join(todo)))
    if rc != 0:
        raise RuntimeError('generation failed, see %s' % log)


def prepare_card(basedir, tag):
    """Write this sample's run card and return its path."""
    card = pjoin(basedir, OUTDIR[tag], 'Cards', 'run_card.dat')
    set_run_card(card, **(RUN_CARD_NLO if tag in ('nlo', 'truth')
                          else RUN_CARD_LO))
    set_pt_min_pdg(card, PT_MIN_PDG)
    if tag == 'truth':
        txt = open(card).read()
        # Matched by regex, not by literal: MG5 writes the entry with the
        # template's spacing in a fresh directory and re-emits it with a tab
        # once a run has used the card.
        m = re.search(r'^[^\n!]*=\s*custom_fcts\b', txt, re.M)
        if not m:
            raise RuntimeError('no custom_fcts entry in %s' % card)
        # The loader matches function names CASE-SENSITIVELY against a lowercase
        # table, so the file must spell ``logical function dummy_cuts`` in lower
        # case; ``LOGICAL FUNCTION DUMMY_CUTS`` is rejected with the unhelpful
        # message ``function %s is not designed for overwritting`` -- with a
        # literal, unformatted %s.
        txt = (txt[:m.start()] + ' %s = custom_fcts' % CUTS_FILE_NLO
               + txt[m.end():])
        open(card, 'w').write(txt)
    return card


def run_sample(basedir, tag, nb_core, logdir, run_name='prod'):
    """Generate one sample; returns the path of its LHE."""
    procdir = pjoin(basedir, OUTDIR[tag])
    lhe = sample_lhe(basedir, tag, run_name)
    if lhe:
        print('%s: reusing %s' % (tag, lhe))
        return lhe
    cmd = pjoin(basedir, 'run_%s.cmd' % tag)
    with open(cmd, 'w') as fp:
        fp.write('set run_mode 2\nset nb_core %d\n' % nb_core)
        if tag in ('nlo', 'truth'):
            # aMC@NLO, stopped at the parton level (-p): MC@NLO events, no
            # shower.  The MC@NLO subtraction terms are those of the shower named
            # in the run card, so the shower choice is recorded in meta.json.
            fp.write('launch aMC@NLO -p -f -n %s\n' % run_name)
            exe = pjoin(procdir, 'bin', 'aMCatNLO')
        else:
            fp.write('generate_events %s -f\n' % run_name)
            exe = pjoin(procdir, 'bin', 'madevent')
    # Named after the RUN and not just the sample: a control reuses the sample's
    # directory, and a log named after the sample would overwrite the production
    # run's -- silently, and only noticed later when the production integration
    # error has gone missing from meta.json.
    log = pjoin(basedir, 'run_%s.log' % (tag if run_name == 'prod'
                                         else run_name))
    rc, dt = sh([sys.executable, exe, cmd], log, cwd=procdir,
                tag='sample_%s' % (tag if run_name == 'prod' else run_name))
    copy_log(log, pjoin(logdir, 'run_%s.log.txt'
                        % (tag if run_name == 'prod' else run_name)))
    print('%s: %.0f s (rc=%d)' % (tag, dt, rc))
    lhe = sample_lhe(basedir, tag, run_name)
    if rc != 0 or not lhe:
        raise RuntimeError('sample %s failed, see %s' % (tag, log))
    return lhe


def sample_lhe(basedir, tag, run_name='prod'):
    """The LHE of a finished sample, or ``None``.

    LO/loop-induced (MadEvent) write ``unweighted_events.lhe.gz``; aMC@NLO
    writes ``events.lhe.gz``.  Both are looked for rather than assumed.
    """
    ev = pjoin(basedir, OUTDIR[tag], 'Events', run_name)
    for name in ('unweighted_events.lhe.gz', 'events.lhe.gz',
                 'unweighted_events.lhe', 'events.lhe'):
        p = pjoin(ev, name)
        if os.path.exists(p):
            return p
    return None


def stage_prod(basedir, nb_core, logdir, tags=('nlo', 'lo', 'li', 'truth')):
    os.makedirs(basedir, exist_ok=True)
    generate_outputs(basedir, tags, logdir)
    out = {}
    for tag in tags:
        prepare_card(basedir, tag)
        out[tag] = run_sample(basedir, tag, nb_core, logdir)
        shutil.copy(pjoin(basedir, OUTDIR[tag], 'Cards', 'run_card.dat'),
                    pjoin(logdir, 'run_card_%s.dat' % tag))
    return out


# --------------------------------------------------------------------------
# stage: controls
# --------------------------------------------------------------------------
# A cut that is silently ignored looks exactly like a cut that fires and removes
# nothing -- and in a three-sample stack it would show up as physics.  Each
# control differs from the real run in ONE thing and in nothing else.
CONTROLS = {
    # what the pt(Z) cut removes at NLO, on the sample it acts on natively
    'nlo_no_ptcut': {'tag': 'nlo', 'pt_min_pdg': '{}',
                     'note': 'p p > z z [QCD] with pt_min_pdg = {} (no pt cut)'},
    # the same cut written the loop-induced study's way.  ptheavy is an
    # RunCardLO-only parameter and an OR over the heavy final states, while
    # pt_min_pdg is an AND -- at 2 -> 2 they must give the same number, and this
    # measures that they do rather than arguing it.
    'lo_ptheavy': {'tag': 'lo', 'pt_min_pdg': '{}', 'card': {'ptheavy': '1.0'},
                   'note': 'p p > z z at LO with ptheavy = 1 instead of '
                           'pt_min_pdg = {23: 1}'},
    # what MG5's own NLO lepton-cut defaults would have cost.  At LO they are
    # ptl = 10 / etal = 2.5 / drll = 0.4 and remove 46 % of a four-lepton rate;
    # the NLO card's defaults are a different set and this measures them.
    'truth_default_lepton_cuts': {
        'tag': 'truth',
        'card': {'ptl': '0.0', 'etal': '-1.0', 'drll': '0.0',
                 'drll_sf': '0.0', 'mll': '0.0', 'mll_sf': '30.0'},
        'note': "p p > e+ e- mu+ mu- / a [QCD] with MG5's NLO default "
                "mll_sf = 30 restored"},
}


def stage_controls(basedir, which, nb_core, logdir):
    """Run the controls, each in the sample directory it belongs to.

    Serial by construction: they reuse the production directories, so two at
    once in the same one would fight over ``Cards/run_card.dat``.
    """
    for name in which:
        spec = CONTROLS[name]
        tag = spec['tag']
        run = 'ctrl_' + name
        if sample_lhe(basedir, tag, run):
            print('%s: reusing' % name)
            continue
        prepare_card(basedir, tag)
        card = pjoin(basedir, OUTDIR[tag], 'Cards', 'run_card.dat')
        if 'pt_min_pdg' in spec:
            set_pt_min_pdg(card, spec['pt_min_pdg'])
        if 'card' in spec:
            set_run_card(card, **spec['card'])
        run_sample(basedir, tag, nb_core, logdir, run_name=run)
        shutil.copy(card, pjoin(logdir, 'run_card_%s.dat' % name))
    # leave every card back in its production state, so a later --stage prod
    # rerun cannot inherit a control's setting
    for tag in set(CONTROLS[n]['tag'] for n in which):
        prepare_card(basedir, tag)


def fixed_order_nlo(basedir, nb_core, logdir):
    """``p p > z z [QCD]`` at FIXED ORDER, as a cross-check on the total.

    The MC@NLO event sample's cross section should equal the fixed-order NLO
    one: the MC counterterms integrate to the shower's own O(alpha_s)
    contribution and cancel in the total.  Distributions do not have to agree
    (the parton-level MC@NLO sample carries the subtraction), but the number
    does, and it is the cheapest available check that the event sample really is
    the NLO cross section.
    """
    log = pjoin(basedir, 'run_nlo_fixedorder.log')
    if os.path.exists(log) and log_cross(log):
        print('fixed order: reusing')
        return
    procdir = pjoin(basedir, OUTDIR['nlo'])
    prepare_card(basedir, 'nlo')
    cmd = pjoin(basedir, 'run_nlo_fo.cmd')
    with open(cmd, 'w') as fp:
        fp.write('set run_mode 2\nset nb_core %d\n' % nb_core)
        fp.write('launch NLO -f -n fixedorder\n')
    sh([sys.executable, pjoin(procdir, 'bin', 'aMCatNLO'), cmd], log,
       cwd=procdir, tag='nlo_fixed_order')
    copy_log(log, pjoin(logdir, 'run_nlo_fixedorder.log.txt'))
    print('fixed order: %s' % (log_cross(log),))


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


def stage_madspin(basedir, lhe, modes, nb_core, logdir, reuse=True):
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
            shutil.copy(lhe, src + '.gz')
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
            # second the second.  One (e+ e-) and one (mu+ mu-) per event, never
            # 4e or 4mu -- the same convention as the loop-induced study, so the
            # two sets of figures mean the same thing.
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


# --------------------------------------------------------------------------
# reading numbers back out of the runs
# --------------------------------------------------------------------------
def banner_field(lhe, pattern, cast=float):
    import gzip
    op = gzip.open if lhe.endswith('.gz') else open
    got = None
    with op(lhe, 'rt', errors='replace') as fh:
        for line in fh:
            if '<init>' in line:
                break
            m = re.search(pattern, line)
            if m:
                got = cast(m.group(1))
    return got


def banner_cross(lhe):
    """MadEvent's / aMC@NLO's own integrated cross section, out of the banner."""
    out = {}
    s = banner_field(lhe, r'Integrated weight \(pb\)\s*:\s*([-\d.eE+]+)')
    if s is not None:
        out['sigma_pb'] = s
    n = banner_field(lhe, r'Number of Events\s*:\s*(\d+)', int)
    if n is not None:
        out['nevents'] = n
    return out


def banner_seed(lhe):
    """The seed a sample was ACTUALLY generated with, out of its own banner.

    Not the same thing as what was asked for: MadEvent rewrites ``iseed`` to 0
    at the end of a run, so a second run in the same directory is auto-seeded and
    the value sitting in ``Cards/run_card.dat`` afterwards is a lie.
    """
    return banner_field(lhe, r'([-\d]+)\s*=\s*iseed', int)


def log_cross(logpath):
    """``(sigma, error)`` from a run log, only if the run reached the end.

    Both spellings are accepted: MadEvent prints a "Results Summary" block with
    ``Cross-section : X +- Y``, aMC@NLO prints ``Total cross section: X +- Y pb``.
    A run cancelled mid-refine leaves a log full of plausible intermediate
    numbers, so the terminal marker is required before any number is read.
    """
    if not os.path.exists(logpath):
        return None
    text = open(logpath, errors='replace').read()
    got = None
    i = text.rfind('Results Summary')
    if i >= 0:
        for line in text[i:].split('\n'):
            m = re.search(r'Cross-section\s*:\s*([-\d.eE+]+)\s*\+-\s*([-\d.eE+]+)',
                          line)
            if m:
                got = (float(m.group(1)), float(m.group(2)))
    if got:
        return got
    for line in text.split('\n'):
        m = re.search(r'Total cross section:\s*([-\d.eE+]+)\s*\+-\s*([-\d.eE+]+)',
                      line)
        if m:
            got = (float(m.group(1)), float(m.group(2)))
    return got


def run_cross(basedir, tag, run_name='prod'):
    """``(sigma, error)`` for one run, from the artefacts that run itself left.

    Preferred over :func:`log_cross`, because these live under the run's own
    name and a later run of the same process directory cannot overwrite them:

    * aMC@NLO writes ``Events/<run>/summary.txt``;
    * MadEvent writes ``HTML/<run>/results.html`` with ``s= X &#177 Y (pb)``.

    Falls back to the driver's own log, which is right for a run whose directory
    has not been reused.
    """
    p = pjoin(basedir, OUTDIR[tag], 'Events', run_name, 'summary.txt')
    if os.path.exists(p):
        m = re.search(r'Total cross section:\s*([-\d.eE+]+)\s*\+-\s*([-\d.eE+]+)',
                      open(p, errors='replace').read())
        if m:
            return float(m.group(1)), float(m.group(2))
    p = pjoin(basedir, OUTDIR[tag], 'HTML', run_name, 'results.html')
    if os.path.exists(p):
        m = re.search(r's=\s*([-\d.eE+]+)\s*&#177;?\s*([-\d.eE+]+)\s*\(pb\)',
                      open(p, errors='replace').read())
        if m:
            return float(m.group(1)), float(m.group(2))
    return log_cross(pjoin(basedir, 'run_%s.log'
                           % (tag if run_name == 'prod' else run_name)))


def read_ms_result(logpath):
    for line in open(logpath, errors='replace'):
        if line.startswith('MSRESULT '):
            return eval(line[len('MSRESULT '):])
    return {}


def card_audit(lhe, keys):
    """Read run-card entries back out of an event file's own banner.

    The card as USED, not the template: this is what RESULTS.md's cut audit is
    built from, and the difference matters -- MG5 writes process-dependent
    defaults into the card, and the loop-induced study's largest trap was a set
    of per-lepton defaults nobody asked for.
    """
    import gzip
    op = gzip.open if lhe.endswith('.gz') else open
    out = {}
    with op(lhe, 'rt', errors='replace') as fh:
        inblock = False
        for line in fh:
            if '<MGRunCard>' in line:
                inblock = True
                continue
            if '</MGRunCard>' in line:
                break
            if not inblock:
                continue
            m = re.match(r'^\s*(\S.*?)\s*=\s*([A-Za-z_]\w*)\s*(?:!.*)?$', line)
            if m and m.group(2) in keys:
                out[m.group(2)] = m.group(1)
    return out


# --------------------------------------------------------------------------
# stage: harvest
# --------------------------------------------------------------------------
def harvest_leptonic(files, meta):
    """Four-lepton observables of the MadSpin comparison."""
    import numpy as np
    sys.path.insert(0, _HERE)
    import observables_zz as OZ
    O = OZ.leptonic()

    hist = {}
    meta.setdefault('runs', {})
    meta['bins'] = {k: v.tolist() for k, v in O.BINS.items()}
    meta['observables'] = list(O.BINS)
    for label, path in files.items():
        w, p = O.read_lhe(path)
        obs = O.compute(p)
        rec = meta['runs'].setdefault(label, {})
        neg = int((w < 0).sum())
        rec.update({
            'lhe': path, 'nevents': int(len(w)),
            'sumw': float(w.sum()), 'sigma_from_events': float(w.mean()),
            'sumw2': float((w ** 2).sum()),
            'sigma_mc_error': float(np.sqrt((w ** 2).sum()) / len(w)),
            # MC@NLO events carry both signs; every downstream number has to
            # keep them, so the count travels with the sample.
            'n_negative_weight': neg,
            'negative_weight_fraction': float(neg) / len(w),
            'sum_abs_w': float(np.abs(w).sum()),
            'distinct_abs_weights': int(len(np.unique(np.round(np.abs(w), 12)))),
        })
        for name, edges in O.BINS.items():
            y, e = O.histogram(obs[name], w, edges)
            hist['%s/%s/y' % (label, name)] = y
            hist['%s/%s/e' % (label, name)] = e
        rec['m_ee_range'] = [float(obs['m_ee'].min()), float(obs['m_ee'].max())]
        rec['m_mumu_range'] = [float(obs['m_mumu'].min()),
                               float(obs['m_mumu'].max())]
        rec['pt_ee_min'] = float(obs['pt_ee'].min())
        rec['pt_mumu_min'] = float(obs['pt_mumu'].min())
        # How well the pt(Z) cut is respected on the reconstructed pairs.  At
        # 2 -> 2 it is exact; on an NLO sample a handful of H events sit below
        # it, and the size of that leak has to be measured on both sides of the
        # comparison rather than assumed to cancel.
        below = (obs['pt_ee'] < PT_Z_MIN) | (obs['pt_mumu'] < PT_Z_MIN)
        rec['n_below_pt_cut'] = int(below.sum())
        rec['weight_fraction_below_pt_cut'] = float(w[below].sum() / w.sum())
        # and the mass window, which only the truth sample is cut on
        out_of_window = ((np.abs(obs['m_ee'] - O.M_Z) > O.BW_CUT * O.W_Z)
                         | (np.abs(obs['m_mumu'] - O.M_Z) > O.BW_CUT * O.W_Z))
        rec['n_outside_mass_window'] = int(out_of_window.sum())
        # the four-lepton mass below the on-shell ZZ threshold: reachable only
        # by a mode that draws a virtuality, and then only for an event with
        # recoil, since the reshuffle conserves sqrt(shat)
        sub = obs['m_4l'] < 2 * O.M_Z
        rec['n_below_2mZ'] = int(sub.sum())
        rec['weight_fraction_below_2mZ'] = float(w[sub].sum() / w.sum())

        def moment(v):
            mu = float(np.average(v, weights=w))
            var = float(np.average((v - mu) ** 2, weights=w))
            return [mu, float(np.sqrt(abs(var) / len(w)))]

        rec['moments'] = {
            'cos_theta1': moment(obs['cos_theta1']),
            'cos2_theta1': moment(obs['cos_theta1'] ** 2),
            'cos1cos2': moment(obs['cos1cos2']),
            'cos_phi': moment(np.cos(obs['phi_planes'])),
            'cos_2phi': moment(np.cos(2 * obs['phi_planes'])),
            'm_epmum': moment(obs['m_epmum']),
        }
    return hist


def harvest_stack(files, meta):
    """Production-level ``Z Z`` observables of the physics figure."""
    import numpy as np
    sys.path.insert(0, _HERE)
    import observables_zz as OZ

    hist = {}
    meta['bins_zz'] = {k: v.tolist() for k, v in OZ.BINS_ZZ.items()}
    meta['observables_zz'] = list(OZ.BINS_ZZ)
    meta.setdefault('production', {})
    for label, path in files.items():
        w, z1, z2, extra = OZ.read_lhe_zz(path)
        obs = OZ.compute_zz(z1, z2)
        neg = int((w < 0).sum())
        rec = meta['production'].setdefault(label, {})
        rec.update({
            'lhe': path, 'nevents': int(len(w)),
            'sigma_from_events': float(w.mean()),
            'sigma_mc_error': float(np.sqrt((w ** 2).sum()) / len(w)),
            'n_negative_weight': neg,
            'negative_weight_fraction': float(neg) / len(w),
            'sum_abs_w': float(np.abs(w).sum()),
            'n_events_with_extra_parton': int((extra > 0).sum()),
            'max_extra_partons': int(extra.max()),
            'pt_zz_max': float(obs['pt_zz'].max()),
            'pt_z_min': float(min(obs['pt_z_lead'].min(),
                                  obs['pt_z_sublead'].min())),
            # the double-counting check, on the events rather than on the
            # process definition
            'initial_states': {'%d %d' % k: v for k, v
                               in sorted(OZ.read_lhe_initial_states(path).items())},
        })
        for name, edges in OZ.BINS_ZZ.items():
            y, e = OZ.histogram(obs[name], w, edges)
            hist['%s/%s/y' % (label, name)] = y
            hist['%s/%s/e' % (label, name)] = e
    return hist


def stage_harvest(basedir, modes, datadir, logdir):
    import numpy as np
    sys.path.insert(0, _HERE)
    import observables_zz as OZ
    O = OZ.leptonic()

    meta = {'code_sha': code_sha(), 'm_Z': O.M_Z, 'width_Z': O.W_Z,
            'BW_cut': O.BW_CUT, 'mass_window': [O.M_LO, O.M_HI],
            'processes': PROC, 'pt_z_min': PT_Z_MIN,
            'scale': 'fixed muR = muF = %s GeV (= m_Z)' % MU,
            'run_card_lo': RUN_CARD_LO, 'run_card_nlo': RUN_CARD_NLO,
            'pt_min_pdg': PT_MIN_PDG, 'madspin_modes': modes,
            'nevents_requested': NEVENTS,
            'seeds': {'requested_production': SEED_PROD,
                      'madspin_all_modes': SEED_MADSPIN}}

    prod = {t: sample_lhe(basedir, t) for t in ('nlo', 'lo', 'li')}
    prod = {k: v for k, v in prod.items() if v}
    hist = harvest_stack(prod, meta)

    for tag, path in prod.items():
        meta['seeds']['%s_actual' % tag] = banner_seed(path)
        meta['production'][tag]['banner'] = banner_cross(path)
        got = run_cross(basedir, tag)
        if got:
            meta['production'][tag]['integration_sigma_pb'] = got[0]
            meta['production'][tag]['integration_error_pb'] = got[1]
        meta['production'][tag]['card'] = card_audit(path, AUDIT_KEYS)

    # The four-lepton side: every decayed sample, plus the off-shell reference
    # if it was affordable.  ``truth`` is deliberately keyed the same way the
    # loop-induced study keys it, so the plotting scripts pick it up by name and
    # fall back to a mode-against-mode comparison when it is absent.
    decayed = {}
    tr = sample_lhe(basedir, 'truth')
    if tr:
        decayed['truth'] = tr
    for mode in modes:
        p = pjoin(basedir, 'ms_%s' % mode, 'events_decayed.lhe.gz')
        if os.path.exists(p):
            decayed[mode] = p
    hist.update(harvest_leptonic(decayed, meta))
    if tr:
        meta['runs']['truth']['seed_actual'] = banner_seed(tr)
        meta['runs']['truth']['banner'] = banner_cross(tr)
        meta['runs']['truth']['card'] = card_audit(tr, AUDIT_KEYS)
        got = run_cross(basedir, 'truth')
        if got:
            meta['runs']['truth']['integration_sigma_pb'] = got[0]
            meta['runs']['truth']['integration_error_pb'] = got[1]
        meta['cuts_file_truth'] = CUTS_FILE_NLO

    meta['reported'] = {}
    for mode in modes:
        log = pjoin(basedir, 'ms_%s' % mode, 'madspin.log')
        if os.path.exists(log):
            meta['reported'][mode] = read_ms_result(log)

    meta['controls'] = {}
    for name, spec in CONTROLS.items():
        got = run_cross(basedir, spec['tag'], 'ctrl_' + name)
        lhe = sample_lhe(basedir, spec['tag'], 'ctrl_' + name)
        if not got or not lhe:
            continue
        meta['controls'][name] = {
            'note': spec['note'], 'sigma_pb': got[0], 'error_pb': got[1],
            'seed_actual': banner_seed(lhe),
            'card': card_audit(lhe, AUDIT_KEYS)}

    # the fixed-order NLO cross-check, if it was run
    got = log_cross(pjoin(basedir, 'run_nlo_fixedorder.log'))
    if got:
        meta['fixed_order_nlo'] = {'sigma_pb': got[0], 'error_pb': got[1],
                                   'note': 'p p > z z [QCD], launch NLO (fixed '
                                           'order), same card'}
    if os.path.exists(_TIMINGS or ''):
        meta['wall_times'] = json.load(open(_TIMINGS))

    os.makedirs(datadir, exist_ok=True)
    np.savez_compressed(pjoin(datadir, 'histograms.npz'), **hist)
    with open(pjoin(datadir, 'meta.json'), 'w') as fp:
        json.dump(meta, fp, indent=1, sort_keys=True)
    print('wrote %s' % pjoin(datadir, 'histograms.npz'))
    return meta


# Every run-card entry the RESULTS.md cut audit quotes.  Read back out of each
# event file's own banner, so entries a process hides simply do not appear -- and
# their absence is itself part of the audit.
AUDIT_KEYS = (
    'ptl', 'etal', 'drll', 'drll_sf', 'mll', 'mll_sf', 'mmll', 'mmllmax',
    'ptheavy', 'ptj', 'etaj', 'jetalgo', 'jetradius', 'ptgmin',
    'pt_min_pdg', 'pt_max_pdg', 'mxx_min_pdg', 'eta_min_pdg', 'eta_max_pdg',
    'dsqrt_shat', 'dsqrt_shatmax', 'bwcutoff', 'pdlabel', 'lhaid',
    'ebeam1', 'ebeam2', 'nevents', 'iseed', 'event_norm',
    'fixed_ren_scale', 'fixed_fac_scale', 'scale', 'dsqrt_q2fact1',
    'dsqrt_q2fact2', 'muR_ref_fixed', 'muF_ref_fixed',
    'dynamical_scale_choice', 'parton_shower', 'folding', 'ickkw',
    'req_acc', 'use_syst', 'nhel', 'sde_strategy',
)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['all', 'prod', 'madspin', 'controls', 'harvest'])
    ap.add_argument('--controls', default=','.join(CONTROLS))
    ap.add_argument('--basedir', default='/tmp/zz_nlo_work')
    ap.add_argument('--nb-core', type=int, default=6)
    ap.add_argument('--modes', default=','.join(MODES))
    ap.add_argument('--samples', default='nlo,lo,li,truth')
    ap.add_argument('--data', default=pjoin(_HERE, 'data'))
    ap.add_argument('--logs', default=pjoin(_HERE, 'logs'))
    args = ap.parse_args()

    global _TIMINGS
    os.makedirs(args.basedir, exist_ok=True)
    _TIMINGS = pjoin(args.basedir, 'timings.json')
    os.makedirs(args.logs, exist_ok=True)
    modes = [m for m in args.modes.split(',') if m]
    tags = tuple(t for t in args.samples.split(',') if t)

    if args.stage in ('all', 'prod'):
        stage_prod(args.basedir, args.nb_core, args.logs, tags)

    if args.stage in ('all', 'madspin'):
        lhe = sample_lhe(args.basedir, 'nlo')
        if not lhe:
            raise RuntimeError('the NLO sample has not been generated yet')
        stage_madspin(args.basedir, lhe, modes, args.nb_core, args.logs)

    if args.stage in ('all', 'controls'):
        stage_controls(args.basedir,
                       [c for c in args.controls.split(',') if c],
                       args.nb_core, args.logs)
        fixed_order_nlo(args.basedir, args.nb_core, args.logs)

    if args.stage in ('all', 'harvest'):
        stage_harvest(args.basedir, modes, args.data, args.logs)


if __name__ == '__main__':
    main()
