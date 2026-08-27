#!/usr/bin/env python3
"""Event generation for Fig. 5 of the MadSpin2 paper (``fig:valid_interf``).

The figure is ``Delta phi(e- e+)`` between the electron and the positron of the
``t t~`` decay, for the SMEFT ``O_tG`` *amplitude-level interference* term,
with spin correlations (``spinmode = onshell``) against without
(``spinmode = none``).

This driver only *generates*; it makes no plot.  It writes

  data/histograms.npz  fine (720-bin) ``Delta phi`` histograms per sample:
                       sum of weights, sum of weights squared, raw counts,
                       and the counts split by the sign of the weight
  data/meta.json       the full generation record the plotting step consumes

and it leaves the decayed LHE files under ``--workdir`` (outside the repo --
at the statistics used here they are several GB and are not committable).

What is generated
-----------------
Three production samples, each decayed twice (``onshell`` and ``none``):

``eft_int``  ``p p > t t~ NP=1 NP^2==1`` in ``SMEFTsim_topU3l_MwScheme_UFO``
             restricted so every Wilson coefficient vanishes except
             ``ctGRe``, which is set to **-1** (not +1 -- see ``CTGRE``), with
             ``Lambda = 1 TeV``.  This is ``-2 Re(M_SM^* M_tG)`` alone, i.e.
             the paper's quantity with the overall sign flipped, which is what
             makes its integral positive and what makes the density spinmodes
             runnable at all.  The overall normalisation is arbitrary (it
             scales with ``c_tG / Lambda^2``).
``sm_lo``    ``p p > t t~ NP=0`` in the *same* restricted model, i.e. the SM
             tree-level reference with the EFT vertex switched off.
``sm_nlo``   ``p p > t t~ [QCD]`` in ``loop_sm``.  SMEFTsim v3.0.2 is a
             tree-level UFO (no ``CT_vertices.py``, no ``Fortran/``), so the
             NLO sample cannot live in it; ``loop_sm`` is used with the top
             mass and width set to the SMEFTsim values so the three samples
             describe the same top quark.

Everything else -- decays, cuts, scale, PDF, beam energy, ``bwcutoff`` -- is
identical across the three, by construction: it is read from the module-level
constants below rather than repeated per sample.

Traps this script encodes
-------------------------
* ``fixed_order`` is refused by MadSpin in the spinmodes that reshuffle the
  production (``PA``/``madspin``, see ``FIXED_ORDER_RESHUFFLING_SPINMODES``).
  It is also *not needed* here: the ``sm_nlo`` sample is MC@NLO, whose LHE
  holds individual signed-weight events rather than born/counter-event groups,
  which is what ``fixed_order`` exists to keep together.  It is left off, and
  the script asserts the two spinmodes it does use run.
* ``sm_nlo`` (MC@NLO) carries negative weights of its own.  ``eft_int`` is a
  signed sample by nature; at ``ctGRe = -1`` it happens to come out positive
  event by event, which is exactly why it runs.  The negative fraction is
  measured for every sample and recorded either way.
* The decay is written as one ``decay`` line per ``t``-like particle -- the
  multi-channel positional rule -- and the harvest *checks* that every decayed
  event has exactly one ``e-`` and exactly one ``e+`` instead of assuming it.
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.realpath(__file__))
_root = os.path.split(os.path.split(os.path.split(_here)[0])[0])[0]
if _root not in sys.path:
    sys.path.insert(0, _root)

pjoin = os.path.join

MG5_BIN = pjoin(_root, 'bin', 'mg5_aMC')
MADSPIN_BIN = pjoin(_root, 'MadSpin', 'madspin')


# --------------------------------------------------------------------------
# Shared physics setup.  One place, used by every sample.
#
# Scale: fixed mu_R = mu_F = 173 GeV.  The paper's Sec. 5 default is the CKKW
# back-clustering scale, but every t t~ study of this series uses the fixed
# scale instead, and here it matters more than usual: the point of the figure
# is a ratio between two *decays* of the same production sample, and a
# dynamical scale computed on the generated final state would differ between
# a production LHE and any cross-check made on a decayed one.
#
# PDF: NNPDF2.3 LO, alpha_s(M_Z) = 0.130, lhaid 230000, at 13 TeV -- the
# paper's Sec. 5 default set.  The *NLO* sample deliberately uses the same LO
# set (as the paper's NLO ZZ study does), so that an LO-vs-NLO comparison shows
# a change of perturbative order and not a change of PDF.  Requesting it
# through LHAPDF rather than through the built-in ``nn23lo1`` makes the LO and
# NLO samples use literally the same grid.
#
# Cuts: none.  The paper states no cut for this figure; the process is a
# 2 -> 2 massive final state, so nothing needs a cut to be finite, and the
# leptons are left fully inclusive (no ptl / etal / drll / mmll).
# --------------------------------------------------------------------------
BEAM_ENERGY = 6500.0          # 13 TeV
FIXED_SCALE = 173.0           # GeV, mu_R = mu_F
LHAID = 230000                # NNPDF23_lo_as_0130_qed
BWCUTOFF = 15.0               # run_card bwcutoff; matched by MadSpin BW_cut
BW_CUT = 15.0
MS_NB_CORE = 12               # MadSpin unweighting workers

# The SMEFTsim v3.0.2 values, carried into loop_sm for the NLO sample so the
# three samples share one top quark.
MT = 172.76
WT = 1.33
WW = 2.085
WZ = 2.4952

MULTIPARTICLES = {'p': 'g u c d s u~ c~ d~ s~'}

# --------------------------------------------------------------------------
# The Wilson coefficient, and why its sign is not free.
#
# The interference term is *linear* in c_tG, so flipping the coefficient flips
# the sign of 2 Re(M_SM^* M_tG) event by event and leaves everything else
# identical.  With ``ctGRe = +1`` (the value written in the paper) the
# integrated interference is negative -- and, as this run measured, so is the
# matrix element at *every* generated phase-space point: 100% of the production
# events come out with a negative weight.
#
# That is not merely awkward to normalise (dividing a distribution by a
# negative integral flips the curve as well as scaling it).  It stops MadSpin
# outright: the density spinmodes normalise their accept/reject weight on
# Tr(rho_prod), and ``_check_production_density`` refuses any production event
# with Tr(rho_prod) <= 0 (``MadSpinDegenerateWeight``).  A negative-definite
# interference therefore fails on the very first event in ``spinmode=onshell``.
#
# ``ctGRe = -1`` gives the same distribution with every sign flipped: positive
# integrated interference, positive Tr(rho_prod), and a run that completes.
# The plotted quantity is then -2 Re(M_SM^* M_tG) with respect to the paper's
# current caption, which is written for ctGRe = +1.  ``CTGRE_CHECK`` re-runs
# the *production* alone at +1 with the same seed to demonstrate the flip.
# --------------------------------------------------------------------------
CTGRE = -1.0
CTGRE_CHECK = +1.0
LAMBDA_SMEFT = 1000.0     # GeV, the SMEFTsim restriction card's value

# The restriction card this run builds.  It is ``restrict_spyros.dat`` with the
# ctGRe entry changed from 1.0 to a plain non-special number: MG5's restriction
# machinery turns a coefficient of exactly 1 into a *fixed* parameter (see
# ``fix_parameter_values``: ``value == 1`` -> ``rule_card.add_one``), which
# removes ctGRe from the param_card altogether and makes the sign unsettable.
# Any other non-zero value keeps it external and settable at launch; 0.37 is
# arbitrary and deliberately not shared with any other entry, so MG5's
# ``merge_iden_couplings`` cannot fuse two couplings that only happen to
# evaluate to the same number.
# The restriction cards this run builds: ``restrict_spyros.dat`` with the ctGRe
# entry replaced by the wanted value, one card per sign.
#
# The value goes in the *restriction card* rather than being set in the
# param_card at launch, and that is not cosmetic.  MadSpin builds its density
# matrices from a standalone matrix-element directory
# (``madspin_me``/``madspin_decay``, written by ``output standalone``) and
# ``initialise_f2py_module`` initialises that library from
# ``<madspin_me>/Cards/param_card.dat`` -- the card ``output standalone`` wrote
# from the *model defaults* -- whenever that file exists.  It does exist, always,
# so the run's own card (which MadSpin does write, to ``<path_me>/param_card.dat``
# from the event banner) is never read on this path.  Putting the coefficient in
# the restriction makes the model default and the run agree, so the density
# matrices see the coupling the events were generated with.
#
# ``restrict_spyros.dat`` itself uses ctGRe = 1, and a restriction value of
# exactly 1 makes MG5 *fix* the parameter (``fix_parameter_values`` ->
# ``rule_card.add_one``), deleting it from the param_card entirely.  For the
# primary (-1) card that does not apply; -1 stays a normal external parameter.
RESTRICTION_SOURCE = 'restrict_spyros.dat'
RESTRICTIONS = {'fig5m': CTGRE, 'fig5p': CTGRE_CHECK}
MODEL_SEARCH = [
    os.path.expanduser('~/Desktop/UFOMODEL/SMEFTsim_topU3l_MwScheme_UFO'),
    pjoin(_root, 'models', 'SMEFTsim_topU3l_MwScheme_UFO'),
]

# One ``decay`` line per t-like particle, in the order the particles appear in
# the production process -- the multi-channel positional rule.  The observable
# is Delta phi(e- e+), so both tops are decayed to *electrons* specifically:
# t -> b W+ -> b e+ ve  gives the e+, t~ -> b~ W- -> b~ e- ve~ gives the e-.
DECAYS = [
    'decay t > w+ b, w+ > e+ ve',
    'decay t~ > w- b~, w- > e- ve~',
]

SPINMODES = ['onshell', 'none']

SEED_PROD = 42
SEED_MADSPIN = 42

# Fine, uniform Delta phi grid on [0, pi].  Far finer than any plot will use,
# so the plotting step can rebin without regenerating.
DPHI_NBINS = 720


SAMPLES = [
    dict(
        key='eft_int',
        label=r'SMEFT $\mathcal{O}_{tG}$ interference ($c_{tG}=-1$)',
        model=None,               # filled in by prepare_model()
        restriction='fig5m',
        process='p p > t t~ NP=1 NP^2==1',
        nlo=False,
        param_sets={},
        production_only=False,
        wilson={'ctGRe': CTGRE, 'ctGIm': 0.0,
                'LambdaSMEFT_GeV': LAMBDA_SMEFT,
                'all_other_Wilson_coefficients': 0.0},
        note=('amplitude-level interference. With ctGRe = -1 the sample is '
              '-2 Re(M_SM^* M_tG) relative to the paper caption, which is '
              'written for ctGRe = +1; its integral is positive. '
              'Normalisation arbitrary: it scales linearly with '
              'c_tG / Lambda^2.'),
    ),
    dict(
        key='sm_lo',
        label=r'SM LO',
        model=None,
        restriction='fig5m',
        process='p p > t t~ NP=0',
        nlo=False,
        param_sets={},
        production_only=False,
        wilson={'all_Wilson_coefficients_absent': True,
                'note': 'NP=0 selects the pure SM amplitude; no EFT vertex '
                        'enters, so ctGRe has no effect here'},
        note='SM tree level in the same model, EFT vertex switched off',
    ),
    dict(
        key='sm_nlo',
        label=r'SM NLO (MC@NLO)',
        model='loop_sm',
        process='p p > t t~ [QCD]',
        nlo=True,
        # match the SMEFTsim top so the three samples share one top quark
        # match the SMEFTsim electroweak widths too, so BR(t -> b e nu) is the
        # same number in the LO and the NLO samples and a naive LO/NLO ratio is
        # not off by the difference between two W widths
        param_sets={'MT': MT, 'WT': WT, 'ymt': MT, 'WW': WW, 'WZ': WZ},
        production_only=False,
        wilson={},
        note='MC@NLO; signed weights of its own, unrelated to the EFT ones',
    ),
    dict(
        # Production-only sign cross-check: the same process, the same seed,
        # ctGRe = +1. Every event should come back with the *same* kinematics
        # and the opposite weight, which is the statement that the sample is
        # linear in the Wilson coefficient. It is production-only on purpose:
        # at +1 the density spinmodes refuse to run (Tr(rho_prod) < 0), and
        # demonstrating that refusal is part of the point.
        key='eft_int_ctg_plus',
        label=r'SMEFT $\mathcal{O}_{tG}$ interference ($c_{tG}=+1$, check)',
        model=None,
        restriction='fig5p',
        process='p p > t t~ NP=1 NP^2==1',
        nlo=False,
        param_sets={},
        production_only=True,
        wilson={'ctGRe': CTGRE_CHECK, 'ctGIm': 0.0,
                'LambdaSMEFT_GeV': LAMBDA_SMEFT,
                'all_other_Wilson_coefficients': 0.0},
        note=('sign cross-check only, not a curve on the figure: the paper\'s '
              'ctGRe = +1 sign, same seed as eft_int'),
    ),
]


# --------------------------------------------------------------------------
# Model preparation
# --------------------------------------------------------------------------
def prepare_model(workdir):
    """Copy the SMEFTsim UFO into the work tree and add our restriction cards.

    Copied rather than edited in place so nothing is written into whatever
    shared model directory MG5 resolves ``SMEFTsim_topU3l_MwScheme_UFO`` to,
    and so the exact model used is captured beside the events.

    Returns ``({restriction_name: model_argument}, record)``.
    """
    src = None
    for cand in MODEL_SEARCH:
        if os.path.isdir(cand):
            src = cand
            break
    if src is None:
        raise RuntimeError('SMEFTsim_topU3l_MwScheme_UFO not found; looked in '
                           + ', '.join(MODEL_SEARCH))
    dest_root = pjoin(workdir, 'models')
    dest = pjoin(dest_root, 'SMEFTsim_topU3l_MwScheme_UFO')
    if not os.path.isdir(dest):
        if not os.path.isdir(dest_root):
            os.makedirs(dest_root)
        shutil.copytree(src, dest,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

    source_card = pjoin(dest, RESTRICTION_SOURCE)
    if not os.path.exists(source_card):
        raise RuntimeError('%s missing from %s' % (RESTRICTION_SOURCE, dest))
    source_lines = open(source_card).read().splitlines()

    models = {}
    cards = {}
    for name, value in sorted(RESTRICTIONS.items()):
        out = []
        replaced = 0
        for line in source_lines:
            # entry 15 of the SMEFT block is ctGRe
            m = re.match(r'^(\s*)15(\s+)([-+0-9.eEdD]+)(\s*#\s*ctgre.*)$',
                         line, re.I)
            if m:
                out.append('%s15%s%.6e%s'
                           % (m.group(1), m.group(2), value, m.group(4)))
                replaced += 1
            else:
                out.append(line)
        if replaced != 1:
            raise RuntimeError('expected exactly one ctGRe entry in %s, '
                               'found %d' % (source_card, replaced))
        card = pjoin(dest, 'restrict_%s.dat' % name)
        with open(card, 'w') as fp:
            fp.write('\n'.join(out) + '\n')
        models[name] = '%s-%s' % (dest, name)
        cards[name] = dict(path=card, ctGRe=value)

    record = dict(
        source=src,
        copied_to=dest,
        restriction_source=source_card,
        restriction_cards=cards,
        restriction_note=(
            'restrict_spyros.dat (all Wilson coefficients zero except ctGRe, '
            'LambdaSMEFT = %g GeV) with the ctGRe entry rewritten per sample. '
            'The coefficient is carried by the restriction card, not by a '
            'param_card override, because MadSpin initialises its density '
            'matrix-element library from <madspin_me>/Cards/param_card.dat -- '
            'the card output-standalone wrote from the model defaults -- and '
            'never from the run\'s own card. Model default and run value must '
            'therefore be the same number.' % LAMBDA_SMEFT),
        all_other_wilson_coefficients='zero (inherited from restrict_spyros.dat)',
        LambdaSMEFT_GeV=LAMBDA_SMEFT,
        m_top_GeV=MT, width_top_GeV=WT,
        mass_note=('the SMEFTsim v3.0.2 values are kept as they are; the '
                   'restriction card fixes m_t = %g GeV, Gamma_t = %g GeV, '
                   'and loop_sm is set to the same two numbers so the three '
                   'samples describe one top quark' % (MT, WT)),
    )
    return models, record


# --------------------------------------------------------------------------
# MG5 driving
# --------------------------------------------------------------------------
def _run(cmd, log_path, cwd=None, env=None):
    with open(log_path, 'w') as logf:
        ret = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              cwd=cwd, env=env)
    return ret


def _mg5_env():
    """f2py must be the one from the mg-3.14 pyenv: the bare ``f2py`` first on
    PATH is a homebrew shim with a dead shebang (exit 126)."""
    env = dict(os.environ)
    pyenv_bin = os.path.expanduser('~/.pyenv/versions/mg-3.14/bin')
    if os.path.isdir(pyenv_bin):
        env['PATH'] = pyenv_bin + os.pathsep + env.get('PATH', '')
    return env


def write_mg5_script(path, sample, nevents, proc_dir, seed):
    lines = ['set automatic_html_opening False --no_save',
             'import model %s' % sample['model']]
    for name, definition in MULTIPARTICLES.items():
        lines.append('define %s = %s' % (name, definition))
    lines.append('generate %s' % sample['process'])
    lines.append('output %s' % proc_dir)
    lines.append('launch %s' % proc_dir)
    # The two launch menus do NOT offer the same switches. The aMC@NLO one has
    # order / fixed_order / shower / madspin / reweight / madanalysis and no
    # 'detector' or 'analysis' at all: sending those makes MG5 answer "not
    # valid for current question. Keep it for next question", which then eats
    # the 'done' *and* the card-editing prompt, and every later 'set' lands on
    # the MG5 option parser instead of the run_card -- silently leaving the run
    # on default PDF and scales.
    if sample['nlo']:
        lines.append('order=NLO')
        lines.append('fixed_order=OFF')
        lines.append('shower=OFF')
        lines.append('madspin=OFF')
        lines.append('reweight=OFF')
    else:
        lines.append('shower=OFF')
        lines.append('madspin=OFF')
        lines.append('reweight=OFF')
        lines.append('detector=OFF')
        lines.append('analysis=OFF')
    lines.append('done')

    lines.append('set nevents %d' % nevents)
    lines.append('set iseed %d' % seed)
    lines.append('set use_syst False')
    lines.append('set ebeam1 %s' % BEAM_ENERGY)
    lines.append('set ebeam2 %s' % BEAM_ENERGY)
    lines.append('set pdlabel lhapdf')
    lines.append('set lhaid %d' % LHAID)
    if sample['nlo']:
        # NLO run_card spelling
        lines.append('set fixed_ren_scale True')
        lines.append('set fixed_fac_scale True')
        lines.append('set muR_ref_fixed %s' % FIXED_SCALE)
        lines.append('set muF_ref_fixed %s' % FIXED_SCALE)
        lines.append('set reweight_scale False')
        lines.append('set reweight_pdf False')
        lines.append('set req_acc -1')
        lines.append('set parton_shower PYTHIA8')
    else:
        lines.append('set fixed_ren_scale True')
        lines.append('set fixed_fac_scale True')
        lines.append('set scale %s' % FIXED_SCALE)
        lines.append('set dsqrt_q2fact1 %s' % FIXED_SCALE)
        lines.append('set dsqrt_q2fact2 %s' % FIXED_SCALE)
        lines.append('set bwcutoff %s' % BWCUTOFF)
        # explicitly inclusive leptons/jets: no cut on the figure's observable
        lines.append('set cut_decays False')
    for key, val in sorted(sample['param_sets'].items()):
        lines.append('set %s %s' % (key, val))
    lines.append('done')
    with open(path, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')
    return path


_ABORT_MARKERS = ('NoDiagramException',
                  'command not executed: output',
                  'command not executed: launch',
                  # a 'set' that never reached the card editor leaves the run on
                  # default PDF/scales while still producing an events file, so
                  # it has to be fatal and not a warning
                  'command not executed: set ',
                  'This answer is not valid for current question')


def produce(sample, nevents, workdir, logdir, seed=SEED_PROD, force=False):
    """Generate the production sample; return (lhe_path, banner_info)."""
    proc_dir = pjoin(workdir, 'PROC_%s' % sample['key'])
    lhe = _find_production_lhe(proc_dir)
    if lhe and not force:
        print('  [skip] production already present: %s' % lhe)
        return lhe, proc_dir

    if os.path.exists(proc_dir):
        shutil.rmtree(proc_dir)
    script = pjoin(workdir, 'mg5_%s.dat' % sample['key'])
    write_mg5_script(script, sample, nevents, proc_dir, seed)
    log_path = pjoin(logdir, 'production_%s.log.txt' % sample['key'])
    print('  running mg5_aMC (%s events) -> %s' % (nevents, log_path))
    t0 = time.time()
    # cwd=workdir, not this directory: mg5_aMC and the reweight machinery drop
    # scratch files (``MG5_debug``, ``py.py``, ``events.lhe.rwgt``,
    # ``scale_pdf_dependence.dat``, ``additional_command``) in whatever
    # directory they are launched from, and that directory is inside the repo.
    ret = _run([MG5_BIN, '-f', script], log_path, cwd=workdir,
               env=_mg5_env())
    wall = time.time() - t0
    text = open(log_path, errors='replace').read()
    if ret != 0 or any(m in text for m in _ABORT_MARKERS):
        raise RuntimeError('mg5_aMC failed for %s -- see %s'
                           % (sample['key'], log_path))
    lhe = _find_production_lhe(proc_dir)
    if not lhe:
        raise RuntimeError('no production LHE under %s' % proc_dir)
    print('  production done in %.0f s: %s' % (wall, lhe))
    return lhe, proc_dir


def _find_production_lhe(proc_dir):
    events = pjoin(proc_dir, 'Events')
    if not os.path.isdir(events):
        return None
    for run in sorted(os.listdir(events)):
        run_dir = pjoin(events, run)
        if not os.path.isdir(run_dir):
            continue
        for name in ('unweighted_events.lhe.gz', 'unweighted_events.lhe',
                     'events.lhe.gz', 'events.lhe'):
            cand = pjoin(run_dir, name)
            if os.path.exists(cand):
                return cand
    return None


# --------------------------------------------------------------------------
# MadSpin driving
# --------------------------------------------------------------------------
def write_madspin_card(path, evt_path, spinmode, seed, extra=None):
    lines = ['set spinmode %s' % spinmode,
             'set seed %d' % seed,
             'set BW_cut %s' % BW_CUT,
             'set max_running_process 4',
             'set nb_core %d' % MS_NB_CORE]
    for key, val in sorted((extra or {}).items()):
        lines.append('set %s %s' % (key, val))
    for name, definition in MULTIPARTICLES.items():
        lines.append('define %s = %s' % (name, definition))
    lines.append('import %s' % evt_path)
    lines.extend(DECAYS)
    lines.append('launch')
    with open(path, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')
    return path


def run_madspin(sample_key, prod_lhe, spinmode, workdir, logdir,
                seed=SEED_MADSPIN, extra=None, force=False):
    run_dir = pjoin(workdir, 'MS_%s_%s' % (sample_key, spinmode))
    decayed = _decayed_path(run_dir)
    if decayed and not force:
        print('  [skip] decayed sample already present: %s' % decayed)
        return decayed, pjoin(logdir, 'madspin_%s_%s.log.txt'
                              % (sample_key, spinmode)), None

    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    evt = pjoin(run_dir, 'events.lhe.gz' if prod_lhe.endswith('.gz')
                else 'events.lhe')
    shutil.copy(prod_lhe, evt)
    card = pjoin(run_dir, 'madspin_card.dat')
    write_madspin_card(card, evt, spinmode, seed, extra=extra)

    log_path = pjoin(logdir, 'madspin_%s_%s.log.txt' % (sample_key, spinmode))
    print('  running MadSpin spinmode=%s -> %s' % (spinmode, log_path))
    t0 = time.time()
    ret = _run([MADSPIN_BIN, card], log_path, cwd=run_dir, env=_mg5_env())
    wall = time.time() - t0
    if ret != 0:
        raise RuntimeError('MadSpin failed for %s/%s -- see %s'
                           % (sample_key, spinmode, log_path))
    decayed = _decayed_path(run_dir)
    if not decayed:
        raise RuntimeError('no decayed LHE under %s' % run_dir)
    print('  MadSpin done in %.0f s: %s' % (wall, decayed))
    return decayed, log_path, wall


def truncate_lhe(src, dest, nevents):
    """Copy the first ``nevents`` events of ``src`` into ``dest``, banner and
    all. Used by the fixed_order probe, which only needs a handful."""
    written = 0
    with _open(src) as fin, gzip.open(dest, 'wt') as fout:
        for line in fin:
            if line.strip().startswith('<event'):
                if written >= nevents:
                    break
                written += 1
            fout.write(line)
        fout.write('</LesHouchesEvents>\n')
    return dest


def fixed_order_probe(prod_lhe, workdir, logdir, nevents=2000,
                      spinmodes=('onshell', 'PA', 'madspin')):
    """Actually run ``fixed_order = True`` in each spinmode and record what
    happens, instead of quoting the source.

    ``FIXED_ORDER_RESHUFFLING_SPINMODES = ('PA', 'madspin')`` is expected to
    refuse; ``onshell`` is expected to be allowed. Whether the *sample* needs
    the option is a separate question -- MC@NLO writes individual signed-weight
    events, not born/counter-event groups -- and the probe answers that too, by
    showing whether the option changes anything for these events.
    """
    probe_dir = pjoin(workdir, 'fixed_order_probe')
    if os.path.exists(probe_dir):
        shutil.rmtree(probe_dir)
    os.makedirs(probe_dir)
    small = truncate_lhe(prod_lhe, pjoin(probe_dir, 'events.lhe.gz'), nevents)

    runs = [(sm, True) for sm in spinmodes]
    # the control: the same slice, the same spinmode, fixed_order off. If the
    # option made any difference to an MC@NLO sample the two would not agree.
    runs.append(('onshell', False))

    out = {}
    for spinmode, fixed in runs:
        tag = '%s_fo%s' % (spinmode, 'on' if fixed else 'off')
        run_dir = pjoin(probe_dir, tag)
        os.makedirs(run_dir)
        evt = pjoin(run_dir, 'events.lhe.gz')
        shutil.copy(small, evt)
        card = pjoin(run_dir, 'madspin_card.dat')
        write_madspin_card(card, evt, spinmode, SEED_MADSPIN,
                           extra={'fixed_order': 'True' if fixed else 'False'})
        log_path = pjoin(logdir, 'fixed_order_probe_%s.log.txt' % tag)
        print('  fixed_order probe: spinmode=%s fixed_order=%s'
              % (spinmode, fixed))
        ret = _run([MADSPIN_BIN, card], log_path, cwd=run_dir, env=_mg5_env())
        text = open(log_path, errors='replace').read()
        refusal = None
        m = re.search(r'fixed_order is not available in spinmode=\S+[^\n]*'
                      r'(?:\n(?!\s*$)[^\n]*)*', text)
        if m:
            refusal = re.sub(r'\s+', ' ', m.group(0)).strip()
        decayed = _decayed_path(run_dir)
        rec = dict(
            spinmode=spinmode,
            fixed_order=fixed,
            nevents_in=nevents,
            returncode=ret,
            ran=(ret == 0 and decayed is not None),
            refused=bool(refusal),
            refusal_message=refusal,
            log=os.path.relpath(log_path, _root),
        )
        if decayed:
            xsec = banner_cross_section(decayed)
            rec['decayed_xsec_pb'] = xsec['xsec_pb']
            rec['decayed_xsec_err_pb'] = xsec['xsec_err_pb']
            w = production_weight_stats(decayed)
            rec['n_events_out'] = w['n_events']
            rec['negative_weight_fraction'] = w['negative_weight_fraction']
            rec.update(parse_madspin_log(log_path))
        # 'ran' only means MadSpin exited 0 and wrote a file. With
        # fixed_order on, an MC@NLO event file gets grouped into nothing --
        # the log says 'joint (auto, 0 decaying particle(s))' and
        # '0 written / 0 trials' -- and MadSpin still exits 0 with an EMPTY
        # decayed LHE. So the useful verdict is whether events came out.
        rec['wrote_events'] = bool(rec.get('n_events_out'))
        if rec['refused']:
            rec['verdict'] = 'refused'
        elif rec['ran'] and rec['wrote_events']:
            rec['verdict'] = 'ran, %d events out' % rec['n_events_out']
        elif rec['ran']:
            rec['verdict'] = ('exited 0 but wrote an EMPTY decayed file -- '
                              'silent no-op')
        else:
            rec['verdict'] = 'failed (rc=%d)' % ret
        out[tag] = rec
        print('    -> %s' % rec['verdict'])
    return out


def _decayed_path(run_dir):
    for name in ('events_decayed.lhe.gz', 'events_decayed.lhe'):
        cand = pjoin(run_dir, name)
        if os.path.exists(cand):
            return cand
    return None


# --------------------------------------------------------------------------
# LHE reading.  A small streaming reader: lhe_parser is far too slow at the
# statistics used here.
# --------------------------------------------------------------------------
def _open(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', errors='replace')
    return open(path, errors='replace')


_RE_XSEC = re.compile(r'#\s*Integrated weight \(pb\)\s*:\s*([-+0-9.eEdD]+)')
_RE_XERR = re.compile(r'#\s*Integrated weight error \(pb\)\s*:\s*'
                      r'([-+0-9.eEdD]+)')


def banner_cross_section(path):
    """Cross-section of an LHE, from the ``<init>`` block *and* from the
    human-readable banner comment.

    The two disagree on a decayed file and it matters: MadSpin rewrites the
    ``<init>`` XSECUP/XERRUP to sigma * BR but leaves the production banner's
    ``# Integrated weight (pb)`` comment untouched, so the comment still shows
    the *production* cross-section. The ``<init>`` numbers are the ones to use;
    both are returned so the manifest can show the discrepancy rather than hide
    it.
    """
    comment_xsec = comment_xerr = None
    init = []
    in_init = False
    with _open(path) as fp:
        for line in fp:
            if '<event' in line:
                break
            m = _RE_XSEC.search(line)
            if m:
                comment_xsec = float(
                    m.group(1).replace('d', 'e').replace('D', 'e'))
            m = _RE_XERR.search(line)
            if m:
                comment_xerr = float(
                    m.group(1).replace('d', 'e').replace('D', 'e'))
            if '<init>' in line:
                in_init = True
                init = []
                continue
            if '</init>' in line:
                in_init = False
                continue
            if in_init:
                init.append(line.split())
    init_xsec = init_xerr = None
    if init and len(init) > 1:
        tot = err = 0.0
        ok = False
        for row in init[1:]:
            if len(row) < 4:
                continue
            try:
                tot += float(row[0])
                err += float(row[1]) ** 2
                ok = True
            except ValueError:
                continue
        if ok:
            init_xsec, init_xerr = tot, math.sqrt(err)
    return dict(xsec_pb=(init_xsec if init_xsec is not None else comment_xsec),
                xsec_err_pb=(init_xerr if init_xerr is not None
                             else comment_xerr),
                init_xsec_pb=init_xsec, init_xsec_err_pb=init_xerr,
                banner_comment_xsec_pb=comment_xsec,
                banner_comment_xsec_err_pb=comment_xerr,
                source=('<init> block' if init_xsec is not None
                        else 'banner comment'))


def production_weight_stats(path, keep_weights=False):
    """Sign census of a *production* LHE: how many events carry which sign.

    This is where a negative-definite interference shows up, and it is the
    number that decides whether the density spinmodes can run at all
    (``_check_production_density`` refuses Tr(rho_prod) <= 0, and Tr(rho_prod)
    is the production matrix element on the helicity subspace, i.e. the thing
    whose sign the event weight carries).
    """
    n_pos = n_neg = n_zero = 0
    total = 0.0
    total_abs = 0.0
    weights = [] if keep_weights else None
    header_next = False
    with _open(path) as fp:
        for raw in fp:
            line = raw.strip()
            if line.startswith('<event'):
                header_next = True
                continue
            if header_next:
                fields = line.split()
                header_next = False
                if len(fields) < 6:
                    continue
                try:
                    w = float(fields[2])
                except ValueError:
                    continue
                total += w
                total_abs += abs(w)
                if w > 0:
                    n_pos += 1
                elif w < 0:
                    n_neg += 1
                else:
                    n_zero += 1
                if weights is not None:
                    weights.append(w)
    n = n_pos + n_neg + n_zero
    out = dict(
        n_events=n, n_positive_weight=n_pos, n_negative_weight=n_neg,
        n_zero_weight=n_zero,
        negative_weight_fraction=(n_neg / n if n else 0.0),
        sum_weights=total, sum_abs_weights=total_abs,
        mean_weight=(total / n if n else None),
    )
    if weights is not None:
        out['_weights'] = weights
    return out


def harvest_dphi(path, nbins=DPHI_NBINS):
    """Stream a decayed LHE and histogram ``Delta phi(e- e+)``.

    Returns a dict with the histogram arrays and the bookkeeping the manifest
    needs, including the *check* that every event holds exactly one ``e-`` and
    exactly one ``e+``.
    """
    edges = np.linspace(0.0, math.pi, nbins + 1)
    sumw = np.zeros(nbins)
    sumw2 = np.zeros(nbins)
    count = np.zeros(nbins)
    count_pos = np.zeros(nbins)
    count_neg = np.zeros(nbins)

    n_event = 0
    n_pos = 0
    n_neg = 0
    n_zero = 0
    sum_all = 0.0
    sum_abs = 0.0
    bad_multiplicity = 0
    mult_hist = {}

    in_event = False
    header_done = False
    weight = 0.0
    em = ep = None
    n_em = n_ep = 0

    with _open(path) as fp:
        for raw in fp:
            line = raw.strip()
            if not in_event:
                if line.startswith('<event'):
                    in_event = True
                    header_done = False
                    em = ep = None
                    n_em = n_ep = 0
                continue
            if line.startswith('</event'):
                in_event = False
                n_event += 1
                key = (n_em, n_ep)
                mult_hist[key] = mult_hist.get(key, 0) + 1
                if n_em != 1 or n_ep != 1:
                    bad_multiplicity += 1
                    continue
                dphi = abs(em - ep)
                while dphi > 2 * math.pi:
                    dphi -= 2 * math.pi
                if dphi > math.pi:
                    dphi = 2 * math.pi - dphi
                idx = int(dphi / math.pi * nbins)
                if idx >= nbins:
                    idx = nbins - 1
                sumw[idx] += weight
                sumw2[idx] += weight * weight
                count[idx] += 1
                if weight > 0:
                    count_pos[idx] += 1
                    n_pos += 1
                elif weight < 0:
                    count_neg[idx] += 1
                    n_neg += 1
                else:
                    n_zero += 1
                sum_all += weight
                sum_abs += abs(weight)
                continue
            if line.startswith('<'):
                continue
            fields = line.split()
            if not header_done:
                # NUP IDPRUP XWGTUP SCALUP AQEDUP AQCDUP
                if len(fields) >= 6:
                    try:
                        weight = float(fields[2])
                    except ValueError:
                        weight = 0.0
                    header_done = True
                continue
            if len(fields) < 10:
                continue
            try:
                pdg = int(fields[0])
                status = int(fields[1])
            except ValueError:
                continue
            if status != 1:
                continue
            if pdg == 11:
                n_em += 1
                em = math.atan2(float(fields[7]), float(fields[6]))
            elif pdg == -11:
                n_ep += 1
                ep = math.atan2(float(fields[7]), float(fields[6]))

    return dict(
        edges=edges, sumw=sumw, sumw2=sumw2, count=count,
        count_pos=count_pos, count_neg=count_neg,
        n_event=n_event, n_pos=n_pos, n_neg=n_neg, n_zero=n_zero,
        sum_w=sum_all, sum_abs_w=sum_abs,
        bad_multiplicity=bad_multiplicity,
        mult_hist={'%d_em_%d_ep' % k: v for k, v in sorted(mult_hist.items())},
    )


# --------------------------------------------------------------------------
# Log parsing
# --------------------------------------------------------------------------
def _flat(text):
    return re.sub(r'\s+', ' ', text)


_RE_UNWEIGHT = re.compile(
    r'unweighting (?:scheme|mode)\s*[:=]?\s*([A-Za-z_]+)', re.I)
_RE_UNWEIGHT2 = re.compile(r"resolved .*?unweighting.*?to '?([A-Za-z_]+)'?",
                           re.I)
_RE_OVERWEIGHT = re.compile(
    r'(\d+)\s+events?\s+(?:had|with)\s+.*?weight.*?(?:above|larger)', re.I)
_RE_EFF = re.compile(r'efficiency[^0-9]*([0-9.]+)', re.I)
_RE_BR = re.compile(r'BR\s*[:=]\s*([0-9.eE+-]+)')


_MS_LINE_PATTERNS = [
    ('decay_output', re.compile(r'MadSpin:\s*decay_output\s*=\s*(.+)$')),
    ('unweighting_scheme',
     re.compile(r'MadSpin:\s*unweighting\s*=\s*(.+)$')),
    ('unweight_efficiency_line',
     re.compile(r'(MadSpin unweight efficiency:.*)$')),
    ('overweight_safety_net',
     re.compile(r'(MadSpin overweight safety net:.*)$')),
    ('bw_truncation', re.compile(r'(Breit-Wigner truncation.*)$')),
]

_RE_EFFICIENCY = re.compile(
    r'MadSpin unweight efficiency:\s*([0-9.]+)\s*\((\d+)\s*written'
    r'\s*/\s*(\d+)\s*trials')


def parse_madspin_log(path):
    """Everything the manifest needs out of a MadSpin log.

    The names matter downstream: ``unweighting_scheme`` is the scheme the run
    actually resolved ``auto`` to (which is *not* the same for every spinmode),
    and ``overweight_*`` is the safety net's own report of whether any written
    event carried a non-unit weight, i.e. whether anything was clipped.
    """
    if not path or not os.path.exists(path):
        return {}
    text = open(path, errors='replace').read()
    out = {}
    for line in text.splitlines():
        clean = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
        for name, pattern in _MS_LINE_PATTERNS:
            m = pattern.search(clean)
            if m:
                out[name] = m.group(1).strip()
    m = _RE_EFFICIENCY.search(re.sub(r'\x1b\[[0-9;]*m', '', text))
    if m:
        out['unweight_efficiency'] = float(m.group(1))
        out['n_written'] = int(m.group(2))
        out['n_trials'] = int(m.group(3))
    # the safety net prints "N/M written events carried a non-unit weight"
    m = re.search(r'safety net:\s*(\d+)\s*/\s*(\d+)\s*written events',
                  re.sub(r'\x1b\[[0-9;]*m', '', text))
    if m:
        out['n_overweight_written'] = int(m.group(1))
        out['n_written_total'] = int(m.group(2))
    # anything else that smells of an overweight, verbatim
    over = [re.sub(r'\x1b\[[0-9;]*m', '', ln).strip()
            for ln in text.splitlines()
            if re.search(r'over[- ]?weight|above the maximum|max_weight'
                         r'|larger than the maximum', ln, re.I)]
    if over:
        out['overweight_lines'] = over[-40:]
    return out


def parse_production_log(path):
    """Pull the reported cross-section and its integration error."""
    out = {}
    if not os.path.exists(path):
        return out
    text = open(path, errors='replace').read()
    # LO madevent: "Cross-section :   1.234 +- 0.001 pb"
    for m in re.finditer(r'Cross-section\s*:\s*([-+0-9.eEdD]+)\s*\+-\s*'
                         r'([-+0-9.eEdD]+)\s*pb', text):
        out['xsec_log'] = float(m.group(1))
        out['xerr_log'] = float(m.group(2))
    # aMC@NLO: "total cross section: 1.234 +- 0.005 pb"
    for m in re.finditer(r'[Tt]otal cross[- ]section:\s*([-+0-9.eEdD]+)\s*'
                         r'\+-\s*([-+0-9.eEdD]+)\s*\(?', text):
        out['xsec_log'] = float(m.group(1))
        out['xerr_log'] = float(m.group(2))
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workdir', default='/tmp/smeft_fig5_work',
                    help='where the process dirs and the (large) LHE live')
    ap.add_argument('--nevents-lo', type=int, default=1000000)
    ap.add_argument('--nevents-nlo', type=int, default=500000)
    ap.add_argument('--only', default=None,
                    help='comma-separated sample keys to run')
    ap.add_argument('--stage', default='all',
                    choices=['all', 'produce', 'decay', 'harvest'])
    ap.add_argument('--fixed-order-probe', action='store_true',
                    help='run the fixed_order refusal probe on the NLO sample')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    workdir = os.path.realpath(args.workdir)
    logdir = pjoin(_here, 'logs')
    datadir = pjoin(_here, 'data')
    for d in (workdir, logdir, datadir):
        if not os.path.isdir(d):
            os.makedirs(d)

    wanted = set(args.only.split(',')) if args.only else None
    samples = [s for s in SAMPLES if wanted is None or s['key'] in wanted]

    smeft_models, model_record = prepare_model(workdir)
    for sample in SAMPLES:
        if sample['model'] is None:
            sample['model'] = smeft_models[sample['restriction']]

    meta_path = pjoin(datadir, 'meta.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fp:
            meta = json.load(fp)
    meta.setdefault('setup', {})
    meta['setup'].update(dict(
        sqrt_s_GeV=2 * BEAM_ENERGY,
        beam_energy_GeV=BEAM_ENERGY,
        scale='fixed mu_R = mu_F = %.1f GeV' % FIXED_SCALE,
        mu_R_GeV=FIXED_SCALE, mu_F_GeV=FIXED_SCALE,
        pdf='NNPDF2.3 LO, alpha_s(M_Z)=0.130 (LHAPDF id %d, '
            'NNPDF23_lo_as_0130_qed)' % LHAID,
        lhaid=LHAID, pdlabel='lhapdf',
        pdf_note='the NLO sample deliberately uses the same LO set, so an '
                 'LO/NLO comparison shows a change of order and not of PDF',
        cuts='none (fully inclusive leptons and b quarks; cut_decays False)',
        bwcutoff=BWCUTOFF, BW_cut=BW_CUT,
        decays=DECAYS,
        decay_note='one decay line per t-like particle (multi-channel '
                   'positional rule); both tops decay to electrons so every '
                   'event holds exactly one e- and one e+',
        observable='Delta phi(e-, e+), radians, folded onto [0, pi]',
        dphi_nbins=DPHI_NBINS,
        m_top_GeV=MT, width_top_GeV=WT,
        seed_production=SEED_PROD, seed_madspin=SEED_MADSPIN,
        spinmodes=SPINMODES,
        fixed_order='off -- MC@NLO events are individual signed-weight '
                    'events, not born/counter-event groups, so the option is '
                    'not needed; it is in any case refused by MadSpin in the '
                    'reshuffling spinmodes (PA, madspin)',
        smeft_model=model_record,
        negative_weight_note=(
            'At ctGRe = -1 the O_tG interference is positive over the sampled '
            'phase space: MadSpin evaluates Tr(rho_prod) > 0 for every '
            'production event, including the ~0.02% whose MadEvent weight is '
            'negative -- those are unweighting residuals rather than '
            'negative-matrix-element points, and a MadSpin onshell run on the '
            'negative-weight events alone completes normally. At ctGRe = +1 '
            'every sign flips and spinmode=onshell aborts on the first event '
            'with MadSpinDegenerateWeight, because _check_production_density '
            'accepts only Tr(rho_prod) > 0. The negative-weight fraction is '
            'reported for the production sample and again after each decay; '
            'MadSpin preserves it exactly.'),
        ctGRe_primary=CTGRE,
        ctGRe_cross_check=CTGRE_CHECK,
        sign_convention=('ctGRe = %+g, so the eft_int sample is '
                         '%s2 Re(M_SM^* M_tG); the paper caption is '
                         'written for ctGRe = +1'
                         % (CTGRE, '-' if CTGRE < 0 else '+')),
        generated_by=os.path.relpath(os.path.realpath(__file__), _root),
        workdir=workdir,
    ))
    meta.setdefault('samples', {})

    hist_path = pjoin(datadir, 'histograms.npz')
    hists = {}
    if os.path.exists(hist_path):
        with np.load(hist_path) as z:
            hists = {k: z[k] for k in z.files}

    for sample in samples:
        key = sample['key']
        print('=== %s ===' % key)
        nev = sample.get('nevents')
        if nev is None:
            nev = args.nevents_nlo if sample['nlo'] else args.nevents_lo
        rec = meta['samples'].setdefault(key, {})
        rec.update(dict(
            label=sample['label'],
            process=sample['process'],
            model=sample['model'],
            model_restriction=sample.get('restriction'),
            wilson_coefficients=sample['wilson'],
            perturbative_order='NLO (MC@NLO)' if sample['nlo'] else 'LO',
            param_overrides=sample['param_sets'],
            nevents_requested=nev,
            note=sample['note'],
        ))

        prod_lhe = None
        proc_dir = pjoin(workdir, 'PROC_%s' % key)
        if args.stage in ('all', 'produce'):
            prod_lhe, proc_dir = produce(sample, nev, workdir, logdir,
                                         force=args.force)
        else:
            prod_lhe = _find_production_lhe(proc_dir)
        if prod_lhe:
            xsec = banner_cross_section(prod_lhe)
            xs, xe = xsec['xsec_pb'], xsec['xsec_err_pb']
            rec['production_lhe'] = prod_lhe
            rec['production_xsec_pb'] = xs
            rec['production_xsec_err_pb'] = xe
            rec['production_xsec_detail'] = xsec
            rec.update(parse_production_log(
                pjoin(logdir, 'production_%s.log.txt' % key)))
            rec['production_weights'] = production_weight_stats(prod_lhe)
            print('  production: %d events, %.4f%% negative, xsec = %s +- %s pb'
                  % (rec['production_weights']['n_events'],
                     100.0 * rec['production_weights']['negative_weight_fraction'],
                     xs, xe))

        if sample['nlo'] and args.fixed_order_probe and prod_lhe:
            rec['fixed_order_probe'] = fixed_order_probe(prod_lhe, workdir,
                                                         logdir)

        if sample.get('production_only'):
            rec['production_only'] = True
            rec['spinmodes'] = {}
            print('  production-only sample; no MadSpin stage')
        elif args.stage in ('all', 'decay', 'harvest'):
            rec.setdefault('spinmodes', {})
            for spinmode in SPINMODES:
                srec = rec['spinmodes'].setdefault(spinmode, {})
                decayed = _decayed_path(
                    pjoin(workdir, 'MS_%s_%s' % (key, spinmode)))
                log_path = pjoin(logdir, 'madspin_%s_%s.log.txt'
                                 % (key, spinmode))
                if args.stage in ('all', 'decay'):
                    decayed, log_path, wall = run_madspin(
                        key, prod_lhe, spinmode, workdir, logdir,
                        force=args.force)
                    if wall is not None:
                        srec['madspin_wall_seconds'] = round(wall, 1)
                if not decayed:
                    continue
                srec['spinmode'] = spinmode
                srec['decayed_lhe'] = decayed
                srec['decayed_lhe_bytes'] = os.path.getsize(decayed)
                srec['madspin_log'] = os.path.relpath(log_path, _root)
                srec['seed_madspin'] = SEED_MADSPIN
                srec.update(parse_madspin_log(log_path))
                xsec = banner_cross_section(decayed)
                srec['decayed_xsec_pb'] = xsec['xsec_pb']
                srec['decayed_xsec_err_pb'] = xsec['xsec_err_pb']
                srec['decayed_xsec_detail'] = xsec

                print('  harvesting %s ...' % decayed)
                h = harvest_dphi(decayed)
                tag = '%s_%s' % (key, spinmode)
                hists['edges'] = h['edges']
                for name in ('sumw', 'sumw2', 'count', 'count_pos',
                             'count_neg'):
                    hists['%s_%s' % (tag, name)] = h[name]
                n_signed = h['n_pos'] + h['n_neg']
                srec.update(dict(
                    n_events_in_file=h['n_event'],
                    n_events_histogrammed=int(h['count'].sum()),
                    n_positive_weight=h['n_pos'],
                    n_negative_weight=h['n_neg'],
                    n_zero_weight=h['n_zero'],
                    negative_weight_fraction=(h['n_neg'] / n_signed
                                              if n_signed else 0.0),
                    sum_weights=h['sum_w'],
                    sum_abs_weights=h['sum_abs_w'],
                    mean_weight=(h['sum_w'] / h['n_event']
                                 if h['n_event'] else None),
                    events_without_exactly_one_e_minus_and_one_e_plus=(
                        h['bad_multiplicity']),
                    lepton_multiplicity_census=h['mult_hist'],
                    histogram_keys=['%s_%s' % (tag, n) for n in
                                    ('sumw', 'sumw2', 'count', 'count_pos',
                                     'count_neg')],
                ))
                # How to turn the histogram into a cross-section, spelled out
                # because the LHE convention here is the one that catches
                # people: IDWTUP = -4, so XWGTUP is (up to its sign) the whole
                # cross-section on *every* event and the estimator is the MEAN
                # of the weights, not their sum.
                xs = srec.get('decayed_xsec_pb')
                mean = srec.get('mean_weight')
                srec['normalisation'] = dict(
                    weight_convention=(
                        'LHE IDWTUP = -4: XWGTUP carries the full '
                        'cross-section in pb with the event sign, so '
                        'sigma = sum(w) / N, not sum(w)'),
                    dsigma_dphi=(
                        'd sigma / d Delta phi in bin i = sumw[i] / '
                        'n_events_in_file / bin_width_rad, in pb/rad'),
                    sigma_total=('sum(sumw) / n_events_in_file = %s pb, which '
                                 'matches the decayed LHE <init> XSECUP'
                                 % mean),
                    bin_width_rad=math.pi / DPHI_NBINS,
                    n_events_in_file=h['n_event'],
                    mean_weight_over_init_xsec=(mean / xs if xs else None),
                    statistical_error=(
                        'per-bin variance is sumw2[i]; the error on '
                        'd sigma / d Delta phi in bin i is sqrt(sumw2[i]) / '
                        'n_events_in_file / bin_width_rad'),
                    sign=('sumw may be negative bin by bin for an '
                          'interference sample; that is physical, not a bug'),
                )
                print('    %d events, %.4f%% negative, sum(w) = %.6g pb'
                      % (h['n_event'],
                         100.0 * h['n_neg'] / max(1, n_signed),
                         h['sum_w']))
                if h['bad_multiplicity']:
                    print('    WARNING: %d events without exactly one e-/e+'
                          % h['bad_multiplicity'])

        with open(meta_path, 'w') as fp:
            json.dump(meta, fp, indent=1, sort_keys=True, default=str)
        np.savez_compressed(hist_path, **hists)

    print('\nmeta   : %s' % meta_path)
    print('hists  : %s' % hist_path)


if __name__ == '__main__':
    main()
