#!/usr/bin/env python3
"""Measure ``m_tt`` near the ``2 m_t`` threshold for a ``2 -> 2`` production:
MadSpin's spinmodes against a doubly-resonant off-shell MadGraph truth, on
``p p > t t~`` with **no** recoiling jet.

The control experiment for the ``t t~ j`` study
-----------------------------------------------
This is the sibling of ``MadSpin/validation/mtt_threshold``, which asks the same
question of ``p p > t t~ j``.  Everything -- model, PDF, beams, scales,
``bwcutoff``, the observable, the binning, the harvester, the plotting -- is the
same code, imported from that directory and re-pointed at a different process.
The one thing that changes is the production multiplicity, and that is the whole
experiment.

In the ``t t~ j`` study the off-shell spinmodes populate ``m_tt < 2 m_t``.  The
mechanism is ``Event.reshuffle_production``, which hands **every** top-level
final-state momentum to RAMBO's ``mass_shuffle`` at fixed ``sqrt(shat)`` -- for
``p p > t t~ j`` that is ``t``, ``t~`` *and the recoiling jet*.  The jet's
momentum is rescaled by the same ``chi`` as the tops', so the ``t t~``
four-momentum is not preserved and ``m_tt`` moves::

    m_tt^2(after)  = shat - 2 sqrt(shat) * chi * E_j        (massless recoil)
    m_tt^2(before) = shat - 2 sqrt(shat) *       E_j

Take the jet away and that term goes with it.  For ``p p > t t~`` the two tops
are the entire final state, so ``m_tt = sqrt(shat)`` identically -- before the
reshuffle and after it.  ``mass_shuffle`` holds ``sqrt(shat)`` fixed by
construction, therefore it cannot move ``m_tt`` at all, for any drawn pair of
virtualities, in any of the density modes.  ``onshell`` never reshuffles in the
first place, and ``madspin_v1`` rebuilds the point from the decay-chain topology
holding ``shat`` and the production tree's invariants fixed -- and with two
final-state particles the ``t t~`` invariant *is* ``shat``.

So the expectation is that **every** spinmode is structurally empty below
``2 m_t``, not just ``onshell``.  This driver measures that rather than asserting
it: ``pair_delta`` (inherited unchanged) walks the production and decayed LHE
files in step, checks the pairing with ``max |Delta sqrt(shat)| = 0``, and
reports the ``Delta m_tt`` moments and the threshold-crossing counts per mode.
If any mode does move ``m_tt``, that is the result and the figure shows it.

What differs from the ``t t~ j`` setup, and why
-----------------------------------------------
* ``PRODUCTION_PROCESS = 'p p > t t~'``, ``TRUTH_PROCESS =
  'p p > t t~, t > w+ b, t~ > w- b~'``.
* **no jet cuts.**  ``ptj``/``etaj`` in the ``t t~ j`` run card exist to make
  the real-emission process finite; there is no jet in this final state for them
  to act on, so carrying them over would record a cut in ``meta.json`` that
  applies to nothing.  This sample is inclusive, and the truth's acceptance is
  the production sample's for the same reason it was there: ``cut_decays`` is
  off, so nothing cuts on the top decay products either.
* everything else -- ``sm``, LO, 6.5+6.5 TeV, ``nn23lo1``, fixed
  ``mu_R = mu_F = 173 GeV``, ``bwcutoff = 15`` matched to ``BW_cut = 15``,
  ``MB = 4.7`` and no ``b`` in ``p``/``j`` -- is imported verbatim.

Usage
-----
    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 run_mtt_threshold.py --stage prod  --nevents-prod 1000000 \
            --basedir /tmp/w --nb-core 8
    python3 run_mtt_threshold.py --stage truth --nevents-truth 1000000 \
            --truth-runs 5 --basedir /tmp/w --nb-core 8
    python3 run_mtt_threshold.py --stage madspin \
            --modes madspin,PA,onshell,madspin_v1 --basedir /tmp/w --nb-core 8
    python3 run_mtt_threshold.py --stage harvest --basedir /tmp/w --cross-check

``--truth-runs N`` is new here and is a convenience, not a change of method:
MG5 refuses more than 1M events in one ``generate_events``
(``madevent_interface.check_nb_events``), so the ``t t~ j`` study got its 5M by
launching the same output directory five times with consecutive seeds and
passing the extra files to ``--truth-lhe``.  This does the launching in the
driver and hands the files to the inherited ``--truth-lhe`` path, so the pooling
arithmetic is still ``harvest_many``'s.
"""

from __future__ import absolute_import
from __future__ import division

import importlib.util
import os
import subprocess
import sys
import time

pjoin = os.path.join

_here = os.path.dirname(os.path.realpath(__file__))
_sibling = pjoin(os.path.dirname(_here), 'mtt_threshold', 'run_mtt_threshold.py')


def _load_sibling():
    """Import the ``t t~ j`` driver as a module, by path.

    Loading it rather than copying it is the point: the harvester, the
    ``m_tt`` reconstruction, the ``Delta m_tt`` pairing and its
    ``sqrt(shat)`` check, the histogram grid and the whole ``main`` flow are
    then literally the same code on both sides of the comparison, and cannot
    drift apart.  The process-dependent module globals are re-bound below.
    """
    if not os.path.exists(_sibling):
        raise SystemExit('cannot find the t t~ j driver at %s' % _sibling)
    spec = importlib.util.spec_from_file_location(
        'run_mtt_threshold_ttj', _sibling)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['run_mtt_threshold_ttj'] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load_sibling()

# ---------------------------------------------------------------- the process
PRODUCTION_PROCESS = 'p p > t t~'
TRUTH_PROCESS = 'p p > t t~, t > w+ b, t~ > w- b~'

# The run card, minus the two jet cuts: there is no jet.  ``bwcutoff`` is the
# one entry that is a parameter of the figure rather than a setting -- it fixes
# how far below ``2 m_t`` the truth's Breit-Wigner tails reach, and it is
# matched to MadSpin's ``BW_cut`` in the same convention (15 widths in the
# mass).
RUN_CARD = {k: v for k, v in base.RUN_CARD.items() if k not in ('ptj', 'etaj')}

base.PRODUCTION_PROCESS = PRODUCTION_PROCESS
base.TRUTH_PROCESS = TRUTH_PROCESS
base.RUN_CARD = RUN_CARD
# ``main()`` uses the module's ``_here`` for the default ``--outdir``; point it
# at this directory so ``data/`` lands beside this file and not beside the
# ``t t~ j`` study's.
base._here = _here

# Re-export the shared setup so a reader of this file sees the whole
# configuration without having to open the sibling.
MODEL = base.MODEL
MULTIPARTICLES = base.MULTIPARTICLES
DECAYS = base.DECAYS
BWCUTOFF = base.BWCUTOFF
MODES = base.MODES
CONTROLS = base.CONTROLS
MTT_LO, MTT_HI, MTT_NBINS = base.MTT_LO, base.MTT_HI, base.MTT_NBINS
MT_POLE, WT_POLE, TWO_MT = base.MT_POLE, base.WT_POLE, base.TWO_MT
SEED_PROD, SEED_TRUTH, SEED_MADSPIN = (base.SEED_PROD, base.SEED_TRUTH,
                                       base.SEED_MADSPIN)

# Inherited verbatim, and named here so the reuse is visible.
edges = base.edges
delta_edges = base.delta_edges
harvest = base.harvest
harvest_many = base.harvest_many
pair_delta = base.pair_delta
cross_check = base.cross_check


# --------------------------------------------------------------------------
# Extra truth runs
# --------------------------------------------------------------------------
def _write_extra_truth_script(path, procdir, nevents, nb_core, seed):
    """A ``launch`` of an existing truth output directory, with a new seed.

    Same process, same cards, a different random seed: an independent sample of
    the same distribution, which is what ``harvest_many`` pools.  The run card
    settings are re-applied because ``launch`` re-opens the cards each time.
    """
    lines = ['set automatic_html_opening False --no_save',
             'set nb_core %d' % nb_core,
             'launch %s' % procdir,
             'madspin=OFF', 'shower=OFF', 'detector=OFF', 'analysis=OFF',
             'done',
             'set nevents %d' % nevents,
             'set iseed %d' % seed]
    for key, val in RUN_CARD.items():
        lines.append('set %s %s' % (key, val))
    lines.append('done')
    with open(path, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')


def run_truth_extra(basedir, nruns, nevents, nb_core):
    """Launch runs 02..``nruns`` of the truth process; return their LHE paths.

    MG5 caps one ``generate_events`` at 1M events, so more statistics means more
    runs.  Seeds are ``SEED_TRUTH + i``, consecutive and recorded in
    ``meta.json`` by the caller.
    """
    procdir = pjoin(basedir, 'TRUTH')
    out = []
    for i in range(2, nruns + 1):
        lhe = pjoin(procdir, 'Events', 'run_%02d' % i,
                    'unweighted_events.lhe.gz')
        if os.path.exists(lhe):
            print('truth run_%02d: reusing %s' % (i, lhe))
            out.append(lhe)
            continue
        script = pjoin(basedir, 'truth_%02d.mg5' % i)
        _write_extra_truth_script(script, procdir, nevents, nb_core,
                                  SEED_TRUTH + i - 1)
        log = pjoin(basedir, 'truth_%02d.log' % i)
        print('truth run_%02d: generating %d events (log %s)'
              % (i, nevents, log))
        sys.stdout.flush()
        t0 = time.time()
        with open(log, 'w') as fp:
            ret = subprocess.call(
                [pjoin(base._root, 'bin', 'mg5_aMC'), '-f', script],
                stdout=fp, stderr=subprocess.STDOUT)
        print('truth run_%02d: %.0f s, rc=%d' % (i, time.time() - t0, ret))
        sys.stdout.flush()
        if ret != 0 or not os.path.exists(lhe):
            raise RuntimeError('truth run_%02d failed, see %s' % (i, log))
        out.append(lhe)
    return out


def main():
    """``base.main()``, with ``--truth-runs`` folded into ``--truth-lhe``.

    ``base.main`` parses its own argv, so the extra option is stripped out here
    and its product -- the paths of the additional truth runs -- is appended to
    the ``--truth-lhe`` list the inherited harvester already understands.
    """
    argv = list(sys.argv[1:])
    nruns = 1
    if '--truth-runs' in argv:
        i = argv.index('--truth-runs')
        nruns = int(argv[i + 1])
        del argv[i:i + 2]

    # Peek at the two options ``run_truth_extra`` needs.  ``base.main`` re-parses
    # everything itself; this only reads.
    def _opt(name, default):
        return argv[argv.index(name) + 1] if name in argv else default
    basedir = _opt('--basedir', '/tmp/mtt_threshold_work')
    nb_core = int(_opt('--nb-core', 8))
    nev_truth = int(_opt('--nevents-truth', 1000000))
    stage = _opt('--stage', 'all')

    extra = []
    if nruns > 1 and stage in ('all', 'truth', 'harvest'):
        if stage == 'harvest':
            # Do not generate during a harvest: only collect what is on disk.
            procdir = pjoin(basedir, 'TRUTH', 'Events')
            for i in range(2, nruns + 1):
                cand = pjoin(procdir, 'run_%02d' % i,
                             'unweighted_events.lhe.gz')
                if os.path.exists(cand):
                    extra.append(cand)
        else:
            # run_01 has to exist before ``launch <procdir>`` can make run_02:
            # the output directory is what the extra runs re-launch.  This is a
            # no-op when it is already on disk.
            base.run_truth(basedir, nev_truth, nb_core)
            extra = run_truth_extra(basedir, nruns, nev_truth, nb_core)

    if extra:
        if '--truth-lhe' in argv:
            i = argv.index('--truth-lhe')
            argv[i + 1] = ','.join([argv[i + 1]] + extra)
        else:
            argv += ['--truth-lhe', ','.join(extra)]

    sys.argv = [sys.argv[0]] + argv
    base.main()

    # The inherited harvester records one truth seed because the t t~ j study's
    # extra runs were launched by hand outside it.  Here the driver launched
    # them, so it knows their seeds and writes them down -- along with the two
    # things about this setup that are absences and would otherwise leave no
    # trace in meta.json at all.
    if stage in ('all', 'harvest'):
        outdir = _opt('--outdir', pjoin(_here, 'data'))
        meta_path = pjoin(outdir, 'meta.json')
        if os.path.exists(meta_path):
            import json
            with open(meta_path) as fp:
                meta = json.load(fp)
            meta['truth_nruns'] = nruns
            meta['seed_truth_runs'] = [SEED_TRUTH + i for i in range(nruns)]
            meta['cuts'] = (
                'none.  The final state is t t~ with no jet, so the ptj/etaj '
                'cuts the p p > t t~ j study needs to make the real emission '
                'finite have nothing to act on and are not set.  cut_decays is '
                'off, so nothing cuts on the top decay products either and the '
                'truth sample\'s acceptance is the production sample\'s.')
            meta['sibling_study'] = (
                'MadSpin/validation/mtt_threshold -- the same measurement on '
                'p p > t t~ j.  This driver imports that one\'s harvester, '
                'binning and pairing code and re-points it at the 2 -> 2 '
                'process; the plotting scripts do the same with its figure.')
            with open(meta_path, 'w') as fp:
                json.dump(meta, fp, indent=2, sort_keys=True)
            print('meta.json: recorded truth seeds %s and the cut note'
                  % meta['seed_truth_runs'])

        # The inherited harvester copies the log and script of truth run_01
        # only, because that is the one it launched.  Runs 02..N were launched
        # here, so their provenance is copied here.  ``.log.txt``, not
        # ``.log``: the repository's .gitignore has a blanket ``*.log`` rule
        # and a copy under that name would silently never be committed.
        import shutil
        logdir = pjoin(outdir, 'logs')
        if os.path.isdir(logdir):
            for i in range(2, nruns + 1):
                for src, dest in (
                        (pjoin(basedir, 'truth_%02d.log' % i),
                         'mg5_truth_run%02d.log.txt' % i),
                        (pjoin(basedir, 'truth_%02d.mg5' % i),
                         'mg5_truth_run%02d_script.dat' % i)):
                    if os.path.exists(src):
                        shutil.copy(src, pjoin(logdir, dest))


if __name__ == '__main__':
    main()
