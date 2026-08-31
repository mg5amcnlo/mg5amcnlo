#!/usr/bin/env python3
"""The NLO truth with the four singly-resonant Born diagrams REMOVED.

``p p > e+ e- mu+ mu- / a e+ e- mu+ mu- [QCD]`` -- the same NLO calculation as
the study's truth, with the leptons forbidden as internal propagators, which
takes the Born content from 6 diagrams per subprocess to the 2 doubly-resonant
``q qbar > z z`` ones (checked: MG5 reports 16 born diagrams over 8 subprocesses
against the full process's 48).  Everything else is the study's own: the same
``run_zz_nlo.py`` helpers, the same NLO run card, the same
``zz_equivalent_cuts_nlo.f``, ``pt_min_pdg = {23: 1}``, fixed mu = m_Z,
nn23lo1 / 230000, 6500+6500 GeV, bwcutoff = 15.

This is the direct NLO measurement of what the LO study measures at LO.

    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 run_nlo_zzonly.py
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..'))
_ZZNLO = os.path.join(_ROOT, 'MadSpin', 'validation', 'zz_nlo')
pjoin = os.path.join

spec = importlib.util.spec_from_file_location(
    'run_zz_nlo', pjoin(_ZZNLO, 'run_zz_nlo.py'))
R = importlib.util.module_from_spec(spec)
sys.modules['run_zz_nlo'] = R
spec.loader.exec_module(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basedir', default=os.path.expanduser(
        '~/Documents/madspin_validation_samples/t131_qq_diagrams/nlo'))
    ap.add_argument('--nblocks', type=int, default=8)
    ap.add_argument('--per-block', type=int, default=50000)
    ap.add_argument('--seed0', type=int, default=7321)
    ap.add_argument('--nb-core', type=int, default=16)
    args = ap.parse_args()

    base = args.basedir
    logs = pjoin(base, '..', 'logs_nlo')
    for d in (base, logs):
        os.makedirs(d, exist_ok=True)
    R._TIMINGS = pjoin(base, 'timings.json')
    # The driver's ``truth`` tag is REPOINTED rather than a new tag added.
    # Three of its branches key off the literal string 'truth' -- prepare_card
    # attaches zz_equivalent_cuts_nlo.f only for it, run_sample picks
    # bin/aMCatNLO over bin/madevent only for it, and set_run_card picks the NLO
    # card only for it -- so a new tag would silently get the LO treatment on
    # all three.  The redirection is safe because this driver writes into its
    # own basedir and runs nothing else.
    R.PROC['truth'] = 'p p > e+ e- mu+ mu- / a e+ e- mu+ mu- [QCD]'
    R.OUTDIR['truth'] = 'pp4l_zzonly_nlo'
    TAG = 'truth'

    meta = {'proc': R.PROC[TAG], 'sha': R.code_sha(),
            'per_block': args.per_block, 'nblocks': args.nblocks,
            'seed0': args.seed0, 'basedir': base,
            'cuts_file': R.CUTS_FILE_NLO, 'pt_min_pdg': R.PT_MIN_PDG,
            'blocks': {}}
    t0 = time.time()
    R.generate_outputs(base, (TAG,), logs)
    for i in range(args.nblocks):
        run = 'z%02d' % (i + 1)
        seed = args.seed0 + i
        R.RUN_CARD_NLO['nevents'] = args.per_block
        R.RUN_CARD_NLO['iseed'] = seed
        card = R.prepare_card(base, TAG)
        if i == 0:
            shutil.copy(card, pjoin(logs, 'run_card_zzonly.dat'))
        p = R.run_sample(base, TAG, args.nb_core, logs, run_name=run)
        meta['blocks'][run] = {'lhe': p, 'seed_requested': seed,
                               'seed_actual': R.banner_seed(p),
                               'banner': R.banner_cross(p)}
        print(run, p, meta['blocks'][run]['seed_actual'], flush=True)
    meta['wall_seconds'] = time.time() - t0
    json.dump(meta, open(pjoin(base, '..', 'seeds_and_runs_nlo.json'), 'w'),
              indent=1)
    print('DONE %.0f s' % meta['wall_seconds'])


if __name__ == '__main__':
    main()
