#!/usr/bin/env python3
"""Second, independently seeded q qbar sample: production, four MadSpin modes,
and a reseeded four-lepton NLO truth.

Everything except the seeds is the first sample's: the same run_zz_nlo.py
driver, hence the same process, cuts (pt_min_pdg = {23: 1}, zz_equivalent_cuts_nlo.f
on the truth), fixed scale mu = m_Z, nn23lo1 / lhaid 230000, 6500+6500 GeV,
bwcutoff = 15 and BW_cut = 15.

Seeds
-----
sample 1 (claude/ms-qq-coefficients)   production 4321, MadSpin 7777,
                                       truth blocks 4321..4324
sample 2 (this run)                    production 8765, MadSpin 1357,
                                       truth blocks 5321..5324

Everything durable: the work tree is under ~/Documents/madspin_validation_samples.
"""
import importlib.util
import json
import os
import sys
import time

HERE = ('/Users/omattelaer/Documents/git_workspace/madgraph5/.claude/'
        'worktrees/qqseed/MadSpin/validation/zz_nlo')
BASEDIR = os.path.expanduser(
    '~/Documents/madspin_validation_samples/t130_qq_seed_check/work')
LOGS = os.path.expanduser(
    '~/Documents/madspin_validation_samples/t130_qq_seed_check/logs')

SEED_PROD_2 = 8765
SEED_MADSPIN_2 = 1357
SEED_TRUTH_2 = [5321, 5322, 5323, 5324]
N_PROD = 200000
N_TRUTH_BLOCK = 50000
NB_CORE = 16
MODES = ['madspin', 'onshell', 'none', 'PA']

spec = importlib.util.spec_from_file_location(
    'run_zz_nlo', os.path.join(HERE, 'run_zz_nlo.py'))
R = importlib.util.module_from_spec(spec)
sys.modules['run_zz_nlo'] = R
spec.loader.exec_module(R)

for d in (BASEDIR, LOGS):
    os.makedirs(d, exist_ok=True)
R._TIMINGS = os.path.join(BASEDIR, 'timings.json')

# --- the only changes: the seeds -----------------------------------------
R.SEED_PROD = SEED_PROD_2
R.SEED_MADSPIN = SEED_MADSPIN_2
R.NEVENTS = N_PROD
R.RUN_CARD_NLO['iseed'] = SEED_PROD_2
R.RUN_CARD_LO['iseed'] = SEED_PROD_2

print('root      :', R._ROOT)
print('cuts file :', R.CUTS_FILE_NLO)
print('pt_min_pdg:', R.PT_MIN_PDG)
print('BW_cut    :', R.BW_CUT, ' bwcutoff:', R.RUN_CARD_NLO['bwcutoff'])
print('seeds     : prod %d  madspin %d  truth %s'
      % (SEED_PROD_2, SEED_MADSPIN_2, SEED_TRUTH_2))
sys.stdout.flush()

meta = {'seeds': {'production': SEED_PROD_2, 'madspin': SEED_MADSPIN_2,
                  'truth_blocks': SEED_TRUTH_2},
        'nevents': {'prod': N_PROD, 'truth_block': N_TRUTH_BLOCK},
        'nb_core': NB_CORE, 'basedir': BASEDIR}

t0 = time.time()

# ---------------------------------------------------------------- outputs
print('=== stage output: p p > z z [QCD] and p p > e+ e- mu+ mu- / a [QCD] ===',
      flush=True)
R.generate_outputs(BASEDIR, ('nlo', 'truth'), LOGS)

# ------------------------------------------------------------- production
print('=== stage prod: p p > z z [QCD], %d events, iseed %d ==='
      % (N_PROD, SEED_PROD_2), flush=True)
R.RUN_CARD_NLO['nevents'] = N_PROD
R.RUN_CARD_NLO['iseed'] = SEED_PROD_2
R.prepare_card(BASEDIR, 'nlo')
lhe = R.run_sample(BASEDIR, 'nlo', NB_CORE, LOGS)
meta['prod_lhe'] = lhe
meta['prod_seed_actual'] = R.banner_seed(lhe)
meta['prod_banner'] = R.banner_cross(lhe)
print('prod:', lhe, meta['prod_seed_actual'], meta['prod_banner'], flush=True)

# ---------------------------------------------------------------- madspin
print('=== stage madspin, seed %d ===' % SEED_MADSPIN_2, flush=True)
meta['madspin'] = R.stage_madspin(BASEDIR, lhe, MODES, NB_CORE, LOGS)

# ------------------------------------------------------------------ truth
print('=== stage truth: 4 x %d, iseed %s ===' % (N_TRUTH_BLOCK, SEED_TRUTH_2),
      flush=True)
meta['truth'] = {}
for i, seed in enumerate(SEED_TRUTH_2):
    run = 'b%02d' % (i + 1)
    R.RUN_CARD_NLO['nevents'] = N_TRUTH_BLOCK
    R.RUN_CARD_NLO['iseed'] = seed
    R.prepare_card(BASEDIR, 'truth')
    p = R.run_sample(BASEDIR, 'truth', NB_CORE, LOGS, run_name=run)
    meta['truth'][run] = {'lhe': p, 'seed_requested': seed,
                          'seed_actual': R.banner_seed(p),
                          'banner': R.banner_cross(p)}
    print('truth %s: %s seed_actual=%s' % (run, p, meta['truth'][run]['seed_actual']),
          flush=True)

meta['wall_seconds'] = time.time() - t0
with open(os.path.join(BASEDIR, '..', 'seeds_and_runs.json'), 'w') as fp:
    json.dump(meta, fp, indent=1)
print('DONE in %.0f s' % meta['wall_seconds'])
