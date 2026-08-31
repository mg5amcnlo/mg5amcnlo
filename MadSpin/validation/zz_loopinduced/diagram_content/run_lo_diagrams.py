#!/usr/bin/env python3
"""Is the q qbar f_0 deficit the truth's non-doubly-resonant diagrams?

The NLO truth is ``p p > e+ e- mu+ mu- / a [QCD]``.  Its Born content, read off
``born_conf.inc`` of the generated ``P0_uux_mumemepmup_no_a``, is SIX diagrams
per subprocess, and only TWO of them are the doubly-resonant ``q qbar > z z``
that MadSpin's production sample carries.  The other four are the
singly-resonant ``q qbar > Z*(m_4l) > l+ l- Z``, i.e. a Z radiated off a lepton
leg of an s-channel Drell-Yan Z*.  MadSpin cannot carry them: its production
process has two z in the final state by construction.

This script measures what those four diagrams do to ``f_0``, at LO, where the
doubly-resonant subset can be generated on its own with the required-s-channel
syntax ``p p > z z > e+ e- mu+ mu-``:

  full4l   p p > e+ e- mu+ mu- / a                 all 6 Born diagrams, off shell
  zz4l     p p > e+ e- mu+ mu- / a e+ e- mu+ mu-   the 2 doubly-resonant ones,
                                                   off shell, same window
  zz       p p > z z  + MadSpin             the LO twin of the chain under test

f_0(full4l) - f_0(zz4l) is the diagram-content effect and nothing else: same
order, same PDF, same scale, same window, same estimator, same code path.
f_0(zz4l) - f_0(zz+MadSpin) is what is left over, i.e. the off-shell production
matrix element against MadSpin's on-shell-plus-reshuffle one.

Every cut is applied OFFLINE, identically on all three samples (the mass window
and pt(l+l-) > 1 GeV of the study), so no generator-level cut can act on
different objects on the two sides -- which is the trap T129 found in the g g
study and which the NLO run of this study still carries.

    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 run_lo_diagrams.py --nevents 500000
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
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

PROC = {
    'full4l': 'p p > e+ e- mu+ mu- / a',
    # The required-s-channel form ``p p > z z > e+ e- mu+ mu-`` is rejected by
    # this MG5 ("Invalid \"> A A >\" syntax"), so the doubly-resonant subset is
    # selected the other way round: FORBID the leptons as internal propagators.
    # The four singly-resonant diagrams each carry an internal e or mu line
    # (born_conf.inc: SPROP -11, 11, -13, 13); the two doubly-resonant ones
    # carry a t-channel quark (TPRID 2) and nothing else.  Checked: 6 diagrams
    # become exactly 2.  Both lepton currents are conserved for massless
    # leptons, so the k^mu k^nu / MZ^2 pieces of the two Z propagators drop and
    # the subset is the gauge-invariant double-pole part.
    'zz4l':   'p p > e+ e- mu+ mu- / a e+ e- mu+ mu-',
    'zz':     'p p > z z',
}
OUT = {'full4l': 'lo_full4l', 'zz4l': 'lo_zz4l', 'zz': 'lo_zz'}

# The study's LO run card, plus every lepton cut explicitly off.  A four-lepton
# LO card comes with ptl = 10, etal = 2.5 and drll = 0.4 by default and the
# p p > z z card does not, so leaving them alone would put the two sides of the
# comparison on different selections -- which is the whole point of the exercise.
CARD = dict(R.RUN_CARD_LO)
CARD.update({
    'ebeam1': '6500.0', 'ebeam2': '6500.0',
    'ptl': '0.0', 'etal': '-1.0', 'etalmin': '0.0',
    'drll': '0.0', 'drllmax': '-1.0',
    'mmll': '0.0', 'mmllmax': '-1.0', 'mmnl': '0.0', 'mmnlmax': '-1.0',
    'xptl': '0.0',
    # dict-valued on this card, and the banner parser rejects a bare ``True``
    'mxx_only_part_antipart': "{'default': True}",
})
# A loose generator-level floor on m(l+ l-), well BELOW the study's window
# (54.567 GeV), so that the offline window is the operative cut on every sample
# and the generator never sees a boundary the analysis also sees.  Without it
# the full4l run spends most of its events at m(l+ l-) far off the Z pole.
MXX_MIN = 40.0


def set_dict_entry(path, key, value):
    """Set a dict-valued run-card entry, keeping its comment."""
    lines = open(path).read().split('\n')
    pat = re.compile(r'^(.*?)=\s*%s\s*(!.*)?$' % re.escape(key))
    done = False
    for i, l in enumerate(lines):
        m = pat.match(l)
        if m:
            lines[i] = ' %s = %s %s' % (value, key, m.group(2) or '')
            done = True
    if not done:
        raise RuntimeError('%s not found in %s' % (key, path))
    open(path, 'w').write('\n'.join(lines))


def write_card(basedir, tag, nevents, seed):
    card = pjoin(basedir, OUT[tag], 'Cards', 'run_card.dat')
    kv = dict(CARD)
    kv['nevents'] = nevents
    kv['iseed'] = seed
    # MG5 writes a run card TAILORED to the final state: the four-lepton card
    # has ptl / etal / drll and no ptheavy, the p p > z z card has ptheavy and
    # no lepton entry at all.  Only the entries a card actually carries are
    # written, and which ones those were is printed and stored, so that a
    # setting silently absent is still visible in the record.
    present = set(re.findall(r'=\s*([A-Za-z_]\w*)\s*(?:!|$)',
                             open(card).read(), re.M))
    absent = sorted(set(kv) - present - {'nevents', 'iseed'})
    kv = {k: v for k, v in kv.items() if k in present or k in ('nevents', 'iseed')}
    R.set_run_card(card, **kv)
    # pt_min_pdg is left EMPTY on every sample, p p > z z included: the pt cut
    # of this study is applied offline to the reconstructed pairs on all three,
    # so it cannot act on an on-shell z on one side and on a reconstructed pair
    # on the other.
    R.set_pt_min_pdg(card, '{}')
    if 'mxx_min_pdg' in present:
        set_dict_entry(card, 'mxx_min_pdg',
                       '{11: %s, 13: %s}' % (MXX_MIN, MXX_MIN))
    return card, absent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basedir', default=os.path.expanduser(
        '~/Documents/madspin_validation_samples/t131_qq_diagrams/work'))
    ap.add_argument('--nevents', type=int, default=500000)
    ap.add_argument('--nevents-zz', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=24680)
    ap.add_argument('--seed-madspin', type=int, default=13579)
    ap.add_argument('--nb-core', type=int, default=16)
    ap.add_argument('--tags', default='full4l,zz4l,zz')
    args = ap.parse_args()

    base = args.basedir
    logs = pjoin(base, '..', 'logs')
    for d in (base, logs):
        os.makedirs(d, exist_ok=True)
    R._TIMINGS = pjoin(base, 'timings.json')
    R.OUTDIR.update(OUT)
    R.PROC.update(PROC)
    R.SEED_MADSPIN = args.seed_madspin

    tags = args.tags.split(',')
    meta = {'proc': PROC, 'nevents': args.nevents, 'nevents_zz': args.nevents_zz,
            'seed': args.seed, 'seed_madspin': args.seed_madspin,
            'mxx_min_pdg': MXX_MIN, 'sha': R.code_sha(), 'basedir': base}
    t0 = time.time()
    R.generate_outputs(base, tags, logs)
    for tag in tags:
        n = args.nevents_zz if tag == 'zz' else args.nevents
        _c, absent = write_card(base, tag, n, args.seed)
        meta.setdefault('card_entries_absent', {})[tag] = absent
        print('%s: run-card entries not present on this card: %s'
              % (tag, absent), flush=True)
        shutil.copy(pjoin(base, OUT[tag], 'Cards', 'run_card.dat'),
                    pjoin(logs, 'run_card_%s.dat' % tag))
        lhe = R.run_sample(base, tag, args.nb_core, logs)
        meta.setdefault('lhe', {})[tag] = lhe
        print(tag, lhe, flush=True)
    if 'zz' in tags:
        meta['madspin'] = R.stage_madspin(base, meta['lhe']['zz'],
                                          ['madspin', 'none'],
                                          args.nb_core, logs)
    meta['wall_seconds'] = time.time() - t0
    with open(pjoin(base, '..', 'seeds_and_runs.json'), 'w') as fp:
        json.dump(meta, fp, indent=1)
    print('DONE %.0f s' % meta['wall_seconds'])


if __name__ == '__main__':
    main()
