#!/usr/bin/env python3
"""Drive the three configurations whose mass/virtuality stage is under
investigation, with the instrumented launcher, off ONE production sample.

    PA        density_keep_jacobian = True   (the default)   eps_m = 3.50
    PAnojac   density_keep_jacobian = False                  eps_m = 1.10
    madspin   fully offshell                                 eps_m = 3.31

Same process, seed and event count as
``doc/madspin_unweighting_efficiency/measure_unweighting.py`` (``p p > t t~``,
both tops leptonic, seed 42), so the ``eps_m`` this reproduces is the one in
that table.  The unweighting is ``sequential`` throughout: that is the scheme
whose mass stage has a bound of its own.

Usage::

    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 doc/madspin_pa_mass_stage/probe_pa_mass_stage.py \
        --nevents 100000 --workdir /scratch/dir --outdir doc/madspin_pa_mass_stage/data
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import os
import shutil
import subprocess
import sys
import time

pjoin = os.path.join

_here = os.path.dirname(os.path.realpath(__file__))
_root = os.path.split(os.path.split(_here)[0])[0]
if _root not in sys.path:
    sys.path.insert(0, _root)

from tests.parallel_tests.madspin_comparator import (  # noqa: E402
    MadSpinFactory, SpinModeConfig)

TTBAR_LEPTONIC = dict(
    production_process='p p > t t~',
    decays=['t > b w+, w+ > l+ vl',
            't~ > b~ w-, w- > l- vl~'],
    multiparticles={'p': 'g u d s c u~ d~ s~ c~',
                    'l+': 'e+ mu+', 'vl': 've vm',
                    'l-': 'e- mu-', 'vl~': 've~ vm~'},
    extra_run_card={'ebeam1': 6500, 'ebeam2': 6500},
)

# (key, spinmode, extra ``set`` lines)
CONFIGS = [
    ('PA', 'PA', {}),
    ('PAnojac', 'PA', {'density_keep_jacobian': 'False'}),
    ('madspin', 'madspin', {}),
]


def make_factory(nevents, workdir, seed=42):
    os.makedirs(workdir, exist_ok=True)
    return MadSpinFactory(name='pa_mass_stage', nevents=nevents, seed=seed,
                          base_dir=workdir,
                          extra_madspin_settings={'nb_core': 1},
                          **TTBAR_LEPTONIC)


def run_one(factory, key, spinmode, extra, outdir, unweighting='sequential'):
    """Write the card exactly as the factory would, then run it through the
    instrumented launcher instead of ``MadSpin/madspin``."""
    run_dir = pjoin(factory.base_dir, 'probe_%s' % key)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    evt_path = pjoin(run_dir, 'events.lhe.gz')
    shutil.copy(factory.events_file, evt_path)

    settings = {'unweighting': unweighting}
    settings.update(extra)
    card_path = pjoin(run_dir, 'madspin_card.dat')
    factory._write_madspin_card(card_path, evt_path,
                                SpinModeConfig(key, spinmode), settings)

    stream = pjoin(outdir, '%s.stream.txt' % key)
    log_path = pjoin(outdir, '%s.log' % key)
    env = dict(os.environ, MS_PROBE_OUT=stream)
    start = time.time()
    with open(log_path, 'w') as logf:
        ret = subprocess.call(
            [sys.executable, '-O', pjoin(_here, 'probe_launcher.py'), card_path],
            cwd=run_dir, stdout=logf, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - start
    if ret != 0:
        raise RuntimeError('MadSpin failed for %s, see %s' % (key, log_path))
    print('    %-10s done in %.0f s -> %s (%.1f MB)'
          % (key, wall, stream, os.path.getsize(stream) / 1e6), flush=True)
    return stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nevents', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--only', default=None)
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    factory = make_factory(args.nevents, args.workdir, args.seed)
    factory.produce_events()
    print('production sample: %s' % factory.events_file, flush=True)

    configs = CONFIGS
    if args.only:
        keep = set(args.only.split(','))
        configs = [c for c in configs if c[0] in keep]
    for key, spinmode, extra in configs:
        print('=== %s (%s) %s' % (key, spinmode, extra or ''), flush=True)
        run_one(factory, key, spinmode, extra, outdir)


if __name__ == '__main__':
    main()
