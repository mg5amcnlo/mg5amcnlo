#!/usr/bin/env python3
"""``use_old_dir`` leg of the ms_dir sweep.

``use_old_dir`` is a *different* reuse mechanism from ``ms_dir``: it is read
only in MadSpin/decay.py (the madspin_v1 decay-chain path), where it (a) keeps
production_me/full_me/decay_me instead of deleting them, (b) reloads
production_me/all_ME.pkl instead of regenerating the matrix elements and (c)
skips the ``make clean``.  There is no gridpack and no madspin.pkl, and
run_onshell/run_bridge never look at the option at all.

So the reuse it means is "run twice in the *same* directory", not "point at a
directory somebody else built" -- both runs share one rundir here.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import sweep                                            # noqa: E402


def main():
    sweep.PROD_XSEC = sweep.read_init(sweep.EVENTS)[0]
    print('production <init> xsec = %s' % sweep.PROD_XSEC, flush=True)
    modes = sys.argv[1:] or ['madspin_v1', 'madspin']
    for mode in modes:
        rundir = os.path.join(sweep.WORK, 'runs', '%s__uod' % mode)
        if os.path.exists(rundir):
            shutil.rmtree(rundir)
        os.makedirs(rundir)
        sweep.run_one('%s__use_old_dir__fresh' % mode, mode, 'import_first', '',
                      'fresh', use_old_dir=True, rundir=rundir)
        sweep.run_one('%s__use_old_dir__reuse' % mode, mode, 'import_first', '',
                      'reuse', use_old_dir=True, rundir=rundir)


if __name__ == '__main__':
    main()
