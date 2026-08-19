#!/usr/bin/env python3
"""Cross-spinmode ``ms_dir`` reuse.

Not part of the (spinmode x ordering x fresh/reuse) matrix, but the same class
of silent failure: several of the things an ``ms_dir`` caches are *not* keyed by
spinmode -- ``max_wgt`` (the joint unweighting bound, written by
get_maxwgt_for_onshell for madspin/PA/onshell alike) and the ``decay_<pdg>_<i>``
gridpacks.  So a user who builds an ms_dir with one spinmode and reuses it with
another gets the first mode's cached bound.  This leg builds an ms_dir with one
mode and reuses it with each of the others.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import sweep                                            # noqa: E402


def main():
    sweep.PROD_XSEC = sweep.read_init(sweep.EVENTS)[0]
    builder = sys.argv[1] if len(sys.argv) > 1 else 'onshell'
    reusers = sys.argv[2:] or ['madspin', 'PA', 'madspin_v1', 'none']
    ms_dir = os.path.join(sweep.WORK, 'msdirs', 'xmode__%s' % builder)
    if os.path.exists(ms_dir):
        shutil.rmtree(ms_dir)
    sweep.run_one('xmode__built_by_%s' % builder, builder, 'ms_dir_first',
                  ms_dir, 'fresh')
    for mode in reusers:
        sweep.run_one('xmode__%s_reusing_%s' % (mode, builder), mode,
                      'ms_dir_first', ms_dir, 'reuse')


if __name__ == '__main__':
    main()
