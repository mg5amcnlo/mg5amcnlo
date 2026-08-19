#!/usr/bin/env python3
"""How much does spinmode=none's <init> move from one ms_dir *reuse* to the
next?

The matrix showed none's fresh and reuse cross-sections differing (5.94052 vs
5.97103 pb) where every other spinmode reproduces its fresh number to the last
digit.  The reason is that run_bridge builds the branching ratio from
``event_files[k].cross`` -- the <init> of the decay events *this* run's run.sh
just produced (interface_madspin.py:2010/2021/2027/2032) -- and never reads the
partial width the building run stored, which is what run_onshell now does via
``generate_events(..., output_width=True)`` -> ``_load_partial_width``.

So the question is whether the gap is Monte-Carlo scatter of that
re-measurement or a systematic shift.  Re-run the *same* ms_dir several times
with different seeds and look at the spread.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import sweep                                            # noqa: E402


def main():
    sweep.PROD_XSEC = sweep.read_init(sweep.EVENTS)[0]
    ms_dir = os.path.join(sweep.WORK, 'msdirs', 'none__ms_dir_first')
    assert os.path.exists(ms_dir), 'run the main sweep first'
    orig = sweep.make_card

    for seed in (33, 77, 101, 4242):
        def patched(path, evtfile, spinmode, order, msd, uod, _s=seed):
            return orig(path, evtfile, spinmode, order, msd, uod, seed=_s)
        sweep.make_card = patched
        sweep.run_one('none__reuse_seed%d' % seed, 'none', 'ms_dir_first',
                      ms_dir, 'reuse')
    sweep.make_card = orig


if __name__ == '__main__':
    main()
