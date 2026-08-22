#!/usr/bin/env python3
"""The six spinmodes on the NEW observables, in the user's personal style.

A second, independent rendering of what ``plot_modes_shapes.py`` draws in the
MG7 paper style.  The data, the binning, the reference logic, the structural
zeros and the numbers are shared with that module and with
``plot_zz_nlo_userstyle.py``; only this file's ``main`` differs, because the
drawing itself is the parent's and is called with the shape study's labels.

Usage::

    python3 plot_modes_shapes_userstyle.py [--data DIR] [--out DIR]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import observables_shapes as OS                                  # noqa: E402
from plot_modes_shapes import load, structural_shapes            # noqa: E402
from plot_zz_nlo_userstyle import draw as draw_modes             # noqa: E402

# plot_modes_shapes imports the MG7-style modules, which set the paper rcParams
# (serif / usetex) at import time; plot_zz_nlo_userstyle resets them on import.
# Assert rather than assume, because the import order here is the reverse of the
# parent script's and an rcParams leak would silently expose these PDFs to the
# Type1 minus-subsetting bug they are not checked for.
assert not mpl.rcParams['text.usetex'], (
    'the user style renders without usetex; if that ever changes these figures '
    'become exposed to the Type1 minus-subsetting bug and need their own check')

TITLE = ('$pp \\to ZZ$ [QCD] (MC@NLO) $\\to e^{+}e^{-}\\mu^{+}\\mu^{-}$, '
         '13 TeV — shape-study observables')


def draw(d, obs, outdir):
    return draw_modes(d, obs, outdir, labels=OS.LABELS_SHAPES_4L,
                      logy=OS.LOGY_SHAPES_4L,
                      structural_fn=structural_shapes, title=TITLE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots_userstyle'))
    args = ap.parse_args()
    d = load(args.data)
    for obs in OS.SHAPE_OBS_4L:
        draw(d, obs, args.out)


if __name__ == '__main__':
    main()
