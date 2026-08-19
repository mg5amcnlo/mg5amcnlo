#!/usr/bin/env python3
"""Render the same synthetic plots with the user's plotter and the MG7 one.

Only here to make the style difference visible; it is not a physics test.  The
real script cannot be run in this repository (it imports ``LHEParser``, which
lives elsewhere, and wants LHE files), so the two plotting functions are taken
straight out of the source files -- no copy of them here -- by exec'ing the
module with a stub ``LHEParser`` standing in for the import.  Nothing from that
stub is ever called: only ``plot_hist_with_ratio`` and
``plot_hist_with_ratio_multi`` run, and they touch numpy and matplotlib only.

    render_style_demo.py <plot_script.py> <out_prefix>

Run it once per script, in separate processes: rcParams are global state.
"""

import os
import sys
import types

import numpy as np


def load(path):
    """exec ``path`` as a module, with LHEParser stubbed out."""
    stub = types.ModuleType('LHEParser')
    stub.EventFile = stub.FourMomentum = object
    sys.modules['LHEParser'] = stub
    ns = {'__name__': 'plotmod', '__file__': path}
    exec(compile(open(path).read(), path, 'exec'), ns)
    return ns


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(prefix) or '.', exist_ok=True)
    mod = load(src)

    rng = np.random.RandomState(20260819)
    n = 2000000
    # a cos(theta)-like shape, and a second sample drawn from the same pdf so
    # the ratio sits at 1 to within the statistics
    a = np.arccos(rng.uniform(-1.0, 1.0, n))
    b = np.arccos(rng.uniform(-1.0, 1.0, n))
    c = np.arccos(rng.uniform(-1.0, 1.0, n))

    mod['plot_hist_with_ratio'](
        a, b, bins=32, xlabel=r'$\theta$',
        outname=prefix + '_hist_with_ratio.png', legend='onshell')

    mod['plot_hist_with_ratio_multi'](
        [{'key': 'ref', 'label': 'density', 'values': a},
         {'key': 's1', 'label': 'onshell', 'values': b},
         {'key': 's2', 'label': 'onshell (v1)', 'values': c}],
        bins=32, xlabel=r'$\theta$',
        outname=prefix + '_hist_with_ratio_multi.png')

    print('wrote %s_hist_with_ratio.png and %s_hist_with_ratio_multi.png'
          % (prefix, prefix))


if __name__ == '__main__':
    main()
