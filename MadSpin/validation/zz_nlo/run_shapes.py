#!/usr/bin/env python3
"""Harvest the shape observables from the samples this study already generated.

**Generates nothing.**  Every event file it reads was written by
``run_zz_nlo.py``; the paths come out of ``data/meta.json``, which records where
each one lives, and a missing file is a hard error rather than something to work
around -- a shape comparison with one sample silently absent is worse than no
comparison.

It writes

    data/histograms_shapes.npz    the raw histograms, production and four-lepton
    data/meta.json                updated in place, additively

and nothing else.  ``meta.json`` is updated rather than replaced because it is
``run_zz_nlo.py --stage harvest`` that owns it: re-running that stage will drop
the keys this script adds, so the order is *harvest first, then this*.  The keys
added are all prefixed ``shapes``.

Usage::

    python3 run_shapes.py [--data DIR] [--basedir DIR]

``--basedir`` re-roots the LHE paths recorded in ``meta.json``, for the case
where the samples were moved after they were made.  Without it the recorded
absolute paths are used as they stand.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import observables_zz as OZ                                      # noqa: E402
import observables_shapes as OS                                  # noqa: E402

PROD = ('lo', 'nlo', 'li')
MODES = ('madspin', 'PA', 'onshell', 'none', 'madspin_v1', 'onshell_v1')


def reroot(path, basedir):
    """Put ``path`` under ``basedir``, keeping everything below the old root.

    The recorded paths look like ``/tmp/zz_nlo_work2/ppzz_lo/Events/...``; the
    piece to keep is everything from the sample directory down.
    """
    if not basedir:
        return path
    parts = path.replace('\\', '/').split('/')
    # the sample directory is the one whose parent is the old basedir: find the
    # last component that also exists under the new basedir
    for i in range(len(parts)):
        cand = os.path.join(basedir, *parts[i:])
        if os.path.exists(cand):
            return cand
    return path


def ebeam_of(meta):
    """The beam energy the samples were made with, out of the audited card."""
    for src in (meta.get('production', {}).get('lo', {}),
                meta.get('production', {}).get('nlo', {})):
        card = src.get('card') or {}
        if 'ebeam1' in card:
            return float(card['ebeam1'])
    return OS.EBEAM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--basedir', default=None,
                    help='re-root the LHE paths recorded in meta.json')
    args = ap.parse_args()

    mpath = os.path.join(args.data, 'meta.json')
    meta = json.load(open(mpath))
    ebeam = ebeam_of(meta)

    hist = {}
    meta['shapes_bins'] = {k: v.tolist() for k, v in OS.BINS_SHAPES.items()}
    meta['shapes_bins_4l'] = {k: v.tolist()
                              for k, v in OS.BINS_SHAPES_4L.items()}
    meta['shapes_observables'] = list(OS.SHAPE_OBS)
    meta['shapes_observables_4l'] = list(OS.SHAPE_OBS_4L)
    meta['shapes_ebeam'] = ebeam
    meta['shapes_m_top'] = OS.M_TOP
    meta['shapes_m_slices'] = list(OS.M_SLICES)
    meta['shapes_reweight_edges'] = OS.REWEIGHT_EDGES.tolist()
    meta['shapes_samples'] = {}

    # ---------------- production side ----------------
    obs, wgt, mzz = {}, {}, {}
    for tag in PROD:
        path = reroot(meta['production'][tag]['lhe'], args.basedir)
        if not os.path.exists(path):
            raise SystemExit('missing sample %s: %s -- this script generates '
                             'nothing, so the file has to be there' % (tag, path))
        w, z1, z2, _ = OZ.read_lhe_zz(path)
        obs[tag] = OS.compute_shapes(z1, z2, ebeam)
        wgt[tag] = w
        mzz[tag] = OZ.mass(z1 + z2)
        meta['shapes_samples'][tag] = {'lhe': path, 'nevents': int(len(w))}
        print('%-5s %6d events from %s' % (tag, len(w), path))

    # the m(ZZ)-reweighted twins: LO and LI forced onto NLO's m(ZZ) spectrum,
    # so what survives is matrix element rather than parton luminosity
    rw = {'nlo': wgt['nlo']}
    for tag in ('lo', 'li'):
        rw[tag] = OS.mzz_reweight(mzz[tag], wgt[tag], mzz['nlo'], wgt['nlo'],
                                  OS.REWEIGHT_EDGES)

    for tag in PROD:
        for name, edges in OS.BINS_SHAPES.items():
            y, e = OS.histogram(obs[tag][name], wgt[tag], edges)
            hist['%s/%s/y' % (tag, name)] = y
            hist['%s/%s/e' % (tag, name)] = e
            y, e = OS.histogram(obs[tag][name], rw[tag], edges)
            hist['%s_rw/%s/y' % (tag, name)] = y
            hist['%s_rw/%s/e' % (tag, name)] = e

    # The measured statement that the Collins-Soper angle and the parent
    # study's cos theta* are the SAME observable on a 2 -> 2 sample.  Recorded
    # rather than asserted, because the whole reason the CS variable is in this
    # module is to report that it buys nothing there.
    meta['shapes_cs_vs_parent'] = {}
    for tag in PROD:
        path = reroot(meta['production'][tag]['lhe'], args.basedir)
        w, z1, z2, _ = OZ.read_lhe_zz(path)
        parent = OZ.compute_zz(z1, z2)['abs_cos_theta_star']
        d = np.abs(obs[tag]['abs_cos_theta_cs'] - parent)
        meta['shapes_cs_vs_parent'][tag] = {
            'max_abs_difference': float(d.max()),
            'mean_abs_difference': float(d.mean()),
            'pt_zz_max': float(OZ.pt(z1 + z2).max())}

    # How much NLO weight sits above the 2 -> 2 boundary of pt/m
    v = obs['nlo']['pt_over_m']
    meta['shapes_pt_over_m_above_half'] = {
        'nlo_weight_fraction': float(wgt['nlo'][v > 0.5].sum()
                                     / wgt['nlo'].sum()),
        'lo_max': float(obs['lo']['pt_over_m'].max()),
        'li_max': float(obs['li']['pt_over_m'].max()),
        'nlo_max': float(v.max())}

    # ---------------- four-lepton side ----------------
    LEP = OZ.leptonic()
    decayed = {}
    tr = meta.get('runs', {}).get('truth', {}).get('lhe')
    if tr:
        decayed['truth'] = reroot(tr, args.basedir)
    for mode in MODES:
        p = meta.get('runs', {}).get(mode, {}).get('lhe')
        if p:
            decayed[mode] = reroot(p, args.basedir)
    for key, path in decayed.items():
        if not os.path.exists(path):
            raise SystemExit('missing decayed sample %s: %s' % (key, path))
        w, p4 = LEP.read_lhe(path)
        o = OS.compute_shapes_4l(p4, ebeam)
        meta['shapes_samples'][key] = {'lhe': path, 'nevents': int(len(w))}
        for name, edges in OS.BINS_SHAPES_4L.items():
            y, e = OS.histogram(o[name], w, edges)
            hist['%s/%s/y' % (key, name)] = y
            hist['%s/%s/e' % (key, name)] = e
        print('%-12s %6d events from %s' % (key, len(w), path))

    os.makedirs(args.data, exist_ok=True)
    np.savez_compressed(os.path.join(args.data, 'histograms_shapes.npz'),
                        **hist)
    with open(mpath, 'w') as fp:
        json.dump(meta, fp, indent=1, sort_keys=True)
    print('wrote %s' % os.path.join(args.data, 'histograms_shapes.npz'))
    print('updated %s' % mpath)


if __name__ == '__main__':
    main()
