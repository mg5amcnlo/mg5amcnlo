#!/usr/bin/env python3
"""Compare native nn23nlo and Pythia PDF15 inputs on the existing ttbar Born sample.

Usage: pdf15_input_audit.py STUDY PYTHIA OUTPUT
Requires the isolated rooted-gluon table from mode1_gluon_real_validation.
"""
import argparse
import ctypes as C
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from analyse import grouped
from delta_evaluator import DeltaEvaluator
from delta_native_pdf import NativePDF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('study', type=Path)
    parser.add_argument('pythia', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    study, pythia, out = args.study.resolve(), args.pythia.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = study / 'pdf15_input_audit'
    work.mkdir(parents=True, exist_ok=True)
    bridge = study / 'libpythia_pdf.so'
    lib = C.CDLL(str(bridge))
    lib.pythia_pdf_init.argtypes = [C.c_char_p, C.c_char_p]
    lib.pythia_pdf_xf.argtypes = [C.c_int, C.c_double, C.c_double]
    lib.pythia_pdf_xf.restype = C.c_double
    assert lib.pythia_pdf_init(str(pythia / 'share/Pythia8/xmldoc').encode(), b'15') == 0
    @lru_cache(maxsize=20000)
    def pdf15(id, x, q):
        return lib.pythia_pdf_xf(id, x, q)
    born_file = study / 'branch_exact.born.jsonl'
    born = [grouped(json.loads(line)) for line in born_file.open()]
    table = study / 'mode1_gluon_real_validation/mode1_incoming_gluon_root.f'
    qs = [10., 20., 50., 100.]
    variants = ['native_grouped', 'pdf15_grouped', 'native_definite', 'pdf15_definite']
    values = np.zeros((len(born), len(qs), len(variants)))
    channels = np.array([b['ids'][0] == 21 for b in born])
    with NativePDF(study / 'TT', work / 'native') as native:
        with DeltaEvaluator(work=work / 'delta', table=table) as evaluator:
            for i, b in enumerate(born):
                definite = dict(b, flavours=[[id] for id in b['ids']])
                for j, q in enumerate(qs):
                    for k, (state, pdf) in enumerate([(b, native), (b, pdf15),
                            (definite, native), (definite, pdf15)]):
                        values[i, j, k] = evaluator.evaluate(state, q, pdf)['delta']
            native_provenance = native.provenance
            delta_provenance = evaluator.provenance
    assert np.isfinite(values).all()
    assert np.array_equal(values[channels, :, 0], values[channels, :, 2])
    assert np.array_equal(values[channels, :, 1], values[channels, :, 3])
    reference = np.load(study / 'mode1_study/matched_pdf15.mode1.npz')
    qi = [np.flatnonzero(reference['q'] == q)[0] for q in qs]
    ref = reference['refcdf'][:, qi]
    previous = np.load(study / 'mode1_gluon_real_validation/matched_pdf15.npz')
    assert np.array_equal(values[:, :, 0], previous['common'][:, qi, 0])
    assert np.array_equal(ref, previous['refcdf'][:, qi])
    rows = []
    for channel, mask in [('all', np.ones(len(born), bool)), ('gg', channels), ('qqbar', ~channels)]:
        for j, q in enumerate(qs):
            means = values[mask, j].mean(axis=0)
            shower = float(ref[mask, j].mean())
            rows.append(dict(channel=channel, q=q, n=int(mask.sum()),
                delta=dict(zip(variants, means.tolist())), shower=shower,
                delta_over_shower=dict(zip(variants, (means/shower).tolist())),
                pdf15_over_native_grouped=float(means[1]/means[0]),
                definite_over_grouped_native=float(means[2]/means[0]),
                definite_over_grouped_pdf15=float(means[3]/means[1])))
    native_grid = study / 'TT/lib/Pdfdata/NNPDF23nlo_as_0119_qed_mem0.grid'
    pythia_grid = pythia / 'share/Pythia8/pdfdata/NNPDF23_nlo_as_0119_qed_0000.dat'
    result = dict(rows=rows,
        scope='Same corrected mode1 table, common cutoff, Born states, scales, caller and reference shower; change only PDF provider and/or grouped versus definite incoming flavour. No new showers.',
        validation=['Native grouped Delta reproduces previous corrected-table common-cut calculation bitwise',
                    'Reference CDF reproduces previous corrected-table study bitwise',
                    'Gluon channel is bitwise unchanged by grouped/definite flavour switch',
                    'All probability outputs are finite'],
        native_grid_header=native_grid.read_text().splitlines()[:11],
        pythia_grid=str(pythia_grid), native_provenance=native_provenance,
        delta_provenance=delta_provenance,
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [native_grid, pythia_grid, bridge, born_file, Path(__file__)]})
    (out / 'pdf15_input_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(work / 'common.npz', values=values, q=qs, variants=variants,
                        channels=channels, reference=ref)
    for row in rows:
        print(row['channel'], row['q'], row['delta_over_shower'],
              'PDF15/native', row['pdf15_over_native_grouped'],
              'native definite/grouped', row['definite_over_grouped_native'])


if __name__ == '__main__':
    main()
