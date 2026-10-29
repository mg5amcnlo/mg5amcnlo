#!/usr/bin/env python3
"""Diagnostic incoming-gluon square root, leaving the production table untouched.

Usage: mode1_gluon_normalization.py STUDY_DIRECTORY OUTPUT_DIRECTORY
The isolated table copy changes only the return value for gluons of typeII/IF.
This tests a normalization hypothesis and is not a proposed production fix.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from analyse import grouped
from delta_evaluator import DeltaEvaluator, ROOT
from delta_native_pdf import NativePDF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('study', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    study, out = args.study.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = study / 'mode1_study/gluon_normalization'
    work.mkdir(parents=True, exist_ok=True)
    original = ROOT / 'vendor/SudGen/sudakov_r515_PDFmode1.f.gz'
    original_text = gzip.decompress(original.read_bytes()).decode()
    target = '      pysudakov=tmp\n'
    assert original_text.count(target) == 1
    replacement = target + '''      if(abs(id).eq.21.and.(itype.eq.1.or.itype.eq.3))then
        pysudakov=sqrt(max(tmp,0d0))
      endif
'''
    table = work / 'mode1_incoming_gluon_root.f'
    table.write_text(original_text.replace(target, replacement))
    baseline_file = study / 'archive_validation/PDFmode1.npz'
    with np.load(baseline_file) as loaded:
        baseline = {k: loaded[k] for k in loaded.files}
    born_file = study / 'branch_exact.born.jsonl'
    born = [grouped(json.loads(line)) for line in born_file.open()]
    qs = [20., 50., 100.]
    indices = [int(np.flatnonzero(baseline['q'] == q)[0]) for q in qs]
    results = np.zeros((len(born), len(qs), 3))
    with NativePDF(study / 'TT', work / 'pdf') as pdf:
        with DeltaEvaluator(work=work / 'build', table=table) as evaluator:
            for i, b in enumerate(born):
                for j, q in enumerate(qs):
                    result = evaluator.evaluate(b, q, pdf)
                    results[i, j] = [result['delta'], result['isr'], result['fsr']]
            table_provenance = evaluator.provenance
            pdf_provenance = pdf.provenance
    raw = baseline['common'][:, indices, :]
    # The isolated change must leave final-state factors and qqbar intact.
    assert np.array_equal(results[:, :, 2], raw[:, :, 2])
    assert np.array_equal(results[~baseline['channels']], raw[~baseline['channels']])
    ref = baseline['refcdf'][:, indices]
    np.savez_compressed(work / 'common_cut.npz', raw=raw, diagnostic=results,
                        reference=ref, q=qs, channels=baseline['channels'])
    rows = []
    rng = np.random.default_rng(12732)
    for channel, mask in [('all', np.ones(len(born), bool)),
                          ('gg', baseline['channels']), ('qqbar', ~baseline['channels'])]:
        count = int(mask.sum())
        boots = rng.integers(0, count, size=(500, count))
        for j, q in enumerate(qs):
            d, r, p = results[mask, j, 0], raw[mask, j, 0], ref[mask, j]
            rows.append(dict(channel=channel, q_GeV=q, born=count,
                raw_mode1=float(r.mean()), diagnostic=float(d.mean()), shower=float(p.mean()),
                raw_ratio=float(r.mean()/p.mean()), diagnostic_ratio=float(d.mean()/p.mean()),
                diagnostic_ratio_se=float((d[boots].mean(axis=1)/p[boots].mean(axis=1)).std(ddof=1)),
                raw_isr=float(raw[mask, j, 1].mean()),
                diagnostic_isr=float(results[mask, j, 1].mean()),
                unchanged_fsr=float(results[mask, j, 2].mean())))
    provenance = dict(rows=rows, modification=replacement,
        definition='Common stopping cut only; sqrt of incoming-gluon typeII/IF table return in an isolated copy; all other entries and caller unchanged.',
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [original, baseline_file, born_file, Path(__file__)]},
        modified_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
        table_provenance=table_provenance, pdf_provenance=pdf_provenance,
        validation='FSR components and all qqbar results are bitwise unchanged.',
        caution='This is a hypothesis test, not a universal root prescription: a full-radiator probability need not factor into identical colour-line probabilities for arbitrary geometry. The source API contract must be fixed explicitly before production use.')
    (out / 'gluon_normalization.json').write_text(json.dumps(provenance, indent=2) + '\n')
    lines = ['Isolated incoming-gluon normalization diagnostic', '',
             'Only incoming gluon table values (II/IF) are replaced by sqrt(max(T,0)).',
             'The production table and caller are untouched; no FF/FSR transformation.',
             'Reference: existing Pythia8.313 Simple, alphaS=.118, PDF13, stable ttbar.',
             'This is a common-cutoff test, not the actual-first-emission comparison.', '',
             'channel q[GeV] rawMode1 diagnostic SimpleCDF rawRatio diagnosticRatio']
    for row in rows:
        lines.append(f'{row["channel"]:7} {row["q_GeV"]:6.0f} '
                     f'{row["raw_mode1"]:.6f} {row["diagnostic"]:.6f} '
                     f'{row["shower"]:.6f} {row["raw_ratio"]:.6f} '
                     f'{row["diagnostic_ratio"]:.6f}+/-{row["diagnostic_ratio_se"]:.6f}')
    lines += ['', provenance['validation'], '', provenance['caution'], '',
              'Reproduce: python tests/validation/delta_noemission/mode1_gluon_normalization.py STUDY_DIRECTORY OUTPUT_DIRECTORY']
    (out / 'gluon_normalization.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
