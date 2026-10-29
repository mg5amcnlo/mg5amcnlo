#!/usr/bin/env python3
"""Independently reconstruct DY factors from raw tables and the shower PDF.

Usage: singlet_dy_algebra_audit.py STUDY_DIRECTORY PYTHIA_INSTALL OUTPUT_DIRECTORY
Run singlet_analysis.py first. This reads its saved probabilities but does not
call delta_leg_probability or DeltaEvaluator. No new showers are generated.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np


TABLE_ORDER = ['default', 'PDFmode2', 'PDFmode2_root', 'PDFmode1', 'PDFmode1_root']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('study', type=Path)
    parser.add_argument('pythia', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    study, install, out = args.study.resolve(), args.pythia.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bridge = study/'analysis/pythia_pdf.so'
    pdf = C.CDLL(str(bridge))
    pdf.pythia_pdf_init.argtypes = [C.c_char_p, C.c_char_p]
    pdf.pythia_pdf_xf.argtypes = [C.c_int, C.c_double, C.c_double]
    pdf.pythia_pdf_xf.restype = C.c_double
    assert pdf.pythia_pdf_init(str(install/'share/Pythia8/xmldoc').encode(), b'15') == 0
    # The table checks the QCD grid masses; non-QCD slots are unused here.
    masses = (C.c_double*21)(.33, .33, .5, 1.5, 4.8, *([0.]*16))
    functions, libraries = {}, {}
    input_files = [bridge, install/'share/Pythia8/pdfdata/NNPDF23_nlo_as_0119_qed_0000.dat']
    for name in ['default', 'PDFmode1', 'PDFmode2']:
        path = study/'analysis'/name/'delta.so'
        libraries[name] = C.CDLL(str(path))
        f = libraries[name].pysudakov_
        f.argtypes = [C.POINTER(C.c_double), C.POINTER(C.c_double),
                      C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_double)]
        f.restype = C.c_double
        functions[name] = f
        input_files += [path, study/'analysis'/name/'sudakov.f']

    def table(name, q, mass, pdg):
        a, b, i, t = C.c_double(q), C.c_double(mass), C.c_int(pdg), C.c_int(1)
        return functions[name](C.byref(a), C.byref(b), C.byref(i), C.byref(t), masses)

    rows = []
    for sample in ['dy_fixed', 'dy_damped']:
        born_file = study/(sample+'.born.jsonl')
        values_file = study/'analysis'/(sample+'.npz')
        input_files += [born_file, values_file]
        born = [json.loads(line) for line in born_file.open()]
        with np.load(values_file) as z:
            cuts, values = z['q'], z['values']
        assert len(born) == len(values)
        errors = {name: 0. for name in functions}
        active = endpoints = 0
        maximum_mass_relative_error = 0.
        for ib, b in enumerate(born):
            assert b['ids'][0] == -b['ids'][1] and 1 <= abs(b['ids'][0]) <= 5
            assert b['ids'][2:] == [11, -11]
            # Independent massless two-beam invariant, avoiding boosted-vector
            # subtraction in the native evaluator's mass construction.
            mass = (4*b['p'][0][0]*b['p'][1][0])**.5
            if sample.endswith('fixed'):
                maximum_mass_relative_error = max(maximum_mass_relative_error,
                    abs(mass-b['scalup'])/mass)
            for jq, q in enumerate(cuts[ib]):
                if q >= b['scalup']:
                    assert (values[ib, jq, :, 0] == 1).all()
                    endpoints += 1
                    continue
                assert q >= 1.01, 'Raw table identity excludes native low-scale continuation'
                factors = [pdf.pythia_pdf_xf(pdg, x, q)/
                           pdf.pythia_pdf_xf(pdg, x, b['scalup'])
                           for pdg, x in zip(b['ids'][:2], b['x'])]
                for name in functions:
                    ratios = [table(name, q, mass, pdg)/table(name, b['scalup'], mass, pdg)
                              for pdg in b['ids'][:2]]
                    direct = np.prod(np.clip(np.array(factors)*ratios, 0, 1))
                    actual = values[ib, jq, TABLE_ORDER.index(name), 0]
                    assert np.isfinite(direct) and abs(direct-actual) < 1.e-10
                    errors[name] = max(errors[name], abs(direct-actual))
                active += 1
        row = dict(sample=sample, born=len(born), active_born_cut_points=active,
            endpoint_born_cut_points=endpoints, maximum_absolute_probability_error=errors,
            maximum_fixed_start_vs_beam_invariant_relative_error=maximum_mass_relative_error)
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(rows=rows,
        definition='Direct product clip(Fa*Ra)*clip(Fb*Rb), using raw Fortran table calls, '
                   'Pythia PDF15 and definite incoming flavours. No native Delta caller is invoked.',
        mass_definition='M=sqrt(4*Ebeam_parton_A*Ebeam_parton_B) for massless collinear Born incoming partons',
        table_order_in_saved_arrays=TABLE_ORDER,
        checks=['Every active tested cutoff is at least1.01GeV',
                'Independent table/PDF product agrees with every saved active DY probability within1e-10',
                'Every saved probability equals1 for cutoffs at or above its starting scale'],
        limitation='This checks table arguments and caller algebra, not the physical fidelity of the archived tables or the reference-shower measurement.',
        script_sha256=sha(__file__),
        input_sha256={str(p): sha(p) for p in input_files})
    (out/'dy_algebra_audit.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
