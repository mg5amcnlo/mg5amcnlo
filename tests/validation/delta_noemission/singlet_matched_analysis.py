#!/usr/bin/env python3
"""Extend the unchanged singlet analysis to all three archived PDF modes.

Arguments are those of singlet_analysis.py. A derived implementation with
explicitly checked replacements is saved in the study directory for audit.
The original driver and its historical provenance remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import sys
import types


NAMES = ['default', 'PDFmode0', 'PDFmode0_root', 'PDFmode2',
         'PDFmode2_root', 'PDFmode1', 'PDFmode1_root']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if len(sys.argv) < 4:
        raise SystemExit('Usage: singlet_matched_analysis.py STUDY PYTHIA OUTPUT [analysis options]')
    template = Path(__file__).with_name('singlet_analysis.py')
    original = template.read_text()
    study, out = Path(sys.argv[1]).resolve(), Path(sys.argv[3]).resolve()
    shower_provenance_path = study/'shower_provenance.json'
    shower_provenance = json.loads(shower_provenance_path.read_text())
    error_samples = sorted(name for name, data in shower_provenance['samples'].items()
        if any(message.startswith('Error') and count for message,count in data['messages'].items()))
    assert all(message=='Error in SimpleSpaceShower::pT2nearThreshold: xPDF = 0'
        for data in shower_provenance['samples'].values()
        for message,count in data['messages'].items() if message.startswith('Error') and count)
    replacements = [
        ("NAMES = ['default', 'PDFmode2', 'PDFmode2_root', 'PDFmode1', 'PDFmode1_root']",
         'NAMES = '+repr(NAMES), 1),
        ('for mode in [1, 2]:', 'for mode in [0, 1, 2]:', 3),
        ("colors = dict(default='#777777',", "colors = dict(default='#777777', PDFmode0='#009E73', PDFmode0_root='#009E73',", 1),
        ("table in ['PDFmode1', 'PDFmode2']", "table in ['PDFmode0', 'PDFmode1', 'PDFmode2']", 1),
        ("if r['table'] in ['default', 'PDFmode1', 'PDFmode1_root']:",
         "if r['table'] in NAMES:", 1),
        ('BORN = EVALUATORS = PDF = None',
         'BORN = EVALUATORS = PDF = None\nTHRESHOLD_ERROR_SAMPLES = '+repr(error_samples), 1),
        ("born=len(ids)))", "born=len(ids), threshold_qualified_born=(int(sum(z['q'][mask, j] <= np.sqrt(2.)*4.8)) if name in THRESHOLD_ERROR_SAMPLES else 0)))", 1),
        ("r['undefined_born']==0 else np.nan", "r['undefined_born']==0 and r['threshold_qualified_born']==0 else np.nan", 1),
        ("'Common cutoff in both calculations; shaded bands: Born-bootstrap statistical errors')",
         "+ ('Common cutoff; Born-bootstrap errors; affected samples omit cuts at/below 6.79 GeV (threshold errors)' if THRESHOLD_ERROR_SAMPLES else 'Common cutoff in both calculations; shaded bands: Born-bootstrap statistical errors'))", 1),
        ('    summarize(arrays, out, args.pdf)',
         "    (out/'analysis_provenance.json').write_text(json.dumps(provenance, indent=2)+'\\n')\n    summarize(arrays, out, args.pdf)", 1),
    ]
    derived = original
    for old, new, count in replacements:
        assert derived.count(old)==count, (old, derived.count(old), count)
        derived = derived.replace(old, new)
    study.mkdir(parents=True, exist_ok=True)
    path = study/'singlet_matched_analysis_impl.py'
    path.write_text(derived)
    # Forked worker functions must be discoverable through their module name.
    module = types.ModuleType('singlet_matched_generated')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(derived, str(path), 'exec'), module.__dict__)
    module.main()
    provenance_path = out/'analysis_provenance.json'
    provenance = json.loads(provenance_path.read_text())
    provenance.update(table_order=NAMES, matched_wrapper_sha256=sha(__file__),
        original_driver_sha256=sha(template), derived_driver_path=str(path),
        derived_driver_sha256=sha(path),
        threshold_error_samples=error_samples,
        shower_provenance_sha256=sha(shower_provenance_path),
        threshold_qualification='Some PDF15 full showers report xPDF=0 in pT2nearThreshold. '
            'Its candidate-scale bound is sqrt(2)*4.8GeV. Rows count Born cutoffs at/below '
            'this bound for affected samples and plot gaps whenever that count is nonzero; '
            'raw numbers are retained only as qualified diagnostics. Error-free samples are not masked.',
        derivation=[dict(original=old, replacement=new, count=count)
                    for old,new,count in replacements],
        pdfmode0_interpretation='The same native consumer and event-dependent PDF factors are retained. '
            'Mode0 already contains fixed-auxiliary-x PDF effects, so numerical agreement does not '
            'establish a consistent PDF-free table contract.')
    provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')


if __name__=='__main__':
    main()
