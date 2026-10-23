"""Strict report gating and labelled vector-book/archive integration checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('macnlops_packaging',
    ROOT / 'docs/audits/package_macnlops_plots.py')
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


def summary():
    # Synthetic metadata is only used in temporary TEST_ONLY reports.
    data = dict(schema=PACKAGE.SCHEMA, complete=True, smoke=False,
                levels=list(PACKAGE.LEVELS), processes={},
                notes=['TEST ONLY: synthetic completed-schema fixture; no physics events'])
    for process in PACKAGE.PROCESSES:
        stages = {stage:dict(complete=True,
            totals={role:dict(events=200000) for role in ('input', 'reference')},
            chunks={role:[dict(events=200000)] for role in ('primary', 'reference')})
            for stage in PACKAGE.LEVELS}
        data['processes'][process] = dict(configuration=dict(expected_events=200000,
            flavours=4 if process == 'wbb' else 5, primary_complete=True,
            reference_complete=True), levels=stages)
    return data


def common_summary():
    data=summary();data['common_source_mode']=True
    for process,record in data['processes'].items():
        config=record['configuration'];config.pop('reference_complete')
        config.update(common_source_mode=True,source_construction='ordinary-mcatnlo-v1',signed_residuals=True)
        for level,stage in record['levels'].items():
            stage.update(common_source_mode=True,signed_residuals=True)
            total=dict(events=200000,negative_events=30000,sumw=100.,sumw2=1000.,effective_events=10.)
            stage['totals']=dict(input=dict(total),reference=dict(total),
                H_negative_retained=dict(events=17,negative_events=17,sumw=-.2,sumw2=.003))
            primary=[dict(events=100000,path='/TEST_ONLY/'+process+'/'+level+'/chunk%d.lhe'%i,
                weight_sequence_sha256=str(i)*64,source_sequence_sha256=str(i+2)*64,
                full_audit=None if level=='first' else dict(path='/TEST_ONLY/audit%d'%i,sha256='a'*64))
                for i in range(2)]
            stage['chunks']=dict(primary=primary,reference=[dict(row,derived_from_primary=True) for row in primary])
    return data


def labelled_pdf(label):
    """Make a single-page text/vector PDF without plotting dependencies."""
    stream = ('BT /F1 12 Tf 30 780 Td (' + label + ') Tj ET\n').encode('ascii')
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
        b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] '
        b'/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
        ('<< /Length %d >>\nstream\n' % len(stream)).encode('ascii') + stream + b'endstream']
    pdf, offsets = bytearray(b'%PDF-1.4\n'), [0]
    for number, content in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(('%d 0 obj\n' % number).encode('ascii') + content + b'\nendobj\n')
    start = len(pdf)
    pdf.extend(b'xref\n0 6\n0000000000 65535 f \n')
    for offset in offsets[1:]:
        pdf.extend(('%010d 00000 n \n' % offset).encode('ascii'))
    pdf.extend(('trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n' % start).encode('ascii'))
    return bytes(pdf)


def fixture(report):
    report.mkdir()
    (report / 'summary.json').write_text(json.dumps(summary()))
    (report / 'index.html').write_text('<!doctype html><h1>TEST ONLY report</h1>\n<p>Original figures</p>\n')
    (report / 'stage_totals.csv').write_text('TEST_ONLY,synthetic\n')
    labels = []
    for process in PACKAGE.PROCESSES:
        for level in PACKAGE.LEVELS:
            directory = report / process / level
            directory.mkdir(parents=True)
            with zipfile.ZipFile(directory / 'histograms.npz', 'w') as archive:
                archive.writestr('TEST_ONLY.txt', 'No physics data in this unit fixture')
            for name in ('jet_pt', 'jet_pt_soft', 'system_pt', 'delta_phi', 'delta_phi_deficit', 'delta_y', 'system_mass'):
                (directory / (name + '.csv')).write_text('TEST_ONLY,synthetic\n')
            for panel in PACKAGE.PANELS:
                label = 'TEST ONLY ' + process + ' ' + level + ' ' + panel
                labels.append(label)
                (directory / (panel + '.pdf')).write_bytes(labelled_pdf(label))
                # The packager preserves PNG bytes; it does not redraw plots.
                (directory / (panel + '.png')).write_bytes(b'TEST ONLY PNG placeholder')
    return labels


class TestMacNLOPSPackagingGates(unittest.TestCase):
    def test_shared_source_omits_reference_cohort_and_records_real_source_count(self):
        data=common_summary();counts=PACKAGE.validate_summary(data)
        self.assertEqual(counts['wbb']['full']['physical_source_events'],200000)
        self.assertEqual(counts['wbb']['full']['negative_H_retained'],17)
        contracts=PACKAGE.comparison_contracts(data)
        self.assertEqual(sum(c['physical_source_events'] for c in contracts.values()),800000)
        self.assertTrue(all(c['signed_residuals'] for c in contracts.values()))
        self.assertTrue(all(c['baseline']=='same primary events and shower' for c in contracts.values()))

    def test_shared_source_rejects_false_baseline_identity_or_residual_policy(self):
        cases=[]
        data=common_summary();data['processes']['dy']['configuration']['reference']=[dict(first='/independent/file')];cases.append(data)
        data=common_summary();data['processes']['dy']['levels']['first']['chunks']['reference'][0]['path']+='.different';cases.append(data)
        data=common_summary();data['processes']['dy']['levels']['first']['chunks']['reference'][0]['derived_from_primary']=False;cases.append(data)
        data=common_summary();data['processes']['dy']['levels']['first']['totals']['reference']['sumw']=101.;cases.append(data)
        data=common_summary();stage=data['processes']['dy']['levels']['full'];stage['chunks']['primary'][0]['source_sequence_sha256']='f'*64;stage['chunks']['reference'][0]['source_sequence_sha256']='f'*64;cases.append(data)
        data=common_summary();record=data['processes']['dy'];record['configuration']['signed_residuals']=False
        for stage in record['levels'].values():stage['signed_residuals']=False
        cases.append(data)
        for index,data in enumerate(cases):
            with self.subTest(case=index),self.assertRaises(PACKAGE.PackagingError):PACKAGE.validate_summary(data)

    def test_rejects_incomplete_or_smoke_without_creating_artifacts(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY-package-gate-') as temporary:
            report = Path(temporary) / 'preview'
            report.mkdir()
            for complete, smoke in ((False, True), (True, True), (False, False), (1, 0)):
                data = summary()
                data.update(complete=complete, smoke=smoke)
                source = json.dumps(data)
                (report / 'summary.json').write_text(source)
                with self.assertRaises(PACKAGE.PackagingError):
                    PACKAGE.package_report(report)
                self.assertEqual((report / 'summary.json').read_text(), source)
                self.assertEqual(sorted(p.name for p in report.iterdir()), ['summary.json'])
                self.assertFalse(report.with_suffix('.zip').exists())

    def test_rejects_wrong_process_stage_counts_and_chunks(self):
        cases = []
        data = summary();del data['processes']['wbb'];cases.append(data)
        data = summary();del data['processes']['zz']['levels']['full'];cases.append(data)
        data = summary();data['levels'] = ['first'];cases.append(data)
        for process in PACKAGE.PROCESSES:
            for level in PACKAGE.LEVELS:
                for role in ('input', 'reference'):
                    data = summary()
                    data['processes'][process]['levels'][level]['totals'][role]['events'] = 199999
                    cases.append(data)
        data = summary();data['processes']['dy']['levels']['full']['chunks']['reference'][0]['events'] = 199999;cases.append(data)
        data = summary();data['processes']['ttbar']['configuration']['primary_complete'] = False;cases.append(data)
        data = summary();data['processes']['wbb']['configuration']['flavours'] = 5;cases.append(data)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(PACKAGE.PackagingError):
                PACKAGE.validate_summary(data)

    def test_archive_cannot_be_inside_report(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY-package-path-') as temporary:
            report = Path(temporary)
            with self.assertRaisesRegex(PACKAGE.PackagingError, 'outside'):
                PACKAGE.package_report(report, report / 'recursive.zip')

    def test_missing_panel_rejects_before_report_mutation(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY-package-missing-') as temporary:
            report = Path(temporary) / 'TEST_ONLY_report'
            fixture(report)
            (report / 'wbb/full/jet_pt_source_signs.pdf').unlink()
            index = (report / 'index.html').read_bytes()
            with self.assertRaisesRegex(PACKAGE.PackagingError, 'Missing or empty'):
                PACKAGE.package_report(report)
            self.assertEqual((report / 'index.html').read_bytes(), index)
            self.assertFalse((report / PACKAGE.BOOK).exists())
            self.assertFalse((report / PACKAGE.MANIFEST).exists())
            self.assertFalse(report.with_suffix('.zip').exists())

    def test_index_links_are_idempotent(self):
        original = '<h1>TEST ONLY</h1><p>Keep this text</p>'
        first = PACKAGE.linked_index(original, 72)
        self.assertEqual(PACKAGE.linked_index(first, 72), first)
        self.assertEqual(first.count(PACKAGE.START), 1)
        self.assertIn('<p>Keep this text</p>', first)


@unittest.skipUnless(all(shutil.which(name) for name in ('pdfunite', 'pdfinfo', 'pdftotext')),
                     'Poppler PDF tools are required for the book integration test')
class TestMacNLOPSVectorBook(unittest.TestCase):
    def test_common_book_and_zip_preserve_shared_cohort_contract(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY-common-book-') as temporary:
            report=Path(temporary)/'TEST_ONLY_shared_report';fixture(report)
            (report/'summary.json').write_text(json.dumps(common_summary()))
            result=PACKAGE.package_report(report)
            self.assertEqual(result['pages'],72)
            manifest=json.loads((report/PACKAGE.MANIFEST).read_text())
            self.assertEqual(manifest['physical_source_events'],800000)
            self.assertTrue(all(c['common_source_mode'] for c in manifest['comparison_contracts'].values()))
            book=json.loads((report/PACKAGE.CONTENTS).read_text())
            self.assertEqual(book['comparison_contracts'],manifest['comparison_contracts'])
            with zipfile.ZipFile(result['archive']) as bundle:
                self.assertIsNone(bundle.testzip())
                self.assertEqual(json.loads(bundle.read(report.name+'/'+PACKAGE.MANIFEST)),manifest)

    def test_labelled_72_page_book_manifest_and_portable_zip(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY-package-book-') as temporary:
            report = Path(temporary) / 'TEST_ONLY_report'
            labels = fixture(report)
            # Nested manifests are ordinary provenance artifacts, not the
            # packager's self-excluded top-level manifest.
            provenance = report / 'provenance'
            provenance.mkdir()
            (provenance / PACKAGE.MANIFEST).write_text('{"TEST_ONLY": true}\n')
            originals = {p.relative_to(report).as_posix():p.read_bytes()
                         for p in report.rglob('*') if p.is_file() and p.name != 'index.html'}
            result = PACKAGE.package_report(report)
            self.assertEqual(result['pages'], 72)
            self.assertEqual(PACKAGE.pdf_pages(report / PACKAGE.BOOK, shutil.which('pdfinfo')), 72)
            extracted = subprocess.check_output(['pdftotext', '-layout',
                str(report / PACKAGE.BOOK), '-'], text=True).split('\f')
            self.assertEqual(len(extracted), 73)
            for expected, actual in zip(labels, extracted):
                self.assertIn(expected, actual)
            for name, contents in originals.items():
                self.assertEqual((report / name).read_bytes(), contents)
            contents = json.loads((report / PACKAGE.CONTENTS).read_text())
            self.assertEqual([p['page'] for p in contents['pages']], list(range(1, 73)))
            manifest = json.loads((report / PACKAGE.MANIFEST).read_text())
            actual_names = {p.relative_to(report).as_posix() for p in report.rglob('*') if p.is_file()}
            self.assertEqual(set(manifest['files']), actual_names - {PACKAGE.MANIFEST})
            for name, entry in manifest['files'].items():
                self.assertEqual(entry['sha256'], hashlib.sha256((report / name).read_bytes()).hexdigest())
            archive = Path(result['archive'])
            self.assertEqual(archive.parent, report.parent)
            with zipfile.ZipFile(archive) as bundle:
                self.assertIsNone(bundle.testzip())
                self.assertEqual(set(bundle.namelist()), {report.name + '/' + name for name in actual_names})
                for name in actual_names:
                    self.assertEqual(bundle.read(report.name + '/' + name), (report / name).read_bytes())
            self.assertEqual(result['archive_sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
