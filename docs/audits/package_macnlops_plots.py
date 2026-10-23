#!/usr/bin/env python3
"""Package a strictly completed four-process MAcNLOPS plot report.

Usage: package_macnlops_plots.py PLOTS_DIRECTORY [--archive /outside/report.zip]
The default ZIP is a sibling of the report directory. No preview override is
provided. Existing plot, CSV, NPZ, and provenance files are preserved.
Nine existing vector panels per process/stage form a 72-page book, including
the fine soft-jet source-sign and rejection-mechanism views.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


PROCESSES = ('dy', 'zz', 'ttbar', 'wbb')
LEVELS = ('first', 'full')
PANELS = (
    'jet_pt_linear_ratio_zoom',
    'system_pt_log_ratio_zoom',
    'delta_phi_linear_ratio_zoom',
    'delta_phi_deficit_log_ratio_zoom',
    'delta_y_linear_ratio_zoom',
    'system_mass_linear_ratio_zoom',
    'jet_pt_source_signs',
    'jet_pt_soft_source_signs',
    'jet_pt_soft_mechanisms',
)
BOOK = 'macnlops_validation_book.pdf'
CONTENTS = 'book_contents.json'
MANIFEST = 'artifact_manifest.json'
SCHEMA = 'macnlops-validation-histograms-v1'
START = '<!-- macnlops-packaging:start -->'
END = '<!-- macnlops-packaging:end -->'


class PackagingError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise PackagingError('Cannot read JSON %s: %s' % (path, error)) from error


def event_count(value, context):
    if type(value) is not int or value != 200000:
        raise PackagingError('%s must contain exactly 200000 events' % context)


def validate_summary(summary):
    if not isinstance(summary, dict) or summary.get('schema') != SCHEMA:
        raise PackagingError('Unknown plot summary schema')
    if summary.get('complete') is not True or summary.get('smoke') is not False:
        raise PackagingError('Final packaging requires complete=True and smoke=False')
    levels = summary.get('levels', [])
    if not isinstance(levels, list) or len(levels) != 2 or not all(
            isinstance(level, str) for level in levels) or set(levels) != set(LEVELS):
        raise PackagingError('Final packaging requires both first and full stages')
    processes = summary.get('processes', {})
    if not isinstance(processes, dict) or set(processes) != set(PROCESSES):
        raise PackagingError('Final packaging requires dy, zz, ttbar and wbb')
    counts = {}
    for process in PROCESSES:
        record = processes[process]
        if not isinstance(record, dict) or not isinstance(record.get('levels'), dict) or \
                set(record['levels']) != set(LEVELS):
            raise PackagingError('Missing first/full stages for ' + process)
        config = record.get('configuration', {})
        if not isinstance(config, dict):
            raise PackagingError('Invalid production configuration for ' + process)
        event_count(config.get('expected_events'), process + ' configured source')
        if config.get('flavours') != (4 if process == 'wbb' else 5):
            raise PackagingError('Wrong flavour scheme for ' + process)
        common=config.get('common_source_mode',False)
        signed=config.get('signed_residuals',config.get('allow_signed_residuals',False))
        if type(common) is not bool or type(signed) is not bool:
            raise PackagingError('Invalid comparison/residual policy for '+process)
        if common and (config.get('source_construction')!='ordinary-mcatnlo-v1' or config.get('reference')):
            raise PackagingError('Shared ordinary comparison must omit independent reference chunks: '+process)
        for role in ('primary', 'reference'):
            if config.get(role + '_complete', True) is not True:
                raise PackagingError('Incomplete %s %s production' % (process, role))
        counts[process] = {}
        for level in LEVELS:
            stage = record['levels'][level]
            context = process + '/' + level
            if not isinstance(stage, dict) or stage.get('complete') is not True:
                raise PackagingError('Incomplete plot stage ' + context)
            totals = stage.get('totals', {})
            if not isinstance(totals, dict):
                raise PackagingError('Invalid event totals for ' + context)
            for role in ('input', 'reference'):
                total = totals.get(role)
                event_count(total.get('events') if isinstance(total, dict) else None,
                            context + ' ' + role)
            chunks = stage.get('chunks', {})
            if not isinstance(chunks, dict):
                raise PackagingError('Invalid chunk accounting for ' + context)
            for role in ('primary', 'reference'):
                rows = chunks.get(role)
                if not isinstance(rows, list) or not rows or any(
                        not isinstance(row, dict) or type(row.get('events')) is not int
                        or row['events'] <= 0 for row in rows):
                    raise PackagingError('Invalid chunk accounting for %s %s' % (context, role))
                event_count(sum(row['events'] for row in rows), context + ' ' + role + ' chunks')
            counts[process][level] = dict(input=200000, reference=200000)
            if common:
                if stage.get('common_source_mode') is not True or stage.get('signed_residuals') is not signed:
                    raise PackagingError('Shared comparison/policy metadata differs: '+context)
                if totals['input']!=totals['reference']:
                    raise PackagingError('Shared baseline/input totals differ: '+context)
                primary,reference=chunks['primary'],chunks['reference']
                if len(primary)!=len(reference):
                    raise PackagingError('Shared baseline chunk count differs: '+context)
                paths=[]
                for original,derived in zip(primary,reference):
                    if derived.get('derived_from_primary') is not True:
                        raise PackagingError('Shared baseline must be derived from primary: '+context)
                    for key in ('path','events','weight_sequence_sha256','source_sequence_sha256','full_audit'):
                        if key not in original or original[key]!=derived.get(key):
                            raise PackagingError('Shared baseline chunk binding differs: '+context+' '+key)
                    paths.append(original['path'])
                if len(set(paths))!=len(paths):
                    raise PackagingError('Shared source repeats a native chunk: '+context)
                residual=totals.get('H_negative_retained',{})
                negative=residual.get('events')
                if type(negative) is not int or negative<0 or residual.get('negative_events')!=negative or (negative and not signed):
                    raise PackagingError('Invalid retained negative-H accounting: '+context)
                counts[process][level].update(physical_source_events=200000,
                    common_source_mode=True,negative_H_retained=negative)
        if common:
            first,full=(record['levels'][level] for level in LEVELS)
            if first['totals']!=full['totals']:
                raise PackagingError('Shared first/full total weights or counts differ: '+process)
            a,b=first['chunks']['primary'],full['chunks']['primary']
            if len(a)!=len(b) or any(any(x[key]!=y[key] for key in (
                    'events','weight_sequence_sha256','source_sequence_sha256')) for x,y in zip(a,b)):
                raise PackagingError('Shared first/full source identities differ: '+process)
    return counts


def comparison_contracts(summary):
    """Keep physical cohort counts distinct from derived baseline curves."""
    contracts={}
    for process,record in summary['processes'].items():
        config=record['configuration'];common=config.get('common_source_mode',False)
        contracts[process]=dict(common_source_mode=common,
            source_construction=config.get('source_construction','legacy independent cohorts'),
            signed_residuals=config.get('signed_residuals',config.get('allow_signed_residuals',False)),
            baseline='same primary events and shower' if common else 'independent reference cohort',
            source_events_before_rejection=200000,physical_source_events=200000 if common else 400000)
    return contracts


def report_files(directory):
    files = []
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise PackagingError('Portable report must contain regular files: ' + str(path))
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise PackagingError('Unsupported report entry: ' + str(path))
    return files


def required_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise PackagingError('Missing or empty plot artifact: ' + str(path))


def validate_report(directory):
    summary = read_json(directory / 'summary.json')
    counts = validate_summary(summary)
    required_file(directory / 'index.html')
    required_file(directory / 'stage_totals.csv')
    pages = []
    for process in PROCESSES:
        for level in LEVELS:
            stage = directory / process / level
            required_file(stage / 'histograms.npz')
            for observable in ('jet_pt', 'jet_pt_soft', 'system_pt', 'delta_phi',
                               'delta_phi_deficit', 'delta_y', 'system_mass'):
                required_file(stage / (observable + '.csv'))
            for panel in PANELS:
                pdf = stage / (panel + '.pdf')
                required_file(pdf)
                required_file(stage / (panel + '.png'))
                pages.append(dict(page=len(pages) + 1, process=process,
                                  stage=level, panel=panel,
                                  path=pdf.relative_to(directory).as_posix(),
                                  sha256=sha256(pdf)))
    report_files(directory)
    return summary, counts, pages


def pdf_pages(path, pdfinfo):
    result = subprocess.run([pdfinfo, str(path)], capture_output=True, text=True)
    match = re.search(r'^Pages:\s+(\d+)\s*$', result.stdout, re.M)
    if result.returncode or match is None:
        raise PackagingError('Cannot inspect PDF %s: %s' % (path, result.stderr.strip()))
    return int(match[1])


def linked_index(original, pages):
    block = (START + '\n<section id="download-book"><h2>Vector PDF book</h2>\n'
             '<p><a href="' + BOOK + '">Download the ' + str(pages) + '-page PDF book</a>; '
             '<a href="' + CONTENTS + '">page contents and panel hashes</a>; '
             '<a href="' + MANIFEST + '">artifact SHA256 manifest</a>.</p>\n'
             '</section>\n' + END)
    if START in original or END in original:
        pattern = re.escape(START) + r'.*?' + re.escape(END)
        updated, count = re.subn(pattern, lambda _: block, original, flags=re.S)
        if count != 1:
            raise PackagingError('Malformed existing packaging links in index.html')
        return updated
    if '</h1>' in original:
        return original.replace('</h1>', '</h1>\n' + block, 1)
    return block + '\n' + original


def package_report(directory, archive=None):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise PackagingError('Not a plot report directory: ' + str(directory))
    archive = Path(archive).resolve() if archive else directory.with_name(directory.name + '.zip')
    try:
        archive.relative_to(directory)
    except ValueError:
        pass
    else:
        raise PackagingError('ZIP must be outside the report directory')
    if archive.suffix.lower() != '.zip' or not archive.parent.is_dir():
        raise PackagingError('ZIP requires an existing parent directory and a .zip suffix')
    summary, counts, pages = validate_report(directory)
    comparisons=comparison_contracts(summary)
    pdfunite, pdfinfo = shutil.which('pdfunite'), shutil.which('pdfinfo')
    if not pdfunite or not pdfinfo:
        raise PackagingError('Install pdfunite and pdfinfo before packaging')
    for page in pages:
        if pdf_pages(directory / page['path'], pdfinfo) != 1:
            raise PackagingError('Book panels must be single-page PDFs: ' + page['path'])

    # All completeness and input checks precede changes to the report. Staging
    # outside it also keeps temporary output out of the archive and manifest.
    with tempfile.TemporaryDirectory(prefix=directory.name + '-package-',
                                     dir=directory.parent) as temporary, \
            tempfile.TemporaryDirectory(prefix=archive.stem + '-zip-',
                                        dir=archive.parent) as archive_temporary:
        staging = Path(temporary)
        result = subprocess.run([pdfunite] + [str(directory / row['path']) for row in pages]
                                + [str(staging / BOOK)], capture_output=True, text=True)
        if result.returncode:
            raise PackagingError('pdfunite failed: ' + result.stderr.strip())
        if pdf_pages(staging / BOOK, pdfinfo) != len(pages):
            raise PackagingError('Combined book page count does not match its contents')
        (staging / CONTENTS).write_text(json.dumps(dict(
            schema='macnlops-validation-book-v1', book=BOOK, pages=pages,
            comparison_contracts=comparisons),
            indent=2, sort_keys=True) + '\n')
        (staging / 'index.html').write_text(linked_index(
            (directory / 'index.html').read_text(), len(pages)))
        replacements = {name:staging / name for name in (BOOK, CONTENTS, 'index.html')}
        originals = {path.relative_to(directory).as_posix():path
                     for path in report_files(directory) if path != directory / MANIFEST}
        artifacts = dict(originals, **replacements)
        manifest = dict(schema='macnlops-validation-artifacts-v1', complete=True,
                        smoke=False, report=directory.name,
                        summary_sha256=sha256(directory / 'summary.json'),
                        expected_events=200000, counts=counts,
                        comparison_contracts=comparisons,
                        physical_source_events=sum(c['physical_source_events'] for c in comparisons.values()),
                        book=dict(path=BOOK, pages=len(pages), contents=CONTENTS),
                        files={name:dict(sha256=sha256(path), bytes=path.stat().st_size)
                               for name,path in sorted(artifacts.items())})
        (staging / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        artifacts[MANIFEST] = staging / MANIFEST
        temporary_zip = Path(archive_temporary) / 'report.zip'
        with zipfile.ZipFile(temporary_zip, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as output:
            for name,path in sorted(artifacts.items()):
                output.write(path, directory.name + '/' + name)
        with zipfile.ZipFile(temporary_zip) as output:
            invalid = output.testzip()
            if invalid:
                raise PackagingError('ZIP integrity check failed: ' + invalid)
        for name in (BOOK, CONTENTS, 'index.html', MANIFEST):
            (staging / name).replace(directory / name)
        temporary_zip.replace(archive)
    return dict(complete=True, smoke=False, book=str(directory / BOOK),
                pages=len(pages), manifest=str(directory / MANIFEST),
                artifact_files=len(manifest['files']), archive=str(archive),
                archive_sha256=sha256(archive))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plots_directory', type=Path)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    try:
        result = package_report(args.plots_directory, args.archive)
    except (PackagingError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
