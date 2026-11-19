"""Input locations and stale-sample checks shared by the single-top analyses."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENTRAL_RUNS = {
    'default': 'C_GLB', 'local': 'C_LOC',
    'local_patch': 'C_Q', 'local_2patch': 'C_QT',
}


def check_run(directory, run):
    """Reject production samples explicitly invalidated by the clustering fix."""
    directory = Path(directory).resolve()
    marker = directory / 'NORMALIZATION_FIX.json'
    if marker.exists() and run in json.loads(marker.read_text()).get('invalid_runs', []):
        raise ValueError(f'{directory}/Events/{run}: superseded by the core '
                         'clustering fix; regenerate this MC@NLO sample before analysis')
    marker = directory / 'NUMERICAL_FIX.json'
    if marker.exists() and run in json.loads(marker.read_text()).get('invalid_runs', []):
        raise ValueError(f'{directory}/Events/{run}: predates the local-recoil '
                         'numerical fixes; regenerate this MC@NLO sample before analysis')


def check_histogram(path):
    """Apply the process marker check to files, including symbolic links."""
    path = Path(path).resolve()
    for parent in path.parents:
        if parent.name == 'Events':
            run = path.relative_to(parent).parts[0]
            check_run(parent.parent, run)
            if (parent / run / 'campaign_done.json').is_file():
                validate_completion(parent.parent, run)
                if path != parent / run / 'plot_PYTHIA8_1_0.HwU':
                    raise ValueError(f'Unrecorded shower output: {path}. '
                                     'Use the HwU recorded by the campaign runner.')
            break
    return path


def validate_completion(directory, run):
    """Check rerun provenance and the files used by the analysis.

    Event-file hashes are checked by the campaign runner. Analysis checks the
    recorded cards/source and the summary/HwU it reads, avoiding a second pass
    over every multi-gigabyte LHE file merely to draw plots.
    """
    from paper.run_g_variations import (SCHEMA, generation_inputs, prepared_files,
                                       check_fixed_clustering, sha256,
                                       source_migration_provenance)
    directory = Path(directory).resolve()
    check_run(directory, run)
    events = directory / 'Events' / run
    try:
        record = json.loads((events / 'campaign_done.json').read_text())
        manifest = json.loads((directory / 'campaign_manifest.json').read_text())
        row = json.loads((directory / 'run_spec.json').read_text())
        inputs = generation_inputs(row)
        cards = prepared_files(directory)
        migration = source_migration_provenance(row, directory, inputs)
        recorded_migration = {name: record[name] for name in
                              ('generation_inputs', 'source_migration_sha256')
                              if name in record}
        valid = (row['run_id'] == run and record.get('schema') == SCHEMA
                 and record.get('inputs') == inputs and manifest.get('inputs') == inputs
                 and record.get('cards_sha256') == cards
                 and manifest.get('prepared_sha256') == cards
                 and recorded_migration == migration)
        check_fixed_clustering(directory / 'SubProcesses/cluster.f')
        valid = valid and sha256(directory / 'SubProcesses/cluster.f') == inputs[
            'source_sha256']['Template/NLO/SubProcesses/cluster.f']
        for filename in ['summary.txt', 'plot_PYTHIA8_1_0.HwU']:
            valid = valid and record.get('output_sha256', {}).get(filename) == sha256(events / filename)
    except (OSError, KeyError, RuntimeError, ValueError) as error:
        raise ValueError(f'Cannot verify corrected sample {events}: {error}') from error
    if not valid:
        raise ValueError(f'Changed or unverified corrected sample: {events}. '
                         'Use a fresh export with the campaign runner.')
    return record


def find_input(path, run=None):
    """Accept an HwU file or process export and select its latest shower."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return check_histogram(path)
    if run is None:
        run = path.name if path.name.startswith(('C_', 'G_', 'P_', 'S_')) else 'run_01'
    check_run(path, run)
    if (path / 'Events' / run / 'campaign_done.json').is_file():
        return check_histogram(path / 'Events' / run / 'plot_PYTHIA8_1_0.HwU')
    candidates = list((path / 'Events' / run).glob('plot_PYTHIA8_*_0.HwU'))
    if not candidates:
        raise ValueError(f'No shower histograms in {path}/Events/{run}. '
                         'Generate the corrected sample first.')
    return check_histogram(max(candidates, key=lambda p: p.stat().st_mtime_ns))


def source_name(path):
    """Repository-relative provenance when possible, absolute for external runs."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
