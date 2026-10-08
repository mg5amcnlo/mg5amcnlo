"""Archive and verify the two-stage ttbar smoke, including native pool replay."""
from pathlib import Path
import hashlib, importlib.util, json, math, re, shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORK = Path((HERE / 'work_directory.txt').read_text().strip())
ARCHIVE = HERE / 'process'
OUT = WORK / 'ttbar'
if OUT.is_dir():
    files = list((OUT / 'Cards').glob('*.dat'))
    files += list((OUT / 'Events/two_stage').glob('*'))
    names = ['log_MINT1.txt', 'log_MINT2.txt', 'res_1.dat', 'res_2.dat',
             'ampli_grids', 'grid.MC_integer', 'ampli_pool.dat', 'ampli_job.dat', 'input_app.txt']
    for name in names:
        files += list((OUT / 'SubProcesses').glob('P*/G*/' + name))
    for src in files:
        if not src.is_file():
            continue
        dst = ARCHIVE / src.relative_to(OUT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    assert not list((OUT / 'SubProcesses').glob('P*/G*/log_MINT0.txt'))
    assert not list((OUT / 'SubProcesses').glob('P*/G*/res_0.dat'))
else:
    OUT = ARCHIVE

spec = importlib.util.spec_from_file_location('prior_native_audit', ROOT / 'validation/ttbar_unfolded_300k_20261007/analyze.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
report = audit.analyze(ARCHIVE, 'ampli', 2000, 'two_stage',
                       HERE / 'sources/madgraph/various/ampli_pool.py')
assert report['complete_expected_sample'] and report['integration_complete']
assert not report['fatal_logs']
manifest = report['production_manifest']
assert manifest['integration_stages'] == ['survey', 'generation']
assert manifest['survey_min_iterations'] == 4
assert report['stage_logs'] == {'0': 0, '1': 5, '2': 5}
surveys = []
for c in manifest['channels']:
    parent = ARCHIVE / 'SubProcesses' / c['subprocess'] / ('GF' + c['channel'])
    lines = (parent / 'ampli_grids').read_text().splitlines()
    assert lines[0].split() == ['MG5_AMPLICOL', '3', '1']
    values = list(map(float, lines[3].split()))
    nvalues = int(lines[1].split()[-1])
    absolute = values[0] + values[4]
    pvirt = max(.001, min(.999, values[4] / absolute))
    expected = max(values[-2] / (1-pvirt), values[-1] / pvirt)
    text = (parent / 'log_MINT1.txt').read_text()
    observations = re.findall(r'AmpliCol survey iteration, cumulative trials, ABS, signed, relative error:\s*(\d+)\s*(\d+)\s*(' + audit.NUMBER + r')\s*(' + audit.NUMBER + r')\s*(' + audit.NUMBER + r')', text)
    iteration, total, measured_absolute, signed, relerr = observations[-1]
    assert int(iteration) == c['survey_iterations'] >= 4
    assert float(relerr) <= .03
    assert math.isclose(float(measured_absolute), absolute, rel_tol=1e-12)
    kept, evaluated = map(int, re.search(r'AmpliCol survey retained statistical points, total evaluations:\s*(\d+)\s*(\d+)', text).groups())
    assert kept == c['survey_points'] and evaluated == int(total) and evaluated > kept
    for batch in c['batches']:
        log = (ARCHIVE / batch['directory'] / 'log_MINT2.txt').read_text()
        seed = re.search(r'AmpliCol saved survey stream maxima, initial production envelope:\s*(' + audit.NUMBER + r')\s*(' + audit.NUMBER + r')\s*(' + audit.NUMBER + r')', log)
        assert math.isclose(float(seed[3]), expected, rel_tol=1e-12)
        assert all(batch['adaptation']['mask'])
    surveys.append(dict(channel=c['subprocess'] + '/' + c['channel'], iterations=int(iteration),
                        relative_absolute_error=float(relerr), retained_points=kept,
                        total_survey_points=evaluated, initial_envelope=expected))
started = json.loads((HERE / 'started.json').read_text())
for name, digest in started['source_sha256'].items():
    assert hashlib.sha256((HERE / 'sources' / name).read_bytes()).hexdigest() == digest
    if OUT != ARCHIVE:
        exported = OUT / ('SubProcesses/' + Path(name).name if name.startswith('Template/') else 'bin/internal/' + Path(name).name)
        assert hashlib.sha256(exported.read_bytes()).hexdigest() == digest
report['two_stage_validation'] = dict(no_stage_zero=True, all_survey_iterations_at_least_four=True,
                                     saved_stream_maxima_seed_production=True,
                                     current_sources_match_export=True, surveys=surveys)
(HERE / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(integration=report['integration'], production=report['production'],
                     surveys=surveys, native_validation=report['native_validation']), indent=2))
