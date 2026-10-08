"""Replay pool/rate/collection checks and verify the bounded iteration schedule."""
from pathlib import Path
import hashlib, json, math, re
import analyze_common

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'ampli'
PRIOR = ROOT / 'validation/ttbar_two_stage_300k_20261007'
report = analyze_common.analyze(OUT, 'ampli', 10000, 'bounded_10k',
                                 HERE / 'sources/madgraph/various/ampli_pool.py')
assert report['complete_expected_sample'] and report['integration_complete'] and not report['fatal_logs']
assert report['stage_logs']['0'] == 0 and report['stage_logs']['1'] == 5
assert report['production_manifest']['integration_stages'] == ['survey', 'generation']
assert (OUT / 'SubProcesses/randinit').read_text().strip() == 'r=19727'
started = json.loads((HERE / 'ampli_started.json').read_text())
for name, digest in started['source_sha256'].items():
    assert hashlib.sha256((HERE / 'sources' / name).read_bytes()).hexdigest() == digest
for name, digest in started['exported_source_sha256'].items():
    assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest

def close(x, y):
    assert math.isclose(x, y, rel_tol=1e-11, abs_tol=1e-12), (x, y)

details = []
for channel in report['production_manifest']['channels']:
    parent = Path('SubProcesses') / channel['subprocess'] / ('GF' + channel['channel'])
    saved = (OUT / parent / 'ampli_grids').read_text().splitlines()
    assert saved[0].split() == ['MG5_AMPLICOL', '3', '1']
    values = list(map(float, saved[3].split()))
    absolute = values[0] + values[4]
    pvirt = max(.001, min(.999, values[4] / absolute))
    initial_envelope = max(values[-2] / (1-pvirt), values[-1] / pvirt)
    assert channel['survey_iterations'] >= 4
    survey = analyze_common.parse_result(OUT / parent / 'res_1.dat')
    assert survey['absolute_error_pb'] <= .03 * absolute
    # The survey algorithm and seed are unchanged; its actual grids/maxima
    # must reproduce the pathological state from the stopped 300K run.
    assert (OUT / parent / 'ampli_grids').read_bytes() == (PRIOR / 'partial' / parent / 'ampli_grids').read_bytes()
    assert (OUT / parent / 'grid.MC_integer').read_bytes() == (PRIOR / 'partial' / parent / 'grid.MC_integer').read_bytes()
    for batch in channel['batches']:
        directory = OUT / batch['directory']
        rows = (directory / 'ampli_pool.dat').read_text().splitlines()
        target, quota = map(int, rows[3].split()[:2])
        n_epoch = int(rows[1].split()[2])
        epochs = [list(map(float, line.split())) for line in rows[6:6+n_epoch]]
        assert int(epochs[0][3]) == max(1024, min(8192, target))
        close(epochs[0][9], min(initial_envelope, absolute))
        assert epochs[0][10] >= initial_envelope * (1-1e-12)
        for previous, current in zip(epochs, epochs[1:]):
            assert current[3] <= 2 * previous[3]
        assert all(e[2] == e[3] and e[1] >= e[2] for e in epochs)
        assert all(batch['adaptation']['mask'])
        text = (directory / 'log_MINT2.txt').read_text()
        forecast = re.findall(r'AmpliCol native effective events, remaining, nonzero acceptance:\s*([^\n]+)', text)
        assert len(forecast) == n_epoch-1
        for i, row in enumerate(forecast):
            effective, remaining, efficiency = map(float, row.split())
            close(remaining, max(1., target-effective))
            expected = int(2*epochs[i][3])
            if efficiency > 0:
                expected = max(min(1024, expected), min(expected, math.ceil(1.1*remaining/efficiency)))
            assert epochs[i+1][3] == expected
        details.append(dict(directory=batch['directory'], envelope_over_absolute=initial_envelope/absolute,
                            initial_envelope=initial_envelope, initial_cutoff=epochs[0][9],
                            iteration_targets=[int(e[3]) for e in epochs],
                            previous_scheduler_initial_target=max(1024, math.ceil(target*max(1.,initial_envelope/absolute)))))

report['bounded_scheduler_validation'] = dict(
    same_survey_and_maxima_as_stopped_300k=True, saved_maximum_retained_in_first_envelope=True,
    bounded_first_iterations=True, forecasts_reproduced=True, growth_cap_verified=True,
    shrinking_iteration_count=sum(sum(b < a for a,b in zip(d['iteration_targets'],d['iteration_targets'][1:])) for d in details),
    workers=details)
(HERE / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(integration=report['integration'], production=report['production'],
                     native_tail=report['native_validation']['maximum_native_tail_fraction'],
                     collected_tail=report['native_validation']['collected_tail_fraction'],
                     scheduler=report['bounded_scheduler_validation']), indent=2))
