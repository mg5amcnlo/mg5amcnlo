"""Isolate the job-size change using an exact copy of the saved ttbar survey."""
from pathlib import Path
import datetime, hashlib, json, os, pickle, re, shutil, subprocess, sys, tempfile, time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / 'validation/ttbar_bounded_300k_20261007'
SOURCE = Path((PRIOR / 'work_directory.txt').read_text().strip()) / 'ampli'
WORK = Path(tempfile.mkdtemp(prefix='mg5-ttbar-30k-jobs-'))
OUT = WORK / 'ampli'
RUN = 'jobs30k_300k'
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
assert not (HERE / 'ampli_started.json').exists(), 'Preserve existing benchmark evidence'
sys.path.insert(0, str(ROOT))
from madgraph.various.banner import RunCardNLO

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def omit_old_workers(directory, names):
    return [n for n in names if re.fullmatch(r'GF[0-9.]+_[0-9]+', n)]

shutil.copytree(SOURCE, OUT, symlinks=True, ignore=omit_old_workers)
assert not list((OUT / 'SubProcesses').glob('P*/GF*_*'))
# Result reporting sorted the saved channel list by rate. Restore its recorded
# allocation order in this isolated copy, so a generation-only restart uses
# precisely the previous CDF and initial quotas with the same seed.
status_path = OUT / 'SubProcesses/job_status.pkl'
with status_path.open('rb') as stream:
    jobs = pickle.load(stream)
jobs.sort(key=lambda job: job['ampli_allocation_order'])
assert [job['ampli_allocation_order'] for job in jobs] == list(range(5))
allocation_order = [dict(subprocess=j['p_dir'], channel=j['channel'],
                         allocation_order=j['ampli_allocation_order'],
                         initial_quota=j['ampli_initial_quota']) for j in jobs]
with status_path.open('wb') as stream:
    pickle.dump(jobs, stream)
(HERE / 'allocation_order.json').write_text(json.dumps(allocation_order, indent=2) + '\n')
for name in ('ampli_production.json', 'nevents_unweighted'):
    (OUT / 'SubProcesses' / name).unlink(missing_ok=True)
(HERE / 'work_directory.txt').write_text(str(WORK) + '\n')
card = RunCardNLO(str(OUT / 'Cards/run_card.dat'))
card['iseed'] = 19727
card['nevt_job'] = 30000
assert card['nevents'] == 300000 and card['req_acc'] == -1.
assert card['folding'] == [1, 1, 1]
card.write(str(OUT / 'Cards/run_card.dat'))
for name in ('run_card.dat', 'param_card.dat', 'FKS_params.dat'):
    shutil.copy2(OUT / 'Cards' / name, HERE / ('ampli_' + name))

previous = json.loads((PRIOR / 'ampli_started.json').read_text())
for name, expected in previous['source_sha256'].items():
    assert digest(ROOT / name) == expected, name
    destination = HERE / 'sources' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, destination)
for name, expected in previous['exported_source_sha256'].items():
    assert digest(OUT / name) == expected, name

def survey_state():
    result = {}
    for name in ('ampli_grids', 'grid.MC_integer', 'res_1.dat', 'log_MINT1.txt'):
        for path in (OUT / 'SubProcesses').glob('P*/GF*/' + name):
            if '_' in path.parent.name:
                continue
            result[str(path.relative_to(OUT))] = dict(sha256=digest(path), mtime_ns=path.stat().st_mtime_ns)
    assert len(result) == 20
    return result

before = survey_state()
for name, state in before.items():
    assert state['sha256'] == digest(SOURCE / name), name
cmd = HERE / 'ampli.cmd'
cmd.write_text('set automatic_html_opening False --no_save\n'
               'set notification_center False --no_save\n'
               'set run_mode 2 --no_save\nset nb_core 5 --no_save\n'
               f'launch aMC@NLO -f -p --only_generation --name={RUN}\nquit\n')
meta = dict(backend='ampli', run_name=RUN, work_directory=str(WORK), cores=5,
            previous_benchmark=str(PRIOR), expected_events=300000, req_acc=-1,
            folding=[1, 1, 1], seed=19727, nevt_job=30000, expected_initial_workers=13,
            fresh_export=False, fresh_survey=False, generation_only=True,
            restored_saved_allocation_order=True, allocation_order=allocation_order,
            source_export=str(SOURCE), survey_before=before,
            source_sha256=previous['source_sha256'],
            exported_source_sha256=previous['exported_source_sha256'],
            started=datetime.datetime.now(datetime.timezone.utc).isoformat())
(HERE / 'ampli_started.json').write_text(json.dumps(meta, indent=2) + '\n')
start = time.monotonic()
with (HERE / 'ampli.log').open('w') as log:
    result = subprocess.run([sys.executable, '-O', str(OUT / 'bin/aMCatNLO'), str(cmd)],
                            cwd=OUT, stdout=log, stderr=subprocess.STDOUT, env=ENV)
after = survey_state()
meta.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            cli_returncode=result.returncode, wall_seconds=time.monotonic()-start,
            summary_exists=(OUT / 'Events' / RUN / 'summary.txt').exists(),
            events_exist=(OUT / 'Events' / RUN / 'events.lhe.gz').exists(),
            survey_after=after, survey_unchanged=(before == after))
(HERE / 'ampli_finished.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps({k: v for k, v in meta.items() if k not in ('survey_before', 'survey_after')}, indent=2), flush=True)
assert result.returncode == 0 and meta['events_exist'] and meta['summary_exists']
assert meta['survey_unchanged'], 'Generation-only restart changed the saved survey'
