"""Fresh 300K two-stage AmpliCol run with the preceding benchmark's cards."""
from pathlib import Path
import datetime, hashlib, json, os, resource, shutil, subprocess, sys, tempfile, time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent / 'ttbar_unfolded_300k_20261007'
WORK = Path(tempfile.mkdtemp(prefix='mg5-ttbar-two-stage-300k-'))
OUT = WORK / 'ampli'
RUN = 'benchmark_300k'
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
sys.path.insert(0, str(ROOT))
from madgraph.various.banner import RunCardNLO

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

(HERE / 'work_directory.txt').write_text(str(WORK) + '\n')
generate = HERE / 'generate.cmd'
generate.write_text('set automatic_html_opening False --no_save\n'
                    'import model loop_sm\ngenerate p p > t t~ [QCD]\n'
                    f'output {OUT} -f\nquit\n')
with (HERE / 'generate.log').open('w') as log:
    subprocess.run([sys.executable, str(ROOT / 'bin/mg5_aMC'), str(generate)],
                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=ENV, check=True)
assert (OUT / 'bin/aMCatNLO').exists()
assert not list((OUT / 'SubProcesses').glob('P*/G*/ampli_grids'))
for name in ('run_card.dat', 'FKS_params.dat', 'param_card.dat'):
    shutil.copy2(PREVIOUS / 'ampli/Cards' / name, OUT / 'Cards' / name)
# MG5 resets the persistent card's seed to zero after launching. Restore the
# explicit seed recorded in the previous run's banner, before launching anew.
card = RunCardNLO(str(OUT / 'Cards/run_card.dat'))
card['iseed'] = 19727
assert card['nevents'] == 300000 and card['req_acc'] == -1.
assert card['folding'] == [1, 1, 1] and card['nevt_job'] == 2500
card.write(str(OUT / 'Cards/run_card.dat'))
for name in ('run_card.dat', 'FKS_params.dat', 'param_card.dat'):
    shutil.copy2(OUT / 'Cards' / name, HERE / ('ampli_' + name))
cmd = HERE / 'ampli.cmd'
cmd.write_text('set automatic_html_opening False --no_save\n'
               'set notification_center False --no_save\n'
               'set run_mode 2 --no_save\nset nb_core 5 --no_save\n'
               f'launch aMC@NLO -f -p --name={RUN}\nquit\n')
sources = ['Template/NLO/SubProcesses/' + name for name in
           ('ampli_mint_adapter.f90', 'simple_integrator.f90', 'driver_mintMC.f',
            'mint_module.f90', 'integrator_helpers.f90', 'genps_fks_radiation.f')]
sources += ['madgraph/interface/amcatnlo_run_interface.py', 'madgraph/various/ampli_pool.py',
            'madgraph/iolibs/export_fks.py']
export_hashes = {}
for name in sources:
    target = HERE / 'sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)
    if name.startswith('Template/'):
        exported = 'SubProcesses/' + Path(name).name
    elif name.endswith('export_fks.py'):
        continue
    else:
        exported = 'bin/internal/' + Path(name).name
    assert digest(ROOT / name) == digest(OUT / exported), exported
    export_hashes[exported] = digest(OUT / exported)
meta = dict(backend='ampli', run_name=RUN, work_directory=str(WORK), cores=5,
            previous_benchmark=str(PREVIOUS), fresh_export=True, fresh_survey=True,
            expected_events=300000, req_acc=-1, folding=[1, 1, 1], seed=19727,
            started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            source_branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
            source_sha256={name: digest(ROOT / name) for name in sources},
            exported_source_sha256=export_hashes,
            cards_sha256={name: digest(OUT / 'Cards' / name) for name in
                          ('run_card.dat', 'FKS_params.dat', 'param_card.dat')})
(HERE / 'ampli_started.json').write_text(json.dumps(meta, indent=2) + '\n')
print('Launching fresh 300K two-stage run in ' + str(OUT), flush=True)
start = time.monotonic()
usage0 = resource.getrusage(resource.RUSAGE_CHILDREN)
with (HERE / 'ampli.log').open('w') as log:
    result = subprocess.run([sys.executable, '-O', str(OUT / 'bin/aMCatNLO'), str(cmd)],
                            cwd=OUT, stdout=log, stderr=subprocess.STDOUT, env=ENV)
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
meta.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            cli_returncode=result.returncode, wall_seconds=time.monotonic()-start,
            child_user_cpu_seconds=usage.ru_utime-usage0.ru_utime,
            child_system_cpu_seconds=usage.ru_stime-usage0.ru_stime,
            summary_exists=(OUT / 'Events' / RUN / 'summary.txt').exists(),
            events_exist=(OUT / 'Events' / RUN / 'events.lhe.gz').exists())
(HERE / 'ampli_finished.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta, indent=2), flush=True)
assert result.returncode == 0 and meta['events_exist'] and meta['summary_exists']
