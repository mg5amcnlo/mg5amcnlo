"""Validate the generation scheduler on the same ttbar survey as the stopped run."""
from pathlib import Path
import datetime, hashlib, json, os, resource, shutil, subprocess, sys, tempfile, time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = ROOT / 'validation/ttbar_two_stage_300k_20261007'
WORK = Path(tempfile.mkdtemp(prefix='mg5-ttbar-bounded-iterations-'))
OUT = WORK / 'ampli'
RUN = 'bounded_10k'
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
for name in ('run_card.dat', 'FKS_params.dat', 'param_card.dat'):
    shutil.copy2(PREVIOUS / ('ampli_' + name), OUT / 'Cards' / name)
card = RunCardNLO(str(OUT / 'Cards/run_card.dat'))
card['iseed'] = 19727
card['nevents'] = 10000
assert card['req_acc'] == -1. and card['folding'] == [1, 1, 1] and card['nevt_job'] == 2500
card.write(str(OUT / 'Cards/run_card.dat'))
cmd = HERE / 'ampli.cmd'
cmd.write_text('set automatic_html_opening False --no_save\n'
               'set notification_center False --no_save\n'
               'set run_mode 2 --no_save\nset nb_core 3 --no_save\n'
               f'launch aMC@NLO -f -p --name={RUN}\nquit\n')
sources = ['Template/NLO/SubProcesses/' + name for name in
           ('ampli_mint_adapter.f90', 'simple_integrator.f90', 'driver_mintMC.f',
            'mint_module.f90', 'integrator_helpers.f90', 'genps_fks_radiation.f')]
sources += ['madgraph/interface/amcatnlo_run_interface.py', 'madgraph/various/ampli_pool.py']
export_hashes = {}
for name in sources:
    target = HERE / 'sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)
    exported = ('SubProcesses/' if name.startswith('Template/') else 'bin/internal/') + Path(name).name
    assert digest(ROOT / name) == digest(OUT / exported), exported
    export_hashes[exported] = digest(OUT / exported)
meta = dict(backend='ampli', run_name=RUN, work_directory=str(WORK), cores=3,
            previous_benchmark=str(PREVIOUS), expected_events=10000, req_acc=-1,
            folding=[1, 1, 1], seed=19727, fresh_export=True, fresh_survey=True,
            started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_sha256={name: digest(ROOT / name) for name in sources},
            exported_source_sha256=export_hashes)
(HERE / 'ampli_started.json').write_text(json.dumps(meta, indent=2) + '\n')
start = time.monotonic()
with (HERE / 'ampli.log').open('w') as log:
    result = subprocess.run([sys.executable, '-O', str(OUT / 'bin/aMCatNLO'), str(cmd)],
                            cwd=OUT, stdout=log, stderr=subprocess.STDOUT, env=ENV)
meta.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            cli_returncode=result.returncode, wall_seconds=time.monotonic()-start,
            summary_exists=(OUT / 'Events' / RUN / 'summary.txt').exists(),
            events_exist=(OUT / 'Events' / RUN / 'events.lhe.gz').exists())
(HERE / 'ampli_finished.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta, indent=2), flush=True)
assert result.returncode == 0 and meta['events_exist'] and meta['summary_exists']
