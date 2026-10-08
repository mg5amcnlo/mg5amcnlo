from pathlib import Path
import datetime, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = Path(tempfile.mkdtemp(prefix='mg5-ampli-two-stage-ttbar-'))
OUT = WORK / 'ttbar'
sys.path.insert(0, str(ROOT))
from madgraph.various.banner import RunCardNLO

(HERE / 'work_directory.txt').write_text(str(WORK) + '\n')
generate = HERE / 'generate.cmd'
generate.write_text('set automatic_html_opening False --no_save\n'
                    'import model loop_sm\ngenerate p p > t t~ [QCD]\n'
                    f'output {OUT} -f\nquit\n')
env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
with (HERE / 'generate.log').open('w') as log:
    subprocess.run([sys.executable, str(ROOT / 'bin/mg5_aMC'), str(generate)],
                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
assert (OUT / 'bin/aMCatNLO').exists()
settings = dict(nevents=2000, req_acc=-1., nevt_job=-1, iseed=19728,
                ebeam1=6500., ebeam2=6500., pdlabel='nn23nlo', parton_shower='PYTHIA8',
                folding=[1, 1, 1], born_spreading=False, event_norm='average',
                reweight_scale=[False], reweight_pdf=[False], store_rwgt_info=False)
card = RunCardNLO(str(OUT / 'Cards/run_card.dat'))
for key, value in settings.items():
    card[key] = value
card.write(str(OUT / 'Cards/run_card.dat'))
fks = OUT / 'Cards/FKS_params.dat'
fks.write_text(fks.read_text().replace('#NLOPSIntegrator\n0', '#NLOPSIntegrator\n1')
               .replace('#UsePolyVirtual\n.False.', '#UsePolyVirtual\n.True.'))
param = OUT / 'Cards/param_card.dat'
param.write_text(re.sub(r'(?mi)^(\s*DECAY\s+6\s+)\S+', r'\g<1>0.000000e+00', param.read_text()))
cmd = HERE / 'run.cmd'
cmd.write_text('set automatic_html_opening False --no_save\n'
               'set notification_center False --no_save\n'
               'set run_mode 2 --no_save\nset nb_core 3 --no_save\n'
               'launch aMC@NLO -f -p --name=two_stage\nquit\n')
sources = ['Template/NLO/SubProcesses/' + name for name in
           ['ampli_mint_adapter.f90', 'simple_integrator.f90', 'driver_mintMC.f', 'mint_module.f90']]
sources += ['madgraph/interface/amcatnlo_run_interface.py', 'madgraph/various/ampli_pool.py']
meta = dict(settings=settings, work_directory=str(WORK), cores=3,
            started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources})
for name in sources:
    target = HERE / 'sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)
(HERE / 'started.json').write_text(json.dumps(meta, indent=2) + '\n')
start = time.monotonic()
with (HERE / 'run.log').open('w') as log:
    result = subprocess.run([sys.executable, '-O', str(OUT / 'bin/aMCatNLO'), str(cmd)],
                            cwd=OUT, stdout=log, stderr=subprocess.STDOUT, env=env)
meta.update(returncode=result.returncode, wall_seconds=time.monotonic()-start,
            events_exist=(OUT / 'Events/two_stage/events.lhe.gz').exists())
(HERE / 'finished.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta, indent=2), flush=True)
assert meta['events_exist']
