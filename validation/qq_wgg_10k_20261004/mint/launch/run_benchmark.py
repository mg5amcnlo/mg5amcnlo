from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path('/export/tmp/rikkert/git/mg5amcnlo/MCcntRefactor_Sfun_Granny')
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from madgraph.various.banner import RunCardNLO

parser = argparse.ArgumentParser()
parser.add_argument('backend', choices=('mint', 'ampli'))
args = parser.parse_args()
out = WORK / args.backend
shutil.copytree(WORK / 'base', out, symlinks=True)
settings = dict(nevents=10000, req_acc=0.01, nevt_job=2500,
                iseed=19721, ebeam1=6500., ebeam2=6500.,
                pdlabel='nn23nlo', parton_shower='PYTHIA8',
                folding=[2, 1, 1], born_spreading=False,
                reweight_scale=[False], reweight_pdf=[False],
                store_rwgt_info=False, event_norm='average',
                fixed_ren_scale=False, fixed_fac_scale=False,
                dynamical_scale_choice=[-1], jetalgo=-1., jetradius=0.4, ptj=30., etaj=4.5)
accuracy_conversion = None
if args.backend == 'ampli':
    # MINT's NLO+PS accuracy is relative to the absolute rate, while this
    # AmpliCol planner uses the signed rate. Match nominal absolute targets
    # using the completed reference integration; preserve every other setting.
    reference = sorted((WORK / 'mint/SubProcesses/P0_udx_wpgg').glob('GF*/res_1.dat'))
    if len(reference) != 8:
        raise RuntimeError('Expected eight completed MINT integration channels')
    rates = [[float(x) for x in p.read_text().split()] for p in reference]
    absolute = sum(row[0] for row in rates)
    signed = sum(row[2] for row in rates)
    settings['req_acc'] = 0.01 * absolute / abs(signed)
    accuracy_conversion = dict(
        method='MINT reference ABS/signed ratio; match nominal absolute-rate target',
        mint_req_acc=0.01, mint_absolute_pb=absolute, mint_signed_pb=signed,
        ampli_req_acc=settings['req_acc'], nominal_target_pb=0.01 * absolute,
        reference_files=[str(p.relative_to(WORK)) for p in reference])
    (WORK / 'accuracy_conversion.json').write_text(json.dumps(accuracy_conversion, indent=2) + '\n')
card = RunCardNLO(str(out / 'Cards/run_card.dat'))
for key, value in settings.items():
    card[key] = value
card.write(str(out / 'Cards/run_card.dat'))
fks = (out / 'Cards/FKS_params.dat').read_text()
fks = fks.replace('#NLOPSIntegrator\n0', '#NLOPSIntegrator\n' + str(int(args.backend.startswith('ampli'))))
fks = fks.replace('#UsePolyVirtual\n.False.', '#UsePolyVirtual\n.True.')
(out / 'Cards/FKS_params.dat').write_text(fks)
for name in ('run_card.dat', 'FKS_params.dat', 'param_card.dat'):
    shutil.copy2(out / 'Cards' / name, WORK / (args.backend + '_' + name))
cmd = WORK / (args.backend + '.cmd')
cmd.write_text('set automatic_html_opening False --no_save\n'
               'set notification_center False --no_save\n'
               'set run_mode 2 --no_save\nset nb_core 5 --no_save\n'
               'launch aMC@NLO -f -p --name=benchmark_10k\nquit\n')
metadata = dict(backend=args.backend, settings=settings, cores=5,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (out / 'SubProcesses').iterdir()
                               if p.name in ('mint_module.f90', 'ampli_mint_adapter.f90',
                                             'simple_integrator.f90', 'driver_mintMC.f',
                                             'genps_fks_radiation.f')})
metadata['accuracy_conversion'] = accuracy_conversion
metadata['process'] = 'quark-antiquark > w+ g g Born component of complete p p > w+ j j [QCD]'
metadata['selected_subprocesses'] = (out / 'SubProcesses/subproc.mg').read_text().split()
metadata['source_sha256'].update({str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [out / 'bin/internal/amcatnlo_run_interface.py', out / 'bin/internal/ampli_pool.py']})
(WORK / (args.backend + '_started.json')).write_text(json.dumps(metadata, indent=2) + '\n')
start = time.perf_counter()
with (WORK / (args.backend + '.log')).open('w') as log:
    result = subprocess.run([sys.executable, '-O', str(out / 'bin/aMCatNLO'), str(cmd)],
                            cwd=out, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, 'OMP_NUM_THREADS': '1',
                                 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
metadata.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                wall_seconds=time.perf_counter()-start,
                child_user_cpu_seconds=usage.ru_utime, child_system_cpu_seconds=usage.ru_stime,
                cli_returncode=result.returncode,
                summary_exists=(out / 'Events/benchmark_10k/summary.txt').exists(),
                events_exist=(out / 'Events/benchmark_10k/events.lhe.gz').exists())
(WORK / (args.backend + '_finished.json')).write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata, indent=2), flush=True)
