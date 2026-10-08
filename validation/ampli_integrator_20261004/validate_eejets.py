"""Standalone one-channel backend smoke test.

Generation is recorded in generate.cmd. This script deliberately supplies a
synthetic two-channel normalization (twice the tested channel rate) at stage2;
it checks the event path, not a physical total cross section or coordinator.
Run: python validate_eejets.py --born --channel 2.0
"""
from pathlib import Path
import argparse
import subprocess
import shutil
import xml.etree.ElementTree as ET
import sys

ROOT=Path('/export/tmp/rikkert/git/mg5amcnlo/MCcntRefactor_Sfun_Granny')
sys.path.insert(0,str(ROOT))
from madgraph.various import banner
BASE=Path(__file__).resolve().parent
OUTPUT=BASE/'eejets'
PDIR=OUTPUT/'SubProcesses/P0_epem_uux'

parser=argparse.ArgumentParser()
parser.add_argument('--born',action='store_true')
parser.add_argument('--channel',default='1.0')
parser.add_argument('--stages',default='0,1,2')
parser.add_argument('--split',type=int,default=0)
parser.add_argument('--legacy-matching',action='store_true')
parser.add_argument('--integrator',type=int,choices=(0,1),default=1)
args=parser.parse_args()
job=PDIR/('GF'+args.channel+('_%d'%args.split if args.split else ''))
job.mkdir(exist_ok=True)
label=('mint' if args.integrator==0 else 'ampli')+('_born' if args.born else '_plain')+('_legacy' if args.legacy_matching else '')
card_path=job/'FKS_params.dat'
if card_path.is_symlink(): card_path.unlink()
fks=(OUTPUT/'Cards/FKS_params.dat').read_text().replace('#NLOPSIntegrator\n1\n','#NLOPSIntegrator\n%d\n'%args.integrator)
if args.legacy_matching: fks=fks.replace('#MCExplicitKLSum\n.true.\n','#MCExplicitKLSum\n.false.\n')
card_path.write_text(fks)
for name in ('driver_mintMC.f','mint_module.f90','FKSParams.f90','makefile_fks_dir','integrator_helpers.f90','simple_integrator.f90','ampli_mint_adapter.f90','genps_fks_radiation.f'):
    shutil.copy2(ROOT/'Template/NLO/SubProcesses'/name,OUTPUT/'SubProcesses'/name)

card=banner.RunCardNLO(str(OUTPUT/'Cards/run_card.dat'))
card['born_spreading']=args.born
card.write(str(OUTPUT/'Cards/run_card.dat'))
commands=[('treatcards',[str(OUTPUT/'bin/aMCatNLO'),'treatcards'],OUTPUT),
          ('source',['make','-j4'],OUTPUT/'Source'),
          ('mintmc',['make','-j4','madevent_mintMC'],PDIR)]
for name,command,cwd in commands:
    path=BASE/('eejets_'+label+'_'+name+'.log')
    with path.open('w') as log:
        result=subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
    print(name,result.returncode,path,flush=True)
    if result.returncode:
        print(path.read_text()[-6000:],flush=True)
        raise SystemExit(result.returncode)

for stage in map(int,args.stages.split(',')):
    folding='1 1 1' if stage==0 else '2 2 2'
    text='1024 4\n0.2\n1 -0.1\n-1 -0.1\n1\n1\n%s\n%d\n%s\nall\n' % (args.channel,stage,folding)
    if stage==2:
        a,ea,s,es=map(float,(job/'res_1.dat').read_text().split()[:4])
        text+='0 10\n20 %.16e %.16e %.16e %.16e\n1\n0 %.16e %.16e %.16e %.16e\n' % (2*a,2*ea,2*s,2*es,2*a,2*ea,2*s,2*es)
    (job/'input_app.txt').write_text(text)
    (BASE/('eejets_'+label+'_stage%d.input'%stage)).write_text(text)
    path=BASE/('eejets_'+label+'_stage%d.log'%stage)
    with path.open('w') as log:
        result=subprocess.run(['bash','ajob1',args.channel,'F',str(args.split),str(stage)],cwd=PDIR,stdout=log,stderr=subprocess.STDOUT)
    print('stage',stage,'status',result.returncode,'log',job/('log_MINT%d.txt'%stage),flush=True)
    if result.returncode:
        print((job/('log_MINT%d.txt'%stage)).read_text()[-6000:],flush=True)
        raise SystemExit(result.returncode)
    if stage<2:
        if not (job/('res_%d.dat'%stage)).exists():
            raise RuntimeError('Fortran stopped without producing stage result')
        print((job/('res_%d.dat'%stage)).read_text(),flush=True)
    else:
        events=ET.fromstring((job/'events.lhe').read_text()).findall('event')
        weights=[float(event.text.split()[2]) for event in events]
        assert len(events)==10,(len(events),weights)
        assert all(abs(abs(weight)-2*a)<1e-7*max(1,2*a) for weight in weights),weights
        print('events',len(events),'weights',weights,flush=True)
