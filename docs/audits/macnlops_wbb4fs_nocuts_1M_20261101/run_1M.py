#!/usr/bin/env python3
"""Fresh native-IR MC@NLO-Delta W+bbbar4FS, no generation/analysis cuts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS')
BASE=Path(__file__).resolve().parent
PROCESS=BASE/'processes/Wbb_4FS'
PYTHIA8='/nfs/home/rikkert/Downloads/pythia8'
LHAPDF='/nfs/home/rikkert/LHAPDF/bin/lhapdf-config'
HEPTOOLS='/export/slow1/rikkert/mg5amcnlo/HEPTools/lib'
CORES=min(32,len(os.sched_getaffinity(0)))
LHAID=334700
P=dict(key='Wbb',analysis='wbb',directory='Wbb_4FS',model='loop_sm',
       process='p p > w+ b b~ QCD=2 QED=1 [QCD]',flavours=4,
       born_partons=2,seed=4010101,generation_cuts='none')
STAGES=('export','cards','generate','diag','shower')
sys.path[:0]=[str(ROOT),str(ROOT/'docs/audits')]
import run_macnlops_campaign as campaign

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)

def mg5(name,exe,cwd,commands):
    log=BASE/'logs'/(name+'.log');cmd=BASE/'commands'/(name+'.cmd')
    cmd.write_text('\n'.join(commands)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MAKEFLAGS='-j1',TMPDIR=str(BASE/'tmp'))
    with log.open('w') as out:
        result=subprocess.run([sys.executable,str(exe),str(cmd)],cwd=cwd,env=env,
                              stdout=out,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} exited {result.returncode}; see {log}')
    # The interactive command interpreter can log an error and still exit zero.
    text=log.read_text(errors='replace')
    if any(marker in text for marker in ['Traceback (most recent call last)',
        'Error detected in','interrupted in sub-command:', 'with error:']):
        raise RuntimeError(f'{name} logged an error; see {log}')

def export(args):
    if (PROCESS/'bin/aMCatNLO').is_file():return
    incoming='g u d s c u~ d~ s~ c~'
    mg5('Wbb_export',ROOT/'bin/mg5_aMC',BASE,[
        'set automatic_html_opening False --no_save','set auto_update 0 --no_save',
        f'set nb_core {CORES} --no_save','set run_mode 2 --no_save',
        f'set ninja {HEPTOOLS} --no_save',f'set collier {HEPTOOLS} --no_save',
        f'set lhapdf {LHAPDF} --no_save',f'set pythia8_path {PYTHIA8} --no_save',
        'import model loop_sm',f'define p = {incoming}',f'define j = {incoming}',
        'generate '+P['process'],f'output {PROCESS} -nojpeg','quit'])
    if not (PROCESS/'bin/aMCatNLO').is_file():raise RuntimeError('Incomplete process export')
    sources=[p for folder in ['Template/NLO/SubProcesses','Template/NLO/MCatNLO/srcPythia8']
             for p in (ROOT/folder).glob('*') if p.is_file()]
    dump(BASE/'validation/source_epoch.json',dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        requested_events=1000000,no_generation_cuts=True,no_analysis_cuts=True,
        process=P,pythia8=PYTHIA8,
        sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        python_sha256={name:sha(ROOT/name) for name in ['madgraph/various/macnlops.py',
                         'madgraph/various/banner.py','madgraph/interface/amcatnlo_run_interface.py']},
        pythia_library_sha256=sha(Path(PYTHIA8)/'lib/libpythia8.so')))

def cards(args):
    from madgraph.various import banner,shower_card
    config=dict(events_per_source=args.events,req_acc=-1,
                nevt_job=min(10000,max(1000,args.events//20)),
                beam_energy_GeV=6500.,lhaids={'4':LHAID})
    seed=P['seed']+args.seed_offset-1
    values=campaign.configure(PROCESS,dict(P,seed=seed),True,config)
    path=PROCESS/'Cards/run_card.dat'
    card=banner.RunCardNLO(str(path));card.remove_all_cut()
    # Remove inactive default cuts as well; none of the cuts are custom.
    for name,value in dict(maxjetflavor=4,ptj=0.,etaj=-1.,ptl=0.,etal=-1.,
                           mll=0.,mll_sf=0.,etagamma=-1.,custom_fcts=[],
                           jetalgo=-1.,jetradius=.5).items():card.set(name,value,user=True)
    card.write(str(path));values.update({name:card[name] for name in card.cuts_parameter})
    params=PROCESS/'SubProcesses/FKS_params.dat'
    text,n=re.subn(r'(?m)(^#DeltaCommonScale\s*\n)[^\n]+',lambda m:m[1]+'.true.',params.read_text())
    if n!=1:raise RuntimeError('Missing DeltaCommonScale')
    params.write_text(text)
    sc=shower_card.ShowerCard(str(PROCESS/'Cards/shower_card.dat'))
    from madgraph.various import macnlops
    continuation_options={'6:mayDecay':'off','24:mayDecay':'off',
        'SpaceShower:nQuarkIn':'4','TimeShower:nGluonToQuark':'4',
        'StandardModel:alphaSnfmax':'5',
        'SpaceShower:alphaSvalue':'0.118','TimeShower:alphaSvalue':'0.118'}
    continuation_options.update({f'{pid}:m0':str(mass) for pid,mass in
                                 macnlops.hard_model_masses(PROCESS).items()})
    nsplit=50 if args.events>=10000 and args.events%50==0 else 4 if args.events%4==0 else 1
    for name,value in dict(nsplit_jobs=nsplit,combine_td=True,h_stable=True,wp_stable=True,
                           pythia8_options=continuation_options,
                           includepaths=str(PROCESS/'MCatNLO/srcCommon'),
                           analyse='HwU.o py8an_HwU_wbb.o wbb_observables.o fjcore.o').items():sc[name]=value
    sc.write(str(PROCESS/'Cards/shower_card.dat'))
    for name in ['py8an_HwU_wbb.f','wbb_observables.cc']:
        source=BASE/'analysis'/name
        if source.is_file():shutil.copy2(source,PROCESS/'MCatNLO/PY8Analyzer'/name)
        else:print('Full-shower analysis source pending:',source,flush=True)
    snapshot=BASE/'cards'/args.run;snapshot.mkdir(parents=True,exist_ok=True)
    for name in ['run_card.dat','param_card.dat','shower_card.dat','amcatnlo_configuration.txt']:
        shutil.copy2(PROCESS/'Cards'/name,snapshot/name)
    shutil.copy2(params,snapshot/params.name)
    dump(snapshot/'settings.json',dict(events=args.events,seed=seed+1,no_generation_cuts=True,
                                      no_analysis_cuts=True,PDF_ID=LHAID,values=values,nsplit_jobs=nsplit))
    print(json.dumps(dict(run=args.run,events=args.events,seed=seed+1,PDF_ID=LHAID,
                         generation_cuts='none',analysis_cuts='none',nsplit_jobs=nsplit)),flush=True)

def generate(args):
    path=PROCESS/'Events'/args.run/'events.lhe.gz'
    if path.is_file():return
    mg5(args.run+'_generate',PROCESS/'bin/aMCatNLO',PROCESS,[f'set nb_core {CORES}',
        'set run_mode 2',f'launch aMC@NLO --parton -f -n {args.run}'])
    if not path.is_file():raise RuntimeError('No collected event file')

def diag(args):
    source=PROCESS/'Events'/args.run/'events.lhe.gz'
    backup=BASE/'backup'/(args.run+'_events.lhe.gz')
    if backup.is_file():return
    subprocess.run([sys.executable,str(BASE/'analysis/diag_weights.py'),str(source),str(backup),
                    str(BASE/'results'/(args.run+'_lhe.npz')),
                    str(BASE/'results'/(args.run+'_lhe.json')),'Wbb'],check=True,cwd=BASE)

def shower(args):
    mg5(args.run+'_shower',PROCESS/'bin/aMCatNLO',PROCESS,[f'set nb_core {CORES}',
        'set run_mode 2',f'shower {args.run} -f'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events',type=int,default=1000000)
    parser.add_argument('--run',default='run_1M_wbb4fs_nocuts')
    parser.add_argument('--seed-offset',type=int,default=10000)
    parser.add_argument('stages',nargs='*',default=list(STAGES))
    args=parser.parse_args()
    if set(args.stages)-set(STAGES):parser.error('Unknown stage')
    for stage in STAGES:
        if stage in args.stages:
            print('==',args.run,stage,flush=True);globals()[stage](args)

if __name__=='__main__':main()
