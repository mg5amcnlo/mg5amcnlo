#!/usr/bin/env python3
"""Fresh native-t_c validation: export/cards/generate/diag/shower/analyse.

Each stage is explicit. Existing exports and collected events are adopted;
cards never change implicitly during generation or first-emission analysis.
"""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_BASE = Path('/export/slow1/rikkert/macnlops_native_tc_1M_20261101')
PYTHIA = Path('/nfs/home/rikkert/Downloads/pythia8')
LHAPDF = '/nfs/home/rikkert/LHAPDF/bin/lhapdf-config'
HEPTOOLS = '/export/slow1/rikkert/mg5amcnlo/HEPTools/lib'
sys.path[:0] = [str(ROOT), str(ROOT/'docs/audits')]
import run_macnlops_campaign as campaign

PROCESSES = [
    dict(key='EE2j', directory='EE2Jets_5FS', model='loop_sm-no_b_mass',
         process='e+ e- > j j QCD=0 QED=2 [QCD]', flavours=5, born_partons=2, seed=4320101, code=6),
    dict(key='DY', directory='DY_5FS', model='loop_sm-no_b_mass',
         process='p p > e+ e- [QCD]', flavours=5, born_partons=0, seed=4320201, code=1),
    dict(key='H', directory='Higgs_HEFT_5FS',
         model='/export/slow1/rikkert/mg5amcnlo/models/HC_NLO_X0_UFO-heft',
         process='p p > x0 / t QNP=1 QCD=2 QED=0 [QCD]', flavours=5, born_partons=0, seed=4320301, code=2),
    dict(key='ttbar', directory='TTbar_5FS', model='loop_sm-no_b_mass',
         process='p p > t t~ [QCD]', flavours=5, born_partons=2, seed=4320401, code=3),
    dict(key='Wj', directory='WPlusJet_5FS', model='loop_sm-no_b_mass',
         process='p p > w+ j QCD=1 QED=1 [QCD]', flavours=5, born_partons=1, seed=4320501, code=5,
         generation_cuts='wjet'),
    dict(key='EE3j', directory='EE3Jets_5FS', model='loop_sm-no_b_mass',
         process='e+ e- > j j j QCD=1 QED=2 [QCD]', flavours=5, born_partons=3, seed=4320601, code=7),
    dict(key='EE4j', directory='EE4Jets_5FS', model='loop_sm-no_b_mass',
         process='e+ e- > j j j j QCD=2 QED=2 [QCD]', flavours=5, born_partons=4, seed=4320701, code=8),
    dict(key='Wbb', directory='Wbb_4FS', model='loop_sm',
         process='p p > w+ b b~ QCD=2 QED=1 [QCD]', flavours=4, born_partons=2, seed=4320801, code=4,
         generation_cuts='none'),
]
STAGES = ('export', 'cards', 'generate', 'diag', 'shower', 'analyse')
CORE = [*campaign.PHYSICS_FILES,
        'Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc',
        'Template/NLO/SubProcesses/mcatnlo_delta_scales.f90',
        'Template/NLO/SubProcesses/macnlops_capi.f',
        'Template/NLO/SubProcesses/weight_lines.f']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def epoch():
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        git_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=ROOT)).hexdigest(),
        source_sha256={name:sha(ROOT/name) for name in CORE if (ROOT/name).is_file()},
        workflow_sha256={str(p.relative_to(HERE)):sha(p) for p in HERE.glob('*') if p.is_file()},
        pythia_libraries_sha256={p.name:sha(p) for p in (PYTHIA/'lib').glob('*.so')},
        pythia_modified=False, source='delta_native_tc',
        source_construction='mcatnlo-delta-native-tc-v1')


def mg5(a, p, name, commands, export=False):
    process = a.base/'processes'/p['directory']
    cmd = a.base/'commands'/(p['key']+'_'+name+'.cmd')
    log = a.base/'logs'/(p['key']+'_'+name+'.log')
    cmd.write_text('\n'.join(commands)+'\n')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MAKEFLAGS='-j1', TMPDIR=str(a.base/'tmp'))
    exe, cwd = (ROOT/'bin/mg5_aMC',a.base) if export else (process/'bin/aMCatNLO',process)
    with log.open('w') as output:
        result=subprocess.run([sys.executable,str(exe),str(cmd)],cwd=cwd,env=env,
                              stdout=output,stderr=subprocess.STDOUT)
    content=log.read_text(errors='replace')
    if result.returncode or any(s in content for s in (
        'Traceback (most recent call last)', 'Error detected in', 'interrupted in sub-command:', 'with error:')):
        raise RuntimeError(f'{name} failed; see {log}')


def export(a,p):
    process=a.base/'processes'/p['directory']
    if not (process/'bin/aMCatNLO').is_file():
        incoming='g u d s c '+('b ' if p['flavours']==5 else '')+'u~ d~ s~ c~'+(' b~' if p['flavours']==5 else '')
        mg5(a,p,'export',[
            'set automatic_html_opening False --no_save','set auto_update 0 --no_save',
            f'set nb_core {a.cores} --no_save','set run_mode 2 --no_save',
            f'set ninja {HEPTOOLS} --no_save',f'set collier {HEPTOOLS} --no_save',
            f'set lhapdf {LHAPDF} --no_save',f'set pythia8_path {PYTHIA} --no_save',
            'import model '+p['model'],'define p = '+incoming,'define j = '+incoming,
            'generate '+p['process'],f'output {process} -nojpeg','quit'],export=True)
    if not (process/'bin/aMCatNLO').is_file(): raise RuntimeError('Incomplete export')
    target=a.base/'validation'/p['key']/'export_epoch.json'
    if not target.exists(): dump(target,dict(epoch(),process=p,adopted_existing_export=True))


def cards(a,p):
    from madgraph.various import banner
    process=a.base/'processes'/p['directory']
    if (process/'Events'/a.run/'events.lhe.gz').exists():
        raise RuntimeError('Collected run already exists; use another --run before changing cards')
    ee=p['key'].startswith('EE')
    config=dict(events_per_source=a.events,req_acc=-1,
        nevt_job=min(10000,max(1000,a.events//20)),beam_energy_GeV=45.594 if ee else 6500.,
        lhaids={'5':331700,'4':334700})
    campaign.configure(process,dict(p,seed=a.seed-1 if a.seed is not None else p['seed']-1),True,config)
    path=process/'Cards/run_card.dat'; card=banner.RunCardNLO(str(path))
    if ee:
        card.remove_all_cut()
        values=dict(lpp1=0,lpp2=0,pdlabel='nn23nlo',fixed_ren_scale=True,fixed_fac_scale=True,
            fixed_qes_scale=True,mur_ref_fixed=91.188,muf_ref_fixed=91.188,
            qes_ref_fixed=91.188,dynamical_scale_choice=[-2],custom_fcts=[])
        if p['born_partons']>2:
            cut=process/'Cards'/f'durham_{p["born_partons"]}j_cuts.f'
            from specification import durham_cut
            cut.write_text(durham_cut(p['born_partons']))
            values['custom_fcts']=[str(cut)]
    elif p['key']=='Wj': values=dict(ptj=20.,etaj=-1.,jetalgo=-1.,jetradius=.5)
    elif p['key']=='Wbb': values=dict(custom_fcts=[],maxjetflavor=4)
    else: values={}
    for name,value in values.items():card.set(name,value,user=True)
    card.write(str(path))
    fks=process/'SubProcesses/FKS_params.dat'
    text,n=re.subn(r'(?m)(^#DeltaCommonScale\s*\n)[^\n]+',lambda m:m[1]+'.true.',fks.read_text())
    if n!=1: raise RuntimeError('Missing DeltaCommonScale')
    fks.write_text(text)
    install_analysis(a,p)
    snapshot=a.base/'cards'/p['key']/a.run;snapshot.mkdir(parents=True,exist_ok=True)
    for source in (process/'Cards').glob('*'):
        if source.is_file() and source.suffix in ('.dat','.txt','.f'):shutil.copy2(source,snapshot/source.name)
    shutil.copy2(fks,snapshot/fks.name)
    dump(snapshot/'settings.json',dict(epoch(),process=p,events=a.events,run=a.run,
        seed=a.seed if a.seed is not None else p['seed'],cores=a.cores,
        cards_sha256={x.name:sha(x) for x in snapshot.iterdir() if x.is_file()}))
    print('Cards ready:',snapshot,flush=True)


def install_analysis(a,p):
    """Only shower/analysis cards change here; run/FKS/param cards stay frozen."""
    from madgraph.various import shower_card,macnlops
    from specification import fortran_hook
    process=a.base/'processes'/p['directory']
    target=process/'MCatNLO/PY8Analyzer'
    target.mkdir(exist_ok=True)
    (target/'py8an_HwU_native_tc.f').write_text(fortran_hook(p['code']))
    helper=HERE/'observables.cc'
    if helper.is_file():shutil.copy2(helper,target/'native_tc_observables.cc')
    # Remove stale analysis objects when adopting a pilot prepared before this hook.
    for stem in ('py8an_HwU_native_tc','native_tc_observables'):
        (target/(stem+'.o')).unlink(missing_ok=True)
    options={'6:mayDecay':'off','24:mayDecay':'off','25:mayDecay':'off',
        'SpaceShower:nQuarkIn':str(p['flavours']), 'TimeShower:nGluonToQuark':str(p['flavours']),
        'StandardModel:alphaSnfmax':'5','SpaceShower:alphaSvalue':'0.118',
        'TimeShower:alphaSvalue':'0.118','SpaceShower:alphaSorder':'1',
        'TimeShower:alphaSorder':'1','PDF:lepton':'off'}
    options.update({f'{pid}:m0':str(mass) for pid,mass in macnlops.hard_model_masses(process).items()})
    sc=shower_card.ShowerCard(str(process/'Cards/shower_card.dat'))
    nsplit=50 if a.events>=10000 and a.events%50==0 else 4 if a.events%4==0 else 1
    for name,value in dict(hadronize=False,ue_enabled=False,primordialkt=False,qed_shower=False,
        nevents=-1,nsplit_jobs=nsplit,pdfcode=1,combine_td=True,h_stable=True,wp_stable=True,
        pythia8_options=options,includepaths=str(process/'MCatNLO/srcCommon'),
        analyse='HwU.o py8an_HwU_native_tc.o native_tc_observables.o fjcore.o').items():sc[name]=value
    sc.write(str(process/'Cards/shower_card.dat'))


def generate(a,p):
    process=a.base/'processes'/p['directory']
    source=process/'Events'/a.run/'events.lhe.gz'
    if source.is_file():return
    proof=epoch();dump(a.base/'validation'/p['key']/(a.run+'_generation_epoch.json'),proof)
    mg5(a,p,a.run+'_generate',[f'set nb_core {a.cores}','set run_mode 2',f'launch aMC@NLO --parton -f -n {a.run}'])
    if not source.is_file():raise RuntimeError('Missing collected event file')


def diag(a,p):
    from histogram_analysis import diagnose
    diagnose(a,p)


def shower(a,p):
    process=a.base/'processes'/p['directory']
    if not (a.base/'results'/p['key']/a.run/'first.json').is_file():
        raise RuntimeError('Run diag before full shower; diagnostic weight columns are required')
    install_analysis(a,p)
    if not (process/'MCatNLO/PY8Analyzer/native_tc_observables.cc').is_file():
        raise RuntimeError('Observable helper is missing')
    dump(a.base/'validation'/p['key']/(a.run+'_shower_epoch.json'),dict(epoch(),
        shower_card_sha256=sha(process/'Cards/shower_card.dat')))
    mg5(a,p,a.run+'_shower',[f'set nb_core {a.cores}','set run_mode 2',f'shower {a.run} -f'])


def analyse(a,p):
    from histogram_analysis import analyse_full
    analyse_full(a,p)
    from audit import audit
    from report import report
    audit(a.base,p,a.run,a.events)
    report(a.base,p,a.run)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--process',nargs='+',choices=[p['key'] for p in PROCESSES],default=['EE2j','DY'])
    parser.add_argument('--events',type=int,default=1000000)
    parser.add_argument('--run',default='run_1M_native_tc')
    parser.add_argument('--cores',type=int,default=min(32,len(os.sched_getaffinity(0))))
    parser.add_argument('--seed',type=int)
    parser.add_argument('stages',nargs='*',choices=STAGES,default=list(STAGES))
    a=parser.parse_args();a.base=a.base.resolve()
    if a.events<1 or a.cores<1:parser.error('events and cores must be positive')
    if a.seed is not None and len(a.process)>1:parser.error('--seed requires one process')
    for folder in ('commands','logs','cards','tmp','processes','validation','results','build'):
        (a.base/folder).mkdir(parents=True,exist_ok=True)
    for p in PROCESSES:
        if p['key'] not in a.process:continue
        for stage in STAGES:
            if stage in a.stages:
                print(p['key'],a.run,stage,flush=True);globals()[stage](a,p)


if __name__=='__main__':main()
