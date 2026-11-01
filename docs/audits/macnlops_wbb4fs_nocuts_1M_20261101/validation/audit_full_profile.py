#!/usr/bin/env python3
"""Initialize the actual stock continuation card; generate no events."""
from pathlib import Path
import argparse,datetime,hashlib,json,math,os,re,subprocess,sys
BASE=Path(__file__).resolve().parents[1]
ROOT=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS')
PROCESS=BASE/'processes/Wbb_4FS'
PYTHIA=Path('/nfs/home/rikkert/Downloads/pythia8')
sys.path.insert(0,str(ROOT))
from madgraph.various import macnlops

def bind(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())
def values(path):
    out={}
    for line in path.read_text().splitlines():
        line=line.split('!',1)[0].split('#',1)[0].strip()
        if '=' in line:
            key,value=line.split('=',1);out[key.strip()]=value.strip()
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',default='run_1M_wbb4fs_nocuts')
    ap.add_argument('--shower-index',type=int,default=3)
    ap.add_argument('--expected-jobs',type=int,default=50)
    a=ap.parse_args()
    rundir=PROCESS/'MCatNLO'/f'RUN_PYTHIA8_{a.shower_index}'
    cards=sorted(rundir.glob('run_*/Pythia8.cmd'),key=lambda p:int(p.parent.name[4:]))
    assert len(cards)==a.expected_jobs,(len(cards),a.expected_jobs)
    profiles=[]
    for card in cards:
        row=values(card)
        for key in ['Beams:LHEF','Random:seed']:row.pop(key,None)
        profiles.append(row)
    assert all(row==profiles[0] for row in profiles),'Continuation jobs have different physics cards'
    # Read the very same first split input the full driver uses, while restoring
    # its relative pathname with an absolute one for this standalone probe.
    source=(cards[0].parent/values(cards[0])['Beams:LHEF']).resolve()
    assert source.is_file(),source
    prefix='main' if '1M' in a.run else 'pilot'
    output=BASE/'validation'/f'{prefix}_full_effective_settings.cmnd'
    command=[str(BASE/'validation/probe_continuation'),str(cards[0]),str(source),str(output)]
    env=macnlops.native_library_environment(PYTHIA)
    env['PYTHIA8DATA']=str(PYTHIA/'share/Pythia8/xmldoc')
    proc=subprocess.run(command,env=env,capture_output=True,text=True)
    log=BASE/'validation'/f'{prefix}_full_probe.log';log.write_text(proc.stdout+proc.stderr)
    assert proc.returncode==0,(proc.returncode,log)
    actual=json.loads(re.search(r'^AUDITJSON (.*)$',proc.stdout,re.M)[1])
    expected={'SpaceShower:nQuarkIn':4,'TimeShower:nGluonToQuark':4,'StandardModel:alphaSnfmax':5,
        'SpaceShower:alphaSorder':1,'TimeShower:alphaSorder':1,'SpaceShower:alphaSvalue':.118,
        'TimeShower:alphaSvalue':.118,'SpaceShower:alphaSuseCMW':0,'TimeShower:alphaSuseCMW':0,
        'PartonLevel:MPI':0,'HadronLevel:Hadronize':0,'TimeShower:globalRecoil':0,
        'Beams:setProductionScalesFromLHEF':1,'Beams:setDipoleShowerStartingScalesFromLHEF':1,
        '24:mayDecay':0,'SpaceShower:pT0Ref':0.,'SpaceShower:pTmin':.5,'TimeShower:pTmin':.5}
    expected.update({str(pid)+':m0':mass for pid,mass in macnlops.hard_model_masses(PROCESS).items() if str(pid)+':m0' in actual})
    for key,value in expected.items():assert math.isclose(actual[key],value,rel_tol=1e-14,abs_tol=1e-14),(key,actual[key],value)
    settings=json.loads((BASE/'validation'/('setup_audit.json' if prefix=='pilot' else 'setup_audit_'+a.run+'.json')).read_text())
    inputs=cards+[source,output,BASE/'validation/probe_continuation',BASE/'validation/probe_continuation.cc',PYTHIA/'lib/libpythia8.so']
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=a.run,passed=True,
        full_run_directory=rundir.name,all_job_physics_cards_identical=True,jobs=len(cards),
        probe='Read the actual first continuation card, replace only its relative LHE path by the absolute same split input, restore zero pT0Ref as in the full driver, initialize stock Pythia without next(), inspect effective settings/masses.',
        generated_events=0,command=command,actual=actual,expected=expected,
        hard_PDF_alpha_s_at91_1876=settings['pdf']['alpha_s_Q']['91.1876'],
        hard_PDF_alpha_s_at91_188=settings['pdf']['alpha_s_Q']['91.188'],
        first_emission_alpha_s_MZ=.1365,first_emission_alpha_s_order=1,full_alpha_s_MZ=actual['SpaceShower:alphaSvalue'],
        stock_alphaSnfmax=5,inputs={str(p):bind(p) for p in inputs},
        limits='Hard/PDF four-flavour scheme and four physical shower flavours, with stock nfmax5 coupling/trial running. This does not claim the historical patched fully nf4 shower. This initialization check does not test shower distributions.')
    path=BASE/'validation'/f'{prefix}_full_profile_audit.json';path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('PASS',a.run,rundir.name,'all',len(cards),'physics cards identical; actual alpha_s .118, physical species4, stock nfmax5, mb4.7; no generated events')
    print(path)
if __name__=='__main__':main()
