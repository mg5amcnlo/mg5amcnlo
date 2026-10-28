#!/usr/bin/env python3
"""Replay fixed Born ensembles with recovered historical SudGen settings.

Usage: singlet_matched_showers.py ORIGINAL_WORK NEW_WORK PYTHIA_INSTALL OUTPUT
Run PDF13 (historical default) and PDF15 controls, all fixed/damped samples.
The original driver, Born files and old results are never modified.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from singlet_run_showers import ROOT, sha
from singlet_pdf13_control import settings

HISTORICAL = {
    'SpaceShower:alphaSvalue': '.130', 'TimeShower:alphaSvalue': '.1365',
    'SpaceShower:alphaSorder': '1', 'TimeShower:alphaSorder': '1',
    'SpaceShower:alphaSuseCMW': 'off', 'TimeShower:alphaSuseCMW': 'off',
    'SpaceShower:renormMultFac': '1.', 'TimeShower:renormMultFac': '1.',
    'SpaceShower:factorMultFac': '1.', 'TimeShower:factorMultFac': '1.',
    'StandardModel:alphaSnfmax': '6', 'SpaceShower:nQuarkIn': '5',
    'SpaceShower:samePTasMPI': 'off', 'SpaceShower:ecmPow': '0.',
    'SpaceShower:ecmRef': '7000.',
    'SpaceShower:phiIntAsym': 'off', 'SpaceShower:phiPolAsym': 'off',
    'SpaceShower:phiPolAsymHard': 'off', 'TimeShower:phiPolAsym': 'off',
    'TimeShower:phiPolAsymHard': 'off',
    'TimeShower:dampenBeamRecoil': 'on', 'TimeShower:globalRecoil': 'off',
    'TimeShower:nMaxGlobalRecoil': '2', 'TimeShower:globalRecoilMode': '0',
    'TimeShower:nMaxGlobalBranch': '-1', 'TimeShower:nPartonsInBorn': '-1',
    'TimeShower:limitPTmaxGlobal': 'off', 'TimeShower:weightGluonToQuark': '4',
    'SpaceShower:MEcorrections': 'off', 'TimeShower:MEcorrections': 'off',
    'SpaceShower:rapidityOrder': 'off',
}
MASSES = {1: 0., 2: 0., 3: 0., 4: 0., 5: 4.8}
SAMPLES = ['dy_fixed', 'dy_damped', 'higgs_fixed', 'higgs_damped']


def equal_value(actual, expected):
    if expected in ['on', 'off']:
        return actual == expected
    return float(actual) == float(expected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('original', type=Path); p.add_argument('work', type=Path)
    p.add_argument('pythia', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--pdf', type=int, nargs='+', choices=[13,15], default=[13,15])
    p.add_argument('--reuse', action='store_true')
    args = p.parse_args()
    original, work, install, out = [x.resolve() for x in [args.original,args.work,args.pythia,args.output]]
    work.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    source_template = ROOT/'tests/validation/delta_noemission/singlet_shower_probe.cc'
    prior_file = original/'shower_provenance.json'
    prior = json.loads(prior_file.read_text())
    assert sha(source_template)==prior['source_sha256']
    template = source_template.read_text()
    source_copy = work/'singlet_shower_probe_original.cc'; source_copy.write_text(template)
    settings_audit = ROOT/'docs/audits/delta_noemission/singlet_matched/settings_audit.json'
    audited_expected = json.loads(settings_audit.read_text())['rerun_expected_shared_settings']
    started = time.monotonic()
    jobs, builds, pilots = [], {}, []
    for pdf in args.pdf:
        directory = work/f'pdf{pdf}'; directory.mkdir(exist_ok=True)
        destination = out/f'pdf{pdf}'; destination.mkdir(exist_ok=True)
        override = dict(HISTORICAL, **{'PDF:pSet':str(pdf)})
        overrides = '\n'.join('  setting(p,"'+k+' = '+v+'");' for k,v in override.items())
        overrides += '\n'+'\n'.join(f'  setting(p,"{i}:m0 = {m}");' for i,m in MASSES.items())
        overrides += '\n  p.settings.forceParm("SpaceShower:pT0Ref",0.);\n'
        needle = '  setting(p,"Random:seed = "+std::to_string(seed));'
        assert template.count(needle)==1
        source_text = template.replace(needle, overrides+needle)
        needle = '    p.settings.writeFile(prefix+".settings.cmnd",true);'
        assert source_text.count(needle)==1
        snapshot = r'''    {
      std::ofstream particle(prefix+".particle_masses.json");
      particle << std::setprecision(17) << "{";
      for(int id=1;id<=6;++id)particle<<(id>1?",":"")<<"\""<<id<<"\":"<<p.particleData.m0(id);
      particle<<"}\n";
      if(p.settings.parm("SpaceShower:pT0Ref")!=0.)throw std::runtime_error("pT0 not zero");
      for(int id=1;id<=4;++id)if(p.particleData.m0(id)!=0.)throw std::runtime_error("Historical mass not zero");
    }
'''
        source_text = source_text.replace(needle, snapshot+needle)
        source = directory/'singlet_matched_probe.cc'
        if args.reuse and source.exists():
            assert source.read_text()==source_text, 'Cannot reuse trials with changed physics source'
        source.write_text(source_text)
        exe = directory/'singlet_matched_probe'
        build = ['c++','-O2','-std=c++17',str(source),'-I'+str(install/'include'),
                 '-L'+str(install/'lib'),'-Wl,-rpath,'+str(install/'lib'),'-lpythia8','-o',str(exe)]
        subprocess.run(build,check=True)
        builds[str(pdf)] = dict(command=build,source=str(source),source_sha256=sha(source),executable_sha256=sha(exe))
        for old in prior['jobs']:
            name = old['sample']; assert name in SAMPLES
            lhe = Path(old['command'][3]); assert sha(lhe)==old['input_lhe_sha256']
            prefix = directory/f'{name}_shard{old["shard"]:02d}'
            repeats = old['audit']['repeats']
            command = [str(exe),str(install/'share/Pythia8/xmldoc'),'shower',str(lhe),str(prefix),str(repeats),str(old['seed'])]
            jobs.append(dict(pdf=pdf,sample=name,shard=old['shard'],first=old['first'],last=old['last'],
                prefix=str(prefix),original_prefix=old['prefix'],seed=old['seed'],repeats=repeats,
                command=command,input_lhe_sha256=sha(lhe)))
        for name in ['dy_fixed','higgs_fixed']:
            old = next(j for j in prior['jobs'] if j['sample']==name and j['shard']==0)
            text = Path(old['command'][3]).read_text(); events=list(re.finditer(r'<event>.*?</event>',text,re.S))
            lhe = directory/(name+'_pilot.lhe')
            lhe.write_text(text[:events[0].start()]+'\n'.join(e.group() for e in events[:10])+text[events[-1].end():])
            prefix = directory/(name+'_pilot')
            command=[str(exe),str(install/'share/Pythia8/xmldoc'),'shower',str(lhe),str(prefix),'30',str(old['seed'])]
            pilots.append(dict(pdf=pdf,sample=name,first=0,last=10,prefix=str(prefix),original_prefix=old['prefix'],
                seed=old['seed'],repeats=30,command=command,input_lhe_sha256=sha(lhe)))

    def run(job, allow_reuse=False):
        log = Path(job['prefix']+'.log')
        if not (allow_reuse and args.reuse and log.exists() and 'failures=0' in log.read_text()):
            with log.open('w') as stream:
                subprocess.run(job['command'],stdout=stream,stderr=subprocess.STDOUT,check=True)
        text = log.read_text()
        assert 'PROOF SimpleShowerModel=1 SimpleSpaceShower=1 SimpleTimeShower=1' in text
        match=re.search(r'AUDIT process_checks=(\d+) maximum_relative_Born_replay_difference=(\S+) emitted=(\d+) Born=(\d+) repeats=(\d+) failures=(\d+)',text)
        assert match,text[-3000:]
        checks,difference,emitted,born,repeats,failures=match.groups()
        assert int(born)==job['last']-job['first'] and int(repeats)==job['repeats'] and int(failures)==0
        assert int(checks)==int(born)*int(repeats) and float(difference)<1e-10
        old,new=settings(job['original_prefix']+'.settings.cmnd'),settings(job['prefix']+'.settings.cmnd')
        expected=dict(audited_expected,**{'PDF:pSet':str(job['pdf']),'SpaceShower:pT0Ref':'0.'})
        for k,v in expected.items(): assert equal_value(new[k],v),(k,new[k],v)
        assert new['SpaceShower:pdfMode']==new['TimeShower:pdfMode']=='0'
        assert new['SpaceShower:pTmaxMatch']=='1'
        diff={k:[old.get(k),new.get(k)] for k in sorted(old.keys()|new.keys()) if old.get(k)!=new.get(k)}
        assert set(diff).issubset(set(expected)|{'SpaceShower:pTmin'}),diff
        masses=json.loads(Path(job['prefix']+'.particle_masses.json').read_text())
        assert all(masses[str(i)]==m for i,m in MASSES.items()),masses
        messages=Counter()
        for n,msg in re.findall(r'^\s*\|\s*(\d+)\s+((?:Warning|Error|Abort).*?)\s*\|',text,re.M):messages[msg]+=int(n)
        errors = {msg:n for msg,n in messages.items() if msg.startswith(('Error','Abort'))}
        # A zero/negative gluon PDF can make the NLO PDF15 heavy-threshold
        # evolution return below sqrt(2)*mQ. Preserve and flag this diagnostic;
        # never describe that control as error-free. PDF13 must remain clean.
        allowed = {'Error in SimpleSpaceShower::pT2nearThreshold: xPDF = 0'}
        assert not errors or (job['pdf']==15 and set(errors)<=allowed), errors
        job['audit']=dict(process_checks=int(checks),maximum_relative_Born_replay_difference=float(difference),
            emitted=int(emitted),born=int(born),repeats=int(repeats),failures=int(failures),messages=dict(messages),logged_errors=errors,
            settings_differences=diff,particle_masses=masses,configured_ISR_pTmin=float(new['SpaceShower:pTmin']),
            log_sha256=sha(log),settings_sha256=sha(job['prefix']+'.settings.cmnd'))
        return job

    for job in pilots:
        run(job, True)
        print('PILOT passed:',job['pdf'],job['sample'],job['audit']['messages'],flush=True)
    (work/'pilot_provenance.json').write_text(json.dumps(pilots,indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,j,True) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            j=f.result();print(f'{i}/{len(jobs)} completed: PDF{j["pdf"]} {j["sample"]} shard{j["shard"]}, {time.monotonic()-started:.1f}s',flush=True)
    samples={}
    for pdf in args.pdf:
        directory=work/f'pdf{pdf}';destination=out/f'pdf{pdf}'
        samples[str(pdf)]={}
        for name in SAMPLES:
            prefix=directory/name
            born0=[json.loads(l) for l in (original/(name+'.born.jsonl')).open()]
            chosen=sorted([j for j in jobs if j['pdf']==pdf and j['sample']==name],key=lambda j:j['shard'])
            borns=[];ntrial=0;emitted=0;messages=Counter()
            with Path(str(prefix)+'.born.jsonl').open('w') as bo,Path(str(prefix)+'.trials.jsonl').open('w') as tr:
                for j in chosen:
                    messages.update(j['audit']['messages'])
                    for line in Path(j['prefix']+'.born.jsonl').open():
                        b=json.loads(line);b['event']+=j['first'];assert b['event']==len(borns) and b==born0[b['event']]
                        borns.append(b);bo.write(json.dumps(b,separators=(',',':'))+'\n')
                    for line in Path(j['prefix']+'.trials.jsonl').open():
                        t=json.loads(line);t['event']+=j['first'];assert (t['event'],t['trial'])==divmod(ntrial,j['repeats']) and t['ok']
                        assert 0<=t['pt']<=borns[t['event']]['scalup']*(1+1e-10)
                        assert (t['type']==2 and t['pt']>0) or (t['type']==0 and t['pt']==0)
                        emitted+=t['pt']>0;ntrial+=1;tr.write(json.dumps(t,separators=(',',':'))+'\n')
            assert len(borns)==len(born0) and ntrial==prior['samples'][name]['trials']
            assert sha(str(prefix)+'.born.jsonl')==sha(original/(name+'.born.jsonl'))
            for suffix in ['settings.cmnd','particle_masses.json']:
                shutil.copyfile(chosen[0]['prefix']+'.'+suffix,str(prefix)+'.'+suffix)
                shutil.copyfile(str(prefix)+'.'+suffix,destination/(name+'.'+suffix))
            samples[str(pdf)][name]=dict(born=len(borns),trials=ntrial,emitting=emitted,no_emission=ntrial-emitted,
                original_born_equal_bitwise=True,messages=dict(messages),
                maximum_relative_Born_replay_difference=max(j['audit']['maximum_relative_Born_replay_difference'] for j in chosen),
                files={str(prefix)+'.'+suffix:sha(str(prefix)+'.'+suffix) for suffix in ['born.jsonl','trials.jsonl','settings.cmnd','particle_masses.json']})
        pdfprov=dict(pdf=pdf,samples=samples[str(pdf)],jobs=[j for j in jobs if j['pdf']==pdf])
        (directory/'shower_provenance.json').write_text(json.dumps(pdfprov,indent=2)+'\n')
    result=dict(pythia_version='8.313',samples=samples,jobs=jobs,pilots=pilots,builds=builds,
        definition='Replay existing 1000 Born states per process, fixed/damped starts, 300 repetitions; recovered historical settings; PDF13 primary and PDF15 control.',
        historical_settings=HISTORICAL,forced_settings={'SpaceShower:pT0Ref':0.},particle_masses=MASSES,
        intentional_differences=['Physical backward pdfMode=0 versus forward archive mode',
            'External Born process and per-event starts retained; pTmaxMatch=1 enforces them',
            'Beams production-scale flags stay on to read identical commonQ starts; historical dummy generator uses internal starts',
            'QED off: dedicated historical ISR Sudakov routine trials QCD only',
            'Hadronization, MPI and primordial kT disabled; first QCD emission observable',
            'Pythia8.313 version explicitly authorized; historical generation executable unrecorded'],
        source_sha256={str(p):sha(p) for p in [source_template,source_copy,Path(__file__),settings_audit]},
        library_sha256=sha(install/'lib/libpythia8.so'),prior_provenance_sha256=sha(prior_file),
        pdf_grid_sha256={str(pdf):sha(install/'share/Pythia8/pdfdata'/grid) for pdf,grid in
            [(13,'NNPDF23_lo_as_0130_qed_0000.dat'),(15,'NNPDF23_nlo_as_0119_qed_0000.dat')]},
        elapsed_seconds=time.monotonic()-started,reused_existing_shards=args.reuse,
        internal_ISR_cutoff=json.loads(settings_audit.read_text())['internal_ISR_cutoff'],
        pdf15_error_qualification='Logged xPDF=0 errors are retained and counted, never silently discarded. '
            'The source invokes this heavy-threshold routine only below sqrt(2)*mQ, bounded by6.7883GeV '
            'for bottom and1.6971GeV for charm in this setup. Values atlowercuts remainqualified; '
            'all events succeed. The primary PDF13 results must contain no Error/Abort diagnostics.',
        checks=['All Born JSON files match original Born files byte for byte',
            'Every shard verifies explicit historical settings and zero input masses1-4 after initialization',
            'Every trial successful; first accepted QCD emission ISR and below start',
            'All process-level Born replay audits pass; Simple classes checked at runtime',
            'All repetitions included in reference CDF; no emission-conditioned selection'])
    for path in [work/'shower_provenance.json',out/'shower_provenance.json']:
        path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(samples,indent=2),flush=True)

if __name__=='__main__':main()
