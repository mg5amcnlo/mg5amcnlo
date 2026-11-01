#!/usr/bin/env python3
"""Isolated native source-prefix controls; never mutate source campaigns."""
from pathlib import Path
import concurrent.futures,datetime,hashlib,itertools,json,shutil,subprocess,sys
ROOT=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS');sys.path.insert(0,str(ROOT))
from madgraph.various import macnlops
from madgraph.various.banner import RunCardNLO
BASE=Path(__file__).resolve().parent;FROZEN=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101/processes/Wbb_4FS')
CAMPAIGNS=Path('/export/slow1/rikkert')
SPECS=[('DY','macnlops_native_ir_1M_20261031','DY_5FS','run_1M_native_ir',250),('H_HEFT','macnlops_native_ir_1M_20261031','Higgs_HEFT_5FS','run_1M_native_ir',500),('TTbar','macnlops_native_ir_1M_20261031','TTbar_5FS','run_1M_native_ir',250),('EE2','macnlops_native_ir_jets_1M_20261031','EE2Jets_5FS','run_1M_native_ir_jets',300),('EE3','macnlops_native_ir_jets_1M_20261031','EE3Jets_5FS','run_1M_native_ir_jets',300),('EE4','macnlops_native_ir_jets_1M_20261031','EE4Jets_5FS','run_1M_native_ir_jets',150),('Wj','macnlops_native_ir_jets_1M_20261031','WPlusJet_5FS','run_1M_native_ir_jets',300),('Wbb','macnlops_wbb4fs_1M_nocuts_20261101','Wbb_4FS','run_1M_wbb4fs_nocuts',500)]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def prefix(source,target,count):
 with macnlops.LHEStream(source) as stream:
  blocks=list(itertools.islice(stream,count));header=stream.header
 assert blocks
 target.parent.mkdir(parents=True,exist_ok=True);target.write_text(header+''.join(blocks)+'</LesHouchesEvents>\n')
 return len(blocks)
def prepare(spec):
 key,campaign,directory,run,cap=spec;original=CAMPAIGNS/campaign/'processes'/directory;clone=BASE/'processes'/key;summary=original/'Events'/run/'macnlops_first_emission.json';record=json.loads(summary.read_text());groups={}
 for channel in record['channels']:
  p=Path(channel['directory']);groups.setdefault(p.parent.name,{})
  group=p.name.split('_')[0]
  if group not in groups[p.parent.name] and len(groups[p.parent.name])<2:groups[p.parent.name][group]=channel
 clone.parent.mkdir(parents=True,exist_ok=True)
 if not clone.exists():subprocess.run(['rsync','-aL','--exclude=OLE_order.olc','--exclude=/Events/','--exclude=/HTML/','--exclude=/MCatNLO/RUN_*/','--exclude=/SubProcesses/P*/GF*/',str(original)+'/',str(clone)+'/'],check=True)
 for name in ['weight_lines.f','montecarlocounter.f','driver_macnlops.f','macnlops_capi.f','macnlops_evaluator.f']:shutil.copy2(FROZEN/'SubProcesses'/name,clone/'SubProcesses'/name)
 shutil.copy2(FROZEN/'MCatNLO/srcPythia8/MacNLOPSMatch.cc',clone/'MCatNLO/srcPythia8/MacNLOPSMatch.cc')
 c=RunCardNLO(str(original/'Cards/run_card.dat'));chosen=[]
 direct_input=None
 if not Path(record['channels'][0]['directory']).is_dir():
  prefix_key={'DY':'DY','H_HEFT':'H','TTbar':'ttbar'}[key]
  backup=CAMPAIGNS/campaign/'backup'/f'{prefix_key}_{run}_events.lhe.gz'
  target=BASE/'direct_inputs'/key/'events.lhe';count=prefix(backup,target,1500)
  direct_input=dict(source=bind(backup),subset=bind(target),events=count,mode='Direct native G/RDK reevaluation on already-sampled frozen endpoint+hardBorn records; raw source channels removed by campaign cleanup.')
  groups={}
 for provider,gfs in groups.items():
  for group,ch in gfs.items():
   source=Path(ch['directory']);destination=clone/'SubProcesses'/provider/source.name
   destination.mkdir(parents=True,exist_ok=True);count=prefix(source/'events.lhe',destination/'events.lhe',cap)
   reference=BASE/'production_reference'/key/provider/source.name/'events.lhe';assert prefix(source/'events_macnlops.lhe',reference,count)==count
   shutil.copy2((source/'res_1').resolve(),destination/'res_1')
   chosen.append(dict(provider=provider,group=group,channel=source.name,seed=ch['seed'],events=count,
      source=bind(source/'events.lhe'),source_summary=bind(source/'events_macnlops.json'),production_matched=bind(source/'events_macnlops.lhe'),
      subset=bind(destination/'events.lhe'),production_prefix=bind(reference),channel_integrals=ch['channel']))
 baseline={name:sha(clone/name) for name in ['SubProcesses/weight_lines.f','SubProcesses/montecarlocounter.f','SubProcesses/driver_macnlops.f','SubProcesses/macnlops_capi.f','SubProcesses/macnlops_evaluator.f','MCatNLO/srcPythia8/MacNLOPSMatch.cc']}
 scale={k:c[k] for k in ['lpp1','lpp2','fixed_ren_scale','fixed_fac_scale','mur_ref_fixed','muf_ref_fixed','mur_over_ref','muf_over_ref','dynamical_scale_choice','lhaid','pdlabel']}
 result=dict(key=key,original=str(original),clone=str(clone),source_summary=bind(summary),samples=chosen,direct_matched_input=direct_input,
      scale_configuration=scale,baseline_matching_sources=baseline,
      generation_sources=macnlops.source_files(original),
      control='No-PDF/fixed-scale ee must returnG=1; dynamic pp checks native Born model,coupling order and carrier PDFs. All inputs are exact first-N prefixes with original seeds; no event selection/reseed.',
      source_epoch='Production native-IR source; unweighted hard construction unchanged. Baseline uses frozen Wbb matching-only sources, independent of pending Born-factor implementation.')
 print(key,len(chosen),'jobs',sum(x['events'] for x in chosen),'events',flush=True);return result

def main():
 assert not (BASE/'baseline_manifest.json').exists()
 with concurrent.futures.ThreadPoolExecutor(4) as pool:processes=list(pool.map(prepare,SPECS))
 result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),processes=processes,
   frozen_baseline_process=str(FROZEN),Pythia_library=bind('/nfs/home/rikkert/Downloads/pythia8/lib/libpythia8.so'),
   current_python=bind(ROOT/'madgraph/various/macnlops.py'),script=bind(Path(__file__)))
 (BASE/'baseline_manifest.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print('Prepared',len(processes),'isolated processes,',sum(len(p['samples']) for p in processes),'native jobs',flush=True)
if __name__=='__main__':main()
