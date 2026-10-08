"""Check archived folded DY generation with bounded native iterations."""
from pathlib import Path
import hashlib,json,math,re,sys
sys.dont_write_bytecode=True
import analyze_common
HERE=Path(__file__).resolve().parent;OUT=HERE/'process';NAME='bounded2000'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=json.loads((HERE/'started.json').read_text());finish=json.loads((HERE/'finished.json').read_text())
 assert finish['returncode']==0 and finish['events_exist']
 assert start['source_sha256']==start['exported_sha256']==finish['source_sha256_after']
 for p,expected in start['source_sha256'].items():assert digest(HERE/'sources'/p)==expected
 assert start['settings']['folding']==[2,2,2] and start['settings']['iseed']==19731 and start['cores']==2
 assert start['checkpoints_before']==finish['checkpoints_after']
 assert len(start['checkpoints_before'])==16
 for path,expected in start['checkpoints_before'].items():assert digest(OUT/path)==expected
 log=(HERE/'run.log').read_text()
 assert 'Surveying channel rates and adapting grids' not in log
 assert 'Adapting integration grids' not in log
 assert 'Generating channel event samples' in log
 assert not list((OUT/'SubProcesses').glob('P*/G*/log_MINT0.txt'))
 report=analyze_common.analyze(OUT,'ampli',2000,NAME,HERE/'sources/madgraph/various/ampli_pool.py')
 assert report['complete_expected_sample'] and report['native_validation']['all_native_and_collected_tail_fractions_below_one_percent']
 assert report['production']['grid_updates']>0
 assert report['stage_logs']['2']==len(report['native_validation']['workers'])
 iterations=[];NUMBER=analyze_common.NUMBER
 def close(a,b):assert math.isclose(a,b,rel_tol=2.e-10,abs_tol=2.e-11),(a,b)
 for channel in report['production_manifest']['channels']:
  parent=OUT/'SubProcesses'/channel['subprocess']/('GF'+channel['channel'])
  saved=list(map(analyze_common.number,(parent/'ampli_grids').read_text().splitlines()[3].split()))
  absolute=saved[0]+saved[4];pvirt=max(.001,min(.999,saved[4]/absolute))
  envelope=max(saved[-2]/(1.-pvirt),saved[-1]/pvirt);cutoff=min(envelope,absolute)
  for batch in channel['batches']:
   assert batch['adaptation']['mask']==[1,1,1,1,0,0,0]
   text=(OUT/batch['directory']/'ampli_pool.dat').read_text().splitlines()
   nepoch=int(text[1].split()[2]);epochs=[row.split() for row in text[6:6+nepoch]]
   targets=[int(e[3]) for e in epochs]
   assert targets[0]==max(1024,min(8192,batch['generated_target']))
   close(analyze_common.number(epochs[0][9]),cutoff)
   assert analyze_common.number(epochs[0][10])>=envelope*(1.-1.e-12)
   workerlog=(OUT/batch['directory']/'log_MINT2.txt').read_text()
   match=re.search(r'AmpliCol native surveyed maximum, initial storage cutoff:\s*('+NUMBER+r')\s*('+NUMBER+')',workerlog)
   assert match;close(analyze_common.number(match[1]),envelope);close(analyze_common.number(match[2]),cutoff)
   forecasts=re.findall(r'AmpliCol native effective events, remaining, nonzero acceptance:\s*('+NUMBER+r')\s*('+NUMBER+r')\s*('+NUMBER+')',workerlog)
   assert len(forecasts)==len(epochs)-1
   for previous,next_target,(_,remaining,efficiency) in zip(targets,targets[1:],forecasts):
    remaining,efficiency=map(analyze_common.number,(remaining,efficiency))
    expected=2*previous
    if efficiency>0 and 1.1*remaining/efficiency<expected:expected=max(1024,math.ceil(1.1*remaining/efficiency))
    assert next_target==expected,(next_target,expected)
   iterations.append(dict(directory=batch['directory'],nonzero_targets=targets,nonzero_counts=[int(e[2]) for e in epochs],trials=[int(e[1]) for e in epochs],initial_storage_cutoff=cutoff,saved_envelope=envelope,final_first_epoch_envelope=analyze_common.number(epochs[0][10])))
 banner=(OUT/'Events'/NAME/(NAME+'_tag_1_banner.txt')).read_text()
 assert re.search(r'^\s*19731\s*=\s*iseed\s*!',banner,re.M)
 assert (OUT/'SubProcesses/randinit').read_text().strip()=='r=19731'
 report['bounded_iteration_checks']=dict(generation_only=True,checkpoints_unchanged=True,all_masks_1111000=True,first_batch_bounded_1024_to_8192=True,initial_storage_cutoff_is_minimum_survey_rate_and_envelope=True,first_epoch_preserves_survey_envelope_floor=True,remaining_event_forecasts_and_growth_ceiling_reproduced=True,iterations=iterations,source_snapshots_match=True)
 (HERE/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 result=dict(events=report['lhe']['events'],negative_events=report['lhe']['negative'],generation_trials=report['production']['trials'],generation_cpu_seconds=report['production']['cpu_seconds'],native_iterations=report['production']['native_iterations'],grid_updates=report['production']['grid_updates'],collection_rounds=report['production']['collection_rounds'],collected_tail_fraction=report['native_validation']['collected_tail_fraction'],max_native_tail_fraction=report['native_validation']['maximum_native_tail_fraction'],max_channel_tail_bound=report['native_validation']['maximum_final_channel_tail_bound'],signed_rate_pb=report['integration']['signed_pb'],signed_error_pb=report['integration']['signed_error_pb'],absolute_rate_pb=report['integration']['absolute_pb'],folding=[2,2,2],adaptation_mask=[1,1,1,1,0,0,0],unchanged_survey_checkpoints=16,first_batch_and_storage_cutoff_verified=True,first_epoch_survey_envelope_floor_verified=True,remaining_event_forecasts_verified=True,source_snapshots_match=True,seed=19731,cores=2,iterations=iterations)
 (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
