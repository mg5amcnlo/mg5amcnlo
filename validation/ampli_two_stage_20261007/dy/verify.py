"""Reproduce the folded DY two-stage validation from this compact archive."""
from pathlib import Path
import argparse,hashlib,json,math,re,sys
sys.dont_write_bytecode=True
import analyze_common as common

HERE=Path(__file__).resolve().parent
NUMBER=common.NUMBER

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b,label):
 assert math.isclose(a,b,rel_tol=2.e-10,abs_tol=2.e-11),(label,a,b)

def verify_run(name):
 base=HERE/name;out=base/'drell_yan';source=HERE/'sources/madgraph/various/ampli_pool.py'
 start=json.loads((base/(name+'_started.json')).read_text())
 finish=json.loads((base/(name+'_finished.json')).read_text())
 assert finish['returncode']==0 and finish['events_exist']
 assert start['settings']['folding']==[2,2,2]
 assert start['settings']['req_acc']==-1
 assert start['source_sha256']==finish['source_sha256_after']
 exported=json.loads((HERE/'exported_source_sha256.json').read_text())
 for key,expected in start['source_sha256'].items():
  assert digest(HERE/'sources'/key)==expected,(key,'source drift')
  assert exported[key]['sha256']==expected,(key,'export differs')
 results=common.analyze(out,'ampli',start['settings']['nevents'],name,source)
 assert results['complete_expected_sample']
 assert not list((out/'SubProcesses').glob('P*/G*/log_MINT0.txt'))
 assert not list((out/'SubProcesses').glob('P*/G*/res_0.dat'))
 log=(base/(name+'.log')).read_text()
 assert 'Adapting integration grids' not in log
 if name.startswith('fresh'):
  assert 'Surveying channel rates and adapting grids' in log
 else:
  assert 'Surveying channel rates and adapting grids' not in log
 assert 'Generating channel event samples' in log
 manifest=results['production_manifest'];surveys=[];envelopes=[]
 for channel in manifest['channels']:
  parent=out/'SubProcesses'/channel['subprocess']/('GF'+channel['channel'])
  survey=common.parse_result(parent/'res_1.dat')
  assert survey['iterations']>=4
  assert survey['absolute_error_pb']<=.03*survey['absolute_pb']*(1.+1.e-12)
  rows=re.findall(r'AmpliCol survey iteration, cumulative trials, ABS, signed, relative error:\s*(\d+)\s+(\d+)\s*('+NUMBER+r')\s*('+NUMBER+r')\s*('+NUMBER+')',(parent/'log_MINT1.txt').read_text())
  assert len(rows)>=4 and int(rows[-1][0])==survey['iterations']
  assert common.number(rows[-1][4])<=.03
  text=(parent/'ampli_grids').read_text().splitlines()
  assert text[0].split()==['MG5_AMPLICOL','3','1']
  ndim,iconfig,sector,nvalues=map(int,text[1].split());folds=list(map(int,text[2].split()))
  assert folds==[1]*(ndim-3)+[2,2,2]
  values=list(map(common.number,text[3].split()))
  assert len(values)==2*nvalues+2
  mean_abs=values[0]+values[4];probability=max(.001,min(.999,values[4]/mean_abs))
  max_novi,max_virt=values[-2:]
  envelope=max(max_novi/(1.-probability),max_virt/probability)
  close(mean_abs,survey['absolute_pb'],'saved survey ABS')
  surveys.append(dict(channel=str(parent.relative_to(out)),iterations=int(survey['iterations']),retained_statistical_points=int(survey['points_or_quota']),total_evaluations=int(rows[-1][1]),relative_absolute_error=survey['absolute_error_pb']/survey['absolute_pb'],saved_stream_maxima=[max_novi,max_virt],initial_envelope=envelope))
  for batch in channel['batches']:
   worker=out/batch['directory'];pooltext=(worker/'ampli_pool.dat').read_text().splitlines()
   epoch=pooltext[6].split()
   close(common.number(epoch[9]),envelope,'first epoch cutoff')
   expected=max(1024,math.ceil(batch['generated_target']*max(1.,envelope/mean_abs)))
   assert int(epoch[3])==expected,('nonzero forecast',epoch[3],expected)
   match=re.search(r'AmpliCol saved survey stream maxima, initial production envelope:\s*('+NUMBER+r')\s*('+NUMBER+r')\s*('+NUMBER+')',(worker/'log_MINT2.txt').read_text())
   assert match
   for actual,target in zip(map(common.number,match.groups()),(max_novi,max_virt,envelope)):close(actual,target,'printed stream bound')
   envelopes.append(dict(directory=batch['directory'],first_nonzero_target=expected,initial_envelope=envelope))
 for relative,expected in finish['checkpoints_after'].items():assert digest(out/relative)==expected,('archived checkpoint',relative)
 assert results['production']['grid_updates']>0
 results['two_stage_checks']=dict(no_stage_zero=True,survey_minimum_four_iterations=True,survey_absolute_accuracy_at_most_three_percent=True,first_nonzero_budget_uses_saved_stream_envelope=True,surveys=surveys,envelopes=envelopes)
 (base/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
 return results,start,finish

def main():
 fresh,fs,ff=verify_run('fresh2000')
 restart,rs,rf=verify_run('restart2000')
 initial=json.loads((HERE/'first_survey_checkpoint_sha256.json').read_text())
 assert initial==ff['checkpoints_after']==rs['checkpoints_before']==rf['checkpoints_after']
 assert len(initial)==2*len(fresh['production_manifest']['channels'])
 for channel in fresh['production_manifest']['channels']:
  rel=Path('drell_yan/SubProcesses')/channel['subprocess']/('GF'+channel['channel'])
  for filename in ('log_MINT1.txt','res_1.dat'):
   assert digest(HERE/'fresh2000'/rel/filename)==digest(HERE/'restart2000'/rel/filename)
 result=dict(process='p p > e+ e- [QCD]',folding=[2,2,2],cores=3,req_acc=-1,
             fresh_stages=[1,2],restart_stages=[2],checkpoints_unchanged_during_fresh_and_restart=True,
             survey_logs_and_results_unchanged_on_restart=True,source_snapshots_match=True)
 for name,data in [('fresh2000',fresh),('restart2000',restart)]:
  result[name]=dict(events=data['lhe']['events'],negative_events=data['lhe']['negative'],
   generation_trials=data['production']['trials'],generation_cpu_seconds=data['production']['cpu_seconds'],
   production_iterations=data['production']['native_iterations'],grid_updates=data['production']['grid_updates'],
   generation_rounds=data['production']['collection_rounds'],
   collected_tail_fraction=data['native_validation']['collected_tail_fraction'],
   maximum_native_tail_fraction=data['native_validation']['maximum_native_tail_fraction'],
   maximum_final_channel_tail_bound=data['native_validation']['maximum_final_channel_tail_bound'],
   signed_rate_pb=data['integration']['signed_pb'],signed_error_pb=data['integration']['signed_error_pb'],
   absolute_rate_pb=data['integration']['absolute_pb'],surveys=data['two_stage_checks']['surveys'])
 (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
