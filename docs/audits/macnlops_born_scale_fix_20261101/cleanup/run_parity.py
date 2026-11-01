#!/usr/bin/env python3
"""Compare complete native matching records before/after C++ guard cleanup."""
from pathlib import Path
import hashlib,json,os,re,shutil,subprocess,sys,unittest
from unittest.mock import patch
R=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS');sys.path.insert(0,str(R))
from tests.unit_tests.various import test_macnlops_first_emission as first
from madgraph.various import macnlops as mc
B=Path('/export/slow1/rikkert/macnlops_born_scale_fix_20261101');O=B/'cleanup/parity';O.mkdir(exist_ok=True)
SOURCES={'before':B/'cleanup/before/Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc','after':R/'Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
replacement=r'''
  const char* unsupported=getenv("MACNLOPS_STUB_FORCE_PDF_ZERO");
  if(unsupported && atoi(unsupported)>=1 && atoi(unsupported)<=2) {
    const int side=atoi(unsupported)-1;
    ids[side]=5;x[side]=.022904539932296634;*pt=.6;q[0]=*pt;
  }
'''
stub=first.STUB.replace('  *mass=13000.*sqrt(x[0]*x[1]);',replacement+'  *mass=13000.*sqrt(x[0]*x[1]);');assert stub!=first.STUB
SCENARIOS=[
 ('unit',{},[],False,True),
 ('G_product6',{'MACNLOPS_STUB_BORN_COUPLING':'2','MACNLOPS_STUB_BORN_PDF':'3'},[],False,True),
 ('active_bound_unit',{'MACNLOPS_STUB_SOURCE':'1','MACNLOPS_STUB_PS':'0.000001'},[],False,True),
 ('active_bound_G6',{'MACNLOPS_STUB_SOURCE':'1','MACNLOPS_STUB_PS':'0.000001','MACNLOPS_STUB_BORN_COUPLING':'2','MACNLOPS_STUB_BORN_PDF':'3'},[],False,True),
 ('zero_coupling_negativeS',{'MACNLOPS_STUB_BORN_COUPLING':'0'},[],True,True),
 ('zero_pdf_negativeS',{'MACNLOPS_STUB_BORN_PDF':'0'},[],True,True),
 ('unsupported_pdf',{'MACNLOPS_STUB_FORCE_PDF_ZERO':'1'},[],False,True),
 ('unsupported_pdf_and_Gzero',{'MACNLOPS_STUB_FORCE_PDF_ZERO':'1','MACNLOPS_STUB_BORN_COUPLING':'0'},[],False,True),
 ('legacy_v2_invalid_reference',{'MACNLOPS_STUB_BORN_READY':'0','MACNLOPS_STUB_BORN_COUPLING':'nan'},['--s-veto','v2'],False,True),
 ('legacy_v3_invalid_reference',{'MACNLOPS_STUB_BORN_READY':'0','MACNLOPS_STUB_BORN_COUPLING':'nan'},['--s-veto','v3'],False,True),
 ('invalid_reference',{'MACNLOPS_STUB_BORN_READY':'0'},[],False,False),
]
command_records=[];orig_run=subprocess.run
outputs={};fixturebindings=[]
for epoch,source in SOURCES.items():
 destination=O/epoch;destination.mkdir(exist_ok=True);fake=destination/'source_tree';src=fake/'Template/NLO/MCatNLO/srcPythia8';src.mkdir(parents=True,exist_ok=True)
 shutil.copy2(source,src/'MacNLOPSMatch.cc')
 for name in ['MacNLOPS.h','MacNLOPS.cc']:
  target=src/name
  if not target.exists():target.symlink_to(R/'Template/NLO/MCatNLO/srcPythia8'/name)
 tests=fake/'tests'
 if not tests.exists():tests.symlink_to(R/'tests')
 cls=type('Fixture_'+epoch,(unittest.TestCase,),{'REGISTRY':first.TestFirstEmissionNative.REGISTRY,'_class_cleanups':[]})
 def recorded_run(command,*args,**kwargs):
  result=orig_run(command,*args,**kwargs);command_records.append(dict(epoch=epoch,phase='fixture_build',argv=list(map(str,command)),cwd=str(kwargs.get('cwd','')),returncode=result.returncode));return result
 try:
  with patch.object(first,'ROOT',fake),patch.object(first,'STUB',stub),patch.object(first.subprocess,'run',recorded_run):first.TestFirstEmissionNative.setUpClass.__func__(cls)
  original=(cls.work/'events.lhe').read_text().replace('source="delta"','source="delta_native_ir"')
  with mc.LHEStream(cls.work/'events.lhe') as stream:original_events=len(list(stream))
  assert original_events==14
  env={key:value for key,value in cls.env.items() if not key.startswith('MACNLOPS_STUB_')};env['MACNLOPS_STUB_MODE']='3'
  epoch_out={}
  for name,changes,options,negative_s,success in SCENARIOS:
   text=original
   if negative_s:
    temporary=cls.work/'negative.lhe';temporary.write_text(text)
    with mc.LHEStream(temporary) as stream:header,blocks=stream.header,list(stream)
    for i,block in enumerate(blocks):
     if mc.Event(block).kind!='S':continue
     lines=block.splitlines(keepends=True);tokens=lines[1].split();tokens[2]='-'+tokens[2].lstrip('+-');lines[1]=' '.join(tokens)+'\n';blocks[i]=''.join(lines)
    text=header+''.join(blocks)+'</LesHouchesEvents>\n'
   (cls.work/'events.lhe').write_text(text)
   argv=['./match.exe','--input','events.lhe','--output',name+'.lhe','--summary',name+'.json','--routes','routes.dat','--settings',str(cls.settings),'--pdf',cls.pdf,'--seed','7349',*options]
   run=orig_run(argv,cwd=cls.work,env=dict(env,**changes),capture_output=True,text=True)
   command_records.append(dict(epoch=epoch,phase=name,argv=argv,cwd=str(cls.work),environment_changes=changes,returncode=run.returncode))
   assert (run.returncode==0)==success,(epoch,name,run.stdout,run.stderr)
   (destination/(name+'.stdout')).write_text(run.stdout);(destination/(name+'.stderr')).write_text(run.stderr)
   if success:
    out=cls.work/(name+'.lhe');shutil.copy2(out,destination/out.name);shutil.copy2(cls.work/(name+'.json'),destination/(name+'.json'))
    with mc.LHEStream(out) as stream:header,blocks=stream.header,list(stream)
    assert len(blocks)==14
    tags=[mc.xml_attributes(re.search(r'<macnlops_decision\b([^>]*?)/>',block)[1]) for block in blocks]
    if name=='unsupported_pdf_and_Gzero':
     assert any(t['rule']=='pdf_support_no_transfer' for t in tags)
     assert not any(t['rule']=='born_scale_no_transfer' for t in tags)
    epoch_out[name]=dict(bytes=out.read_bytes(),events=len(blocks),rules={rule:sum(t['rule']==rule for t in tags) for rule in set(t['rule'] for t in tags)},sha256=sha(out))
   else:
    assert 'Born-scale factor' in run.stderr+run.stdout
    epoch_out[name]=dict(error='Born-scale factor',returncode=run.returncode)
  outputs[epoch]=epoch_out
  for name in ['match.exe','probe.exe','stub.c','stub.o','stub_profile.txt','routes.dat','shower.cmnd']:
   shutil.copy2(cls.work/name,destination/name)
  fixturebindings.append(dict(epoch=epoch,cpp_source_sha256=sha(source),compiled_cpp_sha256=sha(src/'MacNLOPSMatch.cc'),executable_sha256=sha(destination/'match.exe')))
 finally:cls.doClassCleanups()
results=[]
for name,*_ in SCENARIOS:
 a,b=outputs['before'][name],outputs['after'][name]
 assert a==b,(name,'Before/after event record differs')
 results.append(dict(scenario=name,events=a.get('events',0),complete_LHE_bytes_identical='bytes' in a,sha256=a.get('sha256'),rules=a.get('rules'),error=a.get('error'),returncode=a.get('returncode',0)))
(O/'commands.json').write_text(json.dumps(command_records,indent=2)+'\n')
report=dict(passed=True,scenarios=len(results),successful_event_records=sum(x['events'] for x in results),source_events_per_scenario=14,results=results,fixture_bindings=fixturebindings,environment={key:os.environ.get(key) for key in ['PYTHIA8','PYTHIA8DATA','MACNLOPS_TEST_PDF','LD_LIBRARY_PATH']},script_sha256=sha(Path(__file__)),commands_sha256=sha(O/'commands.json'),production_sources_modified=False)
(O/'parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
