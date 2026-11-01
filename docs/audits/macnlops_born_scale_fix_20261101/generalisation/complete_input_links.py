from pathlib import Path
import os,sys,json,shutil,hashlib,itertools
BASE=Path(__file__).resolve().parent;ROOT=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS');sys.path.insert(0,str(ROOT));from madgraph.various import macnlops
p=BASE/'baseline_manifest.json';manifest=json.loads(p.read_text());shutil.copy2(p,BASE/'baseline_manifest.initial.json')
def bind(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
for proc in manifest['processes']:
 original=Path(proc['original']);clone=Path(proc['clone'])
 if proc['key']=='Wbb':
  summary=json.loads(Path(proc['source_summary']['path']).read_text());c=next(q for q in summary['channels'] if q['directory'].endswith('P0_udx_bbxwp/GF1.0_26'));source=Path(c['directory']);destination=clone/'SubProcesses'/source.parent.name/source.name;destination.mkdir(parents=True,exist_ok=True)
  with macnlops.LHEStream(source/'events.lhe') as stream:blocks=list(stream);header=stream.header
  (destination/'events.lhe').write_text(header+''.join(blocks)+'</LesHouchesEvents>\n');reference=BASE/'production_reference'/proc['key']/source.parent.name/source.name/'events.lhe';reference.parent.mkdir(parents=True,exist_ok=True)
  with macnlops.LHEStream(source/'events_macnlops.lhe') as stream:old=list(stream);header=stream.header
  assert len(old)==len(blocks)==9973;reference.write_text(header+''.join(old)+'</LesHouchesEvents>\n');shutil.copy2((source/'res_1').resolve(),destination/'res_1')
  proc['samples'].append(dict(provider=source.parent.name,group='GF1.0',channel=source.name,seed=c['seed'],events=9973,source=bind(source/'events.lhe'),source_summary=bind(source/'events_macnlops.json'),production_matched=bind(source/'events_macnlops.lhe'),subset=bind(destination/'events.lhe'),production_prefix=bind(reference),channel_integrals=c['channel'],targeted_control='Complete formerly failing source shard; includes actual charm-zero-PDF S event5294.'))
 for row in proc['samples']:
  source=Path(row['source']['path']).parent;destination=Path(row['subset']['path']).parent
  for entry in source.iterdir():
   if not entry.is_symlink():continue
   resolved=entry.resolve()
   try:relative=resolved.relative_to(original)
   except ValueError:continue
   target=clone/relative
   if not target.exists():continue
   path=destination/entry.name
   if path.exists() or path.is_symlink():continue
   path.symlink_to(os.path.relpath(target,destination))
  assert (destination/'FKS_params.dat').is_file() and (destination/'param_card.dat').is_file()
manifest['input_link_note']='Source-job metadata symlinks recreated only to isolated clone targets. Complete original9973-event WbbGF1.0_26 added at root request; no filtering/reseeding.'
p.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print('Complete source jobs',sum(len(q['samples']) for q in manifest['processes']),'events',sum(r['events'] for q in manifest['processes'] for r in q['samples']))
