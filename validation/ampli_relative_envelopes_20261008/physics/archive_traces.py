"""Losslessly gzip completed stream traces, preserving their hashes and paths."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
manifest=HERE/'trace_manifest.json'
existing=json.loads(manifest.read_text())['traces'] if manifest.exists() else []
records={(r['variant'],r['workload']):r for r in existing}
for path in sorted(HERE.glob('*/**/ampli_stream_trials.dat')):
 variant=path.relative_to(HERE).parts[0];label=path.parent.name
 row=json.loads((HERE/variant/'summary.json').read_text())[label]
 rawhash=hashlib.sha256(path.read_bytes()).hexdigest();destination=path.with_suffix('.dat.gz')
 with path.open('rb') as source,destination.open('wb') as output:
  with gzip.GzipFile(filename='',mode='wb',fileobj=output,mtime=0) as compressed:shutil.copyfileobj(source,compressed)
 with gzip.open(destination,'rb') as source:assert hashlib.sha256(source.read()).hexdigest()==rawhash
 records[(variant,label)]=dict(variant=variant,workload=label,archive=str(destination.relative_to(HERE)),
  uncompressed_worker_trace=str(Path(row['worker'])/path.name),uncompressed_bytes=path.stat().st_size,
  uncompressed_sha256=rawhash,archive_bytes=destination.stat().st_size,
  archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
 path.unlink()
manifest.write_text(json.dumps(dict(note='Losslessly compressed archived stream diagnostics; original traces remain in isolated workers. Decompress archives for standalone analysis.',traces=[records[k] for k in sorted(records)]),indent=2)+'\n')
print('Archived',len(records),'traces')
