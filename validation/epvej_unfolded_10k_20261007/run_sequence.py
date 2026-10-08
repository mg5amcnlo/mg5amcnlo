from pathlib import Path
import subprocess,sys,json
WORK=Path(__file__).resolve().parent
failed=False
for backend in ['mint','ampli','mint_tail']:
    print('Launching',backend,flush=True)
    with (WORK/(backend+'_wrapper.log')).open('w') as log:
        result=subprocess.run([sys.executable,str(WORK/'run_benchmark.py'),backend],stdout=log,stderr=subprocess.STDOUT)
    status=WORK/(backend+'_finished.json')
    data=json.loads(status.read_text()) if status.exists() else {}
    print(json.dumps(dict(backend=backend,returncode=result.returncode,events_exist=data.get('events_exist'),wall_seconds=data.get('wall_seconds'))),flush=True)
    failed|=bool(result.returncode) or not data.get('events_exist',False)
raise SystemExit(int(failed))
