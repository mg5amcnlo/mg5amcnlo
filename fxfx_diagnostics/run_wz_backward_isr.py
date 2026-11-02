#!/usr/bin/env python3
"""Run the prepared WZ backward-ISR control and variation independently."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/audits/fxfx_delta_wz_backward_isr'
LOCK = threading.Lock()
STATE = {'controller_pid': os.getpid(), 'stage': 'running', 'samples': {}}


def update(sample=None, **values):
    with LOCK:
        target = STATE if sample is None else STATE['samples'].setdefault(sample, {})
        target.update(values)
        STATE['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = OUTPUT / 'status.json.tmp'
        temporary.write_text(json.dumps(STATE, indent=2) + '\n')
        temporary.replace(OUTPUT / 'status.json')


def execute(sample, process, stage, command):
    log = OUTPUT / f'{stage}_{sample}.log'
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1')
    with log.open('w') as stream:
        child = subprocess.Popen(command, cwd=process, stdout=stream,
                                 stderr=subprocess.STDOUT, env=environment,
                                 start_new_session=True)
        update(sample, stage=stage, child_pid=child.pid, log=str(log), command=command)
        code = child.wait()
    update(sample, child_pid=None, returncode=code)
    if code:
        raise RuntimeError(f'{sample} {stage} failed ({code}): {log}')


def run_sample(name, settings):
    try:
        process = Path(settings['process'])
        run = process / 'Events' / settings['run']
        for relative, expected in settings['source_hashes'].items():
            actual = hashlib.sha256((process / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f'Prepared source changed: {process / relative}')
        execute(name, process, 'generation', settings['generation_command'])
        events = run / 'events.lhe.gz'
        if not events.is_file() or events.stat().st_size == 0:
            raise RuntimeError(f'Generation did not produce {events}')
        summary = (run / 'summary.txt').read_text()
        if 'Number of events generated: 1000000' not in summary:
            raise RuntimeError(f'Unexpected generation summary: {run}')
        update(name, stage='generation_complete', event_file=str(events), summary=summary)
        execute(name, process, 'shower', settings['shower_command'])
        histogram = run / 'plot_PYTHIA8_1_0.HwU'
        if not histogram.is_file() or histogram.stat().st_size == 0:
            raise RuntimeError(f'Shower did not produce {histogram}')
        update(name, stage='complete', histogram=str(histogram))
        print(f'{name}: generation and shower complete', flush=True)
        return True
    except Exception as error:
        update(name, stage='failed', error=str(error), child_pid=None)
        print(f'{name}: {error}', flush=True)
        return False


def main():
    setup = json.loads((OUTPUT / 'setup.json').read_text())
    update()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_sample, name, settings)
                   for name, settings in setup['samples'].items()]
        results = [future.result() for future in futures]
    update(stage='showers_complete' if all(results) else 'failed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
