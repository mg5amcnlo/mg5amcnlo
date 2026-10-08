"""Second small-worker replay: time complete event preparation and pool output."""
import difflib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
paths = json.loads((HERE / 'paths.json').read_text())
name = 'driver_mintMC.f'
path = Path(paths['profile']) / 'SubProcesses' / name
before = (Path(paths['original']) / 'SubProcesses' / name).read_text()
after = before
for old, new in [
    ('      double precision ampli_candidate_factor',
     '      double precision prof_write,prof_pool,prof_t0,prof_t1\n'
     '      data prof_write,prof_pool/0d0,0d0/\n'
     '      double precision ampli_candidate_factor'),
    ('                  call write_current_mcatnlo_event(',
     '                  call cpu_time(prof_t0)\n'
     '                  call write_current_mcatnlo_event('),
    ('                  call ampli_record_candidate_factor(',
     '                  call cpu_time(prof_t1)\n'
     '                  prof_write=prof_write+prof_t1-prof_t0\n'
     '                  call ampli_record_candidate_factor('),
    ('            call ampli_finish_pool',
     '            call cpu_time(prof_t0)\n'
     '            call ampli_finish_pool\n'
     '            call cpu_time(prof_t1)\n'
     '            prof_pool=prof_pool+prof_t1-prof_t0\n'
     "            write(*,*) 'PROFILE full_event_write,pool_output:',\n"
     '     $           prof_write,prof_pool'),
]:
    assert after.count(old) == 1, old
    after = after.replace(old, new)
path.write_text(after)
(HERE / (name + '.patch')).write_text(''.join(difflib.unified_diff(
    before.splitlines(True), after.splitlines(True), fromfile=name, tofile=name)))
