#!/usr/bin/env python3
"""Replay saved MC@NLO grids to check the optional hard-coordinate writer.

The input process is read only. Two private copies use the same saved grids,
input settings and RNG states. Their only compiled setting difference is
STORE_HARD_MOMENTA. The complete output must be byte-identical after removing
the optional XML blocks. This exercises the actual add_write_info/stock mass
reshuffle/event writer, rather than a substitute particle-list fixture.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


PACKET = re.compile(rb'^[ \t]*<macnlops_hard_event\b[^\n]*\n.*?'
                    rb'^[ \t]*</macnlops_hard_event>[ \t]*\n', re.M | re.S)
EVENT = re.compile(rb'<event>.*?</event>', re.S)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(process):
    """Pin source/card/grids/RNG and the original collected source."""
    paths = [process / 'Source/run_card.inc', process / 'Cards/run_card.dat']
    paths += list((process / 'SubProcesses').glob('*.f'))
    paths += list((process / 'SubProcesses').glob('P*/mc_histories.inc'))
    for name in ('mint_grids', 'grid.MC_integer', 'randinit', 'input_app.txt'):
        paths += list((process / 'SubProcesses').rglob(name))
    paths += list((process / 'Events').rglob('events.lhe*'))
    return {str(p.relative_to(process)): sha(p)
            for p in sorted(set(paths)) if p.is_file()}


def clone_ignore(directory, names):
    return [name for name in names if name in ('Events', 'HTML', '.git', 'ME5_debug')
            or name.endswith(('.lhe', '.lhe.gz', '.log', '.top', '.HwU'))
            or name.startswith('log_')]


def run_copy(enabled, args, environment):
    label = 'enabled' if enabled else 'disabled'
    directory = args.work / label
    process = directory / 'process'
    directory.mkdir()
    shutil.copytree(args.process, process, symlinks=True, ignore=clone_ignore)
    card = process / 'Source/run_card.inc'
    setting = re.compile(r'(?m)^\s*STORE_HARD_MOMENTA\s*=\s*\.(?:TRUE|FALSE)\.\s*$')
    old = card.read_text()
    new, count = setting.subn('      STORE_HARD_MOMENTA = %s' %
                             ('.TRUE.' if enabled else '.FALSE.'), old)
    if count != 1:
        raise RuntimeError('Expected one compiled STORE_HARD_MOMENTA setting')
    card.write_text(new)
    with (directory / 'generic_build.log').open('w') as log:
        subprocess.run(['make', '-j1', '../lib/libgeneric.a'],
                       cwd=process / 'Source', env=environment, check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    jobs = []
    for context_index, context in enumerate(args.contexts):
        native = process / 'SubProcesses' / context
        with (directory / (context + '_build.log')).open('w') as log:
            subprocess.run(['make', '-j1', 'madevent_mintMC'], cwd=native,
                           env=environment, check=True, stdout=log,
                           stderr=subprocess.STDOUT)
        for grid_index, grid in enumerate(args.grids):
            target = native / grid
            lines = (target / 'input_app.txt').read_text().splitlines()
            lines[7] = '2 ! hard-annotation writer replay from saved grids'
            lines[10] = '0 %d ! diagnostic source events' % args.events_per_grid
            (target / 'input_app.txt').write_text('\n'.join(lines) + '\n')
            # Unlink the local link before writing so no parent RNG is changed.
            random_file = target / 'randinit'
            random_file.unlink(missing_ok=True)
            seed = args.seed + 100 * context_index + grid_index
            random_file.write_text('r=%d\n' % seed)
            log_path = directory / (context + '_' + grid + '_generation.log')
            start = time.monotonic()
            with log_path.open('w') as log:
                result = subprocess.run([str(native / 'madevent_mintMC')],
                    cwd=target, env=environment,
                    input=(target / 'input_app.txt').read_text(), text=True,
                    stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            if result.returncode:
                raise RuntimeError('Source replay failed: ' + str(log_path))
            events = target / 'events.lhe'
            count = len(EVENT.findall(events.read_bytes()))
            if count != args.events_per_grid:
                raise AssertionError('Incorrect generated count: ' + str(events))
            jobs.append({'context': context, 'grid': grid, 'seed': seed,
                         'events': count, 'event_file': str(events),
                         'event_sha256': sha(events), 'log': str(log_path),
                         'seconds': time.monotonic() - start})
    return {'store_hard_momenta': enabled, 'process': str(process),
            'compiled_card_sha256': sha(card),
            'generic_library_sha256': sha(process / 'lib/libgeneric.a'),
            'jobs': jobs}


def inspect_events(data):
    counts = Counter()
    boosted = Counter()
    changed = Counter()
    for event in EVENT.findall(data):
        match = PACKET.search(event)
        if match is None:
            raise AssertionError('Enabled writer omitted an annotation')
        packet = match.group().splitlines()
        if (b'source="ordinary"' not in packet[0] or
                b'partition="global_sum"' not in packet[0]):
            raise AssertionError('Replay is not a qualified ordinary source')
        metadata = list(map(int, packet[1].split()))
        kind = 'S' if metadata[1] == 1 else 'H'
        counts[kind] += 1
        hard = [line.split() for line in packet[2:-1]]
        nup = int(event.splitlines()[1].split()[0])
        body = [line.split() for line in event.splitlines()[2:2 + nup]]
        external = [row for row in body if int(row[1]) in (-1, 1)]
        if len(hard) != metadata[0] or len(hard) != len(external):
            raise AssertionError('Native external count changed')
        for normal, original in zip(external, hard):
            if [normal[i] for i in (0, 1, 4, 5)] != [original[i] for i in (0, 1, 4, 5)]:
                raise AssertionError('External IDs/status/colours changed')
            if not all(math.isfinite(float(x)) for x in original[6:]):
                raise AssertionError('Nonfinite hard coordinates')
        incoming = [row for row in hard if int(row[1]) == -1]
        energy = sum(float(row[9]) for row in incoming)
        pz = sum(float(row[8]) for row in incoming)
        if abs(pz) > 1.e-3 * energy:
            boosted[kind] += 1
        if any(float(row[10]) != float(native[10])
               for row, native in zip(external, hard)):
            changed[kind] += 1
    return {'event_kinds': dict(counts), 'longitudinally_boosted': dict(boosted),
            'stock_body_has_transported_masses': dict(changed)}


def audit(args):
    args.process = args.process.resolve()
    args.work = args.work.resolve()
    for context in args.contexts:
        for grid in args.grids:
            source_input = args.process / 'SubProcesses' / context / grid / 'input_app.txt'
            if len(source_input.read_text().splitlines()) < 14:
                raise ValueError('Choose a populated event-generation grid: ' + str(source_input))
    args.work.mkdir(parents=True, exist_ok=False)
    before = inventory(args.process)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda enabled: run_copy(enabled, args, environment),
                                    (True, False)))
    comparisons = []
    total = Counter()
    boosted = Counter()
    changed = Counter()
    for enabled, disabled in zip(results[0]['jobs'], results[1]['jobs']):
        first = Path(enabled['event_file']).read_bytes()
        second = Path(disabled['event_file']).read_bytes()
        if PACKET.search(second):
            raise AssertionError('Disabled writer emitted an annotation')
        stripped, count = PACKET.subn(b'', first)
        if count != args.events_per_grid or stripped != second:
            raise AssertionError('Actual event body changed for %s/%s' %
                                 (enabled['context'], enabled['grid']))
        detail = inspect_events(first)
        total.update(detail['event_kinds'])
        boosted.update(detail['longitudinally_boosted'])
        changed.update(detail['stock_body_has_transported_masses'])
        comparisons.append(dict(context=enabled['context'], grid=enabled['grid'],
            ordinary_output_byte_identical=True, stripped_sha256=sha(disabled['event_file']),
            annotations=count, **detail))
    if not all(total[kind] and boosted[kind] and changed[kind] for kind in ('S', 'H')):
        raise AssertionError('Replay must cover boosted H and S with stock mass transport')
    after = inventory(args.process)
    if before != after:
        raise AssertionError('Input process was modified')
    report = {'complete': True, 'pass': True, 'source_process': str(args.process),
              'source_unchanged': True, 'input_fingerprints': before,
              'audit_script': str(Path(__file__).resolve()),
              'audit_script_sha256': sha(__file__),
              'only_compiled_setting_difference': 'STORE_HARD_MOMENTA',
              'same_saved_grids_and_random_states': True,
              'ordinary_outputs_byte_identical': True, 'events': sum(total.values()),
              'event_kinds': dict(total), 'longitudinally_boosted': dict(boosted),
              'stock_body_has_transported_masses': dict(changed),
              'copies': results, 'comparisons': comparisons,
              'seconds': time.monotonic() - start}
    (args.work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('process', type=Path)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--contexts', nargs='+', default=['P0_uux_emep'])
    parser.add_argument('--grids', nargs='+', default=['GF1.0', 'GF2.0'])
    parser.add_argument('--events-per-grid', type=int, default=50)
    parser.add_argument('--seed', type=int, default=897321)
    parser.add_argument('--timeout', type=int, default=300)
    result = audit(parser.parse_args())
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('input_fingerprints', 'copies')}, indent=2))
