#!/usr/bin/env python3
"""Check at most ten saved first-emission points with the native V3 driver.

This only reads completed source/sidecar files and existing native objects.
It compiles temporary evaluator drivers serially; it never exports a process,
generates events, showers events, or changes the completed campaign files.
Run from any directory with ``--output REPORT.json`` to retain the evidence.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from madgraph.various.macnlops_components import branch_candidates
from docs.audits.check_macnlops_source_h import evaluator_source, link_audit_object


CASES = (
    ('DY_5FS', 'P0_ddx_emep', 'ISR', 2),
    ('TTbar_5FS', 'P0_gg_ttx', 'ISR', 2),
    ('TTbar_5FS', 'P0_gg_ttx', 'FSR', 1),
    ('Wbb_4FS', 'P0_dxu_bbxwp', 'FSR', 2),
    ('SingleTop_5FS', 'P0_bdx_tux', 'FSR', 3),
)


def command(arguments, cwd, **kwargs):
    result = subprocess.run(arguments, cwd=cwd, text=True,
                            capture_output=True, timeout=45, **kwargs)
    if result.returncode:
        raise RuntimeError('%s failed:\n%s\n%s' %
                           (arguments[0], result.stdout[-5000:], result.stderr[-5000:]))
    return result.stdout


def selected_points(root):
    points = []
    for folder, directory, kind, count in CASES:
        selected = []
        matching = root / folder / 'Events/delta_1M/macnlops'
        pattern = 'SubProcesses__' + directory + '__*/shower.jsonl'
        for path in sorted(matching.glob(pattern))[:3]:
            with path.open() as stream:
                for line in itertools.islice(stream, 10):
                    record = json.loads(line)
                    if (record['event_kind'] == 'S' and record['emitted'] and
                            record['emission_type'] == kind):
                        selected.append((folder, directory, path, record))
                        if len(selected) == count:
                            break
            if len(selected) == count:
                break
        if len(selected) != count:
            raise RuntimeError('Insufficient saved prefix for %s %s' % (folder, kind))
        points.extend(selected)
    if len(points) > 10:
        raise RuntimeError('This bounded check may evaluate at most ten points')
    return points


def request_for_point(root, folder, directory, path, record):
    process = root / folder / 'SubProcesses' / directory
    registry = json.loads((process / 'mc_histories.json').read_text())
    candidate, = branch_candidates(record, registry, process)
    outer = registry['registry']['outer']
    metadata = (root / folder / 'Source/BornSupport' /
                ('c%d' % outer['context']) / 'metadata.f').read_text()
    configurations = {int(k): int(v) for k, v in re.findall(
        r'DATA MAPCONFIG\(\s*(\d+)\s*\)\s*/\s*(\d+)\s*/', metadata, re.I)}
    config = int(float(re.search(r'__GF([0-9.]+)', path.parent.name)[1]))
    graph = configurations[config]
    if record['source_partition']['source'] != 'delta':
        raise RuntimeError('These saved-point cases require their Delta source')
    flow = record['source_partition']['born_flow']
    real = [record['particles'][i] for i in candidate.permutation]
    born = [record['source_particles'][i] for i in candidate.born_permutation]
    lines = ['MACNLOPS_POINT_V3 1 %d %d' % (candidate.sector, len(real))]
    lines.extend('%d %s' % (p['id'], ' '.join('%.17e' % x for x in p['p']))
                 for p in real)
    lines.append('COMPONENT_LOCAL_V1 %d %d %d 1 1.' % (config, graph, flow))
    lines.append('COMPONENT_BORN_V1 %d' % len(born))
    lines.extend('%d %d %d %s %.17e' % (p['id'], *p['colors'],
                 ' '.join('%.17e' % x for x in p['p']), p['mass']) for p in born)
    lines.append('END_POINT')
    return '\n'.join(lines) + '\n', candidate, (config, graph, flow), registry['sha256']


def build_driver(root, folder, directory, stack, trap_delta=False, source_text=None,
                 delta_matching=True, counter_source_text=None):
    process = root / folder / 'SubProcesses' / directory
    # The existing PDF bridge finds lib/PDFsets relative to the executable.
    # A temporary child of P* keeps that search valid without adding symlinks
    # to, or rebuilding objects in, the completed generated process.
    work = Path(stack.enter_context(tempfile.TemporaryDirectory(
        prefix='.macnlops-v3-endpoints-', dir=process)))
    source = source_text if source_text is not None else evaluator_source()
    # Freeze the matching mode to the saved source. No generation cards change.
    source = source.replace('      call setrun\n',
                            '      call setrun\n      mcatnlo_delta=%s\n' %
                            ('.true.' if delta_matching else '.false.'), 1)
    staged, obj, binary = work / 'driver.f', work / 'driver.o', work / 'check'
    staged.write_text(source)
    command(['gfortran', '-O0', '-c', '-ffixed-line-length-132',
             '-I' + str(process), '-I' + str(root / folder / 'lib'),
             str(staged), '-o', str(obj)], work)
    dry = command(['make', '-n', '-W', 'driver_macnlops.o',
                   'macnlops_evaluator'], process)
    links = [line for line in dry.splitlines() if ' -o macnlops_evaluator ' in line]
    if len(links) != 1:
        raise RuntimeError('Cannot isolate the existing evaluator link command')
    link = shlex.split(links[0])
    link[link.index('-o') + 1] = str(binary)
    link_audit_object(link, obj)
    if trap_delta or counter_source_text is not None:
        counter = (counter_source_text if counter_source_text is not None else
                   (ROOT / 'Template/NLO/SubProcesses/montecarlocounter.f').read_text())
    if trap_delta:
        start = counter.index('      subroutine compute_delta(p,probne)')
        end = counter.index('      end subroutine compute_delta', start)
        end += len('      end subroutine compute_delta')
        counter = counter[:start] + '''      subroutine compute_delta(p,probne)
      implicit none
      double precision p(0:3,nexternal),probne
      error stop 'Unexpected numerical Delta evaluation in V3'
      end subroutine compute_delta''' + counter[end:]
    if trap_delta or counter_source_text is not None:
        trapped_source, trapped_obj = work / 'counter_check.f', work / 'counter_check.o'
        trapped_source.write_text(counter)
        command(['gfortran', '-O0', '-c', '-ffixed-line-length-132',
                 '-I' + str(process), '-I' + str(root / folder / 'lib'),
                 str(trapped_source), '-o', str(trapped_obj)], work)
        link[link.index('montecarlocounter.o')] = str(trapped_obj)
    command(link, process)
    environment = dict(os.environ)
    library_paths = [str(root / folder / 'lib')]
    lhapdf_config = re.search(r'^\s*lhapdf\s*=\s*(\S+)',
                             (root / folder / 'Source/make_opts').read_text(), re.M)
    if lhapdf_config:
        library_paths.append(str(Path(lhapdf_config[1]).resolve().parents[1] / 'lib'))
    if environment.get('LD_LIBRARY_PATH'):
        library_paths.append(environment['LD_LIBRARY_PATH'])
    environment['LD_LIBRARY_PATH'] = ':'.join(library_paths)
    return binary, process, environment


def check(root, trap_delta=False):
    driver = ROOT / 'Template/NLO/SubProcesses/driver_macnlops.f'
    result = dict(schema='macnlops-component-endpoint-check-v1',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
                  scope='saved first-ten-record prefixes; no event generation or showering',
                  closure_relative_tolerance=1.e-7,
                  delta_calculation_trap=trap_delta, points=[])
    with contextlib.ExitStack() as stack:
        binaries = {}
        for folder, directory, path, record in selected_points(root):
            key = (folder, directory)
            if key not in binaries:
                binaries[key] = build_driver(root, folder, directory, stack, trap_delta)
            binary, process, environment = binaries[key]
            request, route, selector, registry_hash = request_for_point(
                root, folder, directory, path, record)
            stdout = command([str(binary)], process, input=request, env=environment)
            records = [line.split() for line in stdout.splitlines()
                       if line.startswith('MACNLOPS_DENSITY_V1 ')]
            if len(records) != 1:
                raise RuntimeError('Expected exactly one density result:\n' + stdout[-5000:])
            fields, = records
            particles = {p['index']: p for p in record['pythia_event']}
            result['points'].append(dict(
                process=folder, sidecar=str(path.relative_to(root)),
                sequence=record['sequence'], emission_type=record['emission_type'],
                radiator_pdg=particles[record['branch_indices']['radiator_before']]['id'],
                pt_evol=record['pt_evol'], native_sector=route.sector,
                endpoint_permutation=list(route.permutation),
                born_permutation=list(route.born_permutation),
                config=selector[0], graph=selector[1], born_flow=selector[2],
                registry_sha256=registry_hash,
                sidecar_record_sha256=hashlib.sha256(json.dumps(record, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest(),
                request_sha256=hashlib.sha256(request.encode()).hexdigest(),
                status=int(fields[2]), densities=list(map(float, fields[3:])),
                diagnostics=[line.strip() for line in stdout.splitlines()
                    if line.startswith(('MACNLOPS_BORN_', 'MACNLOPS_DIAGNOSTIC',
                                        'MACNLOPS_MASS_'))]))
    result['passed'] = all(point['status'] == 0 for point in result['points'])
    return result


def check_ordinary_scale_skip(root):
    """Compare the old/new H-scale preparation at low P and support edges."""
    source = evaluator_source()
    # Ordinary G/P is unchanged by this flag. The density-only support mode
    # makes the explicit CONFIG bounds authoritative for these edge checks.
    source = source.replace('      call setrun\n',
                            '      call setrun\n      macnlops=.false.\n', 1)
    baseline = source.replace('               if(.not.local_component)then\n',
                              '               if(.true.)then\n', 1)
    if baseline == source:
        raise RuntimeError('Cannot isolate the local H-scale preparation')

    def point(sequence, energy):
        final = (1000. - energy) / 2
        transverse = math.sqrt(final**2 - energy**2 / 4)
        rows = [(2, 500., 0., 0., 500.), (-2, 500., 0., 0., -500.),
                (11, final, -energy / 2, transverse, 0.),
                (-11, final, -energy / 2, -transverse, 0.),
                (21, energy, energy, 0., 0.)]
        return '\n'.join(['MACNLOPS_POINT_V3 %d 1 5' % sequence] +
            ['%d %.17e %.17e %.17e %.17e' % row for row in rows] +
            ['COMPONENT_LOCAL_V1 1 1 1 0 1.', 'END_POINT']) + '\n'

    payload = ('MACNLOPS_CONFIG_V1 .999 .999 0 0\n' +
        ''.join(point(i + 1, energy) for i, energy in enumerate((.2, 3., 20.))) +
        'MACNLOPS_CONFIG_V1 .999 .999 100 100\n' + point(4, 20.) +
        'MACNLOPS_CONFIG_V1 .001 .001 0 0\n' + point(5, 20.))
    records = []
    with contextlib.ExitStack() as stack:
        for body in (baseline, source):
            binary, process, environment = build_driver(root, 'DY_5FS',
                'P0_uux_emep', stack, trap_delta=True, source_text=body,
                delta_matching=False)
            stdout = command([str(binary)], process, input=payload, env=environment)
            rows = [line.split() for line in stdout.splitlines()
                    if line.startswith('MACNLOPS_DENSITY_V1 ')]
            if len(rows) != 5 or any(int(row[2]) for row in rows):
                raise RuntimeError('Ordinary support check failed:\n' + stdout[-5000:])
            records.append([list(map(float, row[3:])) for row in rows])
    current = records[1]
    passed = (records[0] == current and current[0][0] == current[0][1] and
              current[1][0] != current[1][1] and current[3][2] == current[4][2] == 0.)
    return dict(schema='macnlops-local-h-scale-skip-check-v1', passed=passed,
        driver_sha256=hashlib.sha256((ROOT /
            'Template/NLO/SubProcesses/driver_macnlops.f').read_bytes()).hexdigest(),
        ordinary_bitwise_identical=records[0] == current, points=5,
        delta_calculation_trap=True,
        source_P_policy='ordinary bogus P retained; scalar explicit support configuration',
        source_energy_GeV=[.2, 3., 20., 20., 20.],
        cut_and_beam_K_zero=current[3][2] == current[4][2] == 0.,
        densities=current,
        scope='temporary drivers and existing DY objects; no generation or shower')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--trap-delta', action='store_true',
                        help='replace numerical Delta evaluation by an abort in temporary objects')
    parser.add_argument('--ordinary-scale-skip', action='store_true',
                        help='compare local H-scale preparation in ordinary G/P and support-edge checks')
    arguments = parser.parse_args()
    root = arguments.campaign_root.resolve()
    result = check_ordinary_scale_skip(root) if arguments.ordinary_scale_skip else check(
        root, arguments.trap_delta)
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text)
    print(text, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
