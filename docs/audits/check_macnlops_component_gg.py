#!/usr/bin/env python3
"""One native gg->ggg point: Born-leg transport and daughter-kernel closure.

With --process use an existing real-only ``g g > g g [real=QCD]`` export.
Without it, build a temporary export with at most two compiler workers.
No integration, source event generation, or Pythia shower is performed.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from docs.audits.check_macnlops_component_endpoints import build_driver, command
from docs.audits.check_macnlops_source_h import evaluator_source
from madgraph.various.macnlops_components import (
    branch_candidates, component_orbit_factors, transport_component_basis)


def fresh_export(work):
    from madgraph.interface.amcatnlo_run_interface import aMCatNLOCmd
    from madgraph.various.banner import RunCardNLO
    process = work / 'process'
    script = work / 'export.cmd'
    script.write_text('generate g g > g g [real=QCD]\noutput %s -f\nquit\n' % process)
    stages = [([sys.executable, str(ROOT / 'bin/mg5_aMC'), str(script)], ROOT, 'export.log')]
    for arguments, cwd, name in stages:
        with (work / name).open('w') as stream:
            result = subprocess.run(arguments, cwd=cwd, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise RuntimeError((work / name).read_text()[-8000:])
    card_path = process / 'Cards/run_card.dat'
    card = RunCardNLO(str(card_path))
    card['parton_shower'], card['mcatnlo_delta'], card['macnlops'] = 'PYTHIA8', True, False
    card['ickkw'], card['pdlabel'] = 0, 'nn23nlo'
    card['ebeam1'] = card['ebeam2'] = 6500.
    card.write(str(card_path))
    interface = aMCatNLOCmd(str(process))
    interface.do_treatcards('run')
    interface.do_treatcards('param')
    (process / 'SubProcesses/analyse_opts').write_text(
        'FO_ANALYSE=analysis_dummy.o dbook.o open_output_files_dummy.o HwU_dummy.o\n')
    for cwd, target, name in [(process / 'Source', [], 'source-build.log'),
                              (process / 'SubProcesses/P0_gg_gg',
                               ['macnlops_evaluator'], 'evaluator-build.log')]:
        with (work / name).open('w') as stream:
            result = subprocess.run(['make', '-j2'] + target, cwd=cwd,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise RuntimeError((work / name).read_text()[-8000:])
    return process


def momentum_particle(momentum, colours):
    return dict(id=21, status=1, mothers=[1, 2], colors=list(colours),
                p=list(momentum), mass=0.)


def request(real, selector, born=None):
    lines = ['MACNLOPS_POINT_V3 1 3 5']
    lines.extend('%d %s' % (p['id'], ' '.join('%.17e' % x for x in p['p'])) for p in real)
    lines.append('COMPONENT_LOCAL_V1 %d %d %d 1 1.' % selector)
    if born is not None:
        lines.append('COMPONENT_BORN_V1 4')
        lines.extend('%d %d %d %s %.17e' % (p['id'], *p['colors'],
                     ' '.join('%.17e' % x for x in p['p']), p['mass']) for p in born)
    return '\n'.join(lines + ['END_POINT']) + '\n'


def evaluate(worker, point):
    binary, directory, environment = worker
    stdout = command([str(binary)], directory, input=point, env=environment)
    rows = [line.split() for line in stdout.splitlines()
            if line.startswith('MACNLOPS_DENSITY_V1 ')]
    if len(rows) != 1 or int(rows[0][2]) != 0:
        raise RuntimeError('Native component failure:\n' + stdout[-8000:])
    return list(map(float, rows[0][3:])), stdout


def synthetic_record(real, born, radiator):
    source_indices, endpoint_indices = [3, 4, 5, 6], [7, 8, 9, 10, 11]
    source = [dict(p, status=-1 if i < 2 else 1) for i, p in enumerate(born)]
    after = [dict(p, colors=list(source[min(i, 3)]['colors'])) for i, p in enumerate(real)]
    colour, anticolour = source[radiator]['colors']
    after[radiator]['colors'], after[4]['colors'] = [colour, 999], [999, anticolour]
    snapshot = []
    for i, p in enumerate(source):
        snapshot.append(dict(p, index=source_indices[i],
                             status=-21 if i < 2 else -23, mothers=[0, 0]))
    for i, p in enumerate(after):
        parent = radiator if i == 4 else i
        snapshot.append(dict(p, index=endpoint_indices[i],
                             status=-21 if i < 2 else 23,
                             mothers=[source_indices[parent], 0]))
    return dict(emitted=True, emission_type='FSR', source_particles=source,
                particles=after, pythia_event=snapshot,
                source_pythia_indices=source_indices, hard_pythia_indices=endpoint_indices,
                branch_indices=dict(radiator_before=source_indices[radiator],
                    radiator_after=endpoint_indices[radiator], emitted=endpoint_indices[4]))


def check(process):
    registry = json.loads((process / 'SubProcesses/P0_gg_gg/mc_histories.json').read_text())
    basis = registry['registry']['outer']['component_basis']
    source_selector = (2, 5, 1)
    if not any(g['config'] == 2 and g['graph'] == 5 for g in basis['graphs']):
        raise RuntimeError('Unexpected pure-QCD gg Born diagram convention')
    source = evaluator_source()
    # Only this temporary audit driver prints its inverse Born. The physical
    # endpoint is fixed first; these rows construct two ancestry interpretations
    # of that same real point, not a second independent phase-space point.
    needle = '            this_config=iconfig\n'
    instrumented = source.replace(needle, needle + '''            if(local_component)then
               do leg=1,nexternal-1
                  write(*,'(a,1x,i6,4(1x,es26.17e3))')
     $                 'AUDIT_NATIVE_BORN_V1',leg,p_born(:,leg)
               enddo
            endif
''', 1)
    # A separate temporary control removes only the native daughter partition
    # from K. Source real/G terms and all numerical maps stay identical.
    unpartitioned = instrumented.replace('      sevmc=fks_Hij(p,i_fks,j_fks)\n',
                                         '      sevmc=1d0\n', 1)
    if unpartitioned == instrumented:
        raise RuntimeError('Cannot isolate the raw-kernel daughter partition')
    energy, incoming = 150., 500.
    final = (2 * incoming - energy) / 2
    transverse = math.sqrt(final**2 - (energy / 2)**2)
    momenta = [[incoming, 0., 0., incoming], [incoming, 0., 0., -incoming],
               [final, -energy / 2, transverse, 0.],
               [final, -energy / 2, -transverse, 0.], [energy, energy, 0., 0.]]
    real = [momentum_particle(p, [0, 0]) for p in momenta]
    result = dict(schema='macnlops-gg-component-point-v1',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  driver_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  registry_sha256=registry['sha256'],
                  scope='one synthetic real point; no integration or Pythia shower',
                  real_momenta=momenta, source_selector=list(source_selector),
                  delta_calculation_trap=True, emitters=[])
    with contextlib.ExitStack() as stack:
        worker = build_driver(process.parent, process.name, 'P0_gg_gg', stack,
                              trap_delta=True, source_text=instrumented)
        full_worker = build_driver(process.parent, process.name, 'P0_gg_gg', stack,
                                   trap_delta=True, source_text=unpartitioned)
        for radiator in (2, 3):
            # The reduced native FKS representative has its Born father in slot4.
            native_order = [0, 1, 3, 2, 4] if radiator == 2 else [0, 1, 2, 3, 4]
            _, stdout = evaluate(worker, request([real[i] for i in native_order], source_selector))
            projected = [list(map(float, row.split()[2:])) for row in stdout.splitlines()
                         if row.startswith('AUDIT_NATIVE_BORN_V1 ')]
            if len(projected) != 4:
                raise RuntimeError('Missing native Born projection')
            born_order = [0, 1, 3, 2] if radiator == 2 else [0, 1, 2, 3]
            born = [momentum_particle(projected[i], colours)
                    for i, colours in zip(born_order, basis['flows'][0])]
            record = synthetic_record(real, born, radiator)
            candidates = branch_candidates(record, registry, 'P0_gg_gg')
            if len(candidates) != 2:
                raise RuntimeError('Expected both certified gg daughter orientations')
            orientations = []
            for route in candidates:
                selector = transport_component_basis(registry, route.born_permutation,
                                                      *source_selector)
                point = request([record['particles'][i] for i in route.permutation], selector,
                                [record['source_particles'][i] for i in route.born_permutation])
                density, _ = evaluate(worker, point)
                full_density, _ = evaluate(full_worker, point)
                normalization, emitters, source_orbit = component_orbit_factors(registry, route.sector)
                orientations.append(dict(orientation=route.orientation,
                    born_permutation=list(route.born_permutation),
                    endpoint_permutation=list(route.permutation), selector=list(selector),
                    normalization=str(normalization), equivalent_emitters=emitters,
                    source_orbit=source_orbit, densities=density,
                    unpartitioned_kernel=full_density[2]))
            total = sum(o['densities'][2] for o in orientations)
            reference = orientations[0]['unpartitioned_kernel']
            symmetry_error = abs(reference - orientations[1]['unpartitioned_kernel']) / abs(reference)
            closure_error = abs(total - reference) / abs(reference)
            if min(total, reference) <= 0 or max(symmetry_error, closure_error) > 1.e-10:
                raise RuntimeError('Numerical gg kernel closure failed: %s' % orientations)
            result['emitters'].append(dict(source_radiator=radiator + 1,
                orientations=orientations, kernel_closure_relative_error=closure_error,
                unpartitioned_exchange_relative_error=symmetry_error))
    transported = result['emitters'][0]['orientations'][0]
    if transported['selector'] == list(source_selector):
        raise RuntimeError('The Born permutation failed to exercise graph/flow transport')
    result['passed'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--process', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with contextlib.ExitStack() as stack:
        process = args.process.resolve() if args.process else fresh_export(Path(
            stack.enter_context(tempfile.TemporaryDirectory(prefix='macnlops-gg-component-'))))
        result = check(process)
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
