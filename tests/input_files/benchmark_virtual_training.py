"""Compare virtual-training CPU cost on reproducible synthetic captures.

Run from the repository root, for example::

    python3 tests/input_files/benchmark_virtual_training.py \
        --reference-ref 849759f5c --repeats 3 --output /tmp/virtual-training.json

The benchmark pins itself to one available CPU and limits BLAS threads before
importing NumPy. Reference Python sources are read from git into a temporary
package; the checkout is never changed. These synthetic timings and residuals
measure training behaviour, not event-generation performance for a process.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]


def polynomial_case(points, nlegs):
    import numpy as np
    catalog = 5 if nlegs == 4 else 6
    nf = 1+nlegs*(nlegs-1)//2+1+(2 if catalog == 6 else 0)
    metadata = dict(version=2, catalog=catalog, nf=nf, norders=1,
                    nlegs=nlegs, nx=2, masses=(0.,)*(nlegs-2)+(173., 173.),
                    pdgs=(21,)*(nlegs-2)+(6, -6), incoming=(1, 1)+(0,)*(nlegs-2))
    rng = np.random.RandomState(4901+nlegs)
    x = rng.uniform(-1, 1, (points, nf))
    x[:, 1] = 0.
    born = rng.uniform(.2, 3., points)
    born[::13] *= -1
    born[::17] = 0.
    virtual = born*(.2+x[:, 0]*x[:, 2]+.6*x[:, 3]**2)
    mint = rng.uniform(size=(points, 2))
    rows = [dict(sample=i+1, channel=i%3+1, iteration=i//200+1, order=1,
                 probability=.5, born=born[i], virtual=virtual[i], baseline=0.,
                 features=x[i].tolist(), x=mint[i].tolist(), bins=8,
                 directory='benchmark', weight=1./points) for i in range(points)]
    test_x = rng.uniform(-1, 1, (2000, nf))
    test_x[:, 1] = 0.
    truth = .2+test_x[:, 0]*test_x[:, 2]+.6*test_x[:, 3]**2
    return metadata, rows, test_x, truth


def conditional_case(directory, points):
    import numpy as np
    rng = np.random.RandomState(412)
    lines = ['VIRTUAL_POOL_SAMPLES 3', '7 11 1 4 2', '21 21 6 -6',
             '0 0 173 173', '1 1 0 0', '3', '1 2 1 -1 1 -1',
             '2 0 -1 1 1 -1', '3 1 1 1 -1 -1']
    momenta = [300., 0., 0., 300., 300., 0., 0., -300.,
               300., 200., 0., 141.38953285, 300., -200., 0., -141.38953285]
    for i in range(points):
        x = [float(rng.uniform(-1, 1))]+[0.]*10
        born = rng.uniform(.2, 1.5)*(-1 if i%13 == 0 else 1)
        selected = 1 if rng.random_sample() < .6 else 3
        full = i%11 == 0
        target = born*(.2*x[0]+(.6*.8-.4*.9 if full else .8 if selected == 1 else -.9))
        q = [.6, 0., .4] if i%37 else [0., 0., 0.]
        row = [i+1, 1, 1+i//300, 1, 8, 1., born, target, 0.]+x+rng.random_sample(2).tolist()
        row += [int(full), -1 if full else selected, 1. if full else .6 if selected == 1 else .4]+q+momenta
        lines.append(' '.join(map(str, row)))
    directory.mkdir()
    (directory/'virtual_pool_samples.dat').write_text('\n'.join(lines)+'\n')
    ledger = ''.join('%d 1 %d\n' % (i//300+1, min(300, points-i))
                     for i in range(0, points, 300))
    (directory/'virtual_pool_iterations.dat').write_text(ledger)


def helicity_case(points, nheads=16):
    import numpy as np
    rng = np.random.RandomState(1042)
    inputs = rng.uniform(-1., 1., (points+2000, 12))
    born = rng.uniform(.2, 2., len(inputs))
    born[::13] *= -1
    born[::31] = 0.
    phase = np.linspace(-1., 1., nheads)
    q = np.exp(.4*inputs[:, :1]*phase)
    q /= q.sum(axis=1)[:, None]
    selected = (rng.uniform(size=len(inputs))[:, None] > np.cumsum(q, axis=1)).sum(axis=1)
    mixing = np.zeros_like(q)
    mixing[np.arange(len(inputs)), selected] = 1.
    mixing[::11] = q[::11]
    heads = (.3*phase + .2*inputs[:, :1] +
             .4*np.sin(2*inputs[:, 1:2]+phase) +
             .3*inputs[:, 2:3]*inputs[:, 3:4]*phase)
    target = born*np.sum(mixing*heads, axis=1)
    return inputs, mixing, born, target


def load_reference(ref, temporary):
    name = '_virtual_training_reference'
    directory = temporary/name
    directory.mkdir()
    (directory/'__init__.py').write_text('')
    for module in ('virtual_pool', 'virtual_pool_polynomial', 'virtual_pool_neural'):
        source = subprocess.check_output(
            ['git', 'show', '%s:madgraph/various/%s.py' % (ref, module)], cwd=ROOT)
        (directory/(module+'.py')).write_bytes(source)
    sys.path.insert(0, str(temporary))
    return importlib.import_module(name+'.virtual_pool')


def measured(repeats, run):
    times = []
    for _ in range(repeats):
        start = time.process_time()
        result = run()
        times.append(time.process_time()-start)
    return dict(cpu_seconds=times, median_cpu_seconds=statistics.median(times)), result


def benchmark(modules, points, conditional_points, repeats, temporary):
    import numpy as np
    results = {}
    for nlegs in (4, 6):
        metadata, rows, x, truth = polynomial_case(points, nlegs)
        name = 'polynomial_%d_legs' % nlegs
        results[name] = dict(points=points, features=metadata['nf'], implementations={})
        predictions = {}
        for label, pool in modules.items():
            poly = importlib.import_module(pool.__package__+'.virtual_pool_polynomial')
            folds = pool._folds(rows, 3)
            timing, (entry, detail) = measured(repeats, lambda: poly.fit_order(rows, folds, metadata, 1))
            prediction = poly.predict(entry, x, np.ones(len(x)), np.zeros(len(x)))
            predictions[label] = prediction
            residual = truth-prediction
            timing.update(relative_l1=float(np.mean(abs(residual))/np.mean(abs(truth))),
                          relative_rms=float(np.linalg.norm(residual)/np.linalg.norm(truth)),
                          selected=detail.get('selected'), actual_terms=len(entry['terms']))
            results[name]['implementations'][label] = timing
        if 'reference' in predictions:
            results[name]['maximum_prediction_difference'] = float(np.max(abs(
                predictions['current']-predictions['reference'])))

    job = temporary/'capture'
    conditional_case(job, conditional_points)
    results['conditional'] = dict(points=conditional_points, implementations={})
    for label, pool in modules.items():
        model = temporary/(label+'_model.dat')
        neural = temporary/(label+'_neural.dat')
        timing, detail = measured(repeats, lambda: pool.train_pool([job], model, neural=True, output_neural=neural))
        timing.update(conditional=detail.get('neural'), polynomial=detail['orders'])
        results['conditional']['implementations'][label] = timing

    for nheads in (16, 64):
        inputs, mixing, born, target = helicity_case(points, nheads)
        name = 'nonlinear_%d_heads' % nheads
        results[name] = dict(points=points, implementations={})
        for label, pool in modules.items():
            neural = importlib.import_module(pool.__package__+'.virtual_pool_neural')
            try:
                import scipy.optimize
            except ImportError:
                results[name]['implementations'][label] = dict(reason='scipy_unavailable')
                continue
            configuration = neural.DEFAULT_CONFIGURATIONS[0]
            timing, entry = measured(repeats, lambda: neural.fit(
                inputs[:points], mixing[:points], born[:points], target[:points],
                np.ones(points), list(range(1, nheads+1)), [1]*nheads, configuration))
            prediction = (born[points:]*entry['target_scale']/entry['born_scale'])*np.sum(
                mixing[points:]*neural.forward(entry, inputs[points:]), axis=1)
            residual = target[points:]-prediction
            timing.update(relative_l1=float(np.mean(abs(residual))/np.mean(abs(target[points:]))),
                          relative_rms=float(np.linalg.norm(residual)/np.linalg.norm(target[points:])),
                          configuration=configuration, optimizer_iterations=entry['optimizer_iterations'])
            results[name]['implementations'][label] = timing
    for result in results.values():
        runs = result['implementations']
        if 'reference' in runs and 'median_cpu_seconds' in runs['reference']:
            result['speedup'] = runs['reference']['median_cpu_seconds']/runs['current']['median_cpu_seconds']
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-ref', help='git revision whose trainer to compare')
    parser.add_argument('--points', type=int, default=3000)
    parser.add_argument('--conditional-points', type=int, default=900)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.points < 600 or args.conditional_points < 600 or args.repeats < 1:
        parser.error('use at least 600 points per case and one repeat')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    sys.path.insert(0, str(ROOT))
    from madgraph.various import virtual_pool
    # Warm imports before timing either implementation, including the optional
    # optimised solver; neural fitting itself retains its SciPy fallback.
    try:
        import scipy.linalg
        import scipy.optimize
    except ImportError:
        pass
    with tempfile.TemporaryDirectory(prefix='virtual_training_') as path:
        temporary = Path(path)
        modules = {}
        if args.reference_ref:
            modules['reference'] = load_reference(args.reference_ref, temporary)
        modules['current'] = virtual_pool
        result = dict(reference_ref=args.reference_ref, single_cpu=True,
                      results=benchmark(modules, args.points, args.conditional_points, args.repeats, temporary))
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
