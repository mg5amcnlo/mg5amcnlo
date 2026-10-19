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
from functools import partial
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


def conditional_case(directory, points, seed=412):
    import numpy as np
    rng = np.random.RandomState(seed)
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


def load_reference(ref, temporary, source=None):
    name = '_virtual_training_reference'
    directory = temporary/name
    directory.mkdir()
    (directory/'__init__.py').write_text('')
    for module in ('virtual_pool', 'virtual_pool_polynomial', 'virtual_pool_neural'):
        relative = Path('madgraph/various')/(module+'.py')
        contents = ((source/relative).read_bytes() if source is not None else
                    subprocess.check_output(['git', 'show', '%s:%s' % (ref, relative)], cwd=ROOT))
        (directory/(module+'.py')).write_bytes(contents)
    sys.path.insert(0, str(temporary))
    return importlib.import_module(name+'.virtual_pool')


def measured(repeats, run):
    times = []
    for _ in range(repeats):
        start = time.process_time()
        result = run()
        times.append(time.process_time()-start)
    return dict(cpu_seconds=times, median_cpu_seconds=statistics.median(times)), result


def _prediction_errors(target, prediction, weight=None):
    import numpy as np
    residual = target-prediction
    weight = np.ones(len(target)) if weight is None else np.asarray(weight)
    weight = weight/weight.sum()
    scale = float(weight@abs(target))
    ordered = np.argsort(abs(residual), kind='stable')
    quantile = np.searchsorted(np.cumsum(weight[ordered]), .99)
    return dict(relative_l1=float(weight@abs(residual)/scale),
                relative_rms=float(np.sqrt((weight@(residual**2))/(weight@(target**2)))),
                relative_p99=float(abs(residual[ordered[min(quantile, len(ordered)-1)]])/scale),
                relative_maximum=float(np.max(abs(residual))/scale))


def _benchmark_polynomial(modules, points, nlegs, repeats):
    import numpy as np
    metadata, rows, x, truth = polynomial_case(points, nlegs)
    result = dict(points=points, features=metadata['nf'], implementations={})
    predictions = {}
    for label, pool in modules.items():
        poly = importlib.import_module(pool.__package__+'.virtual_pool_polynomial')
        folds = pool._folds(rows, 3)
        arguments = dict(strategy='compact') if hasattr(poly, 'COMPACT_RIDGE') else {}
        run = partial(poly.fit_order, rows, folds, metadata, 1, **arguments)
        timing, (entry, detail) = measured(repeats, run)
        prediction = poly.predict(entry, x, np.ones(len(x)), np.zeros(len(x)))
        predictions[label] = prediction
        timing.update(_prediction_errors(truth, prediction))
        timing.update(selected=detail.get('selected'), actual_terms=len(entry['terms']))
        result['implementations'][label] = timing
    if 'reference' in predictions:
        result['maximum_prediction_difference'] = float(np.max(abs(
            predictions['current']-predictions['reference'])))
    return result


def _benchmark_conditional(modules, points, repeats, temporary):
    import numpy as np
    job = temporary/'capture'
    conditional_case(job, points)
    heldout = temporary/'heldout'
    conditional_case(heldout, 2000, seed=719)
    result = dict(points=points, implementations={})
    for label, pool in modules.items():
        model = temporary/(label+'_model.dat')
        neural = temporary/(label+'_neural.dat')
        run = partial(pool.train_pool, [job], model, neural=True, output_neural=neural)
        timing, detail = measured(repeats, run)
        timing.update(conditional=detail.get('neural'), polynomial=detail['orders'])
        poly = importlib.import_module(pool.__package__+'.virtual_pool_polynomial')
        conditional = importlib.import_module(pool.__package__+'.virtual_pool_neural')
        _, training, _ = pool.collect_samples([job])
        metadata, rows, _ = pool.collect_samples([heldout])
        if hasattr(pool, '_physics_rows'):
            metadata, rows, _ = pool._physics_rows(metadata, rows)
        # Replay the training MINT history, then predict independent rows with
        # the final frozen marginals, common model and conditional companion.
        mint_state = poly.Marginals(metadata['nx'], 8)
        for iteration in sorted({r['iteration'] for r in training}):
            mint_state.update([r for r in training if r['iteration'] == iteration])
        mint = np.asarray([mint_state.predict(r) for r in rows])
        x, b, y = (np.asarray([r[k] for r in rows]) for k in ('features', 'born', 'virtual'))
        common = pool.load_model(model)
        if neural.exists():
            fitted = conditional.load_model(neural, model)
            prediction, _ = conditional.predict_model(
                fitted, common, 1, x,
                np.asarray([r['momenta'] for r in rows]).reshape(len(rows), metadata['nlegs'], 4),
                np.asarray([r['q'] for r in rows]), b, mint,
                np.asarray([r['selected_helicity'] for r in rows]),
                np.asarray([r['full_branch'] for r in rows]))
        else:
            prediction = pool.predict(common, 1, x, b, mint)
        timing['heldout'] = _prediction_errors(y, prediction)
        result['implementations'][label] = timing
    return result


def _benchmark_helicity(modules, points, nheads, repeats):
    import numpy as np
    inputs, mixing, born, target = helicity_case(points, nheads)
    result = dict(points=points, implementations={},
                  selection='standalone linear versus nonlinear fits; no automatic fallback in this case')
    try:
        import scipy.optimize
    except ImportError:
        result['implementations'] = {label: dict(reason='scipy_unavailable') for label in modules}
        return result
    for label, pool in modules.items():
        neural = importlib.import_module(pool.__package__+'.virtual_pool_neural')
        linear = hasattr(neural, 'LINEAR_CONFIGURATION')
        configuration = neural.LINEAR_CONFIGURATION if linear else neural.DEFAULT_CONFIGURATIONS[0]
        basis = np.column_stack((inputs, inputs**2)) if linear else inputs
        run = partial(neural.fit, basis[:points], mixing[:points], born[:points],
                      target[:points], np.ones(points), list(range(1, nheads+1)),
                      [1]*nheads, configuration)
        timing, entry = measured(repeats, run)
        prediction = (born[points:]*entry['target_scale']/entry['born_scale'])*np.sum(
            mixing[points:]*neural.forward(entry, basis[points:]), axis=1)
        timing.update(_prediction_errors(target[points:], prediction))
        timing.update(configuration=configuration, optimizer_iterations=entry['optimizer_iterations'])
        result['implementations'][label] = timing
    return result


def benchmark(modules, points, conditional_points, repeats, temporary):
    results = {'polynomial_%d_legs' % nlegs: _benchmark_polynomial(modules, points, nlegs, repeats)
               for nlegs in (4, 6)}
    results['conditional'] = _benchmark_conditional(modules, conditional_points, repeats, temporary)
    for nheads in (16, 64):
        results['nonlinear_%d_heads' % nheads] = _benchmark_helicity(modules, points, nheads, repeats)
    for result in results.values():
        runs = result['implementations']
        if 'reference' in runs and 'median_cpu_seconds' in runs['reference']:
            result['speedup'] = runs['reference']['median_cpu_seconds']/runs['current']['median_cpu_seconds']
    return results


def benchmark_captures(modules, directories, repeats, temporary):
    """Train on earlier native observations and test on their later suffix."""
    import numpy as np
    output = dict(captures=list(map(str, directories)), implementations={})
    current = modules['current']
    captured_metadata, captured_rows, _ = current.collect_samples(directories)
    if captured_metadata is None or not captured_rows:
        raise ValueError('native benchmark needs committed captures')
    for label, pool in modules.items():
        poly = importlib.import_module(pool.__package__+'.virtual_pool_polynomial')
        conditional = importlib.import_module(pool.__package__+'.virtual_pool_neural')
        metadata, rows = captured_metadata, captured_rows
        if metadata['catalog'] == 9 and not hasattr(pool, 'COLOUR_FEATURE_CATALOG'):
            # The same physical observations, projected onto the older input
            # catalog. Keep the native Born and V, probabilities and chronology;
            # only remove the newly captured colour and threshold coordinates.
            first = (3+metadata['nlegs']*(metadata['nlegs']-1)//2+
                     len({m for m in metadata['masses'] if m > 0})-1)
            extra = len(current.threshold_pairs(metadata))+4
            metadata = dict(metadata, version=3, catalog=7, nf=metadata['nf']-extra)
            metadata.pop('colours')
            rows = [dict(r, features=r['features'][:first]+r['features'][first+extra:]) for r in rows]
        if hasattr(pool, '_physics_rows'):
            metadata, rows, _ = pool._physics_rows(metadata, rows)
        train = conditional.chronological_mask(rows, 2./3.)
        earlier = [r for r, keep in zip(rows, train) if keep]
        base = temporary/(label+'_native_model.dat')
        heads = temporary/(label+'_native_heads.dat')
        strategy = dict(strategy='compact') if hasattr(poly, 'COMPACT_RIDGE') else {}

        def run():
            entries, reports = [], []
            folds = pool._folds(earlier, 3)
            for order in range(1, metadata['norders']+1):
                entry, report = poly.fit_order(earlier, folds, metadata, order, **strategy)
                entries.append(entry)
                reports.append(report)
            pool._atomic_write(base, pool._model_text(metadata, entries))
            detail = conditional.train(earlier, metadata, entries, reports, base, heads)
            return entries, reports, detail

        timing, (entries, reports, detail) = measured(repeats, run)
        common = pool.load_model(base)
        fitted = conditional.load_model(heads, base) if heads.exists() else None
        timing.update(training_rows=len(earlier), heldout_rows=int(sum(~train)),
                      common=reports, conditional=detail, heldout=[])
        assignment = np.asarray([r['order'] for r in rows])
        for order, entry in enumerate(entries, 1):
            selected = assignment == order
            subset = [r for r, keep in zip(rows, selected) if keep]
            heldout = ~train[selected]
            x, b, y, w = (np.asarray([r[k] for r in subset]) for k in ('features', 'born', 'virtual', 'weight'))
            if not np.any(y[heldout]):
                continue
            mint = poly.mint_predictions(subset, train[selected])
            if fitted is not None:
                prediction, _ = conditional.predict_model(
                    fitted, common, order, x,
                    np.asarray([r['momenta'] for r in subset]).reshape(len(subset), metadata['nlegs'], 4),
                    np.asarray([r['q'] for r in subset]), b, mint,
                    np.asarray([r['selected_helicity'] for r in subset]),
                    np.asarray([r['full_branch'] for r in subset]))
            else:
                prediction = pool.predict(common, order, x, b, mint)
            timing['heldout'].append(dict(order=order,
                **_prediction_errors(y[heldout], prediction[heldout], w[heldout])))
        output['implementations'][label] = timing
    if 'reference' in output['implementations']:
        output['speedup'] = (output['implementations']['reference']['median_cpu_seconds']/
                             output['implementations']['current']['median_cpu_seconds'])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument('--reference-ref', help='git revision whose trainer to compare')
    reference.add_argument('--reference-source', type=Path,
                           help='frozen checkout sources whose trainer to compare')
    parser.add_argument('--points', type=int, default=3000)
    parser.add_argument('--conditional-points', type=int, default=900)
    parser.add_argument('--captures', type=Path, nargs='+',
                        help='benchmark chronological holdout on these native job directories instead of synthetic cases')
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
        if args.reference_ref or args.reference_source:
            modules['reference'] = load_reference(args.reference_ref, temporary, args.reference_source)
        modules['current'] = virtual_pool
        results = (dict(native=benchmark_captures(modules, args.captures, args.repeats, temporary))
                   if args.captures else
                   benchmark(modules, args.points, args.conditional_points, args.repeats, temporary))
        result = dict(reference_ref=args.reference_ref,
                      reference_source=str(args.reference_source) if args.reference_source else None,
                      single_cpu=True, results=results)
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
