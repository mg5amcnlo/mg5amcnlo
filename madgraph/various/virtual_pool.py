"""Train frozen virtual approximations at the first integration barrier.

The integration controller groups jobs by Born subprocess before calling this
module. No workers communicate while integrating. Importing this module does
not require numpy; a controller without it retains the MINT fallback.
"""

import hashlib
import json
import math
import os
import tempfile
import time


SAMPLES = 'virtual_pool_samples.dat'
ITERATIONS = 'virtual_pool_iterations.dat'
FORMAT_VERSION = 2
FEATURE_CATALOG = 5


def _floats(line):
    result = tuple(float(x.replace('D', 'E').replace('d', 'e')) for x in line.split())
    if not all(math.isfinite(x) for x in result):
        raise ValueError('nonfinite virtual-pool input')
    return result


def _header(stream, magic):
    if stream.readline().strip() != '%s %d' % (magic, FORMAT_VERSION):
        raise ValueError('unsupported virtual-pool file header; restart integration')
    dimensions = tuple(int(x) for x in stream.readline().split())
    if len(dimensions) != 5:
        raise ValueError('invalid virtual-pool dimensions')
    catalog, nf, norders, nlegs, nx = dimensions
    pdgs = tuple(int(x) for x in stream.readline().split())
    masses = _floats(stream.readline())
    incoming = tuple(int(x) for x in stream.readline().split())
    expected = 1 + nlegs*(nlegs-1)//2 + len({m for m in masses if m > 0})
    if (catalog != FEATURE_CATALOG or nf != expected or not 1 <= nf <= 2048
            or not 3 <= nlegs <= 64 or not 1 <= norders <= 1024 or not 1 <= nx <= 256
            or any(len(x) != nlegs for x in (pdgs, masses, incoming))
            or any(x < 0 for x in masses) or any(x not in (0, 1) for x in incoming)
            or sum(incoming) != 2):
        raise ValueError('invalid virtual-pool particle metadata')
    return dict(version=FORMAT_VERSION, catalog=catalog, nf=nf, norders=norders,
                nlegs=nlegs, nx=nx, pdgs=pdgs, masses=masses, incoming=incoming)


def _read_iterations(directory):
    ledger = os.path.join(directory, ITERATIONS)
    if not os.path.isfile(ledger):
        return None
    iterations = {}
    with open(ledger) as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError('%s:%s: invalid iteration ledger' % (ledger, number))
            iteration, accepted, attempted = (int(x) for x in fields)
            if iteration < 1 or accepted not in (0, 1) or attempted < 0 or iteration in iterations:
                raise ValueError('%s:%s: invalid/duplicate iteration' % (ledger, number))
            iterations[iteration] = (accepted, attempted)
    return iterations


def _read_job(directory):
    path = os.path.join(directory, SAMPLES)
    iterations = _read_iterations(directory)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        attempts = sum(points for accepted, points in (iterations or {}).values() if accepted)
        return None, [], attempts, dict(
            directory=directory,
            status='empty_samples' if os.path.isfile(path) else 'missing_samples',
            accepted_attempts=attempts)
    with open(path) as stream:
        metadata = _header(stream, 'VIRTUAL_POOL_SAMPLES')
        rows = []
        seen = set()
        for number, line in enumerate(stream, 6):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != metadata['nf'] + 9 + metadata['nx']:
                raise ValueError('%s:%s: incorrect sample width' % (path, number))
            sample, channel, iteration, order = (int(v) for v in fields[:4])
            probability, born, virtual, baseline, *values = _floats(' '.join(fields[5:]))
            features = values[:metadata['nf']]
            if (sample < 1 or channel < 1 or iteration < 1
                    or not 1 <= order <= metadata['norders']
                    or not 0 < probability <= 1):
                raise ValueError('%s:%s: invalid sample metadata' % (path, number))
            key = (sample, channel, iteration, order)
            if key in seen:
                raise ValueError('%s:%s: duplicate sample/order' % (path, number))
            seen.add(key)
            row = dict(sample=sample, channel=channel, iteration=iteration,
                       order=order, probability=probability, born=born,
                       virtual=virtual, baseline=baseline, features=features,
                       directory=directory)
            row.update(bins=int(fields[4]), x=values[metadata['nf']:])
            if row['bins'] not in (4, 8) or any(v < 0 or v > 1 for v in row['x']):
                raise ValueError('invalid MINT coordinates/bins')
            if rows and (iteration, sample) < (rows[-1]['iteration'], rows[-1]['sample']):
                raise ValueError('unordered pooled sample history')
            rows.append(row)
    ledger = os.path.join(directory, ITERATIONS)
    if iterations is None:
        # A failed/incomplete integration must never contribute uncommitted rows.
        return metadata, [], 0, dict(directory=directory, status='missing_iterations',
                                     discarded_rows=len(rows))
    accepted_rows = [row for row in rows if iterations.get(row['iteration'], (0, 0))[0]]
    attempts = sum(points for accepted, points in iterations.values() if accepted)
    if accepted_rows and attempts == 0:
        raise ValueError('%s: accepted samples but no attempted points' % ledger)
    return metadata, accepted_rows, attempts, dict(
        directory=directory, status='read', captured_rows=len(rows),
        accepted_rows=len(accepted_rows), discarded_rows=len(rows) - len(accepted_rows),
        accepted_attempts=attempts)


def collect_samples(job_dirs, job_groups=None):
    """Read committed rows, normalized by attempts and loop-call probability.

    ``job_groups`` maps directory names to a stable integration-channel-batch
    key. Split replicas of one batch share an attempted-point denominator;
    disjoint batches contribute separate mean residuals. Without this mapping,
    each directory is its own group, suitable for unsplit independent jobs.
    The rows already include their integration factors: never multiply a
    Jacobian, luminosity or helicity weight into them a second time. The
    bookkeeping-only fixed-order wgt_mult must be removed before capture.
    """
    directories = sorted(set(os.path.abspath(os.fspath(p)) for p in job_dirs))
    groups = {os.path.abspath(os.fspath(k)): str(v) for k, v in (job_groups or {}).items()}
    if job_groups is not None and any(p not in groups for p in directories):
        raise ValueError('job_groups must identify every supplied job directory')
    metadata, rows, reports, denominators = None, [], [], {}
    for directory in directories:
        header, accepted, attempted, report = _read_job(directory)
        if header is not None:
            # Workers serialize masses consistently. Keep different Born
            # orderings distinct, even if their external flavours agree.
            if metadata is not None and metadata != header:
                raise ValueError('incompatible Born subprocess metadata in %s' % directory)
            metadata = header
        group = groups.get(directory, directory)
        report['group'] = group
        reports.append(report)
        denominators[group] = denominators.get(group, 0) + attempted
        for row in accepted:
            row['group'] = group
        rows.extend(accepted)
    for row in rows:
        row['weight'] = 1. / denominators[row['group']] / row['probability']
        if not math.isfinite(row['weight']):
            raise ValueError('virtual-pool statistical weight overflow')
    return metadata, rows, reports


def _folds(rows, count):
    """Balance each job/channel/iteration; all orders of a sample stay together."""
    strata = {}
    for row in rows:
        stratum = (row['directory'], row['channel'], row['iteration'])
        strata.setdefault(stratum, set()).add(row['sample'])
    assignments = {}
    for stratum in sorted(strata):
        prefix = repr(stratum).encode('utf8')
        offset = int(hashlib.sha256(prefix).hexdigest()[:8], 16) % count
        samples = sorted(strata[stratum], key=lambda sample: hashlib.sha256(
            prefix + str(sample).encode('ascii')).digest())
        for index, sample in enumerate(samples):
            assignments[stratum + (sample,)] = (index + offset) % count
    return [assignments[(r['directory'], r['channel'], r['iteration'], r['sample'])]
            for r in rows]


def _atomic_write(path, text):
    path = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + os.path.basename(path) + '.', dir=directory)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _model_text(metadata, orders):
    floats = lambda values: ' '.join('%.17e' % x for x in values)
    lines = ['VIRTUAL_POOL_MODEL %d' % FORMAT_VERSION,
             '%d %d %d %d %d' % tuple(metadata[k] for k in
                                      ('catalog', 'nf', 'norders', 'nlegs', 'nx')),
             ' '.join(str(x) for x in metadata['pdgs']), floats(metadata['masses']),
             ' '.join(str(x) for x in metadata['incoming'])]
    for entry in orders:
        lines.append('%d %.17e %.17e %.17e %d %d' %
                     (entry['order'], entry['blend'], entry['born_scale'], entry['target_scale'],
                      len(entry['terms']), entry['degree']))
        lines.extend([floats(entry['lower']), floats(entry['upper'])])
        for term in entry['terms']:
            lines.append('%d %d %.17e' % (term['pair'][0]+1, term['pair'][1]+1, term['coefficient']))
    return '\n'.join(lines) + '\n'


def train_pool(job_dirs, output_model, report_path=None, job_groups=None):
    """Fit one frozen model per coupling order after the first-stage barrier.

    Missing or insufficient captures give an explicit MINT fallback. Malformed
    files or incompatible Born metadata raise ValueError, never a silent merge.
    ``model_written`` indicates whether inference may load the model; ``enabled``
    indicates whether any order selected a nonzero blend. An all-zero model is
    a valid frozen fallback. No held-out evaluation data are accepted here.
    """
    from . import virtual_pool_polynomial as polynomial

    started = time.monotonic()
    metadata, rows, jobs = collect_samples(job_dirs, job_groups)
    report = dict(version=FORMAT_VERSION, status='skipped', enabled=False, model_written=False,
                  family='born_invariant_polynomial', accepted_rows=len(rows), jobs=jobs, orders=[],
                  accepted_samples=len({(r['directory'], r['channel'], r['iteration'], r['sample'])
                                        for r in rows}),
                  objective='physical residual; inverse accepted attempts and loop probability',
                  validation='matched three-fold MINT/model CV; paired L1 gain and RMS guards',
                  tail_margin=polynomial.TAIL_MARGIN, max_coefficients=polynomial.MAX_TERMS,
                  minimum_rows_per_coefficient=polynomial.ROWS_PER_TERM,
                  call_probability='retain first-stage fraction; exact loop outside trained domain')
    if metadata is None:
        report['reason'] = 'no_sample_headers'
    else:
        folds = _folds(rows, 3)
        orders = []
        for order in range(1, metadata['norders'] + 1):
            entry, diagnostics = polynomial.fit_order(rows, folds, metadata, order)
            orders.append(entry)
            diagnostics.update(blend=entry['blend'], reason=entry['reason'])
            report['orders'].append(diagnostics)
        _atomic_write(output_model, _model_text(metadata, orders))
        report.update(metadata=metadata, model_written=True,
                      enabled=any(o['blend'] > 0 for o in orders))
        report['status'] = 'trained' if report['enabled'] else 'fallback'
    report['training_seconds'] = time.monotonic()-started
    if report_path is not None:
        _atomic_write(report_path, json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return report


def load_model(path):
    """Strict Python reader matching the frozen Fortran interchange schema."""
    from .virtual_pool_polynomial import MAX_TERMS

    with open(path) as stream:
        metadata = _header(stream, 'VIRTUAL_POOL_MODEL')
        orders = []
        for expected_order in range(1, metadata['norders'] + 1):
            fields = stream.readline().split()
            if len(fields) != 6:
                raise ValueError('invalid pooled model order header')
            order, nterms, degree = int(fields[0]), int(fields[4]), int(fields[5])
            blend, born_scale, target_scale = _floats(' '.join(fields[1:4]))
            lower, upper = _floats(stream.readline()), _floats(stream.readline())
            if (order != expected_order or not 0 <= blend <= 1 or born_scale <= 0 or target_scale <= 0
                    or not 0 <= nterms <= MAX_TERMS or not 0 <= degree <= 2
                    or len(lower) != metadata['nf'] or len(upper) != metadata['nf']
                    or any(a > b for a, b in zip(lower, upper))):
                raise ValueError('invalid pooled model order metadata')
            if blend and not nterms:
                raise ValueError('active polynomial has no coefficients')
            terms = []
            for _ in range(nterms):
                fields = stream.readline().split()
                if len(fields) != 3:
                    raise ValueError('invalid polynomial term')
                pair = tuple(int(v)-1 for v in fields[:2])
                coefficient, = _floats(fields[2])
                if (min(pair) < -1 or max(pair) >= metadata['nf'] or
                        (pair[0] == -1 and pair[1] != -1) or
                        (degree < 2 and pair[1] >= 0) or (degree == 0 and pair[0] >= 0)):
                    raise ValueError('invalid polynomial indices/degree')
                terms.append(dict(pair=pair, coefficient=coefficient))
            orders.append(dict(order=order, blend=blend, born_scale=born_scale,
                               target_scale=target_scale, degree=degree,
                               lower=lower, upper=upper, terms=terms))
        if stream.read().strip():
            raise ValueError('unexpected trailing pooled model data')
    return dict(metadata=metadata, orders=orders)


def predict(model, order, features, born, mint):
    """Evaluate the frozen convex blend with the same normalization as inputs."""
    import numpy as np
    from .virtual_pool_polynomial import predict as polynomial_predict

    if not 1 <= order <= model['metadata']['norders']:
        raise ValueError('invalid pooled model order')
    entry = model['orders'][order - 1]
    x, born, mint = (np.asarray(values, dtype=float) for values in (features, born, mint))
    if (born.ndim != 1 or x.shape != (len(born), model['metadata']['nf']) or mint.shape != born.shape
            or not all(np.all(np.isfinite(v)) for v in (x, born, mint))):
        raise ValueError('invalid pooled prediction inputs')
    return polynomial_predict(entry, x, born, mint)
