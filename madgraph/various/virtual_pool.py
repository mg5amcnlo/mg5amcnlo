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
MODEL_FORMAT_VERSION = 3
FEATURE_CATALOG = 6
LEGACY_FEATURE_CATALOG = 5
CONDITIONAL_FEATURE_CATALOG = 7
PHYSICS_FEATURE_CATALOG = 8
COLOUR_FEATURE_CATALOG = 9
CAPTURE_FORMAT_VERSION = 4


def scalar_feature_count(metadata):
    return (3 + metadata['nlegs']*(metadata['nlegs']-1)//2 +
            len({m for m in metadata['masses'] if m > 0}) +
            (len(threshold_pairs(metadata)) if metadata['catalog'] in (8, 9) else 0) +
            (4 if metadata['catalog'] == COLOUR_FEATURE_CATALOG else 0))


def threshold_pairs(metadata):
    """Massive outgoing pairs, in the same leg order as native inference."""
    legs = [i for i, (mass, inc) in enumerate(zip(metadata['masses'], metadata['incoming']))
            if mass > 0 and not inc]
    return [(i, j) for k, i in enumerate(legs) for j in legs[k+1:]]


def threshold_features(metadata, momenta):
    """Stable 1/v_ij inputs; unresolved thresholds require exact evaluation."""
    import numpy as np
    p = np.asarray(momenta, dtype=float)
    if p.ndim != 3 or p.shape[1:] != (metadata['nlegs'], 4) or not np.all(np.isfinite(p)):
        raise ValueError('invalid threshold momenta')
    columns = []
    for i, j in threshold_pairs(metadata):
        mi, mj = metadata['masses'][i], metadata['masses'][j]
        scale = np.maximum(np.maximum(np.max(np.abs(p[:, (i, j)]), axis=(1, 2)), mi), mj)
        a, b = p[:, i]/scale[:, None], p[:, j]/scale[:, None]
        energy = a[:, 0]*b[:, 0]
        spatial = np.sum(a[:, 1:]*b[:, 1:], axis=1)
        dot = energy-spatial
        mass = (mi/scale)*(mj/scale)
        delta = dot-mass
        error = 64*np.finfo(float).eps*(np.abs(energy)+np.abs(spatial)+mass)
        if np.any(delta <= error) or np.any(dot <= 0):
            raise ValueError('unresolved massive-pair threshold')
        value = 1/np.sqrt((delta/dot)*((dot+mass)/dot))
        if not np.all(np.isfinite(value)):
            raise ValueError('nonfinite massive-pair velocity')
        columns.append(value)
    return np.column_stack(columns) if columns else np.zeros((len(p), 0))


def _physics_rows(metadata, rows):
    """Build Born carriers and physical inputs without altering captures."""
    import numpy as np
    if metadata['catalog'] == COLOUR_FEATURE_CATALOG:
        # MINT adapts V/B, while the fit carries the exactly known extra
        # coupling. Preserve its original Born when replaying the marginals.
        dressed = [dict(r, mint_born=r.get('mint_born', r['born']),
                        born=r.get('mint_born', r['born'])*r['coupling_factor'])
                   for r in rows]
        return metadata, dressed, 'coupling_carried_colour_dipoles'
    if metadata['catalog'] != CONDITIONAL_FEATURE_CATALOG or not threshold_pairs(metadata):
        return metadata, rows, 'no_additional_threshold_inputs'
    if not rows or any('momenta' not in row for row in rows):
        return metadata, rows, 'threshold_momenta_unavailable'
    if metadata['nf']+len(threshold_pairs(metadata)) > 2048:
        return metadata, rows, 'threshold_feature_budget_exceeded'
    try:
        p = np.asarray([r['momenta'] for r in rows]).reshape(len(rows), metadata['nlegs'], 4)
        values = threshold_features(metadata, p)
    except ValueError:
        return metadata, rows, 'threshold_geometry_unresolved'
    # Keep absolute log(shat) last among scalar coordinates, followed by the
    # signed orientation suffix. Older model catalogs retain their numbering.
    index = scalar_feature_count(metadata)-1
    enriched = [dict(r, features=r['features'][:index]+v.tolist()+r['features'][index:])
                for r, v in zip(rows, values)]
    return dict(metadata, catalog=PHYSICS_FEATURE_CATALOG,
                nf=metadata['nf']+values.shape[1]), enriched, 'massive_pair_inverse_velocity'


def _floats(line):
    result = tuple(float(x.replace('D', 'E').replace('d', 'e')) for x in line.split())
    if not all(math.isfinite(x) for x in result):
        raise ValueError('nonfinite virtual-pool input')
    return result


def _header(stream, magic):
    header = stream.readline().split()
    versions = (2, MODEL_FORMAT_VERSION) if magic == 'VIRTUAL_POOL_MODEL' else (2, 3, CAPTURE_FORMAT_VERSION)
    if (len(header) != 2 or header[0] != magic or
            header[1] not in tuple(str(v) for v in versions)):
        raise ValueError('unsupported virtual-pool file header; restart integration')
    version = int(header[1])
    dimensions = tuple(int(x) for x in stream.readline().split())
    if len(dimensions) != 5:
        raise ValueError('invalid virtual-pool dimensions')
    catalog, nf, norders, nlegs, nx = dimensions
    pdgs = tuple(int(x) for x in stream.readline().split())
    masses = _floats(stream.readline())
    incoming = tuple(int(x) for x in stream.readline().split())
    expected = 1 + nlegs*(nlegs-1)//2 + len({m for m in masses if m > 0})
    if catalog in (FEATURE_CATALOG, CONDITIONAL_FEATURE_CATALOG, PHYSICS_FEATURE_CATALOG, COLOUR_FEATURE_CATALOG):
        expected += 2
    if catalog in (CONDITIONAL_FEATURE_CATALOG, PHYSICS_FEATURE_CATALOG, COLOUR_FEATURE_CATALOG):
        expected += (nlegs-2)*(nlegs-3)//2
    if catalog in (PHYSICS_FEATURE_CATALOG, COLOUR_FEATURE_CATALOG):
        expected += len(threshold_pairs(dict(masses=masses, incoming=incoming)))
    if catalog == COLOUR_FEATURE_CATALOG:
        expected += 4
    if (catalog not in (LEGACY_FEATURE_CATALOG, FEATURE_CATALOG, CONDITIONAL_FEATURE_CATALOG, PHYSICS_FEATURE_CATALOG, COLOUR_FEATURE_CATALOG) or nf != expected or not 1 <= nf <= 2048
            or not 3 <= nlegs <= 64 or not 1 <= norders <= 1024 or not 1 <= nx <= 256
            or any(len(x) != nlegs for x in (pdgs, masses, incoming))
            or any(x < 0 for x in masses) or any(x not in (0, 1) for x in incoming)
            or sum(incoming) not in ((1, 2) if catalog >= CONDITIONAL_FEATURE_CATALOG else (2,))):
        raise ValueError('invalid virtual-pool particle metadata')
    metadata = dict(version=version, catalog=catalog, nf=nf, norders=norders,
                    nlegs=nlegs, nx=nx, pdgs=pdgs, masses=masses, incoming=incoming)
    if catalog == COLOUR_FEATURE_CATALOG:
        colours = tuple(int(x) for x in stream.readline().split())
        if len(colours) != nlegs or any(c == 0 for c in colours):
            raise ValueError('invalid virtual-pool colour representations')
        metadata['colours'] = colours
    if magic == 'VIRTUAL_POOL_SAMPLES' and version >= 3:
        fields = stream.readline().split()
        if len(fields) != 1 or not 0 <= int(fields[0]) <= 65536:
            raise ValueError('invalid native helicity map size')
        nhel = int(fields[0])
        helicities = []
        for actual in range(1, nhel+1):
            fields = tuple(int(v) for v in stream.readline().split())
            if len(fields) != nlegs+2 or fields[0] != actual or fields[1] < 0:
                raise ValueError('invalid complete native helicity map')
            helicities.append(fields)
        metadata.update(nhel=nhel, helicities=tuple(helicities))
    return metadata


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


def _native_selection(fields, metadata):
    """Decode the native branch without applying helicity multiplicities again."""
    extra = _floats(' '.join(fields))
    full, selected, selected_q = extra[:3]
    nhel = metadata['nhel']
    q = extra[3:3+nhel]
    total = sum(q)
    if (full not in (0, 1) or selected != int(selected) or
            (full and (selected != -1 or selected_q != 1)) or
            (not full and nhel and
             (not 1 <= selected <= nhel or not 0 < selected_q <= 1)) or
            any(v < 0 or v > 1 for v in q) or
            (total != 0 and abs(total-1) > 1.e-9) or
            (total != 0 and not full and abs(q[int(selected)-1]-selected_q) > 1.e-9)):
        raise ValueError('invalid native helicity selection/probabilities')
    if any(q[i] and not h[1] for i, h in enumerate(metadata['helicities'])):
        raise ValueError('probability assigned to inactive native helicity')
    return dict(full_branch=bool(full), selected_helicity=int(selected),
                selected_q=selected_q, q=q, momenta=extra[3+nhel:])


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
        conditional = metadata['version'] >= 3
        coupled = metadata['version'] >= 4
        if coupled != (metadata['catalog'] == COLOUR_FEATURE_CATALOG):
            raise ValueError('capture version/physics catalog mismatch')
        prefix = metadata['nf'] + 9 + metadata['nx'] + int(coupled)
        suffix = 3 + metadata['nhel'] + 4*metadata['nlegs'] if conditional else 0
        first_row = 7 + metadata['nhel'] + int(coupled) if conditional else 6
        for number, line in enumerate(stream, first_row):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != prefix + suffix:
                raise ValueError('%s:%s: incorrect sample width' % (path, number))
            sample, channel, iteration, order = (int(v) for v in fields[:4])
            probability, born, virtual, baseline, *values = _floats(' '.join(fields[5:prefix]))
            factor = values.pop(0) if coupled else 1.
            features = values[:metadata['nf']]
            if (sample < 1 or channel < 1 or iteration < 1
                    or not 1 <= order <= metadata['norders']
                    or not 0 < probability <= 1 or factor < 0):
                raise ValueError('%s:%s: invalid sample metadata' % (path, number))
            key = (sample, channel, iteration, order)
            if key in seen:
                raise ValueError('%s:%s: duplicate sample/order' % (path, number))
            seen.add(key)
            row = dict(sample=sample, channel=channel, iteration=iteration,
                       order=order, probability=probability, born=born,
                       virtual=virtual, baseline=baseline, features=features,
                       directory=directory)
            row.update(bins=int(fields[4]), x=values[metadata['nf']:metadata['nf']+metadata['nx']])
            if coupled:
                row['coupling_factor'] = factor
            if conditional:
                row.update(_native_selection(fields[prefix:], metadata))
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


def _model_version(metadata, orders):
    if (metadata.get('version', 2) >= 3 or
            any(order['degree'] == 3 for order in orders)):
        return MODEL_FORMAT_VERSION
    return FORMAT_VERSION


def _model_text(metadata, orders):
    floats = lambda values: ' '.join('%.17e' % x for x in values)
    version = _model_version(metadata, orders)
    lines = ['VIRTUAL_POOL_MODEL %d' % version,
             '%d %d %d %d %d' % tuple(metadata[k] for k in
                                      ('catalog', 'nf', 'norders', 'nlegs', 'nx')),
             ' '.join(str(x) for x in metadata['pdgs']), floats(metadata['masses']),
             ' '.join(str(x) for x in metadata['incoming'])]
    if metadata['catalog'] == COLOUR_FEATURE_CATALOG:
        lines.append(' '.join(str(x) for x in metadata['colours']))
    for entry in orders:
        lines.append('%d %.17e %.17e %.17e %d %d' %
                     (entry['order'], entry['blend'], entry['born_scale'], entry['target_scale'],
                      len(entry['terms']), entry['degree']))
        lines.extend([floats(entry['lower']), floats(entry['upper'])])
        for term in entry['terms']:
            indices = tuple(term['pair'])
            if version == MODEL_FORMAT_VERSION:
                indices += (-1,)*(3-len(indices))
            lines.append(' '.join(str(i+1) for i in indices) + ' %.17e' % term['coefficient'])
    return '\n'.join(lines) + '\n'


def train_pool(job_dirs, output_model, report_path=None, job_groups=None,
               neural=False, output_neural=None):
    """Fit one frozen model per coupling order after the first-stage barrier.

    Missing NumPy or captures give an explicit MINT fallback. Malformed files
    or incompatible Born metadata raise ValueError, never a silent merge.
    ``model_written`` indicates whether inference may load the model; ``enabled``
    indicates whether any order selected a nonzero blend. An all-zero model is
    a valid frozen fallback. Without NumPy no model is installed, so ordinary
    MINT adaptation remains active. No held-out evaluation data are accepted here.
    """
    from . import virtual_pool_polynomial as polynomial

    started = time.monotonic()
    cpu_started = time.process_time()
    if neural:
        output_neural = output_neural or os.path.join(os.path.dirname(os.path.abspath(output_model)),
                                                      'virtual_pool_neural.dat')
        if os.path.exists(output_neural):
            os.unlink(output_neural)
    metadata, rows, jobs = collect_samples(job_dirs, job_groups)
    report = dict(version=FORMAT_VERSION, status='skipped', enabled=False, model_written=False,
                  family='born_compact_physics', accepted_rows=len(rows), jobs=jobs, orders=[],
                  accepted_samples=len({(r['directory'], r['channel'], r['iteration'], r['sample'])
                                        for r in rows}),
                  objective='physical residual; inverse accepted attempts and loop probability',
                  validation='matched three-fold CV and forward iterations; paired L1 gain and RMS guards',
                  tail_margin=polynomial.TAIL_MARGIN, max_coefficients=polynomial.MAX_TERMS,
                  max_cubic_features=polynomial.MAX_CUBIC_FEATURES,
                  max_cubic_design_elements=polynomial.MAX_CUBIC_DESIGN_ELEMENTS,
                  minimum_rows_per_coefficient=polynomial.ROWS_PER_TERM,
                  call_probability='adapt to measured residual; exact loop outside trained domain',
                  boundary_prediction='clamped polynomial continuation; exact residual correction')
    if metadata is None:
        report['reason'] = 'no_sample_headers'
        if neural:
            report['neural'] = dict(model_written=False, enabled=False, reason='no_sample_headers')
    else:
        report['metadata'] = metadata
        try:
            import numpy
        except ImportError:
            # A zero-blend model would also freeze the loop-call budget.
            # No model lets the controller keep ordinary adaptive MINT.
            report.update(status='fallback', reason='numpy_unavailable')
            if neural:
                report['neural'] = dict(model_written=False, enabled=False, reason='numpy_unavailable')
        else:
            metadata, rows, report['physics_inputs'] = _physics_rows(metadata, rows)
            report['metadata'] = metadata
            if metadata['catalog'] == COLOUR_FEATURE_CATALOG:
                report['boundary_prediction'] = 'physical function continuation; exact residual correction'
            folds = _folds(rows, 3)
            orders = []
            for order in range(1, metadata['norders'] + 1):
                entry, diagnostics = polynomial.fit_order(rows, folds, metadata, order, strategy='compact')
                orders.append(entry)
                diagnostics.update(blend=entry['blend'], reason=entry['reason'])
                report['orders'].append(diagnostics)
            _atomic_write(output_model, _model_text(metadata, orders))
            report.update(model_written=True,
                          model_format_version=_model_version(metadata, orders),
                          enabled=any(o['blend'] > 0 for o in orders))
            report['status'] = 'trained' if report['enabled'] else 'fallback'
            if neural:
                from . import virtual_pool_neural
                report['neural'] = virtual_pool_neural.train(
                    rows, metadata, orders, report['orders'], output_model, output_neural)
                if report['neural']['enabled']:
                    report.update(enabled=True, status='trained')
    if not report['model_written']:
        report['call_probability'] = 'adaptive MINT'
    report['training_seconds'] = time.monotonic()-started
    report['training_cpu_seconds'] = time.process_time()-cpu_started
    if report_path is not None:
        _atomic_write(report_path, json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return report


def load_model(path):
    """Strict Python reader matching the frozen Fortran interchange schema."""
    from .virtual_pool_polynomial import MAX_TERMS

    with open(path) as stream:
        metadata = _header(stream, 'VIRTUAL_POOL_MODEL')
        indices_per_term = 3 if metadata['version'] == MODEL_FORMAT_VERSION else 2
        orders = []
        for expected_order in range(1, metadata['norders'] + 1):
            fields = stream.readline().split()
            if len(fields) != 6:
                raise ValueError('invalid pooled model order header')
            order, nterms, degree = int(fields[0]), int(fields[4]), int(fields[5])
            blend, born_scale, target_scale = _floats(' '.join(fields[1:4]))
            lower, upper = _floats(stream.readline()), _floats(stream.readline())
            if (order != expected_order or not 0 <= blend <= 1 or born_scale <= 0 or target_scale <= 0
                    or not 0 <= nterms <= MAX_TERMS or not 0 <= degree <= indices_per_term
                    or len(lower) != metadata['nf'] or len(upper) != metadata['nf']
                    or any(a > b for a, b in zip(lower, upper))):
                raise ValueError('invalid pooled model order metadata')
            if blend and not nterms:
                raise ValueError('active polynomial has no coefficients')
            if metadata['catalog'] in (CONDITIONAL_FEATURE_CATALOG, PHYSICS_FEATURE_CATALOG, COLOUR_FEATURE_CATALOG) and blend:
                ns = scalar_feature_count(metadata)
                if any(a != -1 or b != 1 for a, b in zip(lower[ns:], upper[ns:])):
                    raise ValueError('signed orientation coordinates require fixed bounds')
            terms = []
            for _ in range(nterms):
                fields = stream.readline().split()
                if len(fields) != indices_per_term+1:
                    raise ValueError('invalid polynomial term')
                pair = tuple(int(v)-1 for v in fields[:indices_per_term])
                coefficient, = _floats(fields[-1])
                if (min(pair) < -1 or max(pair) >= metadata['nf'] or
                        any(a == -1 and b >= 0 for a, b in zip(pair, pair[1:])) or
                        sum(i >= 0 for i in pair) > degree):
                    raise ValueError('invalid polynomial indices/degree')
                if (metadata['catalog'] in (CONDITIONAL_FEATURE_CATALOG, PHYSICS_FEATURE_CATALOG, COLOUR_FEATURE_CATALOG) and
                        sum(i >= scalar_feature_count(metadata) for i in pair) > 1):
                    raise ValueError('polynomial term contains multiple orientation factors')
                terms.append(dict(pair=pair, coefficient=coefficient))
            orders.append(dict(order=order, blend=blend, born_scale=born_scale,
                               target_scale=target_scale, degree=degree,
                               lower=lower, upper=upper, terms=terms))
            if metadata['catalog'] == COLOUR_FEATURE_CATALOG:
                orders[-1]['unclipped'] = True
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
