"""Capture and frozen-model codecs for virtual approximation workers.

This module owns file versions, feature catalogs, metadata validation and
committed-point normalization. NumPy is optional and imported only for
physical input transformations.
"""
import math
import os
import tempfile


SAMPLES = 'virtual_pool_samples.dat'
ITERATIONS = 'virtual_pool_iterations.dat'
REPORT_FORMAT_VERSION = 2
MODEL_FORMAT_VERSION = 4
CAPTURE_FORMAT_VERSION = 5
FEATURE_CATALOG = 9

MAX_SURROGATE_FEATURES = 94
MAX_SURROGATE_CHANNELS = 10000


def scalar_feature_count(metadata):
    return (3 + metadata['nlegs']*(metadata['nlegs']-1)//2 +
            len({m for m in metadata['masses'] if m > 0}) +
            len(threshold_pairs(metadata)) + 4)


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
    # MINT uses the native Born. The refinement carries the extra loop
    # coupling explicitly, so preserve both carriers for matched replay.
    dressed = [dict(r, mint_born=r.get('mint_born', r['born']),
                    born=r.get('mint_born', r['born'])*r['coupling_factor'])
               for r in rows]
    return metadata, dressed, 'coupling_carried_colour_dipoles'


def _floats(line):
    result = tuple(float(x.replace('D', 'E').replace('d', 'e')) for x in line.split())
    if not all(math.isfinite(x) for x in result):
        raise ValueError('nonfinite virtual-pool input')
    return result


def _header(stream, magic):
    header = stream.readline().split()
    version = MODEL_FORMAT_VERSION if magic == 'VIRTUAL_POOL_MODEL' else CAPTURE_FORMAT_VERSION
    if (len(header) != 2 or header[0] != magic or
            header[1] != str(version)):
        raise ValueError('unsupported virtual-pool file header; restart integration')
    dimensions = tuple(int(x) for x in stream.readline().split())
    if len(dimensions) != 5:
        raise ValueError('invalid virtual-pool dimensions')
    catalog, nf, norders, nlegs, nx = dimensions
    pdgs = tuple(int(x) for x in stream.readline().split())
    masses = _floats(stream.readline())
    incoming = tuple(int(x) for x in stream.readline().split())
    expected = (scalar_feature_count(dict(nlegs=nlegs, masses=masses, incoming=incoming)) +
                (nlegs-2)*(nlegs-3)//2)
    if (catalog != FEATURE_CATALOG or nf != expected or not 1 <= nf <= 2048
            or not 3 <= nlegs <= 64 or not 1 <= norders <= 1024 or not 1 <= nx <= 256
            or any(len(x) != nlegs for x in (pdgs, masses, incoming))
            or any(x < 0 for x in masses) or any(x not in (0, 1) for x in incoming)
            or sum(incoming) not in (1, 2)):
        raise ValueError('invalid virtual-pool particle metadata')
    metadata = dict(version=version, catalog=catalog, nf=nf, norders=norders,
                    nlegs=nlegs, nx=nx, pdgs=pdgs, masses=masses, incoming=incoming)
    colours = tuple(int(x) for x in stream.readline().split())
    if len(colours) != nlegs or any(c == 0 for c in colours):
        raise ValueError('invalid virtual-pool colour representations')
    metadata['colours'] = colours
    if magic == 'VIRTUAL_POOL_SAMPLES':
        fields = stream.readline().split()
        if len(fields) != 1 or not 4 <= int(fields[0]) <= MAX_SURROGATE_FEATURES:
            raise ValueError('invalid captured surrogate basis')
        metadata['nsurrogate'] = int(fields[0])
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
    return _read_iteration_history(directory)[0]


def _read_iteration_history(directory):
    ledger = os.path.join(directory, ITERATIONS)
    if not os.path.isfile(ledger):
        return None, {}
    iterations, channels = {}, {}
    with open(ledger) as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.split()
            if fields[0] == 'CHANNEL':
                if len(fields) != 4:
                    raise ValueError('%s:%s: invalid stratified iteration' % (ledger, number))
                iteration, channel, attempted = (int(x) for x in fields[1:])
                counts = channels.setdefault(iteration, {})
                if iteration < 1 or channel < 1 or attempted < 0 or channel in counts:
                    raise ValueError('%s:%s: invalid/duplicate stratified channel' % (ledger, number))
                counts[channel] = attempted
                continue
            if len(fields) != 3:
                raise ValueError('%s:%s: invalid iteration ledger' % (ledger, number))
            iteration, accepted, attempted = (int(x) for x in fields)
            if iteration < 1 or accepted not in (0, 1) or attempted < 0 or iteration in iterations:
                raise ValueError('%s:%s: invalid/duplicate iteration' % (ledger, number))
            iterations[iteration] = (accepted, attempted)
    for iteration, counts in channels.items():
        if iteration not in iterations or sum(counts.values()) != iterations[iteration][1]:
            raise ValueError('%s: inconsistent stratified attempt counts' % ledger)
    return iterations, channels


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
    iterations, channel_attempts = _read_iteration_history(directory)
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
        surrogate_width = metadata['nsurrogate']
        prefix = metadata['nf'] + 12 + metadata['nx'] + surrogate_width
        suffix = 3 + metadata['nhel'] + 4*metadata['nlegs']
        first_row = 9 + metadata['nhel']
        for number, line in enumerate(stream, first_row):
            if not line.strip():
                continue
            fields = line.split()
            event_width = 0
            if len(fields) <= prefix+suffix:
                raise ValueError('%s:%s: missing S-event groups' % (path, number))
            groups, = _floats(fields[prefix+suffix])
            if groups != int(groups) or not 0 <= groups <= 10000:
                raise ValueError('invalid S-event group count')
            event_width = 1+2*int(groups)
            if len(fields) != prefix + suffix + event_width:
                raise ValueError('%s:%s: incorrect sample width' % (path, number))
            sample, channel, iteration, order = (int(v) for v in fields[:4])
            probability, born, virtual, baseline, *values = _floats(' '.join(fields[5:prefix]))
            factor = values.pop(0)
            surrogate_gs2 = values.pop(0)
            surrogate_valid = values.pop(0)
            surrogate, values = values[:surrogate_width], values[surrogate_width:]
            if surrogate_gs2 < 0 or surrogate_valid not in (0., 1.) or (surrogate_valid and abs(surrogate[0]-1.) > 1.e-10):
                raise ValueError('invalid captured surrogate inputs')
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
            row['coupling_factor'] = factor
            row.update(surrogate=surrogate, surrogate_valid=bool(surrogate_valid), surrogate_gs2=surrogate_gs2)
            row.update(_native_selection(fields[prefix:prefix+suffix], metadata))
            groups = int(groups)
            event_values = _floats(' '.join(fields[prefix+suffix+1:]))
            soft, flavour = event_values[:groups], event_values[groups:]
            if any(f < 0 for f in flavour) or (groups and abs(sum(flavour)-1.) > 1.e-9):
                raise ValueError('invalid captured S-event flavour fractions')
            row.update(event_soft=soft, event_flavour=flavour)
            if row['bins'] not in (4, 8) or any(v < 0 or v > 1 for v in row['x']):
                raise ValueError('invalid MINT coordinates/bins')
            if rows and (iteration, sample) < (rows[-1]['iteration'], rows[-1]['sample']):
                raise ValueError('unordered pooled sample history')
            same_point = rows and all(row[k] == rows[-1][k] for k in ('sample', 'channel', 'iteration'))
            expected_order = rows[-1]['order']+1 if same_point else 1
            if order != expected_order or (rows and not same_point and rows[-1]['order'] != metadata['norders']):
                raise ValueError('incomplete or unordered coupling orders in surrogate capture')
            if same_point:
                common = ('probability', 'bins', 'x', 'features', 'surrogate', 'surrogate_valid',
                          'surrogate_gs2', 'full_branch', 'selected_helicity', 'selected_q',
                          'q', 'momenta', 'event_soft', 'event_flavour')
                if any(row[k] != rows[-1][k] for k in common):
                    raise ValueError('coupling orders have inconsistent point/helicity/event context')
            rows.append(row)
        if rows and rows[-1]['order'] != metadata['norders']:
            raise ValueError('incomplete final surrogate observation')
    ledger = os.path.join(directory, ITERATIONS)
    if iterations is None:
        # A failed/incomplete integration must never contribute uncommitted rows.
        return metadata, [], 0, dict(directory=directory, status='missing_iterations',
                                     discarded_rows=len(rows))
    accepted_rows = [row for row in rows if iterations.get(row['iteration'], (0, 0))[0]]
    for row in accepted_rows:
        counts = channel_attempts.get(row['iteration'])
        if counts is not None:
            attempted = counts.get(row['channel'], 0)
            if not attempted:
                raise ValueError('%s: sample without stratified channel attempts' % ledger)
            # Warm-up visits channels separately with unit channel weight.
            # Combine their means using the same overall attempted-point
            # denominator as subsequent mixed-channel iterations.
            row['attempt_correction'] = iterations[row['iteration']][1]/attempted
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
    Stratified warm-up rows additionally use their channel's recorded
    attempted-point exposure, without changing their Born/virtual targets.
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
        row['weight'] = row.get('attempt_correction', 1.) / denominators[row['group']] / row['probability']
        if not math.isfinite(row['weight']):
            raise ValueError('virtual-pool statistical weight overflow')
    return metadata, rows, reports


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
    version = MODEL_FORMAT_VERSION
    lines = ['VIRTUAL_POOL_MODEL %d' % version,
             '%d %d %d %d %d' % tuple(metadata[k] for k in
                                      ('catalog', 'nf', 'norders', 'nlegs', 'nx')),
             ' '.join(str(x) for x in metadata['pdgs']), floats(metadata['masses']),
             ' '.join(str(x) for x in metadata['incoming'])]
    lines.append(' '.join(str(x) for x in metadata['colours']))
    for entry in orders:
        lines.append('%d %.17e %.17e %.17e %d %d' %
                     (entry['order'], entry['blend'], entry['born_scale'], entry['target_scale'],
                      len(entry['terms']), entry['degree']))
        lines.extend([floats(entry['lower']), floats(entry['upper'])])
        for term in entry['terms']:
            indices = tuple(term['pair'])
            indices += (-1,)*(3-len(indices))
            lines.append(' '.join(str(i+1) for i in indices) + ' %.17e' % term['coefficient'])
        basis = entry.get('surrogate')
        lines.append('SURROGATE %d %d' % ((basis['nfeatures'], len(basis['channels'])) if basis else (0, 0)))
        if basis:
            lines.append(floats(basis['shared']))
            for channel in basis['channels']:
                lines.append('%d %.17e %s' % (channel['channel'], channel['blend'], floats(channel['coefficient'])))
    return '\n'.join(lines) + '\n'


def load_model(path):
    """Strict Python reader matching the frozen Fortran interchange schema."""
    from .polynomial import MAX_TERMS

    with open(path) as stream:
        metadata = _header(stream, 'VIRTUAL_POOL_MODEL')
        indices_per_term = 3
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
            if blend:
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
                if (min(pair) < -1 or max(pair) >= metadata['nf'] or pair[2] != -1 or
                        any(a == -1 and b >= 0 for a, b in zip(pair, pair[1:])) or
                        sum(i >= 0 for i in pair) > degree):
                    raise ValueError('invalid polynomial indices/degree')
                if sum(i >= scalar_feature_count(metadata) for i in pair) > 1:
                    raise ValueError('polynomial term contains multiple orientation factors')
                terms.append(dict(pair=pair, coefficient=coefficient))
            orders.append(dict(order=order, blend=blend, born_scale=born_scale,
                               target_scale=target_scale, degree=degree,
                               lower=lower, upper=upper, terms=terms, unclipped=True))
            fields = stream.readline().split()
            if len(fields) != 3 or fields[0] != 'SURROGATE':
                raise ValueError('missing pooled surrogate basis')
            nb, nc = map(int, fields[1:])
            if ((nb != 0 and not 4 <= nb <= MAX_SURROGATE_FEATURES) or not 0 <= nc <= MAX_SURROGATE_CHANNELS or
                    (nb == 0 and nc != 0)):
                raise ValueError('invalid pooled surrogate dimensions')
            if orders[-1]['blend'] and not nb and not nterms:
                raise ValueError('active refinement has no coefficients')
            if nb:
                shared = _floats(stream.readline())
                if len(shared) != nb:
                    raise ValueError('invalid shared surrogate coefficients')
                channels, seen = [], set()
                for _ in range(nc):
                    fields = stream.readline().split()
                    if len(fields) != nb+2:
                        raise ValueError('invalid channel surrogate coefficients')
                    channel = int(fields[0])
                    blend, *coefficient = _floats(' '.join(fields[1:]))
                    if channel < 1 or channel in seen or not 0 <= blend <= 1:
                        raise ValueError('invalid/duplicate channel surrogate')
                    seen.add(channel)
                    channels.append(dict(channel=channel, blend=blend, coefficient=coefficient))
                orders[-1]['surrogate'] = dict(nfeatures=nb, shared=shared, channels=channels)
        if stream.read().strip():
            raise ValueError('unexpected trailing pooled model data')
    return dict(metadata=metadata, orders=orders)
