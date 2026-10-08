"""Validate and collect independent, quota-driven AmpliCol workers.

Worker quotas and published rates come from the independent survey. Each
worker starts from its saved proposal, adapts only unfolded directions, and
supplies a reserve with draw-time importance weights and full overweight-tail
diagnostics. Collection uniformly trims each reserve
once; candidates from different workers are never rethresholded together.
"""

import math
import os
import random
import re
import shutil
import sys
import tempfile


# Absolute cross-section mass above the threshold, including the whole
# contribution of each overweight point. Keep the Fortran default in sync.
ALLOWED_OVERWEIGHT_FACTOR = .01


class PoolError(ValueError):
    """An incomplete or incompatible provisional pool."""


class PoolShortage(PoolError):
    """More production points are required before finalizing this channel."""

    def __init__(self, required, available):
        self.required = required
        self.available = available
        super().__init__('AmpliCol channel needs %d events; %d are available' %
                         (required, available))


class PoolOverweight(PoolError):
    """The full overweight-tail fraction fails the requested limit."""

    def __init__(self, observed, allowed):
        self.observed = observed
        self.allowed = allowed
        super().__init__('AmpliCol channel overweight tail %.8g is not below %.8g' %
                         (observed, allowed))


def _real(value):
    result = float(value.replace('D', 'E').replace('d', 'e'))
    if not math.isfinite(result):
        raise PoolError('Nonfinite number in AmpliCol pool')
    return result


def _read_adaptation(stream, trials, path):
    """Validate the trial-count schedule independently of grid-update success."""
    record = stream.readline().split()
    if len(record) != 5:
        raise PoolError('Invalid AmpliCol adaptation counts: %s' % path)
    ndim, updates, interval, batch, points = map(int, record)
    mask = list(map(int, stream.readline().split()))
    if (ndim <= 0 or len(mask) != ndim or any(value not in (0, 1) for value in mask)
            or updates < 0 or interval < 0 or interval > 65536 or batch < 0 or points < 0):
        raise PoolError('Invalid AmpliCol adaptation metadata: %s' % path)
    if not any(mask):
        if any((updates, interval, batch, points)):
            raise PoolError('Inconsistent AmpliCol disabled adaptation: %s' % path)
        return dict(mode='frozen', ndim=ndim, mask=mask, updates=0,
                    interval=0, batch=0, points=0, completed_batches=0)
    if interval == 0:
        raise PoolError('Invalid AmpliCol adaptation interval: %s' % path)
    remaining, expected_batch, completed = trials, interval, 0
    # Jump over capped batches: corrupt large trial counts must not make
    # validation take time proportional to the number of production trials.
    while expected_batch < 65536 and remaining >= expected_batch:
        remaining -= expected_batch
        completed += 1
        expected_batch = min(2 * expected_batch, 65536)
    if expected_batch == 65536:
        completed += remaining // expected_batch
        remaining %= expected_batch
    if batch != expected_batch or points != remaining or updates > completed:
        raise PoolError('Inconsistent AmpliCol adaptation schedule: %s' % path)
    return dict(mode='adaptive_unfolded', ndim=ndim, mask=mask, updates=updates,
                interval=interval, batch=batch, points=points,
                completed_batches=completed)


def read_pool(path):
    """Read and validate ``ampli_pool.dat`` in a worker directory.

    ``path`` can also name the sidecar itself. Candidate rows are pairs of
    (absolute integrand weight, log priority) for legacy version 1. Version 2
    adds normalized reserve correction, raw overweight flag, and absolute
    nominal LHE factor. Version 3 also records the unfolded-grid adaptation
    mask and trial schedule. Candidate weights always refer to their sampling
    proposal at draw time, and rows follow the unmodified candidate LHE order.
    """
    path = os.fspath(path)
    if os.path.isdir(path):
        path = os.path.join(path, 'ampli_pool.dat')
    try:
        with open(path) as stream:
            header = stream.readline().split()
            if len(header) != 2 or header[0] != 'MG5_AMPLI_POOL' or header[1] not in ('1', '2', '3'):
                raise PoolError('Incompatible AmpliCol pool header: %s' % path)
            version = int(header[1])
            counts = stream.readline().split()
            if len(counts) != 3:
                raise PoolError('Invalid AmpliCol pool counts: %s' % path)
            trials, ncandidates = int(counts[0]), int(counts[1])
            cutoff = _real(counts[2])
            if trials < 0 or not 0 <= ncandidates <= trials or cutoff < 0.:
                raise PoolError('Invalid AmpliCol pool budget: %s' % path)
            moments = list(map(_real, stream.readline().split()))
            if len(moments) != 5:
                raise PoolError('Invalid AmpliCol pool moments: %s' % path)
            mean_abs, mean_signed, m2_abs, m2_signed, covariance_sum = moments
            tolerance = 1.e-10 * max(mean_abs, abs(mean_signed), 1.e-300)
            if mean_abs < 0. or abs(mean_signed) > mean_abs + tolerance:
                raise PoolError('Inconsistent signed and absolute pool rates: %s' % path)
            if m2_abs < 0. or m2_signed < 0.:
                raise PoolError('Negative AmpliCol pool variance: %s' % path)
            cov_bound = math.sqrt(m2_abs) * math.sqrt(m2_signed)
            if abs(covariance_sum) > cov_bound * (1. + 1.e-10) + 1.e-300:
                raise PoolError('Invalid AmpliCol pool covariance: %s' % path)
            if trials == 0 and any(moments):
                raise PoolError('Nonzero moments for an empty AmpliCol pool: %s' % path)
            generation = {}
            if version >= 2:
                record = stream.readline().split()
                if len(record) != 6:
                    raise PoolError('Invalid AmpliCol generation record: %s' % path)
                generation = dict(zip(
                    ('generated_target', 'final_quota', 'threshold',
                     'full_trial_tail', 'reserve_tail', 'worst_subset_tail'),
                    [int(record[0]), int(record[1])] + list(map(_real, record[2:]))))
            adaptation = (_read_adaptation(stream, trials, path) if version == 3
                          else dict(mode='frozen'))
            candidates = []
            log_cutoff = math.log(cutoff) if cutoff else -math.inf
            for unused in range(ncandidates):
                row = list(map(_real, stream.readline().split()))
                if len(row) != (2 if version == 1 else 5) or row[0] <= 0.:
                    raise PoolError('Invalid AmpliCol candidate row: %s' % path)
                # u in (0,1): log(w/u) >= log(w), and the writing cutoff.
                if row[1] < max(math.log(row[0]), log_cutoff) - 1.e-11:
                    raise PoolError('Inconsistent AmpliCol candidate priority: %s' % path)
                if version >= 2 and (row[2] < 0. or row[3] not in (0., 1.) or row[4] <= 0.):
                    raise PoolError('Invalid AmpliCol candidate correction: %s' % path)
                candidates.append(tuple(row))
            if stream.read().strip():
                raise PoolError('Unexpected trailing AmpliCol pool data: %s' % path)
    except (OSError, ValueError) as error:
        if isinstance(error, PoolError):
            raise
        raise PoolError('Cannot read AmpliCol pool %s: %s' % (path, error)) from error
    pool = dict(path=os.path.abspath(path), version=version,
                lhe_path=os.path.join(os.path.dirname(os.path.abspath(path)),
                                      'ampli_candidates.lhe'),
                trials=trials, ncandidates=ncandidates, cutoff=cutoff,
                mean_abs=mean_abs, mean_signed=mean_signed,
                m2_abs=m2_abs, m2_signed=m2_signed,
                covariance_sum=covariance_sum, candidates=candidates,
                adaptation=adaptation, **generation)
    if version >= 2:
        _worker_status(pool)
    return pool


def combine_moments(pools):
    """Merge diagnostic trial moments with Chan's moment identities.

    The reported errors use the usual sample-variance expression. Adaptation
    and quota stopping prevent interpreting it as an unbiased uncertainty on
    a production integral; published rates and errors come from the survey.
    """
    count = 0
    mean_abs = mean_signed = m2_abs = m2_signed = covariance_sum = 0.
    for pool in pools:
        n = pool['trials']
        if not n:
            continue
        combined = count + n
        delta_abs = pool['mean_abs'] - mean_abs
        delta_signed = pool['mean_signed'] - mean_signed
        fraction = float(n) / combined
        cross_count = count * fraction
        mean_abs += delta_abs * fraction
        mean_signed += delta_signed * fraction
        m2_abs += pool['m2_abs'] + delta_abs * delta_abs * cross_count
        m2_signed += pool['m2_signed'] + delta_signed * delta_signed * cross_count
        covariance_sum += pool['covariance_sum'] + delta_abs * delta_signed * cross_count
        count = combined
    denominator = count * (count - 1) if count > 1 else 1
    return dict(trials=count, mean_abs=mean_abs, mean_signed=mean_signed,
                m2_abs=m2_abs, m2_signed=m2_signed,
                covariance_sum=covariance_sum,
                absolute=mean_abs, signed=mean_signed,
                error_abs=math.sqrt(max(0., m2_abs) / denominator),
                error_signed=math.sqrt(max(0., m2_signed) / denominator),
                covariance=covariance_sum / denominator)


_EVENT_OPEN = re.compile(r'^\s*<event(?:\s[^>]*)?>\s*$')
_EVENT_CLOSE = re.compile(r'^\s*</event\s*>\s*$')
_TOKENS = re.compile(r'\S+')
_COMPACT_HEADER = re.compile(r'(<header>\r?\n)[ \t]*[0-9]+[ \t]*(\r?\n[ \t]*</header>)')


def _header_count(preamble, quota):
    # read_lhef_header reads the number immediately before </header> with
    # format(1x,i8); reweight_xsec_events then loops over exactly that count.
    # Only touch this compact worker header, not an ordinary XML banner.
    return _COMPACT_HEADER.sub(lambda match: match.group(1) + ' %8d' % quota +
                               match.group(2), preamble)


def _replace_tokens(line, replacements):
    tokens = list(_TOKENS.finditer(line))
    if not tokens or max(replacements) >= len(tokens):
        raise PoolError('Malformed LHE numerical record')
    for index in sorted(replacements, reverse=True):
        token = tokens[index]
        line = line[:token.start()] + replacements[index] + line[token.end():]
    return line


def _read_event_weight(event):
    for line in event.splitlines()[1:]:
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        tokens = line.split()
        if len(tokens) < 6:
            break
        try:
            int(tokens[0])
            int(tokens[1])
            return _real(tokens[2])
        except ValueError as error:
            raise PoolError('Invalid AmpliCol LHE event header') from error
    raise PoolError('Invalid AmpliCol LHE event header')


def _event_weight(event, multiplier):
    """Scale XWGTUP alone; retain particles, scales and mgrwgt verbatim."""
    lines = event.splitlines(keepends=True)
    for index in range(1, len(lines)):
        if not lines[index].strip() or lines[index].lstrip().startswith('#'):
            continue
        tokens = lines[index].split()
        if len(tokens) < 6:
            raise PoolError('Invalid AmpliCol LHE event header')
        try:
            int(tokens[0])
            int(tokens[1])
            weight = _real(tokens[2]) * multiplier
        except ValueError as error:
            raise PoolError('Invalid AmpliCol LHE event header') from error
        if not math.isfinite(weight):
            raise PoolError('Nonfinite finalized AmpliCol event weight')
        lines[index] = _replace_tokens(lines[index], {2: '%.16e' % weight})
        return ''.join(lines)
    raise PoolError('Missing AmpliCol LHE event header')


def _init_rates(preamble, cross_sections, normalization_factor):
    """Replace XSECUP/XERRUP by the coordinator's final per-process rates."""
    rates = {int(label): (float(rate), float(error))
             for label, rate, error in cross_sections}
    lines = preamble.splitlines(keepends=True)
    inside = False
    header = False
    found = set()
    expected = None
    for index, line in enumerate(lines):
        if line.strip() == '<init>':
            inside, header = True, True
        elif line.strip() == '</init>':
            inside = False
        elif inside and line.strip() and not line.lstrip().startswith('#'):
            tokens = line.split()
            if header:
                if len(tokens) != 10:
                    raise PoolError('Invalid AmpliCol LHE init header')
                expected = int(tokens[-1])
                # Native re-evaluation can leave residual weight corrections,
                # including when the requested nominal scale is unity.
                lines[index] = _replace_tokens(line, {8: '-4'})
                header = False
                continue
            if len(tokens) != 4:
                raise PoolError('Invalid AmpliCol LHE process row')
            label = int(tokens[3])
            if label not in rates or label in found:
                raise PoolError('AmpliCol init process labels disagree with final rates')
            rate, error = rates[label]
            lines[index] = _replace_tokens(line, {
                0: '%.16e' % rate, 1: '%.16e' % error,
                # XMAXUP is unused for IDWTUP=-4. Record the nominal global
                # scale rather than retaining the obsolete survey value.
                2: '%.16e' % abs(normalization_factor)})
            found.add(label)
    if inside or expected != len(found) or found != set(rates):
        raise PoolError('Incomplete AmpliCol LHE init block')
    return ''.join(lines)


def _lhe_parts(path):
    """Yield ('preamble'|'event'|'footer', text) without loading the pool."""
    with open(path) as stream:
        lines = []
        in_event = False
        started = False
        for line in stream:
            if _EVENT_OPEN.match(line):
                if in_event:
                    raise PoolError('Nested AmpliCol LHE event: %s' % path)
                if not started:
                    yield 'preamble', ''.join(lines)
                    started = True
                elif any(part.strip() for part in lines):
                    raise PoolError('Unexpected data between AmpliCol LHE events')
                lines = [line]
                in_event = True
            elif _EVENT_CLOSE.match(line):
                if not in_event:
                    raise PoolError('Unmatched AmpliCol LHE event end: %s' % path)
                lines.append(line)
                yield 'event', ''.join(lines)
                lines = []
                in_event = False
            else:
                lines.append(line)
        if in_event:
            raise PoolError('Truncated AmpliCol LHE event: %s' % path)
        remainder = ''.join(lines)
        if '</LesHouchesEvents>' not in remainder:
            raise PoolError('Incomplete AmpliCol LHE file: %s' % path)
        if not started:
            closing = remainder.index('</LesHouchesEvents>')
            yield 'preamble', remainder[:closing]
            remainder = remainder[closing:]
        yield 'footer', remainder


def _weighted_fraction(weights, flags):
    """Compute a tail fraction without overflowing a sum of large weights."""
    maximum = max(weights, default=0.)
    if not maximum:
        return 0.
    scaled = [value / maximum for value in weights]
    return math.fsum(value for value, flag in zip(scaled, flags) if flag) / math.fsum(scaled)


def _worker_status(pool):
    """Independently verify the worker's threshold, corrections and tail mass."""
    if pool.get('version', 1) not in (2, 3):
        raise PoolError('Legacy AmpliCol pool protocol 1 cannot be finalized; regenerate with protocol 3')
    target, quota, threshold = (pool[key] for key in
                                ('generated_target', 'final_quota', 'threshold'))
    if any(isinstance(value, bool) or int(value) != value or value < 0
           for value in (target, quota)) or target < quota:
        raise PoolError('Invalid AmpliCol worker event quota')
    if target and target < (11 * quota + 9) // 10:
        raise PoolError('AmpliCol worker is missing its ten percent reserve')
    if quota == 0 and target:
        raise PoolError('AmpliCol zero-quota worker has a nonempty reserve')
    if not math.isfinite(threshold) or threshold < pool['cutoff'] or threshold <= 0.:
        raise PoolError('Invalid AmpliCol final threshold')
    candidates = pool['candidates']
    retained = []
    log_threshold = math.log(threshold)
    excluded_priority = max((row[1] for row in candidates if not row[2]), default=-math.inf)
    retained_priority = min((row[1] for row in candidates if row[2]), default=math.inf)
    if retained_priority < excluded_priority - 1.e-11:
        raise PoolError('AmpliCol reserve violates candidate priority ordering')
    # Fortran stores huge(1d0) when the first excluded log priority cannot be
    # exponentiated. Preserve its actual rank threshold for membership and
    # corrections; every representable raw weight then remains below it.
    if threshold == sys.float_info.max:
        log_threshold = max(log_threshold, excluded_priority)
    for index, (weight, priority, correction, tail, factor) in enumerate(candidates):
        if bool(tail) != (weight > threshold):
            raise PoolError('AmpliCol raw overweight flag disagrees with the threshold')
        if correction:
            if priority < log_threshold - 1.e-11:
                raise PoolError('AmpliCol reserve contains a rejected priority')
            retained.append(index)
        elif priority > log_threshold + 1.e-11 or tail:
            raise PoolError('AmpliCol reserve omits a passing candidate')
    if len(retained) != target:
        raise PoolError('AmpliCol generated reserve count disagrees with its target')
    if target:
        logs = [max(0., math.log(candidates[index][0]) - log_threshold)
                for index in retained]
        largest = max(logs)
        scaled = [math.exp(value-largest) for value in logs]
        total = math.fsum(scaled)
        for index, value in zip(retained, scaled):
            expected = value * target / total
            if not math.isclose(candidates[index][2], expected, rel_tol=1.e-10, abs_tol=1.e-300):
                raise PoolError('AmpliCol normalized correction disagrees with its threshold')
    weights = [candidates[index][2] * candidates[index][4] for index in retained]
    if not all(math.isfinite(value) and value > 0. for value in weights):
        raise PoolError('Nonfinite AmpliCol corrected event magnitude')
    flags = [bool(candidates[index][3]) for index in retained]
    reserve_tail = _weighted_fraction(weights, flags)
    tail_weights = [value for value, flag in zip(weights, flags) if flag]
    ordinary = sorted(value for value, flag in zip(weights, flags) if not flag)
    if not quota or not tail_weights:
        worst_tail = 0.
    elif len(tail_weights) >= quota:
        worst_tail = 1.
    else:
        subset = tail_weights + ordinary[:quota-len(tail_weights)]
        worst_tail = _weighted_fraction(subset, [True]*len(tail_weights) +
                                        [False]*(quota-len(tail_weights)))
    if pool['trials'] and pool['mean_abs']:
        # Every trial above threshold exceeds the storage cutoff, hence appears
        # in candidates. Dividing before summing avoids a large trial sum.
        tail_mean = math.fsum(row[0] / pool['trials'] for row in candidates if row[3])
        full_tail = tail_mean / pool['mean_abs']
    else:
        if candidates:
            raise PoolError('AmpliCol nonempty candidates have zero trial rate')
        full_tail = 0.
    diagnostics = dict(full_trial_tail=full_tail, reserve_tail=reserve_tail,
                       worst_subset_tail=worst_tail)
    for key, value in diagnostics.items():
        reported = pool[key]
        if not 0. <= reported <= 1. or not math.isclose(reported, value,
                                                       rel_tol=1.e-9, abs_tol=1.e-12):
            raise PoolError('AmpliCol %s diagnostic disagrees with candidate data' % key)
    return retained, diagnostics


def available_events(pools):
    """Count generated reserve events; legacy pools remain diagnostic-only."""
    return sum(sum(row[2] > 0. for row in pool['candidates'])
               if pool.get('version', 1) in (2, 3) else len(pool['candidates'])
               for pool in pools)


def pool_status(pools, quota, minimum_reserve=None):
    """Return validated per-worker diagnostics without consuming randomness."""
    if isinstance(quota, bool) or int(quota) != quota or quota < 0:
        raise PoolError('Invalid AmpliCol final event quota')
    pools = list(pools)
    workers = [_worker_status(pool) for pool in pools]
    if sum(pool['final_quota'] for pool in pools) != quota:
        raise PoolError('AmpliCol final quota disagrees with the survey allocation')
    diagnostics = dict(available=available_events(pools),
                       reserve=sum(pool['generated_target'] for pool in pools),
                       selected=quota)
    for key in ('full_trial_tail', 'reserve_tail', 'worst_subset_tail'):
        diagnostics[key] = max((status[key] for unused, status in workers), default=0.)
    diagnostics['overweight'] = max(diagnostics[key] for key in
                                   ('full_trial_tail', 'reserve_tail', 'worst_subset_tail'))
    diagnostics['max_correction'] = max((row[2] for pool in pools
                                         for row in pool['candidates']), default=1.)
    return diagnostics


def select_candidates(pools, quota, rng=None, minimum_reserve=None):
    """Uniformly trim each independent worker once, preserving its corrections."""
    rng = rng or random
    pools = list(pools)
    diagnostics = pool_status(pools, quota)
    selection = {}
    weights, flags = [], []
    for pool_index, pool in enumerate(pools):
        retained, unused = _worker_status(pool)
        for index in rng.sample(retained, pool['final_quota']):
            row = pool['candidates'][index]
            selection[pool_index, index] = row[2]
            weights.append(row[2] * row[4])
            flags.append(bool(row[3]))
    diagnostics['selected_tail'] = _weighted_fraction(weights, flags)
    diagnostics['sum_corrections'] = math.fsum(selection.values())
    return selection, diagnostics


def finalize_channel(pools, quota, output_path, normalization_factor, rng=None,
                     cross_sections=None, minimum_reserve=None,
                     allowed_overweight=ALLOWED_OVERWEIGHT_FACTOR):
    """Collect exactly the survey-assigned ``quota`` events, atomically.

    Raw LHE candidates have signed unit nominal weights, including any bias
    inverse. ``normalization_factor`` is MG5's updated global absolute rate
    (``average``/``bias``), rate divided by total event count (``sum``), or one
    (``unity``). Residual native corrections multiply that nominal scale.
    ``cross_sections`` gives all LHE process rows as (label, signed rate,error).

    Return selection diagnostics. A quota mismatch, excessive full tail, or
    malformed input leaves any existing final file untouched. Candidate files
    and sidecars are retained so top-ups and finalization can be audited or
    repeated.
    """
    if not math.isfinite(normalization_factor) or normalization_factor < 0.:
        raise PoolError('Invalid global AmpliCol event normalization')
    if not math.isfinite(allowed_overweight) or allowed_overweight <= 0.:
        raise PoolError('Invalid AmpliCol allowed overweight fraction')
    pools = list(pools)
    if not pools:
        raise PoolError('Cannot finalize an AmpliCol channel without pools')
    diagnostics = pool_status(pools, quota)
    if diagnostics['overweight'] >= allowed_overweight:
        raise PoolOverweight(diagnostics['overweight'], allowed_overweight)
    selection, diagnostics = select_candidates(pools, quota, rng)
    if diagnostics['selected_tail'] >= allowed_overweight:
        raise PoolOverweight(diagnostics['selected_tail'], allowed_overweight)
    output_path = os.path.abspath(os.fspath(output_path))
    directory = os.path.dirname(output_path)
    preamble = footer = None
    written = 0
    actual_weights, actual_flags = [], []
    max_worker_tail = 0.
    handle, temporary = tempfile.mkstemp(prefix='.ampli-final-', dir=directory)
    try:
        # Spool selected events first so no partially validated LHE header can
        # replace a previous successful finalization.
        with os.fdopen(handle, 'w+') as events:
            for pool_index, pool in enumerate(pools):
                count = 0
                worker_weights, worker_flags = [], []
                for kind, part in _lhe_parts(pool['lhe_path']):
                    if kind == 'preamble':
                        valid_header = '<init>' in part and '<LesHouchesEvents' in part
                        if pool['ncandidates'] and not valid_header:
                            raise PoolError('Missing AmpliCol candidate LHE header')
                        # Fortran writes its header on the first event. A
                        # legitimate zero-candidate worker therefore has only
                        # a closing tag; borrow a later nonempty worker header.
                        if preamble is None and valid_header:
                            preamble = part
                    elif kind == 'footer' and footer is None:
                        footer = part
                    elif kind == 'event':
                        if count >= pool['ncandidates']:
                            raise PoolError('AmpliCol sidecar/LHE candidate counts disagree: %s' % pool['path'])
                        raw_magnitude = abs(_read_event_weight(part))
                        expected = pool['candidates'][count][4]
                        # MG5 writes candidate XWGTUP with Fortran e14.8.
                        if not math.isclose(raw_magnitude, expected, rel_tol=5.1e-8, abs_tol=1.e-300):
                            raise PoolError('AmpliCol nominal LHE factor disagrees with its sidecar')
                        correction = selection.get((pool_index, count))
                        if correction is not None:
                            worker_weights.append(raw_magnitude * correction)
                            worker_flags.append(bool(pool['candidates'][count][3]))
                            events.write(_event_weight(part, normalization_factor * correction))
                            written += 1
                        count += 1
                if count != pool['ncandidates']:
                    raise PoolError('AmpliCol sidecar/LHE candidate counts disagree: %s' % pool['path'])
                worker_tail = _weighted_fraction(worker_weights, worker_flags)
                if worker_tail >= allowed_overweight:
                    raise PoolOverweight(worker_tail, allowed_overweight)
                max_worker_tail = max(max_worker_tail, worker_tail)
                actual_weights.extend(worker_weights)
                actual_flags.extend(worker_flags)
            diagnostics['selected_tail'] = _weighted_fraction(actual_weights, actual_flags)
            diagnostics['max_worker_selected_tail'] = max_worker_tail
            if diagnostics['selected_tail'] >= allowed_overweight:
                raise PoolOverweight(diagnostics['selected_tail'], allowed_overweight)
            if written != quota:
                raise PoolError('Incomplete AmpliCol final candidate selection')
            if preamble is None:
                raise PoolError('No AmpliCol pool contains an LHE init block')
            if cross_sections is None:
                raise PoolError('Final AmpliCol process rates are required')
            preamble = _init_rates(preamble, cross_sections, normalization_factor)
            preamble = _header_count(preamble, quota)
            events.flush()
            events.seek(0)
            final_handle, final_path = tempfile.mkstemp(prefix='.ampli-lhe-', dir=directory)
            try:
                with os.fdopen(final_handle, 'w') as destination:
                    destination.write(preamble)
                    shutil.copyfileobj(events, destination)
                    destination.write(footer)
                os.replace(final_path, output_path)
            finally:
                if os.path.exists(final_path):
                    os.unlink(final_path)
    finally:
        os.unlink(temporary)
    return diagnostics
