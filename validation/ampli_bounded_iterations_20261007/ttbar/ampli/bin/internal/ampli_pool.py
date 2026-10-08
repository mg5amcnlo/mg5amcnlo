"""Validate native AmpliCol iterations and collect updated channel quotas.

Survey estimates seed generation. Native production iterations update rates;
their historical envelopes and candidate uniforms support deterministic upward
rethresholding before a single uniform trim of each channel's pooled reserve.
The earlier survey-normalized protocols remain readable for legacy samples.
"""

import math
import copy
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
    nominal LHE factor. Version 3 records trial-scheduled grid adaptation;
    version 4 records native nonzero-scheduled iterations and their separate
    envelopes. Raw candidate weights refer to the proposal at draw time, and
    rows follow the unmodified candidate LHE order.
    """
    path = os.fspath(path)
    if os.path.isdir(path):
        path = os.path.join(path, 'ampli_pool.dat')
    try:
        with open(path) as stream:
            header = stream.readline().split()
            if header == ['MG5_AMPLI_POOL', '4']:
                return _read_native_pool(stream, path)
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


def _native_moments(record, trials, path):
    values = list(map(_real, record))
    if len(values) != 5:
        raise PoolError('Invalid AmpliCol iteration moments: %s' % path)
    absolute, signed, m2_abs, m2_signed, covariance = values
    if (absolute < 0. or abs(signed) > absolute*(1.+1.e-10)+1.e-300 or
            min(m2_abs, m2_signed) < 0. or
            abs(covariance) > math.sqrt(m2_abs)*math.sqrt(m2_signed)*(1.+1.e-10)+1.e-300 or
            (trials == 0 and any(values))):
        raise PoolError('Inconsistent AmpliCol iteration moments: %s' % path)
    return dict(zip(('mean_abs', 'mean_signed', 'm2_abs', 'm2_signed', 'covariance_sum'), values))


def _read_native_pool(stream, path):
    counts = list(map(int, stream.readline().split()))
    if len(counts) != 3:
        raise PoolError('Invalid native AmpliCol pool counts: %s' % path)
    trials, ncandidates, nepochs = counts
    if trials < 0 or not 0 <= ncandidates <= trials or not 0 <= nepochs <= trials:
        raise PoolError('Invalid native AmpliCol pool dimensions: %s' % path)
    moments = _native_moments(stream.readline().split(), trials, path)
    row = stream.readline().split()
    if len(row) != 6:
        raise PoolError('Invalid native AmpliCol generation record: %s' % path)
    generation = dict(zip(('generated_target', 'final_quota', 'log_z',
                          'full_trial_tail', 'reserve_tail', 'worst_subset_tail'),
                         [int(row[0]), int(row[1])] + list(map(_real, row[2:]))))
    dimensions = list(map(int, stream.readline().split()))
    if len(dimensions) != 2:
        raise PoolError('Invalid native AmpliCol adaptation counts: %s' % path)
    ndim, updates = dimensions
    mask = list(map(int, stream.readline().split()))
    if (ndim <= 0 or len(mask) != ndim or any(value not in (0, 1) for value in mask)
            or not 0 <= updates <= nepochs or (not any(mask) and updates)):
        raise PoolError('Invalid native AmpliCol adaptation mask: %s' % path)
    epochs = []
    for epoch_id in range(1, nepochs+1):
        row = stream.readline().split()
        if len(row) != 13:
            raise PoolError('Invalid native AmpliCol iteration record: %s' % path)
        identifier, count, nonzero, target = map(int, row[:4])
        cutoff, envelope, threshold = map(_real, row[9:12])
        eligible = int(row[12])
        if (identifier != epoch_id or not 0 < target <= nonzero <= count or
                min(cutoff, envelope, threshold) < 0. or envelope == 0. or
                (eligible and threshold < cutoff) or eligible not in (0, 1)):
            raise PoolError('Invalid native AmpliCol iteration state: %s' % path)
        epochs.append(dict(id=identifier, trials=count, nonzero=nonzero,
            target_nonzero=target, cutoff=cutoff, envelope=envelope,
            threshold=threshold, eligible=bool(eligible),
            **_native_moments(row[4:9], count, path)))
    if sum(epoch['trials'] for epoch in epochs) != trials:
        raise PoolError('AmpliCol iteration trial counts disagree with the pool')
    if (sum(epoch['eligible'] for epoch in epochs) > 8 or
            (ncandidates and not any(epoch['eligible'] for epoch in epochs))):
        raise PoolError('Invalid AmpliCol active iteration history')
    combined = combine_moments(epochs)
    for key, value in moments.items():
        scale = max(abs(value), abs(combined[key]), moments['mean_abs'] if key.startswith('mean_') else 0., 1.e-300)
        if abs(value-combined[key]) > 1.e-8*scale:
            raise PoolError('AmpliCol aggregate moments disagree with its iterations')
    candidates, candidate_epochs = [], []
    for unused in range(ncandidates):
        row = stream.readline().split()
        if len(row) != 6:
            raise PoolError('Invalid native AmpliCol candidate row: %s' % path)
        epoch_id = int(row[0])
        weight, priority, correction, tail, factor = map(_real, row[1:])
        if (not 1 <= epoch_id <= nepochs or weight <= 0. or correction < 0.
                or tail not in (0., 1.) or factor <= 0.):
            raise PoolError('Invalid native AmpliCol candidate state: %s' % path)
        cutoff = epochs[epoch_id-1]['cutoff']
        if priority < max(math.log(weight), math.log(cutoff) if cutoff else -math.inf)-1.e-11:
            raise PoolError('Invalid native AmpliCol storage priority: %s' % path)
        candidate_epochs.append(epoch_id)
        candidates.append((weight, priority, correction, tail, factor))
    if stream.read().strip():
        raise PoolError('Unexpected trailing native AmpliCol pool data: %s' % path)
    event_epochs = sorted(set(candidate_epochs))
    if [epoch['id'] for epoch in epochs if epoch['eligible']] != event_epochs[-8:]:
        raise PoolError('AmpliCol eligibility differs from the last eight event iterations')
    pool = dict(path=os.path.abspath(path), version=4,
        lhe_path=os.path.join(os.path.dirname(os.path.abspath(path)), 'ampli_candidates.lhe'),
        trials=trials, ncandidates=ncandidates, epochs=epochs,
        candidates=candidates, candidate_epochs=candidate_epochs,
        adaptation=dict(mode='adaptive_unfolded' if any(mask) else 'frozen',
                        ndim=ndim, mask=mask, updates=updates,
                        completed_iterations=nepochs, schedule='nonzero_trials'),
        **moments, **generation)
    _native_worker_status(pool)
    return pool


def combine_moments(pools):
    """Merge diagnostic trial moments with Chan's moment identities.

    The reported errors use the usual sample-variance expression. Adaptation
    and quota stopping prevent interpreting it as an unbiased uncertainty on
    a production integral. Native published rates use combine_native_rates;
    older protocols publish the independent survey.
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


def combine_native_rates(survey, pools):
    """Combine the survey once with every completed production iteration.

    Match AmpliCol's ``compute_uncertainty`` and ``update_res_and_unc``:
    iteration errors use M2/N**2, then trial-count weighting includes the
    between-iteration contribution. These are conventional adaptive Monte
    Carlo error estimates; nonzero-count stopping is not claimed unbiased.
    """
    count = int(survey['trials'])
    if count <= 0:
        raise PoolError('AmpliCol survey has no integration trials')
    means = [survey['absolute'], survey['signed']]
    variances = [survey['error_abs']**2, survey['error_signed']**2]
    survey_trials = count
    iterations = 0
    for pool in pools:
        for epoch in pool['epochs']:
            n = epoch['trials']
            if n <= 0:
                raise PoolError('AmpliCol production iteration has no trials')
            combined = count + n
            old_fraction, fraction = count / combined, n / combined
            for index, name in enumerate(('abs', 'signed')):
                delta = epoch['mean_' + name] - means[index]
                epoch_variance = epoch['m2_' + name] / n / n
                variances[index] = (variances[index] * old_fraction**2 +
                    epoch_variance * fraction**2 +
                    delta**2 * old_fraction * fraction / combined)
                means[index] += delta * fraction
            count = combined
            iterations += 1
    return dict(trials=count, absolute=means[0], signed=means[1],
                error_abs=math.sqrt(max(0., variances[0])),
                error_signed=math.sqrt(max(0., variances[1])),
                survey_trials=survey_trials, production_trials=count-survey_trials,
                production_iterations=iterations)


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


def _worst_subset_tail(weights, flags, quota):
    if quota == 0 or not any(flags):
        return 0.
    tail = [weight for weight, flag in zip(weights, flags) if flag]
    if len(tail) >= quota:
        return 1.
    ordinary = sorted(weight for weight, flag in zip(weights, flags) if not flag)
    kept = ordinary[:quota-len(tail)]
    return _weighted_fraction(tail+kept, [True]*len(tail)+[False]*len(kept))


def _native_worker_status(pool, verify=True):
    """Validate native iteration-specific rejection without revisiting physics."""
    candidates, epochs = pool['candidates'], pool['epochs']
    target, quota, log_z = pool['generated_target'], pool['final_quota'], pool['log_z']
    if (target < 0 or quota < 0 or (verify and target < (11*quota+9)//10)):
        raise PoolError('Invalid native AmpliCol event reserve')
    retained, log_corrections = [], []
    active_trials = sum(epoch['trials'] for epoch in epochs if epoch['eligible'])
    active_abs = math.fsum(epoch['mean_abs']*(epoch['trials']/active_trials)
                          for epoch in epochs if epoch['eligible']) if active_trials else 0.
    full_tail_mean = 0.
    for epoch in epochs:
        if not epoch['eligible']:
            continue
        threshold = epoch['threshold']
        log_expected = math.log(epoch['envelope']) + log_z
        if (threshold <= 0. or threshold < epoch['cutoff'] or
                abs(math.log(threshold)-min(log_expected, math.log(sys.float_info.max))) > 1.e-9):
            raise PoolError('AmpliCol iteration threshold disagrees with the native envelope')
    for index, (epoch_id, row) in enumerate(zip(pool['candidate_epochs'], candidates)):
        weight, priority, correction, tail, unused = row
        epoch = epochs[epoch_id-1]
        if not epoch['eligible']:
            if correction or tail:
                raise PoolError('Expired AmpliCol iteration retains event corrections')
            continue
        threshold = epoch['threshold']
        if bool(tail) != (weight > threshold):
            raise PoolError('AmpliCol native tail flag disagrees with its iteration threshold')
        rank = priority-math.log(epoch['envelope'])
        if correction:
            if rank < log_z-1.e-10:
                raise PoolError('AmpliCol native reserve contains a rejected priority')
            retained.append(index)
            log_corrections.append(max(0., math.log(weight)-math.log(threshold)))
        elif rank > log_z+1.e-10 or tail:
            raise PoolError('AmpliCol native reserve omits a passing candidate')
        if tail:
            full_tail_mean += weight/active_trials
    if len(retained) != target:
        raise PoolError('AmpliCol native reserve count disagrees with its target')
    if target:
        maximum = max(log_corrections)
        corrections = [math.exp(value-maximum) for value in log_corrections]
        normalization = target/math.fsum(corrections)
        for index, correction in zip(retained, corrections):
            if not math.isclose(candidates[index][2], correction*normalization,
                                rel_tol=1.e-9, abs_tol=1.e-300):
                raise PoolError('AmpliCol native correction disagrees with its iteration threshold')
    weights = [candidates[index][2]*candidates[index][4] for index in retained]
    flags = [bool(candidates[index][3]) for index in retained]
    if not all(math.isfinite(weight) and weight > 0. for weight in weights):
        raise PoolError('Invalid corrected native AmpliCol event magnitude')
    diagnostics = dict(full_trial_tail=full_tail_mean/active_abs if active_abs else 0.,
        reserve_tail=_weighted_fraction(weights, flags),
        worst_subset_tail=_worst_subset_tail(weights, flags, quota))
    if verify:
        for key, expected in diagnostics.items():
            if (not 0. <= pool[key] <= 1. or
                    not math.isclose(pool[key], expected, rel_tol=1.e-8, abs_tol=1.e-12)):
                raise PoolError('AmpliCol native %s disagrees with candidate data' % key)
    return retained, diagnostics


def tighten_native_pool(pool):
    """Raise the common rejection level enough to remove overweight tails.

    This is deterministic thinning with the original candidate uniforms;
    neither the immutable sidecar nor its integration estimates are changed.
    """
    if pool.get('version') != 4:
        raise PoolError('Native rethresholding requires AmpliCol protocol 4')
    updated = copy.deepcopy(pool)
    log_z = max([pool['log_z']] +
        [math.log(row[0])-math.log(pool['epochs'][epoch_id-1]['envelope'])+1.e-12
         for epoch_id, row in zip(pool['candidate_epochs'], pool['candidates'])
         if pool['epochs'][epoch_id-1]['eligible']])
    updated['log_z'] = log_z
    updated['native_log_z'] = pool.get('native_log_z', pool['log_z'])
    updated['native_generated_target'] = pool.get('native_generated_target', pool['generated_target'])
    for epoch in updated['epochs']:
        if epoch['eligible']:
            value = math.log(epoch['envelope'])+log_z
            epoch['threshold'] = max(epoch['cutoff'], math.exp(min(value, math.log(sys.float_info.max))))
    rows, retained = [], []
    for index, (epoch_id, row) in enumerate(zip(updated['candidate_epochs'], updated['candidates'])):
        weight, priority, unused, unused_tail, factor = row
        epoch = updated['epochs'][epoch_id-1]
        keep = epoch['eligible'] and priority-math.log(epoch['envelope']) > log_z
        tail = epoch['eligible'] and weight > epoch['threshold']
        correction = max(1., weight/epoch['threshold']) if keep else 0.
        if keep:
            retained.append(index)
        rows.append((weight, priority, correction, int(tail), factor))
    total = math.fsum(rows[index][2] for index in retained)
    scale = len(retained)/total if total else 1.
    updated['candidates'] = [(w, priority, correction*scale, tail, factor)
                            for w, priority, correction, tail, factor in rows]
    updated['generated_target'] = len(retained)
    unused, diagnostics = _native_worker_status(updated, verify=False)
    updated.update(diagnostics)
    updated['collection_rethresholded'] = pool.get('collection_rethresholded', False) or log_z > pool['log_z']
    return updated


def _worker_status(pool):
    """Independently verify the worker's threshold, corrections and tail mass."""
    if pool.get('version') == 4:
        return _native_worker_status(pool, verify=not pool.get('collection_rethresholded', False))
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
               if pool.get('version', 1) in (2, 3, 4) else len(pool['candidates'])
               for pool in pools)


def pool_status(pools, quota, minimum_reserve=None):
    """Return validated per-worker diagnostics without consuming randomness."""
    if isinstance(quota, bool) or int(quota) != quota or quota < 0:
        raise PoolError('Invalid AmpliCol final event quota')
    pools = list(pools)
    workers = [_worker_status(pool) for pool in pools]
    if pools and all(pool.get('version') == 4 for pool in pools):
        available = available_events(pools)
        if quota > available:
            raise PoolShortage(quota, available)
        weights = [pool['candidates'][index][2]*pool['candidates'][index][4]
                   for pool, (retained, unused) in zip(pools, workers) for index in retained]
        flags = [bool(pool['candidates'][index][3])
                 for pool, (retained, unused) in zip(pools, workers) for index in retained]
        diagnostics = dict(available=available, reserve=available, selected=quota,
            full_trial_tail=max((status['full_trial_tail'] for unused, status in workers), default=0.),
            reserve_tail=_weighted_fraction(weights, flags),
            worst_subset_tail=_worst_subset_tail(weights, flags, quota),
            max_correction=max((row[2] for pool in pools for row in pool['candidates']), default=1.))
        diagnostics['overweight'] = max(diagnostics[key] for key in
            ('full_trial_tail', 'reserve_tail', 'worst_subset_tail'))
        return diagnostics
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
    """Uniformly trim a native channel once, or each legacy worker separately."""
    rng = rng or random
    pools = list(pools)
    diagnostics = pool_status(pools, quota)
    selection = {}
    weights, flags = [], []
    if pools and all(pool.get('version') == 4 for pool in pools):
        available = [(pool_index, index) for pool_index, pool in enumerate(pools)
                     for index, row in enumerate(pool['candidates']) if row[2] > 0.]
        for pool_index, index in rng.sample(available, quota):
            row = pools[pool_index]['candidates'][index]
            selection[pool_index, index] = row[2]
            weights.append(row[2]*row[4])
            flags.append(bool(row[3]))
        diagnostics['selected_tail'] = _weighted_fraction(weights, flags)
        diagnostics['sum_corrections'] = math.fsum(selection.values())
        return selection, diagnostics
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
    """Collect exactly the coordinator-assigned ``quota`` events, atomically.

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
                if pool.get('version') != 4 and worker_tail >= allowed_overweight:
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
