"""Small, frozen Born-times-logarithm models for barrier-pooled virtuals.

NumPy is imported only when fitting/inference is requested. This module has
no dependency on validation scripts or on a particular physics process.
"""
import itertools
import math
from functools import lru_cache

MAX_TERMS = 512
# Cubic screening is deliberately a single additional objective. Bound the
# temporary dense selection fit as well as the frozen sparse representation.
MAX_CUBIC_FEATURES = 18
MAX_CUBIC_DESIGN_ELEMENTS = 40000000
# Cache complete fold-local designs and both noise scalings within this budget.
MAX_FIT_CACHE_ELEMENTS = 8000000
CUBIC_RIDGE = .01
MIN_ROWS = 60
ROWS_PER_TERM = 5
TAIL_MARGIN = .10
RIDGES = (1.e-4, .01, 1.)
LOSSES = ('huber', 'square', 'born_huber', 'born_square')
BLENDS = (.25, .5, .75, 1.)


def empty_order(order, nf, reason):
    return dict(order=order, blend=0., born_scale=1., target_scale=1., degree=0,
                lower=[0.] * nf, upper=[0.] * nf, terms=[], reason=reason)


def coordinates(x, lower, upper):
    import numpy as np
    x, lo, hi = (np.asarray(a, dtype=float) for a in (x, lower, upper))
    width = hi - lo
    active = width > 1.e-12
    z = np.divide(2. * (x-lo), width, out=np.ones_like(x), where=active)-1.
    allowance = np.where(active, TAIL_MARGIN*width, 1.e-10*np.maximum(1., np.abs(lo)))
    outside = np.any((x < lo-allowance) | (x > hi+allowance), axis=1)
    return np.clip(z, -1., 1.), outside


def pairs_for(x, degree, nscalar=None, basis='full'):
    import numpy as np
    active = np.flatnonzero(np.ptp(x, axis=0) > 1.e-12).tolist()
    if nscalar is not None:
        active = [i for i in active if basis != 'relative' or i != nscalar-1]
    pairs = [(-1, -1)] + [(i, -1) for i in active]
    if degree >= 2:
        pairs += list(itertools.combinations_with_replacement(active, 2))
    if degree == 3:
        pairs = [pair + (-1,) for pair in pairs]
        pairs += list(itertools.combinations_with_replacement(active, 3))
    if nscalar is not None:
        # Preserve the even polynomial and an odd-times-linear block. Products
        # of two signed orientation coordinates would erase this separation.
        pairs = [p for p in pairs if sum(i >= nscalar for i in p) <= 1 and
                 not (sum(i >= nscalar for i in p) and sum(i >= 0 for i in p) > 2)]
    return pairs


def basis_size(x, degree, nscalar=None, basis='full'):
    """Count before allocating a potentially enormous high-multiplicity basis."""
    import numpy as np
    active = np.flatnonzero(np.ptp(x, axis=0) > 1.e-12)
    if nscalar is None:
        # Degrees are at most three; keep this available on Python 3.7 too.
        count = 1
        for factor in range(1, degree+1):
            count = count*(len(active)+factor)//factor
        return len(active), count
    even = int(sum(i < nscalar and (basis != 'relative' or i != nscalar-1) for i in active))
    odd = int(sum(i >= nscalar for i in active))
    count = 1+even+odd
    if degree >= 2:
        count += even*(even+1)//2 + even*odd
    if degree == 3:
        count += even*(even+1)*(even+2)//6
    return even, count


def design(z, pairs):
    import numpy as np
    # Build contiguous columns without a temporary row/factor array per term.
    matrix = np.ones((len(z), len(pairs)), order='F')
    for column, indices in enumerate(pairs):
        for index in indices:
            if index >= 0:
                matrix[:, column] *= z[:, index]
    return matrix


@lru_cache(maxsize=1)
def _positive_solver():
    """Use LAPACK's positive-definite solver when optional SciPy is available."""
    try:
        from scipy.linalg import solve
    except ImportError:
        return None
    return solve


def _ridge_solve(gram, rhs, ridge):
    import numpy as np
    system = gram.copy()
    system.flat[::len(system)+1] += ridge
    solve = _positive_solver()
    if solve is not None:
        return solve(system, rhs, assume_a='pos', check_finite=False,
                     overwrite_a=True)
    return np.linalg.solve(system, rhs)


def fit(x, born, virtual, weight, degree, ridge, loss, basis='full', nscalar=None,
        _cache=None):
    import numpy as np
    if degree not in (1, 2, 3):
        raise ValueError('invalid polynomial degree')
    if basis not in ('full', 'relative') or (basis == 'relative' and degree != 3):
        raise ValueError('invalid polynomial feature subset')
    # Catalog 6's final coordinate is absolute log(shat). A relative-only
    # candidate excludes it at every degree, while retaining full-catalog
    # bounds and index numbering for the common native serialization.
    basis_x = x[:, :-1] if basis == 'relative' and nscalar is None else x
    scalar_active, count = basis_size(basis_x, degree, nscalar, basis)
    if degree == 3 and (scalar_active > MAX_CUBIC_FEATURES or
                       len(x)*count > MAX_CUBIC_DESIGN_ELEMENTS):
        raise ValueError('cubic selection exceeds fit budget')
    if degree < 3 and count > MAX_TERMS:
        raise ValueError('polynomial exceeds coefficient budget')
    geometry = None if _cache is None else _cache.get('geometry')
    if geometry is None:
        pairs = pairs_for(basis_x, degree, nscalar, basis)
        if degree < 3 and len(pairs) > MAX_TERMS:
            raise ValueError('polynomial exceeds coefficient budget')
        lo, hi = x.min(axis=0), x.max(axis=0)
        if nscalar is not None:
            lo[nscalar:], hi[nscalar:] = -1., 1.
        z, _ = coordinates(x, lo, hi)
        matrix = design(z, pairs)
        geometry = pairs, lo, hi, matrix
        if _cache is not None:
            _cache['geometry'] = geometry
    pairs, lo, hi, matrix = geometry
    coeff, norm, bscale, vscale = _solve_design(matrix, born, virtual, weight, ridge, loss, _cache)
    if degree == 3 and len(pairs) > MAX_TERMS:
        # Preserve every constant/linear/quadratic term. Select the additional
        # cubic terms entirely within this fitting fold, then refit the sparse
        # model; held-out targets never influence the feature selection.
        lower = sum(indices[2] < 0 for indices in pairs)
        if lower >= MAX_TERMS:
            raise ValueError('mandatory polynomial terms exceed coefficient budget')
        importance = np.abs(coeff[lower:])*norm[lower:]
        ranked = np.argsort(-importance, kind='stable')[:MAX_TERMS-lower] + lower
        selected = np.concatenate((np.arange(lower), np.sort(ranked)))
        pairs = [pairs[i] for i in selected]
        coeff, norm, bscale, vscale = _solve_design(
            matrix[:, selected], born, virtual, weight, ridge, loss)
    terms = [dict(pair=indices, coefficient=float(value)) for indices, value, size in
             zip(pairs, coeff, norm) if size > 0]
    return dict(degree=degree, lower=lo.tolist(), upper=hi.tolist(), terms=terms,
                born_scale=bscale, target_scale=vscale)


def _solve_design(matrix, born, virtual, weight, ridge, loss, _cache=None):
    import numpy as np
    # A Born-dependent noise scale prevents a few very large integration
    # weights from determining a polynomial for the whole subprocess. Keep
    # V-B*P as the regression residual, including signed and zero Born rows;
    # never construct the potentially singular target V/B.
    born_noise = loss.startswith('born_')
    key = 'born_noise' if born_noise else 'constant_noise'
    prepared = None if _cache is None else _cache.get(key)
    if prepared is None:
        w = weight / weight.sum()
        bscale = float(w @ np.abs(born))
        vscale = float(w @ np.abs(virtual))
        if not math.isfinite(bscale) or not math.isfinite(vscale) or not bscale or not vscale:
            raise ValueError('zero fit scale')
        a = (born/bscale)[:, None] * matrix
        noise = np.maximum(np.abs(born)/bscale, 1.e-12) if born_noise else np.ones(len(born))
        fit_weight = w/noise
        norm = np.sqrt(fit_weight @ (a*a))
        if not np.all(np.isfinite(norm)):
            raise ValueError('nonfinite design scale')
        keep = norm > max(float(norm.max())*1.e-10, 1.e-20)
        if not np.any(keep):
            raise ValueError('empty design')
        a = a[:, keep] / norm[keep]
        y = virtual/vscale
        gram = a.T @ (fit_weight[:, None]*a)
        rhs = a.T @ (fit_weight*y)
        norm[~keep] = 0.
        prepared = a, noise, fit_weight, norm, keep, y, bscale, vscale, gram, rhs
        if _cache is not None:
            _cache[key] = prepared
    a, noise, fit_weight, norm, keep, y, bscale, vscale, gram, rhs = prepared
    irls = np.ones(len(born))
    huber = loss.endswith('huber')
    for iteration in range((20 if born_noise else 8) if huber else 1):
        if iteration:
            ww = fit_weight*irls
            gram = a.T @ (ww[:, None]*a)
            rhs = a.T @ (ww*y)
        coeff = _ridge_solve(gram, rhs, ridge)
        if huber:
            updated = np.minimum(1., .25*noise/np.maximum(np.abs(y-a@coeff), 1.e-20))
            # Identical weights imply an identical next solve. Preserve the
            # prescribed iteration cap for fits which have not converged.
            if np.array_equal(updated, irls):
                break
            irls = updated
    if not np.all(np.isfinite(coeff)):
        raise ValueError('nonfinite fit')
    result = np.zeros(matrix.shape[1])
    result[keep] = coeff/norm[keep]
    return result, norm, bscale, vscale


def predict(entry, x, born, mint, blend=None):
    import numpy as np
    lam = entry.get('blend', 1.) if blend is None else blend
    if not lam:
        return mint.copy()
    z, _ = coordinates(x, entry['lower'], entry['upper'])
    value = design(z, [t['pair'] for t in entry['terms']]) @ np.asarray(
        [t['coefficient'] for t in entry['terms']])
    shared = (born/entry['born_scale'])*value*entry['target_scale']
    answer = mint + lam*(shared-mint)
    # Keep the bounded prediction outside the training box. Native inference
    # requests an exact loop there, so continuation cannot bias the estimator.
    # Falling back to MINT would reintroduce its largest residuals at the edge.
    invalid = ~np.isfinite(answer)
    answer[invalid] = mint[invalid]
    return answer


class Marginals:
    """Literal native fill/regrid recurrence for one channel and order.

    All sampled observations, including exact zero residuals, are retained just
    as in the pooled native path. Sampling probabilities already acted through
    which observations were collected; the native recurrence does not reweight
    them again. The regression/validation objective does use inverse probability.
    """
    def __init__(self, nx, bins):
        import numpy as np
        self.nx, self.bins = nx, bins
        self.average = np.zeros((nx, bins))
        self.count = np.zeros((nx, bins), dtype=int)

    def resize(self, bins):
        import numpy as np
        if bins == self.bins:
            return
        if bins < self.bins:
            self.__init__(self.nx, bins)
            return
        if bins != 2*self.bins:
            raise ValueError('invalid MINT bin transition')
        average = np.zeros((self.nx, bins))
        count = np.zeros_like(average, dtype=int)
        for i in range(self.bins):
            average[:, 2*i+1] = self.average[:, i]
            count[:, 2*i+1] = np.where(self.count[:, i], np.maximum(self.count[:, i]//2, 1), 0)
            average[:, 2*i] = self.average[:, i] if not i else (
                self.average[:, i]+self.average[:, i-1])/2.
            n = self.count[:, i] if not i else self.count[:, i]+self.count[:, i-1]
            count[:, 2*i] = np.where(n, np.maximum(n//(2 if not i else 4), 1), 0)
        self.average, self.count, self.bins = average, count, bins

    def update(self, rows):
        import numpy as np
        if not rows:
            return
        b = np.zeros_like(self.average)
        v = np.zeros_like(b)
        n = np.zeros_like(b, dtype=int)
        cells = np.minimum((np.asarray([r['x'] for r in rows])*self.bins).astype(int), self.bins-1)
        born = np.asarray([r['born'] for r in rows])
        virtual = np.asarray([r['virtual'] for r in rows])
        for dim in range(self.nx):
            b[dim] = np.bincount(cells[:, dim], weights=born, minlength=self.bins)
            v[dim] = np.bincount(cells[:, dim], weights=virtual, minlength=self.bins)
            n[dim] = np.bincount(cells[:, dim], minlength=self.bins)
        mask = b != 0
        first = mask & (self.average == 0)
        later = mask & ~first
        self.average[first] = v[first]/b[first]
        self.average[later] = (n[later]*v[later]/b[later] +
                              self.count[later]*self.average[later])/(n[later]+self.count[later])
        self.count += n

    def predict(self, row):
        import numpy as np
        cells = np.minimum((np.asarray(row['x'])*self.bins).astype(int), self.bins-1)
        return float(self.average[np.arange(self.nx), cells].mean())*row['born']

    def predict_many(self, rows):
        import numpy as np
        cells = np.minimum((np.asarray([r['x'] for r in rows])*self.bins).astype(int), self.bins-1)
        ratios = self.average[np.arange(self.nx), cells].mean(axis=1)
        return ratios*np.asarray([r['born'] for r in rows])


def mint_predictions(rows, include):
    """Fit each worker/channel on the same fold complement, then freeze it."""
    import numpy as np
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r['directory'], r['channel']), []).append(i)
    result = np.zeros(len(rows))
    for indices in groups.values():
        records = [rows[i] for i in indices]
        model = Marginals(len(records[0]['x']), records[0]['bins'])
        # Iterate over the recorded grid schedule even if this fold has no
        # selected observations in one of its iterations.
        for _, entries in itertools.groupby(indices, lambda i: rows[i]['iteration']):
            entries = list(entries)
            bins = {rows[i]['bins'] for i in entries}
            if len(bins) != 1:
                raise ValueError('MINT bin count changed within an iteration')
            model.resize(bins.pop())
            model.update([rows[i] for i in entries if include[i]])
        result[indices] = model.predict_many(records)
    return result


def forward_split(rows):
    """Hold out whole later iterations, independently in each worker/channel.

    Random folds test interpolation within the capture. This split additionally
    tests how a frozen approximation behaves as the integration grid evolves.
    Workers with a single committed iteration only supply fitting observations.
    """
    import numpy as np
    histories = {}
    for row in rows:
        key = (row['directory'], row['channel'])
        histories.setdefault(key, set()).add(row['iteration'])
    cutoffs = {}
    for key, iterations in histories.items():
        levels = sorted(iterations)
        cutoffs[key] = levels[max(0, 2*len(levels)//3-1)]
    return np.asarray([row['iteration'] <= cutoffs[(row['directory'], row['channel'])]
                       for row in rows])


def fit_order(rows, folds, metadata, order):
    indices = [i for i, r in enumerate(rows) if r['order'] == order]
    subset = [rows[i] for i in indices]
    report = dict(order=order, training_rows=len(subset), candidates=[],
                  comparator='cross-fitted frozen native MINT on identical fold complements')
    fallback = lambda reason: (empty_order(order, metadata['nf'], reason), report)
    if len(subset) < MIN_ROWS:
        return fallback('insufficient_training_samples')
    try:
        import numpy as np
    except ImportError:
        return fallback('numpy_unavailable')
    x = np.asarray([r['features'] for r in subset])
    b, y, w = (np.asarray([r[key] for r in subset]) for key in ('born', 'virtual', 'weight'))
    assignment = np.asarray(folds)[indices]
    if not np.any(b) or not np.any(y):
        return fallback('zero_born_or_virtual_order')
    if any(np.count_nonzero(assignment == f) < 10 for f in range(3)):
        return fallback('insufficient_cross_validation')
    # Normalize only common physical units; no per-point Born division.
    scale = max(float(np.max(np.abs(y))), 1.e-100)
    wn = w/w.sum()
    mint = np.zeros(len(subset))
    for fold in range(3):
        valid = assignment == fold
        mint[valid] = mint_predictions(subset, ~valid)[valid]
    if not np.all(np.isfinite(mint)):
        return fallback('unstable_cross_validation_mint')
    base = (y-mint)/scale
    base_l1 = float(wn @ np.abs(base))
    base_second = float(wn @ (base*base))
    target_l1 = float(wn @ np.abs(y/scale))
    target_rms = math.sqrt(float(wn @ (y/scale)**2))
    report.update(cv_folds=3, baseline_cv_l1=base_l1*scale,
                  baseline_cv_rms=math.sqrt(base_second)*scale,
                  baseline_cv_max=float(np.max(np.abs(base)))*scale,
                  target_l1=target_l1*scale, target_rms=target_rms*scale,
                  baseline_cv_relative_l1=base_l1/target_l1 if target_l1 else None,
                  baseline_cv_relative_rms=math.sqrt(base_second)/target_rms if target_rms else None)
    if not base_l1:
        return fallback('cross_validation_prefers_mint')
    earlier = forward_split(subset)
    later = ~earlier
    forward = np.count_nonzero(later) >= 10 and np.count_nonzero(earlier) >= MIN_ROWS
    if forward:
        forward_mint = mint_predictions(subset, earlier)
        fw = w[later]/w[later].sum()
        fb = (y[later]-forward_mint[later])/scale
        forward_l1 = float(fw @ np.abs(fb))
        forward_second = float(fw @ (fb*fb))
        ft_l1 = float(fw @ np.abs(y[later]/scale))
        ft_rms = math.sqrt(float(fw @ (y[later]/scale)**2))
        report['forward_validation'] = dict(
            fitting_rows=int(np.count_nonzero(earlier)), validation_rows=int(np.count_nonzero(later)),
            baseline_l1=forward_l1*scale, baseline_rms=math.sqrt(forward_second)*scale,
            baseline_max=float(np.max(np.abs(fb)))*scale,
            target_l1=ft_l1*scale, target_rms=ft_rms*scale,
            baseline_relative_l1=forward_l1/ft_l1 if ft_l1 else None,
            baseline_relative_rms=math.sqrt(forward_second)/ft_rms if ft_rms else None)
    else:
        report['forward_validation'] = dict(reason='insufficient_distinct_iteration_history')
    nscalar = None
    if metadata.get('catalog') == 7:
        from .virtual_pool import scalar_feature_count
        nscalar = scalar_feature_count(metadata)
    caches = {}
    cache_elements = 0

    def fit_subset(train, partition, degree, ridge, loss, basis, nterms):
        nonlocal cache_elements
        cache = None
        if degree < 3:
            # Every partition has its own scalers, targets and weighted design.
            # Bound persistent arrays before preparing them; larger fits simply
            # recompute their matrices, with identical models and guards.
            key = partition, degree, basis
            cache = caches.get(key)
            if cache is None:
                n = int(np.count_nonzero(train))
                cost = 3*n*nterms + 2*nterms*nterms + 12*n + 6*x.shape[1] + 4*nterms
                if cache_elements+cost <= MAX_FIT_CACHE_ELEMENTS:
                    cache = caches[key] = {}
                    cache_elements += cost
        return fit(x[train], b[train], y[train], w[train], degree, ridge, loss,
                   basis, nscalar, cache)

    eligible = []
    candidate_residuals = {}
    for degree in (1, 2, 3):
        caches.clear()
        cache_elements = 0
        # Cap both work and flexibility before allocating quadratic matrices.
        active, dense_terms = basis_size(x, degree, nscalar)
        nterms = min(dense_terms, MAX_TERMS) if degree == 3 else dense_terms
        smallest = min(np.count_nonzero(assignment != f) for f in range(3))
        if degree < 3 and (nterms > MAX_TERMS or smallest < ROWS_PER_TERM*dense_terms):
            report.setdefault('skipped_degrees', []).append(degree)
            continue
        choices = ([(CUBIC_RIDGE, 'born_square', basis) for basis in
                    (('full', 'relative') if metadata.get('catalog') in (6, 7) else ('full',))]
                   if degree == 3 else ((r, loss, 'full') for r, loss in itertools.product(RIDGES, LOSSES)))
        for ridge, loss, basis in choices:
            if degree == 3:
                basis_x = x[:, :-1] if basis == 'relative' and nscalar is None else x
                active, dense_terms = basis_size(basis_x, 3, nscalar, basis)
                nterms = min(dense_terms, MAX_TERMS)
                screen = dict(basis=basis)
                report.setdefault('cubic_screens', []).append(screen)
                if basis == 'full':
                    report['cubic_screen'] = screen
                if (active > MAX_CUBIC_FEATURES or len(x)*dense_terms > MAX_CUBIC_DESIGN_ELEMENTS or
                        smallest < ROWS_PER_TERM*dense_terms):
                    screen['reason'] = 'fit_budget_or_insufficient_rows'
                    continue
                if not forward or np.count_nonzero(earlier) < ROWS_PER_TERM*dense_terms:
                    screen['reason'] = 'insufficient_forward_history'
                    continue
            proposal = np.zeros(len(subset))
            forward_proposal = None
            try:
                if degree == 3:
                    # Cheap rejection of unhelpful richer fits before the
                    # three additional CV fits. This screen does not replace
                    # any of the matched-fold or forward acceptance guards.
                    simpler = [candidate for candidate in eligible if candidate['degree'] < 3]
                    reference = min(simpler, key=lambda a: a['cv_l1']) if simpler else None
                    if reference is None or 'forward_l1' not in reference:
                        screen['reason'] = 'no_validated_simpler_reference'
                        continue
                    model = fit_subset(earlier, 'forward', degree, ridge, loss, basis, nterms)
                    forward_proposal = predict(model, x[later], b[later], forward_mint[later])
                    fr = (y[later]-forward_proposal)/scale
                    screen_l1 = float(fw @ np.abs(fr))*scale
                    screen_rms = math.sqrt(float(fw @ (fr*fr)))*scale
                    useful = (screen_l1 < .98*reference['forward_l1'] and
                              screen_rms <= 1.05*reference['forward_rms'])
                    screen.update(reason='passed' if useful else 'no_forward_improvement',
                        forward_l1=screen_l1, forward_rms=screen_rms,
                        reference_l1=reference['forward_l1'], reference_rms=reference['forward_rms'])
                    if not useful:
                        continue
                for fold in range(3):
                    valid, train = assignment == fold, assignment != fold
                    model = fit_subset(train, fold, degree, ridge, loss, basis, nterms)
                    proposal[valid] = predict(model, x[valid], b[valid], mint[valid])
                if forward_proposal is None and forward and np.count_nonzero(earlier) >= ROWS_PER_TERM*nterms:
                    model = fit_subset(earlier, 'forward', degree, ridge, loss, basis, nterms)
                    forward_proposal = predict(model, x[later], b[later], forward_mint[later])
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                if degree == 3:
                    screen['reason'] = 'fit_failed'
                continue
            for blend in BLENDS:
                residual = (y-(mint+blend*(proposal-mint)))/scale
                l1 = float(wn @ np.abs(residual))
                second = float(wn @ (residual*residual))
                delta = wn*(np.abs(base)-np.abs(residual))
                # A conservative paired standard-error estimate: no centering
                # that would hide the influence of a single large gain.
                se = float(np.linalg.norm(delta))
                stable = all(float(wn[assignment == f] @ residual[assignment == f]**2) <=
                             1.05**2*float(wn[assignment == f] @ base[assignment == f]**2)
                             for f in range(3))
                accept = (base_l1-l1 > max(.02*base_l1, 2.*se) and
                          second <= base_second and stable)
                entry = dict(degree=degree, ridge=ridge, loss=loss, basis=basis, blend=blend,
                             parameters=nterms, cv_l1=l1*scale,
                             cv_rms=math.sqrt(second)*scale,
                             cv_max=float(np.max(np.abs(residual)))*scale, gain_error=se*scale,
                             cv_relative_l1=l1/target_l1 if target_l1 else None,
                             cv_relative_rms=math.sqrt(second)/target_rms if target_rms else None,
                             eligible=bool(accept))
                if forward_proposal is not None:
                    fr = (y[later]-(forward_mint[later]+blend*(forward_proposal-forward_mint[later])))/scale
                    fl1 = float(fw @ np.abs(fr))
                    fsecond = float(fw @ (fr*fr))
                    # Do not select an interpolation improvement that already
                    # degrades on future iterations from this same capture.
                    forward_ok = fl1 <= forward_l1 and fsecond <= 1.05**2*forward_second
                    accept = accept and forward_ok
                    entry.update(forward_l1=fl1*scale, forward_rms=math.sqrt(fsecond)*scale,
                                 forward_max=float(np.max(np.abs(fr)))*scale,
                                 forward_relative_l1=fl1/ft_l1 if ft_l1 else None,
                                 forward_relative_rms=math.sqrt(fsecond)/ft_rms if ft_rms else None,
                                 forward_eligible=bool(forward_ok), eligible=bool(accept))
                elif forward:
                    entry['forward_reason'] = 'insufficient_fitting_rows_per_coefficient'
                report['candidates'].append(entry)
                if accept:
                    eligible.append(entry)
                    candidate_residuals[id(entry)] = np.abs(residual)
        if degree == 3 and not any(candidate['degree'] == 3 for candidate in report['candidates']):
            report.setdefault('skipped_degrees', []).append(3)
    if not eligible:
        return fallback('cross_validation_prefers_mint')
    best = min(eligible, key=lambda a: a['cv_l1'])
    # Complexity comparisons need their own paired uncertainty. The uncertainty
    # of the gain over MINT can be much larger and would admit a substantially
    # worse linear approximation even when quadratic fits generalize well.
    def comparable(entry):
        paired = scale*float(np.linalg.norm(wn*(candidate_residuals[id(entry)]-
                                               candidate_residuals[id(best)])))
        entry['difference_from_best_error'] = paired
        if entry['cv_rms'] > 1.05*best['cv_rms']:
            return False
        if 'forward_l1' in best:
            if ('forward_l1' not in entry or entry['forward_l1'] > 1.05*best['forward_l1'] or
                    entry['forward_rms'] > 1.05*best['forward_rms']):
                return False
        return entry['cv_l1'] <= best['cv_l1']+paired
    selected = min((a for a in eligible if comparable(a)),
                   key=lambda a: (a['degree'], a['cv_l1']))
    try:
        model = fit(x, b, y, w, selected['degree'], selected['ridge'], selected['loss'], selected['basis'], nscalar)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return fallback('final_fit_failed')
    model.update(order=order, blend=selected['blend'], reason='selected_by_matched_cross_validation')
    report.update(selected=selected, actual_terms=len(model['terms']))
    return model, report
