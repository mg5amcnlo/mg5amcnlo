"""Small, frozen Born-times-logarithm models for barrier-pooled virtuals.

NumPy is imported only when fitting/inference is requested. This module has
no dependency on validation scripts or on a particular physics process.
"""
import itertools
import math

MAX_TERMS = 512
MIN_ROWS = 60
ROWS_PER_TERM = 5
TAIL_MARGIN = .10
RIDGES = (1.e-4, .01, 1.)
LOSSES = ('huber', 'square')
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


def pairs_for(x, degree):
    import numpy as np
    active = np.flatnonzero(np.ptp(x, axis=0) > 1.e-12).tolist()
    pairs = [(-1, -1)] + [(i, -1) for i in active]
    if degree == 2:
        pairs += list(itertools.combinations_with_replacement(active, 2))
    return pairs


def design(z, pairs):
    import numpy as np
    ones = np.ones(len(z))
    return np.column_stack([(ones if a < 0 else z[:, a]) *
                            (ones if b < 0 else z[:, b]) for a, b in pairs])


def fit(x, born, virtual, weight, degree, ridge, loss):
    import numpy as np
    lo, hi = x.min(axis=0), x.max(axis=0)
    pairs = pairs_for(x, degree)
    if len(pairs) > MAX_TERMS:
        raise ValueError('polynomial exceeds coefficient budget')
    w = weight / weight.sum()
    bscale = float(w @ np.abs(born))
    vscale = float(w @ np.abs(virtual))
    if not math.isfinite(bscale) or not math.isfinite(vscale) or not bscale or not vscale:
        raise ValueError('zero fit scale')
    z, _ = coordinates(x, lo, hi)
    a = (born/bscale)[:, None] * design(z, pairs)
    norm = np.sqrt(w @ (a*a))
    if not np.all(np.isfinite(norm)):
        raise ValueError('nonfinite design scale')
    keep = norm > max(float(norm.max())*1.e-10, 1.e-20)
    if not np.any(keep):
        raise ValueError('empty design')
    a = a[:, keep] / norm[keep]
    y = virtual/vscale
    irls = np.ones(len(x))
    for _ in range(8 if loss == 'huber' else 1):
        ww = w*irls
        coeff = np.linalg.solve(a.T @ (ww[:, None]*a) + ridge*np.eye(a.shape[1]),
                                a.T @ (ww*y))
        if loss == 'huber':
            irls = np.minimum(1., .25/np.maximum(np.abs(y-a@coeff), 1.e-20))
    if not np.all(np.isfinite(coeff)):
        raise ValueError('nonfinite fit')
    terms = [dict(pair=pairs[i], coefficient=float(c/norm[i]))
             for i, c in zip(np.flatnonzero(keep), coeff)]
    return dict(degree=degree, lower=lo.tolist(), upper=hi.tolist(), terms=terms,
                born_scale=bscale, target_scale=vscale)


def predict(entry, x, born, mint, blend=None):
    import numpy as np
    lam = entry.get('blend', 1.) if blend is None else blend
    if not lam:
        return mint.copy()
    z, outside = coordinates(x, entry['lower'], entry['upper'])
    value = design(z, [t['pair'] for t in entry['terms']]) @ np.asarray(
        [t['coefficient'] for t in entry['terms']])
    shared = (born/entry['born_scale'])*value*entry['target_scale']
    answer = mint + lam*(shared-mint)
    # The native runtime also evaluates the exact loop on these far-outside
    # points. The fallback here keeps CV and Python/Fortran prediction equal.
    answer[outside | ~np.isfinite(answer)] = mint[outside | ~np.isfinite(answer)]
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
        b = np.zeros_like(self.average)
        v = np.zeros_like(b)
        n = np.zeros_like(b, dtype=int)
        dims = np.arange(self.nx)
        for r in rows:
            cells = np.minimum((np.asarray(r['x'])*self.bins).astype(int), self.bins-1)
            b[dims, cells] += r['born']
            v[dims, cells] += r['virtual']
            n[dims, cells] += 1
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
        for i in indices:
            result[i] = model.predict(rows[i])
    return result


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
    report.update(cv_folds=3, baseline_cv_l1=base_l1*scale,
                  baseline_cv_rms=math.sqrt(base_second)*scale)
    if not base_l1:
        return fallback('cross_validation_prefers_mint')
    eligible = []
    for degree in (1, 2):
        # Cap both work and flexibility before allocating quadratic matrices.
        active = int(np.count_nonzero(np.ptp(x, axis=0) > 1.e-12))
        nterms = 1+active+(active*(active+1)//2 if degree == 2 else 0)
        smallest = min(np.count_nonzero(assignment != f) for f in range(3))
        if nterms > MAX_TERMS or smallest < ROWS_PER_TERM*nterms:
            report.setdefault('skipped_degrees', []).append(degree)
            continue
        for ridge, loss in itertools.product(RIDGES, LOSSES):
            proposal = np.zeros(len(subset))
            try:
                for fold in range(3):
                    valid, train = assignment == fold, assignment != fold
                    model = fit(x[train], b[train], y[train], w[train], degree, ridge, loss)
                    proposal[valid] = predict(model, x[valid], b[valid], mint[valid])
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
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
                entry = dict(degree=degree, ridge=ridge, loss=loss, blend=blend,
                             parameters=nterms, cv_l1=l1*scale,
                             cv_rms=math.sqrt(second)*scale, gain_error=se*scale,
                             eligible=bool(accept))
                report['candidates'].append(entry)
                if accept:
                    eligible.append(entry)
    if not eligible:
        return fallback('cross_validation_prefers_mint')
    best = min(eligible, key=lambda a: a['cv_l1'])
    # Prefer a smaller basis if its quality is statistically indistinguishable.
    selected = min((a for a in eligible if a['cv_l1'] <= best['cv_l1']+best['gain_error']),
                   key=lambda a: (a['degree'], a['cv_l1']))
    try:
        model = fit(x, b, y, w, selected['degree'], selected['ridge'], selected['loss'])
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return fallback('final_fit_failed')
    model.update(order=order, blend=selected['blend'], reason='selected_by_matched_cross_validation')
    report.update(selected=selected, actual_terms=len(model['terms']))
    return model, report
