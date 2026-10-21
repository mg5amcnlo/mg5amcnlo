"""Small, frozen Born-times-logarithm models for barrier-pooled virtuals.

NumPy is imported only when fitting/inference is requested. This module has
no dependency on validation scripts or on a particular physics process.
"""
import itertools
import math
from collections import namedtuple
from functools import lru_cache

MAX_TERMS = 512
COMPACT_RIDGE = 1.e-4
ROWS_PER_TERM = 5
TAIL_MARGIN = .10
BLENDS = (.25, .5, .75, 1.)

_Regression = namedtuple('_Regression', (
    'matrix', 'weight', 'column_scale', 'active', 'target',
    'born_scale', 'target_scale', 'gram', 'rhs'))


def empty_order(order, nf, reason):
    return dict(order=order, blend=0., born_scale=1., target_scale=1., degree=0,
                lower=[0.] * nf, upper=[0.] * nf, terms=[], reason=reason)


def coordinates(x, lower, upper, clip=True):
    import numpy as np
    x, lo, hi = (np.asarray(a, dtype=float) for a in (x, lower, upper))
    width = hi - lo
    active = width > 1.e-12
    z = np.divide(2. * (x-lo), width, out=np.ones_like(x), where=active)-1.
    allowance = np.where(active, TAIL_MARGIN*width, 1.e-10*np.maximum(1., np.abs(lo)))
    outside = np.any((x < lo-allowance) | (x > hi+allowance), axis=1)
    return (np.clip(z, -1., 1.) if clip else z), outside


def pairs_for(x, degree, nscalar=None, basis='full', linear_only=()):
    import numpy as np
    if degree not in (1, 2) or basis not in ('full', 'compact'):
        raise ValueError('invalid residual polynomial basis')
    active = np.flatnonzero(np.ptp(x, axis=0) > 1.e-12).tolist()
    pairs = [(-1, -1)] + [(i, -1) for i in active]
    if degree == 2:
        pairs += ([(i, i) for i in active if nscalar is None or i < nscalar]
                  if basis == 'compact' else
                  list(itertools.combinations_with_replacement(active, 2)))
    if nscalar is not None:
        pairs = [p for p in pairs if sum(i >= nscalar for i in p) <= 1]
    if linear_only:
        pairs = [p for p in pairs if sum(i >= 0 for i in p) <= 1 or
                 not any(i in linear_only for i in p)]
    return pairs


def basis_size(x, degree, nscalar=None, basis='full', linear_only=()):
    """Count compact coefficients without allocating the regression matrix."""
    import numpy as np
    active = np.flatnonzero(np.ptp(x, axis=0) > 1.e-12)
    even = [i for i in active if nscalar is None or i < nscalar]
    if basis == 'compact':
        return len(even), 1+len(active)+(sum(i not in linear_only for i in even) if degree == 2 else 0)
    return len(even), len(pairs_for(x, degree, nscalar, basis, linear_only))


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


def fit(x, born, virtual, weight, degree, ridge, basis='full', nscalar=None,
        linear_only=()):
    """Fit the linear/compact quadratic correction to the surrogate residual."""
    _, count = basis_size(x, degree, nscalar, basis, linear_only)
    if count > MAX_TERMS:
        raise ValueError('polynomial exceeds coefficient budget')
    pairs = pairs_for(x, degree, nscalar, basis, linear_only)
    lo, hi = x.min(axis=0), x.max(axis=0)
    if nscalar is not None:
        lo[nscalar:], hi[nscalar:] = -1., 1.
    z, _ = coordinates(x, lo, hi)
    matrix = design(z, pairs)
    coeff, norm, bscale, vscale = _solve_design(matrix, born, virtual, weight, ridge)
    terms = [dict(pair=indices, coefficient=float(value)) for indices, value, size in
             zip(pairs, coeff, norm) if size > 0]
    return dict(degree=degree, lower=lo.tolist(), upper=hi.tolist(), terms=terms,
                born_scale=bscale, target_scale=vscale, unclipped=True)


def _prepare_regression(matrix, born, virtual, weight):
    """Normalize a Born-carried regression and prepare its first linear system."""
    import numpy as np
    # A Born-dependent noise scale prevents a few very large integration
    # weights from determining a polynomial for the whole subprocess. Keep
    # V-B*P as the regression residual, including signed and zero Born rows;
    # never construct the potentially singular target V/B.
    w = weight / weight.sum()
    bscale = float(w @ np.abs(born))
    vscale = float(w @ np.abs(virtual))
    if not math.isfinite(bscale) or not math.isfinite(vscale) or not bscale or not vscale:
        raise ValueError('zero fit scale')
    a = (born/bscale)[:, None] * matrix
    noise = np.maximum(np.abs(born)/bscale, 1.e-12)
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
    return _Regression(a, fit_weight, norm, keep, y, bscale, vscale, gram, rhs)


def _solve_design(matrix, born, virtual, weight, ridge):
    import numpy as np
    prepared = _prepare_regression(matrix, born, virtual, weight)
    coeff = _ridge_solve(prepared.gram, prepared.rhs, ridge)
    if not np.all(np.isfinite(coeff)):
        raise ValueError('nonfinite fit')
    result = np.zeros(matrix.shape[1])
    result[prepared.active] = coeff/prepared.column_scale[prepared.active]
    return result, prepared.column_scale, prepared.born_scale, prepared.target_scale


def predict(entry, x, born, mint, blend=None, surrogate=None, channels=None, supported=None):
    import numpy as np
    lam = entry.get('blend', 1.) if blend is None else blend
    if not lam:
        return mint.copy()
    z, _ = coordinates(x, entry['lower'], entry['upper'], clip=not entry.get('unclipped', False))
    value = design(z, [t['pair'] for t in entry['terms']]) @ np.asarray(
        [t['coefficient'] for t in entry['terms']])
    shared = (born/entry['born_scale'])*value*entry['target_scale']
    if 'surrogate' in entry:
        from .surrogate import basis_prediction
        if surrogate is None or channels is None:
            raise ValueError('pooled refinement requires the native surrogate inputs and channel')
        base, channel_blend = basis_prediction(entry['surrogate'], surrogate, born, channels)
        shared += base
        lam = lam*channel_blend
        if supported is not None:
            lam = lam*np.asarray(supported, dtype=bool)
    answer = mint + lam*(shared-mint)
    # Physical logarithmic/Coulomb continuation retains an exact residual
    # outside the supported box, where native inference requests exact loops.
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

    def _cells(self, coordinates):
        import numpy as np
        return np.minimum((np.asarray(coordinates)*self.bins).astype(int), self.bins-1)

    def update(self, rows):
        import numpy as np
        if not rows:
            return
        b = np.zeros_like(self.average)
        v = np.zeros_like(b)
        n = np.zeros_like(b, dtype=int)
        cells = self._cells([r['x'] for r in rows])
        born = np.asarray([r.get('mint_born', r['born']) for r in rows])
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
        cells = self._cells(row['x'])
        return float(self.average[np.arange(self.nx), cells].mean())*row.get('mint_born', row['born'])

    def predict_many(self, rows):
        import numpy as np
        cells = self._cells([r['x'] for r in rows])
        ratios = self.average[np.arange(self.nx), cells].mean(axis=1)
        return ratios*np.asarray([r.get('mint_born', r['born']) for r in rows])


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
