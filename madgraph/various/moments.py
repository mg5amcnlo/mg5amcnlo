"""Native fixed-order moment estimates and their central-weight covariance.

No numerical Python dependency is needed.  Coefficients are integrals against
sqrt(2*k+1)*P_k(t); the zeroth coefficient is the in-range cross section.
The covariance is statistical: it does not include reconstruction bias.
"""

import argparse
import math
import os


class MomentsError(ValueError):
    """Invalid or incompatible native moment data."""


def _numbers(line, count):
    try:
        result = [float(item.replace('D', 'E').replace('d', 'e'))
                  for item in line.split()]
    except ValueError as exc:
        raise MomentsError('Invalid numerical row: %s' % line) from exc
    if len(result) != count or not all(math.isfinite(v) for v in result):
        raise MomentsError('Expected %d finite numbers: %s' % (count, line))
    return result


def _legendre(t, count):
    result = [1.0]
    if count > 1:
        result.append(t)
    for k in range(1, count - 1):
        result.append(((2*k + 1)*t*result[k] - k*result[k-1])/(k+1))
    return result


class MomentHistogram:
    """One observable, all weight means, and central statistical covariance."""

    def __init__(self, label, kind, nbins, xmin, xmax, title, means, covariance):
        self.label, self.kind, self.nbins = int(label), kind, int(nbins)
        self.xmin, self.xmax, self.title = float(xmin), float(xmax), title
        self.means, self.covariance = means, covariance
        n = len(means)
        nw = len(means[0]) if n else 0
        if (self.label < 1 or kind not in ('histogram', 'legendre') or
                self.nbins < 1 or not n or not nw or
                not math.isfinite(self.xmin) or not math.isfinite(self.xmax) or
                self.xmin >= self.xmax or '\n' in title):
            raise MomentsError('Invalid observable metadata')
        if kind == 'histogram' and n != self.nbins:
            raise MomentsError('Histogram coefficients must match the bin count')
        if (any(len(row) != nw for row in means) or len(covariance) != n or
                any(len(row) != n for row in covariance)):
            raise MomentsError('Inconsistent coefficient/covariance dimensions')
        if not all(math.isfinite(v) for row in means + covariance for v in row):
            raise MomentsError('Non-finite estimate')
        scale = max([abs(v) for row in covariance for v in row] + [1.e-300])
        for i in range(n):
            if covariance[i][i] < -1.e-10*scale:
                raise MomentsError('Negative covariance diagonal')
            for j in range(i):
                if abs(covariance[i][j]-covariance[j][i]) > 1.e-10*scale:
                    raise MomentsError('Asymmetric covariance')

    @property
    def ncoeff(self):
        return len(self.means)

    def _count(self, modes):
        if self.kind == 'histogram' or modes is None:
            return self.ncoeff
        if not 1 <= modes <= self.ncoeff:
            raise MomentsError('Requested modes must be between 1 and %d' % self.ncoeff)
        return modes

    def density_vector(self, x, modes=None):
        """Linear map from native coefficients to the projected density."""
        n = self._count(modes)
        if not math.isfinite(x):
            raise MomentsError('Non-finite coordinate')
        result = [0.0]*self.ncoeff
        if x < self.xmin or x > self.xmax:
            return result
        length = self.xmax-self.xmin
        if self.kind == 'histogram':
            i = min(self.nbins-1, int((x-self.xmin)/length*self.nbins))
            result[i] = self.nbins/length
        else:
            t = max(-1.0, min(1.0, 2*(x-self.xmin)/length-1))
            for k, value in enumerate(_legendre(t, n)):
                result[k] = math.sqrt(2*k+1)*value/length
        return result

    def integral_vector(self, low, high, modes=None):
        """Exact integral of the retained representation, clipped to its range."""
        n = self._count(modes)
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            raise MomentsError('Invalid integration interval')
        low, high = max(low, self.xmin), min(high, self.xmax)
        result = [0.0]*self.ncoeff
        if high <= low:
            return result
        length = self.xmax-self.xmin
        if self.kind == 'histogram':
            width = length/self.nbins
            for k in range(n):
                overlap = min(high, self.xmin+(k+1)*width)-max(low, self.xmin+k*width)
                result[k] = max(0.0, overlap)/width
        else:
            ta, tb = 2*(low-self.xmin)/length-1, 2*(high-self.xmin)/length-1
            pa, pb = _legendre(ta, n+1), _legendre(tb, n+1)
            result[0] = (high-low)/length
            for k in range(1, n):
                result[k] = ((pb[k+1]-pb[k-1])-(pa[k+1]-pa[k-1]))/(2*math.sqrt(2*k+1))
        return result

    def project(self, vector):
        """Return all projected weight means and the central standard error."""
        values = [sum(vector[k]*self.means[k][j] for k in range(self.ncoeff))
                  for j in range(len(self.means[0]))]
        terms = [vector[i]*self.covariance[i][j]*vector[j]
                 for i in range(self.ncoeff) for j in range(self.ncoeff)]
        variance = math.fsum(terms)
        if variance < -1.e-10*max(math.fsum(abs(v) for v in terms), 1.e-300):
            raise MomentsError('Covariance gives a negative projected variance')
        return values, math.sqrt(max(0.0, variance))

    def evaluate(self, x, modes=None):
        return self.project(self.density_vector(x, modes))

    def integral(self, low, high, modes=None):
        return self.project(self.integral_vector(low, high, modes))

    def signature(self):
        return (self.label, self.kind, self.nbins, self.ncoeff,
                self.xmin, self.xmax, self.title)


class MomentFile:
    def __init__(self, weights, histograms):
        self.weights, self.histograms = list(weights), list(histograms)
        if not self.weights or any('\n' in w for w in self.weights):
            raise MomentsError('Invalid weight labels')
        if len({h.label for h in histograms}) != len(histograms):
            raise MomentsError('Duplicate histogram labels')
        if any(len(h.means[0]) != len(weights) for h in histograms):
            raise MomentsError('Weight dimensions disagree')

    @classmethod
    def read(cls, path):
        try:
            return cls._read(path)
        except MomentsError:
            raise
        except (ValueError, OverflowError) as exc:
            raise MomentsError('Invalid metadata in %s: %s' % (path, exc)) from exc

    @classmethod
    def _read(cls, path):
        with open(path) as stream:
            def line():
                value = stream.readline()
                if not value:
                    raise MomentsError('Unexpected end of %s' % path)
                return value.rstrip('\r\n')

            def expect(text):
                value = line()
                if value != text:
                    raise MomentsError('Expected %s, got %s in %s' % (text, value, path))

            def count(tag):
                value = line().split()
                if len(value) != 2 or value[0] != tag:
                    raise MomentsError('Expected %s count in %s' % (tag, path))
                return int(value[1])

            expect('MOMENTS 1')
            nw = count('NWEIGHTS')
            if nw < 1:
                raise MomentsError('NWEIGHTS must be positive')
            weights = []
            for j in range(nw):
                value = line()
                if not value.startswith('WEIGHT '):
                    raise MomentsError('Expected WEIGHT label')
                weights.append(value[7:])
            nh = count('NHIST')
            if nh < 0:
                raise MomentsError('NHIST must be nonnegative')
            histograms = []
            for ih in range(nh):
                header = line().split()
                if len(header) != 7 or header[0] != 'HIST':
                    raise MomentsError('Invalid HIST header')
                _, label, kind, nbins, nc, xmin, xmax = header
                nc = int(nc)
                if nc < 1:
                    raise MomentsError('Coefficient count must be positive')
                title = line()
                if not title.startswith('TITLE '):
                    raise MomentsError('Expected TITLE')
                expect('MEANS')
                means = [_numbers(line(), nw) for k in range(nc)]
                expect('COVARIANCE')
                covariance = [_numbers(line(), nc) for k in range(nc)]
                expect('ENDHIST')
                histograms.append(MomentHistogram(label, kind, nbins,
                    float(xmin.replace('D', 'E').replace('d', 'e')),
                    float(xmax.replace('D', 'E').replace('d', 'e')),
                    title[6:], means, covariance))
            expect('END')
            if stream.read().strip():
                raise MomentsError('Unexpected data after END')
        return cls(weights, histograms)

    def write(self, path):
        with open(path, 'w') as stream:
            stream.write('MOMENTS 1\nNWEIGHTS %d\n' % len(self.weights))
            for weight in self.weights:
                stream.write('WEIGHT %s\n' % weight)
            stream.write('NHIST %d\n' % len(self.histograms))
            for h in self.histograms:
                stream.write('HIST %d %s %d %d %.17e %.17e\nTITLE %s\nMEANS\n' %
                    (h.label, h.kind, h.nbins, h.ncoeff, h.xmin, h.xmax, h.title))
                for row in h.means:
                    stream.write(' '.join('%.17e' % v for v in row)+'\n')
                stream.write('COVARIANCE\n')
                for row in h.covariance:
                    stream.write(' '.join('%.17e' % v for v in row)+'\n')
                stream.write('ENDHIST\n')
            stream.write('END\n')

    def add(self, other):
        """Sum independent job estimates (already normalized by MINT)."""
        if self.weights != other.weights:
            raise MomentsError('Cannot combine different weight labels')
        if [h.signature() for h in self.histograms] != [h.signature() for h in other.histograms]:
            raise MomentsError('Cannot combine different observable definitions')
        for h, rhs in zip(self.histograms, other.histograms):
            for k in range(h.ncoeff):
                for j in range(len(self.weights)):
                    h.means[k][j] += rhs.means[k][j]
                for j in range(h.ncoeff):
                    h.covariance[k][j] += rhs.covariance[k][j]
        return self

    def write_hwu(self, path, modes=None, bins=None):
        """Project to HwU bin integrals; dy contains statistical error only.

        Optional truncation is applied after merging. Native data stay intact.
        Histogram fallback bookings retain their original binning.
        """
        if bins is not None and bins < 1:
            raise MomentsError('Bin count must be positive')
        with open(path, 'w') as stream:
            columns = ['xmin', 'xmax', self.weights[0], 'dy']+self.weights[1:]
            stream.write('##& '+' & '.join(columns)+'\n\n')
            for h in self.histograms:
                nb = bins if bins is not None and h.kind == 'legendre' else h.nbins
                stream.write('<histogram> %04d " %s "\n' % (nb, h.title))
                width = (h.xmax-h.xmin)/nb
                for i in range(nb):
                    low, high = h.xmin+i*width, h.xmin+(i+1)*width
                    values, error = h.integral(low, high, modes)
                    row = [low, high, values[0], error]+values[1:]
                    stream.write('  '+'  '.join('%.16e' % v for v in row)+'\n')
                stream.write('<\\histogram>\n\n\n')


def combine_files(paths):
    result = None
    for path in paths:
        data = MomentFile.read(path)
        result = data if result is None else result.add(data)
    if result is None:
        raise MomentsError('No native moment files to combine')
    return result


def combine_job_moments(jobs, out):
    """Fixed-order hook: job weights already contain split normalization."""
    paths = []
    for job in jobs:
        name = job['dirname']
        if name.endswith('.HwU'):
            name = name[:-4]+'.Moments'
        elif not name.endswith('.Moments'):
            name = os.path.join(name, 'MADatNLO.Moments')
        paths.append(name)
    result = combine_files(paths)
    result.write(out+'.Moments')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', help='Independent native .Moments files')
    parser.add_argument('--out', required=True, help='Output prefix')
    parser.add_argument('--modes', type=int, help='Retained Legendre modes, selected after merging')
    parser.add_argument('--bins', type=int, help='Output bins for smooth observables')
    args = parser.parse_args(argv)
    result = combine_files(args.inputs)
    result.write(args.out+'.Moments')
    result.write_hwu(args.out+'.HwU', modes=args.modes, bins=args.bins)
    print('Wrote %s.Moments and %s.HwU; dy is statistical only.' % (args.out, args.out))


if __name__ == '__main__':
    main()
