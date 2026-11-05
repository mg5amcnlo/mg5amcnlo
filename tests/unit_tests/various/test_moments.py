"""Numerical contract tests for the fixed-order moments analysis backend.

The drivers use the public Fortran API, so these tests cover accumulation,
normalization and output together rather than duplicating backend routines.
No event generator, PDFs or external numerical Python packages are needed.
"""

import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from madgraph.various import moments


ROOT = Path(__file__).resolve().parents[3]


def read_native(path):
    """Read the deliberately simple native format independently of its reader."""
    lines = iter(line.strip() for line in Path(path).read_text().splitlines()
                 if line.strip())
    if next(lines) != 'MOMENTS 1':
        raise AssertionError('Unexpected moments format')
    nweights = int(next(lines).split()[1])
    weights = [next(lines).split(' ', 1)[1] for _ in range(nweights)]
    nhist = int(next(lines).split()[1])
    histograms = {}
    for _ in range(nhist):
        fields = next(lines).split()
        if fields[0] != 'HIST':
            raise AssertionError('Missing HIST record')
        label, kind, nbins, ncoeff = (int(fields[1]), fields[2],
                                      int(fields[3]), int(fields[4]))
        title = next(lines)
        if not title.startswith('TITLE '):
            raise AssertionError('Missing TITLE record')
        if next(lines) != 'MEANS':
            raise AssertionError('Missing MEANS record')
        means = [[float(x.replace('D', 'E')) for x in next(lines).split()]
                 for _ in range(ncoeff)]
        if next(lines) != 'COVARIANCE':
            raise AssertionError('Missing COVARIANCE record')
        covariance = [[float(x.replace('D', 'E')) for x in next(lines).split()]
                      for _ in range(ncoeff)]
        if next(lines) != 'ENDHIST':
            raise AssertionError('Missing ENDHIST record')
        histograms[label] = {
            'kind': kind, 'nbins': nbins, 'ncoeff': ncoeff,
            'xmin': float(fields[5]), 'xmax': float(fields[6]),
            'title': title[6:], 'means': means, 'covariance': covariance}
    if next(lines) != 'END':
        raise AssertionError('Missing END record')
    return weights, histograms


def read_hwu(path):
    """Return raw bin integrals without relying on the display-density reader."""
    histograms = []
    current = None
    for line in Path(path).read_text().splitlines():
        if line.startswith('<histogram>'):
            current = []
            histograms.append(current)
        elif line.startswith('<\\histogram>'):
            current = None
        elif current is not None and line.strip():
            current.append([float(x.replace('D', 'E')) for x in line.split()])
    return histograms


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestMomentsFortran(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix='mg5-moments-build-')
        cls.addClassCleanup(cls.build.cleanup)
        cls.object = Path(cls.build.name) / 'Moments.o'
        source = ROOT / 'Template/NLO/FixedOrderAnalysis/Moments.f'
        result = subprocess.run(
            ['gfortran', '-c', '-O0', '-g', '-fcheck=all',
             '-ffixed-line-length-none', '-ffpe-trap=invalid,zero,overflow',
             str(source), '-o', str(cls.object)],
            cwd=cls.build.name, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Moments.f compilation failed:\n' +
                               result.stdout + result.stderr)

    def run_driver(self, body, nweights=2, xnorm=1., success=True):
        directory = tempfile.TemporaryDirectory(prefix='mg5-moments-test-')
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        source = path / 'driver.f90'
        source.write_text('''program test_moments
  implicit none
  integer :: i
  character(len=80) :: labels(%d)
  double precision :: w(%d), values(2), x, f
  labels(1) = 'central value'
%s
  call HwU_inithist(%d,labels)
%s
  open(unit=99,file='MADatNLO.HwU',status='replace')
  call HwU_output(99,%s)
  close(99)
end program test_moments
''' % (nweights, nweights,
       "  labels(2) = 'scale variation'" if nweights == 2 else '',
       nweights, body, ('%.17e' % xnorm).replace('e', 'd')))
        executable = path / 'driver'
        compilation = subprocess.run(
            ['gfortran', '-O0', '-g', '-fcheck=all',
             '-ffpe-trap=invalid,zero,overflow', str(source),
             str(self.object), '-o', str(executable)], cwd=path,
            capture_output=True, text=True)
        self.assertEqual(compilation.returncode, 0,
                         compilation.stdout + compilation.stderr)
        result = subprocess.run([str(executable)], cwd=path,
                                capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0,
                                'Invalid configuration should fail')
            return path, result
        return path, result

    def native(self, path):
        candidates = list(path.glob('*.Moments')) + list(path.glob('*.moments'))
        self.assertEqual(len(candidates), 1, 'Expected one native moments file')
        return read_native(candidates[0])

    def test_exact_polynomial_projection_and_normalization(self):
        """Three quadrature nodes project a quadratic exactly into three modes."""
        path, _ = self.run_driver('''
  call Moments_book(1,'quadratic',4,-1d0,1d0,3)
  do i=1,3
    if (i.eq.1) then
      x=-sqrt(3d0/5d0)
      f=5d0/9d0
    elseif (i.eq.2) then
      x=0d0
      f=8d0/9d0
    else
      x=sqrt(3d0/5d0)
      f=5d0/9d0
    endif
    w(1)=3d0*f*(2d0+3d0*x+4d0*x*x)
    w(2)=-0.5d0*w(1)
    call HwU_fill(1,x,w)
    call HwU_add_points
  enddo
  values=1d0
  call HwU_accum_iter(.true.,3,values)
''', xnorm=2.)
        weights, native = self.native(path)
        self.assertEqual(weights, ['central value', 'scale variation'])
        self.assertEqual(native[1]['kind'], 'legendre')
        # xnorm is applied to native coefficients as well as HwU output.
        expected = [40. / 3., 4. * math.sqrt(3.),
                    32. * math.sqrt(5.) / 15.]
        for row, coefficient in zip(native[1]['means'], expected):
            self.assertAlmostEqual(row[0], coefficient, places=11)
            self.assertAlmostEqual(row[1], -.5 * coefficient, places=11)
        bins = read_hwu(path / 'MADatNLO.HwU')[0]
        self.assertEqual(len(bins), 4)
        for lo, hi, value, error, variation in bins:
            integral = (2. * (hi - lo) + 1.5 * (hi ** 2 - lo ** 2) +
                        4. / 3. * (hi ** 3 - lo ** 3))
            self.assertAlmostEqual(value, 2. * integral, places=6)
            self.assertAlmostEqual(variation, -integral, places=6)
            self.assertGreaterEqual(error, 0.)
        self.assertAlmostEqual(sum(row[2] for row in bins), 40. / 3., places=6)
        # The independently implemented Python projection must reproduce the
        # actual Fortran output, including covariance propagation and weights.
        data = moments.MomentFile.read(path / 'MADatNLO.Moments')
        data.write_hwu(path / 'python.HwU')
        for actual, expected_row in zip(read_hwu(path / 'python.HwU')[0], bins):
            for a, e in zip(actual, expected_row):
                self.assertAlmostEqual(a, e, places=11)
        data.write(path / 'roundtrip.Moments')
        weights_again, histograms_again = read_native(path / 'roundtrip.Moments')
        self.assertEqual(weights_again, weights)
        self.assertEqual(histograms_again, native)

    def test_grouped_signed_covariance_and_zero_central_variation(self):
        """Counterevents must cancel before forming any outer product."""
        path, _ = self.run_driver('''
  call Moments_book(1,'signed groups',2,-1d0,1d0,2)
  w=(/4d0,6d0/)
  call HwU_fill(1,-0.25d0,w)
  w=-w
  call HwU_fill(1,0.25d0,w)
  call HwU_add_points
  w=(/2d0,5d0/)
  call HwU_fill(1,0d0,w)
  call HwU_add_points
  w=(/0d0,3d0/)
  call HwU_fill(1,0.5d0,w)
  call HwU_add_points
  values=1d0
  call HwU_accum_iter(.true.,4,values)
''')
        _, native = self.native(path)
        result = native[1]
        expected_means = [[.5, 2.], [-.5 * math.sqrt(3.),
                                     -.375 * math.sqrt(3.)]]
        expected_cov = [[.25, math.sqrt(3.) / 12.],
                        [math.sqrt(3.) / 12., .75]]
        for actual, expected in zip(result['means'], expected_means):
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e, places=12)
        for actual, expected in zip(result['covariance'], expected_cov):
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e, places=12)
        bins = read_hwu(path / 'MADatNLO.HwU')[0]
        self.assertAlmostEqual(bins[0][2], .625, places=7)
        self.assertAlmostEqual(bins[1][2], -.125, places=7)
        # These require the off-diagonal coefficient covariance.
        self.assertAlmostEqual(bins[0][3], .375, places=7)
        self.assertAlmostEqual(bins[1][3], math.sqrt(.265625), places=7)

    def test_rejected_iteration_and_mint_weighted_covariance(self):
        path, _ = self.run_driver('''
  call Moments_book(1,'iterations',2,-1d0,1d0,2)
  w=2d0
  call HwU_fill(1,-0.5d0,w)
  call HwU_add_points
  w=4d0
  call HwU_fill(1,0.5d0,w)
  call HwU_add_points
  values=(/2d0,0d0/)
  call HwU_accum_iter(.true.,2,values)
  w=1000000d0
  call HwU_fill(1,1d0,w)
  call HwU_add_points
  call HwU_accum_iter(.false.,1,values)
  w=2d0
  call HwU_fill(1,0d0,w)
  call HwU_add_points
  w=6d0
  call HwU_fill(1,0d0,w)
  call HwU_add_points
  values=(/1d0,3d0/)
  call HwU_accum_iter(.true.,2,values)
''', nweights=1)
        _, native = self.native(path)
        result = native[1]
        self.assertAlmostEqual(result['means'][0][0], 3.75, places=12)
        self.assertAlmostEqual(result['means'][1][0],
                               .125 * math.sqrt(3.), places=12)
        expected = [[2.3125, .09375 * math.sqrt(3.)],
                    [.09375 * math.sqrt(3.), .421875]]
        for actual, row in zip(result['covariance'], expected):
            for a, e in zip(actual, row):
                self.assertAlmostEqual(a, e, places=12)

    def test_histogram_fallback_upper_edge_and_range(self):
        """Categorical/rate bookings remain ordinary bins, including xmax."""
        path, _ = self.run_driver('''
  call HwU_book(4,'categorical rates',2,0d0,2d0)
  w=8d0
  call HwU_fill(4,0d0,w)
  call HwU_add_points
  w=4d0
  call HwU_fill(4,2d0,w)
  call HwU_add_points
  w=1000d0
  call HwU_fill(4,-0.01d0,w)
  call HwU_fill(4,2.01d0,w)
  call HwU_add_points
  values=1d0
  call HwU_accum_iter(.true.,4,values)
''', nweights=1)
        _, native = self.native(path)
        self.assertEqual(set(native), {4})
        self.assertEqual(native[4]['kind'], 'histogram')
        bins = read_hwu(path / 'MADatNLO.HwU')[0]
        self.assertAlmostEqual(bins[0][2], 2.)
        self.assertAlmostEqual(bins[1][2], 1.)
        expected = [[4., -2. / 3.], [-2. / 3., 1.]]
        for actual, row in zip(native[4]['covariance'], expected):
            for a, e in zip(actual, row):
                self.assertAlmostEqual(a, e, places=12)

    def test_initialization_resets_plots_pending_groups_and_iterations(self):
        path, _ = self.run_driver('''
  call Moments_book(7,'old plot',2,-1d0,1d0,2)
  w=1000d0
  call HwU_fill(7,1d0,w)
  call HwU_add_points
  values=1d0
  call HwU_accum_iter(.true.,2,values)
  call HwU_fill(7,0d0,w)
  call HwU_inithist(1,labels)
  call Moments_book(1,'fresh plot',2,-1d0,1d0,1)
  w=2d0
  call HwU_fill(1,0d0,w)
  call HwU_add_points
  w=4d0
  call HwU_fill(1,0d0,w)
  call HwU_add_points
  call HwU_accum_iter(.true.,2,values)
''', nweights=1)
        _, native = self.native(path)
        self.assertEqual(set(native), {1})
        self.assertEqual(native[1]['title'], 'fresh plot')
        self.assertAlmostEqual(native[1]['means'][0][0], 3.)
        self.assertAlmostEqual(native[1]['covariance'][0][0], 1.)

    def test_zero_variance_and_zero_mint_weights_are_finite(self):
        path, _ = self.run_driver('''
  call Moments_book(1,'constant',3,-1d0,1d0,1)
  w=2d0
  do i=1,2
    call HwU_fill(1,0d0,w)
    call HwU_add_points
  enddo
  values=0d0
  call HwU_accum_iter(.true.,2,values)
  w=4d0
  do i=1,2
    call HwU_fill(1,0d0,w)
    call HwU_add_points
  enddo
  call HwU_accum_iter(.true.,2,values)
''', nweights=1)
        _, native = self.native(path)
        self.assertAlmostEqual(native[1]['means'][0][0], 3.)
        self.assertEqual(native[1]['covariance'][0][0], 0.)

    def test_invalid_range_and_incompatible_weighting_fail(self):
        self.run_driver("  call Moments_book(1,'bad range',2,1d0,1d0,2)",
                        nweights=1, success=False)
        self.run_driver('  call set_error_estimation(2)',
                        nweights=1, success=False)


class TestMomentsPython(unittest.TestCase):
    """Native merging and projection need no Fortran compiler."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='mg5-moments-python-')
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)

    def sample(self, fallback=False):
        histograms = [moments.MomentHistogram(
            1, 'legendre', 3, -1., 1., 'smooth observable',
            [[2., 4.], [.5, -.25], [.2, .4]],
            [[.1, .02, 0.], [.02, .3, .04], [0., .04, .5]])]
        if fallback:
            histograms.append(moments.MomentHistogram(
                9, 'histogram', 2, 0., 2., 'categorical rates',
                [[1., 2.], [3., 6.]], [[.4, -.1], [-.1, .8]]))
        return moments.MomentFile(['central value', 'scale variation'], histograms)

    def test_native_roundtrip_and_d_exponents(self):
        data = self.sample(fallback=True)
        path = self.path / 'sample.Moments'
        data.write(path)
        # Fortran native floating point fields are allowed D exponents.
        path.write_text(path.read_text().replace('e+', 'D+').replace('e-', 'D-'))
        reread = moments.MomentFile.read(path)
        self.assertEqual(reread.weights, data.weights)
        for actual, expected in zip(reread.histograms, data.histograms):
            self.assertEqual(actual.signature(), expected.signature())
            self.assertEqual(actual.means, expected.means)
            self.assertEqual(actual.covariance, expected.covariance)

    def test_projection_clipping_rebinning_and_full_covariance(self):
        root_three = math.sqrt(3.)
        h = moments.MomentHistogram(
            1, 'legendre', 2, -1., 1., 'signed groups',
            [[.5, 2.], [-.5 * root_three, -.375 * root_three]],
            [[.25, root_three / 12.], [root_three / 12., .75]])
        values, error = h.integral(-10., 10.)
        self.assertEqual(values, [.5, 2.])
        self.assertAlmostEqual(error, .5)
        lower, lower_error = h.integral(-10., 0.)
        upper, upper_error = h.integral(0., 10.)
        self.assertAlmostEqual(lower[0], .625)
        self.assertAlmostEqual(upper[0], -.125)
        self.assertAlmostEqual(lower_error, .375)
        self.assertAlmostEqual(upper_error, math.sqrt(.265625))
        self.assertAlmostEqual(lower[0] + upper[0], values[0])
        self.assertEqual(h.evaluate(-2.), ([0., 0.], 0.))
        self.assertEqual(h.integral(2., 3.), ([0., 0.], 0.))
        self.assertEqual(h.integral(0., 0.), ([0., 0.], 0.))
        reduced, reduced_error = h.integral(-1., 0., modes=1)
        self.assertEqual(reduced, [.25, 1.])
        self.assertAlmostEqual(reduced_error, .25)
        # Rebinning must conserve the zeroth coefficient at any resolution.
        file = moments.MomentFile(['central value', 'scale variation'], [h])
        file.write_hwu(self.path / 'rebinned.HwU', bins=17)
        bins = read_hwu(self.path / 'rebinned.HwU')[0]
        self.assertEqual(len(bins), 17)
        self.assertAlmostEqual(math.fsum(row[2] for row in bins), .5, places=13)

    def test_merge_independent_covariance_without_split_renormalization(self):
        first, second = self.sample(), self.sample()
        second.histograms[0].means = [[-1., -2.], [1., 2.], [.4, .8]]
        second.histograms[0].covariance = [[.2, -.03, 0.],
                                           [-.03, .5, .04], [0., .04, .7]]
        first_directory = self.path / 'channel_1'
        second_directory = self.path / 'channel_2'
        first_directory.mkdir()
        second_directory.mkdir()
        first.write(first_directory / 'MADatNLO.Moments')
        second.write(second_directory / 'MADatNLO.Moments')
        out = str(self.path / 'combined')
        # Deliberately nontrivial split fractions: MINT already applied them.
        result = moments.combine_job_moments([
            {'dirname': str(first_directory), 'wgt_frac': .125},
            {'dirname': str(second_directory / 'MADatNLO.HwU'),
             'wgt_frac': .875}], out)
        reread = moments.MomentFile.read(out + '.Moments')
        h = reread.histograms[0]
        expected_means = [[1., 2.], [1.5, 1.75], [.6, 1.2]]
        for actual, expected in zip(h.means, expected_means):
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e, places=14)
        expected_covariance = [[.3, -.01, 0.], [-.01, .8, .08],
                               [0., .08, 1.2]]
        for actual, expected in zip(h.covariance, expected_covariance):
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e, places=14)
        values, error = h.integral(-1., 1.)
        self.assertEqual(values, [1., 2.])
        self.assertAlmostEqual(error, math.sqrt(.3))
        self.assertEqual(result.histograms[0].means, h.means)

    def test_reject_incompatible_weight_or_observable_definitions(self):
        base = self.sample()
        wrong_weights = self.sample()
        wrong_weights.weights.reverse()
        with self.assertRaises(moments.MomentsError):
            base.add(wrong_weights)
        for attribute, value in [('label', 2), ('kind', 'histogram'),
                                 ('nbins', 4), ('xmin', -2.),
                                 ('xmax', 2.), ('title', 'different')]:
            other = self.sample()
            setattr(other.histograms[0], attribute, value)
            with self.subTest(attribute=attribute):
                with self.assertRaises(moments.MomentsError):
                    base.add(other)
        short = self.sample()
        short.histograms[0] = moments.MomentHistogram(
            1, 'legendre', 3, -1., 1., 'smooth observable',
            [[1., 2.]], [[.1]])
        with self.assertRaises(moments.MomentsError):
            base.add(short)
        with self.assertRaises(moments.MomentsError):
            moments.combine_files([])

    def test_reject_malformed_nonfinite_and_trailing_native_data(self):
        original = self.path / 'original.Moments'
        self.sample().write(original)
        lines = original.read_text().splitlines()
        invalid = []
        row = lines[:]
        row[8] = '1.0'  # Missing systematic-weight column.
        invalid.append(row)
        row = lines[:]
        row[8] = 'nan 1.0'
        invalid.append(row)
        row = lines[:]
        row[8] = 'oops 1.0'
        invalid.append(row)
        row = lines[:]
        row[12] = '-1.0 0.02 0.0'  # Negative variance.
        invalid.append(row)
        row = lines[:]
        row[12] = '0.1 0.99 0.0'  # Asymmetry.
        invalid.append(row)
        invalid += [lines[:-1], lines + ['unexpected data'],
                    ['MOMENTS 2'] + lines[1:]]
        for i, row in enumerate(invalid):
            with self.subTest(case=i):
                path = self.path / ('invalid_%d.Moments' % i)
                path.write_text('\n'.join(row) + '\n')
                with self.assertRaises(moments.MomentsError):
                    moments.MomentFile.read(path)

    def test_cli_truncates_only_projection_and_keeps_histogram_bookings(self):
        first, second = self.sample(fallback=True), self.sample(fallback=True)
        first_path, second_path = self.path / 'a.Moments', self.path / 'b.Moments'
        first.write(first_path)
        second.write(second_path)
        out = str(self.path / 'cli')
        result = subprocess.run(
            [sys.executable, str(ROOT / 'madgraph/various/moments.py'),
             str(first_path), str(second_path), '--out', out,
             '--modes', '1', '--bins', '7'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        native = moments.MomentFile.read(out + '.Moments')
        self.assertEqual(native.histograms[0].ncoeff, 3)
        self.assertEqual(native.histograms[0].means,
                         [[4., 8.], [1., -.5], [.4, .8]])
        bins = read_hwu(out + '.HwU')
        self.assertEqual([len(h) for h in bins], [7, 2])
        for row in bins[0]:
            self.assertAlmostEqual(row[2], 4. / 7., places=13)
            self.assertAlmostEqual(row[3], math.sqrt(.2) / 7., places=13)
            self.assertAlmostEqual(row[4], 8. / 7., places=13)
        self.assertEqual([row[2] for row in bins[1]], [2., 6.])
        with self.assertRaises(moments.MomentsError):
            native.write_hwu(self.path / 'invalid.HwU', modes=4)


if __name__ == '__main__':
    unittest.main()
