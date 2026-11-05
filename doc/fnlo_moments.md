# Fixed-order moments analysis (experimental)

This implements a smooth alternative to the HwU histogram library for fixed-order
LO/NLO runs. It follows the projection idea in
[Melnikov, Novikov and Pedron, arXiv:2601.10420](https://arxiv.org/abs/2601.10420).
The first implementation provides finite-interval Legendre moments and retains
the complete central-weight covariance. Other bases and automatic regularization
are research extensions, not implemented production features.

## Selecting the backend

Generate a process with the modified MG5, then set `Cards/FO_analyse_card.dat`:

```text
FO_ANALYSIS_FORMAT = moments
FO_ANALYSE = analysis_moments_template.o
```

Keep the existing library/path entries appropriate for your analysis. The example
in `FixedOrderAnalysis/analysis_moments_template.f` fills ordinary total-rate bins
and a soft/collinear-safe inclusive angular energy-flow distribution. It is
particularly suitable for testing with `e+ e- > u u~ [QCD]` and lepton beams.

For an existing HwU analysis, retain initialization, fills and finalization:

```fortran
      call HwU_inithist(nwgt,weights_info)
      call Moments_book(3,'my observable |T@NLO',40,0d0,200d0,12)
      ! Within analysis_fill, including every event/counterevent:
      call HwU_fill(3,x,wgts)
      ! Within analysis_end:
      call HwU_write_file
```

Only change the bookings of observables to be reconstructed smoothly. The last
argument to `Moments_book` is the number of retained modes, including the constant
mode. The third argument remains the number of presentation bins. Choose the
same mode count, range and observable definition in every integration job.

An unchanged `HwU_book` keeps ordinary histogram bins. Use it for rates, discrete
quantities, and observables requiring literal bin contents. Smooth reconstruction
of a discrete quantity or a Born delta peak is generally inappropriate.

The backend is compiled **instead of** `HwU.f`; do not link both. The existing
`HwU` format and its default backend remain selectable. No changes to matrix
elements, subtraction kernels, phase-space generation or event weights are needed.

## Outputs and postprocessing

Each channel writes:

* `MADatNLO.HwU`: exact bin integrals of the retained representation, usable by the
  existing HwU plotting pipeline, including all scale/PDF weight means.
* `MADatNLO.Moments`: all computed coefficients and the full central-weight
  covariance, with ordinary histogram records included too.

The usual run finalization sums independent native job estimates into
`Events/run_XX/MADatNLO.Moments`, alongside the existing plot outputs. The fixed-order
split-job normalization is already present in the filled weights; it is not
applied a second time. Incompatible labels, weight definitions, bases or ranges
are rejected when native data are combined.

Rebin or truncate a **combined** file without rerunning integration:

```bash
python bin/internal/moments.py Events/run_01/MADatNLO.Moments \
    --out Events/run_01/reconstructed --bins 80 --modes 8
```

From the MG5 source tree, use `python -m madgraph.various.moments` instead. Multiple
input files are summed as independent estimates. Do not pass the combined file
together with its constituent jobs. `--modes` applies to smooth observables only;
it must not exceed the available count of any smooth booking. Ordinary histogram
bookings retain their binning. The output `.Moments` retains **all** input modes;
`--modes` and `--bins` affect only the `.HwU` projection. This allows alternative
regularization decisions without losing the original calculation.

Programmatic access needs only the Python standard library:

```python
from madgraph.various.moments import MomentFile
data = MomentFile.read('Events/run_01/MADatNLO.Moments')
h = next(h for h in data.histograms if h.label == 3)
weight_means, central_stat_error = h.evaluate(50.0)
bin_means, bin_stat_error = h.integral(40.0, 60.0)
```

An integral of a truncated reconstruction estimates the projected distribution's
integral. Its reconstruction bias remains even though its Monte Carlo error can
be small. In particular, very fine presentation bins do not add physical
resolution to a low-order expansion.

## Numerical convention and statistical scope

For `L = xmax-xmin`, `t = 2*(x-xmin)/L-1`, the test functions and reconstruction are

```text
psi_k(x) = sqrt(2*k+1) P_k(t)
a_k      = integral f(x) psi_k(x) dx
f_m(x)   = sum(k=0..m-1) a_k psi_k(x) / L.
```

Consequently `a_0` is the cross section inside the booking range. Bin integrals
are evaluated analytically using Legendre antiderivatives. Endpoints are included;
points outside the range are discarded, not folded into endpoint bins. For
ordinary histogram fallback this includes the upper endpoint, unlike the old
HwU integer-bin edge convention.

All correlated fills between calls to `HwU_add_points` are summed first. Stable
Welford accumulation then estimates their covariance. Empty/cut/out-of-range
groups enter as implicit zeros using the MINT sample count. For independent groups
`z_i`, the within-iteration covariance of the mean is

```text
C = sum_i (z_i - mean(z)) (z_i - mean(z))^T / [N*(N-1)].
```

Iteration estimates use the same scalar weights as MINT/HwU error mode 3 (inverse
standard-error weights, not inverse variances). Covariance combines with the
squares of the normalized scalar weights. Accepted iterations with no entries in
an observable contribute a zero estimate; they are not silently skipped. Error
mode 1 gives equal iteration weights and propagates within-iteration covariance;
it does not reproduce HwU's between-iteration error convention. Modes 0 and 2,
event-generation normalization and PineAPPL are explicitly unsupported.

There are two small plotting lifecycle changes in MINT:

1. Flush every fixed-order subtraction group, even if its summed central weight
   vanishes. Its differential moments or variations can still be nonzero.
2. Discard the first special adaptive warmup plotting iteration, which samples
   channels separately using different sample counts. The pooled plotting API
   cannot normalize it using the final channel's count. Production uses the
   ordinary mixture-sampling path and rewrites warmup outputs.

**The reported errors are statistical working estimates, not calibrated total
error bands.** They exclude omitted modes, reference/regularization uncertainty,
and perturbative scale uncertainty. MINT also uses stratified and adaptive
sampling and data-dependent iteration weights/stopping, so the independent-group
formula does not establish exact frequentist coverage. Independent full-run
replicates are needed to assess calibration. Within-observable central covariance
is retained; cross-observable and cross-weight covariance are not yet stored.

Storage per observable is `O(nweights*m + m^2)`; filling/group accumulation has the
same scaling. Ordinary histogram fallback also stores its central bin covariance.
Large collections of very finely binned fallback histograms can therefore use
more memory and CPU than the original HwU library. Filling is serial, as for HwU.

## Basis investigation: Legendre, Chebyshev and reference adaptation

The family name alone does not determine approximation quality. Legendre and
Chebyshev polynomials up to the same degree span the **same polynomial space**.
With the same measure and correct Gram-matrix inversion, their projected
polynomials and propagated covariances coincide (up to rounding). A numerical
advantage must come from a different measure, envelope, coordinate, regularizer,
conditioning or computational cost.

Legendre is convenient for a uniform measure on a finite interval: tests are
bounded, normalization is simple, and the Gram matrix is identity. Chebyshev
first-kind orthogonality uses the arcsine weight `1/sqrt(1-t^2)`. Directly inserting
that weight into event tests can give unbounded endpoint weights and even infinite
variance for otherwise regular sampled integrands. Alternatively, a reference
envelope equal to the arcsine density gives bounded tests but changes the trial
function space and imposes endpoint behavior. Chebyshev interpolation's excellent
uniform-approximation properties alone do not establish a noisy Monte Carlo MSE
advantage. The [NIST orthogonality table](https://dlmf.nist.gov/18.3) summarizes the
measures for these families.

Candidate extensions, in priority order:

1. **Reference-weighted polynomials in a physical coordinate.** Choose a positive
   normalized `q`, construct `integral q*p_k*p_l = delta_kl`, and reconstruct
   `f=q*sum a_k*p_k`, where `a_k=integral f*p_k`. This keeps the reference envelope
   without forcing the correction into its CDF coordinate. No eventwise `1/q`
   division is required. This has
   [established approximation-theory precedent](https://pierre-olivier.goffard.me/Publications/Goffard_Laub_SLP.pdf).
   Polynomial growth in tails can increase sampling variance and must be tested.
2. **Jacobi/Laguerre/Hermite envelopes.** Jacobi weights can incorporate known
   finite-interval endpoint powers; Laguerre naturally accompanies exponential or
   gamma references on a positive half-line; Hermite accompanies Gaussian
   references. Endpoint logarithms, delta functions and plus distributions need
   explicit physical treatment rather than arbitrary smoothing.
3. **Positive reference mixtures.** Add a broader component to the LO reference to
   cover Born zeros and regions first populated at higher orders. Tempered CDF
   maps are another way to trade peak efficiency against tail resolution.
4. **Global covariance-aware regularization.** Select retained modes after summing
   channels. Compare smooth shrinkage against hard truncation, preserve measured
   normalization, and assess reconstruction bias with independent validation.
5. **Smooth local bases and stable paired evaluation.** Overlapping splines can
   resolve shoulders without high global degree. Stable divided differences can
   help evaluate event/counterevent test-function differences. Physical acceptance
   cuts still introduce discontinuities.

A useful exact example is `q=exp(-x/L)/L`, `f=q*(1+beta*x/L)`. Physical-coordinate
Laguerre polynomials represent it with two terms. CDF rectification gives
`t=1-2*exp(-x/L)`, so `f/q=1-beta*log((1-t)/2)` and normalized Legendre coefficients
fall only as `beta*sqrt(2*k+1)/(k*(k+1))`. On a finite interval the endpoint
singularity moves outside the domain but can leave a narrow boundary layer. This
demonstrates an approximation issue; it does not prove smaller Monte Carlo MSE.

`FixedOrderAnalysis/study_moments_bases.py` is a separate, reproducible numerical
study of these distinctions (NumPy required). This is a research dependency
only: the backend and native-file postprocessor do not require it.

```bash
python Template/NLO/FixedOrderAnalysis/study_moments_bases.py --out study.json
```

With the default seed, same-measure 8-mode Legendre and Chebyshev reconstructions
agree within `5.6e-13`, their covariance kernels within `6.7e-14`, and their measured
integrated MSEs coincide. In the finite exponential-reference toy with 2000
independent reference samples, two physical-coordinate modes have expected
integrated squared error about `0.00138`; 2/4/8/16 CDF-Legendre modes give about
`0.0108/0.00505/0.00668/0.0132`. This illustrates the bias/variance tradeoff for
this target and proposal, not performance in a collider integration.

## Verification and next physics checks

```bash
python -m unittest tests.unit_tests.various.test_FO_analyse_card \
    tests.unit_tests.various.test_moments -v
```

Tests compile the actual Fortran backend and check polynomial projection,
normalization, signed groups, off-diagonal covariance, all weight means, rejected
iterations, scalar MINT iteration weights, implicit zeros, reinitialization,
histogram fallback, native merging and postprocessing.

An actual generated `e+ e- > u u~ [QCD]` fNLO smoke test has also completed with
the shared energy-flow example, 125 GeV lepton beams, seed 12345, `req_acc_FO=0.1`,
and scale reweighting (10 weights). It exercised the special adaptive warmup loop,
two production iterations of 6144 samples, native-file merging, and the standard
HwU/Gnuplot/Matplotlib/HTML/PDF output chain. The NLO rate and energy-flow zeroth
moment both gave `3.09672437649 +/- 0.01664579392 pb`; all 10 full-range weight
integrals agreed to `1.8e-15 pb`. This checks integration and normalization, not
precision phenomenology or uncertainty coverage. A separate fixed-point-count run
also passed. Its generated process and logs are temporary validation artifacts,
not source dependencies.

To reproduce the slower integration check explicitly (requires a working Fortran
and C++ toolchain and the bundled reduction libraries):

```bash
python tests/acceptance_tests/run_moments_smoke.py /tmp/new_moments_process
```

Use `--validate-only` with an existing process directory to check the combined
rate and energy-flow normalization without rerunning generation or integration.

The next substantive benchmark is repeated independent fNLO runs with a smooth
observable, a steep tail and a shoulder, comparing the same CPU cost and physical
resolution. Measure bias in bin integrals/tail cross sections, empirical covariance
and coverage, rather than judging only the appearance of curves. This prototype
does not yet establish a production accuracy or speed advantage over HwU.
