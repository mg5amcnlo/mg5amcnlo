# MC@NLO-Delta with complete native histories

Validation on 23 September 2026 started from `628f316e2`, using fresh NLO
exports and Pythia 8.313. The fixes described below were then applied to
the export templates and those generated outputs. Physical Delta matching
needed an event-colour ownership fix and two shower-interface fixes.

## History sum and event ownership

For a fixed real-flavour configuration and real phase-space point, the
implementation retains the complete sum over labelled native histories
`b = (k,l)`, including histories whose Born provider belongs to another
subprocess directory. For an outer sector `a = (i,j)`, it evaluates

\[
 H_b = P_b\left(S_b R-M_b\right),\qquad
 \widehat H_a = S_a\sum_b H_b,\qquad
 \sum_a S_a=1.
\]

Here `M_b` includes the MC counterterm and its G replacements. The native
PDFs, cuts, coupling orders, colour-flow sampling and counterevent measure
ratios are implicit. Physical MC@NLO-Delta uses `P_b = Delta_b`, evaluated
from that native history; it does not reuse the outer history's Delta in
the inner sum. Native S contributions retain the complementary
`P_b M_b + (1-P_b) S_b R`, in addition to the other NLO terms.

If `K_b H_b` is the complete weight returned using the auxiliary native
integration map, the driver multiplies it by

\[
 \frac{S_a K_a}{K_b}.
\]

`K_a` contains the outer sampling and integration factors. This removes
the auxiliary native real measure while preserving the physical
counterevent-to-real measure ratios inside `H_b`. The sampled colour
flows reproduce the colour-summed formula in expectation.

Native provenance is used for the weights; the outer sector owns the
written event's momenta, colours and shower scales. That ownership must
include the fold: the pair

\[
 \bigl(C_{a,f}, Q^{\rm shower}_{a,f}\bigr)
\]

must be stored and selected together. Later native evaluations or outer
sectors must not reconstruct `C_{a,f}` using a different random draw.

## Fixes

* `scale_module.f90` now stores the sampled H-event colours per outer
  sector and fold, alongside the existing saved shower scales.
  `init_process_module_n1body_wrapper` records them only for the outer
  calculation. Native history evaluations and restoration cannot replace
  them. `pick_unweight_contr` selects this saved assignment and
  `add_write_info.f` writes it without drawing or reconstructing colours.
* `sum_identical_contributions` also compares these colours before merging
  H contributions for unweighting. Different colour assignments are
  different shower events even if their momenta and scalar scales agree.
* The LHE writer rejects negative or non-finite scales on valid colour
  connections. In the initial 1,000-event samples, 25 top-pair events and
  49 W+jet events contained `-1` on such connections. Successful Pythia
  execution alone did not detect this error.
* The Delta shower steering enables
  `Beams:setDipoleShowerStartingScalesFromLHEF` when the installed Pythia
  exposes it. `Beams:setProductionScalesFromLHEF` remains enabled: the
  per-parton production scale and the per-dipole shower scales have
  separate uses. This steering fix is also in commit `2cb55190a`.
* The example analysis reader in `LHEFRead.h` now bounds its header and
  event-weight loops by the available input. Previously an event without
  an optional `<rwgt>` block could repeatedly append the header and hang
  while consuming memory. It reads weights only from the current event.

## Numerical mapping guards

The Delta fixes above initially left `genps_fks.f` unchanged. A subsequent
stability change now shares the algebraically equivalent constructions
between the outer and native maps. The coordinate-dependent branches
remain separate:

| Operation | Implementation after the stability change |
| --- | --- |
| Construct daughter angles from longitudinal and transverse components | The massive forward map now uses this arithmetic in both modes, evaluating the sine from each map's own coordinate. The massless forward map already does this at `628f316e2`. |
| Use the original Born mother direction in the forward boost | Both massive and massless forward maps now use the direction returned by `getangles` for the Born mother. |
| Sum spectator momenta for the inverse recoil and its direction | Both massive and massless inverse maps now sum the outgoing spectators. Forward spectators still carry Born momenta before their boost, so they cannot supply the new real recoil at that stage. |
| Disable the small soft/collinear sampling cutoffs | Keep the native-only exception. An auxiliary inverse must accept a physical point close to another history's singular boundary; changing the outer cutoffs is a separate integration change. |
| Use `tau = x` | Keep the flat auxiliary map native-only; the outer map uses threshold/resonance importance sampling. |
| Use `u = uBorn*(1-x_r**2)` and `y = cos(pi*x_theta)` for massive radiation | Keep the complete native parameterization together with its inverse and Jacobians. Promoting it to the outer map requires a separate change and validation of sampling, counterevents and supported resonance mappings. |

In particular, `sin(pi*x_theta)` is the correct stable sine for the native
angle coordinate. It cannot simply replace the outer sine while keeping
the outer definition of `x_theta`. Removing all guards would change more
than numerical arithmetic.

For the outer massive angle coordinate, writing
`q = cctiny + (1-cctiny)*x_theta**2`, the implementation uses

\[
 y=1-2q,\qquad
 \sin\theta=2\sqrt{q(1-\mathrm{cctiny})(1-x_\theta)(1+x_\theta)}.
\]

The factorization preserves the small transverse component near the
antiparallel endpoint. Both modes normalize the vector components
`L = E_i + |p_j|*y` and `T = |p_j|*sin(theta)` to obtain the emitted
parton's direction relative to the mother. The cutoffs, sampling
coordinates, radial branches and Jacobians retain their previous values.
The event and integration results below predate this stability change.

## Validation configuration

The four processes were:

```
p p > e+ e- [QCD]
p p > t t~ [QCD]
p p > w+ j [QCD]
p p > x0 / t QCD=2 QED=0 QNP=1 [QCD]
```

The first three use `loop_sm-no_b_mass`; Higgs production uses the
`HC_NLO_X0_UFO-heft` model and its HEFT parameter card. Proton beams have
6.5 TeV each. The built-in `nn23nlo` PDF has alpha_s(MZ) = 0.119 and is
matched to LHAPDF 244800 in the shower. Renormalization and factorization
scales are fixed to 91.188, 173, 80.419 and 125 GeV respectively. DY has
`m(e+e-) > 60 GeV`; W+jet uses anti-kT jets with R = 0.4 and pT > 30 GeV.
Top, W and Higgs widths are zero in the hard-process parameter cards.

The final Delta runs request 1,000 events each, 3% integration accuracy,
`ickkw=0`, `folding=[1,1,1]`, and physical `mcatnlo_delta=True`. Scale
reweighting is enabled and PDF reweighting is disabled. Each event has
27 reweights: nine scale combinations in each of three contribution-tag
groups. The shower uses the supplied `py8an_HwU_rates.o` example analysis,
with hadronization, without MPI, QED radiation or primordial kT. Top decay
is enabled in Pythia; W and Higgs remain stable.

Final hard-generation seeds are 10001 for DY, top pair and Higgs, and
92302 for W+jet. The ordinary top-pair check uses seed 92501. Fixed-order
seeds are 10000 for DY, top pair and Higgs, and 11000 for W+jet.

The fixed-order comparisons use the same hard-process inputs and request
1% integration accuracy. These comparisons are checks of the inclusive
rate, not an assertion that Delta and fixed-order predictions agree
beyond NLO.

| Process | MC@NLO-Delta integral [pb] | Fixed-order NLO [pb] |
| --- | ---: | ---: |
| DY | 1905 +/- 5.1 | 1912 +/- 8.7 |
| Top pair | 756.7 +/- 4.0 | 753.8 +/- 3.6 |
| Higgs (HEFT) | 31.01 +/- 0.16 | 30.94 +/- 0.16 |
| W+jet | 16460 +/- 160 | 16950 +/- 110 |

The differences are 0.69, 0.54, 0.31 and 2.52 combined integration
standard deviations respectively. All are below three combined errors.

All 4,000 final LHE events pass checks of colour conservation and complete
dipole coverage. Every colour-connected directed pair has a finite,
positive shower scale and every reweight is finite. The largest relative
four-momentum residual is `2.72e-14`. Unequal dipole scales occur in 54 DY,
898 top-pair, 35 Higgs and 812 W+jet events, so retaining individual dipole
scales matters for these samples.

Pythia tried, selected and accepted all 1,000 final events in each process,
with no reported shower errors. Both Delta scale settings are present in
the generated steering. All 27 reweight columns in the resulting HwU
histograms are finite. Their inclusive rates are 1874.5 +/- 67.7 pb (DY),
745.5 +/- 28.8 pb (top pair), 31.28 +/- 1.03 pb (Higgs), and
15894 +/- 918 pb (W+jet); these errors reflect the small event samples.

An additional ordinary MC@NLO top-pair run, with Delta and reweighting
disabled, generated 100 events and obtained 756.1 +/- 5.1 pb. Pythia
accepted all 100 events using positive scalar shower scales, with neither
Delta scale setting in its steering and without an optional `<rwgt>`
block in the event input.

The matrix-element and MC soft/collinear checks pass, as do pole checks
at all 20 points per process with the existing `1e-5` tolerance. The
following focused regression command passes all 63 tests:

```
python tests/test_manager.py test_born_support test_soft_col_limits \
  test_momentum_maps test_mc_dead_zones test_mc_kernels \
  test_mc_event_colours test_cluster test_pythia8 test_shower_card -t0
```

The new compiled Fortran test exercises the production colour assignment,
wrapper, contribution merging and event selection routines, with array
bounds checks. It changes the insertion draw in a later sector and fold,
interleaves native calls, and verifies that the selected owner retains
its colours. C++ reader tests cover absent headers, absent reweights and
the supported reweight formats; shell tests exercise the production
Pythia steering with and without the modern setting.

These are execution and consistency checks using small event samples.
They do not replace a high-statistics differential comparison, PDF
reweighting validation, or validation of FxFx and other matching modes.
The supplied Pythia 8.3 example driver also stops before analysing the
last selected event; accepted-event counts and analysis counts therefore
differ by one in these 1,000-event shower runs.

Run cards, generation and shower logs, LHE files, HwU rate histograms,
and `verified_lhe_audit.json` are retained locally under
`/export/tmp/rikkert/mg5_delta_628f316e2`. Final Delta samples use the run
name `delta_verified`; fixed-order samples use `delta_fo_check`.

## Stability follow-up checks

The shared forward/inverse arithmetic passes all 13 momentum-map tests
and 36 Born-support, soft/collinear, MC-kernel, dead-zone and colour-owner
regressions:

```
python tests/test_manager.py test_momentum_maps -t0
python tests/test_manager.py test_born_support test_soft_col_limits \
  test_mc_dead_zones test_mc_kernels test_mc_event_colours -t0
```

Three new cases fail on the pre-change `741431161` source and pass with
the shared arithmetic, using the same tolerances on both versions:

| Check | Error before the change | Required tolerance |
| --- | ---: | ---: |
| Outer massive map, nearly stationary sister: opening-angle cosine | 1.10e-6 | 1e-7 |
| Outer massive map, nearly stationary spectator: relative momentum round trip | 1.19e-7 | 1e-7 |
| Massless inverse with a soft spectator: relative Born momentum | 1.48e-8 | 1e-9 |

The additional soft-counterevent tests cover both mapping modes and verify
finite scaled emission vectors, the original Born momenta, and the sampled
opening angle. Supplying the existing `rat_xi` through `input_granny_m2`
reproduces the real momenta, Jacobian and measure. Existing tests continue
to check both massive branches and native/outer counterevent measure ratios.

The top-pair output was recompiled with the updated `genps_fks.f` and used
to generate 100 events each with ordinary MC@NLO (`stable_outer`, seed
92601) and physical Delta (`stable_delta`, seed 92602). The integral
estimates are 758.1 +/- 5.7 pb and 756.9 +/- 4.2 pb respectively. Both
runs pass the existing matrix-element, MC soft/collinear and pole checks
(20/20 pole points at tolerance `1e-5`). The 200 events have conserved
colour and valid shower scales; the largest relative four-momentum
residual is `2.56e-15`. Their cards and logs are saved alongside the
earlier Delta validation outputs. Pythia 8.313 tried, selected and accepted
all 100 events in each run, with no reported shower errors.
