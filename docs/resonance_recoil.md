# Replacing the Granny mass inversion

The production generator uses local recoil in channels previously handled by
the numerical Granny mass inversion. FKS subtraction, its finite soft mismatch,
and native MC subtraction use the same resonance frame. Map, subtraction-limit
and matched event checks are listed below. The earlier fixed-order rates used
insufficient cuts and are withdrawn as finite benchmarks; see the
[precision validation](resonance_recoil_precision.md) for the corrected cuts
and differential analysis. Full shower evolution has not been tested.

The target is the complete replacement of the numerical mass inversion and
finite-difference Jacobian, for every applicable final-state resonance channel.
The reference is [Ježo and Nason, arXiv:1509.09071, section 3](https://arxiv.org/html/1509.09071#S3).

## Selecting a final-state or incoming recoiler

Fixed-order FKS calculations can choose the recoiler for final-state emitters
with `Cards/FKS_params.dat`:

```
#FKSFinalRecoil
1
```

`0` (the default) retains automatic resonance recoil, with global final-state
recoil when no resonance subsystem is selected. `1` and `2` select the
corresponding incoming leg. Incoming recoil requires two incoming particles and
a massless parton with a hadronic PDF on the selected beam. Initial-state
emitters continue to use their existing ISR map. Fixed beams and dressed lepton
beams cannot supply the required variable momentum fraction. MC@NLO runs reject
the incoming recoil option: adapting the shower counterterms is separate work.
Regenerate an existing process output to install the updated Fortran sources.

Callers can instead provide a logical mask in real-emission external-leg labels:

```fortran
      recoilers=.false.
      recoilers(1)=.true.
      call set_fks_recoilers(recoilers,pass)
```

Select exactly one incoming leg, or one or more final-state legs; exclude the
emitter and emitted leg. A final-state mask receives a common recoil boost,
preserving the invariant masses within that system and leaving every other
physical spectator fixed. Set the mask after initializing the current process,
FKS indices, masses and beams. It overrides the card until
`clear_fks_recoilers()` is called and is checked again for each active sector;
callers changing external-leg labels must replace or clear their mask.
An invalid request returns `pass=.false.` without replacing the previous one.
The low-level forward/inverse routines also accept the mask (FF) or incoming
leg and momentum fraction (FI) explicitly.

### Incoming recoil map and measure

Let `pbar_a = xbar_a P_a` be the incoming Born parton and `pbar_j` the
final-state Born emitter. Introduce the auxiliary massless momentum
`Rbar = (1-xbar_a) P_a` and the timelike radiation frame `K=pbar_j+Rbar`.
The ordinary final-state map splits the emitter inside this system, leaving
`R=lambda Rbar` on the beam axis. The physical incoming parton becomes

```
p_a = P_a - R
x_a = 1 - lambda (1-xbar_a).
```

Only this incoming leg and the two daughters change. This has the local FI
recoil structure described in the [Pythia 8.3 manual, equation (117)](https://pythia.org/download/pdf/pythia8300.pdf):
the incoming fraction grows while its direction stays fixed. The radiation
coordinates and sampling remain FKS coordinates. A positive reservoir enforces
`x_a <= 1`, including configurations with `x_a/xbar_a > 2`. The massive map
retains both kinematic solutions. The inverse reconstructs the Born fraction
and recovers its azimuth in the Born CM frame.

Including the integration over the incoming fraction changes the radiation
measure to the ordinary FF measure divided by `lambda`. The real flux and PDF
arguments use the new physical incoming momenta; counterevents use their own
incoming fractions. This extra Jacobian is essential even though the auxiliary
momentum construction uses the existing FF kernels.

`lambda=1` at the soft and massless-collinear endpoints. Consequently the local
soft eikonal, spin-correlated collinear term and Born/virtual normalization use
the existing resonance-frame formulas with `K` as their reference momentum.
The soft endpoint includes the finite angular mismatch below, and the
collinear and soft-collinear endpoints use their corresponding rescaled
cutoffs. The integrated Born/virtual expression is unchanged. In
`d=4-2 epsilon`, the reservoir correction is `lambda^(-1+2 epsilon)`;
`log(lambda)` removes both singular limits, so its epsilon-dependent factor
leaves no additional finite integrated term. Choosing an incoming spectator
for a final-state emitter introduces no PDF factorization counterterm.

### Checks for incoming recoil

```
python3 -m unittest tests.unit_tests.fks.test_recoil_selection \
  tests.unit_tests.fks.test_initial_recoil \
  tests.unit_tests.fks.test_resonance_recoil \
  tests.unit_tests.fks.test_momentum_maps \
  tests.unit_tests.fks.test_pythia8_matching
```

The incoming-recoil tests compile the production routines with runtime checks.
They cover both beams, swapped emitter/emission labels, massive and massless
emitters, both massive solutions, the inverse and spin phase, soft and collinear
counterevents, invalid selections, and the actual endpoint prefactors. The
production generator is checked for its real/counterevent fluxes, momentum
fractions, Born projection and asymmetric longitudinal boost. A massive
one-particle Born final state checks that no spare final-state slot is needed
for the auxiliary reservoir.

An independent integral over the emitted momentum and incoming fraction checks
the entire radiation measure against
`K^2 [1-r+r log(r)]/(16 pi^2)`, with `r=m_j^2/K^2` and `r log(r)=0` at zero
mass. This check includes the extra reservoir Jacobian and both massive
branches. The endpoint tests compare the local and global integrated collinear
expressions and exercise the finite soft correction through the production
prefactors.

The generated process `u d > u d [QCD]` builds both the matrix-element limit
tester and the full fixed-order executable. For each incoming-recoil choice,
all six sectors pass their soft/collinear test summaries. Across the two runs,
239 of 240 nonzero individual samples satisfy the stricter convergence test;
one beam-2 sample becomes numerically unstable at simultaneous very soft and
collinear kinematics. Its daughter-pair invariant approaches external-leg
mass-shell roundoff, while its phase-space weight converges to the endpoint.
The same-seed default-recoil baseline has no failed samples. No test tolerance
was changed. Logs and source hashes are in
`/tmp/mg5_initial_recoil_generated_1t4ke8e2/validation_manifest.json`.
These checks do not constitute an integrated cross-section comparison.

## Radiation map

`SubProcesses/resonance_recoil.f` takes an explicit mask of the resonance's
external descendants, in real-emission leg labels. The mask includes the emitted
leg, whose momentum is zero in the input Born configuration. The existing tree
information in `set_granny` supplies this mask; the aunt may itself be a decay
subtree.

The map sums the descendant momenta to obtain `K`, boosts them into its rest
frame, and invokes the existing massless or massive FSR map with invariant mass
`K^2`. Only the aunt's descendants receive the recoil boost. Boosting the result
back preserves `K`, every invariant within the aunt subtree, and every momentum
outside the resonance. The physical incoming momenta never enter the auxiliary
decay map and remain unchanged. The caller must use the full-process flux.

The inverse uses the same descendant mask, recovers the three radiation
coordinates, and projects onto the local Born momenta. It supports both the
ordinary and native massive parametrizations. The collinear helicity phase is
transported back to the frame in which the spin-correlated Born is evaluated.

`xi` and `y` returned by these routines are resonance-frame variables. Their
energy-divided emission vector is boosted back together with the momenta. They
must not be interpreted as the global CM energy fraction and opening angle.

## Production integration

`generate_momenta_conf_wrapper` selects the existing Granny descendant mask and
calls the generator once. The ordinary Born resonance-mass sampler is used; the
root solve, finite-difference Jacobian and repeated event generation are removed.
`resonance_recoil.inc` carries the selected mask and resonance momentum for the
current sector. The same map generates the event and its soft and collinear
counterevents. The full incoming energy still determines the flux and PDFs.

The Born normalization divides out the local radiation measure. Soft eikonal
normalization and the reduced final-state collinear matrix element use the local
energy fraction and angle. The spin-correlated Born remains in the physical
frame, with its azimuthal phase transported from the resonance frame.

For real multichannel weights, every Born diagram is evaluated at a common
global FSR projection. Using each diagram's separate local projection in an
otherwise unchanged ratio would fail to give a partition of unity at fixed real
momenta. Counterevent and shower weights use the local Born projection. The
common global real projection is correct as a partition, but its sampling
performance near radiative resonance peaks remains a possible optimization.
The common projection is evaluated with a common Born coupling scale, including
for mixed Born orders. Ordinary MC weights and native recoil-group weights use
their local Born coupling scales consistently.

The MC@NLO native history sum groups Born diagrams by their recoil mask. Each
group selects its own tree and projection; group weights sum the corresponding
Born diagrams. The real group weight uses the common projection, while MC and G
terms use that group's Born. Outer generation is replayed after the native sum
to restore its frame and phase-space context. Shower invariants are computed in
the selected resonance frame; matrix elements and colour connections retain
physical momenta. MC G replacement terms do not receive an integrated soft
mismatch.

## Soft-subtraction scheme

The mismatch is combined with the existing FKS cut-based endpoint prefactors.
`resonance_subtraction_scales` supplies the factors below. Tests cover their
covariance, collinear limit, radial integral and algebraic Q-term conversion.
Matrix-element limit tests pass. The earlier coarse fixed-order comparison and
technical-cut variation agreed within their quoted one-percent uncertainties,
but later probes found additional unsubtracted collinear limits admitted by
those cuts. Those rates do not validate the finite mismatch.

Let `Q` be the total incoming momentum, `s=Q^2`, and `K` the resonance momentum.
For a null soft momentum `k`, define the ratio of energy fractions

```
a(k) = xi_K / xi_Q = s (K.k) / (K^2 Q.k).
```

For a massless Born emitter `p_j`, let `a_j=a(p_j)` and
`D_j=E_j(K frame)/E_j(Q frame)=a_j sqrt(K^2/s)`. In its collinear limit,
`1-y_Q = D_j^2 (1-y_K)`. Choosing the local subtraction reference scales as

```
xi_cut,K = a_j xi_cut,Q
delta_K  = delta_Q / D_j^2
```

makes the ordinary integrated final-state collinear Q expression algebraically
identical in the two frames. In particular, substitute these scales and
`E_j,K=D_j E_j,Q` into the expression in `bornsoftvirtual`; all changes cancel.
The implementation retains the existing integrated Born/virtual term and evaluates
the finite soft difference with the same Born matrix element and sector weight.

The radial integral of the exponential difference in eq. (92) can be done
analytically, since the eikonal is homogeneous in the soft energy:

```
integral_0^infinity [exp(-a xi)-exp(-b xi)] dxi/xi = log(b/a).
```

With the rescaled local soft reference the mismatch is therefore proportional
to `log(a_j/a(k))`. For a massless emitter this logarithm vanishes in the
collinear limit; it can be integrated over the remaining soft angles. In a
cut-based implementation its contribution shares the existing
soft-counterevent angular sample and normalized soft-energy sampling. Combining
it with the local soft endpoint logarithm amounts to replacing the cutoff in
that logarithm by `xi_cut,Q * a(k)`, while the soft-collinear endpoint uses
`xi_cut,Q * a_j` and the collinear endpoint uses `delta_Q/D_j^2`.

Massive emitters have no collinear endpoint, so only their soft endpoint changes.
MC G replacement factors retain their physical subtraction limits and do not
acquire the finite integrated mismatch. Native H-only evaluation adds no Born
remnant. Map tests cover massive emitters; the generated cross-section comparison
uses a massless bottom emitter.

## Current numerical checks

```
python3 -m unittest tests.unit_tests.fks.test_resonance_recoil
python3 -m unittest tests.unit_tests.fks.test_momentum_maps
```

The local-map tests compile production routines with `-fcheck=all`. They cover
one and two incoming particles, a massless or massive external aunt, a decaying
aunt, massless and massive emitters, both massive solutions, both native and
ordinary coordinates, boosted resonances, interchanged radiation labels, soft
and collinear counterevents, inverse coordinates and measures, invalid masks,
and the boosted helicity phase. An independent Lorentz-invariant three-body
phase-space recursion checks the integrated radiation measure for all mass
choices. The existing momentum-map regressions also pass after generalizing
the inverse FSR kernels' incoming-momentum sum to one incoming particle.

The production-map check additionally verifies the full-process flux, stored
counterevent measures, native inverse and shower-frame Born momentum using the
actual generator. A partition test uses mixed Born orders and changing coupling
scales to check real partition unity and agreement between outer and native MC/G
weights, for local and foreign Born providers.

The focused map, partition, native-weight, shower-kernel, dead-zone and scale
suites run 60 tests successfully.

A generated `u b > d b w+ QCD=0 [QCD]` process passes 252 nonzero matrix-element
soft/collinear checks and 230 nonzero shower-counterterm checks. With the
complex-mass model, its virtual poles cancel at all 20 sampled points. These
checks alone do not validate the finite mismatch or matched cross section.

### Finite validation process

Use the complex-mass scheme for coloured particles with nonzero width. In
particular, a real-mass top model with a nonzero top width fails the QCD virtual
pole check in this example. The comparison currently uses

```
set complex_mass_scheme True --no_save
import model loop_sm-no_b_mass
generate u b > d b e+ ve QCD=0 [QCD]
```

Keep finite top, W and Z widths (1.4915, 2.085 and 2.4952 GeV in this check).
The external-W variant with zero W width is unsuitable: its crossed real
`g b > W+ b d u~` channel contains a zero-width internal W-minus pole. The
largest sampled weights approach `m(d,u~)=m_W`. Resolving jets does not regulate
this pole. Those earlier integrations were stopped and are not baselines.

Use anti-kt jets with R=0.4 and pT above 20 GeV. Require separate jets with
positive net bottom and positive net down flavour. The validation-only
`tests/input_files/resonance_recoil_cuts.f` supplies that requirement.

The earlier requirement of two jets and any nonzero bottom tag was insufficient.
In this restricted Born process, `g b > d b e+ ve u~` has an unsubtracted limit
with beam-collinear d, while `u g > d b e+ ve b~` has one with beam-collinear b.
Their other underlying Born flavours are not part of the generated process.
Direct matrix-element probes find `Sij=1` and a nonzero limit of `pT^2 |M|^2`
in both cases while the old cuts accept the events. The corrected flavour-jet
requirements reject both limits, as well as a collinear photon-to-bottom-pair
jet with zero net bottom flavour.

These cuts are intended for this explicitly flavour-labelled subprocess, not
generic processes that group bottom and light flavours. FastJet checks cover
resolved jets, collinear QCD radiation and all three rejected configurations.
Cuts are applied to events and counterevents through the same `dummy_cuts` hook.

### Generated results

The previously reported rates `5.148 +/- 0.053`, `5.137 +/- 0.055` and
`5.160 +/- 0.050` pb are **not finite benchmarks**: their cuts admit the extra
collinear limits described above. Their apparent agreement is insufficient.
The [0.1% repeat](resonance_recoil_precision.md) uses the corrected cuts for
both implementations and the FKS-cutoff variation, with differential histograms.

The variation changes `(xicut, deltaO, xiScut, deltaS)` from
`(0.5, 1, 0.5, 1)` to `(0.1, 0.2, 0.2, 0.3)` in `fks_powers.inc`, followed by
recompilation. The earlier runs finished and passed their matrix-element and
virtual-pole checks (20/20 poles at tolerance 1e-5); these tests did not probe
the extra limits outside the selected Born flavour channel.

The leptonic process also passes 359 nonzero Pythia8 subtraction-limit checks
across all Born configurations and FKS sectors. The earlier external-W test
passes the Herwig7 limits quoted above; those local limit tests do not rely on
its divergent inclusive integral.

A focused matched run in the top Born channel exercises both production and
decay recoil groups. Reversing both the FKS-history and recoil-group loops gives
agreement at 8e-15 for 2,000 identical fixed-grid points. A 500-event generation
check contains 367 S and 133 H events, including 11 gluons assigned to top decay.
Stored resonance four-momenta equal the sums of their direct daughters within
1.2e-13 GeV. The matched and reweighting executables compile and link.

These event-generation checks use short integration grids and the older cuts;
they test event structure, not a finite matched cross section or the corrected
observable. The longer all-channel matched integration was
stopped after the focused checks; it was still setting up its first two grids.
Full shower evolution and high-statistics matched distributions remain outside
this validation.

Validation files are outside the checkout in
`/tmp/mg5_resonance_validation_3yhq5u5h`, with separate generated template copies.
The completed precision runs are `single_top_leptonic_baseline_refined`,
`single_top_leptonic_refined`, and `single_top_leptonic_cutcheck`. The matched
checks and events are in `single_top_leptonic_mc_smoke`. Its saved diagnostic
driver and logs record recoil groups and per-point probes; the final executable
is rebuilt with the production driver. The production template contains neither
diagnostic output nor reversed history loops.
