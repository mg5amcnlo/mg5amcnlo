# Implementation plan for the PYTHIA8 subtraction audit

Date: 27 September 2026. Baseline: `a9e1a51c5`, branch
`MCcntRefactor_Sfun`. Packages A and D were implemented on 28 September
2026 for the scoped hard-system PYTHIA8 scalar terms.

Status correction, 9 October 2026: commit `4048df117` resolved the audited
ISR projection mismatch by changing the shared FKS forward/inverse maps and
finite endpoint corrections. This supersedes the separate MC projection
and S-map repair proposed below. Commits `62dc0925b` and `85f6b30aa` retain
the matching-compatible default and stabilize the inverse. See the
[current mapping and validation notes](../native_fks_projection.md).
Full process-level matching validation remains outstanding.

This plan addresses [the PYTHIA 8.318 audit](pythia8318_matching.md). It
preserves the fixed-order azimuthal correlations, the unregularized infrared
behaviour needed for subtraction, and the G-function replacements. The aim
is to repair the shower-dependent kinematics, normalization and resolved
scalar weights while maintaining local cancellation of the real-emission
singularities.

Package A uses exact massive fractions and an implicit energy-conservation
derivative instead of activating the old nonuniform massive soft expansion.
The shared helper, now in `mc_counterterms` in `montecarlocounter.f`, keeps
the support/damping scale consistent. The absolute massive geometric prefactor is shared by all
showers, since it converts the FKS phase-space measure and each shower's
radiation Jacobian already takes its magnitude. Massive squark/gluino
controls and HERWIG6, HERWIG7, PYTHIA6Q and PYTHIA8 measure regressions are
tested on both branches with independent finite-difference determinants.
The 1,800-point reproducer now checks positive signed measures, with maximum
relative error `8.8e-14`. New endpoint tests cover both sides of `1d-5`,
small masses, near-threshold recoil, branch coalescence and small radiator
momentum, including checked/FPE-trapped and poisoned builds.

For package D, the guarded recoil factor is itself symmetric: with
`A=1-r+v`, its denominators are `A*z-v` and `A*(1-z)-v`. They exchange
under daughter interchange, including their identical `XMARGIN=1d-12`
guards. Thus `W_sym=D_rec` for the configured global hard-system recoil.
The production regression sums both `fks_Hij` partitions and both colour
connections, with common Born states, local-dipole support and independently
varied randomized start-scale damping. Scalar gluon weights change; exact
helicity terms, other channels and the G prescriptions are preserved.
The launch path explicitly sets `TimeShower:recoilDeadCone=on`.
The new scalar factor is restricted to the audited two-incoming hard
context and the resolved branch. The analytic collinear branch retains
its exact AP coefficient, without the reference generator's numerical
numerator floor. One-mother/MEC contexts retain their prior kernel.

For package A, writing `E=kn0`, `k=kn`, `e=sqrt(S)*xi/2` and `d=1-y`,
the stable gap is `a=m^2/(E+k)+k*d=E-y*k`. The exact fractions are

\[
 z=\frac{(E-k)^2+2Ekd}{2a(E+e)},\qquad
 1-z=\frac{m^2/(2a)+e}{E+e},\qquad
 t=z(1-z)\,2ea.
\]

Implicit differentiation of
`(2-xi)*E+xi*y*k=(S*(1-xi)+m^2-M^2)/sqrt(S)` gives, with
`g=(2-xi)*k+xi*y*E`,

\[
 \frac{\partial w}{\partial y}=-\frac{2\sqrt S\,\xi k^2}{g},\qquad
 \left|\frac{\partial(z,t)}{\partial(\xi,y)}\right|
 =\frac{2\sqrt S\,k^2}{|g|}z(1-z)^2.
\]

The prefactor uses the same `|g|/k^2` geometry, so the combined measure
remains stable on both branches, near their meeting point and at small
`k`. The retained massless soft/collinear approximations are checked with
their `O(xi)`/`O(1-y)` relative remainder envelopes; the massive formulas
and exact support/damping scale use a `1e-12` relative test tolerance.

Validation command:
`python3 -m unittest tests.unit_tests.fks.test_pythia8_matching`.
These tests concern no-MEC first hard emissions. Active resonance MECs,
alternate recoil/settings contracts, and the full S/H expansion retain the
separate validation requirements below.

## 1. Physics decisions and scope

The distinction in the question is essential. The complete MC subtraction
term must have the singular limits of the fixed-order calculation. Copying
PYTHIA's approximate azimuthal distribution, shower termination or ISR
screening into that term would generally defeat this requirement.

The original MC@NLO construction explicitly discusses replacing the shower's
soft behaviour to obtain a local counterterm, and retaining locally necessary
collinear azimuthal correlations. This supports preserving the *role* of
these ingredients; it does not validate every current implementation detail
or every finite extension of a limiting expression. See
[Frixione–Webber, section 5 and appendix A.5, especially equations
(A.82)–(A.87)](https://arxiv.org/pdf/hep-ph/0204244).

Use three separate requirements:

1. **Local subtraction:** the complete native counterterm reproduces the
   fixed-order soft, collinear and soft-collinear limits, including spin and
   colour correlations, in its actual integration measure.
2. **Shower-dependent resolved part:** the inverse Born map, luminosity,
   phase-space measure, support and applicable finite scalar shower weights
   agree with the selected PYTHIA shower prescription.
3. **Matching:** the expansion of the full S/H construction has the stated
   accuracy for infrared-safe observables. Differences deliberately retained
   in requirement 1 need their own angular-cancellation or infrared-power
   analysis. They must not be declared higher order merely because they are
   finite after integration.

The first implementation covers QCD radiation from the hard Born system,
Simple showers, global II ISR recoil, and the first global FSR emission with
the existing local-dipole maximum. Keep FKS integration and ordinary NLO
subtraction. Preserve the current history, identical-particle and colour-flow
normalizations unless a dedicated test demonstrates an error.

The audited reference is 8.318; the configured local installation is 8.313.
Validate both explicitly. This is not a request to upgrade PYTHIA. Resonance
decay matching, FxFx/UNLOPS and alternative shower/recoil algorithms need
separate validation before using the new projection path. Existing QED and
SUSY support must not change through an unguarded QCD/PYTHIA8 modification.

### Disposition of the audit findings

| Finding | Decision | Completion condition |
|---|---|---|
| Uninitialized `xjacPY8` threshold | Fix immediately; validate the newly activated endpoint expressions | Initialization-independent results and stable endpoint transitions |
| Negative massive-FSR scalar prefactor | Correct the geometric sign | Both physical FKS solutions reproduce a positive phase-space density |
| ISR Born fractions, recoil and PDFs | Implement a separate PYTHIA Born projection and propagate it through the complete raw MC contribution, including a locally subtracted S integral | Inverse/forward closure, correct amplitudes/PDFs, local endpoint cancellation and Born-differential S cancellation |
| Missing gluon recoil dead cone | Include in the resolved scalar term with the correct daughter-ordering sum | Agreement with the weighted PYTHIA sum and unchanged fixed-order limits |
| Different azimuthal model | Preserve exact Born-helicity interference and its G interpolation | Pointwise collinear limits and the appropriate angular-integral checks |
| Shower cutoffs and ISR `pT0` screening | Preserve the unregularized subtraction | Infrared cancellation plus a separately measured cutoff/screening dependence |
| G soft/collinear replacements | Preserve the construction, including one replacement per native history | Correct limits, overlap subtraction, S/H accounting and transition-region checks |
| Massive shower charm/bottom versus massless hard kernels | Treat as a flavour-scheme choice; do not copy shower mass terms indiscriminately | Separate massless and massive reference comparisons |
| `bogus_probne_fun`, mode 2 | Retain as an explicit diagnostic; use `P=1` for ordinary production/reference matching | Physical defaults and a separately tested debug mode |
| MEC applicability and implicit shower settings | Make the hard-production comparison contract explicit | Generated settings and actual initialized settings agree |
| `nPartonsInBorn=-1` | Follow-up event-classification check | Correct H/S recoil treatment for representative exported samples |
| `z`, `t`, full Jacobian magnitude and upper bounds already agreeing | Retain; add regressions around the fixes | No extra normalization factor or second Jacobian |
| Labelled versus symmetric `g -> gg` kernels | Retain existing counting initially | Compare complete daughter-ordering and colour-connection sums |

## 2. Baseline evidence and code ownership

Running `python3 docs/audits/check_pythia8318.py` on the baseline reproduced:

| Check | Baseline result |
|---|---:|
| Allowed FSR samples | 1,800 |
| Maximum absolute `z` error | `5.50e-15` |
| Maximum relative `t` error | `2.63e-13` |
| Maximum relative scalar-Jacobian magnitude error | `1.96e-11` |
| Maximum FSR Born momentum error / 1,000 GeV | `2.25e-14` |
| Accepted negative massive scalar coefficients | 12 |
| ISR Born fractions required by the reference | `(0.12, 0.20)` |
| ISR Born fractions currently used | `(0.1473576795, 0.1628690142)` |
| Current ISR PDF fractions | `(0.2455961325, 0.1628690142)` |
| Required ISR mother/other-beam PDF fractions | `(0.20, 0.20)` |
| First `xjacPY8`, ordinary / poisoned unset locals | `28658.6904700 / 74170.0669377` |

These are source-level tests, not showered event validation. The existing
script intentionally asserts the presence of two defects. Convert its
fixtures into tests of corrected behaviour; preserve the historical results
in the audit. Do not leave an unconditional `assert negative` or an assertion
that poisoned initialization must change the answer in the post-fix check.

Main implementation sites:

| File | Relevant responsibility |
|---|---|
| [`montecarlocounter.f`](../../Template/NLO/SubProcesses/montecarlocounter.f) | `compute_MCsubtraction_kl`, `xmcsubt_connection`, kernels, `get_mbar`, `xfact_ileg*`, shower invariants, `compute_gfun`, PYTHIA radiation variables, support and damping |
| [`genps_fks.f`](../../Template/NLO/SubProcesses/genps_fks.f) | FKS generation/inversion and counterevents; keep these as FKS maps |
| [`genps_fks_helpers.f`](../../Template/NLO/SubProcesses/genps_fks_helpers.f) | Shared geometry, momentum utilities and FKS coordinate reconstruction; includes the former kinematics module |
| [`fks_singular.f`](../../Template/NLO/SubProcesses/fks_singular.f) | Native MC/G/real assembly, luminosities, records, grouping, folding, reweighting and event selection |
| [`driver_mintMC.f`](../../Template/NLO/SubProcesses/driver_mintMC.f) | Native-history activation, Born-flow sampling, cuts, scales and H redistribution |
| [`mc_native_runtime.f`](../../Template/NLO/SubProcesses/mc_native_runtime.f) | Native Born evaluators and colour-flow results |
| [`weight_lines.f`](../../Template/NLO/SubProcesses/weight_lines.f) | Persistent weight, momentum, flavour and event metadata |
| [`makefile_fks_dir`](../../Template/NLO/SubProcesses/makefile_fks_dir), [`export_fks.py`](../../madgraph/iolibs/export_fks.py) | Build dependencies and export of any new production module |
| [`born_support.py`](../../madgraph/iolibs/born_support.py) | Generated native-context support, if its interface needs extending |
| [`MCatNLO_MadFKS_PYTHIA8.Script`](../../Template/NLO/MCatNLO/Scripts/MCatNLO_MadFKS_PYTHIA8.Script), [`shower_card.py`](../../madgraph/various/shower_card.py), [`Pythia83.cc`](../../Template/NLO/MCatNLO/srcPythia8/Pythia83.cc) | Shower settings, hard-system classification and runtime validation |

The [archived recoil module](../pythia8_shower_subtraction.f90) contains a
candidate II inverse, and its [Born adapter](../pythia8_shower_born.f)
contains spin-basis handling worth reusing. Extract and review the relevant
pieces. The prototype has additional kernels, mass choices and topology
support, and calls `sborn` directly in its adapter. Activating it wholesale
would bypass the present native-history contract.

## 3. Establish the matching contract before wiring in the ISR map

### 3.1 Keep the pieces distinguishable

For a native labelled history `h`, write schematically

\[
 D_h=D_h^{\mathrm{raw},G}+D_h^{\mathrm{replacement}},\qquad
 H_h=P_h\,[S_h R-D_h].
\]

Here `D_raw,G` includes the existing scalar and exact helicity terms with
their G factors and shower-start damping. `D_replacement` denotes the
soft/collinear/overlap contributions supplied by the FKS routines. This
notation suppresses their different counterevent measures and projections;
the implementation must retain those distinctions.

The corresponding S construction includes `P_h D_h` and
`(1-P_h) S_h R`, together with the ordinary NLO Born, virtual and integrated
subtraction terms. In the code, the raw MC pair is types 12/13, real
redistribution uses types 1/11, and the G replacement appears in the paired
types 4–6/8–10. Types 4–6 also contain ordinary FKS terms: they cannot be
treated as pure G records without separating their coefficients.

Produce a short algebra-to-code ledger specifying, for each piece:

- its real and Born momentum arguments and frame;
- incoming flavours, PDF arguments, flux and coupling order;
- colour-flow probability, connection factor, history/orbit factor and
  identical-particle denominator;
- radiation Jacobian versus integration proposal weight;
- G factor, start-scale damping, `P_h`, cuts and event projection.

Derive the S/H expansion from this ledger. Test inclusive conservation and
observable-level cancellation separately. Equality of sums of scalar event
weights establishes the former only.

### 3.2 State explicitly where shower equality is required

Let `K_h` be the perturbative first-emission density in the real measure and
`beta_h(Phi_R)` its shower Born projection. If the subtraction is split into
pieces `D_h,a` with S projections `pi_h,a`, its first-order residual has the
schematic form

\[
 \mathcal E[O]=\sum_h\int d\mu_R\left\{
 K_h[O_R-O(\beta_h)]
 -\sum_a D_{h,a}[O_R-O(\pi_{h,a})]\right\},
\]

for the `P=1` reference, with counterevent measures pulled back consistently.
Include the real redistribution when studying a general `P`. This is a
diagnostic identity, not an assumption that each G-replacement record has
the same Born projection as the shower.

For each deliberately retained difference, establish whether the residual:

1. cancels after the required angular/history integration;
2. is an infrared power correction with a stated resolution limit; or
3. contains an unresolved finite first-order mismatch requiring a separate
   matching correction or a documented restriction of the accuracy claim.

In particular, zero azimuthal average of a kernel difference does not imply
zero integral against a finite-angle observable or nonuniform cuts. A finite
G transition width is not an extra power of `alpha_s`. Keep these effects
visible during validation instead of forcing the full counterterm to equal
PYTHIA point by point.

## 4. Work package A: numerical and massive-branch fixes

These fixes can be delivered before the projection work.

### A1. Initialize and validate the PYTHIA endpoint threshold

1. Give `xjacPY8` an explicit threshold. Start from `1d-5`, the value already
   used by `zPY8` and `xiPY8`, and check it together with
   `dinvariants_dFKS`. A shared named constant is useful only after verifying
   which switches actually need the same value.
2. Check the *combined* `z`, `t` and Jacobian expansion. Initializing `tiny`
   activates formulae that were normally bypassed by zero-filled storage;
   initialization alone is therefore insufficient.
3. Sweep `xi` and `1-y` independently and together across each switch, from
   about `1e-3` to `1e-12` where double precision is meaningful. Include
   massless and massive radiators, composite recoil, a small radiator mass,
   and the approach to Born threshold.
4. Compare analytic limits with a stable invariant expression or a
   higher-precision reference. Use finite-difference determinants only in
   the interior, where their step-size dependence can be controlled.
5. Check continuity at the switches and the product used by the regulated
   kernel. A small relative error is not a useful test when the reference
   quantity vanishes; use scaled absolute errors there.
6. Compare optimized and checked builds, including the audit's
   `-finit-real=inf` build. Add a signaling-NaN/FPE-trap fixture with all other
   required state initialized. Test failures must reach the harness rather
   than being hidden by `add_wgt` skipping NaNs.

Acceptance: the supplied hard-point values are unchanged apart from intended
fixes, all initialized builds agree, and endpoint errors have a measured
envelope consistent with the retained expansion order. Set endpoint
tolerances from that envelope, not from the interior sample's `1e-11` error.

### A2. Correct the massive geometric prefactor

The offending factor is

\[
 A=2-\xi\left(1-\frac{E_r}{|\mathbf p_r|}y\right).
\]

`xfact_ileg3` retains its sign while the radiation Jacobian and FKS measure
use absolute determinants. Derive the prefactor on both physical branches
and replace this geometric sign by the appropriate absolute determinant.
For the audited expression this means `abs(A)` in the prefactor, with the
remaining positive factors unchanged.

Do not apply `abs` to the complete MC contribution: colour/spin interference
and matched event weights can legitimately be signed. Do not reject the
second physical solution or alter the support to conceal it.

The helper is used by several showers and massive kernels. Either establish
the same determinant convention for every caller and fix the shared helper,
or initially limit the corrected prefactor to the PYTHIA8 branch and record
the remaining callers for review. Include a SUSY massive-kernel control.

Required fixtures:

- the accepted `m=M=173 GeV`, `sqrt(S)=1000 GeV`,
  `z=0.23921928965797884`, `t=41764.76709871376 GeV^2` point;
- the other negative points in the deterministic audit sample;
- ordinary first-solution points and the neighbourhood where branches meet;
- physical points with a small radiator momentum, using stable products if
  `kn0/kn` and the separate Jacobian become poorly conditioned.

Acceptance: the quoted coefficient becomes `+0.313987919506...` for
`N_p=1`; all 1,800 allowed samples agree in sign as well as magnitude with
the independent measure. Branch multiplicities remain unchanged.

## 5. Work package B: reconstruct the PYTHIA ISR Born state

### B1. Add a narrow projection interface

Introduce a small production module, provisionally
`Template/NLO/SubProcesses/pythia8_projection.f90`. Its initial public
contract should take real momenta, incoming hadron momenta or beam energies,
the native external ordering and emitter, and return:

```text
status: allowed / outside physical support / invalid input
p_born_mc, x_born_mc(2), x_real(2)
z, t, recoil transformation, Born-to-lab frame information
transverse reference for the helicity phase
history / emitter / external-label mapping
```

Keep this kinematic helper independent of PDF evaluation, Born amplitudes,
G functions and the global FKS COMMON blocks. Distinguish a genuine dead
zone from a numerical failure. Do not silently substitute the FKS projection
when inversion fails at an otherwise physical point.

For beam A emission, require

\[
 \bar x_A=z x_A^R,\qquad \bar x_B=x_B^R,
 \qquad z=1-\xi,\qquad
 t=\frac{\hat s_R\xi^2(1-y)}2.
\]

Exchange the beam roles for beam B. Undo the complete II recoil
boost/rotation of every hard final particle. Recovering the Born invariant
mass or applying a longitudinal boost alone is insufficient.

Use the archived II inverse as a starting point, including its asymmetric
beam-B reference-axis convention. Review it against both versioned PYTHIA
sources. Add the corresponding forward map for round trips and the S
integration in package C. Keep `generate_momenta_initial_inverse` as the
FKS inverse; other NLO terms still require it.

### B2. Evaluate the resolved Born and spin terms consistently

1. Pass `p_born_mc` explicitly to the fixed-real-point raw PYTHIA8 evaluator
   instead of replacing the process-wide `/pborn/` state. Apply C0 before
   changing the existing fixed-Born S evaluator to use this path.
2. Use `sborn_native` and the active history's Born provider. Invalidate the
   local `calculatedBorn` cache when changing momenta; verify the native
   provider cache keys include the numerical point, relevant model state
   and spin request.
3. Recompute Born-flow weights at this point. The FKS Born flow probabilities
   generally differ at finite recoil. Keep the actual sampling probability
   with the draw and divide by it exactly once.
4. Retain the exact helicity interference. Transform the transverse reference
   and Born momenta together into the evaluator's convention, including the
   beam-2 rotation and the special `2 -> 1` case. Check against the existing
   FKS helicity expression in the emitting collinear limit.
5. Restore ordinary FKS Born/spin state before calling the exact soft and
   collinear replacements. A new MC Born evaluation must not contaminate a
   cached FKS counterterm or the next native history.

For validation, sum all colour flows explicitly. For production, either
sample a separate flow for the raw MC block or use a proposal with support
for both FKS and MC flow distributions. Reusing a FKS draw without checking
support and retaining its original denominator can bias the result.
`include_born_flow_weight` currently changes several prefactors together;
do not overwrite the G/real flow normalization while correcting the raw MC
one.

### B3. PDFs, measures, support and scales

Call `get_mc_lum` with `x_born_mc` for the fixed-real-point raw term.
For beam A its PDF arguments must become
`(x_born_mc(1)/z, x_born_mc(2)) = x_real`.
Keep the native backward-evolution flavour assignment, including
flavour-changing histories; the Born daughter flavour and PDF mother
flavour are not interchangeable.

The audit's `xlum_mc_fact=(1-xi)/z` is one for massless PYTHIA ISR. It does
not correct a wrong Born point. Retain the existing `1/z` factor exactly
once in the complete backward-evolution density. Test ordinary `f(x)` versus
`xf(x)` conventions explicitly with a simple nonconstant PDF fixture.

Check the normalization against

\[
 d\mu_n=dx_A\,dx_B\,\frac{d\Phi_n}{2\hat s_n},\qquad
 J_{\rm ISR}=\frac1{32\pi^3(1-z)}.
\]

The current `xfact*xjac` plus FKS measure already has the audited magnitude.
Keep that representation for H evaluation if its derivation remains valid
with the corrected Born map. A direct shower-measure implementation is an
alternative representation; do not multiply the two together. No recovery
of Born integration-channel coordinates is needed to evaluate the inverse
Born state or its matrix element.

Recompute local dipole invariants, eligible connections and Born shower-start
bounds from `p_born_mc`. Revisit `compute_shower_scale_nbody`,
`get_dead_zone`, `compute_damping_weight`, cuts and stored S-event SCALUP as
one contract. Preserve the established randomized start-scale distribution:
its averaged survival probability is the smooth damping already used.

Do not let `passcuts_nbody` evaluated only at the FKS Born point discard a
nonzero MC contribution whose shower Born passes the relevant generation
cuts. Test both directions of cut migration. Keep ordinary FKS cuts and
their cancellation intact; define the allowed generation domain explicitly.

Acceptance includes both beams, unequal beam energies and fractions,
arbitrary azimuth, `2 -> 1`, `2 -> 2` and several-spectator systems. Verify
momentum conservation, on-shell masses, spectator internal invariants,
forward/inverse closure, PDF flavours/fractions, and the audit's scattering
invariant `2 p_A.p_3 / s_B = 0.35` rather than `0.2651103574...`.

## 6. Work package C: make the S contribution Born-differentially correct

Validate the S integral together with the ISR H-density repair. A new S
integration is conditional on that validation; different intermediate Born
coordinates alone do not establish that the existing S integral is wrong.
The [S-event projection note](../../MCatNLO_S_event_projection.tex) considers
integrating the same corrected real density under two Born projections.
That comparison must be distinguished from the present S implementation,
which evaluates its Born amplitude at the generated output Born point.

### C0. Check the existing global-recoil factorization first

Global recoil is part of the matching contract. The
[PYTHIA aMC@NLO documentation](https://pythia.org/latest-manual/aMCatNLOMatching.html)
requires global timelike recoil to support the subtraction construction.
The configured global II ISR also admits factorization of the hard-system
phase space. Test the complete product of radiation Jacobian, `xfact`,
real/Born phase-space and flux factors, and luminosity convention before
introducing any extra conversion.

For the massless ISR interior, with `S_R=shat_n1`, `xi=xi_FKS` and
`z=1-xi`, the present code gives

```text
xjac = S_R*xi**2/2
xfact = 4*xi*(1-y)/(z*S_R*N_p)
J_FKS_rad = S_R*xi/(64*pi**3), including the real/Born flux ratio.
```

Consequently

\[
 J_{\rm FKS,rad}\,
 \frac{\mathrm{xfact}\,\mathrm{xjac}}{\xi^2(1-y)}
 =\frac{\mathrm{xjac}}{16\pi^3zN_p}.
\]

Multiplication by `g_s^2*P(z)/t` gives the shower radiation measure,
including the backward `1/z`, expressed in FKS radiation coordinates.
This is an existing conversion, not a missing Jacobian. In the current S
term the amplitude is already `B(b)`, and `get_mc_lum` uses the backward
fractions `(b_A/z,b_B)` for emission from A. Such a fixed-Born construction
does not need a Born-amplitude ratio merely because its auxiliary FKS real
point differs from the shower-generated real point.

The compiled factors reproduce this interior identity on 30 actual global
II emissions from the installed PYTHIA 8.313 to `7.8e-16` relative accuracy;
see [the independent shower check](check_pythia8_global_isr.py). This does
not yet validate the full S integral: derive and compare all radiation
domains, PDF support, generation cuts, partner/flow sums, starting-scale
weights and G/azimuthal completions at fixed `b`. In particular, distinguish
FKS real incoming-fraction bounds from shower mother-fraction bounds.

If that complete fixed-Born identity holds, retain the existing S
construction. Correct the H density at its actual real event separately,
then verify the joint expansion. Global recoil preserves the spectator
measure but does not by itself identify every beam-dependent Born invariant
or PDF fraction in the two inverses at a fixed real point. The actual ISR
emission check still finds differences there.

Use C1a/C1b only if a new common real-density evaluator changes what is
integrated in S, or if C0 establishes a residual requiring correction.
Do not add a projection correction to an S weight already representing the
target fixed-Born shower integral.

### C1. Define the projected raw addition

For the G-weighted raw piece defined from the corrected real MC density,
the desired S addition is

\[
 I_{\rm raw}(b)=\sum_h\int d\mu_R\,
 P_h D_h^{\mathrm{raw},G}(\Phi_R)
 \delta_B\bigl(b,\beta_h(\Phi_R)\bigr).
\]

If this same corrected density is inserted in the present FKS-projected S
assembly, its addition instead contains `delta_B(b,pi_FKS(Phi_R))`.
Changing the common amplitude or `bjx` evaluation can therefore require an
S correction. This is not a proof that the original fixed-Born S integrand
failed C0. Updating H redistribution alone does not change the S projection.

Retain the existing Born generator. The desired radiation integral at fixed
generated Born point `b` can be written with the forward shower map:

\[
 I_{\rm raw}(b)=\sum_h\int dz\,dt\,d\phi\,
 J_h^{\rm PS}(b,z,t,\phi)\,
 P_h D_h^{\mathrm{raw},G}\bigl(F_h^{\rm PS}(b,z,t,\phi)\bigr).
\]

Use the physical measure convention of package B. The Born integration
proposal remains the existing one; its sampling weight is applied once.
`J_PS` replaces the raw S radiation conversion in this expression, rather
than multiplying the existing FKS radiation conversion again.

For H, continue to generate real momenta in FKS coordinates and evaluate the
raw density using the inverse shower map. These are two integrations of
the same density, with the S integral holding the shower parent fixed.

If a new S integration is needed, initially apply it only to ISR. For the
audited global FSR map, the FKS and shower Born projections already agree, so its existing
S integration can be retained after the numerical fixes. Verify that
agreement as a control instead of rewriting the FSR generator.

Keep the FKS G replacements and ordinary NLO counterterms in their original
counterevent maps and measures in the recommended construction below. Moving
those globally to `p_born_mc` would change the local subtraction prescription.
The ledger in section 3 must show the resulting separate projections
explicitly. The displayed raw integrals require regularization; they are
not instructions to integrate a singular contribution on its own.

### C1a. Finite map correction when the S projection must change

The S-event regularization issue is the same as in
[Herwig work package F2](herwig730_implementation_plan.md#f2-fixed-shower-born-integrated-contribution).
For the same corrected real density on both sides, define

\[
 \rho_h^X(b,u)=J_h^X(b,u)\,
 [P_hD_h^{\mathrm{raw},G}]\bigl(F_h^X(b,u)\bigr),
 \qquad X\in\{\mathrm{PS},\mathrm{FKS}\}.
\]

These densities include fluxes, PDFs, history/connection factors, support
and the physical radiation measure. Both maps hold the same external Born
state `b` fixed, but generally produce different real points. In particular,
the raw density evaluated on `F_FKS(b,u)` still uses the corrected inverse
shower Born `beta_h(F_FKS(b,u))` for its amplitude and luminosity; this is
generally different from the output state `b`. Do not compare a corrected
shower kernel with the old defective FKS-based kernel.

Construct the fully FKS-subtracted S weight with this corrected density,
denoted `S_FKS[D_corrected]`. Then implement

\[
 \begin{split}
 \Delta S_{\rm raw}(b)
   &=\sum_h\int du\,[\rho_h^{\rm PS}(b,u)-\rho_h^{\rm FKS}(b,u)],\\
 S_{\rm new}(b)
   &=S_{\rm FKS}[D_{\rm corrected}](b)+\Delta S_{\rm raw}(b).
 \end{split}
\]

The first term retains the FKS cancellation and integrated counterterms;
the second changes the Born projection of the raw MC contribution. Terms
whose projection is retained, including the G replacements in this plan,
do not receive this map correction.

Before calling the difference finite, derive a common radiation chart and
align the leading soft, hard-collinear and overlapping coefficients, with
the full measures, PDF arguments, spin phases and regulator conventions.
For a single endpoint, `rho_X=A_X(b,u)/u` has an integrable difference when
the smooth numerators have the same limit `A_PS(b,0)=A_FKS(b,0)`. For the
double endpoint, equality is required along both singular edges and their
intersection, not just at the soft-collinear corner or after azimuthal
integration. Include finite boundary/domain conversion terms if required
by the alignment. A common random-number input alone is insufficient.

Only the two evaluations of the changed raw piece need identical unresolved
coefficients for this difference. The complete counterterm, including the
retained G and azimuthal completions, must still reproduce the exact
fixed-order limits. Preserve the unregularized low-scale continuation.

Evaluate the two densities at correlated radiation points and combine them
at the same output `b` before folding, absolute weights or unweighting.
Prove local absolute integrability and measure the variance. The audit's
resolved-point checks do not yet establish these endpoint properties.

### C1b. Alternative: direct FKS subtraction at fixed shower Born

It is also possible to regularize the shower-mapped integral directly.
Apply the FKS endpoint operators to its full radiation density, including
Jacobians, PDFs and support. On a normalized soft/collinear chart the
double-subtracted numerator has the structure

```text
A(b,u,v) - A(b,0,v) - A(b,u,0) + A(b,0,0),
```

over `u*v`, with the appropriate single-endpoint contributions, cutoff
functions and boundary logarithms. See
[FKS equations (4.37)-(4.38)](https://arxiv.org/pdf/0908.4272).

If this changes the local counterterms to `C_PS`, derive the finite
conversion of their integrated contribution:

```text
V + I_C^PS = (V + I_C^FKS) + (I_C^PS - I_C^FKS).
```

Equal poles do not prove that the last term vanishes. Include ISR PDF
convolutions and endpoint/domain terms, or demonstrate their cancellation
for the chosen construction. Reusing the existing FKS counterevents with
only a new Jacobian does not establish this result. Apply the section 3
ledger to any completion whose counterterm prescription is changed.

### C2. Preserve cancellation and folding

1. Refactor raw MC evaluation into an evaluator returning a density and
   metadata, plus separate S/H assembly calls. If C0 validates the original
   fixed-Born S construction, retain that S evaluator with the required
   independent kernel fixes. In C1a retain the corrected
   FKS-projected type-12 addition and add only `rho_PS-rho_FKS`. In C1b
   replace that addition with the directly subtracted shower-map integral.
   Adding a full new raw integral to the retained old one would double
   count it. The H path must not add a second G replacement.
2. At fixed `b`, parametrize the new S radiation integral with the existing
   available radiation random variables, including their proposal Jacobian
   and all physical inverse branches. Derive a common endpoint
   parametrization for C1a's difference, or C1b's local FKS subtraction and
   integrated conversion. Do not assume equal random numbers imply equal
   measures.
3. Evaluate G functions on the real point belonging to each term. The raw
   evaluations in `rho_PS` and `rho_FKS` use their respective real points;
   the FKS replacements retain theirs. Prove the matching of unresolved
   coefficients with PDFs, spin phases, coupling factors and cuts included.
4. Sum the canceling S pieces at the same output Born state before taking
   absolute values for MINT/unweighting. Preserve the cancellation for
   every tested radiation fold, not just after averaging independent runs.
5. Test `sum_identical_contributions`, `fill_mint_function_NLOPS`,
   `pick_unweight_contr` and `update_shower_scale_Sevents_v2`. A folded raw
   contribution must retain the starting-scale/colour information associated
   with its actual sampling prescription.
6. Measure integration variance and negative-weight fractions. Keeping the
   total cross section unchanged while making the absolute integrand
   divergent is not an acceptable implementation.

A useful independent diagnostic is the projection-transport contribution

\[
 \Delta\langle O\rangle_{\rm transport}
 =\sum_h\int d\mu_R\,P_hD_h^{\mathrm{raw},G}
 \left[O(\beta_h)-O(\pi_h^{\rm FKS})\right].
\]

With common domains and regulators, this is the correction from the old
projection to the new one. Use it to test the fixed-Born implementation.
Do not implement it by blindly emitting two independent, unweighted S
samples at different Born states: their individual weights can remain
singular even when the observable difference is integrable. C1a instead
constructs the correction at fixed output Born state with locally aligned
endpoints. Test its equality to this observable-level diagnostic.

### C3. Store the momenta actually used

Extend the weight-record interface to accept explicit MC evaluation
momenta/frame information where necessary. Avoid temporarily overwriting
FKS COMMON blocks merely to make `add_wgt` copy different arrays.

| Contribution | Physical output state | Born used for raw MC evaluation | PDF arguments for its coefficient |
|---|---|---|---|
| Raw H subtraction, type 13 | Original physical real event | `beta_h(Phi_R)` | Backward mother/other beam at that history's real fractions |
| Existing fixed-Born raw S term, if validated by C0 | Generated Born `b` | `b` | Backward mother `b_A/z` and other Born fraction for beam A |
| Direct raw S addition, or C1a's `+rho_PS` piece | Fixed generated shower Born `b` | `b` | Backward mother `x_born/z` and the other Born fraction |
| C1a's retained raw FKS S addition and compensating `-rho_FKS` | Same output Born `b` | `beta_h(F_FKS(b,u))` | Backward mother/other beam from that MC Born |
| G replacement S/H pair | Existing FKS counterevent / physical real event | Its exact FKS limiting Born | Existing FKS replacement prescription |
| Ordinary NLO and real redistribution | Existing prescribed states | Existing FKS prescription | Existing NLO prescription |

For C1a, retain each evaluation's amplitude/PDF arguments until its signed
contribution has been combined or reweighted; equal output Born momenta do
not imply equal coefficient metadata. Preserve the corresponding generated
real companions if the reweighting format needs them. For type 13, store
the MC Born used in the amplitude as its Born reweighting companion, while
retaining the actual real momenta for event output. The G-replacement
companion remains FKS-derived.

`bjx` records the fractions used by the luminosity coefficient. These need
not equal the incoming fractions of the S event written to the LHE file:
backward evolution uses the mother PDF while the output Born contains the
daughter. Do not fix the PDF bug by changing the physical S beam momenta to
the PDF mother momenta.

Check allocation/copying in `weight_lines`, momentum deduplication in
`fill_rwgt_lines`, the PDF cache in `include_PDF_and_alphas`, and the LHE
writer/reader. New metadata must survive record cloning, history
permutations, boosts and reweighting. A fresh evaluation and a stored-record
evaluation must agree at changed PDF and scale choices.

### C4. Native-history and scale consistency

`repartition_MC_H` currently transforms native record companions into the
outer frame and retains outer real event momenta. Apply that frame change
to the *explicit* MC Born companion as well; do not reconstruct it later
from `p1_cnt`. Retain native provider/flavour identities for evaluation and
outer ownership for the redistributed H event where the current prescription
requires it.

Preserve the normalization over all labelled histories and the existing
native-to-outer real-measure conversion. Do not combine this work with a
change to the directory/history redistribution proposal. If a generated
process hits a current history-coverage guard, report that as a separate
validation limitation instead of skipping its missing histories.

Reconcile the start-scale sampling of the corrected raw term with the S
event actually passed to PYTHIA, including sampled colour connections. A
check of `get_dead_zone` alone does not verify this. For a controlled Born
state, sample the implemented scale distribution and check that
`Pr(Q_start > sqrt(t))` reproduces `compute_damping_weight`.

If the fixed-Born S integral requires a different radiation parameterization
from the H integral, keep their integration proposal weights separate.
Sharing the physical density does not authorize sharing a proposal Jacobian.
This distinction also applies when stripping a native H integration measure
and applying the outer one in `repartition_MC_H`.

Acceptance: the raw S integral and shower no-emission coefficient agree
differentially in Born rapidity and scattering angle in the scalar
reference setup. Inclusive weights, folding, event kinematics, colour
sampling and reweighting also close. The raw-H-only correction is not ready
for production before these tests pass.

## 7. Work package D: include the gluon recoil weight correctly

### D1. Implement a scalar weight with explicit applicability

Add a helper for the `g -> gg` recoil dead-cone factor in the no-MEC scalar
kernel. Use the active global recoil mass

\[
 M^2=\left(\sum_{a\ne r,e}p_a\right)^2,
 \quad r=\frac{M^2}{S},\quad
 v=\frac{s_p}{S},\quad s_p=\frac{t}{z(1-z)},
\]

\[
 x_1=(1-r+v)z,\qquad x_2=1+r-v,\qquad
 D_{\rm rec}=1-\frac{r}{x_1+x_2-1-r}
                   \frac{1+r-x_2}{1-r-x_1}.
\]

Match the reference source's numerical guards and their accepted-branch
semantics; do not invent broad clamping that hides an out-of-domain map.
Several massless spectators can give nonzero `M`. The local colour partner
mass is therefore not the correct input under global recoil.

Apply the helper only to the appropriate PYTHIA8 scalar gluon channel when
`recoilDeadCone` is enabled and the reference MEC path does not replace this
kernel. Retain the exact helicity correction and the FKS G replacements.
Do not multiply every `xkernazi`, every channel, or the full S/H weight by
this factor.

### D2. Combine identical daughters before changing the AP partition

The current symmetric AP kernel plus `fks_Hij`, connection factors and
history counting cannot be replaced by one labelled PYTHIA kernel without
a normalization derivation. The weighted identity to reproduce is

\[
 P_{\rm end}(z)D_{\rm rec}(z)
 +P_{\rm end}(1-z)D_{\rm rec}(1-z),\qquad
 P_{\rm end}(z)=\frac{C_A}{2}\frac{1+z^3}{1-z}.
\]

Under identical support and an identical underlying Born/connection, a
candidate multiplier for the existing symmetric scalar density is

\[
 W_{\rm sym}(z)=
 \frac{P_{\rm end}(z)D_{\rm rec}(z)
       +P_{\rm end}(1-z)D_{\rm rec}(1-z)}
      {P_{\rm end}(z)+P_{\rm end}(1-z)}.
\]

This is a proposed implementation device, not a new PYTHIA formula. Verify
that the actual `fks_Hij` weights and labelled history sum meet its premises.
Where supports, colour assignments or start-scale weights differ, form the
complete weighted sum with those factors explicitly before repartitioning
it over the existing FKS histories. Do not use `D_rec(z)` alone as a common
multiplier for a symmetric AP kernel, and do not add a second factor of two.

Acceptance tests:

- the ordered audit fixture gives `D_rec=0.7530955643618137` at
  `S=1e6 GeV^2`, `M=400 GeV`, `z=0.8`, `t=15000 GeV^2`;
- the *complete* two-daughter production sum matches the complete weighted
  shower sum, including a massive composite of massless spectators;
- `M -> 0` gives one; the fixed-`z` massless collinear limit preserves the
  AP and helicity singular coefficients;
- the soft G replacement continues to supply the exact wide-angle limit;
- inactive channels/settings give no change; both local-dipole and
  randomized starting-scale boundaries behave consistently.

## 8. Work package E: protect the intentional differences

### E1. Azimuthal terms

Retain `bornbarstilde`, `Qterms_reduced_spacelike`,
`Qterms_reduced_timelike` and `gfactazi`. Test a gluon Born leg with nonzero
helicity interference at several azimuths, both ISR orientations, and an
FSR gluon. A quark channel with vanishing helicity-interference term is an
important control, since PYTHIA's ISR colour-interference bias can still
operate there.

At fixed nonsoft energy fraction, take the collinear limit and compare the
regulated real matrix element with the complete native subtraction before
azimuthal averaging. Also integrate over a full azimuth to verify the zero
mean of the spin-interference contribution with the correct Born state and
measure. Test numerical behaviour near an azimuth where the interference
changes sign.

Use `phiPolAsym`, `phiPolAsymHard` and ISR `phiIntAsym` toggles in diagnostic
shower runs to separate scalar and angular differences. Preserve the chosen
physical shower settings by default. Turning all shower asymmetries off
does not remove the need for the fixed-order spin term in subtraction.

For resolved angular observables, explicitly measure the retained model
difference. If it contributes to a claimed exact first-order observable
coefficient, it needs an additional derived correction; zero angular mean
alone is not sufficient. Such a correction must preserve local fixed-order
limits and is a separate work item, not a replacement of `bornbarstilde`.

### E2. Low scales

Keep the perturbative `dt/t` behaviour and the FKS limiting terms below the
physical shower termination scale. Do not insert a `t > pTmin^2` requirement
or replace `t` by `t+pT0^2` in the subtraction.

Compare to the unregularized perturbative shower reference for the formal
kernel tests. Compare to the finite-cutoff generator separately. For
screening, distinguish its hard-region correction of relative size
`pT0^2/t` from the nonuniform region `t` comparable to `pT0^2`. Sweep shower
cutoffs/screening with hard observable cuts held fixed and test the approach
to the perturbative reference. Do not describe the low-scale difference as
uniformly negligible for observables measured at the cutoff itself.

Use fixed coupling or an explicitly specified perturbative continuation for
the reference below the shower scale; a validation limit should not be
dominated by an unrelated running-coupling singularity.

### E3. G replacements

Preserve the existing default parameters and limiting roles:
`gfactsf -> 0` in the soft region, `gfactcl -> 0` in the collinear region,
and the exact helicity contribution selected by `gfactazi`. The current raw
soft term is switched off for `xi <= 0.01`; that region must be covered by
the replacements, including outside raw angular support.

Check the actual coefficients, not just the function values:

- raw scalar: `gfactsf` times its start damping;
- raw helicity: `gfactsf*gfactazi` times its start damping;
- soft replacement: `(1-gfactsf)*P_h` after the existing connection-averaged
  boundary adjustment;
- collinear and soft-collinear replacements:
  `(1-gfactcl)*(1-gfactsf)*P_h`, with their existing relative overlap sign.

`compute_MCsubtraction_kl` changes the returned `gfactsf` after evaluating
the raw kernels so that the G replacement also damps at the start boundary.
Do not use that returned value retrospectively for the raw term. Make the
two roles explicit in the ledger or use separate local names when refactoring.

Require exactly one replacement per native history. Use its native `P_h`,
FKS limiting amplitudes, luminosities and counterevent measure ratios. H
redistribution acts on the resulting complete native contribution.

Scan soft, collinear and simultaneous limits, including angular regions
where the raw shower has no support and the two massive FKS branches.
Vary G transition parameters within their valid range and measure the
observable effect separately from ordinary numerical subtraction parameters.
Do not require exact independence of arbitrary finite G widths without a
proof, and do not accept uncanceled singularities as a matching variation.

### E4. Heavy flavours

Retain the hard calculation's mass and PDF scheme. A massless five-flavour
real matrix element requires its massless collinear counterterm even when
PYTHIA uses nonzero charm/bottom shower masses. Adding `m_Q^2/t`, forced
threshold conversion or a pair-production threshold to that counterterm
would not be a generic fix.

Provide separate reference fixtures for massless hard/shower limits and for
massive final-state hard particles. Preserve the ordinary no-MEC scalar
`Q -> Qg` kernel identified in the source; do not add a massive AP term just
because the continuing radiator is massive.

Catalogue the residual threshold effects in the massive shower comparison.
A genuinely massive `g -> Q Qbar` implementation requires its own map,
measure, hard singular structure and flavour-scheme derivation. It is not
part of copying the massless-gluon recoil correction in package D. Existing
heavy-flavour processes should not be declared unsupported solely because
their hard and shower mass prescriptions differ.

## 9. Work package F: make settings and the no-emission factor explicit

### F1. Separate physical and diagnostic `P_h`

For ordinary matching, use `P_h=1`. For the physical Delta prescription,
retain `compute_delta` and test its expansion. The sufficient perturbative
property is `Delta=1+O(alpha_s)`; the current coupling-independent debug
function has no such expansion. See [Frederix et al., equations
(3.1)–(3.9)](https://arxiv.org/pdf/2002.12716).

Replace the hidden `data itype/2/` production choice by an explicit
diagnostic option, defaulting to the current mode 3 behaviour when Delta is
off. Keep modes 1/2 available for testing the native-history redistribution.
Route the option through the existing FKS parameter mechanism if it is
intended for users, and record the selected mode in the run information.

For each mode, evaluate `P_h` once per native history and use the same value
for its raw MC, G replacement and real redistribution. Delta must not be
multiplied by the artificial function. Preserve the `(1-P_h) S_h R` S term
in diagnostic runs. Test a point below 0.5 GeV, one between 0.5 and 10 GeV,
and one above 10 GeV.

This default change concerns an artificial debugging weight, separately
from preserving unregularized low-scale subtraction and the G functions.
Report any distribution change caused by it independently of the ISR and
recoil fixes.

### F2. Record and verify the actual shower contract

Have the launch path or a validation driver record the initialized values
that affect the first hard emission:

- PYTHIA version, shower model, global FSR controls and ISR dipole recoil;
- `pTmaxMatch`, `pTmaxFudge`, global/local maximum restrictions and SCALUP;
- recoil dead cone and `weightGluonToQuark`;
- effective hard-system MEC eligibility, including `MEextended` and the
  distinction between the first emission and later emissions;
- azimuthal switches, infrared cutoffs, screening, shower masses and PDFs;
- coupling settings and any nonzero optional finite-kernel parameters.

Make relied-upon defaults such as `SpaceShower:dipoleRecoil=off` explicit.
Preserve azimuthal and low-scale settings in physical showering; scalar
diagnostic settings belong to the validation setup. Version-gate settings
introduced in 8.318 so that 8.313 is not given unknown parameters. Detect
failed setting reads instead of silently accepting a different shower.

For the hard two-mother systems in scope, verify the actual no-MEC path.
Do not use `MEafterFirst=off` as evidence that the first emission has no MEC.
One-mother production/decay systems with active MECs require a separate
prescription or an explicit no-MEC configuration for that comparison.

As a separate follow-up, replace or justify the literal
`nPartonsInBorn=-1` using generated hard-process multiplicity information
and the driver's H/S classification. Mixed Born multiplicities need a
per-process or per-event solution. Validate the first S emission and the
subsequent H recoil selection separately; the latter is not the explanation
for the audited first-order ISR discrepancy.

## 10. Validation programme and acceptance gates

### Gate 1: compiled production routines

Extend the existing numerical test infrastructure rather than validating
only the archived prototype. Suggested new test modules are
`tests/unit_tests/fks/test_pythia8_projection.py` and
`tests/unit_tests/fks/test_pythia8_matching.py`, with small Fortran fixtures
under `tests/input_files/`.

| Test family | Required coverage | Initial acceptance target |
|---|---|---|
| Threshold/sign regressions | Audit fixtures, both massive branches, initialization variants | Correct sign and build-independent values |
| ISR projection | Both beams, asymmetric beams, arbitrary azimuth, several spectators | Interior dimensionless momentum/invariant errors below `1e-9`; tighter at well-conditioned fixtures |
| Measure | Independent Cartesian determinant or phase-space factorization, both flux/PDF conventions | Interior relative error below `1e-8`, with finite-difference convergence checked |
| PDFs | Nonconstant flavour-dependent fixture; `q -> qg`, `g -> gg` and flavour-changing ISR | Correct arguments/flavours and one backward `1/z` factor |
| Recoil weight | Ordered fixture and full permutation/connection sum | Agreement at `1e-9` away from guards |
| State isolation | Alternate FKS/MC Born points, local/native providers and histories | Fresh and cached evaluations agree; traversal order has no effect |
| S/H assembly | Raw, G and real pieces with `P=1`, debug `P`, physical Delta | Correct coefficient ledger and no double counting |
| Record handling | Different MC/FKS Born points, boosts, permutations, cloning, PDF/scale reweighting | Stored and fresh calculations agree |
| Singular limits | Fixed azimuth, soft/collinear paths and their overlap | Leading regulated real/counterterm difference tends to zero |
| S projection correction | Fixed output Born, both ISR beams, nonconstant PDFs, spin phases, all singular edges and the overlap | Locally integrable `rho_PS-rho_FKS`, including any boundary conversion; stable absolute integral and measured variance |
| Global-FSR projection control | Audited global map, matching domains and both physical massive branches | Zero integrated map correction within numerical accuracy after the independent sign fix |

Use mixed absolute/relative tolerances, scaled by the hard invariant or
momentum norm. Endpoint tolerances need the measured conditioning and
expansion error from A1. Do not silently exclude branch boundaries or
threshold regions from the reported coverage.

For massless singular channels, useful diagnostics include the approach of
`xi^2*(1-y)*(R_h-D_h)` to zero, with a nonzero limiting Born normalization.
For massive radiators, test the soft coefficient without imposing a
nonexistent massless collinear divergence. Check the complete regulated
integrand as well as this coefficient: vanishing leading error alone does
not prove finite variance.

### Gate 2: real process amplitudes and exported builds

Use the following small process set, with stable hard particles and
documented generation cuts:

| Process class | What it probes |
|---|---|
| Drell–Yan | ISR fractions, rapidity projection and a quark-spin control |
| Gluon-fusion colour-singlet production with the chosen model specified | Initial gluon helicity terms and the `2 -> 1` convention |
| `pp -> t tbar` | Massive FSR branches, ISR scattering projection and several Born channels |
| `pp -> W + jet` | Gluon radiation, Born cuts, colour flows and asymmetric/flavour-changing ISR |
| A hard final gluon with at least two recoil spectators | Nonzero composite recoil mass even when individual spectators are massless |
| A massive SUSY/QCD kernel control and a non-PYTHIA shower control | Shared-helper regressions |

Compile fresh exports so the new module's export list and Fortran dependency
order are exercised. Existing generated directories do not automatically
pick up template changes. Test both an ordinary local Born provider and a
native-history provider in the shared Born support path.

At selected physical real points, log complete native weights before and
after H repartitioning, including their measure conversion. Test independence
of the outer sampling history after the complete permitted history sum.
Do not require equality of arbitrary individual labelled gluon histories.

For S events, compare the subtracted/integrated raw projected addition and
the reference no-emission loss in Born rapidity and a Born scattering angle,
using common regulators or an explicitly finite difference. Do not histogram
two separately divergent integrals without their cancellations. The audit's
fixed-real-point density, once corrected, must be tested with both possible
S projections to expose any residual. Test the original fixed-Born S
evaluator separately under C0; its use of FKS radiation variables is not a
failure condition. Include a cut boundary crossed by the two Born projections.

For C1a, check agreement with an independent fixed-shower-Born reference and
the transport diagnostic in C2. With identical full real domains and
regulators, the Born integral of the projection correction is zero, while
its Born-rapidity and scattering-angle bins need not vanish. Verify this
inclusive identity before imposing projection-dependent cuts. Compare local
endpoint behaviour before integration and folding, including nonuniform PDFs
and azimuth-sensitive terms. Validate any finite boundary conversion
explicitly. For C1b, also validate the finite integrated-counterterm bridge.

### Gate 3: independent PYTHIA checks

The existing `tests/pythia8_subtraction/test_maps.cc` uses PYTHIA vector
utilities and transcribed forward formulae. It is useful independent
algebra, but it does not call the shower's branching algorithm. Label it
accordingly when reusing it for production projection tests.

Add a version-pinned test driver that observes actual first hard emissions,
recording the pre-emission Born state, emitter/connection, post-emission
state and evolution variables. Use a supported hook where sufficient, or a
small documented test-only source instrumentation if the relevant state is
not exposed. Do not modify the user's installed shower for this purpose.

For each reference version:

1. Verify the reconstruction of recorded first-emission Born states.
2. Isolate scalar weights using documented diagnostic settings and a common
   coupling/PDF prescription. Include MEC and recoil-weight applicability.
3. Compare upper support and the sampled start-scale distribution.
4. Restore physical azimuth, masses, cutoffs and screening, and report their
   deliberately retained differences separately.

A finite-coupling first-emission histogram contains a Sudakov and competing
channels. It is not directly the bare `alpha_s P/t` density. Extract the
first-order coefficient with a controlled coupling scan or compare the
instrumented branching weights and competition explicitly.

### Gate 4: expansion, infrared behaviour and numerical performance

Build an observable-level expansion harness for the actual S/H ledger.
Use `P=1` first, then physical Delta. Scale the emission coupling by a
parameter `lambda`, keep the PDF prescription fixed, and factor out the
Born coupling power consistently. Compare the matched expansion with NLO
for smooth infrared-safe weights and hard binned observables.

For an exactly covered reference component, the remaining difference
relative to Born should start at `lambda^2`. For retained G, angular,
mass-scheme and cutoff effects, isolate the coefficient of `lambda` and
show its claimed cancellation or resolution dependence. A coupling scan
with finite physical cutoffs alone cannot prove the absence of a
cutoff-suppressed first-order term.

Include Born-system rapidity, a hard-system scattering angle, recoil
transverse momentum above a fixed hard cut, and an infrared-safe
azimuth-sensitive jet observable. The total rate (`O=1`) is a useful
normalization test but cannot see the ISR projection error.

Check stability under ordinary FKS subtraction-parameter variations,
radiation folding and sampling changes. Separately scan G transition
parameters and shower infrared parameters as matching/infrared studies.
Do not require an unshowered MC@NLO differential distribution to equal NLO:
the first-order shower contribution belongs in that comparison.

Record convergence, absolute integral, variance, negative-weight fraction
and CPU cost. A change that closes the algebra but destroys the finite
absolute S integrand must be revised before production use.

### Existing regression commands

After the corresponding code changes, run the updated audit and relevant
existing suites, adding the new tests above:

```sh
python3 docs/audits/check_pythia8318.py
python3 -m unittest \
  tests.unit_tests.fks.test_momentum_maps \
  tests.unit_tests.fks.test_mc_kernels \
  tests.unit_tests.fks.test_mc_dead_zones \
  tests.unit_tests.fks.test_soft_col_limits \
  tests.unit_tests.fks.test_mc_history_weights \
  tests.unit_tests.fks.test_mc_event_colours \
  tests.unit_tests.fks.test_pdf_luminosity_cache \
  tests.unit_tests.fks.test_mcatnlo_delta_scales
python3 -m unittest \
  tests.unit_tests.iolibs.test_born_support \
  tests.unit_tests.iolibs.test_export_fks \
  tests.unit_tests.various.test_shower_card
```

The archived prototype tests are supplementary; passing them does not
establish that production calls the corrected routines. Full process and
shower tests also need their process cards, settings, seeds, library
versions and statistical uncertainties recorded with the results.

## 11. Delivery order and definition of completion

| Change set | Deliverable | Dependency / release condition |
|---|---|---|
| 1 | Audit fixtures, explicit settings ledger, threshold fix and endpoint validation | Can be delivered independently |
| 2 | Massive prefactor sign fix and both-branch tests | Independent of ISR work; shared callers checked |
| 3 | Pure II inverse/forward module and production-facing tests | Reconstructed momenta only; no partial activation in event weights |
| 4 | Native Born/spin, flow, PDF and scale evaluation at the MC projection | Depends on 3; validate as an isolated evaluator |
| 5 | Validated fixed-Born S weight: retain the existing construction if C0 passes, otherwise derive the required correction; records, folding and complete ISR activation | Depends on 3/4, the section 3 derivation and any applicable C1 endpoint checks; release the consistent S/H pair together |
| 6 | Recoil dead-cone scalar correction with complete gluon-ordering sum | Can be developed independently; release after singular-limit and normalization tests |
| 7 | Explicit diagnostic no-emission option, physical default and settings checks | Keep its numerical impact separate from changes 5/6 |
| 8 | Process-level expansion/shower validation and updated audit status | Required before claiming the corrected matching prescription is validated |
| Follow-up | H/S multiplicity classification, heavy-threshold or active-MEC extensions | Separate applicability and validation; no silent extension of the first release's scope |

Maintain an audit-to-test checklist. A finding is closed only when its
production path passes the relevant test, or when it is explicitly retained
with the mathematical reason and the tested accuracy domain. For the three
protected ingredients, completion means preserving fixed-order cancellation
and documenting their matching role, not obtaining literal equality to the
configured finite-cutoff shower.

The concrete release requirements are: deterministic endpoint behaviour;
positive scalar massive measures on both branches; correct ISR recoil,
Born/PDF arguments and Born-differential raw S addition; the properly summed
gluon recoil weight; intact exact azimuth/G/infrared limits; consistent event
and reweighting records; and an observable-level expansion report that
distinguishes repaired errors from retained matching effects.

The baseline audit reproducer and the 30-emission PYTHIA 8.313 global-ISR
kinematic/measure/PDF-argument check were executed. Packages A and D now
have production fixes and compiled scalar regressions. The ISR projection
repair was subsequently implemented through the shared mapping changes
noted above; full process-level S/H matching validation remains outstanding.
