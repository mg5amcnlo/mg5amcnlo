# Native radiation projection

`repartition_MC_H` needs the native Born projection and radiation limits at a
fixed real point. It does not need the random numbers that could have generated
the projected Born point in a particular integration channel.

## Measure cancellation

For a fixed native Born point, write the generated real measure as
`B * J_real` and a counterevent measure as `B * J_counter`. Here `B` contains
the Born chart Jacobian, Born phase-space weight and Bjorken sampling factors.
The radiation Jacobians, radiation phase-space factors and flux are retained in
`J`; in particular, the flux can differ between ISR real and counterevents.

The complete generated H weight is linear in these measures. Dividing by its
real measure cancels `B` from both real terms and G replacements, leaving
`J_counter / J_real`. Thus `generate_native_momenta` can set the two input Born
measure factors to one and use the existing radiation generator. The driver's
`xinorm_ev * xi_i_fks_ev` normalization, symmetry-factor cancellation and outer
measure multiplication remain in place. Physical shower Jacobians are unchanged.

`invert_fks_radiation` recovers three radiation coordinates and the projected
Born momenta. ISR also determines the Born Bjorken fractions and longitudinal
boost. Massive FSR retains the native radial/angle coordinates covering both
solutions. The generated real point must agree with the supplied point within
the existing relative tolerance. No Born configuration search, Breit-Wigner
inverse, Born s/t-channel inverse or full phase-space replay remains in the
inner history loop. The saved outer coordinates are still replayed once after
the sum to restore the event owner.

## Global initial-state recoil

The ISR map in `genps_fks.f` is shared by fixed-order integration, all shower
choices, and native history projection. It uses massless incoming partons. For
emitter `j`, define `z = 1 - xi`, where `xi = 2 E_rad / sqrt(s)` and `y` is the
cosine of the radiation angle relative to that emitter in the real partonic CM.
The Born and real fractions obey

```
x_j = bar_x_j / z,       x_spectator = bar_x_spectator,
s = bar_s / z,           xi_max = 1 - bar_x_j,
ycm = ycm_born - idir * log(z) / 2,       idir = 3 - 2*j.
```

The angular domain remains `-1 <= y <= 1`; the endpoint no longer depends on
`y`. A lower bound on the real invariant mass restricts `xi` from below through
`max(0, 1 - tau_born / tau_lower_bound)`. Forward and inverse generation use
the same bounds helper, including the separately stored lepton `1-x` near the
beam endpoint.

All hard final momenta are transformed in the underlying Born CM. With
`p_plus = E + idir*pz`, `p_minus = E - idir*pz`, and two transverse components,
the common Lorentz transformation is

```
A = 1 + xi*(1-y)/(2*z),
b = -xi*sqrt((1-y)*(1+y))/(2*sqrt(z)) * (cos(phi), sin(phi)),

p_plus'  = A*p_plus,
p_T'     = p_T + b*p_plus,
p_minus' = (p_minus + 2*b.p_T + b.b*p_plus)/A.
```

It preserves each hard mass and the spectator's null direction. It reduces to
the identity in both the soft and emitter-collinear limits. The incoming
energies in this frame are `sqrt(bar_s)/(2*z)` for the emitter and
`sqrt(bar_s)/2` for the spectator. The radiation has
`k_plus/xi = sqrt(bar_s)*(1+y)/(2*z)`,
`k_minus/xi = sqrt(bar_s)*(1-y)/2`, and
`k_T/xi = sqrt(s)*sqrt((1-y)*(1+y))/2 * (cos(phi), sin(phi))`.
These energy-divided components remain defined at the soft endpoint.

`boost_isr_recoil` implements this transformation and its algebraic inverse.
This is the massless Pythia initial-initial recoil construction expressed in
light-cone components, including the orientation of the hard final state.
The lepton chart without event projection applies the same recoil and then
boosts to the real CM; it retains its existing sampling of real incoming
fractions and reduced Born mass.

### Analytically integrated FKS terms

Changing the hard Lorentz transformation adds no phase-space determinant.
The change of Bjorken variables has determinant `1/z`, so in four dimensions
the radiation factor multiplying the hadronic Born phase space is

```
s / (4*pi)^3 * xi/z * dxi dy dphi.
```

This is the existing ISR measure. The generator omits `xi` because the FKS
prefactors supply it separately. In `4-2*epsilon` dimensions the same measure
has the factors `s^(1-epsilon)`, `xi^(1-2*epsilon)` and
`(1-y*y)^(-epsilon)`; the new hard recoil does not change them. The soft
momenta and the full collinear map are unchanged, so the existing analytic
soft, collinear and soft-collinear kernels still apply.

The physical endpoint must nevertheless be propagated to the finite terms.
For example, at `h = 1 - bar_x_j`,

```
integral_0^h dxi xi^(-1-2*epsilon)
  = -1/(2*epsilon) + log(h) - epsilon*log(h)^2 + O(epsilon^2).
```

`fill_FKS_commons` installs the new `xiimax` and `xinorm` for the real point
and its counterevents. `compute_prefactors_n1body` already converts the
numerical subtraction range `min(h, xiScut_used)` to the common analytic
cutoff `xicut_used`, including its finite logarithms and mixed terms. Thus
neither an extra recoil Jacobian nor a change to the integrated kernels is
needed. The common analytic cutoffs are retained; replacing them by `h` only
in the integrated terms would change the finite cross section.

## COMMON and saved-state audit

The audit follows the removed inverse/full-generator call chain, the retained
radiation call chain, and consumers in the native evaluator and clustering.

| State | Handling |
| --- | --- |
| `/to_mass/`, `/fks_indices/`, native Born metadata | The existing native activation and process initialization run before each projection. |
| `/pborn/`, `/pborn_l/`, `/pborn_ev/` | All three are filled directly from the same projected Born point. The forward radiation generator reads `/pborn_l/`; Born and limit evaluators use the others. |
| `/ctau_lower_bound/` | All three entries are set to the native physical mass threshold during projection and radiation generation. The incoming values are saved and restored on return. No cut-dependent or topology-dependent sampling threshold is inherited. |
| `/to_ee_omx1/` | Saved, set to zero during the supported hadron/fixed-beam projection, and restored. Otherwise stale lepton endpoint values could alter ISR bounds near a beam endpoint. Dressed lepton beams remain unsupported in this path. |
| `/cnbody/`, `/c_skip_only_event_phsp/` | Saved, set to generate the real point and all applicable limits, and restored on return. |
| `/to_use_evpr/`, `/to_mconfigs/` | Set to event projection and native configuration 1 for the subsequent evaluation. The outer forward generation reinstalls the owner's values. |
| `/counterevnts/` | Momenta and obsolete weights are initialized; inactive momenta and Jacobians remain invalid. Valid momenta and weights come from the radiation generator. Massive second solutions invalidate all counterevents. |
| `/fksvariables/`, `/cxiifkscnt/`, `/cxi_i_hat/`, `/cxiimaxev/`, `/cxiimaxcnt/`, `/cxinormev/`, `/cxinormcnt/` | Filled by the existing radiation generator and `fill_FKS_commons`. Counterevent entries are usable only when their Jacobian is positive; unused entries are not physical data. |
| `/cbjorkenx/`, `/cbjrk12_ev/`, `/cbjrk12_cnt/`, `/parton_cms_ev/`, `/parton_cms_cnt/`, `/pev/` | Refreshed by `fill_FKS_commons` for each valid real/limit point. The Born boost is set even when no counterevent exists. |
| `/parton_cms_stuff/` | The radiation generator resets the frame state; the evaluator's existing `set_cms_stuff` calls select the required real/limit frame. The removed inverse's premature `set_cms_stuff(-100)` call is gone. |
| `/cxij_aor/`, `/cgenps_fks/`, `/virtgranny_boost/` | The radiation generator resets/fills the spin phase and FSR momentum/boost data. Native generation zeros the FSR momentum data before generation. ISR does not write the FSR-only blocks: the driver explicitly saves and restores both of those blocks around the sum, including when the outer owner is ISR. |
| `/cnocntevents/`, `/c_isolsign/` | Computed by the forward radiation generator, including the massive second solution. |
| `/sctests/`, `/cxiyfix/`, `/c_fnlo_nlops/` | Existing run/test controls; read without changing them. |
| Born chart state: `/to_itree/`, `/ciconfig0/`, `/c_qmass_qwidth/`, `/c_vegas_x/`, `/born_trees/`, `/c_conflictingBW/`, `/to_phase_space_s_channel/` | Not read by native radiation projection or its downstream weight evaluation. Native topology tables used by clustering are installed by `mc_sync_native_tables`. The ordinary outer generator still initializes its chart state. |
| Granny chart state: `/c_granny_res/`, `/to_virtgranny/`, `/cgrannyrange/`, `/c_rat_xi/`, `/write_granny_resonance/` | Not required by native radiation generation, which passes `input_granny_m2=.false.`. Their owner values are left to the ordinary outer generator; no inner resonance chart is constructed. |

The retained radiation routines have no explicit mutable `SAVE` cache. Their `DATA`
arrays specify fixed soft/collinear limits and are read-only. The unused
`xiimax_save` and `xjactmp` SAVE declarations were removed. Caches in
`generate_momenta_conf` and `set_tau_min` are no longer visited by inner
histories; the existing native epoch/configuration checks still refresh them
when outer generation resumes. Clustering's topology cache is also invalidated
by the existing native epoch mechanism. The context activation sequence and
`calculatedBorn=.false.` resets are retained.

The driver's existing explicit restoration of colour flow, shower scales,
matching flags, G factors and prefactors remains. The new routine has no saved
local cache; its locals are assigned on each call even with `-fno-automatic`.
It restores its temporary input controls even on a rejected
projection. Rejected projections stop the history evaluation rather than using
partially filled event data.

## Checks

`test_native_projection_and_shared_state` compiles the production routines with
runtime checks. It covers ISR from either beam, massless FSR and both massive
FSR solutions, asymmetric beam energies, both history orders, and deliberately
poisoned COMMON values. It compares Born/real/counterevent momenta, radiation
variables, Bjorken fractions, spin phases, validity flags and measure ratios.
A nonunit reference Born factor checks the cancellation directly. Existing
radiation-map and H-weight regression tests remain in use.

`test_initial_state_recoil_and_endpoints` also compares the production ISR map
with an independent boost/rotation construction for both emitters, massive
hard daughters, asymmetric fractions and beam endpoints. It checks the
inverse, measures, common singular Born projections, real mass thresholds,
and the lepton recoil chart. `test_initial_state_fks_finite_integrals` integrates
analytic test functions through the production phase space and FKS
prefactors, varying both subtraction cutoffs and physical endpoints.

A fixed-order check on 2026-10-01 generated `u u~ > e+ e- [QCD]` with
`loop_sm`, 6.5 TeV proton beams, built-in `nn23nlo` PDFs, fixed renormalization
and factorization scales of 91.188 GeV, and the default `mll_sf = 30 GeV`
cut. Grid setup used 3 iterations of 2,000 points, followed by 3 iterations
of 15,000 points. Scale/PDF reweighting was disabled.

| Map | `xicut` | `deltaI` | Cross section (pb) |
| --- | ---: | ---: | ---: |
| New ISR map | 0.5 | 1.0 | 521.31 +/- 3.93 |
| Original map at `3fff30cc7` | 0.5 | 1.0 | 519.52 +/- 3.86 |
| New ISR map, changed analytic cutoffs | 0.1 | 0.2 | 519.35 +/- 4.47 |

The numerical subtraction cutoffs stayed at `xiScut = 0.5`, `deltaS = 1`.
These results agree within their Monte Carlo uncertainties. The generated
soft/collinear tests passed in all four FKS sectors, and all 20 virtual-pole
checks passed at tolerance `1e-5`. This checks fixed-order subtraction;
consistency with the other shower mappings is deferred.
