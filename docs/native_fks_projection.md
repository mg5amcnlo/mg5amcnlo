# FKS phase-space points and native projection

`repartition_MC_H` needs the native Born projection and radiation limits at a
fixed real point. It does not need the random numbers that could have generated
the projected Born point in a particular integration channel.

## Phase-space source layout

The phase-space stages use explicit Born and complete phase-space points. A
small data module owns the active point read by generation, subtraction and
matching. Sources in `Template/NLO/SubProcesses` are grouped as follows:

| Source and module | Responsibility |
| --- | --- |
| `fks_phase_space_data.f` — `fks_phase_space_data` | Active Born, real and counterevent data, radiation variables, frame and recoil information, and sampling bounds. Its only dependency is `nexternal.inc`. |
| `genps_fks.f` — `fks_phase_space` | Point types, public generation stages, native projection, and private orchestration, flux and counterevent bookkeeping. |
| `genps_fks_sampling.f` — `fks_born_sampling` | Private Born chart, invariant masses, t-channel sampling, incoming fractions and lepton-beam sampling. |
| `genps_fks_radiation.f` — `fks_radiation_maps` | Public forward and inverse FSR/ISR kernels, including the coupled map without event projection. |
| `genps_fks_helpers.f` — `fks_phase_space_helpers` | Shared kinematic functions and massive final-state radiation bounds. |

The generation modules contain their procedures, so callers obtain their
interfaces from the compiler. The main module keeps implementation routines
private. Its public API includes the staging routines, `generate_momenta`,
`generate_native_momenta` and
`invert_fks_radiation`.

`reset_fks_kinematics` applies the same invalid-state sentinels before both
radiation-generation paths. `get_massive_fsr_bounds` supplies the common bounds
and normalization for the massive forward map and its final/initial-recoil
inverses, including the stable massless-recoil expressions. Local recoil
wrappers remain in `resonance_recoil.f` and `initial_recoil.f`.
They import the radiation and helper modules without depending on the main
orchestration module. Sampling and radiation both depend on the helpers; the
main module uses the sampling, radiation and helper modules.

`initialize_fks_phase_space` installs the Born sampling chart once through
`initialize_born_chart`. The tree, branch counts and one-body flag are private
sampling-module state, replacing `/born_trees/`; individual point generation
owns its mass and invariant-mass work arrays. Sampling no longer repeats
chart initialization for each point.
Initialization takes only the channel index. The unused random-coordinate
COMMON and duplicate cached channel index have been removed; coordinates
are passed to sampling and radiation directly.

Generation and evaluation routines import the active fields from
`fks_phase_space_data` with `USE ... ONLY`. This replaces the corresponding
COMMON declarations in every reader and writer, including the recoil wrappers,
kinematics and subtraction modules, event writers, reweighting and standalone
EW Sudakov routines. Types, dimensions and storage therefore have one owner.
Recoil defaults are initialized in the data module instead of BLOCK DATA.
The data module has no dependency on the generation or evaluation modules,
so these imports introduce no circular dependency.

`fks_phase_space_point` remains the value used to save and restore a complete
point. The data module still represents one active point; this migration does
not make the calculation reentrant. Run controls and process metadata retain
their existing owners. The lepton endpoint block `/to_ee_omx1/` also remains a
COMMON because the separately compiled PDF library shares its storage; point
capture and restoration continue to include those values. `/to_use_evpr/`
and `/cnbody/` remain run-context COMMONs.

The exporter links all these sources into each subprocess and
`makefile_fks_dir` builds them through `GENPS_OBJECTS`, with the data module
built before every importer. The event-reweighting, standalone EW and Utilities
builds also include the data module. Regenerate existing process outputs to
use this source layout.

## Phase-space API

Callers use `fks_phase_space` for the point types and public interfaces.
`fks_born_point` replaces the name `fks_born_state`: it represents a particular
sampled or projected Born point, which can be reused for several radiation
trials. `fks_phase_space_point` represents the result of one such trial,
including its real configuration and subtraction endpoints. Keeping the Born
input as a nested member preserves its independent lifetime, including when a
radiation trial is rejected.

| Type or member | Contents and meaning |
| --- | --- |
| `fks_born_point` | External Born momenta and masses, incoming fractions and boosts, collider and partonic invariants, sampling bounds, lepton endpoint data, and the Born sampling Jacobian and phase-space weight. Sampling provenance records the integration channel, sector, coordinates, beams, masses and native-context epoch. |
| `fks_phase_space_configuration` | One real or counterevent configuration: momenta, validity, weighted Jacobian, beam fractions and boosts, partonic invariants, radiation variables and bounds, and the soft-limit momentum direction. |
| `point%born` | The reusable sampled or projected Born input. Its `xjac` and `xpswgt` exclude radiation factors, flux and the caller weight. Native projected points have unit Born measures and no Born-chart provenance. |
| `point%event`, `point%counterevent(-2:2)` | The real configuration and the FKS soft, collinear and soft-collinear endpoints. Each configuration has its own validity flag. |
| `point%p`, `point%p_lab`, `point%p_cms`, `point%weight` | The momenta and final integration weight returned by the generation stage. `generate_born_event` returns the plain Born measure carried by slot `0` and has no real configuration. |
| `point%radiation_coordinates`, `point%has_radiation`, `point%nbody_only` | The three sampled radiation coordinates and the generation mode. A plain Born event has no radiation coordinates; Born contributions retain the FKS endpoint generation mode. |
| Born projections | The Born momenta used by the legacy evaluators, plus the distinct collinear and reduced Born systems needed without event projection, with explicit availability flags. |
| Frame and recoil data | Spin phase, FSR energies and solution sign, the active beam/CM frame, resonance recoil momentum and membership, initial recoiler, and owning sector and channel. |

Configuration momenta use the generation frame. With event projection this is
the underlying Born CM, which can differ from the real or counterevent CM.
`point%p_lab` follows the existing API's symmetric hadron frame; consumers
apply the extra longitudinal boost for unequal beam energies. The stored
soft-limit direction is `p_i/xi`, and the sampled energy coordinate `xi_hat`
retains its nonzero sampled value at a soft endpoint. These fields therefore
cannot in general be reconstructed from the endpoint momenta alone.

Valid configuration Jacobians include the Born sampling measure, radiation
Jacobian and phase-space factors, flux and caller weight, following the legacy
normalization. The explicit FKS factor `xi_i_fks` remains handled by the
downstream integration weights. They are distinct from the unweighted Born
input measures and from matrix-element or PDF weights.

| Entry point | Responsibility |
| --- | --- |
| `initialize_fks_phase_space(iconfig)` | Install the selected integration channel, Born chart and recoil context. |
| `sample_fks_born_point(ndim, x, born, pass)` | Sample the incoming fractions and Born phase space, returning the complete Born point. |
| `generate_fks_radiation(born, rad, nbody_only, wgt, point, pass)` | Generate radiation and the applicable FKS counterevents from an immutable Born input and three radiation coordinates. |
| `generate_born_contribution(ndim, iconfig, wgt, x, point)` | Generate the Born contribution with its existing endpoint normalization. |
| `generate_born_event(ndim, iconfig, wgt, x, point)` | Generate padded Born momenta and the plain Born measure for S-event output and mass reshuffling. |
| `generate_real_phase_space(ndim, iconfig, wgt, x, point)` | Generate a real point and its subtraction counterevents, reusing `point%born` when its provenance matches. |
| `generate_prepared_fks_point(ndim, x, nbody_only, wgt, point)` | Generate from an initialized channel, reusing `point%born` when its sampling provenance matches. |
| `capture_fks_phase_space(point)` | Capture the active generated data into an existing point. Generation wrappers also record the returned momenta, weight and Born input. |
| `restore_fks_phase_space(point)` | Restore the active generated data for a previously saved point after reinstating its owner and initializing its channel. |

The fixed-order and matched drivers call the contribution entry points. Native
projection constructs the same Born point from the inverse radiation map and
calls `generate_fks_radiation` directly. This shared radiation step installs the
active Born data, handles radiation and counterevent weights, and supplies the
working, lab and partonic-CM momenta. Native projection therefore uses the same
bookkeeping as ordinary generation. The compatibility entry
`generate_momenta` retains its array arguments; `generate_native_momenta`
retains its array arguments and can additionally return a complete point.

The generation stages pass `nbody_only` explicitly through both radiation
paths without temporarily changing `/cnbody/`. The compatibility entry
`generate_momenta` reads that caller-owned flag to retain its existing
behavior. The private projected-radiation kernel takes only the three
radiation coordinates and owns its mass workspace; it no longer receives
a full integration-coordinate array, dimension count or caller workspace.
The bookkeeping helpers also omit arguments they did not use.

Beam preparation copies only available beam coordinates into its two-entry
sampling input. For a hadronic process with a single Born final-state particle,
the mass fixes tau and rapidity uses `x(1)`; no unused `x(0)` is accessed.
The full phase-space input must contain between 3 and 99 coordinates.

The inactive `wgt_cnt` and `pswgt_cnt` arrays and their snapshot copies have
been removed; the actual counterevent measures remain in `jac_cnt` and
`point%counterevent%jacobian`. The fixed-order driver also omits local lab/CM
momentum copies that it never consumed. The complete point still provides
those frames to callers that need them.

S-event writing and MC-mass reshuffling use `generate_born_event`. It pads the
Born momenta with a zero emitted leg and fills counterevent slot `0` with the
plain Born measure, including flux and the caller weight. It carries no FKS
radiation or endpoint normalization. H-event output and `ickkw=4`, which may
also write an H record, retain full phase-space generation.
For a plain Born event, the real configuration is invalid and has a negative
Jacobian sentinel; `point%weight` and the slot `0` Jacobian carry the accepted
Born measure.

Driver reuse requires the same sector and integration channel, matching Born
coordinates, beam settings, external masses, sampling bounds, recoil choice,
test controls and native-context epoch. A changed radiation coordinate alone
does not require resampling. Matching Born flavours across different sectors
is insufficient because their cuts and resonance sampling can differ. A
provenance mismatch makes the contribution wrapper sample a fresh Born point.

Sampling and radiation routines return `pass` for the caller to check. A
rejected generation invalidates its output, event and counterevent slots with
negative sentinels. A valid Born input remains reusable after a failed
radiation trial. The shared radiation API rejects invalid Born points and
points from the coupled path without event projection. Real validity and
endpoint validity are independent: a soft contribution can be available
without a valid real event, and a massive second solution can have a real
event without counterevents. Callers must check the corresponding validity
flag before reading momenta or weights; unused slots are not samples.

For dressed leptons without event projection, `sample_fks_born_point` can
return the soft Born point needed for S-event output. Its
`event_projection=.false.` flag prevents adding radiation through the shared
API: that incoming endpoint parametrization needs a different reduced Born
system for each radiation endpoint. It retains combined generation for those
contributions. The Born contribution also retains the existing FKS endpoint
normalization; separating the public stages does not remove those
radiation-coordinate factors.
At the exact soft endpoint the coupled map copies the unboosted Born momenta
without attempting to infer a radiation direction from zero momentum divided
by zero energy.
It keeps the Born sampling masses separate from the real-leg masses used for
momentum checks and flux. Inserting an emitted leg before the final position
therefore cannot shift the masses used to sample the next counterevent.

## Measure cancellation

For a fixed native Born point, write the generated real measure as
`B * J_real` and a counterevent measure as `B * J_counter`. Here `B` contains
the Born chart Jacobian, Born phase-space weight and Bjorken sampling factors.
The radiation Jacobians, radiation phase-space factors and flux are retained in
`J`; in particular, the flux can differ between ISR real and counterevents.

The complete generated H weight is linear in these measures. Dividing by its
real measure cancels `B` from both real terms and G replacements, leaving
`J_counter / J_real`. Thus `generate_native_momenta` sets `born%xjac` and
`born%xpswgt` to one and passes that point to `generate_fks_radiation`. The driver's
`xinorm_ev * xi_i_fks_ev` normalization, symmetry-factor cancellation and outer
measure multiplication remain in place. Physical shower Jacobians are unchanged.

`invert_fks_radiation` recovers three radiation coordinates and the projected
Born momenta. ISR also determines the Born Bjorken fractions and longitudinal
boost. Massive FSR retains the native radial/angle coordinates covering both
solutions. The generated real point must agree with the supplied point within
the existing relative tolerance. No Born configuration search, Breit-Wigner
inverse, Born s/t-channel inverse or full phase-space replay remains in the
inner history loop. After the sum, the driver restores the outer owner,
initializes its channel, and restores the saved complete phase-space point.
This restores the active event/counterevent data, radiation variables, spin
and frame data without replaying generation. The snapshot also contains the
additional Born projections used by the coupled dressed-lepton path.

Native generation restores the incoming sampling bounds and lepton endpoint
values before returning. Its optional complete point records those restored
values in `point%bounds` and `point%omx`, matching the active state at return.
The generation inputs remain independently available in `point%born%bounds`
and `point%born%omx` for another radiation trial.

The driver refreshes the snapshot from the active data immediately before
the inner histories. This captures changes made by the outer subtraction
evaluation, including its FSR energy data, while retaining the original
generated momenta and weighted measure.

The point snapshots generated phase-space data. Run settings, cuts, integration
topology tables, matrix elements, PDFs and matching weights retain their
existing owners. Restoring a point therefore requires selecting its original
process/sector and initializing its channel first; it is not a substitute for
those context operations. Restoration checks the sector, FKS emitter/sister
indices and channel against the saved point.


## Global initial-state recoil

The default ISR map in `genps_fks_radiation.f` is shared by fixed-order
integration, all shower choices, and native history projection. It uses massless incoming partons. For
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

`boost_isr_recoil` in `genps_fks_helpers.f` implements this transformation
and its algebraic inverse.
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

`fill_fks_point_data` installs the new `xiimax` and `xinorm` for the real point
and its counterevents. `compute_prefactors_n1body` already converts the
numerical subtraction range `min(h, xiScut_used)` to the common analytic
cutoff `xicut_used`, including its finite logarithms and mixed terms. Thus
neither an extra recoil Jacobian nor a change to the integrated kernels is
needed. The common analytic cutoffs are retained; replacing them by `h` only
in the integrated terms would change the finite cross section.

## Active data and saved-state audit

The audit follows the removed inverse/full-generator call chain, the retained
radiation call chain, and consumers in the native evaluator and clustering.
The table retains the old block names to identify the corresponding fields.
All generated point blocks listed below now live in `fks_phase_space_data`,
except the PDF bridge `/to_ee_omx1/`. Run controls and the remaining shared
chart context retain their existing storage; the former `/born_trees/`
fields are now private sampling-module data.

| Former point block or retained context block | Handling |
| --- | --- |
| `/to_mass/`, `/fks_indices/`, native Born metadata | The existing native activation and process initialization run before each projection. |
| `/pborn/`, `/pborn_l/`, `/pborn_ev/` | The shared radiation step fills all three from the supplied Born point. The forward radiation kernel reads `p_born_l`; Born and limit evaluators use the others. |
| `/pborn_coll/`, `/pborn_norad/` | The coupled map fills these additional Born projections. The complete point records their availability and restores their values. |
| `/ctau_lower_bound/` | All three entries are set to the native physical mass threshold during projection and radiation generation. The incoming values are saved and restored on return. No cut-dependent or topology-dependent sampling threshold is inherited. |
| `/to_ee_omx1/` | Saved, set to zero during the supported hadron/fixed-beam projection, and restored. Otherwise stale lepton endpoint values could alter ISR bounds near a beam endpoint. Dressed lepton beams remain unsupported in this path. |
| `/cnbody/` | Caller-owned contribution mode. The compatibility entry reads it; staged and native generation pass `nbody_only` explicitly without writing the COMMON. |
| `/to_use_evpr/`, `/to_mconfigs/` | Set to event projection and native configuration 1 for the subsequent evaluation. Outer channel initialization and point restoration restore the owner's values. |
| `/counterevnts/` | Inactive momenta and Jacobians remain invalid. The unused legacy weight arrays have been removed. Valid momenta and weights come from the radiation generator. Massive second solutions invalidate all counterevents. |
| `/fksvariables/`, `/cxiifkscnt/`, `/cxi_i_hat/`, `/cxiimaxev/`, `/cxiimaxcnt/`, `/cxinormev/`, `/cxinormcnt/` | Filled by the existing radiation generator and `fill_fks_point_data`. Counterevent entries are usable only when their Jacobian is positive; unused entries are not physical data. |
| `/cbjorkenx/`, `/cbjrk12_ev/`, `/cbjrk12_cnt/`, `/parton_cms_ev/`, `/parton_cms_cnt/`, `/pev/` | Refreshed by `fill_fks_point_data` for each valid real/limit point. The Born boost is set even when no counterevent exists. |
| `/parton_cms_stuff/` | The radiation generator resets the frame state; the evaluator's existing `set_cms_stuff` calls select the required real/limit frame. The removed inverse's premature `set_cms_stuff(-100)` call is gone. |
| `/cxij_aor/`, `/cgenps_fks/` | The shared radiation step resets/fills the spin phase and FSR energies. The complete point snapshot restores these together, including when the outer owner is ISR. |
| `/c_resonance_recoil/`, `/c_initial_recoil/` | The snapshot restores the resonance momentum, mass, membership and recoil flag, together with the selected initial recoiler. |
| `/cnocntevents/`, `/c_isolsign/` | Computed by the forward radiation generator, including the massive second solution. |
| `/sctests/`, `/cxiyfix/`, `/c_fnlo_nlops/` | Existing run/test controls; read without changing them. |
| Born chart state: `/to_itree/`, `/c_qmass_qwidth/`, private sampling chart, `/c_conflictingBW/`, `/to_phase_space_s_channel/` | Not read by native radiation projection or its downstream weight evaluation. Native topology tables used by clustering are installed by `mc_sync_native_tables`. The ordinary outer generator initializes its chart through `initialize_born_chart`; the former `/born_trees/` COMMON is removed. |
| Granny chart state: `/c_granny_res/`, `/to_virtgranny/`, `/cgrannyrange/`, `/c_rat_xi/`, `/write_granny_resonance/` | Not required by native radiation generation. Outer channel initialization owns the resonance chart; inner histories supply projected Born data directly. |

The retained radiation routines have no explicit mutable `SAVE` cache. Their
`DATA` arrays specify fixed soft/collinear limits and are read-only. The unused
`xiimax_save` and `xjactmp` SAVE declarations were removed. Born preparation
recomputes collider energy and mass sums from the current inputs; the former
`generate_momenta_conf` cache is gone. `set_tau_min` retains its existing cache
with native epoch and configuration checks and is not called by inner radiation
histories. Clustering's topology cache also follows the native epoch. The
context activation sequence and `calculatedBorn=.false.` resets are retained.

The driver's existing explicit restoration of colour flow, shower scales,
matching flags, G factors and prefactors remains. The new routine has no saved
local cache; its locals are assigned on each call even with `-fno-automatic`.
It restores its temporary input controls even on a rejected
projection. Rejected projections stop the history evaluation rather than using
partially filled event data.

## Checks

Validation of the phase-space API, module, complete-point and active-data
refactors is limited to build and source checks. The fixed-order and matched
executables compile and link for a generated `u d > u d [QCD]` process using
`make -j4`. The symmetry, soft/collinear-limit, pole-check and event-reweighting
executables also compile and link. The analysis, Sudakov-check, alternative
Binoth interface, Utilities `setcuts` and standalone EW dummy readers compile.
The Utilities check uses the generated subprocess module interfaces; its
existing dependencies on `mint_module` and `mc_native_context` remain.
The review also compiled and linked nine existing Fortran fixtures and the
PYTHIA audit against the current modules, without running their numerical
programs. An additional compiler pass reported no warnings about unused module
imports, unused dummy arguments, uninitialized values or array bounds in
`genps_fks.f`.
Python syntax checks pass for the edited Python files. A source audit found
no remaining active declarations of the migrated COMMON blocks in tracked
sources and confirmed matching field types, shapes and import aliases.
The existing numerical suites described below were not run for these refactors.

`test_native_projection_and_shared_state` compiles the production routines with
runtime checks. It covers ISR from either beam, massless FSR and both massive
FSR solutions, asymmetric beam energies, both history orders, and deliberately
poisoned active values. It compares Born/real/counterevent momenta, radiation
variables, Bjorken fractions, spin phases, validity flags and measure ratios.
A nonunit reference Born factor checks the cancellation directly. Existing
radiation-map and H-weight regression tests remain in use.

The focused harnesses compile the production data, helper and radiation modules.
They select the required main-module procedures into a minimal module while
retaining the production point types and data imports. Their drivers use
module imports, and the PYTHIA audit uses the same extraction support. This
keeps their interfaces tied to the production definitions without requiring
generated Born topology tables.

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
