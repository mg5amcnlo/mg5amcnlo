# MAcNLOPS from the generated measures

## Sampled fixed-flow matching: 25 October implementation

Fresh MAcNLOPS sources can now use `MCExplicitKLSum=False` and
`MCSubtractionAtFixedFlow=True`, with `FKSExplicitSum=False` to sample
the outer FKS contribution groups. The settings are read from
`Cards/FKS_params.dat`; the MAcNLOPS driver no longer overrides them with
an explicit history/colour sum. `FKSExplicitSum` is independent: disabling
it samples a group that may include associated nonsingular sectors, not
necessarily just one numerical term. The alternative globally summed
mode (`MCExplicitKLSum=True`, `MCSubtractionAtFixedFlow=False`) remains
available for the qualified old source prescription. Mixed combinations
are rejected. All large-sample figures below describe that **old global
prescription**, not the new sampled prescription.

The component label is a native sector/history, a Born integration
configuration and diagram, and a Born colour flow. A versioned
`macnlops_component` event packet stores those labels, an immutable
generated-registry fingerprint, the flow-sampling probability, source
insertion variate and damping, and the diagram symmetry factor. H rows
are grouped within this component before unweighting and sign selection;
the source writer certifies the complete physical flavour group. S rows
carry the diagram and flow, but their source integration sector does not
restrict which leg may shower. S grouping checks that its contributing
rows have the same Born kinematics and incoming fractions.

For an actual S emission, the saved Pythia ancestry identifies the Born
radiator, emitted particle and spectator identities. The density route
stays in the exact source Born provider. Identical Born-particle
permutations require a certified transformation of both the diagram and
the oriented colour topology; merely copying their numeric indices is
wrong. Configured interaction trees and colour flows are included in the
registry for this purpose. Ambiguous gluon daughter orientations are
marked with probabilities proportional to their physical kernels. The
native inverse Born state is checked against the actual pre-shuffle S
state, including flavours, colours, masses, Lorentz invariants and
incoming fractions. No Pythia library change is used.

Write `a` for the resulting component and `lambda` for its source scale
and colour-insertion auxiliaries. Let `D_G,a` denote its full G-completed
counterterm, including its physical diagram and flow fractions. Its
undamped hard density is `H_0,a = R_a - D_G,a`. For a Delta source,

\[
 H_{\Delta,a}=\Delta_a H_{0,a},\qquad
 u_a=[-H_{0,a}]_+,\qquad
 v_{S,a}=\min(u_a/K_a,1),\qquad
 a_{H^-,a}=\max(1-K_a/u_a,0).
\]

The surviving negative-H measure is therefore
`Delta_a [u_a-K_a]_+`. These formulas have their continuous operational
limits: no negative numerator means no S veto; at `K_a=0` the S event is
kept and the negative H is retained. Strict positivity remains an optional
diagnostic and fails when a residual is necessary. The nominal source
weights are never divided by Delta or by the flow-sampling probability.
The latter is an importance-sampling factor in generation, not an extra
factor in the physical local density.

The V3 local evaluator keeps Delta source algebra, replays **unit** Delta
for the selected component, and consequently evaluates `H_0,a` without
computing a numerical source Sudakov at the S endpoint. It restores the
source's distinct real and Born diagram fractions and retains both
colour connections of a fixed gluon flow. Ordinary MC@NLO uses the same
local machinery with its genuine bogus-Sudakov `P_a` retained:
the transferred numerator is `[P_a(D_G,a-R_a)]_+`. Replacing that
kinematic profile by one is not an ordinary-source optimization.

Real/Born identical-particle factors also belong to the measure. After
removing the reduced source's outer `F=fac_i fac_j`, the native MC term
still contains `I_B/I_R`. The local evaluator multiplies **all** of
`R,D,K` by `I_R/I_B` to express densities per actual Born emitter.
The reduced source represents `O=F I_B/I_R` equivalent emitters;
the worker checks that `O` is a positive integer. This common conversion
does not change the local veto or H acceptance. It is not a correction
to K alone.

Sampling makes the required Delta factorization meaningful before
taking a negative part; it does not establish exact finite cancellation.
In a positive one-component description, let the actual eligible S
emission density be `beta_a Delta_PS,a K_a`. The difference introduced by
the bounded pair is

\[
 \delta\rho_a=
  [\overline\Delta_a-\beta_a\Delta_{{\rm PS},a}]\min(u_a,K_a),\qquad
 \overline\Delta_a=\mathbb E_{\lambda\mid q,a}[\Delta_a(\lambda)].
\]

The average is over the source auxiliary law before H unweighting. The
algorithm does not evaluate it: the undamped local numerator and kernel
are independent of those auxiliaries.

It vanishes under equality of the source and physical shower factors and
`beta_a=1`. Neither equality is assumed by the implementation. Signed S
coefficients, differing starting-scale laws, shower-kernel correspondence
and subsequent continuations remain relevant. The literal
`(1-R_a/K_a)_+` follows in the special case `D_G,a=K_a`; more generally
the removable negative parts must agree. The native G/spin completion
need not satisfy that condition.

The output LHE header and JSON reports distinguish
`component-undamped-delta-v1`, `component-source-v1` and
`global-source-v1`; the collector rejects mixed prescriptions. Local
evaluation avoids the global history sum and Delta calculation. Immutable
diagram/flow transports are cached, while every event retains a fresh
kinematic density evaluation. Existing checksum and global-route caches
are described below. The earlier global formulas and production evidence
in this note retain their original scope.

## Runtime status: stock Pythia after 21 October 2026

The active runtime now uses the selected Pythia installation without the
MAcNLOPS-specific trial-headroom, certified-PDF-envelope or four-flavour
coupling patches. There is no custom trial-headroom argument, bound-provider
library or MAcNLOPS patch-builder prerequisite. An unrelated existing
global QED patch is outside this change. Here "stock" refers specifically
to removal of the MAcNLOPS modifications.

Incoming and gluon-splitting flavour counts remain four or five according
to the hard/PDF profile. The effective stock `StandardModel:alphaSnfmax`
is **five in both cases**. Consequently the hard W+bb calculation can
still use massive bottoms, an nf4 PDF and four allowed shower flavours,
but its stock shower coupling is not the patched nf4 evolution used in
the frozen study. At a common reference coupling the beta-function
difference starts one power higher in alpha_s; it nevertheless changes
the resummed running and finite predictions. This is not a newly proved
fully consistent four-flavour shower profile.

Ordinary native trial-bound warnings are recorded without rejecting,
dropping or retrying the source event. First-emission metadata records the
warnings from that matching step. This uses the native sampling policy;
it does not certify a universal trial envelope or
identify the accepted native density exactly with the analytic denominator.
Removing a reciprocally compensated trial factor is sampling-neutral only
while its overestimate remains valid. If native overweight acceptance is
clipped, agreement of the leading accepted density with `K_g` must not be
assumed, nor can that discrepancy automatically be assigned to higher
orders. The NLO cancellation argument below remains conditional on that
leading correspondence.

The completed 20 October productions, plots and paper used the earlier
patched runtimes and a now-retired driver that saved first/full endpoints
within one live Pythia evolution. Its driver and production runners are
no longer part of the implementation. Their results and qualifications
remain historical evidence for those frozen configurations, not stock-runtime results.
Sections explicitly marked historical below preserve that record. New
stock runs require freshly measured profiles and matching source/runtime
provenance; old patched checkpoints are not stock qualifications. No new
simulation or audit was performed for this documentation change.

This note derives the rejection for the counterterm and shower that are
actually used. The distinction between the subtraction density `D` and the
accepted shower branching density `K` is essential. The result is

\[
 a(q)=1-\frac{H_-(q)}{K(q)},\qquad
 H_-(q)=[-H(q)]_+,
 \qquad 0\leq H_-(q)\leq K(q).
\]

Here `H` is the **complete generated hard-event density**, after combining
its real, MC, G and P contributions in the partition in which hard events
were generated. The new source construction below changes those
contributions consistently in both streams; it cannot be applied only
to the rejection denominator of an old sample. `K` is the sum of the physical shower histories at the
same real phase-space point, including the distribution of shower starting
scales. It is not a Sudakov-weighted density, a trial-emission envelope,
or a signed colour-correlated counterterm. The familiar `min(1,R/K)` is
the special case `H=R-K`.

The requested endpoint is an ensemble containing the retained original
hard events and Born-like events after zero or one Pythia QCD emission,
followed by rejection. A complete subsequent shower is a separate stage.

The current requested default is to postprocess **one unmodified ordinary
MC@NLO LHE sample**, used by both the control and the rejection branch.
Its ordinary source physics is retained. Operationally the current runner
requires an already qualified annotated sample, measured profile, source
manifest and matching evaluator/card context.
The resolved `D*` source discussed later is a distinct, earlier construction;
adding the artificial no-emission factor to that source would not make it
ordinary MC@NLO. The implemented importer requires a qualified versioned
hard-point annotation and a certified global or sampled source partition. It fails
explicitly when those are absent; it does not retrofit anonymous stock
files. A zero-cancellation fallback is mathematically possible but is not
an implemented generic-import promise. Historical validation results
concern the source and numerical
implementation named in each result, not automatic qualification of a
generic ordinary LHE importer.

## Primary references and scope of the comparison

Nason and Salam formulate event rejection using a common `R_s` for the
subtraction and shower. Their Eq. (3.1) gives acceptance `min(1,R/R_s)`;
their Sec. 4 separately discusses rejection of a branching followed by
continued evolution, which changes the integrated Born contribution.
The finite-cutoff formula also needs the no-emission contribution displayed
in their Eq. (2.3). Their introductory simplification suppresses multiple
singular regions. [Nason and Salam, arXiv:2111.03553](https://arxiv.org/abs/2111.03553).

Che's thesis gives the Born no-emission term explicitly and distinguishes
the inverse-FKS Born point from the original shower Born point. Its
Sec. 4.3 recommends disabling the then-used `UseSudakov` option and explains
a one-ISR implementation for colour-singlet production. Its blanket
positivity claim needs an additional condition on the Born-like density.
[Che, *Implementation of Novel NLO Matching Schemes for pp → ZZ*,
Secs. 3.4, 4.2 and 4.3](https://lup.lub.lu.se/student-papers/record/9229458/file/9229466.pdf).

The related implementation paper explicitly acknowledges residual negative
S weights and the unbalanced removal of H events below the shower cutoff.
It is a colour-singlet ISR study, so its numerical validation does not
prove the same identification of counterterm and shower for arbitrary
coloured processes. [Che and Frederix, arXiv:2605.22108](https://arxiv.org/abs/2605.22108).

The algebra below is independent of those identifications. The existing
local review
`/export/tmp/rikkert/git/macnlops/macnlops_review.tex` also distinguishes
`D` and `K`; it is useful background, rather than a replacement for
checking the generated event and shower contracts.

## Common ordinary input: equality before rejection

### Annotated hard-point presentation and existing stock files

The versioned handoff can attach the pre-MC-reshuffle external hard point
to an otherwise ordinary generated event:

```xml
<macnlops_hard_event version="1" frame="lab" source="ordinary" partition="global_sum">
NUP KIND OWNER NATIVE_SECTOR NATIVE_PROVIDER NATIVE_CONTEXT NATIVE_HISTORY BORN_FLOW MC_KL_SUM FIXED_FLOW FKS_SUM FLAVOUR_COALESCED MAX_BCOL GROUP_SIZE
... NUP ordinary thirteen-field particle rows for the hard external state ...
</macnlops_hard_event>
```

The payload contains numerical values for the fourteen metadata fields,
followed by the external particles in their hard order, with 17-digit
momenta and hard-model masses. `KIND=1` denotes S and `KIND=2` denotes H.
The frame is the original lab frame and the payload contains no auxiliary
resonance annotations. The standard LHE event body, nominal and alternative
weights, shower scales and ordinary G/P source calculation remain their
ordinary values. The additional block records information that the stock
mass reshuffle would otherwise discard.

When present and qualified, the native interface uses this hard point as
the common LHA presentation for both the unvetoed control and the rejection
branch. Their first H endpoint uses the same hard presentation. The
original input event remains provenance; neither branch independently
projects it or changes its weight. This defines the same presentation in
both branches. It is not a claim that it is byte-identical to the native
import of the stock remassed body alone.

`partition="global_sum"` is a claim that must agree with the recorded
source flags and actual contribution grouping before H unweighting.
Necessary flags are `MC_KL_SUM=1`, `FIXED_FLOW=0`,
`FLAVOUR_COALESCED=1` and a positive group size. The generated
`MC_HIST_FLAVOUR_COMPLETE` certificate additionally checks every mapped
native full PDG-row multiset against its outer owner and requires unique
owner PDG rows. Partial overlaps, extra native rows and duplicate physical
owner rows cannot certify a global sign. A dynamic H-group check must also
confirm that all compatible records carrying the selected flavour belong
to the chosen unweighting group. The FKS integration sampling switch can
be false: the separate complete native-history H sum is still required.
Specifically, `repartition_MC_H` checks `MC_HIST_COMPLETE`, enumerates all
native histories and recoil groups, and converts each complete native
weight to the same outer real measure. With fixed-flow sampling disabled,
each history enumerates its Born colours before the complete records are
grouped. The remaining outer factor is a nonnegative FKS partition times
a positive sampling measure. It cannot change the global H sign. Keeping
the outer event colour and shower scale fixed throughout the inner sum
allows those records to be combined. The flavour certificate and dynamic
group check are needed in addition to this algebra; a positive group size
alone does not prove completeness.

`partition="sampled"`, missing metadata, or an unknown version does not
establish a global density. A sampled colour/sector partition requires its
own negative-measure reconstruction. An existing stock file without the
block is not made qualified by silently assuming hard-model masses or
global summation. The current importer rejects missing annotations and
requires the additional qualified component packet for sampled input.
A future zero-cancellation fallback could retain
all signed events there, while a nonzero veto would need the separate
stock-transport and source-partition proof below. New annotated
generation is an information-preserving handoff, not evidence of generic
support for arbitrary older LHE files.

The Python event parser retains this block as an opaque tag. Normal and
weight-only roundtrips, central/alternative weight rescaling and reweight
serialization preserve its numeric tokens and attributes; ordinary parser
serialization normalizes line whitespace. Both memory and external Python
collectors copy complete event blocks byte for byte. The block must follow
the particle rows and contain no comment lines beginning with `#` inside
its XML payload. Momentum-changing or model-changing operations require
an explicit provenance update or invalidation. Fortran collection and
native parsing require their own checks; Python preservation alone does
not qualify those paths.

### Ordinary densities and optional controlled generation

The ordinary source keeps its complete native MC subtraction, including
the G completion, spin-dependent contributions and each history's
artificial no-emission factor. Denote that complete subtraction by
`D_G,h`, its nonnegative real partition by `R_h`, and the artificial factor
by `p_b,h`. In the non-Delta path the existing `bogus_probne_fun` uses
mode 2: zero through 0.5 GeV, a smooth transition, and one from 10 GeV.
The complete hard density is

\[
 H_{\rm ord}(q)=\sum_h p_{b,h}(q)
                    [R_h(q)-D_{G,h}(q)],\qquad
 D_{\rm in}(q)=R(q)-H_{\rm ord}(q).
\]

The ordinary S calculation contains the paired integrated `D_in`, with
its existing FKS subtraction and factorization terms. In particular,
`p_b` must occur consistently in the real H/S split, MC terms and G
replacements. It is not an extra factor to insert only into a post-shower
acceptance denominator. Different inverse histories can have different
`p_b,h`; form the complete H sum before taking its negative part.

For an optional controlled fresh generation, the run-card API separates
source construction from matching workflow preparation:

```text
True = macnlops
False = macnlops_resolved_source
False = mcatnlo_delta
```

`macnlops=True` prepares the measured matching profile and complete source
coalescing. It requires `store_hard_momenta=True`; disabling that annotation
is rejected during run-card validation, before generation. The annotation
records the momenta before shower-mass reshuffling;
the ordinary event body still uses its stock MC-mass reshuffle. The default
`macnlops_resolved_source=False` keeps the ordinary G/P counterterm and
records profile construction `ordinary-mcatnlo-v1`. Setting
`macnlops_resolved_source=True` explicitly selects the earlier resolved
counterterm and profile construction `resolved-convex-v1`. These are
different source definitions and must not share an unqualified source
manifest. These generation flags are not prerequisites for processing an
existing ordinary LHE, and do not convert it after generation.

The previous resolved source instead has

\[
 H^*(q)=\sum_h g_h(q)[R_h(q)-K_{0,h}(q)],
\]

where `g_h` is the physical history gate and `K_0,h` is the scalar physical
kernel. Adding the bogus factor consistently would replace `g_h` by
`p_b,h*g_h` in both source streams. It would still differ from the ordinary
source through the G completion, scalar kernel, physical support and
history gate. A paired change of source counterterm leaves the inclusive
signed S+H integral unchanged; that fact does not make the separate streams
or their showered distributions identical.

To obtain the requested identity, both branches read the same ordinary
LHE and use its exact same events, signed weights, scales, colours and
presentation. Use the same saved native first/full proposals for their
comparison. Before rejection their estimators are then equal event by
event, including their statistical fluctuations. There is no separately
sampled reference normalization. Neither fresh hard generation nor the
resolved source is part of this identity requirement.

An existing input can carry the stock write-time MC-mass
reshuffle: for example their incoming light quarks carry 0.33 or 0.50 GeV
and their massless-model bottom partons can carry 4.80 GeV. That routine
regenerates phase space using the original integration coordinates and
diagram before applying special FKS mass corrections. Evaluating a
massless hard matrix element directly at those written points does not
reconstruct its generating density. Projecting an old event to different
momenta would also defeat exact input identity. An internal pullback must
instead be validated as the actual generator transport inverse, including
its density Jacobian and relation to Pythia's LHA import. Until that is
established the affected negative H contributions must remain. A fresh
ordinary source with a declared common hard-shell writer is an optional
controlled validation case; it does not establish support for stock files.

### MC@NLO-Delta source and its stochastic density contract

The Delta-enabled workflow uses

```text
True = macnlops
False = macnlops_resolved_source
True = mcatnlo_delta
True = store_hard_momenta
```

As for ordinary MAcNLOPS, set `parton_shower=PYTHIA8`, `ickkw=0`,
`event_norm=sum` and `pdlabel=lhapdf`, choose one central PDF member, and
disable scale/PDF reweighting. The first-emission driver currently requires
stock Pythia 8.318. The campaign prepared the summed-history, summed-colour
source (`MCExplicitKLSum=True`, `MCSubtractionAtFixedFlow=False`). The
new sampled fixed-flow mode is described at the start of this note.
The global evaluator still requires equal source-owner auxiliary laws;
the sampled evaluator uses the exact source provider and component instead.

Its construction label is `mcatnlo-delta-v1` and the hard annotation has
`source="delta"`. The unrejected comparison must be labelled **MC@NLO-Delta**
and use that same source and native proposals. It is not the ordinary
bogus-P baseline, nor the legacy resolved source. Delta and
`macnlops_resolved_source=True` must not be combined. An existing Delta
file without the required source-state replay information is not qualified
by relabelling it or drawing replacement source scales.

The normal `macnlops.py` and launch workflows derive this mode from the
run card and generation profile and pass `--mcatnlo-delta` automatically.
The native first-emission driver also accepts that flag explicitly.
The collected output remains
`Events/<run>/macnlops/events.lhe.gz`. Older Delta files lacking the pinned
history registry and replay packet require fresh generation. A derived
endpoint retains the source scales as **inert provenance** and has separate
current continuation scales. It cannot be imported as a fresh matching
source or have its source Delta computed again at the showered point.
Subsequent showering consumes this retained LHE endpoint in a separate
stage, as described under “Continuing the retained LHE endpoint” below.

For a fresh source generated with these flags and `noshower`, the normal
process-directory interface performs the two stages separately:

```text
shower run_01
shower run_01/macnlops
```

The first command generates at most one emission for each S event, applies
H removal and S veto, and writes the single randomized
`Events/run_01/macnlops/events.lhe.gz`. If the initial launch already ran
that stage, start with the second command. The second command showers the
accepted file using its current production and dipole scales, with
hadronisation, MPI, primordial transverse momentum and QED radiation off.
It does not generate another matching proposal or repeat removal/veto.
If `macnlops` already exists, first-stage reruns use `macnlops_2`, etc.; use
the output directory reported by that run for continuation.

Lightweight regression fixtures exercise the paired native Delta source
terms and the Python workflow with controlled native/PDF substitutes.
The bounded native and process-level checks of 23 October are described
below. They test source reconstruction and the separate shower handoff;
precision distribution comparisons remain outstanding.

For one history, suppressing its auxiliary state momentarily, the paired
source terms are

\[
 H_\Delta=\Delta_{\rm src}(R-D_G),\qquad
 D_{{\rm in},\Delta}=\Delta_{\rm src}D_G+(1-\Delta_{\rm src})R.
\]

The S coefficient retains the integrated second expression, together with
its existing regulated virtual, FKS and factorization terms. Multiplying
only old H weights by Delta would omit the compensating S change. The
computed Delta replaces the artificial bogus factor; the two are not
multiplied. This is the paired construction of Eqs. (3.2)-(3.3) of
[Frederix et al., *On the reduction of negative weights in MC@NLO-type
matching procedures*, JHEP 07 (2020) 238](https://arxiv.org/abs/2002.12716).

The native density protocol keeps its `R,D,K` field names. In Delta mode
its returned `D` is defined by `D_out=R-H_Delta`; for one history this is
`(1-Delta_src)R+Delta_src D_G`, rather than the ordinary `D_G` or just
`Delta_src D_G`. Hence `D_out-R` reconstructs the required negative-H
test with the existing convention, while `K_g` is unchanged.

The source Delta is a product of leg probabilities, with reconstructed
colour-line stopping scales, starting scales and ISR PDF factors. It is
not assumed to equal the actual shower's global no-harder-emission
probability at one winning evolution scale. The existing Pythia8 source
path samples one scalar starting scale per native history/recoil and
derives its all-flow directed matrix from that scale. The matrix is shared
across the explicit Born-flow sum; independently sampling its entries or
each flow's Delta would define a different auxiliary law.

Denote all relevant source draws by `lambda`. The complete conditional H
density is

\[
 H_\Delta(q,\lambda)=\sum_h\Delta_h(q,\lambda)
                         [R_h(q)-D_{G,h}(q)].
\]

Histories and flows are summed **before** taking the negative part, while
the source auxiliary variables are integrated **after** it:

\[
 H_{-,\Delta}(q,\lambda)=[-H_\Delta(q,\lambda)]_+,
 \qquad
 E_\lambda H_{-,\Delta}\ne[-E_\lambda H_\Delta]_+
 \quad\hbox{in general}.
\]

Consequently an ordinary deterministic global-H evaluator is insufficient.
Each H record must carry the actual keyed
`(history,recoil,flow,probne)` values used in its source sum, together with
the owning context and required colour-insertion provenance. Replaying
those values reconstructs its conditional source density. A different
new starting-scale draw cannot be used to classify that record's sign.
Contribution groups with different replay packets must not be silently
coalesced into one packet. Complete flavour/history coverage alone does
not certify marginalization over these extra random variables.

For an emitted S point, sample a **new independent** auxiliary state from
the same conditional source-H law `mu(lambda|q)`. It is a mathematical
matching auxiliary, not the observed source SCALUP or an extra shower
emission. Reusing the actual supporting shower start biases the draw:
observing an emission has already conditioned that start on supporting q.
Use a dedicated reproducible auxiliary key; R and `K_g` remain deterministic
at fixed q, while conditional H also depends on that key. A density/cache
identity must therefore include the replay or auxiliary state.

The bounded method remains

\[
 C_\lambda(q)=\min\{H_{-,\Delta}(q,\lambda),K_g(q)\},\qquad
 a_{H^-}=1-C_\lambda/H_{-,\Delta},\qquad
 a_{S,a}=1-g_a C_\lambda/K_g.
\]

Zero-denominator cases keep the corresponding event, as before. `K_g`
retains the physical starting-scale marginal; it is not conditioned on
the new fictitious source auxiliary and receives neither source Delta nor
the bogus factor. Negative H residuals remain where required. The source
gate of the actual S history is still applied; protected and no-emission
S records remain unchanged.

If an outer source owner/channel alpha has a positive partition
`w_alpha(q)` and a distinct auxiliary law `mu_alpha`, the cancellation
density is more generally

\[
 \bar C(q)=\sum_\alpha w_\alpha(q)
       \int\mu_\alpha(d\lambda\mid q)\,C_{\alpha,\lambda}(q).
\]

The S auxiliary sampler must draw that source mixture, or the implementation
must establish equivalence of the joint laws under its native-history
label maps. Using one arbitrary owner's law is not justified merely by
an equal mean H. The shared matrix across flows, per-history/recoil draws,
source partition and any relevant insertion correlations must agree.

The current evaluator checks this condition conservatively before choosing
an S route. For every compatible complete original owner it compares the
full native label maps in the common event order, native context and
sector identities, process restrictions and coupling orders, and the
native-context semantics. Missing or duplicate native draws and different
owner signatures are rejected; the latter requires an explicit
source-owner mixture sampler, which is not implemented. For equal
signatures, native projection, recoil groups and flow basis agree, and
only the order of independent scalar-Q draws can differ. One insertion
uniform remains shared across the whole draw. This check may reject
equivalent identical-spectator permutations rather than assume their
equivalence. The report records the resulting certificate digests.

Let `F_S,g^Delta` be the signed density of eligible first-S endpoints from
the actual Delta source and native shower. The ensemble change is

\[
 \delta\sigma=\int d\mu_R\,\bar C(q)
             [1-F_{S,g}^\Delta(q)/K_g(q)].
\]

Thus this extension enables the **same bounded leading-density matching**;
it is not an exact finite-coupling-density veto. Its formal NLO argument
still requires leading native-density/map/support correspondence, and
full-shower observables still involve their separate H/S continuations.
In the single-history idealization `D_G=K`, exact complete removal would
instead require

\[
 v_{\rm exact}=\frac{(K-R)_+}{K}\,
       \frac{B}{\bar B_\Delta}\,
       \frac{\Delta_{\rm src}}{\Delta_{\rm PS}}.
\]

Equal source and shower Delta factors would remove the inverse-Sudakov
factor in that ideal expression. The implemented bounded `H_minus/K_g`
rule does not make this identification or claim exact rate recovery.

#### 25 October audit: implemented versus undamped Delta veto

The original Nason–Salam acceptance uses the same `K` in the shower and
in `H=R-K`, and rejects an emitted S event with
`v_0=(K-R)_+/K`. Its physical shower Sudakov already multiplies the
emission density; it is not included again in that multiplier.
[Nason and Salam, Eq. (3.1)](https://arxiv.org/abs/2111.03553).
For a Delta source this differs from the implemented conditional rule.
The distinction is meaningful even when both prescriptions have the
same fixed-order expansion.

In the one-history idealization, let `D_G=K`, `u=(K-R)_+`,
`beta=barB_Delta/B`, and use common support. Then

\[
 H_{-,\Delta}=\Delta_{\rm src}u,\qquad
 F_{S,g}^\Delta=\beta\Delta_{\rm PS}K,\qquad
 v_{\rm impl}=\Delta_{\rm src}u/K,\qquad v_0=u/K.
\]

With complete H removal, their endpoint differences from their common
unrejected Delta source are

\[
 \delta\rho_{\rm impl}
   =\Delta_{\rm src}u(1-\beta\Delta_{\rm PS}),\qquad
 \delta\rho_0=u(\Delta_{\rm src}-\beta\Delta_{\rm PS}).
\]

If `Delta_src=Delta_PS` and `beta=1`, the undamped multiplier cancels
exactly while the implemented rule rejects too little S density. In
this limit the implemented rule introduces an extra source-Sudakov
suppression relative to the original multiplier. This can be a large
finite or resummed difference. It is not, by itself, an NLO error:
`Delta_src=1+O(alpha_s)` makes the prescriptions differ beyond NLO at
fixed resolved kinematics, provided their required leading native
kernel correspondence holds. The Delta paper explicitly distinguishes
its fixed-order equivalence from comparable showered results.
[Frederix et al., Eqs. (3.6)–(3.9) and the following discussion](https://arxiv.org/abs/2002.12716).

The H cap must also match the chosen undamped prescription. More
generally, suppose a **common** source factor can be extracted,
`H_minus_Delta=Delta_src*u`, where `u=(D_G-R)_+`, and the actual eligible
S density is `F_S,g^Delta=Delta_src*K_g`. A bounded pair is then

\[
 v_0=\min(u/K_g,1),\qquad
 a_{H^-}^{(0)}=\max(1-K_g/u,0),\qquad
 H_{-,\mathrm{rem}}^{(0)}=\Delta_{\rm src}(u-K_g)_+.
\]

These ratios apply on positive denominators; with no eligible S support
the transfer is zero. They transfer `Delta_src*min(u,K_g)` exactly under
the stated assumptions. Keeping the implemented residual
`(Delta_src*u-K_g)_+` while changing only the S veto would not implement
this bounded pair.

The literal global replacement `v_0=(K_g-R)_+/K_g` is **not adopted**
for the actual general source. The complete `D_G` includes G/spin
completion, native measures and support and is not identical to `K_g`.
Assuming leading native correspondence and each
`Delta_h=1+O(alpha_s)`, replacing only the S veto while keeping the
current H cap gives the leading difference

\[
 \delta\rho_{\rm literal}
   =\min\{(D_G-R)_+,K_g\}-(K_g-R)_+
      +\mathcal O(\alpha_s^{p+2}).
\]

For complete H removal the first term is simply `(D_G-R)_+`.
Equality of the removable negative parts at leading order is required;
otherwise the substitution can affect NLO. All quantities must use
the same physical measure and eligibility support, including the
protected-heavy-flavour gates.

Furthermore, history-dependent source factors cannot generally be
extracted before the global negative part. Even with `D_G,h=K_h`
termwise, the finite example `R_h=(2,6)`, `K_h=(6,1)` and
`Delta_h=(1,0.2)` gives `H_Delta=-3`, whereas total `R=8` and `K_g=7`
give the proposed global veto zero. This is a counterexample to finite
factorization, not a demonstration of a failed fixed-order expansion.

An alternative could use the fully summed undamped completed density
`H_0=sum_h(R_h-D_G,h)` in the S numerator, retaining its G completion.
This could avoid source-Delta evaluation at S emissions, but requires
its own H-removal/residual rule and treatment of conditional signs and
continuation. That global alternative remains unimplemented. The new
component-local prescription at the start of this note retains the labels
before sign selection instead. Exact finite
cancellation would still require the actual eligible S density.
The measured distributions and frozen results use their documented
implemented global rules; none validates the new sampled prescription.

Generation provenance must bind the actual exported `sudakov.f`
interpolation and `MCmasses_PYTHIA8.inc` inputs as well as the cards,
native-history registry and evaluator. Stored H probabilities alone would
not detect changed tables in the new S-auxiliary draws. Source/replay
metadata must survive event collection and randomization. The ordinary-only
source-density cache/gate is not implicitly broadened to Delta inputs.

The lightweight checks in
`tests/unit_tests/various/test_macnlops_delta.py` use exact rational finite
measures. They check paired S compensation, negative-part ordering,
bounded residuals, independent S auxiliaries, owner mixtures and the
distinction between first-point and full-shower equality. They perform no
source generation or showering and do not validate Delta tables, generic
stock input, native sampling or a new Delta production. The frozen paper
and plots remain the earlier non-Delta study.

### Which negative density is actually removed

Negative-part formation is nonlinear. If a generator independently
unweights components `j` and samples auxiliary choices `lambda`, deleting
every negative record adds the density

\[
 C_{\rm input}(q)=\sum_j\int d\lambda\,p_j(\lambda\mid q)
                          [-H_j(q,\lambda)]_+.
\]

In general this is larger than
`[-sum_j integral p_j H_j]_+`. Thus a complete flow-averaged evaluator
cannot classify separately signed sector or sampled-flow records by its
global sign. LHE colour tags need not recover the missing choices: the
native H repartition can draw a different Born flow for every inverse
history while preserving only the outer event colour in the output.

An optional controlled source can combine its intended physical
flavour/history/flow contributions before the absolute value used for H
unweighting. Native
flavour rows must be matched by their actual PDGs, not their table row
number. When this is done for both branches, the negative density is
`H_ord,-=[-H_ord]_+`. If event colours, starting scales or another retained
label still define separately signed H components, that label must remain
in the evaluated density, or its negative parts must be summed explicitly.
An equality of the signed mean alone is insufficient. Existing LHE files
retain their generating partition; postprocessing must reconstruct that
partition's negative measure rather than assume new coalescing rules.
If this cannot be done for a component, choose zero cancellation for that
component and leave both its H and S contributions unchanged. A global
S veto paired with an uncertain negative-H removal is not consistent.

Even a choice `C=[-H_total]_+` requires the **input** negative measure in
the H acceptance: negative records must be kept with probability
`1-C/C_input`, while the S removal targets `C`. Replacing that denominator
by `[-H_total]_+` would delete all negative records and add `C_input`, which
can be larger than the S cancellation. If `C_input` is unknown, the only
certified choice from the signed mean alone is `C=0` in that physical
region. This means retaining every H record and every S proposal there,
not dropping an inconvenient event or applying an unmatched S veto.

### Stock colourless mass transport: conditions for an exact pullback

For an ISR-only stable-ZZ real process, the stock MC-mass setup leaves all
masses passed to phase-space regeneration unchanged: incoming QCD masses
and the FKS emitted mass are set to zero for generation, and the Z masses
are unchanged. With the same coordinates and Born diagram, the regenerated
hard point is therefore identical before the special ISR mass transport.
For DY the daughter lepton masses also change. A two-body s-channel Born
chart preserves its invariant mass and rest-frame directions only if its
sampling bounds and branches are unchanged. Actual forward logs and
roundtrips must check this condition rather than infer it from the tiny
electron mass.

In the real CM, the subsequent ISR routine preserves the emitted three-
momentum `k` and the entire hard recoil four-vector, replaces its emitted
energy by `sqrt(k^2+m_MC^2)`, and reconstructs the incoming pair with the
MC masses. Its total real-CM rapidity is preserved. Given a verified
unchanged preceding regeneration, undoing this step means restoring the
hard emitted energy and reconstructing the hard incoming pair at that
rapidity. This is a conditional inverse of the actual transport, not a
license to project arbitrary LHE events onto a massless shell.

For equal-mass lepton daughters at fixed pair momentum `P`, the candidate
map is covariantly

\[
 \ell_\pm'=P/2+\beta_e(\ell_\pm-P/2),\qquad
 \beta_e=\sqrt{1-4m_e^2/P^2}.
\]

It commutes with a common ISR Lorentz recoil, and its two-body phase-space
factor is `beta_e` in both Born and real measures. That factor cancels in
`H/K` only if numerator and shower use the same transported reference
measure, or Pythia import demonstrably undoes both transports. A numerator
roundtrip by itself does not prove this shower-measure condition.

There is no corresponding blanket cancellation for a massive emitted
sister. As an explicit reference measure, integrate massless beam fractions
and flux, and use coordinates `(M_recoil^2,Y,k_vector)`. Apart from common
constants its real measure is

\[
 d\mu_R\ \propto
 \frac{dM^2\,dY\,d^3k\,d\Phi_B(M)}{W E_k E_R},\qquad
 W=E_k+E_R.
\]

At fixed recoil mass and `k`, changing an emitted hard mass zero to `m`
therefore gives the forward measure ratio
`J_R=beta_e*W_0*|k|/(W_m*sqrt(k^2+m^2))`, whereas the Born lepton ratio is
`J_B=beta_e`. Their ratio contains an additional radiation factor. The
chosen measure here is explicit; massive-incoming flux conventions require
their own conversion. Any native massive-sister proposal must include its
actual inverse and Jacobian. If written H and actual S endpoints instead
lie on disjoint mass shells, they cannot be treated as one density merely
because their observable values are numerically close.

### Ordinary input does not guarantee complete negative-H removal

The ordinary signed counterterm is not the positive scalar shower kernel.
Consequently the identity `H_ord,- <= K` does not follow merely from adding
the bogus factor. The factor may suppress a troublesome region without
proving a probability bound there. Source support can also extend below a
physical shower cutoff or a competing heavy-threshold conversion.

A diagnostic ordinary-source reconstruction gives a concrete example:
the saved W+bb first-step point 517 in chunk 0, with ISR evolution scale
5.96239 GeV, has `R=3.00565582936e10`,
`D_in=8.06870718113e10` and `K=4.57939037499e10` in the common evaluator
units. Its `H_ord,-/K=1.10561689`. The reconstruction uses the complete
ordinary G/P source, summed Born flows and a stable real-amplitude chart;
it leaves the saved point unchanged. This is a probability-bound
counterexample, not a generated common-source production result. Its
diagnostic record is
`/export/tmp/rikkert/macnlops_plots_20261020/analysis/cancellation_kernel_audit/ordinary_bound_chunk5000_wbb/report.json`.

Let `g_h` select only histories whose ordinary leading first-emission
density is represented by the evaluated physical kernel, and write

\[
 K_g(q)=\sum_h g_h(q)K_{0,h}(q).
\]

Here the gate affects the cancellation, not the ordinary source. For a
finite cancellation density `0 <= C <= min(H_ord,-,K_g)`, keep positive H,
keep a negative H record with probability `1-C/H_ord,-`, and accept an
emitted S record from actual history `a` with probability

\[
 a_S(q,a)=1-g_a(q)\frac{C(q)}{K_g(q)}.
\]

Use acceptance one for S when `K_g=0`. At leading matched order the removed
S density is `sum_h g_h K_0,h*C/K_g=C`, which compensates the removed
negative H density. Taking `C=H_ord,-` gives complete H removal only where
its envelope holds. Taking `C=min(H_ord,-,K_g)` retains a signed negative-H
residual where it does not. Deleting every negative H while merely capping
the S veto would not implement this construction.

The selected common ordinary-source workflow uses this signed-residual
choice explicitly. It therefore reduces negative H weights while retaining
the part required by the probability bound. It does not promise removal of
every negative H event. Strict complete removal remains a separate mode
that must stop on a violated bound.

For the exact finite-coupling comparison, denote the signed eligible
first-S density, including its actual source weights, starting-scale
mixture and no-harder-emission probability, by `F_S,g`. Then

\[
 \delta\sigma=\int C(q)
                  \left[1-\frac{F_{S,g}(q)}{K_g(q)}\right]d\mu_R.
\]

Thus exact shared input does not imply an exactly unchanged rate after
rejection. NLO equivalence requires the leading-density expansion and
finiteness conditions; numerical equality at finite shower coupling would
require additional information about `F_S,g`. A large observed discrepancy
requires diagnosis rather than an assumption that it is harmless.

### Numerical real-amplitude stability is a separate requirement

The saved first-S points exposed a leading numerical error in the
massless-fermion HELAS chart. For a negative-z incoming fermion, roundoff
can give simultaneously tiny transverse components and a small positive
`E+pz`. The generic spinor formula then uses `sqrt(E+pz)` and
`(px+i*py)/sqrt(E+pz)`. Its norm can collapse instead of approaching the
physical `2E`, making the real matrix element spuriously near zero and
the veto almost certain. This is a numerical failure of the leading real
density, not a perturbative higher-order effect.

A common proper spatial rotation of every matrix-element input vector
provides a stable chart without changing the physical point. Native maps,
PDF arguments, cuts, scales, shower densities and written momenta remain
in their original frame; only the amplitude evaluation uses the rotated
vectors. Two independent proper rotations must agree, regular source-H
points must remain invariant, and the reconstructed complete ordinary H
must reproduce the actual generating sign and normalization. The earlier
diagnostic rotations restore the affected amplitudes, but they do not by
themselves qualify the new ordinary-source mode or explain every remaining
finite-coupling S/H imbalance.

The generated real chooser implements the correction with local `Q_ME`.
For `N` external momenta it tests `N+1` equally spaced possible z axes in
the original xy plane and chooses the axis maximizing the smallest
`1+pz'/|p|`. At least one candidate is an angle `pi/(N+1)` from every
negative momentum direction. For a physical massless leg this bounds the
spinor denominator away from zero by
`E+pz' >= E*(1-cos(pi/(N+1)))`, up to floating-point error. The chosen
orthogonal matrix has determinant one. Polarization-restricted sectors
retain the caller's basis.

`tests/unit_tests/iolibs/test_real_me_basis.py` independently compiles fresh
DY and ZZ real amplitudes at stored problematic, regular and signed-H
points. It reproduces the original suppression, compares two other proper
rotations, checks input bit patterns and pairwise invariants, verifies
cache restoration, and preserves a true zero after couplings are set to
zero. A compiled balanced `2 -> 6` geometry occupies all six Cartesian
axes and checks that the adaptive basis avoids moving a different leg
onto the pole, preserving determinant one, shells and momentum closure.
Those checks validate the numerical fix; massive final-state and complete
ordinary-source validations remain separate requirements.

The exact identity of the two before-rejection branches is separate from
the perturbative accuracy of the common baseline. In particular the
artificial `p_b` is fixed rather than a perturbative Sudakov tending to
one with the coupling. Its finite transition region and the ordinary
shower cutoff retain the baseline's low-scale matching prescription;
sharing the input does not eliminate those qualifications.

## Definitions with ordinary MSbar PDFs

Choose a positive physical real-emission measure `dμ_R(q)` and a Born
measure `dμ_B(b)`. Write

\[
 B(b)=f_a^{\overline{\rm MS}}(x_a,\mu_F)
       f_b^{\overline{\rm MS}}(x_b,\mu_F)
       \frac{|\mathcal M_B(b;\mu_R)|^2}{\text{flux}}
\]

with spin/colour averages and symmetry factors included consistently.
`R(q)` is the analogous physical tree-level real density. Renormalized
virtual contributions and MSbar mass-factorization remnants are denoted
collectively by `V`. For one common Born projection define

\[
 \bar B_D(b)=B(b)+V(b)+\int D(q)\,d\phi,
 \qquad d\mu_R=d\mu_B\,d\phi.
\]

The integral and `V` use the same regulator/subtraction prescription;
neither is a standalone convergent four-dimensional integral. In
production the finite S density is assembled by the existing FKS
subtraction routines, including their collinear convolutions. There is
no need for a Krk PDF transformation or removal of the MSbar remnants.

For one shower history with radiation variables `r`, let

\[
 d\mu_R(q)=J_h(q)\,d\mu_B(b_h)\,dr_h,
 \qquad dP_h=\kappa_h(r_h\mid b_h)\,dr_h.
\]

The accepted, Sudakov-free shower density in the physical measure is

\[
 K(q)=\sum_{h:q\in\mathrm{Im}(M_h)}
       \frac{B(b_h(q))\,\kappa_h(r_h(q)\mid b_h(q))}{J_h(q)}.
\]

Sum over inverse branches, flavours, colour probabilities, competing
radiators and randomized initialization when they contribute to the same
physical real state. An individual signed colour-correlated dipole is
not a shower probability. For ISR, `κ` contains the backwards-evolution
PDF ratio, schematically

\[
 dP_{b\leftarrow a}=\frac{dt}{t}\frac{\alpha_s(\mu_{R,\rm PS})}{2\pi}
 P_{a\to bc}(z)\,
 \frac{(x/z)f_a(x/z,\mu_{F,\rm PS})}{xf_b(x,\mu_{F,\rm PS})}
 \,dz\,\frac{d\varphi}{2\pi},
\]

together with the actual Pythia support and correction factors. This
equation fixes conventions rather than replacing the process-specific
shower kernel. Pythia uses `xf`, so using a ratio of `f` without its
`1/z` changes the density.

All hard weights use the run's MSbar PDFs, `μ_F` and `μ_R`. For the exact
numerical identity above, `K` uses the shower's own PDF and coupling
definitions and scales. A fixed-order approximation to `K` is sufficient
for NLO equivalence under the additional conditions derived below; it
defines a different choice of higher-order matching terms. PDF-set IDs,
members, shower coupling settings and scale multipliers are part of the
run provenance. PDFs must support a positive probabilistic shower in the
sampled domain; an arbitrary signed PDF ratio does not define a
rejection probability.

## The cancellation derived directly

First take a single common history and work above the shower cutoff. The
one-step shower measure of an S event is

\[
 d\sigma_S=\bar B_D\Delta_K(t_0)\,d\mu_B
  +\frac{\bar B_D}{B}\Delta_K(t)K(q)\,d\mu_R,
\]

where

\[
 \Delta_K(t)=\exp\left[-\int_{t(q')>t}\frac{K(q')}{B}\,d\phi'\right].
\]

The original signed H density is `H=R-D`. Split it **after forming the
complete weight**:

\[
 H=H_+-H_-,\qquad H_+=[H]_+,\quad H_-=[-H]_+.
\]

Removing negative H events adds `H_- dμ_R`. At first order the shower
supplies `K dμ_R`, so compensating removal must satisfy

\[
 K(1-a)=H_-.
\]

This determines the acceptance uniquely wherever `K>0`:

\[
 \boxed{a=1-H_-/K.}
\]

The resulting measure is

\[
 d\sigma_{\rm MAc}=
 \bar B_D\Delta_K(t_0)\,d\mu_B
 +\frac{\bar B_D}{B}\Delta_K(t)(K-H_-)\,d\mu_R
 +H_+\,d\mu_R.
\]

Subtracting the original MC@NLO expression gives the exact resolved
identity

\[
 d\sigma_{\rm MAc}-d\sigma_{\rm MC}
 =H_-\left[1-\frac{\bar B_D}{B}\Delta_K(t)\right]d\mu_R.
\]

If `B=O(α_s^p)`, the sign-separated `H_-` is integrable and
`\bar B_D/B=1+O(α_s)` is a valid local expansion, this difference starts
at `O(α_s^(p+2))`. The same conclusion follows for infrared-safe
observables after expansion and integration. Near Born zeros or in the
immediate vicinity of the shower cutoff the formal expansion need not
be a small numerical correction. Subsequent shower evolution of finite
hard remainders changes them beyond NLO.

If `D=K`, then `H_-=[K-R]_+` and the result reduces to
`a=min(1,R/K)`. If `D≠K`, inserting `min(1,R/D)` instead leaves

\[
 \delta\sigma^{(p+1)}=\int H_-\left(1-\frac K D\right)d\mu_R
\]

in the region `D>R`. It is already of real-emission order, unless an
additional power-suppression argument applies. For example,
`(K,D,R)=(1,0.9,0.8)` requires `a=0.9`, whereas `R/D=8/9` removes the
wrong density. This is an algebraic counterexample; it is not a claim of
a measured error in a published implementation.

### A leading-order shower density is sufficient for NLO matching

It is useful to separate an exact numerical compensation prescription
from the perturbative accuracy it needs. Suppose the implemented
denominator is `K_0`, with

\[
 K_{\rm PS}=K_0+O(\alpha_s^{p+2}),\qquad
 a_0=1-H_-/K_0.
\]

The difference from the original MC construction then becomes

\[
 H_-\left[1-\frac{\bar B_D}{B}\Delta_{\rm PS}
                  \frac{K_{\rm PS}}{K_0}\right]d\mu_R.
\]

This still starts beyond NLO if `K_PS/K_0=1+O(α_s)` in the region where
the veto acts and the earlier integrability conditions hold. It does not
require reproducing the numerical all-order running coupling or the
conditional Sudakov in the denominator. At fixed finite scale ratios,
running `α_s(pT)` versus `α_s(μ_R)`, CMW conversion, and DGLAP-evolved
ratios from the same MSbar PDF set change the one-emission density at the
next perturbative order. They can be retained or omitted as an explicit
choice of higher-order prescription. A large logarithm can make that
formal difference numerically significant.

By contrast, a different recoil map, a missing Jacobian, MEC weight,
azimuthal modulation, kinematic boundary, heavy-quark mass term or ISR
`pT0` screening changes the coefficient of the first power of the shower
coupling. Such differences are not higher order merely because they
vanish in a high-energy or small-mass limit. A separate, quantified
power-correction approximation is needed if they are omitted. In
particular an approximate denominator's finite-event probability bound
must still hold; the perturbative argument does not justify clipping.

## What the present generator actually subtracts

The source has further distinctions beyond a schematic shower kernel:

* `compute_MCsubtraction_kl` in
  `Template/NLO/SubProcesses/montecarlocounter.f` evaluates native Born
  amplitudes and colour connections, restores radiation-coordinate
  factors, and includes starting-scale damping. Its `include_gfun`
  switch controls soft/azimuthal G factors.
* `compute_MC_subt_term` and the native-history driver in
  `Template/NLO/SubProcesses/fks_singular.f` add the MC and G replacement
  contributions with the history's no-emission factor `P`.
* `compute_MCsubtraction_from_gfun` is a signed soft/collinear/
  soft-collinear completion. It is part of the original hard density,
  rather than an independently positive shower kernel.
* At the initial inspection, `bogus_probne_fun` has `itype=2`: an
  artificial `P` transitioning from zero below 0.5 GeV to one above
  10 GeV, also when the physical MC@NLO-Delta switch is false. Thus
  disabling MC@NLO-Delta alone does not imply the idealized `P=1` input.

For a single partition, write its complete G-completed counterterm as
`D_G`. The actual hard density has the schematic form

\[
 H=P(R-D_G),\qquad
 D_{\rm in}\equiv R-H=(1-P)R+P D_G.
\]

The S density contains the matching integrated `D_in`. Therefore the
correct numerator is `H_-=P[D_G-R]_+` for `0≤P≤1`, and the general
formula can equivalently be written

\[
 a=1-\frac{[D_{\rm in}-R]_+}{K_{\rm PS}}.
\]

For several histories with different `P`, form the actual complete H
density before taking its negative part. This avoids assuming that a
single averaged `P` factors out. Keeping the existing `P` or choosing
`P=1` for a new mode are distinct, consistent choices only if event
generation, its S compensation and the post-shower evaluator all use
the same definition. Disabling it only during postprocessing is wrong.

Raw kernels obtained with `include_gfun=.false.` are useful building
blocks for `K`; that switch alone does not certify equality to the
actual accepted Pythia density. Physical support, recoil, masses,
shower corrections, PDFs, coupling, start-scale averaging and the
measure still need to agree.

### External-particle ISR recoil and delayed DY decays

The common ISR inverse acts on every external hard final-state momentum
with the inverse of Pythia's production recoil transformation. This
requirement includes the electron directions in `pp -> e- e+`. An LHA
record may additionally contain a colourless status-2 Z or photon
annotation, although the fixed-order matrix element still describes
the same external electron pair.

Pythia treats such annotated daughters differently from direct external
particles. `SimpleSpaceShower::branch` constructs the common production
transformation `Mtot`, while `PartonLevel::resonanceShowers` subsequently
transports delayed daughters with `RotBstMatrix.bst(oldParent,newParent)`.
Both conventions conserve the parent four-momentum, but they differ by
a rotation in its rest frame. The existing decay restoration correctly
reproduces that native delayed-decay convention. It does not therefore
make the electron angles obey the external-particle inverse used by
the source kernel.

An independent 80-point actual-emitter DY audit exposed this distinction:
the evolution scale and absolute scalar coefficient agreed to
`2.9e-13` and `6.7e-16`, respectively, while the reconstructed original
Born electron/incoming invariants differed by as much as `0.108 sBorn`.
A stable-ZZ control closed the corresponding invariants to `2.1e-13 sBorn`.
Applying the native common `Mtot` directly to the original DY daughters,
then using the unchanged analytic inverse, restored closure to
`4.9e-13 sBorn`. Thus an absolute splitting-coefficient test alone could
not certify this angular map.

The DY validation uses a derived LHA presentation that removes only
colourless status-2 Z/photon annotations for the unique external electron
pair and remaps mother labels. Every external ID, status, colour, momentum,
mass and spin token, the event weight and scale, and all event annotations
are preserved. Already-direct events remain unchanged. The original
hard-source files are retained byte for byte. The same presentation must
be applied to the primary and ordinary MC@NLO reference, with QED and
decay radiation disabled in the specified shower profile.

This change needs no hard-source regeneration: the fixed-order external
Born and real configurations, native subtraction maps, virtual terms,
PDFs and integrated source compensation are unchanged. It makes the
shower presentation implement the external-particle recoil already
assumed by those maps. The source-density and source-H sign audits
therefore remain applicable; stable ZZ requires no corresponding change.

The derived 5,000-event DY pilot completed native evolution, strict
rejection and full-event filtering without changing the frozen shower
sources. Of these records, 4,811 resonance annotations were removed and
189 were already direct. Eighty actual first-ISR points, forty on each
beam, restored original Born invariant closure to `4.4e-13 sBorn`, scale
closure to `2.9e-13` and the absolute scalar coefficient to `6.7e-16`.
Its original input sum `48.095346696 pb` and retained sum
`39.119468620 pb` equal those of the previous presentation at the quoted
precision. The map correction is necessary for the matching contract;
it does not explain the entire finite-coupling normalization offset or
replace the complete production comparison.

### Explicit Pythia heavy-flavour limitations

`mc_pythia8_kernel.f90` provides the ordinary global-II ISR scalar
correction and support factors independently of the MC subtraction. For
heavy Born daughters it uses Pythia 8.318's internal mass floors,
`m_c>=1.2 GeV` and `m_b>=4 GeV`, including when particle-data masses were
set to zero. Above the forced threshold region the accepted AP
coefficients are

\[
 P_{Q\to Qg}=C_F\left[\frac{1+z^2}{1-z}
                         -\frac{2z(1-z)m_Q^2}{t}\right],
\]
\[
 P_{g\to Q\bar Q}=T_R\left[z^2+(1-z)^2
                         +\frac{2z(1-z)m_Q^2}{t}\right].
\]

The routine also includes the ordinary transverse-momentum reality
condition, heavy-daughter `x,z` bounds, and the extra pair-space bound
when a backwards `Q→gQ` creates a heavy sister. A nonzero emitted-sister
mass needs its own inverse map and Jacobian. A scalar weight cannot
repair evaluating a massless hard matrix element at an inconsistent
massive real configuration. The routine returns a distinct unsupported
status if such a map has not been supplied.

For a heavy Born daughter with `t<2m_Q²`, Pythia uses
`pT2nearThreshold`, which forces a normalized `g→QQbar` conversion and
changes the virtuality relation. It is not the ordinary `α_s P(z)/t`
kernel. In this window `Q→Qg` has exactly zero support; the helper returns
`K=0` for that channel. For `g→QQbar` it reports the unsupported conversion
explicitly; an evaluator must not interpret that status as `K=0`.
Changing an XML particle mass does not
remove the compiled internal mass floors. This must be addressed before
claiming arbitrary heavy-flavour coverage.

### Ordinary FSR correspondence and its physical cutoff

The scalar first-FSR contract was checked against Pythia 8.318's
`SimpleTimeShower::pTnext`, `pT2nextQCD` and `branch`, rather than inferred
from the agreement of two source-weight routines. Global recoil mode
zero uses the sum of the hard final-state spectators as the recoiler.
The production inverse FSR map uses precisely that global spectator
momentum and preserves its invariant mass. Its `z,t` variables and
absolute radiation determinant agree with independent forward-branch
phase-space factorization on 1,800 massive and massless points,
including both massive FKS branches.

For a quark of mass `m`, emitted gluon, pair invariant `v`, hard-system
invariant `s` and global spectator invariant `M^2`, write
`E_pair=(s+v-M^2)/(2sqrt(s))` and
`k_B=sqrt(lambda(s,m^2,M^2))/(2sqrt(s))`. Direct two-body phase-space
factorization gives the radiation measure, with unaveraged azimuth,

\[
 \frac{d\Phi_{n+1}}{d\Phi_n\,dz\,dt\,d\phi}
 =\frac{(1-m^2/v)E_{\rm pair}}
        {32\pi^3 k_B z(1-z)}.
\]

The inverse coordinates follow independently from the measured hard-CM
gluon energy: `1-z=E_g/[E_pair(1-m^2/v)]` and
`t=z(1-z)(v-m^2)`. The reconstructed Born radiator points along the
pair momentum in that frame, with energy `(s+m^2-M^2)/(2sqrt(s))`.
The metadata audit
`docs/audits/check_macnlops_fsr_kinematics.py` tests these identities
against the original before-branch radiator and the actual Pythia
evolution scale, without invoking the native inverse map.
On 20 actual `t->t g` or antitop emissions from the full five-flavour
200-event proton-proton top-pair sample, the largest relative
Born-momentum error was `8.88e-15` and the largest relative
evolution-scale error was `4.53e-14`. The same
audit inverts the common longitudinal spectator boost and restores each
before-branch spectator momentum to `4.02e-15` relative precision. It
uses a light-cone rapidity ratio that also remains valid for a massless
composite recoil. This is a pointwise test of actual accepted Pythia
emissions; the independent phase-space factorization establishes the
Jacobian separately.

With MECs, polarization bias and finite extra splitting terms disabled,
Pythia's no-MEC massive quark numerator remains `C_F(1+z^2)/(1-z)`.
There is no additional massive-top AP subtraction to insert into this
FSR scalar coefficient. The quark mass enters the daughter rescaling,
inverse map, Jacobian and support. For a gluon emitter, Pythia's two
ordered colour-end terms contain `(C_A/2)(1+z^3)/(1-z)`. The source's
symmetric kernel and complementary daughter partitions reproduce their
sum. Its `py8_gluon_recoil_weight` also reproduces the actual
`recoilDeadCone=on` factor for a massive global spectator, including the
numerical denominator guards and rejection of negative trial weights.
The complete daughter/connection regression tests that scalar density
and support against the literal Pythia expression.

One further accepted-branch support condition is essential. In
`SimpleTimeShower::branch`, gluon emission from a massive quark rescales
the evolution transverse momentum before applying the physical cutoff:

\[
 M_{qg}^2=m_q^2+\frac{t}{z(1-z)},\qquad
 p_{T,\mathrm{orig}}=\sqrt t\left(1-\frac{m_q^2}{M_{qg}^2}\right),
 \qquad p_{T,\mathrm{orig}}\ge q_{0,\mathrm{FSR}}.
\]

Thus the ordinary condition `t>=q0,FSR^2` is insufficient. For a
173 GeV top, `z=1/2` and `q0,FSR=0.5867333369950678 GeV`, the accepted
evolution scale starts at approximately `sqrt(t)=16.572 GeV`. Below
that boundary the actual accepted kernel is zero even though the
unrestricted AP density is positive. `py8_fsr_ordinary_weight` supplies
this additional support factor for both the generated scalar source
and the rejection denominator; the map-only massive tests cannot
validate it. Five compiled FSR helper tests check the boundary for
charm, bottom and top, charge conjugation and daughter exchange,
massless limits, flavour support and missing massive-pair maps.

For `weightGluonToQuark=1`, massless `g->q qbar` has the ordinary
`T_R[z^2+(1-z)^2]` kernel after summing its two colour ends. The flavour
draw cancels the factor `n_f` in the proposal rate; each individual
colour end has coefficient `1/4` times that polynomial. The compiled
production coefficient divided by an independently factorized forward
phase-space measure agrees with this literal Pythia rate on 12 varied
`z,t` points to `5.19e-15` relative precision. The flavour
draw is limited by `TimeShower:nGluonToQuark`, with at most five flavours;
top-pair conversion is absent. The internal charm/bottom floors used
to divide alpha-s evolution ranges do not veto ordinary pair production
when the configured daughter masses are zero: the flavour draw uses
the configured particle-data mass. The production profile must fix the
same allowed species as the source. For nonzero daughter masses the
actual trial contains `beta=sqrt(1-4m_q^2/M_pair^2)` and the branch
rescales both daughters. That is a different massive-pair inverse map.
The helper explicitly reports it as unsupported rather than multiplying
a massless density by `beta`. A successful source-H numerical audit for
massive top radiation does not establish that separate pair map.

The first-endpoint global map can also be preserved while using Pythia's
native `globalRecoilMode=1`, `nMaxGlobalBranch=1` for a subsequent full
shower. In 8.318, `prepareGlobal` sets `nGlobal=0` and stores all final
coloured hard legs. With `nPartonsInBorn=-1`, none are removed by its
Born-multiplicity test. `Particle::isAncestor` includes the particle
itself. Consequently every initial quark/gluon radiator selects global
recoil in mode 1, just as in mode 0, in both `pTnext` and `branch`.
Its mass, spectator sum, Jacobian and random draws are therefore unchanged
until the first accepted FSR branch, including for a massive bottom or top.
Only a successful branch erases hard lines and increments `nGlobal`;
failed trials do neither. An accepted ISR, including a forced threshold
conversion, already fixes the first endpoint before any subsequent recoil
choice matters. Thus the first-endpoint density proof can be kept while
bounding the number of global FSR recoils in the continuation. This is a
statement about that native setting, not an assertion that all later
emissions retain the global-map kernel derived here. The counters and hard
ancestry used in this argument belong to the live native event. Exporting
its current momenta and shower ceilings to LHE does not preserve those
counters or prove that a fresh Pythia import makes the same recoil choices.

### Historical patched four-flavour W plus bottom-pair production

This subsection records the earlier patched production. Its nf4 coupling
claims and numerical controls do not describe the active stock runtime
defined at the top of this note.

The massive-bottom test uses `loop_sm`, `m_b=4.7 GeV`, a stable W and
`p4 = g u d s c u~ d~ s~ c~`, generating
`p4 p4 > w+ b b~ QCD=2 QED=1 [QCD]` at 13 TeV. Its ordinary MSbar set
is `MSTW2008nlo68cl_nf4/0`, LHAPDF ID 23300, with
`alpha_s(M_Z)=0.114904`. The official grid contains only gluon and
quark flavours one through four. This is a maximum-four-flavour set,
with three-flavour evolution below the charm threshold, rather than
a massless-bottom or five-flavour substitute.
[Official LHAPDF grid metadata](https://lhapdfsets.web.cern.ch/current/MSTW2008nlo68cl_nf4/MSTW2008nlo68cl_nf4.info).

The historical certified shower profile used `SpaceShower:nQuarkIn=4`,
`TimeShower:nGluonToQuark=4` and `StandardModel:alphaSnfmax=4`.
The native source and evaluator receive the two physical splitting
counts as independent integer fields, and check that they describe
the supported common four- or five-flavour profile. The generator
keeps the massive Born bottom legs and their ordinary `b->b g`
FSR, including the exact massive map and physical rescaled cutoff.

Pythia 8.318 cannot realize the last coupling setting without a patch:
its original `AlphaStrong::init` clamps the maximum flavour count to
at least five, and the two simple showers explicitly choose a
five-flavour trial beta function above the bottom threshold. The
optional isolated four-flavour build corrects both. It determines
`Lambda_4` directly from the specified four-flavour coupling at `M_Z`,
matches `Lambda_3` at charm, and keeps the target coupling and QCD trial
evolution at four flavours above charm. The original five- and
six-flavour initialization branches remain unchanged. An independent
compiled regression checks the one-, two- and three-loop formulas,
the absence of a bottom threshold, the factorization into first-order
coupling and higher-order correction, and byte-identical five/six-flavour
values at 66 scale/order points. The two-loop anchor error is below
`4.63e-11` relative. The existing installation is not modified; a second
capability symbol, value 831802, certifies the optional build.

This particular Born process has no outgoing Born gluon. Its first
QCD FSR is therefore massive `b->b g`, not `g->b bbar`. Incoming
four-flavour closure also prevents an ISR bottom mother or emitted
bottom sister. All allowed first-step ISR sisters are massless in this
hard model in the ordinary region above the history boundary. Native
forced charm conversion may still produce a 1.2 GeV sister below that
boundary; it is retained with probability one on its separate mass-shell
component, as derived above. Thus no above-boundary first-step density
call requires the existing unsupported massive-pair/sister inverse maps,
and no such map is approximated in this calculation. The same explicitly selected shower
flavour counts continue to forbid additional bottom-pair conversion
in a subsequent full shower. That higher-order shower choice is part
of the reported four-flavour profile.

The two independent source runs each contain exactly 200,000 events.
Their event-level audits find only incoming flavours one through four
and gluons, one stable W+, and exactly one massive bottom and antibottom
of 4.7 GeV per event. The original-weight sums are `447.9113302 pb`
(resolved source) and `464.5691933 pb` (ordinary reference); the native
integration estimates are separately `451.263 +/- 7.616 pb` and
`474.517 +/- 5.840 pb`. These estimates are not used to rescale the
event files. An initial 100-event first-emission smoke completes native
rejection and checks ten actual massive-bottom FSR inverse maps, with
maximum relative evolution-scale residual `5.46e-13`.

The first large production chunk also exposes a numerical preflight
false rejection: a boosted, known bottom at energy 2711.67 GeV has
`m^2=22.0899988003 GeV^2` from its double-precision components, versus
the nominal `22.09 GeV^2`. The original 256-epsilon hard-event bound is
slightly too tight for the composed boosts. An evaluator-only change
to 4096 epsilons accepts this point without changing its momenta or
density. Compiled regressions still reject a genuine 1.2 GeV emitted
charm on a massless hard shell and a 4.7-to-4.8 GeV bottom-mass shift.
The failed 5,000-event rejection chunk is replayed at its original
random seed from unchanged first/full shower checkpoints. No shower
is rerun; cards, source kernels, generation fingerprints and proposal
hashes remain unchanged. The explicit recovery provenance is
`wbb/diagnostics/numerical_shell_guard/guard_change_provenance.json`
under the large-sample output directory.

## Random starting scales and conditional histories

An S event's actual shower is conditional on its original Born point,
colour and starting scale `Q`. Denote those auxiliary choices by `λ`
with law `p(λ|b)`. The marginal first-order real density is

\[
 K_{\rm marg}(q)=\sum_h\int d\lambda\,
 \frac{B(b_h) p(\lambda|b_h)\kappa_h(q|b_h,\lambda)}{J_h}.
\]

For a scale that only restricts emission to `t<Q`, the integration
supplies the survival probability `Pr(Q>t)`. The generator's
`compute_damping_weight=1-emscafun((qMC-smin)/(smax-smin),1)` represents
such a marginal scale factor when the same distribution is used to
write SCALUP. It must not be replaced by `1` merely because the observed
event has `SCALUP>t`.

A common `a(q)=1-H_-(q)/K_marg(q)` subtracts exactly `H_-` at NLO after
summing the conditional shower samples. Using the same global `H_-`
with only the observed history's conditional kernel generally subtracts
the wrong amount. A conditional construction is possible, but its
numerator must partition `H_-` consistently among those conditional
histories. Different histories retain their own Born points, Sudakovs,
recoil and shower continuation. In particular, exponentiating
`K_marg` does not reconstruct the mixture of the actual conditional
Sudakov factors.

The proof only uses `\bar B(b_h)/B(b_h)=1+O(α_s)` for each history and
the summed first-order measure. It does not require that all inverse
histories share the same Born point.

## Maps, sectors, normalization and rejection semantics

For a showered real event `q=M_h(b_h,r_h)`, an inverse FKS map may give
`(b_F,r_F)=F^{-1}(q)` with `b_F≠b_h`. Regenerating `F(b_F,r_F)=q`
provides the existing FKS evaluators with consistent bookkeeping. The
original hard density must use the projections with which that H sample
was generated. The shower denominator must use the inverse shower
histories that actually populate `q`. Replacing the argument of the
original hard counterterm by `b_h` changes its definition.

If all densities are expressed in FKS coordinates, the common positive
Jacobian cancels:

\[
 \frac{[-H_F]_+}{K_F}
 =\frac{[-J_F H]_+}{J_F K}
 =\frac{[-H]_+}{K}.
\]

A common Jacobian, flux, PDF or grid factor cancels only when it is
actually present in both numerator and denominator. Native projected
point measures deliberately remove the integration Born-chart measure;
raw squared matrix elements cannot be divided by integration weights
without restoring the missing factors.

Sector definitions must also match. In general

\[
 \sum_\alpha[-H_\alpha]_+\ne[-\sum_\alpha H_\alpha]_+.
\]

If the input H sample was generated from the summed native H density,
classify that sum. If it was generated from separately signed sector
densities, removal of every negative sector event requires the sum of
their negative parts and a corresponding shower partition. A metadata
field naming a sector does not by itself establish either construction.

The practical rules are:

1. Preserve the original S ensemble and the original MSbar inputs.
2. Classify each original H record using the complete generating hard
   density. Retain its positive contribution; remove the negative
   contribution only when the compensating construction is valid.
3. Evolve each S record to its first **accepted** QCD shower emission or
   to the cutoff. Internal Pythia trial rejections do not count as
   emissions. Disable additional interactions, QED branchings, decays,
   hadronization and other sources of extra final particles for this
   stage unless the evaluator explicitly includes them.
4. Keep S records with no emission with probability one. For an emitted
   record evaluate the original negative hard density and actual shower
   density at that real state and accept if `u<a`, `u` uniform in `[0,1)`.
5. Reject the **whole proposed S event** on failure. Do not retry its
   Born state until acceptance and do not continue from a rejected
   branching. The latter algorithm changes the shower Sudakov and
   requires a different integrated S contribution.
6. Preserve every accepted event's signed original weight and relative
   alternative weights. Negative S weights remain negative. Uniform
   positive final weights require nonnegative S input as a separate
   property.
7. Preserve the pre-rejection statistical normalization. Rejection
   reduces the cross section. Never normalize the surviving sample back
   to the input cross section or average only over survivors.

For an input file whose estimator is `σ=(1/N)Σ_i w_i`, the output
estimator is `σ_acc=(1/N)Σ_i I_i w_i`, including rejected events as zero.
If an LHE consumer interprets a file of `N_acc` records by averaging
their weights, rescale the surviving weights by `N_acc/N`, or write
zero-weight rejected records. For sum-normalized input weights, do not
rescale them. Mixed file/chunk processing must retain the source file's
normalization and combine chunks according to that contract; treating
each surviving chunk as independently normalized loses the rejection
rate. The estimator's Bernoulli contribution has conditional variance
`Σ_i w_i² a_i(1-a_i)/N²` in the average convention.

H records are already real-emission records. Adding another shower
emission before evaluating their hard classification would evaluate a
different multiplicity. At this endpoint they pass through unchanged
when accepted; a later complete shower can evolve them normally.

## Bounds, infrared finiteness and honest limits

### Disjoint shower mass shells and forced threshold competition

For a massless hard quark, the source real measure and hard remainder
live on a massless real phase space. A Pythia forced threshold conversion
can emit a quark with its internal mass floor even when the model and
particle-data quark mass are zero. Its real configurations belong to a
different mass shell. Treat the complete first-step sample as a measure
on their disjoint union, `μ=μ_hard+μ_forced`. The source H density is zero
on `μ_forced`. Consequently those S events have acceptance one, and the
forced distribution need not be evaluated merely to keep them. On the
massless source manifold, that massive conversion contributes zero
density, rather than an unsupported massless AP factor.

This observation does **not** generally restore the assumption
`Δ=1+O(α_s)` on the common massless manifold. For example, a light-leg
ISR or a final-state emission below another Born leg's heavy threshold
competes with that leg's normalized forced conversion. Its first-emission
density contains the probability that the forced candidate lies below
the ordinary emission scale. At fixed input PDFs that probability need
not approach one when the shower coupling is expanded. The denominator
for an ordinary shared real point must include this competition, or the
H transformation must be changed consistently.

Diagonal colour-singlet `q qbar` production avoids this particular
mixed competition: a light pair has no forced heavy leg; for a heavy
pair, ordinary `Q→Qg` has zero support below the common threshold, and
above it all forced candidates are softer. This statement does not
establish the same property for arbitrary coloured or mixed-flavour
Born systems.

### History gate and the generated counterterm

The new source prescription avoids the forced-competition region in the
S veto, while retaining all Born flavours and the unmodified physical
heavy-threshold shower. Index a native inverse shower history, Born
colour flow and physical subprocess flavour by `h`. Let `R_h` be its
nonnegative partition of the real density, with `sum_h R_h=R`, and let
`K_{0,h}` be its nonnegative ordinary scalar shower density at leading
real-emission order. It includes its native inverse-map measure,
starting-scale average, physical support and actual-flavour PDF factors.
It excludes the source G completion and spin-dependent terms absent
from the configured scalar shower.

For the underlying Born state of each history define

\[
 q_{\mathrm{match},h}=\max\left(q_{0,\mathrm{ISR}},
 q_{0,\mathrm{FSR}},\sqrt2\,m_{\mathrm{int}}(a_h),
 \sqrt2\,m_{\mathrm{int}}(b_h)\right),\qquad
 P_h(q)=\Theta\!\left(t_h(q)-q_{\mathrm{match},h}^{\,2}\right).
\]

Here `t_h` is Pythia's evolution `pT^2` in that history, not the observed
beam-transverse momentum. The internal mass is zero for light quarks
and gluons, `max(m_c,1.2 GeV)` for charm and `max(m_b,4 GeV)` for bottom.
Both incoming Born flavours enter even for a final-state emitting
history: a forced conversion on the other incoming leg can compete
with an ordinary first emission. Using the maximum of the two effective
physical shower cutoffs is conservative; it can be larger than the
individual emitting leg's cutoff. An emission exactly at the boundary
has zero measure and is assigned `P_h=0` by the implementation.

Generate the paired source densities

\[
 D_h^*=(1-P_h)R_h+P_hK_{0,h},\qquad
 H^*=R-\sum_hD_h^*=\sum_hP_h(R_h-K_{0,h}),
 \qquad K_{P,0}=\sum_hP_hK_{0,h}.
\]

The integrated counterpart of `D*` belongs in the S density
`barB_{D*}`. Operationally the source retains raw real records for both
H and S until the physical subprocess flavour is known, then multiplies
them by `P_h` and `1-P_h`, respectively. The paired MC records in H and
S both receive `P_h` and the same scalar shower correction. Applying
this after PDF/flavour lookup is essential: a representative light
subprocess can contain physical charm or bottom PDF channels. Native
kinematics are saved with each record before histories are combined.
Changing the H record alone, or multiplying only an already-summed
luminosity by a representative-flavour gate, would not implement `D*`.
These transfers apply to histories whose underlying Born state passes
the generation cuts. When it does not, the source keeps its full real
term in H and transfers none to S; its ordinary shower density is zero
because that Born state is absent from the S sample. Such additional
nonnegative real terms do not weaken the envelope below.

For nonnegative physical real partitions and scalar kernels the envelope
is now an identity,

\[
 0\le H_-^*=[-H^*]_+
 =\left[K_{P,0}-\sum_hP_hR_h\right]_+\le K_{P,0}.
\]

Take the sign only after forming this complete H sum. Keep positive H,
drop negative H, retain no-emission S, and accept an emitted S event
from its **actual** Born/shower history `h_actual` with probability

\[
 a(q,h_{\mathrm{actual}})
 =1-P_{h_{\mathrm{actual}}}(q)\,
                  \frac{H_-^*(q)}{K_{P,0}(q)}.
\]

Use acceptance one when `K_{P,0}=0`, where the envelope requires
`H_-^*=0`. The gate of the observed S history is indispensable: using
one common acceptance on all S histories would also remove the ungated
histories omitted from `K_{P,0}`. On the common hard manifold, the
leading removed S density is exactly
`sum_h P_h K_{0,h} H_-*/K_{P,0}=H_-*`.

For gated histories every forced incoming-heavy candidate has a lower
evolution scale. Thus the conditional no-earlier-emission probability
in the gated region has the ordinary `1+O(alpha_s)` expansion; the
unknown normalized forced-threshold distribution never enters this
veto. Below the gate the S acceptance is one, including ordinary
emissions competing with forced candidates. A forced conversion whose
emitted floor mass is absent from the massless hard model belongs to
the separate massive component described above. It also passes with
acceptance one, without sending its momenta through a massless inverse
map. Ordinary above-threshold evolution instead uses the configured
particle-data sister mass (`SimpleSpaceShower.cc`, `pT2nextQCD`), so
setting those masses equal to the hard model does not force ordinary
sisters onto the threshold mass shell. A genuinely massive hard sister
still requires the corresponding massive ordinary inverse map; this
source prescription does not supply one.

The rejection changes the `MC[D*,K_PS]` prediction only beyond NLO on
the gated support, assuming `K_PS,h/K_0,h=1+O(alpha_s)`, the finite
sign-separated remainder, and the usual local expansion of `barB/B`.
This statement concerns the rejection of the newly generated source.
It does not imply that replacing the counterterm below the gate is an
NNLO modification of fixed-order NLO.

For an arbitrary unitary one-step shower operator `T`, including forced
conversions and no-emission outcomes, an exact starting point is

\[
 \langle O\rangle_{\mathrm{MC}[D^*,T]}
 -\langle O\rangle_{\mathrm{NLO}}
 =\int d\mu_B\,\bar B_{D^*}(b)
                  [(TO)(b)-O(b)]
  -\sum_h\int d\mu_R\,D_h^*(q)[O(q)-O(b_h)].
\]

Here `T` acts on the S states, while the H states are retained at their
source multiplicity. Reducing this to a usual fixed-order residual
requires expanding the actual shower operator; unitarity alone is not
enough. On an ordinary common mass shell, if that expansion is
`B(TO-O_B)=sum_h integral K_0,h (O_q-O_B)+O(alpha_s^(p+2))`, the source
gives

\[
 \begin{split}
 \langle O\rangle_{\mathrm{MC}[D^*,K_{\rm PS}]}
 -\langle O\rangle_{\mathrm{NLO}}
 ={}&\sum_h\int(1-P_h)(K_{0,h}-R_h)
                 [O(q)-O(b_h)]\,d\mu_R\\
 &+O(\alpha_s^{p+2}),
 \end{split}
\]

The stated leading-kernel matching must also hold in the ordinary
below-gate region for this reduction. If it does not, the expansion
contains the actual first-emission measure, rather than `K_0`. In
particular, at fixed input PDFs a normalized forced conversion can
remain as `alpha_s` is taken to zero. Its contribution to `TO-O_B` is
then not described by an ordinary real-order AP kernel at all. Keep it
in the exact operator identity and state its physical cutoff or heavy
mass power correction separately. A disjoint massive shower component
cannot be identified with the massless hard measure merely because
both contribute to the same observable bin. This qualification applies
to the baseline shower prediction as well as to its rejection stage.

At fixed nonzero matching scales the displayed residual is of order
`alpha_s^(p+1)`, even though it is confined to an unresolved region.
Infrared safety makes `O(q)-O(b_h)` tend to zero, and can make its
integral a power correction when every relevant observable resolution
is much larger than `q_match,h`. Infrared safety alone does not make it
an extra power of the coupling. Exact NLO cancellation for a particular
measurement follows if `O(q)=O(b_h)` throughout the changed region. A
resolved one-jet distribution in a colour-singlet Born process, with
its jet cut above every matching scale, has that property. General
Born-sensitive distributions need not, since an unresolved emission can
change their recoil or bin assignment.

A jet veto below a matching scale is a direct counterexample to an
unqualified NLO claim. At Born level its measurement is one. For a
real emission with `Q_veto<pT<q_match`, the measurement is zero. That
part of the discrepancy is therefore

\[
 -\int_{Q_{\rm veto}<p_T<q_{\rm match}}
           (K_0-R)\,d\mu_R,
\]

which is generally nonzero at real-emission order. The source strategy
must be reported as matching with the stated finite-cutoff power
corrections. A variation of an arbitrary large matching scale would
change the prediction at NLO; using physical shower and heavy-threshold
scales limits that change but does not remove this mathematical limit.

### Alternative auxiliary-shower construction

One constructive generalization uses an auxiliary shower for the negative
H events. Let `h,λ` identify an inverse Born history and its conditional
colour and starting-scale choices, and let

\[
 K_0(q)=\sum_h\int d\lambda\,K_{0,h\lambda}(q),\qquad
 A(q)=\frac{\sum_h\int d\lambda\,
                  K_{0,h\lambda}(q)\Delta_{h\lambda}(t(q))}{K_0(q)}.
\]

Here `K_0` is the ordinary Sudakov-free proposal and `Δ` is the actual
conditional probability of no earlier shower emission, including any
forced candidates from competing legs. Choose a cancellation density
`0≤C≤min(H_-,K_0)`. Accept an emitted S event with `1-C/K_0`. For a
negative H event, select `h,λ` with probability proportional to
`K_{0,hλ}(q)`, evolve an independent copy of its Born shower, and reject
that H event with probability `C/H_-` **only if** the independent shower
has no emission harder than `t(q)`. This implements H rejection
probability `(C/H_-)A` without explicitly evaluating a forced density or
Sudakov.

The S removal is

\[
 \frac{C}{K_0}\sum_h\int d\lambda\,
 K_{0,h\lambda}\Delta_{h\lambda}\frac{\bar B_h}{B_h},
\]

whereas H removal is `C A`. Their difference is therefore

\[
 \frac{C}{K_0}\sum_h\int d\lambda\,
 K_{0,h\lambda}\Delta_{h\lambda}
       \left(1-\frac{\bar B_h}{B_h}\right),
\]

which starts beyond NLO when `C` is a finite real-order density and each
Born ratio has its usual expansion. This argument does not require the
forced no-emission probability to approach one. Conditional history
sampling, including the starting-scale law conditioned on supporting
the real point, is essential; an arbitrary shower of an arbitrarily
clustered H event would not supply `A`.

This is a constructive derivation, not a claim that the current runner
implements auxiliary H showers. The independent algebraic check
`python3 docs/audits/check_macnlops_stratified_measure.py` demonstrates
both the real-order error of ignoring forced competition and its removal
by this generalized H survival rule. It also does not solve an ordinary
massive emitted-parton map whose support actually overlaps a massive
hard model: that common-manifold density still needs the corresponding
inverse map, scalar kernel, PDF factors and physical Jacobian.

### Probability bounds and retained accuracy

An ordinary accept/reject step exists only if `0≤H_-≤K`. Where `K=0`,
it requires `H_-=0`. A violation is not fixed by clipping a negative
acceptance to zero: this removes only `K` while dropping `H_->K`, so it
leaves a real-emission-order excess. Reject invalid densities or retain
the signed residual `-[H_--K]_+` with a consistently changed stream
definition. The latter reduces negative weights without claiming to
remove all of them. Any alternative modification of the subtraction
must also update its integrated S counterpart.

An optional coupled rejection implements that residual without changing
the original S normalization. Set `C=min(H_-,K)`, accept emitted S with
probability `1-C/K`, and retain a negative H record with probability
`1-C/H_-`. Use acceptance one for S when `K=0`. The remaining hard
density is `H_+-H_-+C`; the difference from ordinary MC@NLO is
`C[1-(barB_D/B)Delta_K]`, which has the same NLO cancellation when C is
finite. Where the original envelope holds this is ordinary MAcNLOPS;
where `K=0` the original negative H density remains. The low-level
`Densities` API retains this option for algebraic comparisons. The
resolved-source production runner uses strict rejection: its envelope
is guaranteed by construction, so a bound violation indicates a failed
contract. Capping only the S veto while deleting all negative H records
would miss the compensating term.

Below the shower cutoff, no resolved S emission can compensate removal
of an H contribution. Keep the signed H contribution there, move a
compensation into a properly projected Born term, or quantify an
explicit cutoff power correction before discarding it. No-emission S
records cannot be silently deleted with an unspecified `R/K`. Massive
thresholds can create further shower dead regions that need the same
support analysis.

The integrability of `H=R-D` after a signed azimuthal or colour average
does not imply integrability of `H_±`. For example, a shower missing a
leading `cos(2φ)/t` spin term can have zero azimuthal average while its
positive part integrates logarithmically at small `t`. Soft/azimuthal
G completion can make the separated H remainder finite, but that
completion changes `D` without automatically changing Pythia's `K`.
This is why the distinction is needed even when generation is stable.

Finally, for distinct `D` and `K`, direct expansion of the underlying
MC@NLO construction itself gives

\[
 \langle O\rangle_{{\rm MC}[D,K]}-\langle O\rangle_{\rm NLO}
 =\int(K-D)\,[O(q)-O(b)]\,d\mu_R+O(\alpha_s^{p+2}),
\]

apart from cutoff terms. If `K-D` is localized to an unresolved region,
infrared safety can make this a power correction. It is not an NNLO
term by coupling counting alone. The generalized rejection preserves
the original MC@NLO prediction through NLO under the finite-remainder
assumptions; it does not by itself prove that an arbitrary unmatched
kernel/counterterm pair is exactly NLO accurate. Validation must test
the physical process, its actual maps and scales, and observables at
the resolution where the result will be used.

## Numerical checks of the generated density

The numerical results in this section were recorded before the stock-runtime
change. They remain source/kernel implementation evidence for the named
configurations; native cutoff, envelope and patched-coupling qualifications
do not automatically transfer to a new stock profile.

The independent stream regression
`python3 -m unittest tests.unit_tests.various.test_macnlops_derivation -v`
passes three checks. It assembles paired H/S records for 2,000 overlapping
history configurations and verifies the complete negative-H envelope.
With an ungated competing history whose survival probability remains
0.23, the observed-history gate gives a difference proportional to
`alpha_s^2`, whereas applying the same veto to every history leaves an
`alpha_s` error. Finite radiation-bin measurements independently verify
both the above-gate resolved cancellation and the nonzero sub-threshold
jet-veto residual. These are measure checks, not a replacement for a
Pythia density comparison.

On 20 October 2026 the opt-in generated-process regression passed all four
tests, including an independent comparison to the ordinary hard-event
generation path:

```sh
MG5_MACNLOPS_EXPORT_TESTS=1 python3 -m unittest \
  tests.unit_tests.fks.test_macnlops_evaluator
```

The fixture exports `u u~ > e+ e- [real=QCD]`, compiles its actual matrix
elements, support library and standalone density evaluator, and uses
ordinary `nn23nlo` MSbar PDFs for this source-density check. It does not
generate a matched sample or establish virtual corrections. Both owners
of the `q qbar` real channel give the same density, repeated queries do
not change it, and the original G/P terms change the rejection from its
naive shower-kernel version. The configuration test independently changes
the shower cutoff and beam support while retaining the original `R,D`.

The finite-cutoff configuration uses the shower's effective cutoff,
rather than assuming that the requested `pTmin` equals it. Pythia's
private initialization can raise the cutoff near its coupling pole
without updating the settings object. The one-step driver reproduces
that initialization using Pythia's public `AlphaStrong`, sets the
effective values before initialization, and records them together with
the exact initial beam `xMax` values in each metadata record. For
`α_s(M_Z)=0.120179` at two loops, a requested `pTmin=0.5 GeV` gives
`pTmin_ISR=0.5 GeV` and `pTmin_FSR=0.5867333369950678 GeV`, plus one
floating-point unit of conservative headroom. An independent compiled
`AlphaStrong` check and a real Pythia run with a nested supplied decay
tree agree with those values. The native evaluator's streaming
`MACNLOPS_CONFIG_V1` message applies these cutoffs and beam bounds to
each shower history while keeping the source `R,D` unchanged.

The historical production one-step driver required a separately built
Pythia 8.318 with the compensated trial-headroom patch. The archived patch
was named `pythia8318-trial-headroom.patch`; it is not an active build
prerequisite. For ordinary QCD proposals the patch multiplied the trial
overestimate by a configurable factor `h` and divided its acceptance by
`h`; their product and hence the target branching density were unchanged.
Shower uncertainty variations stayed off. In that historical driver,
native failed overestimates were reported as errors rather than accepted
with a clipped probability. The stock warning policy is different, as
stated at the top of this note.
The patch left the complete normalized `pT2nearThreshold` function
unchanged, checked by the archived isolated build helper, and exported a
capability symbol checked by the historical driver. A forced-only,
100-event bottom-Born test
at fixed seed gives byte-identical one-step events for the original
library and the patched library with headroom one or ten, including the
4 GeV emitted sister mass. This checks that the sampling modification
does not change that physical distribution; it does not supply a
massive-to-massless phase-space map.

The source comparison constructs 25 varied massless real configurations
with different emission scales, angles, Born directions and longitudinal
boosts, and visits both `q qbar` owners and the `qg,gq` channels at each
one. A temporary audit executable calls the **ordinary
`repartition_MC_H` generation routine**, removes its original outer FKS
partition, diagram-channel factor and real phase-space measure, and
compares the resulting H density with the evaluator's `R-D`. Across
these 100 points, the largest difference relative to `max(|R|,|D|,1)`
was `5.43e-14`; the largest H-relative error was `1.57e-11` because of
real/subtraction cancellation. The reusable check is

```sh
python3 docs/audits/check_macnlops_source_h.py \
  PROCESS/SubProcesses/P0_uux_emep
```

This is a one-colour Drell–Yan check. It does not validate the signs of
colour-sampled H events in a process with several Born colour flows.
The production MAcNLOPS mode must sum complete native histories and
Born colour flows **during generation before taking the sign**, so that
the removed negative H density is the same density reconstructed by
the evaluator. A colour-summed evaluator cannot replace the sum of
negative parts of separately colour-sampled input contributions.

The same requirement applies to physical subprocess flavours. Native
Born providers can list identical real flavours in different row orders.
For example, the top-pair `qg` histories list `[d,u,s,c,b]` and
`[u,d,s,c,b]`. Combining records only when their full tables agree
row by row leaves separately signed fragments, even though their
physical-flavour sum is correct. At one actual `ug -> t tbar u` point,
the two fragments were approximately `-6.9850e7` and `+8.7239e7`, while
their complete H density was positive. Removing the negative fragment
using a veto derived from the complete density would be inconsistent.
This failure cannot be cured by weakening the H-sign check.

MAcNLOPS H grouping therefore maps each owner's physical PDG row to the
matching native row in every member before unweighting. Repeated exact
PDG rows are matched by occurrence, preserving a bijection and their
multiplicities. The same lookup is required when removing event bias
and serializing reweight information. Native flavour tables, provider
indices and event-owner labels remain unchanged. A source generated
before this correction must be regenerated when affected; an exact
mean-density comparison alone does not certify its unweighted signs.
An ordinary signed MC@NLO reference retains the linear sum of those
fragments and does not require this nonlinear sign classification.
The four-process static audit found this row-order issue in top-pair
contexts, while all DY, ZZ and Wbb histories already had identical
ordered tables. The supported grouped samples require equal physical
flavour multisets; partially overlapping tables need a further
flavour-resolved grouping construction.

A subsequent read-only audit checked **every actual H event**, of
either sign, in the complete 200,000-event DY, ZZ and Wbb primary
sources. The checked H counts were respectively 32,741, 41,320 and
65,364: all 139,425 signs agreed with the reconstructed complete
`R-D`, and every negative H obeyed its shower cancellation bound.
No sign or bound failures occurred. The reports record unchanged
source SHA256 hashes, the frozen profile and exact evaluator, driver
and audit-script hashes. The fresh top-pair source generated with the
flavour-group correction also passed its complete 200,000-event audit:
all 63,210 H events (32,821 positive and 30,389 negative) were verified,
with maximum negative-H cancellation fraction `0.975506741872776`.
Thus the four primary sources together passed 202,635 actual H checks.
This actual-source check supplements the
mean-density tests; it does not replace the first-shower map and
kernel correspondence checks. The reproducible command is

```sh
python3 docs/audits/check_macnlops_h_signs.py \
  --process-dir PROCESS --expected-events 200000 \
  --output-dir NEW_AUDIT_DIRECTORY SOURCE_EVENTS.lhe.gz
```

The mean-density source-H audit was also extended to
`g g > t t~ [real=QCD]`, with `m_t=173 GeV` and two leading Born colour
flows. Across 100 varied real configurations, the normalized discrepancy
was `1.63e-13`, with H-relative error `1.29e-11`. This verifies that
the evaluator reconstructs the **summed** multicolour H density; it does
not make separately sampled colour signs interchangeable. The audit
command is

```sh
python3 docs/audits/check_macnlops_source_h.py --process ttbar \
  PROCESS/SubProcesses/P0_gg_ttx
```

Separately, the compiled ISR helper tests cover physical AP coefficients,
heavy-quark mass floors and ordinary kinematic bounds, including explicit
unsupported statuses for threshold conversion and a missing massive
emitted-parton map. The production Pythia scalar/FSR regression and the
symmetric initial-state recoil inverse/measure test also pass. These
checks establish their scoped coefficients and maps; none substitutes
for showered process-level validation of a final event sample.

## Histogram protocol for the large validation samples

`docs/audits/plot_macnlops_validation.py` reads the unrejected first and full
LHE chunks, the common primary accept/reject decisions, and an independently
generated ordinary MC@NLO reference. A final run requires all four processes,
both shower levels, and exactly 200,000 primary and 200,000 reference source
events per process. It checks the unchanged original weight sequence and the
preserved source header plus `#aMCatNLO` annotation between first and full
files. Repeated chunks are rejected. A smoke run requires
`--allow-incomplete`, marks every figure and summary as incomplete, and cannot
be presented as the production comparison.

The five distribution curves are the independent reference, the unrejected
resolved source, removal of negative H alone, S veto alone, and their combined
MAcNLOPS decision. All use original signed weights without a survivor
normalization. Total H, positive/negative H and positive/negative S are also
shown separately. Negative logarithmic bins are drawn reflected and dotted;
their original signs remain in the machine-readable results. Bins whose
statistical bands cross zero do not acquire a fictitious positive lower band.
Floating cancellation residues smaller than eight machine epsilons times the
sum of absolute weights are masked only for display and ratio denominators;
the raw sums and the masking rule are retained.

For bin sums A and B, the exported ratio variance is

    Var(A/B) = Var(A)/B^2 + A^2 Var(B)/B^4 − 2 A Cov(A,B)/B^3.

The source stages share proposals, so their covariance is the sum of squared
weights in the intersection of the two selections. The independent reference
has zero covariance with every primary stage. Undefined zero-denominator
ratios are masked. Every main comparison retains an unrestricted ratio panel
and provides a 0.5–1.5 zoom companion with explicit offscale markers. CSV and
NPZ files contain signed bin sums, squared sums, absolute sums, counts, full
stage covariance and unrestricted ratios. Separate underflow, overflow,
undefined-observable and exactly-zero counts prevent excluded phase space
from disappearing from the report. Overall source and retained counts,
negative counts and `N_eff = (sum w)^2/sum w^2` are reported without changing
any weights.

A separate normalization diagnostic conditions on the already generated
proposals. Their expected retained sum is `sum(w_i p_i)`, and the random
veto alone has variance `sum(w_i^2 p_i (1-p_i))`. Both are reported for the
whole sample and each H/S sign component. This conditional variance is not
added to the observed histogram `sumw2`, which already includes the event
selection. It separates random veto fluctuations from a change that is
already expected for the particular finite-coupling shower proposals.

Jets use the FastJet anti-kT algorithm with R=0.4 on stable quarks with
absolute PDG ID at most five and gluons; stable tops are excluded. The Wbb
panels show both all-parton jets and radiation jets obtained by excluding the
two stable bottom IDs before clustering. Thus a Born bottom jet is not
misidentified as the first additional radiation jet. Jet-rapidity panels
require the leading clustered jet to have `pT > 20 GeV`, including both Wbb
jet definitions. Softer and no-jet entries enter the missing-observable flow;
the event totals and original weights remain unchanged. The jet-pT spectrum
has no such cut and retains its zero bin. A rapidity distribution counting
arbitrarily soft jets would not define an infrared-safe observable.
The broad leading-radiation-jet spectrum covers 0–600 GeV. A second
histogram uses the identical clustered jet and resolves 0–50 GeV in
0.5 GeV bins, with separate comparison, mechanism and signed-H panels.
This resolution exposes the soft competition studied in the thesis's
Figs. 10 and 11 without replacing the broad high-energy distribution.
Per-observable CSV and expandable HTML tables give visible-event
fractions, the plotted range and every underflow, overflow and missing
entry with its signed weight; the CSV also records squared-weight sums. Fractions use
event counts, so cancellation of positive and negative weights cannot
hide a range loss. No distribution is rescaled to its visible range.
The hard systems are
the stable electron pair, Z pair, top pair, or W plus bottom pair. Z objects
are ordered by transverse momentum. Wbb azimuthal and rapidity differences
refer to the bottom pair. Wbb also has separate bottom-pair invariant-mass,
transverse-momentum and rapidity panels, alongside the complete Wbb system;
the three-body mass does not replace `m_bb`. The endpoint `pi − abs(delta_phi)` has logarithmic
bins, while its exact zero bin remains in the exported accounting.

The HTML index embeds the run plan and explanatory notes and can copy and
hash linked cards, profiles and audit reports from the input configuration.
The figures do not, by themselves, remove the finite matching-boundary
limitation derived above or certify a subsequent shower's native trial
envelope. The first endpoint and any later native clipping must be identified
separately by the run audits.

For archived full panels, the retained plotter requires one
`macnlops-full-audit-v1` record per original
event, with matching sequence and H/S kind and the declared runtime policy.
The historical patched profile required a protected S first endpoint and
no bound warning for its no-emission S outcome. Stock profiles instead
record `first_endpoint_strict=false` and ordinary native warnings without
rejecting those events. The plotter aggregates emission multiplicities,
exact native diagnostic messages, warning-event counts split by H/S, and
recorded sampler, cache and continuation-policy settings. The primary/reference
setting sets must agree. Compact JSON reports retain SHA256 hashes and
locations of the raw audits without duplicating their potentially large
per-event files in the plot directory.

Plotting is serial by default. `--workers 4` analyzes and renders the
four independent processes in isolated Python subprocesses, with one
thread per numerical backend. Every worker applies the same source-count,
flavour, first/full identity and continuation-audit checks. The parent
validates the complete production dataset, combines results in registry
order and publishes the shared staging directory only when all workers
succeed. Parallelism changes no event weight, histogram or selection.
The 17 plotting regressions include serial/two-worker agreement for
two distinct process-object fixtures, relative-path resolution and
atomic rollback on a failed child identity check. A separate rendering
comparison used actual 5,000-event DY primary and independent reference
chunks: all histogram arrays, summaries and 51 PNG figures were
identical between serial and isolated-worker execution. That comparison
is explicitly labelled incomplete and is not a substitute for the final
200,000-event first/full validation.

A separate read-only report auditor checks the exported arrays without
importing the plotting implementation. For every curve and observable,
the bins plus underflow, overflow and missing entries must reproduce the
event count, signed weight and squared-weight total. It also checks the
signed source partition, the two rejection mechanisms, shared-selection
covariances and identical first/full totals. A complete audit rereads the
eight original source files to anchor their nominal weight sums and
squared sums; native integration errors remain a separate quantity.
Portable per-bin CSV tables and the stage-total table must match the
raw arrays and summary. Eight focused checks cover preview closure,
deliberate bin/variance/flow/table corruptions, rejection of a false
complete-report claim or a missing per-process shower stage, and nominal header weights in the presence of
alternative weights. Passing these
accounting checks does not establish physical agreement with the
reference shower or certify NLO accuracy.

## Running the implementation

These instructions concern the stock runtime described at the top of this
note. Select a Pythia 8.318 prefix directly; do not build or load the
removed MAcNLOPS patches/provider, or supply a trial-headroom option. The
saved effective runtime settings, including `alpha_s_nfmax=5`, must match
the new source profile. The historical paper and its saved patched
profiles must not be relabelled as results from this runtime.

For the ordinary-input comparison, give the supplied unmodified ordinary
LHE file and the same native proposals to the before/after analysis.
Processing must qualify its source partition and mass transport, and
fail explicitly for unsupported input. The current `run_file` also requires
the measured `Cards/macnlops_profile.json`, matching source-state manifest,
generation/evaluator cards, source and PDF hashes, and a valid hard
annotation on every event. It is not an arbitrary older-stock-file or
bare `macnlops=False` importer. Fresh controlled preparation uses
`macnlops=True`, `macnlops_resolved_source=False` and
`store_hard_momenta=True`, preserving the ordinary G/P source physics.
Complete negative-H removal remains a strict probability-bound request.
The CLI's unspecified policy resolves to signed residuals for the ordinary
source; `--signed-residuals` can state that choice explicitly and
`--strict-positive-h` requests complete removal with failure on an invalid
bound. The pure Python `Densities` acceptance API retains its strict
default unless `signed_residuals=True` is passed. A failed strict bound
is not permission to switch the requested policy silently.

### Signed PDF luminosities and numerical source coordinates

The signed-residual runtime also admits finite signed `R` and `K`. Define
`u=max(D-R,0)` and `Kplus=max(K,0)`. The transferable amount is
`C=min(u,Kplus)`: S acceptance is `1-C/K` for `K>0` and one otherwise,
while negative-H acceptance is `1-C/u` for `u>0`. For sampled Delta, `u`
is still the undamped local negative part and the source Delta stays in
the original H weight. A nonpositive `K` leaves the S event and negative-H
residual unchanged. Neither physical densities nor event weights are
replaced by absolute values. In particular a negative nominal S weight
from virtual corrections alone does not trigger this rule; it is the
reconstructed signed leading kernel that matters. The strict API mode
continues to reject negative densities. Equivalent daughter orientations
must share a common nonzero kernel sign, which cancels in their auxiliary
marking probabilities; incompatible signs remain an explicit error.

The massive-FSR second solution can have a valid projected Born but no
soft counterevent. Born identity checks therefore use the inverse map's
Born snapshot, including its incoming fractions, rather than the invalid
counterevent slot. For a single physical recoil particle the MC kernel
uses its on-shell model mass and explicit momentum, avoiding a false
spacelike invariant from subtracting boosted vectors. Annotated hard-input
preflight allows accumulated boost roundoff at the event scale while
retaining the tighter plain-LHE/endpoint check. Local unit-Delta replay
is created only when the MC contribution's FKS partitions are nonzero.

The saved failure-point checks and their limits are documented in
[`audits/macnlops_matching_fixes_20261025.md`](audits/macnlops_matching_fixes_20261025.md).

### One collected MAcNLOPS event file

After successful automatic matching from `launch aMC@NLO -f` or
`shower RUN`, the randomized, collected output is

```text
Events/RUN/macnlops/events.lhe.gz
```

The original ordinary MC@NLO source remains at
`Events/RUN/events.lhe.gz`. The MAcNLOPS output is a separate file; it does
not replace that source. The existing LHE collector combines and randomizes
the retained channel events without renormalizing their signed weights to
the survivor count. Per-channel event files, rejection decisions, metadata
and logs remain available. `Events/RUN/macnlops/summary.json` lists the
collected output and the individual channel records. `collection.json`
records the shuffle seed, original matching seeds, combined counts and
signed weight sums. The combined `<init>` block contains all process IDs;
its cross sections add without survivor-count renormalization, and its
finite-sample channel errors add in quadrature using the original trial
counts, including rejected events.

The explicit CLI publishes the same single-file result under its chosen
output directory. From the generated process directory, for example:

```sh
python3 bin/internal/macnlops.py \
  --process-dir . --pythia8 /path/to/pythia8318 \
  --output-dir matching --seed 734101 --signed-residuals \
  Events/RUN/events.lhe.gz
```

The collected file is `matching/events.lhe.gz`, with its output/channel
listing in `matching/summary.json`. Multiple input channel paths can be
supplied to the same invocation; their individual records are retained
alongside the collected file. This is the retained first-emission endpoint:
H uses its original hard state and S has zero or one accepted QCD emission.
Collection does not add a full shower or hadronisation.

### Continuing the retained LHE endpoint

The first-emission file is an event handoff, not a Pythia checkpoint.
Its signed weights, matching decision and current particle momenta must
survive subsequent processing. Its **current shower ceilings** must also
be distinguished from the original source SCALUP and source Delta scales.
Restarting a retained S event from its old source scale would generate a
second opportunity for radiation in the interval already tested by the
first competing ISR/FSR step.

The endpoint writer distinguishes three cases:

| First endpoint | Global continuation ceiling | Remaining constraints |
| --- | --- | --- |
| H, not showered by the matching step | The effective imported source scale | Preserve the source production/dipole constraints, capped by the global source scale. |
| S with a first emission | The recorded native evolution `pT`, not a jet transverse momentum | Export the native remaining ISR/FSR bounds in the current particle order and cap them by this global ceiling. |
| S with no emission | A strictly positive scale below both measured QCD cutoffs | Production QCD evolution is exhausted; the continuation must not draw another first emission. |

The current numerical sentinel is half the smaller of the measured ISR
and FSR cutoffs. Reusing it assumes the continuation keeps those cutoffs
or increases them; changing the infrared resolution defines a different
shower. Neither negative S weights nor negative H residuals justify
retrying a no-emission event or changing an event weight.

For first endpoints the writer records

```xml
<macnlops_continuation version="1" state="unshowered|emitted|exhausted"
                      scale="..." source_scalup="..."
                      cutoff_isr="..." cutoff_fsr="..."/>
<scales scalup_i="..." scalup_i_j="..."> </scales>
```

In a produced event there is one state string and actual numeric leg
indices. `scalup_i` supplies a positive current per-particle anchor;
`scalup_i_j` is a current directed FSR-dipole bound. Source `<scales>` blocks
move to `<macnlops_source_scales>`, where stock Pythia will not mistake
their old indices for the new event. The hard annotation and, for Delta,
the source replay packet remain source provenance. An emitted S endpoint
must never be replaced by the Born point retained in that annotation.
The nominal weight and alternative weight records are preserved; updating
SCALUP and scale metadata is an intentional part of the event handoff.

For an emitted S event, the implementation reads the live stock 8.318
shower dipole listings at the first-step hook. A temporary numeric-output
facet captures double precision rather than the listings' usual three
decimal places. The writer maps their current radiator/recoiler indices
to the saved external event and rejects ambiguous or unmappable bounds.
This captures an untouched dipole's lower starting scale as well as the
new global ordering ceiling. A scalar SCALUP alone would not retain that
lower bound. This extraction uses the public, version-checked native
listing interface; it does not patch the shower or change trial draws.

Stock 8.318 interprets these quantities differently for ISR and FSR.
`LesHouches.cc` reduces positive scale attributes to a minimum per particle
and accumulates a global maximum while visiting the attributes. That
maximum can exceed the maximum of the final per-particle minima. H Delta
endpoints therefore keep positive global-valued anchors and the source
pair constraints; source zero pairs stay zero, preserving the reader's
ignore-zero production-scale semantics.
`SimpleSpaceShower::prepare` then uses each incoming particle's production
scale. Its ordinary global-recoil ISR is not a separate evolution for
every directed colour connection. `SimpleTimeShower::setupQCDdip` can
additionally recover the directed `scalup_i_j` value. Both showers compete
below the current global ceiling, with their own kinematic bounds.
Consequently a Delta restart must cap **all** active regenerated scale
attributes; lowering only the event-header SCALUP can be undone when the
LHE reader recomputes that global scale.

An external consumer of the derived endpoint, for **either** ordinary or
Delta input, must honor its current production and dipole scales. The
relevant stock settings include

```text
Beams:setProductionScalesFromLHEF = on
Beams:setDipoleShowerStartingScalesFromLHEF = on
SpaceShower:pTmaxMatch = 1
TimeShower:pTmaxMatch = 1
SpaceShower:pTmaxFudge = 1.
TimeShower:pTmaxFudge = 1.
TimeShower:limitPTmaxGlobal = on
LesHouches:setQuarkMass = 0
LesHouches:setLeptonMass = 0
```

These settings concern a **derived-endpoint consumer**, not a change to
the original ordinary-source matching profile and not a reason to label
an ordinary source as MC@NLO-Delta. They require an LHEF 2/3 file and a
reader supporting its production-scale extension, plus the same physical PDF, coupling,
masses, allowed flavours, cutoffs and QCD-only scope as the handoff.
The normal MG5 command `shower <run>/macnlops` recognizes the collected
accepted file, validates its continuation packets and enters the ordinary
shower stage directly. It does not repeat first-emission generation or
removal/veto. It enables both scale-reader switches, uses local recoil
and the recorded cutoffs, preserves the endpoint masses and keeps
hadronisation, MPI, primordial transverse momentum and QED radiation off.
The source run's split files are not reused for this different event
population. The continuation loop processes the requested number of retained
events and stops on a shower failure instead of skipping an accepted input.
Older matched files without the corrected continuation
contract and recorded shower profile must be regenerated from their
original hard source.
The accepted LHE header carries the measured first-stage shower settings,
with the exact input coupling, masses, flavour policy and cutoffs. The
second stage restores these settings before applying its continuation
scale and recoil choices. It also checks the Pythia-library, native
LHAPDF-backend and PDF-data fingerprints, and supplies the validated
library and PDF search paths to the shower job. A later edit to the
process's shower or run card cannot
silently change this physics profile. In particular, the normal shower
defaults of one-loop running and `alphaSvalue = 0.118` must not replace
the first-stage coupling. The native consumer restores
`SpaceShower:pT0Ref = 0` with `forceParm`, because ordinary card reading
clamps that value to a positive stock XML minimum.

Original Delta H records can have `SCALUP = -1`; their directed scale
matrix supplies the physical shower starts. This one sentinel is accepted
only for Delta H sources and retained as inert `source_scalup` provenance.
Every active endpoint scale remains strictly positive. Continuation also
retains the absolute nominal LHE weight: for LHA strategy `-3`, Pythia's
`Info::weight()` supplies only its sign, which is insufficient for the
sum-normalized accepted sample. Both the HepMC-only and Fortran-analysis
drivers enforce one pass over retained inputs and stop on a failed or
skipped input. The Fortran-analysis route also writes one-based event
numbers, since its histogram normalization uses `NEVHEP` as a count.
Import must preserve the current endpoint masses and momenta. In
particular, a native forced-heavy sister can have mass 1.2 or 4 GeV while
the five-flavour particle-data hard mass is zero. The default quark-mass
repair can remass that sister and shift other momenta. Disabling that
repair preserves the supplied event masses; it does not remove native
heavy-flavour evolution thresholds.

Zero is not a reliable exhausted-event sentinel. The stock LHE reader
ignores nonpositive production-scale attributes, while the global-FSR
starting-scale logic can substitute its factorization scale when SCALUP
is nonpositive. The custom continuation marker is descriptive metadata;
a generic stock reader does not act on it. The positive below-cutoff
numeric ceilings, compatible reader settings and unchanged cutoffs carry
the no-further-production-radiation contract.

Even with correct numeric ceilings, an LHE restart is not an assertion of
the same conditional multistep shower distribution. Ordinary particle
records do not restore the beam's selected valence/sea/companion state,
native ancestry and hard-line membership, recoil counters, cached trials
or RNG state. In particular, restarting `globalRecoilMode=1` reinitializes
`nGlobal` and `hardPartons`; it need not reproduce the local/global choice
after a previously accepted FSR branch. The export therefore specifies an
ordered continuation with recorded bounds, not bitwise replay or a proof
of statistical equivalence to an uninterrupted shower.

The frozen first/full comparisons used a retired validation driver that
completed the same live Pythia event after saving its first endpoint.
Those archived outputs describe a different continuation procedure; the
supported workflow writes the accepted LHE and showers it separately.

Lightweight tests cover continuation parsing, preservation through rejection
and collection, shower dispatch and event-loop accounting. Small C++ fixtures
exercise the native listing parser, precision restoration and endpoint scale
mapping without instantiating Pythia. The native driver also passes C++11
syntax checks. The bounded native checks below exercise actual re-import
and production shower evolution. The older frozen production comparisons
do not validate this handoff.

### Bounded native checks, 23 October 2026

Independent generated-matrix-element checks compare the evaluator's Delta
H density with the actual source `repartition_MC_H` contribution, including
native stopping scales and the shipped Sudakov tables. The maximum of
`abs(H_source - (R-D))/max(1,abs(R),abs(D))` is `3.06e-14` for 100 DY points,
`1.06e-14` for 100 top-pair points and `5.88e-12` for 50 massive-bottom
four-flavour Wbb points. Replaying an auxiliary packet and reversing query
order give identical results. Changing the auxiliary seed changes the
Delta H term while leaving the physical real and shower kernel densities
unchanged; setting Delta to zero gives zero H. These checks use real-only
exports rather than integrated/unweighted production samples.

Stock Pythia 8.318 also consumes 620 native endpoints: 200 events from a
fresh full-NLO Delta DY source and 420 controlled ordinary/Delta kinematic
fixtures covering DY-like, top-pair and four-flavour bottom-pair states.
The checks preserve imported momenta, masses, colours and raw LHA weights,
and inspect 2,789 initial directed scale caps and 26,926 subsequent ISR/FSR
branch ceilings. All 60 exhausted fixture endpoints have no further
production radiation. This tests the transport contract; it does not test
the generic HepMC writer's weight normalization or certify rejection
densities for synthetic fixtures.

Fresh full-NLO sources also complete first-emission matching, collection
and the normal process-directory HepMC continuation command:

| Sample | Source events | Accepted S | Accepted H | Negative H removed | S vetoed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary DY | 200 | 172 | 24 | 4 | 0 |
| Delta DY | 200 | 173 | 26 | 0 | 1 |
| Delta top pairs | 100 | 78 | 16 | 4 | 2 |

Every accepted event is written once and its nominal weight agrees
exactly with the retained LHE. The final HepMC signed-weight sums are
`1926.197274`, `1942.928645` and `861.3662992 pb`, respectively; their
cumulative cross-section fields agree to floating-point accumulation
accuracy. Hadronisation, MPI, primordial transverse momentum and QED
radiation are disabled. These independent, small source samples are
workflow checks, not paired cross-section or distribution comparisons.
Negative S weights remain; these checks do not demonstrate that every
possible negative H can be removed without a signed residual.
The separate Fortran HwU rate check for the same 94 accepted top-pair
events gives `861.36630 pb`, agreeing with their signed weight sum within
the histogram's printed precision. This catches an otherwise separate
event-count normalization error in the analysis route.

The artifacts for these bounded checks are under
`/export/tmp/rikkert/macnlops_deep_checks_20261023_0537`, including
`evaluator/summary.json` and `native/native_summary.json`. Small source
samples and selected phase-space points cannot establish a global veto
bound, the frequency of negative H residuals, or precision agreement of
final showered distributions. An LHE restart also retains the conditional
shower-state limitations described above.

### Historical shared-input scientific comparisons

This section describes the frozen study, whose native first/full driver
and production runners have been removed. It preserves the comparison
definition and its accounting evidence, rather than giving commands to
run that retired workflow. New production uses the first-emission matching
and separate LHE continuation documented above.

The frozen comparison started from one ordinary MC@NLO source for each
process. Its complete signed H density retains the ordinary counterterm,
including its G function, spin terms and the ordinary `bogus_sudakov`
profile. Writing the history sum schematically,

\[
 H(q)=\sum_h p_{{\rm bogus},h}(q)\,[R_h(q)-D_{G,h}(q)].
\]

The two compared branches read the **same file**, used the same preferred
hard annotation and shared one native shower of each source event. The
ordinary event body and weight are unchanged. The preferred hard view is
the versioned pre-mass-shuffle annotation; this is an explicitly qualified
import contract, rather than an inverse projection of an arbitrary older
stock LHE file. The global-partition certificate is required before
applying a global negative-H density to separately sampled source records.

The archived production prepared each source, qualified its complete H
population, and performed one native first/full evolution followed by
rejection. Each 200,000-event source was divided into 40 chunks. The
frozen Wbb profile used four incoming/splitting flavours and its nf4 PDF
and massive-bottom hard model, including the patched coupling described
in the historical runtime notes. Distinct process seeds and immutable
checkpoints are recorded in the archived production manifests.

Before releasing **any** shower in a process, all 40 source chunks had to
pass the source-density gate. It evaluated every original H, including
both signs and any zero-weight records, with the exact measured profile.
Finite `R`, `D` and `K`, nonnegative `R` and `K`, and agreement of
`sign(R-D)` with every nonzero H weight are mandatory. A zero global H
with nonzero source weight fails. The aggregate count must match the
entire original source's S/H counts. `H_minus>K` is informational under
the selected signed-residual policy: it is neither a gate failure nor
permission to modify a density. A negative H with `K=0` is recorded as
requiring complete signed survival, with an undefined `H_minus/K` ratio.

Each chunk stores `source_h_densities.jsonl` and an immutable
`gate_report.json`; the full process stores `source_h_sign_gate.json`.
The cache binds the source-chunk bytes, original full-source manifest,
generation profile, exact support/settings, evaluator binaries and
shared libraries. Its point key contains the complete preferred hard
annotation and the process, weight, SCALUP and coupling header fields.
The normal mass-shuffled particle body is deliberately excluded: the H
matching endpoint can have a different body while representing the same
annotated hard point. Every H cache lookup must match that key; an
evolved shower point cannot borrow its source density. S evaluations
remain live. The runner externally pins the gate-report hash; a stale,
changed or orphaned cache never silently falls back to another H RPC.

The frozen native profile explicitly probed 31 off switches covering
hadronisation, hadron decays/rescattering/QED, Bose-Einstein/deuteron and
rope effects, MPI and beam remnants, primordial kT, colour reconnection,
resonance decay/radiation, QED and weak showers, and MPI dipoles. Their
exact names and effective false values are saved in
`support.qcd_only_off_switches`; QCD ISR/FSR remains enabled. Native
history and endpoint guards also reject extra hard systems, photons,
hadrons or changed electroweak particle species. This verifies the
declared QCD parton-level scope independently of plotting selections.

Native heavy-flavour threshold conversions remain part of that shower.
No source event is dropped, masslessly projected or reshowered to remove
them. The first-emission metadata retains the source hard point, actual
native branch indices, emitted species/masses, evolution scale and the
profile's heavy-mass floors. The decision file records the actual
matching boundary; protected sub-boundary S events survive without an
ordinary massless-real density request. Its
`below_matching_boundary_S` count includes all such protected histories
and must not be relabelled as a count of forced conversions alone.
Massive emitted-sister histories can be counted independently from their
native branch records. Negative H outside cancellable K support remains
a signed residual. These records distinguish native threshold activity
from H rejection and from a missing ordinary inverse map.

No H rejection or S veto acts on the baseline. The treatment uses the
declared signed-residual rule, with `C=min(H_minus,K_g)` and the actual
history gate in the S acceptance. Negative H can therefore survive. It is
incorrect to implement the H-only control by deleting every negative H
when the selected treatment keeps a residual. Exact pre-veto equality of
the branches follows from using the same events and native momenta. It is
not a claim that their signed rates after rejection must agree at finite
coupling: that requires the cancellation-density conditions derived above.

For original event weights `w_i`, acceptances `A_i` equal to zero or one,
and the H/S indicators, the four treatment masks are

\[
 I_i=1,\qquad
 M_{H,i}=1_{S,i}+1_{H,i}A_i,\qquad
 M_{S,i}=1_{H,i}+1_{S,i}A_i,\qquad
 M_i=A_i.
\]

The reference is exactly `I`. Each bin, including every underflow,
overflow and missing entry, consequently obeys
`MAcNLOPS = H-only + S-only - input`. Also
`H-only = input - source negative H + retained negative H`. These are
event-selection identities for counts, signed weights and sums of
squared weights; they do not impose cancellation of the integrated H and
S changes. Both first and full stages use the same decision, with no
survivor normalization, and must have identical total counts and weights.

The plotter measures the event covariance directly,
`Cov(a,b)=sum_i w_i^2 M_a,i M_b,i` in each bin. In particular, shared
baseline/input covariance equals their variance, and covariance of a
selected curve with the baseline equals that selected curve's sum of
squared weights. The ratio variance is

\[
 {V_a\over b^2}+{a^2V_b\over b^4}
       -{2a\,\mathrm{Cov}(a,b)\over b^3}.
\]

Thus the input/baseline ratio is identically one with zero paired ratio
uncertainty. Signed cancellations can still produce large treatment
ratios and errors near a zero denominator. Undefined ratios are masked;
negative bins, unrestricted ratios, zoom overflow markers and raw moments
are retained. The conditional-veto diagnostic separately records
`sum(w*p)` and `sum(w^2*p*(1-p))`; this conditional variance is not added
again to the observed histogram `sumw2`.

In `analysis/datasets.json`, each of the four process entries declares
`common_source_mode: true`, `source_construction: "ordinary-mcatnlo-v1"`
and `signed_residuals: true`. Its `primary` list contains all unrejected
first/full chunks, decisions and full-audit paths. It has no `reference`
list: the plotter derives that curve in the primary read, rather than
loading or assuming an independent sample. A nonempty `reference` list in
this mode fails validation. `primary_complete` and `expected_events`
refer to the 200,000 events **before rejection**. DY, ZZ and top pairs are
five flavour; W+bb is genuinely four flavour. A production publication
requires all four processes and both stages. Legacy independent reference
datasets remain supported and retain zero reference cross-covariance.

The scientific panels retain the full observable set: broad and soft
leading radiation-jet spectra, selected jet rapidity, system recoil and
mass, pair angles and rapidity separation, individual-object spectra,
and Wbb b-pair and all-parton-jet controls. Source-sign panels include
the retained negative-H curve when signed residuals are declared. Every
observable exports counts, signed weights, `sumw2`, covariance and range
flows. Native hard-integration errors remain separate source metadata.

The retained plotting and accounting tools can analyze the existing
archived datasets without rerunning the retired native production:

```sh
python3 docs/audits/plot_macnlops_validation.py \
  --datasets /path/to/results/analysis/datasets.json \
  --output-dir /path/to/results/plots --workers 4
python3 docs/audits/audit_macnlops_validation.py \
  /path/to/results/plots \
  --output /path/to/results/analysis/final_histogram_audit.json
python3 docs/audits/package_macnlops_plots.py /path/to/results/plots
```

The auditor independently checks source-weight closure, paired baseline
covariance, partial-H masks, first/full totals, portable flow tables and
ratio uncertainties. Original primary-source weights anchor both input
and baseline totals. A preview needs explicit `--allow-incomplete` and
does not qualify as a final validation. Publication of the plotting
directory is atomic: a failed process or missing stage discards staging
outputs instead of publishing a partial production report. These checks
certify accounting and the declared comparison, not physical-distribution
agreement or exact finite-cutoff NLO accuracy.

Only a complete strict report can be packaged. The book retains nine
selected vector panels per process and stage, including soft-jet signs
and rejection mechanisms: 72 pages in total. Its contents and portable
ZIP manifest explicitly distinguish the shared 200,000-event source from
the derived baseline curve. Four common sources mean 800,000 physical
source events, viewed at two shower stages; counting the baseline as an
independent extra source would be incorrect.

### Historical resolved-source workflow

This section records the earlier resolved-source production and its
independently generated ordinary references. It used the explicit
legacy source selection rather than a shared-input comparison.
Its native first/full driver and production runner have been removed.
That production used an isolated compensated Pythia installation
made with a now-removed patch builder. Reproducing its exact runtime would
require its frozen external archives. The active stock workflow above has
no such prerequisite.

The archived setup used `macnlops_resolved_source=True`,
`mcatnlo_delta=False`, no merging, and sum-normalized events with the
ordinary central LHAPDF member. Both process directories contained exactly
200000 events under `Events/source_200k/events.lhe.gz`; the independent
reference had `macnlops=False`. The retired runner recorded the first
endpoint and completed shower in one native evolution, then applied the
same rejection decision to both records. Each chunk preserved all original
weights without survivor normalization. Immutable
prepared inputs, executable/source/library hashes, native metadata and
per-event rejection decisions accompany the outputs. The archived runner
reused completed checkpoints and withheld a complete report on any failure.
It also checked that all active hard quarks were massless (and that
the bottom was massive in 4FS), and compared every quark mass to the actual
Pythia probe. Card preparation must use keyed access such as
`card['mass'].get((6,))`: a bare integer index addresses the block's list
position and can silently edit a different PDG entry. Before integration,
the production audit compares dependent card masses and widths with the
restricted model as well as with the shower probe.

The retired full-shower driver used Pythia global recoil mode 1 with
`nMaxGlobalBranch=1` and `nPartonsInBorn=-1`. Before the first accepted FSR,
all coloured Born lines belong to `hardPartons`, `nGlobal=0`, and rejected
trials remove no hard line. Consequently its first S emission agrees with
the mode-0 global map used by the source evaluator, including massive
bottom/top emitters. After one accepted global FSR, native evolution uses
local recoil. If ISR wins first, that S matching endpoint is already fixed
before this later FSR choice. Applying the mode-0 global map to every
subsequent emission is not the validation continuation.

In this historical production the compensated ordinary trial headroom
remained 100 throughout. For S
events, all diagnosed trial bounds and nominal auxiliary weights are
checked through the saved first endpoint, including no-emission outcomes.
This chronological check conservatively includes any losing look-ahead
trial below the winning scale. Later native trial-bound warnings, and all
H continuation warnings, are counted per input in
`first.jsonl.full-audit.jsonl`; they do not cause input-event removal.
Their cumulative logger counts are differenced per event, so a recorded
continuation warning cannot contaminate the next S endpoint's check.
This is an explicitly audited native continuation, not a certification of
an analytic late-shower kernel near problematic low-scale PDF ratios.
The previously derived physical-boundary/power limitations still apply
to subthreshold measurements and normalized heavy-flavour conversions.

The retired full writer captured Pythia's actual imported production/decay
tree. It followed resonance ancestry through shower copies and restored
daughter momenta with Pythia's boost convention. This preserved ordinary MC@NLO
reference remassing and recoil before showering. In the earlier resolved
validation the H matching endpoint remained its original source real
point. The annotated common-input workflow instead uses the same preferred
hard presentation for both branches. Full-shower momentum conservation
is audited with Pythia's numerical tolerance; native accumulated roundoff
is retained rather than projected into a different event.

Archived production plotting required 200000 primary and reference events
for each of DY, ZZ, top pairs and W+bb, at both first and full stages. It checked
event identities, weights, rejection decisions and continuation audits
before publishing the report. `--allow-incomplete` is only for visibly
labelled rendering/smoke checks.

Here `21150` selects the installed central MSTW2008nlo90cl member; another
ordinary MSbar set can be selected by its LHAPDF ID. This mode generates
`D*=(1-P)R+P K` as derived above when
`macnlops_resolved_source=True`. Ordinary `macnlops=False` generation
retains its original G/P counterterm and stock writer conventions. The
`macnlops=True` workflow also forces complete
native-history and Born-colour sums before unweighting. Earlier resolved
validation files preserved hard momenta in their ordinary particle rows.
The current ordinary writer retains its standard reshuffle and supplies
the original hard point through the versioned annotation.

Before integration, generation builds and probes the actual one-step
shower. `Cards/macnlops_profile.json` records its effective ISR/FSR
cutoffs, heavy thresholds, beam extraction bounds, particle masses,
PDF member and library hashes. These values populate the Fortran run
include. In the archived runtime the ordinary trial headroom defaulted to
100 and the historical `macnlops_trial_headroom` field selected it before
generation, with reciprocal acceptance compensation. That field and
runtime are no longer part of the active stock interface. Historical
bound failures aborted rather than invoking adaptive retries or uncertainty
weights. Optional input in `Cards/macnlops_custom.cmnd` was restricted to numeric
`PDG:m0`, `SpaceShower:pTmin` and `TimeShower:pTmin` assignments; all
other commands are rejected before Pythia initialization. The hard-model
masses and requested matching cutoffs are supplied last by the runner.

Generation writes source-state manifests beside both the channel files
and the collected file. The version-2 manifests include the matching
profile, settings and source kernel. Older manifests describe a different
source construction and require regeneration. The validated profile uses
proton or antiproton beams and a single central MSbar LHAPDF member.
Ordinary massive emitted sisters need a massive inverse map not provided
by the current ISR evaluator; unsupported kernels fail explicitly.

The archived per-channel outputs retain the original banner, init block,
signed weights, proposal and decision records, native evaluator logs and
effective settings. H events passed through the matching stage without
showering; S events received zero or one accepted QCD emission before
whole-event rejection without retry. The resolved-source bounds removed
negative H, while negative S remained. This historical result is distinct
from the ordinary-source signed-residual prescription used by the current
workflow. Both used the same ordinary LHAPDF set/member and its alpha-s
input, hard-card external masses and no MC PDF scheme transformation.

## Historical certified ordinary-ISR proposal

This section documents the removed proposal extension and the reasons it
was introduced. Its implementation and mathematical qualification apply
to the frozen patched study only. The active stock driver uses native
sampling with recorded warnings and has no bound-provider dependency.

The larger validation campaign exposed a failure of the native cached-PDF
trial envelope even with headroom 100. A first accepted backward
`g -> s sbar` conversion at `pT = 1.11405 GeV` had trial acceptance
`2.7073`, above the physical matching boundary. The mother gluon PDF at
its actual larger momentum fraction was much larger than the cached
value. A fixed multiplier of a cached value cannot guarantee an envelope,
especially when the cached value vanishes. This is a proposal failure;
it does not justify changing the ordinary MSbar PDF, the physical shower
kernel, the source, or an event weight.

The optional capability-831803 library instead bounds the actual PDF
interpolation polynomial on deterministic evolution slabs. Write
`F_d(x,t)` for the native positive daughter component and `F_m(y,t)` for
the mother component, including the native valence/sea tag. For a slab
`I=(L,U]`, it supplies

\[
 D=\max\!\left(\epsilon_{\rm PDF},\inf_{t\in I}F_d(x,t)\right),
 \qquad M_m\ge\sup_{t\in I,\ x\le y\le x_{\max}}F_m(y,t),
 \qquad \frac{F_m(y,t)}{\max(\epsilon_{\rm PDF},F_d(x,t))}
 \le\frac{M_m}{D}.
\]

Here `epsilon_PDF=1e-10` is the existing physical Pythia denominator
floor. For proton u/d the full component is positive valence plus
positive sea, which can differ from the positive part of the total NLO
PDF. The bound callback follows this distinction and Pythia's heavy
antiquark symmetry settings. It does not provide or replace the physical
PDF values used in the final acceptance.

For flavour-preserving channels the additional factor
`C=max(1,M_d/D)` multiplies the integrated trial and divides acceptance.
For flavour-changing channels the trial uses `M_m/D`, including the
mother-flavour selection probability, and the reciprocal factor cancels
in acceptance. Thus integrated trial times normalized z proposal times
flavour probability times acceptance reproduces the intended ordinary
AP kernel times the actual native PDF ratio. The independent regression
checks 5,600 products for gluon, sea-quark, valence-quark and both
flavour-changing families, with three, four and five active mother
flavours and the ordinary massive coefficients. Its maximum relative
product residual is `6.62e-16`.

The initial slab spans half the current upper evolution scale, shortened
by native cutoffs and flavour thresholds. Refinement uses known bounds
before drawing a trial. If an optional tighter daughter bound cannot be
obtained, the implementation consistently restores the original valid
slab. Crossing a slab edge records survival and starts the next slab
there. Consecutive target Sudakov factors multiply, preserving the
original target law. The second-order alpha-s rejection loop stops at
the certified slab floor. Native threshold routing retains priority at
equal edges; an empty domain below a competing endpoint returns before
requesting a PDF rectangle. The fixed 19-event regression fails on its
last event with the earlier implementation and passes all first/full
records with the final version.

The selected LHAPDF logcubic interpolator is reconstructed from its
stored coefficients. Bernstein coefficient hulls enclose each complete
cell and its interior extrema; fixed-x daughter intervals are restricted
in Q before bounding. Duplicated threshold knots and the two-Q-node
bilinear fallback are covered. Outward rounding and a conservative
absolute guard operate under explicitly checked finite-data,
conditioning and floating-point assumptions. This is a numerical
certificate for the selected grids and toolchain, not an arbitrary-
platform exact-real interval-arithmetic claim. Invalid bounds and a
trial acceptance above unity still abort. The optional library rejects
unsupported profiles, including PDF extrapolation, external PDF
pointers, MPI, alternative recoil, MEC and uncertainty variations.

The actual physical PDF-ratio block and normalized forced-heavy sampler
remain byte-identical in both five- and four-flavour libraries. The
ordinary tiny-positive-kernel early return and finite tiny-daughter
retry cap are removed so they cannot discard support of the regulated
target. Full/core driver sources, source weights, physical scales,
PDFs, masses and momentum maps are unchanged.

The local final-version pilots process the complete previously failing
5,000-event reference chunk at its original seed and 200 genuine
four-flavour Wbb events. Both retain every input and its signed weight
at first/full stages, with no native warning. Eighty actual new ISR
points close the absolute kernel coefficient to `6.67e-16` and imported
Born invariants to `6.99e-13`. These are implementation and accounting
checks, not completed production or distribution-agreement results.

Because a failure depends on a generated trial path, keeping successful
chunks from the faulty proposal and replaying only failed chunks would
condition the ensemble on success. The release therefore requires a
fresh replay of all four paired source cohorts under the same finalized
proposal. Original hard events and seeds remain fixed. The runtime
manifest pins the new executable, Pythia and bound-provider libraries,
PDF bytes, mathematical review and local checks, while independently
binding every original source, chunk, setting and frozen driver source.
Whole-cohort completion and final histogram checks remain separate gates.

## Historical process-level validation

The small-sample checks below preceded the larger campaign and its
certified-proposal correction. They document source, map and rejection
tests; they do not certify the earlier cached-PDF proposal globally.

The 20 October independent shower audit of the existing full-NLO Drell–Yan
sample completes all eight nonempty channel files with fixed headroom 100:
1,000 source events, 850 one-step S emissions and 150 unchanged H events.
Two native massive threshold conversions are preserved. The largest hard
momentum imbalance is `2.36e-9 GeV`, and the largest event-scale-normalized
mass-shell residual is `1.86e-13`. A separate headroom-10 scan catches a
native ISR envelope violation and correctly aborts; it is not used for the
validated production profile.

The fresh resolved-source full-NLO Drell–Yan run produces 1,000 events
with MSTW2008nlo90cl/0 and completes strict rejection on all **seven
nonempty** original channel files (the eighth has zero source events).
There are 159 H and 841 S proposals. All 36 negative H events are removed;
123 positive H and 644 S events survive, including 61 negative S events.
The retained cross-section estimator is the original-weight sum
`1553.9530275 pb`; it is not renormalized to the source integral.
The independent audit reproduces every decision and finds maximum hard
momentum imbalance `1.78e-9 GeV` and normalized shell residual `2.78e-13`.

A separately labelled auxiliary validation selects the 90 original
charm- or bottom-Born S events, retaining their complete event blocks,
scales and weights. One fixed independent shower seed gives two native
massive bottom conversions, at evolution scales `5.21156` and `4.78721
GeV`, below their `5.65685 GeV` history boundary. Both pass rejection
with probability one without a massless density request. The same 90
proposals are reused after correcting a too-tight soft-charm shell
roundoff check to use the full hard-system scale; no shower retry is
performed and production outputs remain unchanged. The auxiliary
rejection retains 77 events and rejects 13.

The production and auxiliary independent reports are
`/export/tmp/rikkert/macnlops_validation_20261020/dy_resolved_matching_v2/independent_audit.json`
and `dy_resolved_heavy_validation/independent_audit.json` under the same
validation directory. The mathematical scope is the resolved-source
construction described above, including its explicit physical-cutoff
and threshold corrections; a successful rejection bound alone is not
a proof of exact NLO accuracy for measurements below those scales.


The full-NLO five-flavour `pp -> ttbar` validation independently completes
200 input events with the same ordinary MSbar PDF and the corrected QCD
profile. All 66 H proposals are byte-identical to their source records;
all 38 negative H events are removed. The result retains 105 S and 28 H
events, including 40 negative S events. Its 20 actual massive top-FSR
histories pass the independent inverse-map checks, with maximum relative
evolution-scale error `4.53e-14`; no photons are present. Momentum closure
is better than `1.75e-11 GeV`, and retained input weights sum to
`816.215847 pb`, equal to the declared init sum. This small sample checks
the implementation and support, rather than establishing phenomenological
precision. A restricted gg-only export is not closed under the enabled
five-flavour inverse ISR and correctly fails routing; it is not used as
a successful production sample.

The copied coloured output and report are under
`/export/tmp/rikkert/macnlops_validation_20261020/pp_ttbar_matching`, with
original source provenance retained in the reports. The full overview is
`validation_overview.json` in that validation directory. Focused compiled
checks and all six fresh-export tests pass, including paired real/MC
record conservation, grouped-flavour thresholds, original-source
regression, history-dependent rejection and missing-profile startup
rejection. Ordinary `macnlops=False` source results remain unchanged in
the numerical comparisons.
