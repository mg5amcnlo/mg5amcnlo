# Shared Born support and native MC histories

New NLO outputs use an output-wide Born library and global history registry.
`repartition_MC_H` evaluates histories in their native contexts, including those
whose Born evaluator belongs to another subprocess directory. It implements
`Hhat_a = S_a sum_b P_b (S_b R - M_b)`. The completeness guard checks global
resolution; missing or ambiguous required histories still stop the calculation.

New NLO outputs contain `Source/BornSupport/registry.json`, private provider
sources, native-context metadata, and local compatibility wrappers. HELAS tags
identify equivalent evaluators, with additional checks of labelled external
order, splitting orders, normalization and extra-counterterm roles. IDs are
assigned before export workers start; each worker writes its own metadata file.
User process numbers stay in the context metadata and do not cause a duplicate
numerical provider.
Finalization merges those records before output packaging. Different native
contexts retain separate metadata even if their evaluator is shared.
Existing output directories must be regenerated to receive the library and
registry; no new run-card option is needed.

The registry matches real flavours with fixed incoming ordering and preserves
particle tags and labelled emitter orientations. It records explicit momentum,
flavour, coupling-name and squared-order maps. It records missing or ambiguous
ownership; `born_query(..., require_complete=.true.)` rejects either condition.
Fixed-order export can still be used for deliberately incomplete Born families
and for splitting modes that are unsupported by MC@NLO.

## Fortran API

`mc_born_types` defines `BornModelState`, `BornRequest`, `BornResult`,
`BornMetadata` and `BornHistory`. `mc_born_support` exports:

```fortran
call born_model_dimensions(nreal, ncomplex)
call born_query(provider, context, metadata, status)
call born_evaluate(provider, context, p, state, request, result, status)
```

The query optionally accepts `details` and `require_complete`. `nexternal` is
the real-emission multiplicity, so Born momenta have shape `(0:3,nexternal-1)`.
`request%sector` is the native context's FKS index. The request can additionally
select helicity weights, a colour or charge correlation, and an existing extra
Born counterterm. Charges must be supplied explicitly. Results include split
orders, spin counterterms, diagram/flow weights and requested correlations.
Only local compatibility wrappers copy results into legacy Born COMMON blocks.
Metadata exposes the native flavour count, initialized topology arrays and
Breit-Wigner constraints. `supports_correlations` is false for LO-only exports,
whose legacy order tables have no NLO slots; correlated requests then return
`BORN_MISSING_CORRELATION` instead of entering those unsupported routines.

Status codes are `BORN_OK=0`, `BORN_UNKNOWN_CONTEXT=1`, `BORN_BAD_SECTOR=2`,
`BORN_INVALID_REQUEST=3`, `BORN_MISSING_CORRELATION=4` and
`BORN_INCOMPLETE_HISTORY=5`.

The generated model transfer covers typed COMMON members from `coupl.inc` and
`input.inc`. Providers have private COMMON blocks; model and HELAS symbols are
hidden in the shared library. Evaluation transfers model state on every call,
including cache hits: another provider can have installed different model
COMMON values in the meantime. Each provider retains its own Born amplitudes
with an exact key consisting of context, sector, all four components of every
momentum, and all captured real and complex model values. A hit reconstructs
the contractions, coupling-order buffers and flow weights from those amplitudes.
Correlations therefore use the matching Born without repeating its HELAS calls.
Caller cache flags are never trusted.

Helicity masks are learned separately by each evaluator. Their validity key is
separate from the full-state amplitude key: a consistent positive, nonzero
rescaling of the strong coupling can preserve a learned mask. Export derives
each used coupling's power of \(g_s\) from its UFO expression and checks the
actual HELAS diagrams to prove that each squared coupling-order component
scales homogeneously. If \(c_v=g_s^{d_v}\bar c_v\), this gives

\[
  B_{h,q}(\lambda g_s)=\lambda^{D_q}B_{h,q}(g_s),
  \qquad \lambda>0,
\]

so its zero/nonzero status is unchanged when the normalized couplings
\(\bar c_v\), masses, widths and other relevant inputs remain equal. These
comparisons are exact; normalization roundoff can cause harmless relearning.
Unknown coupling dependencies, nonhomogeneous order components and custom
HELAS dependencies fall back to the full-state check. Unused virtual-model
parameters do not invalidate a certified tree-level mask. The generated
`Source/BornSupport/helicity_signature.json` records the selected mode and
the input indices and coupling powers used by its comparison.

Mask learning checks every squared-order component, avoiding both accidental
cancellation between orders and extra-counterterm masks based only on their
first order. Relative coupling changes and zero crossings still trigger
relearning. Model vectors and result arrays retain their allocated storage.
`BornResult%has_helicities`, `%has_soft`,
`%has_extra` and `%has_single_helicity` indicate which optional values belong
to the current request; allocation alone is not an availability test.
The library does not sample random flows or helicities. Calls are serialized,
as in the legacy Fortran runtime; the API is not a thread-safety guarantee.

## Building and staging

The common `Source` build compiles the support library before subprocess builds.
Dedicated PIC and shared-link flags produce `libmc_born_support.so` on Linux and
`libmc_born_support.dylib` on macOS. Executable link flags stay separate.
Provider implementations compile once; subprocesses compile compatibility
wrappers. HELAS/model dependency changes refresh the private dependency sources.
The common clean target also cleans the DSO and its module files.

Runtime search paths cover the output's relative `lib` directory and a library
beside the executable. Integration and reweight worker transfers include the
DSO. Output tar creation already includes `Source` and `lib`. Linux relocation,
scratch staging and incremental rebuilds are exercised by compiled tests.
The macOS rules have not been executed in this environment.

## Native runtime

Each subprocess has native metadata aliases after its original FKS sectors.
`FKS_INTEGRATED` counts the original integration sectors; `FKS_CONFIGS` includes
the auxiliary aliases. Integration, symmetry and pole loops use the former.
Native activation installs Born topology, masses, colour tables, statistical
factors and coupling-order maps. Context changes invalidate phase-space and
clustering caches. Numerical results for a foreign Born remain runtime-sized;
they are never copied into the owner's Born amplitude COMMON blocks.

For each labelled history, the driver tries native Born configurations in order
and accepts the first that passes inversion and forward momentum checks. The
outer configuration number is not an index into a foreign topology. Auxiliary
maps use physical thresholds, without integration sampling cutoffs that would
exclude a valid point generated by another history. Physical cuts are still
applied to the native counterevent and real event.
For massive final-state radiation the auxiliary coordinates use the radiator
momentum magnitude and opening angle. They cover both kinematic solutions
continuously and avoid cancellation near stationary recoils and branch mergers.
Their Jacobians preserve the physical counterevent-to-real measure ratio.

The complete native H weight includes its G replacements, damping, cuts and
native colour sampling. Dividing by its real phase-space measure removes native
sampling and orbit factors while retaining counterevent-to-real measure ratios.
The outer integration and channel factors are applied once. Replaying the saved
outer random numbers and restoring colours and shower scales reinstates the
event owner after the sum.

The initial outer evaluation emits S records only when a complete H sum will
replace its H records. Inner evaluations emit H records only: ordinary S
counterterms and degenerate remnants are omitted, while nonzero G replacements
are retained. Once damping is known to vanish, real and G-replacement
evaluations are skipped for that H contribution. The native damping prescription, including the ordinary MC@NLO
low-scale damping, is unchanged.

At a fixed outer real point, only the full real matrix element can be shared:

\[
 r_b=\xi_b^2(1-y_b)R(\pi_b p;\theta_b),\qquad
 \widehat H_a=S_a\sum_b P_b(S_bR-M_b).
\]

Here \(\pi_b\) is the explicit labelled permutation and \(\theta_b\) is the
model state installed for that evaluation. A bounded cache uses the exact
local real-evaluator identity, permuted momenta and model state. Both the total
and coupling-order amplitudes are restored on a hit. Evaluations use the
original physical real point, with its labels transformed explicitly, so
inverse/forward roundoff does not create artificial differences between
histories. The native longitudinal frame does not change a summed matrix
element. Polarization-restricted evaluators retain the caller's frame and
require an exact match of those momenta instead. The driver scopes this reuse
to one real point; analytic soft and
collinear limits always retain their native Born projections. No integration
weights, cuts, PDFs, damping factors, or sampled native colours are cached with
the real amplitude.

Real evaluators retain model-aware zero-helicity filtering, but no longer infer
equal helicity amplitudes from equality at a single phase-space point. That
`T_IDENT` shortcut was not invariant under a longitudinal boost for
`g g > W+ d u~`: the two longitudinal-W amplitudes with both gluon helicities
flipped happened to agree at the training point. Removing that inference is
necessary for sharing the physical real amplitude across native frames. The
compiled checks compare the resulting sums and order buffers with an
independent full-helicity evaluation, including boosted points and changed
couplings.

Contribution records retain native provider/context/history IDs for PDF, order
and reweight bookkeeping, separately from the outer event owner. Explicit
flavour maps filter each native PDF group; momentum permutations preserve the
native labels in matrix-element reweight records. The optional provenance suffix
is preserved by the LHE parser and offline scale/PDF reweighter. Event momenta,
colours and shower scales belong to the outer context.

Central luminosities are reused only within one `include_PDF_and_alphas` call,
with exact native-history/FKS, momentum-fraction and factorization-scale keys.
Hits restore the flavour-resolved PDF array as well as its sum and flavour
count. UPC and dressed-lepton beams bypass this cache; scale/PDF reweighting
continues to evaluate its own luminosities. Contribution buffers grow
geometrically, and groups store one membership entry per contribution instead
of a dense contribution-by-group matrix. Group order and signed summation
order are preserved.

Initial validation covered complete DY, ttbar and W+jet integrations with PYTHIA8 matching,
`mcatnlo_delta=False` and no shower. Unsupported splitting modes and resonance
inversions remain unsupported. Subsequent optimization, fixed-order and Delta
checks are recorded in [the performance report](mcatnlo_performance.md).

The compiled provider tests are in
[`test_born_support.py`](../tests/unit_tests/iolibs/test_born_support.py).
The native-runtime validation record is in
[`validation/history_guard_ae5422c5/report.md`](../validation/history_guard_ae5422c5/report.md).
The earlier provider-only results are retained in
[`validation/born_support_ae5422c5/report.md`](../validation/born_support_ae5422c5/report.md).
