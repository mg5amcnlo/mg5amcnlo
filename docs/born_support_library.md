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
hidden in the shared library. Evaluation transfers model state and recomputes
the Born before requesting correlations. It does not trust caller cache flags
or reuse a result at different momenta, couplings or FKS identities. It also
does not sample random flows or helicities. Calls are serialized, as in the
legacy Fortran runtime; the API is not a thread-safety guarantee.

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

Contribution records retain native provider/context/history IDs for PDF, order
and reweight bookkeeping, separately from the outer event owner. Explicit
flavour maps filter each native PDF group; momentum permutations preserve the
native labels in matrix-element reweight records. The optional provenance suffix
is preserved by the LHE parser and offline scale/PDF reweighter. Event momenta,
colours and shower scales belong to the outer context.

Validation covers complete DY, ttbar and W+jet integrations with PYTHIA8 matching,
`mcatnlo_delta=False` and no shower. Unsupported splitting modes and resonance
inversions remain unsupported. Physical-Delta and showered comparisons are
outside this validation.

The compiled provider tests are in
[`test_born_support.py`](../tests/unit_tests/iolibs/test_born_support.py).
The native-runtime validation record is in
[`validation/history_guard_ae5422c5/report.md`](../validation/history_guard_ae5422c5/report.md).
The earlier provider-only results are retained in
[`validation/born_support_ae5422c5/report.md`](../validation/born_support_ae5422c5/report.md).
