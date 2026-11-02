# Clean Reference Fixtures

The release-facing reference bundle consists of:

- `physics-v2.json`: exact model, process, phase-space, topology, resolved-axis,
  reduction, and high-precision matrix-element records;
- `legacy-fortran-v2.json`: independent built-in-SM evidence from the pinned
  Fortran AmpliCol color probe;
- `analytic-oracles-v2.json`: independent closed-form evidence for the scalar
  and scalar-gravity cases;
- `reference-fixture-v2.manifest.json`: the manifest-last atomic commit marker
  and digest for the other three documents.

Every nondegenerate process has three deterministic generic points and one
quantified high-precision stress point. Values are recorded against complete
physical helicity axes and either physical LC flows or one contracted NLC/full
color component. Structural zeros, normalization, topology, reduction maps,
input hashes, independent-oracle output hashes, and exact dependency provenance
are part of the validated contract.

The built-in-SM ladder covers neutral and charged currents, leptons, mixed- and
same-flavour two-quark-line scattering, pure gluons, top-pair production, and
the retained one-, two-, and three-body Z ladder in LC, NLC, and full color.
The external-model slice covers scalar contact and scalar-gravity interactions.

The bundle was captured from the clean installed candidate identified in
`CAPTURE.toml`. Generated process artifacts are intentionally not committed,
but remain in the ignored revision-scoped `.artifacts/reference-fixture-v2/`
directory for retiming and diagnosis.

Run the capture only behind the external 30 GB memory watchdog. The capture
tool validates both independent evidence sets and all bundle digests before it
publishes the manifest. CI independently rebuilds the pinned Fortran evidence
and compares it semantically with the tracked v2 document.

`physics-v1.json` and `legacy-fortran-v1.json` remain solely as focused inputs
for legacy fixture-reader regression tests. They are not release evidence and
are not consumed by current generation/runtime integration tests.
