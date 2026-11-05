# Shared Schemas

These JSON Schemas are the normative wire contracts shared by the Python
generator, Rusticol, the native SDK, and installed-package self-tests.

- `artifact-manifest-v3.schema.json` describes the only top-level process
  artifact accepted by pyAmpliCol 0.1.
- `runtime-physics-v1.schema.json` describes either resolved public helicity
  and color axes or compact on-the-fly public particle/parameter metadata;
  resolved artifacts also carry reductions and selectors.
- `reference-physics-v2.schema.json` is the repository/sdist-only contract for
  exact decimal physics baselines, complete resolved axes, structural zeros,
  model-derived external spin/color/mass metadata, topology, canonical
  reduction-group mappings, normalization, payload hashes, and capture
  provenance.
- `reference-oracle-evidence-v2.schema.json` records independent analytic or
  pinned-Fortran evidence. Evidence can certify resolved cells, per-helicity
  aggregates, or totals according to the oracle's actual semantics; it must
  not promote diagnostic contraction partitions into physical color flows.
  This is a developer validation format, not a runtime artifact format.
- `reference-fixture-bundle-v1.schema.json` is the final commit marker for a
  captured fixture and its evidence documents. It names and hashes every JSON
  member; readers reject data files until this manifest is present and valid.

Evaluator plans remain private payloads. The top-level manifest nevertheless
records their canonical required runtime capabilities so a loader can reject
an unsupported backend before opening evaluator bytes. Direct SymJIT f64
applications, optional Symbolica evaluator state, compiled libraries, and
their checksums, producer ABI, and target compatibility are covered by the
manifest; their internal representation is not part of the public API.

Loaders must perform checks that JSON Schema cannot express before opening a
payload:

1. Resolve every payload beneath the artifact root without following a
   symlink outside it.
2. Reject absolute paths, `..` traversal, duplicate normalized paths,
   missing or unexpected executable payloads, and non-regular files.
3. Verify every byte size and SHA-256 digest before deserializing or loading
   executable state.
4. Check package, process-artifact, runtime-physics, compiled-model,
   evaluator runtime capabilities, relevant evaluator serialization ABI,
   C-ABI, target-triple, and CPU-feature compatibility.

Unknown top-level fields are rejected. Deliberate extension data belongs in
the explicit `extensions` objects and must not affect physics or executable
loading.
