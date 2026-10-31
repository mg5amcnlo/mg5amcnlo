Global-recoil gluon shower-denominator fix, 31 October 2026

report.md describes the implementation and its scope. The two-page PDF
compares old/corrected predictions with native no-emission measurements.
validation/ retains fixed-context, low-scale, startup, Q-averaging and
matching-pilot checks. audit/ contains independent numerical references.
The final 90-test log is retained; source/Pythia bindings are explicit.

Complete scripts, builds and raw trials remain at:
/export/slow1/rikkert/macnlops_native_ir_jets_1M_20261031/gluon_fix
No external or historical data was deleted. Installed Pythia, original
source events and production provider/executable files were not modified.
The repository implementation is changed; validation builds are isolated.

changes.patch.gz isolates this fix relative to the earlier fixed-beam /
compact-Born source epoch. Old modified source bytes come from the campaign
validation/source_state.tar.gz; other old files come from baseline commit
36f8ffb5c44cdf9fa5586ab92240e9ee4923aa10. snapshot.json records before/after
hashes. The patch excludes earlier unrelated dirty changes and documentation.

JSON whitespace alone is compacted. Original locations/hashes are pinned
in snapshot.json. Source probability averaging is unchanged; only the
shower-only Q quadrature is refined when a native soft-gluon startup
transition requires it. Quark/ISR definitions and the local-beta model
are unchanged. No full corrected 1M-event shower campaign is claimed.

Repository artifact cleanup: detailed histogram bins and build manifests may
remain only in the complete external campaign. Repeated channel/provenance
payloads were omitted from compact summaries; all 95 numerical CDF rows,
35 component rows, key transfers, reports, plots and exact original hashes
remain available. snapshot.json indexes retained files, transformations and
archived originals. The complete pre-cleanup repository subset is also at:
/export/slow1/rikkert/macnlops_native_ir_jets_1M_20261031/gluon_cleanup_20261031/repository_artifacts_before.tar.gz
Historical source and production hashes were not relabelled.

The later behavior-preserving cleanup is recorded separately in
validation/cleanup_validation.json and validation/cleanup.patch.gz. It
passes 111 tests without skips. Six-provider profiles (96 contexts, including
the same pre-existing excluded density case) and all 1,200 matching-pilot
event streams are byte-identical to the frozen fix. The original fix source
hashes remain unchanged; cleanup records the later source hashes separately.
