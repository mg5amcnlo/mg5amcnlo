Frozen 1M-event common-scale source validation, 31 October 2026.

report.txt records scope, numerical checks, residuals and the matching fixes.
plots/ compares source, S-veto, H-removal and combined weights before/after
showering, plus transfer balance against the previous independent campaign.
validation/ stores per-event/production summaries and exact source provenance.
inputs/ stores actual cards and one full-shower command file per process.
logs/ stores the 82 targeted regression checks.

Raw event files, all split-job histograms/native-cap CSVs, full logs and the
archived invalid intermediate matching outputs remain at:
  /export/slow1/rikkert/macnlops_winner_1M_20261031

Five positive ttbar hard remainders required a deterministic scale-only repair;
validation/hard_remainder_cap_fix.json proves all non-scale event bytes stayed
unchanged. The new generator emits the equivalent common hard fallback.

Pythia was not modified. Exact cancellation is not established: the report
quantifies statistically resolved transfer residuals and shower migration.
The prior remains the fixed-native-Born AP(z)/t^2 approximation.

This is a historical source, superseded by native continuation/global-IR.
The 200-event pilot summary is retained externally at:
  /export/slow1/rikkert/macnlops_winner_20261031/winner_pilot_summary.json
SHA256: 0d0595bb56840a02658887412c31d635446d0f635b874e85a9821286cdeef87d

Repository input .dat/.cmd copies normalize indentation and trailing whitespace;
the token sequence is unchanged. snapshot.json:normalized_files records exact
external originals and raw/normalized SHA256 values. JSON profiles are unchanged.
Test-log copies use .txt names with identical bytes; renamed_files records their
original paths/hashes. Frozen production-source hashes are unchanged.
