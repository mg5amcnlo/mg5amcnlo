MAcNLOPS Sudakov correction: frozen validation, 30 October 2026

This compact snapshot contains the report, first-emission/full-shower plots,
per-process probability and production checks, independent massive-FSR checks,
and original source/table/library provenance for the fresh 1M-event samples.
snapshot.json binds all copied files by SHA-256.

These are unmodified copies of completed campaign artifacts. Paths in the
report and provenance refer to the original campaign:
  /export/slow1/rikkert/macnlops_sudakov_1M_20261030
The full LHE samples, event caches, executables and logs remain outside Git.
Provenance source hashes describe the exact code used for production, before
this documentation/archive cleanup. The earlier macnlops_1M_20261030 campaign
referenced by the report was no longer available when this snapshot was made.

Start with report.txt and plots/sudakov_fix_comparison.pdf. The four-weight
first-emission/full-shower plots are in plots/hard_pt_run_1M_v4.pdf. The v3/v4
comparison uses independent sources; plots/transfer_balance.pdf instead
applies both rejection prescriptions to the same fresh events.

Known limitation: the statistically resolved ttbar transfer-normalization
residual is retained explicitly in the report. This snapshot does not claim
complete local transfer closure.

The adjacent low-pT audit scripts and probes retain the original investigation
code. They require the original campaign data and the Pythia/LHAPDF installation
specified in those scripts. Implementation and regression tests live in
Template/NLO and tests/unit_tests.
