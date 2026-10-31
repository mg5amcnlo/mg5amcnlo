Native-history/global-IR MC@NLO-Delta: 1M events per process

report.txt records the construction, measured results and limits. The PDF
plot books compare source, S-veto, H-removal and combined weights before and
after showering, and compare native continuation with v4 and auxiliary winner.
validation/ contains compact probability, scale, migration and provenance
summaries; inputs/ contains the actual cards and shower commands. Artifact
paths in report.txt refer to the complete campaign below; this repository
retains the compact subset indexed in snapshot.json.

Complete campaign (events, split-job outputs, scripts, detailed probe CSVs,
logs and frozen sources):
  /export/slow1/rikkert/macnlops_native_ir_1M_20261031

Source construction: delta_native_ir / mcatnlo-delta-native-ir-v1.
Run: run_1M_native_ir; seeds DY3220101, H3220201, ttbar3220301.
Baseline commit: 7031e3e04509b80f61d85e53a1ede8e45fda12a0.
Pythia source and libraries were not modified.

Reproduce the exact production source by applying validation/source_changes.patch.gz
to the baseline (gunzip -c validation/source_changes.patch.gz | git apply -).
The complete frozen source/test archive (including the two
then-untracked tests) remains at:
  /export/slow1/rikkert/macnlops_native_ir_1M_20261031/validation/source_state.tar.gz
Its SHA256 and the paths/hashes of omitted duplicate files are recorded under
snapshot.json:archived_files. No external campaign data was deleted.
The archived original snapshot preserves the pre-cleanup file index.
Production-source hashes in initial_manifest.json/provenance.json identify
the code actually used for the 3M runs. Later code cleanup and its checks are
recorded in validation/cleanup_validation.json; these source hashes have not
been relabelled. The cleanup passes 136 targeted tests and agrees with the
validated helper in 597,584 support/scale/status comparisons on all 11,218
saved probe events. The 1M-event campaigns were not regenerated for this
behavior-preserving cleanup.

From the complete campaign directory, run_1M.py KEY --events 1000000
--run run_1M_native_ir --seed-offset 210000 performs export, configuration,
generation/matching, diagnostics and full shower. KEY is DY, H or ttbar.
validate_events.py, scan_scales.py, plot_hard_pt.py, analyse_native.py,
analyse_migration.py and check_production.py audit and analyse the outputs.
build_report.py assembles the report after verifying frozen production hashes.
The stock scale probe source is docs/audits/macnlops_continuation_scale_probe.cc.

Environment:
  PYTHIA8=/nfs/home/rikkert/Downloads/pythia8
  PYTHIA8DATA=/nfs/home/rikkert/Downloads/pythia8/share/Pythia8/xmldoc
  LD_LIBRARY_PATH=/nfs/home/rikkert/LHAPDF/lib:/nfs/home/rikkert/Downloads/pythia8/lib
  OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1

The original 126-test list remains in logs/regression_tests.txt. Quoted
uncertainties are conditional Monte Carlo estimates; channel integration and
beta-normalization errors are separate. Sparse unsampled ratios do not
establish zero physical rates. Paired migration uses the same 50 chunks/jobs,
checked against all four inclusive weights at printed HwU precision.
Original-v4 ttbar retained QEDByOther; winner and native samples disable it.

Repository input .dat/.cmd copies normalize indentation and trailing whitespace;
the token sequence is unchanged. snapshot.json:normalized_files records exact
external originals and raw/normalized SHA256 values. JSON profiles are unchanged.
Test-log copies use .txt names with identical bytes; renamed_files records their
original paths/hashes. Frozen production-source hashes are unchanged.
