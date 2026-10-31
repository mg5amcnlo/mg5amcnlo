W+jet discrepancy investigation, 31 October 2026

report.md separates the demonstrated local signed-S/beta failure, the
gluon-FSR denominator discrepancy, and continuation of an initial shape
mismatch. It records open findings and conditional uncertainty limits.

The existing corrected 1M source, matching and full-shower records are
unchanged. All 1M first-step endpoints were reproduced exactly. All 155,756
nonzero realized transfers were independently continued in four controls;
omitted event transfers are exactly zero. No warning-conditioned production
selection/retry, Pythia modification, or production matching change was made.
A separate 200k positive LO source checks the Born shape. Detailed native
conditional-Q trials check the actual first-shower survival law.

Complete scripts, rawarrays, everytrial, providerbindings, logs and sources:
/export/slow1/rikkert/macnlops_native_ir_jets_1M_20261031/investigation_wj
All scripts and original compactresults are pinned in snapshot.json.
The prior production audit is ../macnlops_native_ir_jets_1M_20261031/.
JSON copies compact whitespace only. No raw or historical data was deleted.

Reproduction entrypoints in the full investigation:
first/classify_native.py and summarize_first.py: registry/transfer partitions;
kernel/cache_born.py and analyse_lo.py: original Born states and independentLO;
warnings/replay_warning_endpoints.py: exact originalseed endpoint replays;
warnings/select_sudakov_contexts.py/run_cdf.py: conditional-Q native tests;
continuation/prepare.py/run.py/summarize.py: full-transfer continuation controls;
make_plots.py: combined scientific plot book. All use local stock Pythia 8.318.
The exact commands, source/provider/library/card hashes, randomstreams and
control parameters are retained in individual provenance files/raw scripts.

Conditional sampling errors omit hard integration, beta and unmeasured native
trial-envelope bias. Selected CDF contexts do not estimate the full population
residual contribution. A negative local beta is not a valid drop-in veto fix.
The independent LO/native Born integral mismatch remains open; no unique source
bug is claimed and no normalization was silently changed.

Repository artifact cleanup: detailed histogram bins and build manifests may
remain only in the complete external campaign. Repeated channel/provenance
payloads were omitted from compact summaries; all 95 numerical CDF rows,
35 component rows, key transfers, reports, plots and exact original hashes
remain available. snapshot.json indexes retained files, transformations and
archived originals. The complete pre-cleanup repository subset is also at:
/export/slow1/rikkert/macnlops_native_ir_jets_1M_20261031/gluon_cleanup_20261031/repository_artifacts_before.tar.gz
Historical source and production hashes were not relabelled.
