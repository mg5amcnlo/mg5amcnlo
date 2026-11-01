W+jet analysis jet threshold 25 -> 40 GeV, 31 October 2026

Same original 1M source/matching events and 50 same-stream shower jobs.
Only the analysis threshold changes. Generation remains pT>20 GeV,
anti-kT R=0.5; analysis keeps |y|<4.5. No signed-weight renormalization.
The matching epoch precedes the global-gluon denominator correction.

report.md explains the result. comparison.json contains paired statistics;
plots/ contains the two-page PDF, previews and numerical plot data.
validation/ records exact reclustering and byte-identical original 25 GeV
histograms for all 50 shower jobs. provenance/ binds inputs and builds.
analysis_sources.tar.gz contains the exact scripts and analysis sources.
JSON copies are whitespace-compacted; original hashes remain recorded.

All large per-event arrays, inputs, builds and logs remain at:
/export/slow1/rikkert/macnlops_native_ir_jets_1M_20261031/wj_analysis_pt40_20261031
No production events, matching decisions, Pythia source or libraries were changed.
