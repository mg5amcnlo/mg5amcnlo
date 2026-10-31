Continuation-scale audit, 31 October 2026

report.txt gives the conclusion, scope and numeric checks. summary.json keeps
all native-import errors and the stratified paired controls; the PDF plots
the before/full-shower H/S transfer ratios. snapshot.json binds these frozen
copies. Raw selected LHEs, native-cap CSVs and logs remain in the original
audit path recorded in snapshot.json. The million-event inputs are unchanged.

No starting-scale prescription was changed by this audit. A separate real
QED configuration error was fixed in the shower-card template: ByOther is
now set with the QED switch. The archived original full-shower settings and
scale-control numbers predate that flag fix and remain unmodified evidence.

Reproduction tools in the adjacent audit directory:
  check_macnlops_continuation_scales.py KEY --campaign CAMPAIGN --output AUDIT
  macnlops_continuation_scale_probe.cc (compile against stock Pythia 8.318)
  summarize_macnlops_scale_audit.py AUDIT

The C++ probe usage is INPUT.lhe Pythia8.cmd OUTPUT.csv inspect|full [qed-off].
The Python scanner makes sample, capped and common H-start LHE controls.
The separate qed-off probe changes only TimeShower:QEDshowerByOther.
Use the installed Pythia and LHAPDF library/data paths in the environment.
The native regression tests exercise common S caps, distinct H dipole caps
above production minima, and zero H dipole stops without external campaign
files. All 60 relevant regression tests pass.
