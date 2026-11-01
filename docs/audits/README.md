# Recent matching validation records

These records cover distinct source, matching and analysis epochs. Later fixes
do not retroactively change the archived samples. Other historical audits
remain in this directory.

| Record | Scope |
|---|---|
| [Native Born-scale correction](macnlops_born_scale_fix_20261101/report.md) | General implementation, regression tests, 19,812 paired first-step events and 1,700 native density points across eight processes. This is not a new 1M-event or full-shower campaign. |
| [Wbb transfer investigation](macnlops_wbb4fs_transfer_investigation_20261101/report.md) | Expected-transfer controls on the frozen no-cut sample, with an independent 200k-event Born control. The local S/B fits are diagnostic and were not added to production matching. |
| [Wbb 4FS no-cut validation](macnlops_wbb4fs_nocuts_1M_20261101/report.md) | Complete 1M-event first-emission and full-shower validation, including the PDF-support guard. This sample predates the general Born-scale correction. |
| [W+jet 40 GeV analysis cut](macnlops_wj_pt40_20261031/report.md) | Paired 25/40 GeV analysis thresholds on the original 1M-event sample. Generation retains its 20 GeV cut; this matching epoch predates the gluon-denominator and Born-scale corrections. |

Each report links its numerical evidence; `snapshot.json` in the same directory
pins compact files and externally retained inputs. Frozen source snapshots,
initial failures and their hashes are preserved.
The archive-specific whitespace attributes preserve those exact source and
run-card bytes; production code retains ordinary whitespace checks.

For the Born-scale implementation, the qualified native-check result is
[validation_summary.json](macnlops_born_scale_fix_20261101/generalisation/validation_summary.json).
The retained strict comparison and initial probe records are explained there
and in the report: native cache roundoff was checked against controls with the
new correction disabled.

The later cleanup epoch has separate
[cleanup validation](macnlops_born_scale_fix_20261101/cleanup_validation.json).
It preserves the earlier runtime evidence and source epochs.
