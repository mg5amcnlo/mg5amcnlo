# AmpliCol validation artifacts

The AmpliCol studies dated 2026-10-04 through 2026-10-08 commit their reports,
scripts, input cards, source snapshots, patches, compact logs, numerical results
and provenance manifests. Source snapshots and their hashes describe the code
used for each experiment, which may differ from the current implementation.

Large generated artifacts remain in the local validation directories and are
not included in Git: LHE samples, candidate pools (`ampli_pool.dat`), saved
AmpliCol grids (`ampli_grids`), per-trial traces, analytic observables, compressed
raw evidence and HTML output. Compiled objects, executables and caches are also
excluded. These files have not been deleted.

References in the individual reports to retained or archived raw evidence mean
the local archive. Links to those artifacts require that archive and will not
resolve in a fresh clone. Checksum manifests remain as provenance; independent
audits that consume raw pools or events, and replays using saved grids, require
the corresponding local artifacts or a regenerated run. Some physics replays
also require the generated process directories and external libraries recorded
in their scripts and provenance. The committed summaries record the checks
performed on those artifacts; they do not make those audits self-contained.
