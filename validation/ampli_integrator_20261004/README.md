# AmpliCol MC@NLO validation, 2026-10-04

Implementation branch: `MCcntRefactor_Sfun_Granny_AmpliColIntegrator`, based
on `85f6b30aa`. Usage and design are in
[`docs/ampli_integrator.md`](../../docs/ampli_integrator.md).

- `unit_tests.log`: final combined run, 73 tests passed.
- `drell_yan.json`: final LHE counts, signs, normalization, reweight payload
  counts, saved configuration, generation-only allocation, and source hashes.
  The accompanying run/restart/integration-only scripts, commands, and logs
  record the actual coordinator runs.
- `eejets.json` and `eejets/`: standalone channel tests for AmpliCol and MINT,
  fixed-order exclusion, and complete Born-spreading calibration. The
  original native-map failure is retained alongside the successful corrected
  run. These standalone tests use a synthetic global normalization; they are
  not a measurement of the full physical process rate.
- `ttbar/`: polynomial-virtual, massive-process run settings, generation and
  launch commands, logs, production summaries, and LHE validation.
- `final_source_sha256.json`: final reviewed workspace hashes. Per-run
  exported hashes are recorded separately: some long-running checks predate
  the final missing-MC_integer-file guard and/or massless map fix. These
  differences are identified in the individual reports and covered by the
  automated tests and native Born-spreading run.

Scripts and command files are execution records containing the original
absolute workspace and temporary output paths. To reproduce elsewhere,
adjust those paths and generate fresh process directories using
`generate.cmd` (or `ttbar/generate.cmd`) before invoking the corresponding
run scripts. The complete temporary outputs remain at
`/tmp/mg5-amplicol-backend-sq5iorrd` and `/tmp/mg5-ampli-ttbar-expldnvr`.
The copied evidence does not depend on keeping those outputs.

Generated-process checks stop at MC@NLO LHE production with PYTHIA8
matching; the parton shower was not run. Event samples are deliberately
small correctness checks, not precision distribution or performance
benchmarks. Generation with an underestimated envelope is tested to fail
explicitly. Finite surveys cannot certify bounds on unseen tails.

Archived text logs and result files have trailing whitespace removed.
The imported Fortran sources received the same whitespace-only cleanup
before committing; final workspace hashes reflect this cleanup.
