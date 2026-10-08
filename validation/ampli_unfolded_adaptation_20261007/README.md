# Unfolded-grid adaptation validation, 2026-10-07

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


AmpliCol production now updates only coordinates with `ifold=1`. Folded
coordinate maps and physics auxiliaries remain fixed. Every completed trial
trains the eligible maps, including rejected and zero-weight points. Batches
start at 1,024 trials and double to a maximum of 65,536. Finer survey grid
resolution is preserved.

Candidate importance weights and priorities retain their draw-time values.
The common acceptance cutoff allows the existing full-weight 1% tail checks
to cover candidates drawn from different grid versions. The survey remains
the source of rates, uncertainties, channel quotas and event normalization.

## Generated-process checks

Process: `p p > e+ e- [QCD]`, loop SM, 13 TeV, `nn23nlo`, PYTHIA8 matching,
folding `2,2,2`, polynomial virtual approximation, Born spreading off,
`event_norm=sum`, `nevt_job=2000`, five cores. No shower was run.

| Quantity | Full workflow | Generation-only restart |
| --- | ---: | ---: |
| Seed | 19723 | 19724 |
| Requested and collected events | 6,000 | 6,000 |
| Worker jobs | 8 | 8 |
| Production grid updates | 11 | 11 |
| Generation trials | 22,969 | 21,524 |
| Generation CPU seconds | 65.0248 | 61.4324 |
| Negative events | 150 | 172 |
| Largest worker tail bound | 0.8994% | 0.9883% |
| Collected full-weight overweight fraction | 0.4469% | 0.5477% |

All workers report the mask `[1,1,1,1,0,0,0]`: four unfolded coordinates
can adapt; three folded coordinates stay fixed. Six workers reached at
least one update, while two small-quota workers finished before 1,024 trials.
The eight saved survey grids and eight MC-integer grids remained unchanged.

Both runs report the same surveyed signed rate, `2092.961926 ± 4.257862 pb`,
and absolute rate `2214.809556 pb`. All surveyed channels satisfy the 3%
relative absolute-rate target. The outputs have `IDWTUP=-4`; corrections
and negative weights survive collection.

The first run used the initial implementation. Review subsequently added
preservation of finer survey fill-grid resolution, reset of adaptation state
when returning to integration, and sequential finite-value checks. The
generation-only restart uses the final source. Separate source snapshots
and fingerprints document this distinction. These runs do not establish a
performance improvement over fixed grids or MINT.

## Independent verification

From the repository root:

```sh
python validation/ampli_unfolded_adaptation_20261007/initial/verify.py
python validation/ampli_unfolded_adaptation_20261007/restart/verify.py
```

The scripts validate POOL3 masks, schedules, quotas, reserve counts,
corrections and tail diagnostics. They reproduce the seeded collection
selection and compare the resulting weight magnitudes with the final LHE
sample, check event counts, negative weights and init normalization, verify
survey grid hashes, and check the exact tested source snapshots. The full
candidate LHE spools remain in the original temporary run directory; the
archive includes their numerical pool metadata and both final LHE samples.

Each run directory contains commands, logs, cards/banners, pool metadata,
saved survey grids, events and `validation.json`. `source_provenance.json`
records final implementation hashes and the original working directory.
`implementation.patch` records the implementation/test changes relative to
the stated base commit, including the preceding survey/collection revision.

## Focused automated checks

100 tests passed in focused groups against the final sources:

| Suite | Tests |
| --- | ---: |
| Sampler | 28 |
| MC@NLO adapter | 16 |
| Pool reader and collector | 27 |
| Python orchestration | 19 |
| LHE payload handling | 6 |
| Backend selection | 4 |

The equivalent combined command is:

```sh
python -m unittest \
  tests.unit_tests.fks.test_ampli_integrator \
  tests.unit_tests.fks.test_ampli_adapter \
  tests.unit_tests.various.test_ampli_pool \
  tests.unit_tests.interface.test_ampli_orchestration \
  tests.unit_tests.fks.test_ampli_lhe \
  tests.unit_tests.fks.test_nlops_integrator_selection
```

The adaptive numerical tests independently check analytic signed/absolute
integrals and event density across map changes, frozen folded maps, unchanged
historical candidate weights, rejected/zero-trial training, preserved survey
resolution, and exact active-checkpoint continuation. The collector tests
also retain strict 1% boundary rejection, nonzero overweight corrections,
bias factors, LHE precision checks and older POOL2 compatibility.
