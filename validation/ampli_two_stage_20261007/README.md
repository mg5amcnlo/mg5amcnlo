# Two-stage AmpliCol MC@NLO validation

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


The implementation now launches only a survey (stage 1) followed directly by
native event generation (stage 2). Each channel's survey starts fresh, trains
grids and auxiliary physics, and requires at least four iterations and at most
3% relative uncertainty on its absolute integral. The final iteration supplies
the saved rates, grids, virtual approximation and stream maxima. Earlier
iterations train the state; their observations are not combined with the final
absolute target after changing the virtual approximation.

Python allocates quotas with 10% reserves. Generation starts with the saved
stream maxima, divided by the stream selection probabilities, and continues
native grid/envelope adaptation and rate estimation. Only coordinates with
`ifold=1` adapt during generation. The existing full-tail checks remain strict:
the full absolute cross section of overweight points must be below 1%, including
the collection checks. This is a functional validation, not a timing comparison.

## Fresh ttbar run

`p p > t t~ [QCD]`, loop SM, 13 TeV, stable 173 GeV tops, `nn23nlo`, PYTHIA8
matching without showering, folding `(1,1,1)`, polynomial virtual approximation,
Born spreading off, seed 19728, 2,000 events, `req_acc=-1`, three cores.

All five channels ran four survey iterations: 15,360 observations in total per
channel, with the final 8,192 contributing to the saved estimate. Relative
absolute errors ranged from 0.903% to 1.196%. There were five stage-1 and five
stage-2 logs and no stage-0 logs or results. Each generation envelope matches
the saved stream maxima and probabilities.

| Quantity | Result |
| --- | ---: |
| Final events | 2,000 |
| Negative events | 395 |
| Signed rate | 683.1833 ± 3.5274 pb |
| Generation observations | 62,094 |
| Production iterations / grid updates | 8 / 3 |
| Largest native full-tail check | 0.5819% |
| Largest final channel subset bound | 0.5787% |
| Collected full overweight fraction | 0.1265% |

The sample uses `IDWTUP=-4` and retains native corrections. Independent replay
reproduces survey-plus-production rates, initial and revised quotas, seeded
selection, and final LHE weight magnitudes. Quotas changed by at most five
events; reserves sufficed without extra workers or threshold tightening.

Run `python validation/ampli_two_stage_20261007/ttbar/verify.py` to reproduce
these checks. The script uses the independent rate/collection verifier from the
preceding 300K benchmark and adds two-stage and saved-maximum assertions.
`ttbar/process/` contains compact numerical artifacts; `ttbar/sources/` and
`started.json` identify the exact implementation. Commands and logs are saved
alongside them; the complete build stays at the path in `work_directory.txt`.

## Folded Drell–Yan and restart

A fresh `p p > e+ e- [QCD]` run with folding `(2,2,2)` generated 2,000 events,
then a generation-only restart produced another 2,000 using the same survey.
All eight channels completed four survey iterations with relative absolute
errors between 0.560% and 0.982%. The generation adaptation mask was
`[1,1,1,1,0,0,0]`: unfolded coordinates continued adapting and folded coordinates
remained fixed.

| Quantity | Fresh run | Generation-only restart |
| --- | ---: | ---: |
| Executed integration stages | 1, 2 | 2 |
| Collected events | 2,000 | 2,000 |
| Production iterations / grid updates | 12 / 4 | 10 / 2 |
| Signed rate (pb) | 2094.5488 ± 4.0376 | 2090.2353 ± 4.3432 |
| Largest native full-tail check | 0% | 0.3182% |
| Collected full overweight fraction | 0% | 0% |

All 16 saved survey/MC-integer files remained byte-identical from their first
write through fresh generation and the restart. The restart also left survey
logs and results unchanged. Envelope reconstruction, initial nonzero-point
forecasts, rate merging, event selection and LHE magnitudes passed verification.
See [the DY report](dy/README.md) and [machine-readable checks](dy/validation.json).

`two_stage_sources.patch` isolates the numerical and orchestration changes
relative to the source snapshots of the preceding 300K ttbar benchmark.

## Focused tests

123 tests pass: 35 sampler, 19 adapter, 33 pool/collector, 26 orchestration,
six LHE and four backend-selection cases. The full run before the final two
adapter regressions is in `unit_tests.log`; all 19 adapter cases were then
rerun in `adapter_additional_tests.log`.

The tests cover fresh survey scheduling without stage 0, the four-iteration
minimum, last-iteration statistics after auxiliary changes, training every fold
image, direct saved-maximum initialization, generation-time adaptation, strict
tail checks, Born calibration within the survey followed by four measurement
iterations, checkpoint incompatibility, generation-only restarts, and unchanged
MINT scheduling. Fortran tests use bounds checking and floating-point traps.

```sh
python -m unittest \
  tests.unit_tests.fks.test_ampli_integrator \
  tests.unit_tests.fks.test_ampli_adapter \
  tests.unit_tests.fks.test_ampli_lhe \
  tests.unit_tests.various.test_ampli_pool \
  tests.unit_tests.interface.test_ampli_orchestration \
  tests.unit_tests.fks.test_nlops_integrator_selection -q
```

New survey checkpoints use version 3. Older three-stage checkpoints must be
regenerated; native production pool format 4 and the event collection contract
are unchanged.
