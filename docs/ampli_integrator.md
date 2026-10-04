# AmpliCol integration for MC@NLO

The branch `MCcntRefactor_Sfun_Granny_AmpliColIntegrator` adds an optional
AmpliCol numerical backend to `madevent_mintMC`. Generate the process output
from this branch and set the following in `Cards/FKS_params.dat`:

```text
#NLOPSIntegrator
1
```

`0` selects MINT and remains the default, including for older cards without
the setting. Fixed-order runs always use MINT. Switching backends requires
fresh integration. Existing exported processes need to be regenerated to
include the new sources and build rules.

The normal `launch aMC@NLO` workflow, run-card folding, `req_acc`, `nevents`,
`nevt_job`, and event normalization settings apply. `nevents=0` runs adaptation
and integration without event generation. The implementation is process
independent; the validation below samples massless, massive, initial-state,
and final-state processes rather than establishing every process/shower
combination.

## Coordination and stages

The Python NLO run manager retains one subprocess/integration channel per
worker. It assigns integer quotas with probabilities
`A_channel / sum(A_channel)`, where `A` is the absolute generation target,
and splits large quotas through `nevt_job`. Workers communicate through the
existing result files and stage boundaries; no inter-worker messaging or
AmpliCol grid merging is introduced.

| Stage | AmpliCol behavior |
| --- | --- |
| 0 | Adapt the sampling grid, MC integer sampling, and virtual approximation using unfolded points. If enabled, calibrate Born spreading and repeat adaptation for the fitted target. |
| 1 | Freeze the learned state; integrate with the requested folding and virtual treatment; measure signed and absolute rates and a production envelope. |
| Coordination | Use the existing Python allocation and global normalization, including split generation jobs. |
| 2 | Restore the frozen state; generate exactly the assigned quota; finalize complete signed LHE events. |

The staged API separates integration statistics from the requested event
count. Integration reports its achieved uncertainty, and reports when its
iteration limit is reached. Generation has a separate trial limit and fails
if it cannot fill the quota. Absolute-rate uncertainties include the
covariance of the nonvirtual and residual-virtual contributions.

`ampli_mint_adapter.f90` preserves the `sigintF` contract, folded contribution
selection, S/H event treatment, shower scales, and MG5 event writing.
`mint_module` remains the owner of shared physics/auxiliary state. During
production neither the sampling map nor auxiliary state is adapted.

## Unweighting and normalization

Production samples the nonvirtual and residual-virtual streams using their
absolute rates. MG5 supplies the selected contribution's sign and the global
event weight; the integrator does not substitute a channel-local cross
section. The existing `sum`, `average`, `unity`, and bias normalization paths
remain in the event writer. Event attributes, scales, internal reweight
records, and later scale/PDF reweighting are preserved.

MG5 uses fixed-envelope rejection, including for quotas of one or two.
The bound is twice the larger of the measured stream-adjusted maximum and
the channel absolute rate. Every accepted event receives a unit correction
factor. AmpliCol's original ranked candidate selection is retained in the
vendored module but is not used by the MC@NLO path: finite-quota tests showed
that ranked selection alone biases the smallest quotas.

Exact rejection sampling requires a valid bound. A finite integration
survey cannot guarantee an unseen tail is bounded. An observed violation
therefore invalidates production with an explicit error asking for more
integration statistics. It does not enlarge the bound after accepting
events or silently emit corrected weighted events. A failed job's candidate
spool is not a completed event sample. A tighter `req_acc` can increase the
survey statistics; rerun integration before attempting generation again.

## Checkpoints and generation-only runs

Each channel keeps `ampli_grids`, containing a versioned sampler, rates,
folding, channel identity, and virtual approximation state. Its companion
files are `grid.MC_integer`, `res_1.dat`, and, when enabled,
`born_spreading.dat`. Preserve these together with the saved Python jobs.
Split workers share the trained files and retain MG5's independent random
streams. The same files are included in the existing cluster transfer
protocol.

For example, from the exported process's `bin/aMCatNLO` interface:

```text
launch aMC@NLO -f -p --only_generation --name=additional_events
```

Here `-p` stops after producing the MC@NLO LHE events. Generation-only runs
can change the event count, seed, job-size limit, output/reweight controls,
or choose among `sum`, `average`, and `unity`. They reject a different
backend, central physics settings, folding, Born spreading, bias mode, FKS
settings, run mode, or parameter card. The parameter card is compared by
checksum. These are stage-boundary restarts, not recovery of a partially
written stage-2 LHE file.

## Validation on 2026-10-04

The combined automated suite passed 73 tests covering the staged sampler,
folding, small-quota sampling, rejected envelope violations, checkpoints,
adapter statistics/virtual streams, LHE preservation, backend selection,
job allocation/restarts, Born spreading, and momentum maps:

```sh
python -m unittest \
  tests.unit_tests.fks.test_ampli_integrator \
  tests.unit_tests.fks.test_ampli_adapter \
  tests.unit_tests.fks.test_ampli_lhe \
  tests.unit_tests.fks.test_nlops_integrator_selection \
  tests.unit_tests.interface.test_ampli_orchestration \
  tests.unit_tests.fks.test_born_spreading \
  tests.unit_tests.fks.test_momentum_maps
```

| Generated-process check | Result |
| --- | --- |
| `e+ e- > u u~ [QCD]`, folding `2,2,2`, native matching, averaged virtual optimization | All three stages; exactly 10 channel events. Signed channel integral `0.129203 ± 0.000522 pb`, compared with MINT `0.129877 ± 0.000547 pb`. |
| Same process, Born spreading and native matching | Full 800,000-point training plus 200,000-point validation; readaptation, table reload, folded integration, and exactly 10 channel events. |
| Same process, fixed order with switch `1` | Two grouped channels still run MINT and produce `mint_grids`. |
| `p p > e+ e- [QCD]`, folding `2,2,2` | Four subprocesses/eight channel jobs; exactly 100 events, including negative weights, zero/one-event quotas, and split workers. All events retain scale and internal reweight records. |
| Drell–Yan normalization/restart | Initial `sum` weights have magnitude `19.486296`; generation-only runs produce 60 `unity` events and 40 `average` events with magnitude `1948.6296`. |
| Drell–Yan, `nevents=0` | Both integration stages complete, with no event-generation stage. |
| `p p > t t~ [QCD]`, folding `2,1,1`, polynomial virtual approximation | Exactly 20 signed events across five production jobs; signed integral `676.4 ± 4.4 pb`. |

These checks produce MC@NLO LHE samples with PYTHIA8 matching; they do not
include running the parton shower. Standalone electron-positron channel
tests supply a synthetic global normalization to exercise the event path;
the hadronic runs test the actual Python coordinator. No broad performance
comparison has been made. Machine-readable results, commands, logs, and
source provenance are in
[`validation/ampli_integrator_20261004`](../validation/ampli_integrator_20261004/).

Born calibration exposed a precision failure in the native massless FSR
inverse map for nearly antiparallel daughters and soft recoil. The branch
uses a stable angular coordinate and factored endpoint expressions there.
The original failing point passes reconstruction in both daughter orders,
and the full native calibration passes without loosening tolerances or
discarding contributions. This map correction also applies to MINT users
of native matching.

## Source provenance

`simple_integrator.f90` and `integrator_helpers.f90` originate from
`~/space/git/AmpliCol/master/SimpleIntegrator` at commit
`61b9cd52f44b4cdfc4773daa197e94eeb482b560`. The additional `staged_integrator`
API reuses its adaptive grids. MG5 supplies `ran2`; there is no second RNG.
The AmpliCol objects are linked into `madevent_mintMC` only. Their compile
flags omit the legacy `-fno-automatic` because imported `PURE` procedures
cannot have implicit `SAVE` variables.
