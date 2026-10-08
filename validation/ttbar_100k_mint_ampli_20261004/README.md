# 100,000-event ttbar comparison

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


Tested commit: `7fa5120a4ab08bd4b8e880643d4a005bf82335d9`, branch
`MCcntRefactor_Sfun_Granny_AmpliColIntegrator`, on 2026-10-04.
Production sources were not changed for this benchmark.

The equal-settings benchmark completed 100,000 events with MINT. AmpliCol
completed integration but aborted event generation on an observed production
bound violation. A second AmpliCol attempt with tighter integration accuracy
also failed during generation. Neither produced a completed 100,000-event
sample. The current adapter's frozen global bound needs improvement before
it can be used reliably for this production workload.

Both runs use `p p > t t~ [QCD]`, 13 TeV, stable tops with mass 173 GeV,
the built-in `nn23nlo` PDF, default dynamic scales (`dynamical_scale_choice=-1`),
PYTHIA8 matching, folding `(2,1,1)`, polynomial virtual optimization, native
matching, `req_acc=0.003`, seed 19721, and `nevt_job=2500`. Born spreading,
scale/PDF reweighting, and storage of internal reweight records are off.
`event_norm=average`. The only input difference between the initial runs is
`#NLOPSIntegrator`. Input cards are archived here; the launcher sets the
external top width to zero. These are MC@NLO hard-event LHE samples; showers
and decays were not run.

## Equal-settings results

| Measurement | MINT | AmpliCol |
| --- | ---: | ---: |
| Requested events | 100,000 | 100,000 |
| Completed LHE events | 100,000 | Aborted |
| Signed cross section, pb | 683.7035 ± 2.7126 | 679.9264 ± 2.2138 |
| Relative signed integration error | 0.3968% | 0.3256% |
| Absolute rate used for allocation, pb | 1108.4611 ± 2.8099 | 1100.4086 ± 2.5251 |
| Adaptation CPU time, s | 45.16 | 74.55 |
| Folded integration CPU time, s | 240.81 | 340.33 |
| Total adaptation + integration CPU time, s | 285.97 | 414.87 |
| Generation trials | 538,210 | No complete run counters |
| Measured generation acceptance | 18.580% | Unavailable for a completed sample |
| Generation CPU time, s | 1117.68 | Aborted |
| Generation events / CPU second | 89.47 | Unavailable |
| Approximate generation wall time, s | 225.8 | Aborted |
| Whole-launch wall time, s, including compilation/checks | 358.13 | 211.16, until failure |
| Negative events | 19,014 / 100,000 | No completed sample |
| Observed production bound exceedances | 499, continued | First exceedance stops run |

Errors are Monte Carlo integration errors, not scale/PDF theory
uncertainties. `req_acc` targets the absolute rate used for event allocation;
cancellations can make the relative error on the signed total larger.
The difference in signed rates is 3.7771 pb, or nominally
1.08 times the quadrature-combined integration error. Absolute rates depend
on the learned virtual approximation as well as the physical integrand.

AmpliCol used 45.1% more integration CPU time for an 18.4% smaller quoted
signed error. The cost measure `T_integration * (error / cross_section)^2`
is 0.004502 CPU s for MINT and 0.004398 CPU s for AmpliCol: approximately
the same integration efficiency in this single pair of runs. This does not
establish a small performance advantage statistically.

MINT has 38,405 effective signed events, using
`N_eff = (sum(weights))^2 / sum(weights^2)`, or 34.36 effective events per
generation CPU second. Its event-sign estimate of the rate is
686.936 ± 2.751 pb, where this error represents finite-sample sign
fluctuations conditional on the integrated absolute normalization. It is
separate from the integration error above; the two errors have not been
combined as independent measurements.

Runs initially executed concurrently with five cores each on a host with
12 logical CPUs. CPU times sum the Fortran worker times; they exclude
compilation and preliminary checks. Wall times are scheduler- and
machine-dependent. Generation wall time is inferred from original worker
log timestamps and integer wall counters. The whole-launch wrapper timing
is measured directly. This is a single-run comparison, without timing error
bars or CPU pinning.

## Bound behavior and AmpliCol failure

AmpliCol aborted in `P0_gg_ttx/GF1.0_2`. An isolated instrumented replay
reproduced the same seed/checkpoints and a byte-identical 224-event candidate
spool. At trial 4,031, the nonvirtual target was 1937.393 against a frozen
bound of 1602.399, a 20.91% excess. The raw weight was 2.418 times the
maximum found in that channel's 15,360-point survey. The same stream
probability factor is applied to the weight and bound, so that factor does
not explain the violation. The original benchmark was not instrumented.
See `bound_diagnosis.json` and `bound_replay.log`.

Using the saved absolute rates and frozen bounds, the predicted overall
AmpliCol acceptance is about 1.96%, assuming valid bounds and quotas
proportional to the channel rates. This is **an estimate**, not a measured
acceptance for a completed run. Details are in
`ampli_estimated_efficiency.json`.

MINT's 499 exceedances represent 0.0927% of generation trials and 0.499%
of accepted events. The current MINT production path accepts a point with
probability `min(1, f / bound)` and gives the selected event its ordinary
global weight. It does not repair the bound or apply an overweight
correction during production. The physical distortion cannot be inferred
from the count: the individual excess ratios are not logged. These
generation exceedances do not change the previously measured integration
rates/errors. Consequently, completed event count and acceptance alone
are not evidence that all bound effects are negligible.

## Additional AmpliCol survey-statistics test

A separate 100,000-event AmpliCol attempt with `req_acc=0.001` tested the
error message's prescribed remedy of more integration statistics. It used
the same physics settings, seed, and five cores. Its accuracy target differs
from the equal-settings comparison above.

| Tighter AmpliCol attempt | Result |
| --- | ---: |
| Signed cross section, pb | 680.8281 ± 0.8060 |
| Relative signed integration error | 0.1184% |
| Absolute rate, pb | 1098.1653 ± 0.9218 |
| Adaptation CPU time, s | 89.39 |
| Folded integration CPU time, s | 2026.01 |
| Adaptation + integration CPU time, s | 2115.41 (35.26 CPU min) |
| Completed 100,000-event sample | No: bound violation |
| Model estimate of overall acceptance | 1.18%; no completed-run measurement |
| Whole-launch wall time until failure, s | 1020.43 |

The two dominant gluon channels each used 523,264 survey points. Generation
then failed in `P0_gg_ttx/GF1.0_4`; that channel used 64,512 survey points,
compared with 15,360 in the initial run. The smaller integration error and
larger survey did not suffice to make the production bound reliable for the
requested event count. Candidate spools from both failed attempts are
incomplete and are not final event samples. The tightly integrated rate
remains consistent with MINT at 1.02 nominal combined standard deviations.
See `comparison_tight.json` and `ampli_tight_estimated_efficiency.json`.

An exact isolated replay of this second failure matched the original
1,091-event candidate spool byte-for-byte. At production trial 24,746, the
nonvirtual target was 3066.0695 against the larger bound of 2215.8338, a
38.37% exceedance. Its raw weight was 3.829 times the nonvirtual survey
maximum. In this retry the virtual stream set the larger global bound,
but the nonvirtual point still exceeded it. See `bound_tight_diagnosis.json`
and `bound_tight_replay.log`. Rate-precision convergence is therefore not
sufficient evidence of a reliable production envelope in this test.

## Evidence and reproduction

- `comparison_initial.json` contains all parsed channel/worker results and
  the final MINT event-file inspection.
- `derived_metrics_initial.json` records rate-precision and negative-weight
  efficiency calculations.
- `final_checks.json` records an independent parse of all 100,000 MINT event
  elements and the compressed event file checksum.
- `experiment.json` and the `*_started.json` / `*_finished.json` wrappers
  record settings, source hashes, timing, and whether final files exist.
- `mint/`, `ampli/`, and `ampli_tight/` contain per-channel results, job inputs, and worker
  logs. The named input cards, command files, and launch logs are alongside
  this report.
- `run_benchmark.py` and `analyze_ttbar_100k.py` reproduce the setup and
  analysis after adjusting their original absolute paths. `generate.cmd`
  records the fresh process export.

Original process outputs are under `/tmp/mg5-ttbar-100k-rd5zxvbr`.
The completed MINT sample is
`mint/Events/benchmark_100k/events.lhe.gz` there. Large LHE files and
compiled process outputs are not copied into this evidence directory.
