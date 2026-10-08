# Independent analytic comparison of stable adaptation/proposal history

2026-10-08

The final experiment uses 16 independent seeds per variant, with 30,000
collected events and a 33,000-event reserve per seed. Baseline and candidate
compile the frozen sources in ../physics/{baseline,candidate}_sources;
their exact hashes are in replicas/summary.json. No production source was
edited for this experiment. Compiled binaries/modules use temporary directories.

The generation target is separable before folding:
  a(x) = 0.1 + exp(-30*x) + 2*I(0.72 <= x < 0.725).
The second coordinate has support [0.1,0.3) union [0.6,0.8), with absolute
density 1+y. Signs are negative on [0.1,0.225) and [0.725,0.8), positive
on the remaining support. The driver evaluates both images of ifold=(1,2).
Sixty percent of attempted folded points have zero weight. The second map
is checked bitwise at 34 fixed folded coordinates after every completed
batch. The first coordinate remains adaptive. Event signs select one of
the two physical fold contributions with probability proportional to its
absolute contribution. All actual zero/rejected draws enter rate moments.

Exact integrals:
  absolute = 0.08313333333333153
  signed   = 0.003583333333333256
Exact normalized event observables:
  P(x<0.1)             = 0.2907471934028287
  P(0.72<=x<0.725)     = 0.07325581396697017
  mean event sign      = 0.04310344827586207

The driver seeds generation with the analytic absolute rate and a uniform
grid. This intentionally isolates generation, rather than testing survey
accuracy. The first nonzero batch is 128 points (public API argument),
instead of the usual production default, to exercise many proposal epochs.

Coverage pilots were discarded from the final statistical comparison.
Two seeds with quota 10K and the ordinary 8192 first batch gave only 4--6
epochs and no skipped updates, including a narrower 0.0005 tail. Raising
quota to 30K and lowering the initial batch to 128 exercised skipped
adaptation in both coverage-pilot seeds. That setup was then frozen.
The final experiment uses fresh seeds (offset 100); no parameters were
chosen using rate/shape agreement. Pilot summaries/logs remain separately
under pilot*, and do not contribute to replicas/summary.json.

Observed schedule coverage:

```text
                          baseline       candidate
  Total trials             5,258,919       5,236,559
  Mean completed batches     10.6875         10.6875
  Mean grid updates           9.6875          9.4375
  Skipped updates                 0               4
  Seeds with skipped update       0            4/16
  Mean eligible batches           8            8.25
  Expired trials             34,544          24,350
```

Each of the four candidate runs with a skip retained nine eligible batches.
The new branch is therefore exercised, although most final replicas do not
skip. Trial totals differ by -0.43%; independent seed sets and the observed
variation do not establish an efficiency gain from this analytic comparison.

Rate results below quote the error on the ensemble mean computed from
between-seed scatter, not the per-run integration error:

```text
                         baseline                 candidate
  absolute mean     0.08309311 +/- 0.00005051  0.08304561 +/- 0.00004008
  signed mean       0.00359410 +/- 0.00001239  0.00357752 +/- 0.00001168
  abs scatter / RMS quoted error     0.845                  0.672
  signed scatter / RMS quoted error  1.029                  0.973
```

The candidate absolute mean is 2.19 empirical standard errors below truth,
or 1.47 errors using its reported per-run uncertainties combined for the
ensemble mean. Its difference from baseline is only 0.74 combined empirical
standard errors. The signed means differ by 0.97 combined empirical errors.
These 16-seed results do not establish bias or prove uncertainty calibration.
The mildly low candidate absolute mean is recorded explicitly, not hidden
behind a broad pass/fail tolerance.

Final, uniformly collected event shapes use the native correction weights:

```text
                          exact       baseline    candidate
  P(x<0.1)              0.29074719   0.29048809  0.29124112
  P(tail interval)      0.07325581   0.07249571  0.07298173
  mean event sign       0.04310345   0.04234269  0.04293735
```

Candidate deviations from the analytic shapes are 0.71, -0.52 and -0.19
empirical standard errors. The baseline tail-bin deviation is -1.99 errors.
All per-seed observables and errors are retained in replicas/records.json.

All 32 runs passed the existing full-trial, reserve and worst-final-subset
1% checks, independently recomputed by Python. Maximum worker checks:
  baseline 0.96662%, candidate 0.98174%.
Each reserve was uniformly trimmed once to exactly 30,000 events. The
maximum selected-sample tail was 0.89986% baseline and 0.85565% candidate.
No favorable repeated selection or modified tail tolerance was used.
This is a standalone integrator/pool diagnostic; it does not write LHE
payloads or replace the physics-worker comparison.

Reproduce from the repository root:

```sh
python validation/ampli_stable_history_20261008/analytic/run.py
```

Validate the retained compressed evidence without rerunning integration:

```sh
python validation/ampli_stable_history_20261008/analytic/verify.py
```

Each final replica retains run.log, diagnostics.json, ampli_pool.dat.gz and
observables.dat.gz. The native pool is immutable. The verifier recreates
its temporary uncompressed copy, independently validates the pool, and
replays the one final selection using the recorded selection seed. It
also records SHA-256 hashes in replicas/evidence_sha256.json.
