# Independent-seed analytic envelope comparison

The baseline and isolated quantile prototype were fixed before these trials. Sixteen independent seeds per variant use the same signed analytic target, survey procedure, initial nonzero budget 128, actual two-image folding of coordinate 2, 60% zero trials, narrow-tail width 0.005, and requested final quota 30,000 plus the 10% reserve. The seed offset is 400; the two variants use disjoint seed sets. No target or schedule parameter was tuned from these results.

The driver and runner are copied from the previous stable-history validation. The runner changes only source paths and the predetermined seed set. Raw pools and observables are retained compressed in `replicas/`; source hashes are in `replicas/summary.json`.

All **32 pools and final collections** pass the production Python reader, strict full-trial/reserve/worst-subset checks and selected-tail check. The folded coordinate map remains bitwise unchanged. This is a validation diagnostic, not a proof of finite-sample unbiasedness or uncertainty coverage.

| Quantity | Exact | Maximum baseline | Quantile prototype |
|---|---:|---:|---:|
| Absolute integral | 0.08313333333 | 0.08308883811 ± 0.00004260940 | 0.08303918744 ± 0.00005072029 |
| Signed integral | 0.00358333333 | 0.00360012666 ± 0.00001187760 | 0.00358669498 ± 0.00001252537 |
| Mean physics trials | — | 310627.4 ± 13431.3 | 342052.2 ± 19328.9 |
| Largest native tail | <1% | 0.99715% | 0.99685% |
| Largest selected tail | <1% | 0.87435% | 0.92180% |

Table uncertainties are empirical standard errors of the 16-run means. The baseline absolute/signed deviations from truth are -1.04/+1.41 empirical standard errors; the quantile deviations are -1.86/+0.27. Absolute-rate scatter divided by RMS quoted error is 0.708/0.882 (baseline/quantile), and signed-rate ratios are 0.979/1.078. Sixteen replicas are insufficient for precise coverage claims.

Corrected final-event observables test the low-coordinate region, narrow-tail region and signed event fraction. Baseline deviations from exact shapes are +1.21, -0.30 and -0.48 empirical standard errors; quantile deviations are +0.03, -0.96 and -0.57. The quantile signed-shape empirical error is larger in this replica set; this alone does not establish a systematic variance change.

Trial totals are 4,970,039 and 5,472,835. The apparent +10.12% quantile work is only 1.34 combined empirical standard errors in mean trials. Because seeds differ and trial counts fluctuate strongly, neither a speedup nor a definite slowdown is established by this analytic sample. Real-worker comparisons use common seeds and are reported separately.

## Existing regression suite against the prototype

The unchanged 57-case integrator suite yields **55 passes and two expected policy-assertion failures**. The failures assert (1) that the mature first proposal still carries an enormous survey maximum floor and (2) that an independently reconstructed historical envelope is a maximum. Both formulas are exactly what the prototype changes. No other failures occurred; full output is saved at `../experiments/quantile/existing_tests.txt`. Before promotion, these assertions must be rewritten to verify the quantile policy independently, with dedicated sparse-bootstrap and retained-outlier cases.

Reproduce the analytic comparison with:

```sh
python validation/ampli_relative_envelopes_20261008/analytic/run.py --seeds 16 --seed-offset 400
```

Copy this artifact directory to scratch first if the recorded results should be preserved.


## Follow-up with paired seeds

For an efficiency comparison, the quantile prototype was rerun with the **same 16 seeds** as the baseline, reusing the baseline records above. This follow-up keeps the algorithm, analytic target and schedule fixed. It is separate from the independent-seed truth diagnostic; its purpose is to reduce noise in comparing work. The runner is `paired.py`, and `paired/efficiency.json` records all per-seed relative changes.

Baseline and quantile totals are **4,970,039** and **5,134,255** trials: the quantile uses **3.30% more**. The mean paired difference is **10,263.5 ± 4,372.0** trials (empirical standard error), or 2.35 standard errors. The mean per-seed percentage change is **+3.63% ± 1.55%**. It is slower in 9 seeds, exactly unchanged in 5, and faster in 2. Approximate two-sided Student-t 95% limits for the mean paired trial difference are +944 to +19,583 trials; the small replica count and adaptive work distribution limit this inference.

This is evidence for a modest efficiency regression on this target, even though the larger +10.12% difference in the independent-seed totals also included seed fluctuations. It argues against assuming a robust quantile is universally better. The real-worker tests and their limited scope must inform the final policy decision.

All 16 additional quantile pools and final selections pass every strict tail check. The largest native and selected tails are 0.99984% and 0.90276%. The largest absolute empirical mean deviations are 1.53 standard errors for rates and 0.41 for final event shapes. These are diagnostics of the unchanged method, not parameter-selection criteria.


## Isolation of the initial survey floor

A third variant keeps historical candidate **maxima** and removes only the asymmetric first-proposal survey floor once more than 200 candidates exist. It was run with the same 16 baseline seeds and all other settings unchanged. See `paired_observed_maximum.py` and `paired_observed_maximum/{records,summary,efficiency}.json`.

Every paired run has **exactly the same trial count, rates, quoted errors, reserve shapes and final selected shapes** as the maximum baseline. Both totals are 4,970,039 trials. All native and collected checks pass; the maximum tail bounds are 0.997150% native and 0.874348% collected. The absolute/signed rate mean deviations remain -1.04/+1.41 empirical standard errors, and final shape deviations remain within 1.21 standard errors.

This isolates the paired +3.30% quantile regression on this analytic target to the relative quantile choices, rather than releasing the initial survey floor. It establishes neither an improvement nor a general equivalence for the floor-removal policy: the analytic target does not expose a consequential floor effect in these runs. The real-worker comparison tests that effect separately.
