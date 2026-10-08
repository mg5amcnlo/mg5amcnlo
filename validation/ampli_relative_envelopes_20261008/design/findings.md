# Relative proposal envelopes and internal stream sampling

This is a design assessment for item 5 of the efficiency review. Production and benchmark conclusions are recorded separately.

## Upstream behavior needs a correction

The upstream `~/space/git/AmpliCol/master/SimpleIntegrator/simple_integrator.f03` has two different envelope calculations. `integral_compute_fmax` (lines 870–905) uses the `max(int(0.05*N),1)`-th largest historically reweighted candidate weight for provisional unweighted counts. `integral_compute_fmax_next_iter` uses a related quantile for the next candidate-storage cutoff.

However, `check_overweight` (lines 794–854) reconstructs the **maximum** of historical candidate weights for each iteration before the final common-rank selection. The current MG5 maximum-based final relative scales therefore follow upstream finalization. Using candidate quantiles for final relative scales is a new optimization, not a restoration of upstream final behavior. Upstream also checks overweight *excess*, whereas MG5 requires the full cross section of overweight events to remain below 1%; that stronger MG5 criterion must remain unchanged.

## What a relative envelope changes

For a fixed sampling proposal `q_k(x)`, physical positive target `f(x)`, and proposal-wide scale `M_k`, a common threshold `z` produces acceptance `min(1, f/(q_k*z*M_k))`. Multiplying each retained event by its native overweight factor `max(1, f/(q_k*z*M_k))` gives expected corrected density `f/(z*M_k)`. Each epoch changes only a scalar coefficient, so pooling epochs preserves the target shape for fixed proposals, scales and threshold. It is not necessary for the relative scale itself to bound all observed weights.

Scaling **all** `M_k` by one common positive constant changes `z` reciprocally and leaves actual thresholds and final events invariant. Only ratios matter. A survey-maximum floor on proposal 1 alone changes those ratios. Applying an equivalent common normalization to every proposal would not change final selection.

This argument is conditional on fixed scales and thresholds. Scales estimated from the same finite sample, rank-based thresholds and optional stopping already make the native finite-quota scheme approximate. A robust scale does not prove unbiasedness. Independent-seed rate and event-shape diagnostics are required. It is preferable to define a deterministic quantile policy in advance than to select whichever relative scales give the largest observed event yield.

## Quantile prototype

The isolated prototype at `../experiments/quantile/simple_integrator.f90` replaces final relative maxima with the top-five-percent candidate order statistic once more than 200 candidates exist. For 200 or fewer candidates it keeps the observed maximum and the initial survey floor. The sparse bootstrap matches the existing storage-cutoff bootstrap; it avoids depending on only a few candidate ranks. The mature scale for the first proposal drops the survey floor because a one-proposal floor otherwise preserves the extreme outlier imbalance that the experiment is meant to test.

The candidate set is storage-biased, so this statistic is a robust relative scale of the **stored pool**, not a 95% quantile of all physics draws or of the cross section. No weight is clipped. Historical Jacobian reweighting uses every retained candidate, including currently event-ineligible history. Batches sharing a proposal share the same scale. Every batch keeps its original storage cutoff, and common rank selection cannot fall below any of them.

The all-trial, generated-reserve, worst-final-subset and actually collected full-weight tail checks remain authoritative. An outlier below the quantile count is still retained, counted in tail mass and assigned its full native correction factor. Rate estimates continue to include every attempted draw with its original proposal weight.

## Stream changes and mixture correctness

The MC@NLO adapter currently chooses virtual/nonvirtual streams using survey absolute-rate fractions, frozen for all generation. Its proposal is the joint density `p_s*q_k(x)`. The current candidate weights already include `1/p_s`; because `p_s` stays fixed, historical reweighting needs only the coordinate-map Jacobian ratio.

A separate envelope for each physical stream changes the expected corrected stream density to `f_s/(z*M_s)`. The resulting mixture generally has the wrong relative stream weights unless event weights or stream quotas explicitly compensate. Therefore separate stream envelopes cannot be inserted as a drop-in extension of the existing per-proposal normalization.

Changing stream probabilities during generation requires storing each candidate's stream and birth probability. Historical reweighting then needs both the coordinate ratio and `p_birth/p_historical`. Shared history additionally needs stream-specific map provenance if the coordinate maps differ. Virtual control-variate fits must remain frozen unless changing their target decomposition is accounted for. This is a larger extension than robust proposal-wide envelopes.

A smaller possible experiment is to choose a different **frozen** stream probability from the survey. If the observed stream maxima are `B_nonvirtual` and `B_virtual`, the observed minimax bound is attained at `p_virtual=B_virtual/(B_nonvirtual+B_virtual)`. Taking the maximum of this and the existing rate fraction avoids undersampling a stream already allocated by rate. It needs no additional history metadata because it remains constant across proposals. It is not a guarantee of optimal final-event efficiency: a virtual contribution below 1% may already be accommodated by the existing full-tail checks, and survey maxima are noisy.

## Regression requirements before promoting a variant

- Exercise the 200/201-candidate bootstrap boundary and the exact kth-largest convention, including ties and fewer than 20 high weights.
- Reconstruct mature historical scales independently from coordinates and birth Jacobians, using the candidate-weight order statistic rather than priorities or candidate correction factors.
- Preserve identical scales for all batches of one proposal, folded-map invariance and the deterministic eight-proposal history rule.
- Prove with an explicit large outlier that quantile scale selection does not remove its full trial mass or overweight correction, even if it blocks completion.
- Verify that all rate moments remain unchanged when only final relative scales are recomputed, and that failed completion probes consume no RNG or training state.
- Keep strict 1% equality rejection, heterogeneous LHE-factor checks, storage-floor protection and final collection checks.
- Compare predetermined independent seeds on signed analytic targets with zeros, a narrow tail, actual folding and at least several distinct adaptive proposals. Inspect rates, uncertainty scatter and corrected final-event shapes, not only trial counts.
- Compare representative real workers against an exact snapshot of the current implementation. Do not infer a complete 300K speedup from worker or retrospective capacity measurements.
