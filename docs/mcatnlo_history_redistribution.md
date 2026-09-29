**MC@NLO history redistribution within existing Born directories**

Design proposal for the `Incomplete MC history sum` issue on
`MCcntRefactor_Sfun`. The recommendation is to redistribute complete H
contributions within the Born contexts available in each `P*` directory,
using an outer partition normalized over that local set. The algebra below
establishes conservation of the summed unshowered weight. Infrared behaviour,
event assignments and integration performance require validation before this
can be adopted as a matching prescription.

The current implementation forms the H weight for an outer FKS history by
summing native histories of the same real process. Some of those histories
require a Born evaluator belonging to another directory. For example, the
real channel \(qg\to t\bar t q\) has projections onto both
\(gg\to t\bar t\) and \(q\bar q\to t\bar t\). The exporter records this
missing local coverage, and the driver stops. The proposed method lets those
histories contribute through their own directories, without making every
Born evaluator available in every directory.

At a fixed physical real-flavour configuration and phase-space point
\(\Phi_R\), denote a complete native H density by

$$
H_b(\Phi_R)=P_b(\Phi_R)
\left[S_b(\Phi_R)R(\Phi_R)-M_b(\Phi_R)\right].
$$

Here \(b\) labels a fully specified native history, \(S_b\) is the original
FKS partition, \(P_b\) is that history's no-emission factor, and \(M_b\)
includes the raw MC counterterm and its G replacement. This notation
suppresses the PDFs, cuts, coupling orders, statistical normalization and
counterevent measures, all of which must remain those of the native
contribution. All densities in a sum must be expressed in the same real
phase-space measure and external-particle ordering.

The equations refer to colour-summed densities. When colour flows are
sampled, the existing native flow probabilities and inverse sampling weights
must be retained; equality to the colour-summed result then holds in
expectation.

The current global redistribution is

$$
\widehat H_a=S_a\sum_b H_b,
\qquad \sum_a S_a=1.
$$

The construction is implemented in
[`repartition_MC_H`](../Template/NLO/SubProcesses/driver_mintMC.f), with the
native terms produced by
[`compute_native_NLOPS_weights`](../Template/NLO/SubProcesses/fks_singular.f).

For each fixed real channel, let \(B_d\) be the histories assigned to
directory \(d\) whose native Born contexts can be evaluated there. These
sets must form a disjoint, exhaustive partition of the labelled histories
across the output. They are defined separately for each real channel; they
do not collect unrelated real matrix elements merely because they share a
directory. Define

$$
H_d=\sum_{b\in B_d}H_b,
\qquad
W_d=\sum_{c\in B_d}S_c,
\qquad
\omega_{a|d}=\frac{S_a}{W_d}\quad(a\in B_d).
$$

The proposed local redistribution is

$$
\boxed{\displaystyle
\widetilde H_a=\omega_{a|d}H_d
=\frac{S_a}{\sum_{c\in B_d}S_c}
  \sum_{b\in B_d}P_b(S_bR-M_b),\qquad a\in B_d.}
$$

This changes only the outer redistribution. The original \(S_b\) inside
each native contribution, the G replacement, the shower damping and the
native Born projection are retained. The same outer coefficient multiplies
every H record belonging to that complete native contribution.

Since \(\sum_{a\in B_d}\omega_{a|d}=1\), the sum in each directory is

$$
\sum_{a\in B_d}\widetilde H_a=H_d.
$$

Consequently,

$$
\sum_d\sum_{a\in B_d}\widetilde H_a
=\sum_d H_d
=\sum_b H_b
=\sum_a\widehat H_a.
$$

The total H density is therefore unchanged at each real point after summing
the directories. With the S-event terms retained, this also preserves the
unshowered distribution for observables depending on the physical momenta
and flavours. In a Monte Carlo implementation, separately sampled
directories realize this identity through their weighted samples.

For a directory containing only one history of the real channel,
\(\omega_{a|d}=1\) and \(\widetilde H_a=H_a\). If several histories are
available locally, their complete contributions are redistributed among
those histories. In the ttbar example, the two underlying Born contributions
can thus be supplied by their respective directories.

The normalization is essential. Dropping foreign histories while retaining
the old outer factor would give

$$
\sum_d\sum_{a\in B_d}S_aH_d
=\sum_d W_dH_d,
$$

which generally differs from \(\sum_d H_d\). Skipping only foreign MC
counterterms while keeping a different real-term sum also fails to implement
the identity. The object redistributed here is always the complete
\(P_b(S_bR-M_b)\).

The history sets must include the labelled permutations and orientations
already represented by the exported history table. A denominator obtained
by summing only reduced FKS representatives is generally insufficient.
Existing orbit factors, Born/real statistical denominators and native-to-outer
measure conversions must be accounted for exactly once. A global metadata
check can establish coverage and ownership without generating additional
matrix elements. The present export logic is in
[`write_mc_history_files`](../madgraph/iolibs/export_fks.py).

Numerically, \(S_a/W_d\) should be evaluated from local relative FKS
weights, cancelling their common global denominator before division. This
avoids dividing two quantities that both approach zero in a limit belonging
to another directory. The current
[`fks_Sij`](../Template/NLO/SubProcesses/fks_Sij.f) explicitly warns about
numerical limits when called for a non-native pair, so summing arbitrary
calls to that routine is not automatically a stable implementation. Boundary
values require consistent limiting expressions; a nonzero \(H_d\) must not
be discarded because the computed \(W_d\) underflows. This local outer
partition should be implemented separately from the original FKS partition
used inside the real and replacement terms.

A statistical alternative can retain the original global scalar H density
for every outer label. At fixed \(\Phi_R\), sample a source directory
\(d\) with probability \(q_d(\Phi_R)>0\), evaluate \(H_d/q_d\), and
choose an outer label \(a\) with probability \(S_a(\Phi_R)\). Then

$$
\mathbb E\!\left[
\mathbf 1_{\mathrm{owner}=a}\frac{H_d}{q_d}
\;\middle|\;\Phi_R\right]
=S_a\sum_d H_d
=\widehat H_a.
$$

This is an estimator for a signed density: the selection probabilities are
nonnegative, while the event weight retains the sign of \(H_d\). The
probabilities must have support wherever a contributing block is nonzero.

For actual integration, the phase-space sampling density must also be
included. For example, choose directory \(d\) with probability
\(\pi_d\), generate a real point with normalized density
\(\rho_d(\Phi_R)\), and choose \(a\) with probability \(S_a\).
The weighted estimator is

$$
w=\frac{H_d(\Phi_R)}{\pi_d\rho_d(\Phi_R)}.
$$

For an observable \(O\), its expectation at a given outer label is

$$
\mathbb E[\mathbf 1_{\mathrm{owner}=a}wO]
=\int d\Phi_R\,O(\Phi_R)S_a(\Phi_R)\sum_d H_d(\Phi_R).
$$

The density \(\rho_d\) represents the complete proposal, including channel
selection and mapping branches. In code, its inverse is represented by the
appropriate integration Jacobians and sampling weights. Those factors must
not be applied again if the native integration weight already includes them.
Likewise, an explicit \(1/\pi_d\) is needed when randomly selecting a
directory, not when adding independently computed directory integrals.

The statistical method must sample complete locally subtracted blocks.
Sampling the real contribution and its counterterm independently can destroy
their numerical cancellation. Even with complete blocks, cancellations
between directories occur statistically and may increase the variance and
negative-weight fraction. Reproducing the original outer label also does
not by itself reproduce its conditional colour and shower-scale assignments;
those would require a separate construction that avoids a foreign Born
evaluation.

Both proposals require an infrared check beyond the conservation identity.
The local weights obey \(0\leq\omega_{a|d}\leq1\), so they do not spoil
absolute integrability if each complete \(H_d\) is already integrable.
However, cancellation that requires terms from different blocks would
invalidate that premise. Limits belonging to another directory must
therefore be tested explicitly, alongside the native soft, collinear and
soft-collinear limits. Sampling efficiency additionally depends on the
second moment under the chosen phase-space proposal.

Conservation of the unshowered density is also distinct from equivalence of
the showered prediction. The shower acts on the assigned event state,
including colour and starting scales, as expressed by the MC@NLO generating
functional in [Frederix et al., arXiv:2002.12716, equations (2.1) and
(3.1)](https://arxiv.org/pdf/2002.12716). The local proposal changes which
outer owners can receive a given native contribution. Its effect on those
assignments, resummation behaviour and negative-weight rates must be checked.
The local redistribution and statistical estimators above are proposals
derived here, rather than prescriptions established by that reference.

The recommended first implementation is the deterministic local
redistribution. It keeps every native counterterm in its existing Born
context and requires no communication between running subprocesses. A
prototype should proceed through these checks:

1. Verify global coverage of labelled histories and unique ownership for
   every real-flavour channel, including permutations and flavour-group maps.
2. Implement and test the local outer partition, including its limiting
   values, while retaining the original partitions inside native weights.
3. Check \(\sum_{a\in B_d}\widetilde H_a=H_d\) using complete physical
   native weights, and compare with the global redistribution wherever all
   histories can be evaluated for a reference calculation.
4. Test native and foreign soft/collinear limits for ttbar and W+jet, with
   Drell–Yan providing a control where the history sum is already complete.
5. Repeat unshowered fixed-order/MC@NLO comparisons and compare integration
   variance and negative-weight rates; then validate event colours, shower
   scales and showered distributions, including the physical Delta path.

The existing incomplete-history guard should be replaced by checks of the
new method's local coverage and global ownership, rather than merely
disabled. The established result at this stage is the algebraic equality of
the summed unshowered densities under the stated assumptions. No numerical
or shower validation of this proposed method has yet been performed.
