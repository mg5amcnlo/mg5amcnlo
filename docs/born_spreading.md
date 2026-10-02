# Born spreading

The published Born-spreading construction introduces a normalized function of
the radiation variables, then suggests a proxy for that function. The
normalization has to be read in the measure actually used by the event
integrator. This implementation derives the measure from the generated
weights and minimizes the sampled negative mass of unfolded S events.

At fixed underlying Born kinematics, write each grouped S-event subprocess
weight as

\[
  S_{\alpha k}(x_\xi,x_y)=b_{\alpha k}(x_\xi,x_y) f_k(u,v)+c_{\alpha k}(x_\xi,x_y),
\]

where \(\alpha\) labels a grouped partonic channel and \(k\) the sampled
Born FKS sector. The residual virtual contribution is sampled separately
and is excluded from \(c\). The Born matrix element
is independent of the radiation coordinates, but its integration weight is
not: `compute_prefactors_nbody` includes `jac_cnt(0)`, which contains the
radiation-map Jacobians. For an initial-state or massless final-state emitter,
the normalized Born radiation measure is

\[
  d\mu_B=4x_\xi x_y\,dx_\xi\,dx_y=du\,dv,
  \qquad u=x_\xi^2,\quad v=x_y^2.
\]

For a massive final-state emitter, only the first solution carries a Born
contribution. Its range is \(0<x_\xi<r\), with
\(r=\xi_{\max}/\texttt{xinorm}\), which depends on the Born kinematics
and radiation angle. The energy Jacobian supplies \(2x_\xi/r\), and
`compute_prefactors_nbody` supplies another \(1/r\). Thus

\[
  d\mu_B=\frac{4x_\xi x_y}{r^2}\,dx_\xi\,dx_y=du\,dv,
  \qquad u=(x_\xi/r)^2,\quad v=x_y^2.
\]

Each FKS sector has its own table of 40 by 40 equal bins in these
Born-measure coordinates. Every table satisfies

\[
  \int_0^1 du\int_0^1 dv\,f_k(u,v)=1,
  \qquad a_{ij}=\frac{1}{1600},\qquad
  \sum_{ij}a_{ij}f_{kij}=1,\quad f_{kij}\ge0\quad\text{for each }k.
\]

This preserves the signed Born integral at fixed Born kinematics, not just
its average over the training sample. Constant endpoint-offset factors in
the radiation Jacobians cancel between the spread and unspread integrals.
The bin and Born FKS sector are saved for each fold, before later
real-emission mappings overwrite the FKS variables. Calibration and
application both use that saved pair.

For each calibration point, fold and grouped partonic channel, let \(b\) be
the Born contribution and \(c\) the sum of the other nonvirtual S terms
before spreading. Type-14 residual virtual weights are not included: the
negative event mass is \([-(bf+c)]_++[-V_{\rm res}]_+\), and the second
term is constant in \(f\). Combining the streams before taking their signs
would allow cancellations unavailable during generation. The MINT importance
weight and physical prefactors are already in these values. With folding
1,1,1, the fitted objective is the variable part of the negative event mass,

\[
  \widehat J(f)=\sum_{\text{sample rows}}[-(b f_{kij}+c)]_+,
  \qquad [z]_+=\max(0,z),
\]

subject to a separate normalization constraint for each sector. Each sector
is solved independently as a separable convex piecewise-linear problem.
A nonzero-\(b\) row changes slope at \(f_{kij}=-c/b\); rows with \(b=0\)
are constant in the objective. The code
sorts these slope intervals by marginal cost per unit of normalized bin mass
\(a_{ij}f_{kij}\), then allocates each sector's unit normalization to its
cheapest intervals. Flat minima are resolved by choosing factors closest to
one. Slopes consistent with cancellation roundoff are set to zero using the
initial slope and summed jumps as their scale; otherwise a numerically flat
tail could attract a large artificial spike. Sectors without training
information retain a unit table.

Step 0 first adapts the ordinary MINT grid, then uses 800,000 MINT points to
fit the factors and 200,000 separate points to validate them. These extra
points skip the exact one-loop residual virtual calculation. The Born-based
average virtual remains in the nonvirtual S weights used by the fit.
Contributions from every fold retain their own radiation bin; the validation
uncertainty is computed across each complete folded MINT point. For larger
folding factors, fitting and validation use the unfolded negative mass as a
proxy, while generation takes signs after folding. The tables are used only when
validation shows a reduction exceeding two estimated standard errors;
otherwise they are set to unit tables. Later MINT steps load the versioned
`born_spreading.dat` sidecar and check its process dimensions and every
sector's normalization. Version 3 stores the sector tables consecutively.
Version 1 used an incorrect radiation measure; version 2 shared a table
across sectors and included residual virtual weights in fitting. Both older
formats are rejected; rerun integration step 0 to refit them.

The factor multiplies each type-2 Born contribution and its scale coefficients
before PDF evaluation, event grouping and absolute values. H-event terms are
unchanged, and the existing shower-scale calculation continues to use the
underlying Born momenta. Set `born_spreading = False` in the run card to
disable calibration and application.

The normalization regression integrates a nonconstant fitted table against
the production ISR, massless FSR and massive FSR soft-map Jacobians at fixed
Born kinematics. It also checks table round-tripping and rejection of the
obsolete formats. The sector regression fits opposite shapes through the
production S-grouping routine, verifies virtual-stream separation and checks
application to the saved sector and fold. The separate MINT restart regression
checks that the first iteration after calibration averages its samples correctly.

For a 13 TeV `p p > t t~ [QCD]` check with Pythia8 matching, mt=173 GeV,
fixed muR=muF=173 GeV, nn23nlo PDFs, MC@NLO-Delta off and folding 1,1,1,
the sector-dependent implementation gives an S-event negative fraction of
8.15 +/- 0.07% without spreading and 2.61 +/- 0.09% with spreading:
a relative reduction of 68.0 +/- 1.1%. The previous shared-table
implementation gave 4.17 +/- 0.09% with the same physics settings.
The signed cross sections are 761.3 +/- 1.4 pb and 764.3 +/- 2.2 pb,
consistent within their statistical errors. Each configuration uses one
million production-weight integration points summed over the five Born
integration channels, with independently adapted grids. Residual virtual
S events are counted separately from nonvirtual S events, and H events are
excluded from the negative-fraction calculation. These are matched
unweighting integrals, not a showered event sample.

Paired measurements on common baseline grids give a negative fraction of
2.79 +/- 0.05% with spreading and a signed change of +1.04 +/- 1.43 pb.
On the restarted spreading grids, the paired signed change is
-0.35 +/- 1.42 pb. The normalization constraints hold separately in every
sector; these finite-sample rate differences are consistent with zero.

References: [Frederix and Torrielli, arXiv:2310.04160](https://arxiv.org/abs/2310.04160);
[Che, *Reducing Negative Weights in MC@NLO by Improved Implementation of Born
Spreading* (Lund University, 2024)](https://lup.lub.lu.se/student-papers/record/9157443).
