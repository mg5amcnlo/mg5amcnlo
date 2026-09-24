# Born spreading

The published Born-spreading construction introduces a normalized function of
the radiation variables, then suggests a proxy for that function. The
normalization has to be read in the measure actually used by the event
integrator. This implementation derives the measure from the generated
weights and directly minimizes the negative S-event mass, rather than using
the proxy.

At fixed underlying Born kinematics, write each grouped S-event subprocess
weight as

\[
  S_\alpha(x_\xi,x_y)=b_\alpha(x_\xi,x_y) f(u,v)+c_\alpha(x_\xi,x_y),
\]

where \(\alpha\) labels a grouped partonic channel. The Born matrix element
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

The table uses 40 by 40 equal bins in these Born-measure coordinates. Its
normalization is

\[
  \int_0^1 du\int_0^1 dv\,f(u,v)=1,
  \qquad a_{ij}=\frac{1}{1600},\qquad
  \sum_{ij}a_{ij}f_{ij}=1,\quad f_{ij}\ge0.
\]

This preserves the signed Born integral at fixed Born kinematics, not just
its average over the training sample. Constant endpoint-offset factors in
the radiation Jacobians cancel between the spread and unspread integrals.
The bin is saved for each fold during the Born calculation, before later
real-emission mappings overwrite the FKS variables. Calibration and
application both use that saved bin.

For each calibration point, fold and grouped partonic channel, let \(b\) be
the Born contribution and \(c\) the sum of the other S terms before
spreading. The MINT importance weight and physical prefactors are already in
these values. The objective is the integrated negative event mass,

\[
  \widehat J(f)=\sum_{\text{sample rows}}[-(b f_{ij}+c)]_+,
  \qquad [z]_+=\max(0,z),
\]

subject to the weighted normalization above. This is a separable convex
piecewise-linear problem. A nonzero-\(b\) row changes slope at
\(f_{ij}=-c/b\); rows with \(b=0\) are constant in the objective. The code
sorts these slope intervals by marginal cost per unit of normalized bin mass
\(a_{ij}f_{ij}\), then allocates the unit normalization to the cheapest
intervals. Flat minima are resolved by choosing factors closest to one.

Step 0 first adapts the ordinary MINT grid, then uses 800,000 MINT points to
fit the factors and 200,000 separate points to validate them. Contributions
from every fold retain their own radiation bin; the validation uncertainty is
computed across each complete folded MINT point. The table is used only when
validation shows a reduction exceeding two estimated standard errors;
otherwise it is set to the unit table. Later MINT steps load the versioned
`born_spreading.dat` sidecar and check its process dimensions. Version 2 uses
the Born measure above. Version-1 tables used an incorrect raw-coordinate
normalization and are rejected; rerun integration step 0 to refit them.

The factor multiplies each type-2 Born contribution and its scale coefficients
before PDF evaluation, event grouping and absolute values. H-event terms are
unchanged, and the existing shower-scale calculation continues to use the
underlying Born momenta. Set `born_spreading = False` in the run card to
disable calibration and application.

The normalization regression integrates a nonconstant fitted table against
the production ISR, massless FSR and massive FSR soft-map Jacobians at fixed
Born kinematics. It also checks table round-tripping and rejection of the
obsolete format. The separate MINT restart regression checks that the first
iteration after calibration averages its samples correctly.

For a 13 TeV `p p > t t~ [QCD]` check with Pythia8 matching, mt=173 GeV,
fixed muR=muF=173 GeV, nn23nlo PDFs, MC@NLO-Delta off and folding 1,1,1,
the corrected implementation gives an S-event negative fraction of
8.06 +/- 0.07% without spreading and 4.17 +/- 0.09% with spreading:
a relative reduction of 48.3 +/- 1.2%. The signed cross sections are
762.6 +/- 1.7 pb and 762.4 +/- 2.0 pb, respectively. Each configuration
uses one million production-weight integration points summed over the five
Born integration channels, with independent adapted grids. Residual virtual
S events are counted separately from nonvirtual S events, and H events are
excluded from the negative-fraction calculation. This measures the matched
unweighting integrals, not a showered event sample. A separate paired test
on common grids gives a signed change of -0.07 +/- 1.33 pb.

References: [Frederix and Torrielli, arXiv:2310.04160](https://arxiv.org/abs/2310.04160);
[Che, *Reducing Negative Weights in MC@NLO by Improved Implementation of Born
Spreading* (Lund University, 2024)](https://lup.lub.lu.se/student-papers/record/9157443).
