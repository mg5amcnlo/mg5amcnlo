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
  S_\alpha(x_\xi,x_y)=B_\alpha f(x_\xi^2,x_y^2)+C_\alpha,
\]

where \(\alpha\) labels a grouped partonic channel. The code's Born
prefactor is independent of the radiation coordinates \(x_\xi,x_y\); the
quadratic radiation-map Jacobians multiply the real and subtraction terms,
not this Born term. Therefore the Born contribution is distributed in the
raw unit MINT coordinates, and its preservation condition is

\[
  \int_0^1 dx_\xi\int_0^1 dx_y\,f(x_\xi^2,x_y^2)=1.
\]

The table is constant in 40 by 40 equal bins of \(u=x_\xi^2\) and
\(v=x_y^2\). If \(f_{ij}\) is a table entry, its raw-coordinate bin area is

\[
  a_{ij}=\left(\sqrt{\frac{i}{40}}-\sqrt{\frac{i-1}{40}}\right)
         \left(\sqrt{\frac{j}{40}}-\sqrt{\frac{j-1}{40}}\right),
  \qquad \sum_{ij}a_{ij}f_{ij}=1,\quad f_{ij}\ge0.
\]

This area factor is needed because uniform bins in the squared variables are
not equal-area bins in the raw MINT coordinates. It also handles the affine
offsets in the FKS energy and polar maps: those offsets do not change the
relative bin areas.

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
`born_spreading.dat` sidecar and check its process dimensions.

The factor multiplies each type-2 Born contribution and its scale coefficients
before PDF evaluation, event grouping and absolute values. H-event terms are
unchanged, and the existing shower-scale calculation continues to use the
underlying Born momenta. Set `born_spreading = False` in the run card to
disable calibration and application.

References: [Frederix and Torrielli, arXiv:2310.04160](https://arxiv.org/abs/2310.04160);
[Che, *Reducing Negative Weights in MC@NLO by Improved Implementation of Born
Spreading* (Lund University, 2024)](https://lup.lub.lu.se/student-papers/record/9157443).
