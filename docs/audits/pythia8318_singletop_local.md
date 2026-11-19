# Single-top local recoil with unmodified Pythia 8.318

Local MC@NLO is implemented for five-flavour, on-shell t-channel single-top
QCD production without merging or MC@NLO-Delta. It uses the stock Pythia
kernel and an FKS G replacement inside MG5. Both ends of each quark
initial-final colour line are included. This stock prescription needs no
Pythia source patch. A separate, opt-in alternative uses a private patched
build and `PythiaLocalCollinearPatch=.true.`; see the
[patch instructions](pythia8318_local_collinear_patch.txt). The existing
stock-recoil process and its G prescription remain unchanged by default.
A second opt-in alternative, `pp_tj_local_2patch`, also corrects the massive
top soft current and removes the remaining local G replacement; see its
[prescription and later-emission limitations](pythia8318_local_2patch.txt).

The analytic subtraction uses the usual massless-incoming five-flavour
approximation, including incoming charm/bottom. Pythia retains its own
particle masses and threshold treatment. The outgoing top mass is retained
in the map, phase-space support and exact FKS soft current.

## Kernel and G prescription

For real incoming `pa`, outgoing colour partner `pf` and massless emission
`k`, the local projection is

```
zI   = 1 - pf.k / pa.(pf+k)
w    = 1-zI
Q2   = 2 pa.k
D    = 2 pBorn_a.pBorn_f
zeta = pa.pf / pa.(pf+k) = 1-zI*Q2/D
pBorn_a = zI*pa
pBorn_f = pf+k-w*pa
```

The radiation measure, including the flux ratio, is
`dzI dQ2 dphi/(32*pi^3)`. The stock quark IF density is

```
D_stock/B = 2*g_s^2*CF*(1+zI^2)/(zI*Q2*(1-zI)).
```

This follows from the IF probability in
[Cabouat and Sjostrand, section 3.2, eq. (23)](https://arxiv.org/html/1710.00391)
and the [8.318 source](https://gitlab.com/Pythia8/releases/-/blob/pythia8318/src/SimpleSpaceShower.cc).
The two FKS orientations partition this density with
`chiI=pf.k/(pf.k+pa.k)` and `chiF=1-chiI`. Real-state PDFs include the
backward incoming rescaling; no additional PDF factor `1/zI` is applied.
Backward gluon conversion uses `TR*(zI^2+(1-zI)^2)` and only the incoming
orientation. It needs no soft G replacement.

The stock density has the correct incoming AP limit. At the massless final
endpoint its coefficient of `2*g_s^2/s_fk` is `2*CF/(1-zeta)`, whereas QCD
requires `CF*(1+zeta^2)/(1-zeta)`. On the top line its soft limit contains the
pair eikonal but misses the top self term:

```
D_stock_soft/B = 2*g_s^2*CF * pa.pf/(pa.k*pf.k)
D_QCD_soft/B   = D_stock_soft/B - g_s^2*CF*m_t^2/(pf.k)^2
```

These differences make bare `R-D_stock` weights non-integrable. They are
handled by the exact FKS replacements, without changing either stock kernel.
In each native history the effective subtraction is schematically

```
D_G = damping * {g(w)*D_stock
      + (1-g(w))*[S + (1-g(1-y_FKS))*(C-SC)]}.
```

`S`, `C` and `SC` carry their native FKS sector functions and endpoint/real
measure ratios. They are not bare scalar kernels evaluated at the real
Born point. The exact soft current supplies the top self term, the exact
collinear counterterm supplies the outgoing AP numerator, and subtracting
`SC` counts their common limit once. No ad hoc massive factor multiplies
the massless-line shower density.

For width `h=PythiaLocalGWidth`, the compact profile is

```
g(v) = 0                              for v <= 0
       u^2/[u^2+(1-u)^2], u=v/h        for 0 < v < h
       1                              for v >= h.
```

The default is `h=0.01`. Crucially, `w=1-zI` vanishes both soft and
final-collinear, including hard outgoing collinear emission. An energy-only
soft profile would miss the latter limit. The angular profile goes to zero
only at the forward FKS collinear endpoint. Both approach zero quadratically.
For gluon conversion both replacement complements remain zero.

`compute_pythia_local_counterterm` multiplies the raw stock density by `g(w)`
and the starting-scale damping. Its returned soft replacement complement is
`(1-g(w))*damping`, so the replacement also disappears above the shower
starting scale. The existing native FKS driver adds the replacement exactly
once with opposite signs in S and H; the complete native contributions are
summed before H repartitioning. The local mode keeps the ordinary MC@NLO
no-emission factor equal to one.

This is the usual G-function accuracy qualification: it restores integrable
weights but does not make Pythia's singular kernels exact. For an infrared-safe
observable the first-order mismatch has the form
`integral (D_stock-D_G)*(O_real-O_Born)`. It is power suppressed as the
transition region shrinks; at finite width it is not automatically of higher
perturbative order. Vary the width to assess this dependence. See
[Alwall et al., eq. (2.117), pp. 48-49](https://arxiv.org/pdf/1405.0301#page=49),
whose G convention is complementary to the shower multiplier used here.

## Running

Regenerate process exports and integration grids. Old grids/events made with
the withdrawn modified-shower prescription must not be reused. For example:

```text
import model loop_sm-no_b_mass
define p = g u c d s b u~ c~ d~ s~ b~
define j = p
generate p p > t j $$ w+ w- [QCD]
```

In `Cards/FKS_params.dat`, set

```text
#FKSPythiaLocalRecoil
1
#FKSExplicitSum
.true.
#MCExplicitKLSum
.true.
#PythiaLocalGWidth
0.01d0
```

Keep `FKSISRMapping=0`, `FKSFinalRecoil=0`, `parton_shower=PYTHIA8`,
`mcatnlo_delta=False` and `ickkw=0`. Unsupported Born structures, resonance
decays, QED corrections and separate initial/final matching jobs are rejected.

The launcher selects stock settings, including `SpaceShower:dipoleRecoil=on`,
`TimeShower:globalRecoil=off`, ISR/FSR on, MECs off and a stable top. It disables
QED/weak radiation and the extra smooth ISR screening (`pT0Ref force=0`),
while retaining the ordinary shower cutoff. It does not force the light,
charm or bottom masses to zero. Pythia handles both ends of a Born quark IF
line through SpaceShower; TimeShower suppresses its duplicate contribution.

## Validation

Fresh 13 TeV exports use `m_t=173 GeV`, `ptj=10 GeV`, NN23NLO PDFs and
`H_T/2` scales. All four subprocesses pass the production ME/MC singular-limit
checks and 20/20 virtual-pole checks. The MC checks include 100 points per
endpoint/history. Additional checks at fixed sampled energy coordinate `x_E^2=1e-4`
and `y_FKS=0.9999` test the soft-collinear overlap, with zero reported MC
failures in every subprocess. The map supplies the energy upper bound
(`xi` for final radiation, `w` for incoming local recoil). Native inversions
clear limit-test flags only while recovering coordinates; the diagnostic
restores those flags for amplitude evaluation so both sides use the same
`1e-12` precision threshold. Leaving the production `1e-6` shortcut active
on only the subtraction side created a false soft offset at one nearly
beam-collinear test point; that diagnostic inconsistency is fixed.

Independent grid setups and full stock-Pythia showers give:

| G width | Inclusive integration (pb) | Showered events | Negative weights |
| --- | --- | --- | --- |
| 0.005 | 132.1 +/- 2.5 | 1000 | 247 |
| 0.01 | 134.7 +/- 1.6 | 1000 | 267 |
| 0.02 | 134.9 +/- 1.7 | 1000 | 238 |

All LHE weights/scales are finite, and every showered event retains one
stable top with the specified mass. MPI and hadronisation are off. The rates
are mutually compatible and compatible at this statistical precision with
the earlier fixed-order reference, `130.7 +/- 0.46 pb`. Coarse showered top
pT/rapidity comparisons with the default have maximum marginal bin pulls of
2.10 and 2.48 for the half/double widths (15 populated bins each). This is a
statistical smoke test, not a precision differential determination of the
remaining G dependence.

Stock Pythia completes all 3000 events but reports internal retries: 20-22
momentum-check failures per sample, 13-21 negative-dipole-mass warnings and
1-6 ISR weight-above-unity warnings. The half-width sample also reports two
near-threshold loop errors. No LHE event is lost. These messages are recorded
in the validation JSON/logs; G functions do not fix shower evolution or its
trial envelopes. There is no custom headroom, source change or new Pythia
parameter hiding these messages.

The focused numerical/support tests and the read-only shower audit can be run
with

```sh
python3 -m unittest tests.unit_tests.fks.test_pythia_local_matching \
  tests.unit_tests.fks.test_pythia_local_recoil \
  tests.unit_tests.fks.test_recoil_selection \
  tests.unit_tests.fks.test_soft_col_limits \
  tests.unit_tests.fks.test_mc_dead_zones \
  tests.unit_tests.various.test_pythia8
python3 docs/audits/check_pythia8_singletop_local.py /path/to/stock/pythia8318
python3 docs/audits/check_pythia8_singletop_local.py /path/to/stock/pythia8318 \
  --lhe /path/to/events.lhe.gz --steering /path/to/Pythia8.cmd
```

The audit verifies official source hashes and links external diagnostic
drivers to the existing library. It never patches, instruments or rebuilds
Pythia. Its first-emission hooks record stock branches and terminate each
diagnostic event after that branch. The optional LHE check uses complete
showers without hooks and records coarse top distributions. The detailed
[validation record](pythia8318_singletop_local_validation.json) keeps the
withdrawn modified-shower results separately as historical data; they do not
validate this implementation.
