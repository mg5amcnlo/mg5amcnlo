# PYTHIA 8.318 versus the production MC subtraction

Audit date: 27 September 2026. Repository revision: `a9e1a51c5`, branch
`MCcntRefactor_Sfun`. This audit changes no production routines.

Status correction, 9 October 2026: the ISR projection finding below is
historical. Commit `4048df117` (1 October) replaced the shared FKS ISR
forward/inverse maps with the Pythia-like recoil and emitter-only Bjorken
rescaling, including finite endpoint corrections. Commit `62dc0925b`
retained this map for shower matching; `85f6b30aa` stabilized its inverse.
See [the current mapping and validation notes](../native_fks_projection.md).
Full process-level QED matching validation remains outstanding.

Implementation update, 28 September 2026: production fixes for findings
**2, 3 and 5** are implemented. The findings and numerical values below
describe the original audit revision. The new compiled regressions are in
`tests/unit_tests/fks/test_pythia8_matching.py`; run them with
`python3 -m unittest tests.unit_tests.fks.test_pythia8_matching`.

- The PYTHIA8 massive geometric factor uses an absolute determinant on
  both physical FKS branches, with a common stable geometry in the
  prefactor and radiation Jacobian. The quoted negative fixture now gives
  `+0.313987919506...`. The absolute geometric prefactor is shared by all
  showers: it converts the FKS phase-space measure, and every shower's
  radiation Jacobian already takes its magnitude. Cross-shower regressions
  check HERWIG6, HERWIG7, PYTHIA6Q and PYTHIA8 on both massive branches,
  including accepted points and independent finite-difference determinants.
  At the quoted negative fixture, HERWIG6, HERWIG7 and PYTHIA6Q veto the
  second solution; those support decisions are preserved by the shared fix.
- The scalar `g -> gg` term includes the guarded global-recoil dead-cone
  factor, and the launch script explicitly supplies `recoilDeadCone=on`.
  Its two guarded denominators exchange under `z -> 1-z`, making the
  factor symmetric. The existing AP kernel, complementary `fks_Hij`
  weights, identical-particle counting and colour connections therefore
  reproduce the summed labelled PYTHIA weights. Production tests include
  two massless spectators with a massive composite recoil and both local
  dipole and randomized start-scale boundaries.
  The correction is restricted to the audited two-incoming hard context;
  one-mother/MEC contexts retain their previous kernel. The analytic
  collinear branch retains its exact AP coefficient, so PYTHIA's guarded
  numerator floor cannot introduce a residual singular coefficient.
- `xjacPY8` has an explicit `tiny=1d-5`. Initializing it also exposed a
  nonuniform massive soft expansion for small masses. The massive fractions,
  evolution variable, Jacobian and support/damping scale now use exact
  energy-conservation expressions that retain finite-`xi` terms and avoid
  cancellation of `E-k` and `1-z`. Endpoint checks use 80-digit references
  and optimized, infinity-initialized and signaling-NaN/FPE-trapped builds.

The updated audit reproducer checks signed scalar measures at all 1,800
FSR points: none are negative and the maximum relative coefficient error
is `8.8e-14`. The first Jacobian is `28658.690470045796` in both ordinary
and infinity-initialized builds. The ordered recoil fixture remains
`0.7530955643618137`; its production scalar sum is tested separately.
These scalar regressions do not establish full process-level S/H matching.
The ISR projection finding was addressed by the later mapping changes noted
above. The deliberately retained angular, infrared, mass-scheme and
G-prescription differences still require their separate validation.

The nine matching tests and 49 selected existing FKS regressions pass. The full
production counter and its module dependencies also compile against fresh
W-process export includes. The broader Born/export/shower-card command
reports 47 tests: 36 pass, seven fail and four raise errors. A diagnostic
rerun reproduced all eleven unsuccessful tests:

- The seven Born-library comparison tests (`test_dy`, `test_extra_born`,
  `test_loonly`, `test_qed_charge_correlations`,
  `test_shared_provider_retains_both_contexts`, `test_ttbar`, `test_wjet`)
  stop at the `homogeneous_G` signature assertion before comparing matrix
  elements. This is a Python compatibility defect in the unchanged
  `madgraph/iolibs/born_support.py`: the tested Python 3.14.6 has no
  `ast.Num`, and the caught `AttributeError` makes every coupling degree
  uncertified, selecting `full_model_state`. An in-memory diagnostic using
  `ast.Constant` and its `value` field certifies all SM coupling degrees.
  The exported Born libraries compile; these seven amplitude comparisons
  remain unverified by the failing suite.
- Three export tests (`testIO_test_wprod_fksew`, `test_w_nlo_gen_qed`,
  `test_z_nlo_gen_qed`) require the absent `loop_qcd_qed_sm` UFO model.
  Automatic download fails with DNS lookup errors and raises
  `MadGraph5Error: Model not found locally and Impossible to connect any of us servers`.
- `test_w_nlo_gen_gosam` requests GoSam with
  `low_mem_multicore_nlo_generation=True`, which the interface rejects.
  The test is marked `@test_manager.bypass_for_py3`; the direct `unittest`
  invocation does not honor that repository-specific bypass, causing the
  fourth error. The repository's custom test runner honors the marker.

These failures occur in unchanged Python code or test setup, independently
of the PYTHIA matching corrections. That broader suite is not reported as
passing. An isolated fresh W-generation/export regression passes.

## Conclusion

**The implemented subtraction is not the exact first-order expansion of the
configured PYTHIA shower.** Much of the radiation kinematics is correct, but
there are explicit counterexamples to equality:

1. The ISR underlying Born projection differs from PYTHIA's, including the
   incoming momentum fractions, scattering invariants, and PDF arguments.
2. A massive FSR prefactor has the wrong sign on the second physical FKS
   solution. The production dead-zone check accepts these points.
3. PYTHIA applies a gluon recoil dead-cone factor absent from the subtraction.
4. Its azimuthal models, heavy-flavour corrections and infrared regularization
   are not reproduced by the implemented AP kernels and G replacements.
5. `xjacPY8` reads an uninitialized local threshold.

These findings concern QCD matching. They do not rely on electroweak NLO
corrections or dressed-lepton beams. They establish failures of an exact
shower-expansion identity, rather than a list of processes that cannot be
generated. Numerical effects on complete matched predictions have not been
measured here.

The [implementation plan](pythia8318_implementation_plan.md) distinguishes
repairs from the azimuthal, low-scale and G-function differences required
for fixed-order subtraction. The S-event clarification below, added on
27 September 2026, specifies how the ISR map repair must coexist with FKS.

## Sources and scope

The official [release tags](https://gitlab.com/Pythia8/releases/-/tags) identify
`pythia8318`, commit `1e5ae092`, released 16 September 2026, as the latest release
at the audit date. The workspace configuration points to 8.313. I downloaded
the official 8.318 source archive and inspected its shower implementations and
XML settings. Archive SHA256:
`a7fe2c2343b911376b1590b71a0125dd562aac267daa5c75d61cf6f7a77fe515`.

Primary PYTHIA sources:

- [SimpleTimeShower.cc, tag pythia8318](https://gitlab.com/Pythia8/releases/-/blob/pythia8318/src/SimpleTimeShower.cc):
  recoil selection 2277–2366; QCD evolution 2668–3030; kinematics 4090–4240;
  gluon polarization 7264–7308.
- [SimpleSpaceShower.cc, tag pythia8318](https://gitlab.com/Pythia8/releases/-/blob/pythia8318/src/SimpleSpaceShower.cc):
  QCD evolution 935–1553; II kinematics 2593–2610 and 2838–2923.
- [Timelike shower settings](https://gitlab.com/Pythia8/releases/-/blob/pythia8318/share/Pythia8/xmldoc/TimelikeShowers.xml)
  and [spacelike shower settings](https://gitlab.com/Pythia8/releases/-/blob/pythia8318/share/Pythia8/xmldoc/SpacelikeShowers.xml).

The `mc_counterterms` module in
`Template/NLO/SubProcesses/montecarlocounter.f` exports `set_QCD_flows`,
`compute_MCsubtraction_kl`, `compute_delta`, and `bogus_probne_fun`.
It also owns the active shower invariants and G functions, previously in
`kinematics_module`. Its history interface includes `prepare_mc_kinematics`,
`fill_father_and_ileg`, `get_qMC`, and `mc_shower_scale_mass`; the history driver
shares the father index and G factors for native-history restoration and FKS
replacement terms. The invariant formulas, massive fractions and G-function
helpers are private. The two identical `get_zeta` implementations are unified.
The former `kinematics_module` has been folded into `fks_phase_space_helpers`
in `genps_fks_helpers.f`. This lower-level phase-space module provides general
momentum and radiation-coordinate utilities without process initialization
or active point state; callers supply the soft direction when recovering
`y_ij` at an endpoint. `scale_module` retains starting
scales and event-output state, including Born-only paths. Scale generation,
storage and partner selection now receive the active emitter mass bound or
father explicitly, so scale generation does not depend on counterterm internals.
Production callers use explicit module interfaces and import only their
required entry points. Born preparation, connection evaluation, splitting
kernels, shower maps and support checks are private module procedures.
Flow validation owns its partner/flow scratch tables; only its special-gluon
flags persist between evaluations. Each history passes an `mc_kernel_limits` value
to the kernels, so evaluating them no longer depends on a prior call that
sets global limit flags. COMMON blocks shared with the surrounding FKS code,
native-Born adapters and Sudakov tables retain their existing layouts.
The relevant PYTHIA procedures include `zPY8`, `xiPY8`, `xjacPY8`, and
`get_dead_zone`; the comparison also uses the inverses in `genps_fks.f`
and `get_mc_lum`, `compute_MC_subt_term`, and `add_wgt` in `fks_singular.f`.
The archived `docs/pythia8_shower_subtraction.f90` is not called by production
and was not substituted for these routines in the numerical checks.

The counterterm driver prepares two barred Born vectors for the selected
flow and correction order once per native history, then passes them to each
distinct colour connection. This removes `/to_amp_split_bornbars/` and the
unused arrays for other flows and orders. The normalization still sums all
leading Born flows in their original order. A double connection
to the same gluon partner retains both weight entries while reusing the kernel
evaluation. `get_mbar` shares its ISR/FSR spinor-ratio calculation through
`mc_born_azimuth_phase`. Delta matching separates stopping-scale reconstruction
and live-connection selection into `get_delta_stopping_scales` and
`get_delta_connections`. Each Born leg has a `delta_connections` record holding
partners, starting/stopping scales, masses and Sudakov types.
`delta_leg_probability` evaluates its Sudakov and PDF factors from explicit
inputs; `compute_delta` multiplies those probabilities and updates event scales.
These structural changes were checked by compilation and linking; the numerical
comparisons recorded below have not been rerun for this refactor.

The comparison concerns the first QCD emission from the hard Born system,
with standard Simple showers, global FSR recoil, global II ISR recoil, and no
MPI/decay radiation superimposed. Subsequent local-recoil emissions and showers
of H events do not determine the first-order shower term multiplying Born.
Resonance-decay matching and FxFx merging are outside this check.

## Settings actually supplied by this branch

`Template/NLO/MCatNLO/Scripts/MCatNLO_MadFKS_PYTHIA8.Script:556` supplies:

| Settings | Value |
|---|---|
| `TimeShower:globalRecoil`, `globalRecoilMode` | on, 2 |
| `TimeShower:nMaxGlobalBranch`, `nMaxGlobalRecoil` | 1, 1 |
| `TimeShower:limitPTmaxGlobal` | on |
| Both showers' `pTmaxMatch`, `pTmaxFudge` | 1, 1 |
| `TimeShower:dampenBeamRecoil` | off |
| `TimeShower:weightGluonToQuark` | 1 |
| `SpaceShower:rapidityOrder` | off |
| Both showers' `alphaSorder`, `alphaSvalue`, `alphaSuseCMW` | 1, 0.118, false |

`SpaceShower:dipoleRecoil` remains at its default **off**. The script does not
disable `TimeShower:recoilDeadCone`, either shower's `phiPolAsym` or
`phiPolAsymHard`, or `SpaceShower:phiIntAsym`; all default to **on**.
It also does not remove the shower cutoffs or ISR `pT0` screening.
The ordinary `Pythia83.cc` driver supplies no override for these settings.

The shower-card defaults in `madgraph/various/shower_card.py:202` are ISR MECs
off, FSR MECs on, `MEextended` off and `MEafterFirst` off. With `MEextended=off`,
`findMEtype` disables MECs for ordinary two-mother hard production. MECs can
still operate for eligible one-mother systems/decays. Setting
`MEafterFirst=off` alone does not disable a first-emission MEC. An exact
comparison must specify which systems receive MECs.

The literal `nPartonsInBorn=-1` also does not implement the documented H/S
multiplicity discrimination by itself. This concerns later treatment of H
events; it is not the counterexample used below.

## What equality requires

Let $d\mu_n=dx_A\,dx_B\,d\Phi_n/(2\hat s_n)$, with PDFs outside the measure.
For one connection, the resolved first-order shower distribution has the form

$$
 d\mu_B B_c(\Phi_B)\,
 \frac{\alpha_s}{2\pi}\frac{dt}{t}\,dz\frac{d\phi}{2\pi}
 P_c(z)\,W_c\,\Theta_c\,L_c.
$$

Here $\Phi_R=F_c(\Phi_B,z,t,\phi)$ must be the **shower** map. $W_c$ includes
mass and azimuthal factors; $\Theta_c$ includes the actual support and start
scale. For ISR, $L_c$ contains the backward PDF ratio, including $1/z$ when
written in terms of ordinary $f(x)$ rather than $xf(x)$.

Matching requires the same distribution after transformation to real phase
space, including its underlying Born point. Equality of $z$ and $t$ alone
does not imply equality of the distribution.

### Radiation variables: correct in the interior

Write $\xi=\xi_{\rm FKS}$, $y=y_{ij}$ and $S=\hat s_R$.
For a massless ISR emission, production returns

$$z=1-\xi,\qquad t=\frac{S\xi^2(1-y)}2.$$

These equal PYTHIA's $z=\hat s_B/\hat s_R$ and $t=(1-z)Q^2$.

For FSR let $m$ be the continuing parton's mass, $M$ the invariant mass of
**all other hard final particles**, and $w=2p_r\cdot p_e$ with $p_e^2=0$.
The implemented expressions are

$$
 z=1-\frac{S\xi(m^2+w)}{w(S+m^2+w-M^2)},\qquad t=z(1-z)w.
$$

For $m=0$, this reduces to the energy fraction of the continuing daughter in
the hard-system rest frame. For $m>0$, it correctly undoes PYTHIA's massive
daughter rescaling. PYTHIA uses a massless AP scalar kernel for $Q\to Qg$
without MECs; adding a massive AP correction there solely because $Q$ is
massive would not reproduce that source code.

`get_qMC` also equals $\sqrt t$ for PYTHIA8 in the generic kinematics. The
generic comment questioning its equality for all showers is not evidence of
a PYTHIA8 mismatch.

### Full Jacobian: right magnitude, with a massive-branch sign defect

Set $s_p=m^2+t/[z(1-z)]$, $\beta_d=(s_p-m^2)/s_p$ and
$\lambda_B=\lambda(S,m^2,M^2)$. Factorizing the daughter-pair phase space gives

$$
 d\mu_R=d\mu_B\,J_{\rm FSR}\,dz\,dt\,d\phi,\qquad
 J_{\rm FSR}=
 \frac{\beta_d(S+s_p-M^2)}{32\pi^3\sqrt{\lambda_B}\,z(1-z)}.
$$

The global boost preserves the recoil mass and the internal spectator
measure. The FKS and PYTHIA global-FSR Born projections agree in this case.
For massless global II ISR, the corresponding measure is
$J_{\rm ISR}=1/[32\pi^3(1-z)]$.

`xjacPY8` is only $|\partial(z,t)/\partial(x,y)|$, but production also contains
`xfact_ileg*` and the FKS measure. Removing the FKS regulating factors, the
coefficient multiplying $g_s^2P(z)/t$ is

$$ C=\frac{\texttt{xfact}\,\texttt{xjac}}{\xi^2(1-y)}. $$

For FSR it must be $1/(16\pi^3 N_pJ_{\rm FSR})$; ISR has the additional $1/z$
from backward evolution. The compiled production expressions reproduce these
**magnitudes**. Thus a missing complete Born integration-coordinate Jacobian
is not the issue identified here.

However, `xfact_ileg3` (`montecarlocounter.f:1299`) retains the sign of

$$ A=2-\xi\left(1-\frac{E_r}{|\mathbf p_r|}y\right). $$

On the second massive FKS solution, $A$ can be negative. `xjacPY8` takes an
absolute value, as does the FKS phase-space weight, leaving an unphysical
negative density. A concrete accepted point, with a sufficiently high start
scale, is

| Quantity | Value |
|---|---:|
| $\sqrt S$, $m$, $M$ in GeV | 1000, 173, 173 |
| $z$ | 0.23921928965797884 |
| $t$ in GeV squared | 41764.76709871376 |
| $\xi$, $y$ | 0.827453585225583, -0.7456076322312254 |
| Production $C$ for $N_p=1$ | -0.3139879195073465 |
| Shower measure requires | +0.3139879195061142 |

This point passes the **compiled production `get_dead_zone`**, including the
local-dipole limit. The ordinary $Q\to Qg$ AP kernel is positive here. The
first-order shower probability cannot have this sign. `git blame` dates the
current prefactor to June 2025, before the recent radiation-inversion commit.

## ISR: an explicit recoil and luminosity mismatch

For emission from beam A, PYTHIA's inverse must recover

$$\bar x_A=z x_A^R,\qquad\bar x_B=x_B^R.$$

It also undoes the II recoil transformation in `SimpleSpaceShower::branch`.
The production FKS inverse instead computes

$$
 \omega=\sqrt{\frac{2-\xi(1+y)}{2-\xi(1-y)}},\qquad
 \bar x_A^{\rm FKS}=x_A^R\sqrt z\,\omega,\quad
 \bar x_B^{\rm FKS}=x_B^R\sqrt z/\omega.
$$

They coincide in the emitting collinear limit, but generally differ.
`get_mbar` evaluates `sborn_native` at this FKS Born point. Its beam-2 rotation
is a coordinate convention; it does not reconstruct the PYTHIA recoil.

An independently constructed PYTHIA II emission has
$x_A^R=x_B^R=0.2$, $z=0.6$, $t=10240\ {\rm GeV}^2$, $\xi=0.4$, $y=0.2$:

| Quantity | PYTHIA inverse | Production |
|---|---:|---:|
| Born $x_A$ | 0.12 | 0.1473576795226014 |
| Born $x_B$ | 0.20 | 0.1628690142091911 |
| $2\bar p_A\cdot\bar p_3/\hat s_B$ | 0.35 | 0.2651103574351323 |
| Mother PDF $x_A$ | 0.20 | 0.2455961325376690 |
| Other-beam PDF $x_B$ | 0.20 | 0.1628690142091911 |

The PDF entries follow the actual `get_mc_lum` prescription
$(\bar x_A^{\rm FKS}/z,\bar x_B^{\rm FKS})`; its additional
$(1-\xi)/z$ equals one. The changed scattering invariant demonstrates more
than a longitudinal-frame difference.

There is no later repair of the event map: `add_wgt` stores PDF fractions in
`bjx`, but stores `p_ev` for type 13. The native-history driver explicitly
assigns the outer real momenta and outer boost to those H records
(`driver_mintMC.f:1281`). Summing/repartitioning FKS histories does not by
itself replace the Born projections used inside their amplitudes and PDFs.

Correcting this requires the PYTHIA ISR Born fractions and recoil projection,
which can be reconstructed from radiation kinematics without recovering Born
integration-channel coordinates.

### Global recoil and the existing fixed-Born Jacobian conversion

The [PYTHIA aMC@NLO documentation](https://pythia.org/latest-manual/aMCatNLOMatching.html)
requires global timelike recoil for the subtraction construction. The
current audit already uses that configuration and global II ISR. Their
phase-space factorizations must be tested as complete products; different
intermediate coordinates alone are not a mismatch.

In particular, for the massless ISR interior, the existing FKS radiation
measure including the real/Born flux conversion is

$$J_{\rm FKS,rad}=\frac{S\xi}{64\pi^3}.$$

With `xjac=S*xi^2/2` and `xfact=4*xi*(1-y)/(z*S*N_p)`,

$$
 J_{\rm FKS,rad}\,
 \frac{\mathrm{xfact}\,\mathrm{xjac}}{\xi^2(1-y)}
 =\frac{\mathrm{xjac}}{16\pi^3zN_p}.
$$

After multiplication by `g_s^2*P(z)/t`, this is the shower radiation
measure expressed in FKS variables, including its backward `1/z`.
The existing S evaluator uses `B(b)` at the output Born state and backward
PDF fractions `(b_A/z,b_B)` for beam A. A separate Born-density ratio is
not required merely to perform this fixed-Born radiation conversion.

This leaves two different validation questions: whether the complete S
integral at fixed `b` has the shower's support and weights, and whether the
H density at a fixed physical real event uses that event's actual shower
parent. The ISR counterexample above concerns the latter. The successful
measure identity does not establish equality of the Born amplitude and PDF
arguments of those two inverse maps.

### S-event projection and FKS regularization

Any ISR repair must also validate the integrated MC addition to S events.
`add_wgt` in `fks_singular.f` copies the FKS `p1_cnt` counterevent into its
Born companion and uses that companion for type 12. This fact alone does
not prove a defect in the present S integral: its coefficient is evaluated
at that generated Born state, as described above.

The integrated raw MC addition needed for a shower parent `b` is

$$
 I_{\rm PS}(b)=\sum_h\int d\mu_R\,
 P_hD_h^{\mathrm{raw},G}(\Phi_R)
 \delta_B\bigl(b,\beta_h(\Phi_R)\bigr).
$$

If the same corrected real density is used in the FKS-projected S
construction, it has `pi_FKS` instead of `beta_h`. These conditionally
integrated densities generally differ even when the scalar real-measure
conversion is correct. This comparison concerns two projections of the
same corrected density, not an automatic diagnosis of the original S
evaluator. Test the existing fixed-Born construction first and retain it
if its full identity, including support, holds.

When an S-projection change is required, FKS still regularizes the integral.
A practical construction retains the
fully FKS-subtracted S weight, using the corrected raw density, and adds

$$
 \Delta S_{\rm raw}(b)=\sum_h\int du\,
 [\rho_h^{\rm PS}(b,u)-\rho_h^{\rm FKS}(b,u)],\qquad
 \rho_h^X=J_h^X[P_hD_h^{\mathrm{raw},G}](F_h^X(b,u)).
$$

Both terms hold the same output `b` fixed. Local finiteness requires a
common endpoint construction with matching soft, collinear and overlap
coefficients, full measures and regulator conventions. Finite domain or
boundary conversion terms must be included where necessary. The alternative
is direct FKS endpoint subtraction at fixed shower Born, with the finite
integrated-counterterm conversion derived explicitly. Details and tests are
in [work package C](pythia8318_implementation_plan.md#6-work-package-c-make-the-s-contribution-born-differentially-correct).

These are conditional repair options; the current audit does not validate
the complete original or proposed S integral. They do not alter the
successful scalar-Jacobian magnitude check. For the audited global
FSR map the two Born projections already agree, so no projection correction
is required there. The massive-sign and gluon recoil-weight findings remain
separate. Preserve the azimuthal, low-scale and G-function completions needed
for the exact fixed-order limits.

## Phase-space boundaries and kernels

### Upper bounds that do agree

For global FSR, PYTHIA uses

$$D_g=(\sqrt S-M)^2-m^2,\qquad
 z_3^\pm=\tfrac12(1\pm\sqrt{1-4t/D_g}),$$

and requires

$$s_p S<z(1-z)(S+s_p-M^2)^2.$$

These are production's `zp3/zm3` and `zp2/zm2` bounds. With
`limitPTmaxGlobal=on`, both also impose the local dipole bound

$$t<\tfrac14[(m_{rk}-m_k)^2-m^2].$$

For massless global ISR, PYTHIA's transverse-momentum reality condition
reduces to $t\le\hat s_B(1-z)^2/z$. Production's square-root inequality is
algebraically the same condition. This agreement assumes the same Born
invariants and start scales; it does not cure the ISR projection mismatch.

The smooth `compute_damping_weight` is consistent with averaging a step
function over the branch's randomized shower-start scale: `damping_inv`
samples the CDF whose survival function is `1-emscafun`. Its mere presence is
not a discrepancy with a shower using a fixed SCALUP for each event.

### Gluon kernels: respect identical-particle counting

Without MECs or mass weights, PYTHIA's labelled $g\to gg$ kernel per colour
end is

$$P_{\rm end}(z)=\frac{C_A}{2}\frac{1+z^3}{1-z}.$$

Production uses the symmetric AP kernel divided by two, together with
`fks_Hij=h_damp(1-z)` for massless global FSR and identical-particle/history
factors. The identity

$$P_{\rm end}(z)+P_{\rm end}(1-z)=\tfrac12P_{gg}^{\rm AP}(z)$$

means a per-labelled-history comparison alone would give a misleading
factor-of-two finding. For permutation-symmetric observables with identical
support, both daughter orderings must be combined. I do **not** identify this
kernel partition alone as an inclusive normalization error.

### Missing gluon recoil weight

`SimpleTimeShower.cc:2927` multiplies gluon emission by a recoil dead-cone
factor when `recoilDeadCone=on` and the active recoiler is massive. Under
global recoil, this is the **composite** recoiler mass, so even several
individually massless spectators can activate it. In terms of
$r=M^2/S$, $v=s_p/S$, $x_1=(1-r+v)z$, and $x_2=1+r-v$, away from numerical
guards the factor is

$$D_{\rm rec}=1-\frac{r}{x_1+x_2-1-r}
                  \frac{1+r-x_2}{1-r-x_1}.$$

Production has no such multiplier. A physical point with
$S=10^6\ {\rm GeV}^2$, $M=400\ {\rm GeV}$, $z=0.8$ and
$t=15000\ {\rm GeV}^2$ gives $D_{\rm rec}=0.7530955644$.
It satisfies the global kinematic bounds; a sufficiently large local dipole
and start scale allow it. This is a finite **first-order** weight, not a
running-coupling correction. Symmetrizing the two gluon daughters does not
remove it.

### Azimuth, masses, and low scales

- **Azimuth:** the code uses Born helicity interference through `bornbarstilde`
  and Q terms, multiplied by `gfactazi`. PYTHIA uses its history/colour-based
  polarization approximation, including an assumed production fraction for
  eligible hard gluons. ISR additionally has a colour-interference azimuthal
  bias. That bias can act for a quark radiator, where production's helicity
  interference term is zero. The models are not the same function.
- **Heavy flavours:** with nonzero shower c/b masses, PYTHIA's backward
  $Q\to Qg$ and $g\to Q\bar Q$ kernels contain $m_Q^2/t$ terms, special
  high-$x$ and threshold bounds, and a forced threshold conversion algorithm.
  Production's ISR AP kernels are massless. `weightGluonToQuark=1` removes
  extra FSR options but retains the pair threshold factor
  $\sqrt{1-4m_Q^2/s_p}$. Shower masses are supplied separately in the launch
  script. These are familiar differences between a massless hard calculation
  and a massive shower; they are not evidence that heavy-flavour QCD processes
  have newly become unsupported.
- **Low scales:** PYTHIA has evolution cutoffs and an extra mass-reduced FSR
  cutoff. ISR evolves with $dt/(t+p_{T0}^2)$, whereas the counterterm contains
  $dt/t$. The usual unregularized perturbative comparison discards these
  infrared power effects, but the literal expansion of the configured
  finite-cutoff generator retains them. Even with fixed $\alpha_s$ they differ.

## G replacements, debugging factors, and version changes

The complete production weight is not just the bare kernel above. The
default integration input uses `Gsoft` parameters `(1,-0.1)` and `Gazi`
parameters `(-1,-0.1)`. The raw soft contribution is switched off for
$\xi\le0.01$, and `compute_native_NLOPS_weights` supplies exact FKS
soft/collinear replacements. Those are matching constructions, not PYTHIA's
branching probability. Their effect must be included when proving NLO
accuracy; they cannot be silently equated to the shower expansion.

In addition, the default non-Delta path currently uses
`bogus_probne_fun`, mode 2: a coupling-independent debugging factor changing
from zero below 0.5 GeV to one above 10 GeV. It multiplies MC and real/G
contributions with an accompanying S-event redistribution. It is not a
physical PYTHIA Sudakov. All explicit hard-point counterexamples above have
$\sqrt t>10$ GeV and survive setting this factor to one.

A physical Sudakov $1+O(\alpha_s)$, running-coupling scale changes and CMW
conversion affect a single-emission expansion only at the next order,
provided the leading coupling is the same. They should not be confused with
the finite, coupling-independent weights and recoil changes above.

Comparing the local 8.313 source with 8.318 shows that the principal recoil
maps, scalar AP weights and recoil dead-cone factor discussed here already
existed in 8.313. New optional timelike `cEmit*`/`cSplit*` finite terms and
`hEmit*`/`hSplit*` higher-order terms are zero by default. Nonzero `c*` settings
would require additional first-order terms; `h*` terms start one order higher.
New resonance-recoil options do not establish a mismatch in the hard-process
first emission by themselves.

## Uninitialized threshold

`montecarlocounter.f:3551` declares a local `tiny` in `xjacPY8` but never
assigns it. The `tiny` in `kinematics_module` is private, and the analogous
constants in `zPY8`, `xiPY8` and `dinvariants_dFKS` are local parameters.
There is no COMMON or saved assignment providing this value.

Compiling the same routine with different initialization of otherwise unset
locals changes the first test point's `xjacPY8` from
`28658.690470045796` to `74170.06693771241`. A normal static-storage build
typically supplies zero storage and selects the generic expression, but this
does not supply the intended endpoint threshold. It should be explicitly
defined and its limit expansions checked together with `zPY8` and `xiPY8`.

## Reproduction and limits of the checks

Run:

```sh
python3 docs/audits/check_pythia8318.py
```

The script uses a temporary build directory and requires `gfortran`. It
extracts and compiles the unchanged production inverses, radiation functions,
prefactors and `get_dead_zone`. Reference forward momenta are generated from
the 8.318 branch formulas; the full Jacobians are independently obtained by
phase-space factorization.

Results over 1,800 allowed FSR points plus the ISR example:

| Check | Result |
|---|---:|
| Maximum absolute error in $z$ | $5.6\times10^{-15}$ |
| Maximum relative error in $t$ | $2.7\times10^{-13}$ |
| Maximum FSR Born momentum error divided by 1000 GeV | $2.3\times10^{-14}$ |
| Maximum relative error in scalar Jacobian **magnitude** | $2.0\times10^{-11}$ |
| Accepted points with a negative massive scalar prefactor | 12 |

The script is a reproducer of the current findings, including the defects.
It does not run a full PYTHIA shower or compare integrated event distributions,
and the sampled errors are not bounds over singular endpoints. Production's
analytic endpoint approximations need a separate stability check after
initializing `tiny`.

A separate check now observes actual first ISR emissions through PYTHIA's
`doVetoISREmission` hook, then runs the compiled production FKS inverse and
scalar factors on the recorded real events:

```sh
python3 docs/audits/check_pythia8_global_isr.py --pythia-prefix /path/to/pythia8313
```

With the installed **8.313** library, seed `270926`, `gg -> t tbar` at
2 TeV, stable tops, global II recoil (`SpaceShower:dipoleRecoil=off`),
FSR/MPI/hadronization disabled, and the explicit settings in the observer:

| Check | Result |
| --- | ---: |
| Actual first global-II gluon emissions | 30 (18 from A, 12 from B) |
| Maximum fixed-Born interior scalar-measure relative error | `7.8e-16` |
| Maximum absolute FKS versus recorded Born fraction difference | `0.0958686481` |
| Maximum absolute difference in `2 p_A.p_t / s_B` | `0.0545348898` |
| Maximum deviation from one of the PDF-argument probe ratio | `0.3011740680` |

For the last maximum, the recorded Born fractions are
`(0.1847173581,0.2905814765)` and the FKS inverse gives
`(0.2358266465,0.2276055036)`. The corresponding scattering invariants are
`0.5143825332` and `0.5689174230`. Thus the different inverse Born states
are also seen with actual global shower emissions, while the existing
interior Jacobian conversion passes.

The compiled `get_mc_lum` returns PDF fractions
`(0.4928291799,0.2276055036)` for this point and `xlum_mc_fact=1`.
The actual shower mother/other-beam fractions are
`(0.3860212806,0.2905814765)`. A probe with `f_A(x)=x`, `f_B(x)=1`
therefore gives a current/required luminosity ratio `1.2766891482`,
although the geometric scalar coefficient agrees. These are analytic
PDF-argument probes, not predictions using a fitted PDF set or full Born
amplitudes. They isolate what the existing measure conversion does not
change at a fixed real event.

This is an 8.313 kinematic and PDF-argument check,
not a new 8.318 event run or an integrated MC@NLO accuracy test. The
fixed-Born measure check uses the analytic phase-space/flux factor above
and compiled `xfact*xjac`; full domains and S/H weights remain to be tested.

For a general observable, an uncompensated difference between a shower
distribution and its subtraction can leave a term of the schematic form

$$\int(d\sigma_{\rm PS}^{(1)}-d\sigma_{\rm MC})[O_R-O_B].$$

A finite difference proportional to $\alpha_s$ is not automatically a
higher-order effect. Agreement of total rates, or of singular limits alone,
does not prove its absence. Establishing NLO+PS accuracy for the new matching
formula therefore requires an observable-level expansion using its actual
maps and all S/H redistribution terms. The explicit map, sign and missing
weight counterexamples already exclude the stronger claim of exact equality
to PYTHIA 8.318.
