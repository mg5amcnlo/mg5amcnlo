# The `q qbar` `f_0` deficit is an NLO effect, and it is not in the diagrams

`SPIN_COEFFICIENTS.md` and `data/numbers_qq_seedcheck.txt` left one number open.
On two independently seeded 200 000-event samples, every spin-correlated
MadSpin mode sits low on `f_0` against the fully off-shell NLO four-lepton
truth:

| mode | sample 1 | sample 2 | combined `D = mode - truth` | z |
|---|---|---|---|---|
| `madspin` | -2.90 | -3.85 | -0.01353 +- 0.00283 | **-4.77** |
| `onshell` | -2.06 | -2.79 | -0.00971 +- 0.00283 | -3.43 |
| `PA`      | -2.97 | -2.19 | -0.01034 +- 0.00283 | -3.65 |

while the `g g` block of the same table has `madspin` at `+0.11` sigma.

This note answers three questions and leaves one open.

* **Setup.** The two sides are cut, scaled, PDF'd and beamed identically.  The
  one real asymmetry -- the `pt` cut acting on an on-shell `z` on one side and
  on a reconstructed pair on the other -- is worth about `6e-5` on `f_0`,
  i.e. 0.5 % of the deficit.  Section 1.
* **Diagrams.** The truth's four singly-resonant Born diagrams per subprocess,
  which MadSpin structurally cannot carry, move `f_0` by
  **`+0.00059 +- 0.00175`** at LO and **`+0.00102 +- 0.00286`** at NLO.
  Removing them from the NLO truth leaves 92 % of the deficit standing, at
  `-4.41` sigma.  They are not the cause; nor is `Z/gamma*` interference,
  which the truth's `/ a` already removes.  Sections 2 and 3.
* **Order.** The whole LO chain -- `p p > z z` + MadSpin against the LO
  four-lepton truth, on one selection -- agrees to **`+0.00164 +- 0.00273`**.
  The deficit **does not exist at LO**.  Section 4.
* **What is left.** Something that exists only on an MC@NLO sample.  The
  best-motivated member of that class: MadSpin builds its spin density matrix
  from the *tree-level* production matrix element (`MadSpin/decay.py` calls
  `ReweightInterface.get_LO_definition_from_NLO` on any `[...]` process), so an
  MC@NLO sample gets LO spin correlations on NLO kinematics.  The NLO
  correction to `f_0` in this fiducial region is `+0.01652 +- 0.00237`; the
  MadSpin chain picks up `+0.00463 +- 0.00314` of it and misses
  `+0.01189 +- 0.00394`, which is the deficit.  Section 5.  This is a
  *consistency* argument, not a direct measurement: nothing here isolates the
  virtual correction to the density matrix on its own, and nothing here
  excludes a purely technical MC@NLO/MadSpin interaction instead.

Everything below is from the study's own `observables.py`, the study's own
window (`|m(l+l-) - m_Z| < 15 Gamma_Z`) and `pt(l+l-) > 1 GeV`, re-imposed
offline on every sample alike.


## 1. Setup: the two sides are the same run card

`ppzz_nlo/Cards/run_card.dat` and `pp4l_nlo/Cards/run_card.dat` of the
sample-2 run differ in exactly two lines:

```
< 200000 = nevents            > 50000 = nevents
<        = custom_fcts        > .../zz_equivalent_cuts_nlo.f = custom_fcts
```

Everything else is byte-identical after whitespace normalisation, including
`pt_min_pdg = {'23': 1.0}` (present on **both**, natively inert on the
four-lepton process and read from there by the cut file), `bwcutoff = 15.0`,
`fixed_ren_scale`/`fixed_fac_scale = True` with `mur_ref_fixed =
muf_ref_fixed = 91.188`, `pdlabel = nn23lo1` / `lhaid = 230000`,
`ebeam1 = ebeam2 = 6500.0`, `parton_shower = PYTHIA8`, and the lepton cuts
`ptl = etal = drll = drll_sf = mll = mll_sf = 0` that would otherwise bite on a
four-lepton NLO card (`mll_sf` defaults to 30 GeV).  Both param cards carry
`MZ = 91.1880` and `WZ = 2.441404`, so MadSpin's `BW_cut = 15` window and the
cut file's `bwcutoff * WZ` window are the same window, and it is the same one
`observables.M_LO/M_HI` re-imposes.

### The T129 trap, measured

The one place where the two sides genuinely see different objects is the `pt`
threshold: on the production side `pt_min_pdg = {23: 1}` acts on the on-shell
`z` **before** MadSpin reshuffles, on the truth side the cut file acts on the
reconstructed pair.  The analysis re-cut can only remove events, so it cannot
put back a `p p > z z` event killed at `pt(z) < 1` whose decayed pair would
have had `pt(l+l-) > 1`.

Bounded from the data.  Weight fraction with `pt(l+l-) < 2 GeV` on either pair,
and `f_0` there:

| | weight fraction | `f_0` in that region |
|---|---|---|
| truth | 0.00115 | +0.390 |
| `madspin` | 0.00085 | +0.270 |

The hole is `2.9e-4` of the weight, and the sign is the helpful one: the events
MadSpin is missing sit at `f_0 ~ 0.39`, well above its own `0.177`, so putting
them back would *shrink* the deficit.  By `2.9e-4 * (0.39 - 0.18) = 6e-5`, or
`2.4e-4` in the absolute worst case where every missing event is purely
longitudinal.  The deficit is `1.35e-2`.  **Excluded by between one and two
orders of magnitude.**

The full `pt(l+l-)` spectrum near threshold confirms there is no cliff: the
truth/`madspin` weight fractions per bin from 1 GeV up are
`0.00099/0.00073`, `0.00204/0.00139`, `0.00547/0.00507`, `0.01321/0.01196`, ...
-- a smooth, sub-per-mille deficit dying out by 5 GeV.

### It is not a shape difference either

Reweighting every MadSpin sample to the truth's distribution, bin by bin, and
recomputing `f_0` (`madspin` row, combined bar against the truth):

| reweighted in | `D` | z |
|---|---|---|
| nothing (sample 2) | -0.01543 | -3.85 |
| `m_4l` (44 bins) | -0.01738 | -4.22 |
| `pt(4l)` (42 bins) | -0.01473 | -3.65 |
| `m_ee` x `m_mumu` (6x6) | -0.01579 | -3.93 |
| `m_ee` x `m_mumu` x `m_4l` (4x4x10) | -0.01762 | -4.28 |
| `pt(4l)` x `m_4l` (4x16) | -0.01677 | -3.83 |
| rapidity of the four-lepton system (13 bins) | -0.01545 | -3.85 |

No kinematic reweighting removes it; several make it worse.  The difference is
in the decay angles at fixed production kinematics.  The pooled `|cos theta|`
distribution says the same thing in one line -- `madspin` is 1.4 % low below
`|cos theta| = 0.4` and 1.1 % high above 0.6, a smooth monotonic tilt from
`(1 - cos^2)` towards `(1 + cos^2)`, not a localised artefact.


## 2. Diagram content: what the truth has that MadSpin cannot carry

The truth is `p p > e+ e- mu+ mu- / a [QCD]`.  The `/ a` is already in the
study's own `run_zz_nlo.PROC`, and the generated code confirms it took: the
four subprocess directories are `P0_{ddx,dxd,uux,uxu}_mumemepmup_no_a` and
there is no photon propagator anywhere.  **So the brief's leading hypothesis --
a `gamma*` admixture in the truth -- is removed by construction, not by
argument.**  There is no residual: a forbidden particle in MG5 is forbidden in
every internal line of every Born, real and virtual diagram.

What *does* remain is read off `born_conf.inc` of
`P0_uux_mumemepmup_no_a`.  Six Born diagrams per subprocess:

* **Diagrams 5, 6** -- `TPRID = 2`, a t-channel quark, with two s-channel `Z`
  propagators (`SPROP = 23`) hanging off it.  These are `q qbar > z z` with
  both `z` off shell.  This is what `p p > z z` + MadSpin carries.
* **Diagrams 1-4** -- `SPROP = 23` for one lepton pair, then a **lepton**
  propagator (`SPROP = -11, 11, -13, 13`), then another `SPROP = 23` whose
  invariant mass is `m_4l`.  These are singly-resonant: an s-channel Drell-Yan
  `Z*` at `m_4l` decaying to a lepton pair with a `Z` radiated off one lepton
  leg.  Four leptons, but only one of the two *pairs* comes out of a single `Z`
  propagator; the other is not a `Z` at all.  MadSpin's production process has
  two `z` in the final state by construction and can never produce these.

The `g g` contrast is *not* that `g g` lacks such diagrams.  It has them:
`gg4l/SubProcesses/PV0_0_1_gg_epemmupmum_no_a/helas_calls_ampb_1.f` builds
exactly the same singly-resonant lepton-line currents (`W(1,9)`, `W(1,11)`,
`W(1,14)`, `W(1,16)`) and contracts them with `R2_GGZ_0`, and it carries a
`g g > H > Z Z > 4l` diagram on top (`VVS1_3`/`VVS1_0`).  What is true is that
the physical `g g > Z*` amplitude vanishes (Furry plus Landau-Yang), which the
generated counterterms show explicitly -- they come in `+R2_GGZDOWN` /
`-R2_GGZDOWN` pairs that cancel between up- and down-type loops -- and the
Higgs one is far off shell at `m_4l >= 2 m_Z`.  So the `g g` truth *is*
effectively doubly-resonant only.  That would have been a good explanation of
the contrast, if the diagrams mattered.  They do not.


## 3. What the extra diagrams are worth: `+0.0006 +- 0.0018` at LO

Measured, not argued.  The doubly-resonant subset can be generated on its own
by forbidding the leptons as internal propagators:

```
generate p p > e+ e- mu+ mu- / a                 # 6 Born diagrams / subprocess
generate p p > e+ e- mu+ mu- / a e+ e- mu+ mu-   # 2, checked with MG5
```

(The required-s-channel spelling `p p > z z > e+ e- mu+ mu-` is rejected by this
MG5 with *Invalid "> A A >" syntax*.  Forbidding the internal leptons reaches
the same two diagrams from the other side, and both lepton currents are
conserved for massless leptons, so the `k^mu k^nu / MZ^2` pieces of the two `Z`
propagators drop and the subset is the gauge-invariant double-pole part.)

800 000 LO events each, the study's LO run card, **no generator-level cut on
either** (`pt_min_pdg = {}`, all lepton cuts zeroed, a loose `mxx_min_pdg` floor
at 40 GeV well below the window), the study's window and `pt` cut applied
offline to both alike -- so the T129 trap of section 1 is absent from this
comparison by construction:

| sample | sigma [pb] | `f_0` (both) | `f_00` | `f_00 - f_0 f_0` | `f_TT` | `C_kk` |
|---|---|---|---|---|---|---|
| `full4l` (6 diagrams) | 0.020948 | +0.1742 +- 0.0012 | +0.0618 +- 0.0027 | +0.0315 +- 0.0027 | +0.7135 +- 0.0034 | -0.628 +- 0.035 |
| `zz4l` (2 diagrams) | 0.020945 | +0.1736 +- 0.0012 | +0.0582 +- 0.0027 | +0.0281 +- 0.0027 | +0.7111 +- 0.0034 | -0.655 +- 0.035 |

The `full4l` row is the LO reference line of Table 1 of `spin_definitions.tex`,
which is where it is quoted. It is on the table's common selection: the window
and the `pt` cut are applied offline, which reaches the same selection as the
truth rows' `custom_fcts` file because both act on the reconstructed pairs, and
the run's only generator-level threshold -- a 40 GeV floor on the same-flavour
pairs -- sits below the window's 54.567 GeV edge and is measured inert (13.4 %
of the surviving events have `m(e+ mu-) < 40 GeV`, against 12.9 % of the NLO
truth's).

```
full4l - zz4l   f_0 (both)  +0.00059 +- 0.00175   +0.34 sigma
                f_TT        +0.00237 +- 0.00476   +0.50 sigma
                C_kk        +0.02727 +- 0.04883   +0.56 sigma
```

The four singly-resonant diagrams are worth `1.4e-4` of the cross section in
this window and `+0.0006 +- 0.0018` on `f_0`.  At 95 % the effect is bounded by
`|Delta f_0| < 0.0041`, against a deficit of `0.01353 +- 0.00283`.  Their
central value even has the *right* sign for the hypothesis -- they do make the
truth slightly higher -- but they are an order of magnitude too small.

### The same measurement at NLO

The restriction survives the NLO tag, so it does not have to be extrapolated
from LO.  `p p > e+ e- mu+ mu- / a e+ e- mu+ mu- [QCD]`, 8 x 50 000 MC@NLO
events, generated by the study's own `run_zz_nlo.py` helpers with the study's
own NLO run card and `zz_equivalent_cuts_nlo.f` attached, against the study's
two 200 000-event samples combined:

| sample | sigma [pb] | `f_0` (both) |
|---|---|---|
| `truth` -- 6 Born diagrams / subprocess | 0.028665 | +0.19068 +- 0.00202 |
| `truth_zzonly` -- 2, doubly resonant | 0.028612 | +0.18966 +- 0.00202 |
| `madspin` | 0.028650 | +0.17715 +- 0.00199 |

```
truth   - truth_zzonly   f_0 (both)  +0.00102 +- 0.00286   +0.35 sigma
madspin - truth          f_0 (both)  -0.01353 +- 0.00283   -4.77 sigma
madspin - truth_zzonly   f_0 (both)  -0.01251 +- 0.00284   -4.41 sigma
```

**Stripping the truth of every diagram MadSpin cannot carry removes 8 % of the
deficit and leaves `-4.41` sigma.**  The extra diagrams cost 0.18 % of the
cross section and 0.35 sigma of `f_0`, and that is the whole of their effect.


## 4. The deficit does not exist at LO

The same 800 000-event LO truth against 200 000 `p p > z z` LO events run
through MadSpin, one offline selection on all of them:

| sample | `f_0` (both) |
|---|---|
| `full4l` -- LO truth, all diagrams | +0.17416 +- 0.00123 |
| `zz4l` -- LO truth, doubly resonant only | +0.17357 +- 0.00123 |
| `p p > z z` + MadSpin `spinmode = madspin` | +0.17253 +- 0.00244 |
| `p p > z z` + MadSpin `spinmode = none` | +0.33225 +- 0.00236 |

```
full4l - zz_madspin   f_0 (both)  +0.00164 +- 0.00273   +0.60 sigma
zz4l   - zz_madspin   f_0 (both)  +0.00104 +- 0.00273   +0.38 sigma
```

The LO twin of the failing NLO comparison **passes**, at 0.6 sigma, with a
bound of `|Delta f_0| < 0.0070` at 95 %.  `spinmode = none` sits at the
isotropic `1/3` as it must.  Nothing in the chain -- the on-shell production
approximation, the Breit-Wigner reshuffling, the decay-side accept/reject, the
frame convention, the estimator -- costs `f_0` anything at LO.


## 5. What is left: MadSpin's density matrix is tree level

`MadSpin/decay.py`, generating the production squared matrix element:

```python
for proc in processes:
    if '[' in proc:
        commandline += reweight_interface.ReweightInterface.get_LO_definition_from_NLO(proc, mgcmd._curr_model)
```

`p p > z z [QCD]` becomes `p p > z z` plus `p p > z z j` -- Born for the
MC@NLO S-events, real for the H-events, no virtual.  So on an MC@NLO sample
MadSpin carries **LO spin correlations on NLO kinematics**.  The truth carries
NLO ones.

That is enough to account for the number.  `f_0` in this fiducial region is
strongly NLO-sensitive, and the chain inherits only the kinematic part of the
shift:

| | LO | NLO | NLO - LO |
|---|---|---|---|
| truth | +0.17416 +- 0.00123 | +0.19068 +- 0.00202 | **+0.01652 +- 0.00237** (+7.0 sigma) |
| `p p > z z` + MadSpin | +0.17253 +- 0.00244 | +0.17715 +- 0.00199 | **+0.00463 +- 0.00314** (+1.5 sigma) |

```
NLO shift the chain misses     +0.01189 +- 0.00394
observed NLO deficit           -0.01353 +- 0.00283
```

The two agree in size and in sign.  (They are not independent -- both are built
from the same four numbers -- so this is a consistency check, not a second
measurement.)

The split of the NLO samples by extra-parton multiplicity points the same way.
The deficit lives in the S-events, which is where the virtual correction lives
and where MadSpin uses the Born matrix element:

| | truth `f_0` | `madspin` `f_0` | `D` | z |
|---|---|---|---|---|
| `njet = 0` (77 % of the weight) | +0.18595 +- 0.00323 | +0.16977 +- 0.00311 | -0.01619 | -3.61 |
| `njet = 1` (23 %) | +0.21520 +- 0.00614 | +0.20694 +- 0.00657 | -0.00826 | -0.92 |

### What the LO test cannot see, and the sharpest remaining lead

The LO closure of section 4 is strong but it is blind to one whole class of
explanation: LO events are unit weight and have no MC@NLO structure at all --
no S/H split, no counter-events, no FKS subtraction.  So "the deficit does not
exist at LO" excludes everything about the MadSpin chain that is
order-independent, and leaves *anything specific to an MC@NLO sample*.  The
tree-level density matrix is one member of that class, and the best-motivated
one; it is not the only one.

Splitting the NLO comparison by the sign of the event weight shows the
structure is there:

| | truth `f_0` | `madspin` `f_0` | `D` | z | share of the unsigned weight, truth / `madspin` |
|---|---|---|---|---|---|
| `w > 0` | +0.18560 +- 0.00179 | +0.17559 +- 0.00178 | -0.01001 | -3.97 | 0.926 / 0.934 |
| `w < 0` | +0.12725 +- 0.00642 | +0.15508 +- 0.00671 | +0.02783 | +3.00 | 0.074 / 0.066 |

The two subsets disagree in *opposite* directions and do not cancel.  This is
descriptive, not diagnostic: the truth's counter-events are those of
`p p > 4l` and the production sample's are those of `p p > z z`, two different
FKS decompositions, so there is no reason for the subsets to match
individually -- only the signed sums have to.  But it is where an MC@NLO-only
effect would live, and it is the one place a further check could still separate
"the NLO density matrix" from "the MadSpin/aMC@NLO interface".

### The `g g` contrast

This explains the `g g` contrast without needing the diagrams.  The
loop-induced `g g > z z` truth **is** a Born-level calculation -- one loop,
but no QCD corrections on top of it -- and its events are unit weight with no
S/H split.  The production matrix element MadSpin re-derives for the density
matrix is therefore of the *same order* as the truth's, and there is no
higher-order correction to the density matrix to miss and no MC@NLO structure
to mishandle.  The mechanism predicts exactly zero there, and the `g g`
`madspin` row is `+0.11` sigma.

### How much the `g g` block actually excludes

Less than it looks, and this qualifies the contrast rather than the diagnosis.
On the `g g` sample the `madspin` pull is `+0.00037 +- 0.00350`.  That bar
excludes an **absolute** effect of the `q qbar` size (`-0.01353`) at 3.9 sigma,
but the `q qbar` deficit is `-7.1 %` of `f_0`, and `-7.1 %` of the `g g`
truth's `f_0 = 0.0669` is `-0.0047`, which the `g g` bar excludes at only
**1.4 sigma**.  So "the `g g` block shows nothing like it" is a strong
statement about an absolute shift and a weak one about a proportional shift.
It remains a genuine null for the interpretation above, because the mechanism
there predicts *exactly zero* for `g g`, not a scaled-down effect.


## 6. Status

| hypothesis | verdict | number |
|---|---|---|
| seed fluctuation | excluded (base commit) | two samples, `chi2/1 = 0.26-0.45` |
| run-card / cut / scale / PDF asymmetry | excluded | cards differ in `nevents` and `custom_fcts` only |
| `pt` cut on `z` vs on the pair (T129) | excluded | `6e-5` (worst case `2.4e-4`), 0.5 % of the deficit |
| kinematic shape (`m_4l`, `m_ll`, `pt_4l`, `y`) | excluded | reweighting moves the pull by at most 0.4 |
| `Z/gamma*` interference | absent by construction | `/ a` in the truth's own process definition |
| singly-resonant four-lepton diagrams | excluded | `+0.00059 +- 0.00175` (LO), `+0.00102 +- 0.00286` (NLO); removing them leaves `-4.41` sigma |
| anything in the MadSpin chain that exists at LO | excluded | LO twin agrees at `+0.60 sigma`, bound `< 0.0070` |
| **NLO correction to the spin density matrix** | **surviving, consistent** | misses `+0.01189 +- 0.00394` of a `+0.01652 +- 0.00237` NLO shift |
| MC@NLO-specific handling (S/H, counter-events) | **not excluded** | LO events are unit weight, so section 4 is blind to it |

**Localised: all of it.**  The deficit is not in the setup, not in the
diagrams, not in the kinematics and not in anything the chain does that
survives at LO.  It is entirely an MC@NLO-only effect, and 92 % of it survives
stripping the truth of every diagram MadSpin cannot carry.

**Directly measured: none of it.**  Section 5 is an accounting identity plus a
code reading, not an experiment, and it does not separate the tree-level
density matrix from any other MC@NLO-only mechanism.  What would close it is a
fixed-order NLO polarisation extraction of `q qbar > Z Z` with and without the
virtual correction to the density matrix, which is not something this
validation can do with MadSpin as it stands.

Nothing here is a MadSpin bug.  MadSpin is documented to reconstruct spin
correlations from a tree-level production matrix element; the finding measures
what that costs when the input sample is MC@NLO and the reference is a genuine
NLO calculation.  If the paper quotes `f_0` from a MadSpin-decayed NLO sample,
this is the number to quote as the method's systematic: **`0.0135` on
`f_0 = 0.19`, i.e. 7 %.**


## Reproducing

```
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
cd MadSpin/validation/zz_loopinduced/diagram_content

# sections 1, 5 -- no generation at all, reads the existing NLO samples
python3 nlo_breakdown.py            # -> ../data/numbers_qq_breakdown.txt

# sections 3, 4 -- 800k + 800k + 200k LO events, 804 s on 16 cores
python3 run_lo_diagrams.py --nevents 800000 --nevents-zz 200000
python3 harvest_lo_diagrams.py      # -> ../data/numbers_lo_diagrams.txt

# section 3, at NLO -- 8 x 50k MC@NLO events, 345 s on 16 cores
python3 run_nlo_zzonly.py --nblocks 8 --per-block 50000
python3 harvest_nlo_zzonly.py       # -> ../data/numbers_nlo_zzonly.txt
```

Seeds: LO production 24680, LO MadSpin 13579, NLO blocks 7321-7328.  The two
existing 200 000-event `q qbar` samples they are compared against are the study's
own (production 4321 / 8765, MadSpin 7777 / 1357, truth blocks 4321-4324 /
5321-5324).

Events, cards and logs are durable under
`~/Documents/madspin_validation_samples/t131_qq_diagrams/`.  No MadSpin source
file and no `zz_nlo` file is modified; `run_lo_diagrams.py` and
`run_nlo_zzonly.py` import `../../zz_nlo/run_zz_nlo.py` and reuse its run
cards, its cut file and its helpers unchanged.
