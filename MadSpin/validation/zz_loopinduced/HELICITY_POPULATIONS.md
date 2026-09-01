# The helicity populations, read off the LHE

A check of the paper sentence that explains the opposite signs of `C_kk`
between the two `ZZ` production mechanisms. Everything here comes from
`helicity_populations.py`, which writes
`data/numbers_helicity_populations.txt` and `.json`.

The sentence under test:

> A second noteworthy feature is the markedly different behaviour of the
> loop-induced and NLO production mechanisms, including the opposite sign of
> the spin-correlation coefficient. In the `q qbar` channel, chirality
> conservation along the open quark line favours the production of transverse
> `Z`-boson pairs with **opposite helicities**, leading to a negative value of
> `C_kk`. By contrast, in the loop-induced `gg` channel, the closed quark loop
> relaxes this constraint, while the dominant `J_z = 0` gluon configuration
> favours **equal-helicity** transverse states, resulting in a positive value
> of `C_kk`.

## 1. Why re-measuring `C_kk` would have tested nothing

`spin_definitions.tex` Eq. (ckk) *defines*

```
C_kk = <S_k(1) S_k(2)> = P(++) + P(--) - P(+-) - P(-+)
```

with each `S_k` on its own `Z`'s helicity axis. "Opposite helicities give a
negative `C_kk`" is therefore an identity, not a result: extracting `C_kk` from
the decay angles a second time re-confirms a sign that the definition already
guarantees. The physical content of the sentence is the claim about the
**populations** `P(lambda_1, lambda_2)` themselves, and about the two
*mechanisms* offered for them -- chirality conservation on one side, the
dominance of `J_z = 0` gluons on the other. Those are what is measured below,
and all three are measured **independently of any decay angle**.

## 2. `SPINUP` is real, at LO and loop-induced; it is `9` at NLO

Column 13 of an LHE particle line is `SPINUP`. It is populated with genuine
`-1 / 0 / +1` on both undecayed production samples used here:

| sample | process | `SPINUP` on the two `z` |
|---|---|---|
| `t131 .../lo_zz/Events/prod/unweighted_events.lhe.gz` | `p p > z z`, LO | `-1, 0, +1` |
| `t118 .../ggzz/Events/prod/unweighted_events.lhe.gz`  | `g g > z z`, loop induced | `-1, 0, +1` |
| `t130 .../ms_madspin/events.lhe.gz` | `p p > z z [QCD]`, MC@NLO | `9` on every line |

So the answer to "did MadEvent write real helicities" is **yes for the two
samples that carry the comparison, and no for the MC@NLO one**. The aMC@NLO
writer emits `0.9000E+01` -- unknown -- for every particle in every event, as
one would expect from a calculation whose events are not tied to one helicity
configuration.

That is not a limitation here, because the two mechanisms being contrasted are
*both* available in a form that carries helicities: the loop-induced `gg`
sample is its own final word, and on the `q qbar` side the LO sample is a
faithful stand-in -- its `C_kk` of `-0.651 +- 0.001` sits `0.08 sigma` from the
NLO truth's `-0.645 +- 0.080` (`data/numbers_qq.txt`). The one number where LO
and NLO `q qbar` genuinely differ is `f_TT` (0.710 against 0.685, `3.1 sigma`),
which is the `f_0` deficit of `QQ_F0_DEFICIT.md` and not a helicity-sign
effect.

### Why the numbers can be believed

`matrix1.f` picks the helicity written into `SPINUP` by importance sampling,

```fortran
SUMHEL = SUMHEL + DABS(TS(I)) / ANS
IF (RHEL .LT. SUMHEL) THEN
   HEL_SELECTED = I
```

i.e. with probability `|M_i|^2 / sum_j |M_j|^2`. On an **unweighted** file the
frequency of a helicity configuration is therefore its physical fraction.
`TS` is a squared amplitude on both samples -- tree level on the `q qbar` side,
`|M_loop|^2` on the `gg` one -- so the `DABS` is inert and the sampling is a
genuine probability. Both files carry a single constant weight and no negative
weights, so the bars below are ordinary binomial ones.

The **frame** comes for free. MadEvent evaluates the matrix element in the
partonic centre-of-mass frame and boosts to the lab only afterwards -- in
`unwgt.f` the `Boost momentum to lab frame` block runs *after*
`get_helicities`. Both samples are `2 -> 2` at the hard level, so the partonic
CM frame **is** the `ZZ` rest frame, and `SPINUP` is the helicity along each
`Z`'s own direction in it. That is exactly the axis of Eq. (ckk).

### The closure that proves both claims

`t118 ms_madspin` is MadSpin run on the very `ggzz` file read here, and `t131
ms_madspin` is MadSpin run on the very `lo_zz` file read here. So the
`SPINUP` numbers and the published decay-angle numbers are two measurements of
the same events by completely disjoint routes -- one counts helicity labels,
the other fits `eta_l`-diluted angular moments through a factor `4/eta_l^2 =
83.15`:

| | | `SPINUP` | decay angles | |
|---|---|---|---|---|
| `q qbar` | `f_0`  | `+0.17446 +- 0.00066` | `+0.1725 +- 0.0024` | `+0.8 sigma` |
| | `f_TT` | `+0.70999 +- 0.00101` | `+0.7120 +- 0.0066` | `-0.3 sigma` |
| | `C_kk` | `-0.65100 +- 0.00120` | `-0.7214 +- 0.0682` | `+1.0 sigma` |
| `gg` | `f_0`  | `+0.07121 +- 0.00053` | `+0.0673 +- 0.0025` | `+1.5 sigma` |
| | `f_TT` | `+0.90776 +- 0.00065` | `+0.9166 +- 0.0071` | `-1.2 sigma` |
| | `C_kk` | `+0.52271 +- 0.00178` | `+0.4599 +- 0.0723` | `+0.9 sigma` |

Six agreements, none past `1.5 sigma`. Had the helicity sampling not been a
probability, or had `SPINUP` referred to some other axis, this table would not
close. It also happens to be a rather sharp independent validation of MadSpin
itself: the decayed sample reproduces the production-level helicity populations
of its own parent file to well within the errors.

## 3. The populations

`P(lambda_1, lambda_2)`, 200 000 events each, fractions of the **total** rate
(the `3 x 3` matrix sums to one, so the longitudinal entries are included and
`P_eq + P_opp + f_0`-terms is the whole of it).

### `q qbar -> ZZ` (LO `p p > z z`)

| | `l2 = +1` | `l2 = 0` | `l2 = -1` |
|---|---|---|---|
| `l1 = +1` | `0.01473 +- 0.00027` | `0.05727 +- 0.00052` | `0.34090 +- 0.00106` |
| `l1 =  0` | `0.05820 +- 0.00052` | `0.05890 +- 0.00053` | `0.05725 +- 0.00052` |
| `l1 = -1` | `0.33960 +- 0.00106` | `0.05840 +- 0.00052` | `0.01476 +- 0.00027` |

```
P(++) + P(--)   EQUAL      0.02949 +- 0.00038
P(+-) + P(-+)   OPPOSITE   0.68050 +- 0.00104
EQUAL - OPPOSITE          -0.65100 +- 0.00111     -586.9 sigma
```

**Opposite helicities dominate, by a factor of 23.** The four transverse
entries are not close: the two opposite-helicity cells hold 34 % of the *total*
rate each, the two equal-helicity cells 1.5 % each.

### `g g -> ZZ` (loop induced, box)

| | `l2 = +1` | `l2 = 0` | `l2 = -1` |
|---|---|---|---|
| `l1 = +1` | `0.35605 +- 0.00107` | `0.01081 +- 0.00023` | `0.09654 +- 0.00066` |
| `l1 =  0` | `0.01048 +- 0.00023` | `0.05017 +- 0.00049` | `0.01054 +- 0.00023` |
| `l1 = -1` | `0.09599 +- 0.00066` | `0.01025 +- 0.00023` | `0.35918 +- 0.00107` |

```
P(++) + P(--)   EQUAL      0.71523 +- 0.00101
P(+-) + P(-+)   OPPOSITE   0.19252 +- 0.00088
EQUAL - OPPOSITE          +0.52271 +- 0.00134     +390.1 sigma
```

**Equal helicities dominate, by a factor of 3.7.** The matrix is the `q qbar`
one with its two diagonals exchanged.

So the population claim of the sentence is **true on both sides**, and it is
true by a margin that no amount of statistics could reverse.

## 4. The initial state: both mechanisms are also directly measurable

`SPINUP` is written for the *incoming* partons too, which makes the two
mechanistic clauses testable rather than merely plausible. With parton 1 the
one travelling along `+z`, `J_z = lambda_1 - lambda_2`.

A units trap worth recording: MadGraph's `NHEL` -- what lands in `SPINUP` --
is `+-1` for a **fermion** and means helicity `+-1/2`, while for a **vector**
it is the helicity itself. Read naively, the `q qbar` `|J_z| = 1` comes out as
a spurious `|J_z| = 2`.

### `q qbar`: chirality conservation, measured

```
J_z = +1   0.49952 +- 0.00112
J_z = -1   0.50048 +- 0.00112
J_z =  0   0.00000 +- 0.00000        exactly zero events out of 200 000
```

Every single event has the quark and the antiquark in **opposite** helicity
states, i.e. aligned spins along the beam and `|J_z| = 1`. The `J_z = 0`
initial state is not merely suppressed, it is absent -- which is chirality
conservation on a massless open quark line, exactly as the sentence says, and
it is a clean `200 000 / 200 000`.

### `gg`: the `J_z = 0` claim, measured, and it is the *cause*

```
J_z =  0   0.75829 +- 0.00096
J_z = +2   0.12111 +- 0.00073
J_z = -2   0.12060 +- 0.00073
```

`J_z = 0` is dominant, at **75.8 %** of the rate. It is available at all here
only because gluons can pair to `J_z = 0`, which massless quarks cannot.

Better still, splitting the `Z` populations *by* the initial `J_z` shows that
this is not a coincidence of two separately true statements but the actual
mechanism:

| gluon configuration | rate | `P_eq` | `P_opp` | `f_TT` | `C_kk` |
|---|---|---|---|---|---|
| `J_z = 0`   | 0.7583 | `0.88090 +- 0.00083` | `0.04647 +- 0.00054` | 0.927 | `+0.83443 +- 0.00123` |
| `|J_z| = 2` | 0.2417 | `0.19550 +- 0.00180` | `0.65072 +- 0.00217` | 0.846 | `-0.45521 +- 0.00364` |

The `J_z = 0` subsample is *overwhelmingly* equal-helicity (`C_kk = +0.83`);
the `|J_z| = 2` subsample flips over and looks like `q qbar` (`C_kk = -0.46`,
against the `q qbar` `-0.65`). The positive `gg` `C_kk` is therefore the
`J_z = 0` component winning a competition against a `|J_z| = 2` component that
pulls the other way -- which is precisely the causal chain the sentence
asserts, now with the two pieces separated and each measured.

## 5. Verdict, clause by clause

| clause | verdict |
|---|---|
| "opposite sign of the spin-correlation coefficient" between the two mechanisms | **Confirmed.** `-0.651 +- 0.001` against `+0.523 +- 0.002` from `SPINUP` alone; `-0.645 +- 0.080` against `+0.380 +- 0.072` from the decay angles. |
| `q qbar`: "chirality conservation along the open quark line" | **Confirmed, directly.** 200 000 / 200 000 events have `lambda_q = -lambda_qbar`, `J_z = 0` strictly empty. |
| `q qbar`: "favours the production of **transverse** `Z`-boson pairs" | **Refuted as a distinguishing statement.** `f_TT` is 0.710 (LO) / 0.685 (NLO) for `q qbar` against **0.908** for `gg`. The loop-induced channel is the *more* transverse of the two. See below. |
| `q qbar`: "with opposite helicities" | **Confirmed.** `P_opp = 0.681 +- 0.001` against `P_eq = 0.029 +- 0.000`, a factor 23. |
| `q qbar`: "leading to a negative value of `C_kk`" | **True but tautological** -- it is Eq. (ckk) read aloud. Harmless as exposition; it is not evidence. |
| `gg`: "the closed quark loop relaxes this constraint" | **Confirmed with qualification.** Something is certainly relaxed -- `P_eq` goes from 0.029 to 0.715 -- but the operative relaxation measured here is in the *initial* state (gluons reach `J_z = 0`, quarks cannot), not in the loop's own chirality. The clause is loose rather than wrong. |
| `gg`: "the dominant `J_z = 0` gluon configuration" | **Confirmed, directly, and it was testable after all.** 75.8 % of the rate. |
| `gg`: "favours equal-helicity transverse states, resulting in a positive `C_kk`" | **Confirmed, and causally.** Conditional on `J_z = 0`: `P_eq = 0.881`, `C_kk = +0.834`. Conditional on `|J_z| = 2` it flips to `C_kk = -0.455`. |

Two things the brief flagged, resolved:

* **The `f_TT` doubt was right.** Transversity does not distinguish the two
  mechanisms in the direction the sentence implies; it distinguishes them in
  the *opposite* direction. What distinguishes them is the *relative* helicity
  within the transverse sector. The sentence needs rewording even though its
  conclusion is correct.
* **The `J_z = 0` doubt was wrong, and pleasantly so.** The clause is not only
  testable, it is the best-supported clause in the sentence: the gluon
  helicities are in the LHE, `J_z = 0` is measured to dominate, and
  conditioning on it reproduces the equal-helicity preference while
  conditioning against it reverses the sign of `C_kk`. This need not be
  attributed to the citation; this work measures it.

## 6. A convention caveat, for the citation and not for the physics

`C_kk` here is `<S_k(1) S_k(2)>` with each `S_k` on **its own** `Z`'s helicity
axis (`spin_definitions.tex`, Eq. (ckk) and the line following it). The
quantum-information and `t tbar` literature more often uses a **common** `k`
axis for both particles -- `k` the direction of particle 1 in the pair rest
frame, with particle 2's spin projected on the *same* `k` rather than on its
own `-k`. Because that flips the sign of the second projection, the two
conventions give `C_kk` of **opposite sign** for the same events.

Nothing above depends on this: the populations are convention-free, and the
draft sentence is internally consistent with the convention this work declares.
But the sentence asserts signs and cites `Javurkova:2024bwa`, so it is worth
confirming which convention that reference uses before the signs are compared
across the citation. If it uses the common-axis one, either the sign statement
or an explicit remark about the axis convention will need to be added.

## 7. A draft rewording

Only the `q qbar` transversity clause has to move, and the `gg` side can be
strengthened now that `J_z` is measured rather than asserted:

> A second noteworthy feature is the markedly different behaviour of the
> loop-induced and NLO production mechanisms, including the opposite sign of
> the spin-correlation coefficient. Both mechanisms produce predominantly
> transverse `Z` pairs -- indeed the loop-induced channel more so, `f_TT =
> 0.91` against `0.69` -- so it is not transversity but the *relative*
> helicity within the transverse sector that separates them. In the `q qbar`
> channel, chirality conservation on the open quark line forces the initial
> state into `|J_z| = 1` and favours transverse `Z` pairs of **opposite**
> helicity, `P(+-) + P(-+) = 0.68` against `P(++) + P(--) = 0.03`, hence a
> negative `C_kk`. In the loop-induced `gg` channel the closed quark loop
> imposes no such constraint on the external helicities, and the initial state
> can reach `J_z = 0`, which it does in 76 % of the rate; that configuration
> strongly favours **equal**-helicity transverse states (`P(++) + P(--) =
> 0.88` within it), and it is what makes `C_kk` positive overall
> [`Javurkova:2024bwa`].

If a shorter version is wanted, the minimum repair is to delete "favours the
production of transverse `Z`-boson pairs with" and keep "favours `Z`-boson
pairs with opposite helicities", which removes the false contrast without
adding anything.

## Re-running

```
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
python3 helicity_populations.py     # ~4 s, reads the two LHE files, writes data/
```

No sample was generated for this check. It reuses, unmodified:

* `~/Documents/madspin_validation_samples/t118_zz_loopinduced/ggzz/Events/prod/unweighted_events.lhe.gz` (200 k, loop-induced `g g > z z`)
* `~/Documents/madspin_validation_samples/t131_qq_diagrams/work/lo_zz/Events/prod/unweighted_events.lhe.gz` (200 k, LO `p p > z z`)
* `~/Documents/madspin_validation_samples/t130_qq_seed_check/work/ms_madspin/events.lhe.gz`, read only far enough to establish that MC@NLO writes `SPINUP = 9`.
