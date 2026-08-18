# Results

Base: `bdf383554`, the tip of `claude/ms-pure-interference` (PR #351), which
carries the polarisation restriction of #349 and the pure-interference mode.
Nothing else merged in.

13 TeV, LO, `p p > t t~`, NNPDF23LO (`nn23lo1`), `me_frame = [1,2]` (the
partonic CM, run-card default).  MadSpin `spinmode = onshell`, `BW_cut = 15`,
`max_weight_ps_point = 400`, `decay t > b w+, w+ > l+ vl` and the conjugate,
`l = e, mu`.  **50 000 production events per sample, eleven samples, a
different MadEvent seed for each** so that all eleven are statistically
independent and errors add in quadrature.

The interference samples run with the joint accept/reject (the mode forces it);
the four fully polarised diagonal samples keep `onshell`'s own `sequential`
default, which is what PR #360 now also selects under `auto`.

`tests/test_manager.py test_madspin -t0`: **269 tests, OK.**

One line was added to the branch for this test: a `logger.info` reporting the
joint maximum weight in `get_maxwgt_for_onshell` (section 4).  Pure logging.

## 1. The decomposition, and the counting checked against the code

The restriction is a **product of per-index conditions**
(`DensityMatrix._restriction_row_mask`).  On a two-state basis a per-particle
condition closed under `(bra,ket) -> (ket,bra)` has exactly three forms --
`D+ = {+1}`, `D- = {-1}` (symmetric, one entry each) and
`I = ({+1},{-1})` (cross, two entries).  Three per particle, `3 x 3 = 9` in the
product: 4 diagonal-diagonal, 4 with one interference factor, 1 with two --
`4*1 + 4*2 + 1*4 = 16`, the whole joint `4 x 4` matrix, each entry once.

`check_blocks.py` verifies exactly that **against the code**, not against the
prose: two random hermitian 2x2 matrices, `tensor_product`, the nine
restrictions through `set_hel_restriction` / `_restriction_row_mask` /
`scalar_multiplication`.  Output:

* the nine masks are pairwise disjoint and cover all 16 entries (asserted);
* each block's contraction is real (imaginary part `0.000000` on all nine);
* the nine sum to the unrestricted contraction, relative difference **3.5e-8**;
* the two coarse masks the test actually uses reproduce the sum of their three
  blocks (relative difference 0 and 1.8e-8).

So the counting in the task statement is right, and it is right *because* the
mask stays a per-index product -- which is the design decision recorded in
section 13.9 of `MADSPIN_SEQUENTIAL_PLAN.md`.

## 2. A bug: the multi-particle syntax is unreachable from a card

`(I,I)` needs two `pure_interference` entries in one card,

    set pure_interference t = + - ; t~ = + -

and this **cannot be written**.  `extended_cmd.Cmd.precmd` splits every card
line on `;` and runs the pieces as separate commands, so only `t = + -` reaches
the option and ` t~ = + -` is swallowed **silently** -- no error, and the run
proceeds as a single-particle interference sample.  `_pure_interference` splits
its option string on `;` and is plainly written for several particles, so the
intent is there; the separator just collides with the command splitter.

Consequences, in order of severity:

1. a user asking for a two-particle interference gets a **different, valid-
   looking sample** with no warning.  `_validate_pure_interference`'s own
   warning prints the pdgs it ended up with (`... is ON for particle(s) 6`),
   which is the only trace;
2. the `(I,I)` block is not directly reachable.

This does not invalidate anything below -- the block is obtained by
subtraction, see section 3 -- but it should be fixed before #351 is used.  A
one-character fix (accept `,` as well as `;` in `_pure_interference`, or
accumulate repeated `set pure_interference` lines instead of overwriting) would
do it.  I did **not** patch it here: this is a validation branch.

## 3. Which sample carries which block

| block | production | MadSpin card | tag |
|---|---|---|---|
| `(D+,D+)` | `p p > t{+} t~{+}` | -- | `pp` |
| `(D+,D-)` | `p p > t{+} t~{-}` | -- | `pm` |
| `(D-,D+)` | `p p > t{-} t~{+}` | -- | `mp` |
| `(D-,D-)` | `p p > t{-} t~{-}` | -- | `mm` |
| `(I,D+)`  | `p p > t   t~{+}` | `pure_interference t = + -` | `i_tbp` |
| `(I,D-)`  | `p p > t   t~{-}` | `pure_interference t = + -` | `i_tbm` |
| `(D+,I)`  | `p p > t{+} t~`   | `pure_interference t~ = + -` | `i_tp` |
| `(D-,I)`  | `p p > t{-} t~`   | `pure_interference t~ = + -` | `i_tm` |
| `(I,D+)+(I,D-)+(I,I)` | `p p > t t~` | `pure_interference t = + -` | `x_t` |
| `(D+,I)+(D-,I)+(I,I)` | `p p > t t~` | `pure_interference t~ = + -` | `x_tb` |

Blocks 5-8 work because `_validate_pure_interference` only refuses a brace on
the leg the *cross* restriction is asked for -- a brace on the **other** leg is
allowed, and is what selects `D+`/`D-` there -- and because
`hel_restriction_trace` keeps the normalising trace symmetric, so it equals the
braced parent's own cross-section and cancels.  That cancellation is why nine
blocks coming from six different production samples add up with no relative
factor between them.  The half-braced parents come out at
252.28 / 252.29 / 252.11 / 252.51 pb, i.e. `sigma(++) + sigma(-+) = 168.5 +
83.65 = 252.2` pb, as they must.

The nine-term total is assembled two independent ways:

    route 1 :  4 diagonal  +  x_t   +  (D+,I) + (D-,I)
    route 2 :  4 diagonal  +  x_tb  +  (I,D+) + (I,D-)

## 4. Normalisation: one number, derived and then measured

An ordinary MadSpin sample redraws until acceptance, so every production event
writes exactly one event of weight `sigma/N`.  The pure-interference mode
cannot redraw; it draws once, keeps with probability `|W|/max_weight` and writes
`+- sigma_ref`.  The two schemes therefore differ by the single factor
`max_weight / c`, with `c = <W>` over the decay phase space -- the constant
whose event-independence is what makes redraw-until-accept unbiased in the first
place (section 13.7b of the plan).

* `max_weight` was not recoverable from the output: the sequential scheme logs
  its per-slot bounds, the joint one logged nothing, and the `ms_dir` route that
  writes `max_wgt` to disk also switches the decay pool to the gridpack path.
  **One `logger.info` line was added** to `get_maxwgt_for_onshell`.  It is a
  genuine gap, not just a test convenience.
* `c` is **measured**: the unpolarised reference is run with
  `set unweighting joint`, where the unweighting efficiency is `c/max_weight`.

      eff = 0.323725   max_weight = 6.937097e-10   ->  c = 2.245713e-10 +- 0.45%

  and `c` is a decay-side constant (the production density matrix cancels
  between the restricted contraction and its normalising trace), so the same `c`
  applies to all eleven samples.  That is the load-bearing assumption for blocks
  5-8, whose parents are braced, so it was measured again rather than assumed
  (`run_c_check.sh`, ordinary joint mode, 10k events):

  | sample | `max_weight` | eff | `c = eff * max_weight` | / unpolarised |
  |---|---|---|---|---|
  | `p p > t t~` (50k) | 6.9371e-10 | 0.32373 | 2.2457e-10 | 1 |
  | `p p > t t~{+}` | 1.2182e-09 | 0.18380 | 2.2394e-10 | 0.9972 |
  | `p p > t{+} t~` | 1.2821e-09 | 0.17690 | 2.2680e-10 | 1.0099 |
  | `p p > t t~` (5k pilot) | 6.5171e-10 | 0.34350 | 2.2387e-10 | 0.9969 |

  Three different production processes, maximum weights differing by a factor
  two, the same `c` to **1%** -- which is also the precision of each entry
  (`1/sqrt(N_written)`).

The resulting factor is a **prediction**, never fitted.  How well it holds is
measured in section 6.

## 5. Total rate -- unchanged, as it must be

The interference blocks have no diagonal entry, so their restricted trace
vanishes identically and they carry no cross-section.  All five write
`XSECUP = 0` into `<init>` (checked: `0.00000` for every one), and their summed
event weights are

    the 5 interference blocks : +0.02029 +- 0.05552 pb      (0.37 sigma from 0)

against 23.76 pb of total rate -- i.e. **zero to 2.3e-3 of the rate**.  Each
sample's own `z = S / sqrt(sum w^2)` check, over its own events:

| block | kept / read | z | trials above `max_weight` |
|---|---|---|---|
| `(I,D+)` | 1728 / 50000 (3.46%) | -1.35 | 0 |
| `(I,D-)` | 1514 / 50000 (3.03%) | +1.90 | 0 |
| `(D+,I)` | 1702 / 50000 (3.40%) | -0.10 | 0 |
| `(D-,I)` | 1623 / 50000 (3.25%) | -0.02 | 0 |
| `x_t`    | 4084 / 50000 (8.17%) | +0.44 | 0 |
| `x_tb`   | 4291 / 50000 (8.58%) | -0.84 | 0 |

All below 2 sigma, no overweight event anywhere, so the one mechanism that could
have biased `S` was never active.

The diagonal total is therefore also the nine-block total:

|  | sum of the 4 diagonal | unpolarised | ratio |
|---|---|---|---|
| production | 504.5122 +- 0.1563 pb | 504.7490 +- 0.2851 pb | 0.999531 +- 0.000644 (-0.73 sigma) |
| after MadSpin | 23.75506 +- 0.00736 pb | 23.76365 +- 0.01342 pb | 0.999639 +- 0.000645 (-0.56 sigma) |

## 6. The closure -- the deliverable

Per-bin `chi2` of the sum against the unpolarised sample, 20 bins, statistical
errors only, both samples' errors included.  `k` is the best-fit scale of the
interference contribution; **`k = 1` is the prediction of section 4** and is
quoted, not used.

| observable | `chi2` 4 diagonal | `chi2` 9 blocks (r1) | `chi2` 9 blocks (r2) | `k` | previous test |
|---|---|---|---|---|---|
| `cos theta_k(l+)` | 18.3 | **21.8** | 20.5 | -0.29 +- 0.48 | 24.6 |
| `cos theta_k(l-)` | 15.7 | **8.4** | 22.5 | 1.26 +- 0.59 | 12.1 |
| `C_kk` | 17.0 | **20.1** | 18.0 | -0.90 +- 0.74 | 8.4 |
| **`C_nn`** | **531.7** | **25.4** | **27.0** | **1.035 +- 0.053** | 398.4 |
| `C_rr` | 21.3 | **21.2** | 21.2 | 0.02 +- 0.44 | 11.4 |
| **`cos phi_ll`** | **239.8** | **11.6** | **15.0** | **1.126 +- 0.083** | 159.4 |
| `cos theta_n(l+)` | 17.8 | **16.5** | 16.7 | 0.25 +- 0.53 | 23.0 |
| **`Delta phi(l+,l-)`** | **317.4** | **14.0** | **12.5** | **1.067 +- 0.068** | 245.0 |
| `pT(t)` (control) | 15.1 | **11.4** | 10.0 | 0.70 +- 0.67 | 13.8 |
| `m(t t~)` (control) | 12.6 | **15.4** | 12.1 | -0.66 +- 0.71 | 17.6 |

Per-bin statistical precision: 2.0-3.2% on the unpolarised sample, 1.5-2.8% on
the nine-block sum.

**The three observables that failed now close.**  `C_nn` goes from
`chi2 = 531.7` to `25.4`, `cos phi_ll` from `239.8` to `11.6`, `Delta phi` from
`317.4` to `14.0` -- all on 20 bins, i.e. `chi2/ndf` of 1.27, 0.58, 0.70.  The
maximum per-bin deviation on `C_nn` drops from 0.65 to 0.19 (that 0.19 is the
`cos = -1` edge bin, which has a 30% error; `cos phi_ll` and `Delta phi` drop
from 0.144 / 0.159 to 0.038 / 0.038).  Route 2, which shares only the four
diagonal samples with route 1, gives `27.0 / 15.0 / 12.5` -- the same answer from
statistically independent interference samples.

**Everything that already closed stays closed.**  The seven other entries all sit
between 8.4 and 21.8 on 20 bins, and the two controls at 11.4 and 15.4: adding
the interference does not disturb the diagonal observables or the production
kinematics.

Means, and the exact shift the interference produces
(`<O>_9 - <O>_4 = (sum_int w O)/(sum_4 w)`, exact because `sum_int w = 0`):

| observable | 4 diagonal | interference | 9 blocks | unpolarised | pull(4) | pull(9) |
|---|---|---|---|---|---|---|
| `<C_kk>` | 0.03584 +- 0.00077 | -0.00034 +- 0.00062 | 0.03547 +- 0.00100 | 0.03919 +- 0.00148 | -2.01 | -2.09 |
| **`<C_nn>`** | 0.00018 +- 0.00079 | **+0.03626 +- 0.00090** | 0.03641 +- 0.00120 | 0.03793 +- 0.00149 | **-22.40** | **-0.79** |
| `<C_rr>` | -0.00045 +- 0.00079 | +0.00247 +- 0.00091 | 0.00202 +- 0.00120 | 0.00190 +- 0.00149 | -1.39 | +0.06 |
| **`<cos phi_ll>`** | 0.03557 +- 0.00136 | +0.03839 +- 0.00145 | 0.07390 +- 0.00199 | 0.07902 +- 0.00256 | **-14.98** | -1.58 |
| **`<Delta phi>`** | 1.82465 +- 0.00213 | -0.07051 +- 0.00491 | 1.75264 +- 0.00304 | 1.74577 +- 0.00405 | **+17.24** | +1.36 |
| `<pT(t)>` | 120.212 +- 0.174 | +0.267 +- 0.351 | 120.376 +- 0.275 | 119.864 +- 0.348 | +0.89 | +1.15 |
| `<m(t t~)>` | 526.115 +- 0.400 | +0.602 +- 1.218 | 526.268 +- 0.554 | 524.996 +- 0.787 | +1.27 | +1.32 |

The 20-sigma `C_nn` deficit of the previous test (here -22.4 sigma) is closed to
-0.8 sigma, and the interference supplies `+0.03626 +- 0.00090` against a deficit
of `0.03775 +- 0.00168`.

Plots: `plots/closure_summary.pdf` (the three observables side by side, blue =
4 diagonal blocks, red = all 9), and `plots/closure_<observable>.pdf` for each
of the ten, upper panel with the components and the total, ratio panel against
the unpolarised sample carrying both sums.

### 6a. A null test the interference has to pass

The interference integrates to zero over the decay phase space **at every
production point**, so its contribution to a purely production-level observable
must vanish bin by bin, not just in total.  `chi2` of the interference-only
histogram against zero, 20 bins:

| observable | `chi2` vs 0 |
|---|---|
| `pT(t)` | 12.1 |
| `m(t t~)` | 13.5 |
| `cos theta_k(l+)` | 23.0 |
| `C_kk` | 14.9 |
| `C_rr` | 24.4 |
| `cos theta_n(l+)` | 20.1 |
| **`C_nn`** | **1686.2** |
| **`cos phi_ll`** | **738.5** |
| **`Delta phi`** | **1192.7** |

The production-level observables are flat at zero; the signal sits exactly where
it should.

### 6b. Where the effect lives: the `(I,I)` block

`plots/blocks_cnn.pdf` shows the nine blocks separately.  The four blocks with a
*single* interference factor -- `(I,D+)`, `(I,D-)`, `(D+,I)`, `(D-,I)` -- are
flat at zero in `C_nn`; the whole effect is the doubly-interfering `(I,I)`
block, which is exactly the statement
`<sigma_n sigma_n> = 2 Re[rho(+-,-+)] - 2 Re[rho(++,--)]`: both entries have
`i != j` on **both** legs, and both sit in `(I,I)` and nowhere else.

`(I,I)` is available twice, from two disjoint sets of samples:

    (I,I) = x_t  - (I,D+) - (I,D-)        and        (I,I) = x_tb - (D+,I) - (D-,I)

They agree bin by bin: `chi2/20` of **18.3** on `C_nn`, 15.7 on `C_kk`, 18.9 on
`cos phi_ll`, 26.4 on `Delta phi` (bottom panel of the `blocks_*` figures).

## 7. Verdict

**The closure passes.  Diagonal + interference reproduces the unpolarised
sample.**  Specifically:

* the 9-block decomposition is 3 forms per particle, 9 disjoint blocks tiling
  the 16-entry joint density matrix -- verified against the code, not assumed;
* the total rate is unchanged: the five interference blocks sum to
  `+0.020 +- 0.056 pb` against 23.76 pb, every `z` below 2, no overweight event;
* `C_nn` `chi2` 531.7 -> **25.4**, `cos phi_ll` 239.8 -> **11.6**,
  `Delta phi` 317.4 -> **14.0**, on 20 bins, and `<C_nn>` from `-22.4 sigma` to
  `-0.8 sigma`;
* the seven observables that already closed stay closed (8.4 to 21.8 on 20
  bins), including both production-level controls;
* the normalisation is a **prediction**: `max_weight / c`, with `c` measured on
  three different production processes and agreeing to 1%.  The best-fit scale
  of the interference is `1.035 +- 0.053` (`C_nn`), `1.067 +- 0.068`
  (`Delta phi`), `1.126 +- 0.083` (`cos phi_ll`) -- consistent with 1, with a
  mild (1.5 sigma on the most constraining pair, and the three are strongly
  correlated) preference for a value above it;
* the effect is carried entirely by the `(I,I)` block, obtained two independent
  ways that agree bin by bin.

Nothing found here blocks the physics of #351.

**What does need fixing before it ships** (section 2): the `;` separator makes
the multi-particle `pure_interference` syntax unreachable from a card, and fails
*silently* -- a card asking for two particles produces a single-particle sample
with no error.  That is a correctness-of-user-intent bug, not a physics bug.

Second, smaller: `max_weight` (the joint accept/reject bound) is not reported
anywhere, and in this mode it is not an internal detail -- it is part of the
sample's normalisation, since the mode deliberately does not unweight it away.
The `<MGPureInterference>` banner block is the natural place for it.  This test
added a `logger.info`; the banner would be better.

### What was not tested

* only `spinmode = onshell` here (the plan's end-to-end validation used
  `madspin`, so between the two the mode has been run in both an on-shell and an
  off-shell density mode; `PA` and `full` were not run);
* LO, one process, no polarised beams, no vector `{0}`/`{T}` interference in a
  closure (that is the plan's `w+ w-` test), no `fixed_order`;
* the per-bin statistical precision is 1.5-3.2%, so an interference effect below
  ~1.5% of a bin would not be visible in the observables that close;
* `k` is quoted from a single-parameter fit per observable and the three
  interference-sensitive observables are strongly correlated, so "consistent
  with 1" is one measurement, not three;
* the interference samples keep 3-9% of the production events, so a 50 000-event
  parent gives 1500-4300 written events per block.  Statistics on the individual
  `(I,D+-)` blocks are correspondingly thinner than on the diagonal ones.
