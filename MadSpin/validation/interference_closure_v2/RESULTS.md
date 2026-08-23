# Results -- interference closure, second run

Base: `d19d8a293`, the tip of `claude/ms-pure-interference` (PR #351), which
carries the polarisation restriction of #349 and the **reworked** pure-
interference mode (card-nameable diagonal blocks, accumulating `set` lines,
fully weighted output).  Nothing else merged in, and **no line of the branch was
changed for this test** -- unlike the first run, which had to add a `logger.info`
to recover `max_weight`.

13 TeV, LO, `p p > t t~`, NNPDF23LO (`nn23lo1`), `me_frame = [1,2]` (the
partonic CM, run-card default).  MadSpin `spinmode = onshell`, `BW_cut = 15`,
`max_weight_ps_point = 400`, `decay t > b w+, w+ > l+ vl` and the conjugate,
`l = e, mu`.  **Ten samples: 50 000 production events for the unpolarised
reference and for each of the four diagonal blocks, 20 000 for each of the five
interference blocks**, a different MadEvent seed for each so that all ten are
statistically independent and errors add in quadrature.  350 000 production
events, against the first run's 550 000 plus 35 000 more for its `c` measurement
-- see section 7.

`tests/test_manager.py test_madspin -t0`: **291 tests, OK** (re-measured on this
base).

## 1. What a card can name -- settled before anything was generated

`check_blocks.py` part B drives a real `MadSpinInterface` through the same
`precmd` a card line goes through, for all nine blocks of the joint `(h_t,
h_tbar)` density matrix, and checks not only whether the card is accepted but
whether the restriction it produces **is the block meant** (the spec is compared
to the one part A built from `set_hel_restriction` / `_restriction_row_mask`).

| block | card | result |
|---|---|---|
| `(D+,D+)` `(D+,D-)` `(D-,D+)` `(D-,D-)` | `t = + +` / `t~ = - -`, etc. | **refused** by `_validate_pure_interference`: *"every pure_interference entry names a DIAGONAL block, so nothing interferes"* |
| `(I,D+)` | `t = + -` / `t~ = + +` | named, spec `(((1,),(-1,)), (1,))` |
| `(I,D-)` | `t = + -` / `t~ = - -` | named, spec `(((1,),(-1,)), (-1,))` |
| `(D+,I)` | `t = + +` / `t~ = + -` | named, spec `((1,), ((1,),(-1,)))` |
| `(D-,I)` | `t = - -` / `t~ = + -` | named, spec `((-1,), ((1,),(-1,)))` |
| `(I,I)` | `t = + -` / `t~ = + -` | named, spec `(((1,),(-1,)), ((1,),(-1,)))` |

**The expectation in the task statement is exactly what the code does**:
the five blocks carrying at least one `I` index come from `set
pure_interference` lines; the four diagonal-diagonal blocks are not
interference at all and still come from #349's production braces, because the
mode requires at least one particle to carry a disjoint pair.

**`(I,I)` is now named directly, and that is the main cleanliness win.**  The
first run could only reach it as `x_t - (I,D+) - (I,D-)`; that subtraction is
what forced eleven samples, two half-braced production processes and two
independent assembly routes.  All of that is gone: the five interference
samples here run on one and the same unpolarised `p p > t t~`, and the nine
blocks are added, never subtracted.

Part B also confirms the three refusals the rework promises: the `;` spelling
raises instead of truncating silently, a partial overlap (`{+,-}` against
`{-}`) raises, and two `set pure_interference` lines accumulate to
`6 = + - ; -6 = + -`.

Part A is unchanged from the first run and still passes: nine pairwise disjoint
masks covering all 16 entries once, each block's contraction real, the nine
summing to the unrestricted contraction to a relative 3.5e-8.

## 2. Normalisation: nothing to reconstruct

The output is fully weighted -- `w = sigma_ref * BR * W / c`, every trial kept
-- so under MG5's `IDWTUP = -4` convention **one rule serves every sample in
this test**:

    contribution of a bin, in pb  =  sum_(events in bin) w / N_file

`analyse_interference.py:check_normalisation` asserts it per sample rather than
assuming it, and all ten pass:

* the five ordinary samples have a constant `XWGTUP` across the file equal to
  their `<init>` `XSECUP` (relative difference < 1e-5);
* the five interference samples carry `XSECUP = 0`, hold **exactly `N_read`
  events** (20000/20000 each -- fully weighted, nothing dropped, `keep = 1.0000`
  in the log), and their file weight sum reproduces the `S` recorded in the
  `<MGPureInterference>` banner block.

The old `max_weight / c` machinery -- `analyse_interference.py:329`, the added
`logger.info`, the `run_c_check.sh` runs on braced processes -- is **entirely
gone from the analysis**.  `c` is now reported per run in the banner, which
turns what used to be a load-bearing assumption into a free consistency check:

| sample | `c` (banner) | `z` | dead trials |
|---|---|---|---|
| `(I,D+)` | 2.254096e-10 | +0.07 | 0 |
| `(I,D-)` | 2.252648e-10 | -0.40 | 0 |
| `(D+,I)` | 2.258415e-10 | +0.11 | 0 |
| `(D-,I)` | 2.258074e-10 | +0.22 | 0 |
| `(I,I)`  | 2.248850e-10 | -0.77 | 0 |

Five independent measurements spanning **0.42%** -- consistent with `c` being a
decay-side constant, and consistent with the 2.2457e-10 the first run measured
a completely different way (from the unweighting efficiency of a `joint` run).
Each run also prints its analytic candidate `1/(prod_denominators*sym_decay)`,
which agrees with the measured value to 0.1%.

## 3. Total rate -- unchanged, as it must be

The interference blocks have no diagonal entry, so their restricted trace
vanishes identically and they carry no cross-section.  All five write
`XSECUP = 0` into `<init>` (asserted), and **each one integrates to zero on its
own**:

| block | integral [pb] | sigma from 0 |
|---|---|---|
| `(I,D+)` | +0.000602 +- 0.008294 | +0.07 |
| `(I,D-)` | -0.003393 +- 0.008414 | -0.40 |
| `(D+,I)` | +0.000870 +- 0.008251 | +0.11 |
| `(D-,I)` | +0.001841 +- 0.008354 | +0.22 |
| `(I,I)`  | -0.023723 +- 0.030634 | -0.77 |
| **all 5** | **-0.023803 +- 0.034870** | **-0.68** |

i.e. zero to **1.0e-3 of the 23.77 pb total rate**, with no overweight and no
dead-weight event anywhere (`Trials with a dead weight: 0` on all five).  Note
this is now a *per-block* statement: the first run could only make it for the
coarse `x_t` / `x_tb` combinations plus four thin half-braced samples.

The diagonal total is therefore also the nine-block total:

|  | sum of the 4 diagonal | unpolarised | ratio |
|---|---|---|---|
| production | 504.5122 +- 0.1563 pb | 504.7490 +- 0.2851 pb | 0.999531 +- 0.000644 (-0.73 sigma) |
| after MadSpin | 23.75506 +- 0.00736 pb | 23.76511 +- 0.01342 pb | 0.999577 +- 0.000644 (-0.66 sigma) |

## 4. The closure -- the deliverable

**Notation, once, for the whole file.**  `C_kk`, `C_nn`, `C_rr` name the
**per-event products** `cos theta^i(l+) cos theta^j(l-)` -- one number per
event.  That product is what is histogrammed, what every figure draws and what
every `chi2` below is computed on.  The spin-correlation **coefficient** is the
*mean* of it (up to the standard `1/9` of the two leptonic analysing powers),
i.e. one number per sample; those are the `<...>` rows, and only those.  The
distinction matters because a block whose coefficient vanishes still has a cross
section and a perfectly non-zero histogram -- what vanishes for it is the first
moment.  The figures label the axis `... (mean -> C_ij)` for the same reason.

Per-bin `chi2` of the sum against the unpolarised sample, 20 bins, statistical
errors only, both samples' errors included.  `k` is the best-fit scale of the
interference contribution; **`k = 1` is now a prediction with nothing measured
or fitted standing behind it** -- there is no constant between the samples at
all -- and it is quoted, never used.

| observable | `chi2` 4 diagonal | `chi2` 9 blocks | (first run: 4 -> 9) | `k` |
|---|---|---|---|---|
| `cos theta_k(l+)` | 26.5 | 30.3 | 18.3 -> 21.8 | -1.27 +- 0.77 |
| `cos theta_k(l-)` | 20.5 | 23.7 | 15.7 -> 8.4 | -1.46 +- 0.92 |
| `C_kk` | 26.4 | 30.2 | 17.0 -> 20.1 | -3.32 +- 1.23 |
| **`C_nn`** | **500.3** | **24.8** | **531.7 -> 25.4** | **0.988 +- 0.048** |
| `C_rr` | 11.9 | 13.0 | 21.3 -> 21.2 | 0.24 +- 0.48 |
| **`cos phi_ll`** | **234.3** | **14.4** | **239.8 -> 11.6** | **1.137 +- 0.081** |
| `cos theta_n(l+)` | 21.4 | 21.5 | 17.8 -> 16.5 | -0.09 +- 0.81 |
| **`Delta phi(l+,l-)`** | **339.4** | **16.0** | **317.4 -> 14.0** | **1.064 +- 0.062** |
| `pT(t)` (control) | 15.1 | 13.1 | 15.1 -> 11.4 | 0.80 +- 0.93 |
| `m(t t~)` (control) | 12.6 | 12.7 | 12.6 -> 15.4 | 0.23 +- 0.80 |

**The three observables that failed close, on the same numbers as the first
run.**  `C_nn` goes from `chi2 = 500.3` to `24.8`, `cos phi_ll` from `234.3` to
`14.4`, `Delta phi` from `339.4` to `16.0` -- `chi2/ndf` of 1.24, 0.72, 0.80 on
20 bins, against the first run's 25.4 / 11.6 / 14.0.  Every "before" number
agrees with the first run's within the fluctuation of the reference sample
(the four diagonal samples are literally the same runs -- same seeds, same
cards; only the unpolarised reference differs, because the first run had to put
it in `joint` mode to measure `c` and this one does not).

The four entries that already closed and both controls stay between 11.9 and
30.3 on 20 bins.  The three largest -- `cos theta_k(l+)` 30.3, `C_kk` 30.2,
`cos theta_k(l-)` 23.7 -- are `chi2/ndf` of 1.5, 1.5, 1.2, i.e. p-values of
0.07, 0.07, 0.26; they sit at 26.5 / 26.4 / 20.5 *before* the interference is
added too, so this is the reference sample fluctuating, not the interference
doing damage.  The `k` values in those rows (`-3.32 +- 1.23` on `C_kk`) are
fits of noise against noise: the interference contribution to `C_kk` is
`-0.00032 +- 0.00033`, so there is no signal for `k` to scale.

Means, and the exact shift the interference produces
(`<O>_9 - <O>_4 = (sum_int w O)/(sum_4 w)`, exact because `sum_int w = 0`):

| observable | 4 diagonal | interference | 9 blocks | unpolarised | pull(4) | pull(9) |
|---|---|---|---|---|---|---|
| `<C_kk>` | 0.03584 +- 0.00077 | -0.00032 +- 0.00033 | 0.03555 +- 0.00084 | 0.03815 +- 0.00147 | -1.40 | -1.54 |
| **`<C_nn>`** | 0.00018 +- 0.00079 | **+0.03657 +- 0.00059** | 0.03679 +- 0.00099 | 0.03625 +- 0.00148 | **-21.52** | **+0.30** |
| `<C_rr>` | -0.00045 +- 0.00079 | **+0.00104 +- 0.00066** | 0.00060 +- 0.00103 | 0.00393 +- 0.00150 | -2.59 | -1.84 |
| **`<cos phi_ll>`** | 0.03557 +- 0.00136 | +0.03729 +- 0.00096 | 0.07293 +- 0.00167 | 0.07834 +- 0.00255 | **-14.80** | -1.77 |
| **`<Delta phi>`** | 1.82465 +- 0.00213 | -0.07712 +- 0.00313 | 1.74928 +- 0.00257 | 1.74398 +- 0.00405 | **+17.64** | +1.11 |
| `<pT(t)>` | 120.212 +- 0.174 | -0.270 +- 0.223 | 120.063 +- 0.227 | 119.864 +- 0.348 | +0.89 | +0.48 |
| `<m(t t~)>` | 526.115 +- 0.400 | -0.702 +- 0.735 | 525.940 +- 0.470 | 524.996 +- 0.787 | +1.27 | +1.03 |

Against the first run's targets:

| | first run | this run |
|---|---|---|
| `<C_nn>` interference | +0.03626 +- 0.00090 | **+0.03657 +- 0.00059** |
| `<C_rr>` interference | +0.00247 +- 0.00091 | **+0.00104 +- 0.00066** |
| `<C_nn>` pull, 4 blocks | -22.40 | -21.52 |
| `<C_nn>` pull, 9 blocks | -0.79 | **+0.30** |

`<C_nn>` agrees between the two runs at 0.3 sigma and is measured **1.5x more
precisely from 2.5x fewer production events**.  `<C_rr>` moves by 1.3 sigma of
the combined error; both determinations are within 2 sigma of zero, which is
what `C_rr` is expected to be at this precision -- it is the one entry where
the two runs' central values are not obviously the same number, and neither is
significant.

The identity `<cos phi_ll> = <C_kk> + <C_rr> + <C_nn>` holds to 2e-16 on every
combination (4 diagonal, 9 blocks, unpolarised), which is a closure of the
weighted bookkeeping itself.

### 4a. The null test on the production-level observables

The interference integrates to zero over the decay phase space **at every
production point**, so its contribution to a purely production-level observable
must vanish bin by bin, not just in total.  `chi2` of the five interference
blocks summed, against zero, 20 bins:

| observable | `chi2` vs 0 | (first run) |
|---|---|---|
| **`pT(t)`** | **14.9** | 12.1 |
| **`m(t t~)`** | **31.2** | 13.5 |
| `cos theta_k(l+)` | 22.5 | 23.0 |
| `C_kk` | 15.9 | 14.9 |
| `C_rr` | 29.2 | 24.4 |
| `cos theta_n(l+)` | 20.1 | 20.1 |
| **`C_nn`** | **4546.6** | 1686.2 |
| **`cos phi_ll`** | **1602.8** | 738.5 |
| **`Delta phi`** | **2996.3** | 1192.7 |

`pT(t)` is flat at zero (14.9/20).  `m(t t~)` gives 31.2/20, p = 0.053, driven
by a single +3.2 sigma bin and a single -2.5 sigma bin out of 20 (the per-bin
pulls are printed under `null test, bin by bin` in
`plots/closure_numbers.txt`); at this precision
that is a 1.9-sigma-equivalent fluctuation, and the same observable's *closure*
`chi2` is 12.7/20, so nothing propagates.  Flagged rather than smoothed over.

The signal channels are 2.5-2.7x larger in `chi2` than in the first run purely
because the per-bin interference errors are **1.6x smaller** (median ratio 0.63
on `C_nn`, and `chi2` scales as the inverse square) -- the physics is the same
size, resolved better.

### 4b. Where the effect lives: the `(I,I)` block, measured directly

`plots/blocks_cnn.pdf` shows all nine blocks separately, and now every one of
them is its own sample with no subtraction anywhere.  Null test of each group
against zero, 20 bins:

| observable | `(I,I)` alone | the four single-`I` blocks summed |
|---|---|---|
| `C_kk` | 11.3 | 13.0 |
| **`C_nn`** | **5421.7** | **17.2** |
| **`cos phi_ll`** | **2021.8** | **19.7** |
| **`Delta phi`** | **2753.9** | **332.2** |
| `pT(t)` | 19.2 | 16.2 |

For `C_nn` and `cos phi_ll` the four blocks with a *single* interference factor
are flat at zero and the whole effect is the doubly-interfering `(I,I)` block --
exactly the statement `<sigma_n sigma_n> = 2 Re[rho(+-,-+)] - 2 Re[rho(++,--)]`:
both entries have `i != j` on **both** legs, and both sit in `(I,I)` and nowhere
else.  This is now a direct measurement of `(I,I)`, not a difference of three
samples.

`Delta phi(l+,l-)` in the lab is the exception: the four single-`I` blocks give
332.2/20 there, i.e. a genuinely non-zero contribution.  That is not a defect of
this run -- the same sum on the *first* run's committed histograms gives 94.9/20
on the same observable (and 8-25 on every other), so both runs see it and this
one simply resolves it better.  It is expected physics: a single-leg cross term
is a transverse polarisation of one top, which does move a lab-frame azimuthal
separation, unlike the frame-defined spin-correlation coefficients.

## 5. Cross-check against the first run: the new card spelling

The four mixed blocks are the only ones whose *recipe* changed -- their diagonal
factor used to come from a production brace on the other leg and now comes from
a diagonal `pure_interference` entry.  `compare_to_v1.py` compares each of them
bin by bin against the first run's committed histograms.  The two sides share no
event, no production process, no card spelling and no normalisation scheme (the
first run reconstructed `max_weight / c` by hand; this one reads `sum w/N_file`
off the file), so this tests all of that at once.  Per-bin `chi2` on 20 bins:

| block | `cos th_k(l+)` | `C_kk` | `C_nn` | `C_rr` | `cos phi` | `Delta phi` | `pT(t)` | `m(tt)` |
|---|---|---|---|---|---|---|---|---|
| `(I,D+)`: card `t~ = + +` vs brace `t~{+}` | 19.9 | 21.8 | 25.6 | 19.7 | 20.8 | 20.8 | 16.0 | 24.8 |
| `(I,D-)`: card `t~ = - -` vs brace `t~{-}` | 33.2 | 28.2 | 26.5 | 25.0 | 22.5 | 20.8 | 20.3 | 24.0 |
| `(D+,I)`: card `t = + +` vs brace `t{+}` | 26.4 | 15.9 | 9.7 | 6.9 | 30.4 | 24.5 | 11.2 | 13.0 |
| `(D-,I)`: card `t = - -` vs brace `t{-}` | 20.2 | 10.1 | 11.0 | 21.1 | 15.4 | 20.9 | 14.4 | 17.1 |
| `(I,I)`: named vs the first run's subtraction | 9.1 | 16.4 | 21.1 | 24.4 | 13.9 | 19.6 | 15.0 | 18.5 |

50 comparisons, all 20 bins: median **20.2**, smallest 6.9, largest 37.4
(`(I,D-)` on `cos theta_k(l-)`, p = 0.01 -- one entry out of 50 at p = 0.01 is
expected).
**The card-named diagonal factor is the same block as the production brace**,
and the directly named `(I,I)` is the same object the first run built by
subtraction.  Full table in `plots/compare_to_v1.txt`.

## 6. Plots

MG7 paper style (`~/Documents/git_workspace/MG7_paper/plotexample/dummyplot.py`:
LaTeX serif, base size 14, step histograms at line width 1.2, fixed
`7*0.75 in` figure width, tableau palette with black/blue/red promoted,
frameless legends, minor tick locators).

* `plots/closure_summary.pdf` -- the three observables that fail without the
  interference, side by side.  Upper panel: unpolarised points, the 4-diagonal
  sum (blue dashed), the 9-block sum (red), the interference sum alone (purple
  dotted).  Ratio panel: both sums against the unpolarised sample, so the blue
  slope and the red flat line sit on top of each other.
* `plots/closure_<observable>.pdf` -- the same, one per observable, all ten.
* `plots/blocks_<observable>.pdf` -- all nine blocks separately in the upper
  panel; lower panel the five interference blocks alone with error bars, which
  is where `(I,I)` visibly carries the whole `C_nn` effect while the four
  single-`I` blocks lie on zero.
* `plots/closure_numbers.txt`, `plots/compare_to_v1.txt` -- every number above.

`data/histograms.npz` (bin edges, `sum w`, `sum w^2`, counts per sample and
observable) and `data/meta.json` are committed, so
`plot_interference.py data/ plots/` regenerates all of it with no MadSpin run.

### 6a. Stacked variant, and its sign convention

The same comparisons drawn as stacks rather than as overlaid curves, written
next to the originals under a `_stacked` suffix in both styles (nothing is
overwritten; `--stacked-only` on either script rebuilds just these).  Same
`data/histograms.npz`, same binning, same axis labels -- only the marks change.

**Sign convention, stated on every stacked figure and worth stating twice.**
Four of the nine blocks are strictly positive in every bin of every observable
here; the five interference blocks are not, and `(I,I)` in particular changes
sign in the middle of the range -- that sign change *is* the effect these plots
exist to show.  A stack that piled `|contribution|` on the previous one would
draw those bins the wrong way up and turn a subtraction into an addition, with
no visible sign that it had.  So:

> in each bin, contributions with a **positive** content are stacked upward
> from zero and contributions with a **negative** content are stacked downward
> from zero.  A band below the axis is a subtraction and can be read as one.
> The **top of the stack is therefore not the total** -- it is the sum of the
> positive contributions alone.  The total is the algebraic sum of the two
> piles and is drawn explicitly as a line, labelled `net: ...`.  **Read the
> line, not the envelope.**

This is the ROOT `THStack` convention.  The alternative -- laying each band
from the running cumulative sum so that the last band's top *is* the total --
keeps the envelope meaningful, but a negative band then overlaps the ones
beneath it, which with nine components is unreadable.  Hence the explicit net
line instead.

The stack's net reproduces the total bin by bin to
**`max |net - total| = 4.4e-16 pb`** across every stacked panel in both styles
(`1.1e-16` relative to the peak bin) -- it is the same addition regrouped, so
the residual is one unit in the last place of a double and nothing more.  The
per-panel table is in `plots/stacked_numbers.txt` and
`plots_userstyle/us_stacked_numbers.txt`.

What was stacked, and what was not:

* `closure_cos_k_p_stacked` -- the four diagonal blocks one by one, plus the
  interference sum as a fifth band.  The best of the set: the four blocks are
  positive everywhere, so this part of the stack is an ordinary one, and the
  two rising and two falling curves that the overlaid version asks the reader
  to add by eye are simply seen to add to the flat unpolarised reference.
* `closure_cnn_stacked`, `closure_dphi_lab_stacked` -- these **do** have
  something to stack, contrary to first impression: `all 9 blocks` is exactly
  `4 diagonal` **+** `5 interference`, and on these two observables the second
  term is 4% of the first and changes sign inside the frame.  `dphi_lab` is the
  clearest demonstration of the convention anywhere in this directory: the
  interference band sits *above* the diagonal block below `Delta phi ~ 2.1` and
  hangs *below zero* above it, and the net line tracks the unpolarised points
  through both.
* `blocks_cnn_stacked`, `blocks_dphi_lab_stacked` -- all nine stacked in the
  upper panel; the five interference blocks stacked alone on their own scale in
  the lower one, which is the panel the convention actually earns its place on.
  There `(I,I)` fills the frame on both sides of zero and the four single-`I`
  blocks are hairlines, which is the same conclusion as section 4b, read off a
  picture instead of a `chi2`.
* The 9-block stack is only 0.4-4.3% negative by area (`cnn` and `dphi_lab` are
  the 4% cases, everything else is well under 1%); the interference-only stack
  is 102-135% negative -- i.e. the positive and negative piles nearly cancel,
  which is the same statement as "the interference carries no cross section"
  from section 3.  Both columns are tabulated in `plots/stacked_numbers.txt`.
* `pt_t` and `m_tt` were **not** stacked.  Their closure figures are log-scale
  and their interference is a null test consistent with zero in every bin, so a
  stack would be a solid slab plus noise on a log axis, where the sign
  convention cannot be drawn at all.
* One deliberate departure from the unstacked figures: `blocks_*_stacked` gives
  each block **one** colour across both of its panels.  The unstacked
  `blocks_figure` draws `(I,I)` green above and purple below, which is harmless
  when the marker shape also changes but not when the only thing identifying a
  band is its fill.

## 7. Statistics: how much cheaper the fully weighted mode really is

Measured, on the interference contribution to three means, comparing this run's
5 x 20 000 production events against the first run's 5 x 50 000:

| | first run (250k prod. events) | this run (100k) | variance per production event |
|---|---|---|---|
| `<C_nn>` | +- 0.00090 | +- 0.00059 | **5.8x smaller** |
| `<cos phi_ll>` | +- 0.00145 | +- 0.00096 | **5.7x smaller** |
| `<Delta phi>` | +- 0.00491 | +- 0.00313 | **6.1x smaller** |

**This is about 6x, not the 30-40x the task statement expected.**  The
accept/reject kept 3-9% of production events, so the fully weighted mode does
write 11-30x more events per production event -- but they are *weighted*
events whose weights vary, so the variance per written event goes up by roughly
3x and the net gain is ~6x.  The saving is real and it is what let this run use
350 000 production events instead of 585 000 while measuring the key number 1.5x
more precisely; it just is not an order of magnitude.

The bigger saving is structural: ten samples instead of eleven, one assembly
route instead of two, five production processes instead of six, and no separate
`c`-measurement runs at all.

## 8. Verdict

**The closure passes, cleanly, and the new syntax expresses exactly what it
claims.**

* the card names exactly the five blocks with an `I` index, `(I,I)` included,
  and refuses the four diagonal-diagonal ones -- verified against the code
  before generating, and each accepted spec asserted equal to the intended
  block;
* **no subtraction and no reconstruction of the normalisation anywhere.**  One
  rule, `pb = sum_bin(w)/N_file`, asserted per sample; `c` recorded in the
  banner and consistent to 0.42% across five independent runs;
* the total rate is unchanged and **each of the five interference blocks
  integrates to zero on its own** (all below 0.8 sigma; the five together
  `-0.024 +- 0.035 pb`, 1.0e-3 of the rate), every `keep` exactly 1.0000, no
  dead weight;
* `C_nn` `chi2` 500.3 -> **24.8**, `cos phi_ll` 234.3 -> **14.4**, `Delta phi`
  339.4 -> **16.0** on 20 bins, against the first run's 25.4 / 11.6 / 14.0, and
  `<C_nn>` from `-21.5 sigma` to `+0.3 sigma`;
* the interference supplies `<C_nn> = +0.03657 +- 0.00059` against the first
  run's `+0.03626 +- 0.00090` -- 0.3 sigma apart, 1.5x more precise from 2.5x
  fewer events;
* the null test holds: `pT(t)` 14.9/20; `m(t t~)` 31.2/20 (p = 0.053, one +3.2
  sigma bin, flagged, and that observable's own closure is 12.7/20);
* the best-fit scale of the interference is `0.988 +- 0.048` (`C_nn`),
  `1.064 +- 0.062` (`Delta phi`), `1.137 +- 0.081` (`cos phi_ll`) -- consistent
  with the predicted 1, and now a prediction with *no measured constant behind
  it at all*;
* the card-named diagonal factor reproduces #349's production brace bin by bin
  on all four mixed blocks, and the directly named `(I,I)` reproduces the first
  run's subtraction (50 comparisons, median `chi2` 20.1 on 20 bins).

**Nothing found here blocks #351.**  Both defects the first run reported are
fixed and verified fixed: the `;` spelling now raises instead of truncating
silently, and `max_weight` -- along with `c`, `N_read`, the reference sigma and
the zero-cross-section check -- is in the `<MGPureInterference>` banner block,
where the first run said it belonged.  The `logger.info` that run had to add is
no longer needed.

### What was not tested

* only `spinmode = onshell` here (`PA` and `full` were not run; the plan's own
  end-to-end validation uses `madspin`);
* LO, one process, no polarised beams, no vector `{0}`/`{T}` interference in a
  closure (that is the plan's `w+ w-` test), no `fixed_order`;
* per-bin statistical precision on the unpolarised reference is 1.1-4.4%
  depending on observable and bin (1.7-2.3% on `cos phi_ll` and `Delta phi`), so
  an interference effect below ~1.5% of a bin would not be visible in the
  observables that close.  The reference is what limits every `chi2` here -- the
  per-bin interference errors are 1.6x smaller than the first run's;
* `k` is a single-parameter fit per observable and the three
  interference-sensitive observables are strongly correlated, so "consistent
  with 1" is one measurement, not three;
* the interference samples are 20 000 production events each, so the individual
  single-`I` blocks are measured to about +- 0.008 pb in rate; a per-block
  effect below that would not show.
