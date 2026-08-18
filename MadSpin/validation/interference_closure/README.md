# MadSpin interference closure on `p p > t t~`

Does **diagonal + interference** reproduce the unpolarised sample?

The companion test (`../polarization_closure/`) summed the four fully polarised
samples `t{+-} t~{+-}` and found that the total rate closes exactly, that every
observable built from the diagonal of the production density matrix closes, and
that `C_nn`, `cos phi_ll` and `Delta phi(l+,l-)` do **not** -- because for a
massive fermion the helicity basis is `{-1,+1}`, so the four polarised samples
exhaust the *diagonal* of `rho_prod(i,j)` while the unpolarised sample is the
full double sum.  The deficit is the `i != j` interference.

PR #351 adds a mode that produces exactly those `i != j` terms.  This test
generates them and adds them back.

## 1. The decomposition, stated before running

`W = sum_i sum_j rho_prod(i,j) rho_dec(i,j)`, the joint index `i = (h_t,
h_tbar)`.  The restriction the code applies is a **product of per-index
conditions** (`DensityMatrix._restriction_row_mask`), so it is a per-particle
statement.  On a two-state basis a per-particle condition that is closed under
`(bra,ket) -> (ket,bra)` -- the closure that makes the contraction real -- can
take exactly three inequivalent forms:

| form | spelling in the code | entries of that particle's 2x2 block |
|---|---|---|
| `D+` | symmetric `{+1}` | `(+,+)` |
| `D-` | symmetric `{-1}` | `(-,-)` |
| `I`  | cross `({+1},{-1})` | `(+,-)` and `(-,+)` |

Three per particle, `3 x 3 = 9` in the product: **4 diagonal-diagonal** (the
previous test's four samples, 1 entry each), **4 with one interference factor**
(2 entries each) and **1 with two** (4 entries) -- `4 + 8 + 4 = 16`, the whole
`4 x 4` joint matrix, each entry exactly once.

`check_blocks.py` verifies that claim **against the code** rather than against
this prose: it builds two random hermitian 2x2 matrices, tensor-products them,
applies the nine restrictions through `set_hel_restriction` /
`_restriction_row_mask` / `scalar_multiplication`, and asserts that the nine
masks are pairwise disjoint, that they cover all 16 entries, that each block's
contraction is real, and that the nine add up to the unrestricted contraction
(relative difference 3e-8).  It passes.

## 2. Which sample carries which block

A block is obtained by running MadSpin over a production sample that contains
the amplitudes it needs, with the restriction on top.  The production braces
give the *symmetric* part of the restriction and the normalising trace; the
card's `pure_interference` gives the cross part.

| block | production process | MadSpin card |
|---|---|---|
| `(D+,D+)` | `p p > t{+} t~{+}` | -- |
| `(D+,D-)` | `p p > t{+} t~{-}` | -- |
| `(D-,D+)` | `p p > t{-} t~{+}` | -- |
| `(D-,D-)` | `p p > t{-} t~{-}` | -- |
| `(I,D+)`  | `p p > t   t~{+}` | `set pure_interference t = + -` |
| `(I,D-)`  | `p p > t   t~{-}` | `set pure_interference t = + -` |
| `(D+,I)`  | `p p > t{+} t~`   | `set pure_interference t~ = + -` |
| `(D-,I)`  | `p p > t{-} t~`   | `set pure_interference t~ = + -` |
| `(I,I)`   | *not directly reachable -- see below* | |

Two things make this work, and both are properties of the implementation rather
than of the test:

* the **trace restriction is separate from the contraction restriction**
  (`hel_restriction_trace`, section 13.4 of `MADSPIN_SEQUENTIAL_PLAN.md`).  For
  `(I,D+)` the trace restriction is the antitop brace `{+}`, which is exactly
  the parent sample's own cross-section, so the `Tr_S rho_prod` in the
  denominator of the reweighting cancels against the parent's density.  That
  cancellation is why the nine blocks add up with no relative factor between
  them despite coming from six different production samples.
* `_validate_pure_interference` only refuses a brace on the leg the *cross*
  restriction is asked for.  A brace on the **other** leg is allowed, and is
  what selects `D+` or `D-` there.

### The `(I,I)` block, and a bug found on the way

`(I,I)` needs two cross entries in one card,

    set pure_interference t = + - ; t~ = + -

and that **cannot be written**.  `extended_cmd.Cmd.precmd` splits every card
line on `;` and executes the pieces as separate commands, so only `t = + -`
ever reaches the option and ` t~ = + -` is silently swallowed (it is not even an
error).  `_pure_interference` splits its option string on `;` and is plainly
written to accept several particles, so the intent is there; the separator just
collides with the command splitter.  This is reported as a finding, not worked
around in the implementation.

The block is obtained instead from the coarser mask that *is* reachable.  With
no brace on the antitop, its per-particle restriction is `None` -- the full 2x2
block -- so

    x_t   :  p p > t t~,  set pure_interference t  = + -   =  (I,D+) + (I,D-) + (I,I)
    x_tb  :  p p > t t~,  set pure_interference t~ = + -   =  (D+,I) + (D-,I) + (I,I)

(also verified against the code in `check_blocks.py`).  The nine-term total is
then assembled two independent ways,

    route 1 :  4 diagonal  +  x_t   +  (D+,I) + (D-,I)
    route 2 :  4 diagonal  +  x_tb  +  (I,D+) + (I,D-)

and `(I,I)` itself is available twice, as `x_t - (I,D+) - (I,D-)` and as
`x_tb - (D+,I) - (D-,I)`, which is a further cross-check.

## 3. Normalising a pure-interference sample

This is the one place where a constant has to be supplied, and it is *derived*,
not fitted.

An ordinary MadSpin sample redraws the decay configuration until one is
accepted: every production event yields exactly one written event, and the
per-event weight is `sigma_decayed / N`.  That is unbiased because
`<W>_decay-phase-space = c` is the *same constant for every production event*
(section 13.7b of the plan).

The pure-interference mode cannot redraw -- its mean weight is zero and the
quantity that must be allowed to vary is `<|W|>`, the local size of the
interference.  It draws once, keeps the event with probability
`|W| / max_weight`, and writes `+- sigma_parent * BR`.  For any observable `O`,

    kept sum of  sign * w * O   ->   (sigma_ref / N_read) * sum_p Int W O dOmega / max_weight

against the redraw scheme's

    sum over events of w * O    ->   (sigma / N)          * sum_p Int W O dOmega / c

so an interference sample has to be multiplied by

    max_weight / c

to sit on the same footing as the diagonal ones.  Nothing else.

* `max_weight` is the joint accept/reject bound of that run.  It was not
  recoverable from the output (the sequential scheme logs its per-slot bounds,
  the joint one logged nothing, and the `ms_dir` route that writes `max_wgt` to
  disk also switches the decay pool to the gridpack path).  This branch adds one
  `logger.info` line in `get_maxwgt_for_onshell` -- pure logging, no behaviour
  change -- and the analysis reads it back.
* `c` is **measured, not assumed**: the unpolarised reference sample is run with
  `set unweighting joint`, where the unweighting efficiency is `c / max_weight`,
  so `c = eff * max_weight`.  `c` is a decay-side constant (the production
  density matrix cancels between the contraction and its normalising trace), so
  it is common to all eleven samples.
* the resulting factor is a *prediction*.  `plot_interference.py` also reports
  the best-fit scale `k` of the interference contribution against the
  unpolarised sample; `k = 1` is the prediction, and how close it comes is
  quoted in `RESULTS.md` rather than used anywhere.

## 4. Running it

    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    bash    MadSpin/validation/interference_closure/run_interference.sh <workdir> 50000
    python3 MadSpin/validation/interference_closure/analyse_interference.py <workdir> data/
    python3 MadSpin/validation/interference_closure/plot_interference.py   data/ plots/

`analyse_interference.py` writes `data/histograms.npz` (bin edges, sum of
weights, sum of weights squared and raw counts per sample and observable) and
`data/meta.json`; both are committed, so `plot_interference.py` regenerates
every plot and every number without re-running MadSpin.

Settings: 13 TeV, LO, NNPDF23LO, `me_frame = [1,2]` (the partonic CM, the
run-card default), `spinmode = onshell`, `BW_cut = 15`,
`max_weight_ps_point = 400`, `decay t > b w+, w+ > l+ vl` and the conjugate with
`l = e, mu`, 50 000 production events per sample, a different MadEvent seed for
each of the eleven samples so that they are statistically independent and errors
simply add in quadrature.

The interference samples run with the joint accept/reject -- the mode forces it,
and it is cheap here because the mode draws the decay configuration once per
production event whatever the acceptance.  The four fully polarised diagonal
samples keep `onshell`'s own default (`sequential`); under `joint` they cost
200-6000 trials per event (`../polarization_closure/RESULTS.md`, section 5).

## 5. Weighted histogramming

The interference samples carry **signed** weights of constant magnitude, summing
to zero.  Every histogram is filled with weights and every error bar is
`sqrt(sum w^2)` per bin (never `sqrt(N)`); the raw `sum w`, `sum w^2` and entry
count are all kept in the `.npz`.  Sums and differences of samples add their
`sum w^2` (the eleven samples have independent MadEvent seeds).
