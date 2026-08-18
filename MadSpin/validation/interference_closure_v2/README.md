# MadSpin interference closure on `p p > t t~` -- second run

Does **diagonal + interference** reproduce the unpolarised sample?

The companion test (`../polarization_closure/`) summed the four fully polarised
samples `t{+-} t~{+-}` and found that the total rate closes exactly, that every
observable built from the diagonal of the production density matrix closes, and
that `C_nn`, `cos phi_ll` and `Delta phi(l+,l-)` do **not** -- because for a
massive fermion the helicity basis is `{-1,+1}`, so the four polarised samples
exhaust the *diagonal* of `rho_prod(i,j)` while the unpolarised sample is the
full double sum.  The deficit is the `i != j` interference.

PR #351 adds a mode that produces exactly those `i != j` terms.  The first run
of this test (`../interference_closure/`) generated them and added them back,
and the closure passed.  **This is the same test redone on the reworked
interface**, where two things that the first run had to work around are gone.

## 0. What changed, and what it buys

| | first run | this run |
|---|---|---|
| `(I,I)` block | not nameable (`;` was silently truncated); obtained by **subtraction**, `x_t - (I,D+) - (I,D-)` | **named directly** from the card |
| diagonal factor of a mixed block, e.g. the `D+` of `(I,D+)` | a production **brace on the other leg**, `p p > t t~{+}` | a diagonal `pure_interference` entry, `set pure_interference t~ = + +` |
| production processes needed | six (`t t~`, four braced pairs, two half-braced) | **five** (`t t~`, four braced pairs) -- all five interference blocks run on the *same* unpolarised process |
| samples | **eleven**, plus four extra runs to measure `c` | **ten** |
| how the blocks are combined | two routes, sharing only the four diagonal samples, cross-checked against each other | **one** sum; nothing is subtracted, so there is nothing to cross-check against |
| normalisation of an interference sample | `max_weight / c`, with `max_weight` read from a `logger.info` added for the test and `c` measured from the unweighting efficiency of a separate reference run | **none.** `pb = sum_bin(w)/N_file` for every sample in the test |
| output | accept/reject, 3-9% of production events kept, unit-magnitude signed weights | fully weighted, every trial kept, `w = sigma_ref*BR*W/c` |

`check_blocks.py` **part B** establishes the first two rows against the code
before anything is generated -- see section 2.

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

`check_blocks.py` **part A** verifies that claim against the code rather than
against this prose: it builds two random hermitian 2x2 matrices,
tensor-products them, applies the nine restrictions through
`set_hel_restriction` / `_restriction_row_mask` / `scalar_multiplication`, and
asserts that the nine masks are pairwise disjoint, that they cover all 16
entries, that each block's contraction is real, and that the nine add up to the
unrestricted contraction (relative difference 3.5e-8).  It passes.

## 2. What the card can name -- established before generating

`check_blocks.py` **part B** drives a real `MadSpinInterface` through the same
`precmd` a card line goes through, for all nine blocks, and reports whether the
card names each one and whether the restriction it produces is the intended
one.  The answer:

| | card lines | result |
|---|---|---|
| `(D+,D+)` `(D+,D-)` `(D-,D+)` `(D-,D-)` | `t = + +` / `t~ = - -` etc. | **refused** -- `_validate_pure_interference`: "every entry names a DIAGONAL block, so nothing interferes" |
| `(I,D+)` | `t = + -` / `t~ = + +` | named, spec `(((1,),(-1,)), (1,))` |
| `(I,D-)` | `t = + -` / `t~ = - -` | named, spec `(((1,),(-1,)), (-1,))` |
| `(D+,I)` | `t = + +` / `t~ = + -` | named, spec `((1,), ((1,),(-1,)))` |
| `(D-,I)` | `t = - -` / `t~ = + -` | named, spec `((-1,), ((1,),(-1,)))` |
| `(I,I)` | `t = + -` / `t~ = + -` | named, spec `(((1,),(-1,)), ((1,),(-1,)))` |

so **exactly the five blocks carrying an `I` index come from the card**, and
the four diagonal-diagonal ones -- which are not interference at all -- still
come from production braces, as in the polarisation-closure test.  The spec of
each of the five is asserted equal to the block spec part A built, so "named"
means "names *that* block", not merely "accepted".

**`(I,I)` is now named directly.**  That is the main cleanliness win: the first
run could only reach it as `x_t - (I,D+) - (I,D-)`, which is what forced eleven
samples and two independent assembly routes.

Part B also checks the three refusals the reworked interface promises: the `;`
spelling raises instead of truncating, a partial overlap (`{+,-}` against
`{-}`) raises, and two `set pure_interference` lines accumulate to
`6 = + - ; -6 = + -`.

## 3. Which sample carries which block

| block | production process | MadSpin card |
|---|---|---|
| `(D+,D+)` | `p p > t{+} t~{+}` | -- |
| `(D+,D-)` | `p p > t{+} t~{-}` | -- |
| `(D-,D+)` | `p p > t{-} t~{+}` | -- |
| `(D-,D-)` | `p p > t{-} t~{-}` | -- |
| `(I,D+)`  | `p p > t t~` | `set pure_interference t = + -` + `set pure_interference t~ = + +` |
| `(I,D-)`  | `p p > t t~` | `... t = + -` + `... t~ = - -` |
| `(D+,I)`  | `p p > t t~` | `... t = + +` + `... t~ = + -` |
| `(D-,I)`  | `p p > t t~` | `... t = - -` + `... t~ = + -` |
| `(I,I)`   | `p p > t t~` | `... t = + -` + `... t~ = + -` |

All five interference samples run on the **same unpolarised production
process**, each from its own MadEvent run with its own seed.  Nothing about
their normalisation depends on a braced parent's cross-section cancelling
against the restricted trace, which is what the first run had to argue (and
then measure) for its half-braced parents.

What still makes the nine add up is the **trace restriction being separate from
the contraction restriction** (`hel_restriction_trace`, section 13.4 of
`MADSPIN_SEQUENTIAL_PLAN.md`): an interference block has no diagonal entry, so
its own trace vanishes and cannot normalise it; the symmetric part does.

## 4. Normalisation -- there is none to establish

An ordinary MadSpin sample redraws the decay configuration until one is
accepted, so every production event yields exactly one written event and the
per-event weight is the sample's cross-section (MG5 writes LHE with
`IDWTUP = -4`: the cross-section is the *mean* of the weights).

The pure-interference mode used to accept/reject too, against a `max_weight`
that never reached the output, and write unit-magnitude signed weights.  It no
longer does.  It draws once, keeps every trial, and writes

    w = sigma_ref * BR * W / c

with `W` the signed production/decay convolution of that event and `c = <W>`
the decay-side constant the maximum-weight scan already measures.  So

    contribution of a bin, in pb  =  sum_(events in bin) w  /  N_file

is **one rule that serves every sample in this test**.  For a diagonal sample it
is `sigma x (fraction of events in the bin)`; for an interference sample it is
the interference contribution, signed, with `mean(w) = 0` over the whole file.

`analyse_interference.py:check_normalisation` asserts that convention on every
sample instead of assuming it:

* ordinary samples: `XWGTUP` is constant across the file, and equals the
  `<init>` `XSECUP` to 1e-5;
* interference samples: `XSECUP = 0` in `<init>`, the file holds exactly
  `N_read` events (fully weighted -- nothing dropped), and the sum of the file's
  weights reproduces the `S` the `<MGPureInterference>` banner block recorded.

`c` is recorded per run in that banner block.  It is a decay-side constant, so
the five independent measurements of it (one per interference run) agreeing is
a free consistency check, reported in `plots/closure_numbers.txt`.

## 5. Running it

    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
    python3 MadSpin/validation/interference_closure_v2/check_blocks.py
    bash    MadSpin/validation/interference_closure_v2/run_interference.sh <workdir> 50000 20000
    python3 MadSpin/validation/interference_closure_v2/analyse_interference.py <workdir> data/
    python3 MadSpin/validation/interference_closure_v2/plot_interference.py   data/ plots/

`analyse_interference.py` writes `data/histograms.npz` (bin edges, sum of
weights, sum of weights squared and raw counts per sample and observable) and
`data/meta.json`; both are committed, so `plot_interference.py` regenerates
every plot and every number without re-running MadSpin.

Settings: 13 TeV, LO, NNPDF23LO, `me_frame = [1,2]` (the partonic CM, the
run-card default), `spinmode = onshell`, `BW_cut = 15`,
`max_weight_ps_point = 400`, `decay t > b w+, w+ > l+ vl` and the conjugate with
`l = e, mu`.  **50 000 production events for the reference and each of the four
diagonal samples, 20 000 for each of the five interference samples**, a
different MadEvent seed for every sample so that all ten are statistically
independent and errors add in quadrature.  350 000 production events in total,
against the first run's 11 x 50 000 = 550 000 plus 35 000 more for the `c`
measurement -- see `RESULTS.md` section 7 for the measured variance reduction
that allows it.

The interference samples run with the joint scheme (the mode forces it); the
four fully polarised diagonal samples keep `onshell`'s own `sequential`
default.

## 6. Weighted histogramming

The interference samples carry **signed, varying** weights summing to zero.
Every histogram is filled with weights and every error bar is `sqrt(sum w^2)`
per bin (never `sqrt(N)`); the raw `sum w`, `sum w^2` and entry count are all
kept in the `.npz`.  Sums of samples add their `sum w^2` (the ten samples have
independent MadEvent seeds).
