# Closure test of the MadSpin production-polarisation restriction

`p p > t t~` at 13 TeV, LO, dileptonic, MadSpin `spinmode = madspin` (density).

Validation of PR #349 (`claude/ms-density-pol-restrict`, commit `30e0596f7`,
which contains #355 -- the polarisation is defined on the `me_frame` axis, i.e.
the partonic CM, not the lab).

## What is being tested

For a massive fermion the helicity basis is `{-1,+1}`, so the four processes

    p p > t{+} t~{+}    p p > t{+} t~{-}    p p > t{-} t~{+}    p p > t{-} t~{-}

exhaust the **diagonal** of the production spin-density matrix `rho(i,j)` in the
joint `(h_t, h_tbar)` index.  Decayed through MadSpin, the accept/reject weight
of a polarised sample is (`MADSPIN_SEQUENTIAL_PLAN.md`, section 1)

    sum_i  rho(i,i) rho_dec(i,i)          restricted to the one allowed i

while the unpolarised `p p > t t~` uses the full double sum

    sum_i sum_j  rho(i,j) rho_dec(i,j).

So **the sum of the four polarised samples is not expected to reproduce the
unpolarised one**.  The difference is exactly the `i != j` interference between
different production helicity amplitudes.  That is physics, not a bug, and the
point of the test is to check that it shows up *only* where it should:

| observable | depends on | closure expected |
|---|---|---|
| total rate | `sum_i rho(i,i)` -- the trace, no off-diagonal term | **yes, exactly** |
| `cos theta_k` of each lepton against its parent's helicity axis | the diagonal of that particle's reduced density matrix (the partner index is traced, which kills its off-diagonal part) | **yes** |
| `C_kk = cos theta_k(l+) * cos theta_k(l-)` | `rho(i,i)` only, both indices diagonal | **yes** |
| `C_nn`, `C_rr` (transverse spin correlations) | `rho(+-,-+)` etc. -- transverse spin needs coherence between `h = +1` and `h = -1` | **no** |
| `cos phi_ll = l+ . l-` in the parent rest frames | `= C_kk + C_rr + C_nn`, so it carries the transverse part | **no** |
| `Delta phi(l+, l-)` in the lab | same, plus the boost | **no** |
| `pT(t)`, `m(t t~)` | production kinematics only, the decay density matrix integrates out | **yes** -- null test of the machinery |

A failure in the last row (or in the total rate) would mean the machinery is
broken; a deviation in `C_nn` / `C_rr` / `cos phi` is the interference and is
expected to be large.

## Frames and definitions

* `me_frame` in the run card is `1,2` (`frame_id = 6`), the partonic CM.  Since
  this is LO `p p > t t~` with nothing else in the final state, the partonic CM
  is the `t t~` rest frame, and the boost from the lab to it is purely
  longitudinal, so the beam axis is preserved.
* `k` = direction of the `t` in the `t t~` rest frame (the helicity axis).
* `n`, `r` = the usual normal / transverse axes built from `k` and the beam:
  `r = sign(cos T) (z - cos T k)/sin T`, `n = sign(cos T) (z x k)/sin T`.
* Each lepton is boosted to the `t t~` rest frame first and then into its
  parent's rest frame, so that the quantisation axis really is the parent
  direction in `me_frame`.
* `cos theta_k(l-)` is measured against the **antitop** helicity axis (`-k`).

## Samples

See `run_closure.sh`.  Same run card, same MadEvent seed (`iseed = 4321`), same
MadSpin seed (`7777`), same MadSpin card for all five.  Each polarised process
is a separate MadEvent output -- a polarised amplitude cannot be extracted from
a single unpolarised run.

Event counts are **not** identical across samples, on purpose: the
opposite-helicity samples carry half the cross section of the like-helicity ones
and are ~5x slower in the MadSpin accept/reject, so `N` proportional to `sigma`
(50k / 50k / 20k / 20k) is the optimal allocation and keeps the per-event weight
nearly uniform across the four samples that get summed.

Note that QCD is parity conserving, so `|M(+,+)|^2 = |M(-,-)|^2` and
`|M(+,-)|^2 = |M(-,+)|^2` point by point.  With one shared seed, MadEvent
therefore produces **byte-identical production momenta** for `pp`/`mm` and for
`pm`/`mp` (verified on the LHE files -- only the helicity column differs).  The
error bars quoted on the sum add the two members of each parity pair *linearly*
(the fully-correlated bound) rather than in quadrature; the uncorrelated value
is also reported in `plots/closure_numbers.txt`.

## Reproducing

    export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"   # f2py + meson + ninja
    bash MadSpin/validation/polarization_closure/run_closure.sh /some/workdir

then

    python3 MadSpin/validation/polarization_closure/analyse_closure.py \
            /some/workdir MadSpin/validation/polarization_closure/plots closure

## Results

See `RESULTS.md` and `plots/`.
