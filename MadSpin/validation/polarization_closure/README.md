# Closure test of the MadSpin production-polarisation restriction

`p p > t t~` at 13 TeV, LO, dileptonic, MadSpin `spinmode = onshell` (a density
spinmode -- see "Which spinmode, and why" below).

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

## Which spinmode, and why

`_density_spinmode()` (`MadSpin/interface_madspin.py`) covers `madspin`/`full`,
`PA` and `onshell`.  All of them go through
`calculate_matrix_element_from_density`, i.e. through the `set_hel_restriction`
mask that #349 adds and the `_frame_boost` that #355 fixes.  `onshell` differs
from `madspin` only in that it does **not** Breit-Wigner-reshuffle the
virtualities; the spin-density convolution being validated here is identical.

The test uses `onshell` because the offshell mode is unusable for a polarised
production on this process.  Measured accept/reject cost, same cards, same
machine:

Accept/reject trials per written event, identical cards otherwise:

| production | `madspin` (offshell, `unweighting=joint`) | `onshell` (`unweighting=sequential`) |
|---|---|---|
| `p p > t t~` (unpolarised) | **4.05** | **4.11** |
| `p p > t{+} t~{+}` | **213** | **4.43** |
| `p p > t{-} t~{-}` | **204** | **4.44** |
| `p p > t{+} t~{-}` | -- | **4.45** |
| `p p > t{-} t~{+}` | **5800 - 6300** | **4.47** |

The `t{-} t~{+}` figure was **1066** in an earlier `madspin` run that differed
only in `nevents`, i.e. only in the size of the max-weight probe.  A bound that
moves by a factor 6 with the probe size is set by a long tail, not by a stable
maximum, and at 6000 trials/event the run does not finish.  See `RESULTS.md`
section 5 for the full discussion; in short it is a **CPU cost, not a bias**
(an over-estimated bound is always safe, and no overflow warning was emitted in
any run), but it is what forced the switch to `onshell`.

## Samples

See `run_closure.sh`.  Same run card, same number of events (**50 000 unweighted
events per sample**, 250 000 in total), same MadEvent seed (`iseed = 4321`),
same MadSpin seed (`7777`), same MadSpin card for all five.  Each polarised
process is a separate MadEvent output -- a polarised amplitude cannot be
extracted from a single unpolarised run.

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

**`RESULTS.md`** -- the numbers and the verdict.  Raw output in
`plots/closure_numbers.txt`, plots in `plots/`, overview in
`plots/closure_summary.png`.

The whole test was run **twice**: once in `spinmode onshell` (`plots/`, the
quoted result) and once in the default offshell mode, `spinmode madspin` with
`set unweighting sequential` (`plots_offshell/`).  Both give the same answer;
section 6 of `RESULTS.md` compares them and records one caveat that only shows
up in the offshell+sequential pass.

Short version: the total rate closes to 6e-4 (0.00 sigma before the decay,
-0.38 sigma after); every diagonal observable closes at the 2-4% per-bin
statistical precision (`chi2/ndf` 1.2, 0.6, 0.4); both controls close
(`chi2/ndf` 0.7, 0.9); and the only deviations are in the transverse spin
correlation `C_nn` (removed entirely, 20 sigma) and the two observables that
inherit it, `cos phi_ll` and `Delta phi(l+,l-)`.  That is the off-diagonal
interference, and it is quantitatively accounted for.
