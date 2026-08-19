# Pure interference under `spinmode = PA` and `spinmode = madspin`

Validation of the two output paths PR #363 ships, on the two spinmodes its own
caveat left unvalidated.

**Base**: `claude/ms-interference-pa-validation`, branched from
`origin/claude/ms-interference-unweighted` at
`1cb7c0d68 MadSpin plan: record the 13.18 regression and the final test count`
(PR #363, stacked on #351).  Nothing was merged in.

**Verdict**: both output paths give the same physics under `PA` and under
`madspin` as under `onshell`, to within 1.7 sigma of the committed closure
numbers, on 50 000 production events per combination.  Nothing here blocks
#363 or #351.  Two things are flagged rather than fixed: a latent fragility in
how a failed reshuffle is told apart from a signed weight (section 4), and two
pre-existing `fixed_order` defects that are present with the mode switched off
and are therefore not #363's (section 6).

---

## 1. What was run

`p p > t t~` at 13 TeV, NNPDF23LO, `me_frame = [1,2]`, **50 000** production
events (`iseed = 4321`, `use_syst = False`), `BW_cut = 15`,
`max_weight_ps_point = 400`, `decay t > b w+, w+ > l+ vl` and the conjugate
with `l = e, mu`, 8 cores.  The card asks for the `(I, I)` block, the same one
section 13.16 of the plan measured:

    set pure_interference t  = + -
    set pure_interference t~ = + -

The production sample is generated **once** and all six combinations decay that
same file through the standalone MadSpin front end, so the six results are
correlated event by event and the only thing that varies between them is
MadSpin.  `run_pa_madspin.sh` does the sweep, `analyse_pa_madspin.py` reads
only the decayed LHE and its `<MGPureInterference>` banner block.  A separate
`generate_events` run (section 5) checks the integrated path.

| tag | spinmode | `pure_interference_output` | exit | events in file / read | decayed file |
|---|---|---|---|---|---|
| `onshell_w` | onshell | weighted   | 0 | 50000 / 50000 | written |
| `onshell_u` | onshell | unweighted | 0 |  3832 / 50000 | written |
| `pa_w`      | **PA**  | weighted   | 0 | 50000 / 50000 | written |
| `pa_u`      | **PA**  | unweighted | 0 |  2678 / 50000 | written |
| `ms_w`      | **madspin** | weighted   | 0 | 50000 / 50000 | written |
| `ms_u`      | **madspin** | unweighted | 0 |  3458 / 50000 | written |

The two `onshell` rows are the control that this harness reproduces the
committed numbers; they are not the result.

## 2. The zero-cross-section check

Recomputed from the file, not read from the log.  `XSECUP = 0` in `<init>` in
all six, `IDWTUP = -4`, no dead trials anywhere, and no `|W|` above the
maximum-weight bound in any of the three unweighted runs (`Trials above max|W|`
= 0).

| tag | `S` | `sqrt(sum w^2)` | `z` | `mean(w)` [pb] | `mean(w)/err` | dead |
|---|---|---|---|---|---|---|
| `onshell_w` | `-1.432938e+03` | `9.814129e+02` | **-1.460** | `-2.866e-02 +- 1.963e-02` | -1.46 | 0 |
| `onshell_u` | `+6.315191e+01` | `1.954651e+02` | **+0.323** | `+1.648e-02 +- 5.101e-02` | +0.32 | 0 |
| `pa_w`      | `+1.251789e+02` | `9.899761e+02` | **+0.126** | `+2.504e-03 +- 1.980e-02` | +0.13 | 0 |
| `pa_u`      | `+5.490695e+01` | `1.578556e+02` | **+0.348** | `+2.050e-02 +- 5.894e-02` | +0.35 | 0 |
| `ms_w`      | `+1.885699e+03` | `9.818527e+02` | **+1.921** | `+3.771e-02 +- 1.964e-02` | +1.92 | 0 |
| `ms_u`      | `+3.125995e+02` | `1.802190e+02` | **+1.735** | `+9.040e-02 +- 5.209e-02` | +1.74 | 0 |

`mean(w)` is the sample's own cross-section under `IDWTUP = -4`; the largest
excursion is `ms_w` at 1.9 sigma, i.e. 0.16% of the reference `sigma*BR = 23.76`
pb.  Every entry is compatible with zero.

## 3. Weight shape

| tag | `w > 0` | `w < 0` | `w == 0` | distinct `w` | distinct `\|w\|` | `\|w\|` [pb] |
|---|---|---|---|---|---|---|
| `onshell_w` | 24768 | 25232 | 0 | 49991 | 49983 | continuous |
| `onshell_u` |  1926 |  1906 | 0 | **2** | **1** | 3.157596 |
| `pa_w`      | 25041 | 24959 | 0 | 49988 | 49969 | continuous |
| `pa_u`      |  1348 |  1330 | 0 | **2** | **1** | 3.050386 |
| `ms_w`      | 25092 | 24908 | 0 | 49992 | 49987 | continuous |
| `ms_u`      |  1780 |  1678 | 0 | **2** | **1** | 3.064701 |

Both signs are present everywhere.  The `unweighted` output holds **exactly two
signed weight values and one magnitude** under all three spinmodes -- the
property the mode advertises.  (The `weighted` counts fall a few short of 50000
only because a handful of weights coincide to 1e-9 relative; that is the
resolution of the comparison, not a feature of the sample.)

The magnitude is taken from the run, not the probe.  The probe's `<|W|>` came
out **8.2% high under `PA`** and **8.7% high under `madspin`** (6.1% under
`onshell`) -- banner `probe x` = 0.9243, 0.9198 and 0.9425, the factor by which
the written weights were corrected onto the run's own
`(N_file/N_drawn) * max|W|`.  That is the 13.17 correction doing real work on
spinmodes it had not been measured on, and it is the same size as the 9.4%
recorded for `onshell`, so the reason for it (`<|W|>` is not a decay-side
constant, and the probe sees ~80 production events) carries over unchanged.

## 4. `c` measured against its analytic candidate

This is the item the plan flagged: the analytic form
`c = 1/(prod_denominators * sym_decay)` holds only where the chain carries no
reshuffling jacobian, so off `onshell` the code must **measure** `c`.  It does,
and the measurement says exactly how wrong the analytic form is:

| tag | `c` measured | rel. err | analytic candidate | ratio |
|---|---|---|---|---|
| `onshell_w` | `2.250141e-10` | 0.134% | `2.255914e-10` | **0.997441** |
| `onshell_u` | `2.251101e-10` | 0.134% | `2.255914e-10` | **0.997867** |
| `pa_w`      | `2.155229e-10` | 0.137% | `2.255914e-10` | **0.955368** |
| `pa_u`      | `2.154331e-10` | 0.136% | `2.255914e-10` | **0.954970** |
| `ms_w`      | `2.157431e-10` | 0.143% | `2.255914e-10` | **0.956344** |
| `ms_u`      | `2.153361e-10` | 0.143% | `2.255914e-10` | **0.954540** |

Under `onshell` the ratio is 1 to within 2 sigma of the 0.13% measurement (the
plan's 13.16 got 1.0012 on an independent sample).  Under `PA` and `madspin` it
is **0.955**, i.e. `<jac> = 0.955`, some 33 sigma from 1.  So the analytic form
is genuinely inapplicable there and the code is right not to use it; it is
reported in the banner as the cross-check it is meant to be.  `PA` and
`madspin` agree with each other to 0.1%, which is what they should do -- both
carry a Breit-Wigner sampling jacobian over the same window, they just put it
in different places (inside `wgt` for `madspin`, in the outer `jac` for `PA`).

## 5. The physics -- the result

The interference contribution to an observable is `sum_i w_i O_i / N_file`
divided by the reference `sigma*BR`, for both output shapes (for `weighted`
`N_file = N_read`; for `unweighted` the written magnitude carries
`<|W|> = (N_file/N_drawn) * max|W|`, in which `N_file` cancels).  Against the
committed closure numbers -- `interference_closure_v2/RESULTS.md` section 6,
five interference blocks, 5 x 50k events:

### `<C_nn>`, committed `+0.03657 +- 0.00059`

| tag | this run | pull |
|---|---|---|
| `onshell_w` | `+0.036964 +- 0.000322` | +0.59 |
| `onshell_u` | `+0.037828 +- 0.000644` | +1.44 |
| **`pa_w`**  | **`+0.037353 +- 0.000335`** | **+1.15** |
| **`pa_u`**  | **`+0.036016 +- 0.000725`** | **-0.59** |
| **`ms_w`**  | **`+0.037017 +- 0.000326`** | **+0.66** |
| **`ms_u`**  | **`+0.036320 +- 0.000653`** | **-0.28** |

### `<C_rr>`, committed `+0.00104 +- 0.00066`

| tag | this run | pull |
|---|---|---|
| `onshell_w` | `+0.001486 +- 0.000377` | +0.59 |
| `onshell_u` | `+0.002105 +- 0.000862` | +0.98 |
| **`pa_w`**  | **`+0.001499 +- 0.000372`** | **+0.61** |
| **`pa_u`**  | **`+0.003020 +- 0.000993`** | **+1.66** |
| **`ms_w`**  | **`+0.001271 +- 0.000371`** | **+0.31** |
| **`ms_u`**  | **`+0.001706 +- 0.000856`** | **+0.62** |

Every combination agrees with the committed value, the largest pull being
+1.66.  Two supporting observables measured on the same events:
`<cos phi_ll>` comes out `+0.0388 / +0.0394 / +0.0388 / +0.0383 / +0.0382 /
+0.0384` across the six (the closure has `+0.03839 +- 0.00145`), and `<C_kk>`,
which the interference does not populate, is consistent with zero everywhere
(largest `-0.00077 +- 0.00062`).

**This is the result the task asked for: the mode gives the same physics
whichever spinmode computes it, in either output shape.**  The pulls quoted use
the quadrature sum of the two errors; the two samples are independent
(different production runs, different seeds), so that is the right combination.

## 6. The `W <= 0` hunt

### 6a. Where a non-positive `W` can come from

In `_unweight_range` the pure-interference convolution is `test = wgt * jac`,
and `jac` is 1 except in two places:

* `interface_madspin.py:4065-4075` -- `PA` with `density_keep_jacobian` (the
  **default**, `True`): `jac = full_evt.reshuffle_production()`, folded into
  the weight *before* the test;
* `madspin`/`full`: the reshuffle happens inside
  `calculate_matrix_element_from_density`, which calls
  `production.reshuffle_production()` and returns the jacobian as
  `jac_reshuffle`, folded into `wgt` by `get_onshell_evt_and_wgt`.

`Event.reshuffle_production` (`lhe_parser.py:3181`) returns RAMBO's jacobian,
`-1` when `sum(new_masses) > sqrts`, and `0` when `mass_shuffle`'s Newton solve
fails.  So `W <= 0` from a failed reshuffle is a real possibility in principle.

### 6b. What the code does with it -- the guard is gated OFF for this mode

`interface_madspin.py:4081`:

```python
if weighted_decay and signed <= 0:
    ...
    dead = True
```

The `signed <= 0 -> dead` guard is gated on `weighted_decay`, i.e. on
`decay_output = weighted`, **not** on `pure_interference`.  That is deliberate
and the comment says why: outside the interference mode `W` cannot be negative,
so a negative one means a failed reshuffle; inside it, a negative `W` is
physics.  The consequence is that in `pure_interference` the two are **not**
distinguished: only a non-finite `W` is treated as dead (`nb_pi_dead`).  A
`W <= 0` from a failed reshuffle is treated as physics.

The `_dead_trial` backstop does not cover it either.  In the joint loop
`_dead_trial` sits in the `else` branch (line 4124) -- the ordinary
accept/reject -- and is not on the pure-interference path at all, which the
comment at 4109-4112 states outright.

### 6c. So it was measured, then forced

**Measured**, on the 50k runs, with `Event.reshuffle_production` wrapped and
counted (`drive_madspin.py`):

| tag | reshuffle calls | returns `<= 0` | zero | negative | non-finite | internal retries |
|---|---|---|---|---|---|---|
| `onshell_w` / `onshell_u` | 0 | 0 | 0 | 0 | 0 | 0 |
| `pa_w` / `pa_u` | **94016** each | **0** | 0 | 0 | 0 | **0** |
| `ms_w` / `ms_u` | **94016** each | **0** | 0 | 0 | 0 | **0** |

(44016 in the maximum-weight probe plus 50000 in the event loop, per run.)

**It never fires.**  Not once in ~376 000 reshuffles across the four PA/madspin
runs, and the internal retry counter `Event.nb_reshuffle_issue` never moves
either.  So *this run does not test the `W <= 0` handling*, and PA cannot be
called validated on that path by the sweep alone.  There is a mechanism behind
the zero rather than luck: `_draw_mass_value` caps each drawn virtuality at the
remaining `sqrt(shat)` budget (`max_mass = min(pole + bw_cut*width, budget)`),
so `sum(new_masses) <= sqrts` holds by construction and RAMBO's `-1` branch is
unreachable from this direction.

**Forced** (`run_badjac.sh`): `reshuffle_production` made to return a
non-positive value for 10% of trials **in the event loop only** (injecting into
the probe would corrupt `c` and `max|W|` instead of testing the loop), `PA`,
2000 events, against a `p = 0` control on the same events and seed.

| case | injected | events in file | `w == 0` | dead counter | `z` | `<C_nn>` / control |
|---|---|---|---|---|---|---|
| control, weighted   |   0 | 2000 |   0 | 0 | -1.10 | -- |
| control, unweighted |   0 |  116 |   0 | 0 | -0.37 | -- |
| `jac = 0`, weighted   | 200 | 2000 | **200** | 0 | -0.95 | 0.861 (-2.3 sigma) |
| `jac = 0`, unweighted | 204 |  133 | **0** | 0 | +0.43 | 0.902 (-0.8 sigma) |
| `jac = -1`, weighted   | 224 | 2000 | 0 | 0 | +0.44 | **0.709 (-4.7 sigma)** |
| `jac = -1`, unweighted | 228 |  146 | 0 | 0 | -0.66 | **0.671 (-2.4 sigma)** |
| `jac = 0`, `decay_output = weighted`  | 196 | 2000 | 196 | **196** | -- | -- |
| `jac = -1`, `decay_output = weighted` | 184 | 2000 | 184 | **184** | -- | -- |

Reading the rows:

* **`jac = 0`, weighted** -- the trial is written with `w = 0` exactly (200
  injected, 200 zero weights), all 2000 events still written, and it does *not*
  move `z` or `mean(w)`, because a zero weight contributes nothing to either
  `S` or `sum w^2`.  Handled correctly.
* **`jac = 0`, unweighted** -- none of the injected trials reach the file.  The
  accept test is `random()*M >= |W|`, which at `|W| = 0` is true with
  probability 1, so an invalid trial can never be accepted.  Handled correctly.
* **`jac = -1`, either output** -- every monitor the mode has stays silent.
  `z` is +0.44 and -0.66, indistinguishable from the controls; the dead counter
  stays at 0; no zero weights appear.  It *cannot* show up in `z`: `W` is
  already signed with mean zero, so flipping the sign of a random subset leaves
  `S` at zero by construction.  What it does move is the physics -- `<C_nn>`
  drops to 0.71 and 0.67 of the control.  In the unweighted case the injected
  trials are also accepted *more* often than real ones (146 events kept against
  the control's 116), because `|-1 * wgt|` is larger than `|jac * wgt|` for the
  `jac < 1` this process produces.  That is precisely the conflation the task
  was worried about, demonstrated.
* **`decay_output = weighted`** -- the same injections move the dead counter by
  exactly the number injected (196 and 184) and are written with weight 0.
  That is the `signed <= 0` guard doing its job, and it is the contrast that
  shows the guard is what is missing on the interference path.

### 6d. Why it is nevertheless not a live bug

`jac = -1` cannot actually reach either pure-interference path, for a reason
that lives outside them.  Both `_unweight_range:4075` and
`calculate_matrix_element_from_density` call
`reshuffle_production(_allow_retry=True)` -- the default -- and that branch
(`lhe_parser.py:3229-3241`) resamples the resonance masses and recurses
whenever RAMBO reports `jac in (0, -1)`.  Only `_allow_retry=False` callers see
the sentinel, and those are `production_jacobian` and
`_production_jacobian_for`, which belong to the **sequential** schemes --
which `pure_interference` forces off (`interface_madspin.py:3273`, "every
partial weight of a staged scheme is identically zero in this mode").

What *can* reach the caller is a `0`, through the `jac *= reshuffle_decay(...)`
product that runs *after* the retry check, and that is the case both output
paths handle correctly.

**So: no bug to fix in #363, but a fragility to record.**  The pure-interference
paths are protected from a signed invalid trial by a retry policy in
`lhe_parser`, not by anything in the mode itself, and the mode has no monitor
that would notice if that protection were removed -- the `z` test structurally
cannot see it.  If `_allow_retry=False` ever propagates to the joint loop (as
it already has in the sequential scheme), both new output paths would silently
accept invalid trials carrying a sign.  A one-line guard --
`if signed <= 0 and jac <= 0` rather than `if weighted_decay and signed <= 0`
-- would close it, and would be a no-op today.

## 7. `generate_events`, and where the decayed file lands

The sweep uses the standalone front end, which writes `<input>_decayed.lhe.gz`
next to its input.  The integrated path was checked separately: the same card
(`PA`, `weighted`) through `generate_events -f pa_integration`, 2000 events.

* exit 0;
* the decayed events are at
  `Events/pa_integration_decayed_1/unweighted_events.lhe.gz` -- MadSpin writes
  a **new run directory** and leaves the production run in place, which is the
  ordinary MG5 convention and what `interference_closure_v2` reads;
* 2000 events, 12 particles each (i.e. genuinely decayed), `XSECUP = 0`,
  `<MGPureInterference>` present, `z = +0.155`, `c/c_analytic = 0.9578`.

Looking in `Events/pa_integration/` instead finds the undecayed production
sample; that is not a failure, it is the wrong directory.

## 8. `fixed_order` -- two defects, both pre-existing

`fixed_order` was the other path #363 left unvalidated.  A real aMC@NLO
fixed-order sample was not produced (that needs an NLO generation and compile);
instead the production sample was rewrapped into the `<eventgroup>` format
`fixed_order` actually reads -- one born event plus one counter-event per group
-- which exercises the plumbing but not the physics of NLO subtraction.  Even
that is enough to find two problems, and both show up with `pure_interference`
**off**:

| run | spinmode | mode | member 0 | member 1 |
|---|---|---|---|---|
| `fixedorder`      | PA      | `pure_interference` ON  | **4 particles** | **4 particles** |
| `fo_ctrl_onshell` | onshell | `pure_interference` OFF | 12 particles | **4 particles** |
| `fo_ctrl_PA`      | PA      | `pure_interference` OFF | **4 particles** | **4 particles** |
| `fo_ctrl_madspin` | madspin | `pure_interference` OFF | 12 particles | **4 particles** |

(12 particles = the decayed event, 4 = `t t~ g` undecayed.)

1. **The counter-events are never decayed, in every spinmode.**
   `_unweight_range:4157` builds them as
   `[evt.add_decays(decays) for evt in counterevt]`, but `decays` has already
   been drained by the born event's `add_decays` -- `Event.add_decays` copies
   the dict shallowly and `pop(0)`s the caller's lists
   (`lhe_parser.py:3204-3210`), so the second call sees empty lists.
2. **Under `PA`, not even the born event is decayed.**  `fixed_order` forces
   `build_event = True` (`:4046`), so `get_onshell_evt_and_wgt` builds and
   consumes `decays` itself; then the `PA` + `density_keep_jacobian` branch
   (`:4073-4075`) rebuilds `full_evt` from the production and the now-empty
   dict before reshuffling it.

Neither is #363's: with the mode off, none of #363's code runs, and the
responsible `build_event = (not density_method) or self.options['fixed_order']`
line comes from `1bb7ee1de Optimisations` on `origin/madspin_density`, an
ancestor of the PR-stack base `d19d8a293`.  #363's own bookkeeping on top of
this is in fact correct: the pure-interference factor reaches every member of
the group (weights `+5.4694` and `-1.6408`, preserving the `-0.3` ratio the
synthetic counter-event was given), 2000/2000 groups are written, and
`z = +0.83`.  It is the event *content* underneath that is wrong.

**Recommendation**: `fixed_order` should not be advertised as working with the
density spinmodes until this is fixed, and that is a separate change from #363.

## 9. Regression suite

    python3 tests/test_manager.py test_madspin -t0
    Ran 325 tests in 13.470s
    OK

325, matching the count recorded for this base.  Nothing in this directory
touches shipped source.

---

## Files

| file | what |
|---|---|
| `run_pa_madspin.sh` | the six-combination sweep on one shared production sample |
| `run_badjac.sh` | the `W <= 0` fault injection, with `p = 0` controls |
| `run_extras.sh` | the `generate_events` run and the `fixed_order` runs |
| `drive_madspin.py` | `MadSpin/madspin` plus the reshuffle instrumentation and the fault injector; modifies nothing in the tree |
| `analyse_pa_madspin.py` | the sweep numbers (LHE reader and spin basis copied verbatim from `interference_closure_v2/analyse_interference.py`) |
| `analyse_badjac.py` | the fault-injection table |
| `analyse_extras.py` | file placement and the `fixed_order` member counts |
| `data/results_50k.json` | every number in sections 1-5 |
| `data/results_pilot_2k.json` | the 2 000-event pilot, as an independent cross-check |
| `data/badjac.json` | section 6c |
| `data/extras.json` | sections 7-8 |

Reproduce with:

    bash MadSpin/validation/interference_pa_madspin/run_pa_madspin.sh <work> 50000
    bash MadSpin/validation/interference_pa_madspin/run_badjac.sh   <work> 2000 0.10
    bash MadSpin/validation/interference_pa_madspin/run_extras.sh   <work> 2000
    python3 MadSpin/validation/interference_pa_madspin/analyse_pa_madspin.py <work> <out>
    python3 MadSpin/validation/interference_pa_madspin/analyse_badjac.py     <work> <out>
    python3 MadSpin/validation/interference_pa_madspin/analyse_extras.py     <work> <out>
