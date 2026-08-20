# `m(t)` lineshape across every MadSpin spinmode and unweighting scheme

Baseline on the shipped code (`a311d2e64`), before any per-event mass-bound
work.  Raw histograms in `data/`, figures in `plots/` (MG7 paper style) and
`plots_userstyle/` (the user's own style), full numbers in
`plots/lineshape_numbers.txt`.

**Answer up front.** Every unweighting scheme reproduces the lineshape of its
own spinmode, in every spinmode, to within the noise floor set by re-running
one scheme with a different MadSpin seed.  Nothing in the grid is a bug.  The
differences that *are* there are all deliberate: the pole approximation moves
the lineshape by 220 MeV in the mean and by up to 60% in the tails, and
`density_keep_jacobian = False` moves the mean by 33 MeV.

## Setup

| | |
|---|---|
| process | `p p > t t~`, 13 TeV, LO |
| decays | `t > b w+, w+ > l+ vl`; `t~ > b~ w-, w- > l- vl~` (fully leptonic) |
| statistics | 200 000 production events, one madevent sample shared by every cell |
| seeds | madevent + MadSpin seed 42 everywhere; the two replica rows use MadSpin seed 20260820 off the *same* production events |
| `nb_core` | 8 |
| pole / width | `M = 173.000000` GeV, `Gamma = 1.491500` GeV, read off the production banner's param_card |
| BW support | `BW_cut = -1` -> 15, so `|m - M| < 15 Gamma` = `[150.6275, 195.3725]` GeV; **no event in any cell fell outside it** |
| production sigma | 504.449 pb |

**Observable.** The per-event invariant mass `sqrt(E^2 - |p|^2)` of the
status-2 resonance in the decayed LHE, separately for `t` and `t~`.  That is
the virtuality MadSpin assigned to that top in that event.  Not a fitted width,
not a pole parameter.  Means and rms of it are single numbers and appear in the
tables; the histograms are of the quantity itself, and the axis labels say so.
The LHE was read twice per run, once by a fast text reader and once by the
repository's own `lhe_parser`; the two agreed bin for bin in all 22 runs
(`--cross-check`).  As a second, independent check the LHE's own mass column
was compared against the 4-vector mass in every event of every run: the largest
disagreement anywhere is 2.2e-6 GeV.  Total wall time for the 22 runs plus the
production sample: 65 min on 8 cores.

**Binning.** `data/histograms.npz` holds a fine uniform grid (`Gamma/12`, 360
bins) so the committed data is the raw measurement.  The figures group whole
numbers of those into `Gamma/6` in the core (`|m-M| < 3.5 Gamma`), `Gamma/2` on
the shoulders (to `9 Gamma`) and `2 Gamma` in the far tails (to `15 Gamma`) --
70 bins, no edge moved, nothing interpolated.  Per-bin statistical error runs
from **0.69% at the peak to 8.0% in the sparsest bin** (156 entries, at
159.9 GeV).  Every statement below is quoted with its error and none of them
lean on a bin below that.

## Which grid cells are real

22 runs.  Not 6 spinmodes x 4 schemes: most of that product is the same code
path twice.  Every collapse below was **checked bin by bin on the raw
histograms**, not asserted from reading the source, and the mode MadSpin itself
announced in each log is recorded in `data/meta.json`.

### 14 cells with a lineshape

| spinmode | schemes | note |
|---|---|---|
| `madspin` | `joint`, `sequential`, `sequential_global_retry` | 3 real cells |
| `PA` | `joint`, `sequential`, `sequential_global_retry`, `sequential_with_mass` | 4 real cells |
| `PA`, `density_keep_jacobian = False` | the same 4 | a second axis, own figure |
| `madspin_v1` | 1 cell | the `unweighting` entry never reaches `_unweighting_mode`: no line in the log, and the two runs are bit-identical |
| replicas | `madspin/joint`, `PA/joint` at a second seed | the noise floor |

### 4 cells with no lineshape at all

`onshell` under all three schemes, and `none`.  Every event comes out at the
pole mass: `m(t)` spans `[172.9999983181, 173.0000018207]` — a 3.5e-6 GeV
spread, i.e. double-precision reconstruction noise on the 4-vector, not a
distribution.  `onshell` samples no virtuality (`_density_do_reshuffle` is
False, `slot_mass` comes back empty) and `none` does not smear the resonance
either.  Drawing these as a histogram would be a one-bin spike whose height is
set by the binning, so the figures carry a vertical line at the pole instead of
a fake curve.  They are in the data and in the rate table.

### 4 cells that are literally another cell

| run | is bit-identical to | why |
|---|---|---|
| `full` / `joint` | `madspin` / `joint` | `run_madspin` rewrites `spinmode = full` to `'madspin'` before anything else reads it |
| `madspin` / `sequential_with_mass` | `madspin` / `sequential` | `sequential_with_mass` needs a per-particle mass draw, i.e. `PA`; the log says *"the offshell spinmodes do not have (they reshuffle the whole production onto the mass set at once); using sequential instead"* |
| `madspin_v1` / `sequential` | `madspin_v1` / `joint` | no density matrix, so the card entry is inert |
| `none` / `sequential` | `none` / `joint` | same |

"Bit-identical" here means every one of the 360 raw bins matched exactly, for
both `t` and `t~`.  `sequential_with_mass` under `onshell` was not run: it
would fall back the same way, and `onshell` has no lineshape to compare anyway.

## The closure test: does the scheme change the lineshape?

Each cell against its own spinmode's `joint`, `chi2` on 140 bins
(`m(t)` + `m(t~)`), 138 dof.  `sqrt(2*138) = 16.6`, so that is 1 sigma.

| cell | m(t) | m(t~) | combined | n sigma |
|---|---|---|---|---|
| `madspin` / sequential | 58.3/69 | 48.6/69 | 106.9/138 | -1.9 |
| `madspin` / seq. global retry | 53.8/69 | 98.4/69 | 152.2/138 | +0.9 |
| `PA` / sequential | 58.0/69 | 59.5/69 | 117.5/138 | -1.2 |
| `PA` / seq. global retry | 52.6/69 | 77.9/69 | 130.5/138 | -0.5 |
| `PA` / seq. with mass | 52.9/69 | 64.9/69 | 117.8/138 | -1.2 |
| `PA` no-jac / sequential | 54.9/69 | 77.2/69 | 132.1/138 | -0.4 |
| `PA` no-jac / seq. global retry | 53.7/69 | 72.9/69 | 126.6/138 | -0.7 |
| `PA` no-jac / seq. with mass | 76.6/69 | 63.7/69 | 140.3/138 | +0.1 |
| `madspin_v1` (vs `madspin`/joint) | 55.9/69 | 54.0/69 | 109.9/138 | -1.7 |
| **`madspin`/joint, replica seed** | 65.7/69 | 73.5/69 | **139.2/138** | **+0.1** |
| **`PA`/joint, replica seed** | 62.5/69 | 73.1/69 | **135.7/138** | **-0.1** |

The two replica rows are the same scheme with a different MadSpin seed off the
same production events: 139.2 and 135.7.  **Every closure row is at or below
them.**  The largest, `madspin` / sequential global retry at 152.2 (+0.9
sigma), is one bin's worth of fluctuation on `m(t~)` that does not appear on
`m(t)` (98.4 vs 53.8).  Several rows sit *below* 138, which is expected: the
cells share production events, so the independent-error `chi2` is mildly
conservative.

Mean shifts tell the same story.  Combined over `t` and `t~`:

| cell | m(t) [MeV] | m(t~) [MeV] | combined [MeV] | n sigma |
|---|---|---|---|---|
| `madspin` / sequential | +27.2 +- 10.1 | +2.2 +- 10.1 | **+14.7 +- 7.1** | 2.1 |
| `madspin` / seq. global retry | +12.8 | +0.7 | +6.7 +- 7.1 | 0.9 |
| `PA` / sequential | +0.8 | -6.1 | -2.6 +- 7.1 | -0.4 |
| `PA` / seq. global retry | +13.3 | -11.0 | +1.2 +- 7.1 | 0.2 |
| `PA` / seq. with mass | +1.6 | -9.9 | -4.2 +- 7.1 | -0.6 |
| `PA` no-jac / seq. with mass | +41.7 | -22.0 | +9.9 +- 7.2 | 1.4 |
| `madspin_v1` | +9.9 | -4.9 | +2.5 +- 7.1 | 0.4 |
| **`madspin`/joint replica** | +4.5 | -8.2 | **-1.9 +- 7.1** | -0.3 |
| **`PA`/joint replica** | +9.3 | -4.5 | **+2.4 +- 7.1** | 0.3 |

Read the single-resonance columns with care.  `PA` no-jac / seq. with mass is
+41.7 +- 10.1 MeV on `m(t)` alone — 4.1 sigma, and it would look like a real
bias if that were all one looked at.  It is -22.0 MeV on `m(t~)`.  `t` and
`t~` are two independent virtuality draws in the same event, so a scheme that
really biased the lineshape moves both the same way; opposite signs is a
fluctuation, and with 11 comparisons per resonance some of them will reach
3-4 sigma.  Combined, nothing exceeds 2.1 sigma, and the replicas sit at 0.3.

**No unweighting scheme changes the lineshape in any spinmode.**

## The differences that are real, and what each one is

| pair | combined chi2 (138 dof) | mean shift |
|---|---|---|
| `PA` / joint vs `madspin` / joint | **1001.3** | **-218.8 +- 7.1 MeV** |
| `PA` no-jac / joint vs `PA` / joint | 156.0 | **+33.1 +- 7.1 MeV** |
| `madspin_v1` vs `madspin` / joint | 109.9 | +2.5 +- 7.1 MeV |

### 1. The pole approximation, and it is large

`PA` evaluates the production density matrix at on-shell momenta;
`madspin`/`full` evaluates it at the reshuffled off-shell ones
(`_density_pole_approximation`, and the `offshell_density` branch of the
accept/reject).  The difference is exactly the virtuality dependence of the
production matrix element, and it is not small:

| m(t) [GeV] | PA / madspin |
|---|---|
| 152.1 | **1.585 +- 0.135** |
| 155.1 | 1.523 +- 0.112 |
| 158.1 | 1.442 +- 0.088 |
| 173.1 (pole) | 0.995 +- 0.010 |
| 187.9 | 0.735 +- 0.041 |
| 190.9 | **0.696 +- 0.045** |

+59% in the low tail, -30% in the high tail, 0.5% at the pole.  The mean moves
by 219 MeV, 31 sigma.  This is the pole approximation being an approximation,
not a defect — but it is worth knowing that it is a 50% effect on the far
lineshape rather than a percent-level one.

Against the exact Breit-Wigner (drawn on every figure at the `M` and `Gamma`
the run actually used), the two spinmodes sit on opposite sides:

| m(t) [GeV] | madspin / BW | PA / BW |
|---|---|---|
| 152.1 | 0.675 +- 0.045 | 1.070 +- 0.057 |
| 173.1 | 1.009 +- 0.007 | 1.004 +- 0.007 |
| 190.9 | 1.276 +- 0.053 | 0.887 +- 0.044 |
| 193.9 | 1.322 +- 0.063 | 0.927 +- 0.053 |

`PA` is within ~10% of a bare Breit-Wigner everywhere (`chi2` 124.7/69 on
`m(t)` — still 4.8 sigma from it, which is the reshuffling jacobian).  `madspin`
departs from it by -32%/+32%: the off-shell production matrix element suppresses
low virtualities and enhances high ones.  Consistently, the means are
`madspin` 173.165, exact BW 173.002, `PA` 172.950 GeV.

### 2. `density_keep_jacobian = False` shifts the mean by 33 MeV

A coherent +33.1 +- 7.1 MeV (4.7 sigma) with a shape `chi2` of only
156.0/138 (+1.1 sigma) — i.e. at this statistics the effect is visible in the
first moment and not yet resolved bin by bin (the largest single-bin ratio to
`PA`/joint is 1.107 +- 0.076).  This is by design: with the flag off the
reshuffling jacobian is left out of the accept/reject weight and applied
afterwards as a kinematic dressing, so the accepted virtualities are
distributed without it.  Worth recording as the size of that choice.

### 3. `madspin_v1` has the same lineshape as `madspin`, but not the same rate

`chi2` 109.9/138 and a mean shift of +2.5 +- 7.1 MeV: the pre-density
implementation's mass smearing is **statistically indistinguishable** from the
current default's lineshape at 200 000 events.  Its cross section is not:
**24.915137 pb against 23.750990 pb, +4.90%** (BR 0.04939080 vs 0.04708304).
That is the known family split — the legacy path factorises on-shell partial
widths, the `run_onshell` path MC-integrates the partial width including the
off-shell suppression — and it is a rate effect that a unit-area lineshape
figure by construction cannot show, which is why the rates are tabulated
separately.

## Anomalies worth recording

1. **`spinmode = none` gives a different rate from every other cell**:
   23.759797 pb against 23.750990 pb, +0.037%, BR 0.04710049 vs 0.04708304.
   Every other `run_onshell`-family cell agrees to all 8 digits printed.  It is
   tiny and `none` has no accept/reject at all (unweighting efficiency 0.0000),
   so this is most likely a different BR bookkeeping path rather than a
   sampling difference — but it is a real difference and it is not zero.
2. **Per-particle maximum-weight overflows.**  Baseline counts, out of 200 000
   written events:

   | cell | weights above the per-particle bound |
   |---|---|
   | `madspin` / sequential | 2 |
   | `madspin` / seq. global retry | 2 |
   | `PA` / seq. global retry | 3 |
   | every other cell | 0 |

   1.5e-5 of events, and only in the staged schemes — which is where a
   per-particle bound exists at all.  These are exactly the events a per-event
   bound would fix; this is the number to compare against afterwards.  They are
   far too few to be responsible for any of the lineshape numbers above, and
   nothing in the tails is displaced at the level they could cause.

## What this does not cover

- One process (`p p > t t~`), one decay chain, one energy, LO only.  The
  W virtuality was not histogrammed, only the top's.
- `onshell_v1` was not run: it is the fourth non-density spinmode and, like
  `onshell`, keeps the production kinematics, so it has no lineshape either —
  but that was not verified here.
- No polarised production, no `decay_output = weighted`, no `fixed_order`, no
  decay groups (`@` tags).  Each of those forces `joint` and so has no grid to
  compare.
- `sequential_with_mass` under `onshell` was not run (it would fall back to
  `sequential`, and `onshell` has no lineshape).
- `two_stage` was not run: it is a hidden internal scheme
  (`hidden_unweighting_modes`) and is not offered by the card.
- The chi2 treats the bins as independent within a histogram and the two
  samples as independent between histograms.  Both are approximations; the
  replica rows are there precisely so the numbers are read against a measured
  floor rather than against a nominal 1.0.
