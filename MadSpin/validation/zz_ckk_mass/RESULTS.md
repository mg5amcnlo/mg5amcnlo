# `C_kk(m_4l)`: what it costs, and what it buys

Feasibility verdict on a **mass-differential** helicity-sign correlation figure
for `g g -> ZZ` against `q q~ -> ZZ`. The brief asked for the statistics
estimate first and the figure only if the estimate clears the bar.

**Verdict: figure after generation, and the generation is not optional.**
The number the sibling regeneration task needs is

> **`sigma(C_kk) = 30.3 / sqrt(N_eff)` — so 200 000 unit-weight `gg` events give
> `+-0.136` per bin in four equal-statistics bins, at a cost of about
> 3.1 hours on 18 cores.**

That is enough to show that the two mechanisms sit on opposite sides of zero *in
every mass bin*. It is **not** enough to measure the shape of `C_kk(m_4l)`:
resolving a `+-0.10` bin needs 92 000 events per bin, i.e. 370 000 events for
four bins and 5.7 hours. The companion `f_0(m_4l)` costs nothing and is a
far stronger differential observable — 18 sigma of variation on the statistics
that give `C_kk` two.

---

## 1. The statistics, from the real per-event spread

The whole difficulty is the calibration `C_kk = (4/eta_l^2) <cos th1 cos th2> =
83.154 <cos th1 cos th2>`. It multiplies the **error** as well as the value, so
what matters is the per-event spread of the product `cos th1 cos th2`. Measured
on three independent samples, all with the **post-fix** `observables.py` (see
§2):

| sample | `N` | `N_eff` | `sd(c1 c2)` | `k = 83.154 sd` |
|---|---|---|---|---|
| `p p > e+ e- mu+ mu- / a [QCD]`, this study | 200 000 | 145 175 | 0.3637 | **30.24** |
| `p p > z z` LO + MadSpin (`run_12`, 250 000) | 250 000 | 250 000 | 0.3665 | **30.47** |
| `g g > 4l` loop induced, pre-fix histogram | 50 000 | 50 000 | 0.3802 | 31.62 |

So

```
sigma(C_kk) = k / sqrt(N_eff),   k = 30.3
```

and `k` is **flat across production mechanism, order, and the frame fix**, which
it has to be: the spread of `c1 c2` is set by the near-flatness of each
`cos theta` distribution, not by the correlation between them. The brief's
`31.4` estimate was right to within 4 %.

**Events needed per bin.**

| target `sigma(C_kk)` | per bin | 4 bins, total | `gg` wall on 18 cores |
|---|---|---|---|
| `+-0.30` | 10 200 | 41 000 | 0.6 h |
| `+-0.20` | 23 000 | 92 000 | 1.4 h |
| `+-0.15` | 41 000 | 163 000 | 2.5 h |
| `+-0.136` | 50 000 | 200 000 | **3.1 h** |
| `+-0.10` | 92 000 | 367 000 | 5.7 h |
| `+-0.05` | 367 000 | 1 469 000 | 22.7 h |

The wall column uses this study's own measured `gg` cost — `../zz_loopinduced`
generated 50 000 events of `g g > e+ e- mu+ mu- / a [noborn=QCD]` in **54 min on
18 cores, 13h32m CPU**, i.e. **0.97 CPU-seconds per event** — and it is the
loop-induced four-lepton sample that dominates everything. Its `2 -> 2`
counterpart `g g > z z` is 1 min 34 s for the same 50 000 events, 300 times
cheaper per event, and the `q q~` four-lepton side is 88 s per 50 000 on 14
cores. **The `gg` box is the only expensive thing in this study**, exactly as
the brief suspected.

### Two corrections to the arithmetic, both of which matter

**(a) Negative weights cost 38 % more events on the `qq` side.** The MC@NLO
sample harvested here has `N = 200 000` but `N_eff = 145 175`, a ratio of
`0.726`. `N` in the table above is `N_eff`, so an NLO `qq` sample has to be
**1.38 times larger** than a unit-weight one for the same bar. The `gg`
loop-induced sample is unit weight and needs no such factor. All errors here use
`sum w^2 (x - <x>)^2 / (sum w)^2`, which reduces to the ordinary error of the
mean on a unit-weight sample and is correct on a signed one.

**(b) Equal-statistics binning is not free of a coverage decision.** The `m_4l`
spectrum falls steeply, so the top quantile bin runs from ~320 GeV to ~4.4 TeV
while holding a quarter of the rate within a few tens of GeV of its lower edge.
The figure uses a **log** `x` axis and draws each marker at the bin's own
weighted median, with the `x` bar spanning the whole bin; nothing is re-binned
to improve the picture. On a linear axis the four bins collapse into the first
tenth of the pane, which is what the first draft of this figure looked like.

---

## 2. The frame fix, and why every earlier coefficient moved

Everything above and below was computed **after** merging `claude/ms-zz-nlo` at
`0d62a68c`, which fixes a Wigner rotation in `observables.compute`: the helicity
axis was taken in the four-lepton frame while the `l+` was boosted into its
pair's rest frame straight from the lab, and the composition of those two
non-collinear boosts tilted the analysing frame by a median 8 degrees. The
observable was not Lorentz invariant and every rank-2 moment was damped towards
isotropy. `observables.py` now carries an import-time self-test that the
pre-fix version fails immediately.

The brief's inclusive numbers — `gg C_kk = +0.570 +- 0.141`,
`qq~ C_kk = -0.675 +- 0.131` — are **pre-fix and damped towards zero**. This
study reproduces the post-fix `run_12` reference exactly
(`C_kk = -0.758 +- 0.061`, `f_0 = 0.1737 +- 0.0022` on 250 000 events), which is
the check that the machinery here is on the fixed module.

**A new post-fix inclusive number, on the sample that matters.** On 200 000
events of the fully off-shell `p p > e+ e- mu+ mu- / a [QCD]` truth, with the
sibling studies' own cuts and mass window:

```
q q~ continuum, NLO, off shell   C_kk = -0.645 +- 0.080     f_0 = +0.1885 +- 0.0029
```

This is **not** the same number as `run_12`'s `-0.758 +- 0.061`, and the
difference is physics rather than a discrepancy: `run_12` is on-shell `p p > z z`
production, so it has no events at all below `2 m_Z`, and §4 shows that the
region below 200 GeV is exactly where `C_kk` goes to zero. Restricted to
`m_4l > 200 GeV` the two agree: `-0.737 +- 0.087` here against
`-0.791 +- 0.067` for `run_12`, a 0.5 sigma difference on samples that also
differ in order, in off-shellness and in whether any cut was applied at all.

No post-fix `gg` number exists yet, here or anywhere.

---

## 3. The data inventory, which is the part that forces the generation

Reported as a fact regardless of the verdict, because it is useful on its own.

**Gone, and not recoverable.**

* `/tmp/t75run/` — the whole `zz_loopinduced` working tree. The directory
  skeleton survives; **every regular file is missing** (`du` reports 0 B on all
  19 subdirectories). Both `gg4l` (the loop-induced four-lepton truth) and
  `ggzz` (the `2 -> 2` production sample MadSpin decayed) are empty.
* `/tmp/zz_nlo_work/` and `/tmp/zz_nlo_work2/` — same, including
  `ggzz_li` and `pp4l_nlo`.
* Around a dozen further gutted `g g > z z` trees under other scratch
  directories.

**A search of every `.lhe*` on the machine (~8700 files) finds no
`g g > 4 leptons` sample anywhere, and no proc card that ever generated one
outside the swept trees.** The only surviving loop-induced `g g > z z` event
files are four 20 000-event runs of `g g > z{0} z{0} [noborn=QCD]` —
*longitudinally polarised, undecayed* `Z` — which carry no decay angles and
cannot give `C_kk`.

**Still on disk, and usable.**

* `LTS/PROCNLO_loop_sm_7/Events/run_12_decayed_1/events.lhe.gz` — the brief's
  `run_12`. 250 000 events, `p p > z z` at `order = LO`, `spinmode = madspin`,
  `BW_cut = 15`, **no cuts**. It is per-event data with `m_4l`, so it *is*
  usable for a mass-differential study, with two caveats that both matter:
  production is **on shell**, so it is empty below `2 m_Z` and blind to the one
  place `C_kk` does something; and it has no `pt` cut or mass window, so it is
  not bin-for-bin comparable with the cut samples. Harvested and used here as a
  cross-check, not as a curve.
* The same directory holds nine further 250 000-event `p p > z z [QCD]` runs,
  most with a MadSpin-decayed partner. They are the LTS test suite's, not this
  study's, but they are large `qq~`-side per-event data.
* This study's own new `qq` sample, 200 000 events, now stored durably (paths in
  [README.md](README.md)).

**The consequence for the verdict.** A differential-in-`m_4l` coefficient needs
per-event `(m_4l, cos th1, cos th2)`. For the `gg` side that data does not exist
in any form, so the generation cost in §1 is **unavoidable, not optional**. The
sibling regeneration task owns that run; this study's own `gg` generation was
stopped mid-refine as soon as the overlap was known.

### The cheap exact route is blocked, and it is worth knowing why

`C_kk` is a property of the production spin density matrix, and if the diagonal
populations `P(lambda1, lambda2)` could be read off polarised cross sections
directly there would be no `eta_l` dilution at all — the `1/83` penalty is a
property of *measuring* `C_kk` through lepton angles, not of the coefficient.
Four polarised `2 -> 2` runs would then cost about 1/300 of one four-lepton run
per event and carry an error like `1/sqrt(N)` instead of `30/sqrt(N)`.

It does not work on this branch. `madgraph_interface.check_process_format`
refuses a polarisation restriction on a massive particle inside any `[...]`
process:

```
generate g g > z{+} z{+} [noborn=QCD]
InvalidCmd: Polarization restriction can not be used for massive particles
```

The restriction is in an explicit `check(p)` on `p.get('mass') != 'ZERO'`,
applied after `noborn`/`sqrvirt` have already been accepted. (Some other branch
evidently relaxes it — the four surviving `g g > z{0} z{0} [noborn=QCD]` runs
above exist — but `{0}` and `{T}` cannot separate `++` from `+-` and so give
`f_0` and `f_00`, never `C_kk`. Splitting `{+}` from `{-}` is what `C_kk` needs.)
If that restriction were lifted for helicity eigenstates, this entire feasibility
question would go away.

---

## 4. What the differential coefficient actually looks like, on the half that exists

The `gg` curve is not this study's to make, but the `qq` half is finished, and it
answers the "is there anything there?" part of the question.

**Four equal-statistics bins**, 200 000 events, `N_eff/bin ~ 36 300`:

| `m_4l` [GeV] | `N` | `N_eff` | `C_kk` | `f_0` |
|---|---|---|---|---|
| 135 - 212 | 49 257 | 36 840 | `-0.145 +- 0.149` | `+0.2950 +- 0.0055` |
| 212 - 249 | 49 084 | 36 972 | `-0.929 +- 0.155` | `+0.2238 +- 0.0056` |
| 249 - 323 | 49 729 | 36 490 | `-0.731 +- 0.163` | `+0.1456 +- 0.0057` |
| 323 - 4382 | 51 930 | 34 946 | `-0.777 +- 0.171` | `+0.0895 +- 0.0059` |
| inclusive | 200 000 | 145 175 | `-0.645 +- 0.080` | `+0.1885 +- 0.0029` |

Against a constant this is `chi2/ndf = 15.5/3`, i.e. **`C_kk` does depend on
`m_4l`, at about 3.2 sigma** — and the whole of the effect is at threshold.
Finer bins localise it:

```
182 - 200 GeV   C_kk = -0.078 +- 0.204
200 - 220 GeV   C_kk = -0.660 +- 0.184
220 - 250 GeV   C_kk = -0.776 +- 0.177
250 - 300 GeV   C_kk = -0.713 +- 0.186
300 - 400 GeV   C_kk = -0.603 +- 0.203
400 GeV +       C_kk = -0.981 +- 0.237
```

Below `2 m_Z` the correlation is consistent with **zero**; above ~220 GeV it is
flat at about `-0.75`. `run_12` cannot see this — it is on-shell and starts at
182 GeV — and in the bins the two do share they agree to 2 sigma or better.

So the honest reading, and it is a correction to the brief's framing: **the
interesting differential structure is a threshold effect, not a high-mass
evolution.** Above threshold `C_kk` is flat to within the errors at any
statistics anyone is going to generate. The bins that decide the figure are the
low-mass ones, which the equal-statistics binning already puts first, and which
an *on-shell* production sample cannot populate at all.

### `f_0(m_4l)` is the figure that actually works

The same events give `f_0` falling from `0.295` to `0.090` across the four bins
— a spread of `0.20` against a per-bin error of `0.0057`, so about **18 sigma of
variation** on statistics that give `C_kk` barely two. `f_0` carries no `eta_l`
anywhere (it is a rank-2 moment; `eta_l` multiplies only `P1`), and that single
fact is the whole difference. Against a constant, `chi2 = 361` for 3 degrees of
freedom.

If only one differential spin figure is going into the paper, it should be this
one. `C_kk(m_4l)` is the interesting object and the affordable version of it is
a two-point statement — positive for `gg`, negative for `qq~`, in each of four
bins — not a curve.

---

## 5. What is committed, and what is left to do

`plots/ckk_mass.pdf` / `.png` is the **`qq`-only** figure: the machinery,
running, on real post-fix events, with the `gg` slot empty. It is a
demonstration of the layout and the binning, not the physics figure. Adding a
`gg/*` block to `data/events.npz` and re-running

```
python3 plot_ckk_mass.py --nbins 4 --edge-sample gg --check-minus
```

produces the two-curve figure. `--edge-sample gg` matters: the bin edges should
be the quantiles of whichever sample is statistics-limited, and that is the
`gg` one.

Recommended sizing for whoever generates the `gg` sample:

* **200 000 events** for the sign statement at `+-0.136` per bin in four bins.
  Anything less and the `gg` points stop being individually resolved from zero;
* **370 000** if the shape is wanted at `+-0.10`, but §4 says the shape above
  threshold is flat, so the extra 2.6 hours mostly buys precision on a constant;
* generate down to the same `|m_ll - m_Z| < 15 Gamma_Z` window rather than
  on shell — the only bin with anything in it is the one below `2 m_Z`, and an
  on-shell sample has no events there;
* write the LHE somewhere that is not `/tmp`, and harvest **per-event columns**.
  Both sibling studies committed 1-D histograms and both are now unrepeatable
  without regenerating.

## 6. Wall times

| step | wall | notes |
|---|---|---|
| `output` of `p p > e+ e- mu+ mu- / a [QCD]` | 191 s | includes the MadLoop build |
| `output` of `g g > e+ e- mu+ mu- / a [noborn=QCD]` | 41 s | CutTools/IREGI already built |
| `qq`, 4 x 50 000 events | 115 + 98 + 100 + 96 s | 16 cores |
| harvest, 200 000 events | 100 s | one pass, per-event columns |
| `gg`, 50 000 events | (54 min, from `../zz_loopinduced`) | 18 cores, 13h32m CPU |
