# Is a mass-differential `C_kk(m_4l)` figure affordable?

A **feasibility study**, not a finished figure. It asks whether the helicity-sign
correlation between the two `Z`,

```
C_kk = P(++) + P(--) - P(+-) - P(-+) = (4/eta_l^2) <cos th1 cos th2>
```

can be shown **binned in the four-lepton invariant mass**, comparing the
loop-induced `g g` quark box against the `q q~` continuum. The inclusive result
of the two sibling studies is that the two mechanisms have opposite-sign `C_kk`;
whether that develops with `m_4l` is the open question.

The findings, the required statistics and the verdict are in
[RESULTS.md](RESULTS.md). Read that first — the short version is:

> **`sigma(C_kk) = 30.3 / sqrt(N_eff)`.** 200 000 unit-weight events give
> `+-0.136` per bin in four equal-statistics bins, which is enough for the sign
> statement and not enough for the shape. The `gg` events cost about one
> CPU-second each, so 200 000 of them is **3.1 hours on 18 cores**. No `gg`
> event-level data survives anywhere, so that cost is unavoidable rather than
> optional.

## What is here

| | |
|---|---|
| [`run_ckk_mass.py`](run_ckk_mass.py) | generates the two four-lepton samples in seeded 50 000-event batches and harvests them into **per-event columns** |
| [`plot_ckk_mass.py`](plot_ckk_mass.py) | the statistics report, the per-bin numbers and the two-pane figure; runs off `data/` and needs neither MadGraph nor MadSpin |
| `data/` | `ckk_mass.npz` (per-bin values and errors), `numbers.txt`, `plot_meta.json` |
| `plots/` | `ckk_mass.pdf` / `.png` |
| `logs/` | the generation logs |

The `gg` half of the figure is **not** here: the `g g > Z Z` regeneration is
owned by a separate task and this study's own `gg` run was stopped as soon as
that was known, to avoid making the expensive sample twice. Everything in
`plot_ckk_mass.py` already handles both samples; adding the `gg` columns to
`data/events.npz` and re-running it produces the two-curve figure with no
further work.

## The two samples

Neither is decayed by MadSpin. This is a physics figure, so both sides are the
fully off-shell four-lepton matrix element with the two decays correlated by
construction — the same two "truth" processes the sibling studies used:

| tag | process | from |
|---|---|---|
| `gg` | `g g > e+ e- mu+ mu- / a [noborn=QCD]` | [`../zz_loopinduced`](../zz_loopinduced) sample B |
| `qq` | `p p > e+ e- mu+ mu- / a [QCD]` | [`../zz_nlo`](../zz_nlo) truth |

The run cards are **not re-typed**: `run_ckk_mass.py` imports
`RUN_CARD_COMMON` / `RUN_CARD_B_ONLY` from `run_zz_loopinduced.py` and
`RUN_CARD_NLO` / `PT_MIN_PDG` from `run_zz_nlo.py`, together with both studies'
`custom_fcts` cut files, and changes only `nevents` and `iseed`. So these
samples cannot drift away from the published inclusive numbers through an edit
made in one place and not the other.

Cuts, therefore, are the sibling studies': `pt(l+ l-) > 1 GeV` on each
reconstructed pair and `|m(l+ l-) - m_Z| < 15 Gamma_Z` on each pair, fixed
`muR = muF = m_Z`, `nn23lo1`, 13 TeV. Neither cut is a function of a decay
angle, which is what makes the azimuthal integration behind `C_kk` exact rather
than approximate here.

## Per-event columns, and why

`run_ckk_mass.py --stage harvest` writes `(w, m_4l, cos_theta1, cos_theta2,
cos1cos2, pol0_1, pol0_2)` **per event**. That is the entire reason this
directory exists. The two sibling studies committed 1-D histograms, and a 1-D
histogram of `cos th1 cos th2` cannot be re-binned in `m_4l` afterwards — the
joint information is gone at fill time. Their LHE files lived in `/tmp` and have
since been swept, so for the `gg` side the joint information is gone entirely.

The corollary is that per-event output has to live somewhere durable. This
study's `qq` events do:

```
/Users/omattelaer/Documents/git_workspace/zz_spin_events/qq_pp4l_nlo/
    b01..b04.lhe.gz          4 x 50 000 events, 74 MB
    run_card.dat             the card they were made with
    events_percol.npz        the harvested per-event columns, 9 MB
/Users/omattelaer/Documents/git_workspace/zz_spin_events/
    run12_ppzz_lo_madspin.npz    250 000 events, harvested from LTS (see RESULTS)
```

`events_percol.npz` is 9 MB and is deliberately **not** committed; `data/` holds
the per-bin reduction instead.

## Re-running

```
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
python3 run_ckk_mass.py --stage all --basedir /tmp/ckk_work --nb-core 16 \
        --tags qq,gg --qq-batches 4 --gg-batches 4
python3 plot_ckk_mass.py --nbins 4 --edge-sample gg --check-minus
```

`--stage harvest` alone reads whatever batches finished; they are independent
samples and a short set is a smaller sample, not a broken one.

Two things that cost time here and are worth writing down:

* the model has to be imported **explicitly**. `set auto_convert_model T` as the
  first line of a command file with no `import model` after it leaves
  `_curr_model` unset, and `g g > e+ e- mu+ mu- / a [noborn=QCD]` then dies
  inside `loop_interface.validate_model` with
  `AttributeError: 'NoneType' object has no attribute 'get'` — while `mg5_aMC`
  still **exits 0**. `generate_output` therefore checks that `SubProcesses`
  exists rather than trusting the return code. The `[QCD]` sibling process is
  unaffected, so this looks like a difference between the two samples and is
  not;
* `pdftotext` extracts nothing from these usetex PDFs. To check a label, crop
  and look at the PNG.
