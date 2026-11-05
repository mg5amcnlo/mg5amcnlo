# MadSpin polarisation weights against a polarised NLO+PS sample (`p p > z z`)

Validation of the two ways of getting a polarised observable out of MadGraph at
NLO+PS, on the same process, the same frame and the same generator revision:

- **A — polarised matrix element.** `p p > z{0} z{0} [QCD]`, `me_frame = [3,4]`
  (the ZZ rest frame, `FRAME_ID = 24`), decayed by MadSpin.
- **B — unpolarised matrix element + MadSpin polarisation weights.**
  `p p > z z [QCD]`, decayed by MadSpin with
  `set keep_weight_for_polarization_vector [0, +, -, T]`; the sample is then
  reweighted by `ms_pol_23:0_23:0`.

Both showered with Pythia8 and **stopped after the parton shower**; every plot
is made at that level.

## Setup

| | |
|---|---|
| revision | `6b1f93030` (branch of PR #65, with the MadSpin NLO-frame fix below) |
| beams / PDF | 13 TeV, `nn23lo1` |
| scales | fixed, `muR = muF = 182.376` (2 mZ) |
| accuracy | `req_acc = 0.005`, 100 000 events per sample |
| seeds | A 6001, B 6002 (MadSpin 6002 / 6003) |
| MadSpin | `spinmode = madspin`, `BW_cut = 15`, one decay line `z > lpz lmz` with `lpz = e+ mu+` |
| shower | Pythia8, hadronisation OFF, MPI OFF, QED shower ON, shower PDF = NLO PDF |
| analysis | e+e-mu+mu- events; leptons dressed with photons within dR < 0.1 |

Frame convention: the ZZ rest frame, `cos(theta*)` of the negative lepton of
each Z in that Z's rest frame reached by a pure boost from the ZZ frame, with
respect to the Z flight direction — identical to `zz_pol_run` so the numbers can
be compared with that study.

`[3,4]` was chosen over `[3]` deliberately: with two selected legs no leg is at
rest, so the two tools cannot disagree about the quantisation axis. (For a
one-leg frame, HELAS pins the axis of the at-rest particle to the frame z axis,
and the generator boosts from the partonic c.m. while MadSpin boosts from the
lab; the two differ by a Wigner rotation. With `[3,4]` that difference is a pure
rotation, under which a fixed-helicity |M|^2 is invariant.)

## A prerequisite: MadSpin ignored the frame of an NLO sample

`frame_id` was read from the production run_card **only for LO run_cards**; any
NLO sample was forced to `frame_id = 6` (the partonic c.m.). `RunCardNLO` has
carried `me_frame` since the polarised NLO work, so sample A would have been
decayed with its polarisation projected in a different frame from the one it was
generated in, silently. Fixed in `6b1f93030`
(`frame_and_beampol_from_run_card`); verified on A's own production banner:

    production run_card: RunCardNLO, me_frame=[3, 4] -> MadSpin frame_id=24

## Checks that had to pass before any comparison

| check | result |
|---|---|
| generated cross-sections vs `zz_pol_run` | A 0.7376 +- 0.0037 pb (ref 0.7368); B 12.35 +- 0.05 pb (ref 12.361) |
| `FRAME_ID` read back from `Source/run_card.inc` | 24 (A), 0 (B, unpolarised) |
| MadSpin | 100 000 events written per sample, 8.6 / 7.2 trials per event |
| LHE and HepMC hold the same events in the same order | 100 000 = 100 000; weight ratio constant to 2e-16 |
| MadSpin `(0,0)` weight on the already-polarised sample | equals the nominal weight exactly (max deviation 0) |

The last two matter: the HepMC written by aMC@NLO's Pythia8 driver carries a
single weight, so the polarisation weights are taken from the decayed LHE and
the alignment of the two files has to be proven rather than assumed.

## Result

e+e-mu+mu- selected: 49 885 (A), 50 100 (B).

    sigma(selected), A polarised ME          1.66257e-03 +- 8.8e-06 pb
    sigma(selected), B x ms_pol_23:0_23:0    1.65094e-03 +- 1.6e-05 pb
    ratio B/A                                0.9930 +- 0.0107   (0.65 sigma from 1)

    f00 from the MadSpin weights             0.05908
    f00 from the generated cross-sections    0.05972
    f00, zz_pol_run fixed-order NLO          0.05937

Shape comparison, chi2/ndf, **parton-shower level**:

| observable | chi2/ndf |
|---|---|
| cos(theta*) (both Z) | 12.9 / 19 |
| Delta(phi) between decay planes | 18.1 / 19 |
| pT(ZZ) | 12.0 / 9 |
| m(ZZ) | 13.0 / 8 |
| pT(Z) (both Z) | 16.9 / 7 |

**MadSpin's polarisation weights reproduce the polarised matrix element**, in
rate to 0.7% +- 1.1% and in every distribution at the shower level.

Consistent with `zz_pol_run`, which found MadSpin's (tree-level) density lands
~0.6-0.7 of the way from the LO to the NLO polarisation fractions, i.e. an
expected B/A slightly below 1: it measured 0.05830/0.05937 = 0.982, and 0.9930
+- 0.0107 here is 1.0 sigma from that.

## The one thing that looks like a disagreement and is not

At **LHE** level the same comparison gives chi2/ndf = 67.5/9 for pT(ZZ) and
35.3/7 for pT(Z), with a deficit that fills in monotonically with pT:

    pT(ZZ) bin [GeV]   10-15  15-20  20-30  30-40  40-60  60-80  80-120
    B/A (LHE)          0.21   0.48   0.68   0.82   0.89   0.99   1.04

This is the MC@NLO S/H structure, not physics. At LHE level 77-80% of events sit
at pT(ZZ) < 1 GeV (the Born-like S events) and the first few pT bins are
dominated by cancellation between large opposite-sign weights — sum|w| / |sum w|
reaches **70.9** (A, 5-10 GeV) and 12.3 (B, 10-15 GeV). After the shower, which
is what fills pT(ZZ) in MC@NLO, that ratio is a uniform 1.1-1.2 in every bin and
the comparison is flat. LHE-level recoil distributions of an MC@NLO sample are
not physical predictions; only the showered ones are.

## Environment note (not a code change)

The Pythia8 shower would not link: `cpp_compiler` unset makes `make_opts` use the
**C** driver `clang`, and `MCatNLO/srcPythia8/Makefile`'s `Pythia83` rule adds no
`$(STDLIB)`, so libc++ is never linked and the build dies with undefined
`std::__1::...` symbols while aMC@NLO still exits 0. Worked around per run with
`cpp_compiler = clang++`; the missing `$(STDLIB)` in the link rule is a real bug
and is **not** fixed here.

## Limitations

- One seed per sample. The quoted errors are the Monte Carlo weight errors and
  assume independent bins, which is optimistic for an NLO sample where S and H
  events are correlated.
- Only the `(0,0)` combination was compared against a polarised matrix element.
  The other 15 weights were written but not validated against their own
  polarised samples.
- `me_frame = [3,4]` only. A single-leg frame (`[3]`) is the configuration where
  the generator and MadSpin could genuinely disagree about the axis, and it is
  untested.
