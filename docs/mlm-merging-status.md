# MLM merging in madspace — status, findings and open items

Working notes for the `feat-mlm` branch after merging `main` into it.

## What the feature is

`madspace/src/phasespace/mlm_clustering.cpp` compiles the diagram topologies of a
subprocess into a state machine of possible clustering histories, encoded as a
flat integer array. `madspace/src/kernels/mlm.hpp` walks that state machine for
one phase-space point, picks a clustering history by minimising a clustering
measure at each step, and reads off:

* the renormalisation scale,
* the factorisation scales,
* a clustering scale per outgoing jet (written to the LHE as `pt_clust_i`),
* the index of the diagram the history ends on (for colour flows and resonances
  in the LHE output).

`djb_clus`, `dj_clus`, `compute_scale` and `update_momenta` are ports of the
routines of the same name in `Template/NLO/SubProcesses/cluster.f`.

## Merge with main

The branch was 699 commits behind. Most of the conflict surface was not real:

* The generated instruction-set headers (`compgraphs/*.inc`, the cpu/gpu runtime
  mixins, `python/instruction_set.hpp`) are no longer checked in — main
  generates them from `instruction_set.yaml` via `generate_headers.py`. Only the
  two new `mlm_clustering_{hadronic,leptonic}` YAML entries were kept.
* `madevent.py` was split into `launch.py` / `run_interface.py`; the MLM hunks
  were re-applied to `launch.py`. The matrix element is now built in a loop over
  backends, so `me_inputs` is computed once before that loop.
* `run_card.toml` is now a template filled from `banner.py`, so `"mlm"`,
  `beam.jet_radius` and `beam.max_jet_flavor` are declared there. The MLM
  clustering options other than `max_jet_flavor` (`jet_radius`, `scale_scheme`,
  `alphas_reweighting`, `pdf_reweighting`, `clustering_history`, ...) are
  hidden: the template has no line for them, and `RunCardMG7.write` adds one at
  the end of `[beam]` once they are set or differ from their default.
* `LHECompleter` was refactored on main; main's version was taken. The branch's
  `TODO` about the permutation direction is gone because main fixed exactly
  that, by inverting the permutation before using it.

## Bugs found and fixed

1. **`update_momenta` arguments were swapped.** The kernel called it as
   `(..., p1_win, p2_win, ...)` against the signature
   `(..., i_remove, i_keep, ...)`, so it kept `particle2` and removed
   `particle1`. The state machine names the merged pseudo-particle by the lowest
   bit of the combined mask, i.e. by `particle1`, so every later state referred
   to a slot whose momentum had not been updated. It also made the
   `i_keep < 2` initial-state branch unreachable, and inconsistent with the
   caller's `is_initial = particle1 < 2`.

   The Fortran settles it: in `cluster_one_step`, `i` (removed) runs from 3 and
   `j` (kept) runs `1..i-1`, so the kept index is the lower one and `jwin <= 2`
   means "the kept particle is a beam".

2. **The winner selection was not reset between clustering steps.** Only
   `win_scale` was reset; `win_resonant` and `win_next_state` kept their values.
   After a resonant step, no non-resonant candidate could ever win the next
   step, and the walk would then jump to the previous step's target state.

3. **Nothing guaranteed a winner at all.** `win_next_state` starts at `-1` and is
   only assigned when a candidate beats the current best. A NaN measure never
   compares less than anything, so a step in which every candidate was NaN left
   `state = -1` and the kernel read outside the state machine — an actual crash,
   reproducible at a few events per thousand for strongly asymmetric beam
   energies. The boost and rotation `update_momenta` applies after an
   initial-state clustering can leave a pair exactly collinear, which is where
   the NaNs come from.

   Two fixes: `dj_clus` now zeroes a NaN result the way the Fortran does
   ("prevent numerical inaccuracies"), and the selection takes the first
   candidate unconditionally so a clustering is always chosen.

4. **State machine states were expanded more than once.** In `find_clusterings`
   the terminal branch guarded re-expansion with `if (current_states.size() == 0)`
   but the recursive branch did not, so a state reached along several paths had
   its whole transition list appended once per path. Harmless for the result,
   but the table grows exponentially with multiplicity.

5. **`bw_cutoff` was dead.** It was threaded from the run card all the way into
   the kernel signature and never read: the resonance window was `mass ± width`
   rather than `mass ± bw_cutoff * width`.

6. **External masses were indexed by topology slot, not by external leg.** The
   kernel indexes `external_masses` by leg, but the array was built as
   `incoming_masses ++ outgoing_masses`, which is in slot order. Invisible for
   the test processes, whose permutations only exchange identical gluons, but
   wrong for any permutation that moves a mass between legs. The permutation is
   now applied, and a mismatch between diagrams throws instead of being silently
   ignored.

7. **`alive` was hardcoded to `0xFFFFFF`** regardless of the particle count, and
   an empty (dead-end) state would have been read out of bounds. Both now
   handled.

## Vertex metadata, previously not populated

`is_qcd`, `is_jet1`, `is_jet2`, `mass_index` and the `massive_*` flags were all
hardcoded, so massless QCD was assumed everywhere and the resonance branch was
unreachable. They are now filled from the topology:

* masses and widths come from `Topology::Decay`, and from
  `t_propagator_masses()` for the t-channel chain,
* `Topology` gained `t_propagator_pdg_ids()`, filled alongside the existing
  t-channel masses and widths, because the flavour of a t-channel line changes
  along the chain and is needed to tell a QCD splitting from a QED one,
* the **external** pdg ids are not in `Topology` at all, so `MLMClustering`
  takes them as a new constructor argument (`external_pdg_ids`, plus
  `max_jet_flavor`). `launch.py` passes the `all_pids` it already computes. When
  the argument is omitted the old behaviour — everything is a QCD jet — is kept,
  so existing callers still work.

`is_jet` mirrors `isjet()` in `Template/LO/SubProcesses/reweight.f`
(`|pdg| <= max_jet_flavor`, gluons, and the merged-flavour placeholder 81). A
vertex counts as QCD when all three of its lines carry colour, which is the same
rule `launch.py` already uses in `build_multi_channel_data`. A propagator pdg of
0 means "not supplied" and falls back to deciding from the two children.

## Known differences from madevent, by design

These are the open physics questions, not defects. They are what makes a direct
comparison against the validated madevent MLM disagree.

* **Resonances.** MG5 reads the resonance structure off the SDE integration
  channel, which madspace/madnis does not have. This implementation instead
  prefers a resonant clustering over any non-resonant one, using the
  Breit-Wigner window. That is a hard switch, so a phase-space distribution can
  be discontinuous across the window boundary.

* **The factorisation scale.** The kernel uses a single scale for both beams:
  the smallest QCD clustering scale in the history, capped at mu_R. madevent
  tracks the two QCD lines connected to each beam separately and sets
  `mu_F(beam) = sqrt(pt2[jlast(beam)] * pt2[jcentral(beam)])`, so its two
  factorisation scales differ from each other and neither is the softest
  clustering. There is no equivalent of `jfirst`/`jlast`/`jcentral` here: the
  kernel does not trace parton lines through the event the way `ipartupdate`
  does.

* **The renormalisation scale.** The kernel takes the geometric mean of all
  clustering scales, replacing non-QCD ones by the largest scale. madevent takes
  `(pt2[jlast1] * pt2[jcentral1] * pt2[jlast2] * pt2[jcentral2])^(1/8)`, i.e.
  the geometric mean of four specific scales.

* **No Sudakov reweighting**, which is the other half of what the NLO code does.

## The generation-level merging cut: `phasespace.xqcut`

MLM merging needs the matrix element to stop describing radiation the shower is
meant to produce, which madevent enforces with `xqcut`. madspace had no
equivalent, so its samples reached down to whatever the jet cuts allowed and
Pythia's matching veto had to throw the difference away.

The rule is madevent's, from the "Check xqcut for vertices with jet daughters
only" block of `Template/LO/SubProcesses/reweight.f`: walking the clustering
history that was just selected, any step whose daughter is still a bare
final-state jet has to be at or above `xqcut`, or the event is dropped. The
test does not ask whether the vertex was a QCD one, and it reuses the same
"still a bare external leg" bookkeeping the emission scheme already needs.

The clustering only runs for events that already passed the phase-space cuts,
so the veto comes back as a weight (1 or 0) that multiplies the event weight,
rather than as one of the cuts. An event below `xqcut` ends up at weight zero
and is never unweighted.

`p p > t t~` + `t t~ j` at 13 TeV, everything else held fixed:

| `xqcut` | cross section (pb) |
|--------:|-------------------:|
|       0 |          1462 ± 20 |
|      30 |          1181 ± 17 |
|      60 |           815 ± 13 |

It is off by default (`xqcut = 0`), so turning it on is the only thing that can
change an existing result.

## Validating the merging

Differential jet rates reconstructed from a showered event record are what say
whether the merging works: each multiplicity sample has to switch off where the
next switches on, and their sum has to be smooth across the merging scale. That
tooling lives on its own branch (`claude/mlm-djr-plots`,
`madgraph/various/djr_from_hepmc.py`), since it is useful for madevent just as
much as for madspace.

## Open items

* **The trigger for MLM is `dynamical_scale_choice = "mlm"`, which is probably
  the wrong switch and needs revisiting.** In madevent, merging is turned on by
  `ickkw` (with `xqcut`), while `dynamical_scale_choice = -1` selects the CKKW
  clustering scale. Conflating the two means MLM merging cannot be combined with
  another scale choice, and the clustering scale cannot be used without merging.
  A separate `[matching]`-style switch is the likely shape.

* Whether the sqrt(s) fallback should stay a fallback. It is what madevent does
  and it makes the output veto-safe, but a jet at sqrt(s) carries no
  information; if merging is to work well, the fraction of jets landing there
  needs to come down.

* Pass the selected diagram index through to the matrix element. Neither
  backend takes a per-event diagram yet: madmatrix accepts `diagram_in` and
  ignores it, and the mg7 `api.cpp` rejects it as an unsupported input. Feeding
  it `diagram_in` in place of `random_diagram_in` therefore left madmatrix
  drawing its diagram, and with it the colour flow, at a fixed random number of
  0.5 for every event, so MLM keeps `random_diagram_in` until the backends
  support it. Affects the LHE output only, not the scales.

* The `dj_clus` massless/massive test is `mass > 0` here, where the Fortran uses
  thresholds tied to `maxjetflavor` (`m >= 3` if `maxjetflavor > 4`, else
  `m >= 1`). Equivalent for realistic masses; only differs for a tracked mass
  between 0 and 1 GeV.

* Different diagrams can put different propagators on the same partition of
  external legs (a photon and a Z, say). The compiler keeps whichever has a
  width, since that is the one the resonance test is about, but this is an
  arbitrary choice where the diagrams genuinely disagree.

## Comparison against the madevent MLM

Process `p p > e+ ve j` at 13 TeV, 5000 unweighted events each, with the cuts,
the PDF set (`NNPDF23_lo_as_0130_qed`, lhaid 247000) and the collider energy
matched between the two runs.

* mg7: `output mg7`, then `dynamical_scale_choice = "mlm"` in
  `Cards/run_card.toml`.
* madevent: `output madevent`, then `dynamical_scale_choice = -1` (the CKKW
  back-clustering scale) in `Cards/run_card.dat`. A second madevent run adds
  `ickkw = 1` and `xqcut = 20` — neither is written into the run card for a
  fixed-multiplicity process, so they have to be appended by hand — because
  `addmothers.f` only writes the `<scales>` block when `ickkw > 0`.

The hardest-parton pT distributions of the two runs agree closely, so the
differences below are the scale definitions rather than a different population
of phase space.

### Renormalisation scale

|                    | mean | q05 | q50 | q95 |
|--------------------|-----:|----:|----:|----:|
| mg7 MLM            |  62.1 | 40.9 |  54.1 |  92.7 |
| madevent CKKW      | 101.7 | 81.1 |  87.5 | 172.8 |
| mg7 / pT of the hardest parton      | 1.58 | 0.94 | 1.62 | 2.13 |
| madevent / pT of the hardest parton | 2.89 | 1.42 | 2.87 | 4.15 |

mg7's mu_R is about 0.61 of madevent's. This follows from the definitions and
is reproducible in closed form. For this process there are two clustering
steps: the resonant `(e+, nu) -> W`, and the jet clustered onto a beam. The
first is not a QCD splitting, so mg7 replaces its scale by the largest scale in
the history and takes the geometric mean, giving
`mu_R = sqrt(max(m_W, mT_j) * mT_j)`. Recomputing that directly from the LHE
momenta reproduces mg7's `scale` field exactly for 72.8% of events (all of the
`q qbar` ones), the rest being the case discussed below. madevent instead takes
the geometric mean of the four `jfirst`/`jlast`/`jcentral` scales, of which
three are the W here, so its mu_R sits close to m_W.

Cross sections, same cuts and PDF, no merging on either side: mg7
1557.5 +- 7.2 pb against madevent 1616 +- 8.4 pb, a 3.7% difference driven by
the scale.

### Per-jet clustering scales

When both codes assign a real clustering scale to the jet they agree exactly:
`pt_clust / pT_jet` is 1.000 at every quantile in both. Two differences remain.

**mg7 leaves 27.2% of jets with `pt_clust = 0`; madevent leaves none.** The
cause is a topology whose clustering history contains no QCD splitting at all.
For the `g q > e+ ve q` subprocess one of the two channels is the s-channel
`g q -> q* -> (W)(q)`, whose only available clusterings are `(e+, nu) -> W` and
`(W, jet) -> q`. The second has a colourless parent, so it is not a QCD vertex,
and the branch that books a clustering scale onto a jet only runs for QCD
clusterings. In about 36% of `g q` events that history has the smaller measure
and wins, and the jet comes out with no scale. That also explains the remaining
27% of the mu_R comparison above: both clusterings are non-QCD there, so every
scale is replaced by the largest one and mu_R comes out as m_W exactly.

This matters for merging, not just for bookkeeping: `pt_clust` is what the
matching veto compares against `qcut`, and a jet reported at zero is
indistinguishable from a lepton.

**Non-jet legs were written as 0 by mg7 and as sqrt(s) = 13000 by madevent.**
madevent has an explicit fallback (`ptclus(...) = etot` in
`Template/LO/SubProcesses/reweight.f`) that sets any leg without a jet vertex to
the collider energy, so a veto of the form "reject if any pt_clust < qcut" can
never trip on it.

### The sqrt(s) fallback, now implemented

The kernel now does the same: any outgoing leg that no QCD clustering assigned a
scale to is reported at the collider energy rather than at zero. This needed the
collider energy in the kernel, which it did not have, so `MLMClustering` takes it
as a constructor argument (`cm_energy`, filled by `launch.py` from
`process.e_cm`) and it is threaded through as a new instruction input.

Rerunning the same comparison afterwards:

|                                       |   mg7 | madevent |
|---------------------------------------|------:|---------:|
| jets reported at zero                  |  0.0% |     0.0% |
| jets at the sqrt(s) fallback           | 26.4% |    10.1% |
| `pt_clust / pT_jet`, where assigned    | 1.000 |    1.000 |
| median assigned jet scale              | 29.50 |    30.67 |
| non-jet legs                           | 13000 |    13000 |

The convention now matches exactly, and where both assign a real scale they
still agree exactly. madevent reaches the fallback for 10.1% of its own jets, so
the mechanism is shared rather than an mg7 artefact — but mg7 reaches it 2.6
times as often.

### What is still open on this

The fallback makes the LHE output usable by a matching veto; it does not remove
the reason mg7 lands there so much more often, which is the non-QCD clustering
history described above. Two further options, in increasing order of work:

1. Restrict the clustering to histories that do cluster every jet, by rejecting
   candidate transitions that would leave a jet unclustered — a change to the
   state-machine compiler rather than the kernel.
2. Trace the parton lines through the event the way `ipartupdate` does, which is
   also what would be needed for per-beam factorisation scales.

### Which vertex a jet's scale comes from: `beam.jet_scale_scheme`

madevent takes the **maximum** clustering scale over every jet vertex a leg's
*line* takes part in (`ptclus(leg) = max(ptclus(leg), ...)`, with the line
followed by `ipartupdate`), where the kernel originally took the scale of the
**first** clustering the bare leg took part in and froze it there. Both are now
available, selected by `beam.jet_scale_scheme`:

* `"production"` (the default) - the hardest vertex on the parton line the leg
  belongs to, i.e. the scale at which that line was produced. This is what
  madevent does.
* `"emission"` - the vertex at which the leg itself was emitted, i.e. the
  softest clustering it takes part in.

**Why it matters.** `pt_clust_i` is not read by the jet-matching veto at all:
`madevent_interface.py` turns on `Beams:setProductionScalesFromLHEF` for an MLM
run, so Pythia reads it into `event[i].scale()` and uses it as the **starting
scale of the shower off parton i** (with `pTmaxMatch = 1`). The shipped
`JetMatching.h` uses the kinematical `pT()` for its veto, and only FxFx reads
the per-particle scale, as a tag.

That makes `"production"` the motivated choice. It is the CKKW-style
prescription: reconstruct the branching history and restart the shower off each
line at the scale of the node that produced it, letting the veto handle the
overlap. In MLM the Sudakov suppression *is* the vetoed trial emissions, so
starting the shower at the last ME emission scale instead - which is what
`"emission"` does for a line that carries on - never attempts them, and the
relative normalisation between the n-jet samples comes out wrong.

**What it took.** `"emission"` needs no knowledge of the event beyond the leg
indices; `"production"` needs the parton line followed through the clustering,
which is what `ipartupdate` exists for. The compiler now works out, per
transition and from the colour representations of the mother and the two
daughters, which daughter the line continues into (follow the first, the second,
the harder one, or both after a `g -> q qbar`), and stores it as a third word in
the state machine. The kernel keeps the resulting representative legs per slot,
the equivalent of `ipart`, and compares transverse momenta in the *original*
event when it has to pick the harder daughter, as the Fortran does. Colour
representations come from the `pdg_color_types` map already exported in the
subprocess metadata, with a Standard Model fallback.

Where madevent's `ipartupdate` keys its leading cases on pdg equality
(`idmo.eq.idda1`), the colour-based table gives the same answer for every case
those cover. Where `ipartupdate` stops the run on an unrecognised vertex, the
compiler keeps the line on the first daughter: wrong, but local, and it costs
the clustering scale of one leg rather than the whole run.

The two schemes coincide whenever no parton line carries on past the vertex that
emitted it, which is every leg of a 2 -> 3 process - and is why the earlier
`p p > e+ ve j` comparison could not have seen the difference.
