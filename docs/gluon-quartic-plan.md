# Pure-gluon amplitude optimisation — plan

Branch `claude/gluon-amplitude-optimization-8706f5`. Everything is behind
`set merge_quartic_vertices` (off by default), so nothing changes until it is
set. It takes four values:

| | |
|---|---|
| `False` | off, the default |
| `speed` | the current sums, and the diagram order which allows them |
| `slots` | the order which keeps fewest currents alive; no current sums |
| `auto` | `slots` when the matrix elements go to a gpu backend, `speed` otherwise, decided per output |

It has to be a `set` option and not an `output` one, because the diagram order
is fixed while the diagrams are generated -- by `output` time it is already
too late. The interface pushes it onto `madgraph.merge_quartic_vertices`,
which is what the generation and the exporters read.

`auto` is the exception, and only because `slots` is the `speed` order
reversed: it generates the `speed` order and
`MadGraphCmd.apply_quartic_diagram_order` reverses it at `output` time, once
the backend about to be handed the matrix elements is known. The choice is
made from the *matrix element* exporter rather than the output format, so
`output madevent --me_exporter=<gpu backend>` -- one output feeding two
backends off a single `_curr_matrix_elements` -- follows the gpu.

**speed against slots.** The wavefunction store is a stack frame on cpu and is
per thread on gpu, where at seven gluons it is about 24 kB a thread and caps
occupancy. So the trade goes opposite ways:

| `g g > 5 g` | amplitude calls | slots | per thread |
|---|---|---|---|
| off | 7245 | 268 | 25.1 kB |
| `speed` | 6813 | 259 | 24.3 kB |
| `slots` | 7245 | **199** | **18.7 kB** |

6% more arithmetic for 23% less memory. Measured on cpu `speed` wins; which
one a gpu wants has *not* been measured -- there is no device here.

## Goal

Each colour structure of the 4-gluon vertex carries the *same colour factor*
as the diagram obtained by splitting that vertex into two cubic ones. So the
quartic current and the cubic current carrying that colour factor can be
summed into one current, and the whole subtree below it emitted once instead
of twice — a Berends-Giele style recursion, without colour ordering.

```
TMP = VVVVk_1(W_a,W_b,W_c) + VVV1P0_1(VVV1P0_1(W_a,W_b), W_c)
```
Both carry the same `1/P^2` through the same propagator, so this is a plain
sum. Same for amplitudes.

## Established facts (measured, not assumed)

**1. The colour identity holds exactly.** Grouping every amplitude piece by
its colour vector over the ColorBasis:

| process | amplitude pieces | colour groups | cubic diagrams per group |
|---|---|---|---|
| `g g > g g`     | 6   | 3   | exactly 1 |
| `g g > g g g`   | 45  | 15  | exactly 1 |
| `g g > g g g g` | 510 | 105 | exactly 1 |

Every quartic piece is colour-proportional (±1) to exactly one cubic diagram.
Merged count == pure-cubic diagram count == `(2n-5)!!`. `g g > 4g` also
verified in generated Fortran: folding 405 of 510 amplitudes and zeroing the
sources leaves `|M|^2` bit-identical.

**2. The seed rule — forbid two 3-gluon vertices from sharing a line.**

| process | full | seed | seed % | reconstructed by unrolling | missing |
|---|---|---|---|---|---|
| `g g > g g`       | 4    | 1   | 25%   | 4    | 0 |
| `g g > g g g`     | 25   | 10  | 40%   | 25   | 0 |
| `g g > g g g g`   | 220  | 55  | 25%   | 220  | 0 |
| `g g > g g g g g` | 2485 | 385 | 15.5% | 2485 | 0 |

This is forced, not tuned. Unrolling a quartic vertex always yields two
**adjacent** cubic vertices (joined by the line that replaced the vertex), so
a diagram is reachable iff it has an adjacent cubic pair to contract back.
The diagrams with no such pair are exactly the ones that must be in the seed,
and every diagram either has such a pair or is in the seed — necessary and
sufficient, hence exact coverage.

Note "no 3-gluon vertices **at all**" is the over-restrictive special case: it
loses the diagrams whose cubic vertices are non-adjacent (60 of 220 at six
gluons).

**3. Reconstruction gives matched rootings for free.** A reconstructed diagram
and its quartic partner come from the *same* seed diagram, so they share a
decomposition and every current except the one being summed. This is the
property the whole optimisation needs.

## Already committed

| commit | what |
|---|---|
| `fcd8218b6` | link map (`unroll_quartic_vertices`, verified vs colour algebra, N=2..5) + amplitude sums |
| `5c9cdb301` | why the current sums fail on the un-rewired DAG |
| `1c1722ae8` | revert of the auxiliary-particle generation |
| `98d288f41` | `reroot_diagram`, validated 755/755 — probably NOT needed under the seed rule |
| `25fc6d1cc` | step 1, the seed rule inside `reduce_leglist` |
| `7da06bac7` | step 2+3, `expand_seed_diagrams` and the recorded links |
| `8e634cf9a` | step 4, `TMP = W1 + c*W4` at the amplitudes |

Useful pieces to keep: `get_unrollable_quartic_vertices`,
`unroll_quartic_vertices`, `diagram_colour_signature`, `UnrollDiagramTag`,
`split_quartic_vertex`, `unrolled_diagram`, `get_quartic_amplitude_merges`,
`get_amplitude_merge_lines`.

## Plan

**Step 1 — enforce the seed rule in generation.** DONE, `25fc6d1cc`. Rejected
inside `reduce_leglist`, tracking by leg number which lines a cubic vertex
produced. Measured against the full generation filtered by an independently
written adjacency detector, comparing the diagrams and not only the counts:
1 / 10 / 55 / 385, and 4165 of 34300 at eight gluons.

The closing vertex needs two cases, not one. A real n->0 interaction takes
the legs left over as lines coming in, but a 2->2 like reduction ends on the
*identity* vertex, which only states that its two legs are the two ends of
one line — and that line joins the two vertices which produced them. Missing
that left `g g > g g` with all four of its diagrams.

**Step 2 — reconstruct the full set by unrolling the seed.** DONE,
`1b9474f69`, `expand_seed_diagrams`. Diagram count exactly the baseline
(4/25/220/2485), same diagrams by tag, `|M|^2` bit-identical at four and five
gluons and to 1e-14 at six and seven.

The dedup has to run on the *glued* form. While the identity vertex is still
there the same diagram has several spellings, and a quartic vertex sitting
just in front of it is not yet the last one — which is what decides how ALOHA
indexes its colour structures. Without gluing first, the dedup caught nothing
(40 diagrams for `g g > g g g`) and 8 of 30 recorded colour chains disagreed
with the colour algebra.

**Step 3 — record the link during reconstruction.** DONE, same commit,
`get_quartic_unroll_links`. 3 / 30 / 405 / 6300 links, every one agreeing
target for target with `unroll_quartic_vertices`, which stays as the
independent colour-algebra cross-check.

**Step 4 — the current sum.** DONE at the amplitudes, `8e634cf9a`, and
blocked deeper. `TMP = W1 + c*W4` where a quartic current and its cubic
partner feed the same vertex; the amplitude reads the sum and the quartic
amplitude is never computed:

```
W(20) = W(19)
W(20)%W(:) = W(19)%W(:) + W(12)%W(:)
CALL VVV1_0(W(4),W(5),W(20),GC_10,AMP(33))
```

The sum is shared by every amplitude reading it, so it pays for as many calls
as it has users: 60 amplitude calls for 30 sums at six gluons, 432 for 60 at
seven. Substituting several mothers of one amplitude also produces the
amplitude with all of them substituted, so every subset has to be a merge into
that same target weighing the product of the coefficients — that check is what
keeps the count honest.

Two things it costs. `reuse_outdated_wavefunctions` works out when a slot is
free from the diagrams alone and cannot see the extra read, so it has to be
told, or the two currents get handed the same slot (`W(11) + W(11)`). And the
sums take a slot each, never reused: `NWAVEFUNCS` 51 -> 91 at six gluons.

| | `g g > g g g` | `g g > g g g g` | `g g > 5 g` |
|---|---|---|---|
| helas calls | 94 -> 93 + 7 sums | 637 -> 612 + 30 | 8159 -> 7784 + 60 |
| JAMP lines | 131 -> 101 | 1082 -> 688 | 23672 -> 7864 |
| per call | 34.88 -> 35.12 s | 47.77 -> 46.02 s | 42.16 -> 39.80 s |

**The sum stays at the amplitudes.** Measured
on the reconstructed matrix element (`g g > g g g g`): of the 275 places where
a cubic current is fed by another cubic current *and* the quartic partner
taking the same four lines exists, 225 are the last vertex — the amplitude
sum, which pitfall 7 says buys nothing — and 50 are genuine currents. **None
of the 50 pass the 1:1 consumer test.** At seven gluons, none of 135. Only at
five gluons do all 7 pass.

Why, from a failing pair: quartic current 30 has consumers
`{Wav(3,30), Amp(3,4,6,30)x3, Amp(6,8,30), Amp(4,10,30)}` while its cubic
partner 53 has `{Amp(6,8,53), Amp(4,10,53), Amp(3,12,53)}`. Two correspond;
the rest do not, because the same diagram is rooted differently on the two
sides.

That is forced, not a bug in the reconstruction. Expanding the seed produces
more (seed, choice) instances than there are diagrams, so some diagrams are
reached from several seeds:

| process | seeds | (seed, choice) instances | diagrams |
|---|---|---|---|
| `g g > g g`     | 1   | 4    | 4    |
| `g g > g g g`   | 10  | 40   | 25   |
| `g g > g g g g` | 55  | 340  | 220  |
| `g g > 5 g`     | 385 | 4900 | 2485 |

A diagram carries one decomposition, so it can be rooted to match at most one
of its quartic partners, while the current sum needs the match at *every*
node. Fact 3 above holds per (seed, choice) and breaks under the dedup that
fact 2 requires. Not fixable by choosing the spelling more cleverly: the
counting alone rules it out, and `g g > g g` — the one row with no collision
— is also the one process where the seed rule reaches every partner.

That is why the sum is taken only where the vertex reading it is an
*amplitude*: there is nothing above it, so no consumer can be handed a term
it must not have and the substitution is a plain swap of one argument. A sum
at a current would have to be carried through every current above it, and
those are shared. What would work there is a partial rewrite — hand the sum
only to the consumers which do correspond and leave W1 and W4 serving the
rest — which splits shared consumers and cascades upward. That is a DAG
rewriting problem, not this plan.

The same counting is what leaves the two-substitution cases at seven gluons
on the table: their target has a source with both mothers substituted, but
the source with only the *other* one substituted is spelled with a different
rooting, so the subset check refuses it. 432 of the 864 amplitude calls the
structure allows — and the rest are not simply waiting to be picked up, see
"where to go next".

**Step 5 — validate and time.** DONE. `|M|^2` for `g g > N g`, N=2..5, and
per-call timing from the shipped `check` driver, which already loops
`SMATRIX` when given a second argument (`./check 1000 20000`).

| | `g g > g g g g` | `g g > 5 g` |
|---|---|---|
| flag off | 47.73 / 47.77 / 47.80 s | 42.15 / 42.16 s |
| flag on, before steps 1-4 | 47.76 s | 40.86 s |
| flag on, steps 1-3 only | 48.03 / 48.07 s | 40.41 s |
| flag on, at HEAD | 46.14 / 45.90 s | 39.80 s |

and the code that produces it:

| | helas calls | JAMP lines |
|---|---|---|
| flag off | 637 / 8159 | 1082 / 23672 |
| flag on, before steps 1-4 | 637 / 8159 | 688 / 8012 |
| flag on, steps 1-3 only | 672 / 8216 | 688 / 7864 |
| flag on, at HEAD | 612 + 30 / 7784 + 60 | 688 / 7864 |

So the flag is worth **+3.8% at six gluons and +5.6% at seven**. The JAMP
shrink from `fcd8218b6` carries seven gluons on its own; six gluons only
turns positive with the current sum, because the reconstruction deviates from
the canonical decomposition — which is the whole point — and thereby weakens
the wavefunction CSE, 35 calls at six gluons and 57 at seven. Five gluons is
a wash (34.88 -> 35.12 s): 7 sums against 7 calls is too little to pay for
the 14 extra slots.

`|M|^2` bit-identical at four and five gluons, 1e-15 at six and seven, and
unchanged for `g g > t t~ g g` and `u u~ > g g g`.

With the flag off, `matrix.f` is byte-identical to `3b3ed9e85` for N=2..5.

## Step 6 — madevent, and what it does to AMP2

`29b0c670e`. Two things had to give before the optimisation survived the
madevent path, neither of them about AMP2:

1. **The sum has to be a `CALL`.** Helicity recycling rebuilds the whole DAG
   from the calls alone, so a bare assignment was invisible to it — the summed
   current never entered the graph and the amplitude reading it died on a
   `KeyError`. Written as `CALL SUMW_1(W(a),W(b),W(c))` it is an ordinary
   internal wavefunction with two mothers and everything downstream handles
   it. `sumw_1`/`subw_1` live in `aloha_functions.f`; the coefficient is
   restricted to ±1, which is all it has ever been.
2. **`hel_recycle.add_indices` could not index a statement-initial `AMP(`.**
   The pattern ate the character in front of it and there is none at the
   start of a line, so `AMP(31) = AMP(31) + AMP(1)` came out as
   `AMP(31) = AMP( K,31) + AMP( K,1)`. Latent until now: nothing had ever
   emitted a line beginning with `AMP(`.

**AMP2 is left alone and picks up the merged amplitude**, which is the right
thing and turns out to matter more than the matrix element speedup. The fold
lines run before the AMP2 block, so the channel weight is
`|AMP_cubic + AMP_quartic|^2`. Nothing else was needed: `get_amp2_lines`
already skips any diagram with a four point vertex, so the folded amplitudes
were never referenced.

That skip is the point. With the flag off, at six gluons:

| | amplitudes computed | distinct AMP reaching AMP2 |
|---|---|---|
| flag off | 510 | **105** |
| flag on | 450 (+30 sums) | 105, now carrying all 510 |

so four fifths of the amplitude — every quartic contribution — used to enter
**no** channel weight at all. Folding puts each of them in the channel whose
colour factor it shares, which is exactly where it belongs.

**The integration is not measurably better or worse.** That is the answer, and
it took some care to get to, because `generate_events` wall time is a bad
metric here: the refine stage adapts, and the *same* configuration swings by a
factor four between seeds (217 s to 877 s with the flag off). Every number
below is mean +- standard error over independent seeds.

`g g > g g g`, `generate_events`, 10000 events, three seeds:

| | cross section | rel. error | ME cpu |
|---|---|---|---|
| flag off | 3.680-3.694e+07 pb | 0.327% | 52.7 s |
| flag on | 3.684-3.694e+07 pb | 0.300% | 49.0 s |

`g g > g g g g`, `generate_events`, 2000 events, four seeds, and a survey-only
run — the same fixed number of points both ways, so the error measures the
channel weights and nothing else — over six seeds:

| | rel. error (survey) | rel. error (full) | cpu (full) |
|---|---|---|---|
| flag off | 2.14% | 0.498% | 583 s |
| flag on | 1.93% | 0.473% | 580 s |

Every difference is favourable and none of them is significant:

| | off - on | |
|---|---|---|
| `g g > g g g` error | -8% | 1.8 sigma |
| `g g > g g g` cpu | -7% | 1.1 sigma |
| `g g > g g g g` survey error | -10% | 1.1 sigma |
| `g g > g g g g` full error | -5% | 0.8 sigma |
| `g g > g g g g` full cpu | -1% | 0.0 sigma |

Cross sections agree everywhere. So the honest reading is that handing the
quartic contributions to the channel that shares their colour factor does not
hurt the integration, and may help it slightly — but the seed to seed scatter
is far larger than the effect, and nothing here is established at more than
two sigma. A campaign of tens of seeds would be needed to call it either way.

## Step 7 — madmatrix

`e93dfa1b1`. `SUMW_1`/`SUBW_1` as C++ templates next to `ALOHAOBJ` in
`cpp_hel_amps_h.inc` (they cannot use the `INLINE` macro, which the ALOHA
generated block defines further down the header), and the madmatrix writer
emits them exactly as the Fortran one does.

**The colour amplitudes had to be sorted out first, and that was a live bug.**
`get_color_amplitudes` dropped every merge source from the JAMPs on the
assumption that the caller writes the amplitude sums to put them back. Only
the Fortran writer does, so C++ and python output with the flag set
was quietly losing four fifths of the amplitude. It now takes
`merge_quartic_amplitudes`; a backend which writes no sums keeps those
amplitudes in the JAMPs, where their own colour coefficients give the
identical result. So madmatrix gets the current sums, which really do remove
work, and leaves the amplitude level merges alone — there is no `AMP` array
there to fold into anyway, each amplitude going straight into the JAMPs.

One bug in the shared writer surfaced here: a wavefunction number can be
listed by more than one diagram in the madmatrix matrix element (two objects
for the same current with the mothers ordered differently), so a sum was
written twice — 50 lines for 30 sums at six gluons. Harmless numerically, both
write the same value to the same slot, but wasted. Both writers now write each
sum once.

|M|^2 to 1e-14 at five and six gluons (`FPTYPE=d`; the default mixed
precision build rounds the two to the same value), `CPPProcess.cc`
byte-identical with the flag off.

**Speed is mixed, and worse than Fortran** (after step 8):

| | amp calls | nwf | evt/s (sse4, FPTYPE=d) | |
|---|---|---|---|---|
| `g g > g g g` | 45 -> 38 | 12 -> 19 | 72820 -> 68010 | **-6.6%** |
| `g g > g g g g` | 510 -> 450 | 51 -> 86 | 2696 -> 2748 | **+1.9%** |

There is no JAMP fold here to pay for the extra wavefunctions, so five gluons
loses outright: 7 sums against 7 saved amplitude calls does not cover a
wavefunction array half again as large. Note the slot count with the flag on
is worse in madmatrix than in Fortran (86 against 66 at six gluons), because
its matrix element carries duplicate wavefunctions -- two objects for the same
current with the mothers ordered differently, which MG5 does not merge.

## Step 8 — recycle the slots the sums take

`5b421bc80`. A sum used to get a slot of its own at the end, never reused,
which more than doubled `NWAVEFUNCS`. But a sum is an ordinary producer --
written as soon as the later of its two currents is made, dead after the last
amplitude reading it -- so it goes through `reuse_outdated_wavefunctions` with
everything else. That also lets the cubic current die at the sum rather than
at the amplitude, since the amplitude no longer reads it.

| | flag off | own slots | recycled |
|---|---|---|---|
| `g g > g g g` | 12 | 26 | **19** |
| `g g > g g g g` | 51 | 91 | **66** |
| `g g > g g g g` (madmatrix) | 51 | 111 | **86** |

**A bug in `reuse_outdated_wavefunctions` had to be fixed first, and it was
not introduced here.** The same wavefunction can be listed by more than one
diagram in the madmatrix matrix element, and the allocator handed it a slot
again on the second listing, leaking the first. Harmless while the sums had
their own slots; once they shared the pool, `g g > g g g g` came out 0.2%
wrong. A wavefunction now takes one slot at its first appearance and keeps it
until its last use. Nothing moves with the flag off -- `matrix.f` and
`CPPProcess.cc` are byte-identical for N=2..4 in both backends.

Worth it for madmatrix (-8.4% -> -6.6% at five gluons, +0.9% -> +1.9% at six)
and neutral for Fortran (+3.8% -> +3.9%): there the slot count was never the
bottleneck. The madevent run is unchanged, same cross section and error.

## Results

Everything below is `g g > N g` with the flag off against `speed`, on the
same machine. Standalone Fortran is the shipped `check` driver looping
`SMATRIX`; madmatrix is `check_sa.exe perf` built `FPTYPE=d` on `cpu_128b`
(the default mixed precision build rounds the two to the same value and would
hide any difference). Two runs each, reproducible to about 0.1%.

**Speed**

| | standalone | | madmatrix | |
|---|---|---|---|---|
| | per call | | evt/s | |
| `g g > g g` | 11.00 -> 11.04 s | -0.4% | 875150 -> 878724 | +0.4% |
| `g g > g g g` | 34.90 -> 34.99 s | -0.3% | 72359 -> 66757 | **-7.7%** |
| `g g > g g g g` | 47.35 -> 45.61 s | **+3.7%** | 2699 -> 2784 | **+3.1%** |
| `g g > 5 g` | 43.05 -> 39.98 s | **+7.1%** | 41.75 -> 43.33 | **+3.8%** |

Four gluons is a wash on both (there is nothing to sum: the only quartic
vertex is the whole amplitude). Five gluons loses on madmatrix, where the
wavefunction store grows by half and there is no JAMP fold to pay for it. Six
and seven gluons win on both, and the gain grows with the multiplicity.

**Where it came from.** Three states: the flag off (what an unoptimised build
gives), the branch as it stood before this work (`3b3ed9e85`, only the
amplitude merges of `fcd8218b6`), and the flag on today.

standalone Fortran, `slots / wavefunction calls + sums / amplitude calls`:

| | flag off | before this work | at PR open | now |
|---|---|---|---|---|
| `g g > g g` | 5 / 7 / 6 | 5 / 7 / 6 | 5 / 7 / 6 | 5 / 7 / 6 |
| `g g > g g g` | 12 / 33 / 45 | 12 / 33 / 45 | 19 / 39+7 / 38 | 19 / 39+7 / 38 |
| `g g > g g g g` | 51 / 111 / 510 | 51 / 111 / 510 | 66 / 146+30 / 450 | **78 / 126+30 / 450** |
| `g g > 5 g` | 268 / 898 / 7245 | 268 / 898 / 7245 | 290 / 955+60 / 6813 | **259 / 925+60 / 6813** |

madmatrix, `slots / amplitude calls`:

| | flag off | before this work | at PR open | now |
|---|---|---|---|---|
| `g g > g g` | 5 / 6 | *wrong* | 5 / 6 | 5 / 6 |
| `g g > g g g` | 12 / 45 | *wrong* | 19 / 38 | 19 / 38 |
| `g g > g g g g` | 51 / 510 | *wrong* | 86 / 450 | **78 / 450** |
| `g g > 5 g` | 268 / 7245 | *wrong* | 320 / 6813 | **259 / 6813** |

"wrong" is not a figure of speech: at `3b3ed9e85` `get_color_amplitudes` dropped
every merge source from the JAMPs unconditionally, and only the Fortran writer
wrote the sums putting them back, so with the flag set any C++ or python output
lost 405 of its 510 amplitudes at six gluons, silently. That is fixed here.

Per-call time, standalone Fortran (lower is better) and madmatrix in evt/s
(higher is better):

| | off | before | at PR open | now |
|---|---|---|---|---|
| fortran `g g > g g g g` | 47.77 s | 48.12 s | 45.61 s | **45.25 s (+5.3%)** |
| fortran `g g > 5 g` | 42.19 s | 40.88 s | 39.98 s | **40.10 s (+4.9%)** |
| madmatrix `g g > g g g` | 74229 | *wrong* | 66757 | 67566 (-9.0%) |
| madmatrix `g g > g g g g` | 2651 | *wrong* | 2784 | **2880 (+8.6%)** |

Timings taken in batches, and there is about 1% of drift between batches, so
each column should be read against the `off` measured with it.

**Memory — the wavefunction store**, which is what the optimisation costs.
`NWAVEFUNCS` in Fortran, `nwf` in madmatrix; bytes are 100 per wavefunction in
Fortran (4 complex, 4 reals, one int) and 192 in madmatrix on sse4 in double
(4 complex plus a 4-momentum, over a 2 event vector).

| | off | on, slot each | on, recycled | |
|---|---|---|---|---|
| `g g > g g` | 5 | 5 | 5 | 500 B |
| `g g > g g g` | 12 | 26 | **19** | 1900 B (+58%) |
| `g g > g g g g` | 51 | 91 | **66** | 6600 B (+29%) |
| `g g > 5 g` | 268 | 321 | **290** | 29000 B (+8%) |

madmatrix carries duplicate wavefunctions of its own, so its count with the
flag on is higher: 19 / 86 / 320 at five, six and seven gluons, against
19 / 66 / 290 in Fortran. The relative cost falls as the multiplicity rises
(+58% at five gluons, +19% at seven), which is why the trade turns positive
from six gluons on.

**Work done per call**

| | standalone helas calls | JAMP lines | madmatrix amplitude calls | jamp lines |
|---|---|---|---|---|
| `g g > g g` | 29 -> 29 | 23 -> 20 | 6 -> 6 | 34 -> 34 |
| `g g > g g g` | 94 -> 100 (+7 sums) | 131 -> 101 | 45 -> 38 | 370 -> 314 |
| `g g > g g g g` | 637 -> 642 (+30) | 1082 -> 688 | 510 -> 450 | 8170 -> 7210 |
| `g g > 5 g` | 8159 -> 7844 (+60) | 23672 -> 7864 | 7245 -> 6813 | 231850 -> 218026 |

`|M|^2` is bit-identical at four and five gluons and agrees to 1e-14 at six
and seven, in both backends. With the flag off, `matrix.f` and `CPPProcess.cc`
are byte-identical to before any of this.

## Full sweep -- speed, memory and generation time

All three modes against the flag off, on the same machine, for both series.
Timings are the minimum of five runs of the shipped `check` driver looping
`SMATRIX` on a fixed phase space point; the minimum matters, because a single
run carries about 3% of noise and most of the effects here are smaller than
that. The floor of the method is 0.6%, measured on `g g > t t~`, where `off`
and `speed` produce a byte-identical `matrix.f` and still time 4.000 against
3.988 us. `|M|^2` agrees to 8.5e-15 or better on every row.

*slots* is `NWAVEFUNCS`, the length of the wavefunction array, `TYPE(ALOHA)
W(NWAVEFUNCS)` -- not the number of wavefunctions computed, since
`reuse_outdated_wavefunctions` frees an entry as soon as its last reader has
run (898 wavefunction calls live in 268 slots at seven gluons). One entry is
104 bytes, measured with `storage_size`: four `complex*16`, `P(0:3)` and
`flv_index`, padded. One amplitude is 16 bytes.

**`g g > N g`**

| process | mode | generate | matrix.f | matrix.o | W slots | W array | AMP entries | amps computed | AMP array | per call | speed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g g > 2g | off | 1.8 s | 27 kB | 16 kB | 5 | 0.5 kB | 6 | 6 | 0.1 kB | 5.48 us | - |
|  | speed | 1.5 s | 27 kB | 17 kB | 5 | 0.5 kB | 4 | 6 | 0.1 kB | 5.52 us | -1% |
|  | slots | 1.6 s | 27 kB | 17 kB | 5 | 0.5 kB | 4 | 6 | 0.1 kB | 5.55 us | -1% |
| g g > 3g | off | 2.1 s | 40 kB | 28 kB | 12 | 1.2 kB | 45 | 45 | 0.7 kB | 87.50 us | - |
|  | speed | 1.8 s | 39 kB | 29 kB | 19 | 1.9 kB | 24 | 38 | 0.4 kB | 90.25 us | -3% |
|  | slots | 2.4 s | 40 kB | 29 kB | 12 | 1.2 kB | 16 | 45 | 0.2 kB | 93.00 us | -6% |
| g g > 4g | off | 2.8 s | 181 kB | 141 kB | 51 | 5.2 kB | 510 | 510 | 8.0 kB | 2.37 ms | - |
|  | speed | 2.4 s | 165 kB | 146 kB | 78 | 7.9 kB | 316 | 450 | 4.9 kB | 2.29 ms | +4% |
|  | slots | 2.4 s | 168 kB | 152 kB | 54 | 5.5 kB | 106 | 510 | 1.7 kB | 2.41 ms | -2% |
| g g > 5g | off | 35.2 s | 3.3 MB | 5.8 MB | 268 | 27.2 kB | 7245 | 7245 | 113.2 kB | 141.00 ms | - |
|  | speed | 16.6 s | 2.5 MB | 3.4 MB | 259 | 26.3 kB | 5869 | 6813 | 91.7 kB | 132.75 ms | +6% |
|  | slots | 18.4 s | 2.5 MB | 3.4 MB | 199 | 20.2 kB | 946 | 7245 | 14.8 kB | 134.25 ms | +5% |

**`g g > t t~ N g`**

| process | mode | generate | matrix.f | matrix.o | W slots | W array | AMP entries | amps computed | AMP array | per call | speed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g g > t t~ | off | 1.5 s | 26 kB | 16 kB | 5 | 0.5 kB | 3 | 3 | 0.0 kB | 4.00 us | - |
|  | speed | 1.5 s | 26 kB | 16 kB | 5 | 0.5 kB | 3 | 3 | 0.0 kB | 3.99 us | +0% |
|  | slots | 2.0 s | 26 kB | 16 kB | 5 | 0.5 kB | 3 | 3 | 0.0 kB | 4.09 us | -2% |
| g g > t t~ g | off | 1.8 s | 31 kB | 20 kB | 12 | 1.2 kB | 18 | 18 | 0.3 kB | 30.00 us | - |
|  | speed | 1.6 s | 31 kB | 20 kB | 12 | 1.2 kB | 15 | 15 | 0.2 kB | 29.58 us | +1% |
|  | slots | 1.7 s | 31 kB | 21 kB | 12 | 1.2 kB | 15 | 18 | 0.2 kB | 30.33 us | -1% |
| g g > t t~ 2g | off | 2.6 s | 68 kB | 51 kB | 26 | 2.6 kB | 159 | 159 | 2.5 kB | 377.00 us | - |
|  | speed | 2.4 s | 65 kB | 49 kB | 35 | 3.6 kB | 109 | 126 | 1.7 kB | 343.00 us | +9% |
|  | slots | 2.4 s | 66 kB | 53 kB | 29 | 2.9 kB | 106 | 159 | 1.7 kB | 391.00 us | -4% |
| g g > t t~ 3g | off | 6.2 s | 576 kB | 479 kB | 121 | 12.3 kB | 1890 | 1890 | 29.5 kB | 9.63 ms | - |
|  | speed | 4.9 s | 463 kB | 408 kB | 213 | 21.6 kB | 1159 | 1551 | 18.1 kB | 8.97 ms | +7% |
|  | slots | 5.0 s | 493 kB | 466 kB | 141 | 14.3 kB | 946 | 1890 | 14.8 kB | 9.93 ms | -3% |

Two things worth reading off the amplitude columns.

**The AMP array is recycled too, and it is where `slots` wins.** It used to be
declared at the full diagram count in every mode -- 113 kB at seven gluons,
with `speed` leaving 432 entries written by nobody -- which is what prompted
"Recycling the AMP array" below. Now `slots` runs 7245 amplitude calls through
946 entries at seven gluons, 14.8 kB rather than 113.2, while `speed` only
reaches 5869 because of the order it emits them in.

**`slots` mode computes every amplitude** -- `amps computed` equals `amps
decl` on all eight of its rows. With no current sums nothing is skipped, so it
does the same amplitude work as the baseline *plus* the folds, and buys only a
shorter JAMP block and a shorter W array. That is why it is slower than off
almost everywhere rather than a wash: strictly more arithmetic for less
memory. `speed` is the opposite, skipping amplitudes outright (450 of 510,
6813 of 7245, 1551 of 1890), which is where its 5-8% comes from, and paying in
slots -- 121 to 213 at `g g > t t~ 3g`.

**What it is actually good for.** Generation time and code size, more than
speed. `g g > 5 g` generates in 16.6 s rather than 35.2 s, a 53% cut and
reproducible: 385 seed diagrams unrolled is cheaper than 2485 generated. Its
`matrix.o` goes 5.8 MB to 3.4 MB and its `matrix.f` 3.3 MB to 2.5 MB. Runtime
is 4-9% above six particles and nothing at all below, and the `t t~` series
gains more than the pure gluon one at equal particle count, +9% at
`t t~ 2g` against +4% at `4g`. Peak RSS is flat except at seven gluons,
because the wavefunction store is a stack frame and the code image dominates.

**How far it generalises.** Both series above are gluon-rich, so a third
process was measured as a check: `u u~ > z g g g g`, seven legs like
`g g > t t~ 3g` but with a Z and a quark line, so most of its diagrams have no
four gluon vertex at all. (`q` is not a defined multiparticle, hence `u u~`.)

| process | mode | generate | matrix.f | matrix.o | W slots | AMP entries | amps computed | per call | speed | total/call |
|---|---|---|---|---|---|---|---|---|---|---|
| `g g > t t~ 3g` | off | 6.2 s | 576 kB | 479 kB | 121 | 1890 | 1890 | 9.63 ms | - | 41.8 kB |
| | speed | 4.9 s | 463 kB | 408 kB | 213 | 1159 | 1551 | 8.97 ms | +7% | 39.7 kB (-5%) |
| | slots | 5.0 s | 493 kB | 466 kB | 141 | 946 | 1890 | 9.93 ms | -3% | **29.1 kB (-30%)** |
| `u u~ > z 4g` | off | 2.9 s | 135 kB | 115 kB | 76 | 516 | 516 | 1.58 ms | - | 15.8 kB |
| | speed | 2.3 s | 127 kB | 110 kB | 85 | 391 | 450 | 1.51 ms | +4% | 14.7 kB (-7%) |
| | slots | 2.3 s | 132 kB | 117 kB | 84 | 384 | 516 | 1.65 ms | -5% | 14.5 kB (-8%) |

**The payoff tracks the quartic fraction, exactly.** In `slots` mode the AMP
entry count is `total amplitudes - merge sources`, to within one, on every
process measured:

| | amplitudes | merge sources | AMP in `slots` | saving |
|---|---|---|---|---|
| `g g > 5 g` | 7245 | 6300 (87%) | 946 | -87% |
| `g g > g g g g` | 510 | 405 (79%) | 106 | -79% |
| `g g > t t~ 3g` | 1890 | 945 (50%) | 946 | -50% |
| `u u~ > z 4g` | 516 | 132 (**26%**) | 384 | -26% |

So `u u~ > z 4g` is the weakest case measured, and predictably: there is
simply little to merge. Runtime follows at +4% rather than +7 to +9%, and the
working set at -8% rather than -30 or -75%.

It is also the one process where **`slots` does not reduce the wavefunctions
either** -- 84 against off's 76, worse -- which breaks the pattern from the
gluon-rich processes, where reversing the order always recovered them. On this
topology `slots` is the worst of the three: 5% slower than off and larger in
W, for an AMP count it barely wins over `speed`, 384 against 391. `speed`
still behaves, +4% with generation down 21% and a smaller source and object,
which is also what the `auto` gate picks at seven legs.


## The multiplicity gate

The sweep says the full merging turns over at six external legs, and the same
sweep says it costs slots below that. So `auto` gates on it:
`Amplitude.generate_diagrams` only takes the seed rule when the process has
`madgraph.merge_quartic_min_legs` legs or more, six by measurement. `speed`
and `slots` asked for by name are unconditional -- that is how you get the
merging on a small process anyway.

What is left below the threshold is not nothing, and this was worth measuring
rather than assuming. The *amplitude* merges do not need the seed rule: they
are found from the colour algebra by `unroll_quartic_vertices`, so they still
apply, and they shrink the JAMP block without touching the wavefunctions:

| `g g > g g g` | slots | JAMP temporaries | per call |
|---|---|---|---|
| off | 12 | 72 | 87.75 us |
| `auto` (merges only) | 12 | 42 | **84.25 us**, +4.0% |
| `speed` (full) | 19 | 42 | 87.25 us, -1% |

So below the threshold `auto` is *better* than both -- it keeps the JAMP fold,
which is free, and drops the reordering, which is what costs the seven extra
slots. At four and five legs elsewhere it is neutral rather than positive
(`g g > g g` 5.47 -> 5.48 us, `g g > t t~ g` 30.33 -> 30.42 us, both inside
the 0.6% floor) and never negative, at an unchanged slot count.

Above the threshold nothing changes: `auto` at six legs generates a
byte-identical `matrix.f` to `speed`.

## Recycling the AMP array

`reuse_outdated_wavefunctions` recycles the wavefunctions; the amplitudes were
not recycled at all. `AMP` was declared `COMPLEX*16 AMP(NGRAPHS)` at the full
diagram count in every mode, so seven gluons allocated 113 kB of it and
`speed` left 432 entries written by nobody.

`HelasMatrixElement.get_amplitude_slots` now does for AMP what
`reuse_outdated_wavefunctions` does for W. The enabling change is *where the
merges are written*: they used to be emitted in one block at the very end,
which kept every source alive to the end, and each is now written as soon as
both of its amplitudes exist. Once `AMP(t) = AMP(t) + AMP(s)` has run, `s` is
free.

| process | mode | AMP entries | AMP | W slots | W | total per call |
|---|---|---|---|---|---|---|
| `g g > g g g` | off | 45 | 0.7 kB | 12 | 1.2 kB | 1.9 kB |
| | auto | 16 | 0.2 kB | 12 | 1.2 kB | 1.5 kB (-24%) |
| | speed | 24 | 0.4 kB | 19 | 1.9 kB | 2.3 kB (+20%) |
| | slots | 16 | 0.2 kB | 12 | 1.2 kB | 1.5 kB (-24%) |
| `g g > g g g g` | off | 510 | 8.0 kB | 51 | 5.2 kB | 13.1 kB |
| | speed | 316 | 4.9 kB | 78 | 7.9 kB | 12.9 kB (-2%) |
| | slots | **106** | 1.7 kB | 54 | 5.5 kB | 7.1 kB (**-46%**) |
| `g g > t t~ g g` | off | 159 | 2.5 kB | 26 | 2.6 kB | 5.1 kB |
| | speed | 109 | 1.7 kB | 35 | 3.6 kB | 5.3 kB (+3%) |
| | slots | 106 | 1.7 kB | 29 | 2.9 kB | 4.6 kB (-10%) |
| `g g > 5 g` | off | 7245 | 113.2 kB | 268 | 27.2 kB | 140.4 kB |
| | speed | 5869 | 91.7 kB | 259 | 26.3 kB | 118.0 kB (-16%) |
| | slots | **946** | 14.8 kB | 199 | 20.2 kB | **35.0 kB (-75%)** |

**`slots` reaches the floor and `speed` does not**, and the reason is the
diagram order rather than anything about the allocator. Only (2n-5)!! of the
amplitudes are read by the JAMPs -- 105 at six gluons, 945 at seven -- and
everything else is a merge source which could in principle share a handful of
entries. `speed` emits every seed before its unrollings, so a source is born
early and its target arrives late and the entry cannot be reclaimed in
between: 5869 rather than 945. Reversing that order puts each source next to
its target, so `slots` lands on 946 and 106, one above the floor.

That changes the `slots` case rather a lot **in fortran**: it used to buy 23%
of the wavefunction store and cost 6% more arithmetic, and it now buys 75% of
the whole per-call working set at seven gluons.

**It does not carry over to the gpu backend, which is the one the argument was
about.** madmatrix has no AMP array to recycle -- it keeps a single
`cxtype_sv amp_sv[1]` and each amplitude goes straight into the JAMPs, so it
was already at a floor of one entry, better than the 946 the fortran recycling
reaches. Its per-thread cost is 112 bytes a wavefunction (`pvec_sv` 32,
`w_sv` 64, `aloha_obj` 16) plus `jamp_sv[ncolor]`, and the wavefunctions are
the only thing either mode moves:

| | mode | nwf | madmatrix per thread | fortran per call |
|---|---|---|---|---|
| `g g > g g g g` | off | 51 | 7.5 kB | 13.1 kB |
| | speed | 78 | 10.4 kB (+40%) | 12.9 kB (-2%) |
| | slots | 54 | 7.8 kB (+4%) | 7.1 kB (-46%) |
| `g g > 5 g` | off | 268 | 40.6 kB | 140.4 kB |
| | speed | 259 | 39.6 kB (-2%) | 118.0 kB (-16%) |
| | slots | 199 | 33.0 kB (**-19%**) | 35.0 kB (-75%) |
| `g g > t t~ 3g` | off | 121 | 15.1 kB | 41.8 kB |
| | speed | 213 | 25.2 kB (+67%) | 39.7 kB (-5%) |
| | slots | 141 | 17.3 kB (+14%) | 29.1 kB (-30%) |
| `u u~ > z 4g` | off | 76 | 8.7 kB | 15.8 kB |
| | speed | 85 | 9.7 kB (+11%) | 14.7 kB (-7%) |
| | slots | 84 | 9.6 kB (+10%) | 14.5 kB (-8%) |

So on the gpu backend `slots` is -19% at seven gluons and *worse than off* at
every smaller process measured, and `speed` is worse still, +40% and +67% on
two of the four. `jamp_sv` is mode independent and not small either -- 11.2 kB
at seven gluons, 28% of the total -- which dilutes it further. The only lever
that exists there is the wavefunction count.

The recycling is otherwise correctly supported in madmatrix: `nwf` matches the
fortran slot count on all twelve generations, the current sums land inside it
(`aloha_obj` indices 0..77 against `nwf` 78 at six gluons with 30 sums), and
`|M|^2` is bit identical across the three modes.

**It buys no time.** Measured at `g g > g g g g`, minimum of five: `speed`
2247 -> 2260 us and `slots` 2347 -> 2373 us across the change, both inside the
0.6% floor and if anything marginally the wrong way -- the arrays were already
cache resident at this size, and the merges are now interleaved rather than
batched. This is a memory optimisation, not a speed one.

`NGRAPHS` only ever dimensioned `AMP` inside `matrix.f`, so it simply becomes
the entry count; `ngraphs.inc` keeps the diagram count. Everything reading AMP
afterwards goes through the same map: the JAMPs through
`ProcessExporterFortran.map_color_amplitudes`, and AMP2 through
`get_amplitude_slot_map`. AMP2 was the one to check, since multichannel reads
individual amplitudes -- it reads only the merge *targets*, which are the
entries that stay put, so it is unaffected. Verified on a madevent output at
six gluons: 316 entries written, 316 read, none read that is never written.

`|M|^2` is unchanged on every row.

## The diagram order — measured, and worth a lot

`reuse_outdated_wavefunctions` is a linear scan allocator over lifetimes taken
in **emission order**, so `NWAVEFUNCS` depends on the order the diagrams are
written in. (The wavefunction *set* does not — that is content addressed,
`wavefunctions[wavefunctions.index(new_wf)]` — but the number of slots very
much does.) The shipped expansion emits every seed first and then the
unrollings, which is the worst case for it: a quartic current is made early
and its sum is not consumed until much later.

Four orders were built and run. A sum can only be formed when its quartic
current is emitted before the target amplitude, which is what makes this a
trade rather than a free win:

| `NWAVEFUNCS` / sums | off | seeds first (shipped) | by quartic count | seed then its own unrollings | last discovery |
|---|---|---|---|---|---|
| `g g > g g g` | 12 | 19 / 7 | 19 / 7 | 11 / 3 | **15 / 7** |
| `g g > g g g g` | 51 | 66 / 30 | 64 / 30 | 33 / **0** | **27 / 30** |
| `g g > 5 g` | 268 | 290 / 60 | 314 / 60 | 245 / 60 | **219 / 60** |

Emitting a seed followed by its own unrollings gives the locality but loses
the sums: the fully cubic target is usually claimed by an earlier seed, so the
quartic current arrives after it. **Placing each diagram at its *last*
discovery instead of its first fixes that** — the target then sits after every
seed which can reach it — and takes both: all the sums, and a slot count
*below* the flag off baseline (27 against 51 at six gluons, 219 against 268 at
seven).

**Shipped, once the reason it broke madmatrix was found.** The blocker was not
the order at all:

1. The reconstruction could build the same cubic current with its two mothers
   in either order. `sorted_mothers` leaves them alone, because for two
   identical gluons its key ties and the sort is stable, so they stay two
   objects.
2. **`VVV1P0_1` is antisymmetric under exchanging its two inputs.** Measured:
   `VVV1P0_1(a,b) + VVV1P0_1(b,a) = 0` exactly. So the two are *negatives* of
   each other and write out calls differing by a sign.
3. **`HelasWavefunction.__eq__` compares mothers by sorted number** and calls
   them equal -- "the number for this wavefunction, the pdg code, and the
   interaction id are irrelevant".
4. `export_cpp` renumbers wavefunctions by that equality, so the two landed on
   one number and one slot inside a single matrix element and one of them
   silently carried the wrong sign. The Fortran writer never hits it because
   it does not renumber by equality.

Taking the legs of both unrolled vertices in a canonical order removes every
such pair -- 20 of the 35 extra wavefunctions at six gluons, 30 of 57 at seven
-- so there is nothing left to collide, and the order goes in.

| | wavefunctions | order-flipped twins |
|---|---|---|
| `g g > g g g` | 33 -> 39 | 0 |
| `g g > g g g g` | 111 -> 126 (was 146) | 20 -> 0 |
| `g g > 5 g` | 898 -> 925 (was 955) | 30 -> 0 |

Placing each diagram at its last discovery then puts it after every seed which
can reach it, hence after every quartic current summable into it, so all the
sums survive. `NWAVEFUNCS` at seven gluons ends up **below** the unoptimised
build: 259 against 268.

## The slot ordering, searched

Once the twins are gone the slot count is the same in both backends (78 and
259 at six and seven gluons), so the order is one shared problem rather than a
per-backend one. Six orders were built and measured. The wavefunction count is
the same in all of them -- reordering the diagrams never changes *which*
currents exist, only how long each stays alive:

| order | `g g > g g g` | `g g > g g g g` | `g g > 5 g` | sums kept |
|---|---|---|---|---|
| last discovery (shipped) | 19 | 78 | 259 | yes |
| first discovery | 19 | 81 | 314 | yes |
| by quartic count | 19 | 81 | 314 | yes |
| by quartic count, then last | 19 | 81 | 259 | yes |
| last discovery, then quartic count | 19 | 78 | 259 | yes |
| **reversed last discovery** | **12** | **54** | **199** | **no -- all lost** |

The last row is the interesting one: it is far the best on slots and useless,
because reversing puts each target ahead of the quartic currents which feed
it, so no sum can be built. That is the trade in one line -- the constraint
that makes the sums possible is what costs the slots.

A proper register-pressure greedy was also written: order the diagrams under
the precedence "everything which unrolls to a diagram comes before it", and at
each step take the one leaving fewest currents alive. It buys 19 -> 18 and
78 -> 76 and **nothing at all** at seven gluons, for an O(n^2) pass which
takes generation from 0.88 s to 3.24 s at seven gluons and would cost minutes
at eight. Not worth it; the shipped order is within a couple of slots of what
the search finds.

## The backend-chosen order (`auto`)

`speed` suits a cpu and `slots` suits a gpu, but one generation can feed both,
so `auto` defers the choice to `output`. It works only because the two modes
differ in nothing but the diagram order, and `slots` is `speed` reversed --
verified byte for byte: generating in the `speed` order and reversing at
output time reproduces a native `slots` generation exactly, in both backends,
and a session going standalone_fortran -> standalone -> standalone_fortran reproduces its
first output for the third.

**The choice comes from the matrix element exporter, not the output format.**
`output madevent --me_exporter=<gpu backend>` hands one
`self._curr_matrix_elements` to both exporters in a single `export_processes`
call, so the fortran driver and the gpu matrix elements *cannot* carry
different orders. Keying on the format would give that output the cpu order,
which is the one case the whole thing exists for; keying on the me exporter
gives it `slots` (measured: `NWAVEFUNCS=54` rather than 78 at six gluons).

Two things had to be worked around, both pre-existing and neither specific to
this option:

* **An export mutates the diagrams it is given** -- 345 of 757 vertex leg
  records change on `g g > g g g g`. Reversing mutated diagrams gives an
  equivalent but differently numbered result, so `apply_quartic_diagram_order`
  reverses a copy taken before the first export. `copy.deepcopy` of the
  *amplitudes* is not an option: it drags the model along and trips
  `assert type(col_obj) != array.array` in `color_algebra.create_copy`. The
  diagrams alone copy cleanly and cost 0.011 s at seven gluons.
* **An export also drops the marks saying the diagrams came from a seed** --
  after it, `seed_forbidden_cubic_ids` is empty and `quartic_unroll_tags` has
  0 entries instead of 405. So whether an amplitude may be reordered has to be
  read before the first export, not at the output which wants to reorder.

Two dead ends, both measured: reversing the `HelasMatrixElement` diagrams
instead of the amplitude's trips the lifetime assert in
`reuse_outdated_wavefunctions`, since a wavefunction number has to be first
seen in emission order; and restoring only `from_group` on the base diagrams
is not enough to undo an export.

## Why this is not the default

Defaulting `merge_quartic_vertices` to `auto` was tried and **reverted**. It is
safe on the paths it was built for -- UFO Fortran, madmatrix, the python
exporter -- and each of those is validated on `|M|^2`. Turning it on for
everything found three consumers which read the diagram or amplitude
*structure* rather than the result:

1. **FKS born/real linking.** `link_rb_configs` finds the vertex splitting
   `ij` into `i` and `j` and takes it out. The unrolling re-roots the real
   diagrams and can put that pair in the closing vertex, where there is
   nothing to take out, so the remainder is malformed and no born
   configuration matches it. Same 3 diagrams selected, same tag set, different
   decomposition:

   ```
   off   ((1,2>1),(4,5>4),(1,3,4))     the 4-5 vertex is internal
   auto  ((1,2>1),(1,3>1),(1,4,5))     the 4-5 pair closes the diagram
   ```

   `p p > j j [QCD]` raised `FKSProcessError`. **Fixed** by generating an NLO
   process with the merging off, in `FKSMultiProcess.__init__`, so the option
   is now safe for an NLO user rather than only for the default. `g g > g g
   [QCD]` generates byte-identically with the option set and unset.

2. **The legacy `FortranHelasCallWriter`.** Only `FortranUFOHelasCallWriter`
   emits the amplitude folds which put the merged contributions back. The
   MG4-style writer computes `AMP(1..3)` from `GGGGXX` and then leaves them out
   of the JAMPs -- a **silently wrong** `|M|^2`, not a crash. Not fixed: like
   `export_cpp` and `export_python` it needs `merge_quartic_amplitudes=False`,
   but `get_JAMP_lines` is on the exporter and does not know which writer it
   is paired with.

3. **Anything pinning the diagram order.** Cosmetic but wide: `colorize` and
   `DiagramTag` tests select diagrams by position, and the sextet colour basis
   goes 13 -> 15 because folding amplitudes decomposes the same `|M|^2` over
   more colour structures (`|M|^2` bit-identical, checked with
   `MatrixElementEvaluator`).

### The scan for a fourth

Looked for one, did not find one in the shipped tree, and the search bounds
the remaining risk.

**Every merged-JAMP consumer, against every writer.** `get_color_amplitudes`
has four call sites: `export_cpp` (x2), `export_python` and `madmatrix` all
pass `merge_quartic_amplitudes=False`; only `export_v4`'s three
`get_JAMP_lines*` take the merged default, and every Fortran exporter pairs
with `FortranUFOHelasCallWriter`, which emits the folds. The base
`get_amplitude_merge_lines` returns `[]` and `FortranHelasCallWriter` does not
override `get_matrix_element_calls`, so it is the one writer that silently
drops them -- and it is selected exactly when `self._model_v4_path` is set,
i.e. under `import model_v4`. No other combination in the tree reaches merged
JAMPs without folds.

The property which keeps `merge_quartic_amplitudes=False` safe is that
`get_color_amplitudes` drops the current-sum folded amplitudes
*unconditionally* and only the amplitude merges conditionally -- so a writer
which emits the sums but not the folds still gets consistent JAMPs.

**A mechanical audit of the generated code.** For each `AMP(n)` in a generated
matrix element, whether it is written and whether it is read. Read-never-
written is garbage; written-never-read is a contribution dropped on the floor,
which is the legacy writer's signature. Run with the flag off as a control on
every output -- standalone, matchbox, madevent grouped and not, a decay chain,
helicity-recycled files, `u u~ > g g g`, `g g > t t~ g g`, `u u~ > u u~ g g`,
four to six gluons -- and clean everywhere. The split order path is included:
`ProcessExporterFortranSA` and `ProcessExporterFortranME` both always go
through `get_JAMP_lines_split_order`, whose `amp_orders` lists folded
amplitude numbers, but they never reach the code because the colour amplitudes
no longer mention them.

**The two places which weight or group results.** Subprocess grouping for
`p p > j j` gives the same five directories and byte-identical `configs.inc`
and `coloramps.inc`; only the `g g > g g` matrix element differs.
`find_symmetry`, which feeds the multiplicative `symfact.dat`, keeps the same
equivalence classes and multiplicities -- `[3, 3, 3, 6]` at five gluons and
`[3, 6, 12, 12, 12, 12, 12, 12, 24]` at six, in both settings, with the same
number of channels and the same total weight. Only the representative diagram
indices renumber, which is the renumbering itself.

So the residual risk of this class is two named things: `import model_v4`, and
a third-party plugin supplying its own `helas_exporter` paired with
`export_v4`'s merged JAMPs.

The pattern is that the optimisation changes the *representation* -- diagram
order, rooting, which amplitudes survive into the JAMPs -- and every consumer
which reads representation rather than result has to be checked. Three turned
up in one pass, so defaulting it on wants an audit of those consumers, not
another round of patching outward.

Also fixed on the way, and independent of all this: `link_rb_configs` built
`real_tags` deduplicated but left `good_diags` as it was, then walked the two
in lockstep -- `real_tags.remove(btag)` beside `good_diags.pop(ir)` -- so they
only stayed aligned while the dedup dropped nothing. A no-op on every process
in the test suite, but it made the result order dependent for no reason.

## Where to go next

**Give madmatrix the amplitude sums too.** It is the only backend without
them, because there is no `AMP` array to fold into — each amplitude goes
straight into the JAMPs. That is why it gains 1.9% where Fortran gains 3.9%,
and why five gluons still loses. The sources for one target could be
accumulated into a second `amp_sv` slot before the JAMP lines are written,
which the seed ordering makes possible (the quartic diagrams come first), at
the cost of one accumulator per open target.

Then there are the sums which do not sit at an amplitude, and they need a
node to have exactly one rooting *per merge*, which a diagram list cannot
give. Three ways on, in increasing size:

1. **The second substitution.** Cheapest of the three, ceiling 432 amplitude
   calls at seven gluons, but measured to be at most 282 of them and possibly
   much less. Two things stand in the way and only the first is bookkeeping:

   *Identification.* All 432 targets which have a two-substitution source have
   exactly one of the two singles present; the other is the same diagram
   rooted differently, so it is a different amplitude object with different
   mothers and `match_quartic_mothers` cannot see it. Finding it by its
   diagram and colour chain — the frame `compute_quartic_amplitude_merges`
   already works in — rather than by its mothers would take it.

   *The coefficients have to multiply.* Substituting two mothers also produces
   the amplitude with both substituted, weighing the product of the two
   coefficients, so the merge map has to agree. It does not always: taking the
   double's coefficient over the known single's, the missing single would have
   to weigh -1 for 360 of the 432 targets and +1 for 72, and for **150 of them
   no merge source into that target weighs that at all**. The sign from
   `diagram_colour_signature` does not factorise over two contractions in
   general. `subset_is_merged` already refuses those, which is what keeps the
   present code sound; extending the identification does not remove the check.
2. **Partial CSE.** Keep the DAG, add `TMP = W1 + W4` alongside W1 and W4,
   and split the consumers. Bounded gain: at six gluons only 2 of the 6
   quartic consumers correspond, so it saves 2 subtrees per node out of 50
   nodes.
3. **Drop the diagram list for the currents.** Build the wavefunctions by a
   Berends-Giele recursion over subsets — at each node, cubic pair plus
   quartic, which generates exactly the matchings and never double counts —
   and keep the 220 diagrams only for what they are actually needed for
   (multichannel, `matrix.ps`).

Anything that fragments the diagram list to get a per-rooting copy runs into
pitfall 1, and anything that expands each seed independently double counts —
the fully cubic diagrams get reached once per matching of their adjacency
graph (225 instead of 105 at six gluons).

## Pitfalls — all of these cost real time in the previous session

1. **Do not fragment the diagram list.** Generating one diagram per colour
   structure (220 -> 510) works numerically but changes a user-visible number
   and the MadEvent multichannel. It was reverted for that reason.
2. **`copy.deepcopy` on a `Model` silently breaks colour.** `ColorObject`
   derives from `array.array` and deepcopy degrades it to a plain `array`, so
   the structures stop being recognisable. Use `create_copy`.
3. **Two colour-chain conventions exist.** `helas_objects` colorizes the
   *reconstructed* amplitude (`get_base_amplitude`), whose vertex order can
   differ from the generated diagrams. They agree up to six gluons and diverge
   at seven. Anything mapping chains to amplitudes must use the reconstructed
   one.
4. **Pinning a single colour structure needs colour and Lorentz in the same
   frame.** The sum over the three structures is permutation invariant but the
   individual terms are not, so the structure must be chosen against
   `sorted_mothers` (what ALOHA receives), not the vertex leg order.
5. **The `get_color_amplitudes` filter and the writer emitting the folds must
   land together.** Filtering without emitting silently drops the quartic
   contributions from the JAMPs — this produced a wrong `|M|^2` that survived
   three rounds of debugging because the helas calls looked byte-identical to
   baseline. When a number is wrong, diff the JAMP block first.
6. **A current sum is only legitimate if the consumers correspond 1:1.**
   Summing into a shared current hands the contribution to *every* consumer.
   Measured counter-example on the fragmented structure: quartic current 11
   had consumers {7,9,36,38,60,62} mapping to {1,31,55}, while its partner had
   {1,5,31,34,55,58} — three consumers would have gained a term they must not
   have.
7. **The amplitude sums buy nothing at runtime.** The JAMP CSE already finds
   those pairs (`TMP_JAMP(2) = AMP(1) + AMP(4)`). They do shrink the JAMP block
   (1091 -> 697 lines at six gluons), which helps the optimiser and compile
   time. Expect the speedup to come from the currents, not from these.
   *Measured since:* wrong at seven gluons, where the JAMP block goes 23672 ->
   7864 lines and that alone is +3.1%. Right at six, where it is a wash.
8. **A wavefunction number is not a slot.** `reuse_outdated_wavefunctions`
   hands the same `me_id` to wavefunctions whose lifetimes do not overlap, and
   works those lifetimes out from the diagrams alone. Anything emitting an
   extra read of a wavefunction has to extend the lifetime there, or it reads
   whatever else has since been written into that slot -- which showed up as
   `W(11)%W(:) = W(11)%W(:) + W(11)%W(:)`.

## Measuring

Generation: `set merge_quartic_vertices speed` then `generate g g > g g g g`.
Compare `matrix.f` against a run without the variable. Tests:
`./tests/test_manager.py -p U test_diagram_generation test_color_amp
test_helas_objects test_base_objects` (199, must stay green with the flag off).
