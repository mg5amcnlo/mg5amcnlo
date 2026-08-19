# Supporting the `@` grouping tags in the density spin modes

Design note, written against `madspin_density` with an eye on the staged
`unweighting` schemes of `doc/madspin_sequential_plan.md`.

> **Status.** Sections 3 and 4.1-4.3 are implemented: the density modes
> (`PA`, `onshell`, `madspin`/`full`) honour the tags for the rectangular card
> shape described in section 4.6, and the joint accept/reject is forced while
> they do. Section 4.4 (per-group bounds and `Z_k` tables, so the staged
> `unweighting` schemes keep working) is **not** implemented and is what
> section 5 calls structural. Everything outside that shape still warns and
> falls back to the ungrouped behaviour.

## 1. What the tags mean and where they work

The semi-leptonic *tt* idiom is

```
decay t  > w+ b,  w+ > l+ vl   @1
decay t~ > w- b~, w- > j j     @1
decay t  > w+ b,  w+ > j j     @2
decay t~ > w- b~, w- > l- vl~  @2
```

Lines sharing a tag are used *together*, so only the two charge assignments
exist and no fully leptonic or fully hadronic event is produced. A line without
a tag is common to every group.

The grouping lives in `decay_all_events.get_all_ME`
([MadSpin/decay.py:2869](../MadSpin/decay.py#L2869)): it sorts the branches into
`decay_text_correlated[tag]` and emits one `generate`/`add process` per group, so
the correlation is expressed inside the decayed matrix element. Only
`madspin_v1` instantiates that class ([interface_madspin.py:998](../MadSpin/interface_madspin.py#L998));
`onshell_v1`, `PA`, `onshell`, `madspin`/`full` and `none` all use
`decay_all_events_onshell` / `decay_all_events_density`, whose
`get_decay_command` ([decay.py:4470](../MadSpin/decay.py#L4470)) has no notion of
groups.

### 1.1 It fails silently — measured

`get_decay_command` appends an `@` process number of its own to every branch:

```python
newproc = "add process %s @%i --no_warning=duplicate --standalone;" % (proc, i)
```

so MG5 receives two of them. Its `proc_number_pattern` (`^(.+)@\s*(\d+)\s*(.*)$`,
greedy) binds the *last*, i.e. MadSpin's, at the top level, and the user's is
absorbed as the process number of the sub-decay. Checked directly:

```
IN : generate t* > w+ b, w+ > l+ vl @1 @0 --no_warning=duplicate --standalone
OUT: Process: t*> w+ b
       Decay: w+ > e+/mu+ ve/vm WEIGHTED=2 @1      <- id 1 on the SUB-decay
     top-level id = 0
```

The amplitude is the correct ungrouped one; nothing errors and (before this
branch) nothing warned. The legacy path strips the tag explicitly for the same
generation step ([decay.py:2957](../MadSpin/decay.py#L2957)); the density path
does not.

End-to-end on `p p > t t~`, 2000 events, the card above:

| mode | BR written to the banner | (W,W) categories |
|---|---|---|
| `madspin` (density) | 0.7529 = (BR_l + BR_h)^2 | 760 semi-lep, **1116 fully hadronic, 124 fully leptonic** |
| `madspin_v1` | 0.2963 = 2 BR_l BR_h | 2000 semi-lep, nothing else |

The density cross section is 2.54x the intended one and three quarters of the
sample is the wrong final state.

The warning added on this branch
(`MadSpinInterface._warn_ignored_decay_groups`, called from `do_launch`) closes
the silent part. The rest of this note is about the full feature.

## 2. The structural obstacle

The density modes never build a decayed matrix element. They fill one decay pool
per (particle, channel) and draw each particle's channel independently at run
time in `_draw_one_decay`
([interface_madspin.py:3328](../MadSpin/interface_madspin.py#L3328)):

* one decay line for the pdg -> that line;
* `ids.count(pdg) == nb_decay` -> **positional**, the i-th particle takes the
  i-th line;
* otherwise -> drawn at random, proportional to the channels' cross sections.

There is no object on which a cross-particle correlation can be imposed. The
obvious shape of a fix is therefore to move the channel choice from *per
particle* to *per event*: draw the group once, then let each particle take that
group's line for its pdg.

## 3. Is a per-event group draw correct?

Yes, and for the same reason the current per-particle draw is.

Write `R_g(prod)` for the physical rate of group `g` at a fixed production
event,

    R_g(prod) = Integral dOmega_1..dOmega_n  rho_prod . (x)_k D_k^(g) ,

and `Gamma_{k,g}` for the partial width of the channel group `g` gives to slot
`k`. The unpolarised factorisation of `R_g` is exactly `prod_k Gamma_{k,g}`,
which is a constant of the run, readable off the pools' cross sections. So:

* draw `g` with probability `p_g = prod_k Gamma_{k,g} / sum_h prod_k Gamma_{k,h}`;
* draw each slot from its group-`g` channel pool (already unweighted within the
  channel);
* keep the accept/reject weight **unchanged**.

The proposal density is then the unpolarised decay density over the union of the
groups, and the existing spin-correlation weight supplies the polarisation
dependence — precisely the role the per-channel `cross`-weighted draw plays
today. The group's polarisation enhancement is carried by the accept/reject, not
by `p_g`.

Two consequences worth stating up front:

* the branching ratio becomes `sum_g prod_k Gamma_{k,g} / Gamma_tot^n`, with **no
  factorial**: each group is an explicit assignment, so there is nothing to
  enumerate;
* groups with distinct final states do not interfere, so the incoherent sum over
  groups is exact. `madspin_v1` also treats them incoherently (`add process`).
  This matters for section 7.

## 4. What has to change

### 4.1 Threading the group (contained)

*Implemented.* `_draw_all_decays` draws the group (`_draw_decay_group`) and
passes it to `_draw_one_decay`, which restricts the candidate channels to the
ones that group gives the particle and then applies the existing rules to those,
unchanged. A group supplies exactly one channel per particle -- or one per
identical parent, which the positional rule then deals out -- so restricting the
candidates is the whole of the grouping at run time.

The draw sits at the *top* of `_draw_all_decays` rather than anywhere higher, so
the group is redrawn on every trial of the joint accept/reject along with the
decays it selects. That is what keeps the group part of what is being unweighted:
it is proposed and tested together with the angles, and no stage can normalise it
away. The joint max-weight scan goes through the same entry point, so the bound
is measured over the group mixture too.

`sequential_accept_reject` is the one caller that cannot take a group: it redraws
single slots until they are accepted, which divides `E[w_k | group]` out of the
chain, and that expectation differs between groups. `_sequential_active` refuses
the scheme outright when the decays are grouped (logged, like the `fixed_order`
fallback), and `sequential_accept_reject` raises if it is ever reached anyway.
Lifting that needs section 4.4.

### 4.2 Pool sizing (contained, but the refill needs care)

Today `run_onshell` decides per pdg
([interface_madspin.py:1875-1921](../MadSpin/interface_madspin.py#L1875))
between three shapes:

| `gen_jobs[pdg]['kind']` | when | pool layout |
|---|---|---|
| `simple` | `nb_needed == nb_event` | one merged pool (`cumul=True`), channels mixed by cross section |
| `mult_split` | `nb_needed == nb_mult*nb_event` and `len(list_branches[name]) == nb_mult` | one pool per channel, positional |
| `mult_cumul` | same but a different line count | one merged pool, drawn independently `nb_mult` times |

With groups, slot `k` of pdg `p` consumes channel `c(g,p)` only on the fraction
`p_g` of events, so the pool for `(p,c)` must be sized as

    sum over {g : c(g,p) = c}  p_g  x  nb_event x multiplicity / eff_k

with `eff_k` the ladder efficiency from `_sequential_pool_ladder`.

*Implemented, deliberately cruder.* `p_g` is only known once the partial widths
have been measured, i.e. once the generation this sizing controls has already
run. Rather than add a width pre-pass, the new `grouped` job kind sizes every
channel as if its group were drawn on every event. That over-generates by at most
a factor `|groups|` -- never under-generates, so no refill is forced -- and
`decay_event_mult` scales it down for anyone who minds. A cheap analytic width
estimate would recover the factor later.

The refill machinery is the part that needs attention rather than arithmetic.
`_channel_owner` deals channels out to workers round-robin and
`_open_refill_slice` undersizes the owner's slice by 10% so it runs dry first;
both assume every channel is consumed at a comparable rate. Under groups a rare
group's channel drains slowly and a common one fast, so the "owner runs out
first" heuristic degrades and refills fire unevenly. Nothing breaks — the
deadlock fail-safe and the published-generation protocol are rate-agnostic — but
the sizing margins would want re-tuning.

### 4.3 Branching-ratio bookkeeping (contained for the plain cases, not for BR
equalisation)

The three formulas at
[interface_madspin.py:1934-1942](../MadSpin/interface_madspin.py#L1934) are keyed
on line counts, which under groups no longer mean "channels available to a
slot":

* `mult_split`'s `pwidth / totwidth**nb_mult * factorial(nb_mult)` is the
  *positional* formula. The factorial compensates for generating only one of the
  `n!` assignments of `n` distinct channels to `n` identical parents; under
  groups the assignments are listed explicitly, so it must not be applied.
  (Verified separately that the factorial is right as it stands for the
  positional case: `p p > z z` with `decay z > e+ e-` + `decay z > u u~` gives
  BR = Gamma_ee.Gamma_uu/Gamma_Z^2 x 2 = 0.0081747, matching
  2 x BR_ee x BR_uu, and every event carries exactly one of each pair.)
* `mult_cumul`'s `(sum_c Gamma_c / Gamma_tot)**nb_mult` *is* the
  "each particle independently" formula groups exist to replace; it becomes
  `sum_g prod_k Gamma_{k,g} / Gamma_tot^n`.
* what a group means when a pdg has several *identical* parents is not defined
  by the card syntax at all. `decay z > e+ e- @1` twice? The positional rule and
  the group rule are two different mechanisms competing for the same slot, and
  the design has to pick one — the least surprising choice is that within a
  group the positional rule still applies to identical parents, i.e. a group
  supplies `n` lines per pdg with `n` parents.

`drop_prob_per_pdg`
([interface_madspin.py:1950-1965](../MadSpin/interface_madspin.py#L1950)) is the
uncontained one. It equalises branching ratios across production events that do
not all contain the same decaying species, by dropping events *per pdg* with
probability `1 - BR_pdg / BR_max`. Under groups the branching ratio is a property
of the **group**, not of a pdg: two groups differing in one line have different
total BR, and the drop probability would have to become per (final-state class,
group). That is a different data structure and a different correctness argument,
and it is the piece I would carve out of a first implementation (refuse groups
together with mixed final states, and say so).

### 4.4 The staged `unweighting` schemes (structural)

This is where the cost is.

* **Per-slot bounds.** `get_sequential_maxwgt`
  ([interface_madspin.py](../MadSpin/interface_madspin.py)) returns a flat `maxwgts` list indexed by position in the decay ordering, built
  from one probe vector per production event. Under groups each slot's weight
  distribution depends on the group, so the bound vector becomes one per group:
  `|slots| x |groups|` numbers, and `_combine_maxwgt` needs `|groups|` separate
  populations. The probe budget (`Nevents_for_max_weight x
  max_weight_ps_point`) then has to be split across groups; a rare group gets
  few samples, its bound is badly measured, and the sample is biased in exactly
  the way the overflow counter reports but nobody reads. Sizing the probe per
  group (rather than per event) is a change to the scan loop, not a parameter.

* **`Z_k(m)` tables.** `_z_slot_keys`
  ([interface_madspin.py](../MadSpin/interface_madspin.py)) keys the offshell
  rate factor by `<pdg>_<occurrence>`, with the docstring's
  justification that "slots of one pdg are consecutive and in production order,
  which is also how `_draw_one_decay` picks a decay file". That justification is
  exactly what groups break: the slot no longer determines the channel, the
  (slot, group) pair does. `Z_k` is the running partial width of the channel
  drawn, so the key must become `<pdg>_<occurrence>_<group>` and the table count
  multiplies by `|groups|` — with the same probe-splitting problem, and each fit
  needs enough samples for its quadratic in `ln(m/pole)` to be determined.

* **Cache format.** `_UPFRONT_CACHE_FORMAT` must be bumped: the bound vector and
  the table keys both change meaning, and an `ms_dir` written by today's code
  must not be read back.

* **What does *not* change:** `_decaying_pdgs`, `_density_basis`,
  `_sequential_slots` and the slot-to-particle map are all group-independent —
  the group changes which decay is drawn into a slot, never the layout of the
  density matrix. That is a real simplification and worth stating.

* **The joint scheme needs none of this.** A single bound over the whole chain
  already covers every group; the only cost is acceptance, since the bound is
  set by the loudest group.

### 4.5 `fixed_order`

`fixed_order` forces the joint accept/reject and processes event *groups*
(counter-events) that must all decay consistently. A per-event group draw must be
made once per event group, not once per event. Small, but easy to get wrong and
worth an explicit test.

### 4.6 The shape that is accepted (implemented)

Rectangular: **every group gives exactly `n_part` decay lines for every decaying
particle**, `n_part` being how many of that particle each production event
carries. An untagged line belongs to every group, as in `madspin_v1`.

```
decay t  > w+ b,  w+ > l+ vl   @1        n_part = 1: the semi-leptonic ttbar idiom
decay t~ > w- b~, w- > j j     @1
decay t  > w+ b,  w+ > j j     @2
decay t~ > w- b~, w- > l- vl~  @2

decay t > ... @1 ; decay t > ... @1      n_part = 2: p p > t t t~ t~, the group's
decay t > ... @2 ; decay t > ... @2      two lines dealt to the two tops by the
                                         existing positional rule
```

That single rule subsumes every refusal without a special case of its own: a
group missing a particle, a line count that does not match the multiplicity, and
a particle with both a tagged and an untagged line (the untagged one joins every
group, so that group ends up with one line too many) are all count mismatches.
Refused separately: a multiparticle parent (one name would own several pools),
production events that do not all carry the same particles (`drop_prob_per_pdg`
is per pdg -- section 4.3), `fixed_order`, and `spinmode` `none` / `onshell_v1`.

A refusal warns with its reason and falls back to the ungrouped behaviour rather
than raising: a card that merely over-specifies (a tagged line for a species that
never appears in the events, which MadSpin drops anyway) should not stop a run.

Implemented in `_decay_group_layout` (card only), `_validate_decay_groups`
(against the production events) and `_resolve_decay_groups` (mode, `fixed_order`,
and the conversion to pdg keys).

## 5. Contained or structural?

**Structural**, with a contained subset.

| piece | verdict |
|---|---|
| group draw + threading through `_draw_*` | contained — **done** |
| pool sizing | contained — **done**, at the cost noted in 4.2 |
| BR for the plain (one parent per pdg) case | contained — **done** |
| groups x positional rule for identical parents | **done**: inside a group the positional rule applies unchanged, so `p p > t t t~ t~` works |
| BR equalisation across mixed final states (`drop_prob_per_pdg`) | not contained — **refused**, with a reason |
| per-group bounds and `Z_k` tables in the staged schemes | **structural — not done.** The joint accept/reject is forced instead, and `sequential_accept_reject` raises if it is ever reached with groups |
| `fixed_order` event groups | contained, easy to get wrong — **refused** for now |

What the remaining row would cost: roughly twice the joint-only
implementation, plus a validation campaign, because the bias it can introduce is
a badly measured bound for a rare group -- invisible in a cross-section
comparison, and visible only in a lineshape or in the overflow counter.

## 6. The middle option, which is what landed

Groups are implemented for the joint scheme only, and `_unweighting_mode` falls
back to `joint` when groups are declared — the same way it already falls back
for `fixed_order` and for non-density spin modes, and with the same one-line
announcement. That gets the feature, keeps the tabulated machinery untouched,
and costs the user only acceptance. It also leaves the per-group bound question
answerable later, with a working feature to measure against.

## 7. And the honest comparison

Groups with distinct final states do not interfere, and `madspin_v1` itself adds
them incoherently. So **two MadSpin runs plus a merge is not an approximation of
this feature — it is the same sample**, up to:

1. the relative normalisation of the groups, which `set cross_section` fixes (or
   which the user can compute: the groups' rates are in the ratio
   `prod_k Gamma_{k,g}`), and
2. having to concatenate two LHE files.

So the argument *for* doing the work was never physics reach but ergonomics and
error-proneness: the normalisation step is exactly the sort of thing users get
wrong silently.

Section 6 is what landed: the density modes honour the tags for the rectangular
shape of section 4.6 and force the joint accept/reject while they do. Measured on `p p > t t~`, 2000 events, the card of section 1:

| mode | BR | (W,W) categories |
|---|---|---|
| `madspin` (density), before | 0.7529 | 760 semi-lep, 1116 fully hadronic, 124 fully leptonic |
| `madspin` (density), after | 0.28283 | **2000 semi-lep**, 1015 / 985 by charge |
| `PA` (density), after | 0.28283 | **2000 semi-lep**, 1037 / 963 by charge |
| `madspin_v1` | 0.29635 | 2000 semi-lep, 1001 / 999 by charge |

with the same BR serial (`nb_core 1`) and parallel. It is exactly
`sum_g prod_k Gamma_k / Gamma_tot^2` on this run's own measured widths
(`Gamma_lep = 0.32407`, `Gamma_had = 0.97099`, `Gamma_t = 1.4915` → `0.28281`).
The residual gap to `madspin_v1`'s 0.29635 is not from the grouping: it is the
pre-existing difference between how the two paths measure the partial widths,
and it shows in the ungrouped runs too (`0.7529 = (BR_l + BR_h)^2` with the same
widths).

### 7.1 The same sample — measured

`p p > t t~`, 20000 production events, decayed four times off the *same*
production file: the grouped card in one run, each group alone in a dedicated
run, and the grouped card under `madspin_v1`. The two dedicated runs decayed the
same events, so the reference is built pairwise — for production event *i*, take
`dedic1[i]` with probability `p_1 = sigma_1/(sigma_1+sigma_2)` and `dedic2[i]`
otherwise. That is by construction the mixture the grouped run draws, on
identical production kinematics, so anything left over is the grouping itself.

| | sigma (pb) |
|---|---|
| dedicated 1 (`t > l+ nu`, `t~ > j j`) | 71.26739 |
| dedicated 2 (`t > j j`, `t~ > l- nu`) | 71.24537 |
| **sum** | **142.51276** |
| grouped, one run | 142.54946 &nbsp;&nbsp; ratio **1.000258** |
| `madspin_v1` | 149.48100 &nbsp;&nbsp; ratio 1.048896 |

The grouped run also reports the group shares as `@1 = 0.4999, @2 = 0.5001`
against the `0.50008` the two dedicated cross sections imply.

Means, grouped against the merged reference (`cos*` is the child's angle in its
W rest frame against the W direction in its top's rest frame — the spin
analyser; `prod` is the ttbar spin-correlation handle):

| | grouped | merged | pull |
|---|---|---|---|
| `cos*_lep` | -0.14628 | -0.14162 | -0.9 |
| `cos*_down` | -0.13923 | -0.14408 | +1.0 |
| `cos*_lep · cos*_down` | 0.01894 | 0.01972 | -0.3 |
| `dphi(l, d)` | 1.74943 | 1.75194 | -0.3 |
| `pT(lepton)` | 51.458 | 51.525 | -0.2 |
| `pT(leptonic top)` | 120.235 | 120.254 | -0.0 |
| `m(leptonic top)` | 173.192 | 173.184 | +0.3 |
| lepton-from-top fraction | 0.4996 | 0.5001 | -0.1 |

Two-sample Kolmogorov-Smirnov on the same seven distributions: `D` between
0.0027 and 0.0081, `p` between 0.53 and 1.00. Nothing distinguishes them.

The 2.6e-4 on the cross section is the two sides measuring the same partial
widths in independent MG5 integrations, not a bias. The 4.9% against
`madspin_v1` is the pre-existing difference already noted above: the density
path integrates the 3-body `t > b f f'` and gets `Gamma_lep/Gamma_t = 0.21705`
where the naive `Gamma_t x BR(W)` would give 2/9 = 0.22222, the Breit-Wigner
being truncated by `bwcutoff` and suppressed below threshold. It is 2.3% per
leg, hence 4.7% on the product, and it is there in the ungrouped runs too.

### 7.2 The same again with two parents per pdg — `p p > t t~ t t~`

`n_part = 2`, so each group supplies *two* lines for each pdg and the positional
rule deals them out inside the group:

```
decay t  > w+ b, w+ > l+ vl  @1        decay t  > w+ b, w+ > j j     @2
decay t  > w+ b, w+ > j j    @1        decay t  > w+ b, w+ > j j     @2
decay t~ > w- b~, w- > j j   @1        decay t~ > w- b~, w- > l- vl~ @2
decay t~ > w- b~, w- > j j   @1        decay t~ > w- b~, w- > j j    @2
```

Exactly one lepton, from a top in group 1 and from an antitop in group 2. Eight
pools, four per pdg. Group @1's two `t~` lines are *identical*, which makes this
card a test of the assignment factor as well.

20000 production events, an independent production seed from section 7.1:

| | sigma (pb) |
|---|---|
| dedicated 1 (lepton from a top) | 0.0010721 |
| dedicated 2 (lepton from an antitop) | 0.0010722 |
| **sum** | **0.0021442** |
| grouped, one run | 0.0021416 &nbsp;&nbsp; ratio **0.998801** |

Group shares reported `@1 = 0.5001, @2 = 0.4999` against the 0.49998 the
dedicated cross sections imply. Every pull against the merged reference is
within 1.0 sigma -- `cos*_lep` -0.2, `cos*_had` -0.5, `pT(lepton)` -0.9,
`pT(leptonic top)` -1.0, `m(leptonic top)` +0.3, `HT(4 tops)` 0.0, charge split
-0.1 -- and two-sample KS over the six distributions gives `p` from 0.20 to 1.00.
(A first pass at 5000 events had `cos*_lep` at 2.3 sigma, `p = 0.03`; it is
0.751 here, so that was the fluctuation it looked like.)

Structure, 20000/20000 events in all three samples: exactly one charged lepton;
the lepton comes from a top 0.4996 of the time in the grouped sample against
1.0000 and 0.0000 in the two dedicated ones; and **the leptonic parent is always
the first of its pdg**, which is the positional rule operating inside the group.

The 1.2e-3 on the cross section is again the independent width measurements --
four factors per group here instead of two, and each MadEvent estimate is good
to about 0.1%.

#### The assignment factor, against a path that has none

`n!` counts the ways `n` *distinct* channels go to `n` identical parents; with a
repeated line it counts the same final state twice, so the factor is
`n!/prod_c m_c!`. Group @1 above repeats its `t~` line, which makes that
measurable. Writing the same group as a standalone card two ways:

| | sigma (pb) |
|---|---|
| `t~` line written **twice** -> `mult_split`, factor `2!/2! = 1` | 0.0010777 |
| `t~` line written **once** -> `mult_cumul`, which applies no factor at all | 0.0010767 |
| ratio | **1.000886** |

The old plain `n!` would have put that ratio at 2. Two code paths, one of which
has no assignment factor in it, agreeing to 9e-4.

### 7.3 PA, and the `unweighting` fallback

Grouping forces the joint accept/reject (section 4.1), so both need checking:
that the fallback happens, and that it costs nothing but efficiency.

**The fallback fires and is a no-op.** `PA` (where `unweighting = auto` resolves
to a per-particle scheme) and `madspin` with an explicit `set unweighting
sequential` both log

```
MadSpin: the decay lines are grouped ('@' tags), keeping the joint
accept/reject (unweighting ignored)
```

and `set unweighting sequential` on a grouped card produces event records
*byte-identical* to the same card run at the default -- the option is read,
overridden, and changes nothing.

**PA reproduces the dedicated runs.** Same 20000 `p p > t t~` events, `PA`
throughout, with the dedicated runs put on the joint test too so both sides use
the same accept/reject:

| | sigma (pb) |
|---|---|
| dedicated 1 + dedicated 2 | 71.26739 + 71.24537 = 142.51276 |
| grouped, one run | 142.54946 &nbsp;&nbsp; ratio 1.000258 |

Sharper than the merged comparison, and free of its binomial noise: split the
grouped sample by which group each event came from and compare each half with
*its* dedicated run. Over 3 MadSpin seeds x 2 groups x 5 observables
(`pT(lepton)`, `m(top)`, both spin analysers and their product) — 30
comparisons — three land at 2.0-2.7 sigma and none of them reproduces at another
seed, which is what statistics looks like and not what a bias looks like. Every
KS is above 0.01.

**A pre-existing lineshape bias, found on the way -- since fixed.** The first
PA/sequential comparison put `m(top)` at 7.8 sigma, and it was not the grouping.
Taking one *ungrouped* card, the same production events, and changing nothing
but the accept/reject, on the tree as it then stood:

| `decay t > w+ b, w+ > l+ vl` + `decay t~ > w- b~, w- > j j` | mean `m(top_lep)` |
|---|---|
| `unweighting joint` | 173.16870 |
| `unweighting sequential` | 172.94647 |
| | **7.0 sigma**, KS `p = 0.0000` |

Every angular observable agreed between the two (all within 1.0 sigma); only the
virtuality moved, and it moved *down*. That is the signature of the missing
offshell rate factor `Z_k`: the per-slot stage redraws each decay to acceptance,
which divides `E[w_k | m] = Z_k(m)` out of the accepted mass sets, so the
lineshape relaxes towards the Breit-Wigner instead of the offshell one -- and
since the running width grows with `m`, dropping `Z_k` pulls the mean low. `PA`
showed the same effect at 2.1 sigma, where there is no running width to lose but
the per-slot mass redraw normalises itself the same way.

**The tabulated `Z_k` closed it**, and the closure is measured in
`madspin_sequential_plan.md`: section 10 ("A/B after the fix") puts the offshell
residual at **-0.022 +- 0.024 GeV** against the **-0.248 GeV (-7.8 sigma)** that
was there before, and section 11 measures the PA factor directly by forcing
`_zhat` to 1 (**-0.039 +- 0.016 GeV**, recovered to +0.009 +- 0.015 when it is
restored). So the sequential schemes are no longer biased in the top lineshape,
and the comparisons above are a record of why the tables exist rather than a
live caveat.

What it does *not* change is the argument for forcing joint on grouped cards:
the tables are keyed per slot, not per (slot, group), which is exactly section
4.4.

## 8. And the honest comparison, still

Section 4.4 remains open, and with it the argument above: two runs plus
`set cross_section` still produce the same sample, so what this bought is
ergonomics, not reach.
