# Supporting the `@` grouping tags in the density spin modes

Design note. Written against `madspin_density` (275462e52) with an eye on the
sequential/two-stage unweighting schemes of PR #334
(`claude/madspin-sequential-offshell-rate-factor`, 6c051b6d4).

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

`_draw_all_decays` gains a per-event group index and passes it to
`_draw_one_decay`, which uses it instead of the positional/random logic when
groups are declared. The important consumer is
`sequential_accept_reject`
([interface_madspin.py:4603 on PR #334](../MadSpin/interface_madspin.py)), which
calls `_draw_one_decay` per slot and **redraws single slots** on a rejection:
those redraws must stay inside the group already chosen for the chain. That is a
parameter, not a restructuring.

The group must be redrawn at the same point the chain restarts (the
`while True:` mass-set restart), otherwise a group whose feasibility is
production-dependent would be over-represented.

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

with `eff_k` the ladder efficiency from `_sequential_pool_ladder`
([interface_madspin.py:2189 on PR #334](../MadSpin/interface_madspin.py)). This
is a per-channel weight in `gen_jobs`, which the ladder does not currently carry
(it is per pdg). Contained.

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

### 4.4 The sequential / two-stage unweighting (structural)

This is where the cost is.

* **Per-slot bounds.** `get_sequential_maxwgt`
  ([interface_madspin.py:3878 on PR #334](../MadSpin/interface_madspin.py))
  returns a flat `maxwgts` list indexed by position in the decay ordering, built
  from one probe vector per production event. Under groups each slot's weight
  distribution depends on the group, so the bound vector becomes one per group:
  `|slots| x |groups|` numbers, and `_combine_maxwgt` needs `|groups|` separate
  populations. The probe budget (`Nevents_for_max_weight x
  max_weight_ps_point`) then has to be split across groups; a rare group gets
  few samples, its bound is badly measured, and the sample is biased in exactly
  the way the overflow counter reports but nobody reads. Sizing the probe per
  group (rather than per event) is a change to the scan loop, not a parameter.

* **`Z_k(m)` tables.** `_z_slot_keys`
  ([interface_madspin.py:4400 on PR #334](../MadSpin/interface_madspin.py))
  keys the offshell rate factor by `<pdg>_<occurrence>`, with the docstring's
  justification that "slots of one pdg are consecutive and in production order,
  which is also how `_draw_one_decay` picks a decay file". That justification is
  exactly what groups break: the slot no longer determines the channel, the
  (slot, group) pair does. `Z_k` is the running partial width of the channel
  drawn, so the key must become `<pdg>_<occurrence>_<group>` and the table count
  multiplies by `|groups|` — with the same probe-splitting problem, and each fit
  needs enough samples for its quadratic in `ln(m/pole)` to be determined.

* **Cache format.** `_OFFSHELL_CACHE_FORMAT` must be bumped: the bound vector and
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

## 5. Contained or structural?

**Structural**, with a contained subset.

| piece | verdict |
|---|---|
| group draw + threading through `_draw_*` and the sequential retry | contained |
| pool sizing weights | contained; refill margins want re-tuning |
| BR for the plain (one parent per pdg) case | contained |
| groups x positional rule for identical parents | needs a syntax decision first |
| BR equalisation across mixed final states (`drop_prob_per_pdg`) | not contained — different data structure |
| per-group bounds and `Z_k` tables in sequential/two-stage | **structural** — the tabulated per-slot state multiplies by `\|groups\|` and the probe budget has to be split |
| `fixed_order` event groups | contained, easy to get wrong |

Rough effort: a joint-only implementation (density modes, `unweighting = joint`,
refusing groups with mixed final states and with several identical parents) is a
few hundred lines plus tests — call it a few days. Extending it to the
sequential and two-stage schemes roughly doubles that and adds a validation
campaign, because the bias it can introduce is a badly measured bound for a rare
group, which is invisible in a cross-section comparison and only shows up in a
lineshape or in the overflow counter. A week to two, end to end.

## 6. A cheaper middle option

Implement groups for the joint scheme only, and have `_unweighting_mode` fall
back to `joint` when groups are declared — the same way it already falls back
for `fixed_order` and for non-density spin modes, and with the same one-line
announcement. That gets the feature, keeps the tabulated machinery untouched,
and costs the user only acceptance. It also means the per-group bound question
can be answered later, with a working feature to measure against.

## 7. And the honest comparison

Groups with distinct final states do not interfere, and `madspin_v1` itself adds
them incoherently. So **two MadSpin runs plus a merge is not an approximation of
this feature — it is the same sample**, up to:

1. the relative normalisation of the groups, which `set cross_section` fixes (or
   which the user can compute: the groups' rates are in the ratio
   `prod_k Gamma_{k,g}`), and
2. having to concatenate two LHE files.

That is worth weighing before spending the week. The strongest argument *for*
doing the work is not physics reach but ergonomics and error-proneness: the
normalisation step is exactly the sort of thing users get wrong silently. The
strongest argument against is that the same week spent on the sequential
schemes' per-slot bounds buys more.

Recommendation: ship the warning (done on this branch), document the two-run
recipe (already in `doc/madspin_options.tex`), and treat full support as
optional — and if it is taken up, do section 6 first.
